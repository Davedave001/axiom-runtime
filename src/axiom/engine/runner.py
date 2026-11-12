"""Core Axiom Runtime execution engine with checkpointing, retries, idempotency, and approval gates."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

from axiom.engine.state_machine import ExecutionStateMachine
from axiom.gateway.adapter import ModelAdapter, ModelTimeoutError
from axiom.policy.engine import PolicyEngine
from axiom.schemas.domain import (
    AgentDefinition,
    AgentRun,
    ApprovalRequest,
    ApprovalStatus,
    Checkpoint,
    FailureCategory,
    PolicyDecision,
    RunStatus,
    TaskSubmission,
    ToolCallSpec,
    ToolInvocationRecord,
    ToolInvocationStatus,
    TraceEvent,
    utc_now,
)
from axiom.state.store import StateStore
from axiom.tools.registry import (
    InvalidToolArgumentsError,
    ToolExecutionError,
    ToolRegistry,
)


class WorkerInterruptedError(Exception):
    """Simulated or real worker interruption mid-step (after external tool completion, before step commit)."""


class AxiomRunner:
    """Executes stateful AI agent runs with durable state transitions and idempotency guarantees."""

    def __init__(
        self,
        store: StateStore,
        model_adapter: ModelAdapter,
        tool_registry: ToolRegistry,
        policy_engine: Optional[PolicyEngine] = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.store = store
        self.model_adapter = model_adapter
        self.tool_registry = tool_registry
        self.policy_engine = policy_engine or PolicyEngine(tool_registry)
        self.clock = clock
        self._agents: Dict[str, AgentDefinition] = {}

    def register_agent(self, agent: AgentDefinition) -> None:
        self._agents[agent.name] = agent

    def submit_task(self, submission: TaskSubmission) -> Tuple[AgentRun, bool]:
        """Create a new AgentRun or deduplicate if idempotency_key was already submitted."""
        if submission.agent_name not in self._agents:
            raise KeyError(f"Agent '{submission.agent_name}' is not registered")

        initial_run = AgentRun(
            idempotency_key=submission.idempotency_key,
            agent_name=submission.agent_name,
            goal=submission.goal,
            status=RunStatus.CREATED,
            conversation_history=[
                {"role": "user", "content": submission.goal, "context": submission.input_context}
            ],
        )
        run, created = self.store.create_or_get_run(initial_run)
        if created:
            self._emit_trace(
                run,
                event_type="RUN_CREATED",
                state_from=None,
                state_to=RunStatus.CREATED,
                details={"task_id": submission.task_id, "idempotency_key": submission.idempotency_key},
            )
            self._save_checkpoint(run)
        else:
            self._emit_trace(
                run,
                event_type="DUPLICATE_SUBMISSION_DEDUPLICATED",
                state_from=run.status,
                state_to=run.status,
                failure_category=FailureCategory.DUPLICATE_SUBMISSION,
                details={"idempotency_key": submission.idempotency_key},
            )
        return run, created

    def execute_run(
        self,
        run_id: str,
        approval_timeout_seconds: int = 300,
        interrupt_after_tool_execution: bool = False,
    ) -> AgentRun:
        """Drive an AgentRun forward until it completes, pauses for approval, or fails."""
        run = self.store.get_run(run_id)
        agent = self._agents[run.agent_name]

        if ExecutionStateMachine.is_terminal(run.status):
            return run

        if run.status == RunStatus.WAITING_FOR_APPROVAL:
            return self._check_approval_gate(run, approval_timeout_seconds, interrupt_after_tool_execution)

        # Resume from checkpointed or created state
        if run.status in (RunStatus.CREATED, RunStatus.CHECKPOINTED, RunStatus.RETRYING):
            run = self._transition(run, RunStatus.RUNNING, "STEP_STARTED")

        while not ExecutionStateMachine.is_terminal(run.status):
            if run.current_step >= agent.max_steps:
                return self._fail_run(
                    run,
                    FailureCategory.MAX_STEPS_EXCEEDED,
                    f"Exceeded max_steps ({agent.max_steps})",
                )

            # If recovering from a crash where a pending_tool_call was already selected:
            if run.pending_tool_call is not None:
                run = self._transition(run, RunStatus.WAITING_FOR_TOOL, "RESUMING_PENDING_TOOL")
            else:
                run = self._invoke_model_with_retries(run, agent)
                if ExecutionStateMachine.is_terminal(run.status):
                    return run
                if run.status == RunStatus.COMPLETED:
                    return run

            # Handle WAITING_FOR_TOOL -> Policy check -> Approval or Execution
            if run.status == RunStatus.WAITING_FOR_TOOL and run.pending_tool_call is not None:
                decision = self.policy_engine.evaluate(agent, run.pending_tool_call)
                if decision == PolicyDecision.DENY:
                    return self._fail_run(
                        run,
                        FailureCategory.POLICY_DENIED,
                        f"Policy denied tool '{run.pending_tool_call.tool_name}'",
                    )
                if decision == PolicyDecision.REQUIRE_APPROVAL:
                    approval = ApprovalRequest(
                        run_id=run.run_id,
                        step_index=run.current_step,
                        tool_call=run.pending_tool_call,
                        requested_at=self.clock(),
                        expires_at=self.clock() + timedelta(seconds=approval_timeout_seconds),
                    )
                    self.store.save_approval_request(approval)
                    run.pending_approval_id = approval.approval_id
                    run = self._transition(
                        run,
                        RunStatus.WAITING_FOR_APPROVAL,
                        "APPROVAL_REQUESTED",
                        details={"approval_id": approval.approval_id, "tool": run.pending_tool_call.tool_name},
                    )
                    self._save_checkpoint(run)
                    return run

                # PolicyDecision.ALLOW -> Execute tool
                run = self._execute_tool_with_retries(
                    run,
                    agent,
                    interrupt_after_tool_execution=interrupt_after_tool_execution,
                )
                if ExecutionStateMachine.is_terminal(run.status) or run.status == RunStatus.WAITING_FOR_APPROVAL:
                    return run

                # Move from CHECKPOINTED -> RUNNING for next step
                if run.status == RunStatus.CHECKPOINTED:
                    run = self._transition(run, RunStatus.RUNNING, "NEXT_STEP_READY")

        return run

    def approve_tool_call(
        self,
        run_id: str,
        decided_by: str = "human-operator",
        reason: str = "Approved",
    ) -> AgentRun:
        """Approve a pending human-in-the-loop gate and resume execution."""
        run = self.store.get_run(run_id)
        if run.status != RunStatus.WAITING_FOR_APPROVAL or not run.pending_approval_id:
            raise ValueError(f"Run {run_id} is not waiting for approval")

        approval = self.store.get_approval_request(run.pending_approval_id)
        now = self.clock()
        if now > approval.expires_at:
            approval.status = ApprovalStatus.EXPIRED
            self.store.save_approval_request(approval)
            return self._fail_run(
                run,
                FailureCategory.APPROVAL_TIMEOUT,
                f"Approval request {approval.approval_id} expired at {approval.expires_at.isoformat()}",
            )

        approval.status = ApprovalStatus.APPROVED
        approval.decided_by = decided_by
        approval.decided_at = now
        approval.reason = reason
        self.store.save_approval_request(approval)

        self._emit_trace(
            run,
            event_type="APPROVAL_GRANTED",
            state_from=RunStatus.WAITING_FOR_APPROVAL,
            state_to=RunStatus.WAITING_FOR_APPROVAL,
            details={"approval_id": approval.approval_id, "decided_by": decided_by},
        )

        agent = self._agents[run.agent_name]
        run = self._execute_tool_with_retries(run, agent, interrupt_after_tool_execution=False)
        if run.status == RunStatus.CHECKPOINTED:
            run = self._transition(run, RunStatus.RUNNING, "RESUMED_AFTER_APPROVAL")
            return self.execute_run(run.run_id)
        return run

    def resume_from_checkpoint(self, run_id: str) -> AgentRun:
        """Restore a run from its latest persisted checkpoint after a worker interruption and continue."""
        run = self.store.get_run(run_id)
        checkpoint = self.store.get_latest_checkpoint(run_id)
        if checkpoint is not None:
            run.current_step = checkpoint.step_index
            run.conversation_history = checkpoint.conversation_history
            run.pending_tool_call = checkpoint.pending_tool_call
            run.pending_approval_id = checkpoint.pending_approval_id

        self._emit_trace(
            run,
            event_type="WORKER_RECOVERY_FROM_CHECKPOINT",
            state_from=run.status,
            state_to=run.status,
            details={
                "checkpoint_id": checkpoint.checkpoint_id if checkpoint else None,
                "step_index": run.current_step,
            },
        )

        if run.status == RunStatus.EXECUTING_TOOL and run.pending_tool_call is not None:
            agent = self._agents[run.agent_name]
            run = self._execute_tool_with_retries(run, agent, interrupt_after_tool_execution=False)
            if run.status == RunStatus.CHECKPOINTED:
                run = self._transition(run, RunStatus.RUNNING, "RESUMED_AFTER_CRASH")

        return self.execute_run(run.run_id)

    def _check_approval_gate(
        self,
        run: AgentRun,
        approval_timeout_seconds: int,
        interrupt_after_tool_execution: bool,
    ) -> AgentRun:
        if not run.pending_approval_id:
            return run
        approval = self.store.get_approval_request(run.pending_approval_id)
        if self.clock() > approval.expires_at:
            approval.status = ApprovalStatus.EXPIRED
            self.store.save_approval_request(approval)
            return self._fail_run(
                run,
                FailureCategory.APPROVAL_TIMEOUT,
                f"Approval timed out after {approval_timeout_seconds}s",
            )
        return run

    def _invoke_model_with_retries(self, run: AgentRun, agent: AgentDefinition) -> AgentRun:
        max_attempts = agent.retry_policy.max_attempts
        tools_schema = self.tool_registry.list_mcp_descriptors(agent.allowed_tools)

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.model_adapter.generate(
                    agent=agent,
                    conversation_history=run.conversation_history,
                    available_tools=tools_schema,
                )
                run.total_prompt_tokens += response.prompt_tokens
                run.total_completion_tokens += response.completion_tokens

                if response.tool_call is not None:
                    run.pending_tool_call = response.tool_call
                    run = self._transition(
                        run,
                        RunStatus.WAITING_FOR_TOOL,
                        "MODEL_REQUESTED_TOOL",
                        details={
                            "tool_name": response.tool_call.tool_name,
                            "attempt": attempt,
                        },
                    )
                    self._save_checkpoint(run)
                    return run

                # Final answer reached
                run.final_output = response.final_answer or ""
                run.conversation_history.append(
                    {"role": "assistant", "content": run.final_output}
                )
                run.current_step += 1
                run = self._transition(
                    run,
                    RunStatus.COMPLETED,
                    "RUN_COMPLETED",
                    details={"final_output_length": len(run.final_output)},
                )
                self._save_checkpoint(run)
                return run

            except ModelTimeoutError as exc:
                self._emit_trace(
                    run,
                    event_type="MODEL_TIMEOUT",
                    state_from=run.status,
                    state_to=RunStatus.RETRYING if attempt < max_attempts else RunStatus.FAILED,
                    failure_category=FailureCategory.MODEL_TIMEOUT,
                    details={"attempt": attempt, "max_attempts": max_attempts, "error": str(exc)},
                )
                if attempt < max_attempts:
                    run = self._transition(run, RunStatus.RETRYING, "MODEL_RETRY_SCHEDULED")
                    run = self._transition(run, RunStatus.RUNNING, "MODEL_RETRY_STARTED")
                else:
                    return self._fail_run(
                        run,
                        FailureCategory.MODEL_TIMEOUT,
                        f"Model timed out after {max_attempts} attempts: {exc}",
                    )

        return run

    def _execute_tool_with_retries(
        self,
        run: AgentRun,
        agent: AgentDefinition,
        interrupt_after_tool_execution: bool = False,
    ) -> AgentRun:
        tool_call: Optional[ToolCallSpec] = run.pending_tool_call
        if tool_call is None:
            raise ValueError("Cannot execute tool when pending_tool_call is None")

        if run.status != RunStatus.EXECUTING_TOOL:
            run = self._transition(
                run,
                RunStatus.EXECUTING_TOOL,
                "TOOL_EXECUTION_STARTED",
                details={"tool_name": tool_call.tool_name},
            )
            self._save_checkpoint(run)

        # 1. Validate arguments against Pydantic schema
        try:
            validated_args = self.tool_registry.validate_arguments(
                tool_call.tool_name, tool_call.arguments
            )
        except InvalidToolArgumentsError as exc:
            self._emit_trace(
                run,
                event_type="INVALID_TOOL_ARGUMENTS",
                state_from=run.status,
                state_to=RunStatus.FAILED,
                failure_category=FailureCategory.INVALID_TOOL_ARGUMENTS,
                details={"tool_name": tool_call.tool_name, "validation_error": str(exc)},
            )
            return self._fail_run(
                run,
                FailureCategory.INVALID_TOOL_ARGUMENTS,
                f"Invalid arguments for tool '{tool_call.tool_name}': {exc}",
            )

        # 2. Check idempotency record before invoking external tool
        idem_key = tool_call.compute_idempotency_key(run.run_id, run.current_step)
        existing_record = self.store.get_tool_invocation(idem_key)

        if existing_record is not None and existing_record.status == ToolInvocationStatus.COMPLETED:
            self._emit_trace(
                run,
                event_type="TOOL_IDEMPOTENCY_HIT",
                state_from=run.status,
                state_to=run.status,
                details={"idempotency_key": idem_key, "tool_name": tool_call.tool_name},
            )
            tool_output = existing_record.result
        else:
            max_attempts = agent.retry_policy.max_attempts
            tool_output = None
            for attempt in range(1, max_attempts + 1):
                record = ToolInvocationRecord(
                    run_id=run.run_id,
                    step_index=run.current_step,
                    idempotency_key=idem_key,
                    tool_name=tool_call.tool_name,
                    arguments=validated_args,
                    status=ToolInvocationStatus.PENDING,
                    attempt_count=attempt,
                )
                self.store.save_tool_invocation(record)

                try:
                    tool_output = self.tool_registry.invoke(tool_call.tool_name, validated_args)
                    record.status = ToolInvocationStatus.COMPLETED
                    record.result = tool_output
                    record.completed_at = self.clock()
                    self.store.save_tool_invocation(record)

                    self._emit_trace(
                        run,
                        event_type="TOOL_EXECUTION_COMPLETED",
                        state_from=run.status,
                        state_to=run.status,
                        details={
                            "tool_name": tool_call.tool_name,
                            "attempt": attempt,
                            "idempotency_key": idem_key,
                        },
                    )
                    break
                except ToolExecutionError as exc:
                    record.status = ToolInvocationStatus.FAILED
                    record.error_message = str(exc)
                    self.store.save_tool_invocation(record)

                    self._emit_trace(
                        run,
                        event_type="TOOL_EXECUTION_FAILED",
                        state_from=run.status,
                        state_to=RunStatus.RETRYING if (exc.transient and attempt < max_attempts) else RunStatus.FAILED,
                        failure_category=FailureCategory.TOOL_FAILURE,
                        details={
                            "tool_name": tool_call.tool_name,
                            "attempt": attempt,
                            "transient": exc.transient,
                            "error": str(exc),
                        },
                    )
                    if exc.transient and attempt < max_attempts:
                        run = self._transition(run, RunStatus.RETRYING, "TOOL_RETRY_SCHEDULED")
                        run = self._transition(run, RunStatus.EXECUTING_TOOL, "TOOL_RETRY_STARTED")
                    else:
                        return self._fail_run(
                            run,
                            FailureCategory.TOOL_FAILURE,
                            f"Tool '{tool_call.tool_name}' failed after {attempt} attempts: {exc}",
                        )

        # Simulate worker interruption AFTER external tool completed and recorded idempotency,
        # but BEFORE the run transitions to CHECKPOINTED.
        if interrupt_after_tool_execution:
            self._emit_trace(
                run,
                event_type="WORKER_INTERRUPTED",
                state_from=run.status,
                state_to=run.status,
                failure_category=FailureCategory.WORKER_INTERRUPTION,
                details={"step_index": run.current_step, "idempotency_key": idem_key},
            )
            raise WorkerInterruptedError(
                f"Worker interrupted at step {run.current_step} after tool '{tool_call.tool_name}' completed"
            )

        # Commit step output and checkpoint
        run.conversation_history.append(
            {
                "role": "tool",
                "tool_name": tool_call.tool_name,
                "call_id": tool_call.call_id,
                "output": tool_output,
            }
        )
        run.pending_tool_call = None
        run.pending_approval_id = None
        run.current_step += 1
        run = self._transition(
            run,
            RunStatus.CHECKPOINTED,
            "STEP_CHECKPOINTED",
            details={"completed_step": run.current_step},
        )
        self._save_checkpoint(run)
        return run

    def _transition(
        self,
        run: AgentRun,
        target_status: RunStatus,
        event_type: str,
        failure_category: FailureCategory = FailureCategory.NONE,
        details: Optional[Dict[str, Any]] = None,
    ) -> AgentRun:
        ExecutionStateMachine.validate_transition(run.status, target_status)
        previous_status = run.status
        run.status = target_status
        run = self.store.update_run(run, expected_version=run.version)
        self._emit_trace(
            run,
            event_type=event_type,
            state_from=previous_status,
            state_to=target_status,
            failure_category=failure_category,
            details=details or {},
        )
        return run

    def _fail_run(
        self,
        run: AgentRun,
        category: FailureCategory,
        message: str,
    ) -> AgentRun:
        previous_status = run.status
        ExecutionStateMachine.validate_transition(previous_status, RunStatus.FAILED)
        run.status = RunStatus.FAILED
        run.failure_category = category
        run.error_message = message
        run = self.store.update_run(run, expected_version=run.version)
        self._emit_trace(
            run,
            event_type="RUN_FAILED",
            state_from=previous_status,
            state_to=RunStatus.FAILED,
            failure_category=category,
            details={"error_message": message},
        )
        self._save_checkpoint(run)
        return run

    def _save_checkpoint(self, run: AgentRun) -> None:
        cp = Checkpoint(
            run_id=run.run_id,
            step_index=run.current_step,
            status_at_checkpoint=run.status,
            conversation_history=run.conversation_history,
            pending_tool_call=run.pending_tool_call,
            pending_approval_id=run.pending_approval_id,
        )
        self.store.save_checkpoint(cp)

    def _emit_trace(
        self,
        run: AgentRun,
        event_type: str,
        state_from: Optional[RunStatus],
        state_to: Optional[RunStatus],
        failure_category: FailureCategory = FailureCategory.NONE,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = TraceEvent(
            run_id=run.run_id,
            step_index=run.current_step,
            event_type=event_type,
            state_from=state_from,
            state_to=state_to,
            failure_category=failure_category,
            details=details or {},
            timestamp=self.clock(),
        )
        self.store.append_trace_event(event)
