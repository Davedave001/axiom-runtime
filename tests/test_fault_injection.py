"""End-to-end and Fault-Injection Test Suite for the Research-and-Reporting Agent Demo.

Verifies the happy-path lifecycle plus all 6 failure scenarios specified in the Portfolo doc:
1. Model API timeout
2. Tool failure
3. Worker interruption (with idempotent checkpoint recovery)
4. Duplicate task submission
5. Invalid tool arguments
6. Approval timeout
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from axiom.demo.research_agent import (
    SearchToolArgs,
    build_research_tool_registry,
    create_research_agent_definition,
    default_research_agent_script,
)
from axiom.engine.runner import AxiomRunner, WorkerInterruptedError
from axiom.gateway.adapter import ModelTimeoutError, ScriptedModelAdapter
from axiom.schemas.domain import (
    AgentDefinition,
    FailureCategory,
    ModelResponse,
    PolicyDecision,
    RunStatus,
    TaskSubmission,
    ToolCallSpec,
)
from axiom.state.store import InMemoryStateStore, SQLiteStateStore
from axiom.tools.registry import ToolExecutionError


@pytest.mark.parametrize("store_cls", [InMemoryStateStore, SQLiteStateStore])
def test_happy_path_research_and_reporting_agent_with_approval(store_cls) -> None:
    """Verify full lifecycle: search -> summarize -> pause for approval -> resume -> final report."""
    side_effects: Dict[str, int] = {}
    store = store_cls()
    registry = build_research_tool_registry(side_effects)
    adapter = ScriptedModelAdapter(default_research_agent_script)
    runner = AxiomRunner(store=store, model_adapter=adapter, tool_registry=registry)
    runner.register_agent(create_research_agent_definition())

    submission = TaskSubmission(
        idempotency_key="task-happy-001",
        agent_name="research-and-reporting-agent",
        goal="Research agent runtime reliability and publish executive report",
    )
    run, created = runner.submit_task(submission)
    assert created is True

    # Execute until human approval gate on publish_report_tool
    paused_run = runner.execute_run(run.run_id)
    assert paused_run.status == RunStatus.WAITING_FOR_APPROVAL
    assert paused_run.pending_approval_id is not None
    assert side_effects["search_tool"] == 1
    assert side_effects["summarize_tool"] == 1
    assert side_effects.get("publish_report_tool", 0) == 0

    # Approve and resume to completion
    completed_run = runner.approve_tool_call(
        paused_run.run_id,
        decided_by="david@axiom.dev",
        reason="Verified synthesis accuracy",
    )
    assert completed_run.status == RunStatus.COMPLETED
    assert "Final Report Published" in (completed_run.final_output or "")
    assert side_effects["publish_report_tool"] == 1
    assert len(store.list_checkpoints(completed_run.run_id)) >= 4


def test_fault_1_model_api_timeout_recovery_and_exhaustion() -> None:
    """Scenario 1: Model API timeout — verify bounded retry recovery and terminal classification."""
    side_effects: Dict[str, int] = {}
    store = SQLiteStateStore()
    registry = build_research_tool_registry(side_effects)

    # Part A: Transient timeout on call 1 succeeds on retry call 2
    def transient_timeout_script(
        agent: AgentDefinition, history: List[Dict[str, Any]], call_count: int
    ) -> ModelResponse:
        if call_count == 1:
            raise ModelTimeoutError("Upstream LLM gateway 504 Gateway Timeout")
        return default_research_agent_script(agent, history, call_count)

    runner = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(transient_timeout_script),
        tool_registry=registry,
    )
    runner.register_agent(create_research_agent_definition())
    run, _ = runner.submit_task(
        TaskSubmission(
            idempotency_key="task-timeout-recover",
            agent_name="research-and-reporting-agent",
            goal="Recover from transient model timeout",
        )
    )
    paused = runner.execute_run(run.run_id)
    assert paused.status == RunStatus.WAITING_FOR_APPROVAL
    trace_types = [e.event_type for e in store.get_trace(run.run_id)]
    assert "MODEL_TIMEOUT" in trace_types
    assert "MODEL_RETRY_SCHEDULED" in trace_types

    # Part B: Persistent timeout exhausts max_attempts (3) and fails with MODEL_TIMEOUT
    def persistent_timeout_script(
        agent: AgentDefinition, history: List[Dict[str, Any]], call_count: int
    ) -> ModelResponse:
        raise ModelTimeoutError("Persistent provider outage")

    runner_fail = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(persistent_timeout_script),
        tool_registry=registry,
    )
    runner_fail.register_agent(create_research_agent_definition())
    run_fail, _ = runner_fail.submit_task(
        TaskSubmission(
            idempotency_key="task-timeout-exhausted",
            agent_name="research-and-reporting-agent",
            goal="Fail cleanly after max model retries",
        )
    )
    failed_run = runner_fail.execute_run(run_fail.run_id)
    assert failed_run.status == RunStatus.FAILED
    assert failed_run.failure_category == FailureCategory.MODEL_TIMEOUT


def test_fault_2_tool_failure_retry_and_recovery() -> None:
    """Scenario 2: Tool failure — verify transient tool errors retry and recover cleanly."""
    side_effects: Dict[str, int] = {}
    store = SQLiteStateStore()
    registry = build_research_tool_registry(side_effects)

    attempts = {"search": 0}

    def flaky_search(query: str, max_results: int = 3) -> Dict[str, Any]:
        attempts["search"] += 1
        if attempts["search"] < 3:
            raise ToolExecutionError("Search index connection reset", transient=True)
        side_effects["search_tool"] = side_effects.get("search_tool", 0) + 1
        return {"query": query, "documents": ["Recovered Document A"]}

    registry.register(
        name="search_tool",
        description="Flaky search tool",
        args_schema=SearchToolArgs,
        handler=flaky_search,
        default_policy=PolicyDecision.ALLOW,
    )

    runner = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(default_research_agent_script),
        tool_registry=registry,
    )
    runner.register_agent(create_research_agent_definition())
    run, _ = runner.submit_task(
        TaskSubmission(
            idempotency_key="task-flaky-tool",
            agent_name="research-and-reporting-agent",
            goal="Test transient tool failure recovery",
        )
    )
    paused = runner.execute_run(run.run_id)
    assert paused.status == RunStatus.WAITING_FOR_APPROVAL
    assert attempts["search"] == 3
    assert side_effects["search_tool"] == 1


def test_fault_3_worker_interruption_idempotent_checkpoint_recovery() -> None:
    """Scenario 3: Worker interruption — crash after external tool completes, resume without duplicate execution."""
    side_effects: Dict[str, int] = {}
    store = SQLiteStateStore()
    registry = build_research_tool_registry(side_effects)
    runner = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(default_research_agent_script),
        tool_registry=registry,
    )
    runner.register_agent(create_research_agent_definition())

    run, _ = runner.submit_task(
        TaskSubmission(
            idempotency_key="task-crash-recovery",
            agent_name="research-and-reporting-agent",
            goal="Verify zero duplicate tool execution on worker crash recovery",
        )
    )

    # Simulate worker crash immediately AFTER search_tool completes externally
    with pytest.raises(WorkerInterruptedError):
        runner.execute_run(run.run_id, interrupt_after_tool_execution=True)

    # External side effect happened once before crash
    assert side_effects["search_tool"] == 1

    # Resume from persisted checkpoint; search_tool MUST NOT execute a second time!
    paused = runner.resume_from_checkpoint(run.run_id)
    assert paused.status == RunStatus.WAITING_FOR_APPROVAL
    assert side_effects["search_tool"] == 1  # Idempotency guaranteed!
    assert side_effects["summarize_tool"] == 1

    trace_events = [e.event_type for e in store.get_trace(run.run_id)]
    assert "WORKER_INTERRUPTED" in trace_events
    assert "WORKER_RECOVERY_FROM_CHECKPOINT" in trace_events
    assert "TOOL_IDEMPOTENCY_HIT" in trace_events


def test_fault_4_duplicate_task_submission() -> None:
    """Scenario 4: Duplicate task submission — submitting identical idempotency_key returns existing run."""
    side_effects: Dict[str, int] = {}
    store = SQLiteStateStore()
    registry = build_research_tool_registry(side_effects)
    runner = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(default_research_agent_script),
        tool_registry=registry,
    )
    runner.register_agent(create_research_agent_definition())

    sub1 = TaskSubmission(
        idempotency_key="dup-key-999",
        agent_name="research-and-reporting-agent",
        goal="First submission",
    )
    sub2 = TaskSubmission(
        idempotency_key="dup-key-999",
        agent_name="research-and-reporting-agent",
        goal="Accidental duplicate submission",
    )

    run1, created1 = runner.submit_task(sub1)
    run2, created2 = runner.submit_task(sub2)

    assert created1 is True
    assert created2 is False
    assert run1.run_id == run2.run_id

    trace_categories = [e.failure_category for e in store.get_trace(run1.run_id)]
    assert FailureCategory.DUPLICATE_SUBMISSION in trace_categories


def test_fault_5_invalid_tool_arguments() -> None:
    """Scenario 5: Invalid tool arguments — Pydantic schema validation rejects malformed model tool call."""
    side_effects: Dict[str, int] = {}
    store = SQLiteStateStore()
    registry = build_research_tool_registry(side_effects)

    def bad_args_script(
        agent: AgentDefinition, history: List[Dict[str, Any]], call_count: int
    ) -> ModelResponse:
        return ModelResponse(
            tool_call=ToolCallSpec(
                tool_name="search_tool",
                arguments={"query": "x", "max_results": 999},  # Violates min_length=2 and le=10
            )
        )

    runner = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(bad_args_script),
        tool_registry=registry,
    )
    runner.register_agent(create_research_agent_definition())
    run, _ = runner.submit_task(
        TaskSubmission(
            idempotency_key="task-invalid-args",
            agent_name="research-and-reporting-agent",
            goal="Trigger invalid tool arguments",
        )
    )
    failed = runner.execute_run(run.run_id)
    assert failed.status == RunStatus.FAILED
    assert failed.failure_category == FailureCategory.INVALID_TOOL_ARGUMENTS
    assert side_effects.get("search_tool", 0) == 0


def test_fault_6_approval_timeout() -> None:
    """Scenario 6: Approval timeout — human approval arriving after deadline transitions run to FAILED."""
    side_effects: Dict[str, int] = {}
    store = SQLiteStateStore()
    registry = build_research_tool_registry(side_effects)

    current_time = datetime(2026, 11, 12, 12, 0, 0, tzinfo=timezone.utc)

    def fake_clock() -> datetime:
        return current_time

    runner = AxiomRunner(
        store=store,
        model_adapter=ScriptedModelAdapter(default_research_agent_script),
        tool_registry=registry,
        clock=fake_clock,
    )
    runner.register_agent(create_research_agent_definition())

    run, _ = runner.submit_task(
        TaskSubmission(
            idempotency_key="task-approval-timeout",
            agent_name="research-and-reporting-agent",
            goal="Test approval expiration",
            approval_timeout_seconds=60,
        )
    )
    paused = runner.execute_run(run.run_id, approval_timeout_seconds=60)
    assert paused.status == RunStatus.WAITING_FOR_APPROVAL

    # Advance clock by 120 seconds (past 60s approval deadline)
    current_time = current_time + timedelta(seconds=120)

    expired_run = runner.approve_tool_call(paused.run_id, decided_by="late-operator")
    assert expired_run.status == RunStatus.FAILED
    assert expired_run.failure_category == FailureCategory.APPROVAL_TIMEOUT
    assert side_effects.get("publish_report_tool", 0) == 0
