"""Typed domain schemas for Agent, Task, Run, Tool, Checkpoint, Policy, and Telemetry."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class RunStatus(str, Enum):
    """Lifecycle states of an AgentRun."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    CHECKPOINTED = "CHECKPOINTED"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class PolicyDecision(str, Enum):
    """Policy evaluation outcomes before invoking a tool."""

    ALLOW = "ALLOW"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"


class ToolInvocationStatus(str, Enum):
    """Execution status of a single tool invocation."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ApprovalStatus(str, Enum):
    """State of a human-in-the-loop approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class FailureCategory(str, Enum):
    """Canonical failure taxonomy for observability and evaluation."""

    NONE = "NONE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    TOOL_FAILURE = "TOOL_FAILURE"
    WORKER_INTERRUPTION = "WORKER_INTERRUPTION"
    DUPLICATE_SUBMISSION = "DUPLICATE_SUBMISSION"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    APPROVAL_TIMEOUT = "APPROVAL_TIMEOUT"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    POLICY_DENIED = "POLICY_DENIED"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"


class RetryPolicy(BaseModel):
    """Bounded retry configuration for transient model and tool errors."""

    max_attempts: int = Field(default=3, ge=1)
    initial_backoff_ms: int = Field(default=100, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class AgentDefinition(BaseModel):
    """Declarative specification of an AI Agent."""

    name: str
    system_prompt: str
    model_name: str = "gemini-2.5-pro"
    allowed_tools: List[str] = Field(default_factory=list)
    max_steps: int = Field(default=15, ge=1)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class TaskSubmission(BaseModel):
    """User or upstream request to start an agent run."""

    task_id: str = Field(default_factory=new_id)
    idempotency_key: str
    agent_name: str
    goal: str
    input_context: Dict[str, Any] = Field(default_factory=dict)
    approval_timeout_seconds: int = Field(default=300, ge=1)


class ToolCallSpec(BaseModel):
    """Model-requested tool invocation."""

    call_id: str = Field(default_factory=new_id)
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

    def compute_idempotency_key(self, run_id: str, step_index: int) -> str:
        """Compute deterministic idempotency key for crash-safe deduplication."""
        canonical_args = json.dumps(self.arguments, sort_keys=True, separators=(",", ":"))
        payload = f"{run_id}:{step_index}:{self.tool_name}:{canonical_args}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelResponse(BaseModel):
    """Normalized output from an LLM provider adapter."""

    final_answer: Optional[str] = None
    tool_call: Optional[ToolCallSpec] = None
    raw_tool_arguments: Optional[Any] = None
    reasoning: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


class ToolInvocationRecord(BaseModel):
    """Durable record of a tool invocation for idempotency and audit."""

    invocation_id: str = Field(default_factory=new_id)
    run_id: str
    step_index: int
    idempotency_key: str
    tool_name: str
    arguments: Dict[str, Any]
    status: ToolInvocationStatus = ToolInvocationStatus.PENDING
    attempt_count: int = 1
    result: Optional[Any] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class ApprovalRequest(BaseModel):
    """Human-in-the-loop gate created when PolicyDecision == REQUIRE_APPROVAL."""

    approval_id: str = Field(default_factory=new_id)
    run_id: str
    step_index: int
    tool_call: ToolCallSpec
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    reason: Optional[str] = None


class Checkpoint(BaseModel):
    """Serializable snapshot of an AgentRun at a step boundary."""

    checkpoint_id: str = Field(default_factory=new_id)
    run_id: str
    step_index: int
    status_at_checkpoint: RunStatus
    conversation_history: List[Dict[str, Any]]
    pending_tool_call: Optional[ToolCallSpec] = None
    pending_approval_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class TraceEvent(BaseModel):
    """Structured telemetry event emitted on every lifecycle transition or action."""

    event_id: str = Field(default_factory=new_id)
    run_id: str
    step_index: int
    event_type: str
    state_from: Optional[RunStatus] = None
    state_to: Optional[RunStatus] = None
    failure_category: FailureCategory = FailureCategory.NONE
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class AgentRun(BaseModel):
    """Primary aggregate root representing a stateful agent execution."""

    run_id: str = Field(default_factory=new_id)
    idempotency_key: str
    agent_name: str
    goal: str
    status: RunStatus = RunStatus.CREATED
    current_step: int = 0
    version: int = 1
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    pending_tool_call: Optional[ToolCallSpec] = None
    pending_approval_id: Optional[str] = None
    final_output: Optional[str] = None
    failure_category: FailureCategory = FailureCategory.NONE
    error_message: Optional[str] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
