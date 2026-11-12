"""Deterministic execution state machine for Axiom Runtime."""

from __future__ import annotations

from typing import Dict, Set

from axiom.schemas.domain import RunStatus


class InvalidStateTransitionError(Exception):
    """Raised when an illegal RunStatus transition is attempted."""

    def __init__(self, current_status: RunStatus, target_status: RunStatus) -> None:
        self.current_status = current_status
        self.target_status = target_status
        super().__init__(
            f"Invalid state transition from {current_status.value} to {target_status.value}"
        )


ALLOWED_TRANSITIONS: Dict[RunStatus, Set[RunStatus]] = {
    RunStatus.CREATED: {
        RunStatus.RUNNING,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
    },
    RunStatus.RUNNING: {
        RunStatus.WAITING_FOR_TOOL,
        RunStatus.CHECKPOINTED,
        RunStatus.RETRYING,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.WAITING_FOR_TOOL: {
        RunStatus.WAITING_FOR_APPROVAL,
        RunStatus.EXECUTING_TOOL,
        RunStatus.RETRYING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.WAITING_FOR_APPROVAL: {
        RunStatus.EXECUTING_TOOL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.EXECUTING_TOOL: {
        RunStatus.CHECKPOINTED,
        RunStatus.RETRYING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.CHECKPOINTED: {
        RunStatus.RUNNING,
        RunStatus.EXECUTING_TOOL,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.RETRYING: {
        RunStatus.RUNNING,
        RunStatus.EXECUTING_TOOL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}

TERMINAL_STATES: Set[RunStatus] = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


class ExecutionStateMachine:
    """Validates and executes state transitions on an AgentRun."""

    @staticmethod
    def can_transition(current: RunStatus, target: RunStatus) -> bool:
        return target in ALLOWED_TRANSITIONS.get(current, set())

    @staticmethod
    def validate_transition(current: RunStatus, target: RunStatus) -> None:
        if not ExecutionStateMachine.can_transition(current, target):
            raise InvalidStateTransitionError(current, target)

    @staticmethod
    def is_terminal(status: RunStatus) -> bool:
        return status in TERMINAL_STATES
