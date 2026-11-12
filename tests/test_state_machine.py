"""Unit tests for ExecutionStateMachine, optimistic concurrency control, and SQLite/InMemory stores."""

import pytest

from axiom.engine.state_machine import (
    ExecutionStateMachine,
    InvalidStateTransitionError,
)
from axiom.schemas.domain import AgentRun, RunStatus
from axiom.state.store import (
    ConcurrencyConflictError,
    InMemoryStateStore,
    SQLiteStateStore,
)


def test_valid_state_transitions() -> None:
    ExecutionStateMachine.validate_transition(RunStatus.CREATED, RunStatus.RUNNING)
    ExecutionStateMachine.validate_transition(RunStatus.RUNNING, RunStatus.WAITING_FOR_TOOL)
    ExecutionStateMachine.validate_transition(RunStatus.WAITING_FOR_TOOL, RunStatus.WAITING_FOR_APPROVAL)
    ExecutionStateMachine.validate_transition(RunStatus.WAITING_FOR_APPROVAL, RunStatus.EXECUTING_TOOL)
    ExecutionStateMachine.validate_transition(RunStatus.EXECUTING_TOOL, RunStatus.CHECKPOINTED)
    ExecutionStateMachine.validate_transition(RunStatus.CHECKPOINTED, RunStatus.RUNNING)
    ExecutionStateMachine.validate_transition(RunStatus.RUNNING, RunStatus.COMPLETED)
    assert ExecutionStateMachine.is_terminal(RunStatus.COMPLETED)
    assert ExecutionStateMachine.is_terminal(RunStatus.FAILED)


def test_invalid_state_transition_raises() -> None:
    with pytest.raises(InvalidStateTransitionError):
        ExecutionStateMachine.validate_transition(RunStatus.CREATED, RunStatus.COMPLETED)

    with pytest.raises(InvalidStateTransitionError):
        ExecutionStateMachine.validate_transition(RunStatus.COMPLETED, RunStatus.RUNNING)


@pytest.mark.parametrize("store_cls", [InMemoryStateStore, SQLiteStateStore])
def test_optimistic_concurrency_control(store_cls) -> None:
    store = store_cls()
    run = AgentRun(
        idempotency_key="idem-lock-test",
        agent_name="test-agent",
        goal="Test optimistic lock",
    )
    created_run, is_new = store.create_or_get_run(run)
    assert is_new is True
    assert created_run.version == 1

    created_run.status = RunStatus.RUNNING
    updated_v2 = store.update_run(created_run, expected_version=1)
    assert updated_v2.version == 2

    # Stale update using version 1 must fail with ConcurrencyConflictError
    with pytest.raises(ConcurrencyConflictError):
        store.update_run(created_run, expected_version=1)
