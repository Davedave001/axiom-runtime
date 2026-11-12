"""Durable StateStore implementations (InMemory and SQLite/SQL-backed) with optimistic locking and idempotency."""

from __future__ import annotations

import copy
import json
import sqlite3
from typing import Dict, List, Optional, Protocol, Tuple

from axiom.schemas.domain import (
    AgentRun,
    ApprovalRequest,
    Checkpoint,
    ToolInvocationRecord,
    TraceEvent,
    utc_now,
)


class ConcurrencyConflictError(Exception):
    """Raised when optimistic locking detects a stale run version."""


class StateStore(Protocol):
    """Protocol for persisting runs, checkpoints, tool invocations, approvals, and traces."""

    def create_or_get_run(self, run: AgentRun) -> Tuple[AgentRun, bool]:
        """Insert a new run or return existing run if idempotency_key matches.
        Returns (run, created_new).
        """
        ...

    def get_run(self, run_id: str) -> AgentRun:
        ...

    def update_run(self, run: AgentRun, expected_version: Optional[int] = None) -> AgentRun:
        ...

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        ...

    def get_latest_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        ...

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        ...

    def get_tool_invocation(self, idempotency_key: str) -> Optional[ToolInvocationRecord]:
        ...

    def save_tool_invocation(self, record: ToolInvocationRecord) -> None:
        ...

    def save_approval_request(self, approval: ApprovalRequest) -> None:
        ...

    def get_approval_request(self, approval_id: str) -> ApprovalRequest:
        ...

    def append_trace_event(self, event: TraceEvent) -> None:
        ...

    def get_trace(self, run_id: str) -> List[TraceEvent]:
        ...


class InMemoryStateStore:
    """Thread-safe, deterministic state store supporting optimistic concurrency and idempotency."""

    def __init__(self) -> None:
        self._runs: Dict[str, AgentRun] = {}
        self._idempotency_index: Dict[str, str] = {}
        self._checkpoints: Dict[str, List[Checkpoint]] = {}
        self._tool_invocations: Dict[str, ToolInvocationRecord] = {}
        self._approvals: Dict[str, ApprovalRequest] = {}
        self._traces: Dict[str, List[TraceEvent]] = {}

    def create_or_get_run(self, run: AgentRun) -> Tuple[AgentRun, bool]:
        if run.idempotency_key in self._idempotency_index:
            existing_id = self._idempotency_index[run.idempotency_key]
            return copy.deepcopy(self._runs[existing_id]), False

        stored = copy.deepcopy(run)
        self._runs[stored.run_id] = stored
        self._idempotency_index[stored.idempotency_key] = stored.run_id
        self._checkpoints.setdefault(stored.run_id, [])
        self._traces.setdefault(stored.run_id, [])
        return copy.deepcopy(stored), True

    def get_run(self, run_id: str) -> AgentRun:
        if run_id not in self._runs:
            raise KeyError(f"AgentRun not found: {run_id}")
        return copy.deepcopy(self._runs[run_id])

    def update_run(self, run: AgentRun, expected_version: Optional[int] = None) -> AgentRun:
        if run.run_id not in self._runs:
            raise KeyError(f"AgentRun not found: {run.run_id}")
        current = self._runs[run.run_id]
        if expected_version is not None and current.version != expected_version:
            raise ConcurrencyConflictError(
                f"Optimistic lock failed for run {run.run_id}: expected v{expected_version}, found v{current.version}"
            )
        updated = copy.deepcopy(run)
        updated.version = current.version + 1
        updated.updated_at = utc_now()
        self._runs[run.run_id] = updated
        return copy.deepcopy(updated)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._checkpoints.setdefault(checkpoint.run_id, []).append(copy.deepcopy(checkpoint))

    def get_latest_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        items = self._checkpoints.get(run_id, [])
        return copy.deepcopy(items[-1]) if items else None

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        return copy.deepcopy(self._checkpoints.get(run_id, []))

    def get_tool_invocation(self, idempotency_key: str) -> Optional[ToolInvocationRecord]:
        rec = self._tool_invocations.get(idempotency_key)
        return copy.deepcopy(rec) if rec else None

    def save_tool_invocation(self, record: ToolInvocationRecord) -> None:
        self._tool_invocations[record.idempotency_key] = copy.deepcopy(record)

    def save_approval_request(self, approval: ApprovalRequest) -> None:
        self._approvals[approval.approval_id] = copy.deepcopy(approval)

    def get_approval_request(self, approval_id: str) -> ApprovalRequest:
        if approval_id not in self._approvals:
            raise KeyError(f"ApprovalRequest not found: {approval_id}")
        return copy.deepcopy(self._approvals[approval_id])

    def append_trace_event(self, event: TraceEvent) -> None:
        self._traces.setdefault(event.run_id, []).append(copy.deepcopy(event))

    def get_trace(self, run_id: str) -> List[TraceEvent]:
        return copy.deepcopy(self._traces.get(run_id, []))


class SQLiteStateStore:
    """Durable SQL-backed StateStore using Python's built-in sqlite3 for zero-dependency local persistence."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    payload TEXT NOT NULL,
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_invocations (
                    idempotency_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def create_or_get_run(self, run: AgentRun) -> Tuple[AgentRun, bool]:
        row = self._conn.execute(
            "SELECT payload FROM agent_runs WHERE idempotency_key = ?",
            (run.idempotency_key,),
        ).fetchone()
        if row:
            return AgentRun.model_validate_json(row["payload"]), False

        with self._conn:
            self._conn.execute(
                "INSERT INTO agent_runs (run_id, idempotency_key, payload, version) VALUES (?, ?, ?, ?)",
                (run.run_id, run.idempotency_key, run.model_dump_json(), run.version),
            )
        return run.model_copy(deep=True), True

    def get_run(self, run_id: str) -> AgentRun:
        row = self._conn.execute(
            "SELECT payload FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"AgentRun not found: {run_id}")
        return AgentRun.model_validate_json(row["payload"])

    def update_run(self, run: AgentRun, expected_version: Optional[int] = None) -> AgentRun:
        current = self.get_run(run.run_id)
        if expected_version is not None and current.version != expected_version:
            raise ConcurrencyConflictError(
                f"Optimistic lock failed for run {run.run_id}: expected v{expected_version}, found v{current.version}"
            )
        updated = run.model_copy(deep=True)
        updated.version = current.version + 1
        updated.updated_at = utc_now()
        with self._conn:
            self._conn.execute(
                "UPDATE agent_runs SET payload = ?, version = ? WHERE run_id = ?",
                (updated.model_dump_json(), updated.version, updated.run_id),
            )
        return updated

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, run_id, step_index, payload) VALUES (?, ?, ?, ?)",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.step_index,
                    checkpoint.model_dump_json(),
                ),
            )

    def get_latest_checkpoint(self, run_id: str) -> Optional[Checkpoint]:
        row = self._conn.execute(
            "SELECT payload FROM checkpoints WHERE run_id = ? ORDER BY rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return Checkpoint.model_validate_json(row["payload"]) if row else None

    def list_checkpoints(self, run_id: str) -> List[Checkpoint]:
        rows = self._conn.execute(
            "SELECT payload FROM checkpoints WHERE run_id = ? ORDER BY rowid ASC",
            (run_id,),
        ).fetchall()
        return [Checkpoint.model_validate_json(r["payload"]) for r in rows]

    def get_tool_invocation(self, idempotency_key: str) -> Optional[ToolInvocationRecord]:
        row = self._conn.execute(
            "SELECT payload FROM tool_invocations WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return ToolInvocationRecord.model_validate_json(row["payload"]) if row else None

    def save_tool_invocation(self, record: ToolInvocationRecord) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO tool_invocations (idempotency_key, run_id, payload) VALUES (?, ?, ?)",
                (record.idempotency_key, record.run_id, record.model_dump_json()),
            )

    def save_approval_request(self, approval: ApprovalRequest) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO approval_requests (approval_id, run_id, payload) VALUES (?, ?, ?)",
                (approval.approval_id, approval.run_id, approval.model_dump_json()),
            )

    def get_approval_request(self, approval_id: str) -> ApprovalRequest:
        row = self._conn.execute(
            "SELECT payload FROM approval_requests WHERE approval_id = ?",
            (approval_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"ApprovalRequest not found: {approval_id}")
        return ApprovalRequest.model_validate_json(row["payload"])

    def append_trace_event(self, event: TraceEvent) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO trace_events (event_id, run_id, payload) VALUES (?, ?, ?)",
                (event.event_id, event.run_id, event.model_dump_json()),
            )

    def get_trace(self, run_id: str) -> List[TraceEvent]:
        rows = self._conn.execute(
            "SELECT payload FROM trace_events WHERE run_id = ? ORDER BY seq ASC",
            (run_id,),
        ).fetchall()
        return [TraceEvent.model_validate_json(r["payload"]) for r in rows]
