"""Control Plane Service & FastAPI HTTP Router for Axiom Runtime."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from axiom.engine.runner import AxiomRunner
from axiom.schemas.domain import AgentRun, Checkpoint, TaskSubmission, TraceEvent


class ApprovalDecisionInput(BaseModel):
    decided_by: str = "human-operator"
    reason: str = "Approved via Control Plane API"


class AxiomControlPlane:
    """Service layer exposing AxiomRunner operations to HTTP/FastAPI or CLI handlers."""

    def __init__(self, runner: AxiomRunner) -> None:
        self.runner = runner

    def create_run(self, submission: TaskSubmission) -> Dict[str, Any]:
        run, created = self.runner.submit_task(submission)
        return {
            "run": run.model_dump(mode="json"),
            "created": created,
        }

    def execute_run(self, run_id: str, approval_timeout_seconds: int = 300) -> AgentRun:
        return self.runner.execute_run(run_id, approval_timeout_seconds=approval_timeout_seconds)

    def approve_run(self, run_id: str, decision: ApprovalDecisionInput) -> AgentRun:
        return self.runner.approve_tool_call(
            run_id=run_id,
            decided_by=decision.decided_by,
            reason=decision.reason,
        )

    def resume_run(self, run_id: str) -> AgentRun:
        return self.runner.resume_from_checkpoint(run_id)

    def get_run(self, run_id: str) -> AgentRun:
        return self.runner.store.get_run(run_id)

    def get_trace(self, run_id: str) -> List[TraceEvent]:
        return self.runner.store.get_trace(run_id)

    def get_checkpoints(self, run_id: str) -> List[Checkpoint]:
        return self.runner.store.list_checkpoints(run_id)


def create_fastapi_app(runner: AxiomRunner) -> Any:
    """Build a FastAPI application wired to the given AxiomRunner instance."""
    from fastapi import FastAPI, HTTPException

    control = AxiomControlPlane(runner)
    app = FastAPI(
        title="Axiom Runtime Control Plane API",
        version="0.1.0",
        description="Stateful AI Agent Execution Engine HTTP API",
    )

    @app.post("/runs")
    def submit_run(submission: TaskSubmission) -> Dict[str, Any]:
        try:
            return control.create_run(submission)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/execute", response_model=AgentRun)
    def execute_run(run_id: str, approval_timeout_seconds: int = 300) -> AgentRun:
        try:
            return control.execute_run(run_id, approval_timeout_seconds=approval_timeout_seconds)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/approve", response_model=AgentRun)
    def approve_run(run_id: str, decision: ApprovalDecisionInput) -> AgentRun:
        try:
            return control.approve_run(run_id, decision)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/resume", response_model=AgentRun)
    def resume_run(run_id: str) -> AgentRun:
        try:
            return control.resume_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}", response_model=AgentRun)
    def get_run(run_id: str) -> AgentRun:
        try:
            return control.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/runs/{run_id}/trace", response_model=List[TraceEvent])
    def get_trace(run_id: str) -> List[TraceEvent]:
        return control.get_trace(run_id)

    @app.get("/runs/{run_id}/checkpoints", response_model=List[Checkpoint])
    def get_checkpoints(run_id: str) -> List[Checkpoint]:
        return control.get_checkpoints(run_id)

    return app
