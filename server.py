"""Standalone FastAPI Server Entrypoint for Axiom Runtime (Port 8001)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn
from fastapi.responses import RedirectResponse
from axiom.api.app import create_fastapi_app
from axiom.demo.research_agent import (
    build_research_tool_registry,
    create_research_agent_definition,
    default_research_agent_script,
)
from axiom.engine.runner import AxiomRunner
from axiom.gateway.adapter import ScriptedModelAdapter
from axiom.state.store import SQLiteStateStore

side_effects = {}
runner = AxiomRunner(
    store=SQLiteStateStore(),
    model_adapter=ScriptedModelAdapter(default_research_agent_script),
    tool_registry=build_research_tool_registry(side_effects),
)
runner.register_agent(create_research_agent_definition())

app = create_fastapi_app(runner)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect bare root URL to interactive Swagger API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {
        "service": "axiom-runtime",
        "status": "healthy",
        "registered_agents": ["research-and-reporting-agent"],
    }


if __name__ == "__main__":
    import os
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=8001, log_level="info")
