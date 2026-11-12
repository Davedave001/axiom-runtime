"""Tests for GeminiModelAdapter and AxiomControlPlane API."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from axiom.api.app import ApprovalDecisionInput, AxiomControlPlane
from axiom.demo.research_agent import (
    build_research_tool_registry,
    create_research_agent_definition,
    default_research_agent_script,
)
from axiom.engine.runner import AxiomRunner
from axiom.gateway.adapter import ModelTimeoutError, ScriptedModelAdapter
from axiom.gateway.gemini import GeminiModelAdapter
from axiom.schemas.domain import RunStatus, TaskSubmission
from axiom.state.store import SQLiteStateStore


def test_gemini_adapter_function_call_and_text_parsing() -> None:
    """Verify GeminiModelAdapter serializes MCP tools and parses functionCall & text responses."""
    agent = create_research_agent_definition()
    captured_payloads = []

    def mock_http_post(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        captured_payloads.append(payload)
        if len(captured_payloads) == 1:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search_tool",
                                        "args": {"query": "durable state", "max_results": 2},
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 25},
            }
        return {
            "candidates": [
                {"content": {"parts": [{"text": "Synthesis complete: durable state verified."}]}}
            ],
            "usageMetadata": {"promptTokenCount": 180, "candidatesTokenCount": 40},
        }

    adapter = GeminiModelAdapter(api_key="test-key", http_post=mock_http_post)
    registry = build_research_tool_registry({})
    tools = registry.list_mcp_descriptors(agent.allowed_tools)

    # 1. Function call turn
    resp1 = adapter.generate(
        agent=agent,
        conversation_history=[{"role": "user", "content": "Search durable state"}],
        available_tools=tools,
    )
    assert resp1.tool_call is not None
    assert resp1.tool_call.tool_name == "search_tool"
    assert resp1.tool_call.arguments == {"query": "durable state", "max_results": 2}
    assert resp1.prompt_tokens == 100
    assert "tools" in captured_payloads[0]

    # 2. Final answer turn
    resp2 = adapter.generate(
        agent=agent,
        conversation_history=[
            {"role": "user", "content": "Search durable state"},
            {"role": "tool", "tool_name": "search_tool", "output": {"documents": ["Doc 1"]}},
        ],
        available_tools=tools,
    )
    assert resp2.tool_call is None
    assert resp2.final_answer == "Synthesis complete: durable state verified."


def test_gemini_adapter_timeout_mapping() -> None:
    agent = create_research_agent_definition()

    def timeout_http_post(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        raise TimeoutError("Socket read timed out")

    adapter = GeminiModelAdapter(api_key="test-key", http_post=timeout_http_post)
    with pytest.raises(ModelTimeoutError):
        adapter.generate(agent, [{"role": "user", "content": "hi"}], [])


def test_control_plane_end_to_end_workflow() -> None:
    side_effects: Dict[str, int] = {}
    runner = AxiomRunner(
        store=SQLiteStateStore(),
        model_adapter=ScriptedModelAdapter(default_research_agent_script),
        tool_registry=build_research_tool_registry(side_effects),
    )
    runner.register_agent(create_research_agent_definition())
    control = AxiomControlPlane(runner)

    res = control.create_run(
        TaskSubmission(
            idempotency_key="api-run-001",
            agent_name="research-and-reporting-agent",
            goal="Run via control plane",
        )
    )
    assert res["created"] is True
    run_id = res["run"]["run_id"]

    paused = control.execute_run(run_id)
    assert paused.status == RunStatus.WAITING_FOR_APPROVAL

    completed = control.approve_run(
        run_id,
        ApprovalDecisionInput(decided_by="ops-lead", reason="Approved via API"),
    )
    assert completed.status == RunStatus.COMPLETED
    assert len(control.get_trace(run_id)) >= 6
    assert len(control.get_checkpoints(run_id)) >= 4
