"""First Working Demonstration: Research-and-Reporting Agent with Approval Gates and Fault Injection."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field

from axiom.schemas.domain import (
    AgentDefinition,
    ModelResponse,
    PolicyDecision,
    RetryPolicy,
    ToolCallSpec,
)
from axiom.tools.registry import ToolRegistry


class SearchToolArgs(BaseModel):
    query: str = Field(..., min_length=2, description="Search query string")
    max_results: int = Field(default=3, ge=1, le=10)


class SummarizeToolArgs(BaseModel):
    documents: List[str] = Field(..., min_length=1)
    focus: str = Field(default="key_findings")


class PublishReportToolArgs(BaseModel):
    title: str = Field(..., min_length=3)
    summary_body: str = Field(..., min_length=10)
    channel: str = Field(default="stakeholder-briefings")


def build_research_tool_registry(side_effect_counter: Dict[str, int]) -> ToolRegistry:
    """Create the tool registry for the Research-and-Reporting Agent demo."""
    registry = ToolRegistry()

    def search_handler(query: str, max_results: int = 3) -> Dict[str, Any]:
        side_effect_counter["search_tool"] = side_effect_counter.get("search_tool", 0) + 1
        return {
            "query": query,
            "documents": [
                f"Doc 1: Durable state prevents lost sessions for '{query}'.",
                f"Doc 2: Idempotency keys eliminate duplicate tool calls on worker recovery.",
                f"Doc 3: Human-in-the-loop approval gates secure external side effects.",
            ][:max_results],
        }

    def summarize_handler(documents: List[str], focus: str = "key_findings") -> Dict[str, Any]:
        side_effect_counter["summarize_tool"] = side_effect_counter.get("summarize_tool", 0) + 1
        combined = " | ".join(documents)
        return {
            "focus": focus,
            "synthesis": f"Synthesized ({len(documents)} docs): {combined}",
        }

    def publish_report_handler(title: str, summary_body: str, channel: str = "stakeholder-briefings") -> Dict[str, Any]:
        side_effect_counter["publish_report_tool"] = side_effect_counter.get("publish_report_tool", 0) + 1
        return {
            "status": "PUBLISHED",
            "title": title,
            "channel": channel,
        }

    registry.register(
        name="search_tool",
        description="Search technical corpus for relevant engineering documents.",
        args_schema=SearchToolArgs,
        handler=search_handler,
        default_policy=PolicyDecision.ALLOW,
    )
    registry.register(
        name="summarize_tool",
        description="Synthesize retrieved documents into structured findings.",
        args_schema=SummarizeToolArgs,
        handler=summarize_handler,
        default_policy=PolicyDecision.ALLOW,
    )
    registry.register(
        name="publish_report_tool",
        description="Publish final report externally (requires human approval).",
        args_schema=PublishReportToolArgs,
        handler=publish_report_handler,
        default_policy=PolicyDecision.REQUIRE_APPROVAL,
    )
    return registry


def create_research_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        name="research-and-reporting-agent",
        system_prompt=(
            "You are a production research agent. Search for documents, summarize findings, "
            "request approval to publish the external report, and return the final executive summary."
        ),
        allowed_tools=["search_tool", "summarize_tool", "publish_report_tool"],
        max_steps=10,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_ms=50, backoff_multiplier=2.0),
    )


def default_research_agent_script(
    agent: AgentDefinition,
    history: List[Dict[str, Any]],
    call_count: int,
) -> ModelResponse:
    """Deterministic multi-step trajectory for the Research-and-Reporting Agent."""
    tool_messages = [m for m in history if m.get("role") == "tool"]

    if len(tool_messages) == 0:
        return ModelResponse(
            tool_call=ToolCallSpec(
                tool_name="search_tool",
                arguments={"query": "agent runtime reliability", "max_results": 3},
            ),
            prompt_tokens=120,
            completion_tokens=35,
            latency_ms=85,
        )

    if len(tool_messages) == 1:
        docs = tool_messages[0]["output"]["documents"]
        return ModelResponse(
            tool_call=ToolCallSpec(
                tool_name="summarize_tool",
                arguments={"documents": docs, "focus": "durable_execution"},
            ),
            prompt_tokens=240,
            completion_tokens=45,
            latency_ms=95,
        )

    if len(tool_messages) == 2:
        synthesis = tool_messages[1]["output"]["synthesis"]
        return ModelResponse(
            tool_call=ToolCallSpec(
                tool_name="publish_report_tool",
                arguments={
                    "title": "Axiom Runtime Reliability Report",
                    "summary_body": synthesis,
                    "channel": "engineering-leadership",
                },
            ),
            prompt_tokens=310,
            completion_tokens=50,
            latency_ms=110,
        )

    published_info = tool_messages[2]["output"]
    return ModelResponse(
        final_answer=(
            f"Final Report Published to #{published_info['channel']}: "
            f"{published_info['title']}. All checkpoints and approvals verified."
        ),
        prompt_tokens=380,
        completion_tokens=60,
        latency_ms=90,
    )
