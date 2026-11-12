"""Policy engine controlling tool permissions and human-in-the-loop approval gates."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from axiom.schemas.domain import AgentDefinition, PolicyDecision, ToolCallSpec
from axiom.tools.registry import ToolRegistry


class PolicyEngine:
    """Evaluates whether an agent is permitted to invoke a tool immediately, requires approval, or is denied."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._registry = tool_registry
        self._custom_rules: Dict[str, Callable[[ToolCallSpec], PolicyDecision]] = {}

    def add_rule(self, tool_name: str, rule: Callable[[ToolCallSpec], PolicyDecision]) -> None:
        self._custom_rules[tool_name] = rule

    def evaluate(
        self,
        agent: AgentDefinition,
        tool_call: ToolCallSpec,
    ) -> PolicyDecision:
        if agent.allowed_tools and tool_call.tool_name not in agent.allowed_tools:
            return PolicyDecision.DENY

        try:
            registered = self._registry.get(tool_call.tool_name)
        except KeyError:
            return PolicyDecision.DENY

        if tool_call.tool_name in self._custom_rules:
            return self._custom_rules[tool_call.tool_name](tool_call)

        return registered.default_policy
