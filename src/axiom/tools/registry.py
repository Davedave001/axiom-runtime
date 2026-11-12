"""Typed ToolRegistry with Pydantic schema validation and MCP descriptor export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from axiom.schemas.domain import PolicyDecision


class ToolExecutionError(Exception):
    """Raised when a tool fails during execution."""

    def __init__(self, message: str, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class InvalidToolArgumentsError(Exception):
    """Raised when tool arguments fail Pydantic schema validation."""


@dataclass
class RegisteredTool:
    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., Any]
    default_policy: PolicyDecision = PolicyDecision.ALLOW

    def to_mcp_descriptor(self) -> Dict[str, Any]:
        """Export tool definition in Model Context Protocol (MCP) JSON Schema format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.args_schema.model_json_schema(),
        }


class ToolRegistry:
    """Discovers, validates, and invokes permitted agent tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        handler: Callable[..., Any],
        default_policy: PolicyDecision = PolicyDecision.ALLOW,
    ) -> None:
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            args_schema=args_schema,
            handler=handler,
            default_policy=default_policy,
        )

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        return self._tools[name]

    def list_mcp_descriptors(self, allowed_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = (
            [self._tools[n] for n in allowed_names if n in self._tools]
            if allowed_names is not None
            else list(self._tools.values())
        )
        return [t.to_mcp_descriptor() for t in tools]

    def validate_arguments(self, tool_name: str, raw_arguments: Any) -> Dict[str, Any]:
        tool = self.get(tool_name)
        if not isinstance(raw_arguments, dict):
            raise InvalidToolArgumentsError(
                f"Arguments for tool '{tool_name}' must be a JSON object/dict, got {type(raw_arguments).__name__}"
            )
        try:
            validated = tool.args_schema.model_validate(raw_arguments)
            return validated.model_dump()
        except ValidationError as exc:
            raise InvalidToolArgumentsError(str(exc)) from exc

    def invoke(self, tool_name: str, validated_arguments: Dict[str, Any]) -> Any:
        tool = self.get(tool_name)
        return tool.handler(**validated_arguments)
