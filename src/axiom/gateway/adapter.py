"""LLM Gateway: Provider-agnostic ModelAdapter protocol and deterministic test/fault adapters."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol

from axiom.schemas.domain import AgentDefinition, ModelResponse


class ModelTimeoutError(Exception):
    """Raised when an upstream LLM provider call times out."""


class ModelAdapter(Protocol):
    """Protocol abstracting provider-specific LLM APIs (Gemini, OpenAI, Anthropic)."""

    def generate(
        self,
        agent: AgentDefinition,
        conversation_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
    ) -> ModelResponse:
        ...


class ScriptedModelAdapter:
    """Deterministic model adapter for reproducible workflows and fault-injection testing."""

    def __init__(
        self,
        responder: Callable[[AgentDefinition, List[Dict[str, Any]], int], ModelResponse],
    ) -> None:
        self._responder = responder
        self.call_count = 0

    def generate(
        self,
        agent: AgentDefinition,
        conversation_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
    ) -> ModelResponse:
        self.call_count += 1
        return self._responder(agent, conversation_history, self.call_count)
