"""Google Gemini REST API ModelAdapter for Axiom Runtime.

Uses standard library urllib.request so it works out-of-the-box without external SDK lock-in,
while supporting native function calling (MCP tool schemas) and timeout mapping.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from axiom.gateway.adapter import ModelTimeoutError
from axiom.schemas.domain import AgentDefinition, ModelResponse, ToolCallSpec


class GeminiModelAdapter:
    """Production ModelAdapter for Google Gemini (`generateContent` REST API)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 30.0,
        http_post: Optional[Callable[[str, Dict[str, Any], float], Dict[str, Any]]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or self._default_http_post

    def generate(
        self,
        agent: AgentDefinition,
        conversation_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
    ) -> ModelResponse:
        if not self.api_key and self._http_post == self._default_http_post:
            raise ValueError("GEMINI_API_KEY is required for live Gemini API calls")

        url = f"{self.base_url}/models/{agent.model_name}:generateContent?key={self.api_key}"
        payload = self._build_payload(agent, conversation_history, available_tools)

        start_ts = time.perf_counter()
        try:
            raw_response = self._http_post(url, payload, self.timeout_seconds)
        except TimeoutError as exc:
            raise ModelTimeoutError(f"Gemini request timed out after {self.timeout_seconds}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc).lower():
                raise ModelTimeoutError(f"Gemini request timed out: {exc}") from exc
            raise ModelTimeoutError(f"Gemini network error: {exc}") from exc

        latency_ms = int((time.perf_counter() - start_ts) * 1000)
        return self._parse_response(raw_response, latency_ms)

    @staticmethod
    def _build_payload(
        agent: AgentDefinition,
        conversation_history: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            if role == "user":
                contents.append({"role": "user", "parts": [{"text": str(msg.get("content", ""))}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": str(msg.get("content", ""))}]})
            elif role == "tool":
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.get("tool_name", "tool"),
                                    "response": {"result": msg.get("output")},
                                }
                            }
                        ],
                    }
                )

        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": agent.system_prompt}]},
            "contents": contents,
        }

        if available_tools:
            function_declarations = []
            for tool in available_tools:
                schema = dict(tool.get("inputSchema", {}))
                schema.pop("title", None)
                schema.pop("$schema", None)
                function_declarations.append(
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": schema,
                    }
                )
            payload["tools"] = [{"functionDeclarations": function_declarations}]

        return payload

    @staticmethod
    def _parse_response(data: Dict[str, Any], latency_ms: int) -> ModelResponse:
        usage = data.get("usageMetadata", {})
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        completion_tokens = int(usage.get("candidatesTokenCount", 0))

        candidates = data.get("candidates", [])
        if not candidates:
            return ModelResponse(
                final_answer="",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "functionCall" in part:
                fn = part["functionCall"]
                return ModelResponse(
                    tool_call=ToolCallSpec(
                        tool_name=fn["name"],
                        arguments=fn.get("args", {}),
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                )

        text_chunks = [p.get("text", "") for p in parts if "text" in p]
        return ModelResponse(
            final_answer="".join(text_chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _default_http_post(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
