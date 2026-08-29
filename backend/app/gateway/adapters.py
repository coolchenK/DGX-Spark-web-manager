from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdaptedRequest:
    body: dict[str, Any]
    transformations: tuple[str, ...] = ()


class GatewayAdapter:
    """Translate the public OpenAI contract to a runtime's compatible subset."""

    def adapt_request(self, endpoint: str, body: dict[str, Any]) -> AdaptedRequest:
        return AdaptedRequest(body=dict(body))

    def normalize_error(self, content: bytes, *, status_code: int) -> bytes:
        try:
            payload = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            return content

        details = payload if isinstance(payload, dict) else {}
        message = details.get("message")
        if not isinstance(message, str) or not message:
            message = f"Upstream inference request failed with status {status_code}"
        param = details.get("param")
        code = details.get("code")
        normalized = {
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": param if isinstance(param, str) else None,
                "code": code if isinstance(code, str) else None,
            }
        }
        return json.dumps(normalized, ensure_ascii=False).encode("utf-8")


class LocalOpenAIAdapter(GatewayAdapter):
    """Adapter for local runtimes whose model chat templates use legacy roles."""

    def adapt_request(self, endpoint: str, body: dict[str, Any]) -> AdaptedRequest:
        adapted = dict(body)
        if endpoint != "/v1/chat/completions":
            return AdaptedRequest(body=adapted)
        messages = body.get("messages")
        if not isinstance(messages, list):
            return AdaptedRequest(body=adapted)

        adapted_messages: list[Any] = []
        translated = False
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "developer":
                adapted_messages.append({**message, "role": "system"})
                translated = True
            else:
                adapted_messages.append(message)
        adapted["messages"] = adapted_messages
        transformations = ("messages.role.developer_to_system",) if translated else ()
        return AdaptedRequest(body=adapted, transformations=transformations)


_LOCAL_OPENAI_ADAPTER = LocalOpenAIAdapter()


def adapter_for_runtime(runtime: str) -> GatewayAdapter:
    # Managed runtimes expose OpenAI-compatible HTTP endpoints, while role
    # validation remains controlled by each model's legacy chat template.
    if runtime in {"sglang", "vllm", "llama_cpp"}:
        return _LOCAL_OPENAI_ADAPTER
    return _LOCAL_OPENAI_ADAPTER
