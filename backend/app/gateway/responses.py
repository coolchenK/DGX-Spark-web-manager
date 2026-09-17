"""OpenAI Responses API bridge.

The DGX Spark gateway exposes runtimes over the OpenAI *chat/completions*
contract.  Agent clients such as Codex CLI >= 0.4x speak only the OpenAI
*Responses* contract, so this module translates between the two.

The translation is deliberately stateless: a Responses request is flattened
into a single chat/completions request, and the upstream chat stream is
re-expressed as Responses SSE events.  `store` is not honoured because the
gateway keeps no conversation state; clients always resend full history.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

RESPONSES_ENDPOINT = "/v1/responses"

# Reasoning vocabulary the gateway accepts on the wire.  Runtime backends are
# stricter than the gateway's own validator, so callers are normalised here
# rather than relying on upstream error messages.
_CODE_INPUT_TYPES = {"input_text", "text"}
_CODE_IMAGE_TYPES = {"input_image", "image_url"}
_CODE_AUDIO_TYPES = {"input_audio", "audio_url"}
_CODE_VIDEO_TYPES = {"input_video", "video_url"}


class ResponsesTranslationError(ValueError):
    """Raised when a Responses payload cannot be mapped onto chat/completions."""


# ---------------------------------------------------------------------------
# Reasoning-effort policy
# ---------------------------------------------------------------------------

# Runtimes whose chat template exposes a boolean thinking switch instead of a
# graded effort scale.  Detected from the deployment's saved template kwargs.
_TOGGLE_TEMPLATE_KEYS = {"enable_thinking"}

# vLLM/SGLang Qwen templates accept exactly these effort levels.
_QWEN_SUPPORTED_EFFORTS = {"low", "medium", "xhigh"}

# Codex emits `none` when thinking is switched off, and older/newer clients may
# still send the values the gateway historically accepted.
_QWEN_EFFORT_ALIASES = {
    "none": "none",
    "off": "none",
    "false": "none",
    "disabled": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
    "ultra": "xhigh",
    "ultracode": "xhigh",
    "extreme": "xhigh",
}


def uses_thinking_toggle(template_kwargs: Mapping[str, Any] | None) -> bool:
    """Return True when the deployment toggles thinking with a boolean flag."""
    if not isinstance(template_kwargs, Mapping):
        return False
    return bool(_TOGGLE_TEMPLATE_KEYS.intersection(template_kwargs))


def resolve_reasoning(
    effort: Any,
    *,
    thinking_toggle: bool,
) -> tuple[str | None, bool | None]:
    """Map a Responses reasoning effort onto gateway template controls.

    Returns ``(reasoning_effort, enable_thinking)``.  Toggle-style models expose
    only open/closed, so every active level collapses onto thinking-on and only
    an explicit "none" turns it off.  Graded models use the effort scale, with
    unsupported levels downgraded to the nearest supported level so the request
    never reaches a runtime that would reject it.
    """
    canonical = effort.strip().lower() if isinstance(effort, str) else None
    if canonical in {None, "", "default"}:
        return None, None

    if thinking_toggle:
        # MiniCPM-class models: on/off only, no intensity selection.
        return None, canonical not in {"none", "off", "false", "disabled"}

    resolved = _QWEN_EFFORT_ALIASES.get(canonical)
    if resolved is None:
        # Unknown level: leave it to the gateway rather than guessing.
        return canonical, None
    if resolved == "none":
        return "none", False
    if resolved not in _QWEN_SUPPORTED_EFFORTS:
        resolved = "medium"
    return resolved, None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _flatten_namespace_tools(namespace: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a Codex `namespace` tool into individually named functions.

    Codex groups related agent tools under a namespace.  chat/completions has no
    namespacing concept, so nested tools are exposed under a compound name and
    the original name is recorded for round-tripping.
    """
    prefix = str(namespace.get("name") or "").strip()
    nested = namespace.get("tools")
    if not prefix or not isinstance(nested, list):
        return []
    flattened: list[dict[str, Any]] = []
    for entry in nested:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("type") or "") != "function":
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        flattened.append(
            {
                "type": "function",
                "function": {
                    "name": f"{prefix}__{name}",
                    "description": entry.get("description") or "",
                    "parameters": entry.get("parameters")
                    or {"type": "object", "properties": {}},
                },
                # Remember the original identity so tool calls can be mapped
                # back to what the client actually asked for.
                "x-dgx-namespace": prefix,
                "x-dgx-name": name,
            }
        )
    return flattened


def responses_tools_to_chat(tools: Any) -> list[dict[str, Any]]:
    """Convert Responses tool definitions into chat/completions functions."""
    if not isinstance(tools, list):
        return []
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        tool_type = str(tool.get("type") or "")
        if tool_type == "function":
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description") or "",
                        "parameters": tool.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                    "x-dgx-name": name,
                }
            )
        elif tool_type == "namespace":
            converted.extend(_flatten_namespace_tools(tool))
        elif tool_type in {"web_search", "file_search", "computer_use_preview"}:
            # Server-side tools have no local runtime equivalent; Codex already
            # sends `external_web_access: false` for the offline case.  Expose a
            # descriptive stub so the model can still signal intent instead of
            # silently losing the capability.
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_type,
                        "description": (
                            f"Server-side '{tool_type}' is unavailable on the DGX Spark "
                            "gateway; no results can be returned."
                        ),
                        "parameters": {"type": "object", "properties": {}},
                    },
                    "x-dgx-name": tool_type,
                }
            )
    return converted


def chat_tool_name_to_client_name(chat_name: str) -> str:
    """Strip the namespace prefix added when flattening Codex namespace tools."""
    return chat_name.split("__", 1)[1] if "__" in chat_name else chat_name


def strip_gateway_tool_extensions(tools: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop bookkeeping keys before forwarding tools upstream."""
    cleaned: list[dict[str, Any]] = []
    for tool in tools:
        cleaned.append({k: v for k, v in tool.items() if not k.startswith("x-dgx-")})
    return cleaned


def responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    if isinstance(tool_choice, Mapping):
        if str(tool_choice.get("type") or "") == "function":
            name = tool_choice.get("name")
            if isinstance(name, str) and name:
                return {"type": "function", "function": {"name": name}}
    if tool_choice in {"auto", "none", "required"}:
        return tool_choice
    return None


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _content_to_chat_parts(content: Any) -> tuple[str | list[dict[str, Any]], set[str]]:
    """Convert Responses content items into chat content, tracking modalities."""
    if isinstance(content, str):
        return content, set()
    if not isinstance(content, list):
        return "", set()

    parts: list[dict[str, Any]] = []
    modalities: set[str] = set()
    for item in content:
        if isinstance(item, str):
            parts.append({"type": "text", "text": item})
            continue
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "")
        if item_type in _CODE_INPUT_TYPES:
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append({"type": "text", "text": text})
        elif item_type in _CODE_IMAGE_TYPES:
            url = item.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url")
            if isinstance(url, str) and url:
                modalities.add("image")
                parts.append({"type": "image_url", "image_url": {"url": url}})
        elif item_type in _CODE_VIDEO_TYPES:
            url = item.get("video_url")
            if isinstance(url, Mapping):
                url = url.get("url")
            if isinstance(url, str) and url:
                modalities.add("video")
                parts.append({"type": "video_url", "video_url": {"url": url}})
        elif item_type in _CODE_AUDIO_TYPES:
            # Audio is not exposed by the managed runtimes.
            continue

    if not any(part.get("type") != "text" for part in parts):
        # Pure text: collapse to a plain string for maximum template compatibility.
        return "\n".join(part["text"] for part in parts), modalities
    return parts, modalities


def _flatten_function_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        chunks: list[str] = []
        for entry in output:
            if isinstance(entry, Mapping):
                text = entry.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(entry, str):
                chunks.append(entry)
        return "\n".join(chunks)
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False)


def _join_system_text(chunks: list[str]) -> str:
    return "\n\n".join(chunk for chunk in chunks if chunk)


def _tool_arguments_payload(value: Any) -> str:
    """Return `value` as a JSON document suitable for `tool_calls.arguments`.

    Chat templates parse this field with a JSON decoder, so a freeform payload
    (Codex's `custom_tool_call` sends raw code in `input`) must be carried as a
    JSON string rather than passed through verbatim.
    """
    if isinstance(value, str):
        try:
            json.loads(value)
        except (json.JSONDecodeError, ValueError, TypeError):
            return json.dumps({"input": value}, ensure_ascii=False)
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def responses_input_to_chat_messages(
    input_value: Any,
    *,
    instructions: Any = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Translate a Responses `input` payload into chat messages.

    Every system-class instruction is accumulated and emitted as a single
    leading system message.  Qwen chat templates reject requests whose system
    message is not first, and Responses input routinely carries both
    `instructions` and `developer` items.
    """
    conversation: list[dict[str, Any]] = []
    system_chunks: list[str] = []
    modalities: set[str] = set()

    if isinstance(instructions, str) and instructions.strip():
        system_chunks.append(instructions)

    if isinstance(input_value, str):
        conversation.append({"role": "user", "content": input_value})
        return _prepend_system(system_chunks, conversation), modalities

    if not isinstance(input_value, list):
        return _prepend_system(system_chunks, conversation), modalities

    for item in input_value:
        if isinstance(item, str):
            conversation.append({"role": "user", "content": item})
            continue
        if not isinstance(item, Mapping):
            continue

        item_type = str(item.get("type") or "message")
        if item_type in {"function_call", "custom_tool_call"}:
            name = item.get("name")
            arguments = item.get("arguments")
            if arguments is None:
                arguments = item.get("input")
            conversation.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": str(item.get("call_id") or item.get("id") or "call_0"),
                            "type": "function",
                            "function": {
                                "name": str(name or ""),
                                "arguments": _tool_arguments_payload(arguments),
                            },
                        }
                    ],
                }
            )
        elif item_type in {"function_call_output", "custom_tool_call_output"}:
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or "call_0"),
                    "content": _flatten_function_output(item.get("output")),
                }
            )
        elif item_type in {"reasoning", "reasoning_summary"}:
            # Prior-turn reasoning is not replayed upstream.
            continue
        else:
            role = str(item.get("role") or "user")
            content, item_modalities = _content_to_chat_parts(item.get("content"))
            modalities |= item_modalities
            if content in ("", []) and not item_modalities:
                continue
            if role in {"developer", "system"}:
                # Managed runtimes only understand the classic role set, and the
                # template requires exactly one leading system message.
                if isinstance(content, str):
                    system_chunks.append(content)
                else:
                    system_chunks.append(
                        "".join(
                            str(part.get("text") or "")
                            for part in content
                            if isinstance(part, Mapping)
                        )
                    )
                continue
            conversation.append({"role": role, "content": content})

    return _prepend_system(system_chunks, conversation), modalities


def _prepend_system(
    system_chunks: list[str], conversation: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    system_text = _join_system_text(system_chunks)
    if not system_text:
        return conversation
    return [{"role": "system", "content": system_text}, *conversation]


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def responses_to_chat_request(
    body: Mapping[str, Any],
    *,
    model: str,
    thinking_toggle: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a chat/completions body from a Responses request.

    Returns the upstream request plus a small context dict describing how to
    translate the response back.
    """
    messages, modalities = responses_input_to_chat_messages(
        body.get("input"),
        instructions=body.get("instructions"),
    )
    if not messages:
        raise ResponsesTranslationError("The input field must contain at least one message")

    chat: dict[str, Any] = {"model": model, "messages": messages}

    tools = responses_tools_to_chat(body.get("tools"))
    if tools:
        chat["tools"] = strip_gateway_tool_extensions(tools)
        tool_choice = responses_tool_choice_to_chat(body.get("tool_choice"))
        if tool_choice is not None:
            chat["tool_choice"] = tool_choice
        if isinstance(body.get("parallel_tool_calls"), bool):
            chat["parallel_tool_calls"] = body["parallel_tool_calls"]

    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    resolved_effort, enable_thinking = resolve_reasoning(
        effort, thinking_toggle=thinking_toggle
    )
    template_kwargs: dict[str, Any] = {}
    if resolved_effort is not None:
        chat["reasoning_effort"] = resolved_effort
        template_kwargs["reasoning_effort"] = resolved_effort
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    if template_kwargs:
        chat["chat_template_kwargs"] = template_kwargs

    max_output = body.get("max_output_tokens")
    if isinstance(max_output, int) and not isinstance(max_output, bool) and max_output > 0:
        chat["max_tokens"] = max_output

    chat["stream"] = bool(body.get("stream"))
    if chat["stream"]:
        # Streaming runtimes only emit a usage block when explicitly asked, and
        # agent clients rely on it for context accounting.
        chat["stream_options"] = {"include_usage": True}

    context = {
        "model": model,
        "requested_model": body.get("model"),
        "modalities": modalities,
        "tools": tools,
        "stream": chat["stream"],
        "reasoning_effort": resolved_effort,
        "enable_thinking": enable_thinking,
    }
    return chat, context


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _reasoning_summary_part(text: str = "") -> dict[str, Any]:
    return {"type": "summary_text", "text": text}


def _message_item(text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "message",
        "id": _new_id("msg"),
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call_item(
    *, call_id: str, name: str, arguments: str, status: str = "completed"
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": _new_id("fc"),
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


class ResponsesTurnTranslator:
    """Turns one buffered chat/completions result into a Responses object."""

    def __init__(self, *, model: str, context: Mapping[str, Any] | None = None) -> None:
        self.model = model
        self.context = context or {}

    def _envelope(self, *, status: str, output: list[dict[str, Any]], usage: Any) -> dict[str, Any]:
        return {
            "id": _new_id("resp"),
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": normalize_usage(usage),
        }

    def from_chat_response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        choices = payload.get("choices")
        message: Mapping[str, Any] = {}
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                candidate = first.get("message")
                if isinstance(candidate, Mapping):
                    message = candidate

        output: list[dict[str, Any]] = []
        reasoning = extract_reasoning_text(message)
        if reasoning:
            output.append(
                {
                    "type": "reasoning",
                    "id": _new_id("rs"),
                    "summary": [_reasoning_summary_part(reasoning)],
                    "content": [],
                }
            )

        text = message.get("content")
        if isinstance(text, list):
            text = "".join(
                str(part.get("text") or "")
                for part in text
                if isinstance(part, Mapping)
            )
        if isinstance(text, str) and text:
            output.append(_message_item(text))

        for call in _iter_tool_calls(message):
            output.append(
                _function_call_item(
                    call_id=call["id"],
                    name=call["name"],
                    arguments=call["arguments"],
                )
            )

        finish = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish = str(choices[0].get("finish_reason") or "")
        status = "incomplete" if finish == "length" else "completed"
        return self._envelope(status=status, output=output, usage=payload.get("usage"))


def normalize_usage(usage: Any) -> dict[str, Any]:
    """Map chat usage onto the Responses usage object."""
    if not isinstance(usage, Mapping):
        usage = {}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    details = usage.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(details, Mapping):
        candidate = details.get("reasoning_tokens")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            reasoning_tokens = candidate
    output_details: dict[str, Any] = {"reasoning_tokens": reasoning_tokens}
    if isinstance(details, Mapping) and isinstance(details.get("accepted_prediction_tokens"), int):
        output_details["accepted_prediction_tokens"] = details["accepted_prediction_tokens"]
    return {
        "input_tokens": int(prompt) if isinstance(prompt, (int, float)) else 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": int(completion) if isinstance(completion, (int, float)) else 0,
        "output_tokens_details": output_details,
        "total_tokens": int(prompt or 0) + int(completion or 0),
    }


# Reasoning arrives under different keys depending on the runtime: vLLM uses
# `reasoning`, SGLang uses `reasoning_content`.
_REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_text")


def extract_reasoning_text(message: Mapping[str, Any]) -> str:
    for key in _REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            joined = "".join(
                str(part.get("text") or "")
                for part in value
                if isinstance(part, Mapping)
            )
            if joined:
                return joined
    return ""


def _iter_tool_calls(message: Mapping[str, Any]) -> Iterable[dict[str, str]]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    resolved: list[dict[str, str]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name") or "")
        if not name:
            continue
        arguments = function.get("arguments")
        resolved.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "name": chat_tool_name_to_client_name(name),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(
                    arguments or {}, ensure_ascii=False
                ),
            }
        )
    return resolved


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def sse_event(event_type: str, payload: Mapping[str, Any]) -> bytes:
    """Encode one Responses SSE frame.

    The event name is emitted on the `event:` line and repeated inside the JSON
    payload, matching the OpenAI Responses stream format that agent clients
    parse.
    """
    body = dict(payload)
    body.setdefault("type", event_type)
    return (
        f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"
    ).encode()
