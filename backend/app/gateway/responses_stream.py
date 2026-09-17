"""Streaming translation from chat/completions SSE to Responses SSE.

`ResponsesStreamTranslator` consumes upstream `chat.completion.chunk` events and
emits the Responses event sequence agent clients expect:

    response.created
    response.in_progress
    [ reasoning item lifecycle + summary deltas ]
    [ message item lifecycle + output_text deltas ]
    [ function_call item lifecycle + argument deltas ]
    response.completed

The translator is incremental so deltas reach the client as they arrive, and it
tolerates upstreams that omit `[DONE]` or interleave reasoning and content.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from app.gateway.responses import (
    _reasoning_summary_part,
    chat_tool_name_to_client_name,
    extract_reasoning_text,
    normalize_usage,
    sse_event,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class _ItemState:
    """Lifecycle bookkeeping for one output item."""

    __slots__ = ("index", "item_id", "kind", "text", "call_id", "name", "arguments")

    def __init__(self, index: int, item_id: str, kind: str) -> None:
        self.index = index
        self.item_id = item_id
        self.kind = kind
        self.text = ""
        self.call_id = ""
        self.name = ""
        self.arguments = ""


class ResponsesStreamTranslator:
    """Incrementally converts chat chunks into Responses SSE frames."""

    def __init__(self, *, model: str, response_id: str | None = None) -> None:
        self.model = model
        self.response_id = response_id or _new_id("resp")
        self.created_at = int(time.time())
        self.sequence = 0
        self._next_index = 0
        self._reasoning: _ItemState | None = None
        self._message: _ItemState | None = None
        self._tool_calls: dict[int, _ItemState] = {}
        self._tool_order: list[int] = []
        self._usage: dict[str, Any] = {}
        self._finish_reason = ""
        self._completed = False

    # -- envelope helpers -------------------------------------------------

    def _base_response(
        self, *, status: str, output: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output if output is not None else [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": normalize_usage(self._usage) if status == "completed" else None,
        }

    def _emit(self, event_type: str, payload: dict[str, Any]) -> bytes:
        payload.setdefault("sequence_number", self.sequence)
        self.sequence += 1
        return sse_event(event_type, payload)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> list[bytes]:
        return [
            self._emit(
                "response.created", {"response": self._base_response(status="in_progress")}
            ),
            self._emit(
                "response.in_progress", {"response": self._base_response(status="in_progress")}
            ),
        ]

    def _ensure_reasoning(self) -> list[bytes]:
        if self._reasoning is not None:
            return []
        state = _ItemState(self._next_index, _new_id("rs"), "reasoning")
        self._next_index += 1
        self._reasoning = state
        return [
            self._emit(
                "response.output_item.added",
                {
                    "output_index": state.index,
                    "item": {
                        "type": "reasoning",
                        "id": state.item_id,
                        "summary": [],
                        "content": [],
                    },
                },
            ),
            self._emit(
                "response.reasoning_summary_part.added",
                {
                    "item_id": state.item_id,
                    "output_index": state.index,
                    "summary_index": 0,
                    "part": _reasoning_summary_part(),
                },
            ),
        ]

    def _ensure_message(self) -> list[bytes]:
        if self._message is not None:
            return []
        state = _ItemState(self._next_index, _new_id("msg"), "message")
        self._next_index += 1
        self._message = state
        return [
            self._emit(
                "response.output_item.added",
                {
                    "output_index": state.index,
                    "item": {
                        "type": "message",
                        "id": state.item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                },
            ),
            self._emit(
                "response.content_part.added",
                {
                    "item_id": state.item_id,
                    "output_index": state.index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            ),
        ]

    def _ensure_tool_call(self, key: int, *, call_id: str, name: str) -> list[bytes]:
        state = self._tool_calls.get(key)
        if state is not None:
            return []
        state = _ItemState(self._next_index, _new_id("fc"), "function_call")
        self._next_index += 1
        state.call_id = call_id or _new_id("call")
        state.name = chat_tool_name_to_client_name(name)
        self._tool_calls[key] = state
        self._tool_order.append(key)
        return [
            self._emit(
                "response.output_item.added",
                {
                    "output_index": state.index,
                    "item": {
                        "type": "function_call",
                        "id": state.item_id,
                        "call_id": state.call_id,
                        "name": state.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                },
            )
        ]

    # -- chunk handling ---------------------------------------------------

    def _handle_usage(self, usage: Any) -> None:
        if isinstance(usage, Mapping):
            self._usage = dict(usage)

    def feed(self, chunk: Mapping[str, Any]) -> list[bytes]:
        """Consume one chat.completion.chunk and return any Responses frames."""
        frames: list[bytes] = []
        self._handle_usage(chunk.get("usage"))

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return frames
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return frames

        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            reasoning = extract_reasoning_text(delta)
            if reasoning:
                frames.extend(self._ensure_reasoning())
                assert self._reasoning is not None
                self._reasoning.text += reasoning
                frames.append(
                    self._emit(
                        "response.reasoning_summary_text.delta",
                        {
                            "item_id": self._reasoning.item_id,
                            "output_index": self._reasoning.index,
                            "summary_index": 0,
                            "delta": reasoning,
                        },
                    )
                )

            content = delta.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, Mapping)
                )
            if isinstance(content, str) and content:
                frames.extend(self._ensure_message())
                assert self._message is not None
                self._message.text += content
                frames.append(
                    self._emit(
                        "response.output_text.delta",
                        {
                            "item_id": self._message.item_id,
                            "output_index": self._message.index,
                            "content_index": 0,
                            "delta": content,
                        },
                    )
                )

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    key = call.get("index")
                    if not isinstance(key, int):
                        key = len(self._tool_order)
                    function = call.get("function")
                    function = function if isinstance(function, Mapping) else {}
                    frames.extend(
                        self._ensure_tool_call(
                            key,
                            call_id=str(call.get("id") or ""),
                            name=str(function.get("name") or ""),
                        )
                    )
                    state = self._tool_calls[key]
                    if call.get("id") and not state.call_id:
                        state.call_id = str(call["id"])
                    name_piece = function.get("name")
                    if isinstance(name_piece, str) and name_piece and not state.name:
                        # Some runtimes stream the name incrementally.
                        state.name = chat_tool_name_to_client_name(name_piece)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        state.arguments += arguments
                        frames.append(
                            self._emit(
                                "response.function_call_arguments.delta",
                                {
                                    "item_id": state.item_id,
                                    "output_index": state.index,
                                    "delta": arguments,
                                },
                            )
                        )

        finish = choice.get("finish_reason")
        if isinstance(finish, str) and finish:
            self._finish_reason = finish
        return frames

    def finish(self) -> list[bytes]:
        """Close all open items and emit response.completed."""
        if self._completed:
            return []
        self._completed = True
        frames: list[bytes] = []
        output: list[dict[str, Any]] = []

        if self._reasoning is not None:
            state = self._reasoning
            frames.append(
                self._emit(
                    "response.reasoning_summary_text.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.index,
                        "summary_index": 0,
                        "text": state.text,
                    },
                )
            )
            frames.append(
                self._emit(
                    "response.reasoning_summary_part.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.index,
                        "summary_index": 0,
                        "part": _reasoning_summary_part(state.text),
                    },
                )
            )
            item = {
                "type": "reasoning",
                "id": state.item_id,
                "summary": [_reasoning_summary_part(state.text)],
                "content": [],
            }
            frames.append(
                self._emit(
                    "response.output_item.done",
                    {"output_index": state.index, "item": item},
                )
            )
            output.append(item)

        if self._message is not None:
            state = self._message
            frames.append(
                self._emit(
                    "response.output_text.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.index,
                        "content_index": 0,
                        "text": state.text,
                    },
                )
            )
            frames.append(
                self._emit(
                    "response.content_part.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.index,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": state.text,
                            "annotations": [],
                        },
                    },
                )
            )
            item = {
                "type": "message",
                "id": state.item_id,
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": state.text, "annotations": []}
                ],
            }
            frames.append(
                self._emit(
                    "response.output_item.done",
                    {"output_index": state.index, "item": item},
                )
            )
            output.append(item)

        for key in self._tool_order:
            state = self._tool_calls[key]
            frames.append(
                self._emit(
                    "response.function_call_arguments.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.index,
                        "arguments": state.arguments,
                    },
                )
            )
            item = {
                "type": "function_call",
                "id": state.item_id,
                "call_id": state.call_id,
                "name": state.name,
                "arguments": state.arguments,
                "status": "completed",
            }
            frames.append(
                self._emit(
                    "response.output_item.done",
                    {"output_index": state.index, "item": item},
                )
            )
            output.append(item)

        status = "incomplete" if self._finish_reason == "length" else "completed"
        frames.append(
            self._emit(
                "response.completed",
                {"response": self._base_response(status=status, output=output)},
            )
        )
        return frames

    @property
    def usage(self) -> dict[str, Any]:
        return self._usage


def parse_chat_sse_frames(buffer: bytes) -> tuple[list[dict[str, Any]], bytes]:
    """Split a byte buffer into parsed SSE data payloads plus a remainder.

    Returns the decoded JSON payloads for every complete `data:` event and the
    unparsed tail so callers can keep reading across network chunk boundaries.
    """
    events: list[dict[str, Any]] = []
    while True:
        positions = [(buffer.find(b"\n\n"), 2), (buffer.find(b"\r\n\r\n"), 4)]
        positions = [(index, length) for index, length in positions if index >= 0]
        if not positions:
            break
        index, delimiter_len = min(positions)
        frame = buffer[:index]
        buffer = buffer[index + delimiter_len :]
        for line in frame.replace(b"\r\n", b"\n").split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    return events, buffer
