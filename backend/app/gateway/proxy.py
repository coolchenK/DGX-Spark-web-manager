from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import uuid4

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import sessionmaker

from app.models import Deployment, RequestMetric

GENERATION_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "stop",
}
GENERATION_ENDPOINTS = {"/v1/chat/completions", "/v1/completions"}


def merge_generation_defaults(
    endpoint: str,
    body: dict[str, Any],
    defaults: dict[str, Any],
    *,
    supported: set[str],
) -> tuple[dict[str, Any], list[str]]:
    merged = dict(body)
    if endpoint not in GENERATION_ENDPOINTS:
        return merged, []
    applied: list[str] = []
    for key in sorted(GENERATION_KEYS & supported):
        if key not in defaults or key in merged:
            continue
        if key == "max_tokens" and "max_completion_tokens" in merged:
            continue
        merged[key] = defaults[key]
        applied.append(key)
    return merged, applied


def openai_error(message: str, *, status_code: int, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "param": "model" if status_code == 404 else None,
                "code": code,
            }
        },
    )


def record_request_metric(
    session_factory: sessionmaker,
    *,
    model: str,
    endpoint: str,
    status_code: int,
    started_at: float,
    usage: dict[str, Any] | None = None,
) -> None:
    with session_factory() as db:
        db.add(
            RequestMetric(
                model=model,
                endpoint=endpoint,
                status_code=status_code,
                latency_ms=(time.perf_counter() - started_at) * 1000,
                prompt_tokens=(usage or {}).get("prompt_tokens"),
                completion_tokens=(usage or {}).get("completion_tokens"),
            )
        )
        db.commit()


async def proxy_openai_request(
    request: Request,
    deployment: Deployment,
    endpoint: str,
    body: dict[str, Any],
    on_finished: Callable[[], None] | None = None,
) -> Response:
    body["model"] = deployment.api_model_name
    started_at = time.perf_counter()
    url = f"{deployment.endpoint_url}{endpoint}"
    timeout = httpx.Timeout(connect=5, read=600, write=30, pool=5)
    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    upstream_request = client.build_request("POST", url, json=body)
    try:
        upstream = await client.send(upstream_request, stream=bool(body.get("stream")))
    except httpx.HTTPError as exc:
        await client.aclose()
        record_request_metric(
            request.app.state.database.session_factory,
            model=deployment.api_model_name,
            endpoint=endpoint,
            status_code=502,
            started_at=started_at,
        )
        if on_finished:
            on_finished()
        return openai_error(f"Upstream inference service is unavailable: {exc}", status_code=502)

    if body.get("stream"):

        async def stream_body() -> AsyncIterator[bytes]:
            usage: dict[str, Any] | None = None
            saw_done = False
            pending = b""
            rewrite_pending = b""
            reasoning_buffer = ""
            promoted_tool_call = False
            promote_reasoning_to_content = any(
                isinstance(message, dict)
                and isinstance(message.get("content"), str)
                and "<tool_response>" in message["content"]
                for message in (body.get("messages") or [])
            )
            try:
                async for chunk in upstream.aiter_raw():
                    pending += chunk
                    while b"\n" in pending:
                        raw_line, pending = pending.split(b"\n", 1)
                        line = raw_line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            saw_done = True
                            continue
                        if not payload:
                            continue
                        try:
                            event = json.loads(payload)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                            usage = event["usage"]

                    # httpx can split an SSE line across arbitrary chunks. Keep
                    # a separate rewrite buffer so compatibility transforms see
                    # complete events instead of silently missing split lines.
                    rewrite_pending += chunk
                    complete_length = rewrite_pending.rfind(b"\n") + 1
                    if complete_length <= 0:
                        continue
                    chunk = rewrite_pending[:complete_length]
                    rewrite_pending = rewrite_pending[complete_length:]

                    # Alma's OpenCode Go path uses the OpenAI AI-SDK parser,
                    # which recognizes `reasoning_text`; vLLM/SGLang commonly
                    # emit `reasoning_content`. Preserve the native field and
                    # add the parser alias without changing final text chunks.
                    rewritten = chunk
                    try:
                        text = chunk.decode("utf-8")
                        lines: list[str] = []
                        changed = False
                        for raw_line in text.splitlines(keepends=True):
                            stripped = raw_line.strip()
                            if not stripped.startswith("data:"):
                                lines.append(raw_line)
                                continue
                            raw_payload = stripped[5:].strip()
                            if not raw_payload or raw_payload == "[DONE]":
                                lines.append(raw_line)
                                continue
                            event = json.loads(raw_payload)
                            for choice in event.get("choices") or []:
                                delta = choice.get("delta")
                                if isinstance(delta, dict):
                                    reasoning = (
                                        delta.get("reasoning_text")
                                        or delta.get("reasoning_content")
                                        or delta.get("reasoning")
                                    )
                                    if reasoning:
                                        if promote_reasoning_to_content and not delta.get(
                                            "content"
                                        ):
                                            delta["content"] = reasoning
                                            changed = True
                                        reasoning_buffer += reasoning
                                        # Alma's injected prompt asks Qwen to emit
                                        # JSON calls inside <tool_call> tags, but
                                        # the runtime can leave that markup in the
                                        # reasoning channel. Promote a complete tag
                                        # to an OpenAI tool call so Alma executes it.
                                        match = re.search(
                                            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                                            reasoning_buffer,
                                            re.DOTALL,
                                        )
                                        if match and not promoted_tool_call:
                                            try:
                                                parsed_call = json.loads(match.group(1))
                                                name = parsed_call.get("name")
                                                arguments = parsed_call.get("arguments")
                                                if isinstance(name, str) and isinstance(
                                                    arguments, dict
                                                ):
                                                    tool_call_id = (
                                                        f"chatcmpl-tool-{uuid4().hex[:16]}"
                                                    )
                                                    delta.clear()
                                                    delta["tool_calls"] = [
                                                        {
                                                            "index": 0,
                                                            "id": tool_call_id,
                                                            "type": "function",
                                                            "function": {
                                                                "name": name,
                                                                "arguments": json.dumps(
                                                                    arguments,
                                                                    ensure_ascii=False,
                                                                    separators=(",", ":"),
                                                                ),
                                                            },
                                                        }
                                                    ]
                                                    promoted_tool_call = True
                                                    changed = True
                                            except json.JSONDecodeError:
                                                pass
                                        if not promoted_tool_call and "reasoning_text" not in delta:
                                            delta["reasoning_text"] = reasoning
                                            changed = True
                                # Preserve OpenAI tool semantics. Alma/AI-SDK
                                # must see finish_reason=tool_calls so the tool
                                # step executes and the internal loop proceeds.
                                # The marker is diagnostic only and is ignored by
                                # standard clients.
                                if promoted_tool_call and choice.get("finish_reason") == "stop":
                                    choice["finish_reason"] = "tool_calls"
                                    choice["alma_promoted_tool_step"] = True
                                    changed = True
                                elif choice.get("finish_reason") == "tool_calls":
                                    choice["alma_tool_step"] = True
                                    changed = True
                            if changed:
                                ending = (
                                    "\r\n"
                                    if raw_line.endswith("\r\n")
                                    else "\n"
                                    if raw_line.endswith("\n")
                                    else ""
                                )
                                encoded_event = json.dumps(
                                    event,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                lines.append(f"data: {encoded_event}{ending}")
                            else:
                                lines.append(raw_line)
                        if changed:
                            rewritten = "".join(lines).encode("utf-8")
                    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                        pass
                    yield rewritten
                if rewrite_pending:
                    yield rewrite_pending
            finally:
                await upstream.aclose()
                await client.aclose()
                record_request_metric(
                    request.app.state.database.session_factory,
                    model=deployment.api_model_name,
                    endpoint=endpoint,
                    status_code=(
                        upstream.status_code if upstream.status_code >= 400 or saw_done else 499
                    ),
                    started_at=started_at,
                    usage=usage,
                )
                if on_finished:
                    on_finished()

        headers = {}
        if content_type := upstream.headers.get("content-type"):
            headers["content-type"] = content_type
        return StreamingResponse(stream_body(), status_code=upstream.status_code, headers=headers)

    content = await upstream.aread()
    await upstream.aclose()
    await client.aclose()
    usage: dict[str, Any] | None = None
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
            usage = parsed["usage"]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    record_request_metric(
        request.app.state.database.session_factory,
        model=deployment.api_model_name,
        endpoint=endpoint,
        status_code=upstream.status_code,
        started_at=started_at,
        usage=usage,
    )
    if on_finished:
        on_finished()
    return Response(
        content=content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
