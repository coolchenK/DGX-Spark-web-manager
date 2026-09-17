from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import sessionmaker

from app.gateway.adapters import GatewayAdapter
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
SAMPLING_KEYS = (GENERATION_KEYS - {"max_tokens", "stop"}) | {"seed", "logit_bias"}
GENERATION_ENDPOINTS = {"/v1/chat/completions", "/v1/completions"}
STREAM_HEARTBEAT_SECONDS = 15.0


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
    # Sampling is owned by the deployment; missing settings use runtime defaults.
    for key in SAMPLING_KEYS:
        merged.pop(key, None)
    # SDKs normally flatten extra_body; also strip raw HTTP clients' nested values.
    if isinstance(merged.get("extra_body"), dict):
        merged["extra_body"] = {
            key: value for key, value in merged["extra_body"].items() if key not in SAMPLING_KEYS
        }
    applied: list[str] = []
    for key in sorted(GENERATION_KEYS & supported):
        if key not in defaults or key in merged:
            continue
        if key == "max_tokens" and "max_completion_tokens" in merged:
            continue
        merged[key] = defaults[key]
        applied.append(key)
    return merged, applied


def upstream_inference_timeout() -> httpx.Timeout:
    """Allow long local-agent turns while retaining bounded connection phases."""
    return httpx.Timeout(connect=5, read=1800, write=30, pool=5)


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


def extract_usage_from_json(content: bytes) -> dict[str, Any] | None:
    """Return the OpenAI usage mapping from a JSON response body, if present."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
        return parsed["usage"]
    return None


class UsageScanner:
    """Collect the last usage mapping from an SSE byte stream without altering it.

    The buffer only ever holds the unterminated tail of the stream, so memory
    stays bounded even for very long completions. Callers forward every chunk
    verbatim and read :attr:`usage` once the stream is complete.
    """

    MAX_BUFFER = 64 * 1024

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while True:
            positions = [
                (self._buffer.find(b"\n\n"), 2),
                (self._buffer.find(b"\r\n\r\n"), 4),
            ]
            positions = [(index, length) for index, length in positions if index >= 0]
            if not positions:
                break
            index, delimiter_length = min(positions)
            frame = self._buffer[:index]
            self._buffer = self._buffer[index + delimiter_length:]
            for line in frame.replace(b"\r\n", b"\n").split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                    self.usage = event["usage"]
        if len(self._buffer) > self.MAX_BUFFER:
            self._buffer = self._buffer[-self.MAX_BUFFER:]


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
    adapter: GatewayAdapter,
    on_finished: Callable[[], None] | None = None,
) -> Response:
    body["model"] = deployment.api_model_name
    started_at = time.perf_counter()
    url = f"{deployment.endpoint_url}{endpoint}"
    timeout = upstream_inference_timeout()
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

    if upstream.is_error:
        content = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        content = adapter.normalize_error(content, status_code=upstream.status_code)
        record_request_metric(
            request.app.state.database.session_factory,
            model=deployment.api_model_name,
            endpoint=endpoint,
            status_code=upstream.status_code,
            started_at=started_at,
        )
        if on_finished:
            on_finished()
        return Response(
            content=content,
            status_code=upstream.status_code,
            media_type="application/json",
        )

    if body.get("stream"):

        async def stream_body() -> AsyncIterator[bytes]:
            usage: dict[str, Any] | None = None
            saw_done = False
            pending = b""
            stream_tail = b""
            at_event_boundary = True
            upstream_chunks = upstream.aiter_raw()

            async def next_upstream_chunk() -> bytes:
                return await anext(upstream_chunks)

            chunk_task: asyncio.Task[bytes] | None = asyncio.create_task(next_upstream_chunk())
            try:
                while True:
                    done, _ = await asyncio.wait({chunk_task}, timeout=STREAM_HEARTBEAT_SECONDS)
                    if not done:
                        if at_event_boundary:
                            yield b": keepalive\n\n"
                        continue
                    try:
                        chunk = chunk_task.result()
                    except StopAsyncIteration:
                        chunk_task = None
                        break
                    chunk_task = asyncio.create_task(next_upstream_chunk())
                    stream_tail = (stream_tail + chunk)[-4:]
                    at_event_boundary = stream_tail.endswith((b"\n\n", b"\r\n\r\n"))
                    # httpx may coalesce several upstream SSE events into one raw
                    # network chunk. Split at SSE event boundaries before yielding;
                    # otherwise tool-call argument deltas arrive as one burst.
                    pending += chunk
                    while True:
                        positions = [(pending.find(b"\n\n"), 2), (pending.find(b"\r\n\r\n"), 4)]
                        positions = [(i, n) for i, n in positions if i >= 0]
                        if not positions:
                            break
                        index, delimiter_len = min(positions)
                        frame = pending[:index]
                        pending = pending[index + delimiter_len:]
                        frame_bytes = frame + (b"\r\n\r\n" if delimiter_len == 4 else b"\n\n")
                        for line in frame.replace(b"\r\n", b"\n").split(b"\n"):
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == b"[DONE]":
                                saw_done = True
                                continue
                            try:
                                event = json.loads(payload)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                            if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                                usage = event["usage"]
                        yield frame_bytes
            finally:
                if chunk_task is not None and not chunk_task.done():
                    chunk_task.cancel()
                    try:
                        await chunk_task
                    except asyncio.CancelledError:
                        pass
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
        headers.setdefault("Cache-Control", "no-cache")
        headers.setdefault("X-Accel-Buffering", "no")
        return StreamingResponse(stream_body(), status_code=upstream.status_code, headers=headers)

    content = await upstream.aread()
    await upstream.aclose()
    await client.aclose()
    usage = extract_usage_from_json(content)
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
