from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

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
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                record_request_metric(
                    request.app.state.database.session_factory,
                    model=deployment.api_model_name,
                    endpoint=endpoint,
                    status_code=upstream.status_code,
                    started_at=started_at,
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

