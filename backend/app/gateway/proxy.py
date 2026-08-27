from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)

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
ALMA_TASK_DESCRIPTION_MAX_LENGTH = 80
ALMA_EMPTY_CONTINUATION_RETRY_PROMPT = (
    "The preceding assistant generation was empty. Continue the original task now "
    "using the tool results already present in the conversation. If another tool is "
    "strictly necessary, call it immediately. Otherwise, return the complete final "
    "answer. Do not emit an empty response."
)
ALMA_EMPTY_CONTINUATION_FINAL_PROMPT = (
    "The model returned an empty continuation again. Do not call another tool. Return "
    "the best complete final answer now from the available tool results."
)
ALMA_EMPTY_CONTINUATION_FALLBACK = (
    "模型在工具执行后连续返回了空结果，网关已自动重试但仍未获得有效回答。"
    "请重新发送本次请求。"
)


def _is_alma_tool_continuation(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and (
            message.get("role") == "tool"
            or (
                isinstance(message.get("content"), str)
                and "<tool_response>" in message["content"]
            )
        )
        for message in messages
    )


def _prepare_alma_continuation_retry(
    body: dict[str, Any],
    *,
    final_answer: bool = False,
) -> dict[str, Any]:
    retry = dict(body)
    retry["stream"] = False
    retry.pop("stream_options", None)
    messages = retry.get("messages")
    retry["messages"] = [
        *(messages if isinstance(messages, list) else []),
        {
            "role": "user",
            "content": (
                ALMA_EMPTY_CONTINUATION_FINAL_PROMPT
                if final_answer
                else ALMA_EMPTY_CONTINUATION_RETRY_PROMPT
            ),
        },
    ]
    kwargs = retry.get("chat_template_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
    kwargs["enable_thinking"] = False
    retry["chat_template_kwargs"] = kwargs
    if final_answer:
        retry["tool_choice"] = "none"
    return retry


def _normalize_nonstream_chat_completion(payload: dict[str, Any]) -> bool:
    changed = False
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if normalize_alma_tool_call_arguments(message):
            changed = True
        reasoning = (
            message.get("reasoning_text")
            or message.get("reasoning_content")
            or message.get("reasoning")
        )
        if reasoning and "reasoning_text" not in message:
            message["reasoning_text"] = reasoning
            changed = True
        if message.get("tool_calls") and choice.get("finish_reason") == "stop":
            choice["finish_reason"] = "tool_calls"
            choice["alma_tool_step"] = True
            changed = True
    return changed


def _chat_completion_has_output(payload: dict[str, Any]) -> bool:
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if message.get("tool_calls"):
            return True
    return False


def _merge_usage(total: dict[str, int], payload: dict[str, Any]) -> None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _chat_completion_to_sse(payload: dict[str, Any]) -> bytes:
    common = {
        key: payload[key]
        for key in ("id", "created", "model", "system_fingerprint")
        if key in payload
    }
    common["object"] = "chat.completion.chunk"
    output_choices: list[dict[str, Any]] = []
    terminal_choices: list[dict[str, Any]] = []
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        delta = {
            key: value
            for key, value in message.items()
            if key
            in {
                "role",
                "content",
                "refusal",
                "tool_calls",
                "function_call",
                "reasoning",
                "reasoning_content",
                "reasoning_text",
            }
            and value is not None
        }
        delta.setdefault("role", "assistant")
        index = choice.get("index", len(output_choices))
        output_choices.append({"index": index, "delta": delta, "finish_reason": None})
        terminal_choices.append(
            {
                "index": index,
                "delta": {},
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        )
    events = [
        {**common, "choices": output_choices},
        {**common, "choices": terminal_choices, "usage": payload.get("usage")},
    ]
    rendered = "".join(
        f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
        for event in events
    )
    return f"{rendered}data: [DONE]\n\n".encode()


def normalize_alma_tool_call_arguments(delta: dict[str, Any]) -> bool:
    """Keep Alma Task calls within its client-side schema."""
    changed = False
    for tool_call in delta.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict) or function.get("name") != "Task":
            continue
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            continue
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            continue
        description = arguments.get("description") if isinstance(arguments, dict) else None
        if not isinstance(description, str) or len(description) <= ALMA_TASK_DESCRIPTION_MAX_LENGTH:
            continue
        arguments["description"] = description[:ALMA_TASK_DESCRIPTION_MAX_LENGTH]
        function["arguments"] = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        changed = True
    return changed


def _streamed_tool_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        tool_call
        for choice in event.get("choices") or []
        if isinstance(choice, dict)
        for tool_call in ((choice.get("delta") or {}).get("tool_calls") or [])
        if isinstance(tool_call, dict)
    ]


def _update_streamed_tool_call_states(
    event: dict[str, Any],
    states: dict[int, dict[str, Any]],
) -> bool:
    tool_calls = _streamed_tool_calls(event)
    for tool_call in tool_calls:
        index = tool_call.get("index", 0)
        state = states.setdefault(index, {"name": "", "arguments": [], "functions": []})
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            current_name = state["name"]
            if not current_name.endswith(name):
                state["name"] = current_name + name
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            state["arguments"].append(arguments)
            state["functions"].append(function)
    return bool(tool_calls)


def _normalize_streamed_task_calls(states: dict[int, dict[str, Any]]) -> bool:
    changed = False
    for state in states.values():
        if state.get("name") != "Task":
            continue
        try:
            arguments = json.loads("".join(state["arguments"]))
        except json.JSONDecodeError:
            continue
        description = arguments.get("description") if isinstance(arguments, dict) else None
        if not isinstance(description, str) or len(description) <= ALMA_TASK_DESCRIPTION_MAX_LENGTH:
            continue
        arguments["description"] = description[:ALMA_TASK_DESCRIPTION_MAX_LENGTH]
        functions = state["functions"]
        for function in functions:
            function["arguments"] = ""
        functions[0]["arguments"] = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        changed = True
    return changed


def _sse_line_ending(raw_line: str) -> str:
    if raw_line.endswith("\r\n"):
        return "\r\n"
    if raw_line.endswith("\n"):
        return "\n"
    return ""


def _encode_sse_event(event: dict[str, Any], raw_line: str) -> str:
    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}{_sse_line_ending(raw_line)}"


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


async def _buffered_alma_tool_continuation_response(
    *,
    request: Request,
    deployment: Deployment,
    endpoint: str,
    body: dict[str, Any],
    client: httpx.AsyncClient,
    upstream: httpx.Response,
    started_at: float,
    on_finished: Callable[[], None] | None,
) -> Response:
    current_body = dict(body)
    current_body["stream"] = False
    current_body.pop("stream_options", None)
    current_upstream = upstream
    total_usage: dict[str, int] = {}
    parsed: dict[str, Any] | None = None
    content = b""
    status_code = upstream.status_code
    media_type = upstream.headers.get("content-type", "application/json")

    for attempt in range(3):
        content = await current_upstream.aread()
        status_code = current_upstream.status_code
        media_type = current_upstream.headers.get("content-type", "application/json")
        await current_upstream.aclose()
        try:
            candidate = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            candidate = None
        parsed = candidate if isinstance(candidate, dict) else None
        if parsed is not None:
            _normalize_nonstream_chat_completion(parsed)
            _merge_usage(total_usage, parsed)
        if status_code >= 400 or (parsed is not None and _chat_completion_has_output(parsed)):
            break
        if attempt >= 2:
            break

        logger.warning(
            "Retrying empty Alma tool continuation model=%s attempt=%s",
            deployment.api_model_name,
            attempt + 1,
        )
        current_body = _prepare_alma_continuation_retry(
            current_body,
            final_answer=attempt == 1,
        )
        retry_request = client.build_request(
            "POST",
            f"{deployment.endpoint_url}{endpoint}",
            json=current_body,
        )
        try:
            current_upstream = await client.send(retry_request, stream=False)
        except httpx.HTTPError as exc:
            parsed = None
            status_code = 502
            content = json.dumps(
                {
                    "error": {
                        "message": f"Alma continuation retry failed: {exc}",
                        "type": "server_error",
                    }
                }
            ).encode()
            media_type = "application/json"
            break

    await client.aclose()
    if status_code >= 400 and content:
        response = Response(content=content, status_code=status_code, media_type=media_type)
    else:
        if parsed is None:
            parsed = {
                "id": f"chatcmpl-{uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": deployment.api_model_name,
                "choices": [],
            }
        if not _chat_completion_has_output(parsed):
            logger.error(
                "Alma tool continuation remained empty after retries model=%s",
                deployment.api_model_name,
            )
            parsed["choices"] = [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ALMA_EMPTY_CONTINUATION_FALLBACK,
                    },
                    "finish_reason": "stop",
                }
            ]
        response = Response(
            content=_chat_completion_to_sse(parsed),
            status_code=200,
            media_type="text/event-stream",
        )
    record_request_metric(
        request.app.state.database.session_factory,
        model=deployment.api_model_name,
        endpoint=endpoint,
        status_code=response.status_code,
        started_at=started_at,
        usage=total_usage or None,
    )
    if on_finished:
        on_finished()
    return response


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
    buffer_alma_continuation = (
        endpoint == "/v1/chat/completions"
        and bool(body.get("stream"))
        and _is_alma_tool_continuation(body)
    )
    upstream_body = dict(body)
    if buffer_alma_continuation:
        upstream_body["stream"] = False
        upstream_body.pop("stream_options", None)
    upstream_request = client.build_request("POST", url, json=upstream_body)
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

    if buffer_alma_continuation:
        return await _buffered_alma_tool_continuation_response(
            request=request,
            deployment=deployment,
            endpoint=endpoint,
            body=upstream_body,
            client=client,
            upstream=upstream,
            started_at=started_at,
            on_finished=on_finished,
        )

    if body.get("stream"):

        async def stream_body() -> AsyncIterator[bytes]:
            usage: dict[str, Any] | None = None
            saw_done = False
            rewrite_pending = b""
            reasoning_buffer = ""
            promoted_tool_call = False
            buffered_tool_lines: list[tuple[dict[str, Any] | None, str, bool]] = []
            tool_call_states: dict[int, dict[str, Any]] = {}
            buffering_tool_calls = False
            def flush_tool_lines() -> str:
                normalized = _normalize_streamed_task_calls(tool_call_states)
                rendered = "".join(
                    _encode_sse_event(event, raw_line)
                    if event is not None and (changed or normalized)
                    else raw_line
                    for event, raw_line, changed in buffered_tool_lines
                )
                buffered_tool_lines.clear()
                tool_call_states.clear()
                return rendered

            try:
                async for chunk in upstream.aiter_raw():
                    # httpx can split an SSE line across arbitrary chunks. Keep
                    # a buffer so compatibility transforms see complete events.
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
                    try:
                        text = chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        yield chunk
                        continue

                    output_lines: list[str] = []
                    for raw_line in text.splitlines(keepends=True):
                        stripped = raw_line.strip()
                        if not stripped.startswith("data:"):
                            if buffering_tool_calls:
                                buffered_tool_lines.append((None, raw_line, False))
                            else:
                                output_lines.append(raw_line)
                            continue

                        raw_payload = stripped[5:].strip()
                        if raw_payload == "[DONE]":
                            saw_done = True
                            if buffering_tool_calls:
                                output_lines.append(flush_tool_lines())
                                buffering_tool_calls = False
                            output_lines.append(raw_line)
                            continue
                        if not raw_payload:
                            if buffering_tool_calls:
                                buffered_tool_lines.append((None, raw_line, False))
                            else:
                                output_lines.append(raw_line)
                            continue
                        try:
                            event = json.loads(raw_payload)
                        except (json.JSONDecodeError, AttributeError):
                            if buffering_tool_calls:
                                buffered_tool_lines.append((None, raw_line, False))
                            else:
                                output_lines.append(raw_line)
                            continue
                        if not isinstance(event, dict):
                            output_lines.append(raw_line)
                            continue
                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]

                        changed = False
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta")
                            if isinstance(delta, dict):
                                reasoning = (
                                    delta.get("reasoning_text")
                                    or delta.get("reasoning_content")
                                    or delta.get("reasoning")
                                )
                                if reasoning:
                                    reasoning_buffer += reasoning
                                    # Some Qwen templates leave the requested tool call
                                    # in the reasoning channel. Promote complete tags.
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
                                                delta.clear()
                                                delta["tool_calls"] = [
                                                    {
                                                        "index": 0,
                                                        "id": (f"chatcmpl-tool-{uuid4().hex[:16]}"),
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
                            # Alma/AI-SDK must receive the OpenAI finish reason so
                            # it executes the tool and advances its internal loop.
                            if promoted_tool_call and choice.get("finish_reason") == "stop":
                                choice["finish_reason"] = "tool_calls"
                                choice["alma_promoted_tool_step"] = True
                                changed = True
                            elif choice.get("finish_reason") == "tool_calls":
                                choice["alma_tool_step"] = True
                                changed = True

                        has_tool_delta = _update_streamed_tool_call_states(
                            event,
                            tool_call_states,
                        )
                        if tool_call_states:
                            for choice in event.get("choices") or []:
                                if (
                                    isinstance(choice, dict)
                                    and choice.get("finish_reason") == "stop"
                                ):
                                    choice["finish_reason"] = "tool_calls"
                                    choice["alma_tool_step"] = True
                                    changed = True
                        if has_tool_delta:
                            buffering_tool_calls = True
                        if buffering_tool_calls:
                            buffered_tool_lines.append((event, raw_line, changed))
                            finish_reasons = [
                                choice.get("finish_reason")
                                for choice in event.get("choices") or []
                                if isinstance(choice, dict)
                            ]
                            if any(reason is not None for reason in finish_reasons):
                                output_lines.append(flush_tool_lines())
                                buffering_tool_calls = False
                            continue

                        output_lines.append(
                            _encode_sse_event(event, raw_line) if changed else raw_line
                        )

                    if output_lines:
                        yield "".join(output_lines).encode("utf-8")
                if buffered_tool_lines:
                    yield flush_tool_lines().encode("utf-8")
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
        if isinstance(parsed, dict):
            if isinstance(parsed.get("usage"), dict):
                usage = parsed["usage"]
            changed = False
            for choice in parsed.get("choices") or []:
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    if normalize_alma_tool_call_arguments(message):
                        changed = True
                    if message.get("tool_calls") and choice.get("finish_reason") == "stop":
                        choice["finish_reason"] = "tool_calls"
                        choice["alma_tool_step"] = True
                        changed = True
            if changed:
                content = json.dumps(
                    parsed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
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
