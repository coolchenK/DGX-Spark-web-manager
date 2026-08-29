from __future__ import annotations

import asyncio
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
ALMA_TASK_DESCRIPTION_FALLBACK = "Execute delegated task"
ALMA_CONTINUATION_HEARTBEAT_SECONDS = 15.0
ALMA_HIDDEN_REASONING_BLOCK_RE = re.compile(
    r"<(?P<tag>think|analysis)\b[^>]*>.*?(?:</(?P=tag)\s*>|$)",
    re.IGNORECASE | re.DOTALL,
)
ALMA_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(?P<payload>\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
ALMA_PROTOCOL_TOKEN_RE = re.compile(
    r"<\|channel\|>[^<]*<\|message\|>|<\|[^>]+\|>",
    re.IGNORECASE,
)
ALMA_VISIBLE_WORD_RE = re.compile(r"\w", re.UNICODE)
ALMA_EMPTY_CONTINUATION_RETRY_PROMPT = (
    "The preceding assistant generation was empty or stopped after describing a future "
    "action without performing it. Continue the original task now "
    "using the tool results already present in the conversation. If another tool is "
    "strictly necessary, call it immediately. Otherwise, return the complete final "
    "answer. Do not emit an empty response."
)
ALMA_EMPTY_CONTINUATION_FINAL_PROMPT = (
    "The model returned an empty continuation again. Do not call another tool. Return "
    "the best complete final answer now from the available tool results."
)
ALMA_EMPTY_CONTINUATION_FALLBACK = (
    "模型在工具执行后连续返回了空结果或未执行的状态信息，"
    "网关已自动重试但仍未获得有效回答。请重新发送本次请求。"
)
ALMA_INCOMPLETE_ACTION_PATTERNS = (
    re.compile(
        r"(?:我(?:先|现在|这就|马上)?(?:会|将|去|来)?|现在我|接下来我|这就).{0,32}"
        r"(?:获取|抓|下载|查询|搜索|查看|检查|调用|运行|执行).{0,40}"
        r"(?:稍等|等一下|请稍候|片刻|[~～…])",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:i(?:'ll| will| am going to)|let me|next i(?:'ll| will)).{0,48}"
        r"(?:fetch|download|query|search|look up|inspect|check|call|run|execute).{0,48}"
        r"(?:wait|one moment|hold on|shortly|\.\.\.)",
        re.IGNORECASE | re.DOTALL,
    ),
)


def _content_contains_tool_result(content: Any) -> bool:
    if isinstance(content, str):
        lowered = content.casefold()
        return any(
            marker in lowered
            for marker in (
                "<tool_response>",
                "<tool_result>",
                "<tool-response>",
                "<tool-result>",
            )
        )
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").casefold().replace("_", "-")
        if item_type in {"tool-result", "tool-response"}:
            return True
        if (content.get("tool_call_id") or content.get("toolCallId")) and any(
            key in content for key in ("output", "result", "content")
        ):
            return True
        return any(_content_contains_tool_result(value) for value in content.values())
    if isinstance(content, list):
        return any(_content_contains_tool_result(item) for item in content)
    return False


def is_alma_tool_continuation(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("tool_call_id"):
            return True
        if _content_contains_tool_result(message.get("content")):
            return True
        # Alma 0.4.x can retain the assistant tool call while representing the
        # corresponding result as a later UI-message part. Treat any completed
        # assistant tool step as a continuation even when that result shape is
        # not part of the OpenAI wire schema yet.
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if index < len(messages) - 1:
                return True
    return False


def should_buffer_alma_tool_stream(
    *,
    endpoint: str,
    body: dict[str, Any],
) -> bool:
    if endpoint != "/v1/chat/completions" or not bool(body.get("stream")):
        return False
    return is_alma_tool_continuation(body)


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
        if _promote_tagged_tool_call(message):
            changed = True
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


def _content_has_visible_text(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    visible = ALMA_HIDDEN_REASONING_BLOCK_RE.sub("", content)
    visible = ALMA_TOOL_CALL_BLOCK_RE.sub("", visible)
    visible = ALMA_PROTOCOL_TOKEN_RE.sub("", visible).strip()
    return bool(visible and ALMA_VISIBLE_WORD_RE.search(visible))


def _chat_completion_has_output(payload: dict[str, Any]) -> bool:
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if _content_has_visible_text(content):
            return True
        if message.get("tool_calls"):
            return True
    return False


def _chat_completion_is_incomplete_action_pledge(payload: dict[str, Any]) -> bool:
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        visible = ALMA_HIDDEN_REASONING_BLOCK_RE.sub("", content).strip()
        if len(visible) > 500:
            continue
        if any(pattern.search(visible) for pattern in ALMA_INCOMPLETE_ACTION_PATTERNS):
            return True
    return False


def _chat_completion_has_actionable_output(payload: dict[str, Any]) -> bool:
    return _chat_completion_has_output(
        payload
    ) and not _chat_completion_is_incomplete_action_pledge(payload)


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


def _normalize_alma_task_arguments(arguments: Any) -> bool:
    if not isinstance(arguments, dict):
        return False
    current = arguments.get("description")
    if isinstance(current, str) and current.strip():
        description = current[:ALMA_TASK_DESCRIPTION_MAX_LENGTH]
    else:
        description = ""
        for key in ("subject", "title", "prompt"):
            candidate = arguments.get(key)
            if isinstance(candidate, str) and candidate.strip():
                description = " ".join(candidate.split())
                break
        description = (
            description[:ALMA_TASK_DESCRIPTION_MAX_LENGTH] or ALMA_TASK_DESCRIPTION_FALLBACK
        )
    if current == description:
        return False
    arguments["description"] = description
    return True


def _promote_tagged_tool_call(message: dict[str, Any]) -> bool:
    if message.get("tool_calls"):
        return False
    for key in ("reasoning_text", "reasoning_content", "reasoning", "content"):
        source = message.get(key)
        if not isinstance(source, str):
            continue
        match = ALMA_TOOL_CALL_BLOCK_RE.search(source)
        if match is None:
            continue
        try:
            parsed_call = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed_call, dict):
            continue
        name = parsed_call.get("name")
        arguments = parsed_call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            continue
        message["tool_calls"] = [
            {
                "id": f"chatcmpl-tool-{uuid4().hex[:16]}",
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
        if key == "content":
            message["content"] = f"{source[: match.start()]}{source[match.end() :]}"
        return True
    return False


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
        if not _normalize_alma_task_arguments(arguments):
            continue
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
        if not _normalize_alma_task_arguments(arguments):
            continue
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
        if status_code >= 400 or (
            parsed is not None and _chat_completion_has_actionable_output(parsed)
        ):
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
        if not _chat_completion_has_actionable_output(parsed):
            logger.error(
                "Alma tool continuation remained incomplete after retries model=%s",
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
    timeout = upstream_inference_timeout()
    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    buffer_alma_continuation = should_buffer_alma_tool_stream(
        endpoint=endpoint,
        body=body,
    )
    upstream_body = dict(body)
    if buffer_alma_continuation:
        upstream_body["stream"] = False
        upstream_body.pop("stream_options", None)
    upstream_request = client.build_request("POST", url, json=upstream_body)
    if buffer_alma_continuation:

        async def buffered_continuation_body() -> AsyncIterator[bytes]:
            async def fetch_response() -> Response:
                try:
                    upstream = await client.send(upstream_request, stream=True)
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
                    return openai_error(
                        f"Upstream inference service is unavailable: {exc}",
                        status_code=502,
                    )
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

            response_task = asyncio.create_task(fetch_response())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {response_task},
                        timeout=ALMA_CONTINUATION_HEARTBEAT_SECONDS,
                    )
                    if done:
                        response = response_task.result()
                        if response.status_code >= 400:
                            error_payload: dict[str, Any]
                            try:
                                parsed_error = json.loads(response.body)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                parsed_error = None
                            if isinstance(parsed_error, dict) and isinstance(
                                parsed_error.get("error"), dict
                            ):
                                error_payload = parsed_error
                            else:
                                error_payload = {
                                    "error": {
                                        "message": "Upstream inference request failed",
                                        "type": "server_error",
                                        "code": str(response.status_code),
                                    }
                                }
                            error_payload["error"]["upstream_status"] = response.status_code
                            yield (
                                "data: "
                                + json.dumps(
                                    error_payload,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n\ndata: [DONE]\n\n"
                            ).encode()
                            return
                        yield bytes(response.body)
                        return
                    yield b": alma-keepalive\n\n"
            finally:
                if not response_task.done():
                    response_task.cancel()
                    try:
                        await response_task
                    except asyncio.CancelledError:
                        pass

        return StreamingResponse(
            buffered_continuation_body(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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
            rewrite_pending = b""
            reasoning_buffer = ""
            promoted_tool_call = False
            buffered_tool_lines: list[tuple[dict[str, Any] | None, str, bool]] = []
            tool_call_states: dict[int, dict[str, Any]] = {}
            buffering_tool_calls = False
            upstream_chunks = upstream.aiter_raw()

            async def next_upstream_chunk() -> bytes:
                return await anext(upstream_chunks)

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

            chunk_task: asyncio.Task[bytes] | None = asyncio.create_task(next_upstream_chunk())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        {chunk_task},
                        timeout=ALMA_CONTINUATION_HEARTBEAT_SECONDS,
                    )
                    if not done:
                        yield b": alma-keepalive\n\n"
                        continue
                    try:
                        chunk = chunk_task.result()
                    except StopAsyncIteration:
                        chunk_task = None
                        break
                    chunk_task = asyncio.create_task(next_upstream_chunk())
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
