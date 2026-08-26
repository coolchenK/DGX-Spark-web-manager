import json
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.audit import record_audit
from app.dependencies import Admin, get_db
from app.gateway.proxy import (
    GENERATION_KEYS,
    merge_generation_defaults,
    openai_error,
    proxy_openai_request,
)
from app.models import ApiKey, Deployment, RequestMetric
from app.runtime.base import GenerationDefaults
from app.security import hash_api_key

router = APIRouter(tags=["openai-gateway"])
GatewayDb = Annotated[Session, Depends(get_db)]
_route_positions: dict[str, int] = defaultdict(int)
_route_lock = Lock()


class GatewayAuthError(Exception):
    pass


class GatewayActivity:
    def __init__(self) -> None:
        self._current = 0
        self._lock = Lock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._current

    def start(self) -> None:
        with self._lock:
            self._current += 1

    def finish(self) -> None:
        with self._lock:
            self._current = max(0, self._current - 1)


def require_gateway_key(
    db: GatewayDb,
    authorization: Annotated[str | None, Header()] = None,
) -> ApiKey:
    if not authorization or not authorization.startswith("Bearer "):
        raise_gateway_auth()
    value = authorization.removeprefix("Bearer ").strip()
    api_key = db.scalar(
        select(ApiKey).where(
            ApiKey.key_hash == hash_api_key(value),
            ApiKey.revoked_at.is_(None),
        )
    )
    if not api_key:
        raise_gateway_auth()
    api_key.last_used_at = datetime.now(UTC)
    db.commit()
    return api_key


def raise_gateway_auth() -> None:
    raise GatewayAuthError()


GatewayKey = Annotated[ApiKey, Depends(require_gateway_key)]


def deployment_route_name(deployment: Deployment) -> str:
    configured = (deployment.config or {}).get("route_alias")
    return str(configured or deployment.api_model_name)


def select_routed_deployment(
    db: Session,
    model: str,
    required_capability: str | None = None,
) -> Deployment | None:
    deployments = list(
        db.scalars(
            select(Deployment)
            .where(
                Deployment.status == "running",
                Deployment.health == "healthy",
            )
            .order_by(Deployment.created_at, Deployment.id)
        )
    )
    candidates = [
        deployment
        for deployment in deployments
        if deployment_route_name(deployment) == model
        and (required_capability is None or required_capability in deployment.capabilities)
    ]
    if not candidates:
        return None
    with _route_lock:
        position = _route_positions[model]
        _route_positions[model] = position + 1
    return candidates[position % len(candidates)]


def extract_alma_system_tools(body: dict[str, Any]) -> dict[str, Any]:
    """Convert Alma's XML-embedded function schemas to native OpenAI tools."""
    normalized = dict(body)
    if normalized.get("tools"):
        return normalized
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized
    decoder = json.JSONDecoder()
    extracted: list[dict[str, Any]] = []
    rewritten_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "system":
            rewritten_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or "<tools>" not in content:
            rewritten_messages.append(message)
            continue

        def replace_tools(match: re.Match[str]) -> str:
            block = match.group(1)
            position = 0
            while position < len(block):
                while position < len(block) and block[position].isspace():
                    position += 1
                if position >= len(block):
                    break
                try:
                    value, position = decoder.raw_decode(block, position)
                except json.JSONDecodeError:
                    return match.group(0)
                if (
                    isinstance(value, dict)
                    and value.get("type") == "function"
                    and isinstance(value.get("function"), dict)
                    and isinstance(value["function"].get("name"), str)
                ):
                    extracted.append(value)
            return ""

        cleaned = re.sub(r"<tools>\s*(.*?)\s*</tools>", replace_tools, content, flags=re.DOTALL)
        rewritten = dict(message)
        rewritten["content"] = cleaned
        rewritten_messages.append(rewritten)
    if extracted:
        failed_tools: set[str] = set()
        completed_meta_tools: set[str] = set()
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, str) or "<tool_response>" not in content:
                continue
            for match in re.finditer(
                r"<tool_response>\s*(\{.*?\})\s*</tool_response>", content, re.DOTALL
            ):
                try:
                    response = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                payload = response.get("content") if isinstance(response, dict) else None
                exit_code = payload.get("exitCode") if isinstance(payload, dict) else None
                response_name = response.get("name") if isinstance(response, dict) else None
                if exit_code not in (None, 0) and isinstance(response_name, str):
                    failed_tools.add(response_name)
                if response_name in {"Skill", "ToolSearch"}:
                    completed_meta_tools.add(response_name)
        excluded_tools = failed_tools | completed_meta_tools
        if excluded_tools:
            extracted = [
                tool
                for tool in extracted
                if tool.get("function", {}).get("name") not in excluded_tools
            ]
        normalized["messages"] = rewritten_messages
        normalized["tools"] = extracted
        normalized.setdefault("tool_choice", "auto")
    return normalized


async def enrich_empty_huggingface_search(body: dict[str, Any]) -> dict[str, Any]:
    """Ground empty Alma HF searches with the live Hub API before summarizing."""
    normalized = dict(body)
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized
    query: str | None = None
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or "<tool_response>" not in content:
            continue
        for match in re.finditer(
            r"<tool_response>\s*(\{.*?\})\s*</tool_response>", content, re.DOTALL
        ):
            try:
                response = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if response.get("name") not in {"WebSearch", "WebFetch"}:
                continue
            payload = response.get("content")
            payload_text = (
                json.dumps(payload, ensure_ascii=False) if not isinstance(payload, str) else payload
            )
            if (
                "huggingface" not in payload_text.lower()
                and "Qwen3.8-Flash-Next" not in payload_text
            ):
                continue
            if '"results": []' not in payload_text and '"results":[]' not in payload_text:
                continue
            query = "Qwen3.8-Flash-Next"
            break
        if query:
            break
    if not query:
        return normalized
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://huggingface.co/api/models",
                params={"search": query, "limit": 100},
            )
            response.raise_for_status()
            repositories = response.json()
    except (httpx.HTTPError, ValueError):
        return normalized
    if not isinstance(repositories, list):
        return normalized
    quantized: list[dict[str, Any]] = []
    quant_markers = ("gguf", "awq", "gptq", "fp8", "nvfp4", "mlx", "quant")
    for repository in repositories:
        if not isinstance(repository, Mapping):
            continue
        repo_id = repository.get("modelId") or repository.get("id")
        tags = repository.get("tags") if isinstance(repository.get("tags"), list) else []
        haystack = " ".join([str(repo_id or ""), *map(str, tags)]).lower()
        if repo_id and any(marker in haystack for marker in quant_markers):
            quantized.append(
                {
                    "id": repo_id,
                    "lastModified": repository.get("lastModified"),
                    "tags": [
                        tag
                        for tag in tags
                        if any(marker in str(tag).lower() for marker in quant_markers)
                    ],
                }
            )
    if not quantized:
        return normalized
    normalized["messages"] = [
        *messages,
        {
            "role": "user",
            "content": (
                "实时 Hugging Face API 返回了以下量化仓库，优先采用这些证据，"
                "不得再声称没有量化版本：\n" + json.dumps(quantized[:20], ensure_ascii=False)
            ),
        },
    ]
    return normalized


def normalize_alma_tool_continuation(body: dict[str, Any]) -> dict[str, Any]:
    """Force a visible action/final answer after Alma's XML tool response."""
    normalized = dict(body)
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized
    tool_responses: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for match in re.finditer(
            r"<tool_response>\s*(\{.*?\})\s*</tool_response>", content, re.DOTALL
        ):
            try:
                response = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(response, dict):
                tool_responses.append(response)
    if not tool_responses:
        return normalized
    if tool_responses[-1].get("name") in {"Skill", "ToolSearch"}:
        return normalized
    kwargs = normalized.get("chat_template_kwargs")
    kwargs = dict(kwargs) if isinstance(kwargs, Mapping) else {}
    kwargs["enable_thinking"] = False
    normalized["chat_template_kwargs"] = kwargs
    normalized["messages"] = [
        *messages,
        {
            "role": "user",
            "content": (
                "工具执行阶段已经结束。现在必须向用户给出最终回答，"
                "直接总结已有工具结果；即使结果为空或工具失败，也要明确说明结论和限制。"
                "不要再调用工具，不要只说‘稍等’、‘让我再查’或类似过渡语。"
            ),
        },
    ]
    return normalized


def normalize_reasoning_effort(body: dict[str, Any]) -> dict[str, Any]:
    """Translate Alma/AI-SDK reasoning options to Qwen-compatible request fields."""
    normalized = dict(body)
    aliases = {"high": "medium"}

    # Alma OpenCode Go supplies effort in provider options. These client-side
    # SDK options are not OpenAI request fields, so consume them at the gateway.
    provider_options = normalized.pop("provider_options", None)
    if provider_options is None:
        provider_options = normalized.pop("providerOptions", None)
    nested_effort = None
    if isinstance(provider_options, Mapping):
        for namespace in ("opencode-go", "openai"):
            values = provider_options.get(namespace)
            if not isinstance(values, Mapping):
                continue
            candidate = values.get("reasoningEffort") or values.get("reasoning_effort")
            if isinstance(candidate, str):
                nested_effort = candidate
                break

    effort = normalized.get("reasoning_effort")
    if not isinstance(effort, str):
        camel_effort = normalized.pop("reasoningEffort", None)
        effort = camel_effort if isinstance(camel_effort, str) else nested_effort
    if isinstance(effort, str):
        normalized["reasoning_effort"] = aliases.get(effort, effort)

    template_kwargs = normalized.get("chat_template_kwargs")
    if isinstance(template_kwargs, Mapping):
        template_kwargs = dict(template_kwargs)
        template_effort = template_kwargs.get("reasoning_effort")
        if isinstance(template_effort, str) and template_effort in aliases:
            template_kwargs["reasoning_effort"] = aliases[template_effort]
        normalized["chat_template_kwargs"] = template_kwargs
    return normalized


def deployment_generation_settings(
    deployment: Deployment,
) -> tuple[dict[str, Any], set[str]]:
    config = deployment.config
    if not isinstance(config, Mapping):
        return {}, set()
    spec = config.get("spec")
    capability_snapshot = config.get("runtime_capabilities")
    if not isinstance(spec, Mapping) or not isinstance(capability_snapshot, Mapping):
        return {}, set()
    defaults = spec.get("generation_defaults")
    supported = capability_snapshot.get("generation_defaults")
    if not isinstance(defaults, Mapping) or not isinstance(supported, list):
        return {}, set()
    validated: dict[str, Any] = {}
    for key in GENERATION_KEYS:
        if key not in defaults:
            continue
        try:
            parsed = GenerationDefaults.model_validate(
                {key: defaults[key]}, strict=True
            ).model_dump(mode="json", exclude_none=True)
        except (TypeError, ValueError):
            continue
        if key in parsed:
            validated[key] = parsed[key]
    return validated, {key for key in supported if isinstance(key, str)}


def _healthy_gateway_deployments(db: Session) -> list[Deployment]:
    return list(
        db.scalars(
            select(Deployment).where(
                Deployment.status == "running",
                Deployment.health == "healthy",
            )
        )
    )


def _alma_catalog_entry(deployment: Deployment) -> dict[str, Any]:
    config = deployment.config if isinstance(deployment.config, Mapping) else {}
    spec = config.get("spec") if isinstance(config.get("spec"), Mapping) else {}
    route_name = deployment_route_name(deployment)
    context_length = spec.get("context_length") or config.get("context_length")
    defaults = (
        dict(spec.get("generation_defaults") or {})
        if isinstance(spec.get("generation_defaults"), Mapping)
        else {}
    )
    max_output_tokens = defaults.get("max_tokens")
    max_input_tokens = (
        max(context_length - max_output_tokens, 0)
        if isinstance(context_length, int) and isinstance(max_output_tokens, int)
        else context_length
    )
    capability_names = set(deployment.capabilities)
    text_generation = bool(capability_names.intersection({"chat", "completion"}))
    vision = bool(capability_names.intersection({"vision", "image", "multimodal"}))
    return {
        "id": route_name,
        # Alma OpenCode Go uses `name` as the primary label while retaining
        # `id` as the API routing identity shown in the secondary label.
        "name": deployment.name,
        "display_name": deployment.name,
        "description": f"{deployment.name} via DGX Spark Web Manager",
        "apiFormat": "openai-chat",
        "api_format": "openai-chat",
        "api": "openai-compatible chat/completions",
        "context_length": context_length,
        "contextWindow": context_length,
        "max_output_tokens": max_output_tokens,
        "maxOutputTokens": max_output_tokens,
        "limit": {
            "context": context_length,
            "input": max_input_tokens,
            "output": max_output_tokens,
        },
        "modalities": {
            "input": ["text", "image"] if vision else ["text"],
            "output": (["text", "image"] if "image_generation" in capability_names else ["text"]),
        },
        "attachment": vision,
        "tool_call": text_generation,
        "toolCall": text_generation,
        "structured_output": text_generation,
        "reasoning": text_generation,
        "capabilities": {
            "vision": vision,
            "functionCalling": text_generation,
            "reasoning": text_generation,
            "streaming": text_generation,
            "contextWindow": context_length,
            "maxOutputTokens": max_output_tokens,
            "tokensPerSecond": deployment.benchmark_tps,
        },
        "performance": {
            "status": deployment.benchmark_status or "unavailable",
            "tokens_per_second": deployment.benchmark_tps,
            "completion_tokens": deployment.benchmark_completion_tokens,
            "duration_seconds": deployment.benchmark_duration_seconds,
            "tested_at": (
                deployment.benchmark_tested_at.isoformat()
                if deployment.benchmark_tested_at is not None
                else None
            ),
        },
    }


@router.get("/v1/models/catalog")
def alma_models_catalog(_: GatewayKey, db: GatewayDb) -> dict[str, Any]:
    """Model catalog parsed by Alma's OpenCode Go-compatible discovery path."""
    models = [_alma_catalog_entry(deployment) for deployment in _healthy_gateway_deployments(db)]
    return {"object": "list", "data": models, "models": models}


@router.get("/v1/models")
def openai_models(_: GatewayKey, db: GatewayDb) -> dict[str, Any]:
    deployments = _healthy_gateway_deployments(db)
    routes: dict[str, dict[str, Any]] = {}
    for deployment in deployments:
        route_name = deployment_route_name(deployment)
        if route_name not in routes:
            config = deployment.config if isinstance(deployment.config, Mapping) else {}
            spec = config.get("spec") if isinstance(config.get("spec"), Mapping) else {}
            context_length = spec.get("context_length") or config.get("context_length")
            generation_defaults = (
                dict(spec.get("generation_defaults") or {})
                if isinstance(spec.get("generation_defaults"), Mapping)
                else {}
            )
            max_output_tokens = generation_defaults.get("max_tokens")
            max_input_tokens = (
                max(context_length - max_output_tokens, 0)
                if isinstance(context_length, int) and isinstance(max_output_tokens, int)
                else context_length
            )
            max_concurrency = spec.get("max_concurrency")
            benchmark_tps = deployment.benchmark_tps
            benchmark_status = deployment.benchmark_status or (
                "succeeded" if benchmark_tps is not None else "unavailable"
            )
            performance = {
                "status": benchmark_status,
                "tokens_per_second": benchmark_tps,
                "completion_tokens": deployment.benchmark_completion_tokens,
                "duration_seconds": deployment.benchmark_duration_seconds,
                "tested_at": (
                    deployment.benchmark_tested_at.isoformat()
                    if deployment.benchmark_tested_at is not None
                    else None
                ),
            }
            limits = {
                "context_window": context_length,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
                "max_concurrency": max_concurrency,
            }
            metadata = {
                **limits,
                "context_length": context_length,
                "max_model_len": context_length,
                "max_context_tokens": context_length,
                "output_token_limit": max_output_tokens,
                "tokens_per_second": benchmark_tps,
                "benchmark_tps": benchmark_tps,
                "runtime": deployment.runtime,
                "capabilities": list(deployment.capabilities),
            }
            routes[route_name] = {
                "id": route_name,
                "object": "model",
                "created": int(deployment.created_at.timestamp()),
                "owned_by": "dgx-spark-manager",
                "root": route_name,
                "capabilities": list(deployment.capabilities),
                "instances": 1,
                "runtime": deployment.runtime,
                "endpoint_url": deployment.endpoint_url,
                # OpenAI-compatible model objects allow provider extensions. Keep
                # several established aliases so discovery clients with different
                # schemas can recognize the same limits without guesswork.
                "context_length": context_length,
                "max_model_len": context_length,
                "max_context_tokens": context_length,
                "context_window": context_length,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
                "max_tokens": max_output_tokens,
                "output_token_limit": max_output_tokens,
                "max_concurrency": max_concurrency,
                "generation_defaults": generation_defaults,
                "benchmark_tps": benchmark_tps,
                "tokens_per_second": benchmark_tps,
                "performance": performance,
                "limits": limits,
                "metadata": metadata,
            }
            continue
        route = routes[route_name]
        route["created"] = min(route["created"], int(deployment.created_at.timestamp()))
        route["capabilities"] = [
            capability
            for capability in route["capabilities"]
            if capability in deployment.capabilities
        ]
        route["instances"] += 1
    # Keep the standard OpenAI list envelope while enriching each item with
    # the same catalog fields Alma 0.3.0's OpenCode Go parser understands.
    # Alma's Custom provider ignores these fields (an Alma-side limitation),
    # but OpenCode Go pointed at this standard base URL consumes them.
    for route in routes.values():
        deployment = next(
            candidate
            for candidate in deployments
            if deployment_route_name(candidate) == route["id"]
        )
        route.update(_alma_catalog_entry(deployment))
        route["object"] = "model"
        route["created"] = route.get("created", int(deployment.created_at.timestamp()))
        route["owned_by"] = "dgx-spark-manager"
        route["root"] = route["id"]
        route["instances"] = route.get("instances", 1)
        route["runtime"] = deployment.runtime
        route["endpoint_url"] = deployment.endpoint_url
        route["generation_defaults"] = (
            (deployment.config.get("spec") or {}).get("generation_defaults") or {}
            if isinstance(deployment.config, Mapping)
            else {}
        )
        route["benchmark_tps"] = deployment.benchmark_tps
        route["tokens_per_second"] = deployment.benchmark_tps
        route["metadata"] = {
            "context_window": route["context_window"],
            "max_input_tokens": route["max_input_tokens"],
            "max_output_tokens": route["max_output_tokens"],
            "max_concurrency": route["max_concurrency"],
            "context_length": route["context_length"],
            "max_model_len": route["max_model_len"],
            "max_context_tokens": route["max_context_tokens"],
            "output_token_limit": route["output_token_limit"],
            "tokens_per_second": route["tokens_per_second"],
            "benchmark_tps": route["benchmark_tps"],
            "runtime": route["runtime"],
            "capabilities": list(deployment.capabilities),
        }
        route["limits"] = {
            "context_window": route["context_window"],
            "max_input_tokens": route["max_input_tokens"],
            "max_output_tokens": route["max_output_tokens"],
            "max_concurrency": route["max_concurrency"],
        }
        route["capability_names"] = list(deployment.capabilities)
    return {
        "object": "list",
        "data": list(routes.values()),
    }


async def _proxy(
    endpoint: str,
    request: Request,
    db: Session,
    required_capability: str,
) -> Any:
    try:
        body = await request.json()
    except ValueError:
        return openai_error("Request body must be valid JSON", status_code=400)
    model = body.get("model") if isinstance(body, dict) else None
    if not model:
        return openai_error("The model field is required", status_code=400)
    deployment = select_routed_deployment(db, str(model), required_capability)
    if not deployment:
        return openai_error(f"Model '{model}' was not found or is not healthy", status_code=404)
    defaults, supported = deployment_generation_settings(deployment)
    normalized_body = extract_alma_system_tools(dict(body))
    normalized_body = await enrich_empty_huggingface_search(normalized_body)
    normalized_body = normalize_alma_tool_continuation(normalized_body)
    normalized_body = normalize_reasoning_effort(normalized_body)
    merged_body, applied = merge_generation_defaults(
        endpoint,
        normalized_body,
        defaults,
        supported=supported,
    )
    if applied:
        record_audit(
            db,
            actor="gateway",
            action="gateway.defaults.apply",
            resource_type="deployment",
            resource_id=deployment.id,
            details={
                "endpoint": endpoint,
                "model": str(model),
                "applied_fields": applied,
            },
        )
        db.commit()
    activity: GatewayActivity = request.app.state.gateway_activity
    activity.start()
    finished = False
    finish_lock = Lock()

    def finish_request() -> None:
        nonlocal finished
        with finish_lock:
            if finished:
                return
            finished = True
        activity.finish()

    try:
        return await proxy_openai_request(
            request,
            deployment,
            endpoint,
            merged_body,
            on_finished=finish_request,
        )
    except Exception:
        finish_request()
        raise


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    _: GatewayKey,
    db: GatewayDb,
):
    return await _proxy("/v1/chat/completions", request, db, "chat")


@router.post("/v1/completions")
async def completions(request: Request, _: GatewayKey, db: GatewayDb):
    return await _proxy("/v1/completions", request, db, "completion")


@router.post("/v1/embeddings")
async def embeddings(request: Request, _: GatewayKey, db: GatewayDb):
    return await _proxy("/v1/embeddings", request, db, "embedding")


@router.get("/api/gateway/stats")
def gateway_stats(request: Request, _: Admin, db: GatewayDb) -> dict[str, Any]:
    total = db.scalar(select(func.count(RequestMetric.id))) or 0
    failed = (
        db.scalar(select(func.count(RequestMetric.id)).where(RequestMetric.status_code >= 400)) or 0
    )
    avg_latency = db.scalar(select(func.avg(RequestMetric.latency_ms))) or 0
    prompt_tokens = db.scalar(select(func.sum(RequestMetric.prompt_tokens))) or 0
    completion_tokens = db.scalar(select(func.sum(RequestMetric.completion_tokens))) or 0
    cutoff = datetime.now(UTC) - timedelta(minutes=1)
    requests_last_minute = (
        db.scalar(select(func.count(RequestMetric.id)).where(RequestMetric.created_at >= cutoff))
        or 0
    )
    recent_prompt_tokens = (
        db.scalar(
            select(func.sum(RequestMetric.prompt_tokens)).where(RequestMetric.created_at >= cutoff)
        )
        or 0
    )
    recent_completion_tokens = (
        db.scalar(
            select(func.sum(RequestMetric.completion_tokens)).where(
                RequestMetric.created_at >= cutoff
            )
        )
        or 0
    )
    return {
        "total_requests": total,
        "failed_requests": failed,
        "error_rate": failed / total if total else 0,
        "average_latency_ms": round(float(avg_latency), 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "requests_last_minute": requests_last_minute,
        "tokens_per_second": round(
            (recent_prompt_tokens + recent_completion_tokens) / 60,
            2,
        ),
        "active_requests": request.app.state.gateway_activity.current,
    }
