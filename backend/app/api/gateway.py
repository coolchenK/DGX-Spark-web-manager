import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.audit import record_audit
from app.dependencies import Admin, get_db
from app.gateway.adapters import adapter_for_runtime
from app.gateway.proxy import (
    GENERATION_KEYS,
    UsageScanner,
    extract_usage_from_json,
    merge_generation_defaults,
    openai_error,
    proxy_openai_request,
    record_request_metric,
    upstream_inference_timeout,
)
from app.gateway.responses import (
    RESPONSES_ENDPOINT,
    ResponsesTranslationError,
    ResponsesTurnTranslator,
    responses_to_chat_request,
    uses_thinking_toggle,
)
from app.gateway.responses_stream import (
    ResponsesStreamTranslator,
    parse_chat_sse_frames,
)
from app.models import ApiKey, Deployment, RequestMetric
from app.runtime.base import QWEN_FIXED_CHAT_TEMPLATE_PROFILE, GenerationDefaults
from app.security import hash_api_key
from app.services.model_capabilities import (
    input_modalities,
    runtime_multimodal_parameters,
)
from app.services.upstream_gateway import upstream_request_url

router = APIRouter(tags=["openai-gateway"])
GatewayDb = Annotated[Session, Depends(get_db)]
_route_positions: dict[str, int] = defaultdict(int)
_route_lock = Lock()
RUNTIME_CAPACITY_PROBE_TIMEOUT_SECONDS = 1.0


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
    required_capability: str | set[str] | tuple[str, ...] | None = None,
    *,
    runtime: str | None = None,
) -> Deployment | None:
    required = (
        {required_capability}
        if isinstance(required_capability, str)
        else set(required_capability or ())
    )
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
        and required.issubset(set(deployment.capabilities))
        and (runtime is None or deployment.runtime == runtime)
    ]
    if not candidates:
        return None
    with _route_lock:
        route_key = "\0".join((model, runtime or "", *sorted(required)))
        position = _route_positions[route_key]
        _route_positions[route_key] = position + 1
    return candidates[position % len(candidates)]


class MultimodalRequestError(ValueError):
    pass


def _normalize_media_part(
    part: Mapping[str, Any],
    *,
    target_type: str,
    field: str,
) -> dict[str, Any]:
    normalized = dict(part)
    value = normalized.get(field)
    if isinstance(value, str):
        media = {"url": value}
    elif isinstance(value, Mapping):
        media = dict(value)
    else:
        if normalized.get("file_id"):
            raise MultimodalRequestError(
                "file_id media inputs are not available on this gateway; use a URL or data URL"
            )
        raise MultimodalRequestError(f"{field} must be a URL string or an object containing url")
    if not isinstance(media.get("url"), str) or not str(media["url"]).strip():
        raise MultimodalRequestError(f"{field}.url must be a non-empty string")
    if target_type == "image_url" and "detail" in normalized and "detail" not in media:
        media["detail"] = normalized.pop("detail")
    normalized.pop("file_id", None)
    normalized["type"] = target_type
    normalized[field] = media
    return normalized


def normalize_multimodal_chat_body(body: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    normalized = dict(body)
    messages = normalized.get("messages")
    if not isinstance(messages, list):
        return normalized, set()

    requested: set[str] = set()
    output_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, Mapping):
            output_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            output_messages.append(message)
            continue
        output_parts: list[Any] = []
        for part in content:
            if not isinstance(part, Mapping):
                output_parts.append(part)
                continue
            part_type = str(part.get("type") or "").casefold().replace("-", "_")
            if not part_type:
                if "image_url" in part:
                    part_type = "image_url"
                elif "video_url" in part:
                    part_type = "video_url"
            if part_type == "input_text":
                rewritten = dict(part)
                rewritten["type"] = "text"
                output_parts.append(rewritten)
            elif part_type in {"image_url", "input_image"}:
                requested.add("image")
                output_parts.append(
                    _normalize_media_part(part, target_type="image_url", field="image_url")
                )
            elif part_type in {"video_url", "input_video"}:
                requested.add("video")
                output_parts.append(
                    _normalize_media_part(part, target_type="video_url", field="video_url")
                )
            else:
                output_parts.append(part)
        rewritten_message = dict(message)
        rewritten_message["content"] = output_parts
        output_messages.append(rewritten_message)
    normalized["messages"] = output_messages
    return normalized, requested


def requested_multimodal_runtime(body: Mapping[str, Any]) -> str | None:
    vllm = {"mm_processor_kwargs", "media_io_kwargs"}.intersection(body)
    sglang = {"images_config", "use_audio_in_video"}.intersection(body)
    if vllm and sglang:
        raise MultimodalRequestError(
            "vLLM and SGLang multimodal request parameters cannot be mixed"
        )
    for key in ("mm_processor_kwargs", "media_io_kwargs", "images_config"):
        if key in body and body[key] is not None and not isinstance(body[key], Mapping):
            raise MultimodalRequestError(f"{key} must be an object")
    if "use_audio_in_video" in body and not isinstance(body["use_audio_in_video"], bool):
        raise MultimodalRequestError("use_audio_in_video must be a boolean")
    if vllm:
        return "vllm"
    if sglang:
        return "sglang"
    return None


def normalize_reasoning_effort(
    body: dict[str, Any],
    *,
    chat_template: str | None = None,
) -> dict[str, Any]:
    """Translate standard reasoning_effort to model-compatible template fields."""
    normalized = dict(body)
    aliases = (
        {
            "high": "xhigh",
            "max": "xhigh",
            "ultracode": "xhigh",
            "extreme": "xhigh",
            "minimal": "low",
            "off": "none",
        }
        if chat_template == QWEN_FIXED_CHAT_TEMPLATE_PROFILE
        else {"high": "medium"}
    )

    effort = normalized.get("reasoning_effort")
    if isinstance(effort, str):
        canonical_effort = effort.strip().lower()
        normalized["reasoning_effort"] = aliases.get(canonical_effort, canonical_effort)

    template_kwargs = normalized.get("chat_template_kwargs")
    template_kwargs = dict(template_kwargs) if isinstance(template_kwargs, Mapping) else {}
    if isinstance(effort, str):
        template_kwargs["reasoning_effort"] = normalized["reasoning_effort"]
        if normalized["reasoning_effort"] == "none":
            template_kwargs["enable_thinking"] = False
    elif isinstance(template_kwargs.get("reasoning_effort"), str):
        canonical_effort = template_kwargs["reasoning_effort"].strip().lower()
        template_kwargs["reasoning_effort"] = aliases.get(canonical_effort, canonical_effort)
    if template_kwargs:
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


def deployment_context_limits(deployment: Deployment) -> dict[str, int | None]:
    """Resolve public limits from configured context and live runtime capacity."""
    config = deployment.config if isinstance(deployment.config, Mapping) else {}
    spec = config.get("spec") if isinstance(config.get("spec"), Mapping) else {}
    configured_context = spec.get("context_length") or config.get("context_length")
    if not isinstance(configured_context, int) or isinstance(configured_context, bool):
        configured_context = None

    runtime_capacity: int | None = None
    if deployment.runtime == "sglang":
        try:
            with httpx.Client(
                timeout=RUNTIME_CAPACITY_PROBE_TIMEOUT_SECONDS,
                trust_env=False,
            ) as client:
                response = client.get(
                    f"{deployment.endpoint_url.rstrip('/')}/get_server_info"
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            candidate = payload.get("max_total_num_tokens")
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                runtime_capacity = candidate

    if configured_context is None:
        context_length = runtime_capacity
    elif runtime_capacity is None:
        context_length = configured_context
    else:
        context_length = min(configured_context, runtime_capacity)

    defaults = (
        dict(spec.get("generation_defaults") or {})
        if isinstance(spec.get("generation_defaults"), Mapping)
        else {}
    )
    max_output_tokens = defaults.get("max_tokens")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool):
        max_output_tokens = None
    max_input_tokens = (
        max(context_length - max_output_tokens, 0)
        if context_length is not None and max_output_tokens is not None
        else context_length
    )
    return {
        "context_length": context_length,
        "configured_context_length": configured_context,
        "runtime_token_capacity": runtime_capacity,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
    }


def _healthy_gateway_deployments(db: Session) -> list[Deployment]:
    return list(
        db.scalars(
            select(Deployment).where(
                Deployment.status == "running",
                Deployment.health == "healthy",
            )
        )
    )


def _model_catalog_entry(
    deployment: Deployment,
    *,
    capability_names: list[str] | None = None,
    context_limits: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    route_name = deployment_route_name(deployment)
    limits = dict(context_limits or deployment_context_limits(deployment))
    context_length = limits.get("context_length")
    configured_context_length = limits.get("configured_context_length")
    runtime_token_capacity = limits.get("runtime_token_capacity")
    max_output_tokens = limits.get("max_output_tokens")
    max_input_tokens = limits.get("max_input_tokens")
    capability_set = set(
        deployment.capabilities if capability_names is None else capability_names
    )
    text_generation = bool(capability_set.intersection({"chat", "completion"}))
    vision = "image" in capability_set
    video = "video" in capability_set
    modalities = input_modalities(capability_set)
    multimodal_parameters = (
        runtime_multimodal_parameters(deployment.runtime) if vision or video else []
    )
    return {
        "id": route_name,
        # `name` carries the human-readable label while `id` stays the API
        # routing identity referenced by clients.
        "name": deployment.name,
        "display_name": deployment.name,
        "description": f"{deployment.name} via DGX Spark Web Manager",
        "apiFormat": "openai-chat",
        "api_format": "openai-chat",
        "api": "openai-compatible chat/completions",
        "context_length": context_length,
        "contextWindow": context_length,
        "configured_context_length": configured_context_length,
        "runtime_token_capacity": runtime_token_capacity,
        "max_output_tokens": max_output_tokens,
        "maxOutputTokens": max_output_tokens,
        "limit": {
            "context": context_length,
            "input": max_input_tokens,
            "output": max_output_tokens,
        },
        "modalities": {
            "input": modalities,
            "output": (["text", "image"] if "image_generation" in capability_set else ["text"]),
        },
        "input_modalities": modalities,
        "multimodal_parameters": multimodal_parameters,
        "multimodal": {
            "image_content_type": "image_url" if vision else None,
            "video_content_type": "video_url" if video else None,
            "request_parameters": multimodal_parameters,
        },
        "attachment": vision or video,
        "tool_call": text_generation,
        "toolCall": text_generation,
        "structured_output": text_generation,
        "reasoning": text_generation,
        "capabilities": {
            "vision": vision,
            "video": video,
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




@router.get("/v1/models")
def openai_models(_: GatewayKey, db: GatewayDb) -> dict[str, Any]:
    deployments = _healthy_gateway_deployments(db)
    context_limits = {
        deployment.id: deployment_context_limits(deployment) for deployment in deployments
    }
    routes: dict[str, dict[str, Any]] = {}
    for deployment in deployments:
        route_name = deployment_route_name(deployment)
        deployment_limits = context_limits[deployment.id]
        if route_name not in routes:
            config = deployment.config if isinstance(deployment.config, Mapping) else {}
            spec = config.get("spec") if isinstance(config.get("spec"), Mapping) else {}
            context_length = deployment_limits["context_length"]
            configured_context_length = deployment_limits["configured_context_length"]
            runtime_token_capacity = deployment_limits["runtime_token_capacity"]
            generation_defaults = (
                dict(spec.get("generation_defaults") or {})
                if isinstance(spec.get("generation_defaults"), Mapping)
                else {}
            )
            max_output_tokens = deployment_limits["max_output_tokens"]
            max_input_tokens = deployment_limits["max_input_tokens"]
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
                "configured_context_length": configured_context_length,
                "runtime_token_capacity": runtime_token_capacity,
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
                "max_concurrency": max_concurrency,
            }
            metadata = {
                **limits,
                "context_length": context_length,
                "configured_context_length": configured_context_length,
                "runtime_token_capacity": runtime_token_capacity,
                "max_model_len": context_length,
                "max_context_tokens": context_length,
                "output_token_limit": max_output_tokens,
                "tokens_per_second": benchmark_tps,
                "benchmark_tps": benchmark_tps,
                "runtime": deployment.runtime,
                "capabilities": list(deployment.capabilities),
                "input_modalities": input_modalities(deployment.capabilities),
                "multimodal_parameters": (
                    runtime_multimodal_parameters(deployment.runtime)
                    if set(deployment.capabilities).intersection({"image", "video"})
                    else []
                ),
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
                "configured_context_length": configured_context_length,
                "runtime_token_capacity": runtime_token_capacity,
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
        member_context = deployment_limits["context_length"]
        if isinstance(member_context, int) and (
            not isinstance(route["context_length"], int)
            or member_context < route["context_length"]
        ):
            route["context_length"] = member_context
            route["context_window"] = member_context
            route["max_model_len"] = member_context
            route["max_context_tokens"] = member_context
            max_output_tokens = route.get("max_output_tokens")
            route["max_input_tokens"] = (
                max(member_context - max_output_tokens, 0)
                if isinstance(max_output_tokens, int)
                else member_context
            )
    # Keep the standard OpenAI list envelope while enriching each item with
    # discovery metadata so OpenAI-compatible clients can read context
    # limits and modalities without extra round trips.
    for route in routes.values():
        deployment = next(
            candidate
            for candidate in deployments
            if deployment_route_name(candidate) == route["id"]
        )
        capability_names = list(route["capabilities"])
        route.update(
            _model_catalog_entry(
                deployment,
                capability_names=capability_names,
                context_limits={
                    "context_length": route["context_length"],
                    "configured_context_length": route["configured_context_length"],
                    "runtime_token_capacity": route["runtime_token_capacity"],
                    "max_input_tokens": route["max_input_tokens"],
                    "max_output_tokens": route["max_output_tokens"],
                },
            )
        )
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
            "configured_context_length": route["configured_context_length"],
            "runtime_token_capacity": route["runtime_token_capacity"],
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
            "capabilities": capability_names,
            "input_modalities": input_modalities(capability_names),
            "multimodal_parameters": (
                runtime_multimodal_parameters(deployment.runtime)
                if set(capability_names).intersection({"image", "video"})
                else []
            ),
        }
        route["limits"] = {
            "context_window": route["context_window"],
            "configured_context_length": route["configured_context_length"],
            "runtime_token_capacity": route["runtime_token_capacity"],
            "max_input_tokens": route["max_input_tokens"],
            "max_output_tokens": route["max_output_tokens"],
            "max_concurrency": route["max_concurrency"],
        }
        route["capability_names"] = capability_names
    return {
        "object": "list",
        "data": list(routes.values()),
    }


@router.get("/v1/models/{model:path}")
def openai_model(model: str, key: GatewayKey, db: GatewayDb) -> Any:
    for item in openai_models(key, db)["data"]:
        if item["id"] == model:
            return item
    return openai_error(f"Model '{model}' was not found or is not healthy", status_code=404)


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
    normalized_body = dict(body)
    requested_modalities: set[str] = set()
    requested_runtime: str | None = None
    if endpoint == "/v1/chat/completions":
        try:
            normalized_body, requested_modalities = normalize_multimodal_chat_body(
                normalized_body
            )
            requested_runtime = requested_multimodal_runtime(normalized_body)
        except MultimodalRequestError as exc:
            return openai_error(str(exc), status_code=400)
    required_capabilities = {required_capability, *requested_modalities}
    deployment = select_routed_deployment(
        db,
        str(model),
        required_capabilities,
        runtime=requested_runtime,
    )
    if not deployment:
        base_deployment = select_routed_deployment(db, str(model), required_capability)
        if base_deployment is not None and (requested_modalities or requested_runtime):
            requested = ", ".join(sorted(requested_modalities)) or requested_runtime
            return openai_error(
                f"Model '{model}' does not support the requested multimodal input or "
                f"runtime parameters: {requested}",
                status_code=400,
            )
        # Not hosted here: forward to the configured upstream gateway.
        return await _proxy_fallback(request, db, endpoint, normalized_body)
    adapter = adapter_for_runtime(deployment.runtime)
    normalized_body = adapter.adapt_request(endpoint, normalized_body).body
    defaults, supported = deployment_generation_settings(deployment)
    config = deployment.config if isinstance(deployment.config, Mapping) else {}
    spec = config.get("spec") if isinstance(config.get("spec"), Mapping) else {}
    normalized_body = normalize_reasoning_effort(
        normalized_body,
        chat_template=spec.get("chat_template"),
    )
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
            adapter,
            on_finished=finish_request,
        )
    except Exception:
        finish_request()
        raise


def _fallback_upstream(settings: Any) -> tuple[str, dict[str, str]] | None:
    """Resolve the upstream gateway used for models this manager does not host."""
    base_url = (getattr(settings, "fallback_base_url", None) or "").strip().rstrip("/")
    if not base_url:
        return None
    headers: dict[str, str] = {}
    api_key = (getattr(settings, "fallback_api_key", None) or "").strip()
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return base_url, headers


async def _proxy_fallback(
    request: Request,
    db: GatewayDb,
    endpoint: str,
    body: Mapping[str, Any],
) -> Response:
    """Forward a request for a non-local model to the configured upstream gateway.

    The fallback target speaks the same wire API, so the body is forwarded
    verbatim. `authorization` is supplied by the fallback key when configured,
    otherwise the caller's own bearer token is reused.
    """
    settings = request.app.state.settings
    upstream_target = _fallback_upstream(settings)
    if upstream_target is None:
        return openai_error(
            f"Model '{body.get('model')}' was not found or is not healthy",
            status_code=404,
        )
    base_url, fallback_headers = upstream_target

    forward_body = dict(body)
    forward_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "authorization", "accept-encoding"}
    }
    forward_headers.update(fallback_headers)
    if "authorization" not in forward_headers:
        caller_auth = request.headers.get("authorization")
        if caller_auth:
            forward_headers["authorization"] = caller_auth

    started_at = time.perf_counter()
    client = httpx.AsyncClient(timeout=upstream_inference_timeout(), trust_env=False)
    upstream_request = client.build_request(
        "POST",
        upstream_request_url(base_url, endpoint),
        json=forward_body,
        headers=forward_headers,
    )
    try:
        upstream = await client.send(upstream_request, stream=bool(forward_body.get("stream")))
    except httpx.HTTPError as exc:
        await client.aclose()
        record_request_metric(
            request.app.state.database.session_factory,
            model=str(body.get("model")),
            endpoint=endpoint,
            status_code=502,
            started_at=started_at,
        )
        return openai_error(
            f"Fallback gateway is unavailable: {exc}", status_code=502
        )

    if not forward_body.get("stream"):
        content = await upstream.aread()
        status_code = upstream.status_code
        content_type = upstream.headers.get("content-type", "application/json")
        await upstream.aclose()
        await client.aclose()
        record_request_metric(
            request.app.state.database.session_factory,
            model=str(body.get("model")),
            endpoint=endpoint,
            status_code=status_code,
            started_at=started_at,
            usage=extract_usage_from_json(content),
        )
        return Response(content=content, status_code=status_code, media_type=content_type)

    scanner = UsageScanner()

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                scanner.feed(chunk)
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()
            record_request_metric(
                request.app.state.database.session_factory,
                model=str(body.get("model")),
                endpoint=endpoint,
                status_code=upstream.status_code,
                started_at=started_at,
                usage=scanner.usage,
            )

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if content_type := upstream.headers.get("content-type"):
        headers["content-type"] = content_type
    return StreamingResponse(relay(), status_code=upstream.status_code, headers=headers)


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


async def _translate_responses_stream(
    upstream: httpx.Response,
    translator: ResponsesStreamTranslator,
) -> AsyncIterator[bytes]:
    """Re-express an upstream chat SSE stream as Responses SSE frames."""
    for frame in translator.start():
        yield frame

    buffer = b""
    try:
        async for chunk in upstream.aiter_bytes():
            buffer += chunk
            events, buffer = parse_chat_sse_frames(buffer)
            for event in events:
                for frame in translator.feed(event):
                    yield frame
        # Flush a trailing event that arrived without its final delimiter.
        if buffer.strip():
            events, _ = parse_chat_sse_frames(buffer + b"\n\n")
            for event in events:
                for frame in translator.feed(event):
                    yield frame
        for frame in translator.finish():
            yield frame
        yield b"data: [DONE]\n\n"
    finally:
        await upstream.aclose()


@router.post(RESPONSES_ENDPOINT)
async def responses(
    request: Request,
    _: GatewayKey,
    db: GatewayDb,
):
    """OpenAI Responses API endpoint backed by the managed chat runtimes."""
    try:
        body = await request.json()
    except ValueError:
        return openai_error("Request body must be valid JSON", status_code=400)
    if not isinstance(body, dict):
        return openai_error("Request body must be a JSON object", status_code=400)
    model = body.get("model")
    if not model:
        return openai_error("The model field is required", status_code=400)

    deployment = select_routed_deployment(db, str(model), "chat")
    if not deployment:
        # Not hosted here: hand the request to the configured upstream gateway,
        # which speaks the same Responses wire API.
        return await _proxy_fallback(request, db, RESPONSES_ENDPOINT, body)

    config = deployment.config if isinstance(deployment.config, Mapping) else {}
    spec = config.get("spec") if isinstance(config.get("spec"), Mapping) else {}
    try:
        chat_body, _ = responses_to_chat_request(
            body,
            model=deployment.api_model_name,
            thinking_toggle=uses_thinking_toggle(spec.get("chat_template_kwargs")),
        )
    except ResponsesTranslationError as exc:
        return openai_error(str(exc), status_code=400)

    # Only the model name counts as routing input; drop it before normalising so
    # the deployment's own api_model_name is not mistaken for a model override.
    routing_body = {key: value for key, value in chat_body.items() if key != "model"}
    adapter = adapter_for_runtime(deployment.runtime)
    routing_body = adapter.adapt_request(RESPONSES_ENDPOINT, routing_body).body
    normalized_body = normalize_reasoning_effort(
        routing_body,
        chat_template=spec.get("chat_template"),
    )

    upstream_body = {"model": deployment.api_model_name, **normalized_body}
    timeout = upstream_inference_timeout()
    client = httpx.AsyncClient(timeout=timeout, trust_env=False)
    upstream_request = client.build_request(
        "POST", f"{deployment.endpoint_url}/v1/chat/completions", json=upstream_body
    )
    try:
        upstream = await client.send(upstream_request, stream=bool(upstream_body.get("stream")))
    except httpx.HTTPError as exc:
        await client.aclose()
        return openai_error(
            f"Upstream inference service is unavailable: {exc}", status_code=502
        )

    if upstream.is_error:
        content = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(
            content=adapter.normalize_error(content, status_code=upstream.status_code),
            status_code=upstream.status_code,
            media_type="application/json",
        )

    if not upstream_body.get("stream"):
        try:
            payload = await upstream.aread()
            await upstream.aclose()
        finally:
            await client.aclose()
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return openai_error(
                "Upstream inference service returned an unreadable response", status_code=502
            )
        translated = ResponsesTurnTranslator(model=deployment.api_model_name).from_chat_response(
            parsed
        )
        return JSONResponse(content=translated)

    translator = ResponsesStreamTranslator(model=deployment.api_model_name)
    headers = {
        "content-type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _translate_responses_stream(upstream, translator),
        status_code=upstream.status_code,
        headers=headers,
    )


@router.get("/v1/responses/{response_id}")
def retrieve_response(response_id: str, _: GatewayKey, db: GatewayDb) -> Any:
    """The gateway keeps no response state, so retrieval always misses."""
    return openai_error(
        f"Response '{response_id}' was not found. The DGX Spark gateway does not "
        "persist responses; resend the full conversation instead.",
        status_code=404,
    )


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
