from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Annotated, Any

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
        and (
            required_capability is None
            or required_capability in deployment.capabilities
        )
    ]
    if not candidates:
        return None
    with _route_lock:
        position = _route_positions[model]
        _route_positions[model] = position + 1
    return candidates[position % len(candidates)]


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


@router.get("/v1/models")
def openai_models(_: GatewayKey, db: GatewayDb) -> dict[str, Any]:
    deployments = list(db.scalars(
        select(Deployment).where(
            Deployment.status == "running",
            Deployment.health == "healthy",
        )
    ))
    routes: dict[str, dict[str, Any]] = {}
    for deployment in deployments:
        route_name = deployment_route_name(deployment)
        if route_name not in routes:
            routes[route_name] = {
                "id": route_name,
                "object": "model",
                "created": int(deployment.created_at.timestamp()),
                "owned_by": "dgx-spark-manager",
                "root": route_name,
                "capabilities": list(deployment.capabilities),
                "instances": 1,
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
    merged_body, applied = merge_generation_defaults(
        endpoint,
        dict(body),
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
