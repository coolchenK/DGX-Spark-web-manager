from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.dependencies import Admin, get_db
from app.gateway.proxy import openai_error, proxy_openai_request
from app.models import ApiKey, Deployment, RequestMetric
from app.security import hash_api_key

router = APIRouter(tags=["openai-gateway"])
GatewayDb = Annotated[Session, Depends(get_db)]


class GatewayAuthError(Exception):
    pass


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


@router.get("/v1/models")
def openai_models(_: GatewayKey, db: GatewayDb) -> dict[str, Any]:
    deployments = db.scalars(
        select(Deployment).where(
            Deployment.status == "running",
            Deployment.health == "healthy",
        )
    )
    return {
        "object": "list",
        "data": [
            {
                "id": item.api_model_name,
                "object": "model",
                "created": int(item.created_at.timestamp()),
                "owned_by": "dgx-spark-manager",
                "root": item.api_model_name,
                "capabilities": item.capabilities,
            }
            for item in deployments
        ],
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
    deployment = db.scalar(
        select(Deployment).where(
            Deployment.api_model_name == model,
            Deployment.status == "running",
            Deployment.health == "healthy",
        )
    )
    if not deployment:
        return openai_error(f"Model '{model}' was not found or is not healthy", status_code=404)
    if required_capability not in deployment.capabilities:
        return openai_error(
            f"Model '{model}' does not support {required_capability}", status_code=400
        )
    return await proxy_openai_request(request, deployment, endpoint, dict(body))


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
def gateway_stats(_: Admin, db: GatewayDb) -> dict[str, Any]:
    total = db.scalar(select(func.count(RequestMetric.id))) or 0
    failed = (
        db.scalar(select(func.count(RequestMetric.id)).where(RequestMetric.status_code >= 400)) or 0
    )
    avg_latency = db.scalar(select(func.avg(RequestMetric.latency_ms))) or 0
    prompt_tokens = db.scalar(select(func.sum(RequestMetric.prompt_tokens))) or 0
    completion_tokens = db.scalar(select(func.sum(RequestMetric.completion_tokens))) or 0
    return {
        "total_requests": total,
        "failed_requests": failed,
        "error_rate": failed / total if total else 0,
        "average_latency_ms": round(float(avg_latency), 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
