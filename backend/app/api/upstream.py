"""Administrator endpoints for the upstream OpenAI-compatible gateway.

The upstream is used when a request names a model this manager does not host.
Nothing here ever returns the stored API key: the panel learns only whether a
key is configured.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.services.upstream_gateway import (
    UpstreamGatewayConfig,
    UpstreamModelCache,
    bounded_probe_detail,
    clear_upstream_api_key,
    clear_upstream_gateway,
    fetch_upstream_models,
    read_upstream_exposure,
    resolve_upstream_gateway,
    set_upstream_api_key,
    set_upstream_base_url,
    validate_upstream_base_url,
    write_upstream_exposure,
)

router = APIRouter(prefix="/api/gateway/upstream", tags=["upstream-gateway"])


class UpstreamUpdate(BaseModel):
    base_url: str | None = Field(default=None, max_length=600)
    api_key: str | None = Field(default=None, max_length=4096)


class UpstreamExposureUpdate(BaseModel):
    expose_all: bool
    selected_models: list[str] = Field(default_factory=list)


def _serialize(request: Request, db: DbSession, admin: Any = None) -> dict[str, Any]:
    del admin
    box = request.app.state.secret_box
    settings = request.app.state.settings
    exposure = read_upstream_exposure(db, box)
    config = resolve_upstream_gateway(db, box, settings)
    payload = {
        "expose_all": exposure.expose_all,
        "selected_models": sorted(exposure.selected_models),
    }
    if config is None:
        return {
            "base_url": None,
            "api_key_configured": False,
            "source": "unset",
            "enabled": False,
            **payload,
        }
    return {
        "base_url": config.base_url,
        "api_key_configured": bool(config.api_key),
        "source": config.source,
        "enabled": True,
        **payload,
    }


def _cached_models(
    request: Request, config: UpstreamGatewayConfig
) -> tuple[str, list[dict[str, Any]], str | None]:
    """Return the upstream model list, using the shared cache when possible."""
    cache: UpstreamModelCache = request.app.state.upstream_model_cache
    models = cache.get(config.cache_key)
    if models is not None:
        return "ok", models, None
    try:
        models = fetch_upstream_models(config)
    except (httpx.HTTPError, ValueError) as exc:
        return "unavailable", [], bounded_probe_detail(exc)
    cache.set(config.cache_key, models)
    return "ok", models, None


@router.get("")
def get_upstream(request: Request, db: DbSession, _: Admin) -> dict[str, Any]:
    return _serialize(request, db)


@router.put("")
def put_upstream(
    payload: UpstreamUpdate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    secret_box = request.app.state.secret_box
    provided = payload.model_fields_set
    try:
        if "base_url" in provided:
            if payload.base_url is None or not payload.base_url.strip():
                # An explicit null clears the stored configuration so the
                # environment default applies again.
                clear_upstream_gateway(db)
            else:
                set_upstream_base_url(
                    db,
                    secret_box,
                    validate_upstream_base_url(payload.base_url),
                    api_key=payload.api_key,
                )
        elif "api_key" in provided:
            if payload.api_key:
                set_upstream_api_key(db, secret_box, payload.api_key)
            else:
                clear_upstream_api_key(db)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request.app.state.upstream_model_cache.invalidate()
    result = _serialize(request, db)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="gateway.upstream.update",
        resource_type="gateway",
        details={
            "base_url": result["base_url"],
            "api_key_configured": result["api_key_configured"],
            "source": result["source"],
        },
    )
    db.commit()
    return result


@router.get("/models")
def list_upstream_models(request: Request, db: DbSession, _: Admin) -> dict[str, Any]:
    """List the models the upstream publishes together with their exposure state."""
    config = resolve_upstream_gateway(
        db, request.app.state.secret_box, request.app.state.settings
    )
    if config is None:
        return {"status": "unset", "models": [], "detail": None}

    status_value, models, detail = _cached_models(request, config)
    return {
        "status": status_value,
        "detail": detail,
        "models": [
            {
                "id": raw["id"],
                "exposed": config.allows_model(raw["id"]),
            }
            for raw in models
        ],
    }


@router.put("/exposure")
def put_upstream_exposure(
    payload: UpstreamExposureUpdate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    """Choose which upstream models this gateway advertises and serves."""
    try:
        exposure = write_upstream_exposure(
            db,
            request.app.state.secret_box,
            expose_all=payload.expose_all,
            selected_models=payload.selected_models,
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request.app.state.upstream_model_cache.invalidate()
    record_audit(
        db,
        actor=str(admin["username"]),
        action="gateway.upstream.exposure.update",
        resource_type="gateway",
        details={
            "expose_all": exposure.expose_all,
            "selected_count": len(exposure.selected_models),
        },
    )
    db.commit()
    return {
        "expose_all": exposure.expose_all,
        "selected_models": sorted(exposure.selected_models),
    }


@router.post("/test")
def test_upstream(request: Request, db: DbSession, admin: CsrfAdmin) -> dict[str, Any]:
    started_at = time.perf_counter()
    config = resolve_upstream_gateway(db, request.app.state.secret_box, request.app.state.settings)
    if config is None:
        return {"status": "unset", "latency_ms": 0, "model_count": 0, "detail": None}

    try:
        models = fetch_upstream_models(config)
    except (httpx.HTTPError, ValueError) as exc:
        result: dict[str, Any] = {
            "status": "unavailable",
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "model_count": 0,
            "detail": bounded_probe_detail(exc),
        }
    else:
        result = {
            "status": "ok",
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "model_count": len(models),
            "detail": None,
        }

    record_audit(
        db,
        actor=str(admin["username"]),
        action="gateway.upstream.test",
        resource_type="gateway",
        outcome="success" if result["status"] == "ok" else "failed",
        details={"status": result["status"], "model_count": result["model_count"]},
    )
    db.commit()
    return result
