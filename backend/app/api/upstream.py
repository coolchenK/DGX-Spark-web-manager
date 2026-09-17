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
    bounded_probe_detail,
    clear_upstream_api_key,
    clear_upstream_gateway,
    fetch_upstream_models,
    resolve_upstream_gateway,
    set_upstream_api_key,
    set_upstream_base_url,
    validate_upstream_base_url,
)

router = APIRouter(prefix="/api/gateway/upstream", tags=["upstream-gateway"])


class UpstreamUpdate(BaseModel):
    base_url: str | None = Field(default=None, max_length=600)
    api_key: str | None = Field(default=None, max_length=4096)


def _serialize(request: Request, db: DbSession, admin: Any = None) -> dict[str, Any]:
    del admin
    config = resolve_upstream_gateway(db, request.app.state.secret_box, request.app.state.settings)
    if config is None:
        return {
            "base_url": None,
            "api_key_configured": False,
            "source": "unset",
            "enabled": False,
        }
    return {
        "base_url": config.base_url,
        "api_key_configured": bool(config.api_key),
        "source": config.source,
        "enabled": True,
    }


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
