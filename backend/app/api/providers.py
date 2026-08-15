from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import Provider
from app.services.providers import validate_custom_headers

router = APIRouter(prefix="/api/providers", tags=["providers"])


class ProviderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    base_url: str
    api_key: str = Field(min_length=1)
    default_model: str = Field(min_length=1, max_length=255)
    timeout_seconds: int = Field(default=60, ge=5, le=600)
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    _validate_headers = field_validator("headers")(validate_custom_headers)


class ProviderUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    base_url: str | None = None
    api_key: str | None = None
    default_model: str | None = None
    timeout_seconds: int | None = Field(default=None, ge=5, le=600)
    headers: dict[str, str] | None = None
    enabled: bool | None = None

    _validate_headers = field_validator("headers")(validate_custom_headers)


@router.get("")
def list_providers(request: Request, db: DbSession, _: Admin) -> list[dict[str, Any]]:
    providers = db.scalars(select(Provider).order_by(Provider.name))
    return [request.app.state.provider_service.serialize(item) for item in providers]


@router.post("", status_code=status.HTTP_201_CREATED)
def create_provider(
    payload: ProviderCreate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    try:
        provider = request.app.state.provider_service.create(db, **payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record_audit(
        db,
        actor=str(admin["username"]),
        action="provider.create",
        resource_type="provider",
        resource_id=provider.id,
        details={"name": provider.name, "base_url": provider.base_url},
    )
    db.commit()
    return request.app.state.provider_service.serialize(provider)


@router.patch("/{provider_id}")
def update_provider(
    provider_id: str,
    payload: ProviderUpdate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    provider = db.get(Provider, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    values = payload.model_dump(exclude_unset=True)
    api_key = values.pop("api_key", None)
    if "base_url" in values:
        from app.services.providers import normalize_openai_base_url, validate_provider_url

        validate_provider_url(values["base_url"])
        values["base_url"] = normalize_openai_base_url(values["base_url"])
    for key, value in values.items():
        setattr(provider, key, value)
    if api_key:
        request.app.state.provider_service.update_secret(provider, api_key)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="provider.update",
        resource_type="provider",
        resource_id=provider.id,
    )
    db.commit()
    return request.app.state.provider_service.serialize(provider)


@router.post("/{provider_id}/test")
def test_provider(
    provider_id: str,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict[str, Any]:
    provider = db.get(Provider, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    result = request.app.state.provider_service.test(db, provider)
    record_audit(
        db,
        actor=str(admin["username"]),
        action="provider.test",
        resource_type="provider",
        resource_id=provider.id,
        outcome="success" if result["status"] == "healthy" else "failed",
    )
    db.commit()
    return result


@router.delete("/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_provider(provider_id: str, db: DbSession, admin: CsrfAdmin) -> None:
    provider = db.get(Provider, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    record_audit(
        db,
        actor=str(admin["username"]),
        action="provider.delete",
        resource_type="provider",
        resource_id=provider.id,
        details={"name": provider.name},
    )
    db.delete(provider)
    db.commit()

