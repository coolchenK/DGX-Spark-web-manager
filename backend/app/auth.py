import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.audit import record_audit
from app.dependencies import Admin, CsrfAdmin, DbSession
from app.models import ApiKey
from app.security import create_api_key, hash_api_key

router = APIRouter(tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


@router.post("/api/auth/login")
def login(payload: LoginRequest, request: Request, response: Response, db: DbSession) -> dict:
    settings = request.app.state.settings
    valid = payload.username == settings.admin_username
    valid = request.app.state.password_manager.verify(payload.password) and valid
    source_ip = request.client.host if request.client else None
    if not valid:
        record_audit(
            db,
            actor=payload.username or "unknown",
            action="auth.login",
            resource_type="session",
            outcome="denied",
            source_ip=source_ip,
        )
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    csrf_token = secrets.token_urlsafe(32)
    session_token = request.app.state.session_manager.create(
        {"username": settings.admin_username, "role": "admin", "csrf": csrf_token}
    )
    response.set_cookie(
        "dgx_session",
        session_token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        max_age=settings.session_ttl_seconds,
        path="/",
    )
    response.set_cookie(
        "dgx_csrf",
        csrf_token,
        httponly=False,
        secure=settings.cookie_secure,
        samesite="strict",
        max_age=settings.session_ttl_seconds,
        path="/",
    )
    record_audit(
        db,
        actor=settings.admin_username,
        action="auth.login",
        resource_type="session",
        source_ip=source_ip,
    )
    db.commit()
    return {
        "user": {"username": settings.admin_username, "role": "admin"},
        "csrf_token": csrf_token,
    }


@router.get("/api/auth/me")
def me(admin: Admin) -> dict[str, str]:
    return {"username": str(admin["username"]), "role": str(admin["role"])}


@router.get("/api/auth/session")
def session_status(request: Request) -> dict:
    token = request.cookies.get("dgx_session")
    payload = request.app.state.session_manager.load(token) if token else None
    if not payload or payload.get("role") != "admin":
        return {"authenticated": False, "user": None, "csrf_token": None}
    return {
        "authenticated": True,
        "user": {
            "username": str(payload["username"]),
            "role": str(payload["role"]),
        },
        "csrf_token": str(payload["csrf"]),
    }


@router.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, _: CsrfAdmin) -> None:
    response.delete_cookie("dgx_session", path="/")
    response.delete_cookie("dgx_csrf", path="/")


@router.post("/api/keys", status_code=status.HTTP_201_CREATED)
def create_key(
    payload: ApiKeyCreate,
    request: Request,
    db: DbSession,
    admin: CsrfAdmin,
) -> dict:
    value = create_api_key()
    api_key = ApiKey(
        name=payload.name.strip(),
        prefix=value[:12],
        key_hash=hash_api_key(value),
    )
    db.add(api_key)
    db.flush()
    record_audit(
        db,
        actor=str(admin["username"]),
        action="api_key.create",
        resource_type="api_key",
        resource_id=api_key.id,
        source_ip=request.client.host if request.client else None,
        details={"name": api_key.name, "prefix": api_key.prefix},
    )
    db.commit()
    db.refresh(api_key)
    return {
        "id": api_key.id,
        "name": api_key.name,
        "prefix": api_key.prefix,
        "key": value,
        "created_at": api_key.created_at,
    }


@router.get("/api/keys")
def list_keys(db: DbSession, _: Admin) -> list[dict]:
    keys = db.scalars(select(ApiKey).order_by(ApiKey.created_at.desc()))
    return [
        {
            "id": item.id,
            "name": item.name,
            "prefix": item.prefix,
            "created_at": item.created_at,
            "last_used_at": item.last_used_at,
            "revoked_at": item.revoked_at,
        }
        for item in keys
    ]


@router.delete("/api/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(key_id: str, db: DbSession, admin: CsrfAdmin) -> None:
    api_key = db.get(ApiKey, key_id)
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    if api_key.revoked_at is None:
        api_key.revoked_at = datetime.now(UTC)
        record_audit(
            db,
            actor=str(admin["username"]),
            action="api_key.revoke",
            resource_type="api_key",
            resource_id=api_key.id,
        )
        db.commit()
