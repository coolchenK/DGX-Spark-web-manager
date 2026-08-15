from collections.abc import Generator
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import Settings
from app.security import SessionManager


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Generator[Session, None, None]:
    yield from request.app.state.database.session()


def require_admin(request: Request) -> dict[str, Any]:
    token = request.cookies.get("dgx_session")
    payload = request.app.state.session_manager.load(token) if token else None
    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return payload


def require_csrf(
    request: Request,
    admin: Annotated[dict[str, Any], Depends(require_admin)],
    csrf_header: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> dict[str, Any]:
    manager: SessionManager = request.app.state.session_manager
    if not manager.constant_time_equal(admin.get("csrf"), csrf_header):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token is missing or invalid",
        )
    return admin


DbSession = Annotated[Session, Depends(get_db)]
Admin = Annotated[dict[str, Any], Depends(require_admin)]
CsrfAdmin = Annotated[dict[str, Any], Depends(require_csrf)]

