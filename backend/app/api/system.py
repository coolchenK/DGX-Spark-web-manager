from typing import Any

from fastapi import APIRouter, Request

from app.dependencies import Admin

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("")
def system_snapshot(request: Request, _: Admin) -> dict[str, Any]:
    return request.app.state.system_service.snapshot()

