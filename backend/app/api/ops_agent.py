from typing import Literal, TypedDict

from fastapi import APIRouter, Request

from app.dependencies import Admin
from app.services.ops_agent import (
    OpsAgentProtocolError,
    OpsAgentRemoteError,
    OpsAgentUnavailable,
)

router = APIRouter(prefix="/api/ops-agent", tags=["ops-agent"])


class OpsAgentHealthResponse(TypedDict, total=False):
    status: Literal["ok", "unavailable", "error"]
    protocol_version: int
    detail: str


@router.get("/health")
def agent_health(request: Request, _: Admin) -> OpsAgentHealthResponse:
    try:
        health = request.app.state.ops_agent_client.health()
    except OpsAgentUnavailable:
        return {
            "status": "unavailable",
            "detail": "Host operations agent is unavailable",
        }
    except OpsAgentProtocolError:
        return {
            "status": "error",
            "detail": "Host operations agent returned an invalid response",
        }
    except OpsAgentRemoteError:
        return {
            "status": "error",
            "detail": "Host operations agent rejected the health check",
        }
    return {
        "status": "ok",
        "protocol_version": health.protocol_version,
    }
