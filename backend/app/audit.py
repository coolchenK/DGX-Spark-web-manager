from typing import Any

from fastapi import APIRouter, Query, Request
from sqlalchemy import desc, select

from app.dependencies import Admin, DbSession
from app.models import AuditEvent

router = APIRouter(prefix="/api/audit", tags=["audit"])


def record_audit(
    db: DbSession,
    *,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    outcome: str = "success",
    source_ip: str | None = None,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        actor=actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        source_ip=source_ip,
        details=details or {},
    )
    db.add(event)
    return event


@router.get("")
def list_audit_events(
    request: Request,
    db: DbSession,
    _: Admin,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    events = db.scalars(select(AuditEvent).order_by(desc(AuditEvent.created_at)).limit(limit))
    return [
        {
            "id": event.id,
            "actor": event.actor,
            "action": event.action,
            "resource_type": event.resource_type,
            "resource_id": event.resource_id,
            "outcome": event.outcome,
            "source_ip": event.source_ip,
            "details": event.details,
            "created_at": event.created_at,
        }
        for event in events
    ]

