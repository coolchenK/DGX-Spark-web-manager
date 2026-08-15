from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ModelAsset(TimestampMixin, Base):
    __tablename__ = "model_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), index=True)
    alias: Mapped[str | None] = mapped_column(String(255), unique=True)
    source: Mapped[str] = mapped_column(String(32), default="local", index=True)
    repository_id: Mapped[str | None] = mapped_column(String(255), index=True)
    revision: Mapped[str | None] = mapped_column(String(255))
    commit_hash: Mapped[str | None] = mapped_column(String(255))
    local_path: Mapped[str] = mapped_column(Text, unique=True)
    format: Mapped[str | None] = mapped_column(String(64))
    quantization: Mapped[str | None] = mapped_column(String(64))
    parameter_count: Mapped[str | None] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="available", index=True)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    deployments: Mapped[list[Deployment]] = relationship(back_populates="model")


class Deployment(TimestampMixin, Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    model_id: Mapped[str | None] = mapped_column(ForeignKey("model_assets.id"))
    runtime: Mapped[str] = mapped_column(String(32), index=True)
    container_id: Mapped[str | None] = mapped_column(String(128), unique=True)
    container_name: Mapped[str | None] = mapped_column(String(255), unique=True)
    endpoint_url: Mapped[str] = mapped_column(Text)
    api_model_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    health: Mapped[str] = mapped_column(String(32), default="unknown")
    managed: Mapped[bool] = mapped_column(Boolean, default=False)
    image: Mapped[str | None] = mapped_column(String(255))
    port: Mapped[int | None] = mapped_column(Integer)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    capabilities: Mapped[list[str]] = mapped_column(JSON, default=list)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    model: Mapped[ModelAsset | None] = relationship(back_populates="deployments")


class TaskRecord(TimestampMixin, Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    title: Mapped[str] = mapped_column(String(255))
    idempotency_key: Mapped[str | None] = mapped_column(String(255), unique=True)
    progress: Mapped[float] = mapped_column(Float, default=0)
    completed_bytes: Mapped[int] = mapped_column(Integer, default=0)
    total_bytes: Mapped[int | None] = mapped_column(Integer)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    log: Mapped[str] = mapped_column(Text, default="")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Provider(TimestampMixin, Base):
    __tablename__ = "providers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    base_url: Mapped[str] = mapped_column(Text)
    default_model: Mapped[str] = mapped_column(String(255))
    encrypted_api_key: Mapped[str] = mapped_column(Text)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=60)
    headers: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_test_status: Mapped[str | None] = mapped_column(String(32))
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OperationPlan(TimestampMixin, Base):
    __tablename__ = "operation_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    provider_id: Mapped[str | None] = mapped_column(ForeignKey("providers.id"))
    deployment_id: Mapped[str | None] = mapped_column(ForeignKey("deployments.id"))
    summary: Mapped[str] = mapped_column(Text)
    diagnosis: Mapped[str] = mapped_column(Text)
    risk: Mapped[str] = mapped_column(String(32), default="low")
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    requested_by: Mapped[str] = mapped_column(String(255), default="admin")
    approved_by: Mapped[str | None] = mapped_column(String(255))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(255))
    prefix: Mapped[str] = mapped_column(String(24), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    actor: Mapped[str] = mapped_column(String(255), index=True)
    action: Mapped[str] = mapped_column(String(128), index=True)
    resource_type: Mapped[str] = mapped_column(String(64), index=True)
    resource_id: Mapped[str | None] = mapped_column(String(255))
    outcome: Mapped[str] = mapped_column(String(32), default="success")
    source_ip: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RequestMetric(Base):
    __tablename__ = "request_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model: Mapped[str] = mapped_column(String(255), index=True)
    endpoint: Mapped[str] = mapped_column(String(128))
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

