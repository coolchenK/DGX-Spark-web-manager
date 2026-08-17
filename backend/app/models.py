from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
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
    ops_sessions: Mapped[list[OpsSession]] = relationship(
        back_populates="deployment",
        passive_deletes=True,
        order_by=lambda: (OpsSession.created_at, OpsSession.id),
    )


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
    last_test_result: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, server_default="{}"
    )
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    ops_sessions: Mapped[list[OpsSession]] = relationship(
        back_populates="provider",
        passive_deletes=True,
        order_by=lambda: (OpsSession.created_at, OpsSession.id),
    )


class SecretSetting(Base):
    __tablename__ = "secret_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    encrypted_value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


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

    messages: Mapped[list[OpsMessage]] = relationship(
        back_populates="operation_plan",
        passive_deletes=True,
        order_by=lambda: (OpsMessage.created_at, OpsMessage.id),
    )


class OpsSession(TimestampMixin, Base):
    __tablename__ = "ops_sessions"
    __table_args__ = (
        Index("ix_ops_sessions_status_updated_at", "status", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255))
    provider_id: Mapped[str | None] = mapped_column(
        ForeignKey("providers.id", ondelete="SET NULL"), index=True
    )
    deployment_id: Mapped[str | None] = mapped_column(
        ForeignKey("deployments.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="active")
    requested_by: Mapped[str] = mapped_column(String(255), default="admin")

    provider: Mapped[Provider | None] = relationship(back_populates="ops_sessions")
    deployment: Mapped[Deployment | None] = relationship(back_populates="ops_sessions")
    messages: Mapped[list[OpsMessage]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by=lambda: (OpsMessage.created_at, OpsMessage.id),
    )
    tool_runs: Mapped[list[OpsToolRun]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by=lambda: (OpsToolRun.created_at, OpsToolRun.id),
    )


class OpsMessage(TimestampMixin, Base):
    __tablename__ = "ops_messages"
    __table_args__ = (
        Index("ix_ops_messages_session_created_at", "session_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ops_sessions.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    operation_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("operation_plans.id", ondelete="SET NULL"), index=True
    )

    session: Mapped[OpsSession] = relationship(back_populates="messages")
    operation_plan: Mapped[OperationPlan | None] = relationship(back_populates="messages")


class OpsToolRun(TimestampMixin, Base):
    __tablename__ = "ops_tool_runs"
    __table_args__ = (
        Index(
            "ix_ops_tool_runs_session_status_created_at",
            "session_id",
            "status",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("ops_sessions.id", ondelete="CASCADE")
    )
    tool_name: Mapped[str] = mapped_column(String(128))
    risk: Mapped[str] = mapped_column(String(32), default="read_only")
    status: Mapped[str] = mapped_column(String(32), default="queued")
    arguments_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    agent_job_id: Mapped[str | None] = mapped_column(String(128), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[OpsSession] = relationship(back_populates="tool_runs")


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
