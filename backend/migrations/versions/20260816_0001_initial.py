"""Initial manager schema.

Revision ID: 20260816_0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "20260816_0001"
down_revision = None
branch_labels = None
depends_on = None


legacy_metadata = sa.MetaData()

model_assets = sa.Table(
    "model_assets",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("name", sa.String(255), nullable=False, index=True),
    sa.Column("alias", sa.String(255), unique=True),
    sa.Column("source", sa.String(32), nullable=False, index=True),
    sa.Column("repository_id", sa.String(255), index=True),
    sa.Column("revision", sa.String(255)),
    sa.Column("commit_hash", sa.String(255)),
    sa.Column("local_path", sa.Text(), nullable=False, unique=True),
    sa.Column("format", sa.String(64)),
    sa.Column("quantization", sa.String(64)),
    sa.Column("parameter_count", sa.String(64)),
    sa.Column("size_bytes", sa.Integer(), nullable=False),
    sa.Column("status", sa.String(32), nullable=False, index=True),
    sa.Column("capabilities", sa.JSON(), nullable=False),
    sa.Column("metadata_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

deployments = sa.Table(
    "deployments",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("name", sa.String(255), nullable=False, unique=True),
    sa.Column("model_id", sa.String(36), sa.ForeignKey("model_assets.id")),
    sa.Column("runtime", sa.String(32), nullable=False, index=True),
    sa.Column("container_id", sa.String(128), unique=True),
    sa.Column("container_name", sa.String(255), unique=True),
    sa.Column("endpoint_url", sa.Text(), nullable=False),
    sa.Column("api_model_name", sa.String(255), nullable=False, unique=True, index=True),
    sa.Column("status", sa.String(32), nullable=False, index=True),
    sa.Column("health", sa.String(32), nullable=False),
    sa.Column("managed", sa.Boolean(), nullable=False),
    sa.Column("image", sa.String(255)),
    sa.Column("port", sa.Integer()),
    sa.Column("config", sa.JSON(), nullable=False),
    sa.Column("capabilities", sa.JSON(), nullable=False),
    sa.Column("last_checked_at", sa.DateTime(timezone=True)),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

tasks = sa.Table(
    "tasks",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("type", sa.String(64), nullable=False, index=True),
    sa.Column("status", sa.String(32), nullable=False, index=True),
    sa.Column("title", sa.String(255), nullable=False),
    sa.Column("idempotency_key", sa.String(255), unique=True),
    sa.Column("progress", sa.Float(), nullable=False),
    sa.Column("completed_bytes", sa.Integer(), nullable=False),
    sa.Column("total_bytes", sa.Integer()),
    sa.Column("input_json", sa.JSON(), nullable=False),
    sa.Column("result_json", sa.JSON(), nullable=False),
    sa.Column("error", sa.Text()),
    sa.Column("log", sa.Text(), nullable=False),
    sa.Column("cancel_requested", sa.Boolean(), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True)),
    sa.Column("finished_at", sa.DateTime(timezone=True)),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

providers = sa.Table(
    "providers",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("name", sa.String(255), nullable=False, unique=True),
    sa.Column("base_url", sa.Text(), nullable=False),
    sa.Column("default_model", sa.String(255), nullable=False),
    sa.Column("encrypted_api_key", sa.Text(), nullable=False),
    sa.Column("timeout_seconds", sa.Integer(), nullable=False),
    sa.Column("headers", sa.JSON(), nullable=False),
    sa.Column("enabled", sa.Boolean(), nullable=False),
    sa.Column("last_test_status", sa.String(32)),
    sa.Column("last_tested_at", sa.DateTime(timezone=True)),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

secret_settings = sa.Table(
    "secret_settings",
    legacy_metadata,
    sa.Column("key", sa.String(128), primary_key=True),
    sa.Column("encrypted_value", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

operation_plans = sa.Table(
    "operation_plans",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("provider_id", sa.String(36), sa.ForeignKey("providers.id")),
    sa.Column("deployment_id", sa.String(36), sa.ForeignKey("deployments.id")),
    sa.Column("summary", sa.Text(), nullable=False),
    sa.Column("diagnosis", sa.Text(), nullable=False),
    sa.Column("risk", sa.String(32), nullable=False),
    sa.Column("steps", sa.JSON(), nullable=False),
    sa.Column("status", sa.String(32), nullable=False, index=True),
    sa.Column("requested_by", sa.String(255), nullable=False),
    sa.Column("approved_by", sa.String(255)),
    sa.Column("approved_at", sa.DateTime(timezone=True)),
    sa.Column("result_json", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

api_keys = sa.Table(
    "api_keys",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("name", sa.String(255), nullable=False),
    sa.Column("prefix", sa.String(24), nullable=False, index=True),
    sa.Column("key_hash", sa.String(64), nullable=False, unique=True, index=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("last_used_at", sa.DateTime(timezone=True)),
    sa.Column("revoked_at", sa.DateTime(timezone=True)),
)

audit_events = sa.Table(
    "audit_events",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("actor", sa.String(255), nullable=False, index=True),
    sa.Column("action", sa.String(128), nullable=False, index=True),
    sa.Column("resource_type", sa.String(64), nullable=False, index=True),
    sa.Column("resource_id", sa.String(255)),
    sa.Column("outcome", sa.String(32), nullable=False),
    sa.Column("source_ip", sa.String(64)),
    sa.Column("details", sa.JSON(), nullable=False),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

request_metrics = sa.Table(
    "request_metrics",
    legacy_metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("model", sa.String(255), nullable=False, index=True),
    sa.Column("endpoint", sa.String(128), nullable=False),
    sa.Column("status_code", sa.Integer(), nullable=False),
    sa.Column("latency_ms", sa.Float(), nullable=False),
    sa.Column("prompt_tokens", sa.Integer()),
    sa.Column("completion_tokens", sa.Integer()),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)


def upgrade() -> None:
    legacy_metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    legacy_metadata.drop_all(bind=op.get_bind())
