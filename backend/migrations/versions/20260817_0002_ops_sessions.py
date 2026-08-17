"""Persist AI operations sessions.

Revision ID: 20260817_0002
Revises: 20260816_0001
"""

import sqlalchemy as sa
from alembic import context, op

revision = "20260817_0002"
down_revision = "20260816_0001"
branch_labels = None
depends_on = None

FOREIGN_KEY_NAMING_CONVENTION = {
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"
}


def _foreign_key_name(
    table_name: str,
    column_name: str,
    referred_table: str,
    *,
    legacy_schema: bool,
) -> str:
    deterministic_name = f"fk_{table_name}_{column_name}_{referred_table}"
    bind = op.get_bind()
    if context.is_offline_mode():
        if bind.dialect.name == "postgresql" and legacy_schema:
            return f"{table_name}_{column_name}_fkey"
        return deterministic_name
    if bind.dialect.name == "sqlite":
        return deterministic_name
    for foreign_key in sa.inspect(bind).get_foreign_keys(table_name):
        if (
            foreign_key["constrained_columns"] == [column_name]
            and foreign_key["referred_table"] == referred_table
        ):
            return foreign_key["name"] or deterministic_name
    raise RuntimeError(f"Missing legacy foreign key for {table_name}.{column_name}")


def _replace_legacy_foreign_key(
    table_name: str,
    column_name: str,
    referred_table: str,
    *,
    ondelete: str | None,
    legacy_schema: bool,
) -> None:
    old_name = _foreign_key_name(
        table_name,
        column_name,
        referred_table,
        legacy_schema=legacy_schema,
    )
    new_name = f"fk_{table_name}_{column_name}_{referred_table}"
    with op.batch_alter_table(
        table_name, naming_convention=FOREIGN_KEY_NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(old_name, type_="foreignkey")
        batch_op.create_foreign_key(
            new_name,
            referred_table,
            [column_name],
            ["id"],
            ondelete=ondelete,
        )


def _repair_legacy_orphans() -> None:
    op.execute(
        sa.text(
            "UPDATE deployments SET model_id = NULL "
            "WHERE model_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM model_assets WHERE model_assets.id = deployments.model_id)"
        )
    )
    op.execute(
        sa.text(
            "UPDATE operation_plans SET provider_id = NULL "
            "WHERE provider_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM providers WHERE providers.id = operation_plans.provider_id)"
        )
    )
    op.execute(
        sa.text(
            "UPDATE operation_plans SET deployment_id = NULL "
            "WHERE deployment_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM deployments "
            "WHERE deployments.id = operation_plans.deployment_id)"
        )
    )


def _verify_sqlite_foreign_keys() -> None:
    bind = op.get_bind()
    if context.is_offline_mode() or bind.dialect.name != "sqlite":
        return
    violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
    if violations:
        sample = ", ".join(str(tuple(row)) for row in violations[:10])
        raise RuntimeError(f"SQLite foreign key check failed after migration: {sample}")


def upgrade() -> None:
    _repair_legacy_orphans()
    _replace_legacy_foreign_key(
        "deployments",
        "model_id",
        "model_assets",
        ondelete="SET NULL",
        legacy_schema=True,
    )
    _replace_legacy_foreign_key(
        "operation_plans",
        "provider_id",
        "providers",
        ondelete="SET NULL",
        legacy_schema=True,
    )
    _replace_legacy_foreign_key(
        "operation_plans",
        "deployment_id",
        "deployments",
        ondelete="SET NULL",
        legacy_schema=True,
    )

    op.add_column(
        "providers",
        sa.Column(
            "last_test_result",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.create_table(
        "ops_sessions",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("provider_id", sa.String(36)),
        sa.Column("deployment_id", sa.String(36)),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_by", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["provider_id"], ["providers.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["deployment_id"], ["deployments.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ops_sessions_provider_id", "ops_sessions", ["provider_id"])
    op.create_index("ix_ops_sessions_deployment_id", "ops_sessions", ["deployment_id"])
    op.create_index(
        "ix_ops_sessions_status_updated_at", "ops_sessions", ["status", "updated_at"]
    )

    op.create_table(
        "ops_messages",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("operation_plan_id", sa.String(36)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["ops_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["operation_plan_id"], ["operation_plans.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ops_messages_operation_plan_id", "ops_messages", ["operation_plan_id"]
    )
    op.create_index(
        "ix_ops_messages_session_created_at",
        "ops_messages",
        ["session_id", "created_at"],
    )

    op.create_table(
        "ops_tool_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("tool_name", sa.String(128), nullable=False),
        sa.Column("risk", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("arguments_json", sa.JSON(), nullable=False),
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.Column("agent_job_id", sa.String(128)),
        sa.Column("error", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["ops_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ops_tool_runs_agent_job_id", "ops_tool_runs", ["agent_job_id"])
    op.create_index(
        "ix_ops_tool_runs_session_status_created_at",
        "ops_tool_runs",
        ["session_id", "status", "created_at"],
    )
    _verify_sqlite_foreign_keys()


def downgrade() -> None:
    op.drop_index("ix_ops_tool_runs_session_status_created_at", table_name="ops_tool_runs")
    op.drop_index("ix_ops_tool_runs_agent_job_id", table_name="ops_tool_runs")
    op.drop_table("ops_tool_runs")

    op.drop_index("ix_ops_messages_session_created_at", table_name="ops_messages")
    op.drop_index("ix_ops_messages_operation_plan_id", table_name="ops_messages")
    op.drop_table("ops_messages")

    op.drop_index("ix_ops_sessions_status_updated_at", table_name="ops_sessions")
    op.drop_index("ix_ops_sessions_deployment_id", table_name="ops_sessions")
    op.drop_index("ix_ops_sessions_provider_id", table_name="ops_sessions")
    op.drop_table("ops_sessions")

    _replace_legacy_foreign_key(
        "operation_plans",
        "deployment_id",
        "deployments",
        ondelete=None,
        legacy_schema=False,
    )
    _replace_legacy_foreign_key(
        "operation_plans",
        "provider_id",
        "providers",
        ondelete=None,
        legacy_schema=False,
    )
    _replace_legacy_foreign_key(
        "deployments",
        "model_id",
        "model_assets",
        ondelete=None,
        legacy_schema=False,
    )

    with op.batch_alter_table("providers") as batch_op:
        batch_op.drop_column("last_test_result")
