"""Persist AI operations sessions.

Revision ID: 20260817_0002
Revises: 20260816_0001
"""

import sqlalchemy as sa
from alembic import op

revision = "20260817_0002"
down_revision = "20260816_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
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

    with op.batch_alter_table("providers") as batch_op:
        batch_op.drop_column("last_test_result")
