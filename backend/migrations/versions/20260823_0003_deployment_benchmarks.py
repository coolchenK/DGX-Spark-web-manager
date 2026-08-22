"""Persist deployment TPS benchmarks.

Revision ID: 20260823_0003
Revises: 20260817_0002
"""

import sqlalchemy as sa
from alembic import op

revision = "20260823_0003"
down_revision = "20260817_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.add_column(sa.Column("benchmark_status", sa.String(32)))
        batch_op.add_column(sa.Column("benchmark_tps", sa.Float()))
        batch_op.add_column(sa.Column("benchmark_completion_tokens", sa.Integer()))
        batch_op.add_column(sa.Column("benchmark_duration_seconds", sa.Float()))
        batch_op.add_column(sa.Column("benchmark_tested_at", sa.DateTime(timezone=True)))
        batch_op.add_column(sa.Column("benchmark_error", sa.Text()))


def downgrade() -> None:
    with op.batch_alter_table("deployments") as batch_op:
        batch_op.drop_column("benchmark_error")
        batch_op.drop_column("benchmark_tested_at")
        batch_op.drop_column("benchmark_duration_seconds")
        batch_op.drop_column("benchmark_completion_tokens")
        batch_op.drop_column("benchmark_tps")
        batch_op.drop_column("benchmark_status")
