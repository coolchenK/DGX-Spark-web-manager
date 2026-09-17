"""Persist the latest successful benchmark on model assets.

Revision ID: 20260823_0004
Revises: 20260823_0003
"""

import sqlalchemy as sa
from alembic import op

revision = "20260823_0004"
down_revision = "20260823_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("model_assets") as batch_op:
        batch_op.add_column(sa.Column("benchmark_tps", sa.Float()))
        batch_op.add_column(sa.Column("benchmark_tested_at", sa.DateTime(timezone=True)))

    op.execute(
        sa.text(
            """
            UPDATE model_assets
            SET benchmark_tps = (
                    SELECT deployments.benchmark_tps
                    FROM deployments
                    WHERE deployments.model_id = model_assets.id
                      AND deployments.benchmark_status = 'succeeded'
                      AND deployments.benchmark_tps IS NOT NULL
                    ORDER BY deployments.benchmark_tested_at DESC,
                             deployments.updated_at DESC
                    LIMIT 1
                ),
                benchmark_tested_at = (
                    SELECT deployments.benchmark_tested_at
                    FROM deployments
                    WHERE deployments.model_id = model_assets.id
                      AND deployments.benchmark_status = 'succeeded'
                      AND deployments.benchmark_tps IS NOT NULL
                    ORDER BY deployments.benchmark_tested_at DESC,
                             deployments.updated_at DESC
                    LIMIT 1
                )
            WHERE EXISTS (
                SELECT 1
                FROM deployments
                WHERE deployments.model_id = model_assets.id
                  AND deployments.benchmark_status = 'succeeded'
                  AND deployments.benchmark_tps IS NOT NULL
            )
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("model_assets") as batch_op:
        batch_op.drop_column("benchmark_tested_at")
        batch_op.drop_column("benchmark_tps")
