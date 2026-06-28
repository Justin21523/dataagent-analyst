"""create backtest jobs table

Revision ID: 0003_backtest_jobs
Revises: 0002_workspace_states
Create Date: 2026-06-29
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003_backtest_jobs"
down_revision: str | None = "0002_workspace_states"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    payload_type = JSON().with_variant(JSONB, "postgresql")

    op.create_table(
        "backtest_jobs",
        Column("record_id", String(128), primary_key=True),
        Column("dataset_id", String(128), nullable=True),
        Column("secondary_id", String(128), nullable=True),
        Column("status", String(64), nullable=True),
        Column("payload", payload_type, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=True),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_backtest_jobs_dataset_id", "backtest_jobs", ["dataset_id"])
    op.create_index("ix_backtest_jobs_secondary_id", "backtest_jobs", ["secondary_id"])
    op.create_index("ix_backtest_jobs_status", "backtest_jobs", ["status"])
    op.create_index("ix_backtest_jobs_created_at", "backtest_jobs", ["created_at"])
    op.create_index(
        "ix_backtest_jobs_dataset_created",
        "backtest_jobs",
        ["dataset_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("backtest_jobs")
