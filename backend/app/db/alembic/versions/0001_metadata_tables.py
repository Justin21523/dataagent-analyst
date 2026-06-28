"""create metadata tables

Revision ID: 0001_metadata_tables
Revises:
Create Date: 2026-06-29
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001_metadata_tables"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = [
    "datasets",
    "dataset_versions",
    "models",
    "ml_experiments",
    "reports",
    "drift_reports",
    "agent_runs",
    "agent_jobs",
]


def upgrade() -> None:
    payload_type = JSON().with_variant(JSONB, "postgresql")

    for table_name in TABLES:
        op.create_table(
            table_name,
            Column("record_id", String(128), primary_key=True),
            Column("dataset_id", String(128), nullable=True),
            Column("secondary_id", String(128), nullable=True),
            Column("status", String(64), nullable=True),
            Column("payload", payload_type, nullable=False),
            Column("created_at", DateTime(timezone=True), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )
        op.create_index(f"ix_{table_name}_dataset_id", table_name, ["dataset_id"])
        op.create_index(f"ix_{table_name}_secondary_id", table_name, ["secondary_id"])
        op.create_index(f"ix_{table_name}_status", table_name, ["status"])
        op.create_index(f"ix_{table_name}_created_at", table_name, ["created_at"])
        op.create_index(
            f"ix_{table_name}_dataset_created",
            table_name,
            ["dataset_id", "created_at"],
        )


def downgrade() -> None:
    for table_name in reversed(TABLES):
        op.drop_table(table_name)
