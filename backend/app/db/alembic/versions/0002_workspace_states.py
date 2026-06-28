"""create workspace states table

Revision ID: 0002_workspace_states
Revises: 0001_metadata_tables
Create Date: 2026-06-29
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002_workspace_states"
down_revision: str | None = "0001_metadata_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    payload_type = JSON().with_variant(JSONB, "postgresql")

    op.create_table(
        "workspace_states",
        Column("record_id", String(128), primary_key=True),
        Column("dataset_id", String(128), nullable=True),
        Column("secondary_id", String(128), nullable=True),
        Column("status", String(64), nullable=True),
        Column("payload", payload_type, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=True),
        Column("updated_at", DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_workspace_states_dataset_id", "workspace_states", ["dataset_id"])
    op.create_index("ix_workspace_states_secondary_id", "workspace_states", ["secondary_id"])
    op.create_index("ix_workspace_states_status", "workspace_states", ["status"])
    op.create_index("ix_workspace_states_created_at", "workspace_states", ["created_at"])
    op.create_index(
        "ix_workspace_states_dataset_created",
        "workspace_states",
        ["dataset_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("workspace_states")
