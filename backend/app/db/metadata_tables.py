from datetime import UTC, datetime

from sqlalchemy import JSON, Column, DateTime, Index, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

JSON_PAYLOAD_TYPE = JSON().with_variant(JSONB, "postgresql")

metadata = MetaData()


def _metadata_table(name: str) -> Table:
    table = Table(
        name,
        metadata,
        Column("record_id", String(128), primary_key=True),
        Column("dataset_id", String(128), nullable=True, index=True),
        Column("secondary_id", String(128), nullable=True, index=True),
        Column("status", String(64), nullable=True, index=True),
        Column("payload", JSON_PAYLOAD_TYPE, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=True, index=True),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            default=lambda: datetime.now(UTC),
        ),
    )
    Index(f"ix_{name}_dataset_created", table.c.dataset_id, table.c.created_at)
    return table


DATASETS_TABLE = _metadata_table("datasets")
DATASET_VERSIONS_TABLE = _metadata_table("dataset_versions")
MODELS_TABLE = _metadata_table("models")
ML_EXPERIMENTS_TABLE = _metadata_table("ml_experiments")
REPORTS_TABLE = _metadata_table("reports")
DRIFT_REPORTS_TABLE = _metadata_table("drift_reports")
AGENT_RUNS_TABLE = _metadata_table("agent_runs")
AGENT_JOBS_TABLE = _metadata_table("agent_jobs")
BACKTEST_JOBS_TABLE = _metadata_table("backtest_jobs")
WORKSPACE_STATES_TABLE = _metadata_table("workspace_states")


def create_metadata_tables(engine: Engine) -> None:
    metadata.create_all(engine)
