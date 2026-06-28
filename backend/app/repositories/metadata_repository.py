import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from sqlalchemy import Table, delete, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.app.core.config import Settings
from backend.app.db.metadata_tables import (
    AGENT_JOBS_TABLE,
    AGENT_RUNS_TABLE,
    BACKTEST_JOBS_TABLE,
    DATASET_VERSIONS_TABLE,
    DATASETS_TABLE,
    DRIFT_REPORTS_TABLE,
    ML_EXPERIMENTS_TABLE,
    MODELS_TABLE,
    REPORTS_TABLE,
    WORKSPACE_STATES_TABLE,
    create_metadata_tables,
)
from backend.app.db.session import get_engine


class MetadataRepositoryError(Exception):
    """Raised when metadata persistence fails."""


class MetadataRepository(Protocol):
    def load_registry(self, registry_name: str) -> dict[str, list[dict[str, Any]]]: ...

    def save_registry(
        self,
        registry_name: str,
        registry: dict[str, list[dict[str, Any]]],
    ) -> None: ...


@dataclass(frozen=True)
class RegistrySpec:
    name: str
    key: str
    filename_attr: str
    table: Table
    id_fields: tuple[str, ...]
    dataset_field: str | None = "dataset_id"
    secondary_field: str | None = None


REGISTRY_SPECS: dict[str, RegistrySpec] = {
    "datasets": RegistrySpec(
        name="datasets",
        key="datasets",
        filename_attr="dataset_registry_filename",
        table=DATASETS_TABLE,
        id_fields=("id",),
        dataset_field="id",
    ),
    "dataset_versions": RegistrySpec(
        name="dataset_versions",
        key="versions",
        filename_attr="dataset_version_registry_filename",
        table=DATASET_VERSIONS_TABLE,
        id_fields=("dataset_id", "version_id"),
        secondary_field="version_id",
    ),
    "models": RegistrySpec(
        name="models",
        key="models",
        filename_attr="model_registry_filename",
        table=MODELS_TABLE,
        id_fields=("id",),
    ),
    "ml_experiments": RegistrySpec(
        name="ml_experiments",
        key="experiments",
        filename_attr="ml_experiment_registry_filename",
        table=ML_EXPERIMENTS_TABLE,
        id_fields=("experiment_id",),
    ),
    "reports": RegistrySpec(
        name="reports",
        key="reports",
        filename_attr="report_registry_filename",
        table=REPORTS_TABLE,
        id_fields=("id",),
    ),
    "drift_reports": RegistrySpec(
        name="drift_reports",
        key="reports",
        filename_attr="drift_report_registry_filename",
        table=DRIFT_REPORTS_TABLE,
        id_fields=("report_id",),
    ),
    "agent_runs": RegistrySpec(
        name="agent_runs",
        key="runs",
        filename_attr="agent_run_registry_filename",
        table=AGENT_RUNS_TABLE,
        id_fields=("workflow_id",),
    ),
    "agent_jobs": RegistrySpec(
        name="agent_jobs",
        key="jobs",
        filename_attr="agent_job_registry_filename",
        table=AGENT_JOBS_TABLE,
        id_fields=("job_id",),
    ),
    "backtest_jobs": RegistrySpec(
        name="backtest_jobs",
        key="jobs",
        filename_attr="backtest_job_registry_filename",
        table=BACKTEST_JOBS_TABLE,
        id_fields=("job_id",),
        dataset_field=None,
        secondary_field="suite_type",
    ),
    "workspace_states": RegistrySpec(
        name="workspace_states",
        key="states",
        filename_attr="workspace_state_registry_filename",
        table=WORKSPACE_STATES_TABLE,
        id_fields=("workspace_id",),
        dataset_field="dataset_id",
        secondary_field="active_route",
    ),
}


def get_registry_spec(registry_name: str) -> RegistrySpec:
    try:
        return REGISTRY_SPECS[registry_name]
    except KeyError as exc:
        raise MetadataRepositoryError(f"Unknown metadata registry: {registry_name}") from exc


def create_metadata_repository(settings: Settings) -> MetadataRepository:
    if settings.metadata_backend == "postgres":
        return PostgresMetadataRepository(settings)
    return JsonMetadataRepository(settings)


def ensure_metadata_store(settings: Settings) -> None:
    if settings.metadata_backend == "postgres":
        create_metadata_tables(get_engine(settings.database_url))


class JsonMetadataRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_registry(self, registry_name: str) -> dict[str, list[dict[str, Any]]]:
        spec = get_registry_spec(registry_name)
        path = self._registry_path(spec)
        if not path.exists():
            return {spec.key: []}

        try:
            with path.open("r", encoding="utf-8") as file:
                registry = json.load(file)
        except json.JSONDecodeError as exc:
            raise MetadataRepositoryError(f"{registry_name} registry file is corrupted.") from exc

        self._validate_registry(spec, registry)
        return registry

    def save_registry(
        self,
        registry_name: str,
        registry: dict[str, list[dict[str, Any]]],
    ) -> None:
        spec = get_registry_spec(registry_name)
        self._validate_registry(spec, registry)
        path = self._registry_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.name}.{uuid4().hex}.tmp")

        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(registry, file, ensure_ascii=False, indent=2)

            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _registry_path(self, spec: RegistrySpec) -> Path:
        return self.settings.processed_data_dir / str(getattr(self.settings, spec.filename_attr))

    def _validate_registry(self, spec: RegistrySpec, registry: dict[str, Any]) -> None:
        if spec.key not in registry or not isinstance(registry[spec.key], list):
            raise MetadataRepositoryError(f"{spec.name} registry format is invalid.")


class PostgresMetadataRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = get_engine(settings.database_url)

    def load_registry(self, registry_name: str) -> dict[str, list[dict[str, Any]]]:
        spec = get_registry_spec(registry_name)
        statement = select(spec.table.c.payload).order_by(
            spec.table.c.created_at.asc().nulls_last(),
            spec.table.c.record_id.asc(),
        )

        with self.engine.begin() as connection:
            rows = connection.execute(statement).all()

        return {spec.key: [dict(row.payload) for row in rows]}

    def save_registry(
        self,
        registry_name: str,
        registry: dict[str, list[dict[str, Any]]],
    ) -> None:
        spec = get_registry_spec(registry_name)
        records = registry.get(spec.key)
        if not isinstance(records, list):
            raise MetadataRepositoryError(f"{spec.name} registry format is invalid.")

        now = datetime.now(UTC)
        record_ids = [self._record_id(spec, record) for record in records]

        with self.engine.begin() as connection:
            if record_ids:
                connection.execute(
                    delete(spec.table).where(~spec.table.c.record_id.in_(record_ids))
                )
            else:
                connection.execute(delete(spec.table))

            for record, record_id in zip(records, record_ids, strict=True):
                values = {
                    "record_id": record_id,
                    "dataset_id": self._optional_string(record.get(spec.dataset_field))
                    if spec.dataset_field
                    else None,
                    "secondary_id": self._optional_string(record.get(spec.secondary_field))
                    if spec.secondary_field
                    else None,
                    "status": self._optional_string(
                        record.get("status") or record.get("lifecycle_status")
                    ),
                    "payload": record,
                    "created_at": self._parse_datetime(record.get("created_at")),
                    "updated_at": now,
                }
                if self.engine.dialect.name == "sqlite":
                    sqlite_statement = sqlite_insert(spec.table).values(**values)
                    connection.execute(
                        sqlite_statement.on_conflict_do_update(
                            index_elements=[spec.table.c.record_id],
                            set_={
                                "dataset_id": sqlite_statement.excluded.dataset_id,
                                "secondary_id": sqlite_statement.excluded.secondary_id,
                                "status": sqlite_statement.excluded.status,
                                "payload": sqlite_statement.excluded.payload,
                                "created_at": sqlite_statement.excluded.created_at,
                                "updated_at": sqlite_statement.excluded.updated_at,
                            },
                        )
                    )
                else:
                    statement = postgres_insert(spec.table).values(**values)
                    update_values = {
                        "dataset_id": statement.excluded.dataset_id,
                        "secondary_id": statement.excluded.secondary_id,
                        "status": statement.excluded.status,
                        "payload": statement.excluded.payload,
                        "created_at": statement.excluded.created_at,
                        "updated_at": statement.excluded.updated_at,
                    }
                    connection.execute(
                        statement.on_conflict_do_update(
                            index_elements=[spec.table.c.record_id],
                            set_=update_values,
                        )
                    )

    def _record_id(self, spec: RegistrySpec, record: dict[str, Any]) -> str:
        parts = []
        for field in spec.id_fields:
            value = record.get(field)
            if value is None:
                raise MetadataRepositoryError(
                    f"{spec.name} record is missing required id field: {field}"
                )
            parts.append(str(value))
        return ":".join(parts)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
