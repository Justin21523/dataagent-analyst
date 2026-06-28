from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.app.core.config import Settings
from backend.app.db.metadata_tables import create_metadata_tables
from backend.app.db.session import get_engine
from backend.app.repositories.metadata_repository import (
    JsonMetadataRepository,
    PostgresMetadataRepository,
)


def test_json_metadata_repository_round_trips_registry(tmp_path: Path) -> None:
    settings = Settings(processed_data_dir=tmp_path)
    repository = JsonMetadataRepository(settings)

    registry = {
        "models": [
            {
                "id": "model-1",
                "dataset_id": "dataset-1",
                "model_name": "logistic_regression",
                "status": "ready",
                "created_at": "2026-06-29T00:00:00+00:00",
            }
        ]
    }

    repository.save_registry("models", registry)

    assert repository.load_registry("models") == registry


def test_json_metadata_repository_round_trips_workspace_state(tmp_path: Path) -> None:
    settings = Settings(processed_data_dir=tmp_path)
    repository = JsonMetadataRepository(settings)

    registry = {
        "states": [
            {
                "workspace_id": "default",
                "active_route": "model-workbench",
                "dataset_id": "dataset-1",
                "created_at": "2026-06-29T00:00:00+00:00",
                "updated_at": "2026-06-29T00:00:00+00:00",
            }
        ]
    }

    repository.save_registry("workspace_states", registry)

    assert repository.load_registry("workspace_states") == registry


def test_json_metadata_repository_round_trips_backtest_job(tmp_path: Path) -> None:
    settings = Settings(processed_data_dir=tmp_path)
    repository = JsonMetadataRepository(settings)

    registry = {
        "jobs": [
            {
                "job_id": "job-1",
                "suite_type": "api",
                "status": "success",
                "created_at": "2026-06-29T00:00:00+00:00",
                "events": [],
            }
        ]
    }

    repository.save_registry("backtest_jobs", registry)

    assert repository.load_registry("backtest_jobs") == registry


def test_json_metadata_repository_uses_unique_temporary_files_for_parallel_saves(
    tmp_path: Path,
) -> None:
    settings = Settings(processed_data_dir=tmp_path)
    repository = JsonMetadataRepository(settings)

    def save_workspace_state(index: int) -> None:
        repository.save_registry(
            "workspace_states",
            {
                "states": [
                    {
                        "workspace_id": "default",
                        "active_route": f"route-{index}",
                        "dataset_id": "dataset-1",
                        "created_at": "2026-06-29T00:00:00+00:00",
                        "updated_at": "2026-06-29T00:00:00+00:00",
                    }
                ]
            },
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(save_workspace_state, range(20)))

    registry = repository.load_registry("workspace_states")

    assert len(registry["states"]) == 1
    assert registry["states"][0]["active_route"].startswith("route-")


def test_sql_metadata_repository_upserts_records(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'metadata.db'}"
    settings = Settings(database_url=database_url, metadata_backend="postgres")
    create_metadata_tables(get_engine(database_url))
    repository = PostgresMetadataRepository(settings)

    repository.save_registry(
        "models",
        {
            "models": [
                {
                    "id": "model-1",
                    "dataset_id": "dataset-1",
                    "model_name": "logistic_regression",
                    "status": "ready",
                    "created_at": "2026-06-29T00:00:00+00:00",
                }
            ]
        },
    )
    repository.save_registry(
        "models",
        {
            "models": [
                {
                    "id": "model-1",
                    "dataset_id": "dataset-1",
                    "model_name": "random_forest",
                    "status": "ready",
                    "created_at": "2026-06-29T00:00:00+00:00",
                }
            ]
        },
    )

    models = repository.load_registry("models")["models"]

    assert len(models) == 1
    assert models[0]["model_name"] == "random_forest"


def test_sql_metadata_repository_handles_composite_dataset_version_id(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'metadata.db'}"
    settings = Settings(database_url=database_url, metadata_backend="postgres")
    create_metadata_tables(get_engine(database_url))
    repository = PostgresMetadataRepository(settings)

    repository.save_registry(
        "dataset_versions",
        {
            "versions": [
                {
                    "dataset_id": "dataset-1",
                    "version_id": "v1",
                    "version_index": 1,
                    "kind": "original",
                    "file_path": "data/raw/sample.csv",
                    "row_count": 2,
                    "column_count": 2,
                    "columns": ["a", "b"],
                    "content_hash": "hash",
                    "created_at": "2026-06-29T00:00:00+00:00",
                }
            ]
        },
    )

    versions = repository.load_registry("dataset_versions")["versions"]

    assert len(versions) == 1
    assert versions[0]["dataset_id"] == "dataset-1"
    assert versions[0]["version_id"] == "v1"


def test_sql_metadata_repository_upserts_workspace_state(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'metadata.db'}"
    settings = Settings(database_url=database_url, metadata_backend="postgres")
    create_metadata_tables(get_engine(database_url))
    repository = PostgresMetadataRepository(settings)

    repository.save_registry(
        "workspace_states",
        {
            "states": [
                {
                    "workspace_id": "default",
                    "active_route": "data-upload",
                    "dataset_id": "dataset-1",
                    "created_at": "2026-06-29T00:00:00+00:00",
                    "updated_at": "2026-06-29T00:00:00+00:00",
                }
            ]
        },
    )
    repository.save_registry(
        "workspace_states",
        {
            "states": [
                {
                    "workspace_id": "default",
                    "active_route": "model-workbench",
                    "dataset_id": "dataset-1",
                    "created_at": "2026-06-29T00:00:00+00:00",
                    "updated_at": "2026-06-29T00:00:00+00:00",
                }
            ]
        },
    )

    states = repository.load_registry("workspace_states")["states"]

    assert len(states) == 1
    assert states[0]["active_route"] == "model-workbench"


def test_sql_metadata_repository_upserts_backtest_job(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'metadata.db'}"
    settings = Settings(database_url=database_url, metadata_backend="postgres")
    create_metadata_tables(get_engine(database_url))
    repository = PostgresMetadataRepository(settings)

    repository.save_registry(
        "backtest_jobs",
        {
            "jobs": [
                {
                    "job_id": "job-1",
                    "suite_type": "api",
                    "status": "queued",
                    "created_at": "2026-06-29T00:00:00+00:00",
                    "events": [],
                }
            ]
        },
    )
    repository.save_registry(
        "backtest_jobs",
        {
            "jobs": [
                {
                    "job_id": "job-1",
                    "suite_type": "api",
                    "status": "success",
                    "created_at": "2026-06-29T00:00:00+00:00",
                    "events": [],
                }
            ]
        },
    )

    jobs = repository.load_registry("backtest_jobs")["jobs"]

    assert len(jobs) == 1
    assert jobs[0]["status"] == "success"
