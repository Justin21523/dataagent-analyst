#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import Settings  # noqa: E402
from backend.app.db.metadata_tables import create_metadata_tables  # noqa: E402
from backend.app.db.session import get_engine  # noqa: E402
from backend.app.repositories.metadata_repository import (  # noqa: E402
    REGISTRY_SPECS,
    JsonMetadataRepository,
    PostgresMetadataRepository,
    RegistrySpec,
)

ARTIFACT_FIELDS = {
    "datasets": ("file_path",),
    "dataset_versions": ("file_path",),
    "models": ("model_path", "evaluation_artifacts_path"),
    "reports": ("markdown_path",),
}


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    project_root = args.project_root.resolve()

    json_settings = Settings(
        processed_data_dir=source_dir,
        project_root=project_root,
    )
    postgres_settings = Settings(
        database_url=args.database_url,
        metadata_backend="postgres",
        processed_data_dir=source_dir,
        project_root=project_root,
    )

    json_repository = JsonMetadataRepository(json_settings)

    planned: dict[str, list[dict[str, Any]]] = {}
    missing_artifacts = []

    for registry_name, spec in REGISTRY_SPECS.items():
        registry = json_repository.load_registry(registry_name)
        records = registry.get(spec.key, [])
        planned[registry_name] = records
        missing_artifacts.extend(find_missing_artifacts(registry_name, records, project_root))

    print("Metadata migration plan")
    print(f"  source_dir={source_dir}")
    print(f"  database_url={sanitize_database_url(args.database_url)}")

    for registry_name, records in planned.items():
        print(f"  {registry_name}: {len(records)} record(s)")

    if missing_artifacts:
        print("Missing artifact references:")
        for missing_artifact in missing_artifacts:
            print(f"  {missing_artifact}")

        if args.fail_on_missing_artifacts:
            raise SystemExit("Missing artifact references found.")

    if args.dry_run:
        print("Dry run complete. No Postgres rows were changed.")
        return

    create_metadata_tables(get_engine(args.database_url))
    postgres_repository = PostgresMetadataRepository(postgres_settings)

    for registry_name, records in planned.items():
        spec = REGISTRY_SPECS[registry_name]
        current_registry = postgres_repository.load_registry(registry_name)
        merged_records = merge_records(spec, current_registry.get(spec.key, []), records)
        postgres_repository.save_registry(registry_name, {spec.key: merged_records})

    print("Metadata migration complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate JSON metadata registries into PostgreSQL.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing JSON registry files.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve artifact paths.",
    )
    parser.add_argument(
        "--database-url",
        default=Settings().database_url,
        help="PostgreSQL DATABASE_URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print migration counts without writing to PostgreSQL.",
    )
    parser.add_argument(
        "--fail-on-missing-artifacts",
        action="store_true",
        help="Fail if registry records point to missing artifact files.",
    )
    return parser.parse_args()


def merge_records(
    spec: RegistrySpec,
    existing_records: list[dict[str, Any]],
    incoming_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {record_id(spec, record): record for record in existing_records}
    for record in incoming_records:
        merged[record_id(spec, record)] = record
    return list(merged.values())


def record_id(spec: RegistrySpec, record: dict[str, Any]) -> str:
    return ":".join(str(record[field]) for field in spec.id_fields)


def find_missing_artifacts(
    registry_name: str,
    records: list[dict[str, Any]],
    project_root: Path,
) -> list[str]:
    missing = []
    for record in records:
        for field in ARTIFACT_FIELDS.get(registry_name, ()):
            relative_path = record.get(field)
            if not relative_path:
                continue
            if not (project_root / str(relative_path)).exists():
                record_name = record.get("id") or record.get("report_id")
                missing.append(f"{registry_name}:{record_name}:{field}")
    return missing


def sanitize_database_url(database_url: str) -> str:
    if "@" not in database_url:
        return database_url
    scheme_and_auth, host_part = database_url.rsplit("@", 1)
    scheme = scheme_and_auth.split("://", 1)[0]
    return f"{scheme}://***:***@{host_part}"


if __name__ == "__main__":
    main()
