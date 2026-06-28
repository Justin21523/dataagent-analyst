import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.app.core.config import Settings
from backend.app.schemas.bundle_schema import (
    BundleExportRequest,
    BundleExportResponse,
    BundleImportResponse,
)
from backend.app.services.dataset_service import DatasetService


class BundleServiceError(Exception):
    """Raised when bundle export/import fails."""


class BundleService:
    manifest_version = "1.0"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dataset_service = DatasetService(settings)

    def export_bundle(self, request: BundleExportRequest) -> BundleExportResponse:
        self.dataset_service.get_dataset_detail(request.dataset_id)
        dataset_record = self.dataset_service._find_dataset_record(request.dataset_id)
        versions = self.dataset_service.list_dataset_versions(request.dataset_id).versions
        models = (
            self._filter_records(
                self._load_json(
                    self.settings.processed_data_dir / self.settings.model_registry_filename
                ),
                "models",
                request.dataset_id,
            )
            if request.include_models
            else []
        )
        reports = (
            self._filter_records(
                self._load_json(
                    self.settings.processed_data_dir / self.settings.report_registry_filename
                ),
                "reports",
                request.dataset_id,
            )
            if request.include_reports
            else []
        )
        drift_reports = (
            self._filter_records(
                self._load_json(
                    self.settings.processed_data_dir / self.settings.drift_report_registry_filename
                ),
                "reports",
                request.dataset_id,
            )
            if request.include_drift_reports
            else []
        )

        bundle_id = uuid4().hex
        created_at = datetime.now(UTC)
        bundle_path = self.settings.bundles_dir / f"{bundle_id}.zip"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        written_artifacts: set[str] = set()

        manifest = {
            "manifest_version": self.manifest_version,
            "bundle_id": bundle_id,
            "created_at": created_at.isoformat(),
            "dataset": dataset_record,
            "versions": [version.model_dump(mode="json") for version in versions],
            "models": models,
            "reports": reports,
            "drift_reports": drift_reports,
        }

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            self._write_artifact(archive, dataset_record.get("file_path"), written_artifacts)

            for version in versions:
                self._write_artifact(archive, version.file_path, written_artifacts)

            for model in models:
                self._write_artifact(archive, model.get("model_path"), written_artifacts)
                self._write_artifact(
                    archive,
                    model.get("evaluation_artifacts_path"),
                    written_artifacts,
                )

            for report in reports:
                self._write_artifact(archive, report.get("markdown_path"), written_artifacts)

        return BundleExportResponse(
            bundle_id=bundle_id,
            dataset_id=request.dataset_id,
            bundle_path=str(bundle_path.relative_to(self.settings.project_root)),
            manifest=manifest,
            created_at=created_at,
        )

    def import_bundle(self, file_content: bytes) -> BundleImportResponse:
        if not file_content:
            raise BundleServiceError("Bundle file is empty.")

        import_id = uuid4().hex
        import_dir = self.settings.bundles_dir / f"import_{import_id}"
        import_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = import_dir / "bundle.zip"
        bundle_path.write_bytes(file_content)

        with zipfile.ZipFile(bundle_path, "r") as archive:
            self._validate_archive(archive)
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

            if manifest.get("manifest_version") != self.manifest_version:
                raise BundleServiceError("Unsupported bundle manifest version.")

            old_dataset = manifest["dataset"]
            new_dataset_id = uuid4().hex
            id_mapping = {old_dataset["id"]: new_dataset_id}
            extracted_root = import_dir / "artifacts"
            extracted_root.mkdir(parents=True, exist_ok=True)

            for member in archive.namelist():
                if member.startswith("artifacts/") and not member.endswith("/"):
                    archive.extract(member, import_dir)

            dataset_record = dict(old_dataset)
            dataset_record["id"] = new_dataset_id
            dataset_record["name"] = f"{old_dataset['name']}_imported"
            dataset_record["file_path"] = self._imported_artifact_path(
                old_dataset.get("file_path"),
                extracted_root,
            )
            dataset_record["stored_filename"] = Path(dataset_record["file_path"]).name
            dataset_record["created_at"] = datetime.now(UTC).isoformat()
            dataset_record["updated_at"] = dataset_record["created_at"]

            self._append_records(
                self.settings.processed_data_dir / self.settings.dataset_registry_filename,
                "datasets",
                [dataset_record],
            )

            version_records = []

            for version in manifest.get("versions", []):
                version_record = dict(version)
                version_record["dataset_id"] = new_dataset_id
                version_record["file_path"] = self._imported_artifact_path(
                    version.get("file_path"),
                    extracted_root,
                )
                version_records.append(version_record)

            self._append_records(
                self.settings.processed_data_dir / self.settings.dataset_version_registry_filename,
                "versions",
                version_records,
            )

            model_records = []

            for model in manifest.get("models", []):
                new_model_id = uuid4().hex
                id_mapping[model["id"]] = new_model_id
                model_record = dict(model)
                model_record["id"] = new_model_id
                model_record["dataset_id"] = new_dataset_id
                model_record["model_path"] = self._imported_artifact_path(
                    model.get("model_path"),
                    extracted_root,
                )
                model_record["evaluation_artifacts_path"] = self._imported_artifact_path(
                    model.get("evaluation_artifacts_path"),
                    extracted_root,
                )
                model_record["lifecycle_status"] = "candidate"
                model_records.append(model_record)

            self._append_records(
                self.settings.processed_data_dir / self.settings.model_registry_filename,
                "models",
                model_records,
            )

        return BundleImportResponse(
            bundle_id=import_id,
            imported_dataset_id=new_dataset_id,
            id_mapping=id_mapping,
            imported_records={
                "versions": len(version_records),
                "models": len(model_records),
            },
            warnings=[],
        )

    def _filter_records(
        self,
        registry: dict[str, list[dict[str, Any]]],
        key: str,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        return [
            record for record in registry.get(key, []) if record.get("dataset_id") == dataset_id
        ]

    def _load_json(self, path: Path) -> dict[str, list[dict[str, Any]]]:
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _append_records(
        self,
        path: Path,
        key: str,
        records: list[dict[str, Any]],
    ) -> None:
        registry = self._load_json(path) or {key: []}
        registry.setdefault(key, [])
        registry[key].extend(records)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(registry, file, ensure_ascii=False, indent=2)

        temporary_path.replace(path)

    def _write_artifact(
        self,
        archive: zipfile.ZipFile,
        relative_path: str | None,
        written_artifacts: set[str],
    ) -> None:
        if not relative_path:
            return

        if relative_path in written_artifacts:
            return

        artifact_path = self.settings.project_root / relative_path

        if not artifact_path.exists():
            return

        archive.write(artifact_path, arcname=f"artifacts/{relative_path}")
        written_artifacts.add(relative_path)

    def _validate_archive(self, archive: zipfile.ZipFile) -> None:
        if "manifest.json" not in archive.namelist():
            raise BundleServiceError("Bundle manifest is missing.")

        for member in archive.namelist():
            path = Path(member)

            if path.is_absolute() or ".." in path.parts:
                raise BundleServiceError("Bundle contains an unsafe path.")

    def _imported_artifact_path(
        self,
        original_relative_path: str | None,
        extracted_root: Path,
    ) -> str | None:
        if not original_relative_path:
            return None

        source = extracted_root / original_relative_path

        if not source.exists():
            raise BundleServiceError(f"Bundle artifact is missing: {original_relative_path}")

        target = self.settings.data_dir / "imported" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

        return str(target.relative_to(self.settings.project_root))
