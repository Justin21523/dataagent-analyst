from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BundleExportRequest(BaseModel):
    dataset_id: str = Field(description="Dataset to export")
    include_models: bool = Field(default=True)
    include_reports: bool = Field(default=True)
    include_drift_reports: bool = Field(default=True)


class BundleExportResponse(BaseModel):
    bundle_id: str = Field(description="Bundle ID")
    dataset_id: str = Field(description="Dataset ID")
    bundle_path: str = Field(description="Bundle path relative to project root")
    manifest: dict[str, Any] = Field(description="Bundle manifest")
    created_at: datetime = Field(description="Creation time")


class BundleImportResponse(BaseModel):
    bundle_id: str = Field(description="Imported bundle ID")
    imported_dataset_id: str = Field(description="Imported dataset ID")
    id_mapping: dict[str, str] = Field(description="Old-to-new ID mapping")
    imported_records: dict[str, int] = Field(description="Imported record counts")
    warnings: list[str] = Field(description="Import warnings")
