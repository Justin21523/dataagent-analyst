from datetime import datetime

from pydantic import BaseModel, Field


class ReportSummary(BaseModel):
    id: str = Field(description="Report ID")
    dataset_id: str = Field(description="Dataset ID")
    title: str = Field(description="Report title")
    markdown_path: str = Field(description="Report markdown file path")
    status: str = Field(description="Report status")
    created_at: datetime = Field(description="Report creation time")


class ReportDetail(ReportSummary):
    markdown_content: str = Field(description="Markdown report content")


class ReportGenerateResponse(BaseModel):
    message: str = Field(description="Report generation message")
    report: ReportDetail = Field(description="Generated report detail")


class ReportListResponse(BaseModel):
    dataset_id: str = Field(description="Dataset ID")
    reports: list[ReportSummary] = Field(description="Report list")
    total: int = Field(description="Total report count")
