from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class BacktestStep(BaseModel):
    name: str = Field(description="Backtest step name")
    status: str = Field(description="Step status")
    duration_seconds: float = Field(default=0.0, description="Step duration in seconds")
    payload_path: str | None = Field(default=None, description="Relative payload path")
    error: str | None = Field(default=None, description="Step error message")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Step metadata")


class BacktestAssertion(BaseModel):
    name: str = Field(description="Assertion name")
    status: str = Field(description="Assertion status")
    expected: Any = Field(default=None, description="Expected value")
    actual: Any = Field(default=None, description="Actual value")
    message: str | None = Field(default=None, description="Assertion message")


class BacktestPayloadSummary(BaseModel):
    name: str = Field(description="Payload file name")
    path: str = Field(description="Payload path relative to run directory")
    size_bytes: int = Field(description="Payload file size")


class BacktestScreenshotSummary(BaseModel):
    name: str = Field(description="Screenshot file name")
    path: str = Field(description="Screenshot path relative to run directory")
    size_bytes: int = Field(description="Screenshot file size")


class BacktestRunSummary(BaseModel):
    run_id: str = Field(description="Backtest run ID")
    status: str = Field(description="Run status")
    created_at: str | None = Field(default=None, description="Run creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Run metadata")
    step_count: int = Field(default=0, description="Total step count")
    failed_step_count: int = Field(default=0, description="Failed step count")
    assertion_count: int = Field(default=0, description="Total assertion count")
    failed_assertion_count: int = Field(default=0, description="Failed assertion count")
    screenshot_count: int = Field(default=0, description="Screenshot count")
    payload_count: int = Field(default=0, description="Payload count")
    has_summary: bool = Field(default=False, description="Whether summary.md exists")
    error: str | None = Field(default=None, description="Artifact read error")


class BacktestRunDetail(BaseModel):
    summary: BacktestRunSummary = Field(description="Run summary")
    steps: list[BacktestStep] = Field(default_factory=list, description="Run steps")
    assertions: list[BacktestAssertion] = Field(
        default_factory=list,
        description="Run assertions",
    )
    payloads: list[BacktestPayloadSummary] = Field(
        default_factory=list,
        description="Payload files",
    )
    screenshots: list[BacktestScreenshotSummary] = Field(
        default_factory=list,
        description="Screenshot files",
    )
    summary_markdown: str | None = Field(default=None, description="Run markdown summary")


class BacktestRunListResponse(BaseModel):
    runs: list[BacktestRunSummary] = Field(description="Backtest runs")
    total: int = Field(description="Total run count")


class BacktestPayloadDetail(BaseModel):
    run_id: str = Field(description="Backtest run ID")
    name: str = Field(description="Payload file name")
    payload: Any = Field(description="Payload JSON")


class BacktestSuiteSummary(BaseModel):
    suite_id: str = Field(description="Suite ID")
    suite: str = Field(description="Suite name")
    status: str = Field(description="Suite status")
    created_at: str | None = Field(default=None, description="Suite creation timestamp")
    runs: list[dict[str, Any]] = Field(default_factory=list, description="Suite runs")
    error: str | None = Field(default=None, description="Artifact read error")


class BacktestSuiteListResponse(BaseModel):
    suites: list[BacktestSuiteSummary] = Field(description="Backtest suite summaries")
    total: int = Field(description="Total suite count")


BacktestSuiteType = Literal["api", "scenario", "ui", "full"]


class BacktestJobRequest(BaseModel):
    suite_type: BacktestSuiteType = Field(description="Backtest suite to execute")
    sample_file: str | None = Field(default=None, description="Sample CSV path")
    backend_port: int = Field(default=8010, ge=1, le=65535, description="Backend port")
    frontend_port: int = Field(default=5174, ge=1, le=65535, description="Frontend port")
    keep_servers: bool = Field(default=False, description="Keep managed services running")


class BacktestJobEvent(BaseModel):
    event_id: str = Field(description="Event ID")
    job_id: str = Field(description="Backtest job ID")
    event_type: str = Field(description="queued, running, log, command, completed, or failed")
    status: str = Field(description="Event status")
    message: str = Field(description="Event message")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event payload")
    created_at: datetime = Field(description="Event creation time")


class BacktestJobSummary(BaseModel):
    job_id: str = Field(description="Backtest job ID")
    suite_type: BacktestSuiteType = Field(description="Backtest suite type")
    status: str = Field(description="queued, running, success, or failed")
    created_at: datetime = Field(description="Job creation time")
    started_at: datetime | None = Field(default=None, description="Job start time")
    finished_at: datetime | None = Field(default=None, description="Job finish time")
    event_count: int = Field(description="Job event count")
    run_ids: list[str] = Field(default_factory=list, description="Produced run IDs")
    suite_ids: list[str] = Field(default_factory=list, description="Produced suite IDs")
    error_message: str | None = Field(default=None, description="Error message")


class BacktestJobDetail(BacktestJobSummary):
    request: BacktestJobRequest = Field(description="Original backtest job request")
    events: list[BacktestJobEvent] = Field(description="Backtest job events")
    result: dict[str, Any] | None = Field(default=None, description="Backtest job result")


class BacktestJobStartResponse(BaseModel):
    message: str = Field(description="Job start message")
    job: BacktestJobSummary = Field(description="Started job summary")


class BacktestJobListResponse(BaseModel):
    jobs: list[BacktestJobSummary] = Field(description="Backtest jobs")
    total: int = Field(description="Total job count")


class BacktestJobEventListResponse(BaseModel):
    job_id: str = Field(description="Backtest job ID")
    events: list[BacktestJobEvent] = Field(description="Backtest job events")
    total: int = Field(description="Total event count")
