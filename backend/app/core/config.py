from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "DataAgent Analyst API"
    app_version: str = "0.1.0"
    app_env: str = "development"
    api_prefix: str = "/api"
    log_level: str = "INFO"

    max_upload_size_mb: int = 10
    dataset_registry_filename: str = "datasets_registry.json"
    dataset_version_registry_filename: str = "dataset_versions_registry.json"
    model_registry_filename: str = "models_registry.json"
    ml_experiment_registry_filename: str = "ml_experiments_registry.json"
    report_registry_filename: str = "reports_registry.json"
    agent_run_registry_filename: str = "agent_runs_registry.json"
    agent_job_registry_filename: str = "agent_jobs_registry.json"
    backtest_job_registry_filename: str = "backtest_jobs_registry.json"
    drift_report_registry_filename: str = "drift_reports_registry.json"
    workspace_state_registry_filename: str = "workspace_states_registry.json"
    llm_enabled: bool = False
    llm_base_url: str = "http://127.0.0.1:8080"
    llm_model: str = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    llm_timeout_seconds: int = 20
    llm_temperature: float = 0.2
    llm_max_tokens: int = 700

    database_url: str = "postgresql://dataagent:dataagent_dev_password@127.0.0.1:5432/dataagent"
    metadata_backend: Literal["json", "postgres"] = "json"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    infrastructure_checks_enabled: bool = False
    infrastructure_timeout_seconds: float = 2.0

    explainability_max_sample_size: int = 200
    explainability_max_background_size: int = 100
    explainability_max_permutation_repeats: int = 20
    explainability_max_beeswarm_features: int = 15
    explainability_max_local_features: int = 15

    allowed_origins: list[str] = [
        "http://localhost",
        "http://127.0.0.1",
    ]
    allowed_origin_regex: str = r"^http://(localhost|127\.0\.0\.1):\d+$"

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    reports_dir: Path = data_dir / "reports"
    models_dir: Path = data_dir / "models"
    vector_store_dir: Path = data_dir / "vector_store"
    bundles_dir: Path = data_dir / "bundles"
    backtests_dir: Path = data_dir / "backtests"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def ensure_runtime_directories(settings: Settings) -> None:
    # 集中建立執行期間會使用的資料夾，避免各 service 分散處理路徑。
    runtime_directories = [
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.reports_dir,
        settings.models_dir,
        settings.vector_store_dir,
        settings.bundles_dir,
        settings.backtests_dir,
    ]

    for directory in runtime_directories:
        # parents=True 可一次建立上層資料夾；exist_ok=True 讓重複啟動不報錯。
        directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    # 使用快取避免每次 request 都重新讀取環境變數與設定。
    return Settings()
