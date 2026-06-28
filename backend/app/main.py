import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes_agent_jobs import router as agent_jobs_router
from backend.app.api.routes_agents import router as agents_router
from backend.app.api.routes_ai_insights import router as ai_insights_router
from backend.app.api.routes_analysis import router as analysis_router
from backend.app.api.routes_backtests import router as backtests_router
from backend.app.api.routes_bundles import router as bundles_router
from backend.app.api.routes_datasets import router as datasets_router
from backend.app.api.routes_drift import router as drift_router
from backend.app.api.routes_eda import router as eda_router
from backend.app.api.routes_explainability import router as explainability_router
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_ml import router as ml_router
from backend.app.api.routes_ml_workbench import router as ml_workbench_router
from backend.app.api.routes_reports import router as reports_router
from backend.app.api.routes_visualizations import router as visualizations_router
from backend.app.api.routes_workspaces import router as workspaces_router
from backend.app.core.config import ensure_runtime_directories, get_settings
from backend.app.core.error_handlers import register_exception_handlers
from backend.app.core.logging import configure_logging
from backend.app.core.middleware import request_context_middleware
from backend.app.repositories.metadata_repository import ensure_metadata_store

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    # 所有設定集中從 Settings 讀取，避免在不同檔案硬編碼設定值。
    settings = get_settings()

    # 啟動時先建立 logging 與 runtime 資料夾。
    configure_logging(settings.log_level)
    ensure_runtime_directories(settings)
    ensure_metadata_store(settings)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend API for DataAgent Analyst.",
    )

    # CORS 讓本機前端可以安全呼叫 FastAPI。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_origin_regex=settings.allowed_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )

    # Request context middleware 統一加入 request ID 與處理時間。
    app.middleware("http")(request_context_middleware)

    # 統一 HTTP、validation 與未預期錯誤的 response 格式。
    register_exception_handlers(app)

    # 所有 API route 統一掛在 /api 底下，方便前後端維護。
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(ai_insights_router, prefix=settings.api_prefix)
    app.include_router(analysis_router, prefix=settings.api_prefix)
    app.include_router(backtests_router, prefix=settings.api_prefix)
    app.include_router(agents_router, prefix=settings.api_prefix)
    app.include_router(agent_jobs_router, prefix=settings.api_prefix)
    app.include_router(bundles_router, prefix=settings.api_prefix)
    app.include_router(datasets_router, prefix=settings.api_prefix)
    app.include_router(drift_router, prefix=settings.api_prefix)
    app.include_router(eda_router, prefix=settings.api_prefix)
    app.include_router(explainability_router, prefix=settings.api_prefix)
    app.include_router(visualizations_router, prefix=settings.api_prefix)
    app.include_router(ml_router, prefix=settings.api_prefix)
    app.include_router(ml_workbench_router, prefix=settings.api_prefix)
    app.include_router(reports_router, prefix=settings.api_prefix)
    app.include_router(workspaces_router, prefix=settings.api_prefix)

    @app.get("/", tags=["Root"])
    def read_root() -> dict[str, str]:
        # Root endpoint 提供最基本的導覽資訊。
        return {
            "message": "DataAgent Analyst API",
            "docs_url": "/docs",
            "health_url": f"{settings.api_prefix}/health",
        }

    logger.info("DataAgent Analyst API application created")

    return app


app = create_app()
