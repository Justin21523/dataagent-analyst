import socket
from datetime import UTC, datetime
from time import perf_counter
from urllib.parse import urlparse

import httpx

from backend.app.core.config import Settings
from backend.app.schemas.infrastructure_schema import (
    InfrastructureServiceCheck,
    InfrastructureStatusResponse,
)
from backend.app.services.local_llm_service import LocalLLMService


class InfrastructureService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local_llm_service = LocalLLMService(settings)

    def check_status(self) -> InfrastructureStatusResponse:
        # 本機開發預設不檢查外部服務，避免測試環境被基礎設施綁住。
        if not self.settings.infrastructure_checks_enabled:
            return InfrastructureStatusResponse(
                status="disabled",
                checks_enabled=False,
                services=[
                    self._disabled_check("postgresql"),
                    self._disabled_check("qdrant"),
                    self._disabled_check("llama.cpp"),
                ],
                checked_at=datetime.now(UTC),
            )

        postgres_check = self._check_postgresql()
        qdrant_check = self._check_qdrant()
        llm_check = self._check_llm()

        required_checks = [
            postgres_check,
            qdrant_check,
        ]

        if self.settings.llm_enabled:
            required_checks.append(llm_check)

        overall_status = (
            "healthy" if all(check.healthy is True for check in required_checks) else "degraded"
        )

        return InfrastructureStatusResponse(
            status=overall_status,
            checks_enabled=True,
            services=[
                postgres_check,
                qdrant_check,
                llm_check,
            ],
            checked_at=datetime.now(UTC),
        )

    def _check_postgresql(self) -> InfrastructureServiceCheck:
        parsed_url = urlparse(self.settings.database_url)
        host = parsed_url.hostname
        port = parsed_url.port or 5432

        if not host:
            return InfrastructureServiceCheck(
                service="postgresql",
                status="unhealthy",
                healthy=False,
                message="DATABASE_URL does not contain a valid host.",
                latency_ms=None,
            )

        started_at = perf_counter()

        try:
            with socket.create_connection(
                (host, port),
                timeout=self.settings.infrastructure_timeout_seconds,
            ):
                pass
        except OSError as exc:
            return InfrastructureServiceCheck(
                service="postgresql",
                status="unhealthy",
                healthy=False,
                message=f"PostgreSQL TCP connection failed: {exc}",
                latency_ms=self._latency_ms(started_at),
                details={
                    "host": host,
                    "port": port,
                },
            )

        return InfrastructureServiceCheck(
            service="postgresql",
            status="healthy",
            healthy=True,
            message="PostgreSQL TCP connection succeeded.",
            latency_ms=self._latency_ms(started_at),
            details={
                "host": host,
                "port": port,
            },
        )

    def _check_qdrant(self) -> InfrastructureServiceCheck:
        started_at = perf_counter()
        ready_url = f"{self.settings.qdrant_url.rstrip('/')}/readyz"

        headers = {}

        if self.settings.qdrant_api_key:
            headers["api-key"] = self.settings.qdrant_api_key

        try:
            response = httpx.get(
                ready_url,
                headers=headers,
                timeout=self.settings.infrastructure_timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            return InfrastructureServiceCheck(
                service="qdrant",
                status="unhealthy",
                healthy=False,
                message=f"Qdrant readiness check failed: {exc}",
                latency_ms=self._latency_ms(started_at),
                details={
                    "ready_url": ready_url,
                },
            )

        return InfrastructureServiceCheck(
            service="qdrant",
            status="healthy",
            healthy=True,
            message="Qdrant readiness check succeeded.",
            latency_ms=self._latency_ms(started_at),
            details={
                "ready_url": ready_url,
            },
        )

    def _check_llm(self) -> InfrastructureServiceCheck:
        started_at = perf_counter()
        llm_status = self.local_llm_service.check_status()

        if not llm_status.enabled:
            return InfrastructureServiceCheck(
                service="llama.cpp",
                status="disabled",
                healthy=None,
                message=llm_status.message,
                latency_ms=self._latency_ms(started_at),
                details={
                    "base_url": llm_status.base_url,
                    "model": llm_status.model,
                },
            )

        return InfrastructureServiceCheck(
            service="llama.cpp",
            status="healthy" if llm_status.online else "unhealthy",
            healthy=llm_status.online,
            message=llm_status.message,
            latency_ms=self._latency_ms(started_at),
            details={
                "base_url": llm_status.base_url,
                "model": llm_status.model,
            },
        )

    def _disabled_check(self, service: str) -> InfrastructureServiceCheck:
        return InfrastructureServiceCheck(
            service=service,
            status="disabled",
            healthy=None,
            message="Infrastructure checks are disabled.",
            latency_ms=None,
        )

    def _latency_ms(self, started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 2)
