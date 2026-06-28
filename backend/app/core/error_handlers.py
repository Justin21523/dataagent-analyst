import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    # 集中註冊 exception handlers，避免 main.py 堆滿錯誤處理邏輯。
    app.add_exception_handler(StarletteHTTPException, _starlette_http_exception_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


async def _starlette_http_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):
        raise exc

    return await http_exception_handler(request, exc)


async def _request_validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc

    return await validation_exception_handler(request, exc)


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    raw_detail = exc.detail
    message = raw_detail if isinstance(raw_detail, str) else "Request failed."
    details = None if isinstance(raw_detail, str) else raw_detail

    return _build_error_response(
        request=request,
        status_code=exc.status_code,
        code=f"http_{exc.status_code}",
        message=message,
        details=details,
        legacy_detail=raw_detail,
        headers=dict(exc.headers) if exc.headers is not None else None,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    details = jsonable_encoder(exc.errors())

    return _build_error_response(
        request=request,
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details=details,
        legacy_detail=details,
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = _get_request_id(request)

    # Server error 記錄完整 exception，但不把內部 stack trace 回傳前端。
    logger.exception(
        "Unhandled API error | request_id=%s | path=%s",
        request_id,
        request.url.path,
        exc_info=exc,
    )

    return _build_error_response(
        request=request,
        status_code=500,
        code="internal_error",
        message="Internal server error.",
        details=None,
        legacy_detail="Internal server error.",
    )


def _build_error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any,
    legacy_detail: Any,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _get_request_id(request)

    content = {
        "detail": jsonable_encoder(legacy_detail),
        "error": {
            "code": code,
            "message": message,
            "details": jsonable_encoder(details),
        },
        "request_id": request_id,
        "path": request.url.path,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers,
    )


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
