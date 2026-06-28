from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import RequestResponseEndpoint


async def request_context_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    # 優先沿用 client 傳入的 request ID，否則由後端產生。
    request_id = request.headers.get("X-Request-ID", "").strip() or uuid4().hex
    request.state.request_id = request_id

    started_at = perf_counter()
    response = await call_next(request)
    process_time_ms = (perf_counter() - started_at) * 1000

    # Response header 方便前端、log 與 API client 對應同一次 request。
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

    return response
