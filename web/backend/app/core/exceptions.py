from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import settings


class AppException(Exception):
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "Internal server error"
    code: str = "internal_error"

    def __init__(self, detail: str | None = None, code: str | None = None) -> None:
        if detail is not None:
            self.detail = detail
        if code is not None:
            self.code = code
        super().__init__(self.detail)


class NotFoundException(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found"
    code = "not_found"


class ForbiddenException(AppException):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Forbidden"
    code = "forbidden"


class UnauthorizedException(AppException):
    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Unauthorized"
    code = "unauthorized"


class ConflictException(AppException):
    status_code = status.HTTP_409_CONFLICT
    detail = "Conflict"
    code = "conflict"


def problem_response(
    status_code: int,
    detail: str,
    code: str = "error",
    instance: str | None = None,
) -> JSONResponse:
    body: dict = {
        "type": "about:blank",
        "title": detail,
        "status": status_code,
        "detail": detail,
        "code": code,
    }
    if instance:
        body["instance"] = instance
    return JSONResponse(content=body, status_code=status_code)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return problem_response(
        status_code=exc.status_code,
        detail=exc.detail,
        code=exc.code,
        instance=str(request.url),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = [
        {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        content={
            "type": "about:blank",
            "title": "Validation Error",
            "status": status.HTTP_422_UNPROCESSABLE_CONTENT,
            "detail": "Request validation failed",
            "code": "validation_error",
            "errors": errors,
        },
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if settings.DEBUG:
        raise exc
    return problem_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error",
        code="internal_error",
        instance=str(request.url),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
