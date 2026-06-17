from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.database import engine, init_db
from app.core.exceptions import register_exception_handlers
from app.core.logging_ import setup_logging
from app.core.metrics import setup_metrics


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(debug=settings.DEBUG)
    logger = structlog.get_logger()
    logger.info("Starting Test Metal API", debug=settings.DEBUG)
    await init_db()
    yield
    await engine.dispose()
    logger.info("Test Metal API stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="Metallurgical regression & Pareto optimization API",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if not settings.DEBUG:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

    register_exception_handlers(app)

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            content={
                "type": "about:blank",
                "title": "Not Found",
                "status": 404,
                "detail": "The requested resource was not found",
                "code": "not_found",
            },
            status_code=404,
        )

    setup_metrics(app)
    app.include_router(api_v1_router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict:
        return {"app": settings.APP_NAME, "version": "1.0.0"}

    return app


app = create_app()
