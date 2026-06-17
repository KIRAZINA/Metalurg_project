from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@router.get("/ready")
async def readiness() -> dict:
    checks: dict[str, str | bool] = {"status": "ok"}
    # Database check
    try:
        from app.core.database import engine

        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        checks["database"] = str(e)
        checks["status"] = "degraded"

    # Redis check
    try:
        import redis.asyncio as redis

        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = True
    except Exception as e:
        checks["redis"] = str(e)
        checks["status"] = "degraded"

    # MinIO check
    try:
        from app.infrastructure.storage import StorageClient

        storage = StorageClient()
        async with await storage._get_client() as client:
            await client.list_buckets()
        checks["minio"] = True
    except Exception as e:
        checks["minio"] = str(e)
        checks["status"] = "degraded"

    return checks
