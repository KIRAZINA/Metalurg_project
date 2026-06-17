from fastapi import APIRouter

from app.api.v1 import auth, datasets, health, optimizations, regressions, reports, tasks, ws

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["health"])
api_v1_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_v1_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_v1_router.include_router(
    regressions.router, prefix="/regressions", tags=["regressions"]
)
api_v1_router.include_router(
    optimizations.router, prefix="/optimizations", tags=["optimizations"]
)
api_v1_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_v1_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_v1_router.include_router(ws.router, tags=["websocket"])
