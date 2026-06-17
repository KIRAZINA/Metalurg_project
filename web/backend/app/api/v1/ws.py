import asyncio
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.core.database import async_session
from app.domain.task import AsyncTask

router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_progress_ws(websocket: WebSocket, task_id: str):
    await websocket.accept()
    try:
        parsed = uuid.UUID(task_id)
    except ValueError:
        await websocket.send_json({"error": "Invalid task ID"})
        await websocket.close()
        return

    last_status = None
    last_progress = -1
    try:
        while True:
            async with async_session() as db:
                result = await db.execute(
                    select(AsyncTask).where(AsyncTask.id == parsed)
                )
                task = result.scalar_one_or_none()

            if not task:
                await websocket.send_json({"error": "Task not found"})
                await websocket.close()
                return

            status = task.status
            progress = task.progress
            if status != last_status or progress != last_progress:
                await websocket.send_json({
                    "status": status,
                    "progress": progress,
                    "error_message": task.error_message,
                    "result_ref": task.result_ref,
                })
                last_status = status
                last_progress = progress

            if status in ("SUCCESS", "FAILURE"):
                await websocket.close()
                return

            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
