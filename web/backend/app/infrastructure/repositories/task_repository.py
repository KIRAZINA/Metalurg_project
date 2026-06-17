import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.task import AsyncTask


class TaskRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, task_id: uuid.UUID) -> AsyncTask | None:
        return await self.db.get(AsyncTask, task_id)

    async def create(self, task: AsyncTask) -> AsyncTask:
        self.db.add(task)
        await self.db.flush()
        await self.db.refresh(task)
        return task

    async def update_progress(
        self,
        task_id: uuid.UUID,
        status: str,
        progress: int = 0,
        error_message: str | None = None,
        result_ref: dict | None = None,
    ) -> None:
        values: dict = {"status": status, "progress": progress}
        if error_message is not None:
            values["error_message"] = error_message
        if result_ref is not None:
            values["result_ref"] = result_ref
        await self.db.execute(
            update(AsyncTask)
            .where(AsyncTask.id == task_id)
            .values(**values)
        )

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[AsyncTask], int]:
        query = select(AsyncTask).where(AsyncTask.user_id == user_id)
        count_result = await self.db.execute(query)
        total = len(count_result.scalars().all())
        query = query.order_by(AsyncTask.created_at.desc())
        query = query.offset(offset).limit(limit)
        result = await self.db.execute(query)
        items = list(result.scalars().all())
        return items, total
