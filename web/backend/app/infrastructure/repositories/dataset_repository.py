import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.dataset import Dataset


class DatasetRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, dataset_id: uuid.UUID) -> Dataset | None:
        return await self.db.get(Dataset, dataset_id)

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
        status_filter: str | None = None,
        sort: str | None = "-created_at",
    ) -> tuple[list[Dataset], int]:
        query = select(Dataset).where(Dataset.user_id == user_id)
        count_query = select(Dataset).where(Dataset.user_id == user_id)

        if status_filter:
            query = query.where(Dataset.status == status_filter)
            count_query = count_query.where(Dataset.status == status_filter)

        if sort and sort.startswith("-"):
            col = getattr(Dataset, sort[1:], Dataset.created_at)
            query = query.order_by(col.desc())
        else:
            query = query.order_by(Dataset.created_at.desc())

        total_result = await self.db.execute(count_query)
        total = len(total_result.scalars().all())

        query = query.offset(offset).limit(limit)
        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return items, total

    async def create(self, dataset: Dataset) -> Dataset:
        self.db.add(dataset)
        await self.db.flush()
        await self.db.refresh(dataset)
        return dataset

    async def update_status(
        self, dataset_id: uuid.UUID, status: str, error_message: str | None = None
    ) -> None:
        values: dict = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        await self.db.execute(
            update(Dataset).where(Dataset.id == dataset_id).values(**values)
        )

    async def delete(self, dataset_id: uuid.UUID) -> None:
        dataset = await self.get_by_id(dataset_id)
        if dataset:
            await self.db.delete(dataset)
            await self.db.flush()

    async def find_by_hash(self, file_hash: str) -> Dataset | None:
        result = await self.db.execute(
            select(Dataset).where(Dataset.file_hash_sha256 == file_hash).limit(1)
        )
        return result.scalar_one_or_none()
