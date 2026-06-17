import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.regression import RegressionModel


class RegressionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, regression_id: uuid.UUID) -> RegressionModel | None:
        return await self.db.get(RegressionModel, regression_id)

    async def list_for_dataset(
        self, dataset_id: uuid.UUID
    ) -> list[RegressionModel]:
        result = await self.db.execute(
            select(RegressionModel)
            .where(RegressionModel.dataset_id == dataset_id)
            .order_by(RegressionModel.r_squared.desc())
        )
        return list(result.scalars().all())

    async def create(self, model: RegressionModel) -> RegressionModel:
        self.db.add(model)
        await self.db.flush()
        await self.db.refresh(model)
        return model

    async def create_many(
        self, models: list[RegressionModel]
    ) -> list[RegressionModel]:
        self.db.add_all(models)
        await self.db.flush()
        for m in models:
            await self.db.refresh(m)
        return models

    async def delete_by_dataset(self, dataset_id: uuid.UUID) -> None:
        await self.db.execute(
            select(RegressionModel).where(
                RegressionModel.dataset_id == dataset_id
            )
        )
        models = await self.list_for_dataset(dataset_id)
        for m in models:
            await self.db.delete(m)
