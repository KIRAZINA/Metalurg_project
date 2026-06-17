import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.optimization import ParetoOptimization, ParetoPoint


class OptimizationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(
        self, optimization_id: uuid.UUID
    ) -> ParetoOptimization | None:
        return await self.db.get(ParetoOptimization, optimization_id)

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[ParetoOptimization], int]:
        query = select(ParetoOptimization).where(
            ParetoOptimization.user_id == user_id
        )
        count_result = await self.db.execute(query)
        total = len(count_result.scalars().all())
        query = query.order_by(ParetoOptimization.created_at.desc())
        query = query.offset(offset).limit(limit)
        result = await self.db.execute(query)
        items = list(result.scalars().all())
        return items, total

    async def create(
        self, optimization: ParetoOptimization
    ) -> ParetoOptimization:
        self.db.add(optimization)
        await self.db.flush()
        await self.db.refresh(optimization)
        return optimization

    async def list_points(
        self,
        optimization_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[ParetoPoint], int]:
        query = select(ParetoPoint).where(
            ParetoPoint.optimization_id == optimization_id
        )
        count_result = await self.db.execute(query)
        total = len(count_result.scalars().all())
        query = query.order_by(ParetoPoint.rank.asc().nullslast())
        query = query.offset(offset).limit(limit)
        result = await self.db.execute(query)
        items = list(result.scalars().all())
        return items, total

    async def create_points(
        self, points: list[ParetoPoint]
    ) -> list[ParetoPoint]:
        self.db.add_all(points)
        await self.db.flush()
        return points
