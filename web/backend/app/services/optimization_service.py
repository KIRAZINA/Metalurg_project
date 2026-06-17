import uuid

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.dataset import Dataset
from app.domain.optimization import ParetoOptimization
from app.infrastructure.repositories.optimization_repository import (
    OptimizationRepository,
)
from app.schemas.optimization import (
    ParetoOptimizationListResponse,
    ParetoOptimizationResponse,
    ParetoPointListResponse,
    ParetoPointResponse,
)


class OptimizationService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = OptimizationRepository(db)

    async def create(
        self,
        dataset_id: uuid.UUID,
        user_id: uuid.UUID,
        name: str | None,
        targets: dict,
        mode: str,
        n_points: int,
    ) -> ParetoOptimizationResponse:
        dataset = await self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        if dataset.status != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset status is '{dataset.status}', must be 'ready'",
            )

        optimization = ParetoOptimization(
            dataset_id=dataset_id,
            user_id=user_id,
            name=name,
            targets=targets,
            mode=mode,
            n_points=n_points,
            status="pending",
        )
        optimization = await self.repo.create(optimization)
        return ParetoOptimizationResponse.model_validate(optimization)

    async def get_by_id(
        self, optimization_id: uuid.UUID, user_id: uuid.UUID
    ) -> ParetoOptimizationResponse:
        optimization = await self.repo.get_by_id(optimization_id)
        if not optimization:
            raise HTTPException(status_code=404, detail="Optimization not found")
        if optimization.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        return ParetoOptimizationResponse.model_validate(optimization)

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> ParetoOptimizationListResponse:
        items, total = await self.repo.list_for_user(
            user_id=user_id,
            offset=(page - 1) * page_size,
            limit=page_size,
        )
        return ParetoOptimizationListResponse(
            items=[ParetoOptimizationResponse.model_validate(o) for o in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def list_points(
        self,
        optimization_id: uuid.UUID,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 100,
    ) -> ParetoPointListResponse:
        optimization = await self.repo.get_by_id(optimization_id)
        if not optimization:
            raise HTTPException(status_code=404, detail="Optimization not found")
        if optimization.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")

        items, total = await self.repo.list_points(
            optimization_id=optimization_id,
            offset=(page - 1) * page_size,
            limit=page_size,
        )
        return ParetoPointListResponse(
            items=[ParetoPointResponse.model_validate(p) for p in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def delete(self, optimization_id: uuid.UUID, user_id: uuid.UUID) -> None:
        optimization = await self.repo.get_by_id(optimization_id)
        if not optimization:
            raise HTTPException(status_code=404, detail="Optimization not found")
        if optimization.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        await self.db.delete(optimization)
        await self.db.flush()
