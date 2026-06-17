import json
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.domain.task import AsyncTask
from app.domain.user import User
from app.infrastructure.repositories.task_repository import TaskRepository
from app.schemas.common import PaginationParams, pagination_dependency
from app.schemas.optimization import (
    ParetoOptimizationListResponse,
    ParetoOptimizationResponse,
    ParetoPointListResponse,
)
from app.services.optimization_service import OptimizationService
from app.workers.tasks import run_optimization

router = APIRouter()


@router.post("", response_model=ParetoOptimizationResponse, status_code=status.HTTP_201_CREATED)
async def create_optimization(
    dataset_id: uuid.UUID,
    name: str | None = None,
    targets: str = "{}",
    mode: str = "after",
    n_points: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ParetoOptimizationResponse:
    service = OptimizationService(db)
    optimization = await service.create(
        dataset_id=dataset_id,
        user_id=current_user.id,
        name=name,
        targets=json.loads(targets),
        mode=mode,
        n_points=n_points,
    )

    async_task_id = uuid.uuid4()
    task_repo = TaskRepository(db)
    async_task = AsyncTask(
        id=async_task_id,
        user_id=current_user.id,
        task_type="run_optimization",
        status="PENDING",
    )
    await task_repo.create(async_task)
    await db.commit()

    run_optimization.delay(str(optimization.id), str(async_task_id), str(current_user.id))
    return optimization


@router.get("", response_model=ParetoOptimizationListResponse)
async def list_optimizations(
    pagination: PaginationParams = Depends(pagination_dependency),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ParetoOptimizationListResponse:
    service = OptimizationService(db)
    return await service.list_for_user(
        user_id=current_user.id,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/{optimization_id}", response_model=ParetoOptimizationResponse)
async def get_optimization(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ParetoOptimizationResponse:
    service = OptimizationService(db)
    return await service.get_by_id(optimization_id, current_user.id)


@router.get("/{optimization_id}/points", response_model=ParetoPointListResponse)
async def list_optimization_points(
    optimization_id: uuid.UUID,
    pagination: PaginationParams = Depends(pagination_dependency),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ParetoPointListResponse:
    service = OptimizationService(db)
    return await service.list_points(
        optimization_id=optimization_id,
        user_id=current_user.id,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.delete("/{optimization_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_optimization(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    service = OptimizationService(db)
    await service.delete(optimization_id, current_user.id)
