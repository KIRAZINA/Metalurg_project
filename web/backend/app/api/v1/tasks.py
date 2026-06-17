import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.domain.user import User
from app.infrastructure.repositories.task_repository import TaskRepository
from app.schemas.common import PaginationParams, pagination_dependency
from app.schemas.task import TaskListResponse, TaskResponse

router = APIRouter()


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    pagination: PaginationParams = Depends(pagination_dependency),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskListResponse:
    repo = TaskRepository(db)
    items, total = await repo.list_for_user(
        user_id=current_user.id,
        offset=pagination.offset,
        limit=pagination.page_size,
    )
    return TaskListResponse(
        items=[TaskResponse.model_validate(t) for t in items],
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TaskResponse:
    repo = TaskRepository(db)
    task = await repo.get_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return TaskResponse.model_validate(task)
