import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.domain.task import AsyncTask
from app.domain.user import User
from app.infrastructure.repositories.task_repository import TaskRepository
from app.schemas.common import PaginationParams, pagination_dependency
from app.schemas.dataset import DatasetListResponse, DatasetResponse, DatasetUpdate
from app.services.dataset_service import DatasetService

router = APIRouter()


@router.post("", response_model=DatasetResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetResponse:
    service = DatasetService(db)
    dataset = await service.create_from_upload(
        file=file, name=name, description=description, user=current_user
    )
    return dataset


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    pagination: PaginationParams = Depends(pagination_dependency),
    status_filter: str | None = None,
    sort: str | None = "-created_at",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetListResponse:
    service = DatasetService(db)
    return await service.list_for_user(
        user_id=current_user.id,
        page=pagination.page,
        page_size=pagination.page_size,
        status_filter=status_filter,
        sort=sort,
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetResponse:
    service = DatasetService(db)
    return await service.get_by_id(dataset_id, current_user.id)


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: uuid.UUID,
    data: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DatasetResponse:
    service = DatasetService(db)
    return await service.update(dataset_id, current_user.id, data)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    service = DatasetService(db)
    await service.delete(dataset_id, current_user.id)


@router.post("/{dataset_id}/regressions", status_code=status.HTTP_202_ACCEPTED)
async def trigger_regression(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    from app.workers.tasks import process_dataset

    async_task_id = uuid.uuid4()
    task_repo = TaskRepository(db)
    async_task = AsyncTask(
        id=async_task_id,
        user_id=current_user.id,
        task_type="process_dataset",
        status="PENDING",
    )
    await task_repo.create(async_task)
    await db.commit()

    process_dataset.delay(str(dataset_id), str(async_task_id), str(current_user.id))
    return {"task_id": str(async_task_id), "status": "accepted"}


@router.get("/{dataset_id}/regressions")
async def list_regressions_for_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    from app.services.regression_service import RegressionService

    service = RegressionService(db)
    models = await service.list_for_dataset(dataset_id, current_user.id)
    return {"items": models}
