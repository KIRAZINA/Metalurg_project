import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.dataset import Dataset
from app.domain.user import User
from app.infrastructure.repositories.dataset_repository import DatasetRepository
from app.infrastructure.storage import StorageClient, build_s3_key, get_storage
from app.schemas.dataset import DatasetListResponse, DatasetResponse, DatasetUpdate

ALLOWED_EXTENSIONS = {".xls", ".xlsx"}
XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
XLSX_MAGIC = b"PK\x03\x04"


class DatasetService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = DatasetRepository(db)

    async def validate_upload(self, file: UploadFile) -> None:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        header = await file.read(8)
        await file.seek(0)
        if ext == ".xls" and header[:8] != XLS_MAGIC:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is not a valid .xls file",
            )
        if ext == ".xlsx" and header[:4] != XLSX_MAGIC:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is not a valid .xlsx file",
            )

    async def create_from_upload(
        self,
        file: UploadFile,
        name: str,
        description: str | None,
        user: User,
    ) -> DatasetResponse:
        await self.validate_upload(file)

        storage: StorageClient = get_storage()
        await storage.ensure_bucket()

        upload_result = await storage.upload_file(
            file, f"temp/{uuid.uuid4()}_{file.filename}"
        )

        existing = await self.repo.find_by_hash(upload_result["hash_sha256"])
        if existing:
            await storage.delete(upload_result["key"])
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate file already exists as dataset '{existing.name}'",
            )

        s3_key = build_s3_key(
            user.id, file.filename or "unknown", upload_result["hash_sha256"]
        )
        await storage.copy_object(upload_result["key"], s3_key)
        await storage.delete(upload_result["key"])

        dataset = Dataset(
            user_id=user.id,
            name=name,
            description=description,
            original_filename=file.filename or "unknown",
            s3_key=s3_key,
            file_size_bytes=upload_result["size"],
            file_hash_sha256=upload_result["hash_sha256"],
            status="uploading",
        )
        dataset = await self.repo.create(dataset)
        return DatasetResponse.model_validate(dataset)

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
        status_filter: str | None = None,
        sort: str | None = "-created_at",
    ) -> DatasetListResponse:
        items, total = await self.repo.list_for_user(
            user_id=user_id,
            offset=(page - 1) * page_size,
            limit=page_size,
            status_filter=status_filter,
            sort=sort,
        )
        return DatasetListResponse(
            items=[DatasetResponse.model_validate(d) for d in items],
            total=total,
            page=page,
            page_size=page_size,
        )

    async def get_by_id(
        self, dataset_id: uuid.UUID, user_id: uuid.UUID
    ) -> DatasetResponse:
        dataset = await self.repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        return DatasetResponse.model_validate(dataset)

    async def delete(self, dataset_id: uuid.UUID, user_id: uuid.UUID) -> None:
        dataset = await self.repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        storage = get_storage()
        await storage.delete(dataset.s3_key)
        await self.repo.delete(dataset_id)

    async def update(
        self, dataset_id: uuid.UUID, user_id: uuid.UUID, data: DatasetUpdate
    ) -> DatasetResponse:
        dataset = await self.repo.get_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        if data.name is not None:
            dataset.name = data.name
        if data.description is not None:
            dataset.description = data.description
        await self.db.flush()
        await self.db.refresh(dataset)
        return DatasetResponse.model_validate(dataset)
