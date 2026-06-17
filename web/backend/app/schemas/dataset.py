from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    VALIDATING = "validating"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class DatasetResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    original_filename: str
    file_size_bytes: int
    row_count: int | None
    column_count: int | None
    status: DatasetStatus
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class DatasetListResponse(BaseModel):
    items: list[DatasetResponse]
    total: int
    page: int
    page_size: int
