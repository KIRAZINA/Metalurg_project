from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.common import PaginatedResponse


class TaskResponse(BaseModel):
    id: UUID
    user_id: UUID
    task_type: str
    status: str
    progress: int
    result_ref: dict | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TaskListResponse(PaginatedResponse):
    items: list[TaskResponse]
