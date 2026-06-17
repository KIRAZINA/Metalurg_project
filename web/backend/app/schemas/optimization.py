from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.common import PaginatedResponse


class ParetoOptimizationResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    user_id: UUID
    name: str | None
    targets: dict
    mode: str
    n_points: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ParetoOptimizationListResponse(PaginatedResponse):
    items: list[ParetoOptimizationResponse]


class ParetoPointResponse(BaseModel):
    id: UUID
    ratio: float
    total_input: float
    total_output: float
    efficiency: float
    inputs: dict
    outputs: dict
    is_dominated: bool
    rank: int | None

    model_config = {"from_attributes": True}


class ParetoPointListResponse(PaginatedResponse):
    items: list[ParetoPointResponse]
