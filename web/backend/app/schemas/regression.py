from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class RegressionModelResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    x_column: str
    y_column: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    x_min: float
    x_max: float
    confidence: str
    is_feasible: bool | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RegressionModelListResponse(BaseModel):
    items: list[RegressionModelResponse]
    total: int
