import uuid

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.dataset import Dataset
from app.domain.regression import RegressionModel
from app.schemas.regression import RegressionModelResponse


class RegressionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def check_access(self, model: RegressionModel, user_id: uuid.UUID) -> None:
        dataset = await self.db.get(Dataset, model.dataset_id)
        if not dataset or dataset.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")

    async def list_for_dataset(
        self, dataset_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[RegressionModelResponse]:
        dataset = await self.db.get(Dataset, dataset_id)
        if not dataset or dataset.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        result = await self.db.execute(
            select(RegressionModel)
            .where(RegressionModel.dataset_id == dataset_id)
            .order_by(RegressionModel.r_squared.desc())
        )
        models = result.scalars().all()
        return [RegressionModelResponse.model_validate(m) for m in models]

    async def save_results(
        self, dataset_id: uuid.UUID, models: list
    ) -> list[RegressionModel]:
        saved: list[RegressionModel] = []
        for m in models:
            confidence = "high"
            if m.r2 < 0.6:
                confidence = "low"
            elif m.r2 < 0.8:
                confidence = "medium"
            is_feasible = m.r2 > 0.3
            row = RegressionModel(
                dataset_id=dataset_id,
                x_column=m.x_col,
                y_column=m.y_col,
                slope=m.slope,
                intercept=m.intercept,
                r_squared=m.r2,
                p_value=m.pvalue_slope,
                std_err=m.stderr_slope,
                ci_lower=m.conf_int_slope_low,
                ci_upper=m.conf_int_slope_high,
                n_obs=m.nobs,
                x_min=float(m.x.min()),
                x_max=float(m.x.max()),
                confidence=confidence,
                is_feasible=is_feasible,
            )
            self.db.add(row)
            saved.append(row)
        await self.db.flush()
        return saved
