import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.domain.regression import RegressionModel
from app.domain.user import User
from app.schemas.regression import RegressionModelResponse

router = APIRouter()


@router.get("/{regression_id}", response_model=RegressionModelResponse)
async def get_regression(
    regression_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RegressionModel:
    model = await db.get(RegressionModel, regression_id)
    if not model:
        raise HTTPException(status_code=404, detail="Regression model not found")
    from app.services.regression_service import RegressionService

    svc = RegressionService(db)
    await svc.check_access(model, current_user.id)
    return model
