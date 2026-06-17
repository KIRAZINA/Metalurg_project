import csv
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.domain.dataset import Dataset
from app.domain.optimization import ParetoOptimization
from app.domain.regression import RegressionModel
from app.domain.user import User
from app.infrastructure.repositories.optimization_repository import OptimizationRepository

router = APIRouter()


@router.get("/regression/{regression_id}.csv")
async def export_regression_csv(
    regression_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    model = await db.get(RegressionModel, regression_id)
    if not model:
        raise HTTPException(status_code=404, detail="Regression model not found")
    dataset = await db.get(Dataset, model.dataset_id)
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "metric", "value"
    ])
    writer.writerow(["x_column", model.x_column])
    writer.writerow(["y_column", model.y_column])
    writer.writerow(["slope", model.slope])
    writer.writerow(["intercept", model.intercept])
    writer.writerow(["r_squared", model.r_squared])
    writer.writerow(["p_value", model.p_value])
    writer.writerow(["std_err", model.std_err])
    writer.writerow(["ci_lower", model.ci_lower])
    writer.writerow(["ci_upper", model.ci_upper])
    writer.writerow(["n_obs", model.n_obs])
    writer.writerow(["x_min", model.x_min])
    writer.writerow(["x_max", model.x_max])
    writer.writerow(["confidence", model.confidence])
    writer.writerow(["is_feasible", model.is_feasible])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=regression_{regression_id}.csv"
        },
    )


@router.get("/optimization/{optimization_id}.csv")
async def export_optimization_csv(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    opt = await db.get(ParetoOptimization, optimization_id)
    if not opt:
        raise HTTPException(status_code=404, detail="Optimization not found")
    if opt.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    repo = OptimizationRepository(db)
    points, _ = await repo.list_points(optimization_id, offset=0, limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "ratio", "total_input", "total_output", "efficiency",
        "is_dominated", "rank", "inputs", "outputs",
    ])
    for p in points:
        writer.writerow([
            str(p.id), p.ratio, p.total_input, p.total_output, p.efficiency,
            p.is_dominated, p.rank, str(p.inputs), str(p.outputs),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=optimization_{optimization_id}.csv"
        },
    )
