import asyncio
import io
import uuid

import pandas as pd
from sqlalchemy import select
from test_metal.config import ProjectConfig
from test_metal.core.models import OLSResult
from test_metal.core.optimization import InverseRegression, ParetoOptimizer
from test_metal.pipeline import run_pipeline

from app.core.database import async_session
from app.domain.dataset import Dataset
from app.domain.optimization import ParetoPoint
from app.domain.regression import RegressionModel
from app.infrastructure.repositories.optimization_repository import (
    OptimizationRepository,
)
from app.infrastructure.repositories.task_repository import TaskRepository
from app.infrastructure.storage import get_storage
from app.infrastructure.task_queue import celery_app
from app.services.regression_service import RegressionService


def _update_async_task_db(task_id: str, status: str, progress: int = 0, error: str | None = None, result: dict | None = None) -> None:
    async def _inner():
        async with async_session() as db:
            repo = TaskRepository(db)
            await repo.update_progress(
                task_id=uuid.UUID(task_id),
                status=status,
                progress=progress,
                error_message=error,
                result_ref=result,
            )
            await db.commit()
    asyncio.run(_inner())


@celery_app.task(bind=True, name="process_dataset")
def process_dataset(self, dataset_id: str, async_task_id: str, user_id: str) -> dict:
    return asyncio.run(_process_dataset_async(self, dataset_id, async_task_id, user_id))


async def _process_dataset_async(task, dataset_id: str, async_task_id: str, user_id: str) -> dict:
    async with async_session() as db:
        dataset = await db.get(Dataset, dataset_id)
        if not dataset:
            _update_async_task_db(async_task_id, "FAILURE", 0, error="Dataset not found")
            return {"error": "Dataset not found"}

        try:
            dataset.status = "processing"
            await db.commit()
            _update_async_task_db(async_task_id, "PROGRESS", 10)

            storage = get_storage()
            content = await storage.download_bytes(dataset.s3_key)
            _update_async_task_db(async_task_id, "PROGRESS", 20)

            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            if len(df) > 10_000:
                raise ValueError(f"Dataset too large: {len(df)} rows")
            _update_async_task_db(async_task_id, "PROGRESS", 40)

            config = ProjectConfig()
            result = run_pipeline(df=df, config=config)
            _update_async_task_db(async_task_id, "PROGRESS", 80)

            reg_service = RegressionService(db)
            await reg_service.save_results(dataset.id, result.models)

            dataset.row_count = len(df)
            dataset.column_count = len(df.columns)
            dataset.status = "ready"
            await db.commit()
            _update_async_task_db(
                async_task_id, "SUCCESS", 100,
                result={"dataset_id": dataset_id, "models_count": len(result.models)},
            )
            return {
                "dataset_id": str(dataset.id),
                "models_count": len(result.models),
            }

        except Exception as e:
            dataset.status = "failed"
            dataset.error_message = str(e)
            await db.commit()
            _update_async_task_db(async_task_id, "FAILURE", 0, error=str(e))
            raise


@celery_app.task(bind=True, name="run_optimization")
def run_optimization(self, optimization_id: str, async_task_id: str, user_id: str) -> dict:
    return asyncio.run(_run_optimization_async(self, optimization_id, async_task_id, user_id))


async def _run_optimization_async(task, optimization_id: str, async_task_id: str, user_id: str) -> dict:
    async with async_session() as db:
        opt_repo = OptimizationRepository(db)
        optimization = await opt_repo.get_by_id(uuid.UUID(optimization_id))
        if not optimization:
            _update_async_task_db(async_task_id, "FAILURE", 0, error="Optimization not found")
            return {"error": "Optimization not found"}

        try:
            optimization.status = "processing"
            await db.flush()
            _update_async_task_db(async_task_id, "PROGRESS", 10)

            result = await db.execute(
                select(RegressionModel).where(
                    RegressionModel.dataset_id == optimization.dataset_id
                )
            )
            models = result.scalars().all()
            if not models:
                raise ValueError("No regression models found for this dataset")
            _update_async_task_db(async_task_id, "PROGRESS", 30)

            ols_results = [
                OLSResult(
                    x_col=m.x_column,
                    y_col=m.y_column,
                    intercept=m.intercept,
                    slope=m.slope,
                    stderr_intercept=0.0,
                    stderr_slope=m.std_err,
                    pvalue_intercept=0.0,
                    pvalue_slope=m.p_value,
                    r2=m.r_squared,
                    df_resid=m.n_obs - 2,
                    nobs=m.n_obs,
                    conf_int_intercept_low=m.ci_lower,
                    conf_int_intercept_high=m.ci_upper,
                    conf_int_slope_low=m.ci_lower,
                    conf_int_slope_high=m.ci_upper,
                    x=pd.Series(dtype=float),
                    y=pd.Series(dtype=float),
                    y_hat=pd.Series(dtype=float),
                    mean_ci_low=pd.Series(dtype=float),
                    mean_ci_high=pd.Series(dtype=float),
                )
                for m in models
            ]
            _update_async_task_db(async_task_id, "PROGRESS", 50)

            element_targets: dict[str, tuple[str, float]] = {}
            for element, cfg in optimization.targets.items():
                x_col = cfg.get("x_column", "")
                target_value = cfg.get("target_value", 0.0)
                element_targets[element] = (x_col, target_value)

            if not element_targets:
                raise ValueError("No valid element targets defined")

            inverse = InverseRegression(ols_results)
            optimizer = ParetoOptimizer(inverse)
            solutions = optimizer.generate_pareto_front(
                element_targets, optimization.n_points
            )
            _update_async_task_db(async_task_id, "PROGRESS", 70)

            filtered = ParetoOptimizer.filter_pareto_front(solutions)
            dominated_ids = {s.solution_id for s in solutions} - {s.solution_id for s in filtered}

            points = []
            for sol in solutions:
                is_dominated = sol.solution_id in dominated_ids
                rank = None if is_dominated else 1
                points.append(
                    ParetoPoint(
                        optimization_id=optimization.id,
                        ratio=sol.solution_id / max(optimization.n_points - 1, 1),
                        total_input=sol.total_impurity_input,
                        total_output=sol.total_impurity_output,
                        efficiency=sol.efficiency,
                        inputs=sol.input_values,
                        outputs=sol.output_values,
                        is_dominated=is_dominated,
                        rank=rank,
                    )
                )

            await opt_repo.create_points(points)
            optimization.status = "completed"
            await db.flush()
            _update_async_task_db(
                async_task_id, "SUCCESS", 100,
                result={"optimization_id": optimization_id, "points_count": len(points)},
            )
            return {
                "optimization_id": optimization_id,
                "points_count": len(points),
            }

        except Exception as e:
            optimization.status = "failed"
            optimization.error_message = str(e)
            await db.flush()
            _update_async_task_db(async_task_id, "FAILURE", 0, error=str(e))
            raise
