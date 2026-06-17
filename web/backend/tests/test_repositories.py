import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.dataset import Dataset
from app.domain.optimization import ParetoOptimization, ParetoPoint
from app.domain.regression import RegressionModel
from app.domain.task import AsyncTask
from app.domain.user import User
from app.infrastructure.repositories.dataset_repository import (
    DatasetRepository,
)
from app.infrastructure.repositories.optimization_repository import (
    OptimizationRepository,
)
from app.infrastructure.repositories.regression_repository import (
    RegressionRepository,
)
from app.infrastructure.repositories.task_repository import TaskRepository


@pytest_asyncio.fixture
async def user(db_session: AsyncSession) -> User:
    user = User(
        email="repo-test@example.com",
        hashed_password="hashed_pw",
        full_name="Repo Test",
        role="analyst",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def dataset(db_session: AsyncSession, user: User) -> Dataset:
    ds = Dataset(
        user_id=user.id,
        name="Test Dataset",
        original_filename="test.xlsx",
        s3_key=f"datasets/{user.id}/ab/testhash.xlsx",
        file_size_bytes=1000,
        file_hash_sha256="testhash123",
        status="ready",
    )
    db_session.add(ds)
    await db_session.flush()
    await db_session.refresh(ds)
    return ds


class TestDatasetRepository:
    async def test_create_and_get(self, db_session: AsyncSession, user: User) -> None:
        repo = DatasetRepository(db_session)
        ds = Dataset(
            user_id=user.id,
            name="Create Test",
            original_filename="test.xlsx",
            s3_key="datasets/key.xlsx",
            file_size_bytes=500,
            file_hash_sha256="hash123",
            status="uploading",
        )
        created = await repo.create(ds)
        assert created.id is not None
        assert created.name == "Create Test"

        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_find_by_hash(
        self, db_session: AsyncSession, dataset: Dataset
    ) -> None:
        repo = DatasetRepository(db_session)
        found = await repo.find_by_hash(dataset.file_hash_sha256)
        assert found is not None
        assert found.id == dataset.id

    async def test_list_for_user(
        self, db_session: AsyncSession, user: User, dataset: Dataset
    ) -> None:
        repo = DatasetRepository(db_session)
        items, total = await repo.list_for_user(user.id)
        assert total >= 1
        assert len(items) >= 1

    async def test_delete(
        self, db_session: AsyncSession, dataset: Dataset
    ) -> None:
        repo = DatasetRepository(db_session)
        await repo.delete(dataset.id)
        fetched = await repo.get_by_id(dataset.id)
        assert fetched is None


class TestRegressionRepository:
    async def test_create_and_list(
        self, db_session: AsyncSession, dataset: Dataset
    ) -> None:
        repo = RegressionRepository(db_session)
        model = RegressionModel(
            dataset_id=dataset.id,
            x_column="S_before",
            y_column="S_after",
            slope=0.5,
            intercept=1.0,
            r_squared=0.85,
            p_value=0.001,
            std_err=0.1,
            ci_lower=0.3,
            ci_upper=0.7,
            n_obs=100,
            x_min=0.0,
            x_max=1.0,
            confidence="high",
            is_feasible=True,
        )
        created = await repo.create(model)
        assert created.id is not None

        models = await repo.list_for_dataset(dataset.id)
        assert len(models) == 1
        assert models[0].x_column == "S_before"


class TestOptimizationRepository:
    async def test_create_optimization_and_points(
        self, db_session: AsyncSession, user: User, dataset: Dataset
    ) -> None:
        repo = OptimizationRepository(db_session)
        opt = ParetoOptimization(
            dataset_id=dataset.id,
            user_id=user.id,
            targets={"S_after": 0.05},
            n_points=50,
        )
        created = await repo.create(opt)
        assert created.id is not None

        points = [
            ParetoPoint(
                optimization_id=created.id,
                ratio=1.5,
                total_input=10.0,
                total_output=15.0,
                efficiency=1.5,
                inputs={"S_before": 0.1},
                outputs={"S_after": 0.15},
                is_dominated=False,
                rank=1,
            ),
            ParetoPoint(
                optimization_id=created.id,
                ratio=1.2,
                total_input=12.0,
                total_output=14.4,
                efficiency=1.2,
                inputs={"S_before": 0.12},
                outputs={"S_after": 0.14},
                is_dominated=True,
                rank=2,
            ),
        ]
        await repo.create_points(points)

        items, total = await repo.list_points(created.id)
        assert total == 2
        assert len(items) == 2

    async def test_list_optimizations(
        self, db_session: AsyncSession, user: User, dataset: Dataset
    ) -> None:
        repo = OptimizationRepository(db_session)
        opt = ParetoOptimization(
            dataset_id=dataset.id,
            user_id=user.id,
            targets={"Si_after": 0.1},
        )
        await repo.create(opt)
        items, total = await repo.list_for_user(user.id)
        assert total >= 1


class TestTaskRepository:
    async def test_create_and_update(
        self, db_session: AsyncSession, user: User
    ) -> None:
        repo = TaskRepository(db_session)
        task = AsyncTask(
            id=uuid.uuid4(),
            user_id=user.id,
            task_type="regression",
            status="PENDING",
        )
        created = await repo.create(task)
        assert created.id == task.id

        await repo.update_progress(
            task.id, status="PROGRESS", progress=50
        )
        updated = await repo.get_by_id(task.id)
        assert updated is not None
        assert updated.status == "PROGRESS"
        assert updated.progress == 50
