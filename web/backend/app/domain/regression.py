import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.core.db_types import GUID


class RegressionModel(Base):
    __tablename__ = "regression_models"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    x_column: Mapped[str] = mapped_column(String(100), nullable=False)
    y_column: Mapped[str] = mapped_column(String(100), nullable=False)
    slope: Mapped[float] = mapped_column(nullable=False)
    intercept: Mapped[float] = mapped_column(nullable=False)
    r_squared: Mapped[float] = mapped_column(nullable=False)
    p_value: Mapped[float] = mapped_column(nullable=False)
    std_err: Mapped[float] = mapped_column(nullable=False)
    ci_lower: Mapped[float] = mapped_column(nullable=False)
    ci_upper: Mapped[float] = mapped_column(nullable=False)
    n_obs: Mapped[int] = mapped_column(Integer, nullable=False)
    x_min: Mapped[float] = mapped_column(nullable=False)
    x_max: Mapped[float] = mapped_column(nullable=False)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False)
    is_feasible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (UniqueConstraint("dataset_id", "x_column", "y_column"),)

    dataset = relationship("Dataset", back_populates="regression_models")
