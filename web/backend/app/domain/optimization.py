import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.core.db_types import GUID


class ParetoOptimization(Base):
    __tablename__ = "pareto_optimizations"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    targets: Mapped[dict] = mapped_column(JSON, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="after")
    n_points: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    dataset = relationship("Dataset", back_populates="pareto_optimizations")
    points = relationship(
        "ParetoPoint", back_populates="optimization", cascade="all, delete-orphan"
    )


class ParetoPoint(Base):
    __tablename__ = "pareto_points"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    optimization_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("pareto_optimizations.id", ondelete="CASCADE"),
        nullable=False,
    )
    ratio: Mapped[float] = mapped_column(nullable=False)
    total_input: Mapped[float] = mapped_column(nullable=False)
    total_output: Mapped[float] = mapped_column(nullable=False)
    efficiency: Mapped[float] = mapped_column(nullable=False)
    inputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    outputs: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_dominated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (Index("idx_pareto_points_opt", "optimization_id"),)

    optimization = relationship("ParetoOptimization", back_populates="points")
