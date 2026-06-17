from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    excel_header_row: int = 3
    excel_usecols: str = "B:CN"
    missing_threshold: float = 0.5
    r2_high: float = 0.8
    r2_medium: float = 0.6
    r2_min_feasible: float = 0.3
    slope_min_abs: float = 1e-10
    outputs_dir: Path = Path("outputs")
    log_filename: str = "run.log"
