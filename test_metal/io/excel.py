"""Excel ingestion helpers."""

from pathlib import Path

import pandas as pd

from test_metal.config import ProjectConfig

_ENGINES: dict[str, str] = {
    ".xls": "xlrd",
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
}


def resolve_engine(path: Path) -> str | None:
    """Return the pandas engine for a spreadsheet suffix (None = let pandas infer)."""
    return _ENGINES.get(path.suffix.lower())


def load_excel(
    path: Path | str,
    *,
    header_row: int | None = None,
    usecols: str | None = None,
    engine: str | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    """Load an Excel file using the engine matching its extension."""
    cfg = config or ProjectConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    resolved_engine = engine or resolve_engine(p)
    return pd.read_excel(
        p,
        header=header_row if header_row is not None else cfg.excel_header_row,
        usecols=usecols or cfg.excel_usecols,
        engine=resolved_engine,
    )
