from pathlib import Path

import pandas as pd

from test_metal.config import ProjectConfig


def load_excel(
    path: Path | str,
    *,
    header_row: int | None = None,
    usecols: str | None = None,
    config: ProjectConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ProjectConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_excel(
        p,
        header=header_row if header_row is not None else cfg.excel_header_row,
        usecols=usecols or cfg.excel_usecols,
        engine="openpyxl",
    )
