from pathlib import Path
import pandas as pd

def load_excel(file_path: str, header_row: int = 3, usecols: str = "B:CN") -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return pd.read_excel(p, header=header_row, usecols=usecols)
