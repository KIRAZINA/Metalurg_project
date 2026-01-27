import pandas as pd

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")

def clean_missing(df: pd.DataFrame, col_threshold: float = 0.5) -> pd.DataFrame:
    thresh = int(len(df) * col_threshold)
    out = df.dropna(axis=1, thresh=thresh)
    return out.fillna(out.mean(numeric_only=True))

def preprocess(df: pd.DataFrame, col_threshold: float = 0.5) -> pd.DataFrame:
    return clean_missing(to_numeric(df), col_threshold=col_threshold)
