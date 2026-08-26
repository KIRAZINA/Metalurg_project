"""DataFrame preprocessing: numeric coercion and missing-value handling."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.apply(pd.to_numeric, errors="coerce")
    n_coerced = int((coerced.isna() & df.notna()).sum().sum())
    if n_coerced:
        logger.info("Coerced %d non-numeric cells to NaN", n_coerced)
    return coerced


def clean_missing(df: pd.DataFrame, col_threshold: float = 0.5) -> pd.DataFrame:
    thresh = int(len(df) * col_threshold)
    before = df.shape
    out = df.dropna(axis=1, thresh=thresh)
    dropped = [c for c in df.columns if c not in out.columns]
    filled = out.fillna(out.mean(numeric_only=True))
    logger.info(
        "clean_missing: %s -> %s (dropped %d columns below %.0f%% non-null threshold: %s)",
        before,
        filled.shape,
        len(dropped),
        col_threshold * 100,
        dropped or "none",
    )
    return filled


def preprocess(df: pd.DataFrame, col_threshold: float = 0.5) -> pd.DataFrame:
    return clean_missing(to_numeric(df), col_threshold=col_threshold)
