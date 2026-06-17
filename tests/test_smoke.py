"""Integration smoke test — automatically discovered by pytest."""

import numpy as np
import pandas as pd

from test_metal.features import (
    COLUMN_NAMES,
    PREDICTORS_AFTER,
    PREDICTORS_BEFORE,
    TARGET_AFTER,
    TARGET_BEFORE,
)
from test_metal.preprocessing import preprocess
from test_metal.testing_utils import train_linear, train_tree


def _synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {c: rng.random(n) for c in COLUMN_NAMES}
    data["sample_number"] = rng.integers(0, 1000, size=n)
    return pd.DataFrame(data)


def test_smoke_run():
    df = _synthetic_df()
    dfp = preprocess(df)

    after_lin = train_linear(dfp, PREDICTORS_AFTER, TARGET_AFTER)
    after_tree = train_tree(dfp, PREDICTORS_AFTER, TARGET_AFTER)
    before_lin = train_linear(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
    before_tree = train_tree(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)

    assert "mse" in after_lin
    assert "r2" in after_lin
    assert "model" in after_tree
    assert "model" in before_lin
    assert all(
        isinstance(v, float)
        for v in [after_lin["mse"], after_tree["mse"], before_lin["mse"], before_tree["mse"]]
    )
