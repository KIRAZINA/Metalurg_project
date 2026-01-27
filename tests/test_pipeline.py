import pandas as pd
import numpy as np
from test_metal.features import COLUMN_NAMES, PREDICTORS_AFTER, TARGET_AFTER, PREDICTORS_BEFORE, TARGET_BEFORE
from test_metal.preprocessing import preprocess
from test_metal.modeling import train_linear, train_tree

def synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {c: rng.random(n) for c in COLUMN_NAMES}
    data["sample_number"] = rng.integers(0, 1000, size=n)
    return pd.DataFrame(data)

def test_preprocess_shapes():
    df = synthetic_df()
    dfp = preprocess(df)
    assert not dfp.isna().any().any()
    assert set(PREDICTORS_AFTER + [TARGET_AFTER]).issubset(set(dfp.columns))

def test_models_after():
    df = synthetic_df()
    dfp = preprocess(df)
    lin = train_linear(dfp, PREDICTORS_AFTER, TARGET_AFTER)
    tree = train_tree(dfp, PREDICTORS_AFTER, TARGET_AFTER)
    assert "mse" in lin and "r2" in lin and "cv_r2_mean" in lin
    assert "mse" in tree and "r2" in tree

def test_models_before():
    df = synthetic_df()
    dfp = preprocess(df)
    lin = train_linear(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
    tree = train_tree(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
    assert "mse" in lin and "r2" in lin and "cv_r2_mean" in lin
    assert "mse" in tree and "r2" in tree
