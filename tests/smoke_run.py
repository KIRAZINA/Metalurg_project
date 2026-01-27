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

def main():
    df = synthetic_df()
    dfp = preprocess(df)
    after_lin = train_linear(dfp, PREDICTORS_AFTER, TARGET_AFTER)
    after_tree = train_tree(dfp, PREDICTORS_AFTER, TARGET_AFTER)
    before_lin = train_linear(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
    before_tree = train_tree(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
    print("OK", after_lin["mse"], after_tree["mse"], before_lin["mse"], before_tree["mse"])

if __name__ == "__main__":
    main()
