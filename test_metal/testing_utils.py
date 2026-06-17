"""
Scikit-learn based modeling utilities for testing and experimentation.

Contains train/test split helpers, linear regression with cross-validation,
and decision tree training. Used primarily by the test suite.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor


def split(
    df: pd.DataFrame, predictors: list[str], target: str, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_data = df[predictors]
    y = df[target]
    return train_test_split(x_data, y, test_size=test_size, random_state=seed)  # type: ignore[no-any-return]


def train_linear(df: pd.DataFrame, predictors: list[str], target: str) -> dict[str, object]:
    x_train, x_test, y_train, y_test = split(df, predictors, target)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, df[predictors], df[target], cv=5, scoring="r2")
    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "cv_r2_mean": float(np.mean(cv)),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_tree(
    df: pd.DataFrame,
    predictors: list[str],
    target: str,
    seed: int = 42,
    max_depth: int | None = None,
) -> dict[str, object]:
    x_train, x_test, y_train, y_test = split(df, predictors, target, seed=seed)
    model = DecisionTreeRegressor(random_state=seed, max_depth=max_depth)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"model": model, "mse": mse, "r2": r2, "y_test": y_test, "y_pred": y_pred}
