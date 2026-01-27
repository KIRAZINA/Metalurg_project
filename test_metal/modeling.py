from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def split(df: pd.DataFrame, predictors: List[str], target: str, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[predictors]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed)

def train_linear(df: pd.DataFrame, predictors: List[str], target: str) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = split(df, predictors, target)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, df[predictors], df[target], cv=5, scoring="r2")
    return {"model": model, "mse": mse, "r2": r2, "cv_r2_mean": float(np.mean(cv)), "y_test": y_test, "y_pred": y_pred}

def train_tree(df: pd.DataFrame, predictors: List[str], target: str, seed: int = 42, max_depth: int | None = None) -> Dict[str, object]:
    X_train, X_test, y_train, y_test = split(df, predictors, target, seed=seed)
    model = DecisionTreeRegressor(random_state=seed, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"model": model, "mse": mse, "r2": r2, "y_test": y_test, "y_pred": y_pred}

@dataclass
class OLSResult:
    x_col: str
    y_col: str
    intercept: float
    slope: float
    stderr_intercept: float
    stderr_slope: float
    pvalue_intercept: float
    pvalue_slope: float
    r2: float
    df_resid: float
    nobs: int
    conf_int_intercept_low: float
    conf_int_intercept_high: float
    conf_int_slope_low: float
    conf_int_slope_high: float
    x: pd.Series
    y: pd.Series
    y_hat: pd.Series
    mean_ci_low: pd.Series
    mean_ci_high: pd.Series

def fit_ols(df: pd.DataFrame, x_col: str, y_col: str, alpha: float = 0.05) -> OLSResult:
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Missing columns for regression: {x_col}, {y_col}")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if x.empty:
        raise ValueError(f"No valid data for regression {y_col} ~ {x_col}")
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    params = model.params
    bse = model.bse
    conf = model.conf_int(alpha)
    pred = model.get_prediction(X).summary_frame(alpha=alpha)
    intercept = float(params["const"])
    slope = float(params[x_col])
    stderr_intercept = float(bse["const"])
    stderr_slope = float(bse[x_col])
    conf_int_intercept_low = float(conf.loc["const", 0])
    conf_int_intercept_high = float(conf.loc["const", 1])
    conf_int_slope_low = float(conf.loc[x_col, 0])
    conf_int_slope_high = float(conf.loc[x_col, 1])
    r2 = float(model.rsquared)
    df_resid = float(model.df_resid)
    nobs = int(model.nobs)
    logging.info("OLS %s ~ %s: intercept=%.6f slope=%.6f r2=%.6f", y_col, x_col, intercept, slope, r2)
    return OLSResult(
        x_col=x_col,
        y_col=y_col,
        intercept=intercept,
        slope=slope,
        stderr_intercept=stderr_intercept,
        stderr_slope=stderr_slope,
        pvalue_intercept=float(model.pvalues["const"]),
        pvalue_slope=float(model.pvalues[x_col]),
        r2=r2,
        df_resid=df_resid,
        nobs=nobs,
        conf_int_intercept_low=conf_int_intercept_low,
        conf_int_intercept_high=conf_int_intercept_high,
        conf_int_slope_low=conf_int_slope_low,
        conf_int_slope_high=conf_int_slope_high,
        x=x,
        y=y,
        y_hat=pred["mean"],
        mean_ci_low=pred["mean_ci_lower"],
        mean_ci_high=pred["mean_ci_upper"],
    )
