import logging

import pandas as pd
import statsmodels.api as sm

from test_metal.core.models import OLSResult


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
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()
    params = model.params
    bse = model.bse
    conf = model.conf_int(alpha)
    pred = model.get_prediction(x_with_const).summary_frame(alpha=alpha)
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
    logging.info(
        "OLS %s ~ %s: intercept=%.6f slope=%.6f r2=%.6f", y_col, x_col, intercept, slope, r2
    )
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
