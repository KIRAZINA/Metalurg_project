from dataclasses import dataclass

import pandas as pd


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


@dataclass
class OptimizationResult:
    element: str
    target_output: float
    required_input: float
    predicted_output: float
    r2_score: float
    is_feasible: bool
    confidence: str
    notes: list[str]


@dataclass
class ParetoOptimum:
    solution_id: int
    input_values: dict[str, float]
    output_values: dict[str, float]
    total_impurity_input: float
    total_impurity_output: float
    efficiency: float
