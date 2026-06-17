from pathlib import Path
from typing import Any

import pandas as pd

from test_metal.core.models import OLSResult, ParetoOptimum


def save_regression_csv(models: list[OLSResult], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for res in models:
        rows.append(
            {
                "x_col": res.x_col,
                "y_col": res.y_col,
                "intercept": res.intercept,
                "slope": res.slope,
                "stderr_intercept": res.stderr_intercept,
                "stderr_slope": res.stderr_slope,
                "pvalue_intercept": res.pvalue_intercept,
                "pvalue_slope": res.pvalue_slope,
                "r2": res.r2,
                "df_resid": res.df_resid,
                "nobs": res.nobs,
                "conf_int_intercept_low": res.conf_int_intercept_low,
                "conf_int_intercept_high": res.conf_int_intercept_high,
                "conf_int_slope_low": res.conf_int_slope_low,
                "conf_int_slope_high": res.conf_int_slope_high,
            }
        )
    report_df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(path, index=False)


def save_optimization_csv(pareto_front: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pareto_front.to_csv(path, index=False)


def build_pareto_rows(pareto_solutions: list[ParetoOptimum]) -> pd.DataFrame:
    pareto_rows: list[dict[str, Any]] = []
    for sol in pareto_solutions:
        row: dict[str, Any] = {"solution_id": sol.solution_id}
        for elem, val in sol.input_values.items():
            row[f"{elem}_input"] = val
        for elem, val in sol.output_values.items():
            row[f"{elem}_output"] = val
        row["total_impurity_input"] = sol.total_impurity_input
        row["total_impurity_output"] = sol.total_impurity_output
        row["efficiency_%"] = sol.efficiency
        pareto_rows.append(row)
    return pd.DataFrame(pareto_rows)
