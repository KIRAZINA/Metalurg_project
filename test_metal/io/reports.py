"""CSV report writers and the formulas/calculation-details document."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from test_metal.features import optimization_model_columns

if TYPE_CHECKING:
    from pathlib import Path

    from test_metal.config import ProjectConfig
    from test_metal.core.models import OLSResult, ParetoOptimum

logger = logging.getLogger(__name__)


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


def save_optimization_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_pareto_rows(pareto_solutions: list[ParetoOptimum]) -> pd.DataFrame:
    """Build a DataFrame of Pareto-front rows including per-element metric columns.

    Per-element metric name and sign convention depend on the element's domain
    class (from ``optimization_model_columns()``):

    - **reduction** elements (e.g. Sulfur): column is ``{label_short}_reduction_pct``,
      formula is ``(input − output) / |input| × 100``. Positive = more reduction
      (larger is better, by name).
    - **additive** elements (e.g. Silicon): column is ``{label_short}_growth_pct``,
      formula is ``(output − input) / |input| × 100``. Positive = more growth
      (larger is worse, by name).

    Both formulas can be written as a single line of code:
    ``value = sign * (output − input) / |input| × 100`` where sign is
    ``-1`` for reduction and ``+1`` for additive (so the value's sign always
    matches the column name's meaning).

    The column name carries the behavior, so a reader looking at a value does
    not need to remember a sign convention: a value of +245 in
    ``Silicon (Si)_growth_pct`` reads as "245% growth" by name, and a value of
    +8 in ``Sulfur (S)_reduction_pct`` reads as "8% reduction" by name. The
    asymmetric formulas are the price of that honest naming; the alternative
    of one shared formula with a generic ``_change_pct`` name was rejected
    because it leaves the "which direction is good" question to the reader
    to infer from context (see §12.3 of PROJECT_ARCHITECTURE.md).
    """
    element_class_by_label: dict[str, str] = {
        label: cls for label, _x, _y, cls in optimization_model_columns()
    }

    def metric_columns(elem: str) -> tuple[str, float]:
        """Return (column_name, sign) for the given element label.

        The metric is computed as ``sign * (output − input) / |input| × 100``.
        The sign aligns the value's sign with the column name's meaning:

        - reduction elements: sign = −1, so positive value = (in − out) is
          positive = "more removed."
        - additive elements: sign = +1, so positive value = (out − in) is
          positive = "more added."
        - neutral / unknown: sign = +1, raw signed difference.
        """
        cls = element_class_by_label.get(elem)
        if cls == "reduction":
            return f"{elem}_reduction_pct", -1.0
        if cls == "additive":
            return f"{elem}_growth_pct", 1.0
        return f"{elem}_change_pct", 1.0

    pareto_rows: list[dict[str, Any]] = []
    for sol in pareto_solutions:
        row: dict[str, Any] = {"solution_id": sol.solution_id}
        for elem, val in sol.input_values.items():
            row[f"{elem}_input"] = val
        for elem, val in sol.output_values.items():
            row[f"{elem}_output"] = val
        all_elems = set(sol.input_values) | set(sol.output_values)
        for elem in sorted(all_elems):
            in_val: float | None = sol.input_values.get(elem)
            out_val: float | None = sol.output_values.get(elem)
            col_name, sign = metric_columns(elem)
            if in_val is not None and in_val != 0 and out_val is not None:
                row[col_name] = sign * (out_val - in_val) / abs(in_val) * 100
            else:
                row[col_name] = float("nan")
        row["total_impurity_input"] = sol.total_impurity_input
        row["total_impurity_output"] = sol.total_impurity_output
        pareto_rows.append(row)
    df = pd.DataFrame(pareto_rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "solution_id",
                "total_impurity_input",
                "total_impurity_output",
            ]
        )
    return df


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(f):
        return "nan"
    return f"{f:.6g}"


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = " | "
    line_sep = "-+-".join("-" * w for w in widths)
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers)), line_sep]
    for row in rows:
        lines.append(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def write_calculations_report(
    path: Path,
    *,
    source_file: str,
    mode: str,
    config: ProjectConfig,
    rows_raw: int,
    cols_raw: int,
    rows_clean: int,
    cols_clean: int,
    dropped_columns: list[str],
    models: list[OLSResult],
    single_element_report: pd.DataFrame | None,
    pareto_front: pd.DataFrame | None,
    pareto_solutions: list[ParetoOptimum],
) -> None:
    """Write ``calculations.md`` documenting formulas, per-model equations and results."""
    path.parent.mkdir(parents=True, exist_ok=True)

    model_by_xy: dict[tuple[str, str], OLSResult] = {(m.x_col, m.y_col): m for m in models}
    element_models: dict[str, OLSResult | None] = {}
    for label, x_col, y_col, _cls in optimization_model_columns():
        element_models[label] = model_by_xy.get((x_col, y_col))

    lines: list[str] = []
    add = lines.append
    add("# Test Metal — Calculations & Formulas")
    add("")
    add("This document records the statistical formulas, the fitted regression")
    add("equations, the inverse-regression (required-input) calculations and the")
    add("Pareto-optimization details produced by a single command-line run.")
    add("")

    add("## 1. Run configuration")
    add(f"- Input file: `{source_file}`")
    add(f"- Mode: `{mode}` (target: `steel_S_{mode}`)")
    add(f"- Excel header row: {config.excel_header_row}")
    add(f"- Excel column range: `{config.excel_usecols}`")
    add(
        f"- Missing-value threshold: {config.missing_threshold:.2f} "
        f"(columns with fewer non-null values than {config.missing_threshold:.0%} of rows are dropped)"
    )
    add(
        f"- R² thresholds: high > {config.r2_high}, medium > {config.r2_medium}, "
        f"min feasible > {config.r2_min_feasible}"
    )
    add(f"- Minimum |slope| to be treated as a real relationship: {config.slope_min_abs}")
    add("")

    add("## 2. Dataset")
    add(f"- Raw frame: **{rows_raw} rows × {cols_raw} columns**")
    add(f"- After preprocessing: **{rows_clean} rows × {cols_clean} columns**")
    if dropped_columns:
        add(f"- Dropped columns (below missing threshold): {', '.join(dropped_columns)}")
    else:
        add("- Dropped columns: none")
    add("")

    add("## 3. Formulas")
    add("### 3.1 Ordinary Least Squares regression (per predictor `x` → target `y`)")
    add("```")
    add("y = β0 + β1 · x")
    add("β1 = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)")
    add("β0 = ȳ - β1 · x̄")
    add("R² = 1 - SS_res / SS_tot                         (R² = r² for simple regression)")
    add("SE(β1) = sqrt(σ² / Σ((xi - x̄)²)),  σ² = Σei² / (n - 2)")
    add("t = β1 / SE(β1),  p = 2 · (1 - T_cdf(|t|, n - 2))")
    add("95% CI(β1) = β1 ± t_{0.975, n-2} · SE(β1)")
    add("95% prediction band ~ ŷ ± t_{0.975, df_resid} · SE_pred")
    add("```")
    add("")
    add("### 3.2 Inverse regression (required input for a target output)")
    add("```")
    add("x_required = (y_target - β0) / β1")
    add("Feasible  iff  x_min ≤ x_required ≤ x_max   AND   R² > R²_min (0.3)")
    add("Confidence: R² > 0.8 → high | R² > 0.6 → medium | otherwise low")
    add("Guard: |β1| < slope_min_abs → no correlation (NaN, infeasible)")
    add("```")
    add("")
    add("### 3.3 Pareto optimization (grid search)")
    add("```")
    add("For ratio r ∈ [0, 1] and each element i:")
    add("  target_at_xmin_i = β0_i + β1_i·x_min_i   (model output at observed x_min)")
    add("  target_at_xmax_i = β0_i + β1_i·x_max_i   (model output at observed x_max)")
    add("  target_i(r) = target_at_xmin_i + r · (target_at_xmax_i - target_at_xmin_i)")
    add("  required_input_i(r) = (target_i(r) - β0_i) / β1_i")
    add("  total_input  = Σ required_input_i")
    add("  total_output = Σ target_i(r)")
    add("Per-element metric — depends on the element's domain class:")
    add("  reduction elements (e.g. Sulfur):")
    add("    Sulfur (S)_reduction_pct = (input − output) / |input| × 100")
    add("    positive = more removed (larger is better, by column name)")
    add("  additive elements (e.g. Silicon):")
    add("    Silicon (Si)_growth_pct   = (output − input) / |input| × 100")
    add("    positive = more added (larger is worse, by column name)")
    add("Solutions with non-finite or non-feasible required_input are skipped.")
    add("```")
    add("")
    add("**Why two formulas, not one.** Earlier versions of the pipeline computed")
    add("a single `efficiency = (total_input − total_output) / total_input × 100`")
    add("column. That mixed S (a reduction, where output < input) with Si (an additive")
    add("element, where output > input) into one signed number, so a negative value")
    add("did not mean the process was bad — it meant additive growth outweighed")
    add("reduction. The column was structurally misleading, not just unlabeled.")
    add("")
    add("An interim version used a single shared formula `(input − output) / |input| × 100`")
    add("and called the result `{elem}_change_pct` for every element. That preserved")
    add("information but left the direction-of-good ambiguous: positive in S's column")
    add("means 'good' (reduced) while positive in Si's column means 'expected'")
    add("(grown) — and the column name didn't tell the reader which was which.")
    add("The current version uses a column name that names the *behavior*")
    add("(`_reduction_pct` for S, `_growth_pct` for Si) and a per-element-class")
    add("formula sign so that the value's sign matches the column name's meaning.")
    add("The element class is set in `OPTIMIZATION_ELEMENTS` (features.py).")
    add("")
    add("### 3.4 Pareto dominance filter")
    add("```")
    add("Candidate C is dominated iff ∃ solution S with:")
    add("  S.total_input ≤ C.total_input  AND  S.total_output ≤ C.total_output")
    add("  AND (strictly < on at least one objective).")
    add("Non-dominated solutions are kept and sorted by ascending total_input.")
    add("```")
    add("")

    add("## 4. Fitted regression models")
    add(f"Total models: {len(models)}")
    headers = ["#", "Model", "Intercept β0", "Slope β1", "R²", "p(slope)", "n"]
    rows: list[list[str]] = []
    for i, model in enumerate(models, 1):
        rows.append(
            [
                str(i),
                f"{model.y_col} ~ {model.x_col}",
                _fmt(model.intercept),
                _fmt(model.slope),
                _fmt(model.r2),
                _fmt(model.pvalue_slope),
                str(model.nobs),
            ]
        )
    add(_format_table(headers, rows))
    add("")

    add("## 5. Single-element optimization (inverse regression)")
    if single_element_report is None or single_element_report.empty:
        add("No optimizable S/Si before→after models were available; optimization skipped.")
    else:
        headers = [
            "Element",
            "Target y",
            "Model",
            "β0",
            "β1",
            "R²",
            "Required input x = (y - β0)/β1",
            "Predicted y",
            "Feasible",
            "Confidence",
            "Notes",
        ]
        rows = []
        for _, r in single_element_report.iterrows():
            m = element_models.get(str(r["element"]))
            rows.append(
                [
                    str(r["element"]),
                    _fmt(r["target_output"]),
                    (f"{m.y_col} ~ {m.x_col}" if m else "n/a"),
                    (_fmt(m.intercept) if m else ""),
                    (_fmt(m.slope) if m else ""),
                    (_fmt(m.r2) if m else ""),
                    _fmt(r["required_input"]),
                    _fmt(r["predicted_output"]),
                    "yes" if r["is_feasible"] else "no",
                    str(r["confidence"]),
                    str(r["notes"]),
                ]
            )
        add(_format_table(headers, rows))
    add("")

    add("## 6. Pareto front (non-dominated solutions)")
    add("- Solutions generated: see `optimization_report_pareto_front.csv`")
    add("- Full candidate sweep: see `pareto_candidates_full_sweep.csv`")
    add(f"- Non-dominated solutions: {len(pareto_solutions)}")
    if pareto_solutions:
        inputs = [s.total_impurity_input for s in pareto_solutions]
        outputs = [s.total_impurity_output for s in pareto_solutions]
        add(f"- Total input range: {_fmt(min(inputs))} .. {_fmt(max(inputs))}")
        add(f"- Total output range: {_fmt(min(outputs))} .. {_fmt(max(outputs))}")
        for label, _x, _y, cls in optimization_model_columns():
            metric_values: list[float] = []
            for s in pareto_solutions:
                if label not in s.input_values or label not in s.output_values:
                    continue
                ev_in: float | None = s.input_values.get(label)
                ev_out: float | None = s.output_values.get(label)
                if ev_in is None or ev_in == 0 or ev_out is None:
                    continue
                if cls == "reduction":
                    metric_values.append((ev_in - ev_out) / abs(ev_in) * 100)
                elif cls == "additive":
                    metric_values.append((ev_out - ev_in) / abs(ev_in) * 100)
                else:
                    metric_values.append((ev_in - ev_out) / abs(ev_in) * 100)
            if not metric_values:
                continue
            if cls == "reduction":
                col_name = f"{label}_reduction_pct"
                direction = "positive = more removed (larger is better, by column name)"
            elif cls == "additive":
                col_name = f"{label}_growth_pct"
                direction = "positive = more added (larger is worse, by column name)"
            else:
                col_name = f"{label}_change_pct"
                direction = "positive = net reduction, negative = net addition"
            add(
                f"- {col_name} range: {_fmt(min(metric_values))} .. {_fmt(max(metric_values))}% "
                f"({direction})"
            )
        best = pareto_solutions[0]
        add("")
        add("Recommended low-input solution:")
        add(f"- total_impurity_input = {_fmt(best.total_impurity_input)}")
        add(f"- total_impurity_output = {_fmt(best.total_impurity_output)}")
        add(f"- inputs = {best.input_values}")
        add(f"- outputs = {best.output_values}")
        best_metrics: dict[str, float] = {}
        for label, _x, _y, cls in optimization_model_columns():
            bc_in: float | None = best.input_values.get(label)
            bc_out: float | None = best.output_values.get(label)
            if bc_in is None or bc_in == 0 or bc_out is None:
                continue
            if cls == "reduction":
                best_metrics[f"{label}_reduction_pct"] = (bc_in - bc_out) / abs(bc_in) * 100
            elif cls == "additive":
                best_metrics[f"{label}_growth_pct"] = (bc_out - bc_in) / abs(bc_in) * 100
            else:
                best_metrics[f"{label}_change_pct"] = (bc_in - bc_out) / abs(bc_in) * 100
        best_metrics_fmt = {k: _fmt(v) for k, v in best_metrics.items()}
        add(f"- per-element metrics = {best_metrics_fmt}")
    add("")
    add("---")
    add("_Formulas and calculation details are also written to `run.log` for the full ")
    add("step-by-step trace of the run._")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved calculations/formulas report: %s", path)
