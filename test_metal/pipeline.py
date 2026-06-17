import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from matplotlib.figure import Figure

from test_metal.config import ProjectConfig
from test_metal.core.models import OLSResult
from test_metal.core.optimization import InverseRegression, ParetoOptimizer
from test_metal.core.regression import fit_ols
from test_metal.features import (
    COLUMN_NAMES,
    PREDICTORS_AFTER,
    PREDICTORS_BEFORE,
    TARGET_AFTER,
    TARGET_BEFORE,
    build_optimization_targets,
)
from test_metal.io.excel import load_excel
from test_metal.io.pdf import create_combined_pdf
from test_metal.io.reports import build_pareto_rows, save_regression_csv
from test_metal.plotting import regression_ci_plot
from test_metal.preprocessing import preprocess


@dataclass
class PipelineResult:
    models: list[OLSResult]
    single_element_report: pd.DataFrame | None = None
    pareto_front: pd.DataFrame | None = None
    figures: dict[str, Figure] = field(default_factory=dict)


def run_pipeline(
    df: pd.DataFrame,
    *,
    config: ProjectConfig | None = None,
    mode: Literal["after", "before"] = "after",
    x_columns: list[str] | None = None,
    y_column: str | None = None,
) -> PipelineResult:
    cfg = config or ProjectConfig()
    if mode == "after":
        default_x = PREDICTORS_AFTER
        default_y = TARGET_AFTER
    else:
        default_x = PREDICTORS_BEFORE
        default_y = TARGET_BEFORE
    xs = x_columns if x_columns else default_x
    y = y_column if y_column else default_y
    missing = [c for c in xs + [y] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")
    dfp = preprocess(df, col_threshold=cfg.missing_threshold)
    models: list[OLSResult] = []
    figures_dict: dict[str, Figure] = {}
    for x in xs:
        try:
            res = fit_ols(dfp, x, y)
            models.append(res)
            fig, _ = regression_ci_plot(
                res.x,
                res.y,
                res.y_hat,
                res.mean_ci_low,
                res.mean_ci_high,
                res.r2,
                f"Linear Regression {y} ~ {x}",
                x,
                y,
                str(cfg.outputs_dir),
                f"{y}_vs_{x}",
                save_png=False,
            )
            figures_dict[f"{y}_vs_{x}"] = fig
        except Exception:
            logging.exception("Error calculating regression %s ~ %s", y, x)
    if not models:
        raise RuntimeError("Failed to build any regression model")

    inverse = InverseRegression(models)
    optimizer = ParetoOptimizer(inverse)
    optimization_targets = build_optimization_targets(models, dfp)

    single_element_report: pd.DataFrame | None = None
    pareto_front: pd.DataFrame | None = None

    if optimization_targets:
        single_rows: list[dict[str, Any]] = []
        for element, (x_col, target) in optimization_targets.items():
            opt_result = inverse.optimize_single_element(element, x_col, target)
            single_rows.append(
                {
                    "element": opt_result.element,
                    "target_output": opt_result.target_output,
                    "required_input": opt_result.required_input,
                    "predicted_output": opt_result.predicted_output,
                    "r2_score": opt_result.r2_score,
                    "is_feasible": opt_result.is_feasible,
                    "confidence": opt_result.confidence,
                    "notes": "; ".join(opt_result.notes) if opt_result.notes else "",
                }
            )
        if single_rows:
            single_element_report = pd.DataFrame(single_rows)
        pareto_solutions = optimizer.generate_pareto_front(optimization_targets, n_points=100)
        pareto_solutions = optimizer.filter_pareto_front(pareto_solutions)
        if pareto_solutions:
            pareto_front = build_pareto_rows(pareto_solutions)

    return PipelineResult(
        models=models,
        single_element_report=single_element_report,
        pareto_front=pareto_front,
        figures=figures_dict,
    )


def run_pipeline_with_io(
    excel_path: Path,
    config: ProjectConfig | None = None,
    **kwargs: Any,
) -> PipelineResult:
    cfg = config or ProjectConfig()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    df = load_excel(excel_path, config=cfg)
    if len(df.columns) > len(COLUMN_NAMES):
        df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES[: len(df.columns)]
    result = run_pipeline(df, config=cfg, **kwargs)
    report_path = cfg.outputs_dir / "regression_report.csv"
    save_regression_csv(result.models, report_path)
    if result.single_element_report is not None:
        single_path = cfg.outputs_dir / "optimization_report_single_element.csv"
        result.single_element_report.to_csv(single_path, index=False)
    if result.pareto_front is not None:
        pareto_path = cfg.outputs_dir / "optimization_report_pareto_front.csv"
        result.pareto_front.to_csv(pareto_path, index=False)
    if result.figures:
        figures_list = list(result.figures.values())
        combined_pdf_path = cfg.outputs_dir / "all_regressions.pdf"
        create_combined_pdf(figures_list, str(combined_pdf_path))
    return result
