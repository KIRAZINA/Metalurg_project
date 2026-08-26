"""End-to-end analysis pipeline: regression suite plus Pareto optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from test_metal.config import ProjectConfig
from test_metal.core.optimization import InverseRegression, ParetoOptimizer
from test_metal.core.regression import fit_ols
from test_metal.features import (
    COLUMN_NAMES,
    PREDICTORS_AFTER,
    PREDICTORS_BEFORE,
    TARGET_AFTER,
    TARGET_BEFORE,
    build_optimization_targets,
    optimization_model_columns,
)
from test_metal.io.excel import load_excel
from test_metal.io.pdf import create_combined_pdf
from test_metal.io.reports import (
    build_pareto_rows,
    save_optimization_csv,
    save_regression_csv,
    write_calculations_report,
)
from test_metal.plotting import (
    heatmap_corr,
    plot_pareto_front,
    regression_ci_plot,
)
from test_metal.preprocessing import preprocess

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from test_metal.core.models import OLSResult, ParetoOptimum

logger = logging.getLogger(__name__)


def _fit_inverse_models(
    dfp: pd.DataFrame,
    models: list[OLSResult],
    figures: dict[str, Figure],
    config: ProjectConfig,
) -> None:
    """Ensure the before→after S/Si models used by inverse optimization exist."""
    existing = {(m.x_col, m.y_col) for m in models}
    for _label, x_col, y_col, _cls in optimization_model_columns():
        if (x_col, y_col) in existing or x_col not in dfp or y_col not in dfp:
            continue
        try:
            res = fit_ols(dfp, x_col, y_col)
            models.append(res)
            logger.info(
                "Inverse model fitted: %s ~ %s | R2=%.4f slope=%.6g p=%.3g n=%d",
                y_col,
                x_col,
                res.r2,
                res.slope,
                res.pvalue_slope,
                res.nobs,
            )
            fig, _ = regression_ci_plot(
                res.x,
                res.y,
                res.y_hat,
                res.mean_ci_low,
                res.mean_ci_high,
                res.r2,
                f"Inverse model {y_col} ~ {x_col}",
                x_col,
                y_col,
                str(config.outputs_dir),
                f"{y_col}_vs_{x_col}",
                save_png=False,
            )
            figures[f"{y_col}_vs_{x_col}"] = fig
        except Exception:
            logger.exception("Inverse model %s ~ %s failed -- skipped", y_col, x_col)


@dataclass
class PipelineResult:
    models: list[OLSResult]
    single_element_report: pd.DataFrame | None = None
    pareto_front: pd.DataFrame | None = None
    pareto_solutions: list[ParetoOptimum] = field(default_factory=list)
    pareto_candidates: pd.DataFrame | None = None
    preprocessed: pd.DataFrame | None = None
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

    logger.info(
        "Pipeline start: mode=%s predictors=%d (%s...) target=%s rows=%d",
        mode,
        len(xs),
        xs[0],
        y,
        len(df),
    )
    logger.info(
        "Formulas: y = beta0 + beta1*x | inverse x=(y_target-beta0)/beta1 | "
        "Pareto efficiency=(sum_input-sum_output)/sum_input*100"
    )
    dfp = preprocess(df, col_threshold=cfg.missing_threshold)

    models: list[OLSResult] = []
    figures: dict[str, Figure] = {}
    for i, x in enumerate(xs, start=1):
        try:
            res = fit_ols(dfp, x, y)
            models.append(res)
            logger.info(
                "Model %d/%d fitted: %s ~ %s | R2=%.4f slope=%.6g p=%.3g n=%d",
                i,
                len(xs),
                y,
                x,
                res.r2,
                res.slope,
                res.pvalue_slope,
                res.nobs,
            )
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
            figures[f"{y}_vs_{x}"] = fig
        except Exception:
            logger.exception("Error calculating regression %s ~ %s -- skipped", y, x)

    if not models:
        raise RuntimeError("Failed to build any regression model")
    logger.info("Fitted %d/%d regression models successfully", len(models), len(xs))

    # In default mode (no explicit predictors/target) also fit the S/Si
    # before->after models so inverse optimization can run end-to-end.
    if x_columns is None and y_column is None:
        _fit_inverse_models(dfp, models, figures, cfg)

    inverse = InverseRegression(
        models,
        r2_high=cfg.r2_high,
        r2_medium=cfg.r2_medium,
        r2_min_feasible=cfg.r2_min_feasible,
        slope_min_abs=cfg.slope_min_abs,
    )
    optimizer = ParetoOptimizer(inverse)
    optimization_targets = build_optimization_targets(models, dfp)
    if optimization_targets:
        logger.info(
            "Optimization targets (observed minima): %s",
            {
                elem: {"column": col, "target": tgt}
                for elem, (col, tgt) in optimization_targets.items()
            },
        )
    else:
        logger.info("No S/Si before-after model pairs found; optimization skipped")

    single_element_report: pd.DataFrame | None = None
    pareto_front: pd.DataFrame | None = None
    pareto_solutions: list[ParetoOptimum] = []
    filtered: list[ParetoOptimum] = []

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
        filtered = optimizer.filter_pareto_front(pareto_solutions)
        logger.info(
            "Pareto front: %d candidates -> %d non-dominated solutions",
            len(pareto_solutions),
            len(filtered),
        )
        pareto_front = build_pareto_rows(filtered)
        pareto_candidates = build_pareto_rows(pareto_solutions)

    return PipelineResult(
        models=models,
        single_element_report=single_element_report,
        pareto_front=pareto_front,
        pareto_solutions=filtered,
        pareto_candidates=pareto_candidates,
        preprocessed=dfp,
        figures=figures,
    )


def run_pipeline_with_io(
    excel_path: Path,
    config: ProjectConfig | None = None,
    **kwargs: Any,
) -> PipelineResult:
    cfg = config or ProjectConfig()
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    logger.info(
        "Loading Excel file: %s",
        excel_path.resolve() if excel_path.exists() else excel_path,
    )
    df = load_excel(excel_path, config=cfg)
    if len(df.columns) > len(COLUMN_NAMES):
        logger.info(
            "Truncating columns %d -> %d to match canonical schema",
            len(df.columns),
            len(COLUMN_NAMES),
        )
        df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES[: len(df.columns)]
    logger.info("Loaded frame: %d rows x %d columns", len(df), len(df.columns))

    result = run_pipeline(df, config=cfg, **kwargs)
    dfp = result.preprocessed if result.preprocessed is not None else df
    dropped_columns = [c for c in df.columns if c not in dfp.columns]

    report_path = cfg.outputs_dir / "regression_report.csv"
    save_regression_csv(result.models, report_path)
    artifacts.append(report_path)
    logger.info("Saved regression report (%d models): %s", len(result.models), report_path)

    if result.single_element_report is not None:
        single_path = cfg.outputs_dir / "optimization_report_single_element.csv"
        save_optimization_csv(result.single_element_report, single_path)
        artifacts.append(single_path)
        logger.info(
            "Saved single-element optimization report (%d rows): %s",
            len(result.single_element_report),
            single_path,
        )

    if result.pareto_front is not None:
        pareto_path = cfg.outputs_dir / "optimization_report_pareto_front.csv"
        save_optimization_csv(result.pareto_front, pareto_path)
        artifacts.append(pareto_path)
        logger.info(
            "Saved Pareto front report (%d solutions): %s",
            len(result.pareto_front),
            pareto_path,
        )

    if result.pareto_candidates is not None:
        candidates_path = cfg.outputs_dir / "pareto_candidates_full_sweep.csv"
        save_optimization_csv(result.pareto_candidates, candidates_path)
        artifacts.append(candidates_path)
        logger.info(
            "Saved full Pareto candidate sweep (%d solutions): %s",
            len(result.pareto_candidates),
            candidates_path,
        )

    if result.figures:
        figures_list = list(result.figures.values())
        combined_pdf_path = cfg.outputs_dir / "all_regressions.pdf"
        create_combined_pdf(figures_list, str(combined_pdf_path))
        artifacts.append(combined_pdf_path)
        logger.info(
            "Saved combined PDF with %d regression plots: %s",
            len(figures_list),
            combined_pdf_path,
        )

    if result.pareto_front is not None:
        pareto_plot_path = plot_pareto_front(
            result.pareto_solutions, str(cfg.outputs_dir), "pareto_front"
        )
        if pareto_plot_path is not None:
            artifacts.append(pareto_plot_path)
            logger.info(
                "Saved Pareto front plot (%d solutions): %s",
                len(result.pareto_solutions),
                pareto_plot_path,
            )

    corr_cols = [
        c
        for c in (
            "steel_S_before",
            "steel_S_after",
            "steel_Si_before",
            "steel_Si_after",
            "sulfur_reduction_ratio",
            TARGET_AFTER,
        )
        if c in dfp.columns
    ]
    if corr_cols:
        heatmap_paths = heatmap_corr(
            dfp,
            corr_cols,
            "Feature correlation (S/Si focus)",
            str(cfg.outputs_dir),
            "correlation_heatmap",
        )
        artifacts.append(heatmap_paths["png"])
        if "pdf" in heatmap_paths:
            artifacts.append(heatmap_paths["pdf"])
        logger.info("Saved correlation heatmap: %s", heatmap_paths["png"])

    calc_path = cfg.outputs_dir / "calculations.md"
    write_calculations_report(
        calc_path,
        source_file=str(excel_path),
        mode=kwargs.get("mode", "after"),
        config=cfg,
        rows_raw=len(df),
        cols_raw=len(df.columns),
        rows_clean=len(dfp),
        cols_clean=len(dfp.columns),
        dropped_columns=dropped_columns,
        models=result.models,
        single_element_report=result.single_element_report,
        pareto_front=result.pareto_front,
        pareto_solutions=result.pareto_solutions,
    )
    artifacts.append(calc_path)

    logger.info("Run complete. Artifacts written to %s:", cfg.outputs_dir.resolve())
    for path in artifacts:
        logger.info("  - %s", path.name)
    return result
