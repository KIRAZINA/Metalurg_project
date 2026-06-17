import logging
from pathlib import Path
from typing import Any

import pandas as pd

from test_metal.core.models import OLSResult, OptimizationResult, ParetoOptimum
from test_metal.core.optimization import InverseRegression, ParetoOptimizer


def generate_optimization_report(
    inverse_regressor: InverseRegression,
    element_targets: dict[str, tuple[str, float]],
    output_path: str,
    optimizer: ParetoOptimizer | None = None,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("Generating optimization report")
    results = []
    for element, (x_col, target) in element_targets.items():
        opt_result = inverse_regressor.optimize_single_element(element, x_col, target)
        results.append(
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
    if optimizer is None:
        optimizer = ParetoOptimizer(inverse_regressor)
    pareto_solutions = optimizer.generate_pareto_front(element_targets, n_points=50)
    pareto_solutions = optimizer.filter_pareto_front(pareto_solutions)
    logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
    if results:
        df_single = pd.DataFrame(results)
        df_single.to_csv(output_path.replace(".csv", "_single_element.csv"), index=False)
        logger.info(
            f"Single element optimization saved: {output_path.replace('.csv', '_single_element.csv')}"
        )
    if pareto_solutions:
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
        df_pareto = pd.DataFrame(pareto_rows)
        df_pareto.to_csv(output_path.replace(".csv", "_pareto_front.csv"), index=False)
        logger.info(f"Pareto front saved: {output_path.replace('.csv', '_pareto_front.csv')}")
    return pd.DataFrame(results) if results else pd.DataFrame()


def run_full_optimization(
    ols_results: list[OLSResult],
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting input parameter optimization")
    from test_metal.features import (
        build_optimization_targets,
    )

    inverse_regressor = InverseRegression(ols_results)
    optimization_targets = build_optimization_targets(ols_results, df)
    if not optimization_targets:
        logger.info("Optimization not applied: target parameters (S, Si) not found")
        return
    opt_report_path = Path(output_dir) / "optimization_report.csv"
    single_results = []
    for element, (x_col, target) in optimization_targets.items():
        opt_result = inverse_regressor.optimize_single_element(element, x_col, target)
        single_results.append(
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
    if single_results:
        df_single = pd.DataFrame(single_results)
        single_path = str(opt_report_path).replace(".csv", "_single_element.csv")
        df_single.to_csv(single_path, index=False)
        logger.info("Single element optimization saved: %s", single_path)
    optimizer = ParetoOptimizer(inverse_regressor)
    pareto_solutions = optimizer.generate_pareto_front(optimization_targets, n_points=100)
    pareto_solutions = optimizer.filter_pareto_front(pareto_solutions)
    logger.info("Found %d Pareto-optimal solutions", len(pareto_solutions))
    if pareto_solutions:
        pareto_rows = []
        for sol in pareto_solutions:
            row: dict[str, object] = {"solution_id": sol.solution_id}
            for elem, val in sol.input_values.items():
                row[f"{elem}_input"] = val
            for elem, val in sol.output_values.items():
                row[f"{elem}_output"] = val
            row["total_impurity_input"] = sol.total_impurity_input
            row["total_impurity_output"] = sol.total_impurity_output
            row["efficiency_%"] = sol.efficiency
            pareto_rows.append(row)
        df_pareto = pd.DataFrame(pareto_rows)
        pareto_path = str(opt_report_path).replace(".csv", "_pareto_front.csv")
        df_pareto.to_csv(pareto_path, index=False)
        logger.info("Pareto front saved: %s", pareto_path)
        logger.info("Best solution (minimum impurities at input):")
        best = pareto_solutions[0]
        for elem, val in best.input_values.items():
            logger.info("  %s (input): %.6f", elem, val)
        for elem, val in best.output_values.items():
            logger.info("  %s (output): %.6f", elem, val)
        logger.info("  Total impurity at input: %.6f", best.total_impurity_input)
        logger.info("  Total impurity at output: %.6f", best.total_impurity_output)
        logger.info("  Purification efficiency: %.2f%%", best.efficiency)


__all__ = [
    "OptimizationResult",
    "ParetoOptimum",
    "InverseRegression",
    "ParetoOptimizer",
    "generate_optimization_report",
    "run_full_optimization",
]
