import logging

import numpy as np

from test_metal.core.models import OLSResult, OptimizationResult, ParetoOptimum


class InverseRegression:
    def __init__(self, ols_results: list[OLSResult]):
        self.models = {res.x_col: res for res in ols_results}
        self.logger = logging.getLogger(__name__)

    def predict_required_input(self, x_col: str, target_output: float) -> tuple[float, bool, str]:
        if x_col not in self.models:
            raise ValueError(f"Model for {x_col} not found")
        model = self.models[x_col]
        if abs(model.slope) < 1e-10:
            return np.nan, False, "no_correlation"
        required_input = (target_output - model.intercept) / model.slope
        x_min, x_max = model.x.min(), model.x.max()
        is_within_range = bool(x_min <= required_input <= x_max)
        if model.r2 > 0.8:
            confidence = "high"
        elif model.r2 > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        if not is_within_range:
            self.logger.warning(
                "Required input %s=%.6f is outside observed range [%.6f, %.6f]",
                x_col,
                required_input,
                x_min,
                x_max,
            )
        is_feasible = is_within_range and model.r2 > 0.3
        return required_input, is_feasible, confidence

    def optimize_single_element(
        self, element: str, x_col: str, target_output: float
    ) -> OptimizationResult:
        model = self.models[x_col]
        required_input, is_feasible, confidence = self.predict_required_input(x_col, target_output)
        predicted_output = model.intercept + model.slope * required_input
        notes: list[str] = []
        if required_input < 0:
            notes.append("Required value is negative (impossible in reality)")
            is_feasible = False
        if not np.isfinite(required_input):
            notes.append("Cannot find solution - linearity violated")
            is_feasible = False
        x_min, x_max = model.x.min(), model.x.max()
        if required_input < x_min:
            notes.append(f"Value below minimum from data ({x_min:.6f})")
        elif required_input > x_max:
            notes.append(f"Value above maximum from data ({x_max:.6f})")
        return OptimizationResult(
            element=element,
            target_output=target_output,
            required_input=required_input if np.isfinite(required_input) else np.nan,
            predicted_output=predicted_output if np.isfinite(predicted_output) else np.nan,
            r2_score=model.r2,
            is_feasible=is_feasible,
            confidence=confidence,
            notes=notes,
        )


class ParetoOptimizer:
    def __init__(self, inverse_regressor: InverseRegression):
        self.inverse = inverse_regressor
        self.logger = logging.getLogger(__name__)

    def generate_pareto_front(
        self,
        element_targets: dict[str, tuple[str, float]],
        n_points: int = 50,
    ) -> list[ParetoOptimum]:
        pareto_solutions: list[ParetoOptimum] = []
        element_ranges: dict[str, tuple[float, float]] = {}
        for element, (x_col, _base_target) in element_targets.items():
            model = self.inverse.models[x_col]
            y_min = float(model.y.min())
            y_max = float(model.y.max())
            element_ranges[element] = (y_min, y_max)
        elements = list(element_targets.keys())
        if not elements:
            return []
        for i in range(n_points):
            ratio = i / max(1, n_points - 1)
            input_values: dict[str, float] = {}
            output_values: dict[str, float] = {}
            total_input = 0.0
            total_output = 0.0
            for element in elements:
                x_col, base_target = element_targets[element]
                y_min, y_max = element_ranges[element]
                target = y_min + ratio * (base_target - y_min)
                try:
                    required_input, _, _ = self.inverse.predict_required_input(x_col, target)
                    if np.isfinite(required_input) and required_input >= 0:
                        input_values[element] = required_input
                        output_values[element] = target
                        total_input += required_input
                        total_output += target
                except Exception as e:
                    self.logger.warning(f"Error optimizing {element}: {e}")
                    continue
            if input_values:
                efficiency = (
                    ((total_input - total_output) / total_input * 100) if total_input > 0 else 0.0
                )
                pareto_solutions.append(
                    ParetoOptimum(
                        solution_id=i,
                        input_values=input_values,
                        output_values=output_values,
                        total_impurity_input=total_input,
                        total_impurity_output=total_output,
                        efficiency=efficiency,
                    )
                )
        return pareto_solutions

    @staticmethod
    def filter_pareto_front(solutions: list[ParetoOptimum]) -> list[ParetoOptimum]:
        if not solutions:
            return []
        pareto: list[ParetoOptimum] = []
        for candidate in solutions:
            is_dominated = False
            for other in solutions:
                if other is candidate:
                    continue
                if (
                    other.total_impurity_input <= candidate.total_impurity_input
                    and other.total_impurity_output <= candidate.total_impurity_output
                    and (
                        other.total_impurity_input < candidate.total_impurity_input
                        or other.total_impurity_output < candidate.total_impurity_output
                    )
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(candidate)
        return sorted(pareto, key=lambda x: x.total_impurity_input)
