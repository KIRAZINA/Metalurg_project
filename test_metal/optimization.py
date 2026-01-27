"""
Module for optimizing input parameters based on regression models.
Determines minimum input impurity values to achieve target output values.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from test_metal.modeling import OLSResult


@dataclass
class OptimizationResult:
    """Result of optimization for a single element."""
    element: str  # Element name (S, Si)
    target_output: float  # Target value at output
    required_input: float  # Required value at input
    predicted_output: float  # Predicted value at output according to model
    r2_score: float  # R² of regression model
    is_feasible: bool  # Is it possible to achieve target value
    confidence: str  # Confidence level (high/medium/low)
    notes: List[str]  # Additional notes


@dataclass
class ParetoOptimum:
    """Point on the Pareto front of optimal solutions."""
    solution_id: int
    input_values: Dict[str, float]  # Required input values
    output_values: Dict[str, float]  # Predicted output values
    total_impurity_input: float  # Total impurity at input
    total_impurity_output: float  # Total impurity at output
    efficiency: float  # Cleaning efficiency (%)


class InverseRegression:
    """Solving the inverse regression problem: find input by output."""
    
    def __init__(self, ols_results: List[OLSResult]):
        """
        Initialize inverse regression.
        
        Args:
            ols_results: List of OLS regression results in form y_after ~ x_before
        """
        self.models = {res.x_col: res for res in ols_results}
        self.logger = logging.getLogger(__name__)
    
    def predict_required_input(self, x_col: str, target_output: float) -> Tuple[float, bool, str]:
        """
        Determine required input value to achieve target output value.
        
        Args:
            x_col: Name of input variable
            target_output: Target value for output variable
            
        Returns:
            (required_input, is_feasible, confidence_level)
        """
        if x_col not in self.models:
            raise ValueError(f"Model for {x_col} not found")
        
        model = self.models[x_col]
        
        # Inverse function: x = (y - intercept) / slope
        if abs(model.slope) < 1e-10:
            return np.nan, False, "no_correlation"
        
        required_input = (target_output - model.intercept) / model.slope
        
        # Check feasibility
        x_min, x_max = model.x.min(), model.x.max()
        is_within_range = x_min <= required_input <= x_max
        
        # Confidence level depends on R²
        if model.r2 > 0.8:
            confidence = "high"
        elif model.r2 > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        if not is_within_range:
            self.logger.warning(
                "Required input %s=%.6f is outside observed range [%.6f, %.6f]",
                x_col, required_input, x_min, x_max
            )
        
        is_feasible = is_within_range and model.r2 > 0.3
        
        return required_input, is_feasible, confidence
    
    def optimize_single_element(
        self, 
        element: str, 
        x_col: str, 
        target_output: float
    ) -> OptimizationResult:
        """
        Optimize for a single element.
        
        Args:
            element: Element name (S, Si, etc.)
            x_col: Name of input variable
            target_output: Target value at output
            
        Returns:
            OptimizationResult with recommendations
        """
        model = self.models[x_col]
        required_input, is_feasible, confidence = self.predict_required_input(x_col, target_output)
        
        # Predict output value according to model
        predicted_output = model.intercept + model.slope * required_input
        
        notes = []
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
            notes=notes
        )


class ParetoOptimizer:
    """Building Pareto front of optimal input parameter combinations."""
    
    def __init__(self, inverse_regressor: InverseRegression):
        """
        Initialize.
        
        Args:
            inverse_regressor: InverseRegression object with models
        """
        self.inverse = inverse_regressor
        self.logger = logging.getLogger(__name__)
    
    def generate_pareto_front(
        self,
        element_targets: Dict[str, Tuple[str, float]],  # {element_name: (x_col, target_output)}
        n_points: int = 50
    ) -> List[ParetoOptimum]:
        """
        Build Pareto front of optimal solutions for multiple elements.
        
        Args:
            element_targets: Dictionary of target values for elements
            n_points: Number of points for varying (linear space)
            
        Returns:
            List of ParetoOptimum points on the front
        """
        pareto_solutions = []
        
        # For each element get range of possible target values
        element_ranges = {}
        for element, (x_col, base_target) in element_targets.items():
            model = self.inverse.models[x_col]
            y_min, y_max = model.y.min(), model.y.max()
            element_ranges[element] = (y_min, y_max)
        
        # Generate combinations of target values according to Pareto principle
        # Vary target values in their ranges
        elements = list(element_targets.keys())
        
        if not elements:
            return []
        
        # Create grid of target values
        for i in range(n_points):
            ratio = i / max(1, n_points - 1)
            
            input_values = {}
            output_values = {}
            total_input = 0
            total_output = 0
            
            for element in elements:
                x_col, base_target = element_targets[element]
                y_min, y_max = element_ranges[element]
                
                # Interpolate target value from minimum to base target
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
                efficiency = ((total_input - total_output) / total_input * 100) if total_input > 0 else 0
                
                pareto_solutions.append(ParetoOptimum(
                    solution_id=i,
                    input_values=input_values,
                    output_values=output_values,
                    total_impurity_input=total_input,
                    total_impurity_output=total_output,
                    efficiency=efficiency
                ))
        
        return pareto_solutions
    
    @staticmethod
    def filter_pareto_front(solutions: List[ParetoOptimum]) -> List[ParetoOptimum]:
        """
        Keep only Pareto-optimal solutions (no domination).
        Solution A dominates B if it is better in all criteria.
        """
        if not solutions:
            return []
        
        pareto = []
        for candidate in solutions:
            is_dominated = False
            
            for other in solutions:
                if other is candidate:
                    continue
                
                # other dominates candidate if:
                # 1. other has fewer input impurities
                # 2. other has fewer output impurities
                if (other.total_impurity_input <= candidate.total_impurity_input and
                    other.total_impurity_output <= candidate.total_impurity_output and
                    (other.total_impurity_input < candidate.total_impurity_input or
                     other.total_impurity_output < candidate.total_impurity_output)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto.append(candidate)
        
        return sorted(pareto, key=lambda x: x.total_impurity_input)


def generate_optimization_report(
    inverse_regressor: InverseRegression,
    element_targets: Dict[str, Tuple[str, float]],
    output_path: str
) -> pd.DataFrame:
    """
    Generate optimization report.
    
    Args:
        inverse_regressor: InverseRegression object with models
        element_targets: Target values for elements
        output_path: Path to save report
        
    Returns:
        DataFrame with optimization results
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating optimization report")
    
    # Single optimization for each element
    results = []
    for element, (x_col, target) in element_targets.items():
        opt_result = inverse_regressor.optimize_single_element(element, x_col, target)
        results.append({
            "element": opt_result.element,
            "target_output": opt_result.target_output,
            "required_input": opt_result.required_input,
            "predicted_output": opt_result.predicted_output,
            "r2_score": opt_result.r2_score,
            "is_feasible": opt_result.is_feasible,
            "confidence": opt_result.confidence,
            "notes": "; ".join(opt_result.notes) if opt_result.notes else ""
        })
    
    # Generate Pareto front
    optimizer = ParetoOptimizer(inverse_regressor)
    pareto_solutions = optimizer.generate_pareto_front(element_targets, n_points=50)
    pareto_solutions = optimizer.filter_pareto_front(pareto_solutions)
    
    logger.info(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
    
    # Save results
    if results:
        df_single = pd.DataFrame(results)
        df_single.to_csv(output_path.replace(".csv", "_single_element.csv"), index=False)
        logger.info(f"Single element optimization saved: {output_path.replace('.csv', '_single_element.csv')}")
    
    if pareto_solutions:
        pareto_rows = []
        for sol in pareto_solutions:
            row = {"solution_id": sol.solution_id}
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
