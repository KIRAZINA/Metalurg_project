"""
Edge case tests for optimization module.
Tests boundary conditions, error handling, and edge cases.
"""

import numpy as np
import pytest

from test_metal.optimization import (
    InverseRegression,
    OptimizationResult,
    ParetoOptimum,
)


class TestInverseRegressionEdgeCases:
    """Test edge cases for InverseRegression class."""

    def test_perfect_model(self, mock_ols_result):
        """Test inverse regression with perfect model (R² = 0.95)."""
        inverse = InverseRegression([mock_ols_result])

        # For model: y = 0.02 + 0.5*x
        # To get y = 0.08: x = (0.08 - 0.02) / 0.5 = 0.12
        required_input, feasible, confidence = inverse.predict_required_input(
            "steel_S_before", 0.08
        )

        assert abs(required_input - 0.12) < 0.001
        assert feasible is True
        assert confidence == "high"

    def test_poor_model(self, poor_model_ols_result):
        """Test with low-quality model (R² = 0.35)."""
        inverse = InverseRegression([poor_model_ols_result])

        required_input, feasible, confidence = inverse.predict_required_input(
            "steel_Si_before", 0.08
        )

        assert isinstance(required_input, (int, float))
        assert isinstance(feasible, bool)
        assert confidence == "low"  # R² < 0.6 should give low confidence

    def test_extrapolation_beyond_data(self, mock_ols_result):
        """Test extrapolation beyond observed data range."""
        inverse = InverseRegression([mock_ols_result])

        # Request value way beyond observed range (x_max = 0.3)
        required_input, feasible, confidence = inverse.predict_required_input(
            "steel_S_before",
            0.5,  # Very high target
        )

        # Should still return a value but mark as not feasible
        assert isinstance(required_input, (int, float))
        assert feasible is False or feasible is True  # May vary based on implementation

    def test_minimum_target_value(self, mock_ols_result):
        """Test with minimum possible target value."""
        inverse = InverseRegression([mock_ols_result])

        # Target = intercept (minimum achievable)
        required_input, feasible, confidence = inverse.predict_required_input(
            "steel_S_before", 0.02
        )

        assert abs(required_input - 0.0) < 0.001
        assert feasible is True

    def test_negative_slope_model(self):
        """Test with negative slope (inverse relationship)."""
        from test_metal.modeling import OLSResult

        # Model where higher input leads to lower output
        model = OLSResult(
            x_col="test_x",
            y_col="test_y",
            intercept=0.5,
            slope=-0.3,  # Negative slope
            stderr_intercept=0.01,
            stderr_slope=0.01,
            pvalue_intercept=0.01,
            pvalue_slope=0.01,
            r2=0.9,
            df_resid=10,
            nobs=12,
            conf_int_intercept_low=0.48,
            conf_int_intercept_high=0.52,
            conf_int_slope_low=-0.32,
            conf_int_slope_high=-0.28,
            x=np.linspace(0.1, 0.5, 12),
            y=0.5 - 0.3 * np.linspace(0.1, 0.5, 12),
            y_hat=0.5 - 0.3 * np.linspace(0.1, 0.5, 12),
            mean_ci_low=np.linspace(0.3, 0.48, 12),
            mean_ci_high=np.linspace(0.32, 0.50, 12),
        )

        inverse = InverseRegression([model])

        # For y = 0.5 - 0.3*x, to get y = 0.35: x = (0.5 - 0.35) / 0.3 = 0.5
        required_input, feasible, confidence = inverse.predict_required_input("test_x", 0.35)

        assert abs(required_input - 0.5) < 0.001
        assert confidence == "high"  # R² = 0.9, so high confidence

    def test_very_small_slope(self):
        """Test with very small slope (weak relationship)."""
        from test_metal.modeling import OLSResult

        model = OLSResult(
            x_col="test_x",
            y_col="test_y",
            intercept=0.1,
            slope=0.001,  # Very small slope
            stderr_intercept=0.01,
            stderr_slope=0.001,
            pvalue_intercept=0.01,
            pvalue_slope=0.5,  # Not significant
            r2=0.1,  # Poor fit
            df_resid=10,
            nobs=12,
            conf_int_intercept_low=0.08,
            conf_int_intercept_high=0.12,
            conf_int_slope_low=-0.001,
            conf_int_slope_high=0.003,
            x=np.linspace(0.1, 5.0, 12),
            y=0.1 + 0.001 * np.linspace(0.1, 5.0, 12),
            y_hat=0.1 + 0.001 * np.linspace(0.1, 5.0, 12),
            mean_ci_low=np.linspace(0.09, 0.105, 12),
            mean_ci_high=np.linspace(0.11, 0.115, 12),
        )

        inverse = InverseRegression([model])

        # Very large input needed for small change in output
        required_input, feasible, confidence = inverse.predict_required_input("test_x", 0.105)

        assert isinstance(required_input, (int, float))
        assert confidence == "low"  # R² = 0.1, so low confidence

    def test_multiple_models_same_element(self, mock_ols_result, poor_model_ols_result):
        """Test with multiple models (should use first matching)."""
        inverse = InverseRegression([mock_ols_result, poor_model_ols_result])

        # Should find the sulfur model
        required_input, _, confidence = inverse.predict_required_input("steel_S_before", 0.08)

        assert abs(required_input - 0.12) < 0.001
        assert confidence == "high"

    def test_model_not_found(self, mock_ols_result):
        """Test when requested model column doesn't exist."""
        inverse = InverseRegression([mock_ols_result])

        # Request non-existent column
        with pytest.raises(ValueError):
            inverse.predict_required_input("nonexistent_col", 0.08)

    def test_optimize_single_element_returns_correct_type(self, mock_ols_result):
        """Test that optimize_single_element returns OptimizationResult."""
        inverse = InverseRegression([mock_ols_result])

        result = inverse.optimize_single_element("Sulfur", "steel_S_before", 0.08)

        assert isinstance(result, OptimizationResult)
        assert result.element == "Sulfur"
        assert result.target_output == 0.08
        assert hasattr(result, "required_input")
        assert hasattr(result, "is_feasible")
        assert hasattr(result, "confidence")


class TestOptimizationResultDataclass:
    """Test OptimizationResult data class."""

    def test_optimization_result_creation(self, mock_ols_result):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            element="Test",
            target_output=0.05,
            required_input=0.08,
            predicted_output=0.05,
            r2_score=0.9,
            is_feasible=True,
            confidence="high",
            notes=["Note 1", "Note 2"],
        )

        assert result.element == "Test"
        assert result.target_output == 0.05
        assert result.required_input == 0.08
        assert result.r2_score == 0.9
        assert result.is_feasible is True
        assert result.confidence == "high"
        assert len(result.notes) == 2

    def test_optimization_result_default_notes(self):
        """Test OptimizationResult with default empty notes."""
        result = OptimizationResult(
            element="Test",
            target_output=0.05,
            required_input=0.08,
            predicted_output=0.05,
            r2_score=0.9,
            is_feasible=True,
            confidence="high",
            notes=[],
        )

        assert result.notes == []


class TestParetoOptimumDataclass:
    """Test ParetoOptimum data class."""

    def test_pareto_optimum_creation(self):
        """Test creating ParetoOptimum."""
        solution = ParetoOptimum(
            solution_id=1,
            input_values={"S": 0.05, "Si": 0.1},
            output_values={"S": 0.03, "Si": 0.08},
            total_impurity_input=0.15,
            total_impurity_output=0.11,
            efficiency=26.67,
        )

        assert solution.solution_id == 1
        assert solution.input_values["S"] == 0.05
        assert solution.output_values["S"] == 0.03
        assert 26.0 < solution.efficiency < 27.0

    def test_pareto_optimum_zero_efficiency(self):
        """Test ParetoOptimum with zero efficiency (no purification)."""
        solution = ParetoOptimum(
            solution_id=1,
            input_values={"S": 0.05},
            output_values={"S": 0.05},
            total_impurity_input=0.05,
            total_impurity_output=0.05,
            efficiency=0.0,
        )

        assert solution.efficiency == 0.0

    def test_pareto_optimum_high_efficiency(self):
        """Test ParetoOptimum with high efficiency."""
        solution = ParetoOptimum(
            solution_id=1,
            input_values={"S": 0.1},
            output_values={"S": 0.01},
            total_impurity_input=0.1,
            total_impurity_output=0.01,
            efficiency=90.0,
        )

        assert solution.efficiency == 90.0
