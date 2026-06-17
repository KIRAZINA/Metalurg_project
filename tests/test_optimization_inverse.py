import numpy as np
import pandas as pd
import pytest

from test_metal.core.models import OLSResult, OptimizationResult
from test_metal.core.optimization import InverseRegression


def _make_model(
    x_col: str = "steel_S_before",
    y_col: str = "steel_S_after",
    intercept: float = 0.02,
    slope: float = 0.5,
    r2: float = 0.95,
    n: int = 12,
    x_start: float = 0.1,
    x_end: float = 0.3,
):
    x_data = np.linspace(x_start, x_end, n)
    y_data = intercept + slope * x_data
    return OLSResult(
        x_col=x_col,
        y_col=y_col,
        intercept=intercept,
        slope=slope,
        stderr_intercept=0.001,
        stderr_slope=0.01,
        pvalue_intercept=0.05,
        pvalue_slope=0.001,
        r2=r2,
        df_resid=float(n - 2),
        nobs=n,
        conf_int_intercept_low=intercept - 0.002,
        conf_int_intercept_high=intercept + 0.002,
        conf_int_slope_low=slope - 0.01,
        conf_int_slope_high=slope + 0.01,
        x=pd.Series(x_data),
        y=pd.Series(y_data),
        y_hat=pd.Series(y_data),
        mean_ci_low=pd.Series(y_data - 0.01),
        mean_ci_high=pd.Series(y_data + 0.01),
    )


class TestInverseRegressionPredict:
    def test_perfect_model_correct_input(self):
        inverse = InverseRegression([_make_model()])
        required, feasible, confidence = inverse.predict_required_input("steel_S_before", 0.08)
        assert abs(required - 0.12) < 0.001
        assert feasible is True
        assert confidence == "high"

    def test_poor_model_low_confidence(self):
        model = _make_model(r2=0.35)
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("steel_S_before", 0.08)
        assert confidence == "low"

    def test_medium_model_medium_confidence(self):
        model = _make_model(r2=0.7)
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("steel_S_before", 0.08)
        assert confidence == "medium"

    def test_extrapolation_beyond_data_not_feasible(self):
        model = _make_model()
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("steel_S_before", 0.5)
        assert feasible is False

    def test_minimum_target(self):
        model = _make_model(
            x_col="steel_S_before", intercept=0.0, slope=1.0, n=5, x_start=0.0, x_end=0.3
        )
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("steel_S_before", 0.0)
        assert feasible is True

    def test_negative_slope_model(self):
        model = _make_model(x_col="test_x", intercept=0.5, slope=-0.3, r2=0.9)
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("test_x", 0.35)
        assert abs(required - 0.5) < 0.001
        assert confidence == "high"

    def test_very_small_slope(self):
        model = _make_model(x_col="test_x", intercept=0.1, slope=0.001, r2=0.1)
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("test_x", 0.105)
        assert isinstance(required, (int, float))
        assert confidence == "low"

    def test_model_not_found_raises(self):
        inverse = InverseRegression([_make_model()])
        with pytest.raises(ValueError):
            inverse.predict_required_input("nonexistent", 0.08)

    def test_near_zero_slope_returns_nan(self):
        model = _make_model(slope=1e-12, r2=0.01)
        inverse = InverseRegression([model])
        required, feasible, confidence = inverse.predict_required_input("steel_S_before", 0.08)
        assert np.isnan(required)
        assert feasible is False
        assert confidence == "no_correlation"


class TestInverseRegressionOptimize:
    def test_optimize_single_element_returns_correct_type(self):
        inverse = InverseRegression([_make_model()])
        result = inverse.optimize_single_element("Sulfur", "steel_S_before", 0.08)
        assert isinstance(result, OptimizationResult)
        assert result.element == "Sulfur"
        assert result.target_output == 0.08

    def test_optimize_negative_input_adds_note(self):
        model = _make_model(intercept=0.5, slope=1.0, r2=0.9)
        inverse = InverseRegression([model])
        result = inverse.optimize_single_element("Sulfur", "steel_S_before", 0.0)
        if result.required_input < 0:
            assert any("negative" in n.lower() for n in result.notes)

    def test_optimize_unfeasible_due_to_r2(self):
        model = _make_model(r2=0.2)
        inverse = InverseRegression([model])
        result = inverse.optimize_single_element("Sulfur", "steel_S_before", 0.08)
        assert result.is_feasible is False

    def test_optimize_preserves_model_fields(self):
        inverse = InverseRegression([_make_model()])
        result = inverse.optimize_single_element("Sulfur", "steel_S_before", 0.08)
        assert result.r2_score == 0.95
        assert isinstance(result.notes, list)


class TestInverseRegressionMultiModel:
    def test_multiple_models_uses_correct_one(self):
        s_model = _make_model(x_col="steel_S_before", slope=0.5)
        si_model = _make_model(x_col="steel_Si_before", slope=0.4)
        inverse = InverseRegression([s_model, si_model])
        required, _, _ = inverse.predict_required_input("steel_S_before", 0.08)
        assert abs(required - 0.12) < 0.001
        required_si, _, _ = inverse.predict_required_input("steel_Si_before", 0.08)
        assert abs(required_si - (0.08 - 0.02) / 0.4) < 0.001
