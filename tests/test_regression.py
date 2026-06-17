import numpy as np
import pandas as pd
import pytest

from test_metal.core.models import OLSResult
from test_metal.core.regression import fit_ols


def _perfect_data(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, n)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.05, n)
    return pd.DataFrame({"x": x, "y": y})


class TestFitOLS:
    def test_returns_olsresult(self):
        df = _perfect_data()
        result = fit_ols(df, "x", "y")
        assert isinstance(result, OLSResult)

    def test_coefficients_match_known_values(self):
        n = 100
        true_intercept = 0.5
        true_slope = 2.0
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 1, n)
        y = true_intercept + true_slope * x + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"x": x, "y": y})
        result = fit_ols(df, "x", "y")
        assert abs(result.intercept - true_intercept) < 0.1
        assert abs(result.slope - true_slope) < 0.1

    def test_r_squared_high_for_linear_data(self):
        df = _perfect_data()
        result = fit_ols(df, "x", "y")
        assert result.r2 > 0.9

    def test_confidence_interval_contains_true_value(self):
        df = _perfect_data()
        result = fit_ols(df, "x", "y")
        assert result.conf_int_slope_low < 2.0 < result.conf_int_slope_high

    def test_nobs_matches_input(self):
        df = _perfect_data(75)
        result = fit_ols(df, "x", "y")
        assert result.nobs == 75

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError):
            fit_ols(df, "x", "y")

    def test_empty_data_raises_value_error(self):
        df = pd.DataFrame({"x": [np.nan, np.nan], "y": [1, 2]})
        with pytest.raises(ValueError):
            fit_ols(df, "x", "y")

    def test_single_row_raises_error(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with pytest.raises(KeyError):
            fit_ols(df, "x", "y")

    def test_x_col_not_in_df(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        with pytest.raises(KeyError):
            fit_ols(df, "x", "y")

    def test_y_col_not_in_df(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(KeyError):
            fit_ols(df, "x", "y")

    def test_zero_variance_x_raises_error(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [2.0, 3.0, 4.0]})
        with pytest.raises(KeyError):
            fit_ols(df, "x", "y")
