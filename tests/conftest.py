"""
Shared fixtures and configuration for all tests.
"""

import numpy as np
import pandas as pd
import pytest

from test_metal.features import COLUMN_NAMES
from test_metal.modeling import OLSResult


@pytest.fixture
def synthetic_data(n: int = 200):
    """Generate synthetic data for testing."""
    rng = np.random.default_rng(42)
    data = {c: rng.random(n) for c in COLUMN_NAMES}
    data["sample_number"] = rng.integers(0, 1000, size=n)
    return pd.DataFrame(data)


@pytest.fixture
def preprocessed_data(synthetic_data):
    """Generate preprocessed synthetic data."""
    from test_metal.preprocessing import preprocess

    return preprocess(synthetic_data)


@pytest.fixture
def mock_ols_result():
    """Create a mock OLS regression result for testing."""
    x_data = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    y_data = 0.02 + 0.5 * x_data

    return OLSResult(
        x_col="steel_S_before",
        y_col="steel_S_after",
        intercept=0.02,
        slope=0.5,
        stderr_intercept=0.001,
        stderr_slope=0.01,
        pvalue_intercept=0.05,
        pvalue_slope=0.001,
        r2=0.95,
        df_resid=3,
        nobs=5,
        conf_int_intercept_low=0.018,
        conf_int_intercept_high=0.022,
        conf_int_slope_low=0.48,
        conf_int_slope_high=0.52,
        x=np.arange(len(x_data)),
        y=y_data,
        y_hat=y_data,
        mean_ci_low=y_data - 0.01,
        mean_ci_high=y_data + 0.01,
    )


@pytest.fixture
def poor_model_ols_result():
    """Create a low-quality OLS result (R² < 0.5)."""
    x_data = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    y_data = np.array([0.05, 0.06, 0.08, 0.07, 0.09, 0.08, 0.10])

    return OLSResult(
        x_col="steel_Si_before",
        y_col="steel_Si_after",
        intercept=0.05,
        slope=0.15,
        stderr_intercept=0.02,
        stderr_slope=0.05,
        pvalue_intercept=0.2,
        pvalue_slope=0.3,
        r2=0.35,  # Poor R²
        df_resid=5,
        nobs=7,
        conf_int_intercept_low=0.01,
        conf_int_intercept_high=0.09,
        conf_int_slope_low=-0.01,
        conf_int_slope_high=0.31,
        x=np.arange(len(x_data)),
        y=y_data,
        y_hat=y_data * 0.7,  # Poor fit
        mean_ci_low=y_data - 0.05,
        mean_ci_high=y_data + 0.05,
    )


@pytest.fixture
def two_element_models():
    """Create two related OLS models for multi-objective optimization."""
    # Sulfur model
    s_model = OLSResult(
        x_col="steel_S_before",
        y_col="steel_S_after",
        intercept=0.02,
        slope=0.5,
        stderr_intercept=0.001,
        stderr_slope=0.01,
        pvalue_intercept=0.05,
        pvalue_slope=0.001,
        r2=0.85,
        df_resid=10,
        nobs=12,
        conf_int_intercept_low=0.018,
        conf_int_intercept_high=0.022,
        conf_int_slope_low=0.48,
        conf_int_slope_high=0.52,
        x=np.linspace(0.1, 0.3, 12),
        y=0.02 + 0.5 * np.linspace(0.1, 0.3, 12),
        y_hat=0.02 + 0.5 * np.linspace(0.1, 0.3, 12),
        mean_ci_low=np.linspace(0.03, 0.16, 12),
        mean_ci_high=np.linspace(0.06, 0.18, 12),
    )

    # Silicon model
    si_model = OLSResult(
        x_col="steel_Si_before",
        y_col="steel_Si_after",
        intercept=0.05,
        slope=0.4,
        stderr_intercept=0.002,
        stderr_slope=0.015,
        pvalue_intercept=0.08,
        pvalue_slope=0.002,
        r2=0.78,
        df_resid=10,
        nobs=12,
        conf_int_intercept_low=0.046,
        conf_int_intercept_high=0.054,
        conf_int_slope_low=0.37,
        conf_int_slope_high=0.43,
        x=np.linspace(0.1, 0.4, 12),
        y=0.05 + 0.4 * np.linspace(0.1, 0.4, 12),
        y_hat=0.05 + 0.4 * np.linspace(0.1, 0.4, 12),
        mean_ci_low=np.linspace(0.07, 0.22, 12),
        mean_ci_high=np.linspace(0.08, 0.24, 12),
    )

    return [s_model, si_model]


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory for tests."""
    output_path = tmp_path / "outputs"
    output_path.mkdir(exist_ok=True)
    return str(output_path)
