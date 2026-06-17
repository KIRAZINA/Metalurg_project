import pandas as pd

from test_metal.core.models import OLSResult
from test_metal.features import ColumnName, build_optimization_targets, find_models_for_element


class TestFindModelsForElement:
    def test_returns_matching_models(self):
        models = [
            OLSResult(
                x_col="steel_S_before",
                y_col="steel_S_after",
                intercept=0.02,
                slope=0.5,
                stderr_intercept=0.001,
                stderr_slope=0.01,
                pvalue_intercept=0.05,
                pvalue_slope=0.001,
                r2=0.95,
                df_resid=10.0,
                nobs=12,
                conf_int_intercept_low=0.018,
                conf_int_intercept_high=0.022,
                conf_int_slope_low=0.49,
                conf_int_slope_high=0.51,
                x=pd.Series([0.1, 0.2, 0.3]),
                y=pd.Series([0.07, 0.12, 0.17]),
                y_hat=pd.Series([0.07, 0.12, 0.17]),
                mean_ci_low=pd.Series([0.06, 0.11, 0.16]),
                mean_ci_high=pd.Series([0.08, 0.13, 0.18]),
            ),
            OLSResult(
                x_col="steel_Si_before",
                y_col="steel_Si_after",
                intercept=0.01,
                slope=0.6,
                stderr_intercept=0.001,
                stderr_slope=0.01,
                pvalue_intercept=0.05,
                pvalue_slope=0.001,
                r2=0.9,
                df_resid=10.0,
                nobs=12,
                conf_int_intercept_low=0.008,
                conf_int_intercept_high=0.012,
                conf_int_slope_low=0.59,
                conf_int_slope_high=0.61,
                x=pd.Series([0.1, 0.2, 0.3]),
                y=pd.Series([0.07, 0.12, 0.17]),
                y_hat=pd.Series([0.07, 0.12, 0.17]),
                mean_ci_low=pd.Series([0.06, 0.11, 0.16]),
                mean_ci_high=pd.Series([0.08, 0.13, 0.18]),
            ),
        ]
        result = find_models_for_element(models, ColumnName.S)
        assert len(result) == 1
        assert result[0].x_col == "steel_S_before"

    def test_returns_empty_when_no_match(self):
        result = find_models_for_element([], ColumnName.S)
        assert result == []


class TestBuildOptimizationTargets:
    def test_returns_targets_for_s_and_si(self):
        s_model = OLSResult(
            x_col="steel_S_before",
            y_col="steel_S_after",
            intercept=0.02,
            slope=0.5,
            stderr_intercept=0.001,
            stderr_slope=0.01,
            pvalue_intercept=0.05,
            pvalue_slope=0.001,
            r2=0.95,
            df_resid=10.0,
            nobs=12,
            conf_int_intercept_low=0.018,
            conf_int_intercept_high=0.022,
            conf_int_slope_low=0.49,
            conf_int_slope_high=0.51,
            x=pd.Series([0.1, 0.2, 0.3]),
            y=pd.Series([0.07, 0.12, 0.17]),
            y_hat=pd.Series([0.07, 0.12, 0.17]),
            mean_ci_low=pd.Series([0.06, 0.11, 0.16]),
            mean_ci_high=pd.Series([0.08, 0.13, 0.18]),
        )
        si_model = OLSResult(
            x_col="steel_Si_before",
            y_col="steel_Si_after",
            intercept=0.01,
            slope=0.6,
            stderr_intercept=0.001,
            stderr_slope=0.01,
            pvalue_intercept=0.05,
            pvalue_slope=0.001,
            r2=0.9,
            df_resid=10.0,
            nobs=12,
            conf_int_intercept_low=0.008,
            conf_int_intercept_high=0.012,
            conf_int_slope_low=0.59,
            conf_int_slope_high=0.61,
            x=pd.Series([0.1, 0.2, 0.3]),
            y=pd.Series([0.07, 0.12, 0.17]),
            y_hat=pd.Series([0.07, 0.12, 0.17]),
            mean_ci_low=pd.Series([0.06, 0.11, 0.16]),
            mean_ci_high=pd.Series([0.08, 0.13, 0.18]),
        )
        df = pd.DataFrame(
            {"steel_S_after": [0.05, 0.08, 0.06], "steel_Si_after": [0.1, 0.15, 0.12]}
        )
        targets = build_optimization_targets([s_model, si_model], df)
        assert "Sulfur (S)" in targets
        assert "Silicon (Si)" in targets
        assert targets["Sulfur (S)"][0] == "steel_S_before"
        assert abs(targets["Sulfur (S)"][1] - 0.05) < 0.001

    def test_no_matching_models_returns_empty(self):
        df = pd.DataFrame({"steel_S_after": [0.05]})
        targets = build_optimization_targets([], df)
        assert targets == {}
