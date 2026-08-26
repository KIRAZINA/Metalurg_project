import pandas as pd

from test_metal.core.models import OLSResult, ParetoOptimum
from test_metal.io.reports import build_pareto_rows, save_optimization_csv, save_regression_csv


def _make_ols(x_col="x", y_col="y") -> OLSResult:
    return OLSResult(
        x_col=x_col,
        y_col=y_col,
        intercept=1.0,
        slope=0.5,
        stderr_intercept=0.1,
        stderr_slope=0.05,
        pvalue_intercept=0.01,
        pvalue_slope=0.001,
        r2=0.95,
        df_resid=10.0,
        nobs=12,
        conf_int_intercept_low=0.8,
        conf_int_intercept_high=1.2,
        conf_int_slope_low=0.4,
        conf_int_slope_high=0.6,
        x=pd.Series([0.1, 0.2, 0.3]),
        y=pd.Series([1.05, 1.10, 1.15]),
        y_hat=pd.Series([1.05, 1.10, 1.15]),
        mean_ci_low=pd.Series([1.0, 1.05, 1.1]),
        mean_ci_high=pd.Series([1.1, 1.15, 1.2]),
    )


class TestSaveRegressionCSV:
    def test_saves_csv_with_correct_columns(self, tmp_path):
        models = [_make_ols("x1", "y"), _make_ols("x2", "y")]
        path = tmp_path / "reports" / "regression.csv"
        save_regression_csv(models, path)
        assert path.exists()
        df = pd.read_csv(path)
        assert list(df.columns) == [
            "x_col",
            "y_col",
            "intercept",
            "slope",
            "stderr_intercept",
            "stderr_slope",
            "pvalue_intercept",
            "pvalue_slope",
            "r2",
            "df_resid",
            "nobs",
            "conf_int_intercept_low",
            "conf_int_intercept_high",
            "conf_int_slope_low",
            "conf_int_slope_high",
        ]
        assert len(df) == 2

    def test_creates_parent_directory(self, tmp_path):
        models = [_make_ols()]
        path = tmp_path / "deep" / "nested" / "report.csv"
        save_regression_csv(models, path)
        assert path.exists()

    def test_empty_models_list(self, tmp_path):
        path = tmp_path / "empty.csv"
        save_regression_csv([], path)
        assert path.parent.exists()


class TestSaveOptimizationCSV:
    def test_saves_dataframe(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "opt.csv"
        save_optimization_csv(df, path)
        assert path.exists()
        result = pd.read_csv(path)
        assert result.shape == (2, 2)

    def test_creates_parent_directory(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "sub" / "opt.csv"
        save_optimization_csv(df, path)
        assert path.exists()


class TestBuildParetoRows:
    def test_builds_dataframe_with_correct_columns(self):
        # Element labels match the production schema (features.OPTIMIZATION_ELEMENTS)
        # so build_pareto_rows can resolve them to their domain class and emit
        # the correct per-element column names.
        solutions = [
            ParetoOptimum(
                solution_id=0,
                input_values={"Sulfur (S)": 0.1, "Silicon (Si)": 0.2},
                output_values={"Sulfur (S)": 0.05, "Silicon (Si)": 0.1},
                total_impurity_input=0.3,
                total_impurity_output=0.15,
                efficiency=50.0,
            ),
            ParetoOptimum(
                solution_id=1,
                input_values={"Sulfur (S)": 0.2},
                output_values={"Sulfur (S)": 0.08},
                total_impurity_input=0.2,
                total_impurity_output=0.08,
                efficiency=60.0,
            ),
        ]
        df = build_pareto_rows(solutions)
        assert "solution_id" in df.columns
        assert "Sulfur (S)_input" in df.columns
        assert "Sulfur (S)_output" in df.columns
        assert "Silicon (Si)_input" in df.columns
        assert "Silicon (Si)_output" in df.columns
        assert "Sulfur (S)_reduction_pct" in df.columns
        assert "Silicon (Si)_growth_pct" in df.columns
        assert "total_impurity_input" in df.columns
        assert "total_impurity_output" in df.columns
        assert "efficiency_%" not in df.columns
        assert len(df) == 2
        # Numeric spot-checks on the per-element metric formula:
        # S is reduction: (0.1 − 0.05)/|0.1|*100 = 50% reduction
        # Si is additive: (0.1 − 0.2)/|0.2|*100 = -50% growth (i.e. 50% growth)
        assert abs(df.iloc[0]["Sulfur (S)_reduction_pct"] - 50.0) < 1e-9
        assert abs(df.iloc[0]["Silicon (Si)_growth_pct"] - (-50.0)) < 1e-9

    def test_empty_list(self):
        df = build_pareto_rows([])
        assert df.empty

    def test_single_solution(self):
        # Iron is not in OPTIMIZATION_ELEMENTS, so it falls back to the neutral
        # change_pct naming. This documents the fallback behavior.
        sol = ParetoOptimum(
            solution_id=0,
            input_values={"Fe": 0.5},
            output_values={"Fe": 0.3},
            total_impurity_input=0.5,
            total_impurity_output=0.3,
            efficiency=40.0,
        )
        df = build_pareto_rows([sol])
        assert df.iloc[0]["Fe_input"] == 0.5
        assert df.iloc[0]["Fe_output"] == 0.3
        # Fe has no registered element class, so it gets the neutral
        # change_pct column (not _reduction_pct or _growth_pct).
        assert "Fe_change_pct" in df.columns
        assert "Fe_reduction_pct" not in df.columns
        assert "Fe_growth_pct" not in df.columns
