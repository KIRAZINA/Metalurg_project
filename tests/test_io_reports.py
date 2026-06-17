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
        solutions = [
            ParetoOptimum(
                solution_id=0,
                input_values={"S": 0.1, "Si": 0.2},
                output_values={"S": 0.05, "Si": 0.1},
                total_impurity_input=0.3,
                total_impurity_output=0.15,
                efficiency=50.0,
            ),
            ParetoOptimum(
                solution_id=1,
                input_values={"S": 0.2},
                output_values={"S": 0.08},
                total_impurity_input=0.2,
                total_impurity_output=0.08,
                efficiency=60.0,
            ),
        ]
        df = build_pareto_rows(solutions)
        assert "solution_id" in df.columns
        assert "S_input" in df.columns
        assert "S_output" in df.columns
        assert "Si_input" in df.columns
        assert "Si_output" in df.columns
        assert "total_impurity_input" in df.columns
        assert "total_impurity_output" in df.columns
        assert "efficiency_%" in df.columns
        assert len(df) == 2

    def test_empty_list(self):
        df = build_pareto_rows([])
        assert df.empty

    def test_single_solution(self):
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
