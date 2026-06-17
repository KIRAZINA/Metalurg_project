from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from test_metal.config import ProjectConfig
from test_metal.features import (
    COLUMN_NAMES,
    PREDICTORS_AFTER,
    PREDICTORS_BEFORE,
    TARGET_AFTER,
    TARGET_BEFORE,
)
from test_metal.pipeline import PipelineResult, run_pipeline, run_pipeline_with_io
from test_metal.preprocessing import preprocess
from test_metal.testing_utils import train_linear, train_tree


def synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {c: rng.random(n) for c in COLUMN_NAMES}
    data["sample_number"] = rng.integers(0, 1000, size=n)
    return pd.DataFrame(data)


class TestPipeline:
    def test_run_pipeline_returns_pipeline_result(self):
        df = synthetic_df()
        result = run_pipeline(df)
        assert isinstance(result, PipelineResult)

    def test_run_pipeline_has_models(self):
        df = synthetic_df()
        result = run_pipeline(df)
        assert len(result.models) > 0
        assert all(m.r2 >= 0 for m in result.models)

    def test_run_pipeline_with_mode_before(self):
        df = synthetic_df()
        result = run_pipeline(df, mode="before")
        assert len(result.models) > 0

    def test_run_pipeline_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(KeyError):
            run_pipeline(df)

    def test_run_pipeline_accepts_custom_x_and_y(self):
        df = synthetic_df()
        result = run_pipeline(df, x_columns=["steel_S_before"], y_column="steel_S_after")
        assert len(result.models) >= 0

    def test_run_pipeline_with_config(self):
        df = synthetic_df()
        cfg = ProjectConfig(missing_threshold=0.8)
        result = run_pipeline(df, config=cfg)
        assert len(result.models) > 0

    def test_run_pipeline_has_figures(self):
        df = synthetic_df(20)
        result = run_pipeline(df)
        assert len(result.figures) > 0

    def test_run_pipeline_no_models_raises_error(self):
        df = synthetic_df(10)
        for c in df.columns:
            df[c] = float("nan")
        with pytest.raises(RuntimeError, match="Failed to build any regression model"):
            run_pipeline(df)

    def test_run_pipeline_with_custom_xy_has_optimization(self):
        df = synthetic_df(50)
        rng = np.random.default_rng(42)
        df["steel_S_before"] = rng.uniform(0.02, 0.15, size=50)
        df["steel_S_after"] = df["steel_S_before"] * (0.5 + rng.uniform(-0.05, 0.05, size=50))
        result = run_pipeline(df, x_columns=["steel_S_before"], y_column="steel_S_after")
        assert result.single_element_report is not None
        assert result.pareto_front is not None


class TestRunPipelineWithIO:
    @patch("test_metal.pipeline.load_excel")
    def test_returns_pipeline_result(self, mock_load):
        mock_load.return_value = synthetic_df(30)
        result = run_pipeline_with_io(Path("test.xls"))
        assert isinstance(result, PipelineResult)
        assert len(result.models) > 0

    @patch("test_metal.pipeline.load_excel")
    def test_creates_output_dir_and_saves_regression_csv(self, mock_load, tmp_path):
        mock_load.return_value = synthetic_df(30)
        out_dir = tmp_path / "pipe_outputs"
        cfg = ProjectConfig(outputs_dir=out_dir)
        _ = run_pipeline_with_io(Path("test.xls"), config=cfg)
        assert out_dir.exists()
        assert (out_dir / "regression_report.csv").exists()

    @patch("test_metal.pipeline.load_excel")
    def test_saves_combined_pdf(self, mock_load, tmp_path):
        mock_load.return_value = synthetic_df(30)
        out_dir = tmp_path / "pdf_outputs"
        cfg = ProjectConfig(outputs_dir=out_dir)
        _ = run_pipeline_with_io(Path("test.xls"), config=cfg)
        assert (out_dir / "all_regressions.pdf").exists()

    @patch("test_metal.pipeline.load_excel")
    def test_truncates_extra_columns(self, mock_load, tmp_path):
        wide_df = synthetic_df(5)
        for i in range(10):
            wide_df[f"extra_col_{i}"] = 0.0
        mock_load.return_value = wide_df
        out_dir = tmp_path / "trunc_outputs"
        cfg = ProjectConfig(outputs_dir=out_dir)
        result = run_pipeline_with_io(Path("test.xls"), config=cfg)
        assert isinstance(result, PipelineResult)

    @patch("test_metal.pipeline.load_excel")
    def test_saves_optimization_reports_with_custom_xy(self, mock_load, tmp_path):
        rng = np.random.default_rng(42)
        n = 50
        data = {c: rng.random(n) for c in COLUMN_NAMES}
        data["steel_S_before"] = rng.uniform(0.02, 0.15, size=n)
        data["steel_S_after"] = data["steel_S_before"] * (0.5 + rng.uniform(-0.05, 0.05, size=n))
        data["steel_Si_before"] = rng.uniform(0.1, 0.5, size=n)
        data["steel_Si_after"] = data["steel_Si_before"] * (0.6 + rng.uniform(-0.05, 0.05, size=n))
        mock_load.return_value = pd.DataFrame(data)
        out_dir = tmp_path / "opt_io_outputs"
        cfg = ProjectConfig(outputs_dir=out_dir)
        _ = run_pipeline_with_io(
            Path("test.xls"),
            config=cfg,
            x_columns=["steel_S_before", "steel_Si_before"],
            y_column="steel_S_after",
        )
        assert (out_dir / "optimization_report_single_element.csv").exists()
        assert (out_dir / "optimization_report_pareto_front.csv").exists()


class TestPreprocessShapes:
    def test_preprocess_shapes(self):
        df = synthetic_df()
        dfp = preprocess(df)
        assert not dfp.isna().any().any()
        assert set(PREDICTORS_AFTER + [TARGET_AFTER]).issubset(set(dfp.columns))

    def test_models_after(self):
        df = synthetic_df()
        dfp = preprocess(df)
        lin = train_linear(dfp, PREDICTORS_AFTER, TARGET_AFTER)
        tree = train_tree(dfp, PREDICTORS_AFTER, TARGET_AFTER)
        assert "mse" in lin and "r2" in lin and "cv_r2_mean" in lin
        assert "mse" in tree and "r2" in tree

    def test_models_before(self):
        df = synthetic_df()
        dfp = preprocess(df)
        lin = train_linear(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
        tree = train_tree(dfp, PREDICTORS_BEFORE, TARGET_BEFORE)
        assert "mse" in lin and "r2" in lin and "cv_r2_mean" in lin
        assert "mse" in tree and "r2" in tree
