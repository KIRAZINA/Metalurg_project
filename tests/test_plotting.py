from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from test_metal.plotting import (  # noqa: E402
    ensure_dir,
    heatmap_corr,
    plot_pareto_front,
    regression_ci_plot,
    regression_plot,
    save_figure_multiformat,
    save_figure_png_only,
)


class TestEnsureDir:
    def test_creates_directory(self, tmp_path):
        d = tmp_path / "new_dir"
        result = ensure_dir(str(d))
        assert d.exists()
        assert result == d

    def test_existing_directory(self, tmp_path):
        result = ensure_dir(str(tmp_path))
        assert result == tmp_path


class TestSaveFigureMultiformat:
    def test_saves_png_and_pdf(self, tmp_path):
        plt.figure()
        plt.plot([1, 2], [3, 4])
        paths = save_figure_multiformat(tmp_path, "test_fig")
        assert "png" in paths
        assert "pdf" in paths
        assert paths["png"].exists()
        assert paths["pdf"].exists()
        plt.close()


class TestSaveFigurePNGOnly:
    def test_saves_png(self, tmp_path):
        plt.figure()
        plt.plot([1, 2], [3, 4])
        path = save_figure_png_only(tmp_path, "test_png")
        assert path.exists()
        assert path.suffix == ".png"
        plt.close()


class TestRegressionPlot:
    def test_returns_path(self, tmp_path):
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = [1.1, 1.9, 3.2]
        path = regression_plot(y_true, y_pred, "title", "x", "y", str(tmp_path), "reg_plot")
        assert Path(path).exists()


class TestRegressionCIPlot:
    def test_returns_figure(self, tmp_path):
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([1.0, 2.0, 3.0])
        y_hat = pd.Series([1.1, 1.9, 3.2])
        ci_low = pd.Series([0.9, 1.8, 3.0])
        ci_high = pd.Series([1.3, 2.0, 3.4])
        fig, paths = regression_ci_plot(
            x,
            y,
            y_hat,
            ci_low,
            ci_high,
            0.95,
            "title",
            "x",
            "y",
            str(tmp_path),
            "ci_plot",
        )
        assert fig is not None
        assert paths == {}

    def test_with_save_png(self, tmp_path):
        x = pd.Series([1.0, 2.0, 3.0])
        y = pd.Series([1.0, 2.0, 3.0])
        y_hat = pd.Series([1.1, 1.9, 3.2])
        ci_low = pd.Series([0.9, 1.8, 3.0])
        ci_high = pd.Series([1.3, 2.0, 3.4])
        fig, paths = regression_ci_plot(
            x,
            y,
            y_hat,
            ci_low,
            ci_high,
            0.95,
            "title",
            "x",
            "y",
            str(tmp_path),
            "ci_plot",
            save_png=True,
        )
        assert "png" in paths
        assert paths["png"].exists()

    def test_unsorted_input(self, tmp_path):
        x = pd.Series([3.0, 1.0, 2.0])
        y = pd.Series([3.0, 1.0, 2.0])
        y_hat = pd.Series([3.2, 1.1, 1.9])
        ci_low = pd.Series([3.0, 0.9, 1.8])
        ci_high = pd.Series([3.4, 1.3, 2.0])
        fig, paths = regression_ci_plot(
            x,
            y,
            y_hat,
            ci_low,
            ci_high,
            0.95,
            "Unsorted",
            "x",
            "y",
            str(tmp_path),
            "unsorted",
        )
        assert fig is not None
        plt.close(fig)


class TestPlotParetoFront:
    def test_returns_path(self, tmp_path):
        class MockSolution:
            def __init__(self, i, o, e):
                self.total_impurity_input = i
                self.total_impurity_output = o
                self.efficiency = e

        solutions = [
            MockSolution(0.5, 0.3, 40.0),
            MockSolution(0.3, 0.2, 33.3),
            MockSolution(0.4, 0.25, 37.5),
        ]
        path = plot_pareto_front(solutions, str(tmp_path), "pareto")
        assert Path(path).exists()

    def test_empty_solutions_list(self, tmp_path):
        path = plot_pareto_front([], str(tmp_path), "empty_pareto")
        assert Path(path).exists()


class TestHeatmapCorr:
    def test_returns_paths(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
        paths = heatmap_corr(df, ["a", "b", "c"], "Correlation", str(tmp_path), "heatmap")
        assert "png" in paths
        assert "pdf" in paths
        assert paths["png"].exists()
        assert paths["pdf"].exists()

    def test_subset_of_columns(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        paths = heatmap_corr(df, ["a", "b"], "Subset", str(tmp_path), "subset_heatmap")
        assert "png" in paths
