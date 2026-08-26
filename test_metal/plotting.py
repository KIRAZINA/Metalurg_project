from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

matplotlib.rcParams["figure.max_open_warning"] = 0


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure_multiformat(out_dir: Path, name: str, dpi: int = 300) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for ext in ("png", "pdf"):
        out_path = out_dir / f"{name}.{ext}"
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        paths[ext] = out_path
    return paths


def save_figure_png_only(out_dir: Path, name: str, dpi: int = 300) -> Path:
    out_path = out_dir / f"{name}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def regression_ci_plot(
    x: pd.Series,
    y: pd.Series,
    y_hat: pd.Series,
    ci_low: pd.Series,
    ci_high: pd.Series,
    r2: float,
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: str,
    name: str,
    save_png: bool = False,
) -> tuple[Figure, dict[str, Path]]:
    out_dir = ensure_dir(output_dir)
    order = np.argsort(x.values)
    xs = x.values[order]
    ys = y.values[order]
    ys_hat = y_hat.values[order]
    low = ci_low.values[order]
    high = ci_high.values[order]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, alpha=0.6, label="Data")
    ax.plot(xs, ys_hat, color="red", label="Regression Line")
    ax.fill_between(xs, low, high, color="red", alpha=0.2, label="95% CI")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.05, 0.95, f"$R^2 = {r2:.3f}$", transform=ax.transAxes, va="top")
    ax.legend()
    paths: dict[str, Path] = {}
    if save_png:
        png_path = save_figure_png_only(out_dir, name)
        paths["png"] = png_path
    return fig, paths


def plot_pareto_front(solutions: list[Any], output_dir: str, name: str) -> Path:
    """Scatter total_input vs total_output, color by S_reduction_pct.

    Earlier versions colored points by a combined ``efficiency`` value, but
    that mixed S-reduction and Si-growth into one signed number with no
    consistent "higher is better" reading (§12.3 of PROJECT_ARCHITECTURE.md).
    The per-element metric ``S_reduction_pct`` (always positive in this data
    set, monotonically increasing with total input) is a defensible color
    scale: it shows how the S-reduction side of the trade-off varies along
    the front without conflating with the Si side. If a future dataset
    drives Si instead of S as the primary metric, the color source should
    be revisited.
    """
    out_dir = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 8))
    inputs = [s.total_impurity_input for s in solutions]
    outputs = [s.total_impurity_output for s in solutions]
    color_values: list[float] = []
    for s in solutions:
        s_in = s.input_values.get("Sulfur (S)")
        s_out = s.output_values.get("Sulfur (S)")
        if s_in not in (None, 0) and s_out is not None:
            color_values.append((s_in - s_out) / abs(s_in) * 100)
        else:
            color_values.append(float("nan"))
    scatter = ax.scatter(inputs, outputs, c=color_values, cmap="viridis", s=50, alpha=0.8)
    ax.set_xlabel("Total Input Impurities")
    ax.set_ylabel("Total Output Impurities")
    ax.set_title("Pareto Front of Optimal Solutions (color = Sulfur reduction %)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sulfur (S) reduction (%) — positive = more removed")
    ax.grid(True, alpha=0.3)
    png_path = save_figure_png_only(out_dir, name)
    plt.close(fig)
    return png_path


def heatmap_corr(
    df: pd.DataFrame, columns: list[str], title: str, output_dir: str, name: str
) -> dict[str, Path]:
    out_dir = ensure_dir(output_dir)
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title(title)
    paths = save_figure_multiformat(out_dir, name)
    plt.close(fig)
    return paths
