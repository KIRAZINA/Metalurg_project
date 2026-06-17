from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


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


def regression_plot(
    y_true: pd.Series,
    y_pred: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_dir: str,
    name: str,
) -> Path:
    out_dir = ensure_dir(output_dir)
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_true, y=y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.3})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    paths = save_figure_multiformat(out_dir, name)
    plt.close()
    return paths["png"]


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
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, alpha=0.6, label="Data")
    plt.plot(xs, ys_hat, color="red", label="Regression Line")
    plt.fill_between(xs, low, high, color="red", alpha=0.2, label="95% CI")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(0.05, 0.95, f"$R^2 = {r2:.3f}$", transform=plt.gca().transAxes, va="top")
    plt.legend()
    paths: dict[str, Path] = {}
    if save_png:
        png_path = save_figure_png_only(out_dir, name)
        paths["png"] = png_path
    return fig, paths


def plot_pareto_front(solutions: list[Any], output_dir: str, name: str) -> Path:
    out_dir = ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 8))
    inputs = [s.total_impurity_input for s in solutions]
    outputs = [s.total_impurity_output for s in solutions]
    efficiencies = [s.efficiency for s in solutions]
    scatter = ax.scatter(inputs, outputs, c=efficiencies, cmap="viridis", s=50, alpha=0.8)
    ax.set_xlabel("Total Input Impurities")
    ax.set_ylabel("Total Output Impurities")
    ax.set_title("Pareto Front of Optimal Solutions")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Efficiency (%)")
    ax.grid(True, alpha=0.3)
    png_path = save_figure_png_only(out_dir, name)
    plt.close(fig)
    return png_path


def heatmap_corr(
    df: pd.DataFrame, columns: list[str], title: str, output_dir: str, name: str
) -> dict[str, Path]:
    out_dir = ensure_dir(output_dir)
    corr = df[columns].corr()
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title(title)
    paths = save_figure_multiformat(out_dir, name)
    plt.close()
    return paths
