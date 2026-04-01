# Test Metal — Linear Regression & Optimization Framework

> A Python tool for analyzing physicochemical properties of steel using linear regression and multi-objective parameter optimization.

[![Tests](https://github.com/KIRAZINA/Metalurg_project/actions/workflows/tests.yml/badge.svg)](https://github.com/KIRAZINA/Metalurg_project/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Docker Support](#docker-support)
- [Quick Start](#quick-start)
- [Optimization Module](#optimization-module)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Test Metal** solves two related problems in steel metallurgy:

1. **Direct problem** — predict output impurity levels (S_after, Si_after) from input material parameters using OLS linear regression.
2. **Inverse problem** — given target output purity levels, find the *minimum required* input parameters (S_before, Si_before) to achieve them.

The inverse problem is solved via:
- **Inverse Regression** — analytically reverses each OLS model
- **Pareto Optimization** — finds all non-dominated trade-off solutions when optimizing S and Si simultaneously

---

## Project Structure

```
Test_metal/
├── main.py                    # Entry point: runs full analysis pipeline
├── setup.py                   # Package setup
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Test configuration
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml         # Docker Compose setup
├── .dockerignore              # Docker ignore file
├── source_data.xls            # Input data (steel heat measurements)
│
├── test_metal/                # Core library package
│   ├── __init__.py
│   ├── features.py            # Column names, target/predictor definitions
│   ├── io.py                  # Excel data loading
│   ├── preprocessing.py       # Data cleaning and numeric conversion
│   ├── modeling.py            # OLS regression fitting (OLSResult dataclass)
│   ├── optimization.py        # Inverse regression + Pareto optimization
│   └── plotting.py            # Regression plots with confidence intervals
│
├── examples/                  # Runnable usage examples
│   └── example_optimization.py    # Full optimization workflow with console output
│
├── tests/                     # Pytest test suite
│   ├── conftest.py            # Shared fixtures (synthetic data, mock models)
│   ├── test_pipeline.py       # Basic preprocessing and modeling tests
│   ├── test_optimization_edge_cases.py  # InverseRegression edge cases
│   ├── test_optimization_pareto.py      # ParetoOptimizer tests
│   └── smoke_run.py           # Integration smoke test
│
├── outputs/                   # Generated results (git-ignored)
│   ├── all_regressions.pdf
│   ├── regression_report.csv
│   ├── optimization_report_single_element.csv
│   ├── optimization_report_pareto_front.csv
│   └── run.log
│
└── .github/workflows/
    └── tests.yml              # CI: runs pytest on Python 3.10/3.11/3.12
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/KIRAZINA/Metalurg_project.git
cd Metalurg_project

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (required for imports)
pip install -e .
```

**Requirements:** Python 3.10+, pandas, scipy, statsmodels, matplotlib, openpyxl, scikit-learn

---

## Docker Support

The project includes Docker configuration for easy deployment and reproducible environments.

### Option 1: Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/KIRAZINA/Metalurg_project.git
cd Metalurg_project

# Run main analysis
docker-compose up test-metal

# Run optimization example
docker-compose --profile example up test-metal-example
```

### Option 2: Using Docker directly

```bash
# Build the Docker image
docker build -t test-metal .

# Run main analysis
docker run -v "$(pwd)/source_data.xls:/app/data/source_data.xls:ro" \
           -v "$(pwd)/outputs:/app/outputs" \
           test-metal

# Run optimization example
docker run -v "$(pwd)/source_data.xls:/app/source_data.xls:ro" \
           -v "$(pwd)/outputs:/app/outputs" \
           test-metal python examples/example_optimization.py
```

**Docker Features:**
- **Isolated Environment**: Python 3.11 with all dependencies pre-installed
- **Volume Mounting**: Data files and outputs are shared between host and container
- **Reproducible**: Same environment across different machines
- **No Local Setup**: No need to install Python or dependencies locally

**Docker Requirements:** Docker Engine 20.10+ and Docker Compose 2.0+

---

## Quick Start

### Run full analysis pipeline

```bash
python main.py --file source_data.xls --output outputs/ --mode after
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | `source_data.xls` | Path to Excel data file |
| `--output` | `outputs/` | Output directory for results |
| `--mode` | `after` | Analysis mode: `after` or `before` |
| `--missing-threshold` | `0.5` | Max fraction of missing values per column (0–1) |

**Generated outputs:**
- `outputs/all_regressions.pdf` — all regression plots
- `outputs/regression_report.csv` — model coefficients and statistics
- `outputs/optimization_report_single_element.csv` — per-element optimization
- `outputs/optimization_report_pareto_front.csv` — Pareto front solutions
- `outputs/run.log` — execution log

### Run optimization example

```bash
python examples/example_optimization.py
```

---

## Optimization Module

### Concept

Given a direct linear OLS model:
```
Y_after = intercept + slope × X_before    (R² = 0.85)
```

The inverse function finds the required input:
```
X_before = (Y_after − intercept) / slope

Example: target S_after = 0.05
S_before = (0.05 − 0.02) / 0.5 = 0.06
```

### InverseRegression

```python
from test_metal.optimization import InverseRegression
from test_metal.modeling import fit_ols

# Build model
model = fit_ols(dfp, x_col="steel_S_before", y_col="steel_S_after")

# Create inverse regressor
inverse = InverseRegression([model])

# Find required input to achieve target output
required_input, is_feasible, confidence = inverse.predict_required_input(
    x_col="steel_S_before",
    target_output=0.05
)
# required_input ≈ 0.082
# is_feasible = True (within observed data range)
# confidence = "high"  (if R² > 0.8)
```

**`OptimizationResult` fields:**

| Field | Description |
|-------|-------------|
| `element` | Element name (S, Si) |
| `target_output` | Target value after melting |
| `required_input` | Required value before melting |
| `r2_score` | Model quality (higher = better) |
| `is_feasible` | Whether target is achievable |
| `confidence` | `"high"` / `"medium"` / `"low"` |
| `notes` | Warnings or remarks |

### ParetoOptimizer

When optimizing multiple elements simultaneously (S + Si), goals may conflict. The Pareto front shows all non-dominated trade-off solutions.

```python
from test_metal.optimization import InverseRegression, ParetoOptimizer

inverse = InverseRegression([s_model, si_model])
optimizer = ParetoOptimizer(inverse)

targets = {
    "Sulfur (S)":   ("steel_S_before",  0.023),
    "Silicon (Si)": ("steel_Si_before", 0.087),
}

solutions = optimizer.generate_pareto_front(targets, n_points=100)
best_solutions = optimizer.filter_pareto_front(solutions)

best = best_solutions[0]
print(f"S_before:    {best.input_values['Sulfur (S)']:.6f}")
print(f"Si_before:   {best.input_values['Silicon (Si)']:.6f}")
print(f"Efficiency:  {best.efficiency:.2f}%")
```

**`ParetoOptimum` fields:**

| Field | Description |
|-------|-------------|
| `input_values` | Required input per element |
| `output_values` | Predicted output per element |
| `total_impurity_input` | Sum of all inputs |
| `total_impurity_output` | Sum of all outputs |
| `efficiency` | Purification rate: `(input − output) / input × 100%` |

**Domination rule:** Solution A dominates B when A has ≤ total input AND ≤ total output, with at least one strict inequality.

### Generate report

```python
from test_metal.optimization import generate_optimization_report

report_df = generate_optimization_report(
    inverse_regressor=inverse,
    element_targets=targets,
    output_path="outputs/optimization_report.csv"
)
```

### Important notes

- Module assumes **linear** dependency. Check R² before trusting results.
- If required input falls outside the observed data range, `is_feasible = False` and a warning is logged.
- R² confidence thresholds: `> 0.8` → high, `0.6–0.8` → medium, `< 0.6` → low.

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=test_metal --cov-report=term-missing

# Run specific test file
pytest tests/test_optimization_edge_cases.py -v

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Integration smoke test
python tests/smoke_run.py
```

**Test suite overview:**

| File | Tests | Coverage target |
|------|-------|----------------|
| `test_pipeline.py` | 4 | preprocessing, basic modeling |
| `test_optimization_edge_cases.py` | ~15 | InverseRegression edge cases |
| `test_optimization_pareto.py` | ~15 | ParetoOptimizer |
| `smoke_run.py` | 1 | full pipeline integration |

**Shared fixtures** (in `tests/conftest.py`):
- `synthetic_data` — 200-row synthetic DataFrame
- `preprocessed_data` — cleaned version
- `mock_ols_result` — high-quality model (R² = 0.95)
- `poor_model_ols_result` — poor model (R² = 0.35)
- `two_element_models` — S + Si models for Pareto tests
- `output_dir` — temporary directory for test outputs

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, tests, and pull requests.

---

## License

MIT — see [LICENSE](LICENSE).

**Author:** [KIRAZINA](https://github.com/KIRAZINA)
