# Test Metal ⚙️

Linear regression analysis and Pareto optimization framework for physicochemical properties of steel.

## Features

- **Linear Regression** — OLS-based analysis of element relationships in steel composition
- **Pareto Optimization** — Multi-objective optimization to find optimal input/output trade-offs
- **Rich CLI** — Full pipeline execution with step-by-step logging of calculations and saved artifacts

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start (CLI)

Run the full pipeline on your Excel data:

```bash
# For .xlsx files (openpyxl engine, selected automatically)
python main.py --file source_data.xlsx --output outputs

# For .xls files (older format; requires the optional xlrd package)
pip install xlrd
python main.py --file source_data.xls --output outputs --usecols "A:ZZ"

# Run with specific predictors and target
python main.py --file data.xlsx --output outputs \
  --x-columns steel_S_before steel_Si_before \
  --y-column steel_S_after
```

The same CLI is available as the installed console script (`test-metal ...`) or via
`python -m test_metal ...`.

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | `source_data.xls` | Path to Excel file |
| `--output` | `outputs` | Output directory (all artifacts are written here) |
| `--mode` | `after` | Analysis mode: `after` or `before` |
| `--x-columns` | (auto) | Predictor columns (space-separated) |
| `--y-column` | (auto) | Target column |
| `--missing-threshold` | `0.5` | Max fraction of missing values per column |
| `--header-row` | `3` | Row number containing column headers |
| `--usecols` | `B:CN` | Column range to read from Excel |

### Logging

Every run logs to stdout **and** to `<output>/run.log`, including:

- Input file, output directory, and resolved arguments
- Preprocessing results (coerced cells, dropped columns, final shape)
- Each fitted model (`y ~ x`, R², slope, p-value, n)
- Inverse-optimization solutions and feasibility/confidence verdicts
- Pareto candidate vs. non-dominated solution counts
- The exact path of every saved artifact

### Programmatic Usage

```python
from pathlib import Path
from test_metal.config import ProjectConfig
from test_metal.pipeline import run_pipeline_with_io

cfg = ProjectConfig(
    excel_header_row=3,
    excel_usecols="B:CN",
    missing_threshold=0.5,
    outputs_dir=Path("outputs"),
)

result = run_pipeline_with_io(
    Path("source_data.xlsx"),
    config=cfg,
    x_columns=["steel_S_before", "steel_Si_before"],
    y_column="steel_S_after",
)

# Access results
for model in result.models:
    print(f"{model.x_col} -> {model.y_col}: R²={model.r2:.3f}")

if result.single_element_report is not None:
    print(result.single_element_report)

if result.pareto_front is not None:
    print(result.pareto_front)
```

### Excel Format Notes

- The read engine is selected by file extension: `.xlsx`/`.xlsm` → `openpyxl`,
  `.xls` → `xlrd` (install it separately for legacy files).
- Column headers must be on the row specified by `--header-row` (default: row 3).
- The pipeline renames the first N columns using the internal `COLUMN_NAMES`
  mapping and drops any columns beyond it.

### Outputs

After running, the `--output` directory contains exactly:

- `regression_report.csv` — All OLS model coefficients and statistics
- `optimization_report_single_element.csv` — Inverse optimization results
- `optimization_report_pareto_front.csv` — Pareto-optimal solutions
- `all_regressions.pdf` — Combined regression plots
- `run.log` — Execution log

An example Pareto-front demo is also available:

```bash
python examples/example_optimization.py   # writes into ./outputs as well
```

## Running Tests

```bash
pytest                                        # full suite with coverage gate (>=85%)
pytest --cov=test_metal --cov-report=term-missing
ruff check .
mypy --strict test_metal/
```

## Project Structure

```
├── test_metal/              # Core library
│   ├── cli.py               #   Argument parsing + logging setup
│   ├── config.py            #   Frozen ProjectConfig (thresholds live here)
│   ├── core/                #   Regression & optimization engines
│   ├── io/                  #   Excel/PDF/CSV report generation
│   ├── features.py          #   Canonical column schema & targets
│   ├── pipeline.py          #   End-to-end analysis pipeline
│   ├── plotting.py          #   Matplotlib figure builders
│   └── preprocessing.py     #   Numeric coercion + missing-value handling
├── tests/                   # Test suite (pytest + Hypothesis)
├── examples/                # Optimization walkthrough script
├── main.py                  # Thin wrapper -> test_metal.cli.main
├── requirements.txt         # Runtime dependencies (used by Dockerfile)
├── pyproject.toml           # Packaging + tool configuration
├── Dockerfile               # Batch container: python main.py ...
├── docker-compose.yml       # One-shot batch runs (incl. example profile)
├── source_data.xls          # Example data (.xls)
└── _test_source.xlsx        # Example data (.xlsx)
```

## Docker (batch run)

```bash
docker compose up --build        # runs main.py on the mounted source_data.xls
docker compose --profile example up --build test-metal-example
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core Library | Python ≥3.10, statsmodels, scikit-learn, pandas, numpy, scipy |
| CLI | argparse, logging |
| Plots/Reports | matplotlib, seaborn, PdfPages, CSV |
| Tests | pytest, pytest-cov, Hypothesis |
| Quality | ruff, mypy --strict, pre-commit |

## License

MIT License

Copyright (c) 2024
