# Test Metal - Linear Regression & Optimization Framework

A tool for analyzing physicochemical properties of materials using linear regression and parameter optimization.

## 🎯 Key Features

- **Data Loading and Preprocessing** from Excel files
- **Building Linear Regression Models** (OLS)
- **Result Visualization** with confidence intervals
- **Parameter Optimization** via inverse regression and Pareto-front
- **Report Generation** with recommendations

## 📁 Project Structure

```
test_metal/
├── features.py          # Feature and target variable definitions
├── io.py               # Data loading from Excel
├── preprocessing.py    # Data preprocessing
├── modeling.py         # Building regression models (OLS)
├── optimization.py     # Optimization modules (InverseRegression, ParetoOptimizer)
└── plotting.py         # Results visualization

main.py                 # Main analysis script
example_optimization.py # Optimization usage examples
tests/                  # Test suite
outputs/               # Analysis results (CSV, plots)
```

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Analysis

```bash
python main.py --file data.xlsx --output outputs/ --mode after
```

**Available Parameters:**
- `--file` (required) - path to Excel file with data
- `--output` - output directory for results (default: `outputs/`)
- `--mode` - analysis mode: `before` or `after` (default: `after`)
- `--missing-threshold` - data missing threshold from 0 to 1 (default: 0.3)

### Parameter Optimization

Optimization usage examples are in `example_optimization.py`:

```python
from test_metal.optimization import InverseRegression, ParetoOptimizer
from test_metal.modeling import fit_ols

# Inverse regression - find minimum parameters
inverse = InverseRegression(model)
min_params = inverse.find_minimums()

# Pareto optimization - find optimal combinations
pareto = ParetoOptimizer(model)
pareto_front = pareto.find_pareto_front()
```

## 📊 Output and Results

Results are saved to the `outputs/` directory:
- **Regression plots** (PDF)
- **Optimization reports** (CSV)
- **Execution logs** (run.log)

## ✅ Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=test_metal

# Quick smoke test
python tests/smoke_run.py
```

## 📋 Requirements

- Python 3.10+
- pandas
- scipy
- matplotlib
- openpyxl

## 📝 License

MIT

## 👨‍💻 Author

Developed as a tool for analyzing physicochemical properties of materials.

---

**Additional Documentation:**
- [OPTIMIZATION_README.md](OPTIMIZATION_README.md) - Complete optimization documentation
- [QUICKSTART_OPTIMIZATION.md](QUICKSTART_OPTIMIZATION.md) - Optimization quick start guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [TESTING.md](TESTING.md) - Testing information
