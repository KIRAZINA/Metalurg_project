# 📊 Optimization Module Documentation

## 🎯 Purpose

The `optimization.py` module is designed to solve the **inverse problem**: given target impurity values at the output (after melting), determine the **minimum required input values** (before melting) to achieve these targets.

### Task:
Find the minimum amount of sulfur (S) and silicon (Si) in raw materials that will provide maximum possible purity of the finished steel.

---

## 🔧 Key Components

### 1. **InverseRegression** — Solving the Inverse Problem

The class transforms direct regression models of the form:
```
Y_after = intercept + slope × X_before
```

Into inverse functions to determine the required input:
```
X_before = (Y_after - intercept) / slope
```

#### Methods:

**`predict_required_input(x_col, target_output)`**
- Determines the required input value to achieve the target output
- Returns: (required_input, is_feasible, confidence_level)

```python
from test_metal.optimization import InverseRegression

inverse = InverseRegression(ols_results)
required_s, feasible, confidence = inverse.predict_required_input(
    "steel_S_before", 
    target_output=0.05
)
# required_s ≈ 0.082
# feasible = True (within data range)
# confidence = "high" (if R² > 0.8)
```

**`optimize_single_element(element, x_col, target_output)`**
- Complete optimization for a single element
- Returns `OptimizationResult` with detailed recommendations

```python
result = inverse.optimize_single_element(
    element="Sulfur (S)",
    x_col="steel_S_before",
    target_output=0.05
)
print(f"Required value: {result.required_input:.6f}")
print(f"Confidence: {result.confidence}")
print(f"Notes: {result.notes}")
```

---

### 2. **ParetoOptimizer** — Multi-Objective Optimization

Builds a **Pareto front of optimal input parameter combinations** for **multiple elements simultaneously** (for example, S and Si together).

#### Problem:
When optimizing multiple elements, a **goal conflict** can arise:
- Reducing S requires one level of input
- Reducing Si requires a different level of input
- Optimal combination is a compromise between two goals

#### Solution:
The Pareto front shows **all non-dominated solutions** — such combinations where improvement in one parameter leads to deterioration in another.

#### Methods:

**`generate_pareto_front(element_targets, n_points=50)`**

```python
optimizer = ParetoOptimizer(inverse_regressor)

targets = {
    "Sulfur (S)": ("steel_S_before", 0.05),
    "Silicon (Si)": ("steel_Si_before", 0.15),
}

pareto_solutions = optimizer.generate_pareto_front(targets, n_points=100)
```

**`filter_pareto_front(solutions)`**
- Keeps only Pareto-optimal solutions
- Removes dominated variants

```python
best_solutions = optimizer.filter_pareto_front(pareto_solutions)

# Best solution (minimum input impurities)
best = best_solutions[0]
print(f"Input values: {best.input_values}")
print(f"Output values: {best.output_values}")
print(f"Purification efficiency: {best.efficiency:.2f}%")
```

---

### 3. **Helper Functions**

**`generate_optimization_report(inverse_regressor, element_targets, output_path)`**

Generates three CSV files:
1. `optimization_report_single_element.csv` — results of single element optimization
2. `optimization_report_pareto_front.csv` — Pareto front solutions
3. Contains all information needed for decision making

```python
from test_metal.optimization import generate_optimization_report

targets = {
    "Sulfur (S)": ("steel_S_before", 0.05),
    "Silicon (Si)": ("steel_Si_before", 0.15),
}

df_report = generate_optimization_report(
    inverse_regressor, 
    targets, 
    "outputs/optimization_report.csv"
)
```

---

## 📈 Usage Examples

### Example 1: Simple optimization for one element

```python
from test_metal.preprocessing import preprocess
from test_metal.modeling import fit_ols
from test_metal.optimization import InverseRegression

# Data preparation
dfp = preprocess(df)

# Build model
s_model = fit_ols(dfp, "steel_S_before", "steel_S_after")

# Inverse regression
inverse = InverseRegression([s_model])

# Find required S_before to achieve S_after = 0.05
required_s, feasible, conf = inverse.predict_required_input("steel_S_before", 0.05)

print(f"To get S_after = 0.05, need S_before = {required_s:.6f}")
print(f"Feasible: {feasible}, Confidence: {conf}")
```

### Example 2: Multi-objective optimization (S + Si)

```python
from test_metal.optimization import InverseRegression, ParetoOptimizer

# Build two models
s_model = fit_ols(dfp, "steel_S_before", "steel_S_after")
si_model = fit_ols(dfp, "steel_Si_before", "steel_Si_after")

inverse = InverseRegression([s_model, si_model])

# Target values
targets = {
    "Sulfur": ("steel_S_before", 0.05),
    "Silicon": ("steel_Si_before", 0.15),
}

# Pareto front
optimizer = ParetoOptimizer(inverse)
pareto = optimizer.generate_pareto_front(targets, n_points=100)
pareto_best = optimizer.filter_pareto_front(pareto)

# Top-3 solutions
for i, sol in enumerate(pareto_best[:3]):
    print(f"\nSolution {i+1}:")
    print(f"  S_before: {sol.input_values.get('Sulfur', 'N/A'):.6f}")
    print(f"  Si_before: {sol.input_values.get('Silicon', 'N/A'):.6f}")
    print(f"  Purification efficiency: {sol.efficiency:.2f}%")
```

### Example 3: Complete report (as in main.py)

```python
# In main.py reports are automatically generated:
# - optimization_report_single_element.csv
# - optimization_report_pareto_front.csv

# View results
import pandas as pd

df_single = pd.read_csv("outputs/optimization_report_single_element.csv")
df_pareto = pd.read_csv("outputs/optimization_report_pareto_front.csv")

print("Single element optimization:")
print(df_single)

print("\nPareto front (top-5 solutions):")
print(df_pareto.head())
```

---

## 📊 Results Interpretation

### OptimizationResult (single element optimization)

| Field | Description |
|-------|-------------|
| `element` | Element name (S, Si) |
| `target_output` | Target value after melting |
| `required_input` | Required value before melting |
| `predicted_output` | Predicted value from model |
| `r2_score` | Model quality (0-1, higher is better) |
| `is_feasible` | Is target value achievable |
| `confidence` | Confidence level (high/medium/low) |
| `notes` | Additional remarks |

### ParetoOptimum (point on Pareto front)

| Field | Description |
|-------|-------------|
| `input_values` | Required input values {element: value} |
| `output_values` | Predicted output values |
| `total_impurity_input` | Total input impurities |
| `total_impurity_output` | Total output impurities |
| `efficiency` | Purification efficiency in % |

**Purification efficiency:**
```
Efficiency = (Input - Output) / Input × 100%

Example: 
Input = 0.15, Output = 0.05
Efficiency = (0.15 - 0.05) / 0.15 × 100% = 66.67%
```

---

## ⚠️ Important Notes

### 1. **Model Linearity**
The module assumes **linear dependence**:
```
Y_after = intercept + slope × X_before
```

If the relationship is non-linear, prediction accuracy will decrease. Check R²!

### 2. **Extrapolation**
If the required input is outside the range of observed data:
- ⚠️ Result becomes less reliable
- Module outputs warning in log
- `is_feasible = False`

### 3. **Model Quality (R²)**
- **R² > 0.8**: high confidence
- **0.6 < R² ≤ 0.8**: medium confidence
- **R² ≤ 0.3**: model not suitable for prediction

### 4. **Physical Constraints**
The module does not account for physical/technological constraints:
- Negative values
- Technological limits
- Element interactions

These checks must be added manually!

---

## 🚀 Running Example

```bash
# Run optimization example
python example_optimization.py

# Output:
# ================================================================
# EXAMPLE: Optimizing input parameters to minimize impurities
# ================================================================
# ...
# Found 45 Pareto-optimal solutions
# Solution #1:
#   S_before: 0.082145
#   Si_before: 0.145230
#   Purification efficiency: 68.52%
# ...
```

---

## 📝 Integration with main.py

In the `main.py` file, the module automatically:

1. ✅ Builds regression models
2. ✅ Creates inverse regressor
3. ✅ Determines target values (minimums from data)
4. ✅ Generates optimization reports
5. ✅ Logs recommendations

Results are saved to `outputs/` folder:
- `optimization_report_single_element.csv`
- `optimization_report_pareto_front.csv`

---

## 🔍 Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `is_feasible = False` | Output exceeds data range | Check target value range |
| `confidence = "low"` | Poor model (R² < 0.6) | Add more data or variables |
| `NaN в required_input` | Slope ≈ 0 (no relationship) | Check X and Y correlation |
| Empty Pareto front | No achievable solutions | Relax target values |

---

## 📚 Additional Information

- **Pareto Optimality**: [Wikipedia](https://en.wikipedia.org/wiki/Pareto_efficiency)
- **Linear Regression**: see `modeling.py`
- **OLS Parameters**: see `OLSResult` in `modeling.py`

