# 🧪 Test Suite Documentation

## Overview

Complete test suite for the Test_metal project with comprehensive coverage of:
- Unit tests for preprocessing and modeling
- Edge case tests for optimization module
- Pareto front generation and filtering tests
- Smoke tests for integration

---

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures for all tests
├── test_pipeline.py                 # Basic pipeline tests
├── smoke_run.py                     # Integration smoke test
├── test_optimization_edge_cases.py  # Edge case tests for InverseRegression
└── test_optimization_pareto.py      # Pareto optimizer tests
```

---

## Running Tests

### Install pytest (if not installed)

```bash
pip install pytest pytest-cov
```

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_optimization_edge_cases.py
```

### Run specific test class

```bash
pytest tests/test_optimization_edge_cases.py::TestInverseRegressionEdgeCases
```

### Run specific test function

```bash
pytest tests/test_optimization_edge_cases.py::TestInverseRegressionEdgeCases::test_perfect_model
```

### Run with markers

```bash
# Run only fast tests
pytest -m "not slow"

# Run only integration tests
pytest -m "integration"

# Run only edge case tests
pytest -m "edge_case"
```

### Run with coverage report

```bash
pytest --cov=test_metal --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`

### Run verbose with output

```bash
pytest -v -s
```

---

## Test Categories

### Unit Tests (Fast)

**File: `tests/test_pipeline.py`**

Tests basic functionality:
- Data preprocessing
- Model training (linear and tree)
- Pipeline integrity

```bash
pytest tests/test_pipeline.py -v
```

### Edge Case Tests (Important!)

**File: `tests/test_optimization_edge_cases.py`**

Tests boundary conditions:
- Perfect models (R² = 0.95)
- Poor models (R² = 0.35)
- Extrapolation beyond data range
- Minimum/maximum target values
- Negative slopes
- Very small slopes
- Model not found errors

Classes:
- `TestInverseRegressionEdgeCases` — 10 critical tests
- `TestOptimizationResultDataclass` — data validation tests
- `TestParetoOptimumDataclass` — data structure tests

```bash
pytest tests/test_optimization_edge_cases.py -v --tb=short
```

### Pareto Optimization Tests

**File: `tests/test_optimization_pareto.py`**

Tests multi-objective optimization:
- Front generation with 1-2 elements
- Different point counts
- Solution structure validation
- Domination filtering
- Integration workflow
- Efficiency values (0-100%)
- Edge cases (single solution, identical solutions)

Classes:
- `TestParetoOptimizerFrontGeneration` — 4 tests
- `TestParetoOptimizerFiltering` — 6 tests
- `TestParetoOptimizerIntegration` — 3 tests
- `TestParetoEdgeCases` — 2 tests

```bash
pytest tests/test_optimization_pareto.py::TestParetoOptimizerFrontGeneration -v
```

### Integration Tests

**File: `tests/smoke_run.py`**

Quick smoke test of entire pipeline:
```bash
python tests/smoke_run.py
```

---

## Test Fixtures (conftest.py)

### Available Fixtures

#### 1. `synthetic_data`
Generated synthetic DataFrame with all COLUMN_NAMES

```python
def test_example(synthetic_data):
    assert synthetic_data.shape[0] == 200
    assert set(COLUMN_NAMES).issubset(set(synthetic_data.columns))
```

#### 2. `preprocessed_data`
Preprocessed synthetic data

```python
def test_example(preprocessed_data):
    assert not preprocessed_data.isna().any().any()
```

#### 3. `mock_ols_result`
High-quality OLS model (R² = 0.95)

```python
def test_example(mock_ols_result):
    inverse = InverseRegression([mock_ols_result])
    required, feasible, conf = inverse.predict_required_input("steel_S_before", 0.08)
```

#### 4. `poor_model_ols_result`
Low-quality OLS model (R² = 0.35)

```python
def test_example(poor_model_ols_result):
    inverse = InverseRegression([poor_model_ols_result])
    _, _, conf = inverse.predict_required_input("steel_Si_before", 0.08)
    assert conf == "low"
```

#### 5. `two_element_models`
Two related models (Sulfur and Silicon)

```python
def test_example(two_element_models):
    inverse = InverseRegression(two_element_models)
    optimizer = ParetoOptimizer(inverse)
    targets = {
        "Sulfur (S)": ("steel_S_before", 0.06),
        "Silicon (Si)": ("steel_Si_before", 0.12),
    }
    solutions = optimizer.generate_pareto_front(targets, n_points=50)
```

#### 6. `output_dir`
Temporary directory for test outputs

```python
def test_example(output_dir):
    report_path = Path(output_dir) / "report.csv"
    # Use temp directory
```

---

## Expected Test Results

### Quick run (no optimization tests)
```bash
pytest tests/test_pipeline.py tests/smoke_run.py
```
Expected: ~4 tests pass, quick execution

### Full test suite
```bash
pytest
```
Expected: ~35+ tests pass, execution time ~10-30 seconds

### Coverage report
```bash
pytest --cov=test_metal --cov-report=term-missing
```
Expected: >80% coverage of optimization module

---

## Debugging Tests

### Run single test with full output
```bash
pytest tests/test_optimization_edge_cases.py::TestInverseRegressionEdgeCases::test_perfect_model -vv -s
```

### Show print statements
```bash
pytest -s
```

### Drop into debugger on failure
```bash
pytest --pdb
```

### Show local variables on error
```bash
pytest -l
```

---

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'test_metal'`

**Solution:** Run tests from project root:
```bash
cd c:\1001110001000111101(1)\Python\Test_metal
pytest
```

### Issue: Fixtures not found

**Solution:** Ensure `conftest.py` is in `tests/` directory

### Issue: Tests fail on first run

**Solution:** Install dependencies:
```bash
pip install pandas numpy statsmodels scikit-learn openpyxl
```

---

## Adding New Tests

### Template for new test file

```python
"""
Tests for [module/feature].
"""

import pytest
from test_metal.optimization import InverseRegression

class TestNewFeature:
    """Test new feature."""
    
    def test_basic_functionality(self, mock_ols_result):
        """Test basic behavior."""
        inverse = InverseRegression([mock_ols_result])
        result = inverse.predict_required_input("steel_S_before", 0.08)
        
        assert result is not None
    
    def test_edge_case(self, poor_model_ols_result):
        """Test edge case."""
        # Your test here
        pass
```

### Guidelines

1. Use fixtures from `conftest.py` when possible
2. Name tests with `test_` prefix
3. Use descriptive names
4. Add docstrings
5. Test both success and failure cases
6. Group related tests in classes

---

## Continuous Integration

### Run tests automatically before commit

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
pytest tests/ || exit 1
```

---

## Performance Benchmarks

Typical test execution times:

| Test Suite | Time | Tests |
|-----------|------|-------|
| test_pipeline.py | 0.5s | 4 |
| test_optimization_edge_cases.py | 2s | 15 |
| test_optimization_pareto.py | 5s | 15 |
| smoke_run.py | 1s | 1 |
| **Total** | **~8s** | **~35** |

---

## Test Metrics

Current coverage targets:

- **test_metal/optimization.py**: 95%+
- **test_metal/modeling.py**: 80%+
- **test_metal/preprocessing.py**: 75%+
- **Overall**: 85%+

---

## Maintenance

### Update fixtures when

- New model types are added
- Column names change
- API changes in optimization module

### Review tests when

- Changing algorithm logic
- Modifying data structures
- Adding new parameters

---

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/how-to-use-pytest-in-your-project.html)
