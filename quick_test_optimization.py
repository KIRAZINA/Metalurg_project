#!/usr/bin/env python3
"""
Script for quick testing of optimization module.
Can be run directly from console:
    python quick_test_optimization.py
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_imports():
    """Check that all modules import without errors."""
    logger.info("=" * 80)
    logger.info("TEST 1: Import verification")
    logger.info("=" * 80)
    
    try:
        from test_metal.optimization import (
            InverseRegression,
            ParetoOptimizer,
            OptimizationResult,
            ParetoOptimum,
            generate_optimization_report
        )
        logger.info("✓ All modules imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_classes_exist():
    """Check that all necessary classes exist."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Class and function verification")
    logger.info("=" * 80)
    
    try:
        from test_metal.optimization import (
            InverseRegression,
            ParetoOptimizer,
            OptimizationResult,
            ParetoOptimum
        )
        
        # Check attributes
        assert hasattr(InverseRegression, 'predict_required_input'), "InverseRegression.predict_required_input not found"
        assert hasattr(InverseRegression, 'optimize_single_element'), "InverseRegression.optimize_single_element not found"
        assert hasattr(ParetoOptimizer, 'generate_pareto_front'), "ParetoOptimizer.generate_pareto_front not found"
        assert hasattr(ParetoOptimizer, 'filter_pareto_front'), "ParetoOptimizer.filter_pareto_front not found"
        
        logger.info("✓ InverseRegression has methods predict_required_input and optimize_single_element")
        logger.info("✓ ParetoOptimizer has methods generate_pareto_front and filter_pareto_front")
        logger.info("✓ OptimizationResult and ParetoOptimum classes defined")
        
        return True
    except (AssertionError, AttributeError) as e:
        logger.error(f"✗ Error: {e}")
        return False


def test_simple_optimization():
    """Attempt to create simple optimization example."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Simple optimization simulation")
    logger.info("=" * 80)
    
    try:
        import numpy as np
        from test_metal.optimization import InverseRegression
        from test_metal.modeling import OLSResult
        
        # Create dummy regression model
        x_data = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        y_data = 0.02 + 0.5 * x_data  # y = 0.02 + 0.5*x
        
        mock_model = OLSResult(
            x_col="steel_S_before",
            y_col="steel_S_after",
            intercept=0.02,
            slope=0.5,
            stderr_intercept=0.001,
            stderr_slope=0.01,
            pvalue_intercept=0.05,
            pvalue_slope=0.001,
            r2=0.95,
            df_resid=3,
            nobs=5,
            conf_int_intercept_low=0.018,
            conf_int_intercept_high=0.022,
            conf_int_slope_low=0.48,
            conf_int_slope_high=0.52,
            x=np.arange(len(x_data)),
            y=y_data,
            y_hat=y_data,
            mean_ci_low=y_data - 0.01,
            mean_ci_high=y_data + 0.01,
        )
        
        # Create inverse regressor
        inverse = InverseRegression([mock_model])
        logger.info("✓ InverseRegression created with model")
        
        # Predict required input
        target_output = 0.08
        required_input, feasible, confidence = inverse.predict_required_input("steel_S_before", target_output)
        
        logger.info(f"  Target output: {target_output}")
        logger.info(f"  Required input: {required_input:.6f}")
        logger.info(f"  Expected input: 0.12 (0.02 + 0.5*x = 0.08 → x = 0.12)")
        logger.info(f"  Feasible: {feasible}")
        logger.info(f"  Confidence: {confidence}")
        
        # Check that result is close to expected
        expected = 0.12
        if abs(required_input - expected) < 0.001:
            logger.info(f"✓ Result is correct (difference < 0.001)")
            return True
        else:
            logger.error(f"✗ Result is incorrect (difference > 0.001)")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_integration():
    """Check integration with main.py."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: main.py integration check")
    logger.info("=" * 80)
    
    try:
        import main
        
        # Check that main.py imports optimization modules
        source = Path("main.py").read_text(encoding="utf-8")
        
        checks = [
            ("from test_metal.optimization import" in source, "Optimization module import"),
            ("InverseRegression" in source, "InverseRegression usage"),
            ("ParetoOptimizer" in source, "ParetoOptimizer usage"),
            ("generate_optimization_report" in source, "generate_optimization_report usage"),
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                logger.info(f"✓ {description}")
            else:
                logger.error(f"✗ {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        logger.error(f"✗ Error checking main.py: {e}")
        return False


def check_documentation():
    """Check for documentation files."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Documentation check")
    logger.info("=" * 80)
    
    files_to_check = [
        ("OPTIMIZATION_README.md", "Full documentation"),
        ("QUICKSTART_OPTIMIZATION.md", "Quick start guide"),
        ("IMPLEMENTATION_SUMMARY.md", "Change summary"),
        ("example_optimization.py", "Usage example"),
    ]
    
    all_found = True
    for filename, description in files_to_check:
        if Path(filename).exists():
            logger.info(f"✓ {description} found ({filename})")
        else:
            logger.error(f"✗ {description} not found ({filename})")
            all_found = False
    
    return all_found


def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "QUICK TEST OF OPTIMIZATION MODULE".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    tests = [
        ("Imports", test_imports),
        ("Classes and methods", test_classes_exist),
        ("Simple optimization", test_simple_optimization),
        ("main.py integration", test_main_integration),
        ("Documentation", check_documentation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary report
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 80)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:<30} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    logger.info("\n" + "=" * 80)
    if passed == total:
        logger.info(f"✓ ALL TESTS PASSED ({passed}/{total})")
        logger.info("Optimization module is ready for use!")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python main.py --file source_data.xls")
        logger.info("  2. Or run example: python example_optimization.py")
        logger.info("  3. Read documentation: QUICKSTART_OPTIMIZATION.md")
        return 0
    else:
        logger.error(f"✗ SOME TESTS FAILED ({total - passed}/{total})")
        logger.error("Check logs above for diagnostics")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
