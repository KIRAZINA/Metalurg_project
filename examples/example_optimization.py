"""
Example of using optimization module to determine minimum input parameters.
"""

import logging
from pathlib import Path
from test_metal.io import load_excel
from test_metal.features import COLUMN_NAMES
from test_metal.preprocessing import preprocess
from test_metal.modeling import fit_ols
from test_metal.optimization import InverseRegression, ParetoOptimizer, generate_optimization_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def example_optimization(file_path: str, output_dir: str):
    """
    Example: determine minimum input values of sulfur and silicon
    to achieve target output values.
    """
    logging.info("=" * 80)
    logging.info("EXAMPLE: Optimizing input parameters to minimize impurities")
    logging.info("=" * 80)
    
    # Load and preprocess data
    logging.info("Loading data from %s", file_path)
    df = load_excel(file_path)
    df.columns = COLUMN_NAMES
    dfp = preprocess(df, col_threshold=0.5)
    logging.info("Data loaded and preprocessed: shape=%s", dfp.shape)
    
    # Build regression models for sulfur and silicon
    logging.info("\nBuilding regression models:")
    logging.info("-" * 80)
    
    results = []
    
    # Model for sulfur: steel_S_after ~ steel_S_before
    try:
        s_model = fit_ols(dfp, "steel_S_before", "steel_S_after")
        results.append(s_model)
        logging.info("✓ Model for sulfur: S_after ~ S_before (R² = %.4f)", s_model.r2)
    except Exception as e:
        logging.error("✗ Error building sulfur model: %s", e)
    
    # Модель для кремния: steel_Si_after ~ steel_Si_before
    try:
        si_model = fit_ols(dfp, "steel_Si_before", "steel_Si_after")
        results.append(si_model)
        logging.info("✓ Model for silicon: Si_after ~ Si_before (R² = %.4f)", si_model.r2)
    except Exception as e:
        logging.error("✗ Error building silicon model: %s", e)
    
    if not results:
        logging.error("Failed to build any models!")
        return
    
    # Inverse regression: determine input parameters
    logging.info("\n" + "=" * 80)
    logging.info("INVERSE REGRESSION: Determining required input values")
    logging.info("=" * 80)
    
    inverse_regressor = InverseRegression(results)
    
    # Target values (minimum values from data)
    s_target = dfp["steel_S_after"].min()
    si_target = dfp["steel_Si_after"].min()
    
    logging.info("\nTarget values (minimize impurities):")
    logging.info("  Target value for S (output): %.6f", s_target)
    logging.info("  Target value for Si (output): %.6f", si_target)
    
    # Optimization for each element separately
    logging.info("\nOptimization for each element separately:")
    logging.info("-" * 80)
    
    s_input, s_feasible, s_conf = inverse_regressor.predict_required_input("steel_S_before", s_target)
    si_input, si_feasible, si_conf = inverse_regressor.predict_required_input("steel_Si_before", si_target)
    
    logging.info("Sulfur (S):")
    logging.info("  Required value at input (S_before): %.6f", s_input)
    logging.info("  Feasible: %s", "YES" if s_feasible else "NO")
    logging.info("  Confidence: %s", s_conf)
    
    logging.info("Silicon (Si):")
    logging.info("  Required value at input (Si_before): %.6f", si_input)
    logging.info("  Feasible: %s", "YES" if si_feasible else "NO")
    logging.info("  Confidence: %s", si_conf)
    
    # Build Pareto front
    logging.info("\n" + "=" * 80)
    logging.info("PARETO FRONT: Optimal combinations of input parameters")
    logging.info("=" * 80)
    
    optimization_targets = {
        "Sulfur (S)": ("steel_S_before", s_target),
        "Silicon (Si)": ("steel_Si_before", si_target),
    }
    
    optimizer = ParetoOptimizer(inverse_regressor)
    pareto_solutions = optimizer.generate_pareto_front(optimization_targets, n_points=100)
    pareto_solutions = optimizer.filter_pareto_front(pareto_solutions)
    
    logging.info(f"\nFound {len(pareto_solutions)} Pareto-optimal solutions\n")
    
    if pareto_solutions:
        # Output top-5 solutions
        for i, sol in enumerate(pareto_solutions[:5], 1):
            logging.info(f"Solution #{i}:")
            logging.info("  Input values:")
            for elem, val in sol.input_values.items():
                logging.info(f"    {elem}: {val:.6f}")
            logging.info("  Output values:")
            for elem, val in sol.output_values.items():
                logging.info(f"    {elem}: {val:.6f}")
            logging.info(f"  Total input impurities: {sol.total_impurity_input:.6f}")
            logging.info(f"  Total output impurities: {sol.total_impurity_output:.6f}")
            logging.info(f"  Purification efficiency: {sol.efficiency:.2f}%")
            logging.info("")
    
    # Save reports
    logging.info("=" * 80)
    logging.info("SAVING REPORTS")
    logging.info("=" * 80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    opt_report_path = Path(output_dir) / "optimization_report.csv"
    opt_df = generate_optimization_report(inverse_regressor, optimization_targets, str(opt_report_path))
    
    # Generate Pareto front scatter plot
    from test_metal.plotting import plot_pareto_front
    plot_path = plot_pareto_front(pareto_solutions, output_dir, "pareto_front")
    
    logging.info(f"✓ Optimization report saved to {opt_report_path}")
    logging.info(f"✓ Pareto front CSV saved to {str(opt_report_path).replace('.csv', '_pareto_front.csv')}")
    if plot_path:
        logging.info(f"✓ Pareto front plot saved to {plot_path}")
    
    logging.info("\n" + "=" * 80)
    logging.info("COMPLETED")
    logging.info("=" * 80)


if __name__ == "__main__":
    # Replace paths with real paths to your files
    file_path = "source_data.xls"
    output_dir = "outputs/optimization"
    
    try:
        example_optimization(file_path, output_dir)
    except FileNotFoundError:
        logging.error(f"File {file_path} not found!")
    except Exception as e:
        logging.exception(f"Error: {e}")
