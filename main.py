import argparse
import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd
from test_metal.io import load_excel
from test_metal.features import COLUMN_NAMES, TARGET_AFTER, TARGET_BEFORE, PREDICTORS_AFTER, PREDICTORS_BEFORE
from test_metal.preprocessing import preprocess
from test_metal.modeling import fit_ols, OLSResult
from test_metal.plotting import regression_ci_plot, create_combined_pdf
from test_metal.optimization import InverseRegression, ParetoOptimizer, generate_optimization_report

def select_columns(mode: str, x_columns: List[str] | None, y_column: str | None) -> tuple[List[str], str]:
    if mode == "after":
        default_x = PREDICTORS_AFTER
        default_y = TARGET_AFTER
    else:
        default_x = PREDICTORS_BEFORE
        default_y = TARGET_BEFORE
    xs = x_columns if x_columns else default_x
    y = y_column if y_column else default_y
    return xs, y

def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

def configure_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "run.log"
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

def run(file_path: str, output_dir: str, report_path: str, mode: str, x_columns: List[str] | None, y_column: str | None, missing_threshold: float) -> None:
    logging.info("Starting linear regression analysis")
    logging.info("Data loading method: pandas.read_excel")
    df = load_excel(file_path)
    logging.info("Data loaded: shape=%s", df.shape)
    df.columns = COLUMN_NAMES
    logging.info("Expected column names assigned")
    x_cols, y_col = select_columns(mode, x_columns, y_column)
    logging.info("Target variable: %s", y_col)
    logging.info("Predictors: %s", ", ".join(x_cols))
    validate_columns(df, x_cols + [y_col])
    logging.info("Preprocessing: converting to numeric format and handling missing values")
    dfp = preprocess(df, col_threshold=missing_threshold)
    logging.info("After preprocessing: shape=%s", dfp.shape)
    results = []
    figures = {}
    figures_list = []
    r2_threshold = 0.5  # Threshold for saving PNG
    for x in x_cols:
        try:
            logging.info("Regression %s ~ %s", y_col, x)
            res = fit_ols(dfp, x, y_col)
            results.append(res)
            title = f"Linear Regression {y_col} ~ {x}"
            xlabel = x
            ylabel = y_col
            save_png = res.r2 >= r2_threshold
            fig, paths = regression_ci_plot(res.x, res.y, res.y_hat, res.mean_ci_low, res.mean_ci_high, res.r2, title, xlabel, ylabel, output_dir, f"{y_col}_vs_{x}", save_png=save_png)
            figures_list.append(fig)
            if save_png and "png" in paths:
                logging.info("PNG plot saved: %s (R² = %.4f)", paths["png"], res.r2)
            else:
                logging.info("Plot added to combined PDF (R² = %.4f)", res.r2)
        except Exception as exc:
            logging.exception("Error calculating regression %s ~ %s: %s", y_col, x, exc)
    if not results:
        raise RuntimeError("Failed to build any regression model")
    
    # Create combined PDF with all regression plots
    if figures_list:
        combined_pdf_path = Path(output_dir) / "all_regressions.pdf"
        create_combined_pdf(figures_list, str(combined_pdf_path))
        logging.info("All regression plots saved to: %s (%d plots)", combined_pdf_path, len(figures_list))
    
    logging.info("Creating summary report: %s", report_path)
    rows = []
    for res in results:
        rows.append(
            {
                "x_col": res.x_col,
                "y_col": res.y_col,
                "intercept": res.intercept,
                "slope": res.slope,
                "stderr_intercept": res.stderr_intercept,
                "stderr_slope": res.stderr_slope,
                "pvalue_intercept": res.pvalue_intercept,
                "pvalue_slope": res.pvalue_slope,
                "r2": res.r2,
                "df_resid": res.df_resid,
                "nobs": res.nobs,
                "conf_int_intercept_low": res.conf_int_intercept_low,
                "conf_int_intercept_high": res.conf_int_intercept_high,
                "conf_int_slope_low": res.conf_int_slope_low,
                "conf_int_slope_high": res.conf_int_slope_high,
            }
        )
    report_df = pd.DataFrame(rows)
    report_df.to_csv(report_path, index=False)
    logging.info("Summary report saved: %s", report_path)
    
    # Optimize input parameters to minimize impurities
    logging.info("Starting input parameter optimization")
    try:
        inverse_regressor = InverseRegression(results)
        
        # Define target values for impurity minimization
        optimization_targets = {}
        
        # Find regressions for sulfur and silicon
        s_models = [r for r in results if "steel_S" in r.x_col and "steel_S_after" == r.y_col]
        si_models = [r for r in results if "steel_Si" in r.x_col and "steel_Si_after" == r.y_col]
        
        if s_models:
            s_target = dfp["steel_S_after"].min()
            optimization_targets["Sulfur (S)"] = (s_models[0].x_col, s_target)
            logging.info("Target value for Sulfur (S): %.6f", s_target)
        
        if si_models:
            si_target = dfp["steel_Si_after"].min()
            optimization_targets["Silicon (Si)"] = (si_models[0].x_col, si_target)
            logging.info("Target value for Silicon (Si): %.6f", si_target)
        
        if optimization_targets:
            # Single element optimization
            opt_report_path = Path(output_dir) / "optimization_report.csv"
            opt_df = generate_optimization_report(inverse_regressor, optimization_targets, str(opt_report_path))
            
            # Building Pareto front
            optimizer = ParetoOptimizer(inverse_regressor)
            pareto_solutions = optimizer.generate_pareto_front(optimization_targets, n_points=100)
            pareto_solutions = optimizer.filter_pareto_front(pareto_solutions)
            
            if pareto_solutions:
                logging.info("Found %d Pareto-optimal solutions", len(pareto_solutions))
                logging.info("Best solution (minimum impurities at input):")
                best = pareto_solutions[0]
                for elem, val in best.input_values.items():
                    logging.info("  %s (input): %.6f", elem, val)
                for elem, val in best.output_values.items():
                    logging.info("  %s (output): %.6f", elem, val)
                logging.info("  Total impurity at input: %.6f", best.total_impurity_input)
                logging.info("  Total impurity at output: %.6f", best.total_impurity_output)
                logging.info("  Purification efficiency: %.2f%%", best.efficiency)
        else:
            logging.info("Optimization not applied: target parameters (S, Si) not found")
    
    except Exception as exc:
        logging.exception("Error during input parameter optimization: %s", exc)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="source_data.xls")
    parser.add_argument("--output", default=str(Path("outputs")))
    parser.add_argument("--report", default=str(Path("outputs") / "regression_report.csv"))
    parser.add_argument("--mode", choices=["after", "before"], default="after")
    parser.add_argument("--x-columns", nargs="*")
    parser.add_argument("--y-column")
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    args = parser.parse_args()
    configure_logging(args.output)
    try:
        run(args.file, args.output, args.report, args.mode, args.x_columns, args.y_column, args.missing_threshold)
    except FileNotFoundError as exc:
        logging.exception("File not found: %s", exc)
    except KeyError as exc:
        logging.exception("Data structure error: %s", exc)
    except Exception as exc:
        logging.exception("Unhandled error: %s", exc)

if __name__ == "__main__":
    main()
