"""End-to-end example: run the full analysis + Pareto optimization on sample data.

This script demonstrates the supported programmatic entry point
(`run_pipeline_with_io`), the same path used by the `test-metal` CLI. It loads an
Excel workbook of ladle/heat chemistry, fits the OLS regression suite, derives
the inverse-regression (S/Si) models, computes the single-element recommendations
and the Pareto-optimal input configurations, then writes every artifact to the
output directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import logging  # noqa: E402

from test_metal.config import ProjectConfig  # noqa: E402
from test_metal.pipeline import run_pipeline_with_io  # noqa: E402


def example_optimization(file_path: str, output_dir: str) -> None:
    """Run the complete analysis pipeline on an Excel dataset."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("test_metal.example")

    logger.info("=" * 80)
    logger.info("Test Metal example: regression + Pareto optimization")
    logger.info("=" * 80)
    logger.info("Input file: %s", file_path)
    logger.info("Output directory: %s", output_dir)

    config = ProjectConfig(outputs_dir=Path(output_dir))
    result = run_pipeline_with_io(Path(file_path), config=config, mode="after")

    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info("Fitted %d regression models", len(result.models))
    if result.single_element_report is not None:
        logger.info("Single-element optimizations:\n%s", result.single_element_report)
    else:
        logger.info("No S/Si optimization targets were available")
    logger.info(
        "Pareto-optimal solutions: %d",
        len(result.pareto_solutions) if result.pareto_solutions else 0,
    )
    logger.info("Artifacts written to: %s", Path(output_dir).resolve())
    logger.info("=" * 80)
    logger.info("COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    file_path = "source_data.xls"
    output_dir = "outputs"

    try:
        example_optimization(file_path, output_dir)
    except FileNotFoundError:
        logging.error("File %s not found!", file_path)
    except Exception as exc:
        logging.exception("Error: %s", exc)
