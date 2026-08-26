"""Command-line interface for the Test Metal analysis pipeline."""

import argparse
import logging
from pathlib import Path

from test_metal.config import ProjectConfig
from test_metal.pipeline import run_pipeline_with_io

logger = logging.getLogger(__name__)


def configure_logging(output_dir: str) -> Path:
    """Route logging to stdout and to ``<output_dir>/run.log``."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = out_path / "run.log"
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test-metal",
        description=(
            "Linear regression and Pareto optimization analysis of "
            "physicochemical properties of steel."
        ),
    )
    parser.add_argument("--file", default="source_data.xls")
    parser.add_argument("--output", default=str(Path("outputs")))
    parser.add_argument("--mode", choices=["after", "before"], default="after")
    parser.add_argument("--x-columns", nargs="*")
    parser.add_argument("--y-column")
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    parser.add_argument("--header-row", type=int, default=3)
    parser.add_argument("--usecols", default="B:CN")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    log_path = configure_logging(args.output)
    logger.info("Test Metal analysis run")
    logger.info("Log file: %s", log_path.resolve())
    logger.info(
        "Input file: %s | output dir: %s | mode: %s",
        args.file,
        args.output,
        args.mode,
    )
    config = ProjectConfig(
        excel_header_row=args.header_row,
        excel_usecols=args.usecols or "B:CN",
        missing_threshold=args.missing_threshold,
        outputs_dir=Path(args.output),
    )
    logger.info(
        "Config: header_row=%d usecols=%s missing_threshold=%.2f r2_high=%.2f "
        "r2_medium=%.2f r2_min_feasible=%.2f",
        config.excel_header_row,
        config.excel_usecols,
        config.missing_threshold,
        config.r2_high,
        config.r2_medium,
        config.r2_min_feasible,
    )
    try:
        run_pipeline_with_io(
            Path(args.file),
            config=config,
            mode=args.mode,
            x_columns=args.x_columns,
            y_column=args.y_column,
        )
    except FileNotFoundError as exc:
        logger.exception("File not found: %s", exc)
    except KeyError as exc:
        logger.exception("Data structure error: %s", exc)
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)


if __name__ == "__main__":
    main()
