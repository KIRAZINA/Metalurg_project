import argparse
import logging
from pathlib import Path

from test_metal.config import ProjectConfig
from test_metal.pipeline import run_pipeline_with_io


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="source_data.xls")
    parser.add_argument("--output", default=str(Path("outputs")))
    parser.add_argument("--report", default=str(Path("outputs") / "regression_report.csv"))
    parser.add_argument("--mode", choices=["after", "before"], default="after")
    parser.add_argument("--x-columns", nargs="*")
    parser.add_argument("--y-column")
    parser.add_argument("--missing-threshold", type=float, default=0.5)
    parser.add_argument("--header-row", type=int, default=3)
    parser.add_argument("--usecols", default="B:CN")
    args = parser.parse_args()
    configure_logging(args.output)
    config = ProjectConfig(
        excel_header_row=args.header_row,
        excel_usecols=args.usecols or "B:CN",
        missing_threshold=args.missing_threshold,
        outputs_dir=Path(args.output),
    )
    x_cols, y_col = args.x_columns, args.y_column
    try:
        run_pipeline_with_io(
            Path(args.file),
            config=config,
            mode=args.mode,
            x_columns=x_cols,
            y_column=y_col,
        )
    except FileNotFoundError as exc:
        logging.exception("File not found: %s", exc)
    except KeyError as exc:
        logging.exception("Data structure error: %s", exc)
    except Exception as exc:
        logging.exception("Unhandled error: %s", exc)


if __name__ == "__main__":
    main()
