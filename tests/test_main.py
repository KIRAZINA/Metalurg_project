import sys
from pathlib import Path
from unittest.mock import patch

import test_metal.__main__  # noqa: F401
from main import configure_logging, main


class TestConfigureLogging:
    def test_creates_log_directory(self, tmp_path):
        configure_logging(str(tmp_path / "logs"))
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "logs" / "run.log").exists()

    def test_does_not_raise_on_existing(self, tmp_path):
        (tmp_path / "logs2").mkdir(parents=True)
        configure_logging(str(tmp_path / "logs2"))
        assert (tmp_path / "logs2" / "run.log").exists()


class TestMain:
    @patch("main.run_pipeline_with_io")
    def test_main_calls_pipeline(self, mock_run):
        test_file = Path(__file__).resolve().parent.parent / "source_data.xls"
        if not test_file.exists():
            test_file = Path("source_data.xls")
        test_args = ["prog", "--file", str(test_file), "--output", str(Path("outputs"))]
        with patch.object(sys, "argv", test_args):
            main()
        mock_run.assert_called_once()

    @patch("main.run_pipeline_with_io")
    def test_main_with_custom_args(self, mock_run):
        test_args = [
            "prog",
            "--file",
            "data.xls",
            "--output",
            "results",
            "--mode",
            "before",
            "--missing-threshold",
            "0.3",
            "--header-row",
            "2",
            "--usecols",
            "A:Z",
            "--x-columns",
            "col1",
            "col2",
            "--y-column",
            "target",
        ]
        with patch.object(sys, "argv", test_args):
            main()
        mock_run.assert_called_once()

    @patch("main.run_pipeline_with_io")
    def test_main_file_not_found_handled(self, mock_run):
        mock_run.side_effect = FileNotFoundError("test error")
        test_args = ["prog", "--file", "missing.xls", "--output", "outs"]
        with patch.object(sys, "argv", test_args):
            main()

    @patch("main.run_pipeline_with_io")
    def test_main_key_error_handled(self, mock_run):
        mock_run.side_effect = KeyError("test error")
        test_args = ["prog", "--file", "data.xls", "--output", "outs"]
        with patch.object(sys, "argv", test_args):
            main()
