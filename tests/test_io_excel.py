from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from test_metal.config import ProjectConfig
from test_metal.io.excel import load_excel


class TestLoadExcel:
    @patch("test_metal.io.excel.Path.exists", return_value=True)
    @patch("test_metal.io.excel.pd.read_excel")
    def test_load_excel_default_config(self, mock_read_excel, mock_exists):
        mock_read_excel.return_value = pd.DataFrame({"a": [1]})
        load_excel("test.xls")
        args, kwargs = mock_read_excel.call_args
        assert kwargs["header"] == 3
        assert kwargs["usecols"] == "B:CN"
        assert kwargs["engine"] == "xlrd"

    @patch("test_metal.io.excel.Path.exists", return_value=True)
    @patch("test_metal.io.excel.pd.read_excel")
    def test_load_excel_engine_by_extension(self, mock_read_excel, mock_exists):
        """Engine is derived from the file suffix, not hardcoded."""
        from test_metal.io.excel import resolve_engine

        mock_read_excel.return_value = pd.DataFrame({"a": [1]})
        load_excel("test.xlsx")
        assert mock_read_excel.call_args.kwargs["engine"] == "openpyxl"
        assert resolve_engine(Path("book.xls")) == "xlrd"
        assert resolve_engine(Path("unknown.csv")) is None

    @patch("test_metal.io.excel.Path.exists", return_value=True)
    @patch("test_metal.io.excel.pd.read_excel")
    def test_load_excel_custom_config(self, mock_read_excel, mock_exists):
        mock_read_excel.return_value = pd.DataFrame({"a": [1]})
        cfg = ProjectConfig(excel_header_row=5, excel_usecols="A:Z")
        load_excel("test.xls", config=cfg)
        args, kwargs = mock_read_excel.call_args
        assert kwargs["header"] == 5
        assert kwargs["usecols"] == "A:Z"

    @patch("test_metal.io.excel.Path.exists", return_value=True)
    @patch("test_metal.io.excel.pd.read_excel")
    def test_load_excel_explicit_overrides_config(self, mock_read_excel, mock_exists):
        mock_read_excel.return_value = pd.DataFrame({"a": [1]})
        cfg = ProjectConfig(excel_header_row=5, excel_usecols="A:Z")
        load_excel("test.xls", config=cfg, header_row=1, usecols="C:D")
        args, kwargs = mock_read_excel.call_args
        assert kwargs["header"] == 1
        assert kwargs["usecols"] == "C:D"

    def test_load_excel_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_excel("nonexistent_file.xlsx")

    @patch("test_metal.io.excel.Path.exists", return_value=True)
    @patch("test_metal.io.excel.pd.read_excel")
    def test_load_excel_path_object(self, mock_read_excel, mock_exists):
        mock_read_excel.return_value = pd.DataFrame({"a": [1]})
        result = load_excel(Path("test.xls"))
        assert isinstance(result, pd.DataFrame)

    @patch("test_metal.io.excel.Path.exists", return_value=True)
    @patch("test_metal.io.excel.pd.read_excel")
    def test_load_excel_returns_dataframe(self, mock_read_excel, mock_exists):
        expected = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_excel.return_value = expected
        result = load_excel("test.xls")
        pd.testing.assert_frame_equal(result, expected)
