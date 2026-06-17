import numpy as np
import pandas as pd
import pytest

from test_metal.preprocessing import clean_missing, preprocess, to_numeric


class TestToNumeric:
    def test_converts_object_columns(self):
        df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
        result = to_numeric(df)
        assert pd.api.types.is_numeric_dtype(result["a"])
        assert pd.api.types.is_numeric_dtype(result["b"]) or result["b"].isna().all()

    def test_handles_mixed_types(self):
        df = pd.DataFrame({"a": [1, "two", 3]})
        result = to_numeric(df)
        assert pd.isna(result.loc[1, "a"])

    def test_preserves_numeric_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = to_numeric(df)
        assert result["a"].tolist() == [1.0, 2.0, 3.0]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = to_numeric(df)
        assert result.empty


class TestCleanMissing:
    def test_drops_columns_exceeding_threshold(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, np.nan, np.nan],
                "b": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = clean_missing(df, col_threshold=0.5)
        assert "a" not in result.columns
        assert "b" in result.columns

    def test_fills_remaining_nans_with_mean(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0],
                "b": [5.0, 6.0, 7.0, 8.0],
            }
        )
        result = clean_missing(df, col_threshold=0.5)
        assert not result.isna().any().any()
        assert result.loc[2, "a"] == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)

    def test_all_nan_column_dropped(self):
        df = pd.DataFrame(
            {
                "a": [np.nan, np.nan, np.nan],
                "b": [1.0, 2.0, 3.0],
            }
        )
        result = clean_missing(df, col_threshold=0.5)
        assert "a" not in result.columns

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = clean_missing(df, col_threshold=0.5)
        assert list(result.columns) == ["a", "b"]

    def test_threshold_zero_keeps_all(self):
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, np.nan],
                "b": [1.0, 2.0, 3.0],
            }
        )
        result = clean_missing(df, col_threshold=0.0)
        assert "a" in result.columns


class TestPreprocess:
    def test_preprocess_integration(self):
        df = pd.DataFrame(
            {
                "a": ["1.0", "2.0", "3.0"],
                "b": [1.0, np.nan, 3.0],
            }
        )
        result = preprocess(df, col_threshold=0.5)
        assert not result.isna().any().any()
        assert result["a"].dtype == float

    def test_preprocess_empty_dataframe(self):
        df = pd.DataFrame()
        result = preprocess(df)
        assert result.empty

    def test_preprocess_all_nan_column(self):
        df = pd.DataFrame(
            {
                "a": [np.nan, np.nan, np.nan],
                "b": [1.0, 2.0, 3.0],
            }
        )
        result = preprocess(df, col_threshold=0.5)
        assert "a" not in result.columns
