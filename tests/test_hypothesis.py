from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
from hypothesis.strategies import floats

from test_metal.preprocessing import clean_missing, to_numeric


@given(
    data_frames(
        columns=[
            column("x", elements=floats(0, 1, allow_nan=False), dtype=float),
            column("y", elements=floats(0, 1, allow_nan=False), dtype=float),
        ],
        index=range_indexes(min_size=5, max_size=15),
    )
)
@settings(max_examples=10)
def test_clean_missing_never_produces_all_nan(df):
    result = clean_missing(df, col_threshold=0.5)
    assert not result.isna().all().all()


@given(
    data_frames(
        columns=[
            column("a", elements=floats(-1e6, 1e6), dtype=float),
            column("b", elements=floats(-1e6, 1e6), dtype=float),
        ],
        index=range_indexes(min_size=3, max_size=10),
    )
)
@settings(max_examples=5)
def test_to_numeric_preserves_shape(df):
    result = to_numeric(df)
    assert result.shape == df.shape


@given(
    data_frames(
        columns=[
            column("x", elements=floats(0, 100, allow_nan=False), dtype=float),
            column("y", elements=floats(0, 100, allow_nan=False), dtype=float),
        ],
        index=range_indexes(min_size=3, max_size=10),
    )
)
@settings(max_examples=5)
def test_clean_missing_preserves_complete_columns(df):
    result = clean_missing(df, col_threshold=0.5)
    for col in df.columns:
        if df[col].notna().all():
            assert col in result.columns
