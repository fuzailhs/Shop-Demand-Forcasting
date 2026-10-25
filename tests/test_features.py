"""Feature engineering tests: column contract and leakage safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features


def make_series(values, start="2025-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", name="date")
    return pd.Series([float(value) for value in values], index=index)


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(11)
    values = 20 + (np.arange(120) % 7) + rng.normal(0, 2, 120)
    return make_series(np.round(values, 1))


def test_feature_columns_are_complete(series):
    frame = features.make_features(series)

    assert set(frame.columns) == {*features.FEATURE_COLUMNS, features.TARGET_COLUMN}
    assert frame[list(features.FEATURE_COLUMNS)].isna().sum().sum() == 0
    assert features.feature_matrix(frame).columns.tolist() == list(
        features.FEATURE_COLUMNS
    )


def test_first_complete_row_needs_a_full_window(series):
    frame = features.make_features(series)

    # 28 prior days are needed before lag_28 and the 28-day mean exist.
    assert frame.index[0] == series.index[features.MIN_HISTORY_DAYS]
    assert len(frame) == len(series) - features.MIN_HISTORY_DAYS


def test_lags_read_only_prior_days(series):
    frame = features.make_features(series)

    for lag in features.LAGS:
        for stamp in frame.index[::13]:
            expected = series.loc[stamp - pd.Timedelta(days=lag)]
            assert frame.loc[stamp, f"lag_{lag}"] == pytest.approx(expected)


def test_rolling_means_exclude_the_current_day(series):
    frame = features.make_features(series)

    for window in features.ROLLING_WINDOWS:
        stamp = frame.index[40]
        window_values = series.loc[
            stamp - pd.Timedelta(days=window) : stamp - pd.Timedelta(days=1)
        ]
        assert frame.loc[stamp, f"roll_mean_{window}"] == pytest.approx(
            window_values.mean()
        )


def test_changing_a_future_value_cannot_alter_past_features(series):
    """The leakage audit the build plan asks for in milestone 3."""
    baseline = features.make_features(series)

    tampered = series.copy()
    cutoff = tampered.index[80]
    tampered.loc[cutoff:] = tampered.loc[cutoff:] + 1000.0

    after = features.make_features(tampered)
    untouched = baseline.loc[baseline.index < cutoff]
    replayed = after.loc[after.index < cutoff]

    pd.testing.assert_frame_equal(untouched, replayed)


def test_changing_a_past_value_does_alter_later_features(series):
    """Guards against a feature pipeline that ignores its inputs entirely."""
    baseline = features.make_features(series)

    tampered = series.copy()
    tampered.loc[tampered.index[60]] += 50.0

    after = features.make_features(tampered)

    stamp = tampered.index[61]
    assert after.loc[stamp, "lag_1"] != baseline.loc[stamp, "lag_1"]
    assert after.loc[stamp, "roll_mean_28"] != baseline.loc[stamp, "roll_mean_28"]


def test_calendar_features_are_correct(series):
    frame = features.make_features(series)

    mask = frame.index.dayofweek == 6
    assert (frame.loc[mask, "day_of_week"] == 6).all()
    assert (frame.loc[mask, "month"] == frame.loc[mask].index.month).all()


def test_forecast_origin_truncates_the_frame(series):
    origin = series.index[100].date()
    frame = features.make_features(series, forecast_origin=origin)

    assert frame.index.max() <= pd.Timestamp(origin)


def test_next_feature_row_matches_the_batch_matrix(series):
    """Recursive prediction must reproduce the training features exactly."""
    next_day, row = features.next_feature_row(series)
    batch = features.make_features(series)

    extended = pd.concat(
        [series, pd.Series([np.nan], index=pd.DatetimeIndex([next_day]))]
    )
    rebuilt = features.make_features(extended).loc[next_day, list(features.FEATURE_COLUMNS)]

    pd.testing.assert_series_equal(row, rebuilt)
    assert next_day == series.index[-1] + pd.Timedelta(days=1)
    assert batch.index[-1] < next_day


def test_gapped_calendar_is_rejected(series):
    gapped = series.drop(series.index[30])

    with pytest.raises(ValueError, match="complete daily calendar"):
        features.make_features(gapped)


def test_series_without_datetime_index_is_rejected():
    with pytest.raises(TypeError):
        features.make_features(pd.Series([1.0, 2.0, 3.0]))


def test_too_little_history_cannot_build_a_feature_row():
    short = make_series(np.arange(10))

    with pytest.raises(ValueError):
        features.next_feature_row(short)
