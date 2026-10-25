"""Leakage-safe feature engineering.

Every feature for day *d* is built only from observations on or before day
*d - 1*. Rolling windows are shifted one day for exactly that reason, and the
training target is never used as an input.

Training and recursive prediction share the single :func:`make_features`
code path, so a one-off scripted check cannot drift from the model.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

LAGS: tuple[int, ...] = (1, 7, 14, 28)
ROLLING_WINDOWS: tuple[int, ...] = (7, 28)
CALENDAR_FEATURES: tuple[str, ...] = ("day_of_week", "month")

FEATURE_COLUMNS: tuple[str, ...] = (
    *(f"lag_{lag}" for lag in LAGS),
    *(f"roll_mean_{window}" for window in ROLLING_WINDOWS),
    *CALENDAR_FEATURES,
)

TARGET_COLUMN = "target"

#: Days of history needed before the first complete feature row exists.
MIN_HISTORY_DAYS = max(*LAGS, *ROLLING_WINDOWS)


def as_daily_series(series: pd.Series) -> pd.Series:
    """Validate that ``series`` is a gapless daily calendar and return floats."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    ordered = series.sort_index()
    if not ordered.index.is_unique:
        raise ValueError("series index must be unique")
    if len(ordered) == 0:
        raise ValueError("series is empty")
    expected = pd.date_range(ordered.index[0], ordered.index[-1], freq="D")
    if len(expected) != len(ordered):
        raise ValueError(
            "series must cover a complete daily calendar; "
            "call build_daily_series with fill_missing=True first"
        )
    ordered = ordered.copy()
    ordered.index = pd.DatetimeIndex(expected, name="date")
    return ordered.astype("float64")


def make_features(
    series: pd.Series, forecast_origin: date | None = None
) -> pd.DataFrame:
    """Return a feature frame indexed by date.

    Columns are :data:`FEATURE_COLUMNS` plus ``target`` (the actual value for
    that date). Rows without a complete feature set are dropped, so the first
    ``MIN_HISTORY_DAYS`` rows are excluded.
    """
    daily = as_daily_series(series)
    frame = pd.DataFrame({TARGET_COLUMN: daily})

    for lag in LAGS:
        frame[f"lag_{lag}"] = frame[TARGET_COLUMN].shift(lag)

    # shift(1) => the window covers days d-1 .. d-window, never day d.
    prior_days = frame[TARGET_COLUMN].shift(1)
    for window in ROLLING_WINDOWS:
        frame[f"roll_mean_{window}"] = prior_days.rolling(
            window, min_periods=window
        ).mean()

    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month

    if forecast_origin is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(forecast_origin)]

    return frame.dropna(subset=list(FEATURE_COLUMNS))


def next_feature_row(series: pd.Series) -> tuple[pd.Timestamp, pd.Series]:
    """Build the feature row for the day after the last observation."""
    daily = as_daily_series(series)
    next_day = daily.index[-1] + pd.Timedelta(days=1)
    extended = pd.concat(
        [daily, pd.Series([np.nan], index=pd.DatetimeIndex([next_day], name="date"))]
    )
    frame = make_features(extended)
    if next_day not in frame.index:
        raise ValueError(
            f"at least {MIN_HISTORY_DAYS} days of history are needed to forecast"
        )
    return next_day, frame.loc[next_day, list(FEATURE_COLUMNS)]


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs in a stable column order."""
    return frame.loc[:, list(FEATURE_COLUMNS)]
