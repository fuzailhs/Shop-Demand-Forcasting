"""Presentation and export tests (FR-10, FR-11)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

import views
from model import ModelConfig, forecast


def make_series(values, start="2025-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", name="date")
    return pd.Series([float(value) for value in values], index=index)


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(4)
    values = 20 + (np.arange(240) % 7) + rng.normal(0, 4, 240)
    series = make_series(values)
    return forecast(series, 7, ModelConfig(), product="Whole Wheat Bread")


# --------------------------------------------------------------------------- #
# Rounding (FR-08)
# --------------------------------------------------------------------------- #


def test_round_units_rounds_half_up_and_clamps_to_zero():
    rounded = views.round_units([1.49, 1.5, 2.6, -3.2])
    assert rounded.tolist() == [1, 2, 3, 0]


def test_clamp_units_does_not_round():
    assert views.clamp_units([-1.5, 2.7]).tolist() == [0.0, 2.7]


# --------------------------------------------------------------------------- #
# Export (FR-11)
# --------------------------------------------------------------------------- #


def test_export_has_exactly_the_required_columns(result):
    frame = pd.read_csv(io.BytesIO(views.build_export(result)))

    assert list(frame.columns) == list(views.EXPORT_COLUMNS)


def test_export_has_one_row_per_forecast_day(result):
    frame = pd.read_csv(io.BytesIO(views.build_export(result)))

    assert len(frame) == result.horizon
    assert frame["horizon"].unique().tolist() == [result.horizon]
    assert frame["product"].unique().tolist() == [result.product]


def test_export_predictions_match_the_displayed_table(result):
    frame = pd.read_csv(io.BytesIO(views.build_export(result)))
    table = views.forecast_table(result)

    assert frame["predicted_units"].tolist() == table["Predicted units"].tolist()
    assert views.export_matches_table(result) is True


def test_export_bounds_match_the_table(result):
    frame = pd.read_csv(io.BytesIO(views.build_export(result)))
    table = views.forecast_table(result)

    assert frame["lower_bound"].tolist() == table["Lower bound"].tolist()
    assert frame["upper_bound"].tolist() == table["Upper bound"].tolist()


def test_export_excludes_history_rows(result):
    frame = pd.read_csv(io.BytesIO(views.build_export(result)))
    dates = pd.to_datetime(frame["forecast_date"])

    assert dates.min() > result.history.index[-1]
    assert len(frame) == len(result.forecast_dates)


def test_export_leaves_bounds_blank_when_unavailable():
    series = make_series(20 + (np.arange(100) % 7))
    short = forecast(series, 7, ModelConfig())

    frame = pd.read_csv(io.BytesIO(views.build_export(short)))

    assert short.lower is None
    assert frame["lower_bound"].isna().all()
    assert frame["upper_bound"].isna().all()
    assert frame["predicted_units"].notna().all()


def test_export_filename_is_readable(result):
    assert views.export_filename(result) == "forecast-whole-wheat-bread-7d.csv"


def test_export_never_contains_negative_units(result):
    frame = pd.read_csv(io.BytesIO(views.build_export(result)))

    assert (frame["predicted_units"] >= 0).all()
    assert (frame["lower_bound"] >= 0).all()
    assert (frame["upper_bound"] >= 0).all()


def test_export_is_utf8_text(result):
    payload = views.build_export(result)
    assert isinstance(payload, bytes)
    assert payload.decode("utf-8").startswith("product,forecast_date")


# --------------------------------------------------------------------------- #
# Tables and chart (FR-10)
# --------------------------------------------------------------------------- #


def test_forecast_table_has_one_row_per_day(result):
    table = views.forecast_table(result)
    assert len(table) == result.horizon
    assert list(table.columns) == [
        "Forecast date",
        "Predicted units",
        "Lower bound",
        "Upper bound",
    ]


def test_history_table_shows_recent_actuals_newest_last(result):
    table = views.history_table(result, days=14)

    assert len(table) == 14
    assert table["Date"].is_monotonic_increasing
    assert table["Date"].iloc[-1] == result.history.index[-1].date().isoformat()


def test_chart_has_history_forecast_and_band(result):
    figure = views.build_forecast_chart(result)
    names = [trace.name for trace in figure.data]

    assert "Actual units sold" in names
    assert "Forecast units (dashed)" in names
    assert "Approximate 80% range" in names


def test_chart_omits_the_band_when_bounds_are_unavailable():
    series = make_series(20 + (np.arange(100) % 7))
    short = forecast(series, 7, ModelConfig())
    figure = views.build_forecast_chart(short)

    names = [trace.name for trace in figure.data]
    assert "Approximate 80% range" not in names
    assert "Forecast units (dashed)" in names


def test_chart_forecast_values_match_the_table(result):
    figure = views.build_forecast_chart(result)
    forecast_trace = next(
        trace for trace in figure.data if trace.name == "Forecast units (dashed)"
    )
    table = views.forecast_table(result)

    assert list(forecast_trace.y) == table["Predicted units"].tolist()


def test_chart_summary_mentions_the_horizon_and_the_table(result):
    text = views.chart_summary_text(result)

    assert "7 forecast days" in text
    assert "table" in text


def test_total_and_average_units(result):
    table = views.forecast_table(result)

    assert views.total_predicted_units(result) == int(table["Predicted units"].sum())
    assert views.average_daily_units(result) == pytest.approx(
        table["Predicted units"].mean(), abs=0.5
    )


def test_baseline_comparison_states_a_winner(result):
    sentence = views.baseline_comparison(result)

    assert "baseline" in sentence.lower() or "model" in sentence.lower()
    assert str(result.evaluation.holdout_days) in sentence


# --------------------------------------------------------------------------- #
# Copy shown to the user
# --------------------------------------------------------------------------- #


def test_limitations_cover_uncertainty_human_review_and_privacy():
    joined = " ".join(views.LIMITATIONS).lower()

    assert "decision support" in joined
    assert "promotions" in joined
    assert "80% interval" in joined
    assert "session" in joined


def test_run_settings_list_the_expected_labels(result):
    labels = [label for label, _ in views.describe_run_settings(result)]

    assert labels[:3] == ["Product", "Forecast horizon", "Model used"]
    assert "Random seed" in labels
    assert "Uncertainty" in labels
