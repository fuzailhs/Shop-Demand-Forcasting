"""Model tests: metrics, baseline, holdout discipline, intervals, fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import model
from model import ModelConfig, evaluate_models, forecast


def make_series(values, start="2025-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", name="date")
    return pd.Series([float(value) for value in values], index=index)


def weekly_series(days: int = 180, base: float = 20.0) -> pd.Series:
    """Exact weekly cycle with no noise: 10,12,14,16,18,30,34 repeating."""
    pattern = [10, 12, 14, 16, 18, 30, 34]
    return make_series([pattern[index % 7] for index in range(days)])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_mae_is_the_average_absolute_miss():
    assert model.mean_absolute_error_units([10, 20, 30], [12, 18, 33]) == pytest.approx(
        (2 + 2 + 3) / 3
    )


def test_mae_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        model.mean_absolute_error_units([1, 2], [1])


def test_wape_divides_total_error_by_total_actuals():
    assert model.weighted_absolute_percentage_error([10, 30], [12, 27]) == pytest.approx(
        5 / 40
    )


def test_wape_is_unavailable_when_actuals_are_all_zero():
    assert model.weighted_absolute_percentage_error([0, 0], [3, 4]) is None


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def test_seasonal_naive_repeats_the_last_week():
    history = weekly_series(days=21)
    result = model.seasonal_naive_forecast(history.to_numpy(), 10)

    last_week = history.to_numpy()[-7:].tolist()
    expected = (last_week * 2)[:10]
    assert result.tolist() == expected


def test_seasonal_naive_uses_the_value_from_seven_days_earlier():
    # Last observed day is index 6 (value 7); day 7 is seven days after day 0.
    history = make_series([1, 2, 3, 4, 5, 6, 7])
    assert model.seasonal_naive_forecast(history.to_numpy(), 3).tolist() == [1.0, 2.0, 3.0]


def test_seasonal_naive_rejects_an_empty_series():
    with pytest.raises(ValueError):
        model.seasonal_naive_forecast([], 7)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #


def test_known_weekly_series_is_predicted_accurately():
    """A noiseless weekly cycle must be reproduced exactly."""
    series = weekly_series(days=200)
    result = forecast(series, 7, ModelConfig(), product="Test")

    pattern = [10, 12, 14, 16, 18, 30, 34]
    first_future_offset = 200  # index of the first forecast day
    expected = [pattern[(first_future_offset + step) % 7] for step in range(7)]

    rounded = np.floor(result.point + 0.5).astype(int)
    assert rounded.tolist() == expected


def test_holdout_is_the_most_recent_block():
    series = weekly_series(days=200)
    evaluation = evaluate_models(series, ModelConfig())

    assert evaluation.available is True
    assert evaluation.holdout_days == 28
    assert evaluation.train_days == 200 - 28
    assert evaluation.residuals is not None
    assert len(evaluation.residuals) == 28


def test_baseline_is_always_reported_alongside_the_model():
    series = weekly_series(days=200)
    evaluation = evaluate_models(series, ModelConfig())

    assert evaluation.mae_baseline is not None
    assert evaluation.mae_model is not None
    assert evaluation.wape_baseline is not None
    assert evaluation.best in {"model", "baseline", "tie"}


def test_delta_mae_sign_matches_the_winner():
    series = make_series(
        20 + (np.arange(200) % 7) + np.random.default_rng(3).normal(0, 8, 200)
    )
    evaluation = evaluate_models(series, ModelConfig())

    assert evaluation.delta_mae == pytest.approx(
        evaluation.mae_model - evaluation.mae_baseline
    )
    if evaluation.best == "model":
        assert evaluation.delta_mae < 0
    elif evaluation.best == "baseline":
        assert evaluation.delta_mae > 0
    else:
        assert evaluation.delta_mae == pytest.approx(0)


def test_evaluation_needs_120_days_and_explains_why():
    series = weekly_series(days=119)
    evaluation = evaluate_models(series, ModelConfig())

    assert evaluation.available is False
    assert "120" in evaluation.reason


def test_evaluation_is_available_at_exactly_120_days():
    evaluation = evaluate_models(weekly_series(days=120), ModelConfig())
    assert evaluation.available is True


# --------------------------------------------------------------------------- #
# Forecast shape, bounds and safety
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("horizon", [7, 30])
def test_forecast_returns_one_row_per_future_day(horizon):
    series = weekly_series(days=200)
    result = forecast(series, horizon, ModelConfig(), product="Test")

    assert len(result.forecast_dates) == horizon
    assert len(result.point) == horizon
    assert result.forecast_dates[0] == series.index[-1].date() + pd.Timedelta(days=1)
    assert result.forecast_dates[-1] == series.index[-1].date() + pd.Timedelta(
        days=horizon
    )


def test_forecast_dates_are_consecutive_and_do_not_reuse_history():
    series = weekly_series(days=150)
    result = forecast(series, 30, ModelConfig())

    dates = pd.to_datetime(result.forecast_dates)
    assert (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all()
    assert dates.min() > series.index[-1]


def test_intervals_are_ordered_and_never_negative():
    series = make_series(
        20 + (np.arange(200) % 7) + np.random.default_rng(5).normal(0, 12, 200)
    )
    result = forecast(series, 30, ModelConfig())

    assert result.lower is not None and result.upper is not None
    assert np.all(result.lower <= result.point)
    assert np.all(result.point <= result.upper)
    assert np.all(result.lower >= 0)
    assert np.all(result.point >= 0)


def test_bounds_are_omitted_with_a_reason_for_short_history():
    series = weekly_series(days=100)  # below the 120-day evaluation gate
    result = forecast(series, 7, ModelConfig())

    assert result.lower is None
    assert result.upper is None
    assert result.interval_note
    assert "backtest" in result.interval_note.lower()


def test_all_zero_history_forecasts_zero_without_crashing():
    series = make_series([0] * 150)
    result = forecast(series, 7, ModelConfig())

    assert np.allclose(result.point, 0.0)
    assert np.all(result.lower == 0) if result.lower is not None else True


def test_forecast_is_reproducible_for_the_same_input_and_seed():
    series = make_series(
        15 + (np.arange(200) % 7) + np.random.default_rng(9).normal(0, 5, 200)
    )
    first = forecast(series, 14, ModelConfig(random_state=42))
    second = forecast(series, 14, ModelConfig(random_state=42))

    np.testing.assert_allclose(first.point, second.point)


def test_horizon_below_one_is_rejected():
    with pytest.raises(ValueError):
        forecast(weekly_series(days=150), 0, ModelConfig())


def test_series_shorter_than_90_days_is_rejected():
    with pytest.raises(ValueError, match="90"):
        forecast(weekly_series(days=60), 7, ModelConfig())


# --------------------------------------------------------------------------- #
# Reliability: documented fallback path
# --------------------------------------------------------------------------- #


def test_model_failure_falls_back_to_the_baseline(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(model, "train_model", boom)

    series = weekly_series(days=200)
    result = forecast(series, 7, ModelConfig())

    assert result.model_used == model.BASELINE_MODEL
    assert result.fallback_reason
    assert result.warnings
    # Baseline output must still be complete and exportable.
    assert len(result.point) == 7


def test_fallback_during_evaluation_still_reports_both_scores(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(model, "train_model", boom)

    evaluation = evaluate_models(weekly_series(days=200), ModelConfig())

    assert evaluation.available is True
    assert evaluation.model_used == model.BASELINE_MODEL
    assert evaluation.fallback_reason
    # The model fell back, so it should tie the baseline exactly.
    assert evaluation.mae_model == pytest.approx(evaluation.mae_baseline)


def test_settings_capture_the_run_configuration():
    result = forecast(weekly_series(days=200), 30, ModelConfig(random_state=7))

    assert result.settings["horizon_days"] == 30
    assert result.settings["random_state"] == 7
    assert result.settings["model_version"] == ModelConfig().model_version
    assert result.settings["history_days"] == 200
