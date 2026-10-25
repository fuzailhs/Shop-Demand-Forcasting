"""Forecasting model, baseline, evaluation and uncertainty.

Design decisions from the technical requirements:

* Primary model: Histogram-based Gradient Boosting Regressor.
* Baseline: seasonal naive (repeat the last observed week).
* Fallback: the baseline whenever training or evaluation fails.
* Time-based holdout of the most recent 28 days; evaluation needs 120 days.
* Recursive one-day-at-a-time prediction so each step may use earlier
  predictions as lags.
* Uncertainty: empirical 10th/90th percentiles of backtest residuals, labelled
  an approximate 80% interval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    as_daily_series,
    feature_matrix,
    make_features,
    next_feature_row,
)

logger = logging.getLogger(__name__)

PRIMARY_MODEL = "hist_gradient_boosting"
BASELINE_MODEL = "seasonal_naive_7"
INTERVAL_LABEL = "approximate 80% interval"

MIN_SERIES_DAYS = 90


@dataclass(frozen=True)
class ModelConfig:
    """All knobs that affect a run, kept together for reproducibility."""

    model_version: str = "hgb-lags-v1"
    baseline_version: str = "seasonal-naive-7-v1"
    random_state: int = 42
    holdout_days: int = 28
    min_days_for_evaluation: int = 120
    min_training_rows: int = 20
    learning_rate: float = 0.06
    max_iter: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 5
    l2_regularization: float = 1.0
    season_length: int = 7
    min_residuals: int = 7
    lower_quantile: float = 0.10
    upper_quantile: float = 0.90


@dataclass
class EvaluationResult:
    """Backtest outcome comparing the model with the seasonal-naive baseline."""

    available: bool
    reason: str | None = None
    mae_model: float | None = None
    mae_baseline: float | None = None
    wape_model: float | None = None
    wape_baseline: float | None = None
    delta_mae: float | None = None
    best: str | None = None
    holdout_days: int = 0
    train_days: int = 0
    residuals: np.ndarray | None = None
    model_used: str = BASELINE_MODEL
    fallback_reason: str | None = None


@dataclass
class ForecastResult:
    """A completed, self-contained forecast run."""

    product: str
    horizon: int
    forecast_dates: list[date]
    point: np.ndarray
    lower: np.ndarray | None
    upper: np.ndarray | None
    history: pd.Series
    model_used: str
    model_version: str
    generated_at: datetime
    interval_label: str
    interval_note: str | None = None
    fallback_reason: str | None = None
    evaluation: EvaluationResult | None = None
    settings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_bounds(self) -> bool:
        return self.lower is not None and self.upper is not None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def mean_absolute_error_units(actual, predicted) -> float:
    """MAE in units: the average size of a miss."""
    actual_array = np.asarray(actual, dtype="float64")
    predicted_array = np.asarray(predicted, dtype="float64")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual_array.size == 0:
        raise ValueError("cannot score an empty series")
    return float(np.mean(np.abs(actual_array - predicted_array)))


def weighted_absolute_percentage_error(actual, predicted) -> float | None:
    """WAPE = sum(|error|) / sum(|actual|); ``None`` when actual totals zero."""
    actual_array = np.asarray(actual, dtype="float64")
    predicted_array = np.asarray(predicted, dtype="float64")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    denominator = float(np.sum(np.abs(actual_array)))
    if denominator == 0.0:
        return None
    return float(np.sum(np.abs(actual_array - predicted_array)) / denominator)


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def seasonal_naive_forecast(values, horizon: int, season_length: int = 7) -> np.ndarray:
    """Repeat the last observed week across the horizon.

    This is the classic seasonal-naive baseline: ``y_hat[t] = y[t - 7]``, which
    for more than seven steps simply repeats the previous seven observations.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    series = np.asarray(values, dtype="float64")
    if series.size == 0:
        raise ValueError("cannot forecast from an empty series")
    season = max(1, min(int(season_length), series.size))
    last_season = series[-season:]
    repeats = int(np.ceil(horizon / season))
    return np.tile(last_season, repeats)[:horizon]


# --------------------------------------------------------------------------- #
# Training and recursive prediction
# --------------------------------------------------------------------------- #


def _build_estimator(config: ModelConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=config.random_state,
    )


def train_model(series: pd.Series, config: ModelConfig) -> HistGradientBoostingRegressor:
    """Fit the regressor on all available history."""
    daily = as_daily_series(series)
    frame = make_features(daily)
    if len(frame) < config.min_training_rows:
        raise ValueError(
            f"only {len(frame)} training rows are available; "
            f"{config.min_training_rows} are required"
        )
    estimator = _build_estimator(config)
    estimator.fit(feature_matrix(frame), frame[TARGET_COLUMN].to_numpy())
    return estimator


def recursive_forecast(
    estimator: HistGradientBoostingRegressor,
    series: pd.Series,
    horizon: int,
    config: ModelConfig,
) -> np.ndarray:
    """Predict one day at a time, appending each prediction to the history."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    working = as_daily_series(series)
    predictions = np.zeros(horizon, dtype="float64")

    for step in range(horizon):
        next_day, row = next_feature_row(working)
        batch = pd.DataFrame([row.to_numpy()], columns=list(FEATURE_COLUMNS))
        value = float(estimator.predict(batch)[0])
        value = max(0.0, value)
        predictions[step] = value
        working = pd.concat(
            [working, pd.Series([value], index=pd.DatetimeIndex([next_day]))]
        )

    return predictions


def _train_and_predict(
    series: pd.Series, horizon: int, config: ModelConfig
) -> tuple[np.ndarray, str, str | None]:
    """Return (predictions, model_used, fallback_reason).

    Any training or prediction failure degrades gracefully to the baseline.
    """
    baseline = seasonal_naive_forecast(series.to_numpy(), horizon, config.season_length)
    try:
        estimator = train_model(series, config)
        predictions = recursive_forecast(estimator, series, horizon, config)
        return predictions, PRIMARY_MODEL, None
    except Exception as exc:  # noqa: BLE001 - documented fallback path
        logger.warning(
            "model_fallback",
            extra={"error_type": type(exc).__name__, "horizon": horizon},
        )
        reason = (
            "The forecasting model could not be trained on this history, so the "
            "seasonal baseline was used instead."
        )
        return baseline, BASELINE_MODEL, reason


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_models(
    series: pd.Series, config: ModelConfig | None = None
) -> EvaluationResult:
    """Backtest on the most recent ``holdout_days`` and compare with baseline."""
    config = config or ModelConfig()
    daily = as_daily_series(series)
    total_days = len(daily)

    if total_days < config.min_days_for_evaluation:
        return EvaluationResult(
            available=False,
            reason=(
                f"Evaluation needs at least {config.min_days_for_evaluation} days "
                f"of complete daily history; this product has {total_days}. "
                "Forecasts still run, but no error metrics are reported."
            ),
            holdout_days=config.holdout_days,
            train_days=total_days,
        )

    split = total_days - config.holdout_days
    train = daily.iloc[:split]
    actual = daily.iloc[split:]

    predictions, model_used, fallback_reason = _train_and_predict(
        train, config.holdout_days, config
    )
    baseline = seasonal_naive_forecast(
        train.to_numpy(), config.holdout_days, config.season_length
    )

    mae_model = mean_absolute_error_units(actual.to_numpy(), predictions)
    mae_baseline = mean_absolute_error_units(actual.to_numpy(), baseline)
    delta = mae_model - mae_baseline
    if mae_model < mae_baseline:
        best = "model"
    elif mae_baseline < mae_model:
        best = "baseline"
    else:
        best = "tie"

    return EvaluationResult(
        available=True,
        mae_model=mae_model,
        mae_baseline=mae_baseline,
        wape_model=weighted_absolute_percentage_error(actual.to_numpy(), predictions),
        wape_baseline=weighted_absolute_percentage_error(actual.to_numpy(), baseline),
        delta_mae=delta,
        best=best,
        holdout_days=config.holdout_days,
        train_days=split,
        residuals=actual.to_numpy() - predictions,
        model_used=model_used,
        fallback_reason=fallback_reason,
    )


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #


def _interval(
    point: np.ndarray, evaluation: EvaluationResult | None, config: ModelConfig
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Empirical residual interval around each point prediction."""
    if evaluation is None or not evaluation.available or evaluation.residuals is None:
        return (
            None,
            None,
            "No uncertainty range is shown because a backtest was not available "
            "for this history length.",
        )

    residuals = np.asarray(evaluation.residuals, dtype="float64")
    if residuals.size < config.min_residuals:
        return (
            None,
            None,
            "No uncertainty range is shown because there were too few backtest "
            "errors to estimate one reliably.",
        )

    lower_offset = float(np.quantile(residuals, config.lower_quantile))
    upper_offset = float(np.quantile(residuals, config.upper_quantile))

    lower = np.maximum(0.0, point + lower_offset)
    upper = np.maximum(0.0, point + upper_offset)

    # Ordering guard: lower <= point <= upper even with skewed residuals.
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return lower, upper, None


def forecast(
    series: pd.Series,
    horizon: int,
    config: ModelConfig | None = None,
    evaluation: EvaluationResult | None = None,
    product: str = "selected product",
) -> ForecastResult:
    """Train on all history and predict ``horizon`` future days."""
    config = config or ModelConfig()
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    daily = as_daily_series(series)
    if len(daily) < MIN_SERIES_DAYS:
        raise ValueError(
            f"{MIN_SERIES_DAYS} days of daily history are required to forecast; "
            f"this product has {len(daily)}."
        )

    scoring = evaluation if evaluation is not None else evaluate_models(daily, config)
    point, model_used, fallback_reason = _train_and_predict(daily, horizon, config)
    lower, upper, interval_note = _interval(point, scoring, config)

    last_day = daily.index[-1].date()
    dates = [last_day + timedelta(days=step) for step in range(1, horizon + 1)]

    settings = {
        "horizon_days": horizon,
        "history_days": len(daily),
        "history_start": daily.index[0].date(),
        "history_end": last_day,
        "holdout_days": config.holdout_days,
        "random_state": config.random_state,
        "model_version": config.model_version,
        "baseline_version": config.baseline_version,
        "interval": INTERVAL_LABEL if lower is not None else "not available",
    }

    warnings: list[str] = []
    if fallback_reason:
        warnings.append(fallback_reason)

    return ForecastResult(
        product=product,
        horizon=horizon,
        forecast_dates=dates,
        point=point,
        lower=lower,
        upper=upper,
        history=daily,
        model_used=model_used,
        model_version=config.model_version,
        generated_at=datetime.now(timezone.utc),
        interval_label=INTERVAL_LABEL,
        interval_note=interval_note,
        fallback_reason=fallback_reason,
        evaluation=scoring,
        settings=settings,
        warnings=warnings,
    )
