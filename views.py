"""Presentation helpers: chart, tables, metric copy and CSV export.

Kept free of Streamlit so the chart/table/export agreement can be tested
directly. The Streamlit layer only renders these objects.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from model import ForecastResult

EXPORT_COLUMNS: tuple[str, ...] = (
    "product",
    "forecast_date",
    "predicted_units",
    "lower_bound",
    "upper_bound",
    "horizon",
    "model_version",
    "generated_at",
)

HISTORY_COLOR = "#1f3a5f"
FORECAST_COLOR = "#c2410c"
BAND_COLOR = "rgba(194, 65, 12, 0.18)"


# --------------------------------------------------------------------------- #
# Rounding helpers
# --------------------------------------------------------------------------- #


def clamp_units(values) -> np.ndarray:
    """Clamp to zero or higher without rounding."""
    return np.maximum(0.0, np.asarray(values, dtype="float64"))


def round_units(values) -> np.ndarray:
    """Whole units, half-up, never below zero (FR-08)."""
    clamped = clamp_units(values)
    return np.floor(clamped + 0.5).astype("int64")


def _round_optional(values) -> np.ndarray | None:
    if values is None:
        return None
    return round_units(values)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def forecast_table(result: ForecastResult) -> pd.DataFrame:
    """Daily forecast table shown to the user; the chart's text alternative."""
    rows = {
        "Forecast date": [day.isoformat() for day in result.forecast_dates],
        "Predicted units": round_units(result.point),
    }
    lower = _round_optional(result.lower)
    upper = _round_optional(result.upper)
    if lower is not None and upper is not None:
        rows["Lower bound"] = lower
        rows["Upper bound"] = upper
    return pd.DataFrame(rows)


def history_table(result: ForecastResult, days: int = 14) -> pd.DataFrame:
    """Most recent actual sales, newest last."""
    tail = result.history.tail(days)
    return pd.DataFrame(
        {
            "Date": [stamp.date().isoformat() for stamp in tail.index],
            "Units sold": round_units(tail.to_numpy()),
        }
    )


def total_predicted_units(result: ForecastResult) -> int:
    return int(round_units(result.point).sum())


def average_daily_units(result: ForecastResult) -> float:
    return float(np.mean(clamp_units(result.point)))


def baseline_comparison(result: ForecastResult) -> str:
    """Plain-language sentence comparing the model with the baseline."""
    evaluation = result.evaluation
    if evaluation is None or not evaluation.available:
        return "No backtest comparison is available for this history length."

    if evaluation.best == "model":
        verdict = (
            "The forecasting model beat the seasonal baseline on the held-out "
            f"{evaluation.holdout_days} days."
        )
    elif evaluation.best == "baseline":
        verdict = (
            "The seasonal baseline beat the forecasting model on the held-out "
            f"{evaluation.holdout_days} days."
        )
    else:
        verdict = (
            "The forecasting model and the seasonal baseline had the same error "
            f"on the held-out {evaluation.holdout_days} days."
        )
    return verdict


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #


def build_forecast_chart(result: ForecastResult, history_days: int = 90) -> go.Figure:
    """History, forecast and uncertainty band in one view.

    Line style and legend text (not colour alone) distinguish the series.
    """
    history = result.history.tail(history_days)
    history_dates = [stamp.date() for stamp in history.index]
    forecast_dates = list(result.forecast_dates)

    figure = go.Figure()

    if result.has_bounds:
        lower = round_units(result.lower)
        upper = round_units(result.upper)
        figure.add_trace(
            go.Scatter(
                x=forecast_dates + forecast_dates[::-1],
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=BAND_COLOR,
                line={"width": 0},
                name="Approximate 80% range",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=history_dates,
            y=round_units(history.to_numpy()),
            mode="lines",
            name="Actual units sold",
            line={"color": HISTORY_COLOR, "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y} units<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=round_units(result.point),
            mode="lines+markers",
            name="Forecast units (dashed)",
            line={"color": FORECAST_COLOR, "width": 2, "dash": "dash"},
            marker={"size": 6, "color": FORECAST_COLOR},
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y} units<extra></extra>",
        )
    )

    if forecast_dates:
        figure.add_vline(
            x=forecast_dates[0],
            line={"color": "#6b7280", "width": 1, "dash": "dot"},
        )
        figure.add_annotation(
            x=forecast_dates[0],
            y=1.02,
            yref="paper",
            text="Forecast starts",
            showarrow=False,
            font={"size": 11, "color": "#4b5563"},
            xanchor="left",
        )

    figure.update_layout(
        title=f"{result.product}: last {len(history)} days and {result.horizon}-day forecast",
        xaxis_title="Date",
        yaxis_title="Units sold",
        template="plotly_white",
        height=460,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(rangemode="tozero")
    return figure


def chart_summary_text(result: ForecastResult) -> str:
    """Text alternative describing the chart's key encodings."""
    parts = [
        f"The chart plots {result.history.tail(90).size} days of actual units sold "
        f"as a solid line, then {result.horizon} forecast days as a dashed line.",
    ]
    if result.has_bounds:
        parts.append(
            "A shaded band marks the approximate 80% range around each forecast."
        )
    else:
        parts.append("No uncertainty band is shown for this run.")
    parts.append("The same values appear in the daily forecast table below.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def build_export(result: ForecastResult) -> bytes:
    """CSV bytes for the planning file (FR-11)."""
    predicted = round_units(result.point)
    lower = _round_optional(result.lower)
    upper = _round_optional(result.upper)
    generated_at = result.generated_at.isoformat(timespec="seconds")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for index, day in enumerate(result.forecast_dates):
        writer.writerow(
            [
                result.product,
                day.isoformat(),
                int(predicted[index]),
                "" if lower is None else int(lower[index]),
                "" if upper is None else int(upper[index]),
                result.horizon,
                result.model_version,
                generated_at,
            ]
        )
    return buffer.getvalue().encode("utf-8")


def export_filename(result: ForecastResult) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "-"
        for character in result.product
    )
    slug = "-".join(part for part in slug.split("-") if part) or "product"
    return f"forecast-{slug}-{result.horizon}d.csv"


def export_matches_table(result: ForecastResult) -> bool:
    """Chart/table/export agreement guard used by tests and the UI."""
    table = forecast_table(result)
    predicted = round_units(result.point)
    if not np.array_equal(table["Predicted units"].to_numpy(), predicted):
        return False
    exported = _parse_export(build_export(result))
    if exported["predicted_units"].isna().any():
        return False
    return exported["predicted_units"].astype("int64").tolist() == predicted.tolist()


def _parse_export(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload), dtype=str)


def export_row_count(result: ForecastResult) -> int:
    return len(result.forecast_dates)


def describe_run_settings(result: ForecastResult) -> list[tuple[str, str]]:
    """Ordered label/value pairs for the run settings panel."""
    settings = result.settings
    used = (
        "Gradient boosting model"
        if result.model_used == "hist_gradient_boosting"
        else "Seasonal baseline"
    )
    return [
        ("Product", result.product),
        ("Forecast horizon", f"{result.horizon} days"),
        ("Model used", used),
        ("Model version", str(result.model_version)),
        (
            "History used",
            f"{settings.get('history_days', 0)} days "
            f"({settings.get('history_start')} to {settings.get('history_end')})",
        ),
        ("Evaluation holdout", f"{settings.get('holdout_days', 0)} days"),
        ("Random seed", str(settings.get("random_state"))),
        ("Uncertainty", result.interval_label if result.has_bounds else "not available"),
        (
            "Generated at",
            result.generated_at.isoformat(timespec="seconds"),
        ),
    ]


LIMITATIONS: tuple[str, ...] = (
    "This is decision support, not an order. You stay responsible for what you buy.",
    "Forecasts assume the recent pattern continues. Promotions, holidays, weather, "
    "price changes, closures and stockouts are not modelled.",
    "The uncertainty range is an approximate 80% interval built from recent "
    "backtest errors. Future demand can fall outside it.",
    "Missing calendar days are treated as zero sales only because you confirmed "
    "that interpretation.",
    "Brand-new products with no sales history cannot be forecast.",
    "Uploaded data stays in this browser session only and is discarded on reset "
    "or when the session expires.",
)
