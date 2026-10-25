"""Shop Demand Forecasting — Streamlit entry point.

Milestone flow (build plan 4 · Experience):

    template -> upload -> validate -> confirm gaps -> pick product
    -> pick horizon -> forecast -> review -> export -> reset

Accessibility notes: every control has a visible label, status is always
communicated with words and an icon (never colour alone), and the chart values
are repeated in a table.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import streamlit as st

import data as data_service
import views
from model import ModelConfig, evaluate_models, forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("shop_demand_forecasting")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
TEMPLATE_PATH = DATA_DIR / "template.csv"
SAMPLE_PATH = DATA_DIR / "sample_sales.csv"

CONFIG = ModelConfig()
HORIZONS = (7, 30)

STATE_KEYS = (
    "file_bytes",
    "file_name",
    "file_hash",
    "validation",
    "gap_confirmed",
    "product",
    "horizon",
    "evaluation",
    "result",
    "run_error",
    "confirm_reset",
    "loaded_source",
)


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #


def init_state() -> None:
    defaults = {
        "file_bytes": None,
        "file_name": None,
        "file_hash": None,
        "validation": None,
        "gap_confirmed": False,
        "product": None,
        "horizon": HORIZONS[0],
        "evaluation": None,
        "result": None,
        "run_error": None,
        "confirm_reset": False,
        "loaded_source": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_run() -> None:
    """Drop cached results so a stale run can never be exported (FR-12)."""
    st.session_state.result = None
    st.session_state.evaluation = None
    st.session_state.run_error = None


def hard_reset() -> None:
    """Clear uploaded data, selections, cached model and results."""
    for key in STATE_KEYS:
        st.session_state.pop(key, None)
    for key in ("upload_csv", "product_select", "horizon_choice"):
        st.session_state.pop(key, None)


def on_product_change() -> None:
    """Product changed: previous gap decision and run no longer apply."""
    clear_run()
    st.session_state.gap_confirmed = False
    # Drop the stale gap answer so a different product asks again.
    st.session_state.pop("gap_choice", None)


def on_horizon_change() -> None:
    """Horizon changed: the cached run is stale."""
    clear_run()


def request_reset() -> None:
    st.session_state.confirm_reset = True


def cancel_reset() -> None:
    st.session_state.confirm_reset = False


def load_sample_file() -> None:
    """Button callback: load the bundled synthetic dataset."""
    if SAMPLE_PATH.exists():
        load_file(SAMPLE_PATH.read_bytes(), "sample_sales.csv", "sample")


def handle_upload() -> None:
    """Uploader callback: validate the newly selected file."""
    uploaded = st.session_state.get("upload_csv")
    if uploaded is None:
        return
    load_file(uploaded.getvalue(), uploaded.name or "upload.csv", "upload")


def load_file(payload: bytes, filename: str, source: str) -> None:
    """Store the upload and validate it; reset everything downstream."""
    started = time.perf_counter()
    result = data_service.validate_sales_csv(payload, filename)
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "validation status=%s duration_ms=%.0f rows=%d issues=%d",
        "ok" if result.ok else "blocked",
        duration_ms,
        result.row_count,
        len(result.issues),
    )

    st.session_state.file_bytes = payload
    st.session_state.file_name = filename
    st.session_state.file_hash = hashlib.sha256(payload).hexdigest()
    st.session_state.validation = result
    st.session_state.loaded_source = source
    st.session_state.gap_confirmed = False
    st.session_state.product = None
    clear_run()


# --------------------------------------------------------------------------- #
# Small render helpers
# --------------------------------------------------------------------------- #


def status_line(kind: str, title: str, body: str) -> None:
    """Status message with an icon, a heading and words — never colour alone.

    Streamlit's status containers ship their own accessible icon, so the
    heading text carries the meaning.
    """
    renderers = {
        "ok": st.success,
        "warn": st.warning,
        "error": st.error,
        "info": st.info,
    }
    renderers[kind](f"**{title}**\n\n{body}")


def step(title: str) -> str:
    """Numbered section heading that counts only the sections actually shown."""
    st.session_state.run_step = st.session_state.get("run_step", 0) + 1
    label = f"{st.session_state.run_step} · {title}"
    st.subheader(label)
    return label


def reset_step_counter() -> None:
    st.session_state.run_step = 0


def render_header() -> None:
    st.title("Shop Demand Forecasting")
    st.caption(
        "Turn one daily sales CSV into a 7-day or 30-day unit-demand forecast "
        "for a single product, with an uncertainty range and a plain-language "
        "accuracy check."
    )
    status_line(
        "info",
        "Decision support, not an order",
        "These forecasts help you judge the next order. They do not place "
        "orders, and they cannot see promotions, holidays, closures or "
        "stockouts.",
    )


def render_template_section() -> None:
    step("Get your data ready")
    st.write(
        "Use three columns: **date** (YYYY-MM-DD), **product** (text) and "
        "**units_sold** (whole number, 0 or more). One row per product per day."
    )

    template_bytes = TEMPLATE_PATH.read_bytes() if TEMPLATE_PATH.exists() else (
        b"date,product,units_sold\n2026-01-15,Whole Wheat Bread,18\n"
    )

    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download CSV template",
            data=template_bytes,
            file_name="sales-template.csv",
            mime="text/csv",
            help="A CSV with the three required headers and one example row.",
            key="download_template",
        )
    with right:
        if SAMPLE_PATH.exists():
            st.download_button(
                "Download sample dataset",
                data=SAMPLE_PATH.read_bytes(),
                file_name="sample_sales.csv",
                mime="text/csv",
                help="A synthetic year of daily sales for six products.",
                key="download_sample",
            )

    if SAMPLE_PATH.exists():
        st.button(
            "Try the sample dataset",
            help="Loads the synthetic sample file without uploading anything.",
            key="load_sample",
            on_click=load_sample_file,
        )


def render_upload() -> None:
    step("Upload sales history")
    st.file_uploader(
        "Upload daily sales CSV (max 20 MB)",
        type=["csv"],
        accept_multiple_files=False,
        help=(
            "Required columns: date, product, units_sold. "
            "Additional columns are ignored with a warning."
        ),
        key="upload_csv",
        on_change=handle_upload,
    )


def render_validation_issues(validation) -> None:
    errors = validation.errors
    warnings = validation.warnings

    if errors:
        lines = []
        for issue in errors:
            field = f"*Field: `{issue.field}`.* " if issue.field else ""
            action = f"**Next step:** {issue.action}" if issue.action else ""
            lines.append(f"- {issue.message} {field}{action}")
        status_line(
            "error",
            f"{len(errors)} problem(s) block the forecast",
            "Fix these in your CSV and upload it again. Your file stays loaded "
            "so you can compare after correcting it.\n\n" + "\n".join(lines),
        )

    if warnings:
        lines = [f"- {issue.message}" for issue in warnings]
        status_line(
            "warn",
            f"{len(warnings)} warning(s) — forecast can continue",
            "\n".join(lines),
        )


def render_data_summary(validation) -> None:
    step("Data check")
    columns = st.columns(3)
    columns[0].metric("Rows", f"{validation.row_count:,}")
    columns[1].metric("Products", validation.product_count)
    if validation.date_min and validation.date_max:
        span = (validation.date_max - validation.date_min).days + 1
        columns[2].metric("Days covered", f"{span:,}")
        st.caption(
            f"File: {validation.filename or 'upload.csv'} · Dates: "
            f"{validation.date_min.isoformat()} to "
            f"{validation.date_max.isoformat()}"
        )
    status_line(
        "ok",
        "Schema and values look valid",
        "Every required column is present, dates are ISO, units are whole "
        "numbers, and there are no duplicate date/product rows.",
    )


def render_product_selection(frame, summaries):
    step("Choose a product")
    eligible = [summary for summary in summaries if summary.eligible]
    unavailable = [summary for summary in summaries if not summary.eligible]

    if not eligible:
        status_line(
            "error",
            "No product has enough history",
            f"Every product needs at least {data_service.MIN_OBSERVATIONS} days "
            "of daily sales. Upload a longer history to continue.",
        )
        return None

    labels = {summary.name: data_service.product_label(summary) for summary in eligible}
    names = [summary.name for summary in eligible]

    current = st.session_state.product
    index = names.index(current) if current in names else 0

    selected = st.selectbox(
        "Product to forecast",
        options=names,
        index=index,
        format_func=lambda name: labels[name],
        on_change=on_product_change,
        help="Products are sorted alphabetically and show days of available history.",
        key="product_select",
    )
    st.session_state.product = selected

    if unavailable:
        with st.expander(f"Unavailable products ({len(unavailable)})"):
            st.write(
                f"These products are disabled because they have fewer than "
                f"{data_service.MIN_OBSERVATIONS} days of history."
            )
            for summary in unavailable:
                st.write(f"- **{summary.name}** — {summary.reason}")

    return selected


def render_gap_confirmation(daily):
    """FR-05: gaps must be confirmed before the model may run."""
    if not daily.missing_dates:
        return True

    step("Missing days need a decision")
    sample = ", ".join(day.isoformat() for day in daily.missing_dates[:5])
    extra = (
        ""
        if len(daily.missing_dates) <= 5
        else f" and {len(daily.missing_dates) - 5} more"
    )
    st.write(
        f"**{daily.product}** is missing {len(daily.missing_dates)} calendar "
        f"day(s) between {daily.start.isoformat()} and {daily.end.isoformat()}: "
        f"{sample}{extra}."
    )
    st.write(
        "A missing day could mean zero sales, a shop closure, a stockout, or "
        "lost data. Confirm the interpretation before forecasting."
    )

    choice = st.radio(
        "How should missing days be treated?",
        options=["Treat missing days as zero sales", "Do not forecast yet"],
        index=0 if st.session_state.gap_confirmed else 1,
        help=(
            "Choosing 'Do not forecast yet' keeps the run blocked until you "
            "upload corrected data."
        ),
        key="gap_choice",
    )

    confirmed = choice == "Treat missing days as zero sales"
    st.session_state.gap_confirmed = confirmed
    if not confirmed:
        status_line(
            "warn",
            "Forecast is paused",
            "Upload a complete daily history, or choose to treat the missing "
            "days as zero sales.",
        )
    return confirmed


def render_horizon() -> int:
    step("Choose a horizon")
    horizon = st.radio(
        "Forecast horizon",
        options=list(HORIZONS),
        index=list(HORIZONS).index(st.session_state.horizon)
        if st.session_state.horizon in HORIZONS
        else 0,
        format_func=lambda value: f"{value} days",
        horizontal=True,
        on_change=on_horizon_change,
        help="The horizon is used in the chart, the table and the exported file.",
        key="horizon_choice",
    )
    st.session_state.horizon = int(horizon)
    return int(horizon)


def run_forecast(daily, horizon: int, warnings: list[str]) -> None:
    started = time.perf_counter()
    try:
        evaluation = evaluate_models(daily.series, CONFIG)
        result = forecast(
            daily.series,
            horizon,
            CONFIG,
            evaluation=evaluation,
            product=daily.product,
        )
    except Exception:  # noqa: BLE001 - never surface a traceback to the user
        logger.exception("forecast status=error horizon=%d", horizon)
        clear_run()
        st.session_state.run_error = (
            "The forecast could not be produced from this history. Try a longer "
            "history, or start over and upload the file again."
        )
        return

    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "forecast status=ok duration_ms=%.0f horizon=%d history_days=%d model=%s",
        duration_ms,
        horizon,
        len(daily.series),
        result.model_used,
    )
    result.warnings = list(warnings) + list(result.warnings)
    st.session_state.evaluation = evaluation
    st.session_state.result = result
    st.session_state.run_error = None


def render_results(result) -> None:
    step("Your forecast")
    status_line(
        "ok",
        f"Forecast ready: {result.horizon} days from "
        f"{result.forecast_dates[0].isoformat()}",
        f"Product: {result.product}. Total predicted demand is "
        f"{views.total_predicted_units(result):,} units.",
    )
    if views.export_matches_table(result):
        st.caption("Chart, table and export values agree.")

    metrics = st.columns(4)
    metrics[0].metric("Total predicted units", f"{views.total_predicted_units(result):,}")
    metrics[1].metric("Average per day", f"{views.average_daily_units(result):.1f}")
    metrics[2].metric("History used", f"{result.settings['history_days']:,} days")
    metrics[3].metric(
        "Model used",
        "Gradient boosting"
        if result.model_used == "hist_gradient_boosting"
        else "Seasonal baseline",
    )

    st.plotly_chart(
        views.build_forecast_chart(result),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(views.chart_summary_text(result))

    st.markdown("#### Daily forecast")
    st.dataframe(
        views.forecast_table(result),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Units are rounded to whole numbers and never go below zero. The lower "
        "and upper bounds are the approximate 80% interval."
    )

    st.markdown("#### Recent actual sales")
    st.dataframe(
        views.history_table(result),
        use_container_width=True,
        hide_index=True,
    )


def render_evaluation(result) -> None:
    st.markdown("#### How accurate is this forecast?")
    evaluation = result.evaluation

    if evaluation is None or not evaluation.available:
        reason = (
            evaluation.reason
            if evaluation is not None and evaluation.reason
            else "No backtest was available for this history."
        )
        status_line("warn", "Accuracy check not available", reason)
        return

    columns = st.columns(4)
    columns[0].metric(
        "Model MAE",
        f"{evaluation.mae_model:.2f} units"
        if evaluation.mae_model is not None
        else "n/a",
    )
    columns[1].metric(
        "Baseline MAE",
        f"{evaluation.mae_baseline:.2f} units"
        if evaluation.mae_baseline is not None
        else "n/a",
    )
    columns[2].metric(
        "Model WAPE",
        f"{evaluation.wape_model:.1%}" if evaluation.wape_model is not None else "n/a",
    )
    columns[3].metric(
        "Baseline WAPE",
        f"{evaluation.wape_baseline:.1%}"
        if evaluation.wape_baseline is not None
        else "n/a",
    )

    st.write(views.baseline_comparison(result))
    st.caption(
        f"MAE is the typical miss in units. WAPE is total error divided by total "
        f"actual units. Both are measured on the most recent "
        f"{evaluation.holdout_days} days, which the model never saw during "
        f"training. The model was trained on {evaluation.train_days} days."
    )
    if evaluation.fallback_reason:
        status_line(
            "warn",
            "Seasonal baseline was used",
            evaluation.fallback_reason,
        )


def render_settings(result) -> None:
    st.markdown("#### Run settings")
    settings = dict(views.describe_run_settings(result))
    st.dataframe(
        {"Setting": list(settings.keys()), "Value": list(settings.values())},
        use_container_width=True,
        hide_index=True,
    )


def render_warnings(result) -> None:
    if not result.warnings:
        return
    lines = "\n".join(f"- {warning}" for warning in result.warnings)
    status_line(
        "warn",
        f"{len(result.warnings)} warning(s) about this forecast",
        lines,
    )


def render_limitations() -> None:
    st.markdown("#### Limitations — read before you order")
    st.markdown("\n".join(f"- {item}" for item in views.LIMITATIONS))


def render_export(result) -> None:
    st.markdown("#### Download")
    st.download_button(
        "Download forecast CSV",
        data=views.build_export(result),
        file_name=views.export_filename(result),
        mime="text/csv",
        type="primary",
        help=(
            "Columns: product, forecast_date, predicted_units, lower_bound, "
            "upper_bound, horizon, model_version, generated_at."
        ),
        key="download_forecast",
    )


def render_reset() -> None:
    st.divider()
    if not st.session_state.confirm_reset:
        st.button(
            "Start over",
            key="start_over",
            on_click=request_reset,
            help="Clear the uploaded file, selections and results.",
        )
        return

    status_line(
        "warn",
        "Start over?",
        "This clears the uploaded file, your selections, the cached model and "
        "any results from this session.",
    )
    left, right = st.columns(2)
    left.button(
        "Yes, clear this session",
        key="confirm_reset_yes",
        type="primary",
        on_click=hard_reset,
    )
    right.button("Cancel", key="confirm_reset_no", on_click=cancel_reset)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(
        page_title="Shop Demand Forecasting",
        page_icon="📈",
        layout="wide",
    )
    init_state()
    reset_step_counter()
    render_header()
    render_template_section()
    render_upload()

    validation = st.session_state.validation
    if validation is None:
        status_line(
            "info",
            "Waiting for a file",
            "Upload a CSV, or select **Try the sample dataset** to see the full "
            "flow with synthetic data.",
        )
        render_reset()
        return

    if not validation.ok:
        step("Data check")
        render_validation_issues(validation)
        render_reset()
        return

    render_validation_issues(validation)
    render_data_summary(validation)

    frame = validation.frame
    if frame is None:
        render_reset()
        return

    summaries = data_service.products_summary(frame)
    product = render_product_selection(frame, summaries)
    if product is None:
        render_reset()
        return

    summaries_by_name = {summary.name: summary for summary in summaries}
    selected_summary = summaries_by_name[product]

    try:
        probe = data_service.build_daily_series(
            frame, product, fill_missing=st.session_state.gap_confirmed
        )
    except data_service.MissingDatesError:
        # Gaps exist and the user has not confirmed the interpretation yet.
        probe = data_service.build_daily_series(frame, product, fill_missing=True)
        if not render_gap_confirmation(probe):
            render_reset()
            return
    except data_service.DataContractError as exc:
        status_line("error", "This product cannot be forecast", str(exc))
        render_reset()
        return

    warnings = data_service.series_warnings(probe)
    for warning in warnings:
        st.caption(f"Data warning: {warning}")

    horizon = render_horizon()

    step("Generate the forecast")
    st.write(
        f"Forecast **{horizon} days** of daily unit demand for "
        f"**{product}** using {probe.length:,} days of daily history "
        f"({selected_summary.observed_days:,} observed)."
    )

    if st.button("Generate forecast", type="primary", key="generate_forecast"):
        clear_run()
        with st.spinner("Training the model and building the forecast…"):
            run_forecast(probe, horizon, warnings)

    if st.session_state.run_error:
        status_line("error", "Forecast failed", st.session_state.run_error)

    result = st.session_state.result
    if result is not None:
        render_warnings(result)
        render_results(result)
        render_evaluation(result)
        render_settings(result)
        render_limitations()
        render_export(result)

    render_reset()


main()
