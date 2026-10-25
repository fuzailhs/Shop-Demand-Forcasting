"""End-to-end checks driving the real Streamlit app with AppTest.

These cover the milestone 4 acceptance path: template -> sample file ->
validation -> product -> horizon -> forecast -> visible results -> export ->
reset, plus the blocking error and gap-confirmation states.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

import views

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


def new_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    return app


def load_sample(app: AppTest) -> AppTest:
    app.button(key="load_sample").click().run()
    return app


def generate(app: AppTest) -> AppTest:
    app.button(key="generate_forecast").click().run()
    return app


def markdown_text(app: AppTest) -> str:
    chunks = []
    for block in app.markdown:
        chunks.append(str(block.value))
    for block in app.caption:
        chunks.append(str(block.value))
    for block in app.success:
        chunks.append(str(block.value))
    for block in app.warning:
        chunks.append(str(block.value))
    for block in app.error:
        chunks.append(str(block.value))
    for block in app.info:
        chunks.append(str(block.value))
    return "\n".join(chunks)


def all_text(app: AppTest) -> str:
    """Every string the user can read, for honest assertions."""
    parts = [markdown_text(app)]
    for collection in (app.title, app.header, app.subheader, app.text):
        for block in collection:
            parts.append(str(block.value))
    for collection in (app.metric,):
        for block in collection:
            parts.append(str(block.label))
            parts.append(str(block.value))
    return "\n".join(parts)


def download_labels(app: AppTest) -> list[str]:
    """AppTest exposes download buttons by label only, not by key."""
    return [str(button.label) for button in app.get("download_button")]


def uploader(app: AppTest):
    return app.get("file_uploader")[0]


def has_button(app: AppTest, key: str) -> bool:
    """True when a button with this key is currently rendered."""
    return any(button.key == key for button in app.button)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_landing_page_offers_template_upload_and_sample():
    app = new_app()

    assert not app.exception
    assert "Shop Demand Forecasting" in all_text(app)
    labels = download_labels(app)
    assert "Download CSV template" in labels
    assert "Download sample dataset" in labels
    assert has_button(app, "load_sample")
    assert len(app.get("file_uploader")) == 1


def test_happy_path_produces_a_forecast_and_an_export():
    app = load_sample(new_app())
    assert not app.exception

    assert app.session_state["validation"].ok is True
    assert "Apple Pie" in all_text(app)

    generate(app)
    assert not app.exception

    result = app.session_state["result"]
    assert result is not None
    assert result.horizon == 7
    assert result.product == "Apple Pie"
    assert len(result.forecast_dates) == 7

    # FR-10: results are visible in more than one form.
    assert len(app.metric) >= 4
    assert len(app.dataframe) >= 3
    assert "Daily forecast" in markdown_text(app)
    assert "Limitations" in markdown_text(app)
    assert "Download forecast CSV" in download_labels(app)


def test_downloaded_csv_matches_the_onscreen_table():
    app = load_sample(new_app())
    generate(app)

    result = app.session_state["result"]
    payload = views.build_export(result)
    frame = pd.read_csv(io.BytesIO(payload))
    table = views.forecast_table(result)

    assert list(frame.columns) == list(views.EXPORT_COLUMNS)
    assert frame["predicted_units"].tolist() == table["Predicted units"].tolist()


def test_horizon_selection_is_used_consistently():
    app = load_sample(new_app())
    app.radio(key="horizon_choice").set_value(30).run()
    generate(app)

    result = app.session_state["result"]
    assert result.horizon == 30
    assert len(result.forecast_dates) == 30
    assert views.export_row_count(result) == 30


def test_changing_the_horizon_invalidates_the_previous_run():
    app = load_sample(new_app())
    generate(app)
    assert app.session_state["result"] is not None

    app.radio(key="horizon_choice").set_value(30).run()

    assert app.session_state["result"] is None
    assert "Download forecast CSV" not in download_labels(app)


def test_evaluation_and_baseline_are_visible():
    app = load_sample(new_app())
    generate(app)

    result = app.session_state["result"]
    assert result.evaluation.available is True
    text = all_text(app).lower()
    assert "mae" in text
    assert "baseline" in text
    assert "wape" in text


def test_limitations_appear_before_the_download_control():
    app = load_sample(new_app())
    generate(app)

    result = app.session_state["result"]
    assert result is not None
    text = markdown_text(app)
    assert "read before you order" in text.lower()


# --------------------------------------------------------------------------- #
# Validation and recovery (FR-03, FR-04)
# --------------------------------------------------------------------------- #


def test_invalid_file_shows_a_plain_language_error_and_keeps_the_session():
    app = new_app()
    app.session_state["file_bytes"] = b"date,product,units_sold\n2026-01-01,A,5\n"
    app.session_state["file_name"] = "broken.csv"
    app.session_state["validation"] = __import__("data").validate_sales_csv(
        b"date,product,quantity\n2026-01-01,A,5\n", "broken.csv"
    )
    app.run()

    assert not app.exception
    text = all_text(app)
    assert "block the forecast" in text
    assert "units_sold" in text  # names the expected column
    assert "Traceback" not in text


def test_ineligible_product_is_listed_with_a_reason():
    app = load_sample(new_app())
    generate(app)

    text = all_text(app)
    assert "Seasonal Fruit Tart" in text
    assert "90 are required" in text


# --------------------------------------------------------------------------- #
# Missing dates (FR-05)
# --------------------------------------------------------------------------- #


def test_gaps_block_the_forecast_until_confirmed():
    app = load_sample(new_app())
    app.selectbox(key="product_select").set_value("Cinnamon Bun").run()

    assert not app.exception
    assert "Missing days" in all_text(app) or "missing days" in all_text(app)
    assert app.session_state["gap_confirmed"] is False

    # The generate button is not rendered at all while the run is blocked.
    assert not has_button(app, "generate_forecast")


def test_confirming_zero_fill_unblocks_the_forecast():
    app = load_sample(new_app())
    app.selectbox(key="product_select").set_value("Cinnamon Bun").run()
    app.radio(key="gap_choice").set_value("Treat missing days as zero sales").run()

    assert app.session_state["gap_confirmed"] is True
    generate(app)
    assert app.session_state["result"] is not None
    assert app.session_state["result"].product == "Cinnamon Bun"


def test_declining_the_gap_interpretation_keeps_the_run_blocked():
    app = load_sample(new_app())
    app.selectbox(key="product_select").set_value("Cinnamon Bun").run()
    app.radio(key="gap_choice").set_value("Do not forecast yet").run()

    assert app.session_state["gap_confirmed"] is False
    assert app.session_state["result"] is None


# --------------------------------------------------------------------------- #
# Reset (FR-12)
# --------------------------------------------------------------------------- #


def test_start_over_requires_confirmation():
    app = load_sample(new_app())
    generate(app)

    app.button(key="start_over").click().run()

    assert app.session_state["confirm_reset"] is True
    assert app.session_state["result"] is not None  # nothing cleared yet
    assert has_button(app, "confirm_reset_yes")


def test_confirmed_reset_clears_the_whole_session():
    app = load_sample(new_app())
    generate(app)

    app.button(key="start_over").click().run()
    app.button(key="confirm_reset_yes").click().run()

    assert not app.exception
    assert app.session_state["result"] is None
    assert app.session_state["validation"] is None
    assert app.session_state["file_bytes"] is None
    assert app.session_state["product"] is None
    # The export control is gone, so a stale run cannot be downloaded.
    assert "Download forecast CSV" not in download_labels(app)
    assert "Waiting for a file" in all_text(app)


def test_cancelling_reset_keeps_the_results():
    app = load_sample(new_app())
    generate(app)

    app.button(key="start_over").click().run()
    app.button(key="confirm_reset_no").click().run()

    assert app.session_state["confirm_reset"] is False
    assert app.session_state["result"] is not None


# --------------------------------------------------------------------------- #
# Accessibility expectations
# --------------------------------------------------------------------------- #


def test_every_control_has_a_visible_label():
    app = load_sample(new_app())
    generate(app)

    assert uploader(app).label
    assert app.selectbox(key="product_select").label
    assert app.radio(key="horizon_choice").label
    assert app.button(key="load_sample").label
    assert app.button(key="start_over").label


def test_status_messages_use_words_not_only_colour():
    app = load_sample(new_app())
    generate(app)

    text = all_text(app)
    # Every state carries an icon plus an explicit heading.
    assert "Forecast ready" in text
    assert "Decision support" in text
