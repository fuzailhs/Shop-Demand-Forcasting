"""Unit tests for the data contract (FR-01 to FR-06)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import data
from conftest import csv_bytes, daily_rows


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_valid_file_is_accepted_with_summary(valid_csv):
    result = data.validate_sales_csv(valid_csv, "sales.csv")

    assert result.ok is True
    assert result.row_count == 200
    assert result.product_count == 1
    assert result.date_min == date(2025, 1, 1)
    assert result.date_max == date(2025, 1, 1) + timedelta(days=199)
    assert result.frame is not None
    assert list(result.frame.columns) == ["date", "product", "units_sold"]
    assert str(result.frame["units_sold"].dtype) == "int64"


def test_header_case_and_whitespace_are_normalised():
    payload = csv_bytes(
        [("2026-01-01", "Bread", "4")],
        header=(" Date ", "PRODUCT", "units_sold "),
    )
    result = data.validate_sales_csv(payload, "sales.csv")

    assert result.ok is True
    assert result.frame is not None
    assert list(result.frame.columns) == ["date", "product", "units_sold"]


def test_product_values_are_trimmed(valid_csv):
    payload = csv_bytes(
        [(day, "  Whole Wheat Bread  ", units) for day, _, units in daily_rows(days=95)]
    )
    result = data.validate_sales_csv(payload, "sales.csv")

    assert result.ok is True
    assert result.frame["product"].unique().tolist() == ["Whole Wheat Bread"]


def test_extra_columns_warn_without_blocking():
    rows = [(day, product, units, "promo") for day, product, units in daily_rows(days=95)]
    payload = csv_bytes(rows, header=("date", "product", "units_sold", "promo_flag"))
    result = data.validate_sales_csv(payload, "sales.csv")

    assert result.ok is True
    assert len(result.warnings) == 1
    assert result.warnings[0].code == "extra_columns_ignored"
    assert "promo_flag" in result.warnings[0].message


# --------------------------------------------------------------------------- #
# Blocking errors
# --------------------------------------------------------------------------- #


def test_missing_column_names_expected_and_received():
    payload = csv_bytes(
        [("2026-01-01", "Bread", "4")], header=("date", "product", "quantity")
    )
    result = data.validate_sales_csv(payload, "sales.csv")

    assert result.ok is False
    assert result.frame is None
    issue = result.errors[0]
    assert issue.code == "missing_columns"
    assert "units_sold" in issue.message
    assert "quantity" in issue.message
    assert issue.field == "units_sold"


def test_misspelled_column_is_not_silently_accepted():
    payload = csv_bytes(
        [("2026-01-01", "Bread", "4")], header=("date", "prodct", "units_sold")
    )
    result = data.validate_sales_csv(payload, "sales.csv")

    assert result.ok is False
    assert result.errors[0].field == "product"


def test_invalid_date_reports_exact_row_numbers():
    rows = daily_rows(days=95)
    rows[4] = ("03/04/2026", "Whole Wheat Bread", "10")
    result = data.validate_sales_csv(csv_bytes(rows), "sales.csv")

    assert result.ok is False
    issue = next(i for i in result.errors if i.code == "invalid_date")
    assert issue.count == 1
    assert issue.rows == "row 6"  # header is row 1, so index 4 is row 6
    assert issue.field == "date"
    assert "YYYY-MM-DD" in issue.action


def test_blank_product_is_rejected():
    rows = daily_rows(days=95)
    rows[0] = (rows[0][0], "   ", "10")
    result = data.validate_sales_csv(csv_bytes(rows), "sales.csv")

    assert result.ok is False
    issue = next(i for i in result.errors if i.code == "blank_product")
    assert issue.rows == "row 2"


def test_negative_zero_and_text_units_are_reported_separately():
    rows = daily_rows(days=95)
    rows[1] = (rows[1][0], rows[1][1], "-4")
    rows[2] = (rows[2][0], rows[2][1], "2.5")
    rows[3] = (rows[3][0], rows[3][1], "many")
    result = data.validate_sales_csv(csv_bytes(rows), "sales.csv")

    assert result.ok is False
    codes = {issue.code for issue in result.errors}
    assert codes == {"negative_units", "decimal_units", "non_numeric_units"}


def test_zero_units_are_valid():
    result = data.validate_sales_csv(csv_bytes(daily_rows(days=95, value=0)), "sales.csv")

    assert result.ok is True


def test_duplicate_date_product_rows_are_rejected():
    rows = daily_rows(days=95)
    rows.append(rows[7])
    result = data.validate_sales_csv(csv_bytes(rows), "sales.csv")

    assert result.ok is False
    issue = result.errors[0]
    assert issue.code == "duplicate_rows"
    assert issue.count == 1
    assert issue.rows == "row 97"


def test_same_date_across_products_is_not_a_duplicate():
    rows = daily_rows(days=95, product="Apple Pie") + daily_rows(
        days=95, product="Bread"
    )
    result = data.validate_sales_csv(csv_bytes(rows), "sales.csv")

    assert result.ok is True
    assert result.product_count == 2


def test_oversized_file_is_rejected_before_parsing():
    payload = b"date,product,units_sold\n" + b"x" * (data.MAX_FILE_BYTES + 1)
    result = data.validate_sales_csv(payload, "huge.csv")

    assert result.ok is False
    assert result.errors[0].code == "file_too_large"
    assert result.errors[0].field == "file"


def test_non_csv_extension_is_rejected(valid_csv):
    result = data.validate_sales_csv(valid_csv, "sales.xlsx")

    assert result.ok is False
    assert result.errors[0].code == "unsupported_file_type"


def test_empty_file_is_rejected():
    result = data.validate_sales_csv(b"   ", "sales.csv")

    assert result.ok is False
    assert result.errors[0].code == "empty_file"


def test_header_only_file_is_rejected():
    result = data.validate_sales_csv(csv_bytes([]), "sales.csv")

    assert result.ok is False
    assert result.errors[0].code == "no_data_rows"


def test_unparseable_bytes_do_not_raise():
    result = data.validate_sales_csv(b"\x00\x00binary\x00", "sales.csv")

    assert result.ok is False
    assert result.errors[0].code in {"unreadable_csv", "missing_columns"}


@pytest.mark.parametrize(
    "builder",
    [
        lambda: csv_bytes([("2026-01-01", "A", "1")], header=("date", "product")),
        lambda: csv_bytes([("03/04/2026", "A", "1")]),
        lambda: csv_bytes([("2026-01-01", " ", "1")]),
        lambda: csv_bytes([("2026-01-01", "A", "-1")]),
        lambda: csv_bytes([("2026-01-01", "A", "1.5")]),
        lambda: csv_bytes([("2026-01-01", "A", "one")]),
        lambda: csv_bytes([("2026-01-01", "A", "1"), ("2026-01-01", "A", "2")]),
        lambda: b"",
    ],
)
def test_every_blocking_error_names_field_and_next_action(builder):
    """PRD metric: 100% of blocking errors identify the field and next action."""
    result = data.validate_sales_csv(builder(), "sales.csv")

    assert result.ok is False
    assert result.errors
    for issue in result.errors:
        assert issue.field, f"{issue.code} is missing a field"
        assert issue.action, f"{issue.code} is missing a next action"
        assert issue.message


# --------------------------------------------------------------------------- #
# Product catalogue (FR-06)
# --------------------------------------------------------------------------- #


def test_products_are_alphabetical_and_flag_short_history():
    rows = (
        daily_rows(days=120, product="zucchini loaf")
        + daily_rows(days=120, product="Apple Pie")
        + daily_rows(days=45, product="Mango Tart")
    )
    result = data.validate_sales_csv(csv_bytes(rows), "sales.csv")
    summaries = data.products_summary(result.frame)

    assert [item.name for item in summaries] == ["Apple Pie", "Mango Tart", "zucchini loaf"]

    eligible = {item.name: item.eligible for item in summaries}
    assert eligible["Apple Pie"] is True
    assert eligible["zucchini loaf"] is True
    assert eligible["Mango Tart"] is False

    short = next(item for item in summaries if item.name == "Mango Tart")
    assert short.observed_days == 45
    assert "90" in short.reason


def test_product_label_shows_history_length():
    result = data.validate_sales_csv(csv_bytes(daily_rows(days=120)), "sales.csv")
    summary = data.products_summary(result.frame)[0]

    assert data.product_label(summary) == "Whole Wheat Bread — 120 days of history"


# --------------------------------------------------------------------------- #
# Daily series (FR-05)
# --------------------------------------------------------------------------- #


def test_gaps_block_until_the_user_confirms():
    rows = daily_rows(days=120, skip={10, 11, 12})
    frame = data.validate_sales_csv(csv_bytes(rows), "sales.csv").frame

    with pytest.raises(data.MissingDatesError) as error:
        data.build_daily_series(frame, "Whole Wheat Bread", fill_missing=False)

    assert len(error.value.missing_dates) == 3
    assert error.value.missing_dates[0] == date(2025, 1, 11)


def test_confirmed_gaps_are_zero_filled_on_a_complete_calendar():
    rows = daily_rows(days=120, skip={10, 11, 12})
    frame = data.validate_sales_csv(csv_bytes(rows), "sales.csv").frame

    daily = data.build_daily_series(frame, "Whole Wheat Bread", fill_missing=True)

    assert daily.length == 120
    assert daily.observed_days == 117
    assert len(daily.missing_dates) == 3
    assert daily.series.loc["2025-01-11"] == 0.0
    # A gapless daily calendar is required by the feature layer.
    assert list(daily.series.index) == list(
        __import__("pandas").date_range("2025-01-01", periods=120, freq="D")
    )


def test_short_history_is_rejected():
    frame = data.validate_sales_csv(csv_bytes(daily_rows(days=89)), "sales.csv").frame

    with pytest.raises(data.InsufficientHistoryError):
        data.build_daily_series(frame, "Whole Wheat Bread", fill_missing=True)


def test_unknown_product_is_rejected(valid_csv):
    frame = data.validate_sales_csv(valid_csv, "sales.csv").frame

    with pytest.raises(data.UnknownProductError):
        data.build_daily_series(frame, "Not A Product", fill_missing=True)


def test_series_warnings_explain_short_and_sparse_history():
    rows = daily_rows(days=120, value=lambda offset: 0 if offset % 2 else 12)
    frame = data.validate_sales_csv(csv_bytes(rows), "sales.csv").frame
    daily = data.build_daily_series(frame, "Whole Wheat Bread", fill_missing=True)

    warnings = data.series_warnings(daily)

    joined = " ".join(warnings)
    assert "Only 120 days of history" in joined
    assert "zero sales" in joined


def test_series_warnings_quiet_for_a_full_year(valid_csv):
    frame = data.validate_sales_csv(
        csv_bytes(daily_rows(days=365, value=lambda offset: 5 + offset % 7)),
        "sales.csv",
    ).frame
    daily = data.build_daily_series(frame, "Whole Wheat Bread", fill_missing=True)

    assert data.series_warnings(daily) == []
