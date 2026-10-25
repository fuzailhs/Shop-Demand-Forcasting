"""CSV validation and daily-series construction.

Implements the data contract from the technical requirements:

* reject files over 20 MB before parsing
* normalise header whitespace and require the three logical fields
* parse dates strictly, trim product values, parse units as non-negative integers
* reject duplicate date/product pairs instead of silently aggregating
* reindex the selected product onto a complete daily calendar, zero-filling only
  after the user confirms that gaps mean zero sales
* require 90 daily observations, warn below 180, recommend 365

This module deliberately has no Streamlit imports so it can be unit tested.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MAX_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = (".csv",)

REQUIRED_COLUMNS: tuple[str, ...] = ("date", "product", "units_sold")

MIN_OBSERVATIONS = 90
WARN_BELOW_OBSERVATIONS = 180
RECOMMENDED_OBSERVATIONS = 365
MOSTLY_ZERO_SHARE = 0.5

ERROR = "error"
WARNING = "warning"

SAMPLE_ROW_LIMIT = 5

# Unambiguous ISO formats only. Day-first formats such as 03/04/2026 are
# rejected on purpose because the meaning cannot be recovered safely.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)

_INT_RE = re.compile(r"^\+?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?\d*\.\d+$")
_NEGATIVE_RE = re.compile(r"^-\s*\d+(?:\.\d+)?$")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _normalize_header(name: object) -> str:
    """Lower-case, strip and collapse internal whitespace in a column name."""
    text = str(name).replace("\ufeff", "")
    return " ".join(text.strip().lower().split())


def _row_labels(labels, limit: int = SAMPLE_ROW_LIMIT) -> str:
    """Render 0-based dataframe indexes as 1-based spreadsheet row numbers."""
    numbers = [int(i) + 2 for i in labels]  # +1 for header, +1 for 1-based rows
    if not numbers:
        return "no rows"
    shown = ", ".join(str(n) for n in numbers[:limit])
    if len(numbers) == 1:
        return f"row {shown}"
    suffix = "" if len(numbers) <= limit else f" (+{len(numbers) - limit} more)"
    return f"rows {shown}{suffix}"


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse ISO dates strictly. Anything else becomes NaT."""
    text = raw.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    for fmt in _DATE_FORMATS:
        pending = parsed.isna() & text.ne("")
        if not pending.any():
            break
        attempt = pd.to_datetime(text[pending], format=fmt, errors="coerce")
        parsed.loc[pending] = attempt
    return parsed.dt.normalize()


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationIssue:
    """One blocking error or non-blocking warning.

    Every issue names the affected field and the next corrective action so the
    UI can meet the "100% of blocking errors identify the field and the next
    action" success metric.
    """

    severity: str
    code: str
    message: str
    field: str | None = None
    action: str | None = None
    count: int | None = None
    rows: str | None = None

    @property
    def is_blocking(self) -> bool:
        return self.severity == ERROR

    def as_text(self) -> str:
        parts = [self.message]
        if self.rows:
            parts.append(f"Affected {self.rows}.")
        if self.action:
            parts.append(self.action)
        return " ".join(parts)


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    frame: pd.DataFrame | None = None
    row_count: int = 0
    product_count: int = 0
    date_min: date | None = None
    date_max: date | None = None
    filename: str | None = None
    size_bytes: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]


@dataclass(frozen=True)
class ProductSummary:
    """Eligibility of one product at selection time (FR-06)."""

    name: str
    observed_days: int
    span_days: int
    start: date
    end: date
    eligible: bool
    reason: str | None = None


@dataclass
class DailySeries:
    """A selected product on a complete daily calendar."""

    product: str
    series: pd.Series
    observed_days: int
    missing_dates: list[date]
    start: date
    end: date
    zero_share: float

    @property
    def length(self) -> int:
        return int(len(self.series))


class DataContractError(ValueError):
    """Base class for recoverable data problems raised by the data service."""


class UnknownProductError(DataContractError):
    pass


class InsufficientHistoryError(DataContractError):
    pass


class MissingDatesError(DataContractError):
    """Raised when a product has calendar gaps the user has not confirmed."""

    def __init__(self, product: str, missing_dates: list[date]) -> None:
        self.product = product
        self.missing_dates = list(missing_dates)
        super().__init__(
            f"{product} is missing {len(self.missing_dates)} calendar day(s)."
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _failed(
    filename: str | None,
    size_bytes: int,
    issues: list[ValidationIssue],
    row_count: int = 0,
) -> ValidationResult:
    return ValidationResult(
        ok=False,
        issues=issues,
        frame=None,
        row_count=row_count,
        filename=filename,
        size_bytes=size_bytes,
    )


def _check_envelope(file_bytes: bytes, filename: str) -> ValidationIssue | None:
    """File-level guards that must run before parsing."""
    lower_name = (filename or "").strip().lower()
    if lower_name and not lower_name.endswith(ALLOWED_SUFFIXES):
        return ValidationIssue(
            severity=ERROR,
            code="unsupported_file_type",
            message=f"'{filename}' is not a CSV file.",
            field="file",
            action="Upload a file that ends in .csv.",
        )
    if len(file_bytes) > MAX_FILE_BYTES:
        megabytes = len(file_bytes) / (1024 * 1024)
        return ValidationIssue(
            severity=ERROR,
            code="file_too_large",
            message=f"The file is {megabytes:.1f} MB, above the 20 MB limit.",
            field="file",
            action="Upload a smaller file or split the history into one file per product.",
        )
    if not file_bytes.strip():
        return ValidationIssue(
            severity=ERROR,
            code="empty_file",
            message="The file is empty.",
            field="file",
            action="Download the template and add at least one row of sales.",
        )
    return None


def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(file_bytes),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
    )


def validate_sales_csv(
    file_bytes: bytes, filename: str = "upload.csv"
) -> ValidationResult:
    """Validate an uploaded CSV against the input contract.

    Returns ``ok=False`` with one blocking issue per distinct problem class.
    ``frame`` is only populated when the file is safe to model.
    """
    size_bytes = len(file_bytes)
    envelope_issue = _check_envelope(file_bytes, filename)
    if envelope_issue is not None:
        return _failed(filename, size_bytes, [envelope_issue])

    try:
        raw = _read_csv(file_bytes)
    except Exception:
        return _failed(
            filename,
            size_bytes,
            [
                ValidationIssue(
                    severity=ERROR,
                    code="unreadable_csv",
                    message="The file could not be read as a CSV with a header row.",
                    field="file",
                    action="Re-save the file as UTF-8 CSV with a header row and try again.",
                )
            ],
        )

    issues: list[ValidationIssue] = []

    # --- header ---------------------------------------------------------- #
    received = [_normalize_header(column) for column in raw.columns]
    rename: dict[str, str] = {}
    duplicates_in_header: set[str] = set()
    for original, normalized in zip(raw.columns, received):
        if normalized in REQUIRED_COLUMNS:
            if normalized in rename.values():
                duplicates_in_header.add(normalized)
            rename[original] = normalized

    missing = [column for column in REQUIRED_COLUMNS if column not in rename.values()]
    if missing or duplicates_in_header:
        expected = ", ".join(REQUIRED_COLUMNS)
        got = ", ".join(column for column in (str(c) for c in raw.columns)) or "none"
        detail = (
            f"Expected columns: {expected}. Received columns: {got}."
        )
        if duplicates_in_header:
            detail = (
                f"Columns {sorted(duplicates_in_header)} appear more than once. "
                + detail
            )
        return _failed(
            filename,
            size_bytes,
            [
                ValidationIssue(
                    severity=ERROR,
                    code="missing_columns",
                    message=f"Missing required column(s): {', '.join(missing)}. {detail}"
                    if missing
                    else f"Ambiguous header. {detail}",
                    field=", ".join(missing) if missing else "header",
                    action=(
                        "Rename your headers to exactly date, product, units_sold "
                        "and upload again."
                    ),
                )
            ],
            row_count=int(len(raw)),
        )

    frame = raw.rename(columns=rename)
    extra_columns = [
        str(original)
        for original in raw.columns
        if rename.get(original) is None
    ]

    if len(frame) == 0:
        return _failed(
            filename,
            size_bytes,
            [
                ValidationIssue(
                    severity=ERROR,
                    code="no_data_rows",
                    message="The file has a header row but no data rows.",
                    field="rows",
                    action="Add at least 90 days of daily sales and upload again.",
                )
            ],
            row_count=0,
        )

    # --- field values ---------------------------------------------------- #
    dates = _parse_dates(frame["date"])
    products = frame["product"].astype(str).str.strip()
    units_text = frame["units_sold"].astype(str).str.strip()

    units = pd.Series(np.nan, index=frame.index, dtype="float64")
    negative_rows: list[int] = []
    decimal_rows: list[int] = []
    non_numeric_rows: list[int] = []

    # Fast path: the overwhelming majority of real files are clean integers.
    # Vectorised numeric conversion keeps a 100,000-row file well inside the
    # 10 second budget; the exact per-row classifier only runs on failure.
    numeric = pd.to_numeric(units_text, errors="coerce")
    has_decimal_point = np.char.find(
        units_text.to_numpy().astype(str), "."
    ) >= 0
    is_clean = bool(
        numeric.notna().all()
        and not has_decimal_point.any()
        and (numeric >= 0).all()
        and np.all(np.mod(numeric.to_numpy(), 1) == 0)
    )

    if is_clean:
        units = numeric
    else:
        for idx, value in units_text.items():
            if _INT_RE.match(value):
                units.at[idx] = float(int(value))
            elif _NEGATIVE_RE.match(value):
                negative_rows.append(idx)
            elif _DECIMAL_RE.match(value):
                decimal_rows.append(idx)
            else:
                non_numeric_rows.append(idx)

    invalid_date_rows = list(dates.index[dates.isna()])
    blank_product_rows = list(products.index[products.eq("")])

    if invalid_date_rows:
        issues.append(
            ValidationIssue(
                severity=ERROR,
                code="invalid_date",
                message=(
                    f"{len(invalid_date_rows)} row(s) have a date that is not "
                    f"YYYY-MM-DD."
                ),
                field="date",
                action="Rewrite those dates as YYYY-MM-DD, for example 2026-01-15.",
                count=len(invalid_date_rows),
                rows=_row_labels(invalid_date_rows),
            )
        )
    if blank_product_rows:
        issues.append(
            ValidationIssue(
                severity=ERROR,
                code="blank_product",
                message=f"{len(blank_product_rows)} row(s) have an empty product name.",
                field="product",
                action="Fill in the product name on every row.",
                count=len(blank_product_rows),
                rows=_row_labels(blank_product_rows),
            )
        )
    if negative_rows:
        issues.append(
            ValidationIssue(
                severity=ERROR,
                code="negative_units",
                message=f"{len(negative_rows)} row(s) have a negative units_sold value.",
                field="units_sold",
                action="Replace negative quantities with 0 or the correct non-negative count.",
                count=len(negative_rows),
                rows=_row_labels(negative_rows),
            )
        )
    if decimal_rows:
        issues.append(
            ValidationIssue(
                severity=ERROR,
                code="decimal_units",
                message=f"{len(decimal_rows)} row(s) have a fractional units_sold value.",
                field="units_sold",
                action="Round every quantity to a whole number of units.",
                count=len(decimal_rows),
                rows=_row_labels(decimal_rows),
            )
        )
    if non_numeric_rows:
        issues.append(
            ValidationIssue(
                severity=ERROR,
                code="non_numeric_units",
                message=(
                    f"{len(non_numeric_rows)} row(s) have units_sold that is not a "
                    f"whole number."
                ),
                field="units_sold",
                action="Enter a whole number such as 18, or 0 for a day with no sales.",
                count=len(non_numeric_rows),
                rows=_row_labels(non_numeric_rows),
            )
        )

    if issues:
        return _failed(filename, size_bytes, issues, row_count=int(len(frame)))

    # --- duplicates ------------------------------------------------------ #
    cleaned = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "product": products.to_numpy(),
            "units_sold": units.to_numpy().astype("int64"),
        }
    )
    duplicate_mask = cleaned.duplicated(subset=["date", "product"], keep="first")
    if duplicate_mask.any():
        duplicate_rows = list(cleaned.index[duplicate_mask])
        return _failed(
            filename,
            size_bytes,
            [
                ValidationIssue(
                    severity=ERROR,
                    code="duplicate_rows",
                    message=(
                        f"{len(duplicate_rows)} duplicate date/product row(s) found. "
                        "Duplicates are rejected rather than summed."
                    ),
                    field="date, product",
                    action=(
                        "Keep one row per product per calendar day and upload again."
                    ),
                    count=len(duplicate_rows),
                    rows=_row_labels(duplicate_rows),
                )
            ],
            row_count=int(len(cleaned)),
        )

    cleaned = cleaned.sort_values(["date", "product"]).reset_index(drop=True)
    cleaned["units_sold"] = cleaned["units_sold"].astype("int64")

    if extra_columns:
        issues.append(
            ValidationIssue(
                severity=WARNING,
                code="extra_columns_ignored",
                message=(
                    "Ignored extra column(s): "
                    + ", ".join(extra_columns)
                    + ". Only date, product and units_sold are used."
                ),
                field=", ".join(extra_columns),
                action="No action needed unless one of these columns was meant to be a required field.",
            )
        )

    return ValidationResult(
        ok=True,
        issues=issues,
        frame=cleaned,
        row_count=int(len(cleaned)),
        product_count=int(cleaned["product"].nunique()),
        date_min=cleaned["date"].min().date(),
        date_max=cleaned["date"].max().date(),
        filename=filename,
        size_bytes=size_bytes,
    )


# --------------------------------------------------------------------------- #
# Product catalogue
# --------------------------------------------------------------------------- #


def products_summary(frame: pd.DataFrame) -> list[ProductSummary]:
    """Return products alphabetically with their available history length.

    Products with fewer than ``MIN_OBSERVATIONS`` observed days are marked
    ineligible with a readable reason (FR-06).
    """
    summaries: list[ProductSummary] = []
    for name, group in frame.groupby("product", sort=False):
        observed = int(len(group))
        start = group["date"].min().date()
        end = group["date"].max().date()
        span = (end - start).days + 1
        eligible = observed >= MIN_OBSERVATIONS
        reason = None
        if not eligible:
            reason = (
                f"Only {observed} day(s) of history; {MIN_OBSERVATIONS} are required."
            )
        summaries.append(
            ProductSummary(
                name=str(name),
                observed_days=observed,
                span_days=span,
                start=start,
                end=end,
                eligible=eligible,
                reason=reason,
            )
        )
    summaries.sort(key=lambda item: item.name.lower())
    return summaries


def product_label(summary: ProductSummary) -> str:
    """Human label used by the product selector."""
    return f"{summary.name} — {summary.observed_days} days of history"


# --------------------------------------------------------------------------- #
# Daily series
# --------------------------------------------------------------------------- #


def build_daily_series(
    frame: pd.DataFrame, product: str, fill_missing: bool
) -> DailySeries:
    """Build a complete daily calendar for one product.

    Raises ``MissingDatesError`` when the series has gaps and ``fill_missing``
    is false: the model must not run until the user confirms that a gap means
    zero sales.
    """
    subset = frame.loc[frame["product"] == product]
    if subset.empty:
        raise UnknownProductError(f"{product!r} is not in the uploaded file.")

    series = (
        subset.set_index("date")["units_sold"].astype("float64").sort_index()
    )
    start = series.index[0]
    end = series.index[-1]
    observed = int(len(series))

    if observed < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"{product} has {observed} day(s) of history; "
            f"{MIN_OBSERVATIONS} are required."
        )

    calendar = pd.date_range(start, end, freq="D")
    missing = [stamp.date() for stamp in calendar.difference(series.index)]
    if missing and not fill_missing:
        raise MissingDatesError(product, missing)

    series = series.reindex(calendar, fill_value=0.0)
    series.index = pd.DatetimeIndex(calendar, name="date")
    zero_share = float((series == 0).mean())

    return DailySeries(
        product=product,
        series=series,
        observed_days=observed,
        missing_dates=missing,
        start=start.date(),
        end=end.date(),
        zero_share=zero_share,
    )


def series_warnings(daily: DailySeries) -> list[str]:
    """Plain-language warnings shown next to the results (FR-10)."""
    warnings: list[str] = []
    if daily.observed_days < WARN_BELOW_OBSERVATIONS:
        warnings.append(
            f"Only {daily.observed_days} days of history. "
            f"Short histories produce less reliable seasonal patterns; "
            f"{RECOMMENDED_OBSERVATIONS} days is preferred."
        )
    if daily.zero_share >= MOSTLY_ZERO_SHARE:
        warnings.append(
            f"{daily.zero_share:.0%} of days recorded zero sales. "
            "Check whether the product was out of stock on those days."
        )
    if daily.missing_dates:
        warnings.append(
            f"{len(daily.missing_dates)} missing calendar day(s) were treated "
            "as zero sales because you confirmed that interpretation."
        )
    return warnings
