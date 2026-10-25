"""Shared fixtures and CSV builders for the test suite."""

from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def csv_bytes(rows, header=("date", "product", "units_sold")) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    if header:
        writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def daily_rows(
    product: str = "Whole Wheat Bread",
    days: int = 200,
    start: date = date(2025, 1, 1),
    value=None,
    skip: set[int] | None = None,
) -> list[tuple[str, str, str]]:
    """One row per day; ``value`` may be a callable of the day offset."""
    skip = skip or set()
    rows: list[tuple[str, str, str]] = []
    for offset in range(days):
        if offset in skip:
            continue
        day = start + timedelta(days=offset)
        units = value(offset) if callable(value) else (value if value is not None else 10)
        rows.append((day.isoformat(), product, str(units)))
    return rows


def weekly_pattern(offset: int, base: int = 20, amplitude: int = 10) -> int:
    """Deterministic 7-day cycle: 1.0x..1.5x of base."""
    day_of_week = offset % 7
    return base + (amplitude if day_of_week >= 5 else 0) + day_of_week


@pytest.fixture
def valid_csv() -> bytes:
    return csv_bytes(daily_rows(days=200))


@pytest.fixture
def sample_csv_bytes() -> bytes:
    return (DATA_DIR / "sample_sales.csv").read_bytes()


@pytest.fixture
def template_csv_bytes() -> bytes:
    return (DATA_DIR / "template.csv").read_bytes()
