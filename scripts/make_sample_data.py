"""Generate the deterministic synthetic dataset and CSV template.

Run from the project root:

    python scripts/make_sample_data.py

The output is committed so the demo and the tests use identical bytes.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

START = date(2025, 10, 25)
DAYS = 365
SEED = 7

# name, base units, yearly fractional trend, weekday multipliers (Mon..Sun), noise sd, skip first N days
PRODUCTS = [
    ("Whole Wheat Bread", 42.0, 0.08, (1.10, 1.06, 1.02, 1.00, 1.12, 1.35, 1.28), 4.5, 0),
    ("Sourdough Loaf", 21.0, 0.06, (1.05, 1.00, 0.98, 0.96, 1.15, 1.40, 1.25), 3.0, 0),
    ("Butter Croissant", 58.0, -0.04, (0.95, 0.92, 0.90, 0.94, 1.10, 1.45, 1.38), 5.5, 0),
    ("Cinnamon Bun", 26.0, 0.03, (1.00, 0.98, 0.96, 0.98, 1.08, 1.30, 1.22), 3.8, 0),
    ("Apple Pie", 14.0, 0.02, (0.90, 0.88, 0.92, 1.00, 1.20, 1.55, 1.42), 2.8, 0),
    ("Chocolate Cookie", 70.0, 0.12, (1.02, 1.00, 1.01, 1.03, 1.10, 1.20, 1.15), 6.5, 0),
    ("Seasonal Fruit Tart", 9.0, 0.0, (0.80, 0.82, 0.85, 0.95, 1.25, 1.60, 1.50), 2.4, 320),
]

# Calendar gaps so the "missing days" confirmation (FR-05) is demonstrable.
GAPS = {
    "Cinnamon Bun": [41, 42, 43, 120, 201],
}

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def build_rows() -> list[tuple[str, str, int]]:
    rng = np.random.default_rng(SEED)
    rows: list[tuple[str, str, int]] = []

    for name, base, trend, weekdays, noise_sd, skip_first in PRODUCTS:
        gaps = set(GAPS.get(name, []))
        for offset in range(skip_first, DAYS):
            if offset in gaps:
                continue
            day = START + timedelta(days=offset)
            divisor = max(1, DAYS - skip_first)
            progress = (offset - skip_first) / divisor
            growth = max(0.35, 1.0 + trend * progress)
            weekly = weekdays[day.weekday()]
            monthly = 1.0 + 0.08 * np.sin(2 * np.pi * (day.month - 1) / 12 + 0.6)
            expected = base * growth * weekly * monthly
            observed = int(round(float(rng.normal(expected, noise_sd))))
            rows.append((day.isoformat(), name, max(0, observed)))

    rows.sort(key=lambda row: (row[0], row[1]))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    with (OUT_DIR / "sample_sales.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "product", "units_sold"])
        writer.writerows(rows)

    with (OUT_DIR / "template.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "product", "units_sold"])
        writer.writerow(["2026-01-15", "Whole Wheat Bread", "18"])

    print(f"Wrote {len(rows):,} rows for {len(PRODUCTS)} products to {OUT_DIR}")


if __name__ == "__main__":
    main()
