"""Measure the PRD performance target: validation + forecast under 10 s.

Generates a 100,000-row file once, then times the same path the app uses
(validate -> build daily series -> evaluate -> forecast) over N runs and reports
the median and p95.

    python scripts/benchmark.py            # 20 runs (the PRD sample size)
    python scripts/benchmark.py --runs 5
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import data  # noqa: E402
from model import ModelConfig, evaluate_models, forecast  # noqa: E402

PRODUCTS = 100
DAYS = 1000
START = date(2023, 1, 1)
TARGET_ROWS = 100_000


def build_large_csv() -> tuple[bytes, str]:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["date", "product", "units_sold"])
    rows = 0
    for product_index in range(PRODUCTS):
        name = f"Product {product_index:03d}"
        for offset in range(DAYS):
            writer.writerow(
                [
                    (START + timedelta(days=offset)).isoformat(),
                    name,
                    (product_index * 7 + offset * 3) % 40,
                ]
            )
            rows += 1
    return buffer.getvalue().encode("utf-8"), f"{rows:,} rows"


def one_run(payload: bytes, product: str) -> float:
    started = time.perf_counter()
    validation = data.validate_sales_csv(payload, "benchmark.csv")
    if not validation.ok:
        raise RuntimeError("benchmark fixture failed validation")
    daily = data.build_daily_series(validation.frame, product, fill_missing=True)
    evaluation = evaluate_models(daily.series, ModelConfig())
    forecast(daily.series, 30, ModelConfig(), evaluation=evaluation, product=product)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--product", default="Product 000")
    args = parser.parse_args()

    payload, description = build_large_csv()
    payload_mb = len(payload) / (1024 * 1024)
    print(f"Fixture: {description} ({payload_mb:.1f} MB) across {PRODUCTS} products")

    one_run(payload, args.product)  # warm caches, not measured
    durations = [one_run(payload, args.product) for _ in range(args.runs)]

    ordered = sorted(durations)
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
    median = statistics.median(durations)
    p95 = ordered[p95_index]

    print(f"Runs: {len(durations)}")
    print(f"Median: {median:.2f} s")
    print(f"p95:    {p95:.2f} s")
    print(f"Max:    {max(durations):.2f} s")
    print(f"Target: under 10.00 s -> {'PASS' if p95 < 10 else 'FAIL'}")
    return 0 if p95 < 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
