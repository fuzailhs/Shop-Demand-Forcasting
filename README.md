# Shop Demand Forecasting

A small web app that turns one daily sales CSV into a **7-day or 30-day
unit-demand forecast** for a single product, with an uncertainty range and an
honest accuracy check against a simple baseline.

Built as a beginner-friendly MVP in the shape the sample project pack asks for:
Python 3.12 + Streamlit, pandas for data handling, scikit-learn for the model,
Plotly for the chart, pytest for tests, Docker for optional packaging.

> **Decision support, not an order.** The forecast helps a store manager judge
> the next purchase. It does not place orders, and it cannot see promotions,
> holidays, closures or stockouts.

---

## Quick start

```powershell
cd shop-demand-forecasting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt

python -m pytest          # 108 tests
streamlit run app.py      # http://localhost:8501
```

Nothing else is needed: no API keys, no database, no external services.

### Fastest way to see the whole flow

1. Open the app and select **Try the sample dataset**.
2. Leave the product on *Apple Pie* and the horizon on *7 days*.
3. Select **Generate forecast**.
4. Review the chart, the daily table, the accuracy check and the limitations.
5. Select **Download forecast CSV**.
6. Select **Start over**, then confirm.

To see the missing-days confirmation, switch the product to **Cinnamon Bun** —
the sample file deliberately leaves five calendar gaps in that series.

---

## What it does

| Step | What happens |
|---|---|
| 1 · Data | Download a template, or load the bundled synthetic dataset |
| 2 · Upload | One CSV, checked for type and size before parsing |
| 3 · Validate | Columns, ISO dates, whole-number units, duplicate date/product rows |
| 4 · Choose | Product list sorted alphabetically, each with its history length |
| 5 · Gaps | Missing calendar days must be confirmed as zero sales, or the run stays blocked |
| 6 · Horizon | 7 days or 30 days |
| 7 · Forecast | Chart, daily table, total units, uncertainty band, holdout accuracy |
| 8 · Export | CSV with `product, forecast_date, predicted_units, lower_bound, upper_bound, horizon, model_version, generated_at` |

### Input contract

| Column | Type | Example | Rule |
|---|---|---|---|
| `date` | Date | `2026-01-15` | ISO `YYYY-MM-DD`, one calendar day per row per product |
| `product` | Text | `Whole Wheat Bread` | Required, trimmed, non-empty |
| `units_sold` | Whole number | `18` | Required, zero or greater, no decimals |

Extra columns are ignored with a visible warning. Duplicate `date` + `product`
rows are rejected rather than silently summed.

---

## How the forecast works

**Features** (all built from days strictly before the day being predicted):

| Group | Values |
|---|---|
| Lags | `lag_1`, `lag_7`, `lag_14`, `lag_28` |
| Rolling means | prior 7 days, prior 28 days — both shifted one day |
| Calendar | day of week, month |

**Model.** `HistGradientBoostingRegressor` with a fixed seed. The baseline is
**seasonal naive** — repeat the last observed week. If training or prediction
raises, the baseline is served instead with a visible warning.

**Training and evaluation.** The latest 28 days are withheld. The model is
trained on everything before them and forecasts them recursively, exactly as it
will forecast the future. Evaluation needs at least 120 days; below that the app
says why metrics are unavailable.

**Prediction.** One day at a time: each prediction is appended to the working
series so later steps can use earlier predictions as lags. Values are clamped at
zero and rounded to whole units for display and export.

**Uncertainty.** Backtest residuals give empirical 10th and 90th percentiles,
applied as offsets around each point prediction. The result is labelled an
**approximate 80% interval** — not a guarantee. Fewer than 7 residuals means no
band, with a stated reason.

**Metrics.**

| Metric | Definition | Reading |
|---|---|---|
| MAE | `mean(|actual − predicted|)` | Typical miss, in units |
| WAPE | `sum(|error|) / sum(actual)` | Share of demand mispredicted |
| Baseline delta | `MAE_model − MAE_baseline` | Negative means the model improved |

The baseline score is always shown, including when the model wins. On the bundled
sample the model beats the baseline for five of six products and loses for
*Chocolate Cookie* — that honest comparison is the point.

---

## Architecture

```
app.py         Streamlit page flow, session state, accessible messages
data.py        CSV validation and daily-series construction
features.py    Leakage-safe lag and calendar features
model.py       Baseline, regressor, metrics, intervals, recursive forecast
views.py       Plotly chart, tables, CSV bytes, metric copy
```

Only `app.py` imports Streamlit; everything else is plain Python and unit
testable. There is no build step, no bundler and no backend.

**Session state.** Uploaded bytes, the validation result, the chosen product and
horizon, the cached evaluation and the forecast result. Changing the product or
horizon clears the result, so a stale run can never be exported. `Start over`
requires confirmation and clears everything.

---

## Testing

```powershell
python -m pytest                                   # everything
python -m pytest tests/test_app.py                 # end-to-end only
python -m pytest tests/test_features.py -q         # leakage audit
```

| Layer | What it proves |
|---|---|
| Unit | Schema checks, strict date parsing, duplicate detection, zero-fill confirmation, MAE/WAPE, rounding, export columns |
| Integration | Happy path, every blocking error class, short history, all-zero series, baseline fallback, reset, no stale export |
| Model | A noiseless weekly cycle is reproduced exactly; the time split is preserved; intervals are ordered |
| End-to-end | Real Streamlit app driven by `AppTest`: template → sample → forecast → export → reset, plus the gap flow |

Two tests exist purely to police leakage: changing a **future** observation must
leave every earlier feature row byte-identical, and changing a **past**
observation must change later features.

### Performance

```powershell
python scripts/benchmark.py --runs 20
```

Measured on a 100,000-row file across 100 products (validated plus forecasted
end to end):

```
Runs: 20
Median: 3.76 s
p95:    3.92 s
Max:    4.00 s
Target: under 10.00 s -> PASS
```

---

## Docker

```powershell
docker build -t shop-demand-forecasting .
docker run --rm -p 8501:8501 shop-demand-forecasting
```

Non-root user, dependencies installed from `requirements.lock.txt`, no writable
volume required, and a health check against `/_stcore/health`. Deploy behind
HTTPS.

---

## Privacy

- Everything runs in the Streamlit process. No POS connection, no accounts, no
  payments, no analytics (`gatherUsageStats = false`).
- Uploaded bytes and derived series live in the session and are discarded on
  reset or session expiry.
- Logs contain status, duration, row counts and model name — never raw rows and
  never product names.

---

## Limitations

- Forecasts assume the recent pattern continues. Promotions, holidays, weather,
  price changes, closures and stockouts are not modelled.
- The uncertainty band is an approximate 80% interval from recent backtest
  errors; real demand can fall outside it.
- Missing days become zero sales only because the manager confirmed that.
- Products with fewer than 90 days of history cannot be forecast.
- One store, one file, one product at a time. No price or inventory
  optimisation.

### Assumptions worth revisiting with a real shop

- Does a missing day mean zero sales, a closure, a stockout or lost data?
- Should the order use expected demand, the upper interval, shelf capacity or
  supplier case size?
- Which promotions, holidays, prices and stock levels could be supplied as
  future inputs?

---

## Project documents

| File | Contents |
|---|---|
| [`PRD.md`](PRD.md) | Problem, user, goals, journey, features, data contract, metrics, risks, release criteria |
| [`FUNCTIONAL.md`](FUNCTIONAL.md) | FR-01 to FR-12 with triggers, flows, error behaviour and Given/When/Then acceptance criteria |
| [`TECHNICAL.md`](TECHNICAL.md) | Architecture, data pipeline, model specification, NFRs, test strategy, traceability |
| `data/template.csv` | The downloadable template |
| `data/sample_sales.csv` | Deterministic synthetic year for seven products |
| `scripts/make_sample_data.py` | Regenerates the sample data |
| `scripts/benchmark.py` | Reproduces the performance measurement |

**Project principle:** a trustworthy baseline with clear limits is more useful
than a complex model that cannot be explained or tested.
