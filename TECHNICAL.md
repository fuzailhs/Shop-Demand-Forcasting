# Technical Requirements — Shop Demand Forecasting MVP

**Input:** `PRD.md` and `FUNCTIONAL.md` · **Target runtime:** Python 3.12
**Stack:** Streamlit · pandas · scikit-learn · Plotly · pytest · Docker (optional)

Constraints fixed by the PRD: one uploaded CSV (max 20 MB), one active session,
no permanent user-data storage, lag-feature regression with a seasonal-naive
fallback, no paid APIs or external AI services.

---

## 1. System context and component responsibilities

```
Browser
  │  uploads CSV bytes, renders results
  ▼
app.py            Streamlit UI: page flow, controls, accessible messages,
  │               session state, orchestration
  ├── data.py     file checks, schema validation, type conversion,
  │               duplicate detection, daily series creation
  ├── features.py lag and calendar features (no future information)
  ├── model.py    time split, baseline, regressor, metrics,
  │               interval estimation, recursive forecast
  └── views.py    Plotly figure, result tables, CSV bytes, metric copy
```

| Module | Responsibilities |
|---|---|
| `app.py` | Page flow, user controls, accessible messages, session state, orchestration, logging |
| `data.py` | File checks, schema validation, type conversion, duplicate detection, daily series creation |
| `features.py` | Lag and calendar features without using future information |
| `model.py` | Time-based split, baseline, regression model, metrics, interval estimation, recursive forecast |
| `views.py` | Plotly figure, result table, metric explanations, CSV byte generation |

Dependency direction is one-way: `app → views → model → features → data`. No
module imports Streamlit except `app.py`, which keeps the whole pipeline unit
testable.

## 2. Data contract, validation and missing-date policy

1. Read the uploaded CSV from an in-memory byte buffer. Reject files over 20 MB
   before parsing (enforced both in the browser and in `data.py`).
2. Normalise header whitespace and letter case. Require the three logical fields
   `date`, `product`, `units_sold`. Additional columns are ignored **with a
   warning that names them**.
3. Parse dates strictly as ISO (`YYYY-MM-DD`, plus `YYYY-MM-DDTHH:MM[:SS]` and
   `YYYY-MM-DD HH:MM:SS`, all normalised to a date). Day-first formats such as
   `03/04/2026` are rejected because the intent is unrecoverable. Trim product
   values. Parse `units_sold` as a non-negative integer.
4. Reject duplicate `date` + `product` pairs rather than silently aggregating
   them.
5. For the selected product, sort by date and reindex to a complete daily
   calendar. Apply zero-filling **only after the user confirms** that a gap means
   zero sales; otherwise raise `MissingDatesError`.
6. Require 90 daily observations. Recommend 365; warn below 180; warn when more
   than 50% of days are zero.

Error reporting contract: every blocking issue carries `field` and `action`, so
the UI can always name the affected field and the next corrective step. Up to
five spreadsheet row numbers are reported per problem class.

## 3. Feature engineering, training, prediction, uncertainty

### Features

| Group | Features |
|---|---|
| Lags | 1, 7, 14, 28 days |
| Rolling means | prior 7 days, prior 28 days |
| Calendar | day of week, month |

Every rolling window is built from `series.shift(1)`, so a feature for day *d*
uses only observations on or before *d − 1*. The first complete feature row
appears after 28 days of history.

### Models

| Role | Model |
|---|---|
| Primary | `HistGradientBoostingRegressor` |
| Baseline | Seasonal naive — the value from 7 days earlier |
| Fallback | Baseline, whenever training or prediction raises |

Fixed `random_state=42`, `learning_rate=0.06`, `max_iter=300`,
`min_samples_leaf=5`, `l2_regularization=1.0`.

### Training and holdout

Expanding-history features are used throughout. The most recent 28 days are
reserved for backtesting when at least 120 days exist; below that the evaluation
section states why metrics are unavailable.

The training matrix and the recursive prediction path call **the same
`make_features` function**, so a scripted drift between them is impossible by
construction.

### Prediction

Predict one day at a time. Append each prediction to the working series so later
steps can use prior predicted lags. Clamp predictions and bounds to zero or
higher; round display and export values to whole units (half-up).

### Uncertainty

Collect backtest residuals (`actual − predicted`) over the holdout, then offset
each point prediction by the empirical 10th and 90th percentiles. The result is
labelled an **approximate 80% interval**, never a guarantee. Ordering is enforced
(`lower ≤ point ≤ upper`) after clamping. With fewer than 7 residuals, bounds are
omitted and a reason is shown.

### Evaluation metrics

| Metric | Definition | Use |
|---|---|---|
| MAE | `mean(|actual − predicted|)` | Typical miss in units; primary readable metric |
| WAPE | `sum(|error|) / sum(actual)` | Scale-relative comparison; `None` when actual totals zero |
| Baseline delta | `MAE_model − MAE_baseline` | Negative means the model improved on the baseline |

The baseline score is always displayed, including when the model wins.

## 4. Module structure and key interfaces

```
validate_sales_csv(file_bytes: bytes, filename: str) -> ValidationResult
products_summary(frame) -> list[ProductSummary]
build_daily_series(frame, product: str, fill_missing: bool) -> DailySeries
make_features(series, forecast_origin: date | None = None) -> DataFrame
next_feature_row(series) -> tuple[Timestamp, Series]
evaluate_models(series, config: ModelConfig) -> EvaluationResult
forecast(series, horizon: int, config, evaluation=None, product=...) -> ForecastResult
build_export(result: ForecastResult) -> bytes
```

Dataclasses: `ValidationIssue`, `ValidationResult`, `ProductSummary`,
`DailySeries`, `ModelConfig`, `EvaluationResult`, `ForecastResult`.

`ModelConfig` is a frozen dataclass, so every knob that affects a run travels
together and can be echoed back in the run settings panel.

## 5. State management and data lifecycle

| Key | Lifetime |
|---|---|
| `file_bytes`, `file_name`, `file_hash` | Until reset or session expiry |
| `validation` | Derived from `file_bytes`; replaced on each upload |
| `gap_confirmed` | Cleared when the product changes |
| `product`, `horizon` | Until reset |
| `evaluation`, `result` | Cleared whenever an input changes |

Rules:

- A model, evaluation or result object only exists for the current input
  configuration. Changing the product or horizon clears it, so a stale export is
  impossible.
- `Start over` requires confirmation and clears every key above.
- Nothing is written to disk, to a database, or to `localStorage`.

## 6. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | Valid file up to 100,000 rows: validation + forecast under 10 s. Measured p95 over 20 runs — see `scripts/benchmark.py`. Current result: **p95 3.92 s**. |
| Privacy | Uploaded and derived data stay in process memory and are removed on reset or session expiry. No raw rows and no product names are logged. |
| Security | Enforce file size and type server-side, parse as data only, render user values through Streamlit (escaped), pin every dependency in `requirements.lock.txt`. |
| Reliability | A model exception falls back to the seasonal-naive result with a user-readable warning. The app never exports a partial or stale run. |
| Accessibility | Keyboard-operable controls, visible focus, programmatic labels, text alternatives to chart encodings, no colour-only meaning. |
| Observability | One log line per validation and per forecast, containing status, duration in ms, row/history counts and model name — never raw data. |
| Reproducibility | Same input, lock file, configuration and seed produce the same point forecast within floating-point tolerance. |

## 7. Test strategy

| Layer | Coverage | File |
|---|---|---|
| Unit | Schema checks, strict date parsing, duplicate detection, zero-fill confirmation, lag/rolling leakage, MAE/WAPE, rounding, export columns | `tests/test_data.py`, `tests/test_features.py`, `tests/test_model.py`, `tests/test_views.py` |
| Integration | Valid CSV happy path, every blocking error class, short history, all-zero series, baseline fallback, reset, no stale export | `tests/test_data.py`, `tests/test_model.py` |
| Model | Synthetic weekly pattern reproduced exactly, time split preserved, feature rows use only prior dates, intervals ordered | `tests/test_model.py` |
| End-to-end | Template to export, chart/table agreement, visible limitations, reset, accessibility labels | `tests/test_app.py` (Streamlit `AppTest`) |

Leakage tests are explicit: perturbing a future observation must leave every
earlier feature row byte-identical, and perturbing a past observation must change
later features.

## 8. Local setup, configuration and deployment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.lock.txt
python -m pytest
streamlit run app.py
```

- Only non-secret configuration lives in version control
  (`.streamlit/config.toml`). No API keys are required.
- `scripts/make_sample_data.py` regenerates the committed synthetic data
  deterministically.
- `scripts/benchmark.py` measures the performance target.
- The Docker image runs as a non-root user, installs from the lock file, and
  exposes a health check against `/_stcore/health`.
- Deployment must use HTTPS. Disable permanent file persistence: the app writes
  nothing, so no volume is required.

## 9. Traceability

| Requirement | Implementation | Test |
|---|---|---|
| FR-01 Template | `app.render_template_section`, `data/template.csv` | `test_app.py::test_landing_page_*` |
| FR-02 Upload | `data.validate_sales_csv` envelope checks, `.streamlit/config.toml` | `test_data.py::test_oversized_*`, `test_non_csv_*` |
| FR-03 Schema | `data.validate_sales_csv` header normalisation | `test_data.py::test_missing_column_*`, `test_header_case_*` |
| FR-04 Field values | `data._parse_dates`, unit classifier, duplicate check | `test_data.py::test_invalid_date_*`, `test_negative_*`, `test_duplicate_*` |
| FR-05 Missing dates | `data.build_daily_series`, `data.MissingDatesError` | `test_data.py::test_gaps_block_*`, `test_app.py::test_gaps_block_*` |
| FR-06 Product choice | `data.products_summary`, `app.render_product_selection` | `test_data.py::test_products_are_alphabetical_*` |
| FR-07 Horizon | `app.render_horizon`, `model.forecast` | `test_app.py::test_horizon_*`, `test_model.py::test_forecast_returns_*` |
| FR-08 Generate | `model.forecast`, `model.recursive_forecast` | `test_model.py::test_forecast_never_negative`, `test_intervals_*` |
| FR-09 Evaluation | `model.evaluate_models` | `test_model.py::test_holdout_*`, `test_baseline_*` |
| FR-10 Display | `views.build_forecast_chart`, `views.forecast_table` | `test_views.py::test_chart_*`, `test_app.py::test_happy_path_*` |
| FR-11 Export | `views.build_export` | `test_views.py::test_export_*` |
| FR-12 Reset | `app.hard_reset`, `app.render_reset` | `test_app.py::test_confirmed_reset_*` |

## 10. Risks, trade-offs and definition of done

| Risk / trade-off | Decision |
|---|---|
| Recursive forecasting compounds error at 30 days | Accepted and disclosed: intervals come from a genuine 28-day recursive backtest, so their width already reflects that compounding. |
| A recursive baseline could be seen as unfair | The baseline repeats the last observed week, which is the standard seasonal-naive multi-step forecast. Both sides are out-of-sample. |
| Gradient boosting is heavier than a moving average | Justified by the 3.9 s p95; the baseline remains the documented fallback if training fails. |
| Strict ISO dates may reject real exports | Deliberate: a wrong date interpretation silently corrupts the series. The error names the format and the rows. |
| Streamlit cannot grey out individual select options | Ineligible products are excluded from the list and shown in an "Unavailable products" panel with the reason, which is more accessible than a dimmed option. |
| Interval coverage depends on history quality | Labelled "approximate 80%", with every limitation visible before download. |

**Definition of done**

- [x] Full happy path and all validation tests pass (108 tests).
- [x] Every FR-01 to FR-12 has at least one passing acceptance test.
- [x] Evaluation uses a time-based holdout and compares with the baseline.
- [x] The user sees units, uncertainty, assumptions and limitations in plain
      language.
- [x] Invalid or risky inputs fail safely and explain the next action.
- [x] Uploaded data is not retained after reset or session expiry.
- [x] Setup, run, test and deployment instructions work from a clean
      environment (verified against `requirements.lock.txt`).
