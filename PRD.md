# Product Requirements Document — Shop Demand Forecasting

**Status:** Approved for MVP · **Version:** 1.0 · **Owner:** Store manager (primary user)
**Build level:** Beginner-friendly MVP · **Estimated scope:** 1–2 weeks

---

## 1. Problem statement

A small independent shop orders inventory using intuition. Daily sales vary for
reasons the manager cannot easily see in a spreadsheet: weekends, a slow month,
a day the product was out of stock. The result is two opposite failures at once —
popular products run out, and slow products sit on the shelf.

Reviewing months of rows by hand is slow, and a single average hides the weekly
pattern that matters most when placing an order.

## 2. Product vision

A browser-based tool that turns one exported sales CSV into a short-term unit
forecast the manager can act on in under a minute, without needing to understand
machine learning.

| | |
|---|---|
| **Primary user** | Store manager or owner |
| **Core job to be done** | *"Before I place an order, help me estimate demand and see how uncertain that estimate is."* |
| **Business value** | Better stock decisions — fewer stockouts, less dead stock |

The app supports judgment. It does not place orders and it does not promise
exact sales.

## 3. Goals, non-goals and assumptions

### Goals

1. Accept a valid CSV and explain any data problem in plain language.
2. Produce a 7-day and 30-day daily forecast for one selected product.
3. Show history, forecast and lower/upper uncertainty bounds in one view.
4. Provide a downloadable forecast CSV for order planning.
5. Report forecast error against a baseline so the manager can judge trust.

### Non-goals

- Automatic replenishment, supplier ordering or inventory optimisation.
- Real-time point-of-sale integration, user accounts, or multi-store analytics.
- Guaranteed predictions, causal claims, or forecasts for brand-new products
  with no sales history.
- Price optimisation, promotions modelling, weather or holiday effects.

### Assumptions

> Assumptions are marked explicitly and must be revisited with a real shop.

- **A1.** The shop can export at least 90 days of daily unit sales; 365 days is
  preferred.
- **A2.** Dates use the ISO format `YYYY-MM-DD`; quantities are non-negative
  whole numbers.
- **A3.** Missing calendar days mean zero sales *only after the user confirms
  that interpretation*.
- **A4.** One store, one user, one uploaded file at a time.
- **A5.** Uploaded data stays in the browser session and is discarded on reset or
  session expiry.

## 4. End-to-end user journey

1. Open the app and download or view the CSV template.
2. Upload a file. The app validates columns, types, duplicate rows, date order
   and history length.
3. Resolve blocking errors; review warnings and confirm how missing dates should
   be treated.
4. Select a product and choose a 7-day or 30-day horizon.
5. Generate the forecast and review the chart, daily table, evaluation and
   limitations.
6. Download a CSV containing forecast date, predicted units, lower bound and
   upper bound.
7. Start over to clear the session.

## 5. MVP features, prioritised

### Must

| # | Feature |
|---|---------|
| M1 | Downloadable CSV template with the three required headers and one example row |
| M2 | Single CSV upload, size- and type-checked before parsing |
| M3 | Schema and value validation with actionable messages |
| M4 | Missing-date confirmation before any modelling |
| M5 | Product selector limited to products with sufficient history |
| M6 | 7-day and 30-day forecast horizons |
| M7 | History + forecast + uncertainty band in one chart |
| M8 | Daily forecast table with total predicted units |
| M9 | Forecast CSV download |
| M10 | Plain-language decision-support disclaimer |

### Should

| # | Feature |
|---|---------|
| S1 | Seasonal-naive baseline comparison |
| S2 | Holdout evaluation with MAE and WAPE |
| S3 | Recent-sales summary |
| S4 | Warning when history is short or mostly zero |

### Could

| # | Feature |
|---|---------|
| C1 | Holiday flag upload |
| C2 | Multiple product comparison |
| C3 | Saved forecast run |

## 6. Input data contract and validation rules

| Column | Type | Example | Rule |
|---|---|---|---|
| `date` | Date | `2026-01-15` | ISO `YYYY-MM-DD`. One calendar day per row per product. |
| `product` | Text | `Whole Wheat Bread` | Required; trimmed and non-empty. |
| `units_sold` | Whole number | `18` | Required; zero or greater, no decimals. |

| Rule | Severity | Behaviour |
|---|---|---|
| File larger than 20 MB | Blocking | Rejected before parsing |
| File is not `.csv` | Blocking | Rejected before parsing |
| Required column missing or misspelled | Blocking | Names expected and received columns |
| Header whitespace or letter case differs | Tolerated | Normalised automatically |
| Extra column present | Warning | Ignored and named |
| Date not ISO | Blocking | Reports count and row numbers |
| Product blank after trimming | Blocking | Reports count and row numbers |
| `units_sold` negative, fractional or non-numeric | Blocking | Reported separately, with row numbers |
| Duplicate `date` + `product` pair | Blocking | Rejected, never silently summed |
| Product has fewer than 90 observed days | Blocking at selection | Product listed as unavailable with the reason |
| Product has calendar gaps | Blocking until confirmed | User chooses "treat as zero sales" or uploads corrected data |
| Product has fewer than 180 days | Warning | Recommends 365 days |
| More than 50% of days are zero | Warning | Suggests checking for stockouts |

## 7. User-visible error states

| State | What the user sees | Next action offered |
|---|---|---|
| Empty start | "Waiting for a file" | Upload a CSV or load the sample |
| Unsupported or oversized file | Named limit and the actual value | Upload a smaller `.csv` |
| Missing column | Expected and received column lists | Rename headers and re-upload |
| Invalid field values | Count plus up to five row numbers per problem class | Fix the named rows |
| Duplicate rows | Count and sample row numbers | Keep one row per product per day |
| Missing dates | Count, date range and sample dates | Confirm zero-sales or re-upload |
| No eligible product | Minimum-history rule restated | Upload a longer history |
| Evaluation unavailable | The day-count requirement and the actual count | None needed — forecast still runs |
| Forecast failure | Plain-language message, no stack trace | Try a longer history or start over |
| Short or sparse history | Warning block above the results | Interpret results with caution |

Shared error behaviour: the uploaded file and any valid selections survive a
recoverable error, focus moves to the error summary, and a technical stack trace
is never shown.

## 8. Success metrics

| Metric | Target | How it is measured |
|---|---|---|
| Task completion | ≥ 4 of 5 test users upload the sample, create a forecast and export it unaided | Moderated session, no prompting |
| System response | < 10 s for a valid file of up to 100,000 rows | Timed p95 over 20 runs on the target deployment |
| Forecast evaluation | Model beats **or** is honestly explained against the seasonal-naive baseline | Holdout MAE and WAPE reported side by side |
| Data feedback | 100% of blocking errors name the affected field and the next action | Automated test over every error class |
| Chart/table/export agreement | 100% of runs | Values compared row by row |

## 9. Risks, mitigations and open questions

| Risk | Mitigation |
|---|---|
| Too little or poor-quality history | Block under 90 days; warn under 180; keep a baseline fallback |
| Users over-trust a point estimate | Show ranges, error metrics and a visible disclaimer |
| Promotions or closures distort demand | Document that external events are not modelled; allow gap review |
| User reads a missing day as a stockout | Force an explicit interpretation before modelling |
| Model is worse than the baseline | Report both scores and state which won |

**Dependencies:** a browser, Python 3.12 runtime, no external services and no
API keys.

**Open questions for a real shop**

- Does a missing day mean zero sales, a closure, lost data or a stockout?
- Should orders use expected demand, the upper interval, shelf capacity or
  supplier case size?
- Which promotions, holidays, prices and stock levels could be supplied as
  future inputs?

## 10. MVP release acceptance criteria

- [ ] The full happy path and all validation tests pass.
- [ ] No uploaded data is retained after reset or session expiry.
- [ ] Output is reproducible for the same input and configuration.
- [ ] Limitations are visible before the download control.
- [ ] Every blocking error names the field and the next action.
- [ ] The chart, the table and the exported CSV agree value for value.
