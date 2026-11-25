# Kestrel returns-risk

Pre-dispatch return-risk model for Kestrel Home Appliances. Scores an order before it
leaves the warehouse and recommends a confirmation call when the probability exceeds the
policy break-even (~11.2%).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: put the 8 data-pack files in ./data  (see data/README.md)
# or: export KESTREL_DATA_DIR=/path/to/pack

python -m src.train      # needs data; writes models/model.joblib
python -m src.predict    # needs data; writes predictions.csv (2,096 rows)
python -m src.evaluate   # needs data; writes reports/*
pytest -q
python -m src.serve      # http://0.0.0.0:8000/  (no data pack required)
```

`models/model.joblib` is committed so `python -m src.serve` works with no training step and
**no data pack on disk**: `/`, `/health`, `/sample`, and `/score` all run from the model
bundle plus `web/sample_order.json`. Holdout numbers in `reports/` come from a model trained
only on orders before `2026-04-01`; `predictions.csv` is from a refit on all cleaned
training rows (10,504).

## No API key, no LLM, no network

Scoring is a local sklearn logistic regression. There is **no API key**, **no LLM call**, and
**no outbound network request** at prediction time. The only runtime dependency for the demo
service is the committed `models/model.joblib` (plus the small `web/` assets). Each `/score`
costs nothing per prediction beyond ordinary CPU.

## Model choice

**No LLM scoring** - no API access in this environment, not reproducible, and not
defensible for a returns-risk model used in ops.

On a **time-based holdout** (train before 2026-04-01, validate April-June 2026; never a
random split):

| Model | Role |
|-------|------|
| **Logistic regression** (one-hot categoricals) | **Selected** - best holdout AUC / PR-AUC / Brier / Rs at p* |
| HistGradientBoosting (depth 3) | Tree competitor; close but behind on this split |
| Single decision tree | Illustration of rules only - not the deliverable |

Exact numbers are in `reports/metrics.json` and `reports/eval_report.md`.

## Data

The raw client pack (`train.csv`, `test_unlabelled.csv`, `customers.csv`, `products.csv`,
and the three spec documents) is **excluded from the repository** because ops-policy section 10
forbids publishing customer and operational data. Drop the eight files into `./data` (or set
`KESTREL_DATA_DIR`) only when you need to retrain or regenerate reports.

`predictions.csv` is committed on purpose: it contains **no customer attributes**, only an
`order_id` and a probability score.

## Data hygiene (before modelling)

1. **Dedupe** 651 `partner_feed` re-imports - keep the `crm` copy -> 10,504 rows.
2. **October 2025** `order_value_inr` inflated 100x by the new payment gateway - divide by 100.
3. **Leakage:** `last_service_event_type == REVERSE_PICKUP` and `pickup_scheduled_at` are
   post-return artefacts (ops-policy section 7). Dropped. Ablation in the eval report.
4. **Service-state mismatch:** train is as-of-export-day; test is at-dispatch
   (`INSTALL_BOOKED` unseen in train). Column excluded.
5. **Pincode `000000`** treated as missing (`pin_missing`), not a real pin.

## Operating threshold

From ops-policy section 4 / 7 (not Ritu's Rs 600):

`p* = C_CALL / (CALL_PREVENT x C_RETURN) = 45 / (0.35 x 1150) ≈ 0.1118`

Margin for hold analysis uses **MARGIN_PCT = 0.25 as an assumption** (not in the pack);
sensitivity at 0.15 / 0.25 / 0.40 is in the report. Prefer the confirmation call over a
blind hold - the pack measures call prevention, not hold prevention.

## Honest limitations

- Best holdout accuracy is ~89.6%; majority class ("predict nothing returns") is already ~88.5%.
  **95% accuracy is unreachable.**
- Delivery-note free text has almost no signal; only `note_missing` is kept.
- Shield members return more often and are high-LTV - calling/holding them has a service-desk cost.
- Holdout metrics are from a fit on data before 2026-04-01; `predictions.csv` is a full-data refit.

## Layout

```
src/           train, predict, evaluate, serve, features, economics
web/           single-page ops UI + sample_order.json (no CDN)
models/        model.joblib
reports/       metrics.json, eval_report.md, data_audit.md
predictions.csv
```
