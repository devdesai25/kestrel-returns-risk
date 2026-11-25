# Service smoke test

Run with the data pack **absent** (`KESTREL_DATA_DIR=/tmp/absent`), the worst case for a reviewer
who has the code but not the client's data. Everything the model needs is inside the committed
bundle, so scoring is unaffected.

| Call | Status | Result |
|------|--------|--------|
| `GET /` | 200 | HTML page, 10129 bytes |
| `GET /health` | 200 | `"{\"status\": \"ok\", \"model_loaded\": true, \"model_version\": \"logit-v1\"}"` |
| `GET /sample` | 200 | one ready-to-post order record |
| `POST /score` (low risk) | 200 | score 0.0954, band Ship |
| `POST /score` (high risk) | 200 | score 0.9331, band Hold |
| `POST /score` (incomplete record) | 400 | JSON error, no traceback |
| `POST /score` (implausible value) | 200 | score 0.0311, band Ship, warning returned |

## Low-risk order (`/sample` verbatim)

```json
{
  "order_id": "KO2610504",
  "score": 0.095372,
  "band": "Ship",
  "threshold": 0.1118,
  "decision": "Low predicted risk - ship as usual.",
  "expected_saving_inr": 38.39,
  "expected_call_cost_inr": 45.0,
  "reasons": [
    {
      "direction": "down",
      "points": 0,
      "text": "Prepaid with a 7-day promised delivery, but this customer has returned 1 of 2 previous orders. The score is still below the call threshold."
    }
  ],
  "model_version": "logit-v1",
  "warnings": []
}
```

## High-risk order (robot vacuum, cash on delivery, 3 of 5 prior orders returned, 9-day promise)

```json
{
  "order_id": "KO2619999",
  "score": 0.933099,
  "band": "Hold",
  "threshold": 0.1118,
  "decision": "High return risk - review before dispatch; prefer a confirmation call over a blind hold.",
  "expected_saving_inr": 375.57,
  "expected_call_cost_inr": 45.0,
  "reasons": [
    {
      "direction": "up",
      "points": 59,
      "text": "This customer has had 3 of their 5 previous orders returned"
    },
    {
      "direction": "up",
      "points": 12,
      "text": "Cash on delivery orders come back about 1 in 5 times, more often than average"
    },
    {
      "direction": "up",
      "points": 8,
      "text": "Promised delivery is 9 days; longer promises come back more often than short ones"
    },
    {
      "direction": "up",
      "points": 2,
      "text": "Order value is about Rs 29,698"
    }
  ],
  "model_version": "logit-v1",
  "warnings": []
}
```

## Implausible input: Rs 21,999 against a Rs 4,048 SKU

The brief warns that `order_value_inr` is unreliable, and the October gateway incident proves it.
A value 5.4x the list price is outside anything in training, where a linear model extrapolates
badly: this record scored 0.0002 before the guard, i.e. "ship", which is exactly the wrong answer
for a cash-on-delivery customer with three prior returns. The service clamps the ratio to the range
real orders occupy and says so, rather than returning a confident wrong answer.

```json
{
  "order_id": "KO2610504",
  "score": 0.03107,
  "band": "Ship",
  "threshold": 0.1118,
  "decision": "Low predicted risk - ship as usual.",
  "expected_saving_inr": 12.51,
  "expected_call_cost_inr": 45.0,
  "reasons": [
    {
      "direction": "down",
      "points": 0,
      "text": "Cash on delivery with a 7-day promised delivery, but this customer has returned 1 of 2 previous orders. The score is still below the call threshold."
    }
  ],
  "model_version": "logit-v1",
  "warnings": [
    "Order value (Rs 21,999) is 5.4x the list price of this SKU, which does not occur in real orders. The score was computed with the value treated as 2.5x list price - please check the order record."
  ]
}
```

## Malformed request

```json
{
  "error": "Missing required fields: order_placed_at, customer_id, sku, sales_channel, payment_mode, discount_pct, qty, order_value_inr, promised_delivery_days, delivery_pincode, is_gift, customer_prior_orders, customer_prior_returns"
}
```

## Tests

```
$ pytest -q
8 passed in 1.69s
```

## Screen

Rendered headlessly at 1400x1050 with the "Score order" button clicked. The result panel populated
from `POST /score` with no console errors. See `reports/ui_low_risk.png` and
`reports/ui_high_risk.png`.
