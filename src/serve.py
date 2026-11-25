"""stdlib HTTP server for returns-risk scoring. No framework, no API key, no LLM."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    MODEL_VERSION,
    OPERATING_THRESHOLD,
    WEB_DIR,
)
from src.economics import per_order_call_expectation
from src.features import matrix_from_frame, score_band, select_reason_payload

SAMPLE_PATH = WEB_DIR / "sample_order.json"

_STATE: dict[str, Any] = {"bundle": None, "meta": None, "sample_row": None}

_FALLBACK_SAMPLE = {
    "order_id": "KO2610504",
    "order_placed_at": "2026-07-01 00:51",
    "customer_id": "KC105196",
    "sku": "KH-IC-03",
    "sales_channel": "web",
    "payment_mode": "prepaid_upi",
    "discount_pct": 13,
    "qty": 1,
    "order_value_inr": 3521.76,
    "promised_delivery_days": 7,
    "delivery_pincode": "440378",
    "is_gift": "N",
    "customer_prior_orders": 2,
    "customer_prior_returns": 1,
    "delivery_note": "Office address, weekdays only",
}


def load_model() -> None:
    if not MODEL_PATH.exists():
        _STATE["bundle"] = None
        _STATE["meta"] = None
        return
    payload = joblib.load(MODEL_PATH)
    _STATE["bundle"] = payload["bundle"]
    _STATE["meta"] = payload["meta"]


def _ensure_sample_row() -> dict[str, Any]:
    """Serve committed sample_order.json; never touch the data pack; never raise."""
    if _STATE["sample_row"] is not None:
        return _STATE["sample_row"]
    row = dict(_FALLBACK_SAMPLE)
    try:
        if SAMPLE_PATH.exists():
            loaded = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("order_id"):
                row = loaded
    except Exception:
        row = dict(_FALLBACK_SAMPLE)
    _STATE["sample_row"] = row
    return row


REQUIRED_ORDER_FIELDS = [
    "order_id",
    "order_placed_at",
    "customer_id",
    "sku",
    "sales_channel",
    "payment_mode",
    "discount_pct",
    "qty",
    "order_value_inr",
    "promised_delivery_days",
    "delivery_pincode",
    "is_gift",
    "customer_prior_orders",
    "customer_prior_returns",
]


def _validate_order(body: dict) -> str | None:
    missing = [f for f in REQUIRED_ORDER_FIELDS if f not in body or body[f] is None]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None


def _order_to_feature_frame(body: dict) -> pd.DataFrame:
    """Build a one-row feature frame using only the request + model-bundle lookups."""
    from src.features import _featurise_frame, fix_october_inflation

    meta = _STATE["meta"] or {}
    products = meta.get("products_by_sku") or {}
    customers = meta.get("customers_by_id") or {}
    defaults = meta.get("defaults") or {
        "state": "KA",
        "city": "UNKNOWN",
        "shield_member": "N",
        "family": "UNKNOWN",
        "list_price_inr": 5000.0,
        "warranty_months": 12.0,
    }

    row = dict(body)
    row.setdefault("delivery_note", None)
    row.setdefault("last_service_event_type", "NONE")
    row.setdefault("pickup_scheduled_at", None)
    row.setdefault("source", "crm")

    sku = str(row.get("sku", ""))
    cid = str(row.get("customer_id", ""))
    prod = products.get(sku, {})
    cust = customers.get(cid, {})

    row["family"] = row.get("family") or prod.get("family") or defaults["family"]
    row["list_price_inr"] = (
        row.get("list_price_inr")
        if row.get("list_price_inr") is not None
        else prod.get("list_price_inr", defaults["list_price_inr"])
    )
    row["warranty_months"] = (
        row.get("warranty_months")
        if row.get("warranty_months") is not None
        else prod.get("warranty_months", defaults["warranty_months"])
    )
    row["state"] = row.get("state") or cust.get("state") or defaults["state"]
    row["city"] = row.get("city") or cust.get("city") or defaults["city"]
    row["shield_member"] = (
        row.get("shield_member") or cust.get("shield_member") or defaults["shield_member"]
    )
    row["signup_date"] = (
        row.get("signup_date") or cust.get("signup_date") or row["order_placed_at"]
    )

    raw = pd.DataFrame([row])
    raw = fix_october_inflation(raw)
    return _featurise_frame(raw)


def occlusion_deltas(
    feat_row: pd.DataFrame, score: float, references: dict[str, Any], bundle
) -> list[tuple[str, float, Any]]:
    from src.config import CATEGORICAL_FEATURES as CAT_COLS

    X0 = matrix_from_frame(feat_row)
    for c in CAT_COLS:
        X0[c] = X0[c].astype(str)
    base = float(score)
    deltas: list[tuple[str, float, Any]] = []
    row = feat_row.iloc[0]

    candidates = [
        "payment_mode",
        "shield_member",
        "customer_prior_returns",
        "promised_delivery_days",
        "sales_channel",
        "family",
        "discount_pct",
        "value",
        "is_gift",
        "pin_missing",
        "state",
        "customer_prior_orders",
        "qty",
    ]
    for field in candidates:
        if field not in FEATURE_COLUMNS:
            continue
        X = X0.copy()
        ref = references.get(field)
        if ref is None:
            continue
        X.loc[X.index[0], field] = ref
        s = float(bundle.predict_proba(X)[:, 1][0])
        deltas.append((field, base - s, row[field]))
    return deltas


PRICE_RATIO_BAND = (0.4, 2.5)


def _guard_value_plausibility(
    body: dict, feat: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """`order_value_inr` is not trustworthy in this pack (the October gateway incident).

    A value far from the SKU's list price makes the linear model extrapolate well outside
    anything it was trained on, which can flip a genuinely risky order to "ship". Clamp the
    ratio to the range real orders occupy and tell the caller we did it.
    """
    warnings: list[str] = []
    row = feat.iloc[0]
    ratio = float(row["price_ratio"])
    lo, hi = PRICE_RATIO_BAND
    if not (lo <= ratio <= hi):
        clamped = float(np.clip(ratio, lo, hi))
        feat.loc[feat.index[0], "price_ratio"] = clamped
        warnings.append(
            f"Order value (Rs {float(row['value']):,.0f}) is {ratio:.1f}x the list price of "
            f"this SKU, which does not occur in real orders. The score was computed with the "
            f"value treated as {clamped:.1f}x list price - please check the order record."
        )
    return feat, warnings


def score_order(body: dict) -> dict[str, Any]:
    bundle = _STATE["bundle"]
    meta = _STATE["meta"]
    if bundle is None or meta is None:
        raise FileNotFoundError("model not found")

    feat, warnings = _guard_value_plausibility(body, _order_to_feature_frame(body))
    X = matrix_from_frame(feat)
    score = float(bundle.predict_proba(X)[:, 1][0])
    score = float(np.clip(score, 1e-6, 1 - 1e-6))

    thr = OPERATING_THRESHOLD
    band, decision = score_band(score, thr)
    value = float(feat.iloc[0]["value"])
    econ = per_order_call_expectation(score, value, threshold=thr)
    deltas = occlusion_deltas(feat, score, meta["references"], bundle)
    rates = meta.get("rate_lookup") or {"overall": float(meta.get("positive_rate", 0.11))}
    flagged = score > thr
    reasons = select_reason_payload(
        deltas, feat.iloc[0], rates, flagged=flagged, max_reasons=4
    )

    return {
        "order_id": str(body.get("order_id")),
        "score": round(score, 6),
        "band": band,
        "threshold": round(thr, 4),
        "decision": decision,
        "expected_saving_inr": econ["expected_saving_inr"],
        "expected_call_cost_inr": econ["expected_call_cost_inr"],
        "reasons": [
            {"direction": r["direction"], "points": r["points"], "text": r["text"]}
            for r in reasons
        ],
        "model_version": meta.get("model_version", MODEL_VERSION),
        "warnings": warnings,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "KestrelReturnsRisk/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: Any, content_type: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            content_type = "application/json"
        elif isinstance(body, bytes):
            data = body
        else:
            data = str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/health":
                ok = _STATE["bundle"] is not None
                self._send(
                    200,
                    {
                        "status": "ok" if ok else "degraded",
                        "model_loaded": ok,
                        "model_version": (
                            (_STATE["meta"] or {}).get("model_version", MODEL_VERSION)
                            if ok
                            else None
                        ),
                    },
                )
                return
            if path == "/sample":
                self._send(200, _ensure_sample_row())
                return
            if path in ("/", "/index.html"):
                html_path = WEB_DIR / "index.html"
                if not html_path.exists():
                    self._send(404, {"error": "web/index.html not found"})
                    return
                self._send(
                    200,
                    html_path.read_bytes(),
                    content_type="text/html; charset=utf-8",
                )
                return
            self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"Unexpected server error: {exc}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path != "/score":
                self._send(404, {"error": "not found"})
                return
            if _STATE["bundle"] is None:
                self._send(503, {"error": "model not found, run python -m src.train"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send(400, {"error": "Request body must be valid JSON"})
                return
            if not isinstance(body, dict):
                self._send(400, {"error": "JSON body must be an object (one order record)"})
                return
            err = _validate_order(body)
            if err:
                self._send(400, {"error": err})
                return
            try:
                result = score_order(body)
                self._send(200, result)
            except FileNotFoundError:
                self._send(503, {"error": "model not found, run python -m src.train"})
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"Could not score order: {exc}"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"Unexpected server error: {exc}"})


def main() -> None:
    load_model()
    port = int(os.environ.get("PORT", "8000"))
    host = "0.0.0.0"
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Kestrel returns-risk scoring -> http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
