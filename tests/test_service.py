"""Smoke-test the stdlib scoring server on an ephemeral port."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    model = ROOT / "models" / "model.joblib"
    if not model.exists():
        # Train once so the service test can run in a fresh checkout after train
        subprocess.run(
            [sys.executable, "-m", "src.train"],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )

    port = _free_port()
    env = {**os.environ, "PORT": str(port), "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.serve"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    last_err = None
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"server exited early: {out}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError(f"server did not become healthy: {last_err}")

    yield url
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_health(server):
    with urllib.request.urlopen(server + "/health") as resp:
        body = json.loads(resp.read().decode())
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_score_sample(server):
    with urllib.request.urlopen(server + "/sample") as resp:
        sample = json.loads(resp.read().decode())
    req = urllib.request.Request(
        server + "/score",
        data=json.dumps(sample).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read().decode())
    assert "score" in body
    assert 0 < body["score"] < 1
    assert body["band"] in ("Ship", "Call before dispatch", "Hold")
    assert "reasons" in body
    assert "expected_saving_inr" in body


def test_score_missing_field_400(server):
    req = urllib.request.Request(
        server + "/score",
        data=json.dumps({"order_id": "X"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 400


def test_implausible_order_value_is_guarded():
    """A value far from list price must not silently flip a risky order to 'ship'."""
    from src.serve import _guard_value_plausibility
    from src.features import _featurise_frame
    import pandas as pd

    raw = pd.DataFrame(
        [
            {
                "order_id": "T1",
                "order_placed_at": "2026-07-01 10:00",
                "customer_id": "C1",
                "sku": "KH-IC-03",
                "sales_channel": "web",
                "payment_mode": "cod",
                "discount_pct": 5,
                "qty": 1,
                "order_value_inr": 21999.0,
                "promised_delivery_days": 5,
                "delivery_pincode": "400001",
                "is_gift": "N",
                "customer_prior_orders": 5,
                "customer_prior_returns": 3,
                "delivery_note": None,
                "last_service_event_type": "NONE",
                "pickup_scheduled_at": None,
                "source": "crm",
                "list_price_inr": 4048.0,
                "family": "Induction Cooktop",
                "warranty_months": 12,
                "signup_date": "2024-01-01",
                "state": "MH",
                "city": "Mumbai",
                "shield_member": "N",
            }
        ]
    )
    feat, warnings = _guard_value_plausibility({}, _featurise_frame(raw))
    assert warnings, "an implausible value must be reported"
    assert feat.iloc[0]["price_ratio"] <= 2.5 + 1e-9
