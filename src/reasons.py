"""Ops-facing reason copy: rate-based sentences, no raw feature names."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Indian state codes used in customers.csv -> clerk-readable names
STATE_NAMES: dict[str, str] = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TG": "Telangana",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

PAYMENT_LABELS = {
    "cod": "Cash on delivery",
    "emi": "EMI",
    "prepaid_upi": "Prepaid UPI",
    "prepaid_card": "Prepaid card",
}

CHANNEL_LABELS = {
    "app": "App",
    "web": "Web",
    "marketplace": "Marketplace",
    "partner_outlet": "Partner outlet",
}

RATE_COLS = ("family", "state", "sales_channel", "payment_mode", "shield_member")


def rate_as_one_in(rate: float) -> str:
    if rate <= 0:
        return "almost never"
    if rate >= 0.5:
        return f"about {rate:.0%} of the time"
    n = max(2, int(round(1.0 / rate)))
    return f"about 1 in {n} times"


def build_rate_lookup(train_df: pd.DataFrame) -> dict[str, Any]:
    """Historical return rates from cleaned training data (for clerk sentences)."""
    y = train_df["returned"].astype(int)
    overall = float(y.mean())
    lookup: dict[str, Any] = {"overall": overall, "n": int(len(train_df))}
    for col in RATE_COLS:
        rates: dict[str, float] = {}
        for val, grp in train_df.groupby(train_df[col].astype(str)):
            rates[str(val)] = float(grp["returned"].astype(int).mean())
        lookup[col] = rates
    # Rank families for "highest of any family" phrasing
    fam = lookup["family"]
    if fam:
        lookup["family_rank"] = {
            k: i + 1
            for i, (k, _) in enumerate(sorted(fam.items(), key=lambda kv: -kv[1]))
        }
    return lookup


def _vs_average(rate: float, overall: float) -> str:
    if rate > overall * 1.15:
        return "more often than average"
    if rate < overall * 0.85:
        return "less often than average"
    return "about as often as average"


def rate_sentence(
    field: str,
    value: Any,
    row: pd.Series,
    rates: dict[str, Any],
    *,
    direction: str = "up",
) -> str:
    """One standalone sentence a warehouse clerk can act on."""
    overall = float(rates.get("overall", 0.11))
    val = value

    if field == "customer_prior_returns":
        n = int(float(val))
        orders = int(float(row.get("customer_prior_orders", 0)))
        if n <= 0:
            return "This customer has no prior returns on record"
        denom = max(orders, n)
        return f"This customer has had {n} of their {denom} previous orders returned"

    if field == "customer_prior_orders":
        orders = int(float(val))
        rets = int(float(row.get("customer_prior_returns", 0)))
        if orders <= 0:
            return "First-time buyer with no order history on file"
        if rets <= 0:
            return f"This customer has {orders} prior orders and no returns so far"
        return f"This customer has had {rets} of their {orders} previous orders returned"

    if field == "promised_delivery_days":
        n = int(float(val))
        if n >= 7:
            return (
                f"Promised delivery is {n} days; longer promises come back "
                "more often than short ones"
            )
        return f"Promised delivery is only {n} days, which usually means lower return risk"

    if field == "payment_mode":
        label = PAYMENT_LABELS.get(str(val), str(val))
        rate = float(rates.get("payment_mode", {}).get(str(val), overall))
        return (
            f"{label} orders come back {rate_as_one_in(rate)}, "
            f"{_vs_average(rate, overall)}"
        )

    if field == "sales_channel":
        label = CHANNEL_LABELS.get(str(val), str(val))
        rate = float(rates.get("sales_channel", {}).get(str(val), overall))
        return (
            f"{label} orders come back {rate_as_one_in(rate)}, "
            f"{_vs_average(rate, overall)}"
        )

    if field == "family":
        name = str(val)
        rate = float(rates.get("family", {}).get(name, overall))
        rank = rates.get("family_rank", {}).get(name)
        base = f"{name}s come back {rate_as_one_in(rate)}"
        if rank == 1:
            return f"{base}, the highest of any family"
        return f"{base}, {_vs_average(rate, overall)}"

    if field == "state":
        code = str(val)
        name = STATE_NAMES.get(code, code)
        rate = float(rates.get("state", {}).get(code, overall))
        return (
            f"Deliveries to {name} return {_vs_average(rate, overall)} "
            f"({rate_as_one_in(rate)})"
        )

    if field == "shield_member":
        rate = float(rates.get("shield_member", {}).get(str(val).upper(), overall))
        if str(val).upper() == "Y":
            return (
                f"Shield member: free returns, and they come back "
                f"{rate_as_one_in(rate)}, {_vs_average(rate, overall)}"
            )
        return "Not a Shield member"

    if field == "discount_pct":
        n = int(float(val))
        if n >= 15:
            return f"Checkout discount is {n}%, which tends to raise return risk"
        return f"Little or no discount ({n}%) at checkout"

    if field == "value":
        n = float(val)
        return f"Order value is about Rs {n:,.0f}"

    if field == "is_gift":
        if str(val).upper() == "Y":
            return "Marked as a gift order"
        return "Not marked as a gift"

    if field == "pin_missing":
        if int(float(val)) == 1:
            return "Delivery pincode was missing (system used the 000000 default)"
        return "Delivery pincode is on file"

    if field == "qty":
        return f"Quantity ordered is {int(float(val))}"

    return "Something about this order raised the risk score"


def safety_summary(row: pd.Series, rates: dict[str, Any]) -> str:
    """Below-threshold explanation, phrased so it never contradicts the facts."""
    pay = str(row.get("payment_mode", ""))
    if pay.startswith("prepaid"):
        pay_txt = "Prepaid"
    elif pay == "cod":
        pay_txt = "Cash on delivery"
    elif pay == "emi":
        pay_txt = "Paid by EMI"
    else:
        pay_txt = "Paid up front"

    rets = int(float(row.get("customer_prior_returns", 0) or 0))
    orders = int(float(row.get("customer_prior_orders", 0) or 0))
    days = int(float(row.get("promised_delivery_days", 5) or 5))

    if rets <= 0:
        return (
            f"{pay_txt}, no prior returns, {days}-day promised delivery - "
            "nothing about this order stands out."
        )
    return (
        f"{pay_txt} with a {days}-day promised delivery, but this customer has "
        f"returned {rets} of {max(orders, rets)} previous orders. The score is still "
        "below the call threshold."
    )


def select_reason_payload(
    deltas: list[tuple[str, float, Any]],
    row: pd.Series,
    rates: dict[str, Any],
    *,
    flagged: bool,
    max_reasons: int = 4,
) -> list[dict[str, Any]]:
    """
    Flagged: reasons that push score UP (delta>=0.01), plus at most one that pulls down.
    Clean: one safety sentence, not a list of negative deltas.
    """
    if not flagged:
        return [
            {
                "direction": "down",
                "points": 0,
                "text": safety_summary(row, rates),
            }
        ]

    ups = [(f, d, v) for f, d, v in deltas if d >= 0.01]
    downs = [(f, d, v) for f, d, v in deltas if d <= -0.01]
    ups.sort(key=lambda t: -t[1])
    downs.sort(key=lambda t: t[1])  # most negative first

    chosen: list[tuple[str, float, Any]] = ups[: max_reasons - (1 if downs else 0)]
    if downs and len(chosen) < max_reasons:
        chosen.append(downs[0])

    out: list[dict[str, Any]] = []
    for field, delta, value in chosen[:max_reasons]:
        points = int(round(delta * 100))
        if abs(points) < 1 and abs(delta) < 0.01:
            continue
        direction = "up" if delta > 0 else "down"
        text = rate_sentence(field, value, row, rates, direction=direction)
        # A neutral statement is not a reason; drop it rather than pad the list.
        if "about as often as average" in text:
            continue
        out.append({"direction": direction, "points": points, "text": text})
    return out
