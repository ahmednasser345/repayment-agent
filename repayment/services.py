"""Simulated external systems: the payment provider and the store's sales integration.

Outages are injected through env.outages: {"payments": 2} makes the next two payment calls time out.
"""

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from . import db as store
from .labels import SERVICE_NAMES

if TYPE_CHECKING:
    from .env import Env


class ServiceDown(Exception):
    pass


def _fail_if_down(env: "Env", service: str) -> None:
    if env.outages.get(service, 0) > 0:
        env.outages[service] -= 1
        raise ServiceDown(f"{SERVICE_NAMES[service]} timed out")


def payments_report(env: "Env", case_id: str) -> dict[str, Any]:
    _fail_if_down(env, "payments")
    case = store.get_case(env.db, case_id)
    merchant_id = case["merchant_id"]
    since = (env.clock.now() - timedelta(days=90)).isoformat()
    applied = store.rows(
        env.db, "SELECT reference, amount, received_at, source FROM payments "
        "WHERE case_id = ? AND status = 'applied' ORDER BY id", case_id)
    pending = store.rows(
        env.db, "SELECT reference, amount, received_at, note FROM payments "
        "WHERE merchant_id = ? AND status = 'pending' ORDER BY id", merchant_id)
    history = store.rows(
        env.db, "SELECT reference, amount, received_at, note FROM payments "
        "WHERE merchant_id = ? AND case_id IS NULL AND received_at >= ? ORDER BY received_at", merchant_id, since)
    return {
        "balance": case["balance"],
        "currency": case["currency"],
        "applied_to_this_case": [dict(row) for row in applied],
        "pending": [dict(row) for row in pending],
        "history_last_90_days": [dict(row) for row in history],
    }


def sales_report(env: "Env", case_id: str) -> dict[str, Any]:
    _fail_if_down(env, "sales")
    case = store.get_case(env.db, case_id)
    newest_first = store.rows(
        env.db, "SELECT month, revenue, orders FROM sales WHERE merchant_id = ? ORDER BY month DESC LIMIT 6",
        case["merchant_id"])
    months = [dict(row) for row in reversed(newest_first)]
    last = [m["revenue"] for m in months[-3:]]
    previous = [m["revenue"] for m in months[-6:-3]]
    last_avg = round(sum(last) / len(last)) if last else 0
    previous_avg = round(sum(previous) / len(previous)) if previous else 0
    return {
        "months": months,
        "last_3_months_avg": last_avg,
        "previous_3_months_avg": previous_avg,
        "change_pct": round((last_avg - previous_avg) / previous_avg * 100, 1) if previous_avg else None,
        "balance_as_pct_of_monthly_revenue": round(case["balance"] / last_avg * 100, 1) if last_avg else None,
    }
