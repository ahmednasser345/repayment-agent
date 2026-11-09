"""Metrics and alerts over all cases: what the dashboard tiles show, and a text report.

    python -m repayment.metrics --db app.db
"""

import argparse
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import db as store
from . import labels
from .config import Limits

EXTERNAL_TOOLS = ("check_payments", "get_sales_summary")
WAITING_TOO_LONG_DAYS = 5
COST_WARNING_SHARE = 0.8
TOOL_FAILURE_ALERT = 0.2


def compute(db: sqlite3.Connection, now: datetime) -> dict[str, Any]:
    """Rates are fractions from 0 to 1. Per-case averages count only cases the AI has worked on."""
    cases = store.rows(db, "SELECT * FROM cases")
    by_state = Counter(case["state"] for case in cases)
    overdue = sum(case["amount_due"] for case in cases)
    recovered = sum(case["amount_due"] - case["balance"] for case in cases)
    runs = store.one(db, "SELECT COUNT(*) AS n, COUNT(DISTINCT case_id) AS cases, COALESCE(SUM(steps), 0) AS steps, "
                         "COALESCE(SUM(llm_calls), 0) AS llm_calls FROM runs")
    promises = Counter(row["status"] for row in store.rows(db, "SELECT status FROM promises"))
    attempts, failed = _external_attempts(db)
    total_cost = sum(case["cost_usd"] for case in cases)
    return {
        "cases_total": len(cases),
        "by_state": {state: by_state.get(state, 0) for state in labels.STATE_LABELS},
        "resolved_rate": _rate(by_state["RESOLVED"], len(cases)),
        "escalation_rate": _rate(by_state["ESCALATED"], len(cases)),
        "escalations_by_reason": dict(Counter(case["last_stop_reason"] or "unknown" for case in cases
                                              if case["state"] == "ESCALATED")),
        "amount_overdue_total": overdue,
        "amount_recovered": recovered,
        "recovery_rate": _rate(recovered, overdue),
        "avg_runs_per_case": round(runs["n"] / runs["cases"], 2) if runs["cases"] else 0.0,
        "avg_steps_per_run": round(runs["steps"] / runs["n"], 2) if runs["n"] else 0.0,
        "avg_llm_calls_per_run": round(runs["llm_calls"] / runs["n"], 2) if runs["n"] else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_case": round(total_cost / runs["cases"], 6) if runs["cases"] else 0.0,
        "total_tokens": sum(case["tokens_in"] + case["tokens_out"] for case in cases),
        "tool_failure_rate": _rate(failed, attempts),
        "promises": {"made": sum(promises.values()), "kept": promises["kept"], "broken": promises["broken"],
                     "kept_rate": _rate(promises["kept"], promises["kept"] + promises["broken"])},
        "waiting_too_long": sum(1 for case in cases if _waiting_too_long(db, case, now)),
        "flagged_messages": store.one(db, "SELECT COUNT(*) AS n FROM messages WHERE flag IS NOT NULL")["n"],
    }


def alerts(db: sqlite3.Connection, now: datetime, limits: Limits) -> list[dict[str, Any]]:
    found = []
    for case in store.rows(db, "SELECT c.*, m.business_name FROM cases c JOIN merchants m ON m.id = c.merchant_id "
                               "ORDER BY c.rowid"):
        name, case_id = case["business_name"], case["id"]
        if _waiting_too_long(db, case, now):
            found.append(_alert("warn", case_id, f"{name}: no reply or payment for more than "
                                                  f"{WAITING_TOO_LONG_DAYS} days."))
        if case["cost_usd"] >= COST_WARNING_SHARE * limits.max_cost_usd_per_case:
            share = round(case["cost_usd"] / limits.max_cost_usd_per_case * 100)
            found.append(_alert("warn", case_id, f"{name}: used {share}% of its AI budget "
                                                  f"(${case['cost_usd']:.2f} of ${limits.max_cost_usd_per_case:.2f})."))
        if case["retry_at"]:
            found.append(_alert("info", case_id, f"{name}: {labels.paused(case, now)}"))
        if case["state"] == "NEEDS_APPROVAL":
            found.append(_alert("warn", case_id, f"{name}: a payment plan needs a manager's decision."))
        if case["state"] == "ESCALATED" and (case["last_stop_reason"] or "").startswith("cap:"):
            found.append(_alert("bad", case_id, f"{name}: {labels.stop_label(case['last_stop_reason'])}"))
    attempts, failed = _external_attempts(db)
    if attempts and failed / attempts > TOOL_FAILURE_ALERT:
        found.append(_alert("bad", None, f"Tool failure rate is {failed / attempts:.0%} ({failed} of {attempts} "
                                         "calls to the payment and sales systems failed)."))
    return found


def report(db: sqlite3.Connection, now: datetime, limits: Limits) -> str:
    m = compute(db, now)
    states = ", ".join(f"{labels.STATE_LABELS[s]} {n}" for s, n in m["by_state"].items() if n)
    p = m["promises"]
    lines = [
        f"Metrics at {labels.today_label(now)}",
        f"  Cases: {m['cases_total']} ({states})",
        f"  Resolved {m['resolved_rate']:.0%} · escalated {m['escalation_rate']:.0%}",
        f"  Money: recovered {labels.money(m['amount_recovered'])} of {labels.money(m['amount_overdue_total'])} "
        f"overdue ({m['recovery_rate']:.0%})",
        f"  AI: {m['avg_runs_per_case']} runs per case, {m['avg_steps_per_run']} steps and "
        f"{m['avg_llm_calls_per_run']} AI calls per run, {m['total_tokens']:,} tokens, "
        f"est. cost ${m['total_cost_usd']:.4f} at paid-tier prices",
        f"  Tool failure rate: {m['tool_failure_rate']:.0%}",
        f"  Promises: {p['made']} made, {p['kept']} kept, {p['broken']} broken",
        f"  Waiting too long: {m['waiting_too_long']} · flagged merchant messages: {m['flagged_messages']}",
        "Alerts:" if (found := alerts(db, now, limits)) else "Alerts: none",
    ]
    lines += [f"  [{a['level']}] {a['text']}" for a in found]
    return "\n".join(lines)


def _waiting_too_long(db: sqlite3.Connection, case: Any, now: datetime) -> bool:
    """WAITING with no merchant activity (a reply or a payment) for more than 5 days."""
    if case["state"] != "WAITING":
        return False
    row = store.one(db, "SELECT MAX(at) AS at FROM (SELECT created_at AS at FROM messages WHERE case_id = ? AND "
                        "sender = 'merchant' UNION ALL SELECT received_at FROM payments WHERE case_id = ? AND "
                        "status = 'applied')", case["id"], case["id"])
    since = datetime.fromisoformat(row["at"] or case["opened_at"])
    return now - since > timedelta(days=WAITING_TOO_LONG_DAYS)


def _external_attempts(db: sqlite3.Connection) -> tuple[int, int]:
    marks = ", ".join("?" for _ in EXTERNAL_TOOLS)
    row = store.one(db, f"SELECT COALESCE(SUM(attempts), 0) AS attempts, COALESCE(SUM(failed_attempts), 0) AS "
                        f"failed FROM steps WHERE tool IN ({marks})", *EXTERNAL_TOOLS)
    return row["attempts"], row["failed"]


def _rate(part: float, whole: float) -> float:
    return round(part / whole, 3) if whole else 0.0


def _alert(level: str, case_id: str | None, text: str) -> dict[str, Any]:
    return {"level": level, "case_id": case_id, "text": text}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the metrics and alerts of a repayment-agent database.")
    parser.add_argument("--db", default="app.db")
    args = parser.parse_args(argv)
    if not Path(args.db).exists():
        print(f"No database at {args.db}. Run python server.py first, or pass --db.")
        return 1
    db = store.connect(args.db)
    latest = store.one(db, "SELECT MAX(updated_at) AS at FROM cases")["at"]
    now = datetime.now(timezone.utc)
    if latest and datetime.fromisoformat(latest) > now:  # the app's clock may have been moved ahead
        now = datetime.fromisoformat(latest)
    print(report(db, now, Limits()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
