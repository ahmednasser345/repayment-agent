"""Every plain-language text the app shows: labels for codes, notices to the merchant, and status sentences."""

import json
from datetime import date, datetime, timedelta
from typing import Any

from .config import Limits

STATE_LABELS = {
    "NEW": "Not started",
    "WORKING": "AI is working",
    "WAITING": "Waiting for the merchant",
    "NEEDS_APPROVAL": "Needs a manager's approval",
    "ON_PLAN": "On a payment plan",
    "RESOLVED": "Paid — closed",
    "ESCALATED": "With a person",
}

STOP_LABELS = {
    "waiting_on_merchant": "Sent a message and is waiting for the merchant.",
    "plan_offered": "Offered a payment plan and is waiting for the merchant to accept or decline.",
    "needs_approval": "Proposed a plan that needs a manager's approval.",
    "escalated": "The AI decided a person should handle this.",
    "resolved": "Paid in full, so code closed the case.",
    "cap:max_steps": "Hit the step limit for one run, so code handed it to a person.",
    "cap:budget": "Hit the AI cost limit for this case, so code handed it to a person.",
    "cap:deadline": "The case deadline passed, so code handed it to a person.",
    "cap:reminders": "No reply after the last reminder, so code handed it to a person.",
    "cap:broken_promises": "Too many broken promises, so code handed it to a person.",
    "cap:missed_installments": "Too many missed plan payments, so code handed it to a person.",
    "cap:messages": "Hit the message limit for this case, so code handed it to a person.",
    "cap:tool_retries": "A system stayed down after several retries, so code handed it to a person.",
    "stuck:repeated_call": "The AI repeated the same action without learning anything, so code stopped it.",
    "stuck:no_progress": "The AI made several steps without progress, so code stopped it.",
    "malformed_output": "The AI kept giving invalid answers, so code stopped it.",
    "internal_error": "Unexpected error — a person must check.",
    "abandoned": "The app stopped during this run; the next event recovered the case.",
}
SERVICE_NAMES = {"payments": "The payment system", "sales": "The store's sales system", "ai": "The AI"}
SERVICE_PHRASES = {"payments": "the payment system", "sales": "the store's sales system", "ai": "the AI"}

TRIGGER_LABELS = {
    "CASE_OPENED": "Installment overdue — case opened",
    "MERCHANT_REPLY": "The merchant replied",
    "PAYMENT_RECEIVED": "A payment arrived",
    "PLAN_ACCEPTED": "The merchant accepted the plan",
    "PLAN_DECLINED": "The merchant declined the plan",
    "REMINDER_DUE": "No reply for a day — reminder due",
    "DUE_TODAY": "A payment is due today — reminder",
    "PROMISE_DUE": "The promised payment date passed",
    "INSTALLMENT_DUE": "A plan payment date passed",
    "HUMAN_DECISION": "A manager decided",
    "TOOL_RETRY": "Retrying after an outage",
    "DEADLINE_PASSED": "The case deadline passed",
}

TOOL_LABELS = {
    "check_payments": "Checked payments",
    "get_sales_summary": "Checked store sales",
    "record_promise": "Recorded a promise to pay",
    "propose_plan": "Proposed a payment plan",
    "send_message": "Sent a message",
    "escalate_to_human": "Handed to a person",
}
OUTCOME_LABELS = {
    "ok": "Done",
    "rejected": "Blocked by code",
    "failed": "System error",
    "malformed": "Invalid AI answer",
    "stuck": "Stopped: repeating itself",
    "skipped": "Skipped",
}

DECISION_LABELS = {
    "ran_agent": "AI ran",
    "handled_by_code": "Handled by code — no AI needed",
    "stored_for_human": "Saved for the person who owns the case",
    "skipped": "Ignored",
    "escalated": "Limit reached — handed to a person (no AI)",
    "failed": "Unexpected error — handed to a person",
}
DETAIL_LABELS = {
    "duplicate_event": "Same event delivered twice — ignored",
    "duplicate_payment": "This payment was already applied",
    "stale": "Nothing new to act on",
    "case_closed": "Case is already closed",
    "already_started": "Case already started",
    "stale_offer": "That offer is no longer open",
    "human_owns_case": "A person owns this case, so the AI did not run",
}

PLAN_STATUS_LABELS = {
    "pending_approval": "Waiting for a manager's approval",
    "offered": "Offered — waiting for the merchant",
    "active": "Active",
    "declined": "Declined by the merchant",
    "rejected": "Rejected by a manager",
    "superseded": "Replaced by a newer plan",
    "outdated": "Out of date — a payment or the passing days changed it",
    "withdrawn": "Withdrawn — the case went to a person",
    "completed": "Completed — paid in full",
    "broken": "Broken — too many missed payments",
}

LIMIT_LABELS = {
    "reminders": "Reminders sent",
    "broken_promises": "Broken promises",
    "missed_installments": "Missed plan payments",
    "messages": "Messages sent by the AI",
    "budget": "AI cost (est.)",
    "tokens": "AI tokens used",
    "deadline": "Days used of the case deadline",
}

FLAG_TEXT = "Instruction-like text from the merchant (ignored by design)"
NOTICE_ESCALATED = "A member of our team will contact you shortly about your balance."
NOTICE_PLAN_REVIEW = "Thanks — we're reviewing a payment plan for you. You'll hear back soon."
NOTICE_OFFER_OUTDATED = ("This offer is out of date: a payment or the passing days changed what is owed. Reply here "
                         "and we'll send you a new one.")


def money(amount: int) -> str:
    return f"{amount:,} EGP"


def day(value: date | str) -> str:
    """Sun 25 Oct"""
    value = date.fromisoformat(value) if isinstance(value, str) else value
    return f"{value:%a} {value.day} {value:%b}"


def day_time(value: datetime) -> str:
    """Tue 10 Nov 14:00"""
    return f"{day(value.date())} {value:%H:%M}"


def today_label(now: datetime) -> str:
    """Sun 8 Nov 2026, 14:00"""
    return f"{day(now.date())} {now.year}, {now:%H:%M}"


def when(moment: datetime, now: datetime) -> str:
    """14:30 today, otherwise Mon 9 Nov 14:30."""
    return f"{moment:%H:%M}" if moment.date() == now.date() else day_time(moment)


def stop_label(reason: str | None) -> str | None:
    if reason is None:
        return None
    if reason.startswith("paused:"):
        service = reason.removeprefix("paused:").removesuffix("_unavailable")
        return f"{SERVICE_NAMES.get(service, 'A system')} is down. Paused; it retries automatically."
    return STOP_LABELS.get(reason, reason)


def detail_label(detail: str | None) -> str | None:
    """Event details are codes (skip reasons, stop reasons) or sentences written by code."""
    if detail is None:
        return None
    return DETAIL_LABELS.get(detail) or stop_label(detail)


def tool_label(tool: str | None) -> str:
    if tool is None:
        return "No tool call"
    return TOOL_LABELS.get(tool, f"Unknown tool: {tool}")


def receipt(amount: int, balance: int) -> str:
    if balance == 0:
        return f"Payment received: {money(amount)}. Your balance is now 0 EGP — thank you! This case is closed."
    return f"Payment received: {money(amount)}. Remaining balance: {money(balance)}."


def plan_confirmed(amount: int, due: str) -> str:
    return f"Plan confirmed. Your next payment is {money(amount)} on {day(due)}."


def installments(plan: Any) -> list[dict[str, Any]]:
    return json.loads(plan["installments_json"])


def next_installment(plan: Any) -> dict[str, Any] | None:
    return next((item for item in installments(plan) if not item["paid"]), None)


def done(case: Any) -> dict[str, Any]:
    paid, total = case["amount_due"] - case["balance"], case["amount_due"]
    if case["balance"] == 0:
        text = f"Done: the balance is 0 EGP. Paid {paid:,} of {money(total)}."
    else:
        text = f"Done when the balance is 0 EGP. Paid {paid:,} of {money(total)}."
    return {"done": case["balance"] == 0, "text": text, "paid": paid, "total": total}


def _payment_day_then_check(due: str, what: str, today: str) -> str:
    """A payment agreed for a later day gets a reminder on that day; the day after, the AI checks it arrived."""
    check = f"on {day(date.fromisoformat(due) + timedelta(days=1))} the AI checks whether {what}."
    if due > today:
        return f"On {day(due)} the AI sends a payment-day reminder; {check}"
    return check[0].upper() + check[1:]


NEXT_DUE = {
    "REMINDER_DUE": "{name} gets a reminder",
    "DUE_TODAY": "{name} gets a payment-day reminder",
    "PROMISE_DUE": "the AI checks whether {name} kept their promise",
    "INSTALLMENT_DUE": "the AI checks whether {name} paid their plan payment",
    "TOOL_RETRY": "the AI retries {name} after the outage",
    "DEADLINE_PASSED": "the case deadline for {name} passes",
}


def next_due(name: str, event_type: str, on: date, days_ahead: int) -> str:
    """Next: Sun 15 Nov (in 6 days), Cairo Crafts gets a payment-day reminder."""
    ahead = "in 1 day" if days_ahead == 1 else f"in {days_ahead} days"
    return f"Next: {day(on)} ({ahead}), {NEXT_DUE[event_type].format(name=name)}."


def paused(case: Any, now: datetime) -> str:
    """Paused: the payment system is down. Retrying at 14:30."""
    service = (case["last_stop_reason"] or "").removeprefix("paused:").removesuffix("_unavailable")
    retry = datetime.fromisoformat(case["retry_at"])
    return f"Paused: {SERVICE_PHRASES.get(service, 'a system')} is down. Retrying at {when(retry, now)}."


def where_it_stands(case: Any, promise: Any, plan: Any, now: datetime) -> str:
    """One sentence for the dashboard: where the case is right now. plan = the newest open plan, if any."""
    state = case["state"]
    if state == "NEW":
        days_late = (now.date() - date.fromisoformat(case["due_date"])).days
        return f"{money(case['balance'])} is {days_late} days late. Nobody has contacted the merchant yet."
    if state == "WORKING":
        return "The AI is working on this case right now."
    if state == "NEEDS_APPROVAL":
        count = len(installments(plan)) if plan else 0
        return f"The AI proposed a {count}-payment plan. A manager must approve or reject it."
    if state == "RESOLVED":
        return "Paid in full."
    if state == "ESCALATED":
        return f"With a person: {case['escalation_reason'] or 'no reason recorded'}"
    if case["retry_at"]:
        return paused(case, now)
    if state == "ON_PLAN" and plan and (item := next_installment(plan)):
        if item["due_date"] < now.date().isoformat():
            return f"On a plan: the {money(item['amount'])} payment due {day(item['due_date'])} is late."
        return f"On a plan: next payment {money(item['amount'])} on {day(item['due_date'])}."
    if promise:
        return f"Merchant promised {money(promise['amount'])} by {day(promise['due_date'])}."
    if plan and plan["status"] == "offered":
        return "A payment plan was offered; waiting for the merchant to accept or decline."
    return "Waiting for the merchant to reply or pay."


def what_happens_next(
    case: Any, promise: Any, plan: Any, last_ours_at: datetime | None, now: datetime, limits: Limits,
) -> str:
    """What retriggers this case next. last_ours_at = when our last message was sent, if it is the newest one."""
    sentence = _next_trigger(case, promise, plan, last_ours_at, now, limits)
    if case["state"] == "RESOLVED":
        return sentence
    return f"{sentence} A payment closes the case as soon as the balance reaches 0."


def _next_trigger(
    case: Any, promise: Any, plan: Any, last_ours_at: datetime | None, now: datetime, limits: Limits,
) -> str:
    state = case["state"]
    fixed = {
        "NEW": "Press Start in the merchant app to open the case.",
        "WORKING": "The AI is working; this updates when it finishes.",
        "NEEDS_APPROVAL": "Nothing happens until a manager approves or rejects the plan.",
        "ESCALATED": "The AI is stopped. A person owns this case and can hand it back to the AI.",
        "RESOLVED": "Nothing — the case is closed.",
    }
    if state in fixed:
        return fixed[state]
    if case["retry_at"]:
        retry = datetime.fromisoformat(case["retry_at"])
        return f"The AI retries at {when(retry, now)} (retry {case['tool_retries']} of {limits.max_tool_retries})."
    today = now.date().isoformat()
    if promise:
        return _payment_day_then_check(promise["due_date"], f"{money(promise['amount'])} arrived", today)
    if state == "ON_PLAN" and plan:
        upcoming = [item for item in installments(plan) if not item["paid"] and item["due_date"] >= today]
        if not upcoming:
            return "The AI acts as soon as the merchant replies."
        item = upcoming[0]
        return _payment_day_then_check(item["due_date"], f"the {money(item['amount'])} payment arrived", today)
    if last_ours_at is None:
        return "The AI acts as soon as the merchant replies."
    remind_at = day_time(last_ours_at + timedelta(hours=limits.reminder_after_hours))
    sent, cap = case["reminders_sent"], limits.max_reminders
    if sent >= cap:
        return f"If the merchant stays silent until {remind_at}, a person takes over ({cap} reminders were sent)."
    return (f"If the merchant stays silent until {remind_at}, the AI sends reminder {sent + 1} of {cap}. "
            f"After {cap} reminders a person takes over.")
