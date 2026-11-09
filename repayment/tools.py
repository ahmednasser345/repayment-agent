"""The AI's only actions: six tools, each scoped by code to the case being worked (no case_id argument).

Every call is checked against a strict JSON schema before it runs, then against policy. A policy problem goes back
to the AI as an error it can fix, and nothing changes. No tool can change a balance, waive money or close a case.
"""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from . import db as store
from . import labels, policy, services
from .states import Actor, State, escalate, idle_state, transition

if TYPE_CHECKING:
    from .env import Env

REASON = {"type": "string", "minLength": 3, "maxLength": 300,
          "description": "One sentence: why this is the right next step. Shown to the operator."}
DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$", "description": "YYYY-MM-DD"}
AMOUNT = {"type": "integer", "minimum": 1, "description": "Whole EGP, no decimals."}


def _schema(**properties: dict[str, Any]) -> dict[str, Any]:
    properties["reason"] = REASON
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


@dataclass
class ToolResult:
    content: dict[str, Any]
    summary: str
    outcome: str = "ok"  # ok | rejected
    fact: str | None = None  # names new information, for progress detection
    ends_run: str | None = None  # the stop reason, when this call ends the run
    attempts: int = 1
    failed_attempts: int = 0


class ToolUnavailable(Exception):
    """An external system kept failing after the in-run retries: the run pauses."""

    def __init__(self, service: str, attempts: int):
        super().__init__(f"{labels.SERVICE_NAMES[service]} did not answer after {attempts} attempts")
        self.service, self.attempts = service, attempts


@dataclass(frozen=True)
class Tool:
    description: str
    schema: dict[str, Any]
    run: Callable[["Env", str, dict[str, Any]], ToolResult]


def check_payments(env: "Env", case_id: str, args: dict[str, Any]) -> ToolResult:
    report, attempts = _call_service(env, "payments", services.payments_report, case_id)
    _remember(env, case_id, "payments", report)
    applied, pending = report["applied_to_this_case"], report["pending"]
    summary = f"Balance {labels.money(report['balance'])}; {len(applied)} applied to this case; {len(pending)} pending"
    if pending:
        summary += " (" + ", ".join(labels.money(p["amount"]) for p in pending) + ")"
    fact = f"payments:{report['balance']}:{len(applied)}:{len(pending)}"
    return ToolResult(report, summary + ".", fact=fact, attempts=attempts, failed_attempts=attempts - 1)


def get_sales_summary(env: "Env", case_id: str, args: dict[str, Any]) -> ToolResult:
    report, attempts = _call_service(env, "sales", services.sales_report, case_id)
    _remember(env, case_id, "sales", report)
    summary = (f"Last 3 months average {labels.money(report['last_3_months_avg'])}, before that "
               f"{labels.money(report['previous_3_months_avg'])} ({report['change_pct']:+}%). The balance is "
               f"{report['balance_as_pct_of_monthly_revenue']}% of a month's sales.")
    latest = report["months"][-1]["month"] if report["months"] else "none"
    return ToolResult(report, summary, fact=f"sales:{latest}", attempts=attempts, failed_attempts=attempts - 1)


def record_promise(env: "Env", case_id: str, args: dict[str, Any]) -> ToolResult:
    db, amount, due = env.db, args["amount"], args["due_date"]
    problems = policy.check_promise(
        amount, due, store.get_case(db, case_id)["balance"], env.clock.now().date(), store.open_promise(db, case_id),
        store.active_plan(db, case_id) is not None, env.policy,
    )
    if problems:
        return _rejected(problems)
    promise_id = db.execute(
        "INSERT INTO promises (case_id, amount, due_date, status, created_at, balance_at_promise) "
        "VALUES (?, ?, ?, 'open', ?, ?)",
        (case_id, amount, due, env.clock.now().isoformat(), store.get_case(db, case_id)["balance"]),
    ).lastrowid
    content = {"promise_id": promise_id, "amount": amount, "due_date": due, "status": "open",
               "next_step": "Confirm it to the merchant with send_message."}
    return ToolResult(content, f"Promise: {labels.money(amount)} by {labels.day(due)}.", fact=f"promise:{promise_id}")


def propose_plan(env: "Env", case_id: str, args: dict[str, Any]) -> ToolResult:
    db, now, message = env.db, env.clock.now(), args["message"].strip()
    items = [{"amount": item["amount"], "due_date": item["due_date"]} for item in args["installments"]]
    problems, needs_approval, why = policy.check_plan(items, store.get_case(db, case_id)["balance"], now.date(),
                                                      env.policy)
    if store.active_plan(db, case_id):
        problems.append("A plan is already active. If it must change, use escalate_to_human.")
    problems += policy.check_message(message, _last_assistant_text(env, case_id), env.policy)
    if problems:
        return _rejected(problems)
    installments = [{"n": n, **item, "paid": False} for n, item in enumerate(items, 1)]
    with store.tx(db):
        db.execute("UPDATE plans SET status = 'superseded' WHERE case_id = ? AND status IN ('offered', "
                   "'pending_approval')", (case_id,))
        plan_id = db.execute(
            "INSERT INTO plans (case_id, status, installments_json, message, needs_approval, why, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (case_id, "pending_approval" if needs_approval else "offered", json.dumps(installments), message,
             int(needs_approval), why, now.isoformat()),
        ).lastrowid
        if needs_approval:
            store.add_message(db, case_id, "system", "notice", labels.NOTICE_PLAN_REVIEW, now)
            transition(env, case_id, State.NEEDS_APPROVAL, Actor.AGENT, labels.STOP_LABELS["needs_approval"])
        else:
            store.add_message(db, case_id, "assistant", "plan_offer", message, now, plan_id=plan_id)
            transition(env, case_id, idle_state(env, case_id), Actor.AGENT, labels.STOP_LABELS["plan_offered"])
    parts = " + ".join(f"{labels.money(i['amount'])} on {labels.day(i['due_date'])}" for i in installments)
    where = "sent to a manager for approval" if needs_approval else "offered to the merchant"
    return ToolResult(
        {"plan_id": plan_id, "status": "pending_approval" if needs_approval else "offered",
         "needs_approval": needs_approval, "why": why},
        f"Plan {parts}: {where}.", fact=f"plan:{plan_id}",
        ends_run="needs_approval" if needs_approval else "plan_offered",
    )


def send_message(env: "Env", case_id: str, args: dict[str, Any]) -> ToolResult:
    db, now, text = env.db, env.clock.now(), args["text"].strip()
    problems = policy.check_message(text, _last_assistant_text(env, case_id), env.policy)
    if problems:
        return _rejected(problems)
    with store.tx(db):
        message_id = store.add_message(db, case_id, "assistant", "text", text, now)
        db.execute("UPDATE cases SET messages_sent = messages_sent + 1 WHERE id = ?", (case_id,))
        transition(env, case_id, idle_state(env, case_id), Actor.AGENT, labels.STOP_LABELS["waiting_on_merchant"])
    short = text if len(text) <= 120 else text[:117] + "..."
    return ToolResult({"sent": True, "message_id": message_id}, f"Sent: \"{short}\"", ends_run="waiting_on_merchant")


def escalate_to_human(env: "Env", case_id: str, args: dict[str, Any]) -> ToolResult:
    reason = args["reason"]
    escalate(env, case_id, "escalated", reason, Actor.AGENT, escalation_reason=f"AI: {reason}")
    return ToolResult({"escalated": True}, f"Handed to a person: {reason}", ends_run="escalated")


TOOLS: dict[str, Tool] = {
    "check_payments": Tool(
        "Ask the payment provider for this case's balance, payments applied to it, transfers the bank has not "
        "confirmed yet (pending), and the merchant's payments from the last 90 days. Use it whenever the merchant "
        "says they paid.",
        _schema(), check_payments),
    "get_sales_summary": Tool(
        "Get the store's sales for the last 6 months, the change between the last 3 months and the 3 before, and "
        "the balance as a percentage of a month's sales. Use it when the merchant says sales are down or asks for "
        "more time.",
        _schema(), get_sales_summary),
    "record_promise": Tool(
        "Record the merchant's commitment to pay an amount by a date (at most 14 days ahead, one open promise at a "
        "time, not while on a plan). Then confirm it with send_message.",
        _schema(amount=AMOUNT, due_date=DATE), record_promise),
    "propose_plan": Tool(
        "Offer a payment plan: 2-3 installments that add up exactly to the balance, the first due within 7 days and "
        "the last within 60 days. Code checks the policy; more than 2 installments or a last date beyond 30 days "
        "needs a manager's approval. message is what the merchant reads. Ends the run.",
        _schema(
            installments={"type": "array", "minItems": 1, "maxItems": 6, "items": {
                "type": "object", "properties": {"amount": AMOUNT, "due_date": DATE},
                "required": ["amount", "due_date"], "additionalProperties": False}},
            message={"type": "string", "minLength": 1, "maxLength": 600},
        ),
        propose_plan),
    "send_message": Tool(
        "Send the merchant a short, friendly message in their language (Arabic or English), with exact amounts and "
        "dates. Ends the run.",
        _schema(text={"type": "string", "minLength": 1, "maxLength": 1000}), send_message),
    "escalate_to_human": Tool(
        "Hand the case to a person: a dispute that payments don't explain, serious hardship, abuse or lawyers, a "
        "request outside policy (discounts, waivers), or when you are unsure. Ends the run.",
        _schema(), escalate_to_human),
}
VALIDATORS = {name: Draft202012Validator(tool.schema) for name, tool in TOOLS.items()}


def definitions() -> list[dict[str, Any]]:
    return [{"name": name, "description": tool.description, "parameters": tool.schema}
            for name, tool in TOOLS.items()]


def validate(name: str, args: Any) -> str | None:
    """None when the call may run; otherwise what is wrong, in words the AI can act on."""
    if name not in TOOLS:
        return f"Unknown tool '{name}'. Use one of: {', '.join(TOOLS)}."
    error = best_match(VALIDATORS[name].iter_errors(args))
    if error is None:
        return None
    where = "/".join(str(part) for part in error.absolute_path)
    return f"Invalid arguments for {name}: {where + ': ' if where else ''}{error.message}"


def execute(env: "Env", case_id: str, name: str, args: dict[str, Any]) -> ToolResult:
    """Run a validated call. Raises ToolUnavailable when an external system stays down."""
    return TOOLS[name].run(env, case_id, _whole_numbers(args))


def _whole_numbers(value: Any) -> Any:
    """JSON numbers may arrive as 9000.0; the schema accepts them as integers, and money must be int."""
    if isinstance(value, dict):
        return {key: _whole_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_whole_numbers(item) for item in value]
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _call_service(env: "Env", service: str, fetch: Callable, case_id: str) -> tuple[dict[str, Any], int]:
    """Call a simulated system, retrying with exponential backoff. Returns (result, attempts used)."""
    limits = env.limits
    for attempt in range(1, limits.max_tool_attempts + 1):
        try:
            return fetch(env, case_id), attempt
        except services.ServiceDown:
            if attempt < limits.max_tool_attempts:
                env.sleep(limits.tool_backoff_seconds * 2 ** (attempt - 1))
    raise ToolUnavailable(service, limits.max_tool_attempts)


def _remember(env: "Env", case_id: str, key: str, result: dict[str, Any]) -> None:
    env.db.execute(
        "INSERT OR REPLACE INTO findings (case_id, key, result_json, created_at) VALUES (?, ?, ?, ?)",
        (case_id, key, json.dumps(result, ensure_ascii=False), env.clock.now().isoformat()),
    )


def _last_assistant_text(env: "Env", case_id: str) -> str | None:
    row = store.one(env.db, "SELECT body FROM messages WHERE case_id = ? AND sender = 'assistant' "
                            "ORDER BY id DESC LIMIT 1", case_id)
    return row["body"] if row else None


def _rejected(problems: list[str]) -> ToolResult:
    content = {"error": "Blocked by policy. Nothing was changed. Fix the call and try again.", "problems": problems}
    return ToolResult(content, "Blocked: " + " ".join(problems), outcome="rejected")
