"""Small helpers so each test reads like the story it checks."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from repayment import db as store
from repayment.clock import SimClock
from repayment.env import Env, build_env
from repayment.events import Event, EventType, Outcome, handle_event
from repayment.llm import ScriptedLLM

START = datetime(2026, 11, 8, 14, 0, tzinfo=timezone.utc)  # a Sunday
TODAY = START.date()


def make_env(*turns: Any, **overrides: Any) -> Env:
    """A seeded in-memory app whose AI plays back `turns`. Backoff sleeps are skipped."""
    return build_env(ScriptedLLM(list(turns)), SimClock(START), sleep=lambda seconds: None, **overrides)


def on(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat()


def say(text: str = "Hi, your installment is overdue. Can you pay this week?") -> tuple[str, dict]:
    return ("send_message", {"text": text, "reason": "Tell the merchant what is owed."})


def call(tool: str, reason: str = "This is the right next step.", **args: Any) -> tuple[str, dict]:
    return (tool, {**args, "reason": reason})


def plan(*installments: tuple[int, int], message: str = "Here is a plan. Tap Accept to confirm.") -> tuple[str, dict]:
    """plan((9000, 7), (9000, 30)) -> propose_plan with amounts due in 7 and 30 days."""
    items = [{"amount": amount, "due_date": on(days)} for amount, days in installments]
    return ("propose_plan", {"installments": items, "message": message, "reason": "Offer a plan within policy."})


def deliver(env: Env, case_id: str, event_type: EventType, payload: dict | None = None,
            event_id: str | None = None) -> Outcome:
    event_id = event_id or f"{event_type.lower()}:{case_id}:{uuid.uuid4().hex[:8]}"
    return handle_event(env, case_id, Event(event_id, event_type, payload or {}))


def start(env: Env, case_id: str) -> Outcome:
    return deliver(env, case_id, EventType.CASE_OPENED, event_id=f"start:{case_id}")


def reply(env: Env, case_id: str, text: str) -> Outcome:
    return deliver(env, case_id, EventType.MERCHANT_REPLY, {"text": text})


def pay(env: Env, case_id: str, amount: int, reference: str | None = None) -> Outcome:
    reference = reference or f"pay-{uuid.uuid4().hex[:8]}"
    return deliver(env, case_id, EventType.PAYMENT_RECEIVED,
                   {"reference": reference, "amount": amount, "source": "bank_transfer"})


def case(env: Env, case_id: str) -> Any:
    return store.get_case(env.db, case_id)


def plans(env: Env, case_id: str) -> list[Any]:
    return store.rows(env.db, "SELECT * FROM plans WHERE case_id = ? ORDER BY id", case_id)


def messages(env: Env, case_id: str) -> list[Any]:
    return store.rows(env.db, "SELECT * FROM messages WHERE case_id = ? ORDER BY id", case_id)


def steps(env: Env, run_id: str | None = None) -> list[Any]:
    if run_id is None:
        return store.rows(env.db, "SELECT * FROM steps ORDER BY rowid")
    return store.rows(env.db, "SELECT * FROM steps WHERE run_id = ? ORDER BY n", run_id)


def llm_calls(env: Env) -> int:
    return len(env.llm.requests)


def case_file_of(env: Env, request: int = -1) -> str:
    """The case file the AI saw on a given call."""
    return env.llm.requests[request][0]["text"]
