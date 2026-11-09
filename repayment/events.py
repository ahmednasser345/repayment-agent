"""The outer loop. Every event goes through handle_event(), which decides whether the AI runs again.

Cases rest in a waiting state between runs and are retriggered by events: merchant replies, payments, timers,
manager decisions and outage retries. Money events are handled by code only; the AI never moves money.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

from . import db as store
from . import labels, policy
from .agent import Trigger, run_agent
from .observability import log
from .states import HUMAN_OWNED, Actor, State, escalate, idle_state, state_of, transition

if TYPE_CHECKING:
    from .env import Env


class EventType(StrEnum):
    CASE_OPENED = "CASE_OPENED"
    MERCHANT_REPLY = "MERCHANT_REPLY"
    PAYMENT_RECEIVED = "PAYMENT_RECEIVED"
    PLAN_ACCEPTED = "PLAN_ACCEPTED"
    PLAN_DECLINED = "PLAN_DECLINED"
    REMINDER_DUE = "REMINDER_DUE"
    DUE_TODAY = "DUE_TODAY"
    PROMISE_DUE = "PROMISE_DUE"
    INSTALLMENT_DUE = "INSTALLMENT_DUE"
    HUMAN_DECISION = "HUMAN_DECISION"
    TOOL_RETRY = "TOOL_RETRY"
    DEADLINE_PASSED = "DEADLINE_PASSED"


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    event_id: str
    type: str
    case_id: str
    decision: str  # ran_agent | handled_by_code | stored_for_human | skipped | escalated | failed
    detail: str
    stop_reason: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class Result:
    decision: str
    detail: str
    stop_reason: str | None = None
    run_id: str | None = None


def handle_event(env: "Env", case_id: str, event: Event) -> Outcome:
    """The single entry point: idempotent, recovers from a crash, and routes each event type."""
    store.get_case(env.db, case_id)  # unknown case -> UnknownCase
    if store.delivered(env.db, event.id):
        seq = _record_delivery(env, case_id, event)
        return _close(env, seq, case_id, event, Result("skipped", "duplicate_event"))
    seq = _record_delivery(env, case_id, event)  # left as 'processing' if the process dies mid-way
    try:
        _recover_from_crash(env, case_id)
        if state_of(env, case_id) == State.RESOLVED:
            if event.type == EventType.MERCHANT_REPLY:
                _store_merchant_text(env, case_id, event.payload.get("text", ""))
            result = Result("skipped", "case_closed")
        else:
            result = HANDLERS[EventType(event.type)](env, case_id, event)
    except Exception as exc:
        # Not marked 'done', so a corrected delivery of the same event still counts. A person checks the case.
        _escalate_quietly(env, case_id, f"{type(exc).__name__}: {exc}"[:300])
        _close(env, seq, case_id, event, Result("failed", "internal_error", "internal_error"), status="failed")
        raise
    return _close(env, seq, case_id, event, result)


def due_events(env: "Env", now: datetime | None = None) -> list[tuple[str, Event]]:
    """Timers, computed from the database at the clock's current time (or at `now`, to look ahead). Ids are
    deterministic, so nothing fires twice; events already handled are left out."""
    now = now or env.clock.now()
    found = []
    for case in store.rows(env.db, "SELECT * FROM cases WHERE state != 'RESOLVED' ORDER BY rowid"):
        events = [event for event in _timers(env, case, now) if not store.delivered(env.db, event.id)]
        if not events:  # a payment-day reminder only when nothing else is due for the case
            events = [event for event in _payment_day(env, case, now) if not store.delivered(env.db, event.id)]
        found += [(case["id"], event) for event in events]
    return found


def next_due(env: "Env", days: int = 14) -> tuple[int, str, Event] | None:
    """What "Let a day pass" would fire first if nothing else happens: (days from now, case id, event)."""
    now = env.clock.now()
    for ahead in range(1, days + 1):
        if found := due_events(env, now + timedelta(days=ahead)):
            case_id, event = found[0]
            return ahead, case_id, event
    return None


def _timers(env: "Env", case: Any, now: datetime) -> list[Event]:
    case_id, state, today = case["id"], case["state"], now.date().isoformat()
    if state not in HUMAN_OWNED and now >= datetime.fromisoformat(case["deadline_at"]):
        return [Event(f"deadline:{case_id}", EventType.DEADLINE_PASSED)]
    found = []
    if case["retry_at"] and datetime.fromisoformat(case["retry_at"]) <= now:
        found.append(retry_event(env, case))
    promise = store.open_promise(env.db, case_id)
    if promise and promise["due_date"] < today:
        found.append(Event(f"promise:{promise['id']}", EventType.PROMISE_DUE, {"promise_id": promise["id"]}))
    if plan := store.active_plan(env.db, case_id):
        for item in labels.installments(plan):
            if not item["paid"] and item["due_date"] < today:
                found.append(Event(f"installment:{plan['id']}:{item['n']}", EventType.INSTALLMENT_DUE,
                                   {"plan_id": plan["id"], "n": item["n"]}))
    last = store.last_message(env.db, case_id)
    if (state == State.WAITING and not promise and not case["retry_at"] and last
            and last["sender"] != "merchant"
            and _hours_since(last["created_at"], now) >= env.limits.reminder_after_hours):
        found.append(Event(f"reminder:{case_id}:{last['id']}", EventType.REMINDER_DUE,
                           {"last_message_id": last["id"]}))
    return found


def _payment_day(env: "Env", case: Any, now: datetime) -> list[Event]:
    """A reminder on the day a promised or plan payment is due, if it was agreed on an earlier day. Not while a
    person owns the case or an outage has paused it."""
    if case["state"] in HUMAN_OWNED or case["retry_at"]:
        return []
    case_id, today = case["id"], now.date().isoformat()
    promise = store.open_promise(env.db, case_id)
    if promise and promise["due_date"] == today and _day_of(promise["created_at"]) < today:
        return [Event(f"due-today:promise:{promise['id']}", EventType.DUE_TODAY, {"promise_id": promise["id"]})]
    plan = store.active_plan(env.db, case_id)
    if plan and plan["activated_at"] and _day_of(plan["activated_at"]) < today:
        for item in labels.installments(plan):
            if not item["paid"] and item["due_date"] == today:
                return [Event(f"due-today:installment:{plan['id']}:{item['n']}", EventType.DUE_TODAY,
                              {"plan_id": plan["id"], "n": item["n"]})]
    return []


def tick(env: "Env") -> list[Outcome]:
    """Fire every due timer. In production a cron calls this every minute.

    Events are handled one at a time, so a case found WORKING here was left by a crash: it goes back to idle, and
    a timer event the crash cut short is delivered again. One failing case does not stop the others."""
    for case in store.rows(env.db, "SELECT id FROM cases WHERE state = 'WORKING'"):
        _recover_from_crash(env, case["id"])
    outcomes = [_fire(env, case_id, event) for case_id, event in _interrupted_timers(env)]
    return outcomes + [_fire(env, case_id, event) for case_id, event in due_events(env)]


def retry_event(env: "Env", case: Any) -> Event:
    """A retry is named after the run that paused (the case's latest run). The timer and the "Retry now" button
    share the id, so a retry runs once; a later outage pauses another run and gets a new id."""
    run = store.one(env.db, "SELECT id FROM runs WHERE case_id = ? ORDER BY rowid DESC LIMIT 1", case["id"])
    return Event(f"retry:{case['id']}:{run['id'] if run else 'none'}", EventType.TOOL_RETRY)


# --- handlers: one per event type -------------------------------------------------------------------------------

def _case_opened(env: "Env", case_id: str, event: Event) -> Result:
    # "Started" means a run finished: if the app died during the first run, the redelivered event must still run.
    finished = store.one(env.db, "SELECT 1 FROM runs WHERE case_id = ? AND status = 'finished'", case_id)
    if state_of(env, case_id) != State.NEW and finished:
        return Result("skipped", "already_started")
    return _run(env, case_id, Trigger(event.type, event.id))


def _merchant_reply(env: "Env", case_id: str, event: Event) -> Result:
    _store_merchant_text(env, case_id, event.payload.get("text", ""))
    return _run(env, case_id, Trigger(event.type, event.id))


def _payment_received(env: "Env", case_id: str, event: Event) -> Result:
    """Code only: apply the money, settle promise and plan, send a receipt, close the case at 0."""
    db, now, payload = env.db, env.clock.now(), event.payload
    reference, amount = payload["reference"], int(payload["amount"])
    existing = store.one(db, "SELECT * FROM payments WHERE reference = ?", reference)
    if existing is not None and existing["status"] == "applied":
        return Result("skipped", "duplicate_payment")
    case = store.get_case(db, case_id)
    balance = max(0, case["balance"] - amount)
    with store.tx(db):
        if existing is not None:  # the bank confirmed a pending transfer
            db.execute("UPDATE payments SET status = 'applied', amount = ?, case_id = ?, received_at = ?, "
                       "note = 'confirmed by the bank' WHERE id = ?",
                       (amount, case_id, now.isoformat(), existing["id"]))
        else:
            db.execute(
                "INSERT INTO payments (merchant_id, case_id, reference, amount, status, source, note, received_at) "
                "VALUES (?, ?, ?, ?, 'applied', ?, NULL, ?)",
                (case["merchant_id"], case_id, reference, amount, payload.get("source", "bank_transfer"),
                 now.isoformat()))
        store.update_case(db, case_id, now, balance=balance)
        _settle_promise(env, case_id)
        _mark_installments_paid(env, case_id)
        store.add_message(db, case_id, "system", "receipt", labels.receipt(amount, balance), now)
        if balance == 0:
            _close_as_resolved(env, case_id)
    detail = f"Applied {labels.money(amount)}; the balance is now {labels.money(balance)}."
    return Result("handled_by_code", detail, "resolved" if balance == 0 else None)


def _plan_accepted(env: "Env", case_id: str, event: Event) -> Result:
    plan, now = _open_offer(env, case_id, event.payload.get("plan_id")), env.clock.now()
    if plan is None:
        return Result("skipped", "stale_offer")
    if _plan_problems(env, case_id, plan):
        with store.tx(env.db):
            env.db.execute("UPDATE plans SET status = 'outdated' WHERE id = ?", (plan["id"],))
            store.add_message(env.db, case_id, "system", "notice", labels.NOTICE_OFFER_OUTDATED, now)
        return Result("skipped", "stale_offer")
    first = labels.next_installment(plan)
    with store.tx(env.db):
        env.db.execute("UPDATE plans SET status = 'active', activated_at = ? WHERE id = ?",
                       (now.isoformat(), plan["id"]))
        # The plan replaces an open promise; otherwise that promise would later count as broken.
        env.db.execute("UPDATE promises SET status = 'superseded' WHERE case_id = ? AND status = 'open'", (case_id,))
        transition(env, case_id, State.ON_PLAN, Actor.SYSTEM, labels.TRIGGER_LABELS[event.type])
        store.add_message(env.db, case_id, "system", "notice",
                          labels.plan_confirmed(first["amount"], first["due_date"]), now)
    return Result("handled_by_code", "The plan is now active.")


def _plan_declined(env: "Env", case_id: str, event: Event) -> Result:
    plan = _open_offer(env, case_id, event.payload.get("plan_id"))
    if plan is None:
        return Result("skipped", "stale_offer")
    with store.tx(env.db):
        env.db.execute("UPDATE plans SET status = 'declined' WHERE id = ?", (plan["id"],))
        if text := (event.payload.get("text") or "").strip():
            _store_merchant_text(env, case_id, text)
    return _run(env, case_id, Trigger(event.type, event.id, {"plan_id": plan["id"]}))


def _reminder_due(env: "Env", case_id: str, event: Event) -> Result:
    case, last, limits = store.get_case(env.db, case_id), store.last_message(env.db, case_id), env.limits
    still_silent = (
        case["state"] == State.WAITING and last is not None and last["sender"] != "merchant"
        and last["id"] == event.payload.get("last_message_id") and not case["retry_at"]
        and store.open_promise(env.db, case_id) is None
        and _hours_since(last["created_at"], env.clock.now()) >= limits.reminder_after_hours
    )
    if not still_silent:
        return Result("skipped", "stale")
    counted = case["counted_event_id"] == event.id  # delivered again after a crash: this reminder is counted
    sent = case["reminders_sent"] + (0 if counted else 1)
    if sent > limits.max_reminders:
        return _cap(env, case_id, "cap:reminders", f"{case['reminders_sent']} reminders sent, no reply")
    store.update_case(env.db, case_id, env.clock.now(), reminders_sent=sent, counted_event_id=event.id)
    return _run(env, case_id, Trigger(event.type, event.id, {"reminder": sent, "of": limits.max_reminders}))


def _due_today(env: "Env", case_id: str, event: Event) -> Result:
    """One friendly reminder on the payment day. Nothing is counted: if the day passes unpaid, PROMISE_DUE or
    INSTALLMENT_DUE takes over."""
    case = store.get_case(env.db, case_id)
    details = _still_due_today(env, case_id, event.payload)
    if details is None or case["state"] in HUMAN_OWNED or case["retry_at"]:
        return Result("skipped", "stale")  # paid, kept, replaced, no longer today, with a person, or paused
    return _run(env, case_id, Trigger(event.type, event.id, details))


def _still_due_today(env: "Env", case_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    today = _today(env)
    if "promise_id" in payload:
        promise = store.one(env.db, "SELECT * FROM promises WHERE id = ? AND case_id = ?", payload["promise_id"],
                            case_id)
        if promise is None or promise["status"] != "open" or promise["due_date"] != today:
            return None
        return {"promise": {"amount": promise["amount"], "due_date": promise["due_date"]}}
    plan = store.get_plan(env.db, case_id, payload.get("plan_id"))
    item = next((i for i in labels.installments(plan) if i["n"] == payload.get("n")), None) if plan else None
    if plan is None or plan["status"] != "active" or item is None or item["paid"] or item["due_date"] != today:
        return None
    return {"plan_id": plan["id"], "installment": item["n"], "amount": item["amount"], "due_date": item["due_date"]}


def _promise_due(env: "Env", case_id: str, event: Event) -> Result:
    promise = store.one(env.db, "SELECT * FROM promises WHERE id = ? AND case_id = ?",
                        event.payload.get("promise_id"), case_id)
    case = store.get_case(env.db, case_id)
    counted = case["counted_event_id"] == event.id  # delivered again after a crash: already marked broken
    if promise is None or promise["due_date"] >= _today(env) or (promise["status"] != "open" and not counted):
        return Result("skipped", "stale")  # kept, replaced by a plan, already handled, or not due yet
    broken = case["broken_promises"] + (0 if counted else 1)
    with store.tx(env.db):
        env.db.execute("UPDATE promises SET status = 'broken' WHERE id = ?", (promise["id"],))
        store.update_case(env.db, case_id, env.clock.now(), broken_promises=broken, counted_event_id=event.id)
    if state_of(env, case_id) in HUMAN_OWNED:
        return Result("stored_for_human", "human_owns_case")
    if broken >= env.limits.max_broken_promises:
        return _cap(env, case_id, "cap:broken_promises", f"{broken} promises broken")
    details = {"promise": {"amount": promise["amount"], "due_date": promise["due_date"]}, "broken_promises": broken}
    return _run(env, case_id, Trigger(event.type, event.id, details))


def _installment_due(env: "Env", case_id: str, event: Event) -> Result:
    plan, n = store.get_plan(env.db, case_id, event.payload.get("plan_id")), event.payload.get("n")
    items = labels.installments(plan) if plan is not None else []
    item = next((i for i in items if i["n"] == n), None)
    case = store.get_case(env.db, case_id)
    counted = case["counted_event_id"] == event.id  # delivered again after a crash: this miss is counted
    if (plan is None or item is None or item["paid"] or item["due_date"] >= _today(env)
            or (plan["status"] != "active" and not counted)):
        return Result("skipped", "stale")
    missed = case["missed_installments"] + (0 if counted else 1)
    with store.tx(env.db):
        store.update_case(env.db, case_id, env.clock.now(), missed_installments=missed, counted_event_id=event.id)
        if missed >= env.limits.max_missed_installments:
            env.db.execute("UPDATE plans SET status = 'broken' WHERE id = ?", (plan["id"],))
    if state_of(env, case_id) in HUMAN_OWNED:
        return Result("stored_for_human", "human_owns_case")
    if missed >= env.limits.max_missed_installments:
        return _cap(env, case_id, "cap:missed_installments", f"{missed} plan payments missed")
    details = {"plan_id": plan["id"], "installment": n, "amount": item["amount"], "due_date": item["due_date"]}
    return _run(env, case_id, Trigger(event.type, event.id, details))


def _human_decision(env: "Env", case_id: str, event: Event) -> Result:
    action, note = event.payload.get("action"), (event.payload.get("note") or "").strip()
    state, now = state_of(env, case_id), env.clock.now()
    if action == "hand_back":
        if state != State.ESCALATED:
            return Result("skipped", "stale")
        store.update_case(env.db, case_id, now, reminders_sent=0, tool_retries=0, retry_at=None,
                          pending_trigger_json=None, handoff_note=note)
        trigger = Trigger(event.type, event.id, {"action": action, "manager_note": note})
        return _run(env, case_id, trigger, Actor.HUMAN)
    pending = store.plans_with_status(env.db, case_id, "pending_approval")
    plan = next((p for p in pending if p["id"] == event.payload.get("plan_id")), pending[0] if pending else None)
    if state != State.NEEDS_APPROVAL or plan is None or action not in ("approve_plan", "reject_plan"):
        return Result("skipped", "stale")
    problems = _plan_problems(env, case_id, plan) if action == "approve_plan" else []
    if action == "approve_plan" and not problems:
        with store.tx(env.db):
            env.db.execute("UPDATE plans SET status = 'offered', decided_by = 'manager', decision_note = ? "
                           "WHERE id = ?", (note, plan["id"]))
            store.add_message(env.db, case_id, "assistant", "plan_offer", plan["message"], now, plan_id=plan["id"])
            transition(env, case_id, State.WAITING, Actor.HUMAN, "A manager approved the plan.")
        return Result("handled_by_code", "A manager approved the plan, so it was offered to the merchant. Waiting "
                      "for them to accept or decline.", "plan_offered")
    # Rejected, or approved after a payment or the passing days made it wrong: the AI proposes again.
    env.db.execute("UPDATE plans SET status = ?, decided_by = 'manager', decision_note = ? WHERE id = ?",
                   ("outdated" if problems else "rejected", note, plan["id"]))
    details = {"action": action, "plan_id": plan["id"], "manager_note": note}
    if problems:
        details["plan_no_longer_fits"] = problems
    return _run(env, case_id, Trigger(event.type, event.id, details), Actor.HUMAN)


def _tool_retry(env: "Env", case_id: str, event: Event) -> Result:
    case = store.get_case(env.db, case_id)
    counted = case["counted_event_id"] == event.id  # delivered again after a crash: the retry already started
    if not (case["retry_at"] or counted) or case["state"] not in (State.WAITING, State.ON_PLAN):
        return Result("skipped", "stale")
    store.update_case(env.db, case_id, env.clock.now(), retry_at=None, counted_event_id=event.id)
    details = {"retry": case["tool_retries"], "of": env.limits.max_tool_retries,
               "original_trigger": json.loads(case["pending_trigger_json"] or "null")}
    return _run(env, case_id, Trigger(event.type, event.id, details))


def _deadline_passed(env: "Env", case_id: str, event: Event) -> Result:
    state = state_of(env, case_id)
    if state == State.ESCALATED:
        return Result("skipped", "stale")
    if state == State.NEEDS_APPROVAL:  # a manager already owns the decision; they can't be overruled by a timer
        return Result("stored_for_human", "human_owns_case")
    deadline = datetime.fromisoformat(store.get_case(env.db, case_id)["deadline_at"])
    if env.clock.now() < deadline:
        return Result("skipped", "stale")
    return _cap(env, case_id, "cap:deadline", f"the case deadline ({labels.day_time(deadline)}) passed")


HANDLERS: dict[EventType, Callable[["Env", str, Event], Result]] = {
    EventType.CASE_OPENED: _case_opened,
    EventType.MERCHANT_REPLY: _merchant_reply,
    EventType.PAYMENT_RECEIVED: _payment_received,
    EventType.PLAN_ACCEPTED: _plan_accepted,
    EventType.PLAN_DECLINED: _plan_declined,
    EventType.REMINDER_DUE: _reminder_due,
    EventType.DUE_TODAY: _due_today,
    EventType.PROMISE_DUE: _promise_due,
    EventType.INSTALLMENT_DUE: _installment_due,
    EventType.HUMAN_DECISION: _human_decision,
    EventType.TOOL_RETRY: _tool_retry,
    EventType.DEADLINE_PASSED: _deadline_passed,
}


# --- helpers ----------------------------------------------------------------------------------------------------

def _run(env: "Env", case_id: str, trigger: Trigger, actor: Actor = Actor.SYSTEM) -> Result:
    """Run the AI, unless a person owns the case. A cap hit before the start escalates without a run."""
    if state_of(env, case_id) in HUMAN_OWNED and actor != Actor.HUMAN:
        return Result("stored_for_human", "human_owns_case")
    stop = run_agent(env, case_id, trigger, actor)
    if stop.run_id is None:
        return Result("escalated", stop.reason, stop.reason)
    return Result("ran_agent", stop.reason, stop.reason, stop.run_id)


def _cap(env: "Env", case_id: str, reason: str, detail: str) -> Result:
    escalate(env, case_id, reason, detail)
    return Result("escalated", reason, reason)


def _escalate_quietly(env: "Env", case_id: str, detail: str) -> None:
    """Best effort after an unexpected error: the original exception matters more than this one."""
    try:
        escalate(env, case_id, "internal_error", detail)
    except Exception:  # noqa: BLE001 - the caller re-raises the original error
        pass


def _fire(env: "Env", case_id: str, event: Event) -> Outcome:
    try:
        return handle_event(env, case_id, event)
    except Exception:  # noqa: BLE001 - recorded as 'failed' and handed to a person; the other timers still run
        return Outcome(event.id, event.type, case_id, "failed", "internal_error", "internal_error")


def _interrupted_timers(env: "Env") -> list[tuple[str, Event]]:
    """Timer events a crash left 'processing'. Other events are sent again by their source (webhook, app)."""
    timers = (EventType.REMINDER_DUE, EventType.DUE_TODAY, EventType.PROMISE_DUE, EventType.INSTALLMENT_DUE,
              EventType.TOOL_RETRY, EventType.DEADLINE_PASSED)
    found = store.rows(env.db, f"SELECT * FROM events WHERE status = 'processing' AND type IN "
                               f"({', '.join('?' for _ in timers)}) ORDER BY seq", *timers)
    return [(row["case_id"], Event(row["event_id"], row["type"], json.loads(row["payload_json"]))) for row in found]


def _record_delivery(env: "Env", case_id: str, event: Event) -> int:
    seq = env.db.execute(
        "INSERT INTO events (event_id, case_id, type, payload_json, status, received_at) "
        "VALUES (?, ?, ?, ?, 'processing', ?)",
        (event.id, case_id, event.type, json.dumps(event.payload, ensure_ascii=False), env.clock.now().isoformat()),
    ).lastrowid
    # An earlier delivery still 'processing' was cut short by a crash; this one takes over.
    env.db.execute("UPDATE events SET status = 'interrupted' WHERE event_id = ? AND status = 'processing' AND seq < ?",
                   (event.id, seq))
    return seq


def _close(env: "Env", seq: int, case_id: str, event: Event, result: Result, status: str = "done") -> Outcome:
    with store.tx(env.db):
        env.db.execute("UPDATE events SET status = ?, decision = ?, detail = ?, run_id = ?, finished_at = ? "
                       "WHERE seq = ?",
                       (status, result.decision, result.detail, result.run_id, env.clock.now().isoformat(), seq))
        log(env, "event", case_id, result.run_id, event_id=event.id, type=event.type, decision=result.decision,
            detail=result.detail, stop_reason=result.stop_reason, payload=event.payload)
    return Outcome(event.id, event.type, case_id, result.decision, result.detail, result.stop_reason, result.run_id)


def _recover_from_crash(env: "Env", case_id: str) -> None:
    """Events are handled one at a time, so a case found WORKING means the app stopped during a run."""
    case = store.get_case(env.db, case_id)
    if case["state"] != State.WORKING:
        return
    now = env.clock.now()
    with store.tx(env.db):
        env.db.execute("UPDATE runs SET status = 'abandoned', stop_reason = 'abandoned', finished_at = ? "
                       "WHERE case_id = ? AND status = 'active'", (now.isoformat(), case_id))
        store.update_case(env.db, case_id, now, active_run_id=None)
        transition(env, case_id, idle_state(env, case_id), Actor.SYSTEM, labels.STOP_LABELS["abandoned"])


def _store_merchant_text(env: "Env", case_id: str, text: str) -> int:
    flag = "instruction_like" if policy.instruction_like(text) else None
    return store.add_message(env.db, case_id, "merchant", "text", text, env.clock.now(), flag=flag)


def _open_offer(env: "Env", case_id: str, plan_id: Any) -> Any:
    plan = store.get_plan(env.db, case_id, plan_id)
    if plan is None or plan["status"] != "offered" or state_of(env, case_id) != State.WAITING:
        return None
    return plan


def _plan_problems(env: "Env", case_id: str, plan: Any) -> list[str]:
    """Policy is checked again when an offer is accepted or approved: since it was proposed, a payment may have
    changed the balance, or its first date may have passed."""
    items = [{"amount": item["amount"], "due_date": item["due_date"]} for item in labels.installments(plan)]
    balance = store.get_case(env.db, case_id)["balance"]
    problems, _, _ = policy.check_plan(items, balance, env.clock.now().date(), env.policy)
    return problems


def _settle_promise(env: "Env", case_id: str) -> None:
    """An open promise is kept once the balance has dropped by its amount since it was made."""
    promise, balance = store.open_promise(env.db, case_id), store.get_case(env.db, case_id)["balance"]
    if promise is not None and promise["balance_at_promise"] - balance >= promise["amount"]:
        env.db.execute("UPDATE promises SET status = 'kept' WHERE id = ?", (promise["id"],))


def _mark_installments_paid(env: "Env", case_id: str) -> None:
    """Installments are paid in order. A plan adds up to the balance when it starts, so the money paid since then
    is the plan's total minus the balance now."""
    plan = store.active_plan(env.db, case_id)
    if plan is None:
        return
    items, covered = labels.installments(plan), 0
    paid = sum(item["amount"] for item in items) - store.get_case(env.db, case_id)["balance"]
    for item in items:
        covered += item["amount"]
        item["paid"] = paid >= covered
    env.db.execute("UPDATE plans SET installments_json = ? WHERE id = ?", (json.dumps(items), plan["id"]))


def _close_as_resolved(env: "Env", case_id: str) -> None:
    db = env.db
    db.execute("UPDATE plans SET status = 'completed' WHERE case_id = ? AND status = 'active'", (case_id,))
    db.execute("UPDATE plans SET status = 'superseded' WHERE case_id = ? AND status IN ('offered', "
               "'pending_approval')", (case_id,))
    db.execute("UPDATE promises SET status = 'kept' WHERE case_id = ? AND status = 'open'", (case_id,))
    store.update_case(db, case_id, env.clock.now(), retry_at=None, pending_trigger_json=None,
                      last_stop_reason="resolved", last_stop_detail="")
    transition(env, case_id, State.RESOLVED, Actor.SYSTEM, labels.STOP_LABELS["resolved"])


def _hours_since(moment: str, now: datetime) -> float:
    return (now - datetime.fromisoformat(moment)) / timedelta(hours=1)


def _today(env: "Env") -> str:
    return env.clock.now().date().isoformat()


def _day_of(moment: str) -> str:
    return datetime.fromisoformat(moment).date().isoformat()
