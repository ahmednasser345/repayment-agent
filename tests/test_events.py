"""The outer loop: duplicate and stale events, closed cases, crash recovery and timers."""

import json

import pytest

from helpers import (START, call, case, deliver, llm_calls, make_env, messages, on, pay, plan, plans, reply, say,
                     start)
from repayment import db as store
from repayment.events import Event, EventType, due_events, handle_event, tick
from repayment.states import Actor, State, transition


def test_the_same_event_delivered_twice_is_handled_once():
    env = make_env([say()])

    first, second = start(env, "case-nile"), start(env, "case-nile")

    assert first.decision == "ran_agent"
    assert (second.decision, second.detail) == ("skipped", "duplicate_event")
    assert llm_calls(env) == 1
    rows = store.rows(env.db, "SELECT status, decision FROM events WHERE event_id = 'start:case-nile' ORDER BY seq")
    assert [tuple(row) for row in rows] == [("done", "ran_agent"), ("done", "skipped")]  # each delivery is visible


def test_a_payment_reference_seen_twice_is_applied_once():
    env = make_env()
    pay(env, "case-nile", 5_000, reference="pay-aaaa1111")

    again = pay(env, "case-nile", 5_000, reference="pay-aaaa1111")  # a new event id, the same bank reference

    assert (again.decision, again.detail) == ("skipped", "duplicate_payment")
    assert case(env, "case-nile")["balance"] == 7_000


def test_replaying_every_event_of_a_case_changes_nothing():
    env = make_env([say()], [call("get_sales_summary")], [plan((9_000, 7), (9_000, 30))])
    start(env, "case-cairo")
    reply(env, "case-cairo", "Sales are slow. Can I pay in two parts?")
    deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": plans(env, "case-cairo")[0]["id"]})
    pay(env, "case-cairo", 9_000)
    before = snapshot(env, "case-cairo")
    delivered = store.rows(env.db, "SELECT event_id, type, payload_json FROM events ORDER BY seq")

    again = [handle_event(env, "case-cairo", Event(row["event_id"], row["type"], json.loads(row["payload_json"])))
             for row in delivered]

    assert [(o.decision, o.detail) for o in again] == [("skipped", "duplicate_event")] * 4
    assert snapshot(env, "case-cairo") == before  # no second run, plan activation or payment
    assert llm_calls(env) == 3


def snapshot(env, case_id: str) -> list[list[tuple]]:
    """Every row an event can change for this case."""
    queries = ["SELECT * FROM cases WHERE id = ?", "SELECT * FROM plans WHERE case_id = ?",
               "SELECT * FROM messages WHERE case_id = ?", "SELECT * FROM payments WHERE case_id = ?",
               "SELECT * FROM promises WHERE case_id = ?", "SELECT * FROM runs WHERE case_id = ?"]
    return [[tuple(row) for row in store.rows(env.db, query, case_id)] for query in queries]


def test_a_reminder_is_stale_once_the_merchant_has_replied():
    env = make_env([say()], [say("Thanks for letting me know!")])
    start(env, "case-nile")
    env.clock.advance(hours=48)
    [(case_id, reminder)] = due_events(env)

    reply(env, "case-nile", "Sorry, paying tomorrow.")
    outcome = handle_event(env, case_id, reminder)

    assert (outcome.decision, outcome.detail) == ("skipped", "stale")
    assert llm_calls(env) == 2


def test_events_for_a_paid_case_are_ignored_but_the_merchant_message_is_kept():
    env = make_env()
    pay(env, "case-redsea", 6_000)

    outcome = reply(env, "case-redsea", "Did you get my payment?")

    assert (outcome.decision, outcome.detail) == ("skipped", "case_closed")
    assert messages(env, "case-redsea")[-1]["body"] == "Did you get my payment?"
    assert start(env, "case-redsea").detail == "case_closed"
    assert llm_calls(env) == 0


def test_a_reply_to_an_escalated_case_is_saved_for_the_person_and_the_ai_stays_off():
    env = make_env([say()], [call("escalate_to_human", reason="The merchant mentions a lawyer.")])
    start(env, "case-giza")
    reply(env, "case-giza", "Talk to my lawyer.")

    outcome = reply(env, "case-giza", "Hello? Anyone there?")

    assert (outcome.decision, outcome.detail) == ("stored_for_human", "human_owns_case")
    assert llm_calls(env) == 2
    assert messages(env, "case-giza")[-1]["body"] == "Hello? Anyone there?"


def test_a_crash_during_a_run_is_recovered_when_the_event_is_delivered_again():
    env = make_env([say()])
    # What a killed process leaves behind: a delivery still 'processing' and a case stuck in WORKING.
    env.db.execute("INSERT INTO events (event_id, case_id, type, payload_json, status, received_at) "
                   "VALUES ('start:case-nile', 'case-nile', 'CASE_OPENED', '{}', 'processing', ?)",
                   (START.isoformat(),))
    transition(env, "case-nile", State.WORKING, Actor.SYSTEM, "run started")
    env.db.execute("INSERT INTO runs (id, case_id, trigger, started_at, status) "
                   "VALUES ('run-crashed', 'case-nile', 'CASE_OPENED', ?, 'active')", (START.isoformat(),))

    outcome = start(env, "case-nile")

    assert (outcome.decision, outcome.stop_reason) == ("ran_agent", "waiting_on_merchant")
    runs = store.rows(env.db, "SELECT id, status FROM runs ORDER BY rowid")
    assert [run["status"] for run in runs] == ["abandoned", "finished"]
    recovered = store.one(env.db, "SELECT * FROM transitions WHERE from_state = 'WORKING' ORDER BY id LIMIT 1")
    assert (recovered["to_state"], recovered["actor"]) == ("WAITING", "system")
    assert case(env, "case-nile")["active_run_id"] is None


def test_a_case_stuck_working_is_recovered_by_any_next_event():
    env = make_env()
    transition(env, "case-alex", State.WORKING, Actor.SYSTEM, "run started")

    outcome = pay(env, "case-alex", 1_000)

    assert outcome.decision == "handled_by_code"
    assert (case(env, "case-alex")["state"], case(env, "case-alex")["balance"]) == ("WAITING", 7_500)


def test_timers_have_fixed_ids_so_a_second_tick_fires_nothing():
    env = make_env([say()], [say("A friendly reminder.")])
    start(env, "case-nile")
    first_message = messages(env, "case-nile")[-1]["id"]
    env.clock.advance(hours=23)
    assert due_events(env) == []

    env.clock.advance(hours=1)

    assert [event.id for _, event in due_events(env)] == [f"reminder:case-nile:{first_message}"]
    assert [o.decision for o in tick(env)] == ["ran_agent"]
    assert tick(env) == []


# --- failures and crashes ---------------------------------------------------------------------------------------

class Crash(BaseException):
    """Stands in for the process dying in the middle of an AI run (not an Exception, so nothing catches it)."""


def crash_then_tick_again(env) -> list:
    with pytest.raises(Crash):
        tick(env)
    env.clock.advance(minutes=1)  # the cron's next tick
    return tick(env)


def test_a_missed_payment_whose_run_crashed_is_counted_once_when_the_timer_fires_again():
    env = make_env([say()], [call("get_sales_summary")], [plan((9_000, 3), (9_000, 20))], Crash(),
                   [say("Your 9,000 EGP payment was due on Wed 11 Nov. Can you pay it today?")])
    start(env, "case-cairo")
    reply(env, "case-cairo", "Sales are slow. Can I pay in two parts?")
    deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": plans(env, "case-cairo")[-1]["id"]})
    env.clock.advance(days=4)

    [again] = crash_then_tick_again(env)

    assert (again.type, again.decision) == ("INSTALLMENT_DUE", "ran_agent")
    cairo = case(env, "case-cairo")
    assert (cairo["state"], cairo["missed_installments"]) == ("ON_PLAN", 1)
    assert plans(env, "case-cairo")[-1]["status"] == "active"


def test_a_broken_promise_whose_run_crashed_is_followed_up_on_the_next_tick():
    env = make_env([say()], [call("record_promise", amount=6_000, due_date=on(2)), say("Noted: 6,000 EGP by Tue.")],
                   Crash(), [say("The 6,000 EGP you promised for Tue 10 Nov has not arrived. Can you pay today?")])
    start(env, "case-nile")
    reply(env, "case-nile", "I'll pay 6,000 by Tuesday.")
    env.clock.advance(days=3)

    [again] = crash_then_tick_again(env)

    assert (again.type, again.decision) == ("PROMISE_DUE", "ran_agent")
    nile = case(env, "case-nile")
    assert (nile["state"], nile["broken_promises"]) == ("WAITING", 1)
    assert "has not arrived" in messages(env, "case-nile")[-1]["body"]


def test_a_reminder_whose_run_crashed_is_sent_on_the_next_tick_and_counted_once():
    env = make_env([say()], Crash(), [say("A friendly reminder: 12,000 EGP is still due.")])
    start(env, "case-nile")
    env.clock.advance(hours=49)

    [again] = crash_then_tick_again(env)

    assert (again.type, again.decision) == ("REMINDER_DUE", "ran_agent")
    assert (case(env, "case-nile")["state"], case(env, "case-nile")["reminders_sent"]) == ("WAITING", 1)
    rows = store.rows(env.db, "SELECT status FROM events WHERE type = 'REMINDER_DUE' ORDER BY seq")
    assert [row["status"] for row in rows] == ["interrupted", "done"]


def test_an_event_that_fails_is_not_marked_done_so_a_corrected_delivery_still_counts():
    env = make_env()
    bad = {"reference": "pay-0000beef", "amount": "12k", "source": "bank_transfer"}
    with pytest.raises(ValueError):
        deliver(env, "case-nile", EventType.PAYMENT_RECEIVED, bad, event_id="payment:pay-0000beef")

    row = store.one(env.db, "SELECT status, decision FROM events WHERE event_id = 'payment:pay-0000beef'")
    assert tuple(row) == ("failed", "failed")
    assert case(env, "case-nile")["state"] == "ESCALATED"  # a person looks at a case whose event failed

    fixed = deliver(env, "case-nile", EventType.PAYMENT_RECEIVED, {**bad, "amount": 12_000},
                    event_id="payment:pay-0000beef")

    assert fixed.decision == "handled_by_code"
    assert (case(env, "case-nile")["state"], case(env, "case-nile")["balance"]) == ("RESOLVED", 0)


def test_one_case_failing_does_not_stop_the_timers_of_the_others():
    env = make_env([say()], [say()], RuntimeError("Gemini 400: a bug on our side"),
                   [say("A friendly reminder: 18,000 EGP is still due.")])
    start(env, "case-nile")
    start(env, "case-cairo")
    env.clock.advance(hours=48)

    outcomes = tick(env)

    assert [(o.case_id, o.decision) for o in outcomes] == [("case-nile", "failed"), ("case-cairo", "ran_agent")]
    assert case(env, "case-nile")["state"] == "ESCALATED"
    assert case(env, "case-cairo")["reminders_sent"] == 1
