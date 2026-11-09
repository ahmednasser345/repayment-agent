"""Whole cases, start to finish: payments, plans, approvals, promises and reminders."""

import pytest

from helpers import (call, case, case_file_of, deliver, llm_calls, make_env, messages, on, pay, plan, plans, reply,
                     say, start, steps)
from repayment import db as store
from repayment import labels
from repayment.events import EventType, tick


def test_a_payment_closes_the_case_by_code_without_an_ai_run():
    env = make_env([say()])
    assert start(env, "case-nile").decision == "ran_agent"
    assert case(env, "case-nile")["state"] == "WAITING"

    outcome = pay(env, "case-nile", 12_000)

    assert (outcome.decision, outcome.run_id) == ("handled_by_code", None)
    assert llm_calls(env) == 1
    assert (case(env, "case-nile")["state"], case(env, "case-nile")["balance"]) == ("RESOLVED", 0)
    assert messages(env, "case-nile")[-1]["body"] == ("Payment received: 12,000 EGP. Your balance is now 0 EGP — "
                                                      "thank you! This case is closed.")


def test_a_two_part_plan_is_offered_then_accepted_and_paid_off_by_code():
    env = make_env([say()], [call("get_sales_summary")], [plan((9_000, 7), (9_000, 30))])
    start(env, "case-cairo")

    outcome = reply(env, "case-cairo", "Sales are slow this month. Can I pay in two parts?")

    assert outcome.stop_reason == "plan_offered"
    offer = plans(env, "case-cairo")[-1]
    assert (offer["status"], case(env, "case-cairo")["state"]) == ("offered", "WAITING")
    assert messages(env, "case-cairo")[-1]["kind"] == "plan_offer"

    accepted = deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": offer["id"]})
    assert accepted.decision == "handled_by_code"
    assert case(env, "case-cairo")["state"] == "ON_PLAN"
    assert messages(env, "case-cairo")[-1]["body"] == "Plan confirmed. Your next payment is 9,000 EGP on Sun 15 Nov."

    pay(env, "case-cairo", 9_000)
    assert [i["paid"] for i in labels.installments(plans(env, "case-cairo")[-1])] == [True, False]
    assert case(env, "case-cairo")["state"] == "ON_PLAN"
    pay(env, "case-cairo", 9_000)
    assert case(env, "case-cairo")["state"] == "RESOLVED"
    assert plans(env, "case-cairo")[-1]["status"] == "completed"
    assert llm_calls(env) == 3


def test_a_plan_beyond_the_limits_waits_for_a_manager_who_approves_it():
    env = make_env([say()], [plan((30_000, 7), (30_000, 37), (30_000, 60))])
    start(env, "case-delta")

    outcome = reply(env, "case-delta", "Orders dropped a lot. I need about 3 months to pay this.")

    assert outcome.stop_reason == "needs_approval"
    pending = plans(env, "case-delta")[-1]
    assert (pending["status"], pending["needs_approval"]) == ("pending_approval", 1)
    assert "manager must approve" in pending["why"]
    assert case(env, "case-delta")["state"] == "NEEDS_APPROVAL"
    assert messages(env, "case-delta")[-1]["body"] == labels.NOTICE_PLAN_REVIEW
    assert deliver(env, "case-delta", EventType.PLAN_ACCEPTED, {"plan_id": pending["id"]}).detail == "stale_offer"

    approved = deliver(env, "case-delta", EventType.HUMAN_DECISION, {"action": "approve_plan", "note": "Fine."})

    assert approved.decision == "handled_by_code"
    offer = plans(env, "case-delta")[-1]
    assert (offer["status"], offer["decided_by"], offer["decision_note"]) == ("offered", "manager", "Fine.")
    last = messages(env, "case-delta")[-1]
    assert (last["kind"], last["body"], last["plan_id"]) == ("plan_offer", offer["message"], offer["id"])
    transition = store.one(env.db, "SELECT * FROM transitions WHERE from_state = 'NEEDS_APPROVAL'")
    assert (transition["to_state"], transition["actor"]) == ("WAITING", "human")
    deliver(env, "case-delta", EventType.PLAN_ACCEPTED, {"plan_id": offer["id"]})
    assert case(env, "case-delta")["state"] == "ON_PLAN"


def test_a_manager_rejection_runs_the_ai_again_with_the_note():
    env = make_env([say()], [plan((30_000, 7), (30_000, 37), (30_000, 60))], [plan((45_000, 7), (45_000, 30))])
    start(env, "case-delta")
    reply(env, "case-delta", "I need about 3 months to pay this.")

    outcome = deliver(env, "case-delta", EventType.HUMAN_DECISION,
                      {"action": "reject_plan", "note": "Too long. Offer two payments within 30 days."})

    assert (outcome.decision, outcome.stop_reason) == ("ran_agent", "plan_offered")
    assert '"manager_note": "Too long. Offer two payments within 30 days."' in case_file_of(env)
    assert [p["status"] for p in plans(env, "case-delta")] == ["rejected", "offered"]
    transition = store.one(env.db, "SELECT * FROM transitions WHERE from_state = 'NEEDS_APPROVAL'")
    assert (transition["to_state"], transition["actor"]) == ("WORKING", "human")


@pytest.mark.parametrize(("bad_plan", "problem"), [
    (plan((9_000, 7), (8_000, 30)), "add up to the balance exactly"),
    (plan((4_500, 7), (4_500, 14), (4_500, 21), (4_500, 28)), "2 to 3 installments"),
    (plan((9_000, 10), (9_000, 30)), "first payment must be due between"),
    (plan((18_000, 7)), "use record_promise"),
], ids=["sum-not-balance", "four-installments", "first-date-too-late", "one-installment"])
def test_a_plan_that_breaks_policy_is_blocked_and_the_ai_fixes_it_in_the_same_run(bad_plan, problem):
    env = make_env([say()], [bad_plan], [plan((9_000, 7), (9_000, 30))])
    start(env, "case-cairo")

    outcome = reply(env, "case-cairo", "Can I pay in two parts?")

    assert outcome.stop_reason == "plan_offered"
    run_steps = steps(env, outcome.run_id)
    assert [s["outcome"] for s in run_steps] == ["rejected", "ok"]
    assert problem in run_steps[0]["summary"]
    assert [p["status"] for p in plans(env, "case-cairo")] == ["offered"]  # the blocked call changed nothing
    tool_turn = env.llm.requests[-1][-1]
    assert tool_turn["role"] == "tool" and tool_turn["results"][0]["is_error"] is True


def test_a_broken_promise_brings_the_ai_back_to_follow_up():
    env = make_env(
        [say()],
        [call("record_promise", amount=12_000, due_date=on(2)), say("Thanks Mona! Noted: 12,000 EGP by Tue 10 Nov.")],
        [say("Hi Mona, a friendly reminder: the 12,000 EGP you promised is due today, Tue 10 Nov.")],
        [say("Hi Mona, the 12,000 EGP did not arrive on Tue 10 Nov. Can you pay today?")],
    )
    start(env, "case-nile")
    reply(env, "case-nile", "I'll pay on Tuesday.")

    env.clock.advance(days=1)
    assert tick(env) == []  # Monday: the open promise holds back no-reply reminders
    env.clock.advance(days=1)
    [on_the_day] = tick(env)
    assert (on_the_day.type, on_the_day.decision, on_the_day.stop_reason) == (
        "DUE_TODAY", "ran_agent", "waiting_on_merchant")
    assert case(env, "case-nile")["broken_promises"] == 0  # the promised day itself is not late yet
    assert tick(env) == []  # one reminder per promise
    env.clock.advance(days=1)
    [outcome] = tick(env)

    assert (outcome.type, outcome.decision) == ("PROMISE_DUE", "ran_agent")
    assert store.one(env.db, "SELECT status FROM promises")["status"] == "broken"
    assert case(env, "case-nile")["broken_promises"] == 1
    assert '"broken_promises": 1' in case_file_of(env)


def test_a_second_broken_promise_goes_to_a_person_without_asking_the_ai():
    env = make_env([say()], [call("record_promise", amount=15_000, due_date=on(1)), say("Noted, thank you.")])
    start(env, "case-giza")  # Giza already broke one promise before
    reply(env, "case-giza", "I'll pay tomorrow, I promise.")

    env.clock.advance(days=2)
    [outcome] = tick(env)

    assert (outcome.decision, outcome.detail) == ("escalated", "cap:broken_promises")
    assert llm_calls(env) == 2
    assert (case(env, "case-giza")["state"], case(env, "case-giza")["broken_promises"]) == ("ESCALATED", 2)
    assert messages(env, "case-giza")[-1]["body"] == labels.NOTICE_ESCALATED


def test_a_silent_merchant_gets_three_reminders_then_a_person_takes_over():
    env = make_env([say("First message.")], [say("Reminder one.")], [say("Reminder two.")], [say("Reminder three.")])
    start(env, "case-nile")

    fired = []
    for day in range(1, 13):  # like pressing "Let a day pass" twelve times
        env.clock.advance(days=1)
        fired += [(day, o.type, o.decision, o.detail) for o in tick(env)]

    ran = ("REMINDER_DUE", "ran_agent", "waiting_on_merchant")
    assert fired == [(1, *ran), (2, *ran), (3, *ran), (4, "REMINDER_DUE", "escalated", "cap:reminders")]
    assert llm_calls(env) == 4
    assert (case(env, "case-nile")["state"], case(env, "case-nile")["reminders_sent"]) == ("ESCALATED", 3)


def test_missed_plan_payments_bring_the_ai_back_then_a_person_takes_over():
    env = make_env([say()], [plan((9_000, 7), (9_000, 30))],
                   [say("Hi Karim, the 9,000 EGP due on Sun 15 Nov did not arrive. Can you pay it this week?")])
    start(env, "case-cairo")
    reply(env, "case-cairo", "Can I pay in two parts?")
    deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": plans(env, "case-cairo")[0]["id"]})

    env.clock.advance(days=8)  # the first payment was due in 7 days
    [first] = tick(env)
    env.clock.advance(days=23)  # and the second in 30
    [second] = tick(env)

    assert (first.type, first.decision, first.stop_reason) == ("INSTALLMENT_DUE", "ran_agent", "waiting_on_merchant")
    assert (second.decision, second.detail) == ("escalated", "cap:missed_installments")
    assert llm_calls(env) == 3
    assert (case(env, "case-cairo")["state"], case(env, "case-cairo")["missed_installments"]) == ("ESCALATED", 2)
    assert plans(env, "case-cairo")[0]["status"] == "broken"


def test_a_person_can_hand_an_escalated_case_back_to_the_ai():
    env = make_env([say()], [call("escalate_to_human", reason="The merchant mentions a lawyer.")],
                   [say("Hi Youssef, my colleague asked me to follow up. Can you pay 15,000 EGP by Friday?")])
    start(env, "case-giza")
    reply(env, "case-giza", "Talk to my lawyer.")
    env.db.execute("UPDATE cases SET reminders_sent = 3 WHERE id = 'case-giza'")  # as if reminders ran out too

    outcome = deliver(env, "case-giza", EventType.HUMAN_DECISION, {"action": "hand_back", "note": "He will pay."})

    assert (outcome.decision, outcome.stop_reason) == ("ran_agent", "waiting_on_merchant")
    assert '"manager_note": "He will pay."' in case_file_of(env)
    moves = store.rows(env.db, "SELECT from_state, to_state, actor FROM transitions WHERE case_id = 'case-giza'")
    assert tuple(moves[-2]) == ("ESCALATED", "WORKING", "human")
    assert (case(env, "case-giza")["state"], case(env, "case-giza")["reminders_sent"]) == ("WAITING", 0)
    assert deliver(env, "case-giza", EventType.HUMAN_DECISION, {"action": "hand_back"}).detail == "stale"


# --- ambiguous claims: the AI verifies with a tool, then answers, asks for proof, or hands over -----------------

def test_a_merchant_who_says_they_paid_is_checked_with_check_payments():
    env = make_env([say()], [call("check_payments", reason="She says she paid, so check before answering.")],
                   [say("We can see your transfer of 8,500 EGP. It is waiting for the bank's confirmation.")])
    start(env, "case-alex")

    outcome = reply(env, "case-alex", "I already paid last week by bank transfer!")

    assert [s["tool"] for s in steps(env, outcome.run_id)] == ["check_payments", "send_message"]
    [checked] = env.llm.requests[-1][-1]["results"]  # what check_payments told the AI
    [transfer] = checked["content"]["pending"]
    assert transfer["amount"] == 8_500
    assert (case(env, "case-alex")["balance"], case(env, "case-alex")["state"]) == (8_500, "WAITING")  # a claim pays 0

    confirmed = deliver(env, "case-alex", EventType.PAYMENT_RECEIVED,
                        {"reference": transfer["reference"], "amount": 8_500})

    assert (confirmed.decision, confirmed.stop_reason) == ("handled_by_code", "resolved")
    assert case(env, "case-alex")["state"] == "RESOLVED" and llm_calls(env) == 3


def test_a_claim_the_payments_do_not_show_makes_the_ai_ask_for_proof():
    ask = "I can't see a payment from you yet. Could you send the transfer reference or a photo of the receipt?"
    env = make_env([say()], [call("check_payments", reason="He says he paid, so check before answering.")], [say(ask)])
    start(env, "case-nile")

    outcome = reply(env, "case-nile", "I paid yesterday.")

    [checked] = env.llm.requests[-1][-1]["results"]
    assert (checked["content"]["applied_to_this_case"], checked["content"]["pending"]) == ([], [])
    assert (outcome.stop_reason, case(env, "case-nile")["state"]) == ("waiting_on_merchant", "WAITING")
    assert (messages(env, "case-nile")[-1]["body"], case(env, "case-nile")["balance"]) == (ask, 12_000)


def test_a_dispute_the_payments_do_not_explain_goes_to_a_person():
    env = make_env([say()], [call("check_payments", reason="He disputes the debt, so check his payments first.")],
                   [call("escalate_to_human", reason="He disputes the debt and no payment explains it.")])
    start(env, "case-giza")

    outcome = reply(env, "case-giza", "I don't owe you anything. I paid everything last month.")

    [checked] = env.llm.requests[-1][-1]["results"]
    assert (checked["content"]["balance"], checked["content"]["pending"]) == (15_000, [])
    assert (outcome.stop_reason, case(env, "case-giza")["state"]) == ("escalated", "ESCALATED")
    assert case(env, "case-giza")["escalation_reason"] == "AI: He disputes the debt and no payment explains it."
    assert messages(env, "case-giza")[-1]["body"] == labels.NOTICE_ESCALATED
    assert reply(env, "case-giza", "Hello?").decision == "stored_for_human" and llm_calls(env) == 3


# --- offers and promises stay true to the balance ---------------------------------------------------------------

def offer_cairo_a_plan(*extra_turns) -> tuple:
    """Cairo is offered 9,000 (in 3 days) + 9,000 (in 30 days) for its 18,000 balance."""
    env = make_env([say()], [call("get_sales_summary")], [plan((9_000, 3), (9_000, 30))], *extra_turns)
    start(env, "case-cairo")
    reply(env, "case-cairo", "Sales are slow this month. Can I pay in two parts?")
    return env, plans(env, "case-cairo")[-1]


def accept(env, offer) -> object:
    return deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": offer["id"]})


def test_a_plan_offer_ends_the_run_with_its_own_stop_reason():
    env, offer = offer_cairo_a_plan()

    run = store.one(env.db, "SELECT stop_reason FROM runs ORDER BY rowid DESC LIMIT 1")

    assert run["stop_reason"] == "plan_offered"
    assert labels.stop_label("plan_offered") == ("Offered a payment plan and is waiting for the merchant to accept "
                                                 "or decline.")


def test_an_offer_that_no_longer_adds_up_to_the_balance_cannot_be_accepted():
    env, offer = offer_cairo_a_plan()
    pay(env, "case-cairo", 15_000)  # 3,000 is left; the offer still adds up to 18,000

    outcome = accept(env, offer)

    assert (outcome.decision, outcome.detail) == ("skipped", "stale_offer")
    assert (plans(env, "case-cairo")[-1]["status"], case(env, "case-cairo")["state"]) == ("outdated", "WAITING")
    assert messages(env, "case-cairo")[-1]["body"] == labels.NOTICE_OFFER_OUTDATED


def test_an_offer_whose_first_date_has_passed_cannot_be_accepted():
    env, offer = offer_cairo_a_plan()
    env.clock.advance(days=4)  # the first payment was due yesterday

    outcome = accept(env, offer)

    assert (outcome.decision, outcome.detail) == ("skipped", "stale_offer")
    assert plans(env, "case-cairo")[-1]["status"] == "outdated"


def test_a_plan_a_payment_made_outdated_goes_back_to_the_ai_when_the_manager_approves_it():
    env = make_env([say()], [call("get_sales_summary")], [plan((30_000, 7), (30_000, 37), (30_000, 60))],
                   [plan((20_000, 7), (20_000, 30))])
    start(env, "case-delta")
    reply(env, "case-delta", "Orders dropped a lot. I need about 3 months to pay this.")
    pay(env, "case-delta", 50_000)  # 40,000 is left while the manager looks at a 90,000 plan

    outcome = deliver(env, "case-delta", EventType.HUMAN_DECISION, {"action": "approve_plan", "note": ""})

    assert (outcome.decision, outcome.stop_reason) == ("ran_agent", "plan_offered")
    old, new = plans(env, "case-delta")
    assert (old["status"], new["status"]) == ("outdated", "offered")
    assert "add up to the balance exactly (40,000 EGP)" in case_file_of(env)  # the AI was told why


def test_when_the_ai_hands_the_case_to_a_person_the_open_offer_is_withdrawn():
    env, offer = offer_cairo_a_plan([call("escalate_to_human", reason="The merchant asks for a discount.")],
                                    [say("Our manager asked me to follow up: can you pay 18,000 EGP on Friday?")])
    reply(env, "case-cairo", "Actually, can you give me a discount instead?")
    assert plans(env, "case-cairo")[-1]["status"] == "withdrawn"
    deliver(env, "case-cairo", EventType.HUMAN_DECISION, {"action": "hand_back", "note": "She pays in full Friday."})

    outcome = accept(env, offer)  # the old offer must not come back to life after the hand-back

    assert (outcome.decision, outcome.detail) == ("skipped", "stale_offer")
    assert case(env, "case-cairo")["state"] == "WAITING"


def test_when_code_hands_the_case_to_a_person_the_open_offer_is_withdrawn_too():
    env, offer = offer_cairo_a_plan()
    env.clock.advance(days=46)

    tick(env)

    assert case(env, "case-cairo")["last_stop_reason"] == "cap:deadline"
    assert plans(env, "case-cairo")[-1]["status"] == "withdrawn"
    assert labels.PLAN_STATUS_LABELS["withdrawn"] == "Withdrawn — the case went to a person"


def test_accepting_a_plan_closes_the_promise_it_replaces():
    env = make_env([say()], [call("record_promise", amount=5_000, due_date=on(3)), say("Noted: 5,000 EGP by Wed.")],
                   [plan((7_500, 7), (7_500, 20))])
    start(env, "case-giza")  # Giza already broke one promise; a second broken one would go to a person
    reply(env, "case-giza", "I'll pay 5,000 by Wednesday.")
    reply(env, "case-giza", "Actually, can I split the whole amount into two payments?")
    deliver(env, "case-giza", EventType.PLAN_ACCEPTED, {"plan_id": plans(env, "case-giza")[-1]["id"]})
    env.clock.advance(days=4)  # the promised date has passed, but the plan replaced the promise

    assert tick(env) == []
    assert store.one(env.db, "SELECT status FROM promises WHERE case_id = 'case-giza'")["status"] == "superseded"
    assert (case(env, "case-giza")["state"], case(env, "case-giza")["broken_promises"]) == ("ON_PLAN", 1)


def test_money_paid_before_a_promise_does_not_count_toward_it():
    env = make_env([say()], [call("record_promise", amount=7_000, due_date=on(3)), say("Noted: 7,000 EGP by Wed.")])
    start(env, "case-nile")
    pay(env, "case-nile", 5_000)  # 7,000 is left
    reply(env, "case-nile", "I'll pay the remaining 7,000 by Wednesday.")

    pay(env, "case-nile", 2_000)  # same simulated minute as the promise

    promise = store.one(env.db, "SELECT status FROM promises WHERE case_id = 'case-nile'")
    assert promise["status"] == "open"  # only 2,000 of the 7,000 arrived after the promise
    pay(env, "case-nile", 5_000)
    assert store.one(env.db, "SELECT status FROM promises WHERE case_id = 'case-nile'")["status"] == "kept"


def test_money_paid_before_a_plan_started_does_not_count_toward_its_installments():
    env = make_env([say()], [plan((7_500, 3), (7_500, 30))])
    start(env, "case-cairo")
    pay(env, "case-cairo", 3_000)  # 15,000 is left
    reply(env, "case-cairo", "Can I pay the rest in two parts?")
    deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": plans(env, "case-cairo")[-1]["id"]})

    pay(env, "case-cairo", 4_500)

    first, second = labels.installments(plans(env, "case-cairo")[-1])
    assert (first["paid"], second["paid"]) == (False, False)  # 4,500 of the first 7,500 arrived since the start
    pay(env, "case-cairo", 3_000)
    assert labels.installments(plans(env, "case-cairo")[-1])[0]["paid"] is True
