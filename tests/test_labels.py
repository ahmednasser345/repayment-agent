"""The plain-language sentences the UI shows verbatim."""

from datetime import timedelta

from helpers import START, call, deliver, make_env, on, plan, reply, say, start
from repayment import db as store
from repayment import labels
from repayment.config import Limits
from repayment.events import EventType


def where_and_next(env, case_id: str) -> tuple[str, str]:
    case = store.get_case(env.db, case_id)
    promise, open_plan, last_ours_at = store.case_context(env.db, case_id)
    now = env.clock.now()
    return (labels.where_it_stands(case, promise, open_plan, now),
            labels.what_happens_next(case, promise, open_plan, last_ours_at, now, Limits()))


ANY_PAYMENT = " A payment closes the case as soon as the balance reaches 0."


def test_stop_and_status_labels_match_the_ui_contract():
    assert labels.stop_label("paused:payments_unavailable") == ("The payment system is down. Paused; it retries "
                                                                "automatically.")
    assert labels.stop_label("paused:sales_unavailable").startswith("The store's sales system is down.")
    assert labels.stop_label("paused:ai_unavailable").startswith("The AI is down.")
    assert labels.stop_label("cap:reminders") == "No reply after the last reminder, so code handed it to a person."
    assert labels.detail_label("stale_offer") == "That offer is no longer open"
    assert labels.detail_label("cap:deadline") == "The case deadline passed, so code handed it to a person."
    assert labels.today_label(START) == "Sun 8 Nov 2026, 14:00"


def test_a_new_case():
    assert where_and_next(make_env(), "case-nile") == (
        "12,000 EGP is 6 days late. Nobody has contacted the merchant yet.",
        "Press Start in the merchant app to open the case." + ANY_PAYMENT)


def test_waiting_for_a_reply():
    env = make_env([say()])
    start(env, "case-nile")
    assert where_and_next(env, "case-nile") == (
        "Waiting for the merchant to reply or pay.",
        "If the merchant stays silent until Mon 9 Nov 14:00, the AI sends reminder 1 of 3. After 3 reminders a "
        "person takes over." + ANY_PAYMENT)


def test_waiting_on_a_promise():
    env = make_env([say()], [call("record_promise", amount=6_000, due_date=on(4)), say("Noted.")])
    start(env, "case-nile")
    reply(env, "case-nile", "6,000 on Thursday.")
    assert where_and_next(env, "case-nile") == (
        "Merchant promised 6,000 EGP by Thu 12 Nov.",
        "On Thu 12 Nov the AI sends a payment-day reminder; on Fri 13 Nov the AI checks whether 6,000 EGP arrived."
        + ANY_PAYMENT)


def test_an_offer_then_a_plan():
    env = make_env([say()], [plan((9_000, 7), (9_000, 30))])
    start(env, "case-cairo")
    reply(env, "case-cairo", "Two parts?")
    assert where_and_next(env, "case-cairo")[0] == ("A payment plan was offered; waiting for the merchant to accept "
                                                    "or decline.")
    offer = store.plans_with_status(env.db, "case-cairo", "offered")[0]
    deliver(env, "case-cairo", EventType.PLAN_ACCEPTED, {"plan_id": offer["id"]})
    assert where_and_next(env, "case-cairo") == (
        "On a plan: next payment 9,000 EGP on Sun 15 Nov.",
        "On Sun 15 Nov the AI sends a payment-day reminder; on Mon 16 Nov the AI checks whether the 9,000 EGP "
        "payment arrived." + ANY_PAYMENT)


def test_needing_approval_and_escalated():
    env = make_env([say()], [plan((30_000, 7), (30_000, 37), (30_000, 60))])
    start(env, "case-delta")
    reply(env, "case-delta", "3 months please.")
    assert where_and_next(env, "case-delta") == (
        "The AI proposed a 3-payment plan. A manager must approve or reject it.",
        "Nothing happens until a manager approves or rejects the plan." + ANY_PAYMENT)

    env = make_env([say()], [call("escalate_to_human", reason="Mentions a lawyer.")])
    start(env, "case-giza")
    reply(env, "case-giza", "Talk to my lawyer.")
    assert where_and_next(env, "case-giza") == (
        "With a person: AI: Mentions a lawyer.",
        "The AI is stopped. A person owns this case and can hand it back to the AI." + ANY_PAYMENT)


def test_paused_by_an_outage():
    env = make_env([say()], [call("check_payments")])
    start(env, "case-alex")
    env.outages["payments"] = 3
    reply(env, "case-alex", "I paid!")
    assert where_and_next(env, "case-alex") == (
        "Paused: the payment system is down. Retrying at 14:15.",
        "The AI retries at 14:15 (retry 1 of 3)." + ANY_PAYMENT)
    assert labels.when(START + timedelta(days=1), START) == "Mon 9 Nov 14:00"


def test_paid_in_full():
    env = make_env()
    deliver(env, "case-redsea", EventType.PAYMENT_RECEIVED, {"reference": "pay-00000001", "amount": 6_000})
    assert where_and_next(env, "case-redsea") == ("Paid in full.", "Nothing — the case is closed.")
    case = store.get_case(env.db, "case-redsea")
    assert labels.done(case)["text"] == "Done: the balance is 0 EGP. Paid 6,000 of 6,000 EGP."
