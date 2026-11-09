"""The HTTP API, served for real on a free port: every route answers with the JSON shape the web app relies on."""

import json
import os
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import pytest

import server
from helpers import START, call, plan, say
from repayment.clock import SimClock
from repayment.llm import ScriptedLLM

STATE_KEYS = {"now", "today_label", "ai", "cases", "metrics", "alerts", "outages"}
AI_KEYS = {"provider", "models", "last_model", "unavailable", "label"}
SUMMARY_KEYS = {"case_id", "merchant", "state", "state_label", "where", "currency", "amount_due", "balance", "paid",
                "due_date", "days_overdue", "story", "suggestions", "busy", "needs_human", "updated_at"}
MERCHANT_KEYS = {"id", "business_name", "owner_name", "first_name", "category", "platform"}
DETAIL_CASE_KEYS = SUMMARY_KEYS | {"opened_at", "deadline_at", "reminders_sent", "broken_promises",
                                   "missed_installments", "messages_sent", "tokens_in", "tokens_out", "cost_usd",
                                   "last_stop_reason", "stop_label", "escalation_reason", "retry_at",
                                   "pending_payments"}
DETAIL_KEYS = {"case", "messages", "plans", "promise", "payments", "limits", "done", "next", "runs", "events",
               "transitions", "logs", "flags"}
MESSAGE_KEYS = {"id", "sender", "kind", "body", "plan_id", "at", "flag"}
PLAN_KEYS = {"id", "status", "status_label", "needs_approval", "why", "message", "created_at", "decision_note",
             "installments"}
INSTALLMENT_KEYS = {"n", "amount", "due_date", "paid"}
PAYMENT_KEYS = {"reference", "amount", "status", "source", "received_at", "note"}
LIMIT_KEYS = {"key", "label", "used", "max", "unit"}
RUN_KEYS = {"run_id", "trigger", "trigger_label", "started_at", "finished_at", "stop_reason", "stop_label",
            "llm_calls", "tokens_in", "tokens_out", "cost_usd", "duration_ms", "models", "status", "steps"}
STEP_KEYS = {"n", "llm_call", "tool", "tool_label", "args", "reason", "outcome", "outcome_label", "summary",
             "tokens_in", "tokens_out", "cost_usd", "latency_ms", "model", "state_before", "state_after", "attempts"}
EVENT_KEYS = {"seq", "event_id", "type", "type_label", "decision", "decision_label", "detail", "at", "run_id"}
OUTCOME_KEYS = {"event_id", "type", "case_id", "decision", "decision_label", "detail", "stop_reason", "stop_label",
                "run_id"}


@pytest.fixture
def serve_api(tmp_path):
    """Serves a fresh app whose AI plays back `turns`; returns request(method, path, body) -> (status, json)."""
    servers = []

    def start(turns: list) -> Callable[..., tuple[int, dict]]:
        app = server.create_app(ScriptedLLM(turns), db_path=str(tmp_path / "app.db"), log_path=None,
                                clock=SimClock(START))
        servers.append(httpd := server.serve(app, 0))
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        request.port = httpd.server_address[1]
        return request

    yield start
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def api(serve_api):
    return serve_api([[say()], [call("get_sales_summary")], [plan((9_000, 7), (9_000, 30))], [say()],
                      [call("check_payments"), say("We can see your transfer.")]])


def ok_outcomes(answer: tuple[int, dict]) -> list[dict]:
    status, body = answer
    assert status == 200 and body["ok"] is True, body
    assert all(set(outcome) == OUTCOME_KEYS for outcome in body["outcomes"])
    return body["outcomes"]


def test_the_state_endpoint_has_the_expected_shape(api):
    status, state = api("GET", "/api/state")

    assert status == 200 and set(state) == STATE_KEYS
    assert set(state["ai"]) == AI_KEYS and state["ai"]["label"] == "AI: Scripted (no AI) · scripted"
    assert state["today_label"] == "Sun 8 Nov 2026, 14:00"
    assert [c["case_id"] for c in state["cases"]] == ["case-nile", "case-cairo", "case-delta", "case-alex",
                                                      "case-giza", "case-redsea"]
    cairo = state["cases"][1]
    assert set(cairo) == SUMMARY_KEYS and set(cairo["merchant"]) == MERCHANT_KEYS
    assert (cairo["state"], cairo["state_label"], cairo["days_overdue"]) == ("NEW", "Not started", 14)
    assert cairo["where"] == "18,000 EGP is 14 days late. Nobody has contacted the merchant yet."
    assert state["outages"] == {"payments": 0, "sales": 0}
    assert {"cases_total", "resolved_rate", "tool_failure_rate", "promises"} <= set(state["metrics"])


def test_a_case_goes_from_start_to_an_accepted_plan_and_a_payment_through_the_api(api):
    [started] = ok_outcomes(api("POST", "/api/cases/case-cairo/start"))
    assert (started["decision_label"], started["stop_label"]) == (
        "AI ran", "Sent a message and is waiting for the merchant.")
    assert api("POST", "/api/cases/case-cairo/reply", {"text": "  "})[0] == 400
    assert api("POST", "/api/cases/case-cairo/reply", {"text": "x" * 2001})[0] == 400
    ok_outcomes(api("POST", "/api/cases/case-cairo/reply", {"text": "Sales are slow. Can I pay in two parts?"}))

    status, detail = api("GET", "/api/cases/case-cairo")

    assert status == 200 and set(detail) == DETAIL_KEYS and set(detail["case"]) == DETAIL_CASE_KEYS
    assert all(set(m) == MESSAGE_KEYS for m in detail["messages"])
    [offer] = detail["plans"]
    assert set(offer) == PLAN_KEYS and all(set(i) == INSTALLMENT_KEYS for i in offer["installments"])
    assert (offer["status"], offer["status_label"]) == ("offered", "Offered — waiting for the merchant")
    assert [limit["key"] for limit in detail["limits"]] == ["reminders", "broken_promises", "missed_installments",
                                                            "messages", "budget", "tokens", "deadline"]
    assert all(set(limit) == LIMIT_KEYS for limit in detail["limits"])
    assert detail["done"] == {"done": False, "text": "Done when the balance is 0 EGP. Paid 0 of 18,000 EGP.",
                              "paid": 0, "total": 18_000}
    newest = detail["runs"][0]
    assert set(newest) == RUN_KEYS and all(set(step) == STEP_KEYS for step in newest["steps"])
    assert [s["tool_label"] for s in newest["steps"]] == ["Checked store sales", "Proposed a payment plan"]
    assert newest["steps"][0]["args"] == {}
    assert all(set(event) == EVENT_KEYS for event in detail["events"])
    assert detail["events"][0]["type"] == "MERCHANT_REPLY"  # newest first
    assert set(detail["transitions"][0]) == {"from", "to", "actor", "note", "at"}
    assert detail["logs"] and json.loads(detail["logs"][0])["kind"] == "transition"
    assert detail["next"].startswith("If the merchant stays silent until Mon 9 Nov 14:00")

    [accepted] = ok_outcomes(api("POST", f"/api/cases/case-cairo/plans/{offer['id']}/accept"))
    assert accepted["decision"] == "handled_by_code"
    assert api("POST", "/api/cases/case-cairo/plans/999/accept")[0] == 400
    assert api("POST", "/api/cases/case-cairo/pay", {"amount": "9000"})[0] == 400
    assert api("POST", "/api/cases/case-cairo/pay", {"amount": 0})[0] == 400
    [paid] = ok_outcomes(api("POST", "/api/cases/case-cairo/pay", {"amount": 9_000}))
    assert paid["detail"] == "Applied 9,000 EGP; the balance is now 9,000 EGP."

    [again] = ok_outcomes(api("POST", "/api/cases/case-cairo/replay"))
    assert (again["decision_label"], again["detail"]) == ("Ignored", "Same event delivered twice — ignored")
    detail = api("GET", "/api/cases/case-cairo")[1]
    assert detail["case"]["state"] == "ON_PLAN" and detail["case"]["where"].startswith("On a plan: next payment")
    assert detail["payments"] and set(detail["payments"][0]) == PAYMENT_KEYS


def test_test_tools_clock_and_reset(api):
    status, detail = api("GET", "/api/cases/case-alex")
    assert detail["case"]["pending_payments"] == 1
    assert api("POST", "/api/cases/case-alex/retry")[0] == 400
    assert api("POST", "/api/cases/case-alex/decision", {"action": "dance"})[0] == 400

    [confirmed] = ok_outcomes(api("POST", "/api/cases/case-alex/confirm-pending"))
    assert confirmed["detail"] == "Applied 8,500 EGP; the balance is now 0 EGP."
    assert (confirmed["stop_reason"], confirmed["stop_label"]) == ("resolved", "Paid in full, so code closed the case.")
    assert api("POST", "/api/cases/case-alex/confirm-pending")[0] == 400
    assert api("POST", "/api/cases/case-alex/pay", {"amount": 100})[0] == 400  # already paid in full

    assert api("POST", "/api/outage", {"service": "payments", "failures": 3}) == (
        200, {"ok": True, "outages": {"payments": 3, "sales": 0}})
    assert api("POST", "/api/outage", {"service": "email", "failures": 3})[0] == 400
    assert api("POST", "/api/clock", {"hours": 0})[0] == 400
    assert ok_outcomes(api("POST", "/api/clock", {"hours": 24})) == []
    assert api("GET", "/api/state")[1]["today_label"] == "Mon 9 Nov 2026, 14:00"

    assert api("POST", "/api/reset") == (200, {"ok": True})
    state = api("GET", "/api/state")[1]
    assert {c["state"] for c in state["cases"]} == {"NEW"} and state["outages"] == {"payments": 0, "sales": 0}


def test_manager_decisions_a_declined_plan_and_a_retry_through_the_api(serve_api):
    api = serve_api([
        [say()], [call("get_sales_summary")], [plan((30_000, 7), (30_000, 37), (30_000, 60))],
        [plan((30_000, 7), (30_000, 37), (30_000, 59))],  # after the manager's rejection
        [call("escalate_to_human", reason="He declined the plan and asks for a discount.")],
        [say("A manager asked me to follow up. When can you pay the 90,000 EGP?")],  # handed back
        [call("check_payments")], [call("check_payments"), say("We can see your transfer of 8,500 EGP.")],
    ])
    ok_outcomes(api("POST", "/api/cases/case-delta/start"))
    [proposed] = ok_outcomes(api("POST", "/api/cases/case-delta/reply", {"text": "I need about 3 months."}))
    assert proposed["stop_label"] == "Proposed a plan that needs a manager's approval."

    [rejected] = ok_outcomes(api("POST", "/api/cases/case-delta/decision",
                                 {"action": "reject_plan", "note": "End it by day 59."}))
    [approved] = ok_outcomes(api("POST", "/api/cases/case-delta/decision", {"action": "approve_plan", "note": "OK"}))
    assert (rejected["decision"], approved["decision"]) == ("ran_agent", "handled_by_code")
    plans = api("GET", "/api/cases/case-delta")[1]["plans"]
    assert [(p["status"], p["decision_note"]) for p in plans] == [("offered", "OK"), ("rejected", "End it by day 59.")]

    [declined] = ok_outcomes(api("POST", f"/api/cases/case-delta/plans/{plans[0]['id']}/decline",
                                 {"text": "Too much. Give me a discount."}))
    assert (declined["type"], declined["stop_reason"]) == ("PLAN_DECLINED", "escalated")
    [handed_back] = ok_outcomes(api("POST", "/api/cases/case-delta/decision", {"action": "hand_back", "note": "Go"}))
    assert (handed_back["decision"], handed_back["stop_reason"]) == ("ran_agent", "waiting_on_merchant")

    api("POST", "/api/outage", {"service": "payments", "failures": 3})
    [paused] = ok_outcomes(api("POST", "/api/cases/case-alex/start"))
    assert paused["stop_label"] == "The payment system is down. Paused; it retries automatically."
    [retried] = ok_outcomes(api("POST", "/api/cases/case-alex/retry"))
    assert (retried["type"], retried["stop_reason"]) == ("TOOL_RETRY", "waiting_on_merchant")


def test_an_accept_that_came_too_early_does_not_block_the_real_one(serve_api):
    api = serve_api([[say()], [call("get_sales_summary")], [plan((30_000, 7), (30_000, 37), (30_000, 60))]])
    ok_outcomes(api("POST", "/api/cases/case-delta/start"))
    ok_outcomes(api("POST", "/api/cases/case-delta/reply", {"text": "I need about 3 months."}))
    plan_id = api("GET", "/api/cases/case-delta")[1]["plans"][0]["id"]

    [early] = ok_outcomes(api("POST", f"/api/cases/case-delta/plans/{plan_id}/accept"))  # still pending approval
    ok_outcomes(api("POST", "/api/cases/case-delta/decision", {"action": "approve_plan", "note": ""}))
    [accepted] = ok_outcomes(api("POST", f"/api/cases/case-delta/plans/{plan_id}/accept"))

    assert (early["decision"], early["detail"]) == ("skipped", "That offer is no longer open")
    assert accepted["decision"] == "handled_by_code"
    assert api("GET", "/api/cases/case-delta")[1]["case"]["state"] == "ON_PLAN"


def test_errors_are_plain_json(api):
    assert api("GET", "/api/nothing-here") == (404, {"error": "Not found."})
    assert api("POST", "/api/nothing-here")[0] == 404
    status, body = api("GET", "/api/cases/case-atlantis")
    assert status == 400 and "case-atlantis" in body["error"]


def test_an_unexpected_error_answers_500_and_the_case_goes_to_a_person(serve_api):
    api = serve_api([])  # an AI with no answers left: the run crashes

    status, body = api("POST", "/api/cases/case-nile/start")

    assert status == 500 and "handed to a person" in body["error"]
    nile = api("GET", "/api/cases/case-nile")[1]["case"]
    assert (nile["state"], nile["last_stop_reason"]) == ("ESCALATED", "internal_error")


def test_without_a_key_the_server_explains_and_exits_with_code_1():
    no_key = {name: value for name, value in os.environ.items() if name not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}

    done = subprocess.run([sys.executable, "server.py"], cwd=Path(server.__file__).parent, env=no_key,
                          capture_output=True, text=True, timeout=60)

    assert done.returncode == 1
    assert "https://aistudio.google.com/apikey" in done.stdout and "python demo.py" in done.stdout


def test_a_payment_must_be_a_whole_number_from_1_to_the_balance(api):
    for amount in (2**63, 12_001, 0):
        status, body = api("POST", "/api/cases/case-nile/pay", {"amount": amount})
        assert status == 400 and "12,000 EGP" in body["error"]
    assert api("POST", "/api/cases/case-nile/plans/99999999999999999999/accept") == (404, {"error": "Not found."})
    nile = api("GET", "/api/cases/case-nile")[1]
    assert (nile["case"]["state"], nile["events"]) == ("NEW", [])  # nothing reached the agent


def test_a_slow_or_broken_request_does_not_block_other_requests(api):
    slow = socket.create_connection(("127.0.0.1", api.port))
    slow.sendall(b"POST /api/cases/case-nile/reply HTTP/1.1\r\nHost: test\r\nContent-Length: 100\r\n\r\n{\"te")
    try:
        assert api("POST", "/api/outage", {"service": "sales", "failures": 0})[0] == 200  # not stuck behind it
    finally:
        slow.close()
    with socket.create_connection(("127.0.0.1", api.port)) as broken:
        broken.sendall(b"POST /api/outage HTTP/1.1\r\nHost: test\r\nContent-Length: abc\r\n\r\n")
        assert broken.recv(100).split(b"\r\n")[0].endswith(b" 400 Bad Request")
