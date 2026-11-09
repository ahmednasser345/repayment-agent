"""The web app: the merchant app and the AI dashboard (web/index.html) on top of a small JSON API.

    python server.py                         # needs GEMINI_API_KEY (or GOOGLE_API_KEY)
    python server.py --port 8001 --db other.db --keep

Every POST becomes an event that goes through events.handle_event, exactly like production; the UI never changes
state directly. POSTs run one at a time. GETs never wait, so the dashboard can watch an AI run while it happens.
"""

import argparse
import json
import os
import re
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from repayment import db as store
from repayment import labels, metrics
from repayment.clock import SimClock, current_minute
from repayment.config import Limits, Policy
from repayment.env import Env, build_env
from repayment.events import Event, EventType, Outcome, handle_event, next_due, retry_event, tick
from repayment.gemini import GeminiLLM
from repayment.llm import LLM
from repayment.seed import seed

INDEX = Path(__file__).resolve().parent / "web" / "index.html"
NO_KEY_MESSAGE = """No Gemini API key found.

This app uses a real AI model (Google Gemini) to read the merchant's messages and decide what to do.
  1. Get a free key at https://aistudio.google.com/apikey
  2. export GEMINI_API_KEY="your-key"      (GOOGLE_API_KEY works too)
  3. python server.py

Without a key you can still run the tests (pytest -q) and the offline demo with scripted AI decisions
(python demo.py)."""


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class App:
    llm: LLM
    db_path: str  # a file: every request opens its own connection to it
    log_path: Path | None
    clock: SimClock
    outages: dict[str, int] = field(default_factory=lambda: {"payments": 0, "sales": 0})
    lock: threading.Lock = field(default_factory=threading.Lock)  # events are processed one at a time
    limits: Limits = field(default_factory=Limits)
    policy: Policy = field(default_factory=Policy)

    def env(self) -> Env:
        return build_env(self.llm, self.clock, self.db_path, seed=False, limits=self.limits, policy=self.policy,
                         outages=self.outages, log_path=self.log_path)


def create_app(llm: LLM, db_path: str = "app.db", fresh: bool = True, log_path: Path | None = Path("logs/agent.jsonl"),
               clock: SimClock | None = None) -> App:
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            Path(db_path + suffix).unlink(missing_ok=True)
    app = App(llm, db_path, log_path, clock or SimClock(current_minute()))
    env = app.env()
    if store.one(env.db, "SELECT 1 FROM cases LIMIT 1") is None:
        seed(env.db, app.clock.now(), app.limits.case_deadline_days)
    env.db.close()
    return app


def serve(app: App, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Start serving in a background thread and return the server (server_address[1] is the real port)."""
    server = ThreadingHTTPServer((host, port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --- reads ------------------------------------------------------------------------------------------------------

def state_view(app: App, env: Env) -> dict[str, Any]:
    now, ai = env.clock.now(), app.llm.status()
    model = ai.get("last_model") or next((m for m in ai["models"] if m not in ai["unavailable"]), None)
    ai["label"] = f"AI: {ai['provider']}" + (f" · {model}" if model else "")
    return {
        "now": now.isoformat(),
        "today_label": labels.today_label(now),
        "ai": ai,
        "cases": [case_summary(env, case) for case in store.rows(env.db, "SELECT * FROM cases ORDER BY rowid")],
        "metrics": metrics.compute(env.db, now),
        "alerts": metrics.alerts(env.db, now, env.limits),
        "outages": dict(app.outages),
    }


def case_summary(env: Env, case: Any) -> dict[str, Any]:
    merchant = store.one(env.db, "SELECT * FROM merchants WHERE id = ?", case["merchant_id"])
    now = env.clock.now()
    promise, plan, _ = store.case_context(env.db, case["id"])
    return {
        "case_id": case["id"],
        "merchant": {"id": merchant["id"], "business_name": merchant["business_name"],
                     "owner_name": merchant["owner_name"], "first_name": merchant["owner_name"].split()[0],
                     "category": merchant["category"], "platform": merchant["platform"]},
        "state": case["state"],
        "state_label": labels.STATE_LABELS[case["state"]],
        "where": labels.where_it_stands(case, promise, plan, now),
        "currency": case["currency"],
        "amount_due": case["amount_due"],
        "balance": case["balance"],
        "paid": case["amount_due"] - case["balance"],
        "due_date": case["due_date"],
        "days_overdue": (now.date() - date.fromisoformat(case["due_date"])).days,
        "story": merchant["story"],
        "suggestions": json.loads(merchant["suggestions_json"]),
        "busy": case["state"] == "WORKING",
        "needs_human": case["state"] in ("NEEDS_APPROVAL", "ESCALATED"),
        "updated_at": case["updated_at"],
    }


def case_detail(env: Env, case_id: str) -> dict[str, Any]:
    db, now, limits = env.db, env.clock.now(), env.limits
    case = _case(env, case_id)
    promise, plan, last_ours_at = store.case_context(db, case_id)
    payments = store.rows(db, "SELECT * FROM payments WHERE case_id = ? OR (merchant_id = ? AND case_id IS NULL "
                              "AND status = 'pending') ORDER BY id", case_id, case["merchant_id"])
    messages = store.rows(db, "SELECT * FROM messages WHERE case_id = ? ORDER BY id", case_id)
    latest_promise = store.one(db, "SELECT * FROM promises WHERE case_id = ? ORDER BY id DESC LIMIT 1", case_id)
    summary = case_summary(env, case) | {
        key: case[key] for key in ("opened_at", "deadline_at", "reminders_sent", "broken_promises",
                                   "missed_installments", "messages_sent", "tokens_in", "tokens_out",
                                   "last_stop_reason", "escalation_reason", "retry_at")
    }
    summary |= {"cost_usd": round(case["cost_usd"], 6), "stop_label": labels.stop_label(case["last_stop_reason"]),
                "pending_payments": sum(1 for p in payments if p["status"] == "pending")}
    return {
        "case": summary,
        "messages": [{"id": m["id"], "sender": m["sender"], "kind": m["kind"], "body": m["body"],
                      "plan_id": m["plan_id"], "at": m["created_at"], "flag": m["flag"]} for m in messages],
        "plans": [_plan_json(p) for p in store.rows(db, "SELECT * FROM plans WHERE case_id = ? ORDER BY id DESC",
                                                     case_id)],
        "promise": {key: latest_promise[key] for key in ("id", "amount", "due_date", "status")}
        if latest_promise else None,
        "payments": [{key: p[key] for key in ("reference", "amount", "status", "source", "received_at", "note")}
                     for p in payments],
        "limits": _limits_json(case, now, limits),
        "done": labels.done(case),
        "next": labels.what_happens_next(case, promise, plan, last_ours_at, now, limits),
        "runs": _runs_json(env, case_id),
        "events": [_event_json(e) for e in store.rows(db, "SELECT * FROM events WHERE case_id = ? ORDER BY seq DESC",
                                                       case_id)],
        "transitions": [{"from": t["from_state"], "to": t["to_state"], "actor": t["actor"], "note": t["note"],
                         "at": t["at"]}
                        for t in store.rows(db, "SELECT * FROM transitions WHERE case_id = ? ORDER BY id", case_id)],
        "logs": [row["line"] for row in store.rows(
            db, "SELECT line FROM (SELECT id, line FROM logs WHERE case_id = ? ORDER BY id DESC LIMIT 200) "
                "ORDER BY id", case_id)],
        "flags": [{"message_id": m["id"], "text": labels.FLAG_TEXT} for m in messages if m["flag"]],
    }


def _plan_json(plan: Any) -> dict[str, Any]:
    return {"id": plan["id"], "status": plan["status"], "status_label": labels.PLAN_STATUS_LABELS[plan["status"]],
            "needs_approval": bool(plan["needs_approval"]), "why": plan["why"], "message": plan["message"],
            "created_at": plan["created_at"], "decision_note": plan["decision_note"],
            "installments": labels.installments(plan)}


def _limits_json(case: Any, now: datetime, limits: Limits) -> list[dict[str, Any]]:
    days_used = (now - datetime.fromisoformat(case["opened_at"])).days
    rows = [
        ("reminders", case["reminders_sent"], limits.max_reminders, "count"),
        ("broken_promises", case["broken_promises"], limits.max_broken_promises, "count"),
        ("missed_installments", case["missed_installments"], limits.max_missed_installments, "count"),
        ("messages", case["messages_sent"], limits.max_messages_per_case, "count"),
        ("budget", round(case["cost_usd"], 4), limits.max_cost_usd_per_case, "usd"),
        ("tokens", case["tokens_in"] + case["tokens_out"], limits.max_tokens_per_case, "tokens"),
        ("deadline", days_used, limits.case_deadline_days, "days"),
    ]
    return [{"key": key, "label": labels.LIMIT_LABELS[key], "used": used, "max": cap, "unit": unit}
            for key, used, cap, unit in rows]


def _runs_json(env: Env, case_id: str) -> list[dict[str, Any]]:
    steps: dict[str, list[dict[str, Any]]] = {}
    for s in store.rows(env.db, "SELECT * FROM steps WHERE case_id = ? ORDER BY run_id, n", case_id):
        steps.setdefault(s["run_id"], []).append({
            "n": s["n"], "llm_call": s["llm_call"], "tool": s["tool"], "tool_label": labels.tool_label(s["tool"]),
            "args": json.loads(s["args_json"]) if s["args_json"] else None, "reason": s["reason"],
            "outcome": s["outcome"], "outcome_label": labels.OUTCOME_LABELS.get(s["outcome"], s["outcome"]),
            "summary": s["summary"], "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
            "cost_usd": round(s["cost_usd"], 6), "latency_ms": s["latency_ms"], "model": s["model"],
            "state_before": s["state_before"], "state_after": s["state_after"], "attempts": s["attempts"],
        })
    return [
        {"run_id": r["id"], "trigger": r["trigger"], "trigger_label": labels.TRIGGER_LABELS.get(r["trigger"]),
         "started_at": r["started_at"], "finished_at": r["finished_at"], "stop_reason": r["stop_reason"],
         "stop_label": labels.stop_label(r["stop_reason"]), "llm_calls": r["llm_calls"], "tokens_in": r["tokens_in"],
         "tokens_out": r["tokens_out"], "cost_usd": round(r["cost_usd"], 6), "duration_ms": r["duration_ms"],
         "models": json.loads(r["models_json"]), "status": r["status"], "steps": steps.get(r["id"], [])}
        for r in store.rows(env.db, "SELECT * FROM runs WHERE case_id = ? ORDER BY rowid DESC", case_id)
    ]


def _event_json(event: Any) -> dict[str, Any]:
    decision = event["decision"]
    unfinished = "Cut short by a crash; delivered again" if event["status"] == "interrupted" else "Not finished"
    return {"seq": event["seq"], "event_id": event["event_id"], "type": event["type"],
            "type_label": labels.TRIGGER_LABELS.get(event["type"], event["type"]), "decision": decision,
            "decision_label": labels.DECISION_LABELS[decision] if decision else unfinished,
            "detail": labels.detail_label(event["detail"]), "at": event["received_at"], "run_id": event["run_id"]}


def outcome_json(outcome: Outcome) -> dict[str, Any]:
    return {"event_id": outcome.event_id, "type": outcome.type, "case_id": outcome.case_id,
            "decision": outcome.decision, "decision_label": labels.DECISION_LABELS[outcome.decision],
            "detail": labels.detail_label(outcome.detail), "stop_reason": outcome.stop_reason,
            "stop_label": labels.stop_label(outcome.stop_reason), "run_id": outcome.run_id}


# --- writes: every one is an event, exactly as in production ----------------------------------------------------

def start(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    _case(env, case_id)
    return _deliver(env, case_id, Event(f"start:{case_id}", EventType.CASE_OPENED))


def reply(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    _case(env, case_id)
    text = _text(body, "text", required=True)
    return _deliver(env, case_id, Event(f"reply:{case_id}:{_hex()}", EventType.MERCHANT_REPLY, {"text": text}))


def pay(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    case, amount = _case(env, case_id), body.get("amount")
    if case["state"] == "RESOLVED":
        raise ApiError("This case is already paid in full.")
    if not _whole(amount) or not 1 <= amount <= case["balance"]:  # refunds of overpayments are out of scope
        raise ApiError(f"The amount must be a whole number from 1 to the balance ({labels.money(case['balance'])}).")
    reference = f"pay-{_hex()}"
    return _deliver(env, case_id, Event(f"payment:{reference}", EventType.PAYMENT_RECEIVED,
                                        {"reference": reference, "amount": amount, "source": "bank_transfer"}))


# Each tap is its own event (like a reply). A tap that came too early is skipped as stale_offer; with a fixed id,
# a later valid tap would then be ignored as a duplicate. A second tap on an active plan is skipped by the plan check.
def accept(app: App, env: Env, body: dict, case_id: str, plan_id: str) -> dict[str, Any]:
    _plan(env, case_id, plan_id)
    return _deliver(env, case_id, Event(f"accept:{case_id}:{plan_id}:{_hex()}", EventType.PLAN_ACCEPTED,
                                        {"plan_id": int(plan_id)}))


def decline(app: App, env: Env, body: dict, case_id: str, plan_id: str) -> dict[str, Any]:
    _plan(env, case_id, plan_id)
    payload = {"plan_id": int(plan_id), "text": _text(body, "text", required=False)}
    return _deliver(env, case_id, Event(f"decline:{case_id}:{plan_id}:{_hex()}", EventType.PLAN_DECLINED, payload))


def decision(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    _case(env, case_id)
    action = body.get("action")
    if action not in ("approve_plan", "reject_plan", "hand_back"):
        raise ApiError("The action must be approve_plan, reject_plan or hand_back.")
    payload = {"action": action, "note": _text(body, "note", required=False)}
    return _deliver(env, case_id, Event(f"decision:{case_id}:{_hex()}", EventType.HUMAN_DECISION, payload))


def retry(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    case = _case(env, case_id)
    if not case["retry_at"]:
        raise ApiError("No retry is pending for this case.")
    return _deliver(env, case_id, retry_event(env, case))  # the timer's id, so this retry cannot run twice


def replay(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    _case(env, case_id)
    last = store.one(env.db, "SELECT * FROM events WHERE case_id = ? ORDER BY seq DESC LIMIT 1", case_id)
    if last is None:
        raise ApiError("This case has no events to replay yet.")
    return _deliver(env, case_id, Event(last["event_id"], last["type"], json.loads(last["payload_json"])))


def confirm_pending(app: App, env: Env, body: dict, case_id: str) -> dict[str, Any]:
    case = _case(env, case_id)
    pending = store.rows(env.db, "SELECT * FROM payments WHERE status = 'pending' AND (case_id = ? OR "
                                 "(merchant_id = ? AND case_id IS NULL)) ORDER BY id", case_id, case["merchant_id"])
    if not pending:
        raise ApiError("There is no pending bank transfer for this case.")
    outcomes = [handle_event(env, case_id, Event(
        f"bank:{p['reference']}", EventType.PAYMENT_RECEIVED,
        {"reference": p["reference"], "amount": p["amount"], "source": p["source"] or "bank_transfer"}))
        for p in pending]
    return {"ok": True, "outcomes": [outcome_json(o) for o in outcomes]}


def advance_clock(app: App, env: Env, body: dict) -> dict[str, Any]:
    hours = body.get("hours")
    if not _whole(hours) or not 1 <= hours <= 168:
        raise ApiError("hours must be a whole number from 1 to 168.")
    app.clock.advance(hours=hours)
    outcomes = [outcome_json(o) for o in tick(env)]
    return {"ok": True, "outcomes": outcomes, "next": next_json(env)}


def next_json(env: Env) -> dict[str, Any] | None:
    """The first thing another day would wake, so "nothing was due" never looks like nothing works."""
    found = next_due(env)
    if found is None:
        return None
    ahead, case_id, event = found
    on = (env.clock.now() + timedelta(days=ahead)).date()
    name = store.merchant_of(env.db, case_id)["business_name"]
    return {"case_id": case_id, "type": event.type, "days_ahead": ahead, "date": on.isoformat(),
            "text": labels.next_due(name, event.type, on, ahead)}


def set_outage(app: App, env: Env, body: dict) -> dict[str, Any]:
    service, failures = body.get("service"), body.get("failures")
    if service not in ("payments", "sales"):
        raise ApiError("The service must be payments or sales.")
    if not _whole(failures) or not 0 <= failures <= 10:
        raise ApiError("failures must be a whole number from 0 to 10.")
    app.outages[service] = failures
    return {"ok": True, "outages": dict(app.outages)}


def reset(app: App, env: Env, body: dict) -> dict[str, Any]:
    store.wipe(env.db)
    app.clock = SimClock(current_minute())
    app.outages.update(payments=0, sales=0)
    seed(env.db, app.clock.now(), app.limits.case_deadline_days)
    return {"ok": True}


CASE = r"/api/cases/([^/]+)"
POST_ROUTES: list[tuple[re.Pattern, Callable[..., dict[str, Any]]]] = [
    (re.compile(CASE + "/start"), start),
    (re.compile(CASE + "/reply"), reply),
    (re.compile(CASE + "/pay"), pay),
    (re.compile(CASE + r"/plans/(\d{1,9})/accept"), accept),
    (re.compile(CASE + r"/plans/(\d{1,9})/decline"), decline),
    (re.compile(CASE + "/decision"), decision),
    (re.compile(CASE + "/retry"), retry),
    (re.compile(CASE + "/replay"), replay),
    (re.compile(CASE + "/confirm-pending"), confirm_pending),
    (re.compile("/api/clock"), advance_clock),
    (re.compile("/api/outage"), set_outage),
    (re.compile("/api/reset"), reset),
]
CASE_DETAIL = re.compile(CASE)


def _deliver(env: Env, case_id: str, event: Event) -> dict[str, Any]:
    return {"ok": True, "outcomes": [outcome_json(handle_event(env, case_id, event))]}


def _case(env: Env, case_id: str) -> Any:
    try:
        return store.get_case(env.db, case_id)
    except store.UnknownCase:
        raise ApiError(f"There is no case called {case_id}.") from None


def _plan(env: Env, case_id: str, plan_id: str) -> Any:
    _case(env, case_id)
    plan = store.get_plan(env.db, case_id, int(plan_id))
    if plan is None:
        raise ApiError(f"Case {case_id} has no plan {plan_id}.")
    return plan


def _text(body: dict, key: str, required: bool, limit: int = 2000) -> str:
    value = body.get(key) or ""
    if not isinstance(value, str):
        raise ApiError(f"{key} must be text.")
    value = value.strip()
    if required and not value:
        raise ApiError(f"Please write a {key} first.")
    if len(value) > limit:
        raise ApiError(f"The {key} is too long ({len(value)} characters; the limit is {limit}).")
    return value


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _hex() -> str:
    return uuid.uuid4().hex[:8]


# --- HTTP -------------------------------------------------------------------------------------------------------

def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        timeout = 10  # seconds; a client that stops sending is dropped

        def do_GET(self) -> None:  # noqa: N802 - the name http.server expects
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_page()
                return
            self._answer(lambda env: self._read(env, path), lock=False)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            for pattern, action in POST_ROUTES:
                if match := pattern.fullmatch(path):
                    try:
                        body = self._body()  # read before taking the lock, so a slow client blocks nobody
                    except ApiError as exc:
                        self._send_json(exc.status, {"error": str(exc)})
                        return
                    self._answer(lambda env: action(app, env, body, *match.groups()), lock=True)
                    return
            self._send_json(404, {"error": "Not found."})

        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _read(self, env: Env, path: str) -> dict[str, Any]:
            if path == "/api/state":
                return state_view(app, env)
            if match := CASE_DETAIL.fullmatch(path):
                return case_detail(env, match.group(1))
            raise ApiError("Not found.", 404)

        def _answer(self, handle: Callable[[Env], dict[str, Any]], lock: bool) -> None:
            try:
                if lock:
                    with app.lock:
                        body = self._with_env(handle)
                else:
                    body = self._with_env(handle)
                self._send_json(200, body)
            except ApiError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - a failed event already handed its case to a person
                traceback.print_exc()
                handed = " The case was handed to a person." if lock else ""
                self._send_json(500, {"error": f"Unexpected error ({type(exc).__name__}).{handed} The server log "
                                               "has the details."})

        def _with_env(self, handle: Callable[[Env], dict[str, Any]]) -> dict[str, Any]:
            env = app.env()  # one connection per request, closed when the request ends
            try:
                return handle(env)
            finally:
                env.db.close()

        def _body(self) -> dict[str, Any]:
            length = self.headers.get("Content-Length") or "0"
            if not length.isdigit() or int(length) > 100_000:
                raise ApiError("Content-Length must be a number of bytes (at most 100,000).")
            try:
                raw = self.rfile.read(int(length))
            except TimeoutError:
                raise ApiError("The request body did not arrive in time.", 408) from None
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                raise ApiError("The request body must be JSON.") from None
            if not isinstance(body, dict):
                raise ApiError("The request body must be a JSON object.")
            return body

        def _send_page(self) -> None:
            if not INDEX.exists():
                self._send_json(404, {"error": "web/index.html is missing."})
                return
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            self._send(status, json.dumps(body, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repayment Assistant: the merchant app and the AI dashboard.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="app.db")
    parser.add_argument("--keep", action="store_true", help="keep the existing database instead of starting fresh")
    args = parser.parse_args(argv)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print(NO_KEY_MESSAGE)
        return 1
    try:  # bind before touching the database, so a second copy can't wipe the data of one already running
        server = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    except OSError as exc:
        print(f"Port {args.port} is busy ({exc.strerror}). Is the app already running? Try --port {args.port + 1}.")
        return 1
    llm = GeminiLLM()
    server.RequestHandlerClass = make_handler(create_app(llm, db_path=args.db, fresh=not args.keep))
    print(f"Repayment Assistant is running at http://127.0.0.1:{args.port}")
    print(f"AI: Gemini. Models, in the order they are tried: {', '.join(llm.models)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
