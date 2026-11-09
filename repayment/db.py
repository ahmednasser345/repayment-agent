"""SQLite storage. Single statements autocommit; tx() makes a group of writes atomic.

Never hold a transaction open across an LLM call: every write here is short.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS merchants (
    id TEXT PRIMARY KEY, business_name TEXT NOT NULL, owner_name TEXT NOT NULL, owner_name_ar TEXT, phone TEXT,
    email TEXT,
    category TEXT, platform TEXT, funded_amount INTEGER, story TEXT, suggestions_json TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY, merchant_id TEXT NOT NULL, currency TEXT NOT NULL, amount_due INTEGER NOT NULL,
    balance INTEGER NOT NULL, due_date TEXT NOT NULL, state TEXT NOT NULL, opened_at TEXT NOT NULL,
    deadline_at TEXT NOT NULL, reminders_sent INTEGER NOT NULL DEFAULT 0,
    broken_promises INTEGER NOT NULL DEFAULT 0, missed_installments INTEGER NOT NULL DEFAULT 0,
    messages_sent INTEGER NOT NULL DEFAULT 0, tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0, active_run_id TEXT,
    last_stop_reason TEXT, last_stop_detail TEXT, escalation_reason TEXT, tool_retries INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT, pending_trigger_json TEXT, handoff_note TEXT, updated_at TEXT NOT NULL,
    counted_event_id TEXT -- the timer event whose bookkeeping is saved; its redelivery must not count again
);
CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, merchant_id TEXT NOT NULL, case_id TEXT, reference TEXT UNIQUE NOT NULL,
    amount INTEGER NOT NULL, status TEXT NOT NULL, source TEXT, note TEXT, received_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sales (
    merchant_id TEXT NOT NULL, month TEXT NOT NULL, revenue INTEGER NOT NULL, orders INTEGER NOT NULL,
    PRIMARY KEY (merchant_id, month)
);
CREATE TABLE IF NOT EXISTS promises (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, amount INTEGER NOT NULL, due_date TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, balance_at_promise INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, status TEXT NOT NULL,
    installments_json TEXT NOT NULL, message TEXT NOT NULL, needs_approval INTEGER NOT NULL, why TEXT,
    created_at TEXT NOT NULL, activated_at TEXT, decided_by TEXT, decision_note TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, sender TEXT NOT NULL, kind TEXT NOT NULL,
    body TEXT NOT NULL, plan_id INTEGER, flag TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, case_id TEXT NOT NULL, type TEXT NOT NULL,
    payload_json TEXT NOT NULL, status TEXT NOT NULL, decision TEXT, detail TEXT, run_id TEXT,
    received_at TEXT NOT NULL, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS events_by_id ON events (event_id);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, case_id TEXT NOT NULL, trigger TEXT NOT NULL, trigger_event_id TEXT,
    started_at TEXT NOT NULL, finished_at TEXT, stop_reason TEXT, stop_detail TEXT,
    steps INTEGER NOT NULL DEFAULT 0, llm_calls INTEGER NOT NULL DEFAULT 0, tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0, duration_ms INTEGER,
    models_json TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL, n INTEGER NOT NULL, case_id TEXT NOT NULL, llm_call INTEGER NOT NULL, tool TEXT,
    args_json TEXT, reason TEXT, outcome TEXT NOT NULL, summary TEXT, tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0, latency_ms INTEGER NOT NULL DEFAULT 0,
    model TEXT, state_before TEXT, state_after TEXT, attempts INTEGER NOT NULL DEFAULT 0,
    failed_attempts INTEGER NOT NULL DEFAULT 0, ts TEXT NOT NULL, PRIMARY KEY (run_id, n)
);
CREATE TABLE IF NOT EXISTS findings (
    case_id TEXT NOT NULL, key TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY (case_id, key)
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, from_state TEXT NOT NULL, to_state TEXT NOT NULL,
    actor TEXT NOT NULL, note TEXT, at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, case_id TEXT, run_id TEXT, kind TEXT NOT NULL,
    line TEXT NOT NULL
);
"""
TABLES = ["merchants", "cases", "payments", "sales", "promises", "plans", "messages", "events", "runs", "steps",
          "findings", "transitions", "logs"]


class UnknownCase(KeyError):
    pass


def connect(path: str = ":memory:") -> sqlite3.Connection:
    db = sqlite3.connect(path, isolation_level=None)
    db.row_factory = sqlite3.Row
    if path != ":memory:":
        db.execute("PRAGMA journal_mode=WAL")  # readers (the dashboard) never block the writer (an AI run)
        db.execute("PRAGMA busy_timeout=10000")
    db.executescript(SCHEMA)
    return db


@contextmanager
def tx(db: sqlite3.Connection) -> Iterator[None]:
    """BEGIN IMMEDIATE ... COMMIT, or ROLLBACK on error. A nested tx() joins the outer one."""
    if db.in_transaction:
        yield
        return
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


def wipe(db: sqlite3.Connection) -> None:
    with tx(db):
        for table in TABLES:
            db.execute(f"DELETE FROM {table}")


def one(db: sqlite3.Connection, sql: str, *args: Any) -> sqlite3.Row | None:
    return db.execute(sql, args).fetchone()


def rows(db: sqlite3.Connection, sql: str, *args: Any) -> list[sqlite3.Row]:
    return db.execute(sql, args).fetchall()


def get_case(db: sqlite3.Connection, case_id: str) -> sqlite3.Row:
    case = one(db, "SELECT * FROM cases WHERE id = ?", case_id)
    if case is None:
        raise UnknownCase(case_id)
    return case


def update_case(db: sqlite3.Connection, case_id: str, now: datetime, **fields: Any) -> None:
    """Set the given columns (names come from code, never from input) and touch updated_at."""
    fields["updated_at"] = now.isoformat()
    assignments = ", ".join(f"{name} = ?" for name in fields)
    db.execute(f"UPDATE cases SET {assignments} WHERE id = ?", (*fields.values(), case_id))


def merchant_of(db: sqlite3.Connection, case_id: str) -> sqlite3.Row:
    return one(db, "SELECT m.* FROM merchants m JOIN cases c ON c.merchant_id = m.id WHERE c.id = ?", case_id)


def add_message(
    db: sqlite3.Connection, case_id: str, sender: str, kind: str, body: str, now: datetime,
    plan_id: int | None = None, flag: str | None = None,
) -> int:
    cursor = db.execute(
        "INSERT INTO messages (case_id, sender, kind, body, plan_id, flag, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (case_id, sender, kind, body, plan_id, flag, now.isoformat()),
    )
    db.execute("UPDATE cases SET updated_at = ? WHERE id = ?", (now.isoformat(), case_id))
    return cursor.lastrowid


def last_message(db: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    return one(db, "SELECT * FROM messages WHERE case_id = ? ORDER BY id DESC LIMIT 1", case_id)


def open_promise(db: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    return one(db, "SELECT * FROM promises WHERE case_id = ? AND status = 'open' ORDER BY id DESC LIMIT 1", case_id)


def plans_with_status(db: sqlite3.Connection, case_id: str, *statuses: str) -> list[sqlite3.Row]:
    marks = ", ".join("?" for _ in statuses)
    return rows(db, f"SELECT * FROM plans WHERE case_id = ? AND status IN ({marks}) ORDER BY id DESC",
                case_id, *statuses)


def active_plan(db: sqlite3.Connection, case_id: str) -> sqlite3.Row | None:
    found = plans_with_status(db, case_id, "active")
    return found[0] if found else None


def get_plan(db: sqlite3.Connection, case_id: str, plan_id: Any) -> sqlite3.Row | None:
    return one(db, "SELECT * FROM plans WHERE id = ? AND case_id = ?", plan_id, case_id)


def case_context(db: sqlite3.Connection, case_id: str) -> tuple[Any, Any, datetime | None]:
    """For status sentences: the open promise, the newest open plan, and when our last message went out (only
    if it is the newest message, i.e. the merchant has not answered it)."""
    plans = plans_with_status(db, case_id, "pending_approval", "offered", "active")
    last = last_message(db, case_id)
    last_ours_at = datetime.fromisoformat(last["created_at"]) if last and last["sender"] != "merchant" else None
    return open_promise(db, case_id), (plans[0] if plans else None), last_ours_at


def delivered(db: sqlite3.Connection, event_id: str) -> bool:
    return one(db, "SELECT 1 FROM events WHERE event_id = ? AND status = 'done' LIMIT 1", event_id) is not None
