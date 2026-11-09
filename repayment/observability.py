"""Structured logs: one JSON line per entry, PII redacted, stored in the logs table and appended to a file."""

import json
import re
import sqlite3
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from .env import Env

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
SEP = r"[\s.()-]{0,2}"  # what people put between digits: "+20 (100) 555-0101", "0100.555.0101"
PHONES = [
    re.compile(rf"(?<![\w+])(?:(?:\+|00)?20{SEP})?\(?0?1[0125](?:{SEP}\d){{8}}(?!\w)"),  # Egyptian mobiles
    re.compile(rf"(?<![\w+])\+\d{{1,3}}(?:{SEP}\d){{7,12}}(?!\w)"),  # any number written with a country code
]
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)  # matched (and logged) as 0-9
LONG_NUMBER = re.compile(r"(?<!\d)\d{10,}(?!\d)")  # account or ID numbers; amounts and dates are shorter


def owner_names(db: sqlite3.Connection) -> tuple[str, ...]:
    """Each owner's full name and every part of it, in Latin and Arabic letters ("Mona Adel", "Mona", "منى")."""
    names: set[str] = set()
    for row in db.execute("SELECT owner_name, owner_name_ar FROM merchants"):
        for full in filter(None, row):
            names.add(full)
            names.update(part for part in full.split() if len(part) >= 3)
    return tuple(sorted(names, key=len, reverse=True))  # longest first: "Mona Adel" before "Mona"


@lru_cache(maxsize=8)
def _name_pattern(names: tuple[str, ...]) -> re.Pattern | None:
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)


def redact(value: Any, names: Iterable[str] = ()) -> Any:
    """Mask names, emails, phone numbers and long digit runs in every string, at any depth."""
    pattern = _name_pattern(tuple(names))
    return _redact(value, pattern)


def _redact(value: Any, names: re.Pattern | None) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, names) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, names) for item in value]
    if not isinstance(value, str):
        return value
    value = EMAIL.sub("[EMAIL]", value.translate(ARABIC_DIGITS))
    for phone in PHONES:
        value = phone.sub("[PHONE]", value)
    value = LONG_NUMBER.sub("[NUMBER]", value)
    return names.sub("[NAME]", value) if names else value


def log(env: "Env", kind: str, case_id: str | None, run_id: str | None = None, **fields: Any) -> dict[str, Any]:
    """Write one redacted JSON line. Kinds: step, run_summary, event, transition."""
    line: dict[str, Any] = {"ts": env.clock.now().isoformat(), "kind": kind, "case_id": case_id}
    if run_id is not None:
        line["run_id"] = run_id
    line = redact({**line, **fields}, owner_names(env.db))
    text = json.dumps(line, ensure_ascii=False, default=str)
    env.db.execute(
        "INSERT INTO logs (ts, case_id, run_id, kind, line) VALUES (?, ?, ?, ?, ?)",
        (line["ts"], case_id, run_id, kind, text),
    )
    if env.log_path is not None:
        env.log_path.parent.mkdir(parents=True, exist_ok=True)
        with env.log_path.open("a", encoding="utf-8") as file:
            file.write(text + "\n")
    return line
