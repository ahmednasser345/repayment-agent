"""Everything a run needs, passed explicitly so tests can swap any part."""

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import db as store
from . import seed as seeding
from .clock import SimClock
from .config import Limits, Policy
from .llm import LLM


@dataclass
class Env:
    db: sqlite3.Connection
    clock: SimClock
    llm: LLM
    limits: Limits = field(default_factory=Limits)
    policy: Policy = field(default_factory=Policy)
    outages: dict[str, int] = field(default_factory=lambda: {"payments": 0, "sales": 0})  # service -> failures left
    sleep: Callable[[float], None] = time.sleep  # backoff between tool attempts; tests pass a no-op
    log_path: Path | None = None


def build_env(llm: LLM, clock: SimClock, db_path: str = ":memory:", seed: bool = True, **overrides: Any) -> Env:
    env = Env(db=store.connect(db_path), clock=clock, llm=llm, **overrides)
    if seed and store.one(env.db, "SELECT 1 FROM cases LIMIT 1") is None:
        seeding.seed(env.db, clock.now(), env.limits.case_deadline_days)
    return env
