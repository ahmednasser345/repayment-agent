"""Case states, who may move a case between them, and transition(): the only code that writes cases.state."""

from enum import StrEnum
from typing import TYPE_CHECKING

from . import db as store
from . import labels
from .observability import log

if TYPE_CHECKING:
    from .env import Env


class State(StrEnum):
    NEW = "NEW"
    WORKING = "WORKING"
    WAITING = "WAITING"
    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    ON_PLAN = "ON_PLAN"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class Actor(StrEnum):
    AGENT = "agent"  # a tool call chosen by the AI
    SYSTEM = "system"  # event handlers, caps and timers
    HUMAN = "human"  # a manager's decision


_S, _A = State, Actor
TRANSITIONS: dict[tuple[State, State], set[Actor]] = {
    (_S.NEW, _S.WORKING): {_A.SYSTEM},
    (_S.WAITING, _S.WORKING): {_A.SYSTEM},
    (_S.ON_PLAN, _S.WORKING): {_A.SYSTEM},
    (_S.NEEDS_APPROVAL, _S.WORKING): {_A.HUMAN},
    (_S.ESCALATED, _S.WORKING): {_A.HUMAN},
    (_S.WORKING, _S.WAITING): {_A.AGENT, _A.SYSTEM},
    (_S.WORKING, _S.ON_PLAN): {_A.AGENT, _A.SYSTEM},
    (_S.WORKING, _S.NEEDS_APPROVAL): {_A.AGENT},
    (_S.WORKING, _S.ESCALATED): {_A.AGENT, _A.SYSTEM},
    (_S.NEW, _S.ESCALATED): {_A.SYSTEM},
    (_S.WAITING, _S.ESCALATED): {_A.SYSTEM},
    (_S.ON_PLAN, _S.ESCALATED): {_A.SYSTEM},
    (_S.NEEDS_APPROVAL, _S.ESCALATED): {_A.SYSTEM},
    (_S.NEEDS_APPROVAL, _S.WAITING): {_A.HUMAN},
    (_S.WAITING, _S.ON_PLAN): {_A.SYSTEM},
    **{(start, _S.RESOLVED): {_A.SYSTEM}
       for start in (_S.NEW, _S.WAITING, _S.NEEDS_APPROVAL, _S.ON_PLAN, _S.ESCALATED)},
}
HUMAN_OWNED = {State.NEEDS_APPROVAL, State.ESCALATED}


class IllegalTransition(Exception):
    pass


def state_of(env: "Env", case_id: str) -> State:
    return State(store.get_case(env.db, case_id)["state"])


def transition(env: "Env", case_id: str, to: State, actor: Actor, note: str = "") -> State:
    """Move the case, record who did it and why, and log it. Returns the previous state."""
    now = env.clock.now()
    with store.tx(env.db):
        current = state_of(env, case_id)
        if actor not in TRANSITIONS.get((current, to), set()):
            raise IllegalTransition(f"{actor} may not move {case_id} from {current} to {to}")
        store.update_case(env.db, case_id, now, state=to.value)
        env.db.execute(
            "INSERT INTO transitions (case_id, from_state, to_state, actor, note, at) VALUES (?, ?, ?, ?, ?, ?)",
            (case_id, current.value, to.value, actor.value, note, now.isoformat()),
        )
        log(env, "transition", case_id, **{"from": current.value, "to": to.value, "actor": actor.value, "note": note})
    return current


def idle_state(env: "Env", case_id: str) -> State:
    """Where a case rests between runs."""
    return State.ON_PLAN if store.active_plan(env.db, case_id) else State.WAITING


def escalate(env: "Env", case_id: str, reason: str, detail: str = "", actor: Actor = Actor.SYSTEM,
             escalation_reason: str | None = None) -> None:
    """Hand the case to a person: the state, the reason shown to them, and a notice to the merchant. An offer still
    open is withdrawn, so it cannot be accepted behind the person's back."""
    now = env.clock.now()
    with store.tx(env.db):
        if state_of(env, case_id) != State.ESCALATED:
            transition(env, case_id, State.ESCALATED, actor, labels.stop_label(reason))
            store.add_message(env.db, case_id, "system", "notice", labels.NOTICE_ESCALATED, now)
        env.db.execute("UPDATE plans SET status = 'withdrawn' WHERE case_id = ? AND status IN ('offered', "
                       "'pending_approval')", (case_id,))
        store.update_case(
            env.db, case_id, now, escalation_reason=escalation_reason or labels.stop_label(reason),
            last_stop_reason=reason, last_stop_detail=detail,
        )
