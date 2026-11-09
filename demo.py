"""Offline demo: every sample merchant takes a different path to the finish line. No API key needed.

    python demo.py
"""

import json
from datetime import date, datetime

import scenarios
from repayment import db as store
from repayment import labels, metrics
from repayment.clock import SimClock, current_minute
from repayment.env import Env, build_env
from repayment.llm import ScriptedLLM

HEADER = """Repayment Assistant - offline demo

The AI decisions below are SCRIPTED (hand-written), so this runs without an API key.
Everything else is the real code: events, retrigger rules, caps, payments, plans, timers and logs.
`python server.py` runs the same code with a real AI (Gemini) reading the merchant's messages and choosing tools.
Each AI run ends with its run summary; after the merchants come the JSON log format and the metrics report.
"""
MERCHANTS = ["nile", "cairo", "delta", "alex", "redsea", "giza"]  # giza moves the clock 8 days: last


def main() -> None:
    print(HEADER)
    start = current_minute()
    env = _new_env(start)
    found = scenarios.build(start.date())
    for key in MERCHANTS:
        show(env, found[key])
    print("A separate, fresh database for the outage test, so the merchants above stay as they are.\n")
    show(_new_env(start), found["outage"])
    show_log_sample(env, "case-nile")
    print(metrics.report(env.db, env.clock.now(), env.limits).replace("Metrics at", "Metrics for the six merchants at"))


def show(env: Env, scenario: scenarios.Scenario) -> None:
    merchant, case = store.merchant_of(env.db, scenario.case_id), store.get_case(env.db, scenario.case_id)
    days_late = (env.clock.now().date() - date.fromisoformat(case["due_date"])).days
    print(f"== {merchant['business_name']} ({merchant['owner_name']}): {labels.money(case['balance'])}, "
          f"{days_late} days late")
    print(f"   {scenario.title}")
    for step, outcomes in scenarios.run(env, scenario):
        print(f"   > {_describe(step)}")
        for outcome in outcomes:
            detail = labels.detail_label(outcome.detail)
            print(f"     {labels.TRIGGER_LABELS[outcome.type]} -> {labels.DECISION_LABELS[outcome.decision]}: {detail}")
            for row in store.rows(env.db, "SELECT * FROM steps WHERE run_id = ? ORDER BY n", outcome.run_id):
                print(f"       {row['n']}. {labels.tool_label(row['tool'])} "
                      f"[{labels.OUTCOME_LABELS[row['outcome']]}]: {row['summary']}")
            if outcome.run_id:
                print(f"       {_run_summary(env, outcome.run_id)}")
    case = store.get_case(env.db, scenario.case_id)
    promise, plan, _ = store.case_context(env.db, scenario.case_id)
    print(f"   Final state: {labels.STATE_LABELS[case['state']]}. "
          f"{labels.where_it_stands(case, promise, plan, env.clock.now())}")
    print(f"   {_log_count(env, scenario.case_id)}\n")


def show_log_sample(env: Env, case_id: str) -> None:
    """Two JSON log lines exactly as stored: the case's first AI step and the summary of that run."""
    step = store.one(env.db, "SELECT run_id, line FROM logs WHERE case_id = ? AND kind = 'step' ORDER BY id", case_id)
    summary = store.one(env.db, "SELECT line FROM logs WHERE run_id = ? AND kind = 'run_summary'", step["run_id"])
    print(f"The JSON log, as stored (names, phones and emails redacted): the first AI step for {case_id}, then the "
          "summary of its run.")
    print(f"{step['line']}\n{summary['line']}\n")


def _run_summary(env: Env, run_id: str) -> str:
    """One line, read back from the run's run_summary log line."""
    run = json.loads(store.one(env.db, "SELECT line FROM logs WHERE run_id = ? AND kind = 'run_summary'",
                               run_id)["line"])
    return (f"Run summary: stop {run['stop_reason']} | steps {run['steps']} | AI calls {run['llm_calls']} | "
            f"tokens {run['tokens_in'] + run['tokens_out']:,} | est. ${run['cost_usd']:.4f} | {run['duration_ms']} ms")


def _log_count(env: Env, case_id: str) -> str:
    kinds = store.rows(env.db, "SELECT kind, COUNT(*) AS n FROM logs WHERE case_id = ? GROUP BY kind "
                               "ORDER BY MIN(id)", case_id)
    counts = ", ".join(f"{row['n']} {row['kind']}" for row in kinds)
    return f"Logged: {sum(row['n'] for row in kinds)} JSON lines ({counts}), PII redacted."


def _describe(step: scenarios.Step) -> str:
    action, *args = step
    return {
        "start": lambda: "The case opens (the Start button)",
        "reply": lambda: f'The merchant writes: "{args[0]}"',
        "pay": lambda: f"A payment of {labels.money(args[0])} arrives",
        "accept": lambda: "The merchant taps Accept on the plan",
        "approve": lambda: f'A manager approves the plan: "{args[0]}"',
        "confirm": lambda: "The bank confirms the pending transfer",
        "outage": lambda: f"Test: the {args[0]} system fails the next {args[1]} calls",
        "days": lambda: f"{args[0]} days pass (the timers run every day)",
        "minutes": lambda: f"{args[0]} minutes pass",
    }[action]()


def _new_env(start: datetime) -> Env:
    return build_env(ScriptedLLM([]), SimClock(start), sleep=lambda seconds: None)


if __name__ == "__main__":
    main()
