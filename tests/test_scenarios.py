"""The scripted scenarios used by demo.py: each merchant reaches a different, expected end."""

import json

import pytest

import demo
import scenarios
from helpers import TODAY, case, make_env
from repayment import db as store

SCENARIOS = scenarios.build(TODAY)


@pytest.mark.parametrize("key", list(SCENARIOS))
def test_each_scenario_reaches_its_expected_state_using_its_whole_script(key):
    scenario = SCENARIOS[key]
    env = make_env()

    scenarios.run(env, scenario)

    assert case(env, scenario.case_id)["state"] == scenario.expected_state
    assert env.llm.turns == []  # every scripted decision was asked for


def test_the_merchants_take_different_paths():
    ends = {}
    for key in ("nile", "cairo", "delta", "alex", "giza", "redsea"):
        env = make_env()
        scenarios.run(env, SCENARIOS[key])
        tools = [row["tool"] for row in store.rows(env.db, "SELECT tool FROM steps ORDER BY rowid")]
        ends[key] = (case(env, SCENARIOS[key].case_id)["last_stop_reason"], tuple(tools))
    assert len(set(ends.values())) == 6
    assert ends["giza"][0] == "cap:broken_promises" and ends["redsea"][0] == "escalated"


def test_the_demo_prints_run_summaries_and_redacted_json_log_lines(capsys):
    demo.main()

    out = capsys.readouterr().out
    assert out.count("Run summary: stop ") == out.count(" -> AI ran: ") > 6  # one summary per AI run
    assert out.count("Logged: ") == 7  # the six merchants and the outage test
    json_lines = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert [line["kind"] for line in json_lines] == ["step", "run_summary"]
    assert "Hi [NAME]," in json_lines[0]["args"]["text"] and "Mona" not in json.dumps(json_lines)
