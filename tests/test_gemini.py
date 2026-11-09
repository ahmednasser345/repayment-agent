"""The Gemini adapter with a fake client and the real google-genai types: no network, no key."""

import json
from datetime import datetime, timedelta, timezone

import httpx2
import pytest
from google.genai import errors, types

from helpers import START, case, make_env, reply, start
from repayment.gemini import GeminiLLM
from repayment.llm import LLMResponse, LLMUnavailable, ToolCall
from repayment.tools import definitions

NOW = datetime(2026, 11, 8, 14, 0, tzinfo=timezone.utc).timestamp()
CASE_FILE = [{"role": "user", "text": "CASE FILE"}]


def answer(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)), finish_reason="STOP")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000, candidates_token_count=100, thoughts_token_count=50),
    )


def function_call(name: str, args: dict, call_id: str = "fc-1", signature: bytes = b"sig") -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args), thought_signature=signature)


def quota_error(per_day: bool, retry_delay: str = "7s") -> errors.ClientError:
    """The body Gemini really sends for a free-tier 429 (checked against a live response)."""
    quota_id = f"GenerateRequestsPer{'Day' if per_day else 'Minute'}PerProjectPerModel-FreeTier"
    return errors.ClientError(429, {"error": {
        "code": 429, "message": "You exceeded your current quota.", "status": "RESOURCE_EXHAUSTED", "details": [
            {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                 "quotaId": quota_id, "quotaValue": "20"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
        ]}})


def server_error(code: int = 503) -> errors.ServerError:
    return errors.ServerError(code, {"error": {"code": code, "message": "The model is overloaded.",
                                               "status": "UNAVAILABLE"}})


class FakeModels:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": [c.model_copy(deep=True) for c in contents],
                              "config": config})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses: list):
        self.models = FakeModels(responses)


def gemini(responses: list, models: tuple[str, ...] = ("model-a", "model-b"), now: float = NOW) -> GeminiLLM:
    llm = GeminiLLM(models=list(models), client=FakeClient(responses), now=lambda: now)
    llm.waits = []
    llm.sleep = llm.waits.append
    return llm


def asked(llm: GeminiLLM) -> list[str]:
    return [request["model"] for request in llm.client.models.requests]


def test_a_whole_case_runs_through_gemini_and_thought_signatures_go_back_unchanged():
    llm = gemini([
        answer(function_call("send_message", {"text": "Hi Salma, 8,500 EGP is overdue.", "reason": "First contact."},
                             "fc-1", b"sig-1")),
        answer(function_call("check_payments", {"reason": "She says she paid."}, "fc-2", b"sig-2")),
        answer(function_call("send_message", {"text": "We see your pending transfer.", "reason": "Reassure."},
                             "fc-3", b"sig-3")),
    ])
    env = make_env()
    env.llm = llm
    start(env, "case-alex")
    outcome = reply(env, "case-alex", "I already paid last week by bank transfer!")

    assert outcome.stop_reason == "waiting_on_merchant"
    config = llm.client.models.requests[0]["config"]
    assert config.tool_config.function_calling_config.mode == "ANY"
    assert config.automatic_function_calling.disable is True
    assert config.max_output_tokens == 2048
    assert (config.http_options.timeout, config.http_options.retry_options.attempts) == (60_000, 1)
    assert config.thinking_config is None
    declaration = {d.name: d for d in config.tools[0].function_declarations}["propose_plan"]
    assert declaration.parameters_json_schema["additionalProperties"] is False

    contents = llm.client.models.requests[-1]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].thought_signature == b"sig-2"
    response = contents[2].parts[0].function_response
    assert (response.id, response.name) == ("fc-2", "check_payments")
    assert response.response["output"]["pending"][0]["amount"] == 8_500
    alex = case(env, "case-alex")
    assert (alex["tokens_in"], alex["tokens_out"]) == (3_000, 450)  # thinking tokens count as output


def test_results_go_back_to_the_right_function_ids_and_user_turns_merge():
    llm = gemini([answer(function_call("check_payments", {"reason": "r"}, "fc-a"),
                         function_call("get_sales_summary", {"reason": "r"}, "fc-b"))])
    first = llm.complete("system", CASE_FILE, definitions())
    conversation = CASE_FILE + [
        {"role": "model", "response": first},
        {"role": "tool", "results": [
            {"call_id": first.calls[0].id, "name": "check_payments", "content": {"balance": 1}, "is_error": False},
            {"call_id": first.calls[1].id, "name": "get_sales_summary", "content": {"error": "x"}, "is_error": True},
        ]},
        {"role": "user", "text": "Reply with a tool call."},
    ]

    contents = llm.contents(conversation)

    assert [c.role for c in contents] == ["user", "model", "user"]
    parts = contents[2].parts
    assert [(p.function_response.id, p.function_response.name) for p in parts[:2]] == [
        ("fc-a", "check_payments"), ("fc-b", "get_sales_summary")]
    assert parts[0].function_response.response == {"output": {"balance": 1}}
    assert parts[1].function_response.response == {"error": {"error": "x"}}
    assert parts[2].text == "Reply with a tool call."


def test_a_daily_quota_moves_to_the_next_model_and_is_remembered_until_it_resets():
    llm = gemini([quota_error(per_day=True), answer(function_call("check_payments", {"reason": "r"})),
                  answer(function_call("check_payments", {"reason": "r"}))])

    first = llm.complete("system", CASE_FILE, definitions())
    llm.complete("system", CASE_FILE, definitions())  # a new run: model-a is not tried again

    assert first.model == "model-b"
    assert asked(llm) == ["model-a", "model-b", "model-b"]
    assert llm.status()["unavailable"] == ["model-a"] and llm.waits == []
    llm.now = lambda: NOW + 24 * 3600  # past 08:00 UTC the next day: the quota is back
    assert llm.status()["unavailable"] == []


def test_a_per_minute_limit_waits_the_server_delay_once_then_tries_again():
    llm = gemini([quota_error(per_day=False, retry_delay="7s"),
                  answer(function_call("check_payments", {"reason": "r"}))])

    llm.complete("system", CASE_FILE, definitions())

    assert asked(llm) == ["model-a", "model-a"] and llm.waits == [7.0]


def test_a_second_rate_limit_moves_to_the_next_model():
    llm = gemini([quota_error(per_day=False, retry_delay="45s"), quota_error(per_day=False),
                  answer(function_call("check_payments", {"reason": "r"}))])

    llm.complete("system", CASE_FILE, definitions())

    assert asked(llm) == ["model-a", "model-a", "model-b"] and llm.waits == [30.0]  # never more than 30 s


def test_a_server_error_is_retried_once_after_two_seconds_then_the_next_model_is_tried():
    llm = gemini([server_error(), server_error(), answer(function_call("check_payments", {"reason": "r"}))])

    result = llm.complete("system", CASE_FILE, definitions())

    assert asked(llm) == ["model-a", "model-a", "model-b"] and llm.waits == [2.0]
    assert result.model == "model-b" and llm.status()["last_model"] == "model-b"


def test_a_missing_model_and_a_timeout_move_on_and_only_the_missing_model_is_remembered():
    llm = gemini([errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}}),
                  httpx2.ReadTimeout("timed out"), answer(function_call("check_payments", {"reason": "r"}))],
                 models=("model-a", "model-b", "model-c"))

    assert llm.complete("system", CASE_FILE, definitions()).model == "model-c"
    assert llm.status()["unavailable"] == ["model-a"]


def test_when_no_model_answers_the_run_pauses_with_a_summary():
    llm = gemini([server_error(), server_error(), quota_error(per_day=True)])

    with pytest.raises(LLMUnavailable, match="model-a: server error 503; model-b: daily quota used up"):
        llm.complete("system", CASE_FILE, definitions())


def test_any_other_client_error_is_our_bug_and_surfaces():
    llm = gemini([errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})])

    with pytest.raises(errors.ClientError):
        llm.complete("system", CASE_FILE, definitions())


def test_a_run_sticks_to_the_model_it_started_with():
    llm = gemini([server_error(), server_error()])
    earlier = LLMResponse([ToolCall("g1", "check_payments", {"reason": "r"})], model="model-b",
                          raw={"content": types.Content(role="model", parts=[types.Part(text="...")]), "ids": {}})
    conversation = CASE_FILE + [{"role": "model", "response": earlier}]

    with pytest.raises(LLMUnavailable):
        llm.complete("system", conversation, definitions())

    assert asked(llm) == ["model-b", "model-b"]  # model-a is never mixed into model-b's run


def test_an_answer_without_a_call_has_no_calls_and_hides_thoughts():
    llm = gemini([answer(types.Part(text="thinking...", thought=True), types.Part(text="I think it is fine."))])

    result = llm.complete("system", CASE_FILE, definitions())

    assert (result.calls, result.text, result.finish) == ([], "I think it is fine.", "STOP")


def test_models_can_be_chosen_with_an_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "gemini-x, gemini-y")
    assert GeminiLLM(client=FakeClient([])).models == ["gemini-x", "gemini-y"]


def test_an_ai_outage_through_gemini_pauses_the_case():
    env = make_env()
    env.llm = gemini([server_error()] * 4)

    outcome = start(env, "case-nile")

    assert outcome.stop_reason == "paused:ai_unavailable"
    nile = case(env, "case-nile")
    assert (nile["state"], nile["retry_at"]) == ("WAITING", (START + timedelta(minutes=15)).isoformat())
    assert json.loads(nile["pending_trigger_json"])["type"] == "CASE_OPENED"
