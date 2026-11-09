"""Gemini behind the LLM interface: function calling, model fallback and rate-limit handling.

    python -m repayment.gemini      # list the models your key can use

The agent loop keeps its own conversation format. This adapter translates it on every call: model turns go back
as Gemini's own Content objects (so thought signatures return unchanged), and tool results become function
responses. A run sticks to one model, because thought signatures only work with the model that made them.
"""

import importlib
import itertools
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .config import DEFAULT_MODELS
from .llm import LLMResponse, LLMUnavailable, ToolCall

MAX_OUTPUT_TOKENS = 2048
TIMEOUT_MS = 60_000
RATE_LIMIT_WAIT = 15.0  # seconds, when the server does not say how long to wait
RATE_LIMIT_WAIT_MAX = 30.0
SERVER_ERROR_WAIT = 2.0
QUOTA_RESET_HOUR_UTC = 8  # free-tier daily quotas reset at midnight Pacific time


class _TryNextModel(Exception):
    """This model cannot answer now. The message ends up in the summary if no model can."""


class GeminiLLM:
    def __init__(self, models: list[str] | None = None, client: Any = None,
                 sleep: Callable[[float], None] = time.sleep, now: Callable[[], float] = time.time):
        from google import genai
        from google.genai import errors, types

        self.types, self.errors = types, errors
        self.models = models or _models_from_env() or list(DEFAULT_MODELS)
        self.client = client or genai.Client()  # reads GEMINI_API_KEY or GOOGLE_API_KEY from the environment
        self.sleep, self.now = sleep, now
        self.unavailable: dict[str, float] = {}  # model -> time (epoch seconds) it may be tried again
        self.last_model: str | None = None
        self._ids = itertools.count(1)
        self._transport_errors = _transport_errors()

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        config, contents, problems = self._config(system, tools), self.contents(messages), []
        for model in self._candidates(messages, problems):
            try:
                response = self._generate(model, contents, config)
            except _TryNextModel as exc:
                problems.append(f"{model}: {exc}")
                continue
            self.last_model = model
            return self.parse(response, model)
        raise LLMUnavailable("No Gemini model answered. " + "; ".join(problems))

    def status(self) -> dict[str, Any]:
        return {"provider": "Gemini", "models": list(self.models), "last_model": self.last_model,
                "unavailable": [model for model in self.models if not self._available(model)]}

    def contents(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Our conversation -> Gemini contents. Consecutive user turns are merged: Gemini wants them to alternate."""
        types, contents, gemini_ids = self.types, [], {}
        for message in messages:
            if message["role"] == "model":
                raw = message["response"].raw or {}
                gemini_ids.update(raw.get("ids", {}))
                if raw.get("content") is not None:
                    contents.append(raw["content"])  # unchanged, thought signatures included
                continue
            if message["role"] == "user":
                parts = [types.Part(text=message["text"])]
            else:
                parts = [self._function_response(result, gemini_ids) for result in message["results"]]
            if contents and contents[-1].role == "user":
                contents[-1] = types.Content(role="user", parts=[*contents[-1].parts, *parts])
            else:
                contents.append(types.Content(role="user", parts=parts))
        return contents

    def parse(self, response: Any, model: str) -> LLMResponse:
        candidate = (response.candidates or [None])[0]
        content = getattr(candidate, "content", None)
        parts = (content.parts if content is not None else None) or []
        calls, gemini_ids, text = [], {}, []
        for part in parts:
            if part.function_call:
                call_id = f"g{next(self._ids)}"
                gemini_ids[call_id] = part.function_call.id
                calls.append(ToolCall(call_id, part.function_call.name or "", dict(part.function_call.args or {})))
            elif part.text and not part.thought:
                text.append(part.text)
        usage = response.usage_metadata
        finish = getattr(candidate, "finish_reason", None)
        return LLMResponse(
            calls=calls,
            text=" ".join(text).strip(),
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
            model=model,
            raw={"content": content if parts else None, "ids": gemini_ids},
            finish=str(getattr(finish, "name", finish) or ""),
        )

    def _candidates(self, messages: list[dict[str, Any]], problems: list[str]) -> list[str]:
        used = next((m["response"].model for m in messages if m["role"] == "model"), None)
        models = [used] if used is not None else self.models  # a run sticks to the model it started with
        for model in models:
            if not self._available(model):
                problems.append(f"{model}: out of quota or not available to this key")
        return [model for model in models if self._available(model)]

    def _available(self, model: str) -> bool:
        return self.unavailable.get(model, 0.0) <= self.now()

    def _generate(self, model: str, contents: list[Any], config: Any) -> Any:
        """At most two attempts on one model; anything else moves on to the next model."""
        for attempt in (1, 2):
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=config)
            except self.errors.ClientError as exc:
                if exc.code == 429 and _per_day_quota(exc):
                    self.unavailable[model] = _next_quota_reset(self.now())
                    raise _TryNextModel("daily quota used up") from exc
                if exc.code == 429 and attempt == 1:
                    self.sleep(min(_retry_delay(exc) or RATE_LIMIT_WAIT, RATE_LIMIT_WAIT_MAX))
                    continue
                if exc.code == 429:
                    raise _TryNextModel("rate limited") from exc
                if exc.code == 404:
                    self.unavailable[model] = math.inf  # for the rest of this process
                    raise _TryNextModel("not available to this key (404)") from exc
                if exc.code == 408:
                    raise _TryNextModel("timed out") from exc
                raise  # any other 4xx means our request is wrong: a bug to surface, not an outage
            except self.errors.ServerError as exc:
                if attempt == 2:
                    raise _TryNextModel(f"server error {exc.code}") from exc
                self.sleep(SERVER_ERROR_WAIT)
            except self._transport_errors as exc:
                raise _TryNextModel(f"timeout or connection error ({type(exc).__name__})") from exc
        raise _TryNextModel("no answer")

    def _config(self, system: str, tools: list[dict[str, Any]]) -> Any:
        types = self.types
        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=t["parameters"]) for t in tools]
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            # One attempt per request: _generate decides about retries and switching models.
            http_options=types.HttpOptions(timeout=TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)),
        )

    def _function_response(self, result: dict[str, Any], gemini_ids: dict[str, str | None]) -> Any:
        types = self.types
        response = {"error": result["content"]} if result["is_error"] else {"output": result["content"]}
        return types.Part(function_response=types.FunctionResponse(
            id=gemini_ids.get(result["call_id"]), name=result["name"], response=response))


def _error_details(exc: Any) -> list[dict[str, Any]]:
    """The google.rpc details (QuotaFailure, RetryInfo, ...) of an API error body: {"error": {"details": [...]}}."""
    body = exc.details if isinstance(exc.details, dict) else {}
    error = body.get("error", body)
    details = error.get("details") if isinstance(error, dict) else None
    return [detail for detail in details or [] if isinstance(detail, dict)]


def _per_day_quota(exc: Any) -> bool:
    """A QuotaFailure whose quotaId names a daily quota, e.g. GenerateRequestsPerDayPerProjectPerModel-FreeTier."""
    return "PerDay" in json.dumps(_error_details(exc))


def _retry_delay(exc: Any) -> float | None:
    for detail in _error_details(exc):
        if detail.get("@type", "").endswith("RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def _next_quota_reset(now: float) -> float:
    moment = datetime.fromtimestamp(now, timezone.utc)
    reset = moment.replace(hour=QUOTA_RESET_HOUR_UTC, minute=0, second=0, microsecond=0)
    if reset <= moment:
        reset += timedelta(days=1)
    return reset.timestamp()


def _models_from_env() -> list[str]:
    return [model.strip() for model in os.environ.get("GEMINI_MODELS", "").split(",") if model.strip()]


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Timeouts and connection failures from whichever HTTP library google-genai uses (httpx2 or httpx)."""
    found: list[type[BaseException]] = [TimeoutError, ConnectionError]
    for name in ("httpx2", "httpx"):
        try:
            found.append(importlib.import_module(name).TransportError)
        except ImportError:
            continue
    return tuple(found)


def main() -> int:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("Set GEMINI_API_KEY first. Get a free key at https://aistudio.google.com/apikey")
        return 1
    from google import genai

    client = genai.Client()  # keep this reference while listing: the pager fetches lazily through it
    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name.removeprefix("models/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
