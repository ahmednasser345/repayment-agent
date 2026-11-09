"""The model behind a small interface, so the loop works the same with Gemini or with a scripted fake.

Conversation format passed to complete():
    {"role": "user", "text": str}
    {"role": "model", "response": LLMResponse}
    {"role": "tool", "results": [{"call_id": str, "name": str, "content": dict, "is_error": bool}]}
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    raw: Any = None  # provider data the adapter needs back on the next call (Gemini: the model's own Content)
    finish: str = ""


class LLMUnavailable(Exception):
    """No model could answer right now. The run pauses and retries later, like any other system that is down."""


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...

    def status(self) -> dict[str, Any]: ...


class ScriptExhausted(RuntimeError):
    pass


class ScriptedLLM:
    """Plays back hand-written decisions, so tests and the demo run without a key.

    Each turn is a list of (tool_name, args), or an exception to raise. An empty list is a reply with no tool call.
    """

    TOKENS_IN, TOKENS_OUT = 1000, 100

    def __init__(self, turns: list[Any]):
        self.turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []  # the conversation sent on each call, for tests
        self._ids = itertools.count(1)

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.requests.append(list(messages))
        if not self.turns:
            raise ScriptExhausted(f"ScriptedLLM: the script ran out of turns after {len(self.requests) - 1} calls")
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        calls = [ToolCall(f"call-{next(self._ids)}", name, args) for name, args in turn]
        return LLMResponse(calls, "", self.TOKENS_IN, self.TOKENS_OUT, "scripted", None, "STOP")

    def status(self) -> dict[str, Any]:
        return {"provider": "Scripted (no AI)", "models": ["scripted"],
                "last_model": "scripted" if self.requests else None, "unavailable": []}
