"""Limits, policy, models and prices. Every number the code enforces lives here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Caps enforced by code. Hitting any of them hands the case to a person."""

    max_steps_per_run: int = 8
    max_llm_calls_per_run: int = 6
    max_malformed_retries: int = 2
    max_no_progress_steps: int = 3
    max_reminders: int = 3
    reminder_after_hours: int = 24
    max_broken_promises: int = 2
    max_missed_installments: int = 2
    max_messages_per_case: int = 15
    max_tool_attempts: int = 3
    tool_backoff_seconds: float = 0.2
    max_tool_retries: int = 3
    tool_retry_minutes: int = 15
    case_deadline_days: int = 45
    max_cost_usd_per_case: float = 0.50
    max_tokens_per_case: int = 400_000


@dataclass(frozen=True)
class Policy:
    """What the AI may agree to without a person."""

    promise_max_days: int = 14
    plan_min_installments: int = 2
    plan_max_installments: int = 3
    plan_first_due_max_days: int = 7
    plan_max_days: int = 60
    auto_approve_max_installments: int = 2
    auto_approve_max_days: int = 30
    min_installment: int = 500
    message_max_chars: int = 600


# Tried in this order. Free tier: about 20 requests a day and 5 a minute, per model.
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]

# USD per 1M tokens (input, output), paid tier, standard. Thinking tokens bill as output.
# Checked 2026-11-08 on https://ai.google.dev/gemini-api/docs/pricing.
PRICES = {
    # Not listed on the pricing page on 2026-11-08, so a cautious estimate: at or above every Flash price below,
    # including 3.8 Flash's price from 2027.
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.8-flash": (0.75, 3.75),  # introductory price until 2026-12-31; 1.50 / 7.50 from 2027-01-01
}
PRICES["scripted"] = PRICES["gemini-3.5-flash"]  # the offline ScriptedLLM, priced so the budget cap is exercised
UNKNOWN_PRICE = (3.00, 15.00)  # deliberately high, so the budget cap still bites for an unlisted model


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES.get(model, UNKNOWN_PRICE)
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
