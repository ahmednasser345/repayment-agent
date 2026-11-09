"""Pure rules: what the AI may promise, offer or say. Tools check these before anything changes."""

import re
from datetime import date, timedelta
from typing import Any, Mapping

from .config import Policy

THREATS = re.compile(
    r"\b(?:lawsuits?|sue[ds]?|suing|courts?|police|legal\s+action|prison|jail|blacklist(?:ed|ing)?)\b"
    r"|محكمة|قضية|الشرطة|بلاغ|محامي",  # Arabic words often carry prefixes, so no word boundaries here
    re.IGNORECASE,
)
INSTRUCTION_LIKE = re.compile("|".join([
    r"\b(?:ignore|disregard)\b.{0,40}?\b(?:previous|prior|your|all)\b.{0,20}?\binstructions?\b",
    r"\bsystem\s+(?:prompt|message|note)\b",
    r"(?:^|\n)\s*system\s*:",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(?:a|an|the|my|if)\b",
    r"\bmark\b.{0,40}?\bas\s+paid\b",
    r"\bset\s+my\s+balance\b",
    r"\b(?:waive|waiving|waived|cancel|cancell?ing)\b.{0,30}?\b(?:debt|balance|loan)\b",
    r"\bapprov(?:e|ed)\b.{0,40}?\b(?:waiv\w*|discount\w*|cancel\w*|clear\w*|paid)\b",
    r"تجاهل.{0,30}التعليمات",  # "ignore ... the instructions"
    r"(?:إلغاء|الغاء|إسقاط|اسقاط).{0,10}(?:الدين|المديونية)",  # "cancel / drop the debt"
]), re.IGNORECASE)


def parse_date(text: Any) -> date | None:
    try:
        return date.fromisoformat(text) if isinstance(text, str) and len(text) == 10 else None
    except ValueError:
        return None


def check_promise(
    amount: int, due_date: str, balance: int, today: date, open_promise: Mapping | None, has_active_plan: bool,
    policy: Policy = Policy(),
) -> list[str]:
    errors = []
    if not isinstance(amount, int) or not 1 <= amount <= balance:
        errors.append(f"The amount must be a whole number from 1 to the balance ({balance:,} EGP).")
    due, latest = parse_date(due_date), today + timedelta(days=policy.promise_max_days)
    if due is None:
        errors.append(f"'{due_date}' is not a valid date. Use YYYY-MM-DD.")
    elif not today <= due <= latest:
        errors.append(f"The date must be between today ({today}) and {latest} ({policy.promise_max_days} days).")
    if open_promise is not None:
        errors.append(f"There is already an open promise ({open_promise['amount']:,} EGP by "
                      f"{open_promise['due_date']}). Wait until it is kept or broken.")
    if has_active_plan:
        errors.append("The merchant is on an active payment plan; promises are not used on a plan.")
    return errors


def check_plan(installments: list[dict], balance: int, today: date, policy: Policy) -> tuple[list[str], bool, str]:
    """Returns (errors, needs_approval, why). why is a plain sentence for the manager and the dashboard."""
    errors, count = [], len(installments)
    low, high = policy.plan_min_installments, policy.plan_max_installments
    if count == 1:
        errors.append("A single payment is not a plan: use record_promise for a single payment.")
    elif not low <= count <= high:
        errors.append(f"A plan must have {low} to {high} installments, not {count}.")
    amounts = [item["amount"] for item in installments]
    if any(amount < policy.min_installment for amount in amounts):
        errors.append(f"Each installment must be at least {policy.min_installment:,} EGP.")
    if sum(amounts) != balance:
        errors.append(f"The installments must add up to the balance exactly ({balance:,} EGP); they add up to "
                      f"{sum(amounts):,} EGP. Discounts and waivers are not allowed.")
    dates = [parse_date(item["due_date"]) for item in installments]
    if None in dates:
        errors.append("Every due_date must be a real date in YYYY-MM-DD format.")
    else:
        errors += _plan_date_errors(dates, today, policy)
    if errors:
        return errors, False, ""
    last_in = (dates[-1] - today).days
    plan = f"{count} payments, the last in {last_in} days."
    rule = (f"The AI may offer up to {policy.auto_approve_max_installments} payments within "
            f"{policy.auto_approve_max_days} days by itself")
    if count > policy.auto_approve_max_installments or last_in > policy.auto_approve_max_days:
        return [], True, f"{plan} {rule}, so a manager must approve."
    return [], False, f"{plan} {rule}, so no approval is needed."


def _plan_date_errors(dates: list[date], today: date, policy: Policy) -> list[str]:
    errors = []
    if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
        errors.append("Due dates must be in order, each one after the previous one.")
    first_by = today + timedelta(days=policy.plan_first_due_max_days)
    if not today <= dates[0] <= first_by:
        errors.append(f"The first payment must be due between today ({today}) and {first_by}.")
    last_by = today + timedelta(days=policy.plan_max_days)
    if dates[-1] > last_by:
        errors.append(f"The last payment must be due by {last_by} ({policy.plan_max_days} days from today).")
    return errors


def check_message(text: str, last_assistant_text: str | None, policy: Policy) -> list[str]:
    errors, body = [], text.strip()
    if not 1 <= len(body) <= policy.message_max_chars:
        errors.append(f"Messages must be 1 to {policy.message_max_chars} characters (this one has {len(body)}).")
    if THREATS.search(body):
        errors.append("Never mention courts, lawyers, police, legal action or other threats. Rewrite it kindly.")
    if last_assistant_text is not None and body == last_assistant_text.strip():
        errors.append("This is the same text as our previous message. Write something new.")
    return errors


def instruction_like(text: str) -> bool:
    """Flags merchant text that tries to give the AI orders. Only a flag: no tool can change money anyway."""
    return bool(INSTRUCTION_LIKE.search(text))
