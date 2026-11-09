# A live run with the real Gemini API

This is a record of one short run against the real Gemini API. Everything below comes from the app's own data:
`GET /api/cases/case-cairo` (runs, steps, messages, plans) and the JSON log lines it stored. It has only been
shortened, not rewritten.

- **When:** Sun 8 Nov 2026, 19:25 UTC.
- **Merchant:** Cairo Crafts (owner Karim). Owes 18,000 EGP, 14 days late. Store sales are down about a third.
- **What was sent:** `POST /api/cases/case-cairo/start`, then one merchant reply, then `GET /api/cases/case-cairo`.
- **Models:** gemini-3.5-flash answered the first run. On the second run it returned 429 because its free daily
  quota was used up, so the adapter switched to gemini-3.5-flash-lite on its own. One gemini-3.5-flash-lite call
  returned 503 (overloaded). The adapter waited 2 seconds and retried it once, and the retry worked.
- **Requests to the Gemini API:** 5 in total. 3 got answers, 1 was the quota error and 1 was the overloaded
  error.

## Run 1: the case opens (trigger `CASE_OPENED`)

| Step | Tool | The AI's stated reason | Result |
|---|---|---|---|
| 1 | `send_message` | "Send initial payment reminder to Karim regarding the overdue balance of 18,000 EGP due on October 25, 2026." | Message sent |

The merchant received:

> Hello Karim, this is a friendly reminder regarding your Cairo Crafts account. Your payment of 18,000 EGP was due
> on 2026-10-25 and is currently overdue. Please let us know when you can make the payment or if you'd like to
> discuss a flexible payment plan. Thank you!

- **Stop reason:** `waiting_on_merchant` ("Sent a message and is waiting for the merchant.")
- **Usage:** 1 LLM call to gemini-3.5-flash. 1,683 tokens in and 834 out (124 for the answer plus 710 for
  thinking). It took 5.7 s and cost an estimated $0.0100.

## Run 2: the merchant replies (trigger `MERCHANT_REPLY`)

Merchant: **"Sales are slow this month. Can I pay in two parts?"**

| Step | Tool | The AI's stated reason | Result (written by code) |
|---|---|---|---|
| 1 | `get_sales_summary` | "The merchant mentioned slow sales and asked for a two-part payment plan, so I am checking the sales summary to evaluate the request." | Last 3 months average 51,000 EGP, before that 77,667 EGP (-34.3%). The balance is 35.3% of a month's sales. |
| 2 | `propose_plan` | "Proposing a 2-installment payment plan over 30 days to accommodate the merchant's request while meeting policy limits." | Plan 9,000 EGP on Sun 15 Nov + 9,000 EGP on Tue 8 Dec: offered to the merchant. |

Code then checked the plan against policy and recorded why: "2 payments, the last in 30 days. The AI may offer up
to 2 payments within 30 days by itself, so no approval is needed." The offer went to the merchant with Accept and
Decline buttons:

> Hello Karim, we understand sales have been slower recently. We can certainly split your 18,000 EGP balance into
> two equal installments: 9,000 EGP due on 15 November 2026, and 9,000 EGP due on 8 December 2026. Please let us
> know if this works for you!

- **Stop reason:** `waiting_on_merchant`. The case is WAITING: "A payment plan was offered; waiting for the
  merchant to accept or decline." (This run came before the `plan_offered` stop reason was added. The same run
  now stops with `plan_offered`.)
- **What happens next (set by code):** "If the merchant stays silent until Tue 10 Nov 19:25, the AI sends reminder
  1 of 3. After 3 reminders a person takes over." (Reminders now go out after one day of silence, so today this
  would say Mon 9 Nov 19:25.)
- **Usage:** 2 LLM calls to gemini-3.5-flash-lite. 4,025 tokens in and 224 out. It took 5.3 s, including the
  quota error, the 503 and the 2-second wait. Estimated cost: $0.0018.

## Totals

| | Run 1 | Run 2 | Whole case |
|---|---|---|---|
| LLM calls (answered) | 1 | 2 | 3 |
| Tokens in / out | 1,683 / 834 | 4,025 / 224 | 5,708 / 1,058 |
| Estimated cost | $0.0100 | $0.0018 | $0.0118 |

The costs are estimates at paid-tier prices (`PRICES` in `repayment/config.py`), so the free tier itself charged
nothing. This case can spend at most $0.50 on the AI.

## What this run shows

- The AI picked its own tools. It checked sales before offering a plan, and it kept the plan within what it may
  offer without a manager.
- Code made the decisions that matter. It checked the plan against policy and sent the offer. It also set the next
  timer. The AI has no tool that changes money or closes a case.
- Model fallback and the retry after a 503 worked against the real API, with no change to the case.
- Stored step arguments are redacted. That is why the stored copy of each message says "Hello [NAME]", while the
  merchant received "Hello Karim".

## The two `run_summary` log lines, exactly as stored

```json
{"ts": "2026-11-08T19:25:00+00:00", "kind": "run_summary", "case_id": "case-cairo", "run_id": "run-e2c7c560", "event": "CASE_OPENED", "stop_reason": "waiting_on_merchant", "stop_detail": "Sent: \"Hello [NAME], this is a friendly reminder regarding your Cairo Crafts account. Your payment of 18,000 EGP was due on 2...\"", "steps": 1, "llm_calls": 1, "tools_used": {"send_message": 1}, "tokens_in": 1683, "tokens_out": 834, "cost_usd": 0.01003, "duration_ms": 5663, "models": ["gemini-3.5-flash"]}
{"ts": "2026-11-08T19:25:00+00:00", "kind": "run_summary", "case_id": "case-cairo", "run_id": "run-9532577e", "event": "MERCHANT_REPLY", "stop_reason": "waiting_on_merchant", "stop_detail": "Plan 9,000 EGP on Sun 15 Nov + 9,000 EGP on Tue 8 Dec: offered to the merchant.", "steps": 2, "llm_calls": 2, "tools_used": {"get_sales_summary": 1, "propose_plan": 1}, "tokens_in": 4025, "tokens_out": 224, "cost_usd": 0.001767, "duration_ms": 5309, "models": ["gemini-3.5-flash-lite"]}
```

## A second live run: verify before answering

- **When:** Sun 8 Nov 2026, evening (Cairo). **Model:** gemini-3.5-flash-lite.
- **Merchant:** Alex Beauty (owner Salma). Owes 8,500 EGP, 9 days late. Has a bank transfer of 8,500 EGP that is
  still waiting for the bank's confirmation.
- **Requests to the Gemini API:** 3, all answered. Estimated cost: $0.0024.

Run 1 (`CASE_OPENED`): one `send_message`, a friendly reminder with the exact amount. Stop: `waiting_on_merchant`.

Run 2 (`MERCHANT_REPLY`), merchant: **"I already paid last week by bank transfer!"**

| Step | Tool | The AI's stated reason | Result (written by code) |
|---|---|---|---|
| 1 | `check_payments` | "The merchant stated they paid last week by bank transfer, so I need to check the payment provider for balance and pending transfers." | Balance 8,500 EGP; 0 applied to this case; 1 pending (8,500 EGP). |
| 2 | `send_message` | "Inform the merchant that we see their transfer pending from Nov 6 and we will update the account once the bank confirms it." | Message sent |

The merchant received:

> Hi Salma, thank you for letting us know! We see your bank transfer of 8,500 EGP from November 6 is currently
> pending bank confirmation (reference: pay-9f98286a). As soon as the bank confirms it, your account will be
> updated automatically. Thanks for your patience!

The AI checked the claim with a tool before answering, and it did not mark anything as paid: it can't. The case
stays WAITING until the bank confirms the transfer (the **Confirm pending bank transfer** test tool), and then code
closes it.
