# Jev + Groq hybrid: support ticket triage (Python)

Jev makes the fast, structured decisions. A free Groq model writes text, but
only for tickets that actually need text.

```
ticket ──► Jev (1 call, 5 typed questions, evaluated in parallel)
             ├─ needs_reply    noul   (P true)
             ├─ department     choice (+ probabilities, confidence)
             ├─ urgency        score 0-3 (+ probabilities, confidence)
             ├─ wants_refund   noul
             └─ sensitive      noul
             ▼
        route()  plain Python, thresholds on the numbers
             ▼
   archive          auto_reply              human_review
   (no LLM call)    Groq drafts a reply     Groq writes a handoff note
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_offline.py      # no keys needed: checks routing and request shapes
python main.py --check      # verifies both keys and that your Groq model exists
python main.py              # real run over 8 sample tickets
python main.py --compare    # also runs the same questions through Groq alone
```

Requires Python 3.10+.

## Keys

- **Groq** (free, no credit card): https://console.groq.com/keys
- **Jev**, pick one:
  - TypeSafe directly: create a key at https://console.typesafe.ai and set `TYPESAFE_API_KEY`.
    Access has been reported as waitlisted early access, so you may need to request it.
  - Vercel AI Gateway: set `AI_GATEWAY_API_KEY`. Same SDK, different base URL and model ID;
    the code switches automatically.

## Cost, honestly

Groq is free (rate-limited). **Jev is not free, but it is very cheap.** TypeSafe lists
about $0.042 per million input tokens with output tokens free. One triage call is
roughly 280 input tokens, so 1,000 tickets is on the order of a cent. Check the
current price before relying on this.

## Groq model note

Groq retired `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` on 2026-08-16,
including on the free tier, so older tutorials will fail with those IDs. The default
here is `openai/gpt-oss-20b` (Groq's suggested replacement for the 8B). Their catalog
changes often, so `python main.py --check` lists what your key can actually use.
Override with `GROQ_MODEL` in `.env`. gpt-oss models are reasoning models; the code
sets `reasoning_effort="low"` to keep free-tier token budgets healthy.

## What to poke at while learning

- **Confidence vs. probability.** Choice and Score answers carry `confidence`,
  derived from the probability distribution. Noul answers only carry P(true).
  Print `d.department_probs` and watch how the shape changes on murky tickets like T-107.
- **Thresholds in `pipeline.py`** are starting guesses. Write 30-50 labeled tickets
  of your own and see where auto-reply goes wrong at 0.80 versus 0.90.
- **`--compare`** shows Groq answering the same questions as JSON. It gives no
  calibrated probabilities, so it can't drive the routing the way Jev can. Compare
  latency and agreement, not lanes.
- **Edit `QUESTIONS`.** TypeSafe advises atomic, narrow questions combined in code.
  Add one and see how the lanes shift.

## Caveats

- `test_offline.py` uses fabricated Jev answers to test my routing logic and request
  handling. It says nothing about how Jev performs on these tickets.
- The live API calls were not run in development (no keys). `--check` is the first
  thing to run.
- Speed and cost multiples quoted for Jev are TypeSafe's own claims. The summary at
  the end of `python main.py` gives you your own numbers.
