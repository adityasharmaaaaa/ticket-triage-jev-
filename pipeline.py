"""Jev + Groq hybrid ticket triage.

Jev makes the fast, structured decisions. Groq writes text,
but only for tickets that actually need text.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from groq import Groq
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from tickets import Ticket

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"

DEPARTMENTS = ("billing", "technical", "account", "sales", "other")

THRESHOLDS = {
    "department_confidence": 0.80,  # min confidence to auto-handle
    "urgency_confidence": 0.50,  # below this the urgency read is a coin flip
    "reply_needed_floor": 0.20,  # P(needs reply) below this: archive, no LLM
    "reply_needed_unsure": 0.70,  # below this: a human should look
    "sensitive": 0.50,  # P(legal / security / churn risk) that forces review
    "urgency_score": 2.5,  # on a 0-3 rubric; above this a human takes over
}

QUESTIONS = {
    "needs_reply": Noul(
        instructions=(
            "Does this message need a reply from support? "
            "(Spam, automated notices, and simple thank-yous do not.)"
        ),
    ),
    "department": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Charges, invoices, refunds, payment failures",
            "technical": "Bugs, outages, errors, integrations, SSO problems",
            "account": "Login, email, password, permissions, account settings",
            "sales": "Pricing, quotes, upgrades, new purchases",
            "other": "Anything else, or unclear",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=[
            "Low: general question, no time pressure",
            "Medium: annoying, but there is a workaround",
            "High: blocking the customer from working",
            "Critical: outage, data loss, or security exposure",
        ],
    ),
    "wants_refund": Noul(instructions="Is the customer asking for money back?"),
    "sensitive": Noul(
        instructions=(
            "Does this involve a legal threat, security incident, harassment, "
            "or a customer about to cancel?"
        ),
    ),
}


def make_jev_client(transport=None) -> tuple[TypeSafeClient, str]:
    """Build a Jev client from whichever key is configured.

    Both routes use the same SDK; only the key, base URL, and model ID differ.
    """
    kwargs = {"transport": transport} if transport is not None else {}
    typesafe_key = os.getenv("TYPESAFE_API_KEY")
    gateway_key = os.getenv("AI_GATEWAY_API_KEY")

    if typesafe_key:
        model = os.getenv("JEV_MODEL", "jev-latest")
        return TypeSafeClient(api_key=typesafe_key, model=model, **kwargs), model
    if gateway_key:
        model = os.getenv("JEV_MODEL", "typesafe-ai/jev")
        client = TypeSafeClient(
            api_key=gateway_key, base_url=GATEWAY_BASE_URL, model=model, **kwargs
        )
        return client, model
    raise SystemExit(
        "No Jev key found. Set TYPESAFE_API_KEY (console.typesafe.ai) or "
        "AI_GATEWAY_API_KEY (Vercel AI Gateway) in your .env file."
    )


@dataclass
class Decision:
    needs_reply: float  # P(true)
    department: str
    department_confidence: float
    department_probs: dict[str, float]
    urgency: float  # 0-3, fractional
    urgency_confidence: float
    wants_refund: float  # P(true)
    sensitive: float  # P(true)
    ms: float


def decide(client: TypeSafeClient, message: str) -> Decision:
    start = time.perf_counter()
    r = client.system_one(state={"message": message}, questions=QUESTIONS)
    ms = (time.perf_counter() - start) * 1000

    dept = r.choices["department"]
    urg = r.scores["urgency"]
    return Decision(
        needs_reply=r.nouls["needs_reply"].noul,
        department=dept.choice,
        department_confidence=dept.confidence,
        department_probs=dept.probabilities,
        urgency=urg.score,
        urgency_confidence=urg.confidence,
        wants_refund=r.nouls["wants_refund"].noul,
        sensitive=r.nouls["sensitive"].noul,
        ms=ms,
    )



ARCHIVE, AUTO_REPLY, HUMAN_REVIEW = "archive", "auto_reply", "human_review"


def route(d: Decision) -> tuple[str, list[str]]:
    t = THRESHOLDS
    reasons: list[str] = []

    # Risk always wins, even if the message looks like noise.
    if d.sensitive >= t["sensitive"]:
        reasons.append(f"sensitive (P={d.sensitive:.2f})")
    if d.urgency >= t["urgency_score"]:
        reasons.append(f"urgency {d.urgency:.1f}/3")
    if reasons:
        return HUMAN_REVIEW, reasons

    if d.needs_reply < t["reply_needed_floor"]:
        return ARCHIVE, ["no reply needed"]

    if d.department_confidence < t["department_confidence"]:
        reasons.append(f"department unsure (confidence {d.department_confidence:.2f})")
    if d.urgency_confidence < t["urgency_confidence"]:
        reasons.append(f"urgency unclear (confidence {d.urgency_confidence:.2f})")
    if d.needs_reply < t["reply_needed_unsure"]:
        reasons.append(f"unclear if reply needed (P={d.needs_reply:.2f})")

    if reasons:
        return HUMAN_REVIEW, reasons
    return AUTO_REPLY, ["confident, low risk"]




def make_groq_client(http_client=None) -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit(
            "GROQ_API_KEY is missing. Create a free key at "
            "https://console.groq.com/keys and add it to your .env file."
        )
    return Groq(api_key=key, http_client=http_client) if http_client else Groq(api_key=key)


def _chat(
    groq: Groq, system: str, user: str, *, max_tokens: int, json_mode: bool = False
) -> tuple[str, float]:
    extra: dict = {}
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        # gpt-oss models reason before answering. Keep it short so free-tier
        # token limits last, and keep the reasoning out of the reply.
        extra.update(reasoning_effort="low", include_reasoning=False)
    if json_mode:
        extra["response_format"] = {"type": "json_object"}

    start = time.perf_counter()
    resp = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
        temperature=0.3,
        **extra,
    )
    ms = (time.perf_counter() - start) * 1000

    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        raise RuntimeError(
            f"Groq returned empty content (finish_reason={choice.finish_reason}). "
            "Reasoning models can use up the token budget; raise max_tokens or "
            "try another model via GROQ_MODEL."
        )
    return text, ms


def _summary(d: Decision) -> str:
    return json.dumps(
        {
            "department": d.department,
            "urgency": round(d.urgency, 1),
            "wants_refund": d.wants_refund >= 0.5,
        }
    )


def draft_reply(groq: Groq, message: str, d: Decision) -> tuple[str, float]:
    return _chat(
        groq,
        "You are a friendly support agent. Write a short reply (under 90 words). "
        "Do not promise refunds or timelines you cannot guarantee. Say which team "
        "is on it. No subject line.",
        f"Ticket:\n{message}\n\nAutomated triage (already decided, do not "
        f"second-guess):\n{_summary(d)}",
        max_tokens=700,
    )


def write_handoff(
    groq: Groq, message: str, d: Decision, reasons: list[str]
) -> tuple[str, float]:
    return _chat(
        groq,
        "You write handoff notes for human support agents. Max 60 words. Plain "
        "text. Cover: what the customer wants, why automation stopped, and a "
        "suggested first step.",
        f"Ticket:\n{message}\n\nAutomated triage:\n{_summary(d)}\n\n"
        f"Why it was escalated:\n{'; '.join(reasons)}",
        max_tokens=600,
    )


# ---------------------------------------------------------------------------
# Optional: the same questions answered by Groq alone, for side-by-side runs.
# ---------------------------------------------------------------------------


@dataclass
class Baseline:
    department: str
    urgency: float
    wants_refund: float
    ms: float


def groq_as_evaluator(groq: Groq, message: str) -> Baseline:
    text, ms = _chat(
        groq,
        "You triage support tickets. Reply with ONLY a JSON object with keys: "
        f"department (one of {', '.join(DEPARTMENTS)}), urgency (number 0-3, where "
        "0=low, 1=medium, 2=high, 3=critical), wants_refund (number 0-1, "
        "probability the customer wants money back).",
        f"Ticket:\n{message}",
        max_tokens=500,
        json_mode=True,
    )
    data = json.loads(text)
    return Baseline(
        department=str(data["department"]),
        urgency=float(data["urgency"]),
        wants_refund=float(data["wants_refund"]),
        ms=ms,
    )


# ---------------------------------------------------------------------------
# One ticket, end to end.
# ---------------------------------------------------------------------------


@dataclass
class TicketResult:
    ticket: Ticket
    decision: Decision
    lane: str
    reasons: list[str]
    text: str | None = None  # reply or handoff note; None when Groq wasn't called
    text_ms: float = 0.0
    baseline: Baseline | None = field(default=None)


def run_ticket(
    jev: TypeSafeClient, groq: Groq, ticket: Ticket, *, compare: bool = False
) -> TicketResult:
    decision = decide(jev, ticket.message)
    lane, reasons = route(decision)
    result = TicketResult(ticket, decision, lane, reasons)

    if lane == AUTO_REPLY:
        result.text, result.text_ms = draft_reply(groq, ticket.message, decision)
    elif lane == HUMAN_REVIEW:
        result.text, result.text_ms = write_handoff(groq, ticket.message, decision, reasons)

    if compare:
        result.baseline = groq_as_evaluator(groq, ticket.message)
    return result
