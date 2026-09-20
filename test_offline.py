"""Offline test. No API keys, no network.

Both services are faked at the HTTP layer, so the real TypeSafe and Groq SDKs
run for real: the TypeSafe SDK validates the fake responses against its own
response models, and the Groq SDK builds real requests that we inspect.

IMPORTANT: the Jev answers below are FABRICATED fixtures shaped like the API's
documented output. They test my routing rules and request/response handling.
They say nothing about how Jev actually performs on these tickets.
"""
import json
import os
import sys

import httpx
import httpx2

os.environ["TYPESAFE_API_KEY"] = "test-typesafe"
os.environ["GROQ_API_KEY"] = "test-groq"
os.environ["GROQ_MODEL"] = "openai/gpt-oss-20b"

from pipeline import (  # noqa: E402
    ARCHIVE, AUTO_REPLY, HUMAN_REVIEW, make_groq_client, make_jev_client, run_ticket,
)
from tickets import TICKETS  # noqa: E402

DEPTS = ["billing", "technical", "account", "sales", "other"]


def dist(**kw):
    return {d: kw.get(d, 0.0) for d in DEPTS}


# needs_reply, dept distribution, dept confidence, urgency level (0-3),
# urgency confidence, refund, sensitive, expected lane
FIXTURES = {
    "T-101": (0.98, dist(billing=0.96, technical=0.01, account=0.01, sales=0.01, other=0.01), 0.95, 0, 0.85, 0.97, 0.03, AUTO_REPLY),
    "T-102": (0.99, dist(billing=0.01, technical=0.97, account=0.01, sales=0.005, other=0.005), 0.96, 3, 0.85, 0.01, 0.10, HUMAN_REVIEW),
    "T-103": (0.97, dist(billing=0.01, technical=0.02, account=0.94, sales=0.01, other=0.02), 0.92, 0, 0.85, 0.01, 0.02, AUTO_REPLY),
    "T-104": (0.99, dist(billing=0.55, technical=0.35, account=0.05, sales=0.02, other=0.03), 0.40, 1, 0.70, 0.95, 0.93, HUMAN_REVIEW),
    "T-105": (0.08, dist(billing=0.02, technical=0.03, account=0.02, sales=0.01, other=0.92), 0.90, 0, 0.90, 0.01, 0.01, ARCHIVE),
    "T-106": (0.99, dist(billing=0.01, technical=0.20, account=0.75, sales=0.01, other=0.03), 0.70, 3, 0.85, 0.01, 0.96, HUMAN_REVIEW),
    "T-107": (0.98, dist(billing=0.01, technical=0.42, account=0.03, sales=0.50, other=0.04), 0.35, 1, 0.75, 0.02, 0.05, HUMAN_REVIEW),
    "T-108": (0.02, dist(billing=0.01, technical=0.01, account=0.01, sales=0.05, other=0.92), 0.90, 0, 0.90, 0.01, 0.02, ARCHIVE),
}
BY_MESSAGE = {t.message: (t, FIXTURES[t.id]) for t in TICKETS}

jev_requests: list[dict] = []
groq_requests: list[dict] = []


# --- fake TypeSafe API ------------------------------------------------------
def typesafe_handler(request: httpx2.Request) -> httpx2.Response:
    assert request.url.path.endswith("/v1/systemone"), request.url.path
    assert request.headers["authorization"] == "Bearer test-typesafe"
    body = json.loads(request.content)
    jev_requests.append(body)
    _, (nr, dept, dconf, lvl, uconf, refund, sens, _) = BY_MESSAGE[body["state"]["message"]]

    # Point mass of 0.85 on the chosen level, 0.05 elsewhere; score = weighted mean.
    probs = {str(i): (0.85 if i == lvl else 0.05) for i in range(4)}
    score = sum(int(k) * v for k, v in probs.items())
    legend = {str(i): f"level {i}" for i in range(4)}

    return httpx2.Response(200, json={
        "model": "jev-latest",
        "usage": {"input_tokens": 280, "output_tokens": 20},
        "answers": {
            "needs_reply": {"type": "noul", "noul": nr},
            "department": {
                "type": "choice",
                "choice": max(dept, key=dept.get),
                "confidence": dconf,
                "probabilities": dept,
            },
            "urgency": {
                "type": "score", "score": score, "confidence": uconf,
                "legend": legend, "probabilities": probs,
            },
            "wants_refund": {"type": "noul", "noul": refund},
            "sensitive": {"type": "noul", "noul": sens},
        },
    })


# --- fake Groq API ----------------------------------------------------------
def groq_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    groq_requests.append(body)
    if body.get("response_format", {}).get("type") == "json_object":
        content = json.dumps({"department": "billing", "urgency": 1, "wants_refund": 0.9})
    else:
        content = "FAKE GROQ TEXT"
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


jev, jev_model = make_jev_client(transport=httpx2.MockTransport(typesafe_handler))
groq = make_groq_client(http_client=httpx.Client(transport=httpx.MockTransport(groq_handler)))

failures = 0
print(f"Jev client model: {jev_model}\n")

for t in TICKETS:
    r = run_ticket(jev, groq, t)
    expected = FIXTURES[t.id][-1]
    ok = r.lane == expected
    failures += not ok
    called = "Groq called" if r.text else "no LLM   "
    print(f"{'✓' if ok else '✗'} {t.id}  {r.lane:<12} {called}  {'; '.join(r.reasons)}"
          + ("" if ok else f"   (expected {expected})"))

# Cost/latency claim in the README: Groq only runs for non-archived tickets.
expected_calls = sum(1 for f in FIXTURES.values() if f[-1] != ARCHIVE)
ok = len(groq_requests) == expected_calls
failures += not ok
print(f"\n{'✓' if ok else '✗'} Groq called {len(groq_requests)}x (expected {expected_calls}); archive lane skipped it")

# Request shape checks on the real Groq SDK output.
ok = all(r["model"] == "openai/gpt-oss-20b" for r in groq_requests)
failures += not ok
print(f"{'✓' if ok else '✗'} every Groq call used the configured model")

ok = all(r.get("reasoning_effort") == "low" and r.get("include_reasoning") is False for r in groq_requests)
failures += not ok
print(f"{'✓' if ok else '✗'} gpt-oss calls send reasoning_effort=low, include_reasoning=false")

ok = all(len(r["questions"]) == 5 for r in jev_requests) and len(jev_requests) == len(TICKETS)
failures += not ok
print(f"{'✓' if ok else '✗'} one Jev call per ticket, five questions each")

# --compare path: Groq JSON mode parsing.
rc = run_ticket(jev, groq, TICKETS[0], compare=True)
ok = rc.baseline is not None and rc.baseline.department == "billing" and rc.baseline.urgency == 1.0
failures += not ok
print(f"{'✓' if ok else '✗'} --compare parses Groq JSON-mode output")
ok = groq_requests[-1].get("response_format") == {"type": "json_object"}
failures += not ok
print(f"{'✓' if ok else '✗'} --compare requests JSON mode")

print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
