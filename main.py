import argparse
import sys
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

from pipeline import (  # noqa: E402  (must come after load_dotenv)
    ARCHIVE,
    AUTO_REPLY,
    GROQ_MODEL,
    HUMAN_REVIEW,
    QUESTIONS,
    make_groq_client,
    make_jev_client,
    run_ticket,
)
from tickets import TICKETS  # noqa: E402


def ms(x: float) -> str:
    return f"{round(x)}ms"


def indent(text: str) -> str:
    return "\n".join("     " + line for line in text.splitlines())


def check() -> int:
    """Verify both keys work and the Groq model ID is still live."""
    jev, jev_model = make_jev_client()
    groq = make_groq_client()

    r = jev.system_one(
        state="ping",
        questions={"t": QUESTIONS["wants_refund"]},
    )
    print(f"Jev OK    model={r.model}  (asked for {jev_model})")

    ids = sorted(m.id for m in groq.models.list().data)
    if GROQ_MODEL in ids:
        print(f"Groq OK   model={GROQ_MODEL} is available")
        return 0
    print(f"Groq key works, but {GROQ_MODEL!r} is not in your model list.")
    print("Available:", ", ".join(ids))
    print("Set GROQ_MODEL in .env to one of these.")
    return 1


def run(compare: bool) -> int:
    jev, jev_model = make_jev_client()
    groq = make_groq_client()
    print(f"Jev: {jev_model}   Groq: {GROQ_MODEL}\n")

    jev_ms, text_ms, base_ms = [], [], []
    lanes = {ARCHIVE: 0, AUTO_REPLY: 0, HUMAN_REVIEW: 0}
    llm_calls = agree = compared = 0

    with jev:
        for t in TICKETS:
            snippet = t.message[:70] + ("…" if len(t.message) > 70 else "")
            print(f'── {t.id}  "{snippet}"')
            try:
                r = run_ticket(jev, groq, t, compare=compare)
            except Exception as e:  # keep going so one failure doesn't hide the rest
                print(f"   ✗ failed: {e}\n")
                continue

            d = r.decision
            jev_ms.append(d.ms)
            lanes[r.lane] += 1
            print(
                f"   Jev ({ms(d.ms)}): dept={d.department} (conf {d.department_confidence:.2f})"
                f"  urgency={d.urgency:.1f}/3  refund={d.wants_refund:.2f}"
                f"  sensitive={d.sensitive:.2f}  needs_reply={d.needs_reply:.2f}"
            )
            print(f"   Lane: {r.lane.upper()}  ({'; '.join(r.reasons)})")

            if r.text is not None:
                llm_calls += 1
                text_ms.append(r.text_ms)
                label = "reply" if r.lane == AUTO_REPLY else "handoff"
                print(f"   Groq {label} ({ms(r.text_ms)}):\n{indent(r.text)}")
            else:
                print("   Groq not called.")

            if r.baseline is not None:
                b = r.baseline
                compared += 1
                base_ms.append(b.ms)
                agree += b.department == d.department
                print(
                    f"   Groq-as-evaluator ({ms(b.ms)}): dept={b.department}"
                    f"  urgency={b.urgency:.1f}/3  refund={b.wants_refund:.2f}"
                )
            print()

    if not jev_ms:
        print("Nothing succeeded. Run `python main.py --check` to diagnose.")
        return 1

    print("══ Summary")
    print(
        f"Lanes: {lanes[AUTO_REPLY]} auto-reply, {lanes[HUMAN_REVIEW]} human review, "
        f"{lanes[ARCHIVE]} archived"
    )
    print(
        f"Jev avg {ms(mean(jev_ms))} per ticket (5 decisions each)"
        + (f" vs Groq text avg {ms(mean(text_ms))}" if text_ms else "")
    )
    print(f"Groq called on {llm_calls}/{len(jev_ms)} tickets. The rest needed no generated text.")
    if compared:
        print(
            f"Compare: Groq-as-evaluator avg {ms(mean(base_ms))} per ticket, "
            f"department agreement with Jev {agree}/{compared}"
        )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Jev + Groq hybrid ticket triage")
    p.add_argument("--check", action="store_true", help="verify keys and Groq model ID, then exit")
    p.add_argument("--compare", action="store_true", help="also run the same questions through Groq alone")
    args = p.parse_args()
    sys.exit(check() if args.check else run(args.compare))
