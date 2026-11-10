"""End-to-end test over the harness's own fixture.

This runs without the credit-union snapshots, so it works from a fresh clone and
checks the whole pipeline: ingest two snapshots, answer a current question, an
"as of" question and a question the site cannot answer, and report the one real
contradiction with the right winner.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "harness"))

from run_eval import load_snapshots                    # noqa: E402
from scoring import score_conflicts                    # noqa: E402

from knowledge_now import KnowledgeNow                  # noqa: E402

FIXTURE = ROOT / "harness" / "selftest"
FAIL: list[str] = []


def want_in(label: str, needle: str, haystack: str) -> None:
    if needle.lower() not in haystack.lower():
        FAIL.append(f"{label}: {needle!r} missing from {haystack!r}")


def want_not_in(label: str, needle: str, haystack: str) -> None:
    if needle.lower() in haystack.lower():
        FAIL.append(f"{label}: {needle!r} should not appear in {haystack!r}")


def main() -> int:
    ks = KnowledgeNow()
    for snap in load_snapshots(FIXTURE / "snapshots"):
        ks.ingest(snap)

    current = ks.answer("What is the 6-month certificate APY?")
    want_in("current value", "3.50%", current.text)
    want_not_in("stale value suppressed", "2.95%", current.text)
    if not current.sources:
        FAIL.append("current value: no source cited")

    # The same question in different words must land on the same number.
    reworded = ks.answer("Current APY on a six month share certificate?")
    want_in("paraphrase", "3.50%", reworded.text)

    historic = ks.answer("What was the 6-month certificate APY in January 2026?",
                         as_of="2026-01-31")
    want_in("as_of value", "2.95%", historic.text)
    want_not_in("as_of not contaminated by today", "3.50%", historic.text)

    fee = ks.answer("How much is a lost debit card replacement?")
    want_in("fee", "$2.50", fee.text)

    unknown = ks.answer("What is the 60-month jumbo certificate rate?")
    if unknown.confident:
        FAIL.append("unpublished rate: should decline, said: " + unknown.text)

    # Asking twice must give the same answer.
    if ks.answer("What is the 6-month certificate APY?").text != current.text:
        FAIL.append("repeat: answer changed between identical calls")

    gold = [{"id": "c1", "topic_keywords": ["6-month", "certificate"],
             "values": ["2.95%", "3.50%"], "correct_value": "3.50%"}]
    report = score_conflicts(gold, ks.conflicts())
    if report["f1"] < 1.0:
        FAIL.append(f"conflicts: f1={report['f1']} ({report})")
    if report["resolution_accuracy"] != 1.0:
        FAIL.append(f"conflicts: picked the wrong winner ({report})")

    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print("  -", f)
        return 1
    print("ok - end-to-end fixture passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
