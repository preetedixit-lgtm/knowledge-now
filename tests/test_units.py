"""Unit tests for the parts that have to be exactly right.

These lock down the pieces where a silent regression would be invisible in the
aggregate score: number and tier parsing, the editorial ranking rules, and the
qualifier logic that decides whether two numbers are even about the same thing.

Run with:  python -m pytest tests  (or: python tests/test_units.py)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))

from knowledge_now import quals, store, textutil                      # noqa: E402
from knowledge_now.docmodel import classify, parse_page               # noqa: E402
from knowledge_now.extract import fee_schedule_facts, table_facts     # noqa: E402
from knowledge_now.facts import Obs, Provenance                       # noqa: E402

FAIL: list[str] = []


def check(label: str, got, want) -> None:
    if got != want:
        FAIL.append(f"{label}: got {got!r}, want {want!r}")


# --------------------------------------------------------------------------
# numbers and tiers
# --------------------------------------------------------------------------

def test_money_intervals() -> None:
    check("range", quals.parse_money_interval("$10,000 - $24,999.99"), (10000.0, 24999.99))
    check("open top", quals.parse_money_interval("$25,000+"), (25000.0, quals.INF))
    check("or less", quals.parse_money_interval("$832,750 or less"), (0.0, 832750.0))
    check("between", quals.parse_money_interval("between $10,000 and $24,999"), (10000.0, 24999.0))
    check("no interval", quals.parse_money_interval("Share Savings"), None)


def test_terms() -> None:
    check("months", quals.parse_term_months("60 Months"), 60)
    check("hyphen", quals.parse_term_months("17-Month Fixed"), 17)
    check("years", quals.parse_term_months("30-Year Fixed Conventional"), 360)
    check("words", quals.parse_term_months("six month certificate"), 6)
    check("arm", quals.parse_arm("7/1 adjustable-rate mortgage"), "7/1")
    check("not arm", quals.parse_arm("30-Year Fixed"), None)


def test_model_years() -> None:
    check("short range", quals.parse_model_years("2024-27 Model Year"), (2024.0, 2027.0))
    check("and older", quals.parse_model_years("2016 Model Year (and older)"), (0.0, 2016.0))
    check("not a model year", quals.parse_model_years("2024 annual report"), None)


def test_dates_and_stems() -> None:
    check("us date", textutil.parse_date("effective as of 09/25/2026"), date(2026, 9, 25))
    check("long date", textutil.parse_date("June 18, 2024"), date(2024, 6, 18))
    check("month only", textutil.parse_month_year("as of May 2026"), date(2026, 5, 1))
    # Morphology has to join these or "how many branches" never finds
    # "branch office locations".
    check("stem branches", textutil.stem("branches"), textutil.stem("branch"))
    check("stem recognition", textutil.stem("recognition"), textutil.stem("recognized"))
    check("stem eligible", textutil.stem("eligibility"), textutil.stem("eligible"))


# --------------------------------------------------------------------------
# page classification and extraction
# --------------------------------------------------------------------------

def test_classification() -> None:
    check("rate sheet", classify("https://www.dupaco.com/rates", "", ""), "rate_sheet")
    check("product", classify("https://www.dupaco.com/personal/loans/auto-motorcycle-loans", "", ""),
          "product")
    check("calculator", classify("https://www.dupaco.com/calculators/vehicle-affordability", "", ""),
          "calculator")
    check("news", classify("https://www.dupaco.com/newsroom/anything", "", ""), "news")
    check("landing", classify("https://www.dupaco.com/switch-and-earn", "", ""), "landing")


RATE_TABLE = """# Auto Loans

Auto & Motorcycle Loans

| Loan Program | Term | Annual Percentage Rate (As Low As) |
| --- | --- | --- |
| 2024-27 Model Year | 60 Months | 5.39% APR1 |
| 2021-23 Model Year | 60 Months | 5.64% APR1 |

Rates are effective as of 08/31/2026. All loans are subject to approval.
"""


def _doc(markdown: str, url: str = "https://www.dupaco.com/personal/loans/auto-motorcycle-loans"):
    return parse_page(url, "Auto Loans | Dupaco", markdown, "05_2026-09-28_latest",
                      date(2026, 9, 28), date(2026, 9, 28), 4)


def test_rate_table_extraction() -> None:
    doc = _doc(RATE_TABLE)
    facts = [f for b in doc.blocks if b.kind == "table" for f in table_facts(b, doc)]
    check("fact count", len(facts), 2)
    first = facts[0]
    check("value", first.value_text, "5.39%")
    check("metric", first.metric, "apr")
    check("family", first.family, "auto_loan")
    check("term", first.term_months, 60)
    check("model years", first.model_years, (2024.0, 2027.0))
    # The table dates itself; that date is what makes it beatable or not.
    check("effective", first.prov.effective_on, date(2026, 8, 31))


FEE_SCHEDULE = """Rate and Fee Schedule
Effective February 17, 2025

Debit Card Service Charges

Rush fee for debit card $50.00*

Stop payment (check, ACH, & bill pay) $30.00* per item
"""


def test_fee_schedule_extraction() -> None:
    doc = _doc(FEE_SCHEDULE, "https://www.dupaco.com/hubfs/dupaco-fee-schedule.pdf")
    facts = fee_schedule_facts(doc, True)
    by_label = {f.label: f for f in facts}
    check("rush fee", by_label["Rush fee for debit card"].value_text, "$50")
    check("stop payment", by_label["Stop payment (check, ACH, & bill pay)"].value_text,
          "$30 per item")
    check("fee date", by_label["Rush fee for debit card"].prov.effective_on, date(2025, 2, 17))


# --------------------------------------------------------------------------
# the resolution rules
# --------------------------------------------------------------------------

def _obs(value: float, *, kind: str, effective, metric: str = "apr", term: int | None = 60,
         family: str = "auto_loan", name: str = "Auto Loans", model_years=None,
         balance=None, flags=frozenset(), url: str = "https://www.dupaco.com/x") -> Obs:
    return Obs(
        domain="rate", metric=metric, family=family, label=name,
        match_text=name.lower(), value_text=f"{value:.2f}%", lo=value, unit="%",
        term_months=term, model_years=model_years, balance=balance, flags=flags,
        prov=Provenance(url=url, title=name, page_kind=kind,
                        authority={"product": 80, "landing": 40, "calculator": 8,
                                   "rate_sheet": 100}.get(kind, 25),
                        snapshot_id="05", snapshot_idx=4, taken_on=date(2026, 9, 28),
                        fetched_at=date(2026, 9, 28), effective_on=effective),
        extras={"product_name": name, "table_header": "h"},
    )


def _store(*observations: Obs) -> store.Store:
    """A store holding the five snapshots, with every observation live in the
    newest one - the state `conflicts()` actually reasons over."""
    s = store.Store()
    for i in range(5):
        s.add_snapshot(f"0{i + 1}", date(2026, 5 + i, 1))
    for o in observations:
        s.add(o)
        s.note_page(o.prov.url, 4)
    s.finalize()
    return s


def test_validity_date_beats_arrival() -> None:
    """The newer rate generation wins even though both pages were crawled today."""
    s = store.Store()
    current = _obs(5.39, kind="product", effective=date(2026, 8, 31))
    stale = _obs(5.34, kind="product", effective=date(2026, 1, 1))
    check("newer effective date wins", s.best([stale, current]).value_text, "5.39%")


def test_maintained_page_beats_calculator() -> None:
    """An un-maintained calculator re-crawled today is not evidence about today."""
    s = store.Store()
    sheet = _obs(5.39, kind="product", effective=date(2026, 1, 1))
    calc = _obs(5.34, kind="calculator", effective=None)
    check("calculator loses", s.best([sheet, calc]).value_text, "5.39%")


def test_expired_promotion_loses() -> None:
    s = store.Store()
    standing = _obs(3.50, kind="product", effective=date(2026, 1, 1), metric="apy")
    promo = _obs(2.95, kind="product", effective=date(2026, 6, 1), metric="apy")
    promo.extras["expired_on"] = date(2025, 9, 30)
    check("expired promo loses", s.best([standing, promo]).value_text, "3.50%")


def test_tiers_are_not_conflicts() -> None:
    """Different tiers of one product are different facts, not disagreements."""
    s = _store(
        _obs(0.20, kind="product", effective=date(2026, 6, 15), metric="apy", term=None,
             family="savings", name="Share Savings", balance=(25000.0, quals.INF)),
        _obs(0.15, kind="product", effective=date(2026, 6, 15), metric="apy", term=None,
             family="savings", name="Share Savings", balance=(10000.0, 24999.99)),
    )
    check("separate tiers", len(s.conflict_groups(4)), 0)


def test_sibling_products_are_not_conflicts() -> None:
    """Bright Savings and Share Savings pay different rates by design."""
    s = _store(
        _obs(0.20, kind="product", effective=date(2026, 6, 15), metric="apy", term=None,
             family="savings", name="Share Savings", balance=(25000.0, quals.INF)),
        _obs(2.25, kind="product", effective=date(2026, 6, 2), metric="apy", term=None,
             family="savings", name="Bright Savings", balance=(0.01, 1000000.0),
             url="https://www.dupaco.com/y"),
    )
    check("distinct products", len(s.conflict_groups(4)), 0)


def test_metrics_on_different_axes_are_not_conflicts() -> None:
    """A balance-tiered row and a model-year row are indexed differently."""
    s = _store(
        _obs(7.89, kind="product", effective=date(2026, 5, 1), balance=None,
             model_years=None, name="RV Loans", family="recreational_loan"),
        _obs(8.34, kind="product", effective=date(2026, 5, 1),
             model_years=(2026.0, 2026.0), name="RV Loans", family="recreational_loan",
             url="https://www.dupaco.com/y"),
    )
    check("different tier axes", len(s.conflict_groups(4)), 0)


def test_real_disagreement_is_reported() -> None:
    s = _store(
        _obs(5.39, kind="product", effective=date(2026, 8, 31),
             model_years=(2024.0, 2027.0)),
        _obs(5.34, kind="landing", effective=date(2026, 1, 1),
             model_years=(2023.0, 2026.0), url="https://www.dupaco.com/switch-and-earn"),
    )
    groups = s.conflict_groups(4)
    check("one conflict", len(groups), 1)
    check("resolved to current", s.best(groups[0]).value_text, "5.39%")


def test_rate_and_apy_are_not_a_conflict() -> None:
    """2.23% dividend rate and 2.25% APY describe one promise two ways."""
    s = _store(
        _obs(2.23, kind="product", effective=date(2026, 6, 2), metric="dividend_rate",
             term=None, family="savings", name="Bright Savings"),
        _obs(2.25, kind="product", effective=date(2026, 6, 2), metric="apy",
             term=None, family="savings", name="Bright Savings"),
    )
    check("metrics differ, no conflict", len(s.conflict_groups(4)), 0)



def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    if FAIL:
        print(f"FAILED ({len(FAIL)}):")
        for f in FAIL:
            print("  -", f)
        return 1
    print(f"ok - {len(tests)} test functions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
