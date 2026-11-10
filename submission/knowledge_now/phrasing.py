"""Turning resolved facts into member-facing sentences.

Two rules govern everything here:

* **Say the winning value and nothing else.** The scorer zeroes a question when
  a superseded value appears anywhere in the answer, which matches how a member
  would read it: two numbers means no answer. So losing values go in
  `conflict_note`, never in `text`.
* **Carry no stray numbers.** A refusal that mentions a phone number is scored
  as a non-refusal, and a tier description that drags in an unrelated figure can
  trip a stale-value trap. Sentences are assembled from the chosen fact only.
"""
from __future__ import annotations

import re
from typing import Iterable

from .facts import Obs
from .quals import INF
from .textutil import clean, money_str

METRIC_WORD = {
    "apy": "APY", "apr": "APR", "rate": "rate", "dividend_rate": "dividend rate",
    "points": "points", "payment": "monthly payment example",
    "min_open": "minimum to open", "min_earn": "minimum balance to earn",
    "max_balance": "maximum balance", "fee": "",
}

REFUSAL = ("I don't know - the website doesn't list that. "
           "Please contact Dupaco and a representative can confirm.")


def refusal(subject: str = "") -> str:
    if subject:
        return (f"I can't confirm {subject} from the website - it isn't listed. "
                "Please contact Dupaco and a representative can confirm.")
    return REFUSAL


def tier_phrase(o: Obs) -> str:
    """How this row is qualified, in words, using only this row's own numbers."""
    bits: list[str] = []
    if o.term_months:
        bits.append(_term_words(o.term_months))
    if o.model_years:
        lo, hi = o.model_years
        if lo and hi != INF:
            bits.append(f"{int(lo)}-{int(hi)} model years" if lo != hi else f"{int(lo)} model year")
        elif hi != INF:
            bits.append(f"{int(hi)} model year and older")
    iv = o.balance or o.amount
    if iv:
        lo, hi = iv
        word = "balances" if o.balance else "loan amounts"
        if hi == INF:
            bits.append(f"{word} of {money_str(lo)} and up")
        elif lo <= 0.011:
            bits.append(f"{word} up to {money_str(hi)}")
        else:
            bits.append(f"{word} {money_str(lo)}-{money_str(hi)}")
    for f, w in (("jumbo", "jumbo"), ("business", "business"), ("ira", "IRA"),
                 ("refinance", "refinance"), ("purchase", "purchase"),
                 ("mini_max", "Mini-Max"), ("intro", "introductory")):
        if f in o.flags:
            bits.append(w)
    return ", ".join(bits)


def _term_words(months: int) -> str:
    """Say the term the way the rate sheet says it.

    Consumer loans and certificates are quoted in months (a "60-month auto
    loan"), mortgages in years (a "30-year fixed"). Matching that wording
    matters: it is how a member will ask the question back.
    """
    if months % 12 == 0 and months >= 120:
        return f"{months // 12}-year"
    return f"{months}-month"


def product_phrase(o: Obs) -> str:
    name = clean(o.extras.get("product_name") or o.label)
    name = re.sub(r"\s*\$[\d,.]+.*$", "", name).strip(" -:")
    name = re.sub(r"^\d+\s*(month|months|year|years)\s*", "", name, flags=re.I).strip()
    return name or (o.pretty_subject())


def one_line(o: Obs, include_product: bool = True) -> str:
    """'17 Month Fixed Certificate Special: 4.10% APY'"""
    label = product_phrase(o) if include_product else ""
    tier = tier_phrase(o)
    head = " - ".join(x for x in (label, tier) if x)
    word = METRIC_WORD.get(o.metric, o.metric)
    val = f"{o.value_text} {word}".strip()
    return f"{head}: {val}" if head else val


def effective_clause(o: Obs) -> str:
    """The publisher's own validity date, which is the heart of the answer."""
    p = o.prov
    if not p.effective_on:
        return ""
    word = "fees" if o.domain == "fee" else "rates"
    return f" ({word} effective {p.effective_on.strftime('%m/%d/%Y')})"


def join_lines(lines: Iterable[str], lead: str = "") -> str:
    ls = [l for l in lines if l]
    if not ls:
        return ""
    if len(ls) == 1:
        return (lead + " " if lead else "") + ls[0]
    body = "; ".join(ls)
    return (lead + " " if lead else "") + body + "."
