"""Qualifier recognisers.

A number on a rate sheet only means something together with its qualifiers: the
term, the balance tier, the loan-amount tier, the vehicle model year. The same
recognisers run over table rows at ingest time *and* over the member's question
at answer time, so "between $10,000 and $24,999" in a question and
"$10,000 - $24,999.99" in a table reduce to the same interval.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from .textutil import WORD_NUM, norm, to_float

INF = math.inf

_NUM = r"\$?\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?"

_TERM_MONTHS = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(?:mo\b|mos\b|month)", re.I)
_TERM_YEARS = re.compile(r"\b(\d{1,2})\s*[- ]?\s*(?:yr\b|yrs\b|year)", re.I)
_TERM_WORD = re.compile(r"\b(" + "|".join(WORD_NUM) + r")[\s-]*(month|year)", re.I)
_ARM = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})\b")


def parse_term_months(text: str) -> Optional[int]:
    """'60 Months' / '17-Month' / '30-Year' / 'six month' -> months."""
    t = norm(text)
    m = _TERM_MONTHS.search(t)
    if m:
        return int(m.group(1))
    m = _TERM_YEARS.search(t)
    if m:
        return int(m.group(1)) * 12
    m = _TERM_WORD.search(t)
    if m:
        n = WORD_NUM.get(m.group(1).lower())
        if n:
            return n * (12 if m.group(2).startswith("year") else 1)
    return None


def parse_arm(text: str) -> Optional[str]:
    """'7/1 ARM' -> '7/1'. ARM products are a different animal from fixed rates."""
    t = norm(text)
    if "arm" in t.split() or "adjustable" in t:
        m = _ARM.search(t)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
        return "arm"
    m = _ARM.search(t)
    if m and int(m.group(1)) in (3, 5, 7, 10) and int(m.group(2)) in (1, 6):
        return f"{m.group(1)}/{m.group(2)}"
    return None


# --------------------------------------------------------------------------
# money intervals
# --------------------------------------------------------------------------

_RANGE = re.compile(_NUM + r"\s*(?:-|to|through|up to)\s*" + _NUM, re.I)
_BETWEEN = re.compile(r"between\s*" + _NUM + r"\s*(?:-|to|and)\s*" + _NUM, re.I)
_PLUS = re.compile(_NUM + r"\s*(?:\+|or more|and (?:up|over|above)|or (?:greater|higher|above))", re.I)
_OR_LESS = re.compile(_NUM + r"\s*(?:or less|or below|and (?:under|below)|or under)", re.I)
_UNDER = re.compile(r"(?:under|below|less than)\s*" + _NUM, re.I)
_OVER = re.compile(r"(?:over|above|more than|at least|minimum of)\s*" + _NUM, re.I)


def parse_money_interval(text: str) -> Optional[tuple[float, float]]:
    """'$10,000 - $24,999.99' -> (10000.0, 24999.99);  '$25,000+' -> (25000, inf).

    Returns None when the text carries no balance/amount interval at all.
    """
    t = norm(text)
    if "$" not in t and not re.search(r"\d{3,}", t):
        return None
    m = _BETWEEN.search(t) or _RANGE.search(t)
    if m:
        lo = to_float(m.group(1), m.group(2))
        hi = to_float(m.group(3), m.group(4))
        if hi >= lo:
            return (lo, hi)
    m = _PLUS.search(t)
    if m:
        return (to_float(m.group(1), m.group(2)), INF)
    m = _OR_LESS.search(t)
    if m:
        return (0.0, to_float(m.group(1), m.group(2)))
    m = _UNDER.search(t)
    if m:
        return (0.0, to_float(m.group(1), m.group(2)))
    m = _OVER.search(t)
    if m:
        return (to_float(m.group(1), m.group(2)), INF)
    return None


def parse_single_amount(text: str) -> Optional[float]:
    """A bare amount a member states about themselves: 'I have $12,000 saved'."""
    m = re.search(r"\$\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?", text)
    if m:
        return to_float(m.group(1), m.group(2))
    m = re.search(r"\b(\d{1,3}(?:,\d{3})+)(?:\.(\d+))?\b", text)
    return to_float(m.group(1), m.group(2)) if m else None


def interval_contains(iv: tuple[float, float], v: float) -> bool:
    return iv[0] - 1e-9 <= v <= iv[1] + 1e-9


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] + 1e-9 and b[0] <= a[1] + 1e-9


# --------------------------------------------------------------------------
# vehicle model years
# --------------------------------------------------------------------------

_MY_RANGE = re.compile(r"\b(19|20)(\d{2})\s*[-]\s*(?:(19|20))?(\d{2})\b")
_MY_ONE = re.compile(r"\b((?:19|20)\d{2})\b")


def parse_model_years(text: str) -> Optional[tuple[float, float]]:
    """'2024-27 Model Year' -> (2024, 2027);  '2016 Model Year (and older)' -> (0, 2016)."""
    t = norm(text)
    if "model year" not in t and "model-year" not in t:
        return None
    m = _MY_RANGE.search(t)
    if m:
        lo = int(m.group(1) + m.group(2))
        hi_c = m.group(3) or m.group(1)
        hi = int(hi_c + m.group(4))
        if hi < lo:                      # '2024-27' -> century carried from lo
            hi = int(str(lo)[:2] + m.group(4))
        if hi >= lo:
            return (float(lo), float(hi))
    m = _MY_ONE.search(t)
    if m:
        y = float(m.group(1))
        if re.search(r"and older|or older|and under", t):
            return (0.0, y)
        if re.search(r"and newer|or newer|\+", t):
            return (y, INF)
        return (y, y)
    return None


def question_model_year(text: str) -> Optional[float]:
    """A model year the member names: '2025 model-year vehicle', 'a 2024 car'."""
    t = norm(text)
    for m in _MY_ONE.finditer(t):
        y = int(m.group(1))
        if 1980 <= y <= 2035:
            return float(y)
    return None


# --------------------------------------------------------------------------
# transaction / variant flags found in a row label
# --------------------------------------------------------------------------

def flags(text: str) -> set[str]:
    t = " " + norm(text) + " "
    out = set()
    if "jumbo" in t:
        out.add("jumbo")
    if "business" in t or "commercial" in t:
        out.add("business")
    if re.search(r"\bira\b|\broth\b|\btraditional\b", t):
        out.add("ira")
    if "special" in t:
        out.add("special")
    if "refinance" in t or "refi " in t:
        out.add("refinance")
    if "purchase" in t:
        out.add("purchase")
    if "introductory" in t or "intro " in t:
        out.add("intro")
    if "mini max" in t or "mini-max" in t:
        out.add("mini_max")
    if re.search(r"share secured|savings secured|secured by", t):
        out.add("secured")
    if "student" in t:
        out.add("student")
    return out
