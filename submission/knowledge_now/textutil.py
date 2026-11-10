"""Text normalisation shared by the extractors and the question parser.

Everything here is deterministic and dependency-free: the same input always
produces the same output, which is what makes the answer path reproducible
(the scorer asks a sample of questions twice and compares).
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Optional

# --------------------------------------------------------------------------
# character level
# --------------------------------------------------------------------------

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"',
           ord(" "): " ", ord(" "): " ", ord(" "): " "}


def clean(s: str) -> str:
    """Unicode dashes/quotes -> ASCII, collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_DASHES).translate(_QUOTES)
    s = s.replace("\\r\\n", " ").replace("\\n", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def norm(s: str) -> str:
    """Lower-cased comparison form. Mirrors the scorer's own normalisation so
    that what we emit matches the way it will be checked."""
    s = clean(s).lower()
    s = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", ",", s)        # keep thousands separators intact
    s = re.sub(r"[^a-z0-9%$.,:/+\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------

PCT_RE = re.compile(r"(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*%")
MONEY_RE = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?")
ANY_NUM_RE = re.compile(r"(\$)?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%)?")


def to_float(whole: str, frac: Optional[str]) -> float:
    return float(whole.replace(",", "") + ("." + frac if frac else ""))


def first_percent(s: str) -> Optional[float]:
    m = PCT_RE.search(s or "")
    return to_float(m.group(1), m.group(2)) if m else None


def first_money(s: str) -> Optional[float]:
    m = MONEY_RE.search(s or "")
    return to_float(m.group(1), m.group(2)) if m else None


def money_str(v: float) -> str:
    return f"${v:,.2f}" if v % 1 else f"${v:,.0f}"


def pct_str(v: float) -> str:
    """Render a rate the way rate sheets do: two decimals."""
    return f"{v:.2f}%"


WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "twentyfour": 24, "thirty": 30,
    "thirtysix": 36, "forty": 40, "fortyeight": 48, "sixty": 60, "seventytwo": 72,
    "eightyfour": 84,
}


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LONG = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_MONTH_YEAR = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b")


def parse_date(s: str) -> Optional[date]:
    """Parse the first date in `s`. Handles 09/25/2026, 2026-09-25, Sep 25, 2026."""
    if not s:
        return None
    m = _ISO.search(s)
    if m:
        return _safe(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _SLASH.search(s)
    if m:
        y = int(m.group(3))
        y += 2000 if y < 100 else 0
        return _safe(y, int(m.group(1)), int(m.group(2)))
    m = _LONG.search(s)
    if m:
        mo = _MONTHS.get(m.group(1)[:3].lower())
        if mo:
            return _safe(int(m.group(3)), mo, int(m.group(2)))
    return None


def parse_month_year(s: str) -> Optional[date]:
    """'May 2026' -> 2026-05-01. Used for 'as of May 2026' style questions."""
    m = _MONTH_YEAR.search(s or "")
    if m:
        mo = _MONTHS.get(m.group(1)[:3].lower())
        if mo:
            return _safe(int(m.group(2)), mo, 1)
    return None


def _safe(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------

STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from get give go had has have
how i if in into is it its just like may me my of on or our please so some tell that the
their there these they this to us want was we were what whats when where which who will
with would you your am pm current currently right now today much many does did about
know would like need find looking tell show me more info information question questions
hi hello thanks thank please help offer offers available
""".split())

_TOKEN_RE = re.compile(r"\$?[0-9][0-9,]*(?:\.[0-9]+)?%?|[a-z][a-z0-9]*")


def tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(norm(s))


def content_tokens(s: str) -> list[str]:
    return [t for t in tokens(s) if t not in STOPWORDS and len(t) > 1]


def slug(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", norm(s))).strip("-")


# --------------------------------------------------------------------------
# stemming
# --------------------------------------------------------------------------

_SUFFIXES = ("ational", "izations", "ization", "ations", "ation", "ingly", "edly",
             "ments", "ment", "ness", "ities", "ity", "ings", "ing", "ies", "ied",
             "ers", "er", "es", "ed", "ly", "s")


def stem(word: str) -> str:
    """Conservative stemmer: enough to join recognition/recognized/recognize and
    branch/branches, without a dependency or a dictionary.

    Retrieval over a single website needs morphological recall far more than it
    needs linguistic precision - a member asking about "branches" must reach a
    page that says "branch office locations".
    """
    w = word.lower()
    if len(w) <= 4 or w[0].isdigit() or w.startswith("$"):
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            if suf in ("ies", "ied"):
                w += "y"
            break
    return w[:6]


def stems(text: str) -> list[str]:
    return [stem(t) for t in tokens(text)]
