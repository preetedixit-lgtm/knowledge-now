"""The unit of knowledge: one observation of one fact, with full provenance.

Nothing is ever overwritten. Ingesting a snapshot appends observations, so the
store keeps the whole history and `answer(as_of=...)` is a filter over it rather
than a different code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .ontology import FAMILY_LABEL
from .textutil import norm


@dataclass(frozen=True)
class Provenance:
    url: str
    title: str
    page_kind: str
    authority: int
    snapshot_id: str
    snapshot_idx: int
    taken_on: date
    fetched_at: Optional[date]
    effective_on: Optional[date]
    quote: str = ""

    @property
    def asserted_on(self) -> date:
        """The best evidence of when this value became true.

        A rate table that says "effective as of 09/25/2026" is self-dating and
        that beats everything else: it is the publisher telling us which
        generation of the rate sheet this is. Failing that we fall back to when
        the crawler last saw the page, and finally to the snapshot date.
        """
        return self.effective_on or self.fetched_at or self.taken_on

    @property
    def self_dated(self) -> bool:
        return self.effective_on is not None

    @property
    def path_words(self) -> str:
        path = self.url.split("dupaco.com", 1)[-1] if "dupaco.com" in self.url else self.url
        return path.replace("/", " ").replace("-", " ").replace("_", " ").lower()


@dataclass
class Obs:
    """One (fact, value, source) triple."""
    domain: str                      # rate | fee | hours | branch | entity | link | text
    metric: str                      # apy | apr | rate | fee | value | ...
    family: Optional[str]            # product family from the ontology
    label: str                       # human-readable identity, e.g. "Share Savings - $25,000+"
    match_text: str                  # everything a question may be matched against
    value_text: str                  # what we will print, e.g. "5.39%"
    lo: Optional[float] = None       # numeric value (low end of a range)
    hi: Optional[float] = None       # high end when the sheet prints a range
    unit: str = ""                   # "%" | "$" | ""
    term_months: Optional[int] = None
    balance: Optional[tuple[float, float]] = None
    amount: Optional[tuple[float, float]] = None
    model_years: Optional[tuple[float, float]] = None
    arm: Optional[str] = None
    flags: frozenset[str] = frozenset()
    prov: Optional[Provenance] = None
    extras: dict = field(default_factory=dict)

    # -- identity ---------------------------------------------------------
    @property
    def fact_key(self) -> tuple:
        """Two observations with the same fact_key are claims about the same
        thing, so if their values differ the website contradicts itself."""
        return (
            self.domain, self.metric, self.family or self.label_core,
            self.term_months, self.arm,
            _round_iv(self.balance), _round_iv(self.amount),
            tuple(sorted(self.flags - {"purchase", "refinance"})),
        )

    @property
    def label_core(self) -> str:
        return norm(self.label)[:60]

    @property
    def value_num(self) -> Optional[float]:
        return self.lo

    @property
    def is_range(self) -> bool:
        return self.hi is not None and self.lo is not None and self.hi != self.lo

    def pretty_subject(self) -> str:
        if self.family and self.family in FAMILY_LABEL:
            return FAMILY_LABEL[self.family]
        return self.label


def _round_iv(iv):
    return None if iv is None else (round(iv[0], 2), round(iv[1], 2))
