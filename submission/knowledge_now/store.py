"""The temporal fact store.

Three ideas do most of the work here:

1. **Visibility.** "The website as it stood on day D" is not "every page we ever
   crawled up to D". For each URL it is that page's newest copy at or before D.
   `visible_at()` precomputes exactly that, which is why a question with
   `as_of` and a question without it run through identical code.

2. **Validity beats arrival.** A rate table that says "effective as of
   09/25/2026" dates itself. A press release that happens to be re-crawled
   today does not become true today. Ranking therefore starts from the
   editorial tier of the page and then uses the fact's own validity date.

3. **Qualifiers are identity, not noise.** A 48-month rate and a 60-month rate
   are different facts, so they never compete. Two sources that claim different
   values for the *same* qualifiers are a contradiction, and that is what
   `conflicts()` reports.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from .facts import Obs
from .ontology import GENERIC, HEAD_WORDS
from .quals import intervals_overlap
from .textutil import stem, tokens

# Editorial tiers. Within a tier the fact's own validity date decides; across
# tiers the more carefully maintained page wins, because a stale calculator
# being re-crawled today does not make its frozen number current.
TIER = {"rate_sheet": 3, "product": 3, "support": 2, "legal": 2, "other": 2,
        "landing": 1, "news": 0, "blog": 0, "calculator": 0}



DEPOSIT_SET = frozenset({"apy", "dividend_rate"})
LOAN_SET = frozenset({"apr", "rate"})


def _tier(o: Obs) -> int:
    return TIER.get(o.prov.page_kind, 2)


# Words that mark a *variant* rather than a product. They are already carried
# as flags, so letting them into the identity makes "Business Platinum Visa" and
# "Business Rewards Visa" look like one product.
_VARIANT_WORDS = frozenset({"business", "commercial", "jumbo", "ira", "roth",
                            "traditional", "special", "personal", "consumer"})


def name_key(o: Obs) -> frozenset[str]:
    """What distinguishes this product from its siblings in the same family."""
    base = HEAD_WORDS.get(o.family or "", frozenset())
    raw = o.extras.get("product_name") or o.label
    return frozenset(t for t in tokens(raw)
                     if t not in GENERIC and t not in base and t not in _VARIANT_WORDS
                     and len(t) > 1 and not t.isdigit() and not t.startswith("$"))


# --------------------------------------------------------------------------
# vocabulary / phrase index
# --------------------------------------------------------------------------

class Vocabulary:
    """What words and product phrases the website actually uses.

    A content word the site never uses anywhere is the strongest possible
    signal that we should decline rather than guess.
    """

    def __init__(self) -> None:
        self.words: set[str] = set()      # words as the site spells them
        self.stems: set[str] = set()      # the same vocabulary, stemmed
        self.phrases: set[str] = set()

    def add_text(self, text: str) -> None:
        self.words.update(tokens(text))

    def add_product(self, label: str) -> None:
        """Record a product name and the multi-word phrases inside it.

        Only phrases whose first and last word carry meaning are stored, so
        "platinum visa" is a phrase the site uses and "the interest" is not.
        """
        ts = [t for t in tokens(label) if len(t) > 1]
        self.words.update(ts)
        for n in (2, 3, 4):
            for i in range(len(ts) - n + 1):
                part = ts[i:i + n]
                if part[0] in GENERIC or part[-1] in GENERIC:
                    continue
                self.phrases.add(" ".join(part))

    def known(self, phrase: str) -> bool:
        return phrase in self.phrases

    def known_word(self, w: str) -> bool:
        """Does the website use this word anywhere?

        Checked against stems as well as spellings, so "highest" counts as known
        when the site says "high". A word that fails this test is a word the
        site has never used, which is the one case where declining is clearly
        better than answering.
        """
        return w in self.words or stem(w) in self.stems


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------

@dataclass
class Snap:
    snapshot_id: str
    taken_on: date
    idx: int


class Store:
    def __init__(self) -> None:
        self.obs: list[Obs] = []
        self.snaps: list[Snap] = []
        self.vocab = Vocabulary()
        self._url_idx: dict[str, list[int]] = defaultdict(list)   # url -> snapshot idxs seen
        self._visible: list[list[int]] | None = None
        self._newest: dict[str, list[int]] = {}
        self._by_domain: dict[str, list[int]] = defaultdict(list)

    # -- ingest -----------------------------------------------------------
    def add_snapshot(self, snapshot_id: str, taken_on: date) -> int:
        idx = len(self.snaps)
        self.snaps.append(Snap(snapshot_id, taken_on, idx))
        return idx

    def add(self, o: Obs) -> None:
        self.obs.append(o)
        self._by_domain[o.domain].append(len(self.obs) - 1)
        self._visible = None

    def note_page(self, url: str, snapshot_idx: int) -> None:
        self._url_idx[url].append(snapshot_idx)

    def finalize(self) -> None:
        """Precompute, for each snapshot, which observations were live then."""
        n = len(self.snaps)
        newest: dict[str, list[int]] = {}
        for url, idxs in self._url_idx.items():
            seen = sorted(set(idxs))
            row = []
            cur = -1
            for s in range(n):
                if s in seen:
                    cur = s
                row.append(cur)
            newest[url] = row
        self._newest = newest
        self._visible = []
        for s in range(n):
            live = []
            for i, o in enumerate(self.obs):
                if o.prov.snapshot_idx == newest.get(o.prov.url, [-1] * n)[s]:
                    live.append(i)
            self._visible.append(live)

    def newest_idx(self, url: str, snap_idx: int) -> int:
        """Which crawl of `url` was the live one on the day of `snap_idx`."""
        if self._visible is None:
            self.finalize()
        row = self._newest.get(url)
        if row is None:
            return -1
        return row[max(0, min(snap_idx, len(row) - 1))]

    # -- visibility -------------------------------------------------------
    def snapshot_index_for(self, as_of: Optional[date]) -> int:
        if as_of is None:
            return len(self.snaps) - 1
        best = -1
        for s in self.snaps:
            if s.taken_on <= as_of:
                best = s.idx
        return best if best >= 0 else 0

    def visible_at(self, snap_idx: int) -> list[int]:
        if self._visible is None:
            self.finalize()
        return self._visible[max(0, min(snap_idx, len(self._visible) - 1))]

    # -- ranking ----------------------------------------------------------
    def rank_key(self, o: Obs) -> tuple:
        p = o.prov
        return (0 if o.extras.get("expired_on") else 1,
                _tier(o), 1 if p.self_dated else 0, p.asserted_on.toordinal(),
                p.authority, p.snapshot_idx)

    def best(self, cands: Iterable[Obs]) -> Optional[Obs]:
        cands = list(cands)
        return max(cands, key=self.rank_key) if cands else None

    # -- conflicts --------------------------------------------------------
    def conflict_groups(self, snap_idx: int) -> list[list[Obs]]:
        """Observations that make incompatible claims about the same fact."""
        live = [o for o in (self.obs[i] for i in self.visible_at(snap_idx))
                if o.domain in ("rate", "fee") and o.lo is not None and _claimable(o)]
        buckets: dict[tuple, list[Obs]] = defaultdict(list)
        for o in live:
            if o.metric in ("points", "payment", "min_open", "min_earn", "max_balance"):
                continue
            # "special" is excluded on purpose: a special-offer rate and the
            # standing rate for the same product and term are exactly the
            # contradiction a member runs into when a promotion is never taken
            # down.
            core = frozenset(o.flags & {"jumbo", "business", "ira", "intro", "mini_max",
                                        "secured", "student"})
            # The metric stays raw on purpose: a dividend rate of 2.23% beside an
            # APY of 2.25%, or a 7.25% note rate beside a 7.29% APR, is the same
            # promise described two ways, not a contradiction.
            buckets[(o.metric, o.family, o.term_months, o.arm, core)].append(o)

        groups: list[list[Obs]] = []
        for key, items in buckets.items():
            if len(items) < 2:
                continue
            # split a bucket by product identity, merging names that share a word
            clusters: list[tuple[set[str], list[Obs]]] = []
            for o in items:
                nk = set(name_key(o))
                hit = None
                for c in clusters:
                    if (not nk) or (not c[0]) or (nk & c[0]):
                        hit = c
                        break
                if hit:
                    hit[0].update(nk)
                    hit[1].append(o)
                else:
                    clusters.append((nk, [o]))
            for _, members in clusters:
                if len(members) < 2:
                    continue
                for sub in _split_by_tier_overlap(members):
                    vals = {round(m.lo, 4) for m in sub}
                    if len(vals) > 1:
                        groups.append(sub)
        return groups


# A payment-illustration row ("P&I $1,637") is not a published price, and a
# marketing sentence with no term or tier ("competitive rate if you carry a
# balance") is not a claim precise enough to contradict anything.
_CALC_LABEL = re.compile(r"^(p&i|p ?& ?i|principal|interest|total|monthly payment|"
                         r"payment|balance|savings)\b", re.I)


def _claimable(o: Obs) -> bool:
    name = o.extras.get("product_name") or o.label
    if _CALC_LABEL.match(name.strip()):
        return False
    if o.extras.get("prose") and not any(
            (o.term_months, o.balance, o.amount, o.model_years)):
        return False
    return True


def _split_by_tier_overlap(members: list[Obs]) -> list[list[Obs]]:
    """Group members whose balance / loan-amount / model-year ranges overlap.

    Non-overlapping tiers are *different* facts; overlapping ones with different
    values are the contradictions we are looking for.
    """
    out: list[list[Obs]] = []
    for m in members:
        placed = False
        for grp in out:
            if all(_tiers_overlap(m, g) for g in grp):
                grp.append(m)
                placed = True
                break
        if not placed:
            out.append([m])
    return out


def _tiers_overlap(a: Obs, b: Obs) -> bool:
    """Are these two rows keyed the same way, and do their tiers meet?

    A row priced by loan amount and a row priced by model year are indexed on
    different dimensions, so they are different facts even when they sit in the
    same table - comparing them manufactures contradictions out of a product's
    own structure.
    """
    dims_a = {attr for attr in ("balance", "amount", "model_years") if getattr(a, attr)}
    dims_b = {attr for attr in ("balance", "amount", "model_years") if getattr(b, attr)}
    if dims_a != dims_b:
        return False
    for attr in dims_a:
        if not intervals_overlap(getattr(a, attr), getattr(b, attr)):
            return False
    return True
