"""A small BM25 index over the site's sentences.

Rate tables answer rate questions. Everything else - who is eligible, what a
programme pays out, how many branches there are - lives in prose, and prose is
where the stale copy lives too. So the index deduplicates identical sentences
across pages and snapshots and keeps every place each one appeared; a sentence
is then dated by the most recent page that still carries it.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from .docmodel import Doc
from .ontology import GENERIC, SYNONYMS
from .textutil import clean, norm, stem, tokens

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(“\"$0-9])")
_JUNK = re.compile(
    r"select a branch|loading locations|sorry, we couldn't find|data points:|"
    r"please select|^\s*go\s*$|cookie|javascript|^table$|^\d+ results$|"
    r"skip to|^menu$|all rights reserved", re.I)


@dataclass
class Occurrence:
    url: str
    page_kind: str
    authority: int
    snapshot_idx: int
    when: date
    title: str = ""
    self_dated: bool = False
    pos: int = 0


@dataclass
class Statement:
    text: str
    toks: tuple[str, ...]          # stems, for scoring
    tf: dict[str, int]
    raw: frozenset = frozenset()   # raw tokens, for exact value checks
    occ: list[Occurrence] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.toks)


class StatementIndex:
    def __init__(self) -> None:
        self.stmts: list[Statement] = []
        self._by_text: dict[str, int] = {}
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.df: dict[str, int] = defaultdict(int)
        self.avgdl = 1.0
        self._total = 0
        # Sentences in the order they appear on each crawl of each page, so a
        # sentence that continues the previous one can be read with it.
        self.doc_seq: dict[tuple[str, int], list[int]] = {}

    # -- build ------------------------------------------------------------
    def add_doc(self, doc: Doc) -> None:
        when = doc.published_on or doc.fetched_at or doc.taken_on
        occ_proto = (doc.url, doc.kind, doc.authority, doc.snapshot_idx, when, doc.title,
                     doc.published_on is not None)
        seq: list[int] = []
        self.doc_seq[(doc.url, doc.snapshot_idx)] = seq
        for b in doc.blocks:
            if b.kind != "text":
                continue
            raw = b.text
            if len(raw) < 12 or _JUNK.search(raw):
                continue
            for sent in _sentences(raw):
                sid = self._add_sentence(sent, occ_proto, len(seq))
                if sid is not None:
                    seq.append(sid)

    def _add_sentence(self, sent: str, proto, pos: int) -> Optional[int]:
        key = norm(sent)
        if len(key) < 12 or len(key) > 420:
            return None
        raw = tokens(key)
        if len(raw) < 4:
            return None
        ts = tuple(stem(t) for t in raw)
        sid = self._by_text.get(key)
        if sid is None:
            tf: dict[str, int] = defaultdict(int)
            for t in ts:
                tf[t] += 1
            sid = len(self.stmts)
            self._by_text[key] = sid
            self.stmts.append(Statement(text=clean(sent), toks=ts, tf=dict(tf),
                                        raw=frozenset(raw)))
            for t in tf:
                self.postings[t].append(sid)
                self.df[t] += 1
            self._total += len(ts)
        st = self.stmts[sid]
        if len(st.occ) < 40:
            st.occ.append(Occurrence(url=proto[0], page_kind=proto[1], authority=proto[2],
                                     snapshot_idx=proto[3], when=proto[4], title=proto[5],
                                     self_dated=proto[6], pos=pos))
        return sid

    def finalize(self) -> None:
        self.avgdl = max(1.0, self._total / max(1, len(self.stmts)))

    def is_ubiquitous(self, word: str, frac: float = 0.12) -> bool:
        """A word the site uses everywhere - its own name, for instance - tells
        us nothing about which sentence answers the question."""
        n = max(1, len(self.stmts))
        return len(self.postings.get(stem(word), ())) > n * frac

    def is_rare(self, word: str, frac: float = 0.004) -> bool:
        """Does this word pin the subject? A word the site barely uses does."""
        n = max(1, len(self.stmts))
        df = len(self.postings.get(stem(word), ()))
        return 0 < df <= max(3, int(n * frac))

    def occurs(self, word: str) -> bool:
        return stem(word) in self.postings

    # -- query --------------------------------------------------------------
    def following(self, occ: Occurrence, count: int = 2) -> list[Statement]:
        """The next sentences on the same page, in reading order."""
        seq = self.doc_seq.get((occ.url, occ.snapshot_idx)) or []
        out = []
        for i in range(occ.pos + 1, min(len(seq), occ.pos + 1 + count)):
            out.append(self.stmts[seq[i]])
        return out

    def statements_with(self, stem_word: str) -> list[Statement]:
        """Every sentence containing a word, regardless of how it scores.

        A count ("22 branch office locations") often sits in a long paragraph of
        boilerplate that keyword scoring buries, so counting questions look the
        word up directly instead of hoping it surfaces in the top results.
        """
        return [self.stmts[i] for i in self.postings.get(stem_word, ())]

    def best_occurrence(self, st: Statement, visible, snap_idx: int):
        best = None
        for o in st.occ:
            if o.snapshot_idx > snap_idx or not visible(o.url, o.snapshot_idx, snap_idx):
                continue
            if best is None or (o.when, o.authority) > (best.when, best.authority):
                best = o
        return best

    def search(self, query: str, visible: Callable[[str, int], bool], snap_idx: int,
               k: int = 30, extra_terms: tuple[str, ...] = ()) -> list[tuple[float, Statement, Occurrence]]:
        qt = [t for t in tokens(norm(query))] + list(extra_terms)
        seen: set[str] = set()
        terms: list[str] = []
        for t in qt:
            for v in (t,) + SYNONYMS.get(t, ()):
                for w0 in v.split():
                    w = stem(w0)
                    if w not in seen and len(w) > 1 and w in self.postings:
                        seen.add(w)
                        terms.append(w)
        n = len(self.stmts) or 1
        # Rarest terms first, and skip terms so common they only add noise.
        terms = sorted(terms, key=lambda t: self.df.get(t, 0))
        scores: dict[int, float] = defaultdict(float)
        budget = 400_000
        for t in terms:
            post = self.postings.get(t)
            if not post:
                continue
            dfv = len(post)
            if dfv > n * 0.35:
                continue
            budget -= dfv
            if budget < 0:
                break
            idf = math.log(1 + (n - dfv + 0.5) / (dfv + 0.5))
            w = idf * (0.45 if t in GENERIC else 1.0)
            for sid in post:
                st = self.stmts[sid]
                f = st.tf.get(t, 0)
                scores[sid] += w * (f * 2.2) / (f + 1.2 * (0.4 + 0.6 * st.length / self.avgdl))
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k * 4]
        out: list[tuple[float, Statement, Occurrence]] = []
        for sid, sc in top:
            st = self.stmts[sid]
            best = None
            for o in st.occ:
                if o.snapshot_idx > snap_idx or not visible(o.url, o.snapshot_idx, snap_idx):
                    continue
                if best is None or (o.when, o.authority) > (best.when, best.authority):
                    best = o
            if best is not None:
                out.append((sc, st, best))
            if len(out) >= k:
                break
        return out


def _strip_bullet(s: str) -> str:
    return re.sub(r"^[-*\u2022]\s*", "", s).strip()


def _sentences(text: str):
    t = clean(text)
    if len(t) <= 240:
        yield t
        return
    for s in SENT_SPLIT.split(t):
        s = s.strip()
        if 12 <= len(s) <= 420:
            yield s
