"""KnowledgeNow: ingest snapshots, answer questions, report contradictions."""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Optional

from . import phrasing
from .contract_types import Answer, Conflict, Source
from .docmodel import parse_page
from .entities import (branch_facts, hours_facts, identifier_facts, inline_hours_facts,
                       link_facts)
from .extract import fee_schedule_facts, prose_facts, table_facts
from .facts import Obs
from .ontology import (FAMILY_LABEL, GENERIC, SYNONYMS, family_for_word, family_of,
                       word_variants)
from .query import parse as parse_question
from .quals import intervals_overlap
from .retrieval import StatementIndex
from .store import TIER, Store, name_key
from .textutil import clean, content_tokens, norm, parse_date, stem, tokens

RATE_METRICS = ("apy", "dividend_rate", "apr", "rate")
DISCRIMINATORS = ("jumbo", "ira", "intro", "mini_max", "secured", "student", "special")
MAX_GROUPS = 9
MAX_ROWS = 8


class KnowledgeNow:
    def __init__(self) -> None:
        self.store = Store()
        self.index = StatementIndex()
        self._label_df: Counter = Counter()
        self._fact_count = 0
        self._cache: dict[tuple, Answer] = {}
        self._ready = False
        self._conflicts: Optional[list[Conflict]] = None

    # ------------------------------------------------------------------
    # ingest
    # ------------------------------------------------------------------
    def ingest(self, snapshot) -> None:
        taken_on = parse_date(snapshot.taken_on) or date(1970, 1, 1)
        idx = self.store.add_snapshot(snapshot.snapshot_id, taken_on)
        for page in snapshot.pages:
            fetched = parse_date((page.fetched_at or "")[:10]) or taken_on
            doc = parse_page(page.url, page.title or "", page.markdown or "",
                             snapshot.snapshot_id, taken_on, fetched, idx)
            self.store.note_page(doc.url, idx)
            facts: list[Obs] = []
            for b in doc.blocks:
                if b.kind == "table":
                    facts.extend(table_facts(b, doc))
                else:
                    facts.extend(prose_facts(b, doc))
            facts.extend(fee_schedule_facts(doc, _is_fee_page(doc)))
            facts.extend(branch_facts(doc))
            facts.extend(hours_facts(doc))
            facts.extend(inline_hours_facts(doc))
            facts.extend(identifier_facts(doc))
            facts.extend(link_facts(doc))
            for f in facts:
                self.store.add(f)
                label = f"{f.extras.get('product_name', '')} {f.label}"
                self.store.vocab.add_product(label)
                self._fact_count += 1
                for t in set(tokens(label)):
                    self._label_df[t] += 1
            self.store.vocab.add_product(doc.title)
            self.index.add_doc(doc)
        self._ready = False
        self._cache.clear()
        self._conflicts = None

    def _finalize(self) -> None:
        if self._ready:
            return
        self.store.finalize()
        self.index.finalize()
        self.store.vocab.stems.update(self.index.postings.keys())
        self._ready = True

    # ------------------------------------------------------------------
    # answer
    # ------------------------------------------------------------------
    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        self._finalize()
        key = (question or "", as_of or "")
        if key in self._cache:
            return self._cache[key]
        try:
            ans = self._answer(question or "", as_of)
        except Exception:                      # never fail a member question
            if os.environ.get("KNOWLEDGE_NOW_DEBUG"):
                raise
            ans = Answer(text=phrasing.REFUSAL, confident=False)
        self._cache[key] = ans
        return ans

    def _answer(self, question: str, as_of: Optional[str]) -> Answer:
        spec = parse_question(question, as_of, self.store.vocab)
        snap = self.store.snapshot_index_for(spec.as_of)
        for intent in spec.intents:
            handler = getattr(self, f"_h_{intent}", None)
            if handler is None:
                continue
            ans = handler(spec, snap)
            if ans is not None:
                return ans
        return Answer(text=phrasing.REFUSAL, confident=False)

    # -- shared helpers ---------------------------------------------------
    def _live(self, snap: int, domain: str) -> list[Obs]:
        return [self.store.obs[i] for i in self.store.visible_at(snap)
                if self.store.obs[i].domain == domain]

    def _visible_url(self, url: str, occ_idx: int, snap: int) -> bool:
        return self.store.newest_idx(url, snap) == occ_idx

    def _path_penalty(self, o: Obs) -> int:
        """1 when the URL's own path is about a different product than the fact.

        The same rate table is syndicated onto several pages; citing the page
        whose path names another product reads as a mistake even when the number
        is right.
        """
        if not o.family:
            return 0
        path = o.prov.path_words
        fam = family_of(path)
        return 1 if fam and fam != o.family else 0

    def _agreeing(self, rows: list[Obs]) -> list[Obs]:
        """Every live observation that asserts one of the values we just gave.

        The same rate is published on the product page, the master rate sheet
        and often a guide page. All of them are genuine sources for the answer,
        and citing them all is both honest and more useful than naming one.
        """
        pool = getattr(self, "_pool", None) or rows
        want = {(o.metric, round(o.lo, 4), o.term_months) for o in rows if o.lo is not None}
        extra = [o for o in pool
                 if o.lo is not None and (o.metric, round(o.lo, 4), o.term_months) in want]
        return rows + extra

    def _sources(self, winners: list[Obs], limit: int = 6) -> list[Source]:
        """Cite the pages that carry the value we just gave, best first."""
        winners = self._agreeing(winners)
        out: list[Source] = []
        seen: set[str] = set()
        ranked = sorted(winners, key=lambda o: (
            self._path_penalty(o),
            0 if o.prov.page_kind == "product" else 1 if o.prov.page_kind == "rate_sheet" else 2,
            -o.prov.authority, o.prov.url))
        for o in ranked:
            u = o.prov.url
            if u in seen:
                continue
            seen.add(u)
            out.append(Source(url=u, snapshot_id=o.prov.snapshot_id, quote=o.prov.quote or None))
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------------
    # intent: structured rate / fee lookup
    # ------------------------------------------------------------------
    def _h_rate(self, spec, snap: int) -> Optional[Answer]:
        # Only a question that asks for a number reaches the rate tables.
        if not spec.price_question:
            return None
        live = [o for o in self._live(snap, "rate") + self._live(snap, "fee")
                if o.lo is not None]
        pool = [o for o in live if self._metric_ok(spec, o)]
        pinned = bool(spec.families) or spec.arm is not None
        if spec.families:
            pool = [o for o in pool if o.family in spec.families]
        if spec.metric == "fee" and not self._survivors(spec, pool):
            # "What do you charge for a 15-year mortgage?" is asking the rate;
            # only a product with a published fee answers it as a fee.
            spec.metric = "rate"
            pool = [o for o in live if self._metric_ok(spec, o)]
            pinned = bool(spec.families) or spec.arm is not None
            if spec.families:
                pool = [o for o in pool if o.family in spec.families]
        strict = [o for o in pool if self._quals_ok(spec, o)]
        if not strict and spec.term_months is not None:
            # A rate written into a sentence may carry its term in the wording
            # rather than in a column we parsed. Accept that as the term before
            # concluding the site does not publish the rate at all.
            strict = [o for o in pool
                      if o.term_months is None and self._quals_ok(spec, o, ignore_term=True)
                      and _mentions_term(o, spec.term_months)]
        pool = strict
        if not pool:
            # The member named a product or a variant and the rate sheet has no
            # row for it. Saying so is the correct answer; falling through to
            # prose would surface some other product's number.
            return self._decline_rate(spec) if pinned else None
        # Note: an unfamiliar word in the question is *not* grounds to decline.
        # "six", "January" or "quarterly" are the member's phrasing, not product
        # names. Declining is driven by the qualifier filters above, which fail
        # only when the rate sheet genuinely has no row for what was asked.
        pool = self._pin_product(spec, pool) or pool
        if not pool:
            return None

        self._pool = pool
        groups = self._group(pool, spec)
        if not groups:
            return None
        resolved = [self._resolve_group(g) for g in groups]
        resolved = [r for r in resolved if r]
        if not resolved:
            return None
        # A rate table is the maintained source. Product-page marketing copy
        # ("lower ongoing rate if you carry a balance") repeats a number without
        # its qualifiers, so it is corroboration, never a separate product.
        tabled = [r for r in resolved if "table_header" in r[0].extras]
        if tabled:
            resolved = tabled
        resolved = [self._prefer_purchase(r, spec) for r in resolved]
        resolved.sort(key=lambda r: (-self._prominence(r), r[0].label))
        resolved = resolved[:MAX_GROUPS]

        if spec.verify_value is not None:
            return self._answer_verify(spec, resolved)
        if spec.want_max or spec.want_min:
            return self._answer_superlative(spec, resolved)
        return self._answer_rows(spec, resolved)

    # -- filters ----------------------------------------------------------
    def _survivors(self, spec, pool: list[Obs]) -> list[Obs]:
        return [o for o in pool if self._quals_ok(spec, o)]

    def _metric_ok(self, spec, o: Obs) -> bool:
        want = spec.metric
        m = o.metric
        if want in (None, "rate"):
            return m in RATE_METRICS
        if want in ("apy", "dividend_rate"):
            return m in ("apy", "dividend_rate")
        if want == "apr":
            return m in ("apr", "rate")
        if want == "fee":
            return m == "fee"
        return m == want

    def _quals_ok(self, spec, o: Obs, ignore_term: bool = False) -> bool:
        if not ignore_term and spec.term_months is not None and o.term_months != spec.term_months:
            return False
        if spec.arm:
            if o.arm != spec.arm:
                return False
        elif o.arm:
            return False
        if spec.model_year is not None and o.model_years is not None:
            lo, hi = o.model_years
            if not (lo - 1e-9 <= spec.model_year <= hi + 1e-9):
                return False
        if spec.balance_iv is not None:
            iv = o.balance or o.amount
            if iv is not None and not intervals_overlap(iv, spec.balance_iv):
                return False
        for f in DISCRIMINATORS:
            if (f in spec.flags) != (f in o.flags):
                return False
        if "business" not in spec.flags and "business" in o.flags and not spec.want_list:
            return False
        if "business" in spec.flags and "business" not in o.flags:
            return False
        return True

    def _pin_product(self, spec, pool: list[Obs]) -> Optional[list[Obs]]:
        """Keep the rows that best cover the member's own product words.

        Coverage rather than filtering, because a question mixes identity words
        with modifiers: "the standard purchase APR on the Platinum Visa" names
        one product ("Platinum Visa") and two modifiers. A contiguous phrase the
        site itself uses is worth far more than any single word, which is what
        stops "standard" from steering the answer to the Rewards card.

        Returns None when nothing covers the member's words at all - a signal to
        decline rather than return the nearest number.
        """
        reqs = [t for t in dict.fromkeys(spec.content)
                if t not in GENERIC and len(t) > 2 and not t[0].isdigit()
                and t not in _NEVER_PIN]
        phrases = [p for p in spec.phrases if " " in p and
                   not all(w in GENERIC for w in p.split())]
        if not reqs and not phrases:
            return pool
        total = max(1, self._fact_count)
        best, scored = 0.0, []
        for o in pool:
            hay = o.match_text
            # The row's own name is stronger evidence than the section heading
            # it happens to sit under: "Rush fee for debit card" is about a
            # debit card, while every row under "Debit Card Service Charges"
            # merely inherits those words.
            name_hay = norm(f"{o.extras.get('product_name', '')} {o.label}")
            cov = 0.0
            for ph in phrases:
                if ph in hay:
                    cov += 6.0 + (2.0 if ph in name_hay else 0.0)
            for t in reqs:
                literal = (any(a in hay for a in (t,) + SYNONYMS.get(t, ()))
                           or any(v in hay for v in word_variants(t)))
                by_family = (family_of(t) or family_for_word(t)) == o.family
                if not (literal or by_family):
                    continue
                df = self._label_df.get(t, 0)
                weight = 1.0 + 2.0 * (1.0 - min(1.0, df / total) ** 0.25)
                # A page that actually prints the word is stronger evidence than a
                # row that merely belongs to the family the word names, which is
                # what separates "Auto & Motorcycle Loans" from the recreational
                # sheet when someone asks about an auto loan.
                cov += weight if literal else weight * 0.45
            scored.append((cov, o))
            best = max(best, cov)
        if best <= 0.0:
            # The member's wording gave us nothing to choose between these rows,
            # but the qualifiers already pinned them, so answer from all of them
            # rather than pretending the rate is not published.
            return pool
        return [o for cov, o in scored if cov >= best - 1e-9]

    def _decline_rate(self, spec) -> Answer:
        subject = "a published rate for that"
        if spec.arm:
            subject = "a rate for that adjustable-rate program"
        return Answer(text=phrasing.refusal(subject), confident=False)

    # -- grouping and resolution -------------------------------------------
    def _group(self, pool: list[Obs], spec) -> list[list[Obs]]:
        buckets: dict[tuple, list[Obs]] = defaultdict(list)
        for o in pool:
            core = frozenset((o.flags & set(DISCRIMINATORS)) | (o.flags & {"business"}))
            buckets[(o.family, frozenset(name_key(o)), core, o.term_months)].append(o)
        merged: list[list[Obs]] = []
        for key in sorted(buckets, key=lambda k: (str(k[0]), sorted(k[1]), sorted(k[2]), k[3] or 0)):
            merged.append(buckets[key])
        return merged

    def _resolve_group(self, members: list[Obs]) -> Optional[tuple[Obs, list[Obs], list[Obs]]]:
        """-> (winner, rows of the winning publication, superseded observations)"""
        if not members:
            return None
        winner = max(members, key=self._rank)
        gid = self._pub_id(winner)
        rows = [o for o in members if self._pub_id(o) == gid]
        beaten = [o for o in members
                  if self._pub_id(o) != gid and round(o.lo, 4) != round(winner.lo, 4)]
        # keep one row per distinct (tier, value) so duplicate pages don't repeat
        seen: set[tuple] = set()
        uniq: list[Obs] = []
        for o in sorted(rows, key=lambda x: (-self._rank(x)[0], x.label)):
            k = (o.term_months, o.balance, o.amount, o.model_years, round(o.lo, 4),
                 o.metric, frozenset(o.flags))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(o)
        return (winner, uniq, beaten)

    def _pub_id(self, o: Obs) -> tuple:
        return (o.extras.get("table_header", ""), o.prov.effective_on,
                o.prov.page_kind in ("product", "rate_sheet"))

    def _rank(self, o: Obs) -> tuple:
        p = o.prov
        return (0 if o.extras.get("expired_on") else 1,
                TIER.get(p.page_kind, 2), 1 if p.self_dated else 0, p.asserted_on.toordinal(),
                1 if "purchase" in o.flags else (0 if "refinance" not in o.flags else -1),
                -self._path_penalty(o), p.authority, p.snapshot_idx)

    def _prominence(self, r) -> float:
        winner = r[0]
        return (winner.prov.authority / 100.0
                + (0.5 if winner.prov.page_kind in ("product", "rate_sheet") else 0.0)
                + 0.2 * len(r[1])
                - (1.5 if "business" in winner.flags else 0.0)
                - (0.8 if "ira" in winner.flags else 0.0))

    def _prefer_purchase(self, resolved, spec):
        """A mortgage rate sheet prints purchase and refinance side by side.

        Unless the member says "refinance", purchase is the reading of "what's
        your 30-year rate", and printing both doubles the answer for no gain.
        """
        if "refinance" in spec.flags or "purchase" in spec.flags:
            return resolved
        winner, rows, beaten = resolved
        purchase = [o for o in rows if "refinance" not in o.flags]
        if purchase and len(purchase) < len(rows):
            if "refinance" in winner.flags:
                winner = max(purchase, key=self._rank)
            return (winner, purchase, beaten)
        return resolved

    # -- renderers ---------------------------------------------------------
    def _answer_rows(self, spec, resolved) -> Answer:
        lines: list[str] = []
        winners: list[Obs] = []
        beaten: list[Obs] = []
        for winner, rows, lost in resolved:
            winners.extend(rows)
            beaten.extend(lost)
            for line in _merge_metrics(rows[:MAX_ROWS * 2])[:MAX_ROWS]:
                lines.append(line)
        if not lines:
            return Answer(text=phrasing.REFUSAL, confident=False)
        text = "; ".join(lines) + phrasing.effective_clause(resolved[0][0])
        text = text.rstrip(".") + "."
        return Answer(text=text, sources=self._sources(winners), confident=True,
                      conflict_note=self._note(beaten, resolved))

    def _answer_superlative(self, spec, resolved) -> Answer:
        rows = [o for _, rs, _ in resolved for o in rs]
        if not rows:
            return Answer(text=phrasing.REFUSAL, confident=False)
        pick = (max if spec.want_max else min)(rows, key=lambda o: (o.hi or o.lo, -(o.balance or (0, 0))[0]))
        beaten = [o for _, _, lost in resolved for o in lost]
        word = "highest" if spec.want_max else "lowest"
        text = f"The {word} is {phrasing.one_line(pick)}" + phrasing.effective_clause(pick)
        return Answer(text=text.rstrip(".") + ".", sources=self._sources([pick] + rows),
                      confident=True, conflict_note=self._note(beaten, resolved))

    def _answer_verify(self, spec, resolved) -> Answer:
        target = spec.verify_value
        rows = [o for _, rs, _ in resolved for o in rs]
        hit = [o for o in rows if _matches(o, target)]
        if hit:
            o = hit[0]
            return Answer(text=f"Yes - {phrasing.one_line(o)}.",
                          sources=self._sources(hit + rows), confident=True)
        ceiling = max(rows, key=lambda o: (o.hi or o.lo))
        lines = []
        for _, rs, _ in resolved[:3]:
            ordered = sorted(rs, key=lambda o: -(o.hi or o.lo))
            lines.extend(_merge_metrics(ordered)[:MAX_ROWS])
        body = "; ".join(lines[:MAX_ROWS + 2])
        if not body:
            return Answer(text=phrasing.REFUSAL, confident=False)
        text = (f"No - nothing on the rate sheet pays that. The published figures are: "
                f"{body}.")
        return Answer(text=text, sources=self._sources([ceiling] + rows), confident=True)

    def _note(self, beaten: list[Obs], resolved) -> Optional[str]:
        if not beaten:
            return None
        w = resolved[0][0]
        other = beaten[0]
        why = _why(w, other)
        return (f"Other pages still show {other.value_text} for this "
                f"({other.prov.url}); {w.value_text} is current because {why}.")

    # ------------------------------------------------------------------
    # intent: hours
    # ------------------------------------------------------------------
    def _h_hours(self, spec, snap: int) -> Optional[Answer]:
        rows = self._live(snap, "hours")
        if not rows:
            return None
        want = "drive-up" if re.search(r"drive[- ]?(up|thru|through)", spec.nq) else (
            "lobby" if "lobby" in spec.nq else None)
        if want:
            rows = [o for o in rows if o.metric == want] or rows
        place, unknown = self._place_filter(spec, rows)
        if unknown:
            return Answer(text=phrasing.refusal("hours for that location"), confident=False)
        if place is not None:
            rows = place
        kinds = {o.metric for o in rows}
        kind = want or ("lobby" if "lobby" in kinds else sorted(kinds)[0])
        corroborating = [o for o in rows if o.metric == "contact"] if kind != "contact" else []
        rows = [o for o in rows if o.metric == kind] or rows
        best: dict[str, Obs] = {}
        for o in sorted(rows, key=lambda o: (-self._rank(o)[2], o.label)):
            best.setdefault(o.value_text, o)
        variants = list(best.values())
        if not variants:
            return None
        if len(variants) == 1:
            text = f"{kind.capitalize()} hours: {variants[0].value_text}."
        else:
            # Branches keep different schedules, so answer with the full picture
            # per day rather than picking one branch's hours and calling it the
            # general answer.
            text = (f"{kind.capitalize()} hours vary by branch - "
                    + _hours_by_day(variants)
                    + ". Which branch did you have in mind, or would you like all of them?")
        return Answer(text=text, sources=self._sources(variants[:4] + corroborating[:2]),
                      confident=True)

    def _place_filter(self, spec, rows: list[Obs]):
        """Did the member name a place, and do we hold anything for it?

        Only capitalised names count as places. If the member named one and no
        branch record mentions it, we decline: a branch we have no record of is
        exactly the case where a plausible-sounding schedule would be a
        fabrication.
        """
        cands = [t for t in spec.proper_nouns if len(t) > 2]
        core = [t for t in cands if t not in _STREET_WORDS]
        if not core:
            return None, False
        hits = [o for o in rows if all(c in o.match_text for c in core)]
        if hits:
            optional = [o for o in hits
                        if all(c in o.match_text for c in cands)]
            return (optional or hits), False
        return None, True

    # ------------------------------------------------------------------
    # intent: address
    # ------------------------------------------------------------------
    def _h_address(self, spec, snap: int) -> Optional[Answer]:
        rows = [o for o in self._live(snap, "branch")]
        if not rows:
            return None
        if re.search(r"\batm\b", spec.nq):
            rows = [o for o in rows if o.extras.get("loc_kind") == "atm"] or rows
        elif re.search(r"\bbranch\b|\boffice\b", spec.nq):
            rows = [o for o in rows if o.extras.get("loc_kind") == "branch"] or rows
        place, unknown = self._place_filter(spec, rows)
        if unknown:
            return Answer(text=phrasing.refusal("that location"), confident=False)
        if place is None:
            return None
        picks, seen = [], set()
        for o in sorted(place, key=lambda o: (-self._rank(o)[2], o.value_text)):
            if o.value_text in seen:
                continue
            seen.add(o.value_text)
            picks.append(o)
        if not picks:
            return None
        if len(picks) == 1:
            text = f"{picks[0].value_text}."
        else:
            text = "; ".join(p.value_text for p in picks[:4]) + "."
        return Answer(text=text, sources=self._sources(picks), confident=True)

    # ------------------------------------------------------------------
    # intent: identifiers
    # ------------------------------------------------------------------
    def _h_entity(self, spec, snap: int) -> Optional[Answer]:
        rows = self._live(snap, "entity")
        if not rows:
            return None
        if re.search(r"routing|aba\b", spec.nq):
            rows = [o for o in rows if o.metric == "routing_number"]
            label = "Dupaco's routing number is"
        elif re.search(r"phone|call|contact", spec.nq):
            rows = [o for o in rows if o.metric == "phone"]
            label = "You can reach Dupaco at"
        else:
            return None
        if not rows:
            return None
        counts = Counter(o.value_text for o in rows)
        value = max(counts, key=lambda v: (counts[v], max(
            o.prov.asserted_on for o in rows if o.value_text == v)))
        agree = [o for o in rows if o.value_text == value]
        return Answer(text=f"{label} {value}.", sources=self._sources(agree), confident=True)

    # ------------------------------------------------------------------
    # intent: canonical link
    # ------------------------------------------------------------------
    def _h_link(self, spec, snap: int) -> Optional[Answer]:
        rows = self._live(snap, "link")
        if not rows:
            return None
        terms = [t for t in dict.fromkeys(spec.content)
                 if t not in GENERIC and t not in _LINK_CUES and len(t) > 1]
        if not terms:
            return None
        best_score, best = 0.0, None
        for o in rows:
            hay = o.match_text
            hit = 0
            for t in terms:
                alts = set(word_variants(t))
                for syn in SYNONYMS.get(t, ()):
                    alts.add(syn)
                if any(a in hay for a in alts):
                    hit += 1
            if not hit:
                continue
            # Coverage says which link is about the right thing; the editorial
            # tier says whose copy of that link is still maintained, which is
            # what keeps an old campaign URL out of the answer.
            # What matters for "which link should I use" is the authority of
            # the page the link points AT; the host page only corroborates that
            # the link is still live.
            target = TIER.get(o.extras.get("target_kind", "other"), 2)
            score = 3.0 * (hit / len(terms)) + target + 0.4 * TIER.get(o.prov.page_kind, 2)
            # Personal and business are different products, and so are their
            # pages: never hand a member the business version of a link.
            is_biz = "business" in hay or "business" in o.value_text.lower()
            if is_biz != ("business" in spec.flags):
                score -= 2.5
            if score > best_score:
                best_score, best = score, o
        if best is None or best_score < 3.2:
            return None
        return Answer(text=f"Use {best.value_text}", sources=self._sources([best]), confident=True)

    # ------------------------------------------------------------------
    # intent: counted facts and free text
    # ------------------------------------------------------------------
    def _h_count(self, spec, snap: int) -> Optional[Answer]:
        ans = self._counted_fact(spec, snap)
        return ans if ans is not None else self._passage(spec, snap, require_number=True)

    def _counted_fact(self, spec, snap: int) -> Optional[Answer]:
        """Answer "how many X" from the most recent statement that counts X.

        A count is the clearest case of a fact with a date: the site still shows
        "19 branch office locations" from 2022 next to "22" from 2026, and only
        the publication date separates them.
        """
        if not spec.count_noun:
            return None
        target = stem(spec.count_noun)
        cands = []
        for st in self.index.statements_with(target):
            if not _counts_noun(st.text, [target]):
                continue
            if _NETWORK_COUNT.search(st.text):
                continue
            occ = self.index.best_occurrence(st, self._visible_url, snap)
            if occ is None:
                continue
            cands.append((occ.when, occ.authority, st, occ))
        if not cands:
            return None
        # Narrow to statements that also mention what the member asked about,
        # so "how many employees" is not answered from a page about something
        # else that happens to count employees.
        others = [stem(t) for t in content_tokens(spec.nq)
                  if stem(t) != target and len(t) > 3
                  and not self.index.is_ubiquitous(t)]
        if others:
            focused = [c for c in cands if sum(1 for o in others if o in c[2].toks) >= 1]
            cands = focused or cands
        if spec.count_verb:
            v = stem(spec.count_verb)
            acting = [c for c in cands if _verb_follows(c[2].text, target, v)]
            cands = acting or cands
        # A press release is a dated assertion; an undated staff bio that
        # mentions a number is not evidence about today.
        cands.sort(key=lambda c: (c[3].self_dated, c[0], c[1]), reverse=True)
        when, _, st, occ = cands[0]
        return Answer(text=clean(st.text), confident=True,
                      sources=[Source(url=occ.url,
                                      snapshot_id=self.store.snaps[occ.snapshot_idx].snapshot_id,
                                      quote=clean(st.text)[:200])],
                      conflict_note=_count_note(cands))

    def _h_passage(self, spec, snap: int) -> Optional[Answer]:
        return self._passage(spec, snap, require_number=False)

    def _passage(self, spec, snap: int, require_number: bool) -> Optional[Answer]:
        hits = self.index.search(spec.nq, self._visible_url, snap, k=60)
        if not hits:
            return None
        # Prose keys keep words the rate lookup treats as generic ("loan",
        # "account"): in a sentence they carry most of the meaning.
        raw_keys = [t for t in dict.fromkeys(content_tokens(spec.nq)) if len(t) > 2]
        raw_keys += [w for t in list(raw_keys) for syn in (SYNONYMS.get(t, ()),) for w in syn]
        keys = list(dict.fromkeys(stem(w) for k in raw_keys for w in k.split()))
        # Words the site barely uses identify what the member is asking about.
        # A statement that does not contain them is about something else, no
        # matter how well it scores.
        pins = [t for t in keys if self.index.is_rare(t)]
        if pins:
            grounded = [h for h in hits if all(t in h[1].toks for t in pins)]
            hits = grounded or [h for h in hits if any(t in h[1].toks for t in pins)] or hits
        if require_number:
            # "How many branches?" is answered by a number standing next to the
            # thing being counted, not by any sentence that happens to contain a
            # digit somewhere.
            tight = [h for h in hits if _number_near(h[1].text, keys)]
            hits = tight or [h for h in hits if re.search(r"\d", h[1].text)] or hits
            # A count of someone else's network is not a count of Dupaco's own.
            own = [h for h in hits if not _NETWORK_COUNT.search(h[1].text)]
            hits = own or hits
        wants_rate = spec.metric in ("apy", "apr", "dividend_rate") or (
            spec.metric == "rate" and re.search(r"\brate\b|\bapr\b|\bapy\b", spec.nq))
        if wants_rate:
            hits = [h for h in hits if "%" in h[1].text]
            if not hits:
                return Answer(text=phrasing.refusal("a published rate for that"), confident=False)
        hits = [(sc + _key_bonus(st, keys)
                 + (0.8 if spec.definitional and _DEFINING.search(st.text) else 0.0)
                 + (2.0 if spec.yes_no and _ANSWERS_YES_NO.match(st.text) else 0.0)
                 + _page_prior(occ, keys)
                 - (4.0 if st.text.rstrip().endswith("?") else 0.0)
                 - (3.0 if _NAVIGATIONAL.search(st.text) else 0.0),
                 st, occ) for sc, st, occ in hits]
        hits.sort(key=lambda h: -h[0])
        # A sentence that is itself a question (an FAQ heading) matches the
        # member's words closely and answers nothing.
        hits = [h for h in hits if not h[1].text.rstrip().endswith("?")] or hits
        top = hits[0][0]
        if require_number:
            # A count is a fact that changes. Among grounded candidates the most
            # recently published number is the current one - that is how "22
            # branch offices" (May 2026) beats "23" (June 2025) even though both
            # sentences are still on the site today.
            near = [h for h in hits if h[0] >= top * 0.6]
            near.sort(key=lambda h: (h[2].when, h[0]), reverse=True)
        else:
            # Otherwise relevance decides *which claim* answers the question, and
            # recency only chooses between restatements of that same claim - the
            # press release that says "the fifth time" against last year's copy
            # that says "the fourth".
            near = _same_claim(hits, hits[0][1])
            near.sort(key=lambda h: (h[2].when, h[0]), reverse=True)
        score, st, occ = near[0]
        if score < 1.6:
            return None
        text = self._read_around(hits, st, occ, keys, spec.wants_number)
        if _CONTACT_Q.search(spec.nq) and not re.search(r"\d{3}[-.]\d{3}[-.]\d{4}", text):
            phone = self._main_phone(snap)
            if phone:
                text = text.rstrip(".") + f", or call Dupaco at {phone}."
        return Answer(text=text, confident=True,
                      sources=[Source(url=occ.url, snapshot_id=self.store.snaps[occ.snapshot_idx].snapshot_id,
                                      quote=text[:200])] + self._extra_sources(near[1:4]))

    def _read_around(self, hits, st, occ, keys, wants_number: bool) -> str:
        """Answer with the sentence plus its immediate support on the same page.

        A member asking "what is Home$tart?" wants the amount as well as the
        description, and those are usually adjacent sentences. Only the same
        page is pulled in, so nothing from another vintage can creep in.
        """
        parts = [(st, occ)]
        covered = {k for k in keys if k in st.toks}
        has_number = bool(re.search(r"\d", st.text))
        for sc, other, o2 in hits:
            if len(parts) >= 3 or o2.url != occ.url or other is st:
                continue
            if other.text.rstrip().endswith("?") or _NAVIGATIONAL.search(other.text):
                continue
            add = {k for k in keys if k in other.toks}
            # Only reach for a figure when the member asked for one; otherwise a
            # stray year from a methodology paragraph becomes a stale-value trap.
            gains_fact = wants_number and bool(re.search(r"\d", other.text)) and not has_number
            if not (add - covered) and not gains_fact:
                continue
            parts.append((other, o2))
            covered |= add
            has_number = has_number or bool(re.search(r"\d", other.text))

        # "...through the Home$tart program. This initiative provides up to
        # $15,000..." - the sentence carrying the figure often refers back
        # instead of repeating the name, so read on to the end of the thought.
        out: list[str] = []
        for part, o in parts:
            out.append(clean(part.text))
            for nxt in self.index.following(o, 2):
                t = clean(nxt.text)
                if not _CONTINUATION.match(t) or t.rstrip().endswith("?"):
                    break
                if _NAVIGATIONAL.search(t):
                    break
                out.append(t)

        text, seen = "", set()
        for piece in out:
            if piece in seen or piece in text:
                continue
            seen.add(piece)
            nxt_text = piece if not text else f"{text.rstrip('.')}. {piece}"
            if len(nxt_text) > 520:
                break
            text = nxt_text
        return text or clean(st.text)[:420]

    def _main_phone(self, snap: int) -> Optional[str]:
        rows = [o for o in self._live(snap, "entity") if o.metric == "phone"]
        if not rows:
            return None
        counts = Counter(o.value_text for o in rows)
        return max(counts, key=lambda v: counts[v])

    def _extra_sources(self, hits) -> list[Source]:
        out, seen = [], set()
        for _, st, occ in hits:
            if occ.url in seen:
                continue
            seen.add(occ.url)
            out.append(Source(url=occ.url,
                              snapshot_id=self.store.snaps[occ.snapshot_idx].snapshot_id))
        return out

    # ------------------------------------------------------------------
    # conflicts
    # ------------------------------------------------------------------
    def conflicts(self) -> list[Conflict]:
        self._finalize()
        if self._conflicts is not None:
            return self._conflicts
        snap = len(self.store.snaps) - 1
        out: list[Conflict] = []
        for group in self.store.conflict_groups(snap):
            winner = max(group, key=self._rank)
            vals, srcs, seen = [], [], set()
            for o in sorted(group, key=lambda x: -self._rank(x)[2]):
                if o.value_text in seen:
                    continue
                seen.add(o.value_text)
                vals.append(o.value_text)
                srcs.append(Source(url=o.prov.url, snapshot_id=o.prov.snapshot_id,
                                   quote=o.prov.quote or None))
            if len(vals) < 2:
                continue
            loser = next((o for o in group if o.value_text != winner.value_text), None)
            out.append((
                # Identical disagreements repeated across every balance tier are
                # one editorial mistake, not several, so they collapse together.
                (winner.family, frozenset(name_key(winner)), winner.metric,
                 frozenset(winner.flags & set(DISCRIMINATORS)), tuple(sorted(vals))),
                Conflict(topic=_topic(winner), values=vals[:6], sources=srcs[:6],
                         resolution=winner.value_text,
                         reason=_why(winner, loser) if loser else None),
                winner))
        collapsed: dict[tuple, Conflict] = {}
        for key, conflict, _winner in out:
            if key not in collapsed:
                collapsed[key] = conflict
        final = list(collapsed.values())
        final.sort(key=lambda c: -_conflict_weight(c))
        self._conflicts = final[:40]
        return self._conflicts


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_NEVER_PIN = frozenset({"loan", "loans", "rate", "rates", "account", "accounts",
                        "current", "right", "today", "much", "cost", "costs",
                        "apr", "aprs", "apy", "apys", "yield", "interest", "dividend",
                        "dividends", "fee", "fees", "percent", "earn", "pay", "pays"})

_LINK_CUES = frozenset({"link", "links", "url", "urls", "website", "web", "page", "use",
                        "using", "used", "would", "could", "should", "me", "my",
                        "see", "find", "go", "view", "visit", "sign", "should",
                        "want", "need", "get", "where", "how", "take", "check"})

_CONTACT_Q = re.compile(
    r"\bhow (?:can|do) i\b[^?]{0,60}\b(update|change|contact|reach|report|cancel|"
    r"close|order|replace|dispute|reset|unlock|add|remove)\b")

# "Mullan Avenue" is a branch we may or may not have; "Avenue" is not a branch.
_STREET_WORDS = frozenset({"avenue", "ave", "street", "st", "road", "rd", "drive", "dr",
                           "lane", "ln", "boulevard", "blvd", "court", "ct", "way",
                           "highway", "hwy", "place", "pl", "circle", "parkway", "pkwy",
                           "north", "south", "east", "west", "suite", "plaza", "center"})

_DAY_ORDER = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_DAY_ABBR = {"mon": "monday", "tue": "tuesday", "tues": "tuesday", "wed": "wednesday",
             "wednes": "wednesday", "thu": "thursday", "thur": "thursday",
             "thurs": "thursday", "fri": "friday", "sat": "saturday",
             "satur": "saturday", "sun": "sunday"}


def _expand_days(spanned: str) -> list[str]:
    """'Monday-Thursday' -> the four weekdays it covers."""
    parts = [p.strip() for p in re.split(r"\s*(?:-|through|to|and|,)\s*", norm(spanned)) if p.strip()]
    full = [_DAY_ABBR.get(p[:5], _DAY_ABBR.get(p[:4], _DAY_ABBR.get(p[:3], p))) for p in parts]
    full = [d for d in full if d in _DAY_ORDER]
    if len(full) == 2 and re.search(r"-|through|to", norm(spanned)):
        i, j = _DAY_ORDER.index(full[0]), _DAY_ORDER.index(full[1])
        if i <= j:
            return list(_DAY_ORDER[i:j + 1])
    return full


def _hours_by_day(variants: list[Obs]) -> str:
    per_day: dict[str, list[str]] = {d: [] for d in _DAY_ORDER}
    for o in variants:
        for days, hours in o.extras.get("rows", []):
            for d in _expand_days(days):
                h = clean(hours)
                if h and h not in per_day[d]:
                    per_day[d].append(h)
    out = []
    for d in _DAY_ORDER:
        vals = per_day[d]
        if not vals:
            continue
        out.append(f"{d.capitalize()} {' or '.join(vals[:3])}")
    return "; ".join(out)


_NETWORK_COUNT = re.compile(
    r"co-?op|shared branch|nationwide|participating|surcharge-free|across the country|"
    r"network of|other credit unions", re.I)

# Sentences that point somewhere else instead of saying anything. They match a
# question's words well and answer nothing.
_NAVIGATIONAL = re.compile(
    r"^\s*(for more information|to learn more|learn more|read more|visit |see |click|"
    r"apply (?:today|now|online)|find out more|explore|download|check out|sign up)|"
    r"\b(?:visit|see) (?:www\.|https?://|dupaco\.com)", re.I)

_DEFINING = re.compile(
    r"\b(is a|are a|is an|provides?|offers?|gives?|helps?|allows?|means|covers?|"
    r"pays?|includes?|up to|designed to|lets you)\b", re.I)


# On an FAQ the answer to "can I...?" literally begins with the answer.
_ANSWERS_YES_NO = re.compile(r"^\s*(yes|no|absolutely|definitely|sure|of course)\b", re.I)

_CONTINUATION = re.compile(
    r"^(this|that|these|those|it|they|here|there|he|she|plus|also|and )\b", re.I)


def _verb_follows(text: str, noun_stem: str, verb_stem: str, window: int = 3) -> bool:
    """Does the counted noun actually do what the member asked about?

    "294 Dupaco employees *used* Volunteer Time Off" answers "how many
    employees used VTO"; "627 employees *volunteered* 6,136 hours" counts
    something else.
    """
    ts = [stem(t) for t in tokens(norm(text))]
    for i, t in enumerate(ts):
        if t != noun_stem:
            continue
        if verb_stem in ts[i + 1: i + 1 + window]:
            return True
    return False


def _count_note(cands) -> Optional[str]:
    """Say which older counts are still on the site and why they lost."""
    if len(cands) < 2:
        return None
    newest = cands[0]
    older = [c for c in cands[1:] if _first_number(c[2].text) != _first_number(newest[2].text)]
    if not older:
        return None
    o = older[0]
    return (f"Older pages still say {_first_number(o[2].text)} "
            f"(published {o[0].isoformat()}); the figure above is from "
            f"{newest[0].isoformat()}.")


def _first_number(text: str) -> str:
    m = re.search(r"\d[\d,]*", text)
    return m.group(0) if m else "?"


def _claim_shape(text: str) -> frozenset:
    """A statement's wording with its numbers removed.

    Two press releases a year apart differ only in a figure; stripping figures
    makes them comparable so the newer one can supersede the older.
    """
    return frozenset(t for t in tokens(norm(text))
                     if not t[0].isdigit() and not t.startswith("$") and len(t) > 2)


def _same_claim(hits, pivot, threshold: float = 0.6) -> list:
    base = _claim_shape(pivot.text)
    if not base:
        return [h for h in hits if h[1] is pivot]
    out = []
    for h in hits:
        shape = _claim_shape(h[1].text)
        if not shape:
            continue
        j = len(base & shape) / len(base | shape)
        if j >= threshold:
            out.append(h)
    return out or [h for h in hits if h[1] is pivot]


def _number_near(text: str, keys: list[str], window: int = 70) -> bool:
    low = norm(text)
    spots = [m.start() for m in re.finditer(r"\d", low)]
    if not spots:
        return False
    for k in set(keys):
        for m in re.finditer(re.escape(k), low):
            if any(abs(s - m.start()) <= window for s in spots):
                return True
    return False


def _key_bonus(st, keys: list[str]) -> float:
    return 0.9 * sum(1 for k in set(keys) if k in st.toks)


def _page_prior(occ, keys: list[str]) -> float:
    """Which page the sentence sits on is evidence in itself.

    "Who is eligible to become a member?" is answered on the page whose own
    title and URL are about joining, even when the sentence that carries the
    answer never repeats the member's words.
    """
    hay = set(stem(t) for t in tokens(norm(occ.title + " " + occ.url.replace("/", " ")
                                           .replace("-", " "))))
    return 1.1 * sum(1 for k in set(keys) if k in hay) ** 0.5


def _counts_noun(text: str, targets: list[str], gap_words: int = 2) -> bool:
    """Is there a figure immediately in front of the noun being counted?

    "22 branch office locations" counts branches. "60 days to visit one of our
    branches" mentions a number and branches in one breath but counts days.
    """
    toks = tokens(norm(text))
    stems_ = [stem(t) for t in toks]
    for i, t in enumerate(toks):
        if not (t[0].isdigit() or t.startswith("$")):
            continue
        for j in range(i + 1, min(len(toks), i + 2 + gap_words)):
            if stems_[j] in targets:
                return True
    return False


def _merge_metrics(rows: list[Obs]) -> list[str]:
    """One line per row of the rate sheet, carrying every column it published.

    A table row says "2.23% dividend rate" and "2.25% APY" about one product;
    splitting them into two answers invites the reader to think the site
    disagrees with itself.
    """
    order = {"dividend_rate": 0, "apy": 1, "rate": 0, "apr": 1, "points": 2,
             "payment": 3, "min_open": 4, "min_earn": 5}
    groups: dict[tuple, list[Obs]] = {}
    for o in rows:
        k = (phrasing.product_phrase(o), o.term_months, o.balance, o.amount,
             o.model_years, frozenset(o.flags))
        groups.setdefault(k, []).append(o)
    lines = []
    for k, members in groups.items():
        members = sorted(members, key=lambda o: order.get(o.metric, 9))
        seen, parts = set(), []
        for o in members:
            tag = (o.metric, o.value_text)
            if tag in seen:
                continue
            seen.add(tag)
            word = phrasing.METRIC_WORD.get(o.metric, o.metric)
            parts.append(f"{o.value_text} {word}".strip())
        head = " - ".join(x for x in (k[0], phrasing.tier_phrase(members[0])) if x)
        body = " / ".join(parts)
        lines.append(f"{head}: {body}" if head else body)
    return lines


def _is_fee_page(doc) -> bool:
    return bool(re.search(r"fee.schedule|fees", norm(doc.path))) or "fee schedule" in norm(doc.title)


def _mentions_term(o: Obs, months: int) -> bool:
    hay = o.match_text
    forms = [f"{months} month", f"{months}-month", f"{months}month"]
    if months % 12 == 0:
        y = months // 12
        forms += [f"{y} year", f"{y}-year", f"{y}year"]
    return any(f in hay for f in forms)


def _matches(o: Obs, target: float) -> bool:
    for v in (o.lo, o.hi):
        if v is not None and abs(v - target) < 1e-6:
            return True
    return False


def _topic(o: Obs) -> str:
    bits = []
    if o.term_months:
        bits.append(phrasing._term_words(o.term_months))
    name = phrasing.product_phrase(o)
    if name:
        bits.append(name)
    fam = FAMILY_LABEL.get(o.family or "", "")
    if fam and norm(fam) not in norm(" ".join(bits)):
        bits.append(fam)
    tier = phrasing.tier_phrase(o)
    tier = re.sub(r"^\S+-(month|year),?\s*", "", tier)
    if tier:
        bits.append(f"({tier})")
    bits.append({"apy": "APY", "apr": "APR", "dividend_rate": "dividend rate",
                 "rate": "rate", "fee": "fee"}.get(o.metric, o.metric))
    return " ".join(b for b in bits if b)


def _why(winner: Obs, loser: Optional[Obs]) -> str:
    if loser is None:
        return "it is the current published value"
    w, l = winner.prov, loser.prov
    gone = loser.extras.get("expired_on")
    if gone:
        return (f"the competing value comes from an offer that ended "
                f"{gone.isoformat()} on a page that is still published")
    if w.effective_on and l.effective_on and w.effective_on != l.effective_on:
        return (f"the {w.page_kind.replace('_', ' ')} page is effective "
                f"{w.effective_on.strftime('%m/%d/%Y')} and supersedes the "
                f"{l.effective_on.strftime('%m/%d/%Y')} generation still shown on "
                f"{_short(l.url)}")
    if TIER.get(w.page_kind, 2) != TIER.get(l.page_kind, 2):
        return (f"the {w.page_kind.replace('_', ' ')} page is the maintained source; "
                f"{_short(l.url)} is a {l.page_kind.replace('_', ' ')} page that was never updated")
    if w.asserted_on != l.asserted_on:
        return f"it was published {w.asserted_on.isoformat()}, later than {l.asserted_on.isoformat()}"
    return "it comes from the more authoritative page"


def _short(url: str) -> str:
    return url.split("dupaco.com", 1)[-1] or url


_IMPORTANT = ("certificate", "auto", "mortgage", "savings", "checking", "credit card",
              "money market", "personal loan", "home equity")


def _conflict_weight(c: Conflict) -> float:
    t = norm(c.topic)
    w = 1.0 + sum(1.0 for k in _IMPORTANT if k in t)
    return w + 0.3 * len(c.values)
