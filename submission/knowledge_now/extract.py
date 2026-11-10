"""Extractors: blocks -> observations.

Rate tables are the authoritative, self-dating part of the site, so they get a
real parser. Everything else (hours, addresses, routing numbers, counts,
canonical links) is pulled out with narrow, auditable patterns, and whatever is
left stays available to the passage index as text.
"""
from __future__ import annotations

import re
from typing import Optional

from . import quals
from .docmodel import Block, Doc
from .facts import Obs, Provenance
from .ontology import family_of, key_for_header, metric_for_header
from .textutil import PCT_RE, clean, first_money, norm, to_float

RATE_METRICS = {"apy", "apr", "rate", "dividend_rate"}
MONEY_METRICS = {"min_open", "min_earn", "max_balance", "payment", "fee"}


# --------------------------------------------------------------------------
# values
# --------------------------------------------------------------------------

def _percents(cell: str) -> list[float]:
    """All percentages in a cell. '12.50% - 19.50%2' -> [12.5, 19.5]."""
    return [to_float(m.group(1), m.group(2)) for m in PCT_RE.finditer(clean(cell))]


def _parse_value(cell: str, metric: str) -> Optional[tuple[str, float, Optional[float], str]]:
    """-> (display, lo, hi, unit) or None when the cell holds no value."""
    c = clean(cell)
    if not c or norm(c) in {"n/a", "na", "-", "--", "not applicable", "none", "variable"}:
        return None
    if metric in RATE_METRICS:
        ps = _percents(c)
        if not ps:
            return None
        lo, hi = min(ps), max(ps)
        disp = f"{lo:.2f}%" if lo == hi else f"{lo:.2f}% - {hi:.2f}%"
        return (disp, lo, hi if hi != lo else None, "%")
    if metric in MONEY_METRICS:
        v = first_money(c)
        if v is None:
            ps = _percents(c)
            if ps and metric == "fee":
                return (f"{ps[0]:.2f}%", ps[0], None, "%")
            return None
        disp = f"${v:,.2f}" if v % 1 else f"${v:,.0f}"
        return (disp, v, None, "$")
    if metric == "points":
        m = re.search(r"\d+(?:\.\d+)?", c)
        return (m.group(0), float(m.group(0)), None, "") if m else None
    return None


# --------------------------------------------------------------------------
# rate / fee tables
# --------------------------------------------------------------------------

def _column_roles(header: list[str]) -> tuple[dict[int, str], dict[int, str], list[str]]:
    """-> (metric columns, key columns, header cells that read as product labels)."""
    metrics: dict[int, str] = {}
    keys: dict[int, str] = {}
    labelish: list[str] = []
    for i, h in enumerate(header):
        h = clean(h)
        if not h:
            continue
        k = key_for_header(h)
        if k:
            keys[i] = k
            continue
        m = metric_for_header(h)
        if m:
            metrics[i] = m
            continue
        if not _TIERISH.match(norm(h)):
            labelish.append(h)       # a header naming the product, e.g. "Platinum Visa"
    if metrics:
        # Unclaimed columns in a table that does have values are row labels.
        for i, h in enumerate(header):
            if i not in metrics and i not in keys:
                keys[i] = "label"
    return metrics, keys, labelish


def table_facts(block: Block, doc: Doc) -> list[Obs]:
    t = block.table
    if t is None or not t.rows:
        return []
    metrics, keys, labelish = _column_roles(t.header)
    if not metrics:
        return []

    base_labels = [l for l in block.labels if l] + labelish
    page_ctx = f"{doc.title} {doc.path.replace('/', ' ').replace('-', ' ')}"
    eff = block.effective_on or doc.page_effective_on
    out: list[Obs] = []

    for row in t.rows:
        if len(row) < 2 or not any(clean(c) for c in row):
            continue
        key_cells = [clean(row[i]) for i in sorted(keys) if i < len(row) and clean(row[i])]
        if not key_cells and not base_labels:
            continue
        key_text = " ".join(key_cells)
        ident = " - ".join([*base_labels[-2:], key_text]) if key_text else " - ".join(base_labels[-2:])
        qual_text = " ".join([*base_labels, key_text])

        term = quals.parse_term_months(key_text) or quals.parse_term_months(" ".join(base_labels))
        arm = quals.parse_arm(key_text) or quals.parse_arm(" ".join(base_labels))
        myears = quals.parse_model_years(key_text) or quals.parse_model_years(" ".join(base_labels))
        flags = quals.flags(qual_text) | quals.flags(page_ctx if "business" in norm(doc.path) else "")

        balance = amount = None
        for i, dim in keys.items():
            if i >= len(row):
                continue
            iv = quals.parse_money_interval(row[i])
            if iv is None:
                continue
            if dim == "balance":
                balance = iv
            elif dim == "loan_amount":
                amount = iv
            elif dim == "label" and amount is None and balance is None:
                # "30-Year Fixed Conventional $100,000 or less" - a loan-size tier
                amount = iv
        if balance is None and amount is None:
            iv = quals.parse_money_interval(key_text)
            if iv:
                balance = iv

        family = (family_of(key_text) or family_of(" ".join(base_labels))
                  or family_of(page_ctx))
        product_name = _product_name(key_cells, base_labels)
        match_text = norm(" ".join([ident, *base_labels, key_text, page_ctx, *t.header]))

        for i, metric in sorted(metrics.items()):
            if i >= len(row):
                continue
            parsed = _parse_value(row[i], metric)
            if parsed is None:
                continue
            disp, lo, hi, unit = parsed
            if metric in MONEY_METRICS and metric != "fee" and metric != "payment":
                # minimums are qualifiers, but also answerable facts
                pass
            if metric == "fee" and (lo > 5000 or _CALC_LABEL.match(product_name.strip())):
                continue          # a calculator result, not a published price
            quote = f"{ident}: {clean(row[i])}".strip(" :")
            out.append(Obs(
                domain="fee" if metric == "fee" else "rate",
                metric=metric, family=family, label=ident or key_text,
                match_text=match_text, value_text=disp, lo=lo, hi=hi, unit=unit,
                term_months=term, balance=balance, amount=amount, model_years=myears,
                arm=arm, flags=frozenset(flags),
                prov=_prov(doc, eff, quote),
                extras={"table_header": " | ".join(t.header), "row": key_text,
                        "labels": tuple(base_labels), "product_name": product_name},
            ))
    return out


# Row keys that describe the tier or the column rather than the product. When a
# row is keyed only by one of these, the product name comes from the heading.
# Labels that mark a calculator's output rather than a product's price.
_CALC_LABEL = re.compile(r"^(p&i|p ?& ?i|principal|interest|total|monthly payment|payment|"
                         r"balance|savings|insurance|taxes|down payment|loan and trade)\b", re.I)

_TIERISH = re.compile(r"^[\s$\d.,%+-]*$|model year|^\d+\s*(month|months|year|years|mo)\b|"
                      r"^(standard|variable|fixed|purchase|refinance|rate|rates|apr|apy|"
                      r"type|types|loan type|program|programs|product|products|category|"
                      r"item|description|account type|balance|balance tier|term|col\d+)$", re.I)


def _product_name(key_cells: list[str], labels: list[str]) -> str:
    """The most specific thing this row is *about*.

    A savings table names the product in its first column; a loan table names
    only the tier ("2024-27 Model Year"), so the product comes from the heading
    above it. Getting this right is what keeps Bright Savings from being
    compared against Share Savings.
    """
    for c in key_cells:
        if c and not _TIERISH.match(norm(c)) and any(ch.isalpha() for ch in c):
            return c
    for l in reversed(labels):
        if l and any(ch.isalpha() for ch in l):
            return l
    return key_cells[0] if key_cells else ""


def _prov(doc: Doc, eff, quote: str) -> Provenance:
    return Provenance(url=doc.url, title=doc.title, page_kind=doc.kind, authority=doc.authority,
                      snapshot_id=doc.snapshot_id, snapshot_idx=doc.snapshot_idx,
                      taken_on=doc.taken_on, fetched_at=doc.fetched_at,
                      effective_on=eff, quote=quote[:300])


# --------------------------------------------------------------------------
# rates stated in prose
# --------------------------------------------------------------------------

_PROSE_RATE = re.compile(
    r"(?P<subject>[A-Za-z0-9][A-Za-z0-9''&./\- ]{2,70}?)\s*"
    r"(?:rate|apy|apr)?\s*[:=]\s*(?P<val>\d{1,2}(?:\.\d+)?\s*%)", re.I)

_PROSE_APY = re.compile(
    r"(?P<subject>[A-Za-z0-9][A-Za-z0-9''&./\- ]{2,70}?)\s+(?:pays|earns|offers|is|at)\s+"
    r"(?:a\s+)?(?P<val>\d{1,2}(?:\.\d+)?\s*%)\s*(?P<metric>apy|apr)\b", re.I)

# "...APY 2.95%" with no punctuation between the product and its rate.
_PROSE_BARE = re.compile(
    r"(?P<subject>[A-Za-z0-9][A-Za-z0-9'':&./\- ]{2,70}?)\s+"
    r"(?P<metric>apy|apr|rate|dividend rate)\s+(?P<val>\d{1,2}(?:\.\d+)?\s*%)", re.I)

# "Lost debit card replacement fee: $2.50"
_PROSE_FEE = re.compile(
    r"(?P<subject>[A-Za-z][A-Za-z0-9'':&./\- ]{2,70}?)\s*"
    r"(?:fee|charge|cost|price)\s*(?:is|of|:|=)\s*\$?\s*(?P<val>\d{1,6}(?:\.\d{2})?)", re.I)

# A promotion that names its own end date stops being true on that date, even
# though the page stays up. This is the single most common way a crawled site
# contradicts itself.
_OFFER_END = re.compile(
    r"(?:offer|promotion|promo|special|rate)?\s*"
    r"(?:end(?:s|ed)|expire[sd]?|valid through|available through|through)\s*"
    r"(?:on\s*)?(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})", re.I)

_EFFECTIVE_INLINE = re.compile(
    r"effective\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+is\s+a\s+variable\s+rate\s+of\s+"
    r"(\d{1,2}(?:\.\d+)?%)\s*-\s*(\d{1,2}(?:\.\d+)?%)", re.I)


def prose_facts(block: Block, doc: Doc) -> list[Obs]:
    """Facts written as sentences rather than tables.

    Kept narrow on purpose: prose is where stale marketing copy lives, so these
    observations carry the page's authority and rarely outrank a rate table.
    """
    txt = block.text
    if not txt or len(txt) > 400:
        return []
    out: list[Obs] = []
    low = norm(txt)
    expired_on = _expiry(txt, doc)
    if not re.search(r"\d\s*%", low):
        return _prose_fees(block, doc, expired_on)
    page_ctx = f"{doc.title} {doc.path.replace('/', ' ').replace('-', ' ')}"
    labels = [l for l in block.labels if l]

    for rx, default_metric in ((_PROSE_RATE, None), (_PROSE_APY, None), (_PROSE_BARE, None)):
        for m in rx.finditer(txt):
            subject = clean(m.group("subject"))
            if len(subject) < 3 or norm(subject) in {"rate", "rates", "the rate"}:
                continue
            metric = (m.groupdict().get("metric") or "").lower()
            if not metric:
                metric = "apy" if "apy" in norm(m.group(0)) else (
                    "apr" if "apr" in norm(m.group(0)) else "rate")
            val = _parse_value(m.group("val"), metric if metric in RATE_METRICS else "rate")
            if not val:
                continue
            disp, lo, hi, unit = val
            term = quals.parse_term_months(subject)
            family = family_of(subject) or family_of(" ".join(labels)) or family_of(page_ctx)
            if family is None and term is None:
                continue
            out.append(Obs(
                domain="rate", metric=metric if metric in RATE_METRICS else "rate",
                family=family, label=subject,
                match_text=norm(" ".join([subject, *labels, page_ctx])),
                value_text=disp, lo=lo, hi=hi, unit=unit, term_months=term,
                arm=quals.parse_arm(subject), flags=frozenset(quals.flags(subject + " " + " ".join(labels))),
                prov=_prov(doc, block.effective_on or doc.page_effective_on, clean(txt)[:300]),
                extras={"prose": True, "expired_on": expired_on},
            ))
    for m in _EFFECTIVE_INLINE.finditer(txt):
        from .textutil import parse_date
        lo = float(m.group(2).rstrip("%"))
        hi = float(m.group(3).rstrip("%"))
        family = family_of(" ".join(labels) + " " + page_ctx) or "credit_card"
        out.append(Obs(
            domain="rate", metric="apr", family=family,
            label=" - ".join(labels[-2:]) or doc.title,
            match_text=norm(" ".join([*labels, page_ctx, "standard variable rate"])),
            value_text=f"{lo:.2f}% - {hi:.2f}%", lo=lo, hi=hi, unit="%",
            flags=frozenset(quals.flags(" ".join(labels))),
            prov=_prov(doc, parse_date(m.group(1)), clean(txt)[:300]),
            extras={"prose": True},
        ))
    return out


def _expiry(text: str, doc: Doc):
    """The date this offer says it stops being valid, if it already has."""
    m = _OFFER_END.search(text)
    if not m:
        return None
    from .textutil import parse_date
    end = parse_date(m.group(1))
    if end and end < doc.taken_on:
        return end
    return None


# A fee schedule is a list, not a table: "Rush fee for debit card $50.00*",
# "Money orders $5.00* each". The label is everything before the amount.
_FEE_LINE = re.compile(
    r"^(?P<label>[A-Za-z][^$]{2,90}?)\s*\$\s*(?P<amt>\d{1,3}(?:,\d{3})*(?:\.\d{2})?)"
    r"\s*\*?\s*(?P<unit>per item|each|per month|per roll|per day|per year|per check|"
    r"per transaction|per page|per copy)?", re.I)
_FEE_PAGE = re.compile(r"fee schedule|fees? and charges|service charges|miscellaneous fees", re.I)


def fee_schedule_facts(doc: Doc, is_fee_page: bool) -> list[Obs]:
    """Every "<service> $<amount>" line on a fee schedule.

    These lines carry no table structure, so the only thing that makes them
    interpretable is the section heading above them and the schedule's own
    effective date - both of which the block already carries.
    """
    out: list[Obs] = []
    for b in doc.blocks:
        if b.kind != "text":
            continue
        heading = " ".join(b.labels[-2:])
        if not (is_fee_page or _FEE_PAGE.search(heading) or _FEE_PAGE.search(b.text)):
            continue
        line = clean(b.text)
        if len(line) > 160 or line.count("$") > 2:
            continue
        m = _FEE_LINE.match(line)
        if not m:
            continue
        label = clean(m.group("label")).strip(" -:*")
        if len(label) < 4 or label.lower().startswith(("the following", "please", "for fees")):
            continue
        try:
            v = float(m.group("amt").replace(",", ""))
        except ValueError:
            continue
        unit = (m.group("unit") or "").lower()
        disp = f"${v:,.2f}" if v % 1 else f"${v:,.0f}"
        if unit:
            disp = f"{disp} {unit}"
        page_ctx = f"{doc.title} {doc.path.replace('/', ' ').replace('-', ' ')}"
        out.append(Obs(
            domain="fee", metric="fee", family=family_of(label),
            label=label,
            match_text=norm(" ".join([label, heading, "fee charge cost price how much",
                                      page_ctx])),
            value_text=disp, lo=v, unit="$",
            flags=frozenset(quals.flags(label + " " + heading)),
            prov=_prov(doc, doc.page_effective_on, line[:200]),
            extras={"fee_schedule": True, "product_name": label, "unit": unit},
        ))
    return out


def _prose_fees(block: Block, doc: Doc, expired_on) -> list[Obs]:
    """Fees stated in a sentence: "Lost debit card replacement fee: $2.50"."""
    txt = block.text
    if "$" not in txt and not re.search(r"\bfee\b|\bcharge\b|\bcost\b", norm(txt)):
        return []
    out: list[Obs] = []
    labels = [l for l in block.labels if l]
    page_ctx = f"{doc.title} {doc.path.replace('/', ' ').replace('-', ' ')}"
    for m in _PROSE_FEE.finditer(txt):
        subject = clean(m.group("subject")).strip(" -:")
        if len(subject) < 3:
            continue
        try:
            v = float(m.group("val"))
        except ValueError:
            continue
        disp = f"${v:,.2f}" if v % 1 else f"${v:,.0f}"
        out.append(Obs(
            domain="fee", metric="fee", family=family_of(subject) or family_of(page_ctx),
            label=f"{subject} fee",
            match_text=norm(" ".join([subject, "fee charge cost", *labels, page_ctx])),
            value_text=disp, lo=v, unit="$",
            flags=frozenset(quals.flags(subject)),
            prov=_prov(doc, block.effective_on or doc.page_effective_on, clean(txt)[:300]),
            extras={"prose": True, "expired_on": expired_on},
        ))
    return out
