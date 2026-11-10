"""Extractors for the parts of a credit-union site that are not rate tables:
branch directory, lobby/drive-up hours, canonical links and identifiers.
"""
from __future__ import annotations

import re

from .docmodel import Doc, classify
from .facts import Obs, Provenance
from .textutil import clean, norm

ADDRESS_RE = re.compile(
    r"^(?P<street>\d{2,6}[^,]{3,60}?),\s*(?P<city>[A-Z][A-Za-z.' ]{2,28}),\s*"
    r"(?P<state>IA|IL|WI|Iowa|Illinois|Wisconsin),?\s*(?P<zip>\d{5})(?:-\d{4})?\.?$")

DAY = r"(?:mon|tues?|wed(?:nes)?|thur?s?|fri|sat(?:ur)?|sun)[a-z]*"
HOURS_LINE = re.compile(
    rf"^(?P<kind>lobby|drive[- ]?up|drive[- ]?thru|atm)?\s*"
    rf"(?P<days>{DAY}(?:\s*(?:-|through|to|and|,)\s*{DAY})*)\s*:?\s*"
    rf"(?P<hours>(?:\d{{1,2}}(?::\d{{2}})?\s*(?:a\.?m\.?|p\.?m\.?)|noon|midnight|closed|"
    rf"atm only|by appointment)[^|]*)$", re.I)
KIND_LINE = re.compile(r"^(lobby|drive[- ]?up|drive[- ]?thru)\b\s*:?\s*$", re.I)

ROUTING_RE = re.compile(r"routing\s*(?:number|#)?\s*[:\-]?\s*\(?(\d{9})\)?", re.I)
PHONE_RE = re.compile(r"\b(?:1[-. ])?\(?(\d{3})\)?[-. ](\d{3})[-. ](\d{4})\b")

_BRANCH_HINT = re.compile(r"\b([A-Z][A-Za-z.']+(?:\s[A-Z][A-Za-z.']+){0,2})\s+(?:Dupaco\s+)?branch\b")
_LOCATED_AT = re.compile(r"\bbranch\s+is\s+located\s+at\s+([^.;]{5,70})", re.I)


def _prov(doc: Doc, eff, quote: str) -> Provenance:
    return Provenance(url=doc.url, title=doc.title, page_kind=doc.kind, authority=doc.authority,
                      snapshot_id=doc.snapshot_id, snapshot_idx=doc.snapshot_idx,
                      taken_on=doc.taken_on, fetched_at=doc.fetched_at,
                      effective_on=eff, quote=quote[:300])


def _date_for(doc: Doc):
    """News and blog pages date themselves with a byline; product pages don't."""
    return doc.published_on if doc.kind in ("news", "blog") else doc.page_effective_on


# --------------------------------------------------------------------------
# branch directory
# --------------------------------------------------------------------------

def branch_facts(doc: Doc) -> list[Obs]:
    out: list[Obs] = []
    blocks = doc.blocks
    for i, b in enumerate(blocks):
        if b.kind != "text":
            continue
        m = ADDRESS_RE.match(b.text.strip())
        if not m:
            continue
        # what kind of location is this? the line above says so
        kind = ""
        name = ""
        for j in range(max(0, i - 3), i):
            t = norm(blocks[j].text)
            if "surcharge-free atm" in t or t.endswith(" atm"):
                kind = "atm"
            elif "branch" in t:
                kind = "branch"
            elif "video teller" in t:
                kind = kind or "video_teller"
            if blocks[j].text and blocks[j].text[0].isupper() and len(blocks[j].text) < 60:
                name = blocks[j].text
        features = ""
        if i + 1 < len(blocks) and "·" in blocks[i + 1].text:
            features = blocks[i + 1].text
        city = clean(m.group("city"))
        addr = clean(b.text).rstrip(".")
        label = f"{city} {kind or 'location'}".strip()
        out.append(Obs(
            domain="branch", metric="address", family=None, label=label,
            match_text=norm(" ".join([city, m.group("street"), name, kind, features,
                                      doc.title, doc.path.replace('/', ' ')])),
            value_text=addr, prov=_prov(doc, _date_for(doc), addr),
            extras={"city": city, "street": clean(m.group("street")), "state": m.group("state"),
                    "zip": m.group("zip"), "loc_kind": kind or "location",
                    "features": features, "name": name},
        ))
    return out


# --------------------------------------------------------------------------
# hours
# --------------------------------------------------------------------------

def hours_facts(doc: Doc) -> list[Obs]:
    out: list[Obs] = []
    blocks = doc.blocks
    i = 0
    context = ""
    while i < len(blocks):
        b = blocks[i]
        if b.kind != "text":
            i += 1
            continue
        t = clean(b.text)
        if "branch" in norm(t) and len(t) < 300:
            context = t
        kind = None
        km = KIND_LINE.match(t)
        if km:
            kind = norm(km.group(1)).replace(" ", "-")
            i += 1
        else:
            hm = HOURS_LINE.match(t)
            if hm and hm.group("kind"):
                kind = norm(hm.group("kind")).replace(" ", "-")
            elif hm:
                kind = "lobby"
            else:
                i += 1
                continue
        rows: list[tuple[str, str]] = []
        while i < len(blocks) and len(rows) < 8:
            tt = clean(blocks[i].text) if blocks[i].kind == "text" else ""
            hm = HOURS_LINE.match(tt)
            if not hm:
                break
            if hm.group("kind") and rows:
                break
            rows.append((clean(hm.group("days")), clean(hm.group("hours"))))
            i += 1
        if not rows:
            i += 1
            continue
        kind = "drive-up" if kind and kind.startswith("drive") else (kind or "lobby")
        sched = "; ".join(f"{d}: {h}" for d, h in rows)
        place = _place(context, doc)
        out.append(Obs(
            domain="hours", metric=kind, family=None,
            label=f"{place} {kind} hours".strip(),
            match_text=norm(" ".join([place, kind, "hours lobby branch office open close",
                                      doc.title, doc.path.replace('/', ' '), context[:120]])),
            value_text=sched, prov=_prov(doc, _date_for(doc), sched),
            extras={"rows": rows, "place": place, "hours_kind": kind},
        ))
    return out


_INLINE_HOURS = re.compile(
    rf"(?P<days>{DAY}[a-z]*\s*(?:-|through|to|and)\s*{DAY}[a-z]*)\s*,?\s*"
    rf"(?P<hours>\d{{1,2}}(?::\d{{2}})?\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:-|to)\s*"
    rf"\d{{1,2}}(?::\d{{2}})?\s*(?:a\.?m\.?|p\.?m\.?|noon))", re.I)


def inline_hours_facts(doc: Doc) -> list[Obs]:
    """Hours written into a sentence rather than listed.

    The contact page states when you can reach Dupaco inside a paragraph; that
    is the answer a member wants when they ask "when are you open?", and the
    page that says it is the one worth citing.
    """
    out: list[Obs] = []
    for b in doc.blocks:
        if b.kind != "text" or len(b.text) > 400:
            continue
        low = norm(b.text)
        if not re.search(r"\bhours?\b|\bopen\b|\breply\b|\breach\b|\bavailable\b|\blobby\b", low):
            continue
        for m in _INLINE_HOURS.finditer(b.text):
            kind = ("lobby" if "lobby" in low else
                    "drive-up" if re.search(r"drive[- ]?up", low) else "contact")
            sched = f"{clean(m.group('days'))}: {clean(m.group('hours'))}"
            out.append(Obs(
                domain="hours", metric=kind, family=None,
                label=f"{kind} hours",
                match_text=norm(" ".join([kind, "hours open contact reach branch office",
                                          doc.title, doc.path.replace('/', ' ')])),
                value_text=sched, prov=_prov(doc, _date_for(doc), clean(b.text)[:200]),
                extras={"rows": [(clean(m.group("days")), clean(m.group("hours")))],
                        "place": "", "hours_kind": kind},
            ))
    return out
def _place(context: str, doc: Doc) -> str:
    m = _LOCATED_AT.search(context)
    if m:
        pm = _BRANCH_HINT.search(context)
        if pm:
            return clean(pm.group(1))
    pm = _BRANCH_HINT.search(context)
    if pm and norm(pm.group(1)) not in {"dupaco", "the", "a", "this", "our"}:
        return clean(pm.group(1))
    return ""


# --------------------------------------------------------------------------
# identifiers and canonical links
# --------------------------------------------------------------------------

def identifier_facts(doc: Doc) -> list[Obs]:
    out: list[Obs] = []
    for b in doc.blocks:
        if b.kind != "text" or len(b.text) > 500:
            continue
        for m in ROUTING_RE.finditer(b.text):
            out.append(Obs(
                domain="entity", metric="routing_number", family=None, label="routing number",
                match_text=norm("routing number aba transit dupaco " + doc.title),
                value_text=m.group(1), prov=_prov(doc, _date_for(doc), clean(b.text)[:200]),
            ))
        if re.search(r"\bphone\b|\bcall\b", norm(b.text)):
            for m in PHONE_RE.finditer(b.text):
                num = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                out.append(Obs(
                    domain="entity", metric="phone", family=None, label="phone number",
                    match_text=norm("phone number call contact dupaco " + doc.title + " " + b.text[:80]),
                    value_text=num, prov=_prov(doc, _date_for(doc), clean(b.text)[:200]),
                ))
    return out


_LINK_PURPOSE = (
    ("certificate rates", ("certificate", "cd")),
    ("savings rates", ("savings",)),
    ("loan rates", ("loan", "rates")),
    ("join online", ("join", "become a member", "open an account")),
    ("branch locator", ("branch", "atm", "location", "locator", "find")),
    ("online banking", ("shine", "online banking", "log in")),
)


def link_facts(doc: Doc) -> list[Obs]:
    """Canonical URLs, so "what link should I use for X" is answered with the
    link the current site actually points at rather than one from an old post."""
    out: list[Obs] = []
    seen: set[tuple[str, str]] = set()
    pairs = [(t, h, b.labels) for b in doc.blocks for (t, h) in b.links]
    for text, href, labels in pairs:
        if not href.startswith("http") or "dupaco" not in href:
            continue
        if any(x in href for x in ("hubfs", ".jpg", ".png", ".jpeg", ".svg", ".pdf", "tel:", "mailto:")):
            continue
        anchor = clean(text)
        if not anchor or len(anchor) > 80:
            continue
        key = (norm(anchor), href)
        if key in seen:
            continue
        seen.add(key)
        out.append(Obs(
            domain="link", metric="url", family=None, label=anchor,
            match_text=norm(" ".join([anchor, href.replace("/", " ").replace("-", " "),
                                      " ".join(labels[-3:]), doc.title,
                                      "link url page where"])),
            value_text=href, prov=_prov(doc, _date_for(doc), f"{anchor} -> {href}"),
            extras={"anchor": anchor, "target_kind": classify(href, anchor, "")},
        ))
    return out
