"""Turn a crawled markdown page into labelled blocks.

The crawler flattens a CMS page, so a rate sheet arrives as:

    Savings Accounts            <- tab group label
    Share Savings               <- tab label
    | Account Type | ... |      <- the table
    | --- | ... |
    | Share Savings | $25,000+ | 0.20% | ... |
    Rates are effective as of 06/15/2026 ...   <- the fact's own validity date
    1Savings Rates: Dividends and ...          <- footnotes

The label lines above a table and the "effective as of" line below it are what
make a row interpretable, so the parser keeps them together as one Block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from .textutil import clean, norm, parse_date, slug

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
SEP_RE = re.compile(r"^\s*\|[\s|:\-]+\|\s*$")
EFFECTIVE_RE = re.compile(r"effective(?:\s+as\s+of)?\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})", re.I)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# Call-to-action lines sit between a product blurb and its rate table but say
# nothing about the product, so they must not become part of a fact's label.
_IS_CTA = re.compile(
    r"^(open|apply|see|view|learn|call|get|start|join|log ?in|compare|find|shop|"
    r"download|read|explore|contact|become|check|visit|schedule|talk|chat|go\b|"
    r"request|order|switch|refer|watch|listen|subscribe|sign)\b", re.I)


# --------------------------------------------------------------------------
# page kind + authority
# --------------------------------------------------------------------------

# Higher authority wins when two sources carry the same validity date. The
# ordering encodes a simple editorial truth: the rate sheet is maintained, a
# calculator or a press release is not.
AUTHORITY = {
    "rate_sheet": 100,   # /rates, /business-rates - the master tables
    "product": 80,       # /personal/..., /business/... product pages
    "support": 55,       # /contact, /about, locator, FAQ
    "landing": 40,       # campaign/promo landing pages
    "legal": 30,         # disclosures, fee schedules, policies
    "news": 20,          # /newsroom/, dated blog posts, press releases
    "blog": 18,          # /learn/blog
    "calculator": 8,     # calculator widgets that embed frozen rates
    "other": 25,
}

_DATED_PATH = re.compile(r"/(19|20)\d{2}/\d{1,2}/\d{1,2}/")


def classify(url: str, title: str, markdown: str) -> str:
    u = norm(url)
    path = u.split("dupaco.com", 1)[-1] if "dupaco.com" in u else u
    if re.search(r"/(business-)?rates/?$", path) or path.endswith("/rates"):
        return "rate_sheet"
    if "calculator" in path or "/calculators" in path:
        return "calculator"
    if "/newsroom" in path or _DATED_PATH.search(path) or "press-release" in path:
        return "news"
    if "/learn/blog" in path or "/learn/" in path or "-resources/" in path:
        return "blog"
    if re.search(r"(privacy|disclosure|terms|policy|agreement|legal|fee-schedule|truth-in-savings)", path):
        return "legal"
    if re.search(r"(-lp-|/lp/|campaign|promo|offer|earn-more|switch-and-earn|balancetransfer)", path):
        return "landing"
    if re.search(r"^/(personal|business)/", path) or "/savings-account" in path or "/loans" in path:
        return "product"
    if re.search(r"(contact|location|branch|atm|about|connect|faq|support|help|join)", path):
        return "support"
    return "other"


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------

@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]

    @property
    def ncol(self) -> int:
        return len(self.header)


@dataclass
class Block:
    kind: str                       # "table" | "text"
    labels: tuple[str, ...]         # heading stack + tab labels directly above
    text: str = ""                  # for kind == "text"
    table: Optional[Table] = None
    effective_on: Optional[date] = None
    trailer: str = ""               # the lines just after a table (footnotes)
    line: int = 0
    links: tuple[tuple[str, str], ...] = ()

    @property
    def label_path(self) -> str:
        return " > ".join(self.labels)


@dataclass
class Doc:
    url: str
    title: str
    kind: str
    authority: int
    fetched_at: Optional[date]
    snapshot_id: str
    taken_on: date
    snapshot_idx: int = 0
    blocks: list[Block] = field(default_factory=list)
    page_effective_on: Optional[date] = None
    published_on: Optional[date] = None
    links: list[tuple[str, str]] = field(default_factory=list)
    path: str = ""

    @property
    def doc_id(self) -> str:
        return slug(self.path or self.url)


_NOISE_HEADERS = (
    "col2", "col3", "creditor", "required signatures", "for office use only",
    "assets", "liabilities", "section 1", "section 2", "section 3",
    "no. shares", "name of bank", "name of creditor", "insurance company",
    "titled in name of", "serial number", "noi(", "noi (", "vin",
    "category | interest | principal | remaining", "monthly breakdown",
    "amount financed", "annual mileage", "privileges",
    "p&i", "principal", "remaining balance", "payment number", "amortization",
)


def _is_noise_table(t: Table) -> bool:
    joined = norm(" | ".join(t.header))
    if not joined or not joined.replace("|", "").strip():
        return True
    if any(k in joined for k in _NOISE_HEADERS):
        return True
    # a table whose cells are almost all empty carries no fact
    cells = [c for r in t.rows for c in r]
    if cells and sum(1 for c in cells if c.strip()) / len(cells) < 0.4:
        return True
    return False


def _strip_md(s: str) -> str:
    s = IMG_RE.sub(" ", s)
    s = LINK_RE.sub(r"\1", s)
    s = s.replace("\\*", "*").replace("**", "").replace("\\#", "#").replace("\\_", "_")
    s = re.sub(r"^\s*[-*\u2022]\s+", "", s)
    return clean(s)


def parse_page(url: str, title: str, markdown: str, snapshot_id: str,
               taken_on: date, fetched_at: Optional[date], snapshot_idx: int = 0) -> Doc:
    kind = classify(url, title, markdown)
    path = url.split("dupaco.com", 1)[-1] if "dupaco.com" in url else url
    doc = Doc(url=url, title=clean(title), kind=kind, authority=AUTHORITY.get(kind, 25),
              fetched_at=fetched_at, snapshot_id=snapshot_id, taken_on=taken_on,
              snapshot_idx=snapshot_idx, path=path or "/")

    lines = markdown.split("\n")
    heading_stack: list[tuple[int, str]] = []
    pending_labels: list[str] = []          # short lines seen since the last block
    i = 0
    first_eff: Optional[date] = None

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        m = HEADING_RE.match(line)
        if m:
            level, htext = len(m.group(1)), _strip_md(m.group(2))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            if htext:
                heading_stack.append((level, htext))
            pending_labels = []
            i += 1
            continue

        # ---- table -------------------------------------------------------
        if ROW_RE.match(line) and i + 1 < len(lines) and SEP_RE.match(lines[i + 1]):
            header = [_strip_md(c) for c in ROW_RE.match(line).group(1).split("|")]
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and ROW_RE.match(lines[j].strip()):
                rows.append([_strip_md(c) for c in ROW_RE.match(lines[j].strip()).group(1).split("|")])
                j += 1
            table = Table(header=header, rows=rows)
            trailer_lines = []
            k = j
            eff = None
            while k < len(lines) and len(trailer_lines) < 8:
                s = lines[k].strip()
                if ROW_RE.match(s):
                    break
                if HEADING_RE.match(s):
                    break
                if s:
                    trailer_lines.append(_strip_md(s))
                    if eff is None:
                        em = EFFECTIVE_RE.search(s)
                        if em:
                            eff = parse_date(em.group(1))
                k += 1
            if not _is_noise_table(table):
                labels = tuple([h for _, h in heading_stack] + pending_labels[-3:])
                doc.blocks.append(Block(kind="table", labels=labels, table=table,
                                        effective_on=eff, trailer=" ".join(trailer_lines), line=i))
            if eff and first_eff is None:
                first_eff = eff
            pending_labels = []
            i = j
            continue

        # ---- plain text ----------------------------------------------------
        if line:
            txt = _strip_md(line)
            blinks = tuple((clean(lm.group(1)), lm.group(2)) for lm in LINK_RE.finditer(raw))
            doc.links.extend(blinks)
            if txt:
                em = EFFECTIVE_RE.search(txt)
                if em and first_eff is None:
                    first_eff = parse_date(em.group(1))
                if (len(txt) <= 70 and not txt.endswith((".", "!", "?", ":"))
                        and len(txt.split()) <= 9 and not _IS_CTA.search(txt)):
                    # short, caption-like: likely a tab/section label for what follows
                    pending_labels.append(txt)
                    if len(pending_labels) > 4:
                        pending_labels.pop(0)
                labels = tuple([h for _, h in heading_stack] + pending_labels[-2:])
                doc.blocks.append(Block(kind="text", labels=labels, text=txt, line=i,
                                        links=blinks))
        i += 1

    doc.page_effective_on = first_eff
    doc.published_on = _published_on(lines)
    return doc


_DATE_LINE = re.compile(
    r"^((?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+(?:19|20)\d{2})$", re.I)


def _published_on(lines: list[str]) -> Optional[date]:
    """A press release or blog post is a *dated statement*.

    The crawler leaves the byline date on its own line near the top. Treating it
    as the fact's validity date is what lets "22 branch offices" (May 2026) beat
    "23 branch offices" (March 2026) when both are still on the site.
    """
    seen = 0
    for raw in lines[:40]:
        s = raw.strip()
        if not s:
            continue
        seen += 1
        if seen > 14:
            break
        m = _DATE_LINE.match(_strip_md(s))
        if m:
            return parse_date(m.group(1))
    return None
