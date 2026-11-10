"""Question -> QuerySpec.

The parser is deliberately literal. Every qualifier it finds becomes a *hard*
constraint, and every content word it cannot account for becomes a requirement
the candidate facts must satisfy. That is what produces honest refusals: a
question about a "7/1 ARM" or a "Mullan Avenue branch" leaves a requirement
nothing in the store can meet, so the system declines instead of answering with
the nearest-looking number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from . import quals
from .ontology import GENERIC, SYNONYMS, families_in, is_price_question, metric_for_question
from .textutil import STOPWORDS, content_tokens, norm, parse_date, parse_month_year, tokens

SUPERLATIVE_MAX = re.compile(
    r"\b(highest|best|top|most|maximum|max|greatest|largest|biggest)\b")
SUPERLATIVE_MIN = re.compile(r"\b(lowest|cheapest|smallest|minimum|least)\b")

LIST_CUES = re.compile(
    r"\b(what|which)\b.{0,30}\b(do you (?:offer|have)|are (?:your|the)|cards|accounts|"
    r"products|options|types|terms|programs)\b|\ball\b|\beach\b|\blist\b|"
    r"\bwhat are the\b|\brates for\b|\boptions\b|\bbreakdown\b|\bcompare\b")

PLURAL_PRODUCT = re.compile(
    r"\b(cards|certificates|accounts|loans|products|options|types|terms|programs|branches)\b")

REFUSAL_HINT = re.compile(r"\bdo (?:you|any)\b|\bis there\b|\bare there\b|\bany\b")

# Deliberately narrow: "when does the Hy-Vee branch close?" is a question about
# a closure announcement, not about opening hours.
HOURS_CUE = re.compile(r"\bhours?\b|\bwhat time\b|\bopen on\b|\bopening times?\b|"
                       r"\bwhen (?:do|does|are) (?:you|your branches)\b")
ADDRESS_CUE = re.compile(r"\baddress\b|\blocated\b|\blocation\b|\bwhere is\b|\bdirections?\b|\bzip\b")
PHONE_CUE = re.compile(r"\bphone\b|\bnumber to call\b|\bcall\b|\bcontact\b|\btelephone\b")
URL_CUE = re.compile(
    r"\burl\b|\blink\b|\bwebsite\b|\bweb ?page\b|"
    r"\bwhere (?:can|do|should) i (?:see|find|go|view|sign|apply|join)\b|"
    r"\b(?:join|sign up|enroll|apply|register|open an account)\b[^.?]{0,30}\bonline\b|"
    r"\bhow (?:do|can) i (?:join|sign up|enroll|register)\b")
# "if I keep $500 in the account" states a balance, not a published tier.
SELF_AMOUNT = re.compile(
    r"\b(?:i (?:have|keep|hold|deposit|save|put)|with|if i|balance of|holding|"
    r"keeping|carrying|my balance is)\b[^?]{0,24}\$?\s*\d")

WANTS_NUMBER = re.compile(
    r"\bhow many\b|\bhow much\b|\bwhat date\b|\bwhen (?:do|does|did|is|are|was|will)\b|"
    r"\bstill open\b|\bnumber of\b|\bwhat year\b|\bhow long\b|\bwhat time\b|"
    r"\bclos(?:e|es|ed|ing)\b")

YES_NO = re.compile(r"\s*(can|do|does|did|is|are|was|were|will|would|may|should|could|have|has)\b")

DEFINITIONAL = re.compile(
    r"\bwhat (?:is|are|does)\b|\btell me about\b|\bexplain\b|\bhow does\b|\bwhat'?s\b")

COUNT_CUE = re.compile(r"\bhow many\b|\bnumber of\b|\bhow much\b.{0,12}\btotal\b")

BRANCH_WORD = re.compile(r"\bbranch(?:es)?\b|\boffice\b|\blobby\b|\bdrive[- ]?up\b|\bdrive[- ]?thru\b|\bATM\b", re.I)

AS_OF_RE = re.compile(
    r"\b(?:as of|in|on|during|back in|by)\s+((?:january|february|march|april|may|june|july|"
    r"august|september|october|november|december)\.?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})", re.I)


@dataclass
class Spec:
    raw: str
    nq: str
    metric: Optional[str] = None
    families: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)     # product phrases found in the store
    unknown: list[str] = field(default_factory=list)     # content words with no corpus support
    term_months: Optional[int] = None
    arm: Optional[str] = None
    model_year: Optional[float] = None
    balance: Optional[float] = None
    balance_iv: Optional[tuple[float, float]] = None
    amount: Optional[float] = None
    flags: set[str] = field(default_factory=set)
    want_max: bool = False
    want_min: bool = False
    want_list: bool = False
    verify_value: Optional[float] = None    # "do any tiers pay 0.35%?"
    as_of: Optional[date] = None
    intents: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    price_question: bool = False
    wants_number: bool = False
    definitional: bool = False
    yes_no: bool = False
    count_noun: Optional[str] = None
    count_verb: Optional[str] = None
    proper_nouns: list[str] = field(default_factory=list)


def _phrase_ngrams(toks: list[str], n_max: int = 4):
    """Longest first, and never single words: a lone word is already a
    requirement, while a phrase the site itself uses is strong evidence of
    *which* product the member means."""
    for n in range(min(n_max, len(toks)), 1, -1):
        for i in range(len(toks) - n + 1):
            yield i, n, " ".join(toks[i:i + n])


def parse(question: str, as_of: Optional[str], phrase_index) -> Spec:
    nq = norm(question)
    spec = Spec(raw=question, nq=nq)

    spec.as_of = parse_date(as_of) if as_of else None
    if spec.as_of is None:
        m = AS_OF_RE.search(question)
        if m:
            spec.as_of = parse_date(m.group(1)) or parse_month_year(m.group(1))

    spec.metric = metric_for_question(nq)
    spec.price_question = is_price_question(nq)
    spec.proper_nouns = _proper_nouns(question)
    spec.wants_number = bool(WANTS_NUMBER.search(nq))
    spec.definitional = bool(DEFINITIONAL.search(nq))
    spec.yes_no = bool(YES_NO.match(nq))
    spec.count_noun, spec.count_verb = _count_noun(nq)
    spec.term_months = quals.parse_term_months(nq)
    spec.arm = quals.parse_arm(nq)
    spec.flags = quals.flags(nq)
    spec.want_max = bool(SUPERLATIVE_MAX.search(nq))
    spec.want_min = bool(SUPERLATIVE_MIN.search(nq))

    iv = quals.parse_money_interval(nq)
    if iv:
        spec.balance_iv = iv
        spec.balance = (iv[0] + min(iv[1], iv[0] * 10 + 1)) / 2 if iv[1] != quals.INF else iv[0]
    if spec.balance_iv is None and SELF_AMOUNT.search(nq):
        amt = quals.parse_single_amount(nq)
        if amt is not None and amt > 0:
            spec.balance = amt
            spec.balance_iv = (amt, amt)
    my = quals.question_model_year(nq)
    if my and ("model" in nq or "vehicle" in nq or "car" in nq or "auto" in nq or "truck" in nq):
        spec.model_year = my

    # "Do any savings tiers pay 0.35% APY?" - the member proposes a value.
    if REFUSAL_HINT.search(nq):
        from .textutil import first_percent
        v = first_percent(nq)
        if v is not None:
            spec.verify_value = v

    # intents (ordered; the engine tries them in order)
    if HOURS_CUE.search(nq) and BRANCH_WORD.search(question):
        spec.intents.append("hours")
    elif HOURS_CUE.search(nq) and not re.search(r"\brate|\bapr|\bapy", nq):
        spec.intents.append("hours")
    if ADDRESS_CUE.search(nq):
        spec.intents.append("address")
    if COUNT_CUE.search(nq):
        spec.intents.append("count")
    if URL_CUE.search(nq):
        spec.intents.append("link")
    if PHONE_CUE.search(nq) and "number" in nq:
        spec.intents.append("entity")
    if re.search(r"routing number|aba\b|account and routing", nq):
        spec.intents = ["entity"] + spec.intents
    spec.intents.append("rate")
    spec.intents.append("passage")

    spec.want_list = bool(LIST_CUES.search(nq) or PLURAL_PRODUCT.search(nq))
    spec.families = families_in(nq)

    # ---- product phrases the store actually knows about ------------------
    toks = [t for t in tokens(nq)]
    claimed = [False] * len(toks)
    for i, n, ph in _phrase_ngrams(toks):
        if any(claimed[i:i + n]):
            continue
        if phrase_index.known(ph):
            spec.phrases.append(ph)
            for k in range(i, i + n):
                claimed[k] = True

    spec.content = [t for t in content_tokens(nq) if t not in GENERIC]
    consumed = set()
    if spec.term_months:
        consumed |= {str(spec.term_months), str(spec.term_months // 12) if spec.term_months % 12 == 0 else ""}
    if spec.model_year:
        consumed.add(str(int(spec.model_year)))
    for ph in spec.phrases:
        consumed |= set(ph.split())
    for f in spec.flags:
        consumed.add(f)

    spec.content = [t for t in spec.content if t not in consumed]
    spec.unknown = [t for t in spec.content
                    if t not in consumed and not phrase_index.known_word(t)
                    and not re.fullmatch(r"[\d$%.,+-]+", t) and len(t) > 2]
    return spec


def expand(token: str) -> tuple[str, ...]:
    return SYNONYMS.get(token, (token,))


_SENTENCE_START = re.compile(r"^[^A-Za-z]*")
_PROPER = re.compile(r"\b([A-Z][A-Za-z']+(?:[- ][A-Z][A-Za-z']+)*)\b")
_NOT_A_PLACE = frozenset("""
dupaco visa mastercard i a an the what which who how when where why can could do
does did is are was were may might shall should will would june july monday tuesday
wednesday thursday friday saturday sunday january february march april may august
september october november december yes no apr apy aprs apys atm cd cds ira hsa heloc us
branch branches office offices lobby bank banking credit union savings checking
rates rate loan loans card cards account accounts member members pay visa shine
free business personal online mobile home auto service services please thank hi
tell give list show need want find get see know hello hey thanks also just
""".split())


def _proper_nouns(question: str) -> list[str]:
    """Capitalised names the member used - the only words we treat as places.

    "Mullan Avenue" and "Galena" are names; "general lobby hours" contains none.
    Using capitalisation rather than a gazetteer means an unlisted branch name
    still reads as a name, which is what makes the honest refusal possible.
    """
    body = question.strip()
    out: list[str] = []
    for m in _PROPER.finditer(body):
        if m.group(1).isupper() and len(m.group(1)) <= 5:
            continue                       # an acronym, not a place name
        for w in re.split(r"[- ]", m.group(1)):
            lw = w.lower().removesuffix("'s").removesuffix("s'")
            if lw not in _NOT_A_PLACE and len(lw) > 2:
                out.append(lw)
    return out


_COUNT_NOUN = re.compile(
    r"how many\s+([a-z0-9'\- ]{2,40}?)\s*"
    r"(?:\bdoes\b|\bdo\b|\bare\b|\bis\b|\bhas\b|\bhave\b|\bused\b|\bwere\b|"
    r"\bwas\b|\bwill\b|\bcan\b|[?.]|$)")


def _count_noun(nq: str) -> tuple[Optional[str], Optional[str]]:
    """The thing being counted in a "how many X" question.

    Knowing that "how many branches" counts branches - and not days, dollars or
    volunteer hours - is what lets the system insist the number it quotes is
    actually standing next to that noun.
    """
    m = _COUNT_NOUN.search(nq)
    if not m:
        return None, None
    words = [w for w in m.group(1).split() if w not in STOPWORDS]
    if not words:
        return None, None
    tail = nq[m.end(1):].strip().split()
    verb = next((w for w in tail[:2] if w not in STOPWORDS and len(w) > 2), None)
    return words[-1], verb
