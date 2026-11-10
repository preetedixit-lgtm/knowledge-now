"""Domain vocabulary for a credit-union website.

This is deliberately *not* a list of answers. It is the small amount of banking
knowledge a careful human reader brings to the page: that a CD is a
certificate, that "car loan" and "auto loan" are the same product, that APY and
dividend rate are different columns of the same row, and that "jumbo",
"business" and "IRA" are variants you must not silently substitute.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------
# metrics (table column -> what the number means)
# --------------------------------------------------------------------------

METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"annual percentage yield|\bapys?\b", "apy"),
    (r"annual percentage rate|\baprs?\b", "apr"),
    (r"dividend rate", "dividend_rate"),
    (r"interest rate", "rate"),
    (r"^rate$|^rate\b|as low as.*rate|^rates$", "rate"),
    (r"^points$", "points"),
    (r"payment example|monthly payment", "payment"),
    (r"min\.? to open|minimum to open", "min_open"),
    (r"min\.? to earn|minimum to earn|min\.? balance|minimum balance", "min_earn"),
    (r"max\.? balance|maximum balance", "max_balance"),
    (r"^fee$|fee amount|^amount$|^cost$|^price$|^charge$", "fee"),
]

# Which metric a question is asking for.
METRIC_QUESTION = [
    (r"\bapys?\b|annual percentage yield", "apy"),
    (r"\baprs?\b|annual percentage rate", "apr"),
    (r"dividend rate", "dividend_rate"),
    (r"\bpoints\b", "points"),
    (r"monthly payment|payment example", "payment"),
    (r"minimum (?:to|needed to) open|min(?:imum)? deposit", "min_open"),
    (r"minimum (?:to|needed to) earn", "min_earn"),
    (r"\bfee\b|\bcost\b|how much (?:is|does|to)\b|\bcharge\b|\bprice\b", "fee"),
    (r"interest rate|\brate\b|how much (?:do|will) i earn|\bearn\b|\byield\b", "rate"),
]

# Rate metrics are interchangeable enough to fall back between: if a member asks
# for "the rate" on a deposit product we may answer with the APY, and vice
# versa, but we must never answer an APR question with an APY.
DEPOSIT_METRICS = ("apy", "dividend_rate", "rate")
LOAN_METRICS = ("apr", "rate")


def metric_for_header(header: str) -> str | None:
    h = header.lower()
    for pat, name in METRIC_PATTERNS:
        if re.search(pat, h):
            return name
    return None


def metric_for_question(q: str) -> str | None:
    for pat, name in METRIC_QUESTION:
        if re.search(pat, q):
            return name
    return None


# --------------------------------------------------------------------------
# key columns (table column -> which qualifier dimension it carries)
# --------------------------------------------------------------------------

KEY_PATTERNS: list[tuple[str, str]] = [
    (r"balance tier|^balance$|balance range", "balance"),
    (r"loan amount|^amount financed$", "loan_amount"),
    (r"^term$|term length|^months$", "term"),
    (r"account type|^account$|^product$|^program$|loan program|^certificate$|"
     r"ira type|card type|^type$|^card$|^loan$|^plan$|^service$|^item$|^description$|"
     r"^fee type$|^transaction$|^feature$|^model year$", "label"),
]


def key_for_header(header: str) -> str | None:
    h = header.lower()
    for pat, name in KEY_PATTERNS:
        if re.search(pat, h):
            return name
    return None


# --------------------------------------------------------------------------
# product families
# --------------------------------------------------------------------------
# Order matters: the first match wins, so put the specific before the general.

FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("heloc", ("heloc", "home equity line")),
    ("home_equity_loan", ("home equity loan", "home equity refinance", "home equity")),
    ("home_improvement_loan", ("home improvement",)),
    ("mortgage", ("mortgage", "home loan", "fixed conventional", "purchase:", "refinance:",
                  "year fixed", "arm", "adjustable-rate", "fha", "va loan", "usda")),
    ("construction_loan", ("construction",)),
    # Recreational vehicles are priced on their own sheet; an "auto loan"
    # question must not be answered from the boat-and-camper table.
    ("recreational_loan", ("recreational", "rv ", "boat", "atv", "snowmobile",
                           "camper", "motorhome", "jet ski", "pontoon")),
    # "model year" is a tier label, not a product name, so it is deliberately
    # absent: a row keyed only by model year takes its product from the heading
    # above it, which is how the recreational-vehicle sheet stays distinct.
    ("auto_loan", ("auto loan", "auto &", "auto and", "motorcycle", "vehicle loan",
                   "car loan", "mini max", "mini-max")),
    ("student_loan", ("student loan", "student refinance", "college")),
    ("personal_loan", ("personal loan", "signature loan", "unsecured loan", "share secured",
                       "savings secured", "line of credit", "overdraft")),
    ("credit_card", ("visa", "credit card", "mastercard", "platinum", "rewards card")),
    ("certificate", ("certificate", "term share", "cd ", " cd", "jumbo special")),
    ("ira", ("ira", "roth", "traditional", "retirement")),
    ("money_market", ("money market", "investor's choice", "investors choice", "achieve")),
    ("hsa", ("health savings", "hsa")),
    ("checking", ("checking", "free checking", "1st rate")),
    ("savings", ("savings", "share savings", "you name it", "holiday club", "launchpad",
                 "bright savings", "high-yield", "high yield", "dopack", "kids account")),
    ("business_loan", ("commercial real estate", "term loans and lines", "sba")),
]

FAMILY_LABEL = {
    "heloc": "home equity line of credit (HELOC)",
    "home_equity_loan": "home equity loan",
    "home_improvement_loan": "home improvement loan",
    "mortgage": "mortgage",
    "construction_loan": "construction loan",
    "auto_loan": "auto loan",
    "recreational_loan": "recreational-vehicle loan",
    "student_loan": "student loan",
    "personal_loan": "personal loan",
    "credit_card": "credit card",
    "certificate": "certificate",
    "ira": "IRA",
    "money_market": "money market",
    "hsa": "health savings account",
    "checking": "checking account",
    "savings": "savings account",
    "business_loan": "business loan",
}

LOAN_FAMILIES = frozenset({
    "heloc", "home_equity_loan", "home_improvement_loan", "mortgage", "construction_loan",
    "auto_loan", "student_loan", "personal_loan", "credit_card", "business_loan",
    "recreational_loan",
})


def family_of(text: str) -> str | None:
    t = " " + text.lower() + " "
    for name, needles in FAMILIES:
        for n in needles:
            if n in t:
                return name
    return None


def families_in(text: str) -> list[str]:
    t = " " + text.lower() + " "
    return [name for name, needles in FAMILIES if any(n in t for n in needles)]


# --------------------------------------------------------------------------
# question-side synonyms -> the words the website actually uses
# --------------------------------------------------------------------------

SYNONYMS: dict[str, tuple[str, ...]] = {
    "cd": ("certificate",),
    "cds": ("certificate",),
    "car": ("auto", "vehicle"),
    "cars": ("auto", "vehicle"),
    "vehicle": ("auto",),
    "truck": ("auto", "vehicle"),
    "mortgage": ("home loan", "fixed conventional"),
    "mortgages": ("home loan",),
    "house": ("home",),
    "home": ("home",),
    "visa": ("visa", "credit card"),
    "creditcard": ("credit card",),
    "atm": ("atm",),
    "cu": ("credit union",),
    "savings": ("savings",),
    "share": ("share",),
    "moneymarket": ("money market",),
    "checkings": ("checking",),
    "chequing": ("checking",),
    "hours": ("hours",),
    "routing": ("routing",),
    "aba": ("routing",),
    # Intent words a member uses where the site uses product words.
    "eligible": ("eligibility", "membership", "join", "qualify", "charter", "counties",
                 "serves residents", "live or work"),
    "who": ("eligibility", "charter"),
    "eligibility": ("membership", "join", "qualify"),
    "join": ("membership", "join"),
    "qualify": ("eligibility", "membership"),
    "member": ("membership",),
    "members": ("membership",),
    "lobby": ("lobby",),
    "wallet": ("wallet",),
}

# Words that mark a *variant* of a product. If the member does not say them we
# must not hand back the variant, and if the member does say them we must not
# hand back the plain version.
DISCRIMINATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("jumbo", ("jumbo",)),
    ("business", ("business", "commercial")),
    ("ira", ("ira", "roth", "traditional ira")),
    ("special", ("special",)),
    ("refinance", ("refinance", "refi")),
    ("purchase", ("purchase",)),
    ("intro", ("introductory", "intro")),
    ("mini_max", ("mini max", "mini-max")),
    ("secured", ("share secured", "savings secured")),
    ("student", ("student",)),
)

# Generic words that must never be treated as a product requirement.
GENERIC = frozenset("""
rate rates apr apy apy1 yield dividend dividends interest percent current now today
loan loans account accounts offer offers product products program programs term terms
month months year years tier tiers balance minimum min max credit union dupaco
what whats how much the a an is are for on in of to me my you your i we please tell
give get know right there any does do did can could would will cu website site page
""".split())


# A question only reaches the structured rate/fee lookup if it actually asks for
# a number. Without this gate "How do I join online?" matches a product word and
# comes back with a rate table.
PRICE_QUESTION = re.compile(
    r"\baprs?\b|\bapys?\b|\brates?\b|\byield\b|\bdividends?\b|\binterest\b|"
    r"\bfees?\b|\bcosts?\b|\bcharges?\b|\bprice\b|\bpoints\b|\bpayment\b|"
    r"\bhow much\b|\bminimum\b|\bpays?\b|\bearn(s|ing)?\b|\bpercent\b|"
    r"\bpremium\b|\bpenalty\b")


def is_price_question(q: str) -> bool:
    return bool(PRICE_QUESTION.search(q))


# Every word that appears inside a family's own vocabulary. Used to tell that
# "cards" is asking about the credit-card family even when a particular rate
# table's labels never spell out "credit cards".
_FAMILY_WORD_INDEX: dict[str, str] = {}
for _name, _needles in FAMILIES:
    for _n in _needles:
        for _w in _n.replace(":", " ").split():
            _FAMILY_WORD_INDEX.setdefault(_w.strip(), _name)


def family_for_word(word: str) -> str | None:
    w = word.lower().strip()
    for cand in (w, w.rstrip("s"), w + "s"):
        if cand in _FAMILY_WORD_INDEX:
            return _FAMILY_WORD_INDEX[cand]
    return None


def word_variants(word: str) -> tuple[str, ...]:
    """Crude morphology so "branches" finds "branch" and vice versa."""
    w = word.lower()
    out = {w}
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        out.add(w[:-2])
    if w.endswith("s") and len(w) > 2:
        out.add(w[:-1])
    else:
        out.add(w + "s")
        out.add(w + "es")
    return tuple(sorted(out))


# The generic nouns inside each family's product names. Strip these and what is
# left is the product's identity: "Bright Savings" -> {bright}, "Share Savings"
# -> {share}. Without this, every savings account looks like the same fact and
# the system reports contradictions between unrelated products.
HEAD_WORDS: dict[str, frozenset[str]] = {
    "savings": frozenset({"savings", "saving", "account", "accounts"}),
    "checking": frozenset({"checking", "account", "accounts"}),
    "certificate": frozenset({"certificate", "certificates", "term", "share", "fixed",
                              "cd", "cds"}),
    "money_market": frozenset({"money", "market", "markets", "account", "accounts"}),
    "credit_card": frozenset({"credit", "card", "cards", "visa", "mastercard"}),
    "mortgage": frozenset({"mortgage", "mortgages", "home", "loan", "loans", "fixed",
                           "conventional", "year", "purchase", "refinance"}),
    "auto_loan": frozenset({"auto", "motorcycle", "vehicle", "car", "loan", "loans",
                            "model", "year"}),
    "recreational_loan": frozenset({"recreational", "loan", "loans", "vehicle"}),
    "personal_loan": frozenset({"personal", "loan", "loans", "signature", "unsecured"}),
    "heloc": frozenset({"heloc", "home", "equity", "line", "credit", "loan"}),
    "home_equity_loan": frozenset({"home", "equity", "loan", "loans", "fixed"}),
    "home_improvement_loan": frozenset({"home", "improvement", "loan", "loans"}),
    "hsa": frozenset({"health", "savings", "account", "hsa"}),
    "ira": frozenset({"ira", "retirement", "account"}),
    "student_loan": frozenset({"student", "loan", "loans"}),
    "business_loan": frozenset({"business", "commercial", "loan", "loans"}),
    "construction_loan": frozenset({"construction", "loan", "loans"}),
}
