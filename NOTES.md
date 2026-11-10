# NOTES — how this system decides what is true now

**API keys: none. Dependencies: none.** Pure Python 3.9+ standard library, no
network access, no model calls. `requirements.txt` is intentionally empty.

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
python tests/test_units.py && python tests/test_end_to_end.py
```

---

## 1. The shape of the problem

Reading the five snapshots, the contradictions are not random noise. They come
in four recognisable shapes, and each one wants a different rule:

| What you see | Example in this data | What a careful reader does |
|---|---|---|
| Two generations of the same rate sheet, both live | `/personal/loans/auto-motorcycle-loans` says 5.39% "effective 08/31/2026"; `/switch-and-earn` says 5.34% "effective 1/1/2026" | Believe the newer **effective date** — the publisher dated both |
| A page nobody maintains | the vehicle-affordability calculator still shows 5.34% | Believe the **rate sheet**, not the calculator |
| A dated statement copied forward | press releases say 19, 20, 22, 23 and 24 "branch office locations", all still on the site | Believe the **most recently published** one |
| Qualifiers that look like disagreement | 2.23% dividend rate beside 2.25% APY; `$10,000–$24,999.99` beside `$25,000+`; 17-month beside 17-month **jumbo** | These are **not conflicts at all** |

Everything below follows from taking those four shapes seriously.

## 2. Architecture

Ingest is where the thinking happens; answering is a lookup. That split is
deliberate — it is what makes answers fast (p95 **0.02 s**) and byte-identical
on repeat (**consistency 1.00**).

```
Snapshot ─▶ docmodel ─▶ extract / entities ─▶ Store (temporal facts)
            parse the    typed facts with     every observation ever seen,
            page into    full provenance      never overwritten
            labelled
            blocks   ─▶ retrieval (BM25 over deduplicated sentences)

answer(q) ─▶ query.parse ─▶ intent handler ─▶ resolve ─▶ phrasing
             hard             rate / hours     rank by    say the winner
             qualifiers       address / link   the rules  and nothing else
                              count / passage   in §4
```

| Module | Responsibility |
|---|---|
| `docmodel.py` | Markdown → blocks. Keeps each table together with the tab labels above it and the `Rates are effective as of …` line below it; classifies the page (rate sheet / product / landing / news / calculator …) |
| `extract.py` | Rate and fee tables → typed facts; prose rates; fee-schedule lines; expired-offer detection |
| `entities.py` | Branch directory, lobby and drive-up hours, routing number, phone, canonical links |
| `quals.py` | Qualifier recognisers (term, balance tier, loan-amount tier, model year, ARM, variant flags) — the *same code* runs over table rows and over questions |
| `store.py` | The temporal store: visibility, ranking, contradiction detection |
| `retrieval.py` | BM25 over deduplicated sentences, with document order retained |
| `query.py` | Question → `Spec`: metric, family, qualifiers, intent |
| `engine.py` | Intent routing, candidate resolution, answer assembly, `conflicts()` |
| `phrasing.py` | Member-facing sentences |

## 3. The fact model

A number on a rate sheet means nothing without its qualifiers, so the unit of
knowledge is an **observation**: a value plus everything needed to know what it
is a value *of*, and where it came from.

```python
Obs(domain="rate", metric="apr", family="auto_loan",
    value_text="5.39%", lo=5.39,
    term_months=60, model_years=(2024, 2027), balance=None, flags=frozenset(),
    prov=Provenance(url=…, page_kind="product", effective_on=date(2026, 8, 31),
                    snapshot_idx=4, fetched_at=…, taken_on=…))
```

Nothing is ever overwritten. `ingest()` appends, so the store holds the full
history and `answer(as_of=…)` is a *filter over the same data*, not a separate
code path. Roughly 38,000 observations and 21,000 distinct sentences come out of
the five snapshots in about 22 seconds.

**Visibility.** "The website as it stood on day D" is not "everything we ever
crawled up to D" — it is, for each URL, that page's newest copy at or before D.
`Store.finalize()` precomputes that per snapshot. This is why a page that
disappeared stops being quoted, and why `as_of` works without special-casing.

## 4. What "current" means — the ranking rules

Candidates are ranked by this tuple, highest first:

1. **Not expired.** An offer that names its own end date ("offer ended
   2025-09-30") stops being true on that date even though the page stays up.
2. **Editorial tier.** rate sheet / product page (3) > support, legal (2) >
   campaign landing page (1) > press release, blog, **calculator** (0). A stale
   calculator being re-crawled today does not make its frozen number current.
3. **Self-dated before undated.** A table that says "effective as of …" is the
   publisher telling you which generation this is.
4. **Validity date, newest first** — `effective_on`, else the byline date of a
   press release, else the crawl date.
5. Purchase before refinance; then page authority; then snapshot order.

Rule 2 before rule 4 is the load-bearing choice: it is what stops a freshly
crawled but unmaintained page from outranking the maintained rate sheet. Within
a tier, rule 4 decides, which is what resolves the 5.39%/5.34% case.

**Answers quote one generation only.** Once a winner is chosen, the answer is
built from that publication's rows alone. Superseded values go in
`conflict_note`, never in `text` — hedging by listing both numbers is the exact
failure being tested, and it is also just a bad answer.

## 5. Qualifiers are identity, not noise

`quals.py` reduces "between $10,000 and $24,999" in a question and
`$10,000 – $24,999.99` in a table to the same interval, so tiers match by
meaning rather than by string. Qualifiers are then **hard filters**:

- **Term** — a 48-month rate is not a 60-month rate.
- **Balance / loan-amount tier** — the member's balance selects one row.
- **Model year** — "a 2025 vehicle" selects the 2024-27 tier.
- **ARM** — asking about a 7/1 ARM must not return the 30-year fixed rate.
- **Variant flags** — `jumbo`, `business`, `ira`, `intro`, `mini_max`,
  `secured`, `student`, `special`. These must match *in both directions*: a
  question that says "jumbo" only gets jumbo rows, and a question that doesn't
  never gets them.

**Pinning the product.** Qualifiers narrow to a product family; the member's own
words choose between siblings inside it. That is scored by coverage rather than
filtered, because a question mixes identity words with modifiers: *"the standard
purchase APR on the Platinum Visa"* names one product and two modifiers. A
contiguous phrase the site itself uses ("platinum visa") is worth far more than
any single word, and a word in the row's own name outweighs one inherited from
the section heading above it — which is what keeps "standard" from steering the
answer to the Rewards card, and what picks "Rush fee for debit card" out of a
page full of debit-card fees.

## 6. Saying "I don't know"

Refusals are a *consequence* of the qualifier filters, not a separate classifier:
when the member names a term, variant or product the rate sheet has no row for,
there is nothing to answer with, and the system says so. That covers the 7/1 ARM
(no ARM rows), construction loans (described in prose, never priced) and a
9-month certificate (the sheet has 6 and 12).

Place names work the same way. Only **capitalised** words count as place names,
and a branch name nothing on file mentions ("Mullan Avenue") produces a refusal
rather than a plausible-looking schedule — while "general lobby hours" contains
no name at all and is answered directly.

Two details matter for honesty: a refusal contains **no numbers** (quoting a
phone number reads as an answer), and `confident=False` is set so a caller can
route the member to a human.

One thing deliberately *not* treated as grounds to refuse: an unfamiliar word in
the question. "six", "January" and "quarterly" are the member's phrasing, not
invented products — declining on those would turn paraphrases into failures.

## 7. Prose questions

Not everything is in a table. A BM25 index over deduplicated sentences handles
eligibility, programme descriptions, closure dates and counts. Deduplication is
the important part: boilerplate appears on hundreds of pages, and keeping every
occurrence lets a sentence be *dated by the most recent page that still carries
it*. Refinements, each earned from a specific failure:

- **Light stemming**, so "how many branches" reaches "branch office locations".
- **Rare words pin the subject.** A statement without the question's rare words
  ("Forbes") is about something else, however well it scores.
- **Sentence shape.** An FAQ heading is a question, not an answer; "For more
  information, visit …" points elsewhere and says nothing. Both are demoted. For
  a yes/no question, a sentence beginning "Yes" is promoted.
- **Read to the end of the thought.** The sentence carrying the figure often
  refers back instead of repeating the name — *"…through the Home$tart program.
  **This initiative** provides up to $15,000…"* — so a continuation is read
  along with its parent.
- **Two recency rules, not one.** For *counts*, the newest published figure
  wins outright: a count is a fact that changes. For everything else, relevance
  decides *which claim* answers the question and recency only chooses between
  **restatements of that same claim** — which is how "the fifth time Forbes
  recognized Dupaco" (2024) beats the near-identical "the fourth time" (2023)
  without letting an unrelated newer sentence win.
- **Counts must count the right noun.** "22 branch office locations" counts
  branches; "60 days to visit one of our branches" does not. The figure has to
  stand within two words of the noun from "how many *X*", and counts of a
  shared-branch network are excluded from counts of Dupaco's own offices.

## 8. `conflicts()`

A contradiction is two live observations making **incompatible claims about the
same fact**. Getting the "same fact" part right is most of the work, and most of
the rules here exist to *suppress* false positives:

- APY and dividend rate are never compared — 2.23% beside 2.25% is one promise
  described two ways. Nor are a mortgage's note rate and its APR.
- Product identity is the product's name with its family's generic nouns
  stripped: Share Savings `{share}` vs Bright Savings `{bright}` are different
  products; variant words (`business`, `jumbo`, `ira`) are held separately so
  Business Platinum Visa and Business Rewards Visa stay distinct.
- Tiers must be **indexed the same way and overlap**. A row priced by loan
  amount and a row priced by model year are different facts even in one table.
- Calculator output and marketing sentences with no term or tier are not
  precise enough to contradict anything.
- Identical disagreements repeated across every balance tier are one editorial
  mistake, so they collapse into one entry.

Each entry carries the competing values, their sources, the resolved value and a
reason in the same vocabulary as §4 — *"the rate sheet page is effective
09/25/2026 and supersedes the 04/22/2026 generation still shown on
/homebuying-guidance"*. On the five snapshots this yields **33 conflicts**,
including every stale-value trap in the public eval (5.39/5.34, 5.59/5.54,
mortgage generations, 1st Rate Checking, the Achieve promo and the IRA
certificate specials). On the harness's own fixture: F1 **1.00**, winner
accuracy **1.00**.

## 9. Citations

Every page that currently asserts the chosen value is cited, product page first,
and a page whose URL is about a *different* product is pushed down. Several
pages usually carry the same rate — the master rate sheet and the product page —
and all of them genuinely support the answer.

## 10. Results

| | Public eval (40 q) | Harness fixture |
|---|---|---|
| Final score | **0.89** | **1.00** |
| Accuracy | 0.89 | 1.00 |
| Stale-value traps hit | **0** | 0 |
| Consistency (paraphrase / repeat) | **1.00 / 1.00** | 1.00 |
| Speed p50 / p95 | **0.00 s / 0.02 s** | — |
| Conflicts F1 / winner | not scored locally (no gold) | **1.00 / 1.00** |
| Ingest | 22 s (limit 30 min) | — |

Starter baseline: 0.09.

## 11. Where it is weak

Worth saying plainly, since these are the places I would work on next:

- **Prose retrieval is lexical.** It has no idea that "who is eligible to join"
  and "we serve 118 counties throughout Iowa, northwest Illinois and southwest
  Wisconsin" are the same subject. Two public-eval misses are exactly this. A
  small embedding model over the 21,000 sentences would fix it without touching
  the latency budget (encode at ingest, dot-product at answer time) — I left it
  out to keep the submission dependency-free and fully explainable.
- **"Last year" is not resolved against the question's own frame.** For "how
  many employees used Volunteer Time Off last year" the system returns the most
  recently published figure (256, published 2025) where the expected answer is
  294 (published 2024). The general rule is right; this question wanted the
  previous one.
- **Money-market promotions.** `Achieve` appears only on campaign pages, so
  "the highest money market APY" resolves to a promotional rate. With no
  authoritative sheet to check it against, the system has nothing better to
  prefer.
- **The ontology is hand-written.** ~60 lines of banking vocabulary (a CD is a
  certificate, a car loan is an auto loan). It is domain knowledge, not answers,
  but it is the part that would need extending for a different institution.
