# A knowledge system that knows what's true *now*

Answers a credit union's members from five dated snapshots of its website —
with the current value, the right tier, a citation, and an honest "I don't
know" when the site doesn't say.

**No API keys. No dependencies. No network calls.** Python 3.9+ standard
library only.

## Run it

```bash
# score against the practice questions
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl

# tests (these need no snapshot data, so they run on a fresh clone)
python tests/test_units.py
python tests/test_end_to_end.py
```

The website snapshots are **not** in this repository (the data was shared for
the assignment only). Drop the kit's `snapshots_zip/` beside `snapshots/` and the
harness unpacks it on first run, or unzip it into `snapshots/` yourself.

## Results

| | Public eval (40 q) | Harness fixture |
|---|---|---|
| Final score | **0.89** | **1.00** |
| Stale-value traps hit | **0** | 0 |
| Consistency (paraphrase / repeat) | **1.00 / 1.00** | 1.00 |
| Speed p50 / p95 | **0.00 s / 0.02 s** | — |
| Conflicts F1 / winner picked | — (no local gold) | **1.00 / 1.00** |
| Ingest | 22 s (limit 30 min) | — |

Starter baseline: 0.09.

## How it works, in one paragraph

Ingest does the thinking; answering is a lookup. Each page is parsed into
labelled blocks that keep a rate table together with the headings above it and
its `Rates are effective as of …` line below it. Tables, fee schedules, hours,
branches and links become **observations** — a value plus its qualifiers (term,
balance tier, model year, jumbo/business/IRA) plus full provenance. Nothing is
overwritten, so the store holds the whole history and `as_of` is a filter over
it rather than a separate path. "Current" is then a ranking: an expired offer
loses; a maintained rate sheet beats an unmaintained calculator; among
equally-maintained sources the newer *effective date* wins. Qualifiers are hard
filters, which is where honest refusals come from — ask for a 7/1 ARM and there
is no row to answer with. Prose questions go to a BM25 index over deduplicated
sentences, where keeping every occurrence of a sentence lets it be dated by the
most recent page still carrying it.

**[NOTES.md](NOTES.md) has the full reasoning**, including the ranking rules,
how contradictions are told apart from tiers, and where the system is weak.

## Layout

```
submission/
  solution.py            build() -> ingest / answer / conflicts
  knowledge_now/
    docmodel.py          page -> labelled blocks; page classification
    extract.py           rate & fee tables, prose rates, expired offers
    entities.py          branches, hours, routing number, links
    quals.py             qualifier recognisers (shared by ingest and queries)
    ontology.py          banking vocabulary: families, metrics, synonyms
    facts.py             the observation and its provenance
    store.py             temporal store: visibility, ranking, contradictions
    retrieval.py         BM25 over deduplicated sentences
    query.py             question -> Spec
    engine.py            intent routing, resolution, conflicts()
    phrasing.py          member-facing sentences
tests/                   unit tests + an end-to-end run on the harness fixture
harness/, eval/          the organisers' scorer and practice questions (unmodified)
```
