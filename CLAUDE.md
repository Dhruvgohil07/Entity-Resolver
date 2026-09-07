# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Two stages implemented, both with tests (`pytest` → 24 passing):

- **`schema.py`** — canonical `Record` model (product-domain scope; `raw_attributes` is the escape
  hatch for unmapped source columns). Committed.
- **`normalize.py`** — `_fold` (NFKD + casefold + whitespace), unit-spelling and brand-alias
  canonicalization, model-number extraction, and the `NormalizedRecord` type (composition:
  `raw: Record` + derived fields).

Everything from `blocking/` onward is still empty scaffolding, and `data/raw/` holds no datasets.
The next work is a **dataset loader** (raw CSV → `Record`) plus the TF-IDF baseline below —
blocking cannot be evaluated on pair completeness until there is loaded data with ground truth.

Commands in this file describe the intended contract — verify a command exists before relying on it,
and update this file as each phase lands.

### Known issues (open)

A code review found real defects in `normalize.py` that are **not yet fixed** — its output is not
yet trustworthy. In brief: three of the six unit regexes are dead (a trailing `\b` after `\.` can
never match before a space), the `cu ft` rule leaves a stray period, the apostrophe rule maps the
*foot* mark to inches (and mangles possessives), and model-number extraction's trailing-delimiter
branch checks for `" - "` but then takes the last token unconditionally — so specs like `1200W` and
`12MP` are returned as model numbers, which would become high-weight false blocking keys.
`schema.py` separately accepts NaN/`inf` prices, whitespace-only `title`/`record_id`, and silently
ignores unknown field names (`extra="ignore"`), all of which matter once a CSV loader exists.

## What this is

An entity-resolution / deduplication service for e-commerce product listings. It answers two questions:

1. Batch: given a catalog of N records, which sets of records refer to the same real-world product?
2. Online: given one new record, which existing records does it likely duplicate, and with what confidence?

The output is not just a boolean. Pairs are routed into auto-merge / auto-reject / human-review bands,
and human decisions from the review queue flow back as training labels.

## Pipeline architecture

The system is a five-stage pipeline. Every stage constrains the ones after it, and the constraints are
the part that requires reading multiple modules to understand:

```
raw records
  -> normalize.py        canonical form: unicode NFKD, case, units, brand aliases, model-number extraction
  -> blocking/           N^2 pairs -> candidate pairs        [sets a HARD RECALL CEILING]
  -> features/           candidate pair -> ~25-35 dim feature vector
  -> model/              vector -> CALIBRATED probability -> cost-based band assignment
  -> cluster/            scored pair graph -> entity clusters
```

Stage boundaries that matter:

- **Blocking sets a ceiling nothing downstream can lift.** A true duplicate pair that no blocker emits
  is unrecoverable — the classifier never sees it. When system recall disappoints, check blocking pair
  completeness *before* touching the model. Blockers are evaluated on two numbers, not accuracy:
  pair completeness (recall of true pairs surviving) and reduction ratio (fraction of N^2 discarded).
  Multiple blockers are unioned because each misses a different failure mode.
- **The classifier consumes pairs, the service consumes entities.** `cluster/` is the translation layer.
  Good pairwise F1 does not imply good clusters — they are scored with different metrics (see below).
- **`normalize.py` runs identically in batch and at serve time.** Any normalization that exists only in
  the training path is a train/serve skew bug. Model-number extraction lives here and is the single
  highest-value signal; it warrants its own tests.

## Invariants

These are correctness traps specific to this problem. Violating them produces results that look good
and are wrong.

- **Split by entity, never by pair.** A pair-level train/test split leaks: the same product appears on
  both sides and metrics inflate badly. Group splits on entity/cluster id.
- **Use PR-AUC and precision@k. Never ROC-AUC.** Post-blocking class imbalance is extreme (thousands of
  negatives per positive); ROC-AUC reads ~0.99 for a useless model.
- **Model output must be calibrated** (isotonic or Platt). The cost model consumes a probability, not a
  ranking score — an uncalibrated score makes the threshold sweep meaningless.
- **Thresholds come from expected cost, not argmax F1.** False merge (fusing two distinct products —
  corrupts the catalog, hard to undo) and false split (a duplicate survives — cheap) have asymmetric
  costs. Two thresholds define three bands: auto-merge, auto-reject, review queue.
- **Missing values are encoded as explicit indicator features.** A null price is not a price mismatch;
  without indicators the model learns garbage from imputed values.
- **Cluster quality is measured with B-cubed precision/recall**, separately from pairwise metrics.
- **Connected components chains.** A~B and B~C with A!~C still merges all three; one bad edge fuses two
  clusters. This failure is intentionally demonstrated, not designed around silently.

## Layout

```
src/dedup/
  schema.py       canonical record model — all datasets normalize into this, keeping later stages dataset-agnostic
  normalize.py    shared by batch and serve paths
  blocking/       standard, sorted_neighborhood, lsh, ann, union
  features/       string, numeric, semantic, missingness
  model/          train, calibrate, threshold (cost model)
  cluster/        components, correlation, agglomerative, bcubed
  synth/          corruption engine for synthetic scale-up
  service/        FastAPI app, HNSW + inverted index, review queue
reports/          blocking table, PR curves, cost curves — the defensible results
```

## Stack choices with a reason

- `rapidfuzz` for string distances, **not** `fuzzywuzzy` — C++ backed, orders of magnitude faster, which
  is load-bearing when computing several metrics over millions of pairs.
- `datasketch` (MinHash-LSH) and `faiss-cpu` (ANN, via `IndexHNSWFlat`) are the two sublinear
  blockers. `faiss-cpu` replaces `hnswlib` — `hnswlib` has no PyPI wheel for any platform and needs
  an MSVC C++ toolchain to build on Windows, which the dev machine doesn't have; verified by a direct
  `pip download --only-binary=:all:` probe, not assumed. `faiss-cpu` ships a real win_amd64 wheel and
  implements the same HNSW algorithm, so `blocking/ann.py` can be built against it with no design
  change from the original hnswlib plan.
- `sentence-transformers` embeddings are complementary to string distance, not a replacement: they catch
  paraphrase and miss fine distinctions (`WH-1000XM4` vs `WH-1000XM5`), string metrics do the opposite.
- LightGBM over the pair features; a fine-tuned cross-encoder is the optional ceiling comparison.
- DuckDB/SQLite record store, FastAPI + uvicorn, htmx review UI (deliberately no Node toolchain).

## Data

- Benchmarks with ground truth: Abt-Buy, Amazon-Google (start here — small, published numbers to compare
  against). DBLP-ACM / DBLP-Scholar if a second domain is needed.
- These benchmarks are *pre-blocked*, so they understate the blocking problem. `synth/` corrupts a large
  catalog (typos, abbreviations, token drop/reorder, unit swaps, price jitter, brand aliasing) to produce
  200k-1M records with known ground truth, which is where blocking becomes genuinely necessary.

## Baseline to beat

TF-IDF char-3gram cosine with a single threshold. Record its F1 before building anything else; every
later stage is justified against it.

## Commands

The dev environment is `.venv` (deliberately Python 3.12, not the machine default — ML wheel
availability). On Windows, prefix with `.venv/Scripts/python -m` if the venv is not active.

These work today:

```bash
# environment
pip install -e ".[dev]"

# tests
pytest                                  # all (24 passing)
pytest tests/test_normalize.py          # one file
pytest tests/test_normalize.py::test_model_number_trailing_convention   # one test
pytest -k model_number                  # by keyword

# lint (line-length 100, target py310)
ruff check src tests
```

Not built yet — intended contract, will fail if invoked:

```bash
# pipeline stages
python -m dedup.blocking.evaluate       # emits the blocker x completeness x reduction table
python -m dedup.model.train
python -m dedup.cluster.evaluate

# service
uvicorn dedup.service.app:app --reload
```

## Commit Messages

Use Conventional Commits format for every commit:

```
<type>(<scope>): <short summary, imperative mood, ≤72 chars>

<body — why this change, not just what changed>

<footer — optional, e.g. "Refs: ADR-3">
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `chore`
Scopes: `blocking`, `features`, `model`, `eval`, `service`, `data-gen`

Rules:

- Subject line imperative ("add geohash blocking pass", not "added" or "adds").
- Body explains reasoning/tradeoff, especially for modeling or threshold changes.
- Reference the CLAUDE.md Decisions Log entry if the commit implements a logged decision.
- No commit bundles unrelated changes — one logical change per commit.
- Do NOT add "Co-Authored-By: Claude" or any Claude/Anthropic attribution line in the commit message.
- Do NOT include the "🤖 Generated with Claude Code" footer or any equivalent.
- All commits should be authored under the user's own git identity — never configure or leave a
  separate Claude author/committer identity in git.
- Commit only when the user explicitly prompts Claude to commit — never as a side effect of finishing
  other work.
