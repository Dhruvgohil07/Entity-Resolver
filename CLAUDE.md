# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Three stages implemented, all with tests (`pytest` → 71 passing):

- **`schema.py`** — canonical `Record` model (product-domain scope; `raw_attributes` is the escape
  hatch for unmapped source columns). Committed.
- **`normalize.py`** — `_fold` (NFKD + casefold + whitespace), unit-spelling and brand-alias
  canonicalization, model-number extraction, and the `NormalizedRecord` type (composition:
  `raw: Record` + derived fields).
- **`data/abt_buy.py`** — Abt-Buy loader: both sides → `Record`, ground-truth pairs → `entity_id`
  by union-find. Tested against committed fixtures, plus one integration test that runs only when
  the benchmark has been downloaded.

Everything from `blocking/` onward is still empty scaffolding. The next work is the **TF-IDF
baseline** below — there is now loaded data with ground truth, so blocking can finally be
evaluated on pair completeness after that.

Commands in this file describe the intended contract — verify a command exists before relying on it,
and update this file as each phase lands.

### Open questions

The code-review defects in `normalize.py` and `schema.py` are fixed, each pinned by a regression
test naming the failure mode. One judgment call in those fixes is worth revisiting against more
data rather than treating as settled:

- **The spec-suffix list in `_SPEC_UNIT_SUFFIXES` is conservative on purpose.** It rejects `1200W`
  and `12MP` as model numbers while deliberately omitting `wh` and `a`, which collide with real
  vendor codes (Bose 161WH, HP Officejet 8500A). A false model number fuses unrelated products into
  one block; a missing one only loses a signal — so the list should grow only against evidence.

Settled by measurement, recorded so it is not re-litigated:

- **`'` folds to inches, not feet.** Typographically `'` is the foot mark, and an earlier pass
  changed it on that basis. Measuring the actual CSVs overturned it: all 249 digit+`'` occurrences
  across `Abt.csv` and `Buy.csv` are screen and driver sizes (`3.0' LCD Display`, `32' to 50' LCD`,
  `4' x 6' Print Paper`, `1-1/8' Dome Tweeter`) and none is a length in feet. Normalize follows the
  source convention, not the typographic one. Revisit only if a source that genuinely sells by the
  foot is added.

## What this is

An entity-resolution / deduplication service for e-commerce product listings. It answers two questions:

1. Batch: given a catalog of N records, which sets of records refer to the same real-world product?
2. Online: given one new record, which existing records does it likely duplicate, and with what confidence?

The output is not just a boolean. Pairs are routed into auto-merge / auto-reject / human-review bands,
and human decisions from the review queue flow back as training labels.

## Pipeline architecture

The system is a five-stage pipeline fed by a per-dataset loader. Every stage constrains the ones
after it, and the constraints are the part that requires reading multiple modules to understand:

```
raw CSV, one shape per dataset
  -> data/               source columns -> canonical Record; ground-truth pairs -> entity_id
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
- **`data/` is the only place that may know a dataset's name.** Everything after it sees `Record`
  objects and nothing else, which is what lets a blocker or feature be written once and evaluated on
  Abt-Buy, Amazon-Google and `synth/` output alike. A per-source branch anywhere downstream is a bug,
  not a shortcut.

## Invariants

These are correctness traps specific to this problem. Violating them produces results that look good
and are wrong.

- **Split by entity, never by pair.** A pair-level train/test split leaks: the same product appears on
  both sides and metrics inflate badly. Group splits on entity/cluster id — `Record.entity_id`, which
  loaders in `data/` populate from the dataset's ground truth for exactly this purpose.
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
  data/           dataset loaders, one module per benchmark — raw CSV to Record, ground truth to entity_id
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
- Download (data/ is gitignored, so a fresh clone starts empty):

  ```bash
  B=https://raw.githubusercontent.com/dchud/ddbench/HEAD/data
  mkdir -p data/raw/abt-buy
  for f in Abt.csv Buy.csv abt_buy_perfectMapping.csv; do
    curl -sS -o "data/raw/abt-buy/$f" "$B/abt-buy/$f"
  done
  ```

  The Leipzig original (`https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip`) has the same three
  files; the older `/file/` path 301-redirects there.
- Measured properties of Abt-Buy, verified by `tests/test_data_abt_buy.py` rather than assumed —
  each was a defect waiting to happen:
  - `Abt.csv` is **cp1252**, not UTF-8 (so are `Amazon.csv` and `GoogleProducts.csv`). A default
    `pd.read_csv` raises on them. Do not "fix" this with `errors="replace"` — that mojibakes
    product names into mismatches instead of failing loudly.
  - **The two sides have different columns.** Abt has no brand column; Buy has `manufacturer` on
    all but 6 of 1092 rows. So `brand` is `None` on every Abt record, which makes brand equality
    useless as a feature here (missing on every pair) and useless as a blocking key.
  - Price is `"$399.00"`, with a thousands comma on 97 of 1008 populated values. 61% of Abt and
    46% of Buy rows leave it empty — empty means `None`, never NaN.
  - 1081 Abt + 1092 Buy records, 1097 ground-truth pairs. Union-find over those pairs gives 1076
    clusters: 1055 of size 2, 21 of size 3, **no runaway component** — so deriving `entity_id` by
    connected components is safe on this dataset. Re-check that distribution before reusing the
    construction anywhere noisier.
  - Every Abt and Buy id appears in the mapping, so there are no unmatched *records* — negatives
    exist only as non-matching pairs.
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
pytest                                  # all (71 passing)
pytest tests/test_normalize.py          # one file
pytest tests/test_normalize.py::test_model_number_trailing_convention   # one test
pytest -k model_number                  # by keyword

# lint (line-length 100, target py310)
ruff check src tests

# load the benchmark (see Data above for the download)
python -c "from dedup.data.abt_buy import load_abt_buy; print(len(load_abt_buy('data/raw/abt-buy')))"
```

Tests run without any dataset present: they use committed fixtures under `tests/fixtures/abt-buy/`.
The single test that reads `data/raw/` skips when the benchmark has not been downloaded, so a green
run does *not* by itself mean the real files were checked — `pytest -rs` reports the skip.

If `tests/fixtures/abt-buy/` ever needs a new shape, regenerate it rather than hand-editing:
`Abt.csv` must stay cp1252-encoded on disk, which an editor will silently undo.

```bash
python tests/fixtures/make_fixtures.py
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

<body — why this change, not just what changed. When the commit covers
several pieces of work, lead with the headline change and cover the rest
as bullets.>

<footer — optional, e.g. "Refs: ADR-3">
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `chore`

Scopes — one per module, added as each lands. In use so far: `schema`, `normalize`, `data`,
`blocking`. Reserved for modules not yet built: `features`, `model`, `eval`, `cluster`, `service`,
`data-gen`. A commit touching no single module (this file, packaging, CI) takes no scope.

Rules:

- Subject line imperative ("add geohash blocking pass", not "added" or "adds").
- Body explains reasoning/tradeoff, especially for modeling or threshold changes.
- If the commit settles or revisits a decision recorded under **Project status > Open questions**,
  update that section in the same commit and say so in the body. That section is this project's
  decisions log; there is no separate one.
- **One commit per work session, not one per logical change.** Batch everything outstanding into a
  single commit. Do not split a session's work into a chain of small ones — that inflates the commit
  count and makes the history harder to read, not easier. The subject names the headline change and
  the body's bullets cover the rest. Split only when the user asks for it, or when one part must
  stay independently revertable (a risky change sitting alongside routine ones).
- Never commit purely to mark progress. If nothing is asked for and nothing is finished, there is no
  commit to make.
- Do NOT add "Co-Authored-By: Claude" or any Claude/Anthropic attribution line in the commit message.
- Do NOT include the "🤖 Generated with Claude Code" footer or any equivalent.
- All commits should be authored under the user's own git identity — never configure or leave a
  separate Claude author/committer identity in git.
- Commit only when the user explicitly prompts Claude to commit — never as a side effect of finishing
  other work.
