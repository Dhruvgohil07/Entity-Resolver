# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Five stages implemented, each with its own tests (`pytest` → 186 passing). One later addition is not
covered — it is called out under the list, not folded into it:

- **`schema.py`** — canonical `Record` model (product-domain scope; `raw_attributes` is the escape
  hatch for unmapped source columns). Committed.
- **`normalize.py`** — `_fold` (NFKD + casefold + whitespace), unit-spelling and brand-alias
  canonicalization, model-number extraction, and the `NormalizedRecord` type (composition:
  `raw: Record` + derived fields).
- **`data/abt_buy.py`** — Abt-Buy loader: both sides → `Record`, ground-truth pairs → `entity_id`
  by union-find. Tested against committed fixtures, plus one integration test that runs only when
  the benchmark has been downloaded.
- **`eval/`** — `metrics.py` (PR curve, PR-AUC, precision@k, threshold evaluation — all taking
  `n_positives_total` so a pruned candidate set cannot flatter recall), `splits.py` (entity-grouped
  splitting, `count_true_pairs`), and `baseline.py` (the TF-IDF baseline; see **Baseline to beat**).
  Result committed at `reports/baseline_tfidf.md`.
- **`blocking/`** — eight modules, every one directly covered by tests: four blocker families
  (`standard` exact keys, `sorted_neighborhood`, `lsh`, `ann`) plus `union.py` (which combines them
  and computes pair completeness / reduction ratio), `pairs.py` (the packed-int64 representation),
  `base.py` (the `Blocker` contract), and `evaluate.py` (the CLI). Union pair completeness
  **0.9928** at reduction ratio 0.9644 on Abt-Buy — the recall ceiling every later stage inherits.
  Result committed at `reports/blocking.md`.

Implemented but **untested**: `data/__init__.py`'s `DATASETS` registry and `load_dataset()`, which
let a CLI resolve a benchmark by name so no later stage has to import a loader module. Its two error
paths — unknown dataset name, and a dataset directory that does not exist because `data/` is
gitignored — were checked by hand and never pinned by a test. Nothing under `tests/` imports it:
`test_data_abt_buy.py` and `test_blocking_evaluate.py` both call `dedup.data.abt_buy` directly, so
the registry is exercised only by the two CLIs (`dedup.eval.baseline`, `dedup.blocking.evaluate`)
and would not fail a test run if it broke.

The remaining directories — `features/`, `model/`, `cluster/`, `synth/`, `service/` — are all still
bare `__init__.py` scaffolding with no modules in them.

Next is **`features/`**, and it is now the only stage that can move: `model/` needs its vectors and
`cluster/` needs scored pairs from the model, so each is blocked on the one before it. `features/`
is unblocked because `blocking/` now emits candidate pairs for it to build vectors from — 84,117 of
them on Abt-Buy, carrying 0.9928 of all true pairs. That number is a **ceiling, not a score**: the
8 true pairs no blocker emitted are unrecoverable, so system recall cannot exceed it however good
the classifier becomes.

Two caveats on that 0.9928, both measured. It is a **full-catalog** figure; on the entity-grouped
split the union reaches 0.9949 on train and 1.0000 on test, and the test number is the ceiling that
actually binds when `features/` and `model/` are compared against the baseline's test-split F1
0.5204. And the blocker parameters were **swept over the same catalog they are scored on**, so it is
a best-of-sweep number, not a clean estimate — small bias here, but unquantified until parameters
are selected on train alone.

Commands in this file describe the intended contract — verify a command exists before relying on it,
and update this file as each phase lands.

### Open questions

Defects found so far in `normalize.py`, `schema.py` and `eval/metrics.py` are fixed, each pinned by
a regression test naming the failure mode. These judgment calls are deliberately left open, to be
settled against more data rather than treated as decided:

- **Evaluation uses the deduplication framing, not record linkage, and that costs comparability.**
  The baseline scores every pair of the combined catalog and calls a pair positive when the two
  records share an `entity_id` — so same-side pairs are candidates, and Abt-Buy has 1118 true pairs
  rather than the 1097 its mapping file ships. Published Abt-Buy figures instead score Abt rows
  against Buy rows only. The dedup framing is the right one for what this service actually does
  (CLAUDE.md's opening question is "given a catalog of N records", not "given two catalogs"), and it
  is what keeps every stage free of a per-source branch. But it means our numbers are near, not
  equal, to the literature's. Revisit when Amazon-Google lands: it has the same two-sided shape, so
  if the gap matters for comparison, the fix is a second reported number computed in `eval/`, never
  a side-aware branch in `blocking/` or `features/`.
- **The spec-suffix list in `_SPEC_UNIT_SUFFIXES` is conservative on purpose.** It rejects `1200W`
  and `12MP` as model numbers while deliberately omitting `wh` and `a`, which collide with real
  vendor codes (Bose 161WH, HP Officejet 8500A). A false model number fuses unrelated products into
  one block; a missing one only loses a signal — so the list should grow only against evidence.

Settled by measurement, recorded so it is not re-litigated:

- **Model-number blocking keys strip separators; the printed code is kept alongside.**
  `normalize.py` grew `model_number_key` next to `model_number` because Abt writes `KXTS208W` and
  Buy writes `KX-TS208W` for the same Panasonic phone. Both are correct as printed vendor codes, so
  neither field wins: `model_number` stays as the source wrote it for the review queue, and the
  stripped form is what blocks. Worth pair completeness 0.3354 → 0.5349 — about a fifth of
  achievable recall on the strongest key there is, previously lost to punctuation. The rule is
  `normalize.code_key`, public and shared: `blocking/standard.py` briefly carried its own copy built
  on `str.lower`, and two implementations that must agree is how train/serve skew starts. It
  decomposes NFKD internally rather than trusting the caller, because its callers disagree — the
  model-number path passes a code taken from the raw title, blocking passes tokens from the folded
  one, and without that a precomposed `Ü` is dropped whole while a decomposed one keeps its base
  letter, keying the same code two ways.
- **Three of the six blockers add no completeness the others do not already have.** Leave-one-out
  marginals against the committed six-blocker union, measured, not estimated:

  | blocker | marginal candidates | marginal PC |
  | --- | ---: | ---: |
  | `standard (model number)` | +0 | +0.0000 |
  | `standard (code tokens)` | +737 | +0.0000 |
  | `standard (rare tokens)` | +15,244 | +0.0089 |
  | `sorted_neighborhood` | +28,474 | +0.0045 |
  | `lsh (minhash)` | +18,317 | +0.0000 |
  | `ann (faiss HNSW)` | +2,345 | +0.0215 |

  The model-number blocker is *entirely* subsumed — every pair it finds, something else finds too,
  which follows from `code_token_keys` indexing every code-shaped token while extraction commits to
  one. LSH earns nothing for 18,317 candidates; `sorted_neighborhood` buys 0.0045 for 28,474.
  Only `ann`, `rare tokens` and `sorted_neighborhood` move the ceiling at all.

  None are deleted. A negative result someone can re-run is evidence; the same claim asserted from
  a deleted experiment is not, and all three are configuration- and dataset-specific. LSH in
  particular was measured with *token* shingles on six-to-ten-token titles, which is close to the
  worst case for Jaccard — `shingles="char"` and `synth/`'s token-drop corruption are both untested
  against it. Re-check every row of this table on `synth/` before dropping anything.
- **`ann` is the load-bearing blocker.** faiss HNSW over char-3gram TF-IDF reaches PC 0.9562 alone,
  beating every exact-key blocker combined, because it needs no shared token at all — it is what
  catches `Bose 161WH` against `Boss 161 Speaker`, a source typo in the brand. Its index is built
  **single-threaded on purpose**: parallel HNSW construction gave 14,608 / 14,603 / 14,602
  candidates across three runs of identical input, and a committed report whose numbers drift is not
  reproducible.
- **What blocking still misses is a different identifier system, not a near-miss.** Of 1,118 true
  pairs, 8 survive nothing. They are two failure modes, and neither is fixable by tuning a window or
  a threshold: (a) vendor SKU against distributor part number — `Canon Color Ink Tank - CL41CL` vs
  `Canon Ink Cartridge For PIXMA iP1600 ... - 0617B002`, two disjoint numbering schemes for one
  product; and (b) a truncated marketplace title carrying no code at all — `LG Over-The-Range White
  Microwave Oven - LMV1680WH` vs `LG 1.6 cu.ft. Over the Range`. Closing (a) needs a
  manufacturer-part-number cross-reference, which is data this project does not have; closing (b)
  needs the description, which `features/` will have and blocking does not.
- **The baseline compares normalized title only; adding description makes it worse.** Measured on
  Abt-Buy: title alone gives test F1 0.5204 / PR-AUC 0.4720, title + description gives 0.4349 /
  0.2898 — and P@10 collapses from 0.600 to 0.000, so the very top of the ranking is what breaks.
  The cause is a measured length asymmetry, not prose quality: Abt descriptions average 249
  characters and are never empty, Buy's average 34 (median 14) and are empty on 40% of rows. So
  concatenation makes an Abt vector that is mostly description face a Buy vector that is mostly
  title, diluting exactly the true pairs it was meant to help. What the two sides do share is
  category vocabulary — `finish` appears in 759 descriptions, `black` in 622, `digital` in 386 —
  which lifts *unrelated* pairs instead. Description is not worthless; it belongs in `features/`
  as its own signal with a missingness indicator, not glued onto the title string.
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
                  __init__.py holds DATASETS, the name -> loader registry
  eval/           metrics (PR-AUC, precision@k), entity-grouped splits, the TF-IDF baseline
  blocking/       standard, sorted_neighborhood, lsh, ann, union + evaluate
                  pairs.py packs candidate pairs as int64 i*n+j — 8 bytes each, so the same
                  code survives the jump to synth/ scale; base.py holds the Blocker contract
  features/       string, numeric, semantic, missingness
  model/          train, calibrate, threshold (cost model)
  cluster/        components, correlation, agglomerative, bcubed
  synth/          corruption engine for synthetic scale-up
  service/        FastAPI app, HNSW + inverted index, review queue
reports/          blocking table, PR curves, cost curves — the defensible results
  baseline_tfidf.md   the TF-IDF number every later stage is measured against
  blocking.md         blocker x completeness x reduction; the union row is the recall ceiling
```

`eval/metrics.py` is where the metric invariants below are actually enforced, so a new stage should
import from it rather than calling sklearn directly — `sklearn.metrics.precision_recall_curve`
divides recall by the positives it can see, which is wrong for every post-blocking candidate set.

The same trap has a second mouth, and `blocking/` will meet both: **any metric whose denominator is
the surviving candidate count rewards discarding candidates.** `precision_at_k` therefore divides by
the requested k, never by how many pairs are left — an audit caught it doing the latter, which let a
floor that discarded 98% of pairs report P@100 = 0.611 for the same 11 hits that score 0.11
unpruned. Pair completeness and reduction ratio must always be reported together for the same
reason: either number alone is trivially gamed by moving the threshold.

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

TF-IDF char-3gram cosine with a single threshold — built, measured, and committed at
`reports/baseline_tfidf.md`. **Abt-Buy, normalized title only: test F1 0.5204** (P 0.4605,
R 0.5982, PR-AUC 0.4720), entity-grouped split at `test_fraction=0.3, seed=0`. Every later stage is
justified against that number.

Protocol, so a later comparison is like-for-like:

- Vectorizer fit on the **train** split only, threshold chosen on the **train** split only, both
  spent on test. Fitting IDF on the full catalog scores better and is a quiet leak.
- Recall is divided by all true pairs in the split, not by the ones that survived into the
  candidate set.
- Scoring is the full N² upper triangle — no blocking, so no recall ceiling above the similarity
  function. That is what makes it the right reference for `blocking/`.
- This is the **deduplication** framing: all 2173 records, a pair is positive when the two records
  share an `entity_id`. That makes 1118 true pairs, not the 1097 the mapping file ships (the 21
  size-3 clusters each imply a third pair), and same-side pairs are candidates. Published Abt-Buy
  figures are for the cross-source task, so they are close but not directly comparable.

## Commands

The dev environment is `.venv` (deliberately Python 3.12, not the machine default — ML wheel
availability). On Windows, prefix with `.venv/Scripts/python -m` if the venv is not active.

These work today:

```bash
# environment
pip install -e ".[dev]"

# tests
pytest                                  # all (186 passing)
pytest tests/test_normalize.py          # one file
pytest tests/test_normalize.py::test_model_number_trailing_convention   # one test
pytest -k model_number                  # by keyword

# lint (line-length 100, target py310)
ruff check src tests

# load the benchmark (see Data above for the download)
python -c "from dedup.data import load_dataset; print(len(load_dataset('abt-buy')))"

# the TF-IDF baseline: prints the report, --out also writes it
python -m dedup.eval.baseline --dataset abt-buy --out reports/baseline_tfidf.md

# the blocking table: blocker x pair completeness x reduction ratio
python -m dedup.blocking.evaluate --dataset abt-buy --out reports/blocking.md
```

Tests run without any dataset present: they use committed fixtures under `tests/fixtures/abt-buy/`.
Three tests read `data/raw/` and skip when the benchmark has not been downloaded — the loader's
integration test in `tests/test_data_abt_buy.py`, the one in `tests/test_eval_baseline.py` that
re-derives the published baseline F1, and the one in `tests/test_blocking_evaluate.py` that
re-derives the published union pair completeness. So a green run does *not* by itself mean the real
files were checked, and in particular does not mean either published number was reproduced —
`pytest -rs` reports the skips.

If `tests/fixtures/abt-buy/` ever needs a new shape, regenerate it rather than hand-editing:
`Abt.csv` must stay cp1252-encoded on disk, which an editor will silently undo.

```bash
python tests/fixtures/make_fixtures.py
```

Not built yet — intended contract, will fail if invoked:

```bash
# pipeline stages
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
`blocking`, `eval`. Reserved for modules not yet built: `features`, `model`, `cluster`, `service`,
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
