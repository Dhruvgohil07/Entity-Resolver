# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

An entity-resolution / deduplication service for e-commerce product listings. It answers two
questions:

1. **Batch:** given a catalog of N records, which sets of records refer to the same real-world
   product?
2. **Online:** given one new record, which existing records does it likely duplicate, and with what
   confidence?

The output is not a boolean. Scored pairs are routed into **auto-merge / review / auto-reject**
bands by a cost model, clustered into entities, and served over a FastAPI + DuckDB API with a review
queue. `README.md` is the detailed code reference; `reports/` holds the measured results;
`docs/DECISIONS.md` is the decisions log -- open questions and measured, settled calls. Read the
relevant entry there before changing anything it covers. `docs/plans/` holds work plans.

## Plans

Every plan made for this project is saved in `docs/plans/` -- roadmaps, implementation plans,
plan-mode output -- not left only in chat.

- One file per plan, named `YYYY-MM-DD-<short-slug>.md` (the date it was made).
- Start with the date, a **Status** line (`proposed`, `in progress`, `done`, `superseded by <file>`)
  and its source; list tasks as `- [ ]` checkboxes with effort, dependencies and a "done when".
- Keep it current: tick tasks and update the status as work lands, in the same commit as the work.
  Supersede a plan with a new file rather than rewriting its history.
- Before starting planned work, check `docs/plans/` for an existing plan that covers it.

## Setup

The dev environment is `.venv`, deliberately **Python 3.12** (not the machine default) for ML wheel
availability. On Windows, run tools as `.venv/Scripts/python -m <tool>` if the venv is not active.

```bash
pip install -e ".[dev]"
```

Benchmark data is gitignored, so a fresh clone starts without it:

```bash
B=https://raw.githubusercontent.com/dchud/ddbench/HEAD/data
mkdir -p data/raw/abt-buy
for f in Abt.csv Buy.csv abt_buy_perfectMapping.csv; do
  curl -sS -o "data/raw/abt-buy/$f" "$B/abt-buy/$f"
done
```

## Commands

```bash
# tests and lint
pytest                                   # full suite
pytest tests/test_normalize.py           # one file
pytest -k model_number                   # by keyword
pytest -rs                               # show skip reasons
ruff check src tests                     # line-length 100, target py310

# report CLIs -- each prints markdown, --out also writes it; --dataset abt-buy | synth-20k | synth-200k
python -m dedup.eval.baseline     --dataset abt-buy --out reports/baseline_tfidf.md
python -m dedup.blocking.evaluate --dataset abt-buy --out reports/blocking.md
python -m dedup.features.evaluate --dataset abt-buy --out reports/features.md
python -m dedup.model.evaluate    --dataset abt-buy --out reports/model.md --save-scorer artifacts/scorer
python -m dedup.cluster.evaluate  --dataset abt-buy --out reports/cluster.md

# service: batch run -> API -> enable online lookup for a run
python -m dedup.service.batch --dataset abt-buy --scorer artifacts/scorer --db data/service.duckdb
DEDUP_DB_PATH=data/service.duckdb uvicorn dedup.service.app:app --reload
python -m dedup.service.build_index --db data/service.duckdb --run-id <run_id> --index-root artifacts/index/<run_id>

# synthetic catalogs (gitignored under data/synth/)
python -m dedup.synth.generate --seed-dataset abt-buy --records 20000 --out data/synth/abt-buy-train-20k --report reports/synth/realism.md
```

Each committed report ends with a "Regenerate with" block recording the exact command and flags that
produced it -- use that, not the examples above, when regenerating a report. Useful flags:
`--cost-false-merge/--cost-false-split/--cost-review` (model, cluster, batch), `--without`,
`--leave-one-out`, `--ann-components`, `--ann-neighbours`, `--lsh-max-neighbours` (blocking),
`--min-similarity` (baseline), `--semantic` (features; downloads ~90 MB of weights).

## Architecture

A per-dataset loader feeds a five-stage pipeline. Each stage constrains the ones after it:

```
raw CSV (one shape per dataset)
  -> data/          source columns -> canonical Record; ground-truth pairs -> entity_id
  -> normalize.py   NFKD, case, units, brand aliases, model-number extraction
  -> blocking/      N^2 pairs -> candidate pairs        [sets a HARD RECALL CEILING]
  -> features/      candidate pair -> feature vector (33 columns, 34 with --semantic)
  -> model/         vector -> CALIBRATED probability -> cost-based band
  -> cluster/       scored pair graph -> entity clusters
  -> service/       DuckDB store, batch runs, review queue, online lookup (FastAPI)
```

```
src/dedup/
  schema.py        canonical Record model -- every dataset maps into it
  normalize.py     shared by batch and serve paths
  data/            loaders + DATASETS registry; notes.py / *_notes.py = what a report may say
                   about a dataset; synthetic.py = JSONL catalogs with hash-checked manifests
  eval/            metrics (PR-AUC, precision@k), entity-grouped splits, TF-IDF baseline
  blocking/        standard, sorted_neighborhood, lsh, ann, union; defaults.py is the shared
                   blocker set (block_split for evaluation, block_unlabeled for real catalogs);
                   pairs.py packs pairs as int64 i*n+j; ann.AnnIndex / standard_index.InvertedIndex
                   are the incrementally-queryable indexes online lookup uses
  features/        string, numeric, semantic, missingness blocks; vectorize.PairFeaturizer
  model/           train (PairScorer = featurizer + LightGBM + calibrator, save/load),
                   calibrate (out-of-fold Platt default; beta, isotonic), threshold (CostModel)
  cluster/         base (expected-cost objective), components, agglomerative (average linkage,
                   the default), correlation, bcubed
  synth/           synthetic catalog generator seeded from Abt-Buy's train split
  service/         store (DuckDB), batch, app (FastAPI), lookup (pure decision logic),
                   review, build_index, schemas
reports/           committed results; reports/synth/ for the synthetic catalogs
```

## Invariants

Correctness traps specific to this problem. Violating them produces results that look good and are
wrong.

- **Split by entity, never by pair.** Use `eval/splits.py`; it groups on `Record.split_group` where
  present (synthetic families) and `entity_id` otherwise. A pair-level split leaks.
- **PR-AUC and precision@k, never ROC-AUC.** Post-blocking imbalance is extreme. Import metrics from
  `eval/metrics.py`, not sklearn directly: every metric takes `n_positives_total`, because sklearn's
  PR curve divides recall by the positives it can see, which flatters any pruned candidate set.
- **Any metric whose denominator is the surviving candidate count rewards discarding candidates.**
  `precision_at_k` divides by k, and pair completeness is always reported with reduction ratio.
- **Blocking sets a ceiling nothing downstream can lift.** When recall disappoints, check blocking
  pair completeness before touching the model.
- **Model output must be calibrated.** `PairScorer.probabilities` refuses to run without a
  calibrator, and calibrators must be strictly monotone (never reorder or flatten pairs).
- **Thresholds come from expected cost, not argmax F1.** `p_hi = 1 - C_review/C_fm`,
  `p_lo = C_review/C_fs`; default cost ratio 20 : 2 : 1. False merges are far more expensive than
  false splits.
- **Missing values get explicit indicator features.** Every imputed column declares a
  `companion_indicator`; the featurizer refuses to build otherwise. Reporting code must also compute
  over covered pairs only -- a fill value is not a mismatch.
- **Fit on train, spend on test.** Vectorizers, featurizers, thresholds and calibrators are fit on
  the train split (calibration out-of-fold) and only applied to test.
- **Cluster quality is B-cubed**, measured separately from pairwise metrics. Connected components
  chains (one bad edge fuses two clusters); that failure is demonstrated in reports, not hidden.

## Conventions

- **`data/` is the only place that may know a dataset's name.** Everything downstream sees `Record`s.
  A per-source branch elsewhere is a bug; `tests/test_data_registry.py` enforces the import rule.
- **`normalize.py` runs identically in batch and at serve time.** Normalization that exists only in
  the training path is train/serve skew. Shared rules live in one place (e.g. `normalize.code_key`);
  never duplicate them in a blocker or feature.
- **Report text is chosen from measured values.** Renderers branch on the measurement rather than
  printing a claim unconditionally; dataset-specific prose belongs in that dataset's `DatasetNotes`.
- **Reproducibility over speed.** The `ann` HNSW index is built single-threaded on purpose (parallel
  builds give different candidate counts); synthetic generation is seeded by SHA-256, never `hash()`.
- **Persisted artifacts are hash-checked.** Scorers, indexes and synthetic catalogs are saved with a
  `manifest.json` and refuse to load on a hash mismatch.
- **Changing `normalize.py` moves every downstream number.** After such a change, regenerate the
  affected reports with their recorded commands and state what moved and why.

## Data gotchas

- `Abt.csv` (and Amazon-Google's CSVs) are **cp1252**, not UTF-8. Never paper over this with
  `errors="replace"`.
- Abt has no brand column, so `brand` is `None` on every Abt record. Empty price means `None`,
  never NaN.
- Evaluation uses the **deduplication framing** (all records, same-side pairs included): Abt-Buy has
  1118 true pairs, not the 1097 its mapping file lists, so numbers are close to but not directly
  comparable with published cross-source figures.
- The committed `synth-20k` / `synth-200k` catalogs are **fixed artifacts**: re-running the generator
  today writes a different catalog because extraction has changed since. Use
  `synth.generate --report-only` to re-render `realism.md` without rewriting the catalog.
- Never hand-edit `tests/fixtures/abt-buy/` (an editor will silently undo the cp1252 encoding);
  regenerate with `python tests/fixtures/make_fixtures.py`.

## Testing

- The suite runs without the benchmark, using committed fixtures. Tests that re-derive published
  figures are marked `@needs_data` and **skip** when `data/raw/` is absent, so a green run alone does
  not prove any published number reproduced -- check `pytest -rs`.
- One test always skips unless the sentence-transformers weights are downloaded (`--semantic` is
  opt-in).
- `tests/test_cli_smoke.py` covers every CLI's plumbing; `tests/test_pipeline_e2e.py` covers the
  handoffs between stages, including the saved-scorer seam between `model/` and `service/`.
- Add a regression test named after the failure mode for every bug fixed.

## Current state

All stages through `service/` (batch dedup, review queue, online lookup) are implemented and tested.
Headline results: TF-IDF baseline test F1 **0.5204**; pipeline test F1 **0.8892** (PR-AUC 0.9472);
blocking union pair completeness **0.9928** on Abt-Buy -- see `reports/` before quoting any of them.

Not built yet: the htmx review UI (no Node toolchain), the Amazon-Google loader, and persistent
entity identity across batch runs (`cluster_id` is only stable within one run). Measured but
unresolved questions are tracked under "Open questions" in `docs/DECISIONS.md`.

## Commit messages

Conventional Commits: `<type>(<scope>): <imperative summary, <=72 chars>`, then a body explaining
*why*, especially for modelling or threshold changes.

- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `data`, `chore`.
- Scopes: `schema`, `normalize`, `data`, `blocking`, `eval`, `features`, `model`, `cluster`,
  `data-gen` (`synth/`), `service`. A commit touching no single module takes no scope.
- If a commit settles or revisits an entry in `docs/DECISIONS.md`, update that entry in the same
  commit and say so in the body.
- One commit per work session: batch outstanding work into a single commit with the headline as the
  subject and the rest as bullets. Split only when asked or when part must stay revertable alone.
- Commit only when the user explicitly asks. Never commit just to mark progress.
- No `Co-Authored-By: Claude`, no "Generated with Claude Code" footer, no Claude/Anthropic
  attribution of any kind. Commits are authored under the user's own git identity.
