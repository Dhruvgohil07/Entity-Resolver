# Entity Resolver — project reference

A personal reference for what exists in this repo, what each file and function does,
and what is still missing. Written 2026-09-08.

For the *rules* the project is built to (invariants, commit conventions, stack
rationale), see `CLAUDE.md` — this file describes the code as it stands, not the
policy it follows.

---

## 1. What this project is

An **entity-resolution / deduplication** system for e-commerce product listings.

The same physical product gets listed many times across vendors, worded differently,
with no shared identifier:

```
Abt:  "Sony Turntable - PSLX350H"
Buy:  "Sony PS-LX350H Belt-Drive Turntable, Black"
```

Same product. Two words in common, model number spelled two ways, one has a price and
the other doesn't. Meanwhile these two are one character apart and are **different**
products:

```
"Sony WH-1000XM4 Wireless Headphones"
"Sony WH-1000XM5 Wireless Headphones"
```

That tension — paraphrase that means *same*, tiny differences that mean *different* —
is the entire problem.

The system answers two questions:

1. **Batch** — given a catalog of N records, which sets of records are the same product?
2. **Online** — given one new record, which existing records does it duplicate, and with
   what confidence?

The output is not a boolean. Pairs are routed into **auto-merge / review / auto-reject**
bands, because a false merge (fusing two real products, corrupting the catalog) costs far
more than a false split (a duplicate survives). Human decisions from the review queue flow
back as training labels.

---

## 2. Status at a glance

| Stage | Directory | State |
| --- | --- | --- |
| Canonical record | `schema.py` | ✅ Done |
| Normalization | `normalize.py` | ✅ Done |
| Dataset loading | `data/` | ✅ Done (Abt-Buy) |
| Evaluation + baseline | `eval/` | ✅ Done |
| Blocking | `blocking/` | ✅ Done (6 blockers + union) |
| Features | `features/` | ✅ Done (33 columns + diagnostics) |
| Model | `model/` | ⬜ Empty scaffolding |
| Clustering | `cluster/` | ⬜ Empty scaffolding |
| Synthetic data | `synth/` | ⬜ Empty scaffolding |
| Service / API / UI | `service/` | ⬜ Empty scaffolding |

**Tests: 250 passing, 1 skipped** — the Abt-Buy benchmark is downloaded so every
integration test actually runs; the single skip is the sentence-transformers column, which
is opt-in and whose weights are not downloaded.

Six of nine stages are built. What is done is the half that decides whether the rest can be
trusted: canonical shape, normalization, honest metrics, an honest baseline, candidate
generation with a measured recall ceiling, and now a pair vector whose missingness handling
is enforced at construction rather than by review.

---

## 3. File structure

```
Entity-Resolver/
├── CLAUDE.md                       project rules, invariants, decisions log
├── README.md                       this file
├── pyproject.toml                  deps, ruff (line 100, py310), pytest config
│
├── data/raw/abt-buy/               the benchmark CSVs (gitignored)
│   ├── Abt.csv                     1081 rows, cp1252 encoded
│   ├── Buy.csv                     1092 rows
│   └── abt_buy_perfectMapping.csv  1097 ground-truth pairs
│
├── reports/                        the defensible results
│   ├── baseline_tfidf.md           TF-IDF baseline — the number to beat
│   └── blocking.md                 blocker × completeness × reduction table
│
├── src/dedup/
│   ├── schema.py                   ✅ canonical Record model
│   ├── normalize.py                ✅ text folding, units, brands, model numbers
│   │
│   ├── data/                       ✅ raw CSV → Record
│   │   ├── __init__.py             DATASETS registry, load_dataset()
│   │   └── abt_buy.py              the Abt-Buy loader
│   │
│   ├── eval/                       ✅ metrics, splits, baseline
│   │   ├── metrics.py              PR curve, PR-AUC, precision@k
│   │   ├── splits.py               entity-grouped train/test splitting
│   │   └── baseline.py             TF-IDF char-3gram cosine baseline
│   │
│   ├── blocking/                   ✅ N² pairs → candidate pairs
│   │   ├── base.py                 the Blocker contract, timing
│   │   ├── pairs.py                packed int64 pair representation
│   │   ├── standard.py             exact-key blocking (3 key functions)
│   │   ├── sorted_neighborhood.py  sort + sliding window
│   │   ├── lsh.py                  MinHash-LSH
│   │   ├── ann.py                  FAISS HNSW over char TF-IDF
│   │   ├── union.py                combine blockers, score them
│   │   └── evaluate.py             CLI → reports/blocking.md
│   │
│   ├── features/                   ✅ pair → 33-column vector + diagnostics
│   │   ├── base.py                 FeatureSpec/FeatureMatrix contract, presence predicates
│   │   ├── string.py               title, vendor code, description, brand similarity
│   │   ├── numeric.py              price and digit-token agreement
│   │   ├── semantic.py             sentence-transformers cosine (opt-in)
│   │   ├── missingness.py          the 8 indicator columns
│   │   ├── vectorize.py            PairFeaturizer: fit/transform, registry validation
│   │   └── evaluate.py             CLI → reports/features.md
│   │
│   ├── model/                      ⬜ vector → calibrated probability → bands
│   ├── cluster/                    ⬜ scored pair graph → entity clusters
│   ├── synth/                      ⬜ corruption engine for scale-up
│   └── service/                    ⬜ FastAPI + htmx review queue
│
└── tests/                          251 tests (250 pass, 1 skips without the ST weights)
    ├── fixtures/                   committed mini-CSVs (tests run with no dataset)
    │   └── make_fixtures.py        regenerates them (Abt.csv must stay cp1252)
    ├── test_schema.py              15
    ├── test_normalize.py           28
    ├── test_data_abt_buy.py        22
    ├── test_eval_metrics.py        17
    ├── test_eval_splits.py         13
    ├── test_eval_baseline.py       14
    ├── test_blocking_pairs.py      13
    ├── test_blocking_blockers.py   16
    ├── test_blocking_evaluate.py   15
    ├── test_features_string.py     18
    ├── test_features_numeric.py    11
    ├── test_features_vectorize.py  18
    └── test_features_evaluate.py   18
```

---

## 4. The pipeline, end to end

```
raw CSV (one shape per dataset)
   │
   ▼  data/          source columns → canonical Record; ground truth → entity_id
   │
   ▼  normalize.py   unicode fold, units, brand aliases, model-number extraction
   │
   ▼  blocking/      N² pairs → candidate pairs        ← HARD RECALL CEILING
   │
   ▼  features/      candidate pair → 33 numbers       ✅ built
   │
   ▼  model/         numbers → calibrated probability → band   ⬜ not built
   │
   ▼  cluster/       scored pair graph → entity clusters       ⬜ not built
```

Three stage boundaries matter more than the rest:

- **Blocking sets a ceiling nothing downstream can lift.** A true pair no blocker emits is
  never scored, never seen by the model, never reaches clustering. When system recall
  disappoints, check blocking *before* touching the model.
- **The classifier scores pairs; the service returns entities.** `cluster/` is the
  translation layer, and good pairwise F1 does not imply good clusters.
- **`data/` is the only place allowed to know a dataset's name.** Everything after it sees
  `Record` objects. A per-source branch downstream is a bug, not a shortcut.

---

## 5. What each file does, function by function

### 5.1 `schema.py` — the canonical record

Defines the one shape every dataset is flattened into. Nothing downstream ever sees a
CSV column name.

| Name | What it is |
| --- | --- |
| `SourceName` | `Literal["abt_buy", "amazon_google", "synthetic"]` — the sources a loader may claim |
| `NonBlankStr` | String type that strips whitespace *then* requires length ≥ 1, so `" "` fails validation instead of travelling downstream |
| `Record` | The canonical product listing (Pydantic model) |
| `Record._price_is_finite_and_non_negative` | Field validator rejecting NaN, infinity and negative prices |

**`Record` fields:**

| Field | Type | Notes |
| --- | --- | --- |
| `record_id` | `NonBlankStr` | Globally unique, source-prefixed: `"abt_buy:abt:10"` |
| `source` | `SourceName` | Which loader produced it |
| `entity_id` | `str \| None` | Ground-truth cluster id; `None` at serve time |
| `title` | `NonBlankStr` | Required — the one field never allowed empty |
| `description` | `str \| None` | `None` = column absent, `""` = column present but empty |
| `brand` | `str \| None` | Same convention |
| `category` | `str \| None` | Same convention |
| `price` | `float \| None` | `None` for missing, never NaN |
| `raw_attributes` | `dict` | Escape hatch for unmapped source columns |

**Three design decisions worth remembering:**

- `extra="forbid"` — a loader passing a misspelled or unmapped field **crashes** rather
  than silently dropping data that would resurface as unexplained recall loss later.
- **`None` vs `""` is load-bearing.** `None` means the source had no such column; `""`
  means the column existed and was empty. `features/missingness.py` will key off this,
  so loaders must never collapse one into the other.
- **NaN is rejected.** NaN survives a bare `v < 0` check and then propagates silently
  through an entire feature vector with no error to trace back.

---

### 5.2 `normalize.py` — canonical text form

Runs **identically in the batch and serve paths**. It is a pure function with no branching
on `record.source`, so it behaves the same for a source it has never seen. Any
normalization that existed only in the training path would be a train/serve skew bug.

| Function | What it does |
| --- | --- |
| `_fold(text)` | Unicode NFKD + `casefold()` + collapse whitespace. `casefold`, not `lower` — it's the correct primitive for case-insensitive comparison |
| `_fold_optional(text)` | `_fold` that passes `None` through |
| `_canonicalize_units(text)` | Applies `_UNIT_RULES`: `"4.2 Cu. Ft."` → `"4.2 cu ft"`, `32'` → `32 in`. Spelling only, never conversion |
| `_normalize_brand(folded)` | Looks up `_BRAND_ALIASES`: `"hewlett-packard"` → `"hp"` |
| `_clean_token(token)` | Strips trailing `,./=` punctuation from a token |
| `_is_spec_token(token)` | True if the token is `<digits><unit>` — `1200W`, `12MP`, `500GB` |
| `_qualifies_as_model_number(token)` | Gate: 4-20 chars, alphanumeric + hyphens, has **both** a letter and a digit, and is not a spec |
| `_extract_model_number(title)` | The main heuristic — returns the vendor code, uppercased |
| `_model_number_key(model_number)` | Comparison form: lowercase, separators stripped. `"KX-TS208W"` → `"kxts208w"` |
| `normalize(record)` | `Record` → `NormalizedRecord`. The only public entry point |

**`NormalizedRecord` fields** — composition, not inheritance. It holds the untouched raw
`Record` plus derived fields, so a later stage wanting the original price never has to
guess whether a field was normalized in place.

| Field | Example |
| --- | --- |
| `raw` | the original `Record`, untouched |
| `normalized_title` | `"sony turntable - pslx350h"` |
| `normalized_description` | folded description, or `None` |
| `normalized_brand` | alias-resolved brand, or `None` |
| `model_number` | `"PSLX350H"` — as the source printed it, for display |
| `model_number_key` | `"pslx350h"` — comparison form, for blocking |

**How `_extract_model_number` decides**, in preference order — each grounded in a
convention observed in real titles:

1. **Trailing** `"<description> - <CODE>"` (Abt/Buy style). The candidate must be the
   token following the *final* `" - "`, not merely the last token — otherwise
   `"Canon PowerShot Camera - 12MP Black"` yields a spec.
2. **Leading** `"<CODE> <description>"` (Google style).
3. **Fallback** — the first qualifying token anywhere in the title.

**Why two model-number fields.** `model_number` preserves the printed form for a human
reading the review queue; `model_number_key` answers "are these the same code". Abt writes
`KXTS208W`, Buy writes `KX-TS208W` — the same Panasonic phone. **Measured: keying blocking
on the stripped form lifted pair completeness from 0.3354 to 0.5349.** A fifth of the
achievable recall on the strongest signal there is, lost to punctuation alone.

**Why the spec list is deliberately short.** `_SPEC_UNIT_SUFFIXES` rejects `1200W` and
`12MP` but intentionally omits `wh` and `a`, which collide with real vendor codes (Bose
161WH, HP Officejet 8500A). A false model number fuses unrelated products into one block;
a missing one only loses a signal. The list grows only against evidence.

**Why `'` folds to inches, not feet.** Typographically `'` is the foot mark. Measuring the
actual CSVs overturned that: all 249 digit+`'` occurrences across `Abt.csv` and `Buy.csv`
are screen and driver sizes (`3.0' LCD Display`, `4' x 6' Print Paper`), and none is a
length in feet. Normalize follows the source convention, not the typographic one.

---

### 5.3 `data/` — dataset loaders

The only place in the codebase permitted to know a dataset's name.

#### `data/__init__.py`

| Name | What it does |
| --- | --- |
| `DatasetSpec` | Frozen dataclass: a dataset's `name`, `default_root`, and `load` callable |
| `DATASETS` | The registry — `{"abt-buy": DatasetSpec(...)}` |
| `load_dataset(name, root=None)` | Resolve a dataset by name, load it with `entity_id` populated. Raises a readable error if the directory is missing (data/ is gitignored, so a fresh clone starts empty) |

The registry is what lets a CLI say `--dataset abt-buy` while `eval/`, `blocking/` and
`model/` never import a loader module or learn a benchmark's name.

#### `data/abt_buy.py`

| Function | What it does |
| --- | --- |
| `_read_rows(path, expected_columns)` | Read a CSV as cp1252, **reject** the file if its column set isn't exactly what's expected |
| `_parse_price(raw)` | `"$1,999.00"` → `1999.0`; `""` → `None`; anything else → `ValueError` |
| `_record(row, side, has_brand_column)` | One CSV row → one `Record` (this call is where schema validation happens) |
| `load_products(root)` | Both sides → `list[Record]`, `entity_id` still unset |
| `load_pairs(root)` | The mapping CSV → `[(abt_record_id, buy_record_id), ...]` |
| `assign_entity_ids(records, pairs)` | **Union-find** over the pair graph → cluster ids. Contains a nested `find()` with path compression |
| `load_abt_buy(root)` | The composed entry point: `load_products` + `load_pairs` + `assign_entity_ids` |

**Measured properties of Abt-Buy** — each verified by a test rather than assumed, and each
one a defect waiting to happen:

- **`Abt.csv` is cp1252, not UTF-8.** A default `pd.read_csv` raises on it. Not "fixed"
  with `errors="replace"` — that mojibakes product names into mismatches instead of
  failing loudly. `latin-1` is also refused: it decodes every byte, so a genuine encoding
  change would corrupt silently rather than raise.
- **The two sides have different columns.** Abt has no brand column at all; Buy has
  `manufacturer` on all but 6 of 1092 rows. So `brand` is `None` on every Abt record —
  which makes brand equality useless both as a feature (missing on every pair) and as a
  blocking key on this benchmark.
- **Price is `"$399.00"`**, with a thousands comma on 97 of 1008 populated values. 61% of
  Abt and 46% of Buy rows leave it empty.
- **Union-find is safe here, and that was checked.** 1097 pairs → 1076 clusters: 1055 of
  size 2, 21 of size 3, **no runaway component**. The size-3 clusters are one Abt product
  against two duplicate Buy listings — genuine. Re-check that distribution before reusing
  connected components on anything noisier.
- **Entity ids are numbered by each cluster's smallest `record_id`**, so they are stable
  between runs. An id that shifted would silently reshuffle a cached train/test split.

---

### 5.4 `eval/` — metrics, splits, baseline

#### `eval/metrics.py`

Two things here are deliberately **not** what a default sklearn call gives you.

| Name | What it does |
| --- | --- |
| `PrecisionRecallCurve` | Frozen dataclass: `thresholds`, `precision`, `recall`, `n_positives_total` |
| `ThresholdPoint` | One operating point: `threshold`, `precision`, `recall`, `f1` (with a readable `__str__`) |
| `_f1(p, r)` | F1, guarded against 0/0 |
| `_as_arrays(scores, labels)` | Coerce to numpy, check shapes, **reject NaN/inf scores** |
| `_resolve_total(labels, n)` | Resolve the recall denominator; raise if the claimed total is below what's present |
| `precision_recall_curve(...)` | P and R at every *reachable* threshold, ranked best first |
| `average_precision(curve)` | PR-AUC as a step-wise sum, matching sklearn's `average_precision_score` |
| `best_f1(curve)` | The highest-F1 operating point on a curve |
| `evaluate_at_threshold(...)` | Apply a fixed threshold (`score >= threshold`) to a scored set |
| `precision_at_k(scores, labels, k)` | Purity of the top-k — the review-queue metric |

**The four traps this module exists to close:**

1. **PR-AUC, never ROC-AUC.** After blocking there are thousands of negatives per positive.
   ROC-AUC divides by that huge negative count, so a model ranking garbage above half the
   true pairs still reads ~0.99.
2. **The candidate set is pruned; the recall denominator is not.** Every function takes
   `n_positives_total` — the true pairs in the whole catalog. Divide by the positives that
   happen to be present and *pruning harder improves your score*, which is exactly
   backwards.
3. **Only reachable operating points are reported.** A threshold admits *every* pair
   sharing that score, so only the last index of each run of equal scores is a real cut.
   Reporting intermediate ranks claims a precision no threshold can deliver — and with
   char-3gram cosine, exact ties are common.
4. **`precision_at_k` divides by the requested k, never by candidates available.** Clamping
   k to `scores.size` turns "precision at 100" into "precision at however many pairs
   survived pruning" — the same backwards incentive as (2). Fewer than k candidates means
   empty seats in the queue, and an empty seat is not a hit. Ties straddling position k are
   resolved **by expectation**, not by array order, so the answer doesn't depend on the
   order the loader emitted rows.

NaN scores are rejected rather than tolerated: `nan >= threshold` is False at every cut, so
a NaN-scored true pair can never be admitted by any threshold while still counting against
recall — the curve quietly reports a maximum that is not reachable.

#### `eval/splits.py`

| Function | What it does |
| --- | --- |
| `group_by_entity(records)` | Bucket records by `entity_id`; **raise** if any record is unlabeled |
| `count_true_pairs(records)` | Σ C(size, 2) over entity groups — the recall denominator |
| `split_by_entity(records, test_fraction=0.3, seed=0)` | Partition into (train, test) so **no entity spans both sides** |
| `true_pair_ids(records)` | Every same-entity pair as frozensets — for tests and debugging only |

**The project's first invariant lives here: split by entity, never by pair.** A pair-level
split puts (A1, A2) in train and (A1, A3) in test — the model has already seen A1's exact
title string at test time. Reported F1 climbs and none of it transfers.

Entity ids are **sorted before shuffling**, so the split depends only on the seed and the
set of entities — not on the order the loader happened to return rows in.

`count_true_pairs` returns **1118** for Abt-Buy, not the 1097 the mapping file ships: the
21 size-3 clusters each imply a third pair by transitivity.

#### `eval/baseline.py`

The TF-IDF char-3gram cosine baseline — the number every later stage owes.

| Name | What it does |
| --- | --- |
| `comparison_text(record, include_description)` | The string being compared, built **through `normalize()`** so the baseline sits on the same canonical text as every later stage |
| `fit_vectorizer(records, ...)` | Fit IDF on these records only — pass the train split, never the catalog |
| `ScoredPairs` | Dataclass of scored pairs + the counts pruning would hide; `recall_ceiling` property |
| `score_pairs(records, vectorizer, ...)` | Cosine over the **full N² upper triangle**, in row chunks |
| `VariantResult` | One text configuration measured end to end |
| `BaselineReport` | The whole run: setup, counts, all variants |
| `run_variant(train, test, ...)` | Fit on train, choose the threshold on train, spend it on test |
| `run_baseline(records, ...)` | Split, then run both variants (title; title + description) |
| `_results_row(variant)` | One markdown table row |
| `_variant_detail(variant, n)` | Per-variant markdown detail block |
| `render_markdown(report)` | The `reports/` artifact — numbers plus the caveats that make them readable |
| `main(argv)` | CLI entry point |

**The protocol is deliberately unflattering to the baseline**, because a baseline that
flatters itself is not a bar:

- Vectorizer fit on **train only**; threshold chosen on **train only**; both spent on test.
  Fitting IDF on the full catalog scores better and is a quiet leak — same class as a
  pair-level split.
- Recall divided by **all** true pairs in the split, not the ones that survived scoring.
- The **test-set oracle** (best F1 chosen with test labels in hand) is printed alongside,
  so the gap is visible rather than accidentally claimed.

`analyzer="char_wb"`, not `"char"`: `char_wb` pads each word before slicing, so a short
model number contributes n-grams anchored to its own boundaries instead of being blended
into its neighbours. Cross-word grams mostly encode word *order* — exactly what vendor
titles reorder freely.

`lowercase=False`: `normalize.py` already casefolded. Leaving sklearn's default on would
bury a second, weaker case rule (`str.lower`, not `casefold`) inside the fitted artifact
where `normalize.py` cannot see it.

---

### 5.5 `blocking/` — candidate generation

The stage that sets the ceiling. It exists because N² doesn't scale: 2,173 records is
**2,359,878** pairs to find **1,118** real matches (1 in 2,100). At 1M records it's 500
billion pairs.

Because the README already covered this stage in less depth, this section leads with the
**concepts** — what blocking is and how each technique works — and then walks the code
file by file. If you only want the function reference, skip to *The code, file by file*.

---

#### The concepts

**The problem, stated plainly.** To find every duplicate in a catalog of N records you
would compare every record against every other — N(N−1)/2 pairs. That number grows with
the *square* of the catalog: doubling the catalog quadruples the work. At Abt-Buy's 2,173
records it is 2.36 million comparisons; at 200,000 records it is 20 billion; at 1 million,
500 billion. You cannot run a similarity function, let alone a model, that many times. And
it is wasteful: almost every one of those pairs is two obviously-unrelated products.

**What blocking is.** Blocking is a cheap first pass that throws away the pairs that
couldn't plausibly be matches, so the expensive stages only ever see a small **candidate
set**. The classic way to do it: compute a cheap *block key* for each record, drop records
that share a key into the same *block* (bucket), and only compare records inside the same
block. Two records that never land in a common block are never compared — that is the
saving, and also the risk.

```
Every record → a block key → records with the same key share a bucket
                              → only within-bucket pairs become candidates

  "Sony ... - PSLX350H"   key: PSLX350H  ┐
  "Sony PS-LX350H ..."    key: PSLX350H  ┘ same bucket → candidate pair ✓
  "Canon PIXMA iP4200"    key: IP4200      different bucket → never compared
```

**Blocking sets a ceiling nothing downstream can lift.** This is the single most important
idea in the stage. If a true duplicate pair does not share any block with any blocker, it
never enters the candidate set — so `features/` never vectorizes it, `model/` never scores
it, `cluster/` never sees it. It is lost permanently, no matter how good the later stages
are. That is why blocking is judged on recall of *true pairs*, and why we measure it
first: when final recall disappoints, the ceiling here is the first thing to check.

**The two numbers, always reported together.** A blocker is not scored like a classifier —
there is deliberately no precision, F1 or accuracy anywhere in this stage, because a
blocker's output is ~99% non-matches *by design* and discarding non-matches is the whole
job. Instead:

- **Pair completeness (PC)** — of all true duplicate pairs, the fraction that survived into
  the candidate set. This is recall of the blocker, and it is the ceiling above. PC = 1.0
  means every true pair is still reachable.
- **Reduction ratio (RR)** — the fraction of the N² possible pairs that were discarded.
  RR = 0.99 means only 1% of pairs remain to be scored. This is the saving.

Neither number means anything alone, and each is trivially gamed by the other's expense:

| Strategy | PC | RR | Useless because |
| --- | --- | --- | --- |
| keep every pair | 1.0000 | 0.0000 | no saving — you're back to N² |
| emit nothing | 0.0000 | 1.0000 | perfect saving, zero recall |

So they are **always printed side by side**, and the job is to push both toward 1.0 at
once — high recall of true pairs *and* a small candidate set. On Abt-Buy the union of all
blockers reaches PC 0.9928 at RR 0.9644.

**Why several blockers, unioned.** No single blocking strategy catches every kind of
duplicate — each has a blind spot baked into how its key works. An exact-key blocker misses
a pair with a typo in the code; a sort-window blocker misses a pair that differs in its
first word; a vector blocker can miss a pair that shares an exact rare code but sits far
apart in embedding space. The fix is not a cleverer single blocker but **several weak ones
that fail differently**, whose candidate sets are then *unioned*. A blocker earns its place
by lifting the union's PC — not by its standalone score. One that raises the candidate
count without lifting the union is pure cost, and the report is built to expose exactly
that (see the marginal-completeness discussion in `union.py` and CLAUDE.md).

##### The four strategies, as ideas

**1. Exact-key blocking** (`standard.py`) — the cheapest and, on clean vendor codes, the
most precise. Compute a key from each record and bucket by exact string equality. The keys
used here: the extracted model number (`PSLX350H`), every code-shaped token in the title
(in case extraction picked the wrong one), and *rare* title tokens (a word appearing in ≤30
records behaves like an accidental identifier; common words like "black" would bucket
everything). Its weakness is that it is *exact*: `KXTS208W` and `KX-TS208W` are the same
phone but share no key — which is exactly why `normalize.py` produces `model_number_key`,
the separator-stripped comparison form, so the reduction happens once, upstream, and both
records key the same way.

**2. Sorted-neighborhood** (`sorted_neighborhood.py`) — designed to *survive a bad key*.
Sort every record by a key (e.g. the normalized title), then slide a window of width `w`
down the sorted order; every pair falling within `w` positions of each other becomes a
candidate. The insight: two records that differ by one character usually still sort next to
each other, so a window catches them even though exact-key blocking would not. Its blind
spot is the mirror image: it only catches differences *late* in the sort key. `Bose 161WH`
and `Boss 161 Speaker` differ in the first word, so they sort far apart and no reasonable
window reaches across the gap.

**3. MinHash + LSH** (`lsh.py`) — approximate set-similarity search, sublinearly. Four
ideas stack up:

- **Shingles** — break each title into a *set* of overlapping pieces, either whole tokens
  or character k-grams. `"sony turntable"` → `{sony, turntable}` (token) or
  `{sony, ony_, ny_t, ...}` (char). Now "similar title" becomes "overlapping sets".
- **Jaccard similarity** — the overlap of two sets: |A ∩ B| / |A ∪ B|. 1.0 if identical, 0
  if disjoint. This is the target similarity, but computing it for all N² pairs is the very
  cost we are avoiding.
- **MinHash** — a compact signature (here 128 numbers) that *estimates* Jaccard without
  storing the sets: the probability that two records' MinHash values agree equals their
  Jaccard similarity. So set comparison becomes cheap integer comparison.
- **LSH (locality-sensitive hashing)** — bands the signatures and hashes each band into
  buckets, arranged so that high-Jaccard records collide (share a bucket) with high
  probability and dissimilar ones almost never do. The `threshold` parameter tunes where
  that collision curve turns on. The net effect: near-duplicate sets are found without ever
  building the N² Jaccard matrix. Its measured blind spot on Abt-Buy is instructive —
  titles are only 6–10 tokens, so token-shingle sets are tiny and Jaccard over them is
  coarse (dropping one token moves it a lot), which is why this blocker adds *zero* marginal
  recall here despite working fine in principle. Character shingles would densify the sets;
  the knob is left exposed so the claim can be re-tested on `synth/`.

**4. Approximate nearest neighbours / HNSW** (`ann.py`) — the strongest single blocker
here (PC 0.9562 alone), and the only one that needs *no shared token at all*. Three ideas:

- **Vector embedding** — turn each title into a point in high-dimensional space. Here that's
  character-3gram TF-IDF, L2-normalized, so that two titles' cosine similarity is just the
  dot product of their vectors. "Similar product" becomes "nearby point".
- **Nearest-neighbour search** — for each record, find its k closest points; those become
  its candidate partners. Because it works in continuous space, it catches pairs a
  character typo would break for every key-based method: `Bose 161WH` and `Boss 161 Speaker`
  have a misspelled brand and *still* land near each other in char-3gram space.
- **HNSW (Hierarchical Navigable Small World)** — doing exact nearest-neighbour search is
  itself N², so HNSW approximates it. It builds a layered proximity graph (a few long-range
  links up top, dense local links at the bottom) and answers a query by greedily walking the
  graph toward the target — sublinear, at the cost of occasionally missing a true neighbour.
  Its knobs: `M` (links per node), `efConstruction` (how hard it works while building),
  `efSearch` (how hard it works per query — the main recall dial), and `k` (neighbours
  returned per record). Built via `faiss.IndexHNSWFlat`.

**The candidate-pair representation** (`pairs.py`) — every blocker emits the *same* thing so
the union is cheap: a sorted, deduplicated array of int64 keys, where an unordered pair of
record indices `(i, j)` with `i < j` is packed as `i * n + j`. A set of Python tuples would
cost >100 bytes per pair (gigabytes at synth scale); packed int64 is 8 bytes each (~160 MB
for the same set), and unioning several blockers becomes a single `np.unique(concatenate)`
sort rather than millions of hash lookups. The `i < j` rule is what makes a pair
*unordered*, so two blockers that both find the same pair don't double-count it.

---

#### The code, file by file

#### `blocking/base.py` — the contract

| Name | What it does |
| --- | --- |
| `BlockerRun` | One blocker's output: `name`, `params`, packed `keys`, `build_seconds`, `query_seconds`; `n_candidates` property |
| `Blocker` | The Protocol every blocker satisfies — `name`, `params`, `run(records)` |
| `Elapsed` | Mutable seconds holder for the timer |
| `timed()` | Context manager measuring one phase |

Build and query are timed **separately** because they scale differently and the distinction
drives real decisions: an index taking 60s to build and 11s to query is fine for a batch
pass and unusable at serve time. One number would hide that.

`params` is a human-readable string reproduced verbatim in the report — a blocking table
whose rows can't be tied back to a configuration is not a result.

#### `blocking/pairs.py` — packed pair keys

| Function | What it does |
| --- | --- |
| `_check_n(n)` | Guard: n must be non-negative and below ~3.04e9 (int64 packing limit) |
| `pack(left, right, n)` | Index pairs → sorted, deduplicated int64 keys. Normalizes each to `(min, max)` and drops self-pairs |
| `pack_block(members, n)` | Every pair *within* one block, packed |
| `unpack(keys, n)` | Inverse of `pack` → `(left, right)` with `left < right` |
| `union(*key_arrays)` | Merge candidate sets from several blockers, deduplicated |
| `total_pairs(n)` | N(N-1)/2 — the reduction-ratio denominator |

Every blocker emits the same thing: a sorted, deduplicated int64 array where pair `(i, j)`
with `i < j` is packed as `i * n + j`.

**Why not a set of tuples**, the obvious choice: a tuple-of-two-ints inside a set costs
well over 100 bytes once the tuple, its two int objects and the set slot are counted. At
the 20M candidate pairs a 200k-record catalog produces, that's multiple gigabytes — packed
int64 is 8 bytes each, ~160 MB. And unioning becomes `np.unique(np.concatenate(...))`, a
sort, instead of millions of Python-level hash lookups.

The `i < j` convention is what makes a pair *unordered*. Without it, the union of two
blockers double-counts every pair they agree on.

#### `blocking/standard.py` — exact-key blocking

| Name | What it does |
| --- | --- |
| `_strip(token)` | Lowercase, remove non-alphanumerics |
| `model_number_keys(records)` | Key on the extracted vendor code. Precise, low coverage |
| `code_token_keys(records)` | Key on **every** code-shaped token in the title, not just the one extraction chose |
| `rare_token_keys(df_cutoff=30)` | Returns a key function blocking on tokens appearing in ≤ `df_cutoff` records |
| `StandardBlocker` | Group by exact key; every pair inside a block is a candidate. `__init__`, `run` |
| `default_blockers(max_block_size=100)` | The three exact-key blockers measurement justified |

**`max_block_size` is not a tuning knob, it's a safety valve.** A block of m records is
m(m-1)/2 pairs, so one runaway bucket can cost more than every other blocker combined — a
single 5,000-record block is 12.5M pairs. Such a block is also worthless: a key shared by
thousands of records isn't discriminating between them.

**Why `code_token_keys` exists:** `_extract_model_number` commits to one token per title,
and when it picks wrong the pair is lost. Indexing every code-shaped token recovers those —
**0.6708 completeness against 0.5349**, at only 2,072 candidates.

**The `df_cutoff` is the whole design** of `rare_token_keys`. Common words ("black",
"digital", "cable") put every unrelated product in one block; rare ones behave like
accidental identifiers. Measured: cutoff 3 → 0.5608, cutoff 10 → 0.7084, cutoff 30 →
0.8694 — with the candidate count growing alongside. That trade is what the report shows.

#### `blocking/sorted_neighborhood.py`

| Name | What it does |
| --- | --- |
| `title_sort_key(record)` | Sort on the normalized title |
| `model_then_title_sort_key(record)` | Sort on the vendor code where one exists, else the title |
| `SortedNeighborhoodBlocker` | Sort, then slide a window of `w` over the order. `__init__`, `run` |

The point is to **survive a bad key**. Exact-key blocking is all-or-nothing — `"sony psl
x350h"` and `"sony pslx350h"` share no key and are never compared. Sorting puts them
adjacent anyway.

The weakness, and it's why this blocker measures worst here (PC 0.6190 at w=20, for the
largest candidate count of any single blocker): it only catches pairs differing *late* in
the string. `"Bose 161WH"` and `"Boss 161 Speaker"` sort far apart no matter how wide the
window. Widening `w` costs candidates linearly and buys very little.

Ties are broken by index, so the candidate set doesn't depend on loader row order.

#### `blocking/lsh.py`

| Name | What it does |
| --- | --- |
| `_shingles(text, mode, char_k)` | Token shingles or character k-shingles as bytes |
| `MinHashLSHBlocker` | MinHash signatures + LSH index → approximate Jaccard neighbours. `__init__`, `run` |

Needs neither a good sort key nor an exact match. Hashes each record's shingle set so
similar sets land in the same bucket, without ever computing the N² Jaccard matrix.

**Measured caveat, recorded because it's surprising:** with token shingles at threshold 0.4
on Abt-Buy, this reaches PC 0.5894 alone and contributes **exactly zero** marginal
completeness to the union — 20,539 extra candidate pairs for no additional true pair.
Product titles are short (6-10 tokens), so a token-shingle set is tiny and Jaccard over it
is coarse: dropping one token moves the similarity a long way.

That's a result about *this configuration on this dataset*, not about LSH. `shingles="char"`
gives a much denser set, and `synth/`'s token-drop corruption is precisely the failure mode
LSH is meant to absorb. Both knobs stay exposed so the claim can be re-tested.

#### `blocking/ann.py`

| Name | What it does |
| --- | --- |
| `AnnBlocker.__init__` | Configure neighbours, HNSW M / efConstruction / efSearch, optional SVD, determinism |
| `AnnBlocker._vectors(records)` | Build the dense float32 matrix, with a memory guard and optional SVD projection |
| `AnnBlocker.run(records)` | Build `faiss.IndexHNSWFlat`, search k+1 neighbours, pack the pairs |

**The strongest single blocker: PC 0.9562 at RR 0.9938** — beating every exact-key blocker
combined. It succeeds where keys fail because it needs no shared token at all: `"Bose
161WH"` and `"Boss 161 Speaker"` have a source typo in the brand and still land near each
other in character-3gram space.

Built on `faiss.IndexHNSWFlat` rather than `hnswlib`, which ships no PyPI wheel for any
platform and needs an MSVC toolchain this machine doesn't have. Same HNSW algorithm.

Three details that are load-bearing:

- **The vectorizer comes from `eval.baseline.fit_vectorizer`**, not a local copy of the
  char-3gram config, so this blocker searches the same text space the baseline is scored
  in. Two definitions of "the vector for a record" drifting apart is train/serve skew.
- **The build is pinned to one thread.** HNSW graph construction is parallelized and the
  order threads link nodes changes the graph — measured, repeated runs on identical input
  gave 14,608 / 14,603 / 14,602 candidates. PC was unmoved, but a committed report whose
  candidate count drifts isn't reproducible.
- **A memory guard, not an OOM.** Dense char-3gram TF-IDF is 59 MB at Abt-Buy's
  2,173 × 6,804 but ~24 GB at 200k records. Past a 2 GB budget it raises with an
  explanation and points at `n_components` (SVD), rather than being killed.

#### `blocking/union.py` — combining and scoring

| Function | What it does |
| --- | --- |
| `true_pair_keys(records)` | Packed keys of every same-entity pair, from ground truth (including pairs transitivity implies) |
| `BlockerScore` | One row of the blocking table |
| `score(run, truth, n_records, n_true_pairs_total)` | Pair completeness + reduction ratio for one blocker |
| `union_run(runs, name)` | Combine every blocker's candidate set into the one the pipeline sees |
| `missed_pairs(candidate_keys, truth)` | The true pairs no blocker emitted — unrecoverable |
| `ground_truth(records)` | The ground-truth pair set and its size, **cross-checked two ways** |

**A blocker is not scored like a classifier.** There is deliberately no precision, F1 or
accuracy anywhere in this module: a blocker's output is ~99% non-duplicates by
construction, and discarding non-duplicates is the job, not an error.

`score()` takes `n_true_pairs_total` as a parameter **and checks it**, rather than deriving
it from `truth`, because that's the exact place this measurement goes wrong. Computing pair
completeness over the pairs that survived blocking makes it identically 1.0 — a
self-fulfilling result that has reached publication.

`ground_truth()` cross-checks two independent computations: `count_true_pairs` sums
C(size, 2) over entity groups, `true_pair_keys` enumerates the pairs. If they disagree, one
is wrong and every completeness number downstream is wrong with it.

#### `blocking/evaluate.py` — the CLI

| Name | What it does |
| --- | --- |
| `default_blocker_set()` | All six blockers, **including the ones that don't earn their place** |
| `BlockingReport` | The whole evaluation: rows, union row, missed examples |
| `evaluate(records, blockers, dataset)` | Run every blocker, score each, score the union, collect missed pairs |
| `_row(s)` | One markdown table row |
| `render_markdown(report)` | The `reports/blocking.md` artifact |
| `main(argv)` | CLI entry point |

`default_blocker_set()` keeps LSH and sorted-neighborhood even though measurement says
neither earns its cost, because **a negative result someone can re-run is evidence, and the
same result asserted from a deleted experiment is not.**

The report prints the *actual titles* of missed pairs, not just the count — the count says
what the ceiling is, the pairs say what to build next.

---

### 5.6 `features/` — candidate pair → feature vector

Turns each candidate pair into 33 numbers (34 with the opt-in semantic column). Shaped like
an sklearn transformer, because the fit/transform boundary is what makes train/serve parity
enforceable rather than aspirational: `fit` sees the **train split only** and never labels,
and the fitted object *is* the serve-time artifact.

#### `features/base.py` — the contract

| Name | What it does |
| --- | --- |
| `FeatureSpec` | One column: `name`, `doc`, `higher_is_similar`, `is_indicator`, `imputed`, `companion_indicator` |
| `FeatureMatrix` | `values` (n_pairs × n_features) carried together with its `specs`; `column(name)` looks up by name |
| `FeatureBlock` | The Protocol every block satisfies — `specs`, `fit(records)`, `transform(records, left, right)` |
| `FILL_VALUE` | What an undefined feature holds (0.0) |
| `has_price` / `has_description` / `has_brand` / `has_model_number` | What "present" means, shared by the blocks and the indicators |

The presence predicates live here, not in the blocks, deliberately. An indicator is a claim
about *another* column, so the predicate gating the feature and the predicate setting the
indicator must be the same code — two copies that drift produce an indicator that lies,
which is worse than no indicator at all.

#### `features/string.py` — the bulk

| Block | Columns |
| --- | --- |
| `TitleBlock` | ratio, token-sort, token-set, partial, token jaccard, containment, length ratio, common prefix |
| `CodeBlock` | model-number exact/prefix, code-token jaccard/count, best code ratio |
| `DescriptionBlock` | description token jaccard, length ratio |
| `BrandBlock` | brand equality |
| `CorpusTextBlock` | IDF-weighted title overlap, title cosine, description cosine, two cross title×description cosines — **the only block with a real `fit`** |

`containment` (shared ÷ *smaller* token set) exists because jaccard punishes a short title
for being short — a five-token marketplace stub inside a twenty-token Abt title scores 0.25
by jaccard and 1.0 by containment.

`code_best_ratio` deliberately scores `WH-1000XM4` vs `WH-1000XM5` at 0.89. That's not a
bug: paired with `model_number_exact` reading **0**, it's what lets the model tell "same
code" from "adjacent code" — a distinction no single column can express.

Code tokens come from `blocking.standard.code_token_keys`, not a local copy. That function
already sits on the public `normalize.code_key`, and a second copy of a rule that must agree
with blocking's is how a serve-time index stops reproducing the batch keys.

`CorpusTextBlock` fits **one** vectorizer over titles *and* descriptions, not one each — the
cross columns compare a title to a description, and a cosine between two separately fitted
spaces is a coincidence, not a similarity.

#### `features/numeric.py`

`PriceBlock` (absolute log ratio, relative difference — both declared as *distances*) and
`DigitTokenBlock` (jaccard and count over multi-digit title tokens). Digit tokens cover a
real gap: `code_token_keys` requires a letter *and* a digit in one token, so a bare `1600`
or `161` is invisible to it.

`log1p`, not `log`: `schema.py` permits a price of 0.0 and `log(0)` is -inf, which would
propagate through the whole vector with nothing pointing back at the cause.

#### `features/missingness.py` — the invariant, as code

Eight indicator columns. Two of them describe **three** states rather than two:

| `price_both_present` | `price_neither_present` | meaning |
| ---: | ---: | --- |
| 1 | 0 | comparable |
| 0 | 0 | exactly one side priced |
| 0 | 1 | nothing to compare |

A single `price_is_missing` column collapses the middle row, and the middle row is the
interesting one. (This is also why `desc_either_missing` was dropped during implementation:
it's exactly `1 - desc_both_present`, carrying no information and adding a perfectly
collinear column. `desc_any_present` splits three ways instead.)

#### `features/semantic.py` — opt-in

One column, `title_embedding_cosine`. `sentence_transformers` is imported *inside* the
methods so importing `dedup.features` stays cheap and offline, and `model_is_cached()` lets
tests and the CLI check for weights without triggering a 90 MB download.

#### `features/vectorize.py` — assembly

`PairFeaturizer` concatenates every block in a fixed order and `validate_registry` rejects a
mis-declared feature set **at construction time**: an imputed column with no companion
indicator, a companion naming something that isn't an indicator, an indicator that is itself
imputed, or a duplicate name. So the missingness invariant is checked by construction, and a
feature added next month inherits the check by being registered.

Column order is part of the trained-model contract — a model that learned column 7 as
`title_common_prefix_ratio` mispredicts silently if column 7 later means something else. The
semantic block appends *last* so turning it on renumbers nothing.

#### `features/evaluate.py` — the diagnostic CLI

Blocks each split separately (cross-split pairs are all negatives by construction), fits on
train, measures on test, and writes `reports/features.md`. Distance-declared columns are
negated before ranking, or `price_abs_log_ratio` would report as useless for being strong.

---

## 6. Results so far

### Baseline — `reports/baseline_tfidf.md`

TF-IDF char-3gram cosine, one threshold, no blocking, no model.

| | Test F1 | Test P | Test R | PR-AUC | P@10 |
| --- | --- | --- | --- | --- | --- |
| **normalized title** | **0.5204** | 0.4605 | 0.5982 | 0.4720 | 0.600 |
| title + description | 0.4349 | — | — | 0.2898 | 0.000 |

Entity-grouped split, `test_fraction=0.3`, `seed=0`. **Every later stage is justified
against 0.5204.**

**Adding description makes it worse, and the reason is measurable**, not a matter of prose
quality: Abt descriptions average 249 characters and are never empty; Buy's average 34
(median 14) and are empty on 40% of rows. Concatenation makes an Abt vector that's mostly
description face a Buy vector that's mostly title, diluting exactly the true pairs it was
meant to help. What the two sides *do* share is category vocabulary — `finish` in 759
descriptions, `black` in 622 — which lifts *unrelated* pairs. Description isn't worthless;
it belongs in `features/` as its own signal with a missingness indicator, not glued onto
the title string.

### Blocking — `reports/blocking.md`

N = 2,173 records, 1,118 true pairs, 2,359,878 possible pairs.

| blocker | params | candidates | PC | RR |
| --- | --- | ---: | ---: | ---: |
| standard (model number) | key=model_number_key | 628 | 0.5349 | 0.9997 |
| standard (code tokens) | code-shaped title tokens | 2,072 | 0.6708 | 0.9991 |
| standard (rare tokens) | title tokens, df≤30 | 29,555 | 0.8694 | 0.9875 |
| sorted_neighborhood | w=20 | 41,097 | 0.6190 | 0.9826 |
| lsh (minhash) | 128p, t=0.4, token | 27,828 | 0.5894 | 0.9882 |
| ann (faiss HNSW) | M=32, ef=100, k=10 | 14,603 | **0.9562** | 0.9938 |
| **union (all)** | — | 84,117 | **0.9928** | 0.9644 |

**PC** = pair completeness (true pairs surviving ÷ all 1,118 ground-truth pairs).
**RR** = reduction ratio (fraction of possible pairs discarded).

**The union row is the only one the rest of the pipeline inherits.** Its 0.9928 is a hard
ceiling on system recall — the **8** true pairs no blocker emitted are never scored by
`features/`, never seen by `model/`, never reach `cluster/`.

Those 8 misses are the design input for the next blocker. They're genuinely hard:

```
"Tivo Wireless Adapter - AG0100"   vs  "Tvio Wireless USB Network Adpator"
"LG DLEX7177RM ... SteamDryer - DLEX7177RD"  vs  "LG 27' Front-Load Electric Dryer..."
"Canon Black Photo Ink Cartridge - CLI8B"    vs  "Canon Ink Cartridge For PIXMA iP4200..."
```

Two source typos in one title, a model number present on one side and absent on the other,
and a vendor part number that shares nothing with the product name.

**Read honestly:** Abt-Buy is *pre-blocked* — two curated ~1,000-record catalogs already
scoped to overlapping product ranges. A high union PC says the benchmark is small and
clean, not that blocking is solved. RR is flattered by a small N: 2.36M pairs is something
a laptop brute-forces, and the baseline does exactly that. `synth/` is where this stage
earns its keep.

---

### Features — `reports/features.md`

33 columns per candidate pair (34 with the opt-in semantic block), fit on train and spent
on test. There is **no F1 here** — `features/` has no classifier in it. What the report
gives is per-column coverage, univariate PR-AUC and class separation, which is what makes a
later disappointment diagnosable: a constant column, a backwards column, and a column
imputed on 98% of pairs all read as "the model didn't improve much" from inside `model/`.

Top of the ranking, on the test split:

| feature | coverage | PR-AUC | pos mean | neg mean | n+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code_token_jaccard` | 0.7456 | **0.6274** | 0.6584 | 0.0032 | 268 |
| `model_number_prefix_ratio` | 0.6917 | 0.5784 | 0.8310 | 0.0431 | 246 |
| `title_tfidf_cosine` | 1.0000 | 0.5222 | 0.6648 | 0.1153 | 341 |
| `model_number_exact` | 0.6917 | 0.4714 | 0.6707 | 0.0005 | 246 |
| `code_best_ratio` | 0.7456 | 0.4646 | 0.9610 | 0.2859 | 268 |
| `cross_title_desc_cosine_max` | 0.9494 | 0.4473 | 0.4246 | 0.0915 | 338 |

Every figure is over the **covered** pairs — the ones where the column actually holds a
value. `n+` is how many of those are true matches.

**Vendor codes dominate, exactly as predicted.** A test asserts they stay at the top — if
they ever don't, the feature is broken, not the claim.

**The cross title×description column works.** It was added for the one failure blocking
can't close: a truncated marketplace title carrying no code. On `LG Over-The-Range White
Microwave Oven - LMV1680WH` vs `LG 1.6 cu.ft. Over the Range` it reads **0.7056** while
every string column reads under 0.32.

**One column points the wrong way, and it isn't a bug.** `desc_len_ratio` reads 0.2213 on
true pairs against 0.4763 on false, over 12,133 covered pairs. That's the dedup framing
showing through — same-side Buy↔Buy pairs are candidates, and Abt descriptions average 249
chars against Buy's 34, so two descriptions of *similar* length are evidence of being
same-side, which here means evidence of being a non-match. A tree model uses that fine; a
human reading the column name does not, which is why the report flags it.

**The first version of this report got that list wrong, and the reason is worth keeping.**
It also listed `brand_equal` as backwards (0.0176 vs 0.0890) — but that was the *diagnostic*
averaging `FILL_VALUE` into the class means, i.e. reading a null brand as a brand mismatch.
That is exactly the error invariant I5 exists to prevent, committed one layer above the
vector where I5 was being enforced. On the 4,506 pairs where a brand actually exists,
`brand_equal` points forwards and strongly: **0.7500 vs 0.3655**. Caught by the
`er-invariants` agent, not by the test suite — the suite checked the invariant inside
`vectorize.py` and never asked whether the report obeyed it. Now pinned by two regression
tests.

**Don't compare these PR-AUCs to the baseline's 0.4720.** `title_tfidf_cosine` scores
higher (0.5222) but is measured over the 18,819 pairs blocking left, not the full N²
triangle — 91% of pairs, almost all negatives, are already gone. That's the candidate set
scoring, not the column. The like-for-like number against **0.5204** arrives with `model/`.

---

## 7. The invariants (why the code looks the way it does)

These are correctness traps specific to entity resolution. Violating them produces results
that look good and are wrong.

| Invariant | Why | Where enforced |
| --- | --- | --- |
| **Split by entity, never by pair** | A pair-level split puts the same product on both sides; metrics inflate badly | `eval/splits.py` |
| **PR-AUC, never ROC-AUC** | Post-blocking imbalance is thousands:1 — ROC-AUC reads 0.99 for a useless model | `eval/metrics.py` |
| **Recall denominator = all true pairs** | Otherwise pruning harder *improves* the score | every metric takes `n_positives_total` |
| **Model output must be calibrated** | The cost model consumes a probability, not a ranking score | ⬜ `model/` |
| **Thresholds from expected cost, not argmax F1** | False merge ≫ false split in cost | ⬜ `model/` |
| **Missing values as explicit indicators** | A null price is not a price mismatch | ✅ `features/` — enforced by `validate_registry`, not by review |
| **Cluster quality via B-cubed** | Pairwise F1 doesn't imply good clusters | ⬜ `cluster/` |
| **Blockers scored on PC + RR only** | Never precision/F1 — a blocker's output is ~99% negatives by design | `blocking/union.py` |
| **`normalize.py` identical in batch and serve** | Anything else is train/serve skew | `normalize.py` (no source branching) |
| **Only `data/` knows a dataset's name** | Lets a blocker be written once, evaluated everywhere | `data/__init__.py` registry |

---

## 8. What's pending

### Immediately next — `model/`

- `train.py` — LightGBM over the pair features, entity-grouped split
- `calibrate.py` — isotonic or Platt. **Non-negotiable**: the cost model needs a real
  probability, not a ranking score
- `threshold.py` — the cost model. Two thresholds → three bands (auto-merge / review /
  auto-reject), derived from asymmetric error costs, not argmax F1

### Then — `cluster/`

- `components.py` — connected components, **including a demonstration of chaining**: A~B
  and B~C with A!~C still merges all three. CLAUDE.md says this failure is intentionally
  shown, not silently designed around
- `correlation.py`, `agglomerative.py` — alternatives that resist chaining
- `bcubed.py` — B-cubed precision/recall, measured separately from pairwise metrics

### Then — `synth/`

Corruption engine: typos, abbreviations, token drop/reorder, unit swaps, price jitter,
brand aliasing → 200k-1M records with known ground truth.

This is where blocking stops being a formality. It's also where the ANN blocker's dense
matrix stops fitting in memory (the `n_components` SVD path exists for exactly this), and
where the baseline's N² scoring becomes impossible — demonstrating rather than asserting
why `blocking/` exists.

### Finally — `service/`

- `app.py` — FastAPI + uvicorn
- HNSW + inverted index for serve-time candidate lookup
- Review queue with an htmx UI (deliberately no Node toolchain)
- DuckDB/SQLite record store
- Human decisions flowing back as training labels

### Also outstanding

- **A second dataset.** `DATASETS` has one entry. Amazon-Google is the natural second
  (also cp1252) and would test whether the "no per-source branching" rule actually holds
- **`reports/`** still needs PR curves and cost curves
- **`normalize.py` open question:** whether `_SPEC_UNIT_SUFFIXES` should grow — it needs
  evidence from more data, not judgment

---

## 9. Commands

The dev environment is `.venv`, deliberately Python 3.12 rather than the machine default
(ML wheel availability). On Windows, prefix with `.venv/Scripts/python -m` if the venv
isn't active.

```bash
# environment
pip install -e ".[dev]"

# tests — 250 passing, 1 skipped
pytest
pytest tests/test_normalize.py                  # one file
pytest -k model_number                          # by keyword
pytest -rs                                      # show skips

# lint (line-length 100, target py310)
ruff check src tests

# download the benchmark (data/ is gitignored, fresh clones start empty)
B=https://raw.githubusercontent.com/dchud/ddbench/HEAD/data
mkdir -p data/raw/abt-buy
for f in Abt.csv Buy.csv abt_buy_perfectMapping.csv; do
  curl -sS -o "data/raw/abt-buy/$f" "$B/abt-buy/$f"
done

# load the benchmark
python -c "from dedup.data import load_dataset; print(len(load_dataset('abt-buy')))"

# regenerate the reports
python -m dedup.eval.baseline --dataset abt-buy --out reports/baseline_tfidf.md
python -m dedup.blocking.evaluate --dataset abt-buy --out reports/blocking.md
python -m dedup.features.evaluate --dataset abt-buy --out reports/features.md

# regenerate test fixtures (Abt.csv must stay cp1252 — an editor will silently undo that)
python tests/fixtures/make_fixtures.py
```

**Not built yet** — these will fail if invoked:

```bash
python -m dedup.model.train
python -m dedup.cluster.evaluate
uvicorn dedup.service.app:app --reload
```
