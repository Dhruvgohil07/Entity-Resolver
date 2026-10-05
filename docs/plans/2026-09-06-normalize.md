# Plan: `src/dedup/normalize.py` — shared normalize step + `NormalizedRecord`

- **Date:** 2026-09-06
- **Status:** done
- **Implemented in:** `0a46fa1` feat(normalize): add shared normalize stage and NormalizedRecord
- **Source:** plan mode (`now-let-s-plan-for-eventual-bumblebee.md`), copied unchanged below this header.
  It records the plan as written before the work began; what shipped can differ -- see the
  commit and `docs/DECISIONS.md`.

## Context

`schema.py` (`Record`) is done and committed. CLAUDE.md names `normalize.py` as
the next stage: "canonical form: unicode NFKD, case, units, brand aliases,
model-number extraction" and calls out model-number extraction as "the single
highest-value signal," warranting its own tests. The module must run
identically in batch and at serve time (train/serve parity invariant) — so it
has to be a pure function of a `Record`, with no source-specific branching
that would silently differ for an unseen serve-time source.

No benchmark data is downloaded yet, so the design is grounded in real rows
pulled from the actual datasets (not assumed shapes):

- **Abt.csv** (`dchud/ddbench`, GitHub): titles end in `"<Description> - <CODE>"`
  — e.g. `"Sony Turntable - PSLX350H"`, `"Bose 27028 161 Bookshelf Pair
  Speakers In White - 161WH"`, `"Panasonic 2-Line Integrated Telephone -
  KXTSC14W"`.
- **Buy.csv** (same repo): same trailing-code convention on many rows
  (`"Netgear ProSafe FS105 Ethernet Switch - FS105NA"`,
  `"Panasonic Black Toner Cartridge - KX-FA83"`), but *not all rows have a
  trailing code* — some embed it mid-title with no delimiter at all
  (`"Netgear ProSafe FS105 Ethernet Switch"`, `"Linksys EtherFast EZXS55W
  Ethernet Switch"`).
- **GoogleProducts.csv** (`git.iti.cs.ovgu.de` mirror): lowercase, and many
  rows *lead* with a vendor SKU token before the description
  (`"cd384-aisk9= cisco ios advanced ip services - complete package - cd"`,
  `"eu063av aba microsoft windows xp professional..."`, `"0202-034 axis
  camera station - license - 1 additional user"`), and many others have no
  extractable code at all (`"learning quickbooks 2007"`,
  `"video studio 11 plus"`).
- **Amazon-Google** common attrs confirmed as `title/name, description,
  manufacturer, price` — matches `Record`'s scope, no surprises there.

Takeaway that shapes the design: there is no single reliable delimiter
convention across sources. A per-source regex would violate dataset-agnosticism
and wouldn't generalize to whatever source a real serve-time record declares.
So model-number extraction is one generic heuristic — "is this token shaped
like a vendor code" — applied the same way regardless of `record.source`,
with a documented preference order grounded in the conventions actually
observed above.

## Design

### `NormalizedRecord` (composition, not inheritance)

```python
class NormalizedRecord(BaseModel):
    """Derived, normalized view of a Record. Never constructed directly by a
    loader -- always produced by normalize(), so batch and serve paths can't
    diverge (CLAUDE.md train/serve parity invariant)."""

    raw: Record

    normalized_title: str
    normalized_description: str | None
    normalized_brand: str | None
    model_number: str | None
```

`raw` keeps the original `Record` reachable (features/ stages will want both
the normalized text and the untouched original, e.g. for raw price). No
`normalized_category` — category isn't used by any string-similarity feature
per CLAUDE.md's feature list; add only when a stage actually needs it.

### `normalize(record: Record) -> NormalizedRecord`

Pure function, no I/O, no branching on `record.source`. Pipeline:

1. **`_fold(text: str) -> str`** — `unicodedata.normalize("NFKD", text)`,
   `str.casefold()` (not `.lower()` — casefold is the correct primitive for
   case-insensitive matching, e.g. German `ß`), collapse repeated whitespace
   via `re.sub(r"\s+", " ", ...)`, `.strip()`. Applied to `title`,
   `description`, `brand`. `None` stays `None` (missingness preserved, not
   collapsed to `""` — same convention `schema.py` already established).

2. **`_canonicalize_units(text: str) -> str`** — small, explicit substitution
   table applied after folding, e.g. `cu. ft.` / `cu ft.` → `cu ft`,
   `"` / `'` / `in.` / `inch` → `in`, `oz.` → `oz`, `lbs.` / `lb.` / `pound` →
   `lb`. Seeded from what's visible in the real titles above (`"4.2 Cu.
   Ft."`, `"24' White..."`); documented as a small, growing table — real
   coverage will expand once the actual benchmark CSVs are downloaded and
   more unit spellings are visible. Not a general unit-conversion system,
   just spelling canonicalization so `"4.2 cu ft"` and `"4.2 cu. ft."` become
   the same token stream for string-similarity features later.

3. **`_BRAND_ALIASES: dict[str, str]`** — lowercase variant → canonical
   brand, e.g. `"hewlett-packard": "hp", "hewlett packard": "hp"`. Seeded
   small (brands seen in the real rows above — sony, bose, panasonic, denon,
   linksys, netgear, belkin, canon, lacie, d-link, logitech, kensington,
   cuisinart, kitchenaid, frigidaire, tripp lite — are already single tokens
   needing no alias); `_normalize_brand` looks up the folded brand in this
   table and falls through to the folded value unchanged if there's no
   entry. Documented as a seed table to extend once real brand columns are
   in hand, not a claim of completeness.

4. **`_extract_model_number(title: str) -> str | None`** — operates on the
   **original, pre-fold** title (case is a real signal for these codes: `EZXS88W`,
   `KX-FA83`). Algorithm:
   - Split on whitespace; strip trailing `,./=` punctuation from each token.
   - A token *qualifies* if `4 <= len(token) <= 20`, it matches
     `^[A-Za-z0-9-]+$`, and it contains **both** a digit and a letter (this
     is what excludes pure words like `"quickbooks"`, pure numbers like
     `"2007"` or a price-like digit run, and years).
   - Preference order, grounded in the three conventions found above:
     a. If the title contains `" - "`, check the token(s) after the *last*
        occurrence — the Abt/Buy trailing-SKU convention. Use it if it
        qualifies.
     b. Else check the *first* token — the Google leading-SKU convention.
        Use it if it qualifies.
     c. Else scan all tokens left-to-right, return the first that qualifies
        (covers Buy's undelimited mid-title case, e.g. `FS105` inside
        `"Netgear ProSafe FS105 Ethernet Switch"`).
     d. Else `None` — no vendor-code-shaped token found (e.g. `"learning
        quickbooks 2007"`, `"video studio 11 plus"`).
   - Return value is `.upper()`'d for a consistent canonical form across
     sources regardless of input casing (Google's `cd384-aisk9` and Abt's
     `PSLX350H` both end up uppercase).
   - Documented as a heuristic, not a guarantee: false negatives (a real
     code the pattern misses) and false positives (an incidental alnum token
     mistaken for a code) are both possible. Revisit precision/recall of
     this specific function once real labeled data is downloaded — flagged
     as a follow-up, not solved here.

5. `normalize()` composes the above: folds title/description/brand, runs
   unit canonicalization on the folded title and description, resolves the
   brand alias, extracts the model number from the *raw* title, and returns
   a `NormalizedRecord`.

### Key decisions and why

- **One generic extraction heuristic, not per-`source` branching.** Keeps
  the function dataset-agnostic and lets it work on a serve-time record from
  a source it's never seen, which per-source regex dispatch could not do —
  this is the direct train/serve-parity consequence of the mixed
  conventions found in real data.
- **Model number extracted from the un-folded title.** Folding is for
  string-similarity features (case/accent-insensitive matching); model
  numbers are compared more like exact/fuzzy codes, and case is part of
  their shape. Both are useful in different downstream features, which is
  why they're separate fields rather than one doing double duty.
- **No regex-only "trust the first hyphen-split half" shortcut.** Rejected
  because Buy.csv shows plenty of titles with *no* trailing code at all —
  a positional-only rule would silently emit garbage instead of `None`.
- **Unit canonicalization is spelling normalization, not unit conversion.**
  Converting `"4.2 cu ft"` to liters is a `features/numeric.py`-scope
  decision (if ever needed); `normalize.py` only makes equivalent spellings
  compare equal as text.

## Files

- **`src/dedup/normalize.py`** (new) — `NormalizedRecord`, `normalize()`,
  and the four private helpers above.
- **`tests/test_normalize.py`** (new) — real examples pulled from the
  research above, not synthetic ones, specifically:
  - NFKD + casefold behavior (accented input folds to a comparable form).
  - Whitespace collapse (tabs/double-spaces).
  - Unit canonicalization on `"4.2 Cu. Ft."`-shaped input.
  - Brand alias hit (seeded alias) + passthrough for an unlisted brand.
  - Model number, trailing-convention: `"Sony Turntable - PSLX350H"` →
    `"PSLX350H"`.
  - Model number, trailing-convention with noisy mid-title digits:
    `"Bose 27028 161 Bookshelf Pair Speakers In White - 161WH"` → `"161WH"`
    (not the mid-title `27028`/`161`, which are pure-digit and don't
    qualify anyway, and not picked because trailing wins).
  - Model number, no-delimiter mid-title fallback:
    `"Netgear ProSafe FS105 Ethernet Switch"` → `"FS105"`.
  - Model number, Google leading-token convention:
    `"cd384-aisk9= cisco ios advanced ip services - complete package - cd"`
    → `"CD384-AISK9"`.
  - Model number, genuinely absent: `"learning quickbooks 2007"` → `None`.
  - Purity: normalizing the same `Record` twice yields equal
    `NormalizedRecord`s.
  - `raw` field round-trips the original `Record` unchanged.
  - `description=None` stays `None` (not folded into `""`).

## Verification

1. `pytest tests/test_normalize.py -v` — all new tests pass.
2. `pytest` (full suite) — `test_schema.py` still green, no regressions.
3. `ruff check src/dedup/normalize.py tests/test_normalize.py` — clean.
4. Manual REPL sanity check: run `normalize()` on 2-3 more real rows from the
   research above that aren't already covered by a test, confirm output
   looks right.
