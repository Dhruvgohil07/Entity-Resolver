"""Abt-Buy's report passages: the claims only this dataset's measurements support.

Each passage replaces a renderer's generic text for the slot it names (see
`notes.py`), and each is the exact text that renderer printed before passages
existed -- so every committed Abt-Buy report regenerates byte for byte. The
measurements behind them are recorded in CLAUDE.md under "Settled by
measurement", and several are re-derived by the tests that read `data/raw/`.

The baseline passage names `{n_records}` and `{n_true_pairs}`, which the
renderer fills in; no other passage is formatted.
"""

from pathlib import Path

from dedup.data.notes import DatasetNotes

NOTES = DatasetNotes(
    baseline_report=Path("reports/baseline_tfidf.md"),
    passages={
        "baseline.framing": """\
- **This is the deduplication framing, not the record-linkage one.** Every pair of the
  {n_records} records is a candidate, and a pair is positive when the two records
  share an `entity_id`. Published Abt-Buy F1 figures are for the cross-source task — Abt
  row against Buy row only — over the 1097 shipped pairs. Transitivity through the size-3
  clusters makes {n_true_pairs} pairs true here, and same-side pairs are in the
  candidate set. Close to the published task, not identical to it; compare accordingly.""",
        "blocking.intro": """\
**These parameters were chosen against these same numbers.** The window, neighbour
count and document-frequency cutoff were selected by sweeping them over this full
catalog, not over a held-out split, so the figures below are fit to this dataset and
are optimistic as an estimate of what these settings would do on an unseen one. See
*Reading this honestly* for the held-out ceilings.""",
        "blocking.provenance": """\
- **Abt-Buy is pre-blocked.** It ships as two curated catalogs of ~1,000 records each,
  already scoped to overlapping product ranges. A high union PC here says the benchmark
  is small and clean, not that blocking is solved. `synth/` — 200k to 1M records with
  known ground truth — is where this stage earns its keep, and where a dense ANN index
  stops fitting in memory.""",
        "blocking.tuning": """\
- **The parameters were tuned on the catalog they are scored on.** No held-out split was
  used to pick the window, neighbour count or df cutoff, so treat the union figure as a
  best-of-sweep number rather than a clean estimate. Measured for comparison on the
  entity-grouped split (`seed=0`), where the same settings give union pair completeness
  **0.9949 on train and 1.0000 on test** — so the selection bias here is small, but it is
  present and unmeasured until parameters are chosen on train alone.""",
        "features.computation": """\
Two things about how these are computed, both of which changed a published number here:

- **Every figure is over the covered pairs only.** Averaging the fill into a class mean
  reads a null as a mismatch — the error CLAUDE.md's missingness invariant exists to
  prevent, and it is just as wrong in a report as in a vector. `brand_equal` is covered on
  2.3% of positives against 24.3% of negatives, so averaging 0.0 in at those rates reported
  it as pointing *backwards*, when on pairs that actually have a brand it points forwards
  and strongly. The same restriction applies to the ranking, where a fill of 0.0 ranks last
  for a similarity column but — once negated — ranks *first* for a distance column, sorting
  an absent price as a perfect price match.
- **`distance` columns are negated before ranking**, so their PR-AUC is comparable with the
  rest. Without it `price_abs_log_ratio` would report as useless for being strong.""",
        "features.narrow_column": """\
The recall denominator stays every true pair in the split, so a narrow column cannot look
strong by being asked less: `brand_equal` can address 8 of 341 true pairs, and its PR-AUC
is bounded near 0.02 accordingly. Read PR-AUC as *how much of the problem this column can
reach*, and the means as *whether it points the right way where it applies*. They answer
different questions and a column can score well on one and badly on the other.""",
        "features.wrong_way": """\
This is **not** a bug, and flipping the sign would be the wrong fix. It is the
deduplication framing showing through: every pair of the combined catalog is a candidate,
so same-side pairs (Buy against Buy) sit in the table alongside the cross-source pairs that
carry almost all the true matches. Abt descriptions average 249 characters against Buy's
34, so two descriptions of *similar* length are evidence of being same-side — which on this
dataset means evidence of being a non-match. A tree model uses that correctly. A human
reading the column name does not, which is why it is called out here.""",
        "features.wrong_way_caution": """\
Check `n+` before drawing a conclusion from any row in this section. A column covered on a
handful of true pairs can land here on noise, and the fix for that is more data, not a sign
flip. `brand_equal` was listed here in an earlier version of this report for a worse reason
than noise: the class means included the fill, so a missing brand was being counted as a
brand mismatch.""",
        "features.tuning": """\
- **The blocker parameters were swept on the full catalog**, so the PC figures inherited
  here are best-of-sweep rather than a clean estimate. Small bias, still unquantified —
  it is recorded as an open question in CLAUDE.md and not resolved by this report.""",
        "model.gain_column.desc_len_ratio": (
            "CLAUDE.md records it pointing backwards univariately on Abt-Buy — a measured "
            "consequence of the deduplication framing — and a tree model uses a backwards "
            "column correctly where a human reading the column name does not."
        ),
    },
)
