"""Build the persisted ann/standard indexes that let a run accept online lookups.

`python -m dedup.service.build_index --db data/service.duckdb --run-id <run_id> --index-root artifacts/index/<run_id>`

A CLI, not a route -- `blocking.ann.AnnIndex.build` refits a TF-IDF
vectorizer over the whole run's records, the same unbounded, request-shaped-
wrong operation that keeps batch execution (`service/batch.py`) out of
`service/app.py` in the first place. This is the explicit "this run now
accepts lookups" transition: `POST /runs/{run_id}/lookup` refuses with 409
until this has run for that `run_id` at least once.

Re-running against an already-enabled run rebuilds both indexes from the
run's *current* `records` table (including anything earlier lookups added)
and overwrites `run_indexes`' pointers -- a legitimate way to defragment the
persisted FAISS graph after many `add()`-triggered incremental writes, not
an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

from dedup.blocking.ann import AnnIndex
from dedup.blocking.standard_index import InvertedIndex
from dedup.normalize import normalize
from dedup.schema import Record
from dedup.service import store


def build_index(conn: duckdb.DuckDBPyConnection, run_id: str, index_root: Path) -> None:
    """Read `run_id`'s full catalog, build and save both indexes, record
    their location in `run_indexes`."""
    rows = store.iter_records(conn, run_id)
    if not rows:
        raise ValueError(f"run {run_id!r} has no records to index")

    records = [normalize(Record.model_validate_json(row.raw_json)) for row in rows]
    record_ids = [row.record_id for row in rows]

    index_root = Path(index_root)
    ann_root = index_root / "ann"
    standard_root = index_root / "standard"

    AnnIndex.build(records, record_ids).save(ann_root)
    InvertedIndex.build(records, record_ids).save(standard_root)

    store.set_run_indexes(
        conn, run_id, ann_root=str(ann_root), standard_root=str(standard_root)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True, help="the DuckDB file to read/update")
    parser.add_argument("--run-id", required=True, help="the run to enable lookup for")
    parser.add_argument(
        "--index-root",
        type=Path,
        required=True,
        help="directory for the persisted ann/ and standard/ index artifacts",
    )
    args = parser.parse_args(argv)

    conn = store.connect(args.db)
    if store.get_run(conn, args.run_id) is None:
        parser.error(f"no run {args.run_id!r} in {args.db}")
    build_index(conn, args.run_id, args.index_root)
    conn.close()

    print(f"run {args.run_id}: indexes built at {args.index_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
