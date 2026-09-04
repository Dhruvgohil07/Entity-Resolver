"""Canonical record model.

Every dataset (Abt-Buy, Amazon-Google, DBLP-*, synthetic corruptions) is mapped
into this shape before touching any later stage -- that is what keeps
normalize/blocking/features/model/cluster dataset-agnostic (CLAUDE.md, Layout).

Not yet designed. Constraints it needs to satisfy, carried over from
CLAUDE.md's Invariants:
  - missing values must survive as explicit signal, not collapse to a default,
    so downstream missingness-indicator features have something to key off;
  - an entity/cluster id slot is required so ground-truth datasets can be
    split by entity, never by pair (pair-level splits leak identity).
"""
