"""Evaluation: ranking metrics, entity-grouped splits, and the TF-IDF baseline.

Everything here is dataset-agnostic -- it consumes `Record` objects and knows
nothing about which benchmark produced them (CLAUDE.md, Layout). The dataset
name is resolved once, in `dedup.data`, and passed through as a string.

The metrics live here rather than in `model/` because `blocking/`, `model/`
and `cluster/` all need the same two things: a precision/recall curve that
stays correct when the candidate set has been pruned, and a split that groups
on entity rather than pair.
"""
