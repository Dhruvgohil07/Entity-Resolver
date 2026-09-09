"""Candidate pair -> ~25-35 dim feature vector: string, numeric, semantic, missingness."""

from dedup.features.base import FILL_VALUE, FeatureBlock, FeatureMatrix, FeatureSpec
from dedup.features.vectorize import PairFeaturizer, pair_labels, validate_registry

__all__ = [
    "FILL_VALUE",
    "FeatureBlock",
    "FeatureMatrix",
    "FeatureSpec",
    "PairFeaturizer",
    "pair_labels",
    "validate_registry",
]
