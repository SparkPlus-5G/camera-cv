"""Sparse representation module."""

from .motion_parallax import MotionParallaxEstimator
from .spatial_coherence import SpatialCoherenceAnalyzer, SparseMotionCloud
from .disturbance_classifier import DisturbanceClassifier, DisturbanceType

__all__ = [
    "MotionParallaxEstimator",
    "SpatialCoherenceAnalyzer",
    "SparseMotionCloud",
    "DisturbanceClassifier",
    "DisturbanceType"
]
