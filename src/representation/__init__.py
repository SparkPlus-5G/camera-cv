"""Sparse representation module."""

from .motion_parallax import (
    MotionParallaxEstimator, 
    EnhancedParallaxEstimator,
    RelativeDepthMap
)
from .spatial_coherence import SpatialCoherenceAnalyzer, SparseMotionCloud
from .disturbance_classifier import DisturbanceClassifier, DisturbanceType
from .depth_layers import DepthLayerAnalyzer, DepthLayer, LayerCoherenceResult

__all__ = [
    "MotionParallaxEstimator",
    "EnhancedParallaxEstimator",
    "RelativeDepthMap",
    "SpatialCoherenceAnalyzer",
    "SparseMotionCloud",
    "DisturbanceClassifier",
    "DisturbanceType",
    "DepthLayerAnalyzer",
    "DepthLayer",
    "LayerCoherenceResult"
]

