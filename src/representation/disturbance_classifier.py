"""
Disturbance classification: scene-level vs localized motion.
Key component for agent-based false positive suppression.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

from config import (
    LOCALIZED_THRESHOLD, SCENE_LEVEL_THRESHOLD,
    DIRECTION_VARIANCE_THRESHOLD
)
from representation.spatial_coherence import SparseMotionCloud


class DisturbanceType(Enum):
    """Classification of observed motion pattern."""
    NONE = "none"              # No significant motion detected
    LOCALIZED = "localized"    # Motion concentrated in small region
    SCENE_LEVEL = "scene_level"  # Correlated motion across scene


@dataclass
class DisturbanceResult:
    """Result of disturbance classification."""
    disturbance_type: DisturbanceType
    confidence: float          # 0-1: confidence in classification
    localized_region: Optional[Tuple[int, int, int, int]]  # (col1, row1, col2, row2) if localized
    explanation: str           # Human-readable explanation


class DisturbanceClassifier:
    """
    Classifies motion patterns as scene-level or localized.
    
    Key distinctions:
    - NONE: Less than 5% of cells have motion
    - LOCALIZED: Motion in <20% of cells, typically clustered
    - SCENE_LEVEL: Correlated motion in >50% of cells
    
    Use case:
    - Scene-level: Environmental response (supports sensor alerts)
    - Localized: Object motion (potentially false positive)
    """
    
    def __init__(self,
                 localized_threshold: float = LOCALIZED_THRESHOLD,
                 scene_level_threshold: float = SCENE_LEVEL_THRESHOLD):
        self.localized_threshold = localized_threshold
        self.scene_level_threshold = scene_level_threshold
    
    def classify(self, sparse_cloud: SparseMotionCloud) -> DisturbanceResult:
        """
        Classify the motion pattern in a sparse motion cloud.
        
        Args:
            sparse_cloud: SparseMotionCloud from SpatialCoherenceAnalyzer
            
        Returns:
            DisturbanceResult with classification and metadata
        """
        active_ratio = sparse_cloud.active_cell_ratio
        coherent_ratio = sparse_cloud.coherent_cell_ratio
        scene_coherence = sparse_cloud.scene_coherence_score
        
        # Check for no motion
        if active_ratio < 0.05:
            return DisturbanceResult(
                disturbance_type=DisturbanceType.NONE,
                confidence=0.9,
                localized_region=None,
                explanation="No significant motion detected in scene"
            )
        
        # Scene-level: high coverage AND high coherence
        if (active_ratio >= self.scene_level_threshold and 
            scene_coherence > 0.5):
            confidence = min(active_ratio, scene_coherence)
            return DisturbanceResult(
                disturbance_type=DisturbanceType.SCENE_LEVEL,
                confidence=confidence,
                localized_region=None,
                explanation=f"Correlated motion across {active_ratio*100:.0f}% of scene "
                           f"with {scene_coherence:.0%} coherence"
            )
        
        # Localized: low coverage OR low coherence
        if active_ratio < self.localized_threshold:
            # Find the localized region
            region = self._find_motion_region(sparse_cloud)
            return DisturbanceResult(
                disturbance_type=DisturbanceType.LOCALIZED,
                confidence=0.8,
                localized_region=region,
                explanation=f"Motion concentrated in {active_ratio*100:.0f}% of scene"
            )
        
        # Intermediate case: some coverage but low coherence
        # This suggests independent object motions, not environment-level
        if scene_coherence < 0.4:
            return DisturbanceResult(
                disturbance_type=DisturbanceType.LOCALIZED,
                confidence=0.6,
                localized_region=None,
                explanation=f"Motion in {active_ratio*100:.0f}% of scene but "
                           f"low coherence ({scene_coherence:.0%}) suggests independent motions"
            )
        
        # Default: scene-level with moderate confidence
        return DisturbanceResult(
            disturbance_type=DisturbanceType.SCENE_LEVEL,
            confidence=0.5,
            localized_region=None,
            explanation=f"Moderate motion coverage ({active_ratio*100:.0f}%) with "
                       f"moderate coherence ({scene_coherence:.0%})"
        )
    
    def _find_motion_region(self, 
                            sparse_cloud: SparseMotionCloud
                            ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find bounding box of motion region for localized disturbances.
        
        Returns:
            (col1, row1, col2, row2) bounding box of active cells
        """
        active_cells = sparse_cloud.active_cells
        
        if not np.any(active_cells):
            return None
        
        # Find rows and columns with motion
        active_rows = np.any(active_cells, axis=1)
        active_cols = np.any(active_cells, axis=0)
        
        row_indices = np.where(active_rows)[0]
        col_indices = np.where(active_cols)[0]
        
        if len(row_indices) == 0 or len(col_indices) == 0:
            return None
        
        return (
            int(col_indices[0]),
            int(row_indices[0]),
            int(col_indices[-1]),
            int(row_indices[-1])
        )
    
    def get_spatial_distribution(self, 
                                  sparse_cloud: SparseMotionCloud) -> dict:
        """
        Analyze spatial distribution of motion.
        
        Returns:
            Dict with distribution metrics
        """
        active_cells = sparse_cloud.active_cells
        rows, cols = active_cells.shape
        
        # Compute center of mass of motion
        if not np.any(active_cells):
            return {
                'center_col': cols / 2,
                'center_row': rows / 2,
                'spread_col': 0,
                'spread_row': 0,
                'is_centered': True,
                'is_edge': False
            }
        
        row_indices, col_indices = np.where(active_cells)
        
        center_row = np.mean(row_indices)
        center_col = np.mean(col_indices)
        spread_row = np.std(row_indices) if len(row_indices) > 1 else 0
        spread_col = np.std(col_indices) if len(col_indices) > 1 else 0
        
        # Check if motion is centered or at edges
        is_centered = (
            0.3 * rows <= center_row <= 0.7 * rows and
            0.3 * cols <= center_col <= 0.7 * cols
        )
        
        is_edge = (
            center_row < 0.2 * rows or center_row > 0.8 * rows or
            center_col < 0.2 * cols or center_col > 0.8 * cols
        )
        
        return {
            'center_col': float(center_col),
            'center_row': float(center_row),
            'spread_col': float(spread_col),
            'spread_row': float(spread_row),
            'is_centered': is_centered,
            'is_edge': is_edge
        }
