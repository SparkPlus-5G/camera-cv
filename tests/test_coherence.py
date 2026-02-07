"""
Unit tests for coherence analysis and disturbance classification.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motion.motion_field import MotionFieldState
from representation.spatial_coherence import SpatialCoherenceAnalyzer, SparseMotionCloud
from representation.disturbance_classifier import (
    DisturbanceClassifier, DisturbanceType, DisturbanceResult
)


class TestSpatialCoherenceAnalyzer:
    """Tests for SpatialCoherenceAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        return SpatialCoherenceAnalyzer(grid_size=(8, 6))
    
    def _create_motion_state(self, magnitude_grid, direction_grid=None):
        """Helper to create MotionFieldState."""
        rows, cols = magnitude_grid.shape
        
        if direction_grid is None:
            direction_grid = np.zeros_like(magnitude_grid)
        
        motion_mask = magnitude_grid > 2.0
        
        return MotionFieldState(
            grid_cols=cols,
            grid_rows=rows,
            magnitude_grid=magnitude_grid,
            direction_grid=direction_grid,
            direction_variance_grid=np.ones_like(magnitude_grid) * 0.1,
            point_count_grid=(magnitude_grid > 0).astype(np.int32) * 10,
            motion_mask=motion_mask,
            global_magnitude_mean=float(np.mean(magnitude_grid[motion_mask])) if np.any(motion_mask) else 0.0,
            global_direction_mean=0.0,
            total_tracked_points=int(np.sum(magnitude_grid > 0) * 10),
            active_cell_count=int(np.sum(motion_mask))
        )
    
    def test_no_motion(self, analyzer):
        """Test with no motion."""
        magnitude_grid = np.zeros((6, 8))
        state = self._create_motion_state(magnitude_grid)
        
        cloud = analyzer.generate_sparse_cloud(state)
        
        assert cloud.scene_coherence_score == 0.0
        assert cloud.motion_intensity == 0.0
        assert cloud.active_cell_ratio == 0.0
    
    def test_full_scene_coherent_motion(self, analyzer):
        """Test with coherent motion across entire scene."""
        magnitude_grid = np.ones((6, 8)) * 20.0
        direction_grid = np.zeros((6, 8))  # All same direction
        state = self._create_motion_state(magnitude_grid, direction_grid)
        
        cloud = analyzer.generate_sparse_cloud(state)
        
        assert cloud.scene_coherence_score > 0.7
        assert cloud.active_cell_ratio == 1.0
        assert cloud.motion_intensity > 0.3
    
    def test_localized_motion(self, analyzer):
        """Test with motion in only a few cells."""
        magnitude_grid = np.zeros((6, 8))
        magnitude_grid[0:2, 0:2] = 20.0  # Motion in corner only
        state = self._create_motion_state(magnitude_grid)
        
        cloud = analyzer.generate_sparse_cloud(state)
        
        # Low active cell ratio
        assert cloud.active_cell_ratio < 0.15
        
        # Coherence might still be high within the region
        assert cloud.coherent_cell_ratio <= cloud.active_cell_ratio


class TestDisturbanceClassifier:
    """Tests for DisturbanceClassifier."""
    
    @pytest.fixture
    def classifier(self):
        return DisturbanceClassifier()
    
    def _create_sparse_cloud(self, active_ratio, coherence_score, motion_intensity=0.5):
        """Helper to create SparseMotionCloud for testing."""
        rows, cols = 6, 8
        total_cells = rows * cols
        active_cells = int(total_cells * active_ratio)
        
        # Create active cell mask
        active_mask = np.zeros((rows, cols), dtype=bool)
        flat_indices = np.random.choice(total_cells, active_cells, replace=False)
        for idx in flat_indices:
            r, c = divmod(idx, cols)
            active_mask[r, c] = True
        
        return SparseMotionCloud(
            grid_cols=cols,
            grid_rows=rows,
            motion_magnitude=np.where(active_mask, 20.0, 0.0),
            motion_direction=np.zeros((rows, cols)),
            direction_variance=np.ones((rows, cols)) * 0.1,
            relative_depth_order=np.full((rows, cols), 0.5),
            temporal_consistency=np.zeros((rows, cols)),
            active_cells=active_mask,
            coherent_cells=active_mask,
            scene_coherence_score=coherence_score,
            motion_intensity=motion_intensity,
            active_cell_ratio=active_ratio,
            coherent_cell_ratio=active_ratio
        )
    
    def test_no_motion(self, classifier):
        """Test classification of no motion."""
        cloud = self._create_sparse_cloud(0.02, 0.0, 0.0)
        
        result = classifier.classify(cloud)
        
        assert result.disturbance_type == DisturbanceType.NONE
    
    def test_localized_motion(self, classifier):
        """Test classification of localized motion."""
        cloud = self._create_sparse_cloud(0.10, 0.3, 0.3)
        
        result = classifier.classify(cloud)
        
        assert result.disturbance_type == DisturbanceType.LOCALIZED
    
    def test_scene_level_motion(self, classifier):
        """Test classification of scene-level motion."""
        cloud = self._create_sparse_cloud(0.70, 0.8, 0.6)
        
        result = classifier.classify(cloud)
        
        assert result.disturbance_type == DisturbanceType.SCENE_LEVEL
    
    def test_high_coverage_low_coherence(self, classifier):
        """Test high coverage but low coherence (independent motions)."""
        cloud = self._create_sparse_cloud(0.60, 0.2, 0.5)
        
        result = classifier.classify(cloud)
        
        # Should be classified as localized due to low coherence
        assert result.disturbance_type == DisturbanceType.LOCALIZED


class TestDisturbanceResult:
    """Tests for DisturbanceResult dataclass."""
    
    def test_result_creation(self):
        """Test creating a DisturbanceResult."""
        result = DisturbanceResult(
            disturbance_type=DisturbanceType.SCENE_LEVEL,
            confidence=0.85,
            localized_region=None,
            explanation="Test explanation"
        )
        
        assert result.disturbance_type == DisturbanceType.SCENE_LEVEL
        assert result.confidence == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
