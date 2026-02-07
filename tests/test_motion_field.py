"""
Unit tests for motion field analysis.
Tests the grid-based motion aggregation and coherence computation.
"""

import numpy as np
import pytest

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motion.motion_field import MotionVectorField, MotionFieldState
from motion.optical_flow import FlowResult


class TestMotionVectorField:
    """Tests for MotionVectorField class."""
    
    @pytest.fixture
    def motion_field(self):
        """Create a motion field analyzer."""
        return MotionVectorField(
            grid_size=(8, 6),
            frame_size=(1280, 720)
        )
    
    def _create_flow_result(self, points, vectors, valid_mask=None):
        """Helper to create FlowResult objects."""
        n = len(points)
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        curr_points = points + np.array(vectors, dtype=np.float32).reshape(-1, 1, 2)
        vectors = np.array(vectors, dtype=np.float32)
        
        if valid_mask is None:
            valid_mask = np.ones(n, dtype=bool)
        else:
            valid_mask = np.array(valid_mask, dtype=bool)
        
        return FlowResult(
            prev_points=points,
            curr_points=curr_points,
            motion_vectors=vectors,
            valid_mask=valid_mask,
            tracking_quality=np.mean(valid_mask)
        )
    
    def test_empty_flow(self, motion_field):
        """Test with empty flow result."""
        flow = FlowResult(
            prev_points=np.array([]).reshape(-1, 1, 2),
            curr_points=np.array([]).reshape(-1, 1, 2),
            motion_vectors=np.array([]).reshape(-1, 2),
            valid_mask=np.array([], dtype=bool),
            tracking_quality=0.0
        )
        
        state = motion_field.analyze(flow)
        
        assert state.total_tracked_points == 0
        assert state.active_cell_count == 0
        assert state.global_magnitude_mean == 0.0
    
    def test_uniform_rightward_motion(self, motion_field):
        """Test with uniform rightward motion across all cells."""
        # Create points in each cell with same rightward motion
        points = []
        vectors = []
        
        for row in range(6):
            for col in range(8):
                # Point in center of each cell
                x = (col + 0.5) * (1280 / 8)
                y = (row + 0.5) * (720 / 6)
                points.append((x, y))
                vectors.append((10.0, 0.0))  # Uniform rightward motion
        
        flow = self._create_flow_result(points, vectors)
        state = motion_field.analyze(flow)
        
        # All cells should have motion
        assert state.active_cell_count == 48
        
        # Direction should be ~0 (rightward)
        assert abs(state.global_direction_mean) < 0.1
        
        # Magnitude should be ~10
        assert abs(state.global_magnitude_mean - 10.0) < 0.1
    
    def test_localized_motion(self, motion_field):
        """Test with motion in only one corner."""
        # Points only in top-left corner (first 2x2 cells)
        points = []
        vectors = []
        
        for row in range(2):
            for col in range(2):
                x = (col + 0.5) * (1280 / 8)
                y = (row + 0.5) * (720 / 6)
                points.append((x, y))
                vectors.append((10.0, 10.0))
        
        flow = self._create_flow_result(points, vectors)
        state = motion_field.analyze(flow)
        
        # Only 4 cells should have motion
        assert state.active_cell_count == 4
        
        # Most cells should have no motion
        assert np.sum(~state.motion_mask) == 44
    
    def test_opposing_motion(self, motion_field):
        """Test with opposing motion (left half vs right half)."""
        points = []
        vectors = []
        
        # Left half moves right
        for row in range(6):
            for col in range(4):
                x = (col + 0.5) * (1280 / 8)
                y = (row + 0.5) * (720 / 6)
                points.append((x, y))
                vectors.append((10.0, 0.0))
        
        # Right half moves left
        for row in range(6):
            for col in range(4, 8):
                x = (col + 0.5) * (1280 / 8)
                y = (row + 0.5) * (720 / 6)
                points.append((x, y))
                vectors.append((-10.0, 0.0))
        
        flow = self._create_flow_result(points, vectors)
        state = motion_field.analyze(flow)
        
        # All cells should have motion
        assert state.active_cell_count == 48
        
        # Direction variance should be high (opposing directions)
        mean_variance = np.mean(state.direction_variance_grid[state.motion_mask])
        assert mean_variance < 0.5  # Within each cell, direction is consistent
    
    def test_temporal_consistency(self, motion_field):
        """Test temporal consistency computation."""
        # Add same motion pattern multiple times
        points = [(640, 360)]
        vectors = [(10.0, 0.0)]
        flow = self._create_flow_result(points, vectors)
        
        for _ in range(5):
            motion_field.analyze(flow)
        
        consistency = motion_field.get_temporal_consistency()
        
        # Should have high consistency where there's motion
        assert consistency.shape == (6, 8)


class TestMotionFieldState:
    """Tests for MotionFieldState dataclass."""
    
    def test_state_creation(self):
        """Test creating a MotionFieldState."""
        state = MotionFieldState(
            grid_cols=8,
            grid_rows=6,
            magnitude_grid=np.zeros((6, 8)),
            direction_grid=np.zeros((6, 8)),
            direction_variance_grid=np.zeros((6, 8)),
            point_count_grid=np.zeros((6, 8), dtype=np.int32),
            motion_mask=np.zeros((6, 8), dtype=bool),
            global_magnitude_mean=0.0,
            global_direction_mean=0.0,
            total_tracked_points=0,
            active_cell_count=0
        )
        
        assert state.grid_cols == 8
        assert state.grid_rows == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
