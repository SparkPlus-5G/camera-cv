"""
Motion vector field analysis.
Aggregates motion vectors into a spatial grid representation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque

from config import (
    COHERENCE_GRID, MOTION_MAGNITUDE_THRESHOLD,
    COHERENCE_HISTORY_FRAMES, FRAME_WIDTH, FRAME_HEIGHT
)
from motion.optical_flow import FlowResult


@dataclass
class CellMotionStats:
    """Motion statistics for a single grid cell."""
    magnitude_mean: float = 0.0
    magnitude_std: float = 0.0
    direction_mean: float = 0.0
    direction_std: float = 0.0
    point_count: int = 0
    has_motion: bool = False


@dataclass
class MotionFieldState:
    """
    Complete motion field state across the spatial grid.
    This is the primary input to the coherence analyzer.
    """
    # Grid dimensions
    grid_cols: int
    grid_rows: int
    
    # Per-cell statistics (shape: rows x cols)
    magnitude_grid: np.ndarray
    direction_grid: np.ndarray
    direction_variance_grid: np.ndarray
    point_count_grid: np.ndarray
    motion_mask: np.ndarray  # Boolean: which cells have significant motion
    
    # Global statistics
    global_magnitude_mean: float = 0.0
    global_direction_mean: float = 0.0
    total_tracked_points: int = 0
    active_cell_count: int = 0
    
    # Raw data for detailed analysis
    all_vectors: Optional[np.ndarray] = None
    all_points: Optional[np.ndarray] = None


class MotionVectorField:
    """
    Aggregates motion vectors into a spatial grid representation.
    
    Provides:
    - Per-cell motion magnitude and direction
    - Direction variance (coherence indicator)
    - Temporal history for consistency analysis
    """
    
    def __init__(self,
                 grid_size: tuple = COHERENCE_GRID,
                 frame_size: tuple = (FRAME_WIDTH, FRAME_HEIGHT),
                 history_length: int = COHERENCE_HISTORY_FRAMES):
        self.grid_size = grid_size  # (cols, rows)
        self.frame_size = frame_size  # (width, height)
        self.history_length = history_length
        
        # Cell dimensions
        self.cell_width = frame_size[0] / grid_size[0]
        self.cell_height = frame_size[1] / grid_size[1]
        
        # Temporal history
        self._history: deque = deque(maxlen=history_length)
    
    def analyze(self, flow_result: FlowResult) -> MotionFieldState:
        """
        Analyze flow result and produce grid-based motion field.
        
        Args:
            flow_result: Result from SparseFlowTracker
            
        Returns:
            MotionFieldState with per-cell motion statistics
        """
        cols, rows = self.grid_size
        
        # Initialize grids
        magnitude_grid = np.zeros((rows, cols), dtype=np.float32)
        direction_grid = np.zeros((rows, cols), dtype=np.float32)
        direction_variance_grid = np.zeros((rows, cols), dtype=np.float32)
        point_count_grid = np.zeros((rows, cols), dtype=np.int32)
        
        # Get valid points and vectors
        valid_mask = flow_result.valid_mask
        if not np.any(valid_mask):
            return self._create_empty_state()
        
        valid_points = flow_result.prev_points[valid_mask].reshape(-1, 2)
        valid_vectors = flow_result.motion_vectors[valid_mask]
        
        # Assign points to grid cells
        cell_indices = self._get_cell_indices(valid_points)
        
        # Aggregate per cell
        for row in range(rows):
            for col in range(cols):
                # Find points in this cell
                cell_mask = (cell_indices[:, 0] == col) & (cell_indices[:, 1] == row)
                cell_vectors = valid_vectors[cell_mask]
                
                if len(cell_vectors) == 0:
                    continue
                
                # Compute cell statistics
                stats = self._compute_cell_stats(cell_vectors)
                
                magnitude_grid[row, col] = stats.magnitude_mean
                direction_grid[row, col] = stats.direction_mean
                direction_variance_grid[row, col] = stats.direction_std
                point_count_grid[row, col] = stats.point_count
        
        # Create motion mask (cells with significant motion)
        motion_mask = magnitude_grid > MOTION_MAGNITUDE_THRESHOLD
        
        # Compute global statistics
        all_magnitudes = np.linalg.norm(valid_vectors, axis=1)
        global_magnitude_mean = float(np.mean(all_magnitudes)) if len(all_magnitudes) > 0 else 0.0
        
        # Global direction (circular mean)
        sin_sum = np.sum(np.sin(np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])))
        cos_sum = np.sum(np.cos(np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])))
        global_direction_mean = float(np.arctan2(sin_sum, cos_sum))
        
        state = MotionFieldState(
            grid_cols=cols,
            grid_rows=rows,
            magnitude_grid=magnitude_grid,
            direction_grid=direction_grid,
            direction_variance_grid=direction_variance_grid,
            point_count_grid=point_count_grid,
            motion_mask=motion_mask,
            global_magnitude_mean=global_magnitude_mean,
            global_direction_mean=global_direction_mean,
            total_tracked_points=len(valid_vectors),
            active_cell_count=int(np.sum(motion_mask)),
            all_vectors=valid_vectors,
            all_points=valid_points
        )
        
        # Add to history
        self._history.append(state)
        
        return state
    
    def _get_cell_indices(self, points: np.ndarray) -> np.ndarray:
        """Convert point coordinates to cell indices."""
        cols = np.clip(
            (points[:, 0] / self.cell_width).astype(int),
            0, self.grid_size[0] - 1
        )
        rows = np.clip(
            (points[:, 1] / self.cell_height).astype(int),
            0, self.grid_size[1] - 1
        )
        return np.column_stack([cols, rows])
    
    def _compute_cell_stats(self, vectors: np.ndarray) -> CellMotionStats:
        """Compute motion statistics for vectors in a cell."""
        if len(vectors) == 0:
            return CellMotionStats()
        
        magnitudes = np.linalg.norm(vectors, axis=1)
        directions = np.arctan2(vectors[:, 1], vectors[:, 0])
        
        # Circular mean for direction
        sin_sum = np.sum(np.sin(directions))
        cos_sum = np.sum(np.cos(directions))
        direction_mean = np.arctan2(sin_sum, cos_sum)
        
        # Circular variance
        r = np.sqrt(sin_sum**2 + cos_sum**2) / len(directions)
        direction_std = np.sqrt(-2 * np.log(max(r, 1e-10)))
        
        return CellMotionStats(
            magnitude_mean=float(np.mean(magnitudes)),
            magnitude_std=float(np.std(magnitudes)),
            direction_mean=float(direction_mean),
            direction_std=float(min(direction_std, np.pi)),  # Cap at pi
            point_count=len(vectors),
            has_motion=np.mean(magnitudes) > MOTION_MAGNITUDE_THRESHOLD
        )
    
    def _create_empty_state(self) -> MotionFieldState:
        """Create an empty motion field state."""
        cols, rows = self.grid_size
        return MotionFieldState(
            grid_cols=cols,
            grid_rows=rows,
            magnitude_grid=np.zeros((rows, cols), dtype=np.float32),
            direction_grid=np.zeros((rows, cols), dtype=np.float32),
            direction_variance_grid=np.zeros((rows, cols), dtype=np.float32),
            point_count_grid=np.zeros((rows, cols), dtype=np.int32),
            motion_mask=np.zeros((rows, cols), dtype=bool),
            global_magnitude_mean=0.0,
            global_direction_mean=0.0,
            total_tracked_points=0,
            active_cell_count=0
        )
    
    def get_temporal_consistency(self) -> np.ndarray:
        """
        Compute temporal consistency of motion across history.
        
        Returns:
            Grid of consistency scores (0-1), where 1 = highly consistent motion
        """
        if len(self._history) < 2:
            return np.zeros((self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        
        # Stack magnitude grids from history
        mag_history = np.stack([s.magnitude_grid for s in self._history])
        
        # Compute coefficient of variation (lower = more consistent)
        mean_mags = np.mean(mag_history, axis=0)
        std_mags = np.std(mag_history, axis=0)
        
        # Avoid division by zero
        cv = np.where(mean_mags > 0, std_mags / mean_mags, 1.0)
        
        # Convert to consistency score (inverse of CV, capped at 1)
        consistency = np.clip(1.0 - cv, 0, 1)
        
        return consistency.astype(np.float32)
    
    def get_history(self) -> list:
        """Get motion field history."""
        return list(self._history)
    
    def reset(self) -> None:
        """Reset history."""
        self._history.clear()
