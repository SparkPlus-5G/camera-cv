"""
Motion parallax-based relative depth estimation.
NOT true metric depth - motion magnitude proxy for relative scene layering.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import COHERENCE_GRID, MOTION_MAGNITUDE_THRESHOLD


@dataclass
class RelativeDepthMap:
    """
    Relative depth ordering based on motion parallax.
    
    IMPORTANT: This is NOT metric depth. It's a relative ordering
    based on the principle that under camera motion, nearby objects
    have larger optical flow than distant objects.
    
    Values range from 0.0 (far/background) to 1.0 (near/foreground).
    """
    depth_grid: np.ndarray  # Shape: (rows, cols), values 0.0-1.0
    confidence_grid: np.ndarray  # Per-cell confidence in depth estimate
    valid_cells: int  # Number of cells with valid depth estimates


class MotionParallaxEstimator:
    """
    Estimates relative depth ordering from differential motion.
    
    Principle:
    - Under camera motion (not object motion), nearby objects have
      larger optical flow magnitudes than distant objects.
    - We compute relative depth as normalized inverse motion magnitude.
    
    Limitations:
    - Only valid when camera is moving
    - Breaks down with independent object motion
    - Provides ordering, not metric depth
    
    Use case:
    - Distinguishing foreground vs background motion patterns
    - Validating whether motion is scene-level (all depths move together)
      vs localized (specific depth layer moves independently)
    """
    
    def __init__(self, grid_size: tuple = COHERENCE_GRID):
        self.grid_size = grid_size  # (cols, rows)
        
        # History for temporal smoothing
        self._depth_history: list = []
        self._max_history = 5
    
    def estimate(self, magnitude_grid: np.ndarray,
                 point_count_grid: np.ndarray) -> RelativeDepthMap:
        """
        Estimate relative depth from motion magnitude grid.
        
        Args:
            magnitude_grid: Per-cell average motion magnitude (rows x cols)
            point_count_grid: Per-cell point counts for confidence
            
        Returns:
            RelativeDepthMap with relative depth ordering
        """
        rows, cols = magnitude_grid.shape
        
        # Find cells with sufficient motion for depth estimation
        valid_mask = magnitude_grid > MOTION_MAGNITUDE_THRESHOLD
        
        if not np.any(valid_mask):
            return RelativeDepthMap(
                depth_grid=np.full((rows, cols), 0.5, dtype=np.float32),
                confidence_grid=np.zeros((rows, cols), dtype=np.float32),
                valid_cells=0
            )
        
        # Normalize magnitudes to 0-1 range
        valid_mags = magnitude_grid[valid_mask]
        mag_min = np.min(valid_mags)
        mag_max = np.max(valid_mags)
        mag_range = mag_max - mag_min
        
        # Compute relative depth (inverse magnitude = larger motion is nearer)
        depth_grid = np.zeros((rows, cols), dtype=np.float32)
        
        if mag_range > 0:
            # Normalize: higher motion = nearer = higher depth value
            normalized_mag = (magnitude_grid - mag_min) / mag_range
            depth_grid = np.where(valid_mask, normalized_mag, 0.5)
        else:
            # All valid cells have same motion - assign middle depth
            depth_grid = np.where(valid_mask, 0.5, 0.5)
        
        # Compute confidence based on point count and motion magnitude
        # More points + higher motion = higher confidence
        max_points = np.max(point_count_grid) if np.max(point_count_grid) > 0 else 1
        point_confidence = point_count_grid / max_points
        motion_confidence = np.where(
            magnitude_grid > MOTION_MAGNITUDE_THRESHOLD,
            np.clip(magnitude_grid / (mag_max + 1e-6), 0, 1),
            0
        )
        confidence_grid = (point_confidence * motion_confidence).astype(np.float32)
        
        # Apply temporal smoothing if we have history
        if len(self._depth_history) > 0:
            depth_grid = self._temporal_smooth(depth_grid, valid_mask)
        
        # Update history
        self._depth_history.append(depth_grid.copy())
        if len(self._depth_history) > self._max_history:
            self._depth_history.pop(0)
        
        return RelativeDepthMap(
            depth_grid=depth_grid,
            confidence_grid=confidence_grid,
            valid_cells=int(np.sum(valid_mask))
        )
    
    def _temporal_smooth(self, depth_grid: np.ndarray,
                         valid_mask: np.ndarray) -> np.ndarray:
        """Apply exponential moving average for temporal smoothing."""
        alpha = 0.3  # Smoothing factor
        
        # Average with history
        history_avg = np.mean(self._depth_history, axis=0)
        
        # Blend current with history
        smoothed = np.where(
            valid_mask,
            alpha * depth_grid + (1 - alpha) * history_avg,
            depth_grid
        )
        
        return smoothed.astype(np.float32)
    
    def get_depth_layers(self, depth_map: RelativeDepthMap,
                         num_layers: int = 3) -> np.ndarray:
        """
        Quantize depth into discrete layers.
        
        Args:
            depth_map: RelativeDepthMap from estimate()
            num_layers: Number of depth layers (default 3: near/mid/far)
            
        Returns:
            Grid of layer indices (0 = far, num_layers-1 = near)
        """
        # Quantize depth values
        layer_grid = np.floor(depth_map.depth_grid * num_layers).astype(np.int32)
        layer_grid = np.clip(layer_grid, 0, num_layers - 1)
        
        return layer_grid
    
    def reset(self) -> None:
        """Reset history."""
        self._depth_history.clear()
