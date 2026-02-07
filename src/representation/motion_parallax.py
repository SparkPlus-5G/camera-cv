"""
Enhanced Motion parallax-based relative depth estimation.
Multi-frame accumulation for stable depth ordering.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

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
    
    # Enhanced: stability metrics
    stability_grid: Optional[np.ndarray] = None  # How stable each cell is over time
    layer_grid: Optional[np.ndarray] = None  # Discrete layer assignment (0=far, 2=near)


@dataclass
class DepthAccumulator:
    """Temporal accumulator for stable depth estimation."""
    depth_ema: np.ndarray  # Exponential moving average of depth
    confidence_ema: np.ndarray  # EMA of confidence
    variance: np.ndarray  # Per-cell variance (stability measure)
    update_count: int = 0


class EnhancedParallaxEstimator:
    """
    Enhanced motion parallax estimator with multi-frame accumulation.
    
    Improvements over basic version:
    - EMA-based temporal smoothing with adaptive alpha
    - Per-cell stability tracking
    - Automatic layer segmentation
    - Motion consistency validation
    
    Principle:
    Under camera motion, nearby objects have larger optical flow magnitudes
    than distant objects. By accumulating motion over multiple frames,
    we build a stable relative depth ordering.
    """
    
    def __init__(self, 
                 grid_size: tuple = COHERENCE_GRID,
                 history_length: int = 15,
                 ema_alpha: float = 0.15,
                 num_layers: int = 3):
        """
        Args:
            grid_size: (cols, rows) for the analysis grid
            history_length: Number of frames to keep in history
            ema_alpha: Smoothing factor for EMA (lower = smoother)
            num_layers: Number of depth layers (default 3: near/mid/far)
        """
        self.grid_size = grid_size  # (cols, rows)
        self.history_length = history_length
        self.ema_alpha = ema_alpha
        self.num_layers = num_layers
        
        rows, cols = grid_size[1], grid_size[0]
        
        # Initialize accumulator
        self._accumulator = DepthAccumulator(
            depth_ema=np.full((rows, cols), 0.5, dtype=np.float32),
            confidence_ema=np.zeros((rows, cols), dtype=np.float32),
            variance=np.zeros((rows, cols), dtype=np.float32),
            update_count=0
        )
        
        # History buffers for variance calculation
        self._depth_history: deque = deque(maxlen=history_length)
        self._magnitude_history: deque = deque(maxlen=history_length)
        
        # Adaptive alpha bounds
        self._min_alpha = 0.05
        self._max_alpha = 0.4
    
    def estimate(self, magnitude_grid: np.ndarray,
                 point_count_grid: np.ndarray) -> RelativeDepthMap:
        """
        Estimate relative depth with multi-frame accumulation.
        
        Args:
            magnitude_grid: Per-cell average motion magnitude (rows x cols)
            point_count_grid: Per-cell point counts for confidence
            
        Returns:
            RelativeDepthMap with accumulated depth ordering
        """
        rows, cols = magnitude_grid.shape
        
        # Find cells with sufficient motion
        valid_mask = magnitude_grid > MOTION_MAGNITUDE_THRESHOLD
        
        if not np.any(valid_mask):
            return self._create_empty_result(rows, cols)
        
        # Compute instantaneous depth from motion magnitude
        instant_depth = self._magnitude_to_depth(magnitude_grid, valid_mask)
        
        # Compute instantaneous confidence
        instant_confidence = self._compute_confidence(
            magnitude_grid, point_count_grid, valid_mask
        )
        
        # Compute adaptive alpha based on motion consistency
        alpha = self._compute_adaptive_alpha(magnitude_grid)
        
        # Update EMA accumulators
        self._update_accumulators(instant_depth, instant_confidence, valid_mask, alpha)
        
        # Add to history for variance calculation
        self._depth_history.append(instant_depth.copy())
        self._magnitude_history.append(magnitude_grid.copy())
        
        # Compute stability (inverse of variance)
        stability_grid = self._compute_stability()
        
        # Compute discrete layers
        layer_grid = self._compute_layers(self._accumulator.depth_ema)
        
        self._accumulator.update_count += 1
        
        return RelativeDepthMap(
            depth_grid=self._accumulator.depth_ema.copy(),
            confidence_grid=self._accumulator.confidence_ema.copy(),
            valid_cells=int(np.sum(valid_mask)),
            stability_grid=stability_grid,
            layer_grid=layer_grid
        )
    
    def _magnitude_to_depth(self, magnitude_grid: np.ndarray,
                            valid_mask: np.ndarray) -> np.ndarray:
        """Convert motion magnitude to relative depth (0=far, 1=near)."""
        rows, cols = magnitude_grid.shape
        depth_grid = np.full((rows, cols), 0.5, dtype=np.float32)
        
        valid_mags = magnitude_grid[valid_mask]
        if len(valid_mags) == 0:
            return depth_grid
        
        mag_min = np.min(valid_mags)
        mag_max = np.max(valid_mags)
        mag_range = mag_max - mag_min
        
        if mag_range > 0.5:  # Sufficient range for meaningful depth
            normalized = (magnitude_grid - mag_min) / mag_range
            depth_grid = np.where(valid_mask, normalized, 0.5)
        else:
            # Insufficient motion range - mark as mid-depth
            depth_grid = np.where(valid_mask, 0.5, 0.5)
        
        return depth_grid.astype(np.float32)
    
    def _compute_confidence(self, magnitude_grid: np.ndarray,
                           point_count_grid: np.ndarray,
                           valid_mask: np.ndarray) -> np.ndarray:
        """Compute per-cell confidence in depth estimate."""
        # Factor 1: Point count (more points = more reliable)
        max_points = np.max(point_count_grid) if np.max(point_count_grid) > 0 else 1
        point_factor = np.clip(point_count_grid / max_points, 0, 1)
        
        # Factor 2: Motion magnitude (higher motion = better parallax estimate)
        max_mag = np.max(magnitude_grid) if np.max(magnitude_grid) > 0 else 1
        magnitude_factor = np.clip(magnitude_grid / max_mag, 0, 1)
        
        # Factor 3: Valid mask
        valid_factor = valid_mask.astype(np.float32)
        
        # Combined confidence
        confidence = point_factor * magnitude_factor * valid_factor
        
        return confidence.astype(np.float32)
    
    def _compute_adaptive_alpha(self, magnitude_grid: np.ndarray) -> np.ndarray:
        """
        Compute per-cell adaptive smoothing factor.
        
        High motion consistency -> lower alpha (smoother)
        Low motion consistency -> higher alpha (more responsive)
        """
        rows, cols = magnitude_grid.shape
        
        if len(self._magnitude_history) < 2:
            return np.full((rows, cols), self.ema_alpha, dtype=np.float32)
        
        # Compute motion variance over recent history
        mag_stack = np.stack(list(self._magnitude_history))
        motion_variance = np.var(mag_stack, axis=0)
        
        # Normalize variance
        max_var = np.max(motion_variance) if np.max(motion_variance) > 0 else 1
        normalized_var = motion_variance / max_var
        
        # Map to alpha range (high variance = high alpha)
        alpha = self._min_alpha + normalized_var * (self._max_alpha - self._min_alpha)
        
        return alpha.astype(np.float32)
    
    def _update_accumulators(self, instant_depth: np.ndarray,
                             instant_confidence: np.ndarray,
                             valid_mask: np.ndarray,
                             alpha: np.ndarray) -> None:
        """Update EMA accumulators with new frame data."""
        # Only update cells with valid data
        update_mask = valid_mask & (instant_confidence > 0.1)
        
        # EMA update for depth
        self._accumulator.depth_ema = np.where(
            update_mask,
            alpha * instant_depth + (1 - alpha) * self._accumulator.depth_ema,
            self._accumulator.depth_ema
        ).astype(np.float32)
        
        # EMA update for confidence
        conf_alpha = np.minimum(alpha * 2, 0.5)  # Confidence adapts faster
        self._accumulator.confidence_ema = np.where(
            update_mask,
            conf_alpha * instant_confidence + (1 - conf_alpha) * self._accumulator.confidence_ema,
            self._accumulator.confidence_ema * 0.95  # Decay inactive cells
        ).astype(np.float32)
    
    def _compute_stability(self) -> np.ndarray:
        """
        Compute per-cell stability based on depth variance over time.
        
        Returns:
            Stability grid (0=unstable, 1=very stable)
        """
        if len(self._depth_history) < 3:
            rows, cols = self._accumulator.depth_ema.shape
            return np.zeros((rows, cols), dtype=np.float32)
        
        # Stack depth history
        depth_stack = np.stack(list(self._depth_history))
        
        # Compute variance
        variance = np.var(depth_stack, axis=0)
        
        # Convert to stability (inverse of variance)
        # Variance of 0.1 = stability of ~0.5
        stability = 1.0 / (1.0 + variance * 10)
        
        # Store for external access
        self._accumulator.variance = variance.astype(np.float32)
        
        return stability.astype(np.float32)
    
    def _compute_layers(self, depth_grid: np.ndarray) -> np.ndarray:
        """
        Segment depth into discrete layers.
        
        Returns:
            Layer grid (0=far, 1=mid, 2=near for 3 layers)
        """
        # Quantize depth into layers
        layer_grid = np.floor(depth_grid * self.num_layers).astype(np.int32)
        layer_grid = np.clip(layer_grid, 0, self.num_layers - 1)
        
        return layer_grid
    
    def _create_empty_result(self, rows: int, cols: int) -> RelativeDepthMap:
        """Create empty result when no motion detected."""
        return RelativeDepthMap(
            depth_grid=np.full((rows, cols), 0.5, dtype=np.float32),
            confidence_grid=np.zeros((rows, cols), dtype=np.float32),
            valid_cells=0,
            stability_grid=np.zeros((rows, cols), dtype=np.float32),
            layer_grid=np.ones((rows, cols), dtype=np.int32)  # Mid layer
        )
    
    def get_layer_statistics(self) -> dict:
        """
        Get statistics about the current depth layers.
        
        Returns:
            Dict with per-layer statistics
        """
        depth_grid = self._accumulator.depth_ema
        confidence_grid = self._accumulator.confidence_ema
        
        layer_grid = self._compute_layers(depth_grid)
        
        stats = {}
        for layer in range(self.num_layers):
            layer_mask = layer_grid == layer
            cell_count = np.sum(layer_mask)
            
            if cell_count > 0:
                avg_confidence = np.mean(confidence_grid[layer_mask])
                avg_depth = np.mean(depth_grid[layer_mask])
            else:
                avg_confidence = 0.0
                avg_depth = 0.0
            
            layer_name = ['far', 'mid', 'near'][layer] if self.num_layers == 3 else f'layer_{layer}'
            stats[layer_name] = {
                'cell_count': int(cell_count),
                'avg_confidence': float(avg_confidence),
                'avg_depth': float(avg_depth),
                'coverage': float(cell_count / depth_grid.size)
            }
        
        return stats
    
    def get_accumulated_depth(self) -> np.ndarray:
        """Get the current accumulated depth map."""
        return self._accumulator.depth_ema.copy()
    
    def get_stability_map(self) -> np.ndarray:
        """Get the current stability map."""
        return self._compute_stability()
    
    def reset(self) -> None:
        """Reset all accumulators and history."""
        rows, cols = self.grid_size[1], self.grid_size[0]
        self._accumulator = DepthAccumulator(
            depth_ema=np.full((rows, cols), 0.5, dtype=np.float32),
            confidence_ema=np.zeros((rows, cols), dtype=np.float32),
            variance=np.zeros((rows, cols), dtype=np.float32),
            update_count=0
        )
        self._depth_history.clear()
        self._magnitude_history.clear()


# Keep backward compatibility with original class name
MotionParallaxEstimator = EnhancedParallaxEstimator
