"""
Spatial coherence analysis and sparse motion cloud generation.
The core sparse representation for agent integration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import (
    COHERENCE_GRID, MOTION_MAGNITUDE_THRESHOLD,
    DIRECTION_VARIANCE_THRESHOLD
)
from motion.motion_field import MotionFieldState
from representation.motion_parallax import MotionParallaxEstimator, RelativeDepthMap
from representation.depth_layers import DepthLayerAnalyzer, LayerCoherenceResult


@dataclass
class SparseMotionCloud:
    """
    Enhanced motion-induced point-cloud-like representation.
    
    Includes multi-frame depth accumulation and layer analysis.
    """
    # Grid dimensions
    grid_cols: int
    grid_rows: int
    
    # Per-cell motion characteristics
    motion_magnitude: np.ndarray      # Average motion per cell (rows x cols)
    motion_direction: np.ndarray      # Dominant direction per cell (radians)
    direction_variance: np.ndarray    # Direction spread per cell (low = coherent)
    relative_depth_order: np.ndarray  # Accumulated depth (0=far, 1=near)
    temporal_consistency: np.ndarray  # Stability over time (0=unstable, 1=stable)
    
    # Boolean masks
    active_cells: np.ndarray          # Cells with significant motion
    coherent_cells: np.ndarray        # Cells with spatially coherent motion
    
    # Aggregate metrics
    scene_coherence_score: float      # 0-1: overall spatial coherence
    motion_intensity: float           # 0-1: normalized motion level
    active_cell_ratio: float          # Fraction of cells with motion
    coherent_cell_ratio: float        # Fraction of cells that are coherent
    
    # Enhanced: Depth layer analysis
    depth_layers: Optional[np.ndarray] = None  # Per-cell layer (0=far, 1=mid, 2=near)
    depth_stability: Optional[np.ndarray] = None  # Per-cell depth stability
    layer_coherence: Optional[LayerCoherenceResult] = None  # Layer analysis result
    depth_anomaly_detected: bool = False  # Quick flag for depth anomaly


class SpatialCoherenceAnalyzer:
    """
    Generates sparse motion cloud and computes coherence metrics.
    
    Enhanced with multi-frame depth accumulation and layer analysis.
    """
    
    def __init__(self, grid_size: tuple = COHERENCE_GRID, enable_layer_analysis: bool = True):
        self.grid_size = grid_size
        self.enable_layer_analysis = enable_layer_analysis
        
        self._parallax = MotionParallaxEstimator(grid_size)
        self._layer_analyzer = DepthLayerAnalyzer(grid_size) if enable_layer_analysis else None
        
        # History for temporal analysis
        self._direction_history: list = []
        self._max_history = 10
    
    def generate_sparse_cloud(self, 
                              motion_field: MotionFieldState,
                              temporal_consistency: Optional[np.ndarray] = None
                              ) -> SparseMotionCloud:
        """
        Generate sparse motion cloud from motion field state.
        
        Args:
            motion_field: MotionFieldState from MotionVectorField analyzer
            temporal_consistency: Optional pre-computed temporal consistency grid
            
        Returns:
            SparseMotionCloud representation
        """
        rows, cols = motion_field.grid_rows, motion_field.grid_cols
        
        # Get relative depth from motion parallax
        depth_map = self._parallax.estimate(
            motion_field.magnitude_grid,
            motion_field.point_count_grid
        )
        
        # Compute temporal consistency if not provided
        if temporal_consistency is None:
            temporal_consistency = self._compute_direction_consistency(
                motion_field.direction_grid
            )
        
        # Identify active cells (significant motion)
        active_cells = motion_field.motion_mask
        
        # Identify coherent cells (low direction variance)
        coherent_cells = (
            active_cells & 
            (motion_field.direction_variance_grid < DIRECTION_VARIANCE_THRESHOLD)
        )
        
        # Compute scene-level coherence score
        scene_coherence = self._compute_scene_coherence(
            motion_field, coherent_cells
        )
        
        # Normalize motion intensity
        max_expected_motion = 50.0  # Pixels
        motion_intensity = min(
            motion_field.global_magnitude_mean / max_expected_motion,
            1.0
        )
        
        # Compute ratios
        total_cells = rows * cols
        active_cell_ratio = np.sum(active_cells) / total_cells
        coherent_cell_ratio = np.sum(coherent_cells) / total_cells
        
        # Enhanced: Depth layer analysis
        depth_layers = None
        depth_stability = None
        layer_coherence = None
        depth_anomaly_detected = False
        
        if self.enable_layer_analysis and self._layer_analyzer is not None:
            # Get layer grid from the enhanced parallax estimator
            if depth_map.layer_grid is not None:
                depth_layers = depth_map.layer_grid
            else:
                # Compute layers from depth grid
                depth_layers = np.floor(depth_map.depth_grid * 3).astype(np.int32)
                depth_layers = np.clip(depth_layers, 0, 2)
            
            # Get stability from parallax estimator
            if depth_map.stability_grid is not None:
                depth_stability = depth_map.stability_grid
            
            # Run layer coherence analysis
            layer_coherence = self._layer_analyzer.analyze(
                depth_layers,
                motion_field.magnitude_grid,
                motion_field.direction_grid,
                active_cells
            )
            depth_anomaly_detected = layer_coherence.depth_anomaly_detected
        
        return SparseMotionCloud(
            grid_cols=cols,
            grid_rows=rows,
            motion_magnitude=motion_field.magnitude_grid,
            motion_direction=motion_field.direction_grid,
            direction_variance=motion_field.direction_variance_grid,
            relative_depth_order=depth_map.depth_grid,
            temporal_consistency=temporal_consistency,
            active_cells=active_cells,
            coherent_cells=coherent_cells,
            scene_coherence_score=scene_coherence,
            motion_intensity=motion_intensity,
            active_cell_ratio=active_cell_ratio,
            coherent_cell_ratio=coherent_cell_ratio,
            # Enhanced fields
            depth_layers=depth_layers,
            depth_stability=depth_stability,
            layer_coherence=layer_coherence,
            depth_anomaly_detected=depth_anomaly_detected
        )
    
    def _compute_scene_coherence(self, 
                                  motion_field: MotionFieldState,
                                  coherent_cells: np.ndarray) -> float:
        """
        Compute overall scene coherence score.
        
        High coherence means:
        1. Many cells have motion
        2. Motion directions are aligned across cells
        3. Motion magnitudes are similar across cells
        
        Returns:
            Score from 0.0 (no coherence) to 1.0 (perfect coherence)
        """
        if motion_field.active_cell_count == 0:
            return 0.0
        
        # Component 1: Spatial coverage (are many cells moving?)
        total_cells = motion_field.grid_rows * motion_field.grid_cols
        coverage = motion_field.active_cell_count / total_cells
        
        # Component 2: Direction alignment
        # Use the active cell directions to compute alignment
        active_mask = motion_field.motion_mask
        if np.sum(active_mask) < 2:
            direction_alignment = 0.0
        else:
            active_directions = motion_field.direction_grid[active_mask]
            
            # Circular variance of directions
            sin_sum = np.sum(np.sin(active_directions))
            cos_sum = np.sum(np.cos(active_directions))
            n = len(active_directions)
            
            # R is concentration (0 = uniform, 1 = all same direction)
            r = np.sqrt(sin_sum**2 + cos_sum**2) / n
            direction_alignment = r
        
        # Component 3: Magnitude uniformity
        active_magnitudes = motion_field.magnitude_grid[active_mask]
        if len(active_magnitudes) > 0 and np.mean(active_magnitudes) > 0:
            cv = np.std(active_magnitudes) / np.mean(active_magnitudes)
            magnitude_uniformity = max(0, 1 - cv)  # Lower CV = more uniform
        else:
            magnitude_uniformity = 0.0
        
        # Combine components (weighted average)
        coherence = (
            0.3 * coverage +
            0.5 * direction_alignment +
            0.2 * magnitude_uniformity
        )
        
        return float(np.clip(coherence, 0, 1))
    
    def _compute_direction_consistency(self, 
                                        direction_grid: np.ndarray) -> np.ndarray:
        """
        Compute per-cell direction consistency over temporal history.
        
        Returns:
            Grid of consistency scores (0-1)
        """
        rows, cols = direction_grid.shape
        
        # Add to history
        self._direction_history.append(direction_grid.copy())
        if len(self._direction_history) > self._max_history:
            self._direction_history.pop(0)
        
        if len(self._direction_history) < 2:
            return np.zeros((rows, cols), dtype=np.float32)
        
        # Stack history and compute circular variance per cell
        dir_stack = np.stack(self._direction_history)  # Shape: (history, rows, cols)
        
        consistency = np.zeros((rows, cols), dtype=np.float32)
        
        for r in range(rows):
            for c in range(cols):
                cell_dirs = dir_stack[:, r, c]
                
                # Skip cells with no motion history
                if np.all(cell_dirs == 0):
                    continue
                
                # Circular variance
                sin_sum = np.sum(np.sin(cell_dirs))
                cos_sum = np.sum(np.cos(cell_dirs))
                n = len(cell_dirs)
                
                r_val = np.sqrt(sin_sum**2 + cos_sum**2) / n
                consistency[r, c] = r_val
        
        return consistency
    
    def reset(self) -> None:
        """Reset history."""
        self._direction_history.clear()
        self._parallax.reset()
