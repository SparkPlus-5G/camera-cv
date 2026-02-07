"""
Depth layer analysis module.
Segments scene into discrete depth layers and analyzes per-layer coherence.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum

from config import COHERENCE_GRID


class DepthLayer(Enum):
    """Depth layer enumeration."""
    FAR = 0       # Background layer (lowest motion)
    MID = 1       # Middle layer  
    NEAR = 2      # Foreground layer (highest motion)


@dataclass
class LayerState:
    """State of a single depth layer."""
    layer: DepthLayer
    cell_count: int  # Number of cells in this layer
    coverage: float  # Fraction of scene covered
    avg_motion: float  # Average motion magnitude
    motion_coherence: float  # Direction coherence within layer
    is_active: bool  # Has significant motion
    dominant_direction: float  # Primary motion direction (radians)


@dataclass
class LayerCoherenceResult:
    """Result of per-layer coherence analysis."""
    layer_states: Dict[DepthLayer, LayerState]
    
    # Cross-layer analysis
    layers_moving_together: bool  # All active layers same direction
    depth_anomaly_detected: bool  # Unexpected layer behavior
    anomaly_layer: Optional[DepthLayer]  # Which layer is anomalous
    anomaly_description: str
    
    # Summary metrics
    total_active_layers: int
    scene_depth_coherence: float  # 0-1: how coherent is motion across depths


class DepthLayerAnalyzer:
    """
    Analyzes motion patterns across discrete depth layers.
    
    Use cases:
    1. Detect "depth anomalies" - when one layer moves differently
    2. Validate scene-level events - all layers should move together
    3. Identify foreground occlusions - near layer blocking mid/far
    
    A scene-level structural response (e.g., earthquake) should show:
    - All depth layers moving in the same direction
    - Similar motion magnitude scaling across layers
    
    A localized disturbance (e.g., person walking) should show:
    - Motion concentrated in one or two layers
    - Different direction from other layers
    """
    
    def __init__(self, 
                 grid_size: tuple = COHERENCE_GRID,
                 num_layers: int = 3,
                 min_layer_coverage: float = 0.05):
        """
        Args:
            grid_size: (cols, rows) for the analysis grid
            num_layers: Number of depth layers
            min_layer_coverage: Minimum fraction of cells for active layer
        """
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.min_layer_coverage = min_layer_coverage
        
        # Motion threshold for "active" layer
        self.motion_threshold = 2.0
        
        # Direction difference threshold for coherence
        self.direction_threshold = np.pi / 4  # 45 degrees
    
    def analyze(self,
                layer_grid: np.ndarray,
                magnitude_grid: np.ndarray,
                direction_grid: np.ndarray,
                active_mask: np.ndarray) -> LayerCoherenceResult:
        """
        Analyze motion coherence within and across depth layers.
        
        Args:
            layer_grid: Per-cell layer assignment (0=far, 2=near)
            magnitude_grid: Per-cell motion magnitude
            direction_grid: Per-cell motion direction (radians)
            active_mask: Boolean mask of cells with significant motion
            
        Returns:
            LayerCoherenceResult with per-layer and cross-layer analysis
        """
        total_cells = layer_grid.size
        
        # Analyze each layer
        layer_states = {}
        active_layers = []
        
        for layer_val in range(self.num_layers):
            layer = DepthLayer(layer_val)
            state = self._analyze_layer(
                layer, layer_val, layer_grid,
                magnitude_grid, direction_grid, active_mask, total_cells
            )
            layer_states[layer] = state
            
            if state.is_active:
                active_layers.append(state)
        
        # Cross-layer analysis
        layers_moving_together = self._check_layers_coherent(active_layers)
        anomaly_layer, anomaly_desc = self._detect_depth_anomaly(active_layers)
        
        # Compute scene depth coherence
        scene_depth_coherence = self._compute_depth_coherence(active_layers)
        
        return LayerCoherenceResult(
            layer_states=layer_states,
            layers_moving_together=layers_moving_together,
            depth_anomaly_detected=anomaly_layer is not None,
            anomaly_layer=anomaly_layer,
            anomaly_description=anomaly_desc,
            total_active_layers=len(active_layers),
            scene_depth_coherence=scene_depth_coherence
        )
    
    def _analyze_layer(self,
                       layer: DepthLayer,
                       layer_val: int,
                       layer_grid: np.ndarray,
                       magnitude_grid: np.ndarray,
                       direction_grid: np.ndarray,
                       active_mask: np.ndarray,
                       total_cells: int) -> LayerState:
        """Analyze a single depth layer."""
        # Get cells in this layer
        layer_mask = layer_grid == layer_val
        cell_count = np.sum(layer_mask)
        coverage = cell_count / total_cells
        
        # Check if layer has significant motion
        active_layer_mask = layer_mask & active_mask
        active_count = np.sum(active_layer_mask)
        
        if active_count < 3:  # Too few active cells
            return LayerState(
                layer=layer,
                cell_count=int(cell_count),
                coverage=float(coverage),
                avg_motion=0.0,
                motion_coherence=0.0,
                is_active=False,
                dominant_direction=0.0
            )
        
        # Compute layer statistics
        layer_magnitudes = magnitude_grid[active_layer_mask]
        layer_directions = direction_grid[active_layer_mask]
        
        avg_motion = float(np.mean(layer_magnitudes))
        
        # Compute direction coherence (circular statistics)
        sin_sum = np.sum(np.sin(layer_directions))
        cos_sum = np.sum(np.cos(layer_directions))
        n = len(layer_directions)
        
        # R is concentration (0 = uniform, 1 = all same direction)
        r = np.sqrt(sin_sum**2 + cos_sum**2) / n
        motion_coherence = float(r)
        
        # Dominant direction
        dominant_direction = float(np.arctan2(sin_sum, cos_sum))
        
        # Is layer significantly active?
        is_active = (
            avg_motion > self.motion_threshold and
            coverage > self.min_layer_coverage
        )
        
        return LayerState(
            layer=layer,
            cell_count=int(cell_count),
            coverage=float(coverage),
            avg_motion=avg_motion,
            motion_coherence=motion_coherence,
            is_active=is_active,
            dominant_direction=dominant_direction
        )
    
    def _check_layers_coherent(self, active_layers: List[LayerState]) -> bool:
        """Check if all active layers are moving in the same direction."""
        if len(active_layers) < 2:
            return True  # Single layer is trivially coherent
        
        # Compare all pairs of layers
        for i, layer_a in enumerate(active_layers):
            for layer_b in active_layers[i+1:]:
                # Angular difference
                diff = abs(layer_a.dominant_direction - layer_b.dominant_direction)
                diff = min(diff, 2 * np.pi - diff)  # Handle wrap-around
                
                if diff > self.direction_threshold:
                    return False
        
        return True
    
    def _detect_depth_anomaly(self, 
                              active_layers: List[LayerState]
                              ) -> Tuple[Optional[DepthLayer], str]:
        """
        Detect if one layer is behaving differently from others.
        
        Returns:
            (anomalous_layer, description) or (None, '')
        """
        if len(active_layers) < 2:
            return None, ""
        
        # Compute average direction
        sin_mean = np.mean([np.sin(l.dominant_direction) for l in active_layers])
        cos_mean = np.mean([np.cos(l.dominant_direction) for l in active_layers])
        avg_direction = np.arctan2(sin_mean, cos_mean)
        
        # Find layer most different from average
        max_diff = 0
        anomaly_layer = None
        
        for layer_state in active_layers:
            diff = abs(layer_state.dominant_direction - avg_direction)
            diff = min(diff, 2 * np.pi - diff)
            
            if diff > max_diff:
                max_diff = diff
                anomaly_layer = layer_state
        
        # Is the difference significant?
        if max_diff > self.direction_threshold and anomaly_layer is not None:
            layer_name = anomaly_layer.layer.name.lower()
            return (
                anomaly_layer.layer,
                f"{layer_name} layer moving in different direction "
                f"({np.degrees(max_diff):.0f}° from scene average)"
            )
        
        # Check for magnitude anomaly (one layer moving much more/less)
        magnitudes = [l.avg_motion for l in active_layers]
        mean_mag = np.mean(magnitudes)
        
        for layer_state in active_layers:
            # Check if magnitude is very different (>2x or <0.5x mean)
            if layer_state.avg_motion > mean_mag * 2.5:
                return (
                    layer_state.layer,
                    f"{layer_state.layer.name.lower()} layer has unusually "
                    f"high motion ({layer_state.avg_motion:.1f} vs avg {mean_mag:.1f})"
                )
            elif layer_state.avg_motion < mean_mag * 0.3 and layer_state.is_active:
                return (
                    layer_state.layer,
                    f"{layer_state.layer.name.lower()} layer has unusually "
                    f"low motion ({layer_state.avg_motion:.1f} vs avg {mean_mag:.1f})"
                )
        
        return None, ""
    
    def _compute_depth_coherence(self, active_layers: List[LayerState]) -> float:
        """
        Compute overall coherence across depth layers.
        
        High score when:
        - All layers moving same direction
        - Motion magnitude scales appropriately with depth
        
        Returns:
            Score from 0.0 (incoherent) to 1.0 (perfectly coherent)
        """
        if len(active_layers) == 0:
            return 0.0
        
        if len(active_layers) == 1:
            # Single layer - use its internal coherence
            return active_layers[0].motion_coherence
        
        # Factor 1: Direction coherence across layers
        sin_vals = [np.sin(l.dominant_direction) for l in active_layers]
        cos_vals = [np.cos(l.dominant_direction) for l in active_layers]
        
        r = np.sqrt(np.mean(sin_vals)**2 + np.mean(cos_vals)**2)
        direction_coherence = r
        
        # Factor 2: Average of per-layer coherence
        avg_layer_coherence = np.mean([l.motion_coherence for l in active_layers])
        
        # Combined score
        coherence = 0.6 * direction_coherence + 0.4 * avg_layer_coherence
        
        return float(np.clip(coherence, 0, 1))
    
    def get_layer_summary(self, result: LayerCoherenceResult) -> str:
        """Generate human-readable summary of layer analysis."""
        lines = []
        
        # Active layers
        active_names = [
            s.layer.name.lower() 
            for s in result.layer_states.values() 
            if s.is_active
        ]
        
        if not active_names:
            return "No significant motion in any depth layer"
        
        lines.append(f"Active layers: {', '.join(active_names)}")
        
        # Coherence status
        if result.layers_moving_together:
            lines.append("All layers moving together (scene-level motion)")
        else:
            lines.append("Layers moving independently (localized motion)")
        
        # Anomaly
        if result.depth_anomaly_detected:
            lines.append(f"⚠️ Anomaly: {result.anomaly_description}")
        
        # Depth coherence score
        lines.append(f"Depth coherence: {result.scene_depth_coherence:.0%}")
        
        return " | ".join(lines)
