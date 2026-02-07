"""
Spatially-distributed FAST feature detection.
Ensures features are distributed evenly across the frame, not clustered.
"""

import cv2
import numpy as np
from typing import Optional

from config import (
    MAX_FEATURES, FAST_THRESHOLD, FAST_NON_MAX_SUPPRESSION,
    FEATURE_GRID_SIZE, FEATURES_PER_CELL
)


class SpatialFeatureDetector:
    """
    Detects FAST features with spatial binning to ensure
    even distribution across the frame.
    
    This prevents feature clustering in high-texture regions
    and ensures motion analysis covers the entire scene.
    """
    
    def __init__(self, 
                 max_features: int = MAX_FEATURES,
                 grid_size: tuple = FEATURE_GRID_SIZE,
                 fast_threshold: int = FAST_THRESHOLD):
        self.max_features = max_features
        self.grid_size = grid_size  # (columns, rows)
        self.fast_threshold = fast_threshold
        self.features_per_cell = max_features // (grid_size[0] * grid_size[1])
        
        # Create FAST detector
        self._detector = cv2.FastFeatureDetector_create(
            threshold=fast_threshold,
            nonmaxSuppression=FAST_NON_MAX_SUPPRESSION
        )
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect features with spatial distribution.
        
        Args:
            frame: Grayscale input frame
            
        Returns:
            Array of keypoint coordinates, shape (N, 1, 2) for optical flow
        """
        if frame is None or frame.size == 0:
            return np.array([], dtype=np.float32).reshape(-1, 1, 2)
        
        h, w = frame.shape[:2]
        cell_h = h // self.grid_size[1]
        cell_w = w // self.grid_size[0]
        
        all_points = []
        
        # Process each cell in the grid
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                # Extract cell region
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < self.grid_size[1] - 1 else h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < self.grid_size[0] - 1 else w
                
                cell = frame[y1:y2, x1:x2]
                
                # Detect features in cell
                keypoints = self._detector.detect(cell)
                
                if len(keypoints) > 0:
                    # Sort by response (stronger features first)
                    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)
                    
                    # Take top N per cell
                    keypoints = keypoints[:self.features_per_cell]
                    
                    # Convert to global coordinates
                    for kp in keypoints:
                        pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                        all_points.append(pt)
        
        if len(all_points) == 0:
            return np.array([], dtype=np.float32).reshape(-1, 1, 2)
        
        # Convert to numpy array in optical flow format
        points = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)
        
        return points
    
    def detect_cells(self, frame: np.ndarray) -> dict:
        """
        Detect features and return per-cell point arrays.
        Useful for spatial coherence analysis.
        
        Returns:
            Dict mapping (col, row) -> array of points
        """
        if frame is None or frame.size == 0:
            return {}
        
        h, w = frame.shape[:2]
        cell_h = h // self.grid_size[1]
        cell_w = w // self.grid_size[0]
        
        cell_points = {}
        
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < self.grid_size[1] - 1 else h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < self.grid_size[0] - 1 else w
                
                cell = frame[y1:y2, x1:x2]
                keypoints = self._detector.detect(cell)
                
                points = []
                if len(keypoints) > 0:
                    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)
                    keypoints = keypoints[:self.features_per_cell]
                    
                    for kp in keypoints:
                        pt = (kp.pt[0] + x1, kp.pt[1] + y1)
                        points.append(pt)
                
                cell_points[(col, row)] = np.array(points, dtype=np.float32)
        
        return cell_points
    
    def get_cell_for_point(self, point: tuple, frame_shape: tuple) -> tuple:
        """Get the grid cell (col, row) for a given point."""
        h, w = frame_shape[:2]
        cell_h = h // self.grid_size[1]
        cell_w = w // self.grid_size[0]
        
        col = min(int(point[0] // cell_w), self.grid_size[0] - 1)
        row = min(int(point[1] // cell_h), self.grid_size[1] - 1)
        
        return (col, row)
