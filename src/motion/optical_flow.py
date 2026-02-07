"""
Sparse Lucas-Kanade optical flow tracking.
Includes forward-backward error checking for robust tracking.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from config import (
    LK_WIN_SIZE, LK_MAX_LEVEL, LK_CRITERIA,
    LK_MIN_EIG_THRESHOLD, FB_ERROR_THRESHOLD
)


@dataclass
class FlowResult:
    """Result of optical flow computation."""
    prev_points: np.ndarray      # Points in previous frame
    curr_points: np.ndarray      # Tracked points in current frame
    motion_vectors: np.ndarray   # Motion vectors (curr - prev)
    valid_mask: np.ndarray       # Boolean mask of valid tracks
    tracking_quality: float      # Percentage of successfully tracked points


class SparseFlowTracker:
    """
    Sparse optical flow tracker using pyramidal Lucas-Kanade.
    
    Uses forward-backward error checking to validate tracks:
    1. Track points forward (prev -> curr)
    2. Track result points backward (curr -> prev)
    3. Reject if forward-backward distance > threshold
    """
    
    def __init__(self,
                 win_size: tuple = LK_WIN_SIZE,
                 max_level: int = LK_MAX_LEVEL,
                 fb_threshold: float = FB_ERROR_THRESHOLD):
        self.win_size = win_size
        self.max_level = max_level
        self.fb_threshold = fb_threshold
        self.criteria = LK_CRITERIA
        
        # Track history for temporal analysis
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
    
    def track(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
              prev_points: np.ndarray) -> FlowResult:
        """
        Track points from prev_frame to curr_frame.
        
        Args:
            prev_frame: Previous grayscale frame
            curr_frame: Current grayscale frame
            prev_points: Points to track, shape (N, 1, 2)
            
        Returns:
            FlowResult with tracked points and motion vectors
        """
        if prev_points is None or len(prev_points) == 0:
            return FlowResult(
                prev_points=np.array([]).reshape(-1, 1, 2),
                curr_points=np.array([]).reshape(-1, 1, 2),
                motion_vectors=np.array([]).reshape(-1, 2),
                valid_mask=np.array([], dtype=bool),
                tracking_quality=0.0
            )
        
        # Ensure correct shape
        prev_points = prev_points.reshape(-1, 1, 2).astype(np.float32)
        
        # Forward tracking: prev -> curr
        curr_points, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_points, None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
            minEigThreshold=LK_MIN_EIG_THRESHOLD
        )
        
        # Backward tracking: curr -> prev (for validation)
        back_points, status_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(
            curr_frame, prev_frame, curr_points, None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
            minEigThreshold=LK_MIN_EIG_THRESHOLD
        )
        
        # Compute forward-backward error
        fb_error = np.linalg.norm(
            prev_points.reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1
        )
        
        # Valid points: forward success AND backward success AND low FB error
        valid_mask = (
            (status_fwd.ravel() == 1) &
            (status_bwd.ravel() == 1) &
            (fb_error < self.fb_threshold)
        )
        
        # Compute motion vectors
        motion_vectors = (curr_points - prev_points).reshape(-1, 2)
        
        # Calculate tracking quality
        tracking_quality = np.sum(valid_mask) / len(valid_mask) if len(valid_mask) > 0 else 0.0
        
        return FlowResult(
            prev_points=prev_points,
            curr_points=curr_points,
            motion_vectors=motion_vectors,
            valid_mask=valid_mask,
            tracking_quality=tracking_quality
        )
    
    def track_continuous(self, frame: np.ndarray, 
                         points: Optional[np.ndarray] = None) -> Optional[FlowResult]:
        """
        Track points continuously across frames.
        
        Args:
            frame: Current grayscale frame
            points: Optional new points to track (resets tracking)
            
        Returns:
            FlowResult if previous frame exists, None on first frame
        """
        if points is not None:
            # Reset with new points
            self._prev_points = points.reshape(-1, 1, 2).astype(np.float32)
        
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return None
        
        if self._prev_points is None or len(self._prev_points) == 0:
            self._prev_frame = frame.copy()
            return None
        
        # Track points
        result = self.track(self._prev_frame, frame, self._prev_points)
        
        # Update state with valid points only
        if np.any(result.valid_mask):
            self._prev_points = result.curr_points[result.valid_mask].reshape(-1, 1, 2)
        else:
            self._prev_points = None
        
        self._prev_frame = frame.copy()
        
        return result
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._prev_frame = None
        self._prev_points = None
    
    @staticmethod
    def compute_motion_stats(motion_vectors: np.ndarray, 
                            valid_mask: np.ndarray) -> dict:
        """
        Compute statistics on motion vectors.
        
        Returns:
            Dict with magnitude_mean, magnitude_std, direction_mean, direction_std
        """
        valid_vectors = motion_vectors[valid_mask]
        
        if len(valid_vectors) == 0:
            return {
                'magnitude_mean': 0.0,
                'magnitude_std': 0.0,
                'direction_mean': 0.0,
                'direction_std': 0.0,
                'count': 0
            }
        
        # Compute magnitudes
        magnitudes = np.linalg.norm(valid_vectors, axis=1)
        
        # Compute directions (radians, -pi to pi)
        directions = np.arctan2(valid_vectors[:, 1], valid_vectors[:, 0])
        
        # Circular mean for directions
        sin_sum = np.sum(np.sin(directions))
        cos_sum = np.sum(np.cos(directions))
        direction_mean = np.arctan2(sin_sum, cos_sum)
        
        # Circular standard deviation
        r = np.sqrt(sin_sum**2 + cos_sum**2) / len(directions)
        direction_std = np.sqrt(-2 * np.log(r)) if r > 0 else np.pi
        
        return {
            'magnitude_mean': float(np.mean(magnitudes)),
            'magnitude_std': float(np.std(magnitudes)),
            'direction_mean': float(direction_mean),
            'direction_std': float(direction_std),
            'count': len(valid_vectors)
        }
