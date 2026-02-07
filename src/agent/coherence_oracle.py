"""
Coherence Oracle - Main orchestrator and agent interface.
Primary entry point for the agentic AI layer.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Any
import threading

from config import PROCESSING_FPS, MIN_FEATURE_COUNT
from camera.capture import FrameCapture
from motion.feature_detector import SpatialFeatureDetector
from motion.optical_flow import SparseFlowTracker
from motion.motion_field import MotionVectorField
from representation.spatial_coherence import SpatialCoherenceAnalyzer, SparseMotionCloud
from representation.disturbance_classifier import DisturbanceClassifier, DisturbanceType


@dataclass
class CoherenceSignal:
    """
    Output signal for the agentic AI layer.
    
    This is the primary interface between the camera vision system
    and the multi-sensor fusion layer.
    """
    timestamp: float
    
    # Core metrics
    scene_coherence: float          # 0.0-1.0: spatial correlation of motion
    disturbance_type: DisturbanceType
    motion_intensity: float         # 0.0-1.0: normalized motion magnitude
    
    # Trust signals for agent decision-making
    visual_confidence: float        # 0.0-1.0: confidence in visual assessment
    false_positive_risk: float      # 0.0-1.0: likelihood sensors have false positive
    
    # Detailed data (optional, for deep analysis)
    sparse_cloud: Optional[SparseMotionCloud] = None
    
    # Diagnostic info
    tracked_features: int = 0
    processing_time_ms: float = 0.0


class CoherenceOracle:
    """
    Main entry point for the Visual Coherence Oracle system.
    
    Orchestrates the full pipeline:
    Camera -> Feature Detection -> Optical Flow -> Motion Field -> 
    Sparse Cloud -> Coherence Signal
    
    Usage:
        oracle = CoherenceOracle(camera_id=0)
        oracle.start()
        
        # Option A: Polling
        signal = oracle.get_coherence_signal()
        
        # Option B: Callback
        def on_signal(signal: CoherenceSignal):
            print(f"Coherence: {signal.scene_coherence:.2f}")
        oracle.set_callback(on_signal)
        
        oracle.stop()
    """
    
    def __init__(self, camera_id: int = 0, include_sparse_cloud: bool = False):
        """
        Initialize the Coherence Oracle.
        
        Args:
            camera_id: Camera device index
            include_sparse_cloud: If True, include full SparseMotionCloud in signals
        """
        self.camera_id = camera_id
        self.include_sparse_cloud = include_sparse_cloud
        
        # Components
        self._capture = FrameCapture(camera_id)
        self._detector = SpatialFeatureDetector()
        self._tracker = SparseFlowTracker()
        self._motion_field = MotionVectorField()
        self._coherence_analyzer = SpatialCoherenceAnalyzer()
        self._classifier = DisturbanceClassifier()
        
        # State
        self._is_running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._latest_signal: Optional[CoherenceSignal] = None
        self._signal_lock = threading.Lock()
        self._callback: Optional[Callable[[CoherenceSignal], Any]] = None
        
        # Feature redetection
        self._redetect_interval = 10  # Frames between feature redetection
        self._frame_count = 0
        self._current_features: Optional[np.ndarray] = None
    
    def start(self) -> bool:
        """
        Start the oracle (opens camera and begins processing).
        
        Returns:
            True if started successfully
        """
        if not self._capture.open():
            return False
        
        self._is_running = True
        self._capture.start_continuous()
        
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the oracle and release resources."""
        self._is_running = False
        
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None
        
        self._capture.close()
    
    def get_coherence_signal(self) -> Optional[CoherenceSignal]:
        """
        Get the most recent coherence signal.
        
        Returns:
            Latest CoherenceSignal, or None if not available
        """
        with self._signal_lock:
            return self._latest_signal
    
    def set_callback(self, callback: Callable[[CoherenceSignal], Any]) -> None:
        """
        Set callback for new coherence signals.
        
        Args:
            callback: Function called with each new CoherenceSignal
        """
        self._callback = callback
    
    def process_single_frame(self) -> Optional[CoherenceSignal]:
        """
        Process a single frame and return coherence signal.
        Use this for synchronous/polling mode without continuous capture.
        
        Returns:
            CoherenceSignal for current frame
        """
        frame_pair = self._capture.get_frame_pair()
        
        if frame_pair is None:
            # Need at least one more frame
            self._capture.read_frame()
            frame_pair = self._capture.get_frame_pair()
            
            if frame_pair is None:
                return None
        
        return self._process_frame_pair(frame_pair[0], frame_pair[1])
    
    def _processing_loop(self) -> None:
        """Background processing loop."""
        target_interval = 1.0 / PROCESSING_FPS
        
        while self._is_running:
            start_time = time.time()
            
            frame_pair = self._capture.get_frame_pair()
            
            if frame_pair is not None:
                signal = self._process_frame_pair(frame_pair[0], frame_pair[1])
                
                if signal is not None:
                    with self._signal_lock:
                        self._latest_signal = signal
                    
                    if self._callback is not None:
                        try:
                            self._callback(signal)
                        except Exception:
                            pass  # Don't crash on callback errors
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _process_frame_pair(self, prev_frame: np.ndarray, 
                            curr_frame: np.ndarray) -> Optional[CoherenceSignal]:
        """Process a frame pair and generate coherence signal."""
        start_time = time.time()
        self._frame_count += 1
        
        # Redetect features periodically or if we have too few
        need_redetect = (
            self._current_features is None or
            len(self._current_features) < MIN_FEATURE_COUNT or
            self._frame_count % self._redetect_interval == 0
        )
        
        if need_redetect:
            self._current_features = self._detector.detect(prev_frame)
        
        if self._current_features is None or len(self._current_features) == 0:
            return self._create_empty_signal(start_time)
        
        # Track features
        flow_result = self._tracker.track(
            prev_frame, curr_frame, self._current_features
        )
        
        # Update features with successfully tracked points
        if np.any(flow_result.valid_mask):
            self._current_features = flow_result.curr_points[flow_result.valid_mask]
        else:
            self._current_features = None
        
        # Analyze motion field
        motion_state = self._motion_field.analyze(flow_result)
        
        # Get temporal consistency
        temporal_consistency = self._motion_field.get_temporal_consistency()
        
        # Generate sparse cloud
        sparse_cloud = self._coherence_analyzer.generate_sparse_cloud(
            motion_state, temporal_consistency
        )
        
        # Classify disturbance
        disturbance = self._classifier.classify(sparse_cloud)
        
        # Compute trust signals
        visual_confidence = self._compute_visual_confidence(
            flow_result, sparse_cloud, motion_state
        )
        false_positive_risk = self._compute_false_positive_risk(
            sparse_cloud, disturbance
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return CoherenceSignal(
            timestamp=time.time(),
            scene_coherence=sparse_cloud.scene_coherence_score,
            disturbance_type=disturbance.disturbance_type,
            motion_intensity=sparse_cloud.motion_intensity,
            visual_confidence=visual_confidence,
            false_positive_risk=false_positive_risk,
            sparse_cloud=sparse_cloud if self.include_sparse_cloud else None,
            tracked_features=motion_state.total_tracked_points,
            processing_time_ms=processing_time
        )
    
    def _compute_visual_confidence(self, flow_result, sparse_cloud, motion_state) -> float:
        """
        Compute confidence in the visual assessment.
        
        High confidence when:
        - Many features tracked
        - Good tracking quality
        - Clear motion pattern (either high or low)
        """
        # Feature count factor
        feature_factor = min(motion_state.total_tracked_points / 200, 1.0)
        
        # Tracking quality factor
        tracking_factor = flow_result.tracking_quality
        
        # Motion clarity factor (high motion or no motion = clear)
        intensity = sparse_cloud.motion_intensity
        motion_clarity = max(intensity, 1 - intensity)  # Clear at extremes
        
        confidence = 0.4 * feature_factor + 0.4 * tracking_factor + 0.2 * motion_clarity
        
        return float(np.clip(confidence, 0, 1))
    
    def _compute_false_positive_risk(self, sparse_cloud, disturbance) -> float:
        """
        Compute likelihood that other sensors have a false positive.
        
        High risk when:
        - Low scene coherence but sensors are alerting
        - Localized motion only
        - Motion at scene edges (likely camera artifact)
        """
        if disturbance.disturbance_type == DisturbanceType.NONE:
            # No visual motion - if sensors are alerting, possible false positive
            return 0.7
        
        if disturbance.disturbance_type == DisturbanceType.LOCALIZED:
            # Localized motion doesn't support scene-level sensor alerts
            return 0.6
        
        # Scene-level motion supports sensor alerts
        # Lower false positive risk with higher coherence
        risk = 1.0 - sparse_cloud.scene_coherence_score
        risk *= 0.5  # Base reduction for scene-level motion
        
        return float(np.clip(risk, 0, 1))
    
    def _create_empty_signal(self, start_time: float) -> CoherenceSignal:
        """Create signal when no features are available."""
        return CoherenceSignal(
            timestamp=time.time(),
            scene_coherence=0.0,
            disturbance_type=DisturbanceType.NONE,
            motion_intensity=0.0,
            visual_confidence=0.1,  # Low confidence
            false_positive_risk=0.5,  # Unknown
            tracked_features=0,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def reset(self) -> None:
        """Reset all internal state."""
        self._tracker.reset()
        self._motion_field.reset()
        self._coherence_analyzer.reset()
        self._current_features = None
        self._frame_count = 0
    
    @property
    def is_running(self) -> bool:
        """Check if oracle is running."""
        return self._is_running
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
