"""
Demo visualizer for the Visual Coherence Oracle.
Shows real-time camera feed with coherence grid overlay.
"""

import sys
from pathlib import Path

# Add src to path BEFORE any local imports
_src_path = str(Path(__file__).parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import cv2
import numpy as np
import time
import argparse

# Now import local modules (these are top-level within src/)
import config
from camera.capture import FrameCapture
from motion.feature_detector import SpatialFeatureDetector
from motion.optical_flow import SparseFlowTracker
from motion.motion_field import MotionVectorField
from representation.spatial_coherence import SpatialCoherenceAnalyzer
from representation.disturbance_classifier import DisturbanceClassifier, DisturbanceType


class CoherenceVisualizer:
    """
    Real-time visualizer for the coherence oracle.
    
    Displays:
    - Camera feed with tracked features
    - 8x6 coherence grid overlay
    - Motion vectors
    - Real-time metrics panel
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        
        # Components
        self.capture = FrameCapture(camera_id)
        self.detector = SpatialFeatureDetector()
        self.tracker = SparseFlowTracker()
        self.motion_field = MotionVectorField()
        self.coherence = SpatialCoherenceAnalyzer()
        self.classifier = DisturbanceClassifier()
        
        # State
        self.features = None
        self.frame_count = 0
        self.redetect_interval = 15
        
        # Colors
        self.COLORS = {
            'feature': (0, 255, 0),      # Green
            'motion': (255, 255, 0),      # Cyan
            'grid': (128, 128, 128),      # Gray
            'active': (0, 200, 0),        # Dark green
            'coherent': (0, 255, 255),    # Yellow
            'text': (255, 255, 255),      # White
            'none': (100, 100, 100),
            'localized': (0, 165, 255),   # Orange
            'scene_level': (0, 255, 0),   # Green
        }
    
    def run(self):
        """Main visualization loop."""
        if not self.capture.open():
            print("ERROR: Could not open camera")
            return
        
        print("Visual Coherence Oracle - Demo Visualizer")
        print("=" * 50)
        print(f"Camera: {self.camera_id}")
        print(f"Resolution: {self.capture.actual_resolution}")
        print(f"Grid: {config.COHERENCE_GRID[0]}x{config.COHERENCE_GRID[1]}")
        print("-" * 50)
        print("Press 'q' to quit, 'r' to reset features")
        print("=" * 50)
        
        prev_frame = None
        fps_start = time.time()
        fps_count = 0
        current_fps = 0
        
        try:
            while True:
                # Capture frame
                color_frame = self.capture.read_frame_color()
                if color_frame is None:
                    continue
                
                gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                self.frame_count += 1
                fps_count += 1
                
                # Calculate FPS
                if time.time() - fps_start >= 1.0:
                    current_fps = fps_count
                    fps_count = 0
                    fps_start = time.time()
                
                # Detect/track features
                need_redetect = (
                    self.features is None or
                    len(self.features) < 50 or
                    self.frame_count % self.redetect_interval == 0
                )
                
                if need_redetect:
                    self.features = self.detector.detect(gray_frame)
                
                # Draw features
                vis_frame = color_frame.copy()
                if self.features is not None:
                    for pt in self.features:
                        cv2.circle(vis_frame, 
                                   (int(pt[0, 0]), int(pt[0, 1])), 
                                   3, self.COLORS['feature'], -1)
                
                # Track if we have prev frame
                sparse_cloud = None
                disturbance = None
                
                if prev_frame is not None and self.features is not None and len(self.features) > 0:
                    start_time = time.time()
                    
                    # Track
                    flow_result = self.tracker.track(prev_frame, gray_frame, self.features)
                    
                    # Update features
                    if np.any(flow_result.valid_mask):
                        self.features = flow_result.curr_points[flow_result.valid_mask]
                        
                        # Draw motion vectors
                        vis_frame = self._draw_motion_vectors(vis_frame, flow_result)
                        
                        # Analyze motion field
                        motion_state = self.motion_field.analyze(flow_result)
                        
                        # Generate sparse cloud
                        temporal = self.motion_field.get_temporal_consistency()
                        sparse_cloud = self.coherence.generate_sparse_cloud(
                            motion_state, temporal
                        )
                        
                        # Classify disturbance
                        disturbance = self.classifier.classify(sparse_cloud)
                        
                        # Draw coherence grid
                        vis_frame = self._draw_coherence_grid(vis_frame, sparse_cloud)
                    else:
                        self.features = None
                    
                    processing_time = (time.time() - start_time) * 1000
                else:
                    processing_time = 0
                
                # Draw metrics panel
                vis_frame = self._draw_metrics_panel(
                    vis_frame, sparse_cloud, disturbance, 
                    current_fps, processing_time
                )
                
                # Show
                cv2.imshow("Visual Coherence Oracle", vis_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.features = None
                    self.motion_field.reset()
                    self.coherence.reset()
                    print("Features reset")
                
                prev_frame = gray_frame
        
        finally:
            self.capture.close()
            cv2.destroyAllWindows()
    
    def _draw_motion_vectors(self, frame, flow_result):
        """Draw motion vectors for valid tracks."""
        valid_mask = flow_result.valid_mask
        if not np.any(valid_mask):
            return frame
        
        prev_pts = flow_result.prev_points[valid_mask].reshape(-1, 2)
        curr_pts = flow_result.curr_points[valid_mask].reshape(-1, 2)
        
        for (px, py), (cx, cy) in zip(prev_pts, curr_pts):
            cv2.arrowedLine(
                frame,
                (int(px), int(py)),
                (int(cx), int(cy)),
                self.COLORS['motion'],
                1,
                tipLength=0.3
            )
        
        return frame
    
    def _draw_coherence_grid(self, frame, sparse_cloud):
        """Draw the coherence grid overlay."""
        h, w = frame.shape[:2]
        rows, cols = sparse_cloud.grid_rows, sparse_cloud.grid_cols
        cell_h = h // rows
        cell_w = w // cols
        
        # Create overlay
        overlay = frame.copy()
        
        for row in range(rows):
            for col in range(cols):
                x1, y1 = col * cell_w, row * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                
                # Color based on cell state
                if sparse_cloud.coherent_cells[row, col]:
                    color = (*self.COLORS['coherent'], 60)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), 
                                  self.COLORS['coherent'][:3], -1)
                elif sparse_cloud.active_cells[row, col]:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                  self.COLORS['active'][:3], -1)
                
                # Draw grid lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                              self.COLORS['grid'], 1)
        
        # Blend overlay
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def _draw_metrics_panel(self, frame, sparse_cloud, disturbance, 
                            fps, processing_time):
        """Draw metrics panel in corner."""
        h, w = frame.shape[:2]
        
        # Panel background
        panel_h = 180
        panel_w = 300
        cv2.rectangle(frame, (w - panel_w - 10, 10), (w - 10, panel_h + 10),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (w - panel_w - 10, 10), (w - 10, panel_h + 10),
                      (128, 128, 128), 1)
        
        # Text metrics
        x = w - panel_w
        y = 35
        line_h = 25
        
        def draw_text(label, value, color=self.COLORS['text']):
            nonlocal y
            cv2.putText(frame, f"{label}: {value}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line_h
        
        draw_text("FPS", f"{fps}")
        draw_text("Processing", f"{processing_time:.1f}ms")
        
        if sparse_cloud is not None:
            draw_text("Coherence", f"{sparse_cloud.scene_coherence_score:.2f}")
            draw_text("Motion", f"{sparse_cloud.motion_intensity:.2f}")
            draw_text("Active Cells", f"{sparse_cloud.active_cell_ratio:.1%}")
        
        if disturbance is not None:
            dist_type = disturbance.disturbance_type.value.upper()
            color = self.COLORS.get(disturbance.disturbance_type.value,
                                    self.COLORS['text'])
            draw_text("Disturbance", dist_type, color)
        
        return frame


def main():
    parser = argparse.ArgumentParser(
        description="Visual Coherence Oracle - Demo Visualizer"
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    args = parser.parse_args()
    
    visualizer = CoherenceVisualizer(camera_id=args.camera)
    visualizer.run()


if __name__ == "__main__":
    main()
