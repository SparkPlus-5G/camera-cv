"""
Frame capture and buffering for camera input.
Handles USB camera connection with graceful error handling.
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple
import threading
import time

from config import (
    FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    FRAME_BUFFER_SIZE
)


class FrameBuffer:
    """
    Thread-safe ring buffer for storing recent frames.
    Used for temporal analysis and motion history.
    """
    
    def __init__(self, max_size: int = FRAME_BUFFER_SIZE):
        self._buffer: deque = deque(maxlen=max_size)
        self._timestamps: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def push(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add a frame to the buffer."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self._buffer.append(frame.copy())
            self._timestamps.append(timestamp)
    
    def get_latest(self, n: int = 1) -> list:
        """Get the N most recent frames."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]
    
    def get_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the two most recent frames for optical flow computation."""
        with self._lock:
            if len(self._buffer) < 2:
                return None
            return self._buffer[-2], self._buffer[-1]
    
    def get_timestamps(self, n: int = 1) -> list:
        """Get the N most recent timestamps."""
        with self._lock:
            if n >= len(self._timestamps):
                return list(self._timestamps)
            return list(self._timestamps)[-n:]
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._timestamps.clear()


class FrameCapture:
    """
    Camera frame capture with automatic configuration.
    Handles USB camera connection and provides grayscale frames.
    """
    
    def __init__(self, camera_id: int = 0, width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT, fps: int = TARGET_FPS):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer = FrameBuffer()
        self._is_running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_color: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
    
    def open(self) -> bool:
        """
        Open the camera connection.
        
        Tries V4L2 backend first (better for Raspberry Pi),
        then falls back to default backend.
        """
        # Try V4L2 backend first (works better on Linux/RPi)
        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        
        if not self._cap.isOpened():
            # Fallback to default backend (GStreamer, etc.)
            self._cap = cv2.VideoCapture(self.camera_id)
        
        if not self._cap.isOpened():
            # Last resort: try different camera indices
            for alt_id in [0, 1, 2]:
                if alt_id != self.camera_id:
                    self._cap = cv2.VideoCapture(alt_id, cv2.CAP_V4L2)
                    if self._cap.isOpened():
                        print(f"Camera found at index {alt_id}")
                        break
        
        if not self._cap.isOpened():
            print("ERROR: Could not open any camera")
            return False
        
        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set MJPEG format for better USB camera performance
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Disable auto-focus if available (reduces latency)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # Read a test frame to confirm camera is working
        ret, _ = self._cap.read()
        if not ret:
            print("WARNING: Camera opened but could not read frame")
            return False
        
        return True
    
    def close(self) -> None:
        """Close the camera connection."""
        self.stop_continuous()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from camera (grayscale).
        Returns None if read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        with self._frame_lock:
            self._last_frame = gray
            self._last_frame_color = frame
        
        # Add to buffer
        self._buffer.push(gray)
        
        return gray
    
    def read_frame_color(self) -> Optional[np.ndarray]:
        """Read a single frame in color (for visualization)."""
        if self._cap is None or not self._cap.isOpened():
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        with self._frame_lock:
            self._last_frame = gray
            self._last_frame_color = frame
        
        self._buffer.push(gray)
        
        return frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently captured frame (grayscale)."""
        with self._frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None
    
    def get_latest_frame_color(self) -> Optional[np.ndarray]:
        """Get the most recently captured frame (color)."""
        with self._frame_lock:
            return self._last_frame_color.copy() if self._last_frame_color is not None else None
    
    def get_frame_pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get two consecutive frames for optical flow."""
        return self._buffer.get_pair()
    
    def get_buffer(self) -> FrameBuffer:
        """Get the frame buffer for temporal analysis."""
        return self._buffer
    
    def start_continuous(self) -> None:
        """Start continuous frame capture in background thread."""
        if self._is_running:
            return
        
        self._is_running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
    
    def stop_continuous(self) -> None:
        """Stop continuous frame capture."""
        self._is_running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
    
    def _capture_loop(self) -> None:
        """Background capture loop."""
        frame_interval = 1.0 / self.fps
        
        while self._is_running:
            start_time = time.time()
            self.read_frame()
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    @property
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Get actual camera resolution (may differ from requested)."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
