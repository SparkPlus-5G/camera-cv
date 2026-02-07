"""
Configuration constants for Visual Coherence Oracle.
Tuned for Raspberry Pi 5B performance constraints.
"""

import cv2

# =============================================================================
# Camera Settings
# =============================================================================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
PROCESSING_FPS = 15  # Process every 2nd frame for performance
FRAME_BUFFER_SIZE = 10  # Ring buffer size for temporal analysis

# =============================================================================
# Feature Detection (FAST)
# =============================================================================
MAX_FEATURES = 500  # Total keypoints across frame
FAST_THRESHOLD = 20  # FAST corner detection threshold
FAST_NON_MAX_SUPPRESSION = True
FEATURE_GRID_SIZE = (8, 6)  # Spatial distribution grid (columns, rows)
FEATURES_PER_CELL = MAX_FEATURES // (FEATURE_GRID_SIZE[0] * FEATURE_GRID_SIZE[1])

# =============================================================================
# Optical Flow (Lucas-Kanade)
# =============================================================================
LK_WIN_SIZE = (21, 21)  # Search window size
LK_MAX_LEVEL = 3  # Pyramid levels
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
LK_MIN_EIG_THRESHOLD = 0.001  # Minimum eigenvalue for point quality
FB_ERROR_THRESHOLD = 1.0  # Forward-backward error threshold for validation

# =============================================================================
# Motion Analysis
# =============================================================================
COHERENCE_GRID = (8, 6)  # Grid cells for coherence analysis (columns, rows)
MOTION_MAGNITUDE_THRESHOLD = 2.0  # Minimum pixels to consider as "motion"
MOTION_MAGNITUDE_MAX = 50.0  # Max expected motion for normalization
COHERENCE_HISTORY_FRAMES = 10  # Temporal window for consistency analysis

# =============================================================================
# Disturbance Classification
# =============================================================================
LOCALIZED_THRESHOLD = 0.20  # <20% of cells with motion = LOCALIZED
SCENE_LEVEL_THRESHOLD = 0.50  # >50% of cells with correlated motion = SCENE_LEVEL
DIRECTION_VARIANCE_THRESHOLD = 0.5  # Radians - low variance = coherent direction

# =============================================================================
# Trust Signal Parameters
# =============================================================================
MIN_FEATURE_COUNT = 50  # Minimum tracked features for valid assessment
LOW_CONFIDENCE_MOTION = 5.0  # Motion below this is uncertain
HIGH_CONFIDENCE_THRESHOLD = 0.7  # Coherence above this = high confidence
