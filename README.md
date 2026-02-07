# Visual Coherence Oracle

A lightweight motion-induced sparse representation system for epistemic validation 
of multi-sensor environmental anomalies on Raspberry Pi 5B.

## Overview

This system generates a sparse, motion-based representation to validate whether 
multi-sensor anomalies (IMU, acoustic, atmospheric) correspond to real environment-level 
structural responses. The camera is NOT a detector—it's a validator.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the visualizer demo
python demo/visualizer.py

# Use in your code
from src.agent.coherence_oracle import CoherenceOracle

oracle = CoherenceOracle(camera_id=0)
signal = oracle.get_coherence_signal()
print(f"Scene coherence: {signal.scene_coherence:.2f}")
print(f"Disturbance type: {signal.disturbance_type}")
```

## Architecture

```
Camera Frame (720p)
    ↓
FAST Feature Detection (500 keypoints)
    ↓
Lucas-Kanade Optical Flow
    ↓
Motion Vector Field (8x6 grid)
    ↓
Spatial Coherence Analysis
    ↓
CoherenceSignal → Agentic AI Layer
```

## Key Outputs

| Metric | Range | Meaning |
|--------|-------|---------|
| `scene_coherence` | 0.0-1.0 | Correlation of motion across spatial cells |
| `motion_intensity` | 0.0-1.0 | Normalized motion magnitude |
| `disturbance_type` | NONE/LOCALIZED/SCENE_LEVEL | Classification of motion pattern |
| `visual_confidence` | 0.0-1.0 | Trust in the visual assessment |
| `false_positive_risk` | 0.0-1.0 | Likelihood other sensors have false positive |

## Performance

- Target: <20ms per frame
- Processing rate: 15 FPS
- Tested on: Raspberry Pi 5B

## Project Structure

```
Camera/
├── src/
│   ├── config.py           # Configuration constants
│   ├── camera/             # Frame capture
│   ├── motion/             # Feature detection & optical flow
│   ├── representation/     # Sparse coherence grid
│   └── agent/              # Oracle interface
├── tests/                  # Unit tests
├── demo/                   # Visualization tools
└── requirements.txt
```
