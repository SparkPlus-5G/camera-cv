"""Agent interface module."""

from .coherence_oracle import CoherenceOracle, CoherenceSignal
from .trust_signals import TrustSignalGenerator

__all__ = [
    "CoherenceOracle",
    "CoherenceSignal",
    "TrustSignalGenerator"
]
