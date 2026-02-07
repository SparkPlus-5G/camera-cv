"""
Trust signal generation for sensor fusion.
Converts coherence signals into actionable trust adjustments.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from agent.coherence_oracle import CoherenceSignal
from representation.disturbance_classifier import DisturbanceType


@dataclass
class TrustAdjustment:
    """Trust adjustment recommendation for a sensor."""
    sensor_name: str
    original_trust: float
    adjusted_trust: float
    multiplier: float
    reason: str


@dataclass
class SuppressionDecision:
    """Decision on whether to suppress an alert."""
    should_suppress: bool
    confidence: float
    reason: str


class TrustSignalGenerator:
    """
    Converts coherence signals into actionable trust adjustments.
    
    Use cases:
    1. Reweight sensor trust based on visual coherence
    2. Suppress false positives when visual doesn't support sensor data
    3. Boost confidence when visual confirms sensor alerts
    
    Logic:
    - Scene coherence HIGH + sensor alert → Boost sensor trust
    - Scene coherence LOW + sensor alert → Suppress (likely false positive)
    - Scene coherence HIGH + no sensor alert → Possible visual-only event
    """
    
    def __init__(self,
                 base_visual_weight: float = 0.3,
                 max_boost: float = 1.5,
                 max_suppression: float = 0.3):
        """
        Args:
            base_visual_weight: Base weight of visual system in fusion (0-1)
            max_boost: Maximum trust multiplier when visual confirms
            max_suppression: Minimum trust multiplier when visual contradicts
        """
        self.base_visual_weight = base_visual_weight
        self.max_boost = max_boost
        self.max_suppression = max_suppression
    
    def compute_sensor_reweight(
        self,
        coherence: CoherenceSignal,
        sensor_alerts: Dict[str, float]
    ) -> Dict[str, TrustAdjustment]:
        """
        Compute per-sensor trust adjustments based on visual coherence.
        
        Args:
            coherence: CoherenceSignal from oracle
            sensor_alerts: Dict mapping sensor name to alert level (0-1)
            
        Returns:
            Dict mapping sensor name to TrustAdjustment
        """
        adjustments = {}
        
        for sensor_name, alert_level in sensor_alerts.items():
            adjustment = self._compute_single_adjustment(
                coherence, sensor_name, alert_level
            )
            adjustments[sensor_name] = adjustment
        
        return adjustments
    
    def _compute_single_adjustment(
        self,
        coherence: CoherenceSignal,
        sensor_name: str,
        alert_level: float
    ) -> TrustAdjustment:
        """Compute trust adjustment for a single sensor."""
        
        scene_coh = coherence.scene_coherence
        visual_conf = coherence.visual_confidence
        disturbance = coherence.disturbance_type
        
        # Default: no adjustment
        multiplier = 1.0
        reason = "No adjustment needed"
        
        # Case 1: High alert + Scene-level visual motion
        if alert_level > 0.5 and disturbance == DisturbanceType.SCENE_LEVEL:
            # Visual confirms sensor - boost trust
            boost = 1.0 + (self.max_boost - 1.0) * scene_coh * visual_conf
            multiplier = min(boost, self.max_boost)
            reason = f"Visual confirms with {scene_coh:.0%} coherence"
        
        # Case 2: High alert + No/Localized visual motion
        elif alert_level > 0.5 and disturbance != DisturbanceType.SCENE_LEVEL:
            # Visual contradicts sensor - suppress
            if visual_conf > 0.5:
                # High confidence in visual = strong suppression
                suppression = self.max_suppression + (1.0 - self.max_suppression) * (1 - visual_conf)
                multiplier = suppression
                reason = f"Visual contradicts: {disturbance.value} motion only"
            else:
                # Low visual confidence = weak suppression
                multiplier = 0.7
                reason = "Low visual confidence, mild reduction"
        
        # Case 3: Low alert + Scene-level visual motion
        elif alert_level < 0.3 and disturbance == DisturbanceType.SCENE_LEVEL:
            # Visual sees something sensors don't - investigate
            multiplier = 1.0  # Don't change sensor trust
            reason = "Visual-only event detected, sensors may be lagging"
        
        # Case 4: Both quiet
        elif alert_level < 0.3 and disturbance == DisturbanceType.NONE:
            multiplier = 1.0
            reason = "Both visual and sensor quiet"
        
        adjusted = alert_level * multiplier
        
        return TrustAdjustment(
            sensor_name=sensor_name,
            original_trust=alert_level,
            adjusted_trust=adjusted,
            multiplier=multiplier,
            reason=reason
        )
    
    def should_suppress_alert(
        self,
        coherence: CoherenceSignal,
        alert_severity: float,
        sensor_name: str = "unknown"
    ) -> SuppressionDecision:
        """
        Decide whether to suppress a sensor alert.
        
        Args:
            coherence: CoherenceSignal from oracle
            alert_severity: Severity of the alert (0-1)
            sensor_name: Name of alerting sensor
            
        Returns:
            SuppressionDecision with recommendation and reasoning
        """
        false_pos_risk = coherence.false_positive_risk
        visual_conf = coherence.visual_confidence
        disturbance = coherence.disturbance_type
        
        # Don't suppress if visual confidence is low
        if visual_conf < 0.4:
            return SuppressionDecision(
                should_suppress=False,
                confidence=visual_conf,
                reason="Visual confidence too low for suppression decision"
            )
        
        # Don't suppress scene-level alerts with scene-level motion
        if disturbance == DisturbanceType.SCENE_LEVEL:
            return SuppressionDecision(
                should_suppress=False,
                confidence=coherence.scene_coherence,
                reason="Scene-level visual motion supports alert"
            )
        
        # Suppress if high false positive risk and no scene-level motion
        if false_pos_risk > 0.6 and disturbance != DisturbanceType.SCENE_LEVEL:
            return SuppressionDecision(
                should_suppress=True,
                confidence=false_pos_risk * visual_conf,
                reason=f"High false positive risk ({false_pos_risk:.0%}), "
                       f"only {disturbance.value} motion detected"
            )
        
        # Suppress mild alerts with no visual motion
        if alert_severity < 0.4 and disturbance == DisturbanceType.NONE:
            return SuppressionDecision(
                should_suppress=True,
                confidence=visual_conf,
                reason="Mild alert with no visual motion"
            )
        
        # Default: don't suppress
        return SuppressionDecision(
            should_suppress=False,
            confidence=0.5,
            reason="Insufficient evidence for suppression"
        )
    
    def compute_fused_confidence(
        self,
        coherence: CoherenceSignal,
        sensor_confidences: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute fused confidence score across all sensors + visual.
        
        Args:
            coherence: CoherenceSignal from oracle
            sensor_confidences: Per-sensor confidence scores
            
        Returns:
            Tuple of (fused_confidence, per_sensor_weights)
        """
        if not sensor_confidences:
            return coherence.scene_coherence, {}
        
        # Compute visual weight based on confidence
        visual_weight = self.base_visual_weight * coherence.visual_confidence
        remaining_weight = 1.0 - visual_weight
        
        # Distribute remaining weight among sensors
        n_sensors = len(sensor_confidences)
        sensor_weight = remaining_weight / n_sensors if n_sensors > 0 else 0
        
        # Compute weighted sum
        fused = visual_weight * coherence.scene_coherence
        weights = {'visual': visual_weight}
        
        for sensor_name, conf in sensor_confidences.items():
            # Adjust sensor contribution by coherence agreement
            adjustment = self._compute_single_adjustment(
                coherence, sensor_name, conf
            )
            adjusted_conf = conf * adjustment.multiplier
            fused += sensor_weight * adjusted_conf
            weights[sensor_name] = sensor_weight
        
        return float(np.clip(fused, 0, 1)), weights
    
    def get_action_recommendation(
        self,
        coherence: CoherenceSignal,
        sensor_alert: float
    ) -> str:
        """
        Get human-readable action recommendation.
        
        Returns:
            Action recommendation string
        """
        disturbance = coherence.disturbance_type
        scene_coh = coherence.scene_coherence
        
        if sensor_alert > 0.7:
            if disturbance == DisturbanceType.SCENE_LEVEL and scene_coh > 0.6:
                return "HIGH ALERT: Visual confirms scene-level disturbance. Recommend action."
            elif disturbance == DisturbanceType.LOCALIZED:
                return "CAUTION: Sensor alert but only localized motion. Possible false positive."
            elif disturbance == DisturbanceType.NONE:
                return "INVESTIGATE: Sensor alert with no visual confirmation. Check sensor health."
        
        elif sensor_alert > 0.4:
            if disturbance == DisturbanceType.SCENE_LEVEL:
                return "MONITOR: Moderate sensor activity with visual scene motion. Watch closely."
            else:
                return "LOW PRIORITY: Moderate sensor activity, no supporting visual motion."
        
        else:
            if disturbance == DisturbanceType.SCENE_LEVEL:
                return "ANOMALY: Visual detects scene motion but sensors quiet. Investigate."
            else:
                return "NORMAL: Both sensors and visual quiet."
