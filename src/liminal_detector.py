"""
Module: liminal_detector.py
Purpose: Detect liminal (transitional) market states
Part of LIMINAL ProtoConsciousness — Adaptive Finance Framework
"""
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class LiminalSignal:
    """Represents a signal indicating liminal state"""
    signal_type: str  # "volatility_spike", "sentiment_flip", "volume_anomaly", etc.
    strength: float  # 0.0 to 1.0
    description: str
    timestamp: str


@dataclass
class LiminalState:
    """Represents the current liminal state of the market"""
    state: str  # "stable", "liminal", "critical"
    liminal_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signals: List[LiminalSignal]
    regime: str  # "bull", "bear", "sideways", "transition"
    timestamp: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "liminal_score": round(self.liminal_score, 4),
            "confidence": round(self.confidence, 4),
            "signals": [
                {
                    "type": s.signal_type,
                    "strength": round(s.strength, 4),
                    "description": s.description
                }
                for s in self.signals
            ],
            "regime": self.regime,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class LiminalDetector:
    """Detects liminal (transitional) states in market dynamics"""

    def __init__(
        self,
        volatility_window: int = 20,
        sentiment_window: int = 10,
        volume_window: int = 24,
        liminal_threshold: float = 0.6,
        critical_threshold: float = 0.8
    ):
        """
        Initialize liminal detector

        Args:
            volatility_window: Window size for volatility calculation
            sentiment_window: Window size for sentiment tracking
            volume_window: Window size for volume analysis
            liminal_threshold: Threshold for liminal state (0-1)
            critical_threshold: Threshold for critical state (0-1)
        """
        self.volatility_window = volatility_window
        self.sentiment_window = sentiment_window
        self.volume_window = volume_window
        self.liminal_threshold = liminal_threshold
        self.critical_threshold = critical_threshold

        # Historical buffers
        self.volatility_history: deque = deque(maxlen=volatility_window)
        self.sentiment_history: deque = deque(maxlen=sentiment_window)
        self.volume_history: deque = deque(maxlen=volume_window)

        # State tracking
        self.current_state = "stable"
        self.state_duration = 0

        LOGGER.info(f"LiminalDetector initialized (thresholds: {liminal_threshold}/{critical_threshold})")

    def detect_volatility_spike(self, current_volatility: float) -> Optional[LiminalSignal]:
        """
        Detect sudden volatility spike

        Args:
            current_volatility: Current volatility value

        Returns:
            LiminalSignal if spike detected, None otherwise
        """
        if len(self.volatility_history) < 5:
            self.volatility_history.append(current_volatility)
            return None

        # Calculate historical mean and std
        hist_mean = np.mean(self.volatility_history)
        hist_std = np.std(self.volatility_history)

        # Handle edge case: all historical values identical
        if hist_std == 0:
            # If new value differs significantly from constant history, it's a spike
            if abs(current_volatility - hist_mean) > 0.1:
                strength = 1.0  # Maximum strength for obvious spike
                self.volatility_history.append(current_volatility)
                return LiminalSignal(
                    signal_type="volatility_spike",
                    strength=strength,
                    description=f"Volatility spike: {current_volatility:.4f} (sudden change from constant {hist_mean:.4f})",
                    timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                )
            self.volatility_history.append(current_volatility)
            return None

        # Z-score
        z_score = (current_volatility - hist_mean) / hist_std

        # Update history
        self.volatility_history.append(current_volatility)

        # Spike detection (z > 2.0)
        if z_score > 2.0:
            strength = min(1.0, z_score / 4.0)  # Normalize to [0, 1]
            return LiminalSignal(
                signal_type="volatility_spike",
                strength=strength,
                description=f"Volatility spike: {current_volatility:.4f} (z={z_score:.2f})",
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            )

        return None

    def detect_sentiment_flip(self, current_sentiment: float) -> Optional[LiminalSignal]:
        """
        Detect sentiment direction change

        Args:
            current_sentiment: Current sentiment value (-1 to 1)

        Returns:
            LiminalSignal if flip detected, None otherwise
        """
        if len(self.sentiment_history) < 3:
            self.sentiment_history.append(current_sentiment)
            return None

        # Check for sign change
        prev_sentiment = np.mean(list(self.sentiment_history)[-3:])

        self.sentiment_history.append(current_sentiment)

        # Detect flip (crossing zero with magnitude > 0.1)
        if (prev_sentiment > 0.1 and current_sentiment < -0.1) or \
           (prev_sentiment < -0.1 and current_sentiment > 0.1):

            strength = abs(current_sentiment - prev_sentiment)
            return LiminalSignal(
                signal_type="sentiment_flip",
                strength=min(1.0, strength),
                description=f"Sentiment flip: {prev_sentiment:.2f} → {current_sentiment:.2f}",
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            )

        return None

    def detect_volume_anomaly(self, current_volume: int) -> Optional[LiminalSignal]:
        """
        Detect anomalous news volume

        Args:
            current_volume: Current news count

        Returns:
            LiminalSignal if anomaly detected, None otherwise
        """
        if len(self.volume_history) < 5:
            self.volume_history.append(current_volume)
            return None

        # Calculate historical percentiles
        hist_array = np.array(self.volume_history)
        p75 = np.percentile(hist_array, 75)
        p95 = np.percentile(hist_array, 95)

        self.volume_history.append(current_volume)

        # High volume anomaly (above 95th percentile)
        if current_volume > p95 and p95 > 0:
            strength = min(1.0, (current_volume - p75) / (p95 - p75 + 1))
            return LiminalSignal(
                signal_type="volume_anomaly",
                strength=strength,
                description=f"High volume: {current_volume} (p95={p95:.0f})",
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            )

        # Low volume anomaly (below 25th percentile)
        p25 = np.percentile(hist_array, 25)
        if current_volume < p25 and p75 > 0:
            strength = min(1.0, (p75 - current_volume) / (p75 - p25 + 1))
            return LiminalSignal(
                signal_type="volume_drought",
                strength=strength * 0.5,  # Less critical than spike
                description=f"Low volume: {current_volume} (p25={p25:.0f})",
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            )

        return None

    def detect_indicator_conflict(
        self,
        sentiment: float,
        volatility: float,
        volume: int
    ) -> Optional[LiminalSignal]:
        """
        Detect conflict between indicators

        Args:
            sentiment: Sentiment score (-1 to 1)
            volatility: Volatility value
            volume: News volume

        Returns:
            LiminalSignal if conflict detected, None otherwise
        """
        # Check for contradictions
        conflicts = []

        # High volatility but neutral sentiment
        if volatility > 0.5 and abs(sentiment) < 0.1:
            conflicts.append("High volatility with neutral sentiment")

        # Strong sentiment but low volume
        if abs(sentiment) > 0.5 and volume < 20:
            conflicts.append("Strong sentiment with low volume")

        # Low volatility but extreme sentiment
        if volatility < 0.2 and abs(sentiment) > 0.7:
            conflicts.append("Low volatility with extreme sentiment")

        if conflicts:
            strength = len(conflicts) / 3.0  # Normalize
            return LiminalSignal(
                signal_type="indicator_conflict",
                strength=strength,
                description="; ".join(conflicts),
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            )

        return None

    def compute_liminal_score(self, signals: List[LiminalSignal]) -> Tuple[float, float]:
        """
        Compute overall liminal score from signals

        Args:
            signals: List of liminal signals

        Returns:
            Tuple of (liminal_score, confidence)
        """
        if not signals:
            return 0.0, 1.0

        # Weighted aggregation
        weights = {
            "volatility_spike": 0.4,
            "sentiment_flip": 0.3,
            "volume_anomaly": 0.15,
            "volume_drought": 0.05,
            "indicator_conflict": 0.1
        }

        total_score = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = weights.get(signal.signal_type, 0.1)
            total_score += signal.strength * weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            liminal_score = total_score / total_weight
        else:
            liminal_score = 0.0

        # Confidence based on number and strength of signals
        confidence = min(1.0, len(signals) / 3.0)

        return liminal_score, confidence

    def classify_state(self, liminal_score: float) -> str:
        """
        Classify market state based on liminal score

        Args:
            liminal_score: Liminal score (0-1)

        Returns:
            State classification
        """
        if liminal_score >= self.critical_threshold:
            return "critical"
        elif liminal_score >= self.liminal_threshold:
            return "liminal"
        else:
            return "stable"

    def detect(
        self,
        sentiment: float,
        volatility: float,
        volume: int,
        regime: str = "unknown"
    ) -> LiminalState:
        """
        Detect liminal state from current market indicators

        Args:
            sentiment: Current sentiment (-1 to 1)
            volatility: Current volatility (0-1)
            volume: Current news volume
            regime: Current market regime (optional)

        Returns:
            LiminalState object
        """
        signals = []

        # Run detection methods
        vol_signal = self.detect_volatility_spike(volatility)
        if vol_signal:
            signals.append(vol_signal)

        sent_signal = self.detect_sentiment_flip(sentiment)
        if sent_signal:
            signals.append(sent_signal)

        vol_anomaly = self.detect_volume_anomaly(volume)
        if vol_anomaly:
            signals.append(vol_anomaly)

        conflict = self.detect_indicator_conflict(sentiment, volatility, volume)
        if conflict:
            signals.append(conflict)

        # Compute liminal score
        liminal_score, confidence = self.compute_liminal_score(signals)

        # Classify state
        state = self.classify_state(liminal_score)

        # Update state tracking
        if state == self.current_state:
            self.state_duration += 1
        else:
            self.state_duration = 0
            self.current_state = state

        # Build liminal state
        liminal_state = LiminalState(
            state=state,
            liminal_score=liminal_score,
            confidence=confidence,
            signals=signals,
            regime=regime,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            metadata={
                "sentiment": round(sentiment, 4),
                "volatility": round(volatility, 4),
                "volume": volume,
                "state_duration": self.state_duration
            }
        )

        LOGGER.info(
            f"Liminal detection: {state} (score={liminal_score:.2f}, "
            f"confidence={confidence:.2f}, signals={len(signals)})"
        )

        return liminal_state

    def reset(self):
        """Reset detector state"""
        self.volatility_history.clear()
        self.sentiment_history.clear()
        self.volume_history.clear()
        self.current_state = "stable"
        self.state_duration = 0
        LOGGER.info("Detector reset")
