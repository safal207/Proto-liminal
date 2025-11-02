"""
Module: market_regime.py
Purpose: Classify and track market regimes (bull/bear/sideways/transition)
Part of LIMINAL ProtoConsciousness — Adaptive Finance Framework
"""
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class RegimeClassification:
    """Represents market regime classification"""
    regime: str  # "bull", "bear", "sideways", "transition"
    confidence: float  # 0.0 to 1.0
    trend_direction: float  # -1.0 to 1.0 (-1=down, 0=neutral, 1=up)
    volatility_level: str  # "low", "medium", "high"
    sentiment_alignment: bool  # True if sentiment aligns with trend
    timestamp: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 4),
            "trend_direction": round(self.trend_direction, 4),
            "volatility_level": self.volatility_level,
            "sentiment_alignment": self.sentiment_alignment,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class MarketRegimeClassifier:
    """Classifies market regimes based on multiple indicators"""

    def __init__(
        self,
        trend_window: int = 50,
        volatility_window: int = 20,
        transition_threshold: float = 0.3
    ):
        """
        Initialize regime classifier

        Args:
            trend_window: Window for trend calculation
            volatility_window: Window for volatility calculation
            transition_threshold: Threshold for transition detection
        """
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.transition_threshold = transition_threshold

        # Historical buffers
        self.price_history: deque = deque(maxlen=max(trend_window, volatility_window))
        self.sentiment_history: deque = deque(maxlen=trend_window)
        self.regime_history: deque = deque(maxlen=10)  # Track regime changes

        # Current state
        self.current_regime = "unknown"
        self.regime_duration = 0

        LOGGER.info(f"MarketRegimeClassifier initialized (windows: trend={trend_window}, vol={volatility_window})")

    def calculate_trend(self, prices: List[float]) -> float:
        """
        Calculate trend direction using linear regression

        Args:
            prices: List of prices

        Returns:
            Trend slope (-1 to 1)
        """
        if len(prices) < 5:
            return 0.0

        # Simple linear regression
        x = np.arange(len(prices))
        y = np.array(prices)

        # Normalize to prevent overflow
        y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)

        # Calculate slope
        slope = np.polyfit(x, y_norm, 1)[0]

        # Normalize slope to [-1, 1]
        return np.clip(slope * 10, -1.0, 1.0)

    def calculate_volatility(self, prices: List[float]) -> float:
        """
        Calculate volatility (normalized standard deviation)

        Args:
            prices: List of prices

        Returns:
            Volatility score (0 to 1)
        """
        if len(prices) < 2:
            return 0.0

        # Calculate returns
        returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-8)

        # Standard deviation of returns
        volatility = np.std(returns)

        # Normalize to [0, 1] (assuming max vol ~0.1 = 10%)
        return np.clip(volatility / 0.1, 0.0, 1.0)

    def calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """
        Calculate price momentum

        Args:
            prices: List of prices
            period: Momentum period

        Returns:
            Momentum score (-1 to 1)
        """
        if len(prices) < period + 1:
            return 0.0

        # Rate of change over period
        current = prices[-1]
        previous = prices[-period - 1]

        if previous == 0:
            return 0.0

        momentum = (current - previous) / previous

        # Normalize to [-1, 1]
        return np.clip(momentum * 10, -1.0, 1.0)

    def classify_volatility_level(self, volatility: float) -> str:
        """
        Classify volatility as low/medium/high

        Args:
            volatility: Volatility score (0-1)

        Returns:
            Volatility level
        """
        if volatility > 0.6:
            return "high"
        elif volatility > 0.3:
            return "medium"
        else:
            return "low"

    def detect_transition(
        self,
        current_trend: float,
        previous_trends: List[float]
    ) -> bool:
        """
        Detect if market is in transition between regimes

        Args:
            current_trend: Current trend direction
            previous_trends: Historical trends

        Returns:
            True if in transition
        """
        if len(previous_trends) < 3:
            return False

        # Check for trend reversal
        prev_avg = np.mean(previous_trends[-3:])

        # Transition if:
        # 1. Trend crossed zero
        # 2. Significant change in direction
        crossed_zero = (prev_avg > 0 and current_trend < 0) or (prev_avg < 0 and current_trend > 0)
        significant_change = abs(current_trend - prev_avg) > self.transition_threshold

        return crossed_zero or significant_change

    def classify_regime(
        self,
        trend: float,
        volatility: float,
        sentiment: float,
        is_transition: bool
    ) -> str:
        """
        Classify market regime

        Args:
            trend: Trend direction (-1 to 1)
            volatility: Volatility score (0 to 1)
            sentiment: Sentiment score (-1 to 1)
            is_transition: Whether in transition state

        Returns:
            Regime classification
        """
        # Transition takes priority
        if is_transition:
            return "transition"

        # Strong trend detection
        if trend > 0.3:
            return "bull"
        elif trend < -0.3:
            return "bear"
        else:
            # Weak trend = sideways
            return "sideways"

    def check_sentiment_alignment(self, trend: float, sentiment: float) -> bool:
        """
        Check if sentiment aligns with trend

        Args:
            trend: Trend direction (-1 to 1)
            sentiment: Sentiment score (-1 to 1)

        Returns:
            True if aligned
        """
        # Both positive or both negative = aligned
        return (trend > 0 and sentiment > 0) or (trend < 0 and sentiment < 0)

    def classify(
        self,
        price: Optional[float] = None,
        sentiment: Optional[float] = None,
        features: Optional[Dict] = None
    ) -> RegimeClassification:
        """
        Classify current market regime

        Args:
            price: Current price (optional if features provided)
            sentiment: Current sentiment (optional if features provided)
            features: Feature dict with vader_compound_avg, etc. (alternative input)

        Returns:
            RegimeClassification object
        """
        # Extract from features if provided
        if features:
            sentiment = features.get('vader_compound_avg', 0.0)
            # Note: price not available in features, use volatility as proxy
            volatility = features.get('vader_pos', 0.0) + features.get('vader_neg', 0.0)
        else:
            volatility = 0.0

        # Update history
        if price is not None:
            self.price_history.append(price)
        if sentiment is not None:
            self.sentiment_history.append(sentiment)

        # Need sufficient history
        if price is not None and len(self.price_history) < 5:
            return RegimeClassification(
                regime="unknown",
                confidence=0.0,
                trend_direction=0.0,
                volatility_level="unknown",
                sentiment_alignment=False,
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                metadata={"insufficient_data": True}
            )

        # Calculate indicators
        if price is not None:
            prices = list(self.price_history)
            trend = self.calculate_trend(prices)
            volatility_calc = self.calculate_volatility(prices)
            momentum = self.calculate_momentum(prices)
        else:
            # Fallback: use sentiment as trend proxy
            if len(self.sentiment_history) >= 5:
                sentiment_list = list(self.sentiment_history)
                trend = self.calculate_trend([s + 1.0 for s in sentiment_list])  # Shift to positive
                volatility_calc = volatility
                momentum = sentiment
            else:
                trend = 0.0
                volatility_calc = volatility
                momentum = 0.0

        # Detect transition
        price_list = list(self.price_history)
        recent_trends = [self.calculate_trend(price_list[i:i+10])
                        for i in range(0, max(1, len(price_list) - 10), 5)
                        if len(price_list[i:i+10]) >= 5]

        is_transition = self.detect_transition(trend, recent_trends) if recent_trends else False

        # Classify regime
        regime = self.classify_regime(trend, volatility_calc, sentiment or 0.0, is_transition)

        # Check sentiment alignment
        sentiment_aligned = self.check_sentiment_alignment(trend, sentiment or 0.0)

        # Confidence calculation
        confidence = min(1.0, len(self.price_history) / self.trend_window)

        # Adjust confidence based on signal clarity
        if abs(trend) > 0.5:
            confidence *= 1.2  # More confident in strong trends
        if is_transition:
            confidence *= 0.7  # Less confident during transitions

        confidence = np.clip(confidence, 0.0, 1.0)

        # Update regime tracking
        if regime != self.current_regime:
            self.regime_duration = 0
            self.current_regime = regime
            self.regime_history.append(regime)
        else:
            self.regime_duration += 1

        # Classify volatility
        vol_level = self.classify_volatility_level(volatility_calc)

        # Build classification
        classification = RegimeClassification(
            regime=regime,
            confidence=confidence,
            trend_direction=trend,
            volatility_level=vol_level,
            sentiment_alignment=sentiment_aligned,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            metadata={
                "volatility": round(volatility_calc, 4),
                "momentum": round(momentum, 4),
                "sentiment": round(sentiment or 0.0, 4),
                "regime_duration": self.regime_duration,
                "is_transition": is_transition
            }
        )

        LOGGER.info(
            f"Regime classification: {regime} (trend={trend:.2f}, "
            f"vol={vol_level}, confidence={confidence:.2f})"
        )

        return classification

    def get_regime_history(self) -> List[str]:
        """Get recent regime history"""
        return list(self.regime_history)

    def reset(self):
        """Reset classifier state"""
        self.price_history.clear()
        self.sentiment_history.clear()
        self.regime_history.clear()
        self.current_regime = "unknown"
        self.regime_duration = 0
        LOGGER.info("Classifier reset")
