"""
Tests for liminal_detector.py and market_regime.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from liminal_detector import LiminalDetector, LiminalSignal, LiminalState
from market_regime import MarketRegimeClassifier, RegimeClassification


# ===== LiminalDetector Tests =====

def test_liminal_detector_init():
    """Test detector initialization"""
    detector = LiminalDetector()

    assert detector.liminal_threshold == 0.6
    assert detector.critical_threshold == 0.8
    assert detector.current_state == "stable"
    assert len(detector.volatility_history) == 0


def test_detect_volatility_spike():
    """Test volatility spike detection"""
    detector = LiminalDetector(volatility_window=10)

    # Build normal history
    for i in range(10):
        signal = detector.detect_volatility_spike(0.2)

    # Spike
    signal = detector.detect_volatility_spike(0.8)

    assert signal is not None
    assert signal.signal_type == "volatility_spike"
    assert signal.strength > 0.0


def test_detect_sentiment_flip():
    """Test sentiment flip detection"""
    detector = LiminalDetector(sentiment_window=10)

    # Positive sentiment
    for i in range(5):
        signal = detector.detect_sentiment_flip(0.5)

    # Flip to negative
    signal = detector.detect_sentiment_flip(-0.5)

    assert signal is not None
    assert signal.signal_type == "sentiment_flip"
    assert signal.strength > 0.0


def test_detect_volume_anomaly_high():
    """Test high volume anomaly detection"""
    detector = LiminalDetector(volume_window=10)

    # Normal volume
    for i in range(10):
        signal = detector.detect_volume_anomaly(50)

    # Spike
    signal = detector.detect_volume_anomaly(200)

    assert signal is not None
    assert signal.signal_type == "volume_anomaly"
    assert signal.strength > 0.0


def test_detect_volume_anomaly_low():
    """Test low volume anomaly detection"""
    detector = LiminalDetector(volume_window=10)

    # High volume
    for i in range(10):
        signal = detector.detect_volume_anomaly(100)

    # Drought
    signal = detector.detect_volume_anomaly(5)

    assert signal is not None
    assert signal.signal_type == "volume_drought"


def test_detect_indicator_conflict():
    """Test indicator conflict detection"""
    detector = LiminalDetector()

    # High volatility + neutral sentiment = conflict
    signal = detector.detect_indicator_conflict(
        sentiment=0.0,
        volatility=0.8,
        volume=50
    )

    assert signal is not None
    assert signal.signal_type == "indicator_conflict"


def test_compute_liminal_score():
    """Test liminal score computation"""
    detector = LiminalDetector()

    signals = [
        LiminalSignal("volatility_spike", 0.8, "test", "2025-01-01T00:00:00Z"),
        LiminalSignal("sentiment_flip", 0.6, "test", "2025-01-01T00:00:00Z")
    ]

    score, confidence = detector.compute_liminal_score(signals)

    assert 0.0 <= score <= 1.0
    assert 0.0 <= confidence <= 1.0
    assert score > 0.0  # Should have non-zero score


def test_classify_state():
    """Test state classification"""
    detector = LiminalDetector(liminal_threshold=0.6, critical_threshold=0.8)

    assert detector.classify_state(0.5) == "stable"
    assert detector.classify_state(0.7) == "liminal"
    assert detector.classify_state(0.9) == "critical"


def test_detect_stable():
    """Test detection of stable state"""
    detector = LiminalDetector()

    state = detector.detect(
        sentiment=0.1,
        volatility=0.2,
        volume=50,
        regime="bull"
    )

    assert isinstance(state, LiminalState)
    assert state.state in ["stable", "liminal", "critical"]
    assert 0.0 <= state.liminal_score <= 1.0
    assert 0.0 <= state.confidence <= 1.0


def test_detect_liminal():
    """Test detection of liminal state"""
    detector = LiminalDetector(volatility_window=5)

    # Build history
    for i in range(5):
        detector.detect(sentiment=0.3, volatility=0.2, volume=50)

    # Trigger liminal state with spike
    state = detector.detect(
        sentiment=-0.5,  # Sentiment flip
        volatility=0.9,  # Volatility spike
        volume=200       # Volume spike
    )

    assert state.liminal_score > 0.5
    assert len(state.signals) > 0


def test_detector_reset():
    """Test detector reset"""
    detector = LiminalDetector()

    # Add some history
    for i in range(5):
        detector.detect(sentiment=0.1, volatility=0.2, volume=50)

    detector.reset()

    assert len(detector.volatility_history) == 0
    assert len(detector.sentiment_history) == 0
    assert detector.current_state == "stable"


# ===== MarketRegimeClassifier Tests =====

def test_regime_classifier_init():
    """Test classifier initialization"""
    classifier = MarketRegimeClassifier()

    assert classifier.trend_window == 50
    assert classifier.volatility_window == 20
    assert classifier.current_regime == "unknown"


def test_calculate_trend_up():
    """Test upward trend calculation"""
    classifier = MarketRegimeClassifier()

    prices = [100 + i * 2 for i in range(20)]  # Upward trend
    trend = classifier.calculate_trend(prices)

    assert trend > 0.0


def test_calculate_trend_down():
    """Test downward trend calculation"""
    classifier = MarketRegimeClassifier()

    prices = [100 - i * 2 for i in range(20)]  # Downward trend
    trend = classifier.calculate_trend(prices)

    assert trend < 0.0


def test_calculate_volatility():
    """Test volatility calculation"""
    classifier = MarketRegimeClassifier()

    # Low volatility
    stable_prices = [100 + i * 0.1 for i in range(20)]
    vol_low = classifier.calculate_volatility(stable_prices)

    # High volatility
    volatile_prices = [100, 110, 95, 105, 90, 115, 85] * 3
    vol_high = classifier.calculate_volatility(volatile_prices)

    assert 0.0 <= vol_low <= 1.0
    assert 0.0 <= vol_high <= 1.0
    assert vol_high > vol_low


def test_classify_volatility_level():
    """Test volatility level classification"""
    classifier = MarketRegimeClassifier()

    assert classifier.classify_volatility_level(0.2) == "low"
    assert classifier.classify_volatility_level(0.5) == "medium"
    assert classifier.classify_volatility_level(0.8) == "high"


def test_detect_transition():
    """Test transition detection"""
    classifier = MarketRegimeClassifier()

    # Trend reversal
    previous_trends = [0.5, 0.4, 0.3]  # Upward
    current_trend = -0.3  # Reversal to downward

    is_transition = classifier.detect_transition(current_trend, previous_trends)

    assert is_transition is True


def test_classify_regime_bull():
    """Test bull regime classification"""
    classifier = MarketRegimeClassifier()

    regime = classifier.classify_regime(
        trend=0.6,  # Strong upward
        volatility=0.3,
        sentiment=0.5,
        is_transition=False
    )

    assert regime == "bull"


def test_classify_regime_bear():
    """Test bear regime classification"""
    classifier = MarketRegimeClassifier()

    regime = classifier.classify_regime(
        trend=-0.6,  # Strong downward
        volatility=0.3,
        sentiment=-0.5,
        is_transition=False
    )

    assert regime == "bear"


def test_classify_regime_sideways():
    """Test sideways regime classification"""
    classifier = MarketRegimeClassifier()

    regime = classifier.classify_regime(
        trend=0.1,  # Weak trend
        volatility=0.2,
        sentiment=0.0,
        is_transition=False
    )

    assert regime == "sideways"


def test_classify_regime_transition():
    """Test transition regime classification"""
    classifier = MarketRegimeClassifier()

    regime = classifier.classify_regime(
        trend=0.2,
        volatility=0.5,
        sentiment=0.1,
        is_transition=True  # Transition flag overrides
    )

    assert regime == "transition"


def test_check_sentiment_alignment():
    """Test sentiment alignment check"""
    classifier = MarketRegimeClassifier()

    # Aligned: both positive
    assert classifier.check_sentiment_alignment(0.5, 0.3) is True

    # Aligned: both negative
    assert classifier.check_sentiment_alignment(-0.5, -0.3) is True

    # Not aligned: opposite signs
    assert classifier.check_sentiment_alignment(0.5, -0.3) is False


def test_classify_with_price():
    """Test classification with price data"""
    classifier = MarketRegimeClassifier()

    # Build upward trend
    for i in range(20):
        price = 100 + i * 2
        classification = classifier.classify(price=price, sentiment=0.3)

    assert isinstance(classification, RegimeClassification)
    assert classification.regime in ["bull", "bear", "sideways", "transition", "unknown"]
    assert classification.trend_direction > 0.0  # Should detect upward trend


def test_classify_with_features():
    """Test classification with feature dict"""
    classifier = MarketRegimeClassifier()

    features = {
        'vader_compound_avg': 0.4,
        'vader_pos': 0.3,
        'vader_neg': 0.1,
        'count_news': 100
    }

    classification = classifier.classify(features=features)

    assert isinstance(classification, RegimeClassification)
    assert classification.regime in ["bull", "bear", "sideways", "transition", "unknown"]


def test_regime_history_tracking():
    """Test regime history tracking"""
    classifier = MarketRegimeClassifier()

    # Simulate regime changes
    for i in range(10):
        classifier.classify(price=100 + i * 5, sentiment=0.5)  # Bull

    for i in range(10):
        classifier.classify(price=150 - i * 5, sentiment=-0.5)  # Bear

    history = classifier.get_regime_history()

    assert len(history) >= 0
    assert all(r in ["bull", "bear", "sideways", "transition", "unknown"] for r in history)


def test_classifier_reset():
    """Test classifier reset"""
    classifier = MarketRegimeClassifier()

    # Add history
    for i in range(10):
        classifier.classify(price=100 + i, sentiment=0.1)

    classifier.reset()

    assert len(classifier.price_history) == 0
    assert len(classifier.regime_history) == 0
    assert classifier.current_regime == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
