"""
Tests for predictor.py and features_builder.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from predictor import BaselinePredictor
from calibrator import apply_calibration


def test_sigmoid():
    """Test sigmoid function"""
    predictor = BaselinePredictor()

    # Test known values
    assert abs(predictor.sigmoid(0.0) - 0.5) < 0.01
    assert predictor.sigmoid(-10.0) < 0.01
    assert predictor.sigmoid(10.0) > 0.99

    # Test monotonicity
    assert predictor.sigmoid(-1.0) < predictor.sigmoid(0.0)
    assert predictor.sigmoid(0.0) < predictor.sigmoid(1.0)


def test_predict_raw():
    """Test raw prediction"""
    predictor = BaselinePredictor(a=2.0, b=0.5, c=0.5, d=0.2)

    # Positive sentiment, high volume
    features = {
        'vader_compound_avg': 0.3,
        'vader_pos': 0.4,
        'vader_neg': 0.1,
        'count_news': 100,
    }

    p_raw = predictor.predict_raw(features)

    # Should be in valid range
    assert 0.0 <= p_raw <= 1.0

    # Should be relatively high (positive sentiment)
    assert p_raw > 0.5


def test_predict_raw_negative():
    """Test raw prediction with negative sentiment"""
    predictor = BaselinePredictor(a=2.0, b=0.5, c=0.5, d=0.2)

    # Negative sentiment
    features = {
        'vader_compound_avg': -0.3,
        'vader_pos': 0.1,
        'vader_neg': 0.4,
        'count_news': 50,
    }

    p_raw = predictor.predict_raw(features)

    # Should be in valid range
    assert 0.0 <= p_raw <= 1.0

    # Should be relatively low (negative sentiment)
    assert p_raw < 0.5


def test_predict_forecast():
    """Test full forecast generation"""
    predictor = BaselinePredictor()

    features = {
        'vader_compound_avg': 0.2,
        'vader_pos': 0.3,
        'vader_neg': 0.1,
        'count_news': 75,
    }

    forecast = predictor.predict(features, event_id="TEST_EVENT")

    # Check required fields
    assert 'ts_generated' in forecast
    assert 'event_id' in forecast
    assert forecast['event_id'] == "TEST_EVENT"
    assert 'p_raw' in forecast
    assert 'p_calibrated' in forecast
    assert 'confidence_band' in forecast
    assert 'features' in forecast
    assert 'explain' in forecast

    # Check probabilities in range
    assert 0.0 <= forecast['p_raw'] <= 1.0
    assert 0.0 <= forecast['p_calibrated'] <= 1.0

    # Check confidence band
    lower, upper = forecast['confidence_band']
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= forecast['p_calibrated'] <= upper


def test_generate_explanations():
    """Test explanation generation"""
    predictor = BaselinePredictor()

    # High positive sentiment
    features = {
        'vader_compound_avg': 0.3,
        'vader_pos': 0.4,
        'vader_neg': 0.1,
        'count_news': 150,
    }

    explanations = predictor.generate_explanations(features)

    assert isinstance(explanations, list)
    assert len(explanations) > 0
    assert any("sentiment ↑" in exp for exp in explanations)
    assert any("volume of news ↑" in exp for exp in explanations)


def test_apply_calibration_without_calibrator():
    """Test calibration without calibrator"""
    p_raw = 0.65
    p_cal = apply_calibration(p_raw, calibrator=None)

    # Without calibrator, should return raw probability
    assert p_cal == p_raw


def test_apply_calibration_with_mock_calibrator():
    """Test calibration with mock calibrator"""
    class MockCalibrator:
        def predict(self, X):
            # Simple mock: slightly reduce probabilities
            return [x * 0.9 for x in X]

    p_raw = 0.65
    calibrator = MockCalibrator()
    p_cal = apply_calibration(p_raw, calibrator=calibrator)

    # Should be calibrated
    assert p_cal < p_raw
    assert abs(p_cal - 0.585) < 0.01


def test_features_builder_basic():
    """Test basic features builder functionality"""
    try:
        from features_builder import FeaturesBuilder

        builder = FeaturesBuilder(window_hours=24)

        # Test sentiment analysis
        text = "This is great news! The market is doing very well."
        sentiment = builder.analyze_sentiment_vader(text)

        assert 'pos' in sentiment
        assert 'neg' in sentiment
        assert 'neu' in sentiment
        assert 'compound' in sentiment

        # All scores should be in valid range
        assert 0.0 <= sentiment['pos'] <= 1.0
        assert 0.0 <= sentiment['neg'] <= 1.0
        assert 0.0 <= sentiment['neu'] <= 1.0
        assert -1.0 <= sentiment['compound'] <= 1.0

    except ImportError:
        pytest.skip("FeaturesBuilder dependencies not available")


def test_features_builder_fallback():
    """Test fallback sentiment analysis"""
    try:
        from features_builder import FeaturesBuilder

        builder = FeaturesBuilder()

        # Positive text
        text_pos = "great success profit growth excellent win"
        sentiment_pos = builder.analyze_sentiment_fallback(text_pos)

        # Negative text
        text_neg = "bad loss decline fail crash bear risk"
        sentiment_neg = builder.analyze_sentiment_fallback(text_neg)

        # Positive should have higher compound than negative
        assert sentiment_pos['compound'] > sentiment_neg['compound']

    except ImportError:
        pytest.skip("FeaturesBuilder not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
