"""Tests for signal extraction module."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from signal_extractor import (
    SignalExtractor,
    Entity,
    ExtractedSignal
)


def test_entity_creation():
    """Test Entity dataclass"""
    entity = Entity(
        name="Bitcoin",
        entity_type="CRYPTO",
        mentions=3,
        confidence=0.9
    )

    assert entity.name == "Bitcoin"
    assert entity.entity_type == "CRYPTO"
    assert entity.mentions == 3
    assert entity.confidence == 0.9

    # Test to_dict
    entity_dict = entity.to_dict()
    assert entity_dict["name"] == "Bitcoin"
    assert entity_dict["type"] == "CRYPTO"


def test_extracted_signal_creation():
    """Test ExtractedSignal dataclass"""
    signal = ExtractedSignal(
        entity="Bitcoin",
        features={"sentiment": 0.8, "relevance": 0.9},
        signal_strength=0.85,
        timestamp="2025-10-26T12:00:00Z"
    )

    assert signal.entity == "Bitcoin"
    assert signal.signal_strength == 0.85
    assert signal.features["sentiment"] == 0.8

    # Test to_dict
    signal_dict = signal.to_dict()
    assert signal_dict["entity"] == "Bitcoin"
    assert signal_dict["signal_strength"] == 0.85


def test_extractor_initialization():
    """Test SignalExtractor initialization"""
    extractor = SignalExtractor()

    assert extractor.entity_keywords is not None
    assert "CRYPTO" in extractor.entity_keywords
    assert "FINANCE" in extractor.entity_keywords
    assert extractor.stats["documents_processed"] == 0


def test_keyword_entity_extraction():
    """Test keyword-based entity extraction"""
    extractor = SignalExtractor(use_spacy=False)

    text = "Bitcoin price surged to new highs amid strong Ethereum demand. The cryptocurrency market is booming."

    entities = extractor.extract_entities(text)

    # Should extract Bitcoin, Ethereum, crypto-related entities
    entity_names = [e.name.lower() for e in entities]

    assert any("bitcoin" in name for name in entity_names)
    assert any("ethereum" in name for name in entity_names)
    assert len(entities) > 0


def test_sentiment_analysis_positive():
    """Test sentiment analysis with positive text"""
    extractor = SignalExtractor()

    text = "The market saw incredible gains today with stocks rising to record highs. Investors are very optimistic about future growth."

    sentiment = extractor.analyze_sentiment(text)

    assert "sentiment" in sentiment
    assert "positive" in sentiment
    assert "negative" in sentiment

    # Should be positive
    assert sentiment["sentiment"] > 0


def test_sentiment_analysis_negative():
    """Test sentiment analysis with negative text"""
    extractor = SignalExtractor()

    text = "Markets crashed today with massive losses across all sectors. Investors fear a major recession is coming."

    sentiment = extractor.analyze_sentiment(text)

    # Should be negative
    assert sentiment["sentiment"] < 0


def test_urgency_calculation():
    """Test urgency score calculation"""
    extractor = SignalExtractor()

    # Recent timestamp (should be urgent)
    from datetime import datetime, timezone, timedelta

    recent = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    urgency_recent = extractor.calculate_urgency(recent, [])

    # Old timestamp (should be less urgent)
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    urgency_old = extractor.calculate_urgency(old, [])

    assert urgency_recent > urgency_old

    # With urgent tags
    urgency_tagged = extractor.calculate_urgency(old, ["breaking", "urgent"])
    assert urgency_tagged > urgency_old


def test_relevance_calculation():
    """Test relevance score calculation"""
    extractor = SignalExtractor()

    entity = Entity(name="Bitcoin", entity_type="CRYPTO", mentions=5)

    # Text with many mentions
    text = "Bitcoin reached new highs. Bitcoin trading volume surged. Bitcoin investors are excited. Bitcoin blockchain technology advancing."
    title = "Bitcoin Price Analysis"

    relevance = extractor.calculate_relevance(entity, text, title)

    # Should have high relevance (in title + many mentions)
    assert relevance > 0.7


def test_signal_strength_calculation():
    """Test signal strength calculation"""
    extractor = SignalExtractor()

    # Strong signal (high relevance, sentiment, urgency)
    strong_features = {
        "relevance": 0.9,
        "sentiment": 0.8,
        "urgency": 0.9,
        "confidence": 0.9,
        "mentions": 0.8
    }

    strength_strong = extractor.calculate_signal_strength(strong_features)

    # Weak signal
    weak_features = {
        "relevance": 0.2,
        "sentiment": 0.1,
        "urgency": 0.2,
        "confidence": 0.3,
        "mentions": 0.1
    }

    strength_weak = extractor.calculate_signal_strength(weak_features)

    assert strength_strong > strength_weak
    assert 0 <= strength_strong <= 1
    assert 0 <= strength_weak <= 1


def test_extract_signals_from_record():
    """Test extracting signals from a news record"""
    extractor = SignalExtractor(use_spacy=False)

    record = {
        "title": "Bitcoin Surges Past $50,000 Amid Strong Demand",
        "text": "Bitcoin price surged past $50,000 today as demand from institutional investors continues to grow. Ethereum also saw gains, rising 8% in 24 hours. The cryptocurrency market capitalization increased by $100 billion.",
        "published_at": "2025-10-26T12:00:00Z",
        "tags": ["crypto", "markets"],
        "feed_url": "https://example.com/feed",
        "lang": "en"
    }

    signals = extractor.extract_signals_from_record(record)

    assert len(signals) > 0

    # Check signal structure
    signal = signals[0]
    assert hasattr(signal, "entity")
    assert hasattr(signal, "features")
    assert hasattr(signal, "signal_strength")
    assert hasattr(signal, "timestamp")

    # Check features
    assert "sentiment" in signal.features
    assert "relevance" in signal.features
    assert "urgency" in signal.features

    # Check signal strength is valid
    assert 0 <= signal.signal_strength <= 1


def test_extract_signals_filters_weak():
    """Test that weak signals are filtered out"""
    extractor = SignalExtractor(use_spacy=False)

    # Record with weak/irrelevant content
    record = {
        "title": "Some random news",
        "text": "This is a short text with no relevant entities or sentiment.",
        "published_at": "2025-01-01T12:00:00Z",
        "tags": [],
        "feed_url": "https://example.com/feed",
        "lang": "en"
    }

    signals = extractor.extract_signals_from_record(record)

    # Should filter out weak signals (strength < 0.3)
    for signal in signals:
        assert signal.signal_strength >= 0.3


def test_extractor_stats_tracking():
    """Test statistics tracking"""
    extractor = SignalExtractor(use_spacy=False)

    record = {
        "title": "Bitcoin and Ethereum Analysis",
        "text": "Bitcoin and Ethereum both saw strong gains in the cryptocurrency market today.",
        "published_at": "2025-10-26T12:00:00Z",
        "tags": [],
        "feed_url": "https://example.com/feed",
        "lang": "en"
    }

    initial_stats = extractor.get_stats()
    assert initial_stats["documents_processed"] == 0

    signals = extractor.extract_signals_from_record(record)

    final_stats = extractor.get_stats()
    assert final_stats["documents_processed"] == 1
    assert final_stats["entities_extracted"] > 0
    assert final_stats["signals_generated"] >= len(signals)


def test_extract_features():
    """Test feature extraction for an entity"""
    extractor = SignalExtractor()

    entity = Entity(name="Bitcoin", entity_type="CRYPTO", mentions=3, confidence=0.9)

    text = "Bitcoin price surged today with strong gains across the cryptocurrency market."
    title = "Bitcoin Surges"
    published_at = "2025-10-26T12:00:00Z"
    tags = ["crypto", "breaking"]

    features = extractor.extract_features(entity, text, title, published_at, tags)

    # Check all required features are present
    required_features = [
        "sentiment",
        "sentiment_positive",
        "sentiment_negative",
        "relevance",
        "urgency",
        "mentions",
        "confidence",
        "text_length"
    ]

    for feature in required_features:
        assert feature in features
        assert isinstance(features[feature], (int, float))
        assert 0 <= features[feature] <= 1 or -1 <= features[feature] <= 1  # Sentiment can be negative


def test_multiple_entity_types():
    """Test extraction of different entity types"""
    extractor = SignalExtractor(use_spacy=False)

    text = """
    Bitcoin and Ethereum led cryptocurrency gains today.
    The Federal Reserve announced new interest rate policy.
    Apple and Microsoft stocks rose on tech sector optimism.
    China and USA continued trade negotiations.
    """

    entities = extractor.extract_entities(text)

    entity_types = {e.entity_type for e in entities}

    # Should detect multiple categories
    assert len(entity_types) > 1
    assert any(t in entity_types for t in ["CRYPTO", "FINANCE", "TECH", "GEOPOLITICS"])


def test_entity_deduplication():
    """Test that duplicate entities are merged"""
    extractor = SignalExtractor(use_spacy=False)

    text = "Bitcoin Bitcoin Bitcoin"  # Same entity multiple times

    entities = extractor.extract_entities(text)

    # Count how many Bitcoin entities
    bitcoin_entities = [e for e in entities if "bitcoin" in e.name.lower()]

    # Should only have one Bitcoin entity (deduplicated)
    assert len(bitcoin_entities) == 1

    # But mentions should be counted
    if bitcoin_entities:
        assert bitcoin_entities[0].mentions >= 3


def test_empty_text_handling():
    """Test handling of empty or missing text"""
    extractor = SignalExtractor()

    record = {
        "title": "",
        "text": "",
        "published_at": "2025-10-26T12:00:00Z",
        "tags": [],
        "feed_url": "https://example.com/feed",
        "lang": "en"
    }

    signals = extractor.extract_signals_from_record(record)

    # Should return empty list for empty text
    assert signals == []


def test_signal_to_dict_serialization():
    """Test that signals can be serialized to dict/JSON"""
    extractor = SignalExtractor(use_spacy=False)

    record = {
        "title": "Bitcoin Analysis",
        "text": "Bitcoin price movement analysis for cryptocurrency traders and investors.",
        "published_at": "2025-10-26T12:00:00Z",
        "tags": ["crypto"],
        "feed_url": "https://example.com/feed",
        "lang": "en"
    }

    signals = extractor.extract_signals_from_record(record)

    if signals:
        signal_dict = signals[0].to_dict()

        # Should be JSON serializable
        json_str = json.dumps(signal_dict)
        assert isinstance(json_str, str)

        # Should round-trip
        loaded = json.loads(json_str)
        assert loaded["entity"] == signal_dict["entity"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
