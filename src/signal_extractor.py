"""
Module: signal_extractor.py
Purpose: Extract structured signals and features from normalized news data
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Optional NLP dependencies
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity from text"""
    name: str
    entity_type: str  # PERSON, ORG, GPE, PRODUCT, EVENT, etc.
    mentions: int = 1
    confidence: float = 1.0
    context: List[str] = None

    def __post_init__(self):
        if self.context is None:
            self.context = []

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.entity_type,
            "mentions": self.mentions,
            "confidence": self.confidence,
            "context": self.context[:3]  # Limit context
        }


@dataclass
class ExtractedSignal:
    """Signal extracted from news data"""
    entity: str
    features: Dict[str, float]
    signal_strength: float
    timestamp: str
    source_type: str = "news"
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "features": self.features,
            "signal_strength": self.signal_strength,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "metadata": self.metadata
        }


class SignalExtractor:
    """Extract signals and features from news data"""

    def __init__(self, use_spacy: bool = False, spacy_model: str = "en_core_web_sm"):
        """
        Initialize signal extractor

        Args:
            use_spacy: Use spaCy for advanced NER (requires model download)
            spacy_model: spaCy model to use
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.nlp = None

        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                LOGGER.info(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                LOGGER.warning(f"spaCy model {spacy_model} not found, falling back to keyword extraction")
                self.use_spacy = False

        # Sentiment analyzer
        self.sentiment_analyzer = None
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            LOGGER.info("VADER sentiment analyzer loaded")

        # Entity keywords for fallback extraction
        self.entity_keywords = self._load_entity_keywords()

        # Statistics
        self.stats = {
            "documents_processed": 0,
            "entities_extracted": 0,
            "signals_generated": 0
        }

    def _load_entity_keywords(self) -> Dict[str, List[str]]:
        """Load predefined entity keywords for fallback extraction"""
        return {
            "CRYPTO": [
                "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
                "blockchain", "defi", "nft", "altcoin", "dogecoin", "cardano",
                "solana", "polkadot", "binance", "coinbase"
            ],
            "FINANCE": [
                "stock", "market", "trading", "nasdaq", "dow jones", "s&p 500",
                "wall street", "fed", "federal reserve", "interest rate",
                "inflation", "gdp", "economy", "recession", "bond"
            ],
            "TECH": [
                "ai", "artificial intelligence", "machine learning", "openai",
                "google", "microsoft", "apple", "amazon", "meta", "tesla",
                "nvidia", "technology", "software", "hardware", "chip"
            ],
            "GEOPOLITICS": [
                "china", "russia", "ukraine", "usa", "europe", "war", "conflict",
                "sanctions", "diplomacy", "nato", "un", "g7", "g20"
            ]
        }

    def extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        if not self.nlp:
            return []

        doc = self.nlp(text[:10000])  # Limit text length
        entities = []

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "MONEY"]:
                entities.append(Entity(
                    name=ent.text,
                    entity_type=ent.label_,
                    confidence=0.9
                ))

        return entities

    def extract_entities_keywords(self, text: str) -> List[Entity]:
        """Extract entities using keyword matching (fallback)"""
        text_lower = text.lower()
        entities = []

        for entity_type, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Count mentions
                    mentions = text_lower.count(keyword)

                    entities.append(Entity(
                        name=keyword.title(),
                        entity_type=entity_type,
                        mentions=mentions,
                        confidence=0.7
                    ))

        return entities

    def extract_entities(self, text: str, title: str = "") -> List[Entity]:
        """
        Extract entities from text

        Args:
            text: Main text content
            title: Title (optional, higher weight)

        Returns:
            List of Entity objects
        """
        combined_text = f"{title} {title} {text}"  # Title weighted 2x

        if self.use_spacy:
            entities = self.extract_entities_spacy(combined_text)
        else:
            entities = self.extract_entities_keywords(combined_text)

        # Deduplicate and merge
        entity_map = {}
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key in entity_map:
                entity_map[key].mentions += entity.mentions
            else:
                entity_map[key] = entity

        return list(entity_map.values())

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text

        Returns:
            Dictionary with sentiment scores
        """
        if self.sentiment_analyzer:
            # VADER sentiment
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                "sentiment": scores["compound"],  # -1 to 1
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"]
            }
        else:
            # Fallback: simple keyword-based sentiment
            positive_words = ["gain", "rise", "up", "growth", "profit", "success", "bull", "positive", "win"]
            negative_words = ["loss", "fall", "down", "decline", "drop", "crash", "bear", "negative", "fail"]

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            total = pos_count + neg_count
            if total == 0:
                sentiment = 0.0
            else:
                sentiment = (pos_count - neg_count) / total

            return {
                "sentiment": sentiment,
                "positive": pos_count / max(total, 1),
                "negative": neg_count / max(total, 1),
                "neutral": 1.0 - (pos_count + neg_count) / max(total, 1)
            }

    def calculate_urgency(self, published_at: str, tags: List[str]) -> float:
        """
        Calculate urgency score based on recency and tags

        Args:
            published_at: ISO timestamp
            tags: Article tags

        Returns:
            Urgency score 0-1
        """
        urgency = 0.5  # Default

        # Recency component
        try:
            pub_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = (now - pub_time).total_seconds() / 3600

            if age_hours < 1:
                urgency += 0.4
            elif age_hours < 6:
                urgency += 0.3
            elif age_hours < 24:
                urgency += 0.1

        except (ValueError, AttributeError):
            pass

        # Tag-based urgency
        urgent_tags = ["breaking", "alert", "urgent", "live", "now"]
        if any(tag.lower() in urgent_tags for tag in tags):
            urgency += 0.1

        return min(1.0, urgency)

    def calculate_relevance(self, entity: Entity, text: str, title: str) -> float:
        """
        Calculate relevance score for entity

        Args:
            entity: Entity object
            text: Full text
            title: Article title

        Returns:
            Relevance score 0-1
        """
        relevance = 0.5

        # Mention frequency
        text_lower = text.lower()
        entity_lower = entity.name.lower()

        mentions = text_lower.count(entity_lower)
        if mentions > 5:
            relevance += 0.3
        elif mentions > 2:
            relevance += 0.2
        elif mentions > 0:
            relevance += 0.1

        # Title presence (very important)
        if entity_lower in title.lower():
            relevance += 0.3

        # Entity type importance
        important_types = ["ORG", "PRODUCT", "CRYPTO", "FINANCE"]
        if entity.entity_type in important_types:
            relevance += 0.1

        return min(1.0, relevance)

    def extract_features(
        self,
        entity: Entity,
        text: str,
        title: str,
        published_at: str,
        tags: List[str]
    ) -> Dict[str, float]:
        """
        Extract features for an entity

        Returns:
            Dictionary of feature values
        """
        sentiment_scores = self.analyze_sentiment(text)

        return {
            "sentiment": sentiment_scores["sentiment"],
            "sentiment_positive": sentiment_scores["positive"],
            "sentiment_negative": sentiment_scores["negative"],
            "relevance": self.calculate_relevance(entity, text, title),
            "urgency": self.calculate_urgency(published_at, tags),
            "mentions": min(entity.mentions / 10.0, 1.0),  # Normalize
            "confidence": entity.confidence,
            "text_length": min(len(text) / 2000.0, 1.0),  # Normalize
        }

    def calculate_signal_strength(self, features: Dict[str, float]) -> float:
        """
        Calculate overall signal strength from features

        Uses weighted combination of key features
        """
        weights = {
            "relevance": 0.35,
            "sentiment": 0.15,  # Absolute value
            "urgency": 0.25,
            "confidence": 0.15,
            "mentions": 0.10
        }

        strength = 0.0
        for feature, weight in weights.items():
            value = features.get(feature, 0.0)
            # Use absolute value for sentiment (strong negative = strong signal)
            if feature == "sentiment":
                value = abs(value)
            strength += value * weight

        return min(1.0, max(0.0, strength))

    def extract_signals_from_record(self, record: Dict) -> List[ExtractedSignal]:
        """
        Extract signals from a single news record

        Args:
            record: News record from collector

        Returns:
            List of ExtractedSignal objects
        """
        text = record.get("text", "")
        title = record.get("title", "")
        published_at = record.get("published_at", datetime.now(timezone.utc).isoformat())
        tags = record.get("tags", [])

        if not text:
            return []

        # Extract entities
        entities = self.extract_entities(text, title)

        if not entities:
            LOGGER.debug("No entities extracted from record")
            return []

        # Generate signals for each entity
        signals = []
        for entity in entities:
            features = self.extract_features(entity, text, title, published_at, tags)
            signal_strength = self.calculate_signal_strength(features)

            # Filter weak signals
            if signal_strength < 0.3:
                continue

            signal = ExtractedSignal(
                entity=entity.name,
                features=features,
                signal_strength=signal_strength,
                timestamp=published_at,
                source_type="news",
                metadata={
                    "entity_type": entity.entity_type,
                    "source_feed": record.get("feed_url", "unknown"),
                    "language": record.get("lang", "und")
                }
            )

            signals.append(signal)
            self.stats["signals_generated"] += 1

        self.stats["documents_processed"] += 1
        self.stats["entities_extracted"] += len(entities)

        return signals

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        min_signal_strength: float = 0.3,
        max_signals: int = 1000
    ) -> Dict:
        """
        Process JSONL file and extract signals

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output signals JSONL
            min_signal_strength: Minimum signal strength threshold
            max_signals: Maximum signals to extract

        Returns:
            Processing statistics
        """
        from utils_io import ensure_parent_dir, safe_write_jsonl

        ensure_parent_dir(output_path)

        signals_written = 0
        entity_counts = Counter()

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if signals_written >= max_signals:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning(f"Line {line_num}: Invalid JSON")
                    continue

                # Extract signals
                signals = self.extract_signals_from_record(record)

                # Write signals
                for signal in signals:
                    if signal.signal_strength >= min_signal_strength:
                        safe_write_jsonl(output_path, signal.to_dict())
                        entity_counts[signal.entity] += 1
                        signals_written += 1

        # Summary stats
        summary = {
            "documents_processed": self.stats["documents_processed"],
            "entities_extracted": self.stats["entities_extracted"],
            "signals_generated": signals_written,
            "unique_entities": len(entity_counts),
            "top_entities": entity_counts.most_common(10)
        }

        LOGGER.info(f"Signal extraction complete: {signals_written} signals from {self.stats['documents_processed']} documents")
        LOGGER.info(f"Top entities: {entity_counts.most_common(5)}")

        return summary

    def get_stats(self) -> Dict:
        """Get extraction statistics"""
        return self.stats.copy()


def main():
    """CLI interface for signal extraction"""
    import argparse

    parser = argparse.ArgumentParser(description="Extract signals from news data")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output signals JSONL")
    parser.add_argument("--min-strength", type=float, default=0.3, dest="min_strength")
    parser.add_argument("--max-signals", type=int, default=1000, dest="max_signals")
    parser.add_argument("--use-spacy", action="store_true", dest="use_spacy")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create extractor
    extractor = SignalExtractor(use_spacy=args.use_spacy)

    # Process
    summary = extractor.process_jsonl(
        input_path=args.input,
        output_path=args.output,
        min_signal_strength=args.min_strength,
        max_signals=args.max_signals
    )

    # Print summary
    print("\n" + "="*60)
    print("Signal Extraction Summary")
    print("="*60)
    print(f"Documents processed: {summary['documents_processed']}")
    print(f"Entities extracted:  {summary['entities_extracted']}")
    print(f"Signals generated:   {summary['signals_generated']}")
    print(f"Unique entities:     {summary['unique_entities']}")
    print("\nTop 10 entities:")
    for entity, count in summary['top_entities']:
        print(f"  {entity:30} {count:4} signals")


if __name__ == "__main__":
    main()
