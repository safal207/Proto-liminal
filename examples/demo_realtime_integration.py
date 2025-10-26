#!/usr/bin/env python3
"""
Real-time integration demo: RSS news → LiminalBD impulses
Shows how Proto-liminal can feed live signals into LiminalBD cellular substrate
"""
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict

from liminal_bridge import LiminalBridge, Signal, create_signal_from_news
from utils_io import ensure_parent_dir


LOGGER = logging.getLogger(__name__)


def extract_signals_from_jsonl(jsonl_path: str, limit: int = 10) -> List[Signal]:
    """
    Extract signals from collected news data

    Args:
        jsonl_path: Path to JSONL file with news
        limit: Maximum number of signals to extract

    Returns:
        List of Signal objects
    """
    signals = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if len(signals) >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning(f"Line {line_num}: JSON decode error: {exc}")
                    continue

                # Extract entity from title or text
                title = record.get("title", "")
                text = record.get("text", "")
                entity = extract_entity(title, text)

                if not entity:
                    continue

                # Calculate signal metrics
                # In real implementation, use NLP for sentiment analysis
                sentiment = estimate_sentiment(text)
                relevance = estimate_relevance(record)
                urgency = estimate_urgency(record)

                signal = create_signal_from_news(
                    entity=entity,
                    sentiment=sentiment,
                    relevance=relevance,
                    urgency=urgency
                )

                signals.append(signal)
                LOGGER.info(f"Extracted signal: {entity} (strength={signal.signal_strength:.2f})")

    except FileNotFoundError:
        LOGGER.error(f"File not found: {jsonl_path}")
        return []

    return signals


def extract_entity(title: str, text: str) -> str:
    """
    Extract main entity from title/text
    Simplified version - in production, use NER (Named Entity Recognition)
    """
    # Look for common crypto entities
    crypto_entities = {
        "bitcoin": "Bitcoin",
        "btc": "Bitcoin",
        "ethereum": "Ethereum",
        "eth": "Ethereum",
        "crypto": "Cryptocurrency",
        "blockchain": "Blockchain Technology"
    }

    # Look for stock market terms
    market_entities = {
        "stock": "Stock Market",
        "market": "Financial Market",
        "fed": "Federal Reserve",
        "inflation": "Inflation",
        "economy": "Economy"
    }

    text_lower = (title + " " + text).lower()

    # Check crypto first
    for keyword, entity in crypto_entities.items():
        if keyword in text_lower:
            return entity

    # Check market terms
    for keyword, entity in market_entities.items():
        if keyword in text_lower:
            return entity

    # Default: use first two words of title
    words = title.split()
    if len(words) >= 2:
        return " ".join(words[:2])

    return "General News"


def estimate_sentiment(text: str) -> float:
    """
    Estimate sentiment (-1 to 1)
    Simplified version - in production, use BERT/RoBERTa sentiment model
    """
    positive_words = ["gain", "rise", "up", "growth", "profit", "success", "bull"]
    negative_words = ["loss", "fall", "down", "decline", "drop", "crash", "bear"]

    text_lower = text.lower()

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    # Normalize to -1 to 1
    sentiment = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, sentiment))


def estimate_relevance(record: Dict) -> float:
    """
    Estimate relevance (0 to 1)
    Based on source quality, tags, etc.
    """
    # Simple heuristic: use text length as proxy
    text_length = len(record.get("text", ""))

    # Longer articles tend to be more substantial
    if text_length > 1000:
        return 0.9
    elif text_length > 500:
        return 0.7
    elif text_length > 200:
        return 0.5
    else:
        return 0.3


def estimate_urgency(record: Dict) -> float:
    """
    Estimate urgency (0 to 1)
    Based on publication time
    """
    published = record.get("published_at")
    if not published:
        return 0.5

    try:
        pub_time = datetime.fromisoformat(published.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_hours = (now - pub_time).total_seconds() / 3600

        # Newer = more urgent
        if age_hours < 1:
            return 0.9
        elif age_hours < 6:
            return 0.7
        elif age_hours < 24:
            return 0.5
        else:
            return 0.3

    except (ValueError, AttributeError):
        return 0.5


def run_demo(
    news_path: str,
    liminal_cli_path: str = "liminal-cli",
    max_signals: int = 10,
    delay_seconds: float = 2.0
):
    """
    Run real-time integration demo

    Args:
        news_path: Path to JSONL news file
        liminal_cli_path: Path to liminal-cli executable
        max_signals: Maximum signals to send
        delay_seconds: Delay between signals
    """
    LOGGER.info("Starting LiminalBD integration demo")
    LOGGER.info(f"News source: {news_path}")
    LOGGER.info(f"Max signals: {max_signals}")

    # Create bridge
    bridge = LiminalBridge(cli_path=liminal_cli_path)
    LOGGER.info("Bridge initialized")

    # Extract signals from news
    LOGGER.info("Extracting signals from news...")
    signals = extract_signals_from_jsonl(news_path, limit=max_signals)

    if not signals:
        LOGGER.warning("No signals extracted. Check news file.")
        return

    LOGGER.info(f"Extracted {len(signals)} signals")

    # Send signals with delay
    print("\n" + "="*60)
    print("Sending signals to LiminalBD...")
    print("="*60 + "\n")

    for i, signal in enumerate(signals, 1):
        print(f"[{i}/{len(signals)}] {signal.entity}")
        print(f"  Strength: {signal.signal_strength:.2f}")
        print(f"  Features: ", end="")
        for key, val in signal.features.items():
            if key != "normalized_sentiment":
                print(f"{key}={val:.2f} ", end="")
        print()

        # Convert to impulse and show pattern
        impulse = signal.to_impulse()
        print(f"  Pattern:  {impulse.pattern}")
        print(f"  Tags:     {', '.join(impulse.tags[:3])}")

        # Send to LiminalBD
        success = bridge.send_signal(signal)

        if success:
            print(f"  Status:   ✓ Sent")
        else:
            print(f"  Status:   ✗ Failed")

        print()

        # Delay before next signal
        if i < len(signals):
            time.sleep(delay_seconds)

    # Show statistics
    print("="*60)
    print("Integration Statistics")
    print("="*60)

    stats = bridge.get_stats()
    print(f"Impulses sent:    {stats['impulses_sent']}")
    print(f"Events received:  {stats['events_received']}")
    print(f"Errors:           {stats['errors']}")

    success_rate = 0
    if len(signals) > 0:
        success_rate = (stats['impulses_sent'] / len(signals)) * 100

    print(f"Success rate:     {success_rate:.1f}%")

    print("\n" + "="*60)
    LOGGER.info("Demo complete")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time RSS → LiminalBD integration demo"
    )
    parser.add_argument(
        "--news",
        default="data/raw/news_latest.jsonl",
        help="Path to JSONL news file"
    )
    parser.add_argument(
        "--liminal-cli",
        default="liminal-cli",
        dest="liminal_cli",
        help="Path to liminal-cli executable"
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=10,
        dest="max_signals",
        help="Maximum signals to send"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between signals (seconds)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Run demo
    run_demo(
        news_path=args.news,
        liminal_cli_path=args.liminal_cli,
        max_signals=args.max_signals,
        delay_seconds=args.delay
    )


if __name__ == "__main__":
    main()
