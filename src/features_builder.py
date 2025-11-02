"""
Module: features_builder.py
Purpose: Build sentiment features from normalized news using VADER
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
import argparse
import glob
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

LOGGER = logging.getLogger(__name__)


class FeaturesBuilder:
    """Build sentiment features from news articles"""

    def __init__(self, window_hours: int = 24):
        """
        Initialize features builder

        Args:
            window_hours: Time window size in hours
        """
        self.window_hours = window_hours

        if SentimentIntensityAnalyzer is None:
            LOGGER.warning("vaderSentiment not installed, using fallback sentiment")
            self.analyzer = None
        else:
            self.analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER

        Args:
            text: Input text

        Returns:
            Dictionary with pos, neg, neu, compound scores
        """
        if self.analyzer is None:
            # Fallback: simple keyword-based sentiment
            return self.analyze_sentiment_fallback(text)

        try:
            scores = self.analyzer.polarity_scores(text)
            return {
                'pos': scores.get('pos', 0.0),
                'neg': scores.get('neg', 0.0),
                'neu': scores.get('neu', 0.0),
                'compound': scores.get('compound', 0.0),
            }
        except Exception as exc:
            LOGGER.error(f"VADER analysis error: {exc}")
            return {'pos': 0.0, 'neg': 0.0, 'neu': 0.0, 'compound': 0.0}

    def analyze_sentiment_fallback(self, text: str) -> Dict[str, float]:
        """
        Fallback sentiment analysis using keywords

        Args:
            text: Input text

        Returns:
            Dictionary with pos, neg, neu, compound scores
        """
        text_lower = text.lower()

        positive_words = [
            'good', 'great', 'excellent', 'positive', 'success', 'win', 'growth',
            'profit', 'gain', 'up', 'rise', 'surge', 'rally', 'bull', 'optimistic'
        ]

        negative_words = [
            'bad', 'poor', 'negative', 'loss', 'fail', 'decline', 'fall', 'drop',
            'crash', 'bear', 'pessimistic', 'risk', 'concern', 'worry', 'fear'
        ]

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total_count = pos_count + neg_count

        if total_count == 0:
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}

        pos_score = pos_count / total_count
        neg_score = neg_count / total_count
        neu_score = 1.0 - (pos_score + neg_score)
        compound = (pos_score - neg_score)

        return {
            'pos': pos_score,
            'neg': neg_score,
            'neu': max(0.0, neu_score),
            'compound': compound,
        }

    def build_features(self, records: List[Dict]) -> List[Dict]:
        """
        Build features from records using time windows

        Args:
            records: List of normalized news records

        Returns:
            List of feature dictionaries
        """
        if not records:
            return []

        # Parse timestamps and sort
        timestamped = []
        for record in records:
            ts_str = record.get('published_at_utc')
            if not ts_str:
                continue

            try:
                if ts_str.endswith('Z'):
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                else:
                    ts = datetime.fromisoformat(ts_str)

                timestamped.append((ts, record))
            except Exception as exc:
                LOGGER.warning(f"Could not parse timestamp: {ts_str} - {exc}")

        if not timestamped:
            return []

        # Sort by timestamp
        timestamped.sort(key=lambda x: x[0])

        # Find time range
        min_ts = timestamped[0][0]
        max_ts = timestamped[-1][0]

        # Generate windows
        current_ts = min_ts
        window_delta = timedelta(hours=self.window_hours)
        features = []

        while current_ts <= max_ts:
            window_start = current_ts
            window_end = current_ts + window_delta

            # Collect records in window
            window_records = [
                rec for ts, rec in timestamped
                if window_start <= ts < window_end
            ]

            if window_records:
                # Compute features for window
                window_features = self.compute_window_features(
                    window_records,
                    window_start,
                    window_end
                )
                features.append(window_features)

            # Move to next window
            current_ts += window_delta

        return features

    def compute_window_features(
        self,
        records: List[Dict],
        window_start: datetime,
        window_end: datetime
    ) -> Dict:
        """
        Compute sentiment features for a time window

        Args:
            records: Records in window
            window_start: Window start timestamp
            window_end: Window end timestamp

        Returns:
            Feature dictionary
        """
        sentiments = []

        for record in records:
            text = record.get('text_clean', record.get('text', ''))
            if not text:
                continue

            sentiment = self.analyze_sentiment_vader(text)
            sentiments.append(sentiment)

        if not sentiments:
            return {
                'window_start': window_start.isoformat().replace('+00:00', 'Z'),
                'window_end': window_end.isoformat().replace('+00:00', 'Z'),
                'count_news': 0,
                'vader_pos': 0.0,
                'vader_neg': 0.0,
                'vader_neu': 0.0,
                'vader_compound_avg': 0.0,
            }

        # Aggregate sentiments
        count = len(sentiments)
        avg_pos = sum(s['pos'] for s in sentiments) / count
        avg_neg = sum(s['neg'] for s in sentiments) / count
        avg_neu = sum(s['neu'] for s in sentiments) / count
        avg_compound = sum(s['compound'] for s in sentiments) / count

        return {
            'window_start': window_start.isoformat().replace('+00:00', 'Z'),
            'window_end': window_end.isoformat().replace('+00:00', 'Z'),
            'count_news': count,
            'vader_pos': avg_pos,
            'vader_neg': avg_neg,
            'vader_neu': avg_neu,
            'vader_compound_avg': avg_compound,
        }

    def save_to_parquet(self, features: List[Dict], output_path: str):
        """
        Save features to Parquet file

        Args:
            features: List of feature dictionaries
            output_path: Output file path
        """
        if pd is None:
            raise ImportError("pandas is required for Parquet export")

        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(features)

        # Save to Parquet
        df.to_parquet(output_path, index=False, engine='pyarrow')

        LOGGER.info(f"Saved {len(features)} feature records to {output_path}")


def main():
    """CLI interface for features builder"""
    parser = argparse.ArgumentParser(
        description="Build sentiment features from normalized news",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/features_builder.py \\
    --inp data/clean/news_norm_*.jsonl \\
    --out data/features/news_features.parquet \\
    --window-hours 24
        """
    )

    parser.add_argument(
        '--inp',
        required=True,
        nargs='+',
        help='Input JSONL file(s) or glob pattern(s)'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output Parquet file'
    )
    parser.add_argument(
        '--window-hours',
        type=int,
        default=24,
        dest='window_hours',
        help='Time window size in hours (default: 24)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Check pandas availability
    if pd is None:
        LOGGER.error("pandas is required. Install with: pip install pandas pyarrow")
        return 1

    # Expand globs
    all_files = []
    for pattern in args.inp:
        matched = glob.glob(pattern)
        if matched:
            all_files.extend(matched)
        else:
            all_files.append(pattern)

    if not all_files:
        LOGGER.error(f"No files matched: {args.inp}")
        return 1

    # Load records
    LOGGER.info(f"Loading records from {len(all_files)} file(s)")
    records = []

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        LOGGER.warning(f"Invalid JSON at {file_path}:{line_num}")

        except FileNotFoundError:
            LOGGER.warning(f"File not found: {file_path}")
        except Exception as exc:
            LOGGER.error(f"Error reading {file_path}: {exc}")

    LOGGER.info(f"Loaded {len(records)} records")

    if not records:
        LOGGER.error("No records to process")
        return 1

    # Build features
    builder = FeaturesBuilder(window_hours=args.window_hours)
    features = builder.build_features(records)

    LOGGER.info(f"Generated {len(features)} feature windows")

    # Save to Parquet
    builder.save_to_parquet(features, args.out)

    # Print summary
    print("\n" + "=" * 60)
    print("Features Builder Summary")
    print("=" * 60)
    print(f"Input files:          {len(all_files)}")
    print(f"Records processed:    {len(records)}")
    print(f"Feature windows:      {len(features)}")
    print(f"Window size:          {args.window_hours} hours")
    print(f"Output file:          {args.out}")

    return 0


if __name__ == "__main__":
    exit(main())
