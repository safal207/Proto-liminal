"""
Module: predictor.py
Purpose: Baseline predictor using sigmoid model on VADER sentiment features
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

from calibrator import load_calibrator, apply_calibration

LOGGER = logging.getLogger(__name__)


class BaselinePredictor:
    """Baseline sigmoid predictor for event probabilities"""

    def __init__(
        self,
        a: float = 2.0,
        b: float = 0.5,
        c: float = 0.5,
        d: float = 0.2,
        calibrator_path: Optional[str] = None
    ):
        """
        Initialize baseline predictor

        Formula: p_raw = sigmoid(a*compound_avg + b*pos - c*neg + d*log(1+count))

        Args:
            a: Weight for compound sentiment
            b: Weight for positive sentiment
            c: Weight for negative sentiment
            d: Weight for news count (log scaled)
            calibrator_path: Path to calibrator model (optional)
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # Load calibrator if available
        self.calibrator = None
        if calibrator_path:
            self.calibrator = load_calibrator(calibrator_path)

        LOGGER.info(f"Initialized BaselinePredictor (a={a}, b={b}, c={c}, d={d})")

    def sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function

        Args:
            x: Input value

        Returns:
            Sigmoid output in [0, 1]
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            # Handle extreme values
            return 0.0 if x < 0 else 1.0

    def predict_raw(self, features: Dict) -> float:
        """
        Compute raw probability from features

        Args:
            features: Feature dictionary with vader_* and count_news

        Returns:
            Raw probability in [0, 1]
        """
        compound_avg = features.get('vader_compound_avg', 0.0)
        pos = features.get('vader_pos', 0.0)
        neg = features.get('vader_neg', 0.0)
        count = features.get('count_news', 0)

        # Compute logit
        logit = (
            self.a * compound_avg +
            self.b * pos -
            self.c * neg +
            self.d * math.log(1 + count)
        )

        # Apply sigmoid
        p_raw = self.sigmoid(logit)

        return p_raw

    def predict(self, features: Dict, event_id: str = "BTC_UP_24H_GT_2PCT") -> Dict:
        """
        Generate forecast from features

        Args:
            features: Feature dictionary
            event_id: Event identifier

        Returns:
            Forecast dictionary
        """
        # Compute raw probability
        p_raw = self.predict_raw(features)

        # Apply calibration if available
        p_calibrated = apply_calibration(p_raw, self.calibrator)

        # Compute confidence band (simple heuristic based on sample size)
        count = features.get('count_news', 0)
        if count > 100:
            margin = 0.1
        elif count > 50:
            margin = 0.15
        elif count > 20:
            margin = 0.2
        else:
            margin = 0.25

        confidence_lower = max(0.0, p_calibrated - margin)
        confidence_upper = min(1.0, p_calibrated + margin)

        # Generate explanations
        explanations = self.generate_explanations(features)

        # Build forecast
        forecast = {
            'ts_generated': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'event_id': event_id,
            'p_raw': round(p_raw, 4),
            'p_calibrated': round(p_calibrated, 4),
            'confidence_band': [round(confidence_lower, 4), round(confidence_upper, 4)],
            'features': {
                'vader_compound_avg': round(features.get('vader_compound_avg', 0.0), 4),
                'vader_pos': round(features.get('vader_pos', 0.0), 4),
                'vader_neg': round(features.get('vader_neg', 0.0), 4),
                'count_news': features.get('count_news', 0),
            },
            'explain': explanations,
        }

        return forecast

    def generate_explanations(self, features: Dict) -> List[str]:
        """
        Generate human-readable explanations

        Args:
            features: Feature dictionary

        Returns:
            List of explanation strings
        """
        explanations = []

        compound_avg = features.get('vader_compound_avg', 0.0)
        pos = features.get('vader_pos', 0.0)
        neg = features.get('vader_neg', 0.0)
        count = features.get('count_news', 0)

        # Sentiment direction
        if compound_avg > 0.1:
            explanations.append("news sentiment ↑")
        elif compound_avg < -0.1:
            explanations.append("news sentiment ↓")

        # Positive sentiment
        if pos > 0.3:
            explanations.append("high positive tone")

        # Negative sentiment
        if neg > 0.3:
            explanations.append("high negative tone")

        # Volume
        if count > 100:
            explanations.append("volume of news ↑")
        elif count < 20:
            explanations.append("volume of news ↓")

        return explanations if explanations else ["neutral indicators"]


def main():
    """CLI interface for predictor"""
    parser = argparse.ArgumentParser(
        description="Baseline predictor for event probabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/predictor.py \\
    --features data/features/news_features.parquet \\
    --out data/forecast/forecast_$(date +%Y%m%d%H).jsonl \\
    --event BTC_UP_24H_GT_2PCT
        """
    )

    parser.add_argument(
        '--features',
        required=True,
        help='Input Parquet file with features'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output JSONL file for forecasts'
    )
    parser.add_argument(
        '--event',
        default='BTC_UP_24H_GT_2PCT',
        help='Event ID (default: BTC_UP_24H_GT_2PCT)'
    )
    parser.add_argument(
        '--calibrator',
        help='Path to calibrator model (optional)'
    )
    parser.add_argument(
        '-a',
        type=float,
        default=2.0,
        help='Weight for compound sentiment (default: 2.0)'
    )
    parser.add_argument(
        '-b',
        type=float,
        default=0.5,
        help='Weight for positive sentiment (default: 0.5)'
    )
    parser.add_argument(
        '-c',
        type=float,
        default=0.5,
        help='Weight for negative sentiment (default: 0.5)'
    )
    parser.add_argument(
        '-d',
        type=float,
        default=0.2,
        help='Weight for log(count) (default: 0.2)'
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

    # Load features
    LOGGER.info(f"Loading features from {args.features}")

    try:
        df = pd.read_parquet(args.features)
        LOGGER.info(f"Loaded {len(df)} feature records")
    except FileNotFoundError:
        LOGGER.error(f"Features file not found: {args.features}")
        return 1
    except Exception as exc:
        LOGGER.error(f"Error loading features: {exc}")
        return 1

    if len(df) == 0:
        LOGGER.error("No features to process")
        return 1

    # Initialize predictor
    predictor = BaselinePredictor(
        a=args.a,
        b=args.b,
        c=args.c,
        d=args.d,
        calibrator_path=args.calibrator
    )

    # Create output directory
    output_dir = Path(args.out).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate forecasts
    LOGGER.info("Generating forecasts")
    forecasts = []

    for idx, row in df.iterrows():
        features = row.to_dict()
        forecast = predictor.predict(features, event_id=args.event)
        forecasts.append(forecast)

    # Write to JSONL
    with open(args.out, 'w', encoding='utf-8') as f:
        for forecast in forecasts:
            f.write(json.dumps(forecast, ensure_ascii=False) + '\n')

    LOGGER.info(f"Wrote {len(forecasts)} forecasts to {args.out}")

    # Print summary
    print("\n" + "=" * 60)
    print("Predictor Summary")
    print("=" * 60)
    print(f"Input features:       {len(df)}")
    print(f"Forecasts generated:  {len(forecasts)}")
    print(f"Event ID:             {args.event}")
    print(f"Model params:         a={args.a}, b={args.b}, c={args.c}, d={args.d}")
    print(f"Output file:          {args.out}")

    if forecasts:
        avg_prob = sum(f['p_calibrated'] for f in forecasts) / len(forecasts)
        print(f"Avg probability:      {avg_prob:.2%}")

    return 0


if __name__ == "__main__":
    exit(main())
