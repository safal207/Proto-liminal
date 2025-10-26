"""
Module: predictor.py
Purpose: Produce probabilistic forecasts with confidence intervals from signals
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


@dataclass
class Forecast:
    """Represents a probabilistic forecast"""
    entity: str
    probability: float  # Main probability (0-1)
    confidence_interval: Tuple[float, float]  # (lower, upper) bounds
    forecast_type: str  # "movement", "sentiment", "volatility"
    horizon: str  # "1h", "6h", "24h", "7d"
    timestamp: str
    features_used: Dict[str, float]
    model_confidence: float = 1.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Ensure probability is in valid range
        self.probability = max(0.0, min(1.0, self.probability))

        # Ensure confidence interval is valid
        lower, upper = self.confidence_interval
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        self.confidence_interval = (lower, upper)

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "probability": self.probability,
            "confidence_interval": list(self.confidence_interval),
            "forecast_type": self.forecast_type,
            "horizon": self.horizon,
            "timestamp": self.timestamp,
            "features_used": self.features_used,
            "model_confidence": self.model_confidence,
            "metadata": self.metadata
        }


class BasePredictor:
    """Base class for predictors"""

    def __init__(self, model_type: str = "logistic"):
        """
        Initialize predictor

        Args:
            model_type: Type of model ("logistic", "random_forest", "simple")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Initialize model
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
        elif model_type == "simple":
            # Rule-based predictor (no training needed)
            self.model = None
            self.is_fitted = True

        self.stats = {
            "predictions_made": 0,
            "entities_tracked": set()
        }

        LOGGER.info(f"Initialized {model_type} predictor")

    def extract_feature_vector(self, signal: Dict) -> np.ndarray:
        """
        Extract feature vector from signal

        Args:
            signal: Signal dictionary

        Returns:
            Feature vector
        """
        features = signal.get("features", {})

        # Core features
        feature_vector = [
            features.get("sentiment", 0.0),
            features.get("sentiment_positive", 0.0),
            features.get("sentiment_negative", 0.0),
            features.get("relevance", 0.0),
            features.get("urgency", 0.0),
            features.get("mentions", 0.0),
            features.get("confidence", 0.0),
            features.get("text_length", 0.0),
            signal.get("signal_strength", 0.0)
        ]

        return np.array(feature_vector)

    def predict_probability_simple(self, signal: Dict) -> float:
        """
        Simple rule-based probability prediction

        Uses signal strength and sentiment as main indicators
        """
        features = signal.get("features", {})
        signal_strength = signal.get("signal_strength", 0.5)
        sentiment = features.get("sentiment", 0.0)
        relevance = features.get("relevance", 0.5)
        urgency = features.get("urgency", 0.5)

        # Positive movement probability
        # Combines signal strength with sentiment direction
        base_prob = signal_strength * 0.5  # 0-0.5 range

        # Add sentiment component
        if sentiment > 0:
            base_prob += sentiment * 0.3
        else:
            base_prob += sentiment * 0.2  # Negative sentiment reduces probability

        # Add urgency boost
        base_prob += urgency * 0.15

        # Add relevance component
        base_prob += relevance * 0.05

        return max(0.0, min(1.0, base_prob))

    def calculate_confidence_interval(
        self,
        probability: float,
        signal: Dict,
        width_factor: float = 0.15
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for probability

        Args:
            probability: Point estimate
            signal: Signal dictionary
            width_factor: Base width of interval

        Returns:
            (lower, upper) bounds
        """
        features = signal.get("features", {})
        confidence = features.get("confidence", 0.7)

        # Interval width inversely proportional to confidence
        width = width_factor * (1.0 - confidence * 0.5)

        lower = max(0.0, probability - width)
        upper = min(1.0, probability + width)

        return (lower, upper)

    def predict_from_signal(
        self,
        signal: Dict,
        horizon: str = "24h",
        forecast_type: str = "movement"
    ) -> Forecast:
        """
        Generate forecast from a single signal

        Args:
            signal: Signal dictionary
            horizon: Forecast horizon
            forecast_type: Type of forecast

        Returns:
            Forecast object
        """
        entity = signal.get("entity", "Unknown")
        features = signal.get("features", {})

        # Get probability prediction
        if self.model_type == "simple" or not self.is_fitted:
            probability = self.predict_probability_simple(signal)
            model_confidence = 0.6  # Lower confidence for simple model
        else:
            # Use trained model
            feature_vector = self.extract_feature_vector(signal).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            probability = self.model.predict_proba(feature_vector_scaled)[0][1]
            model_confidence = 0.8

        # Calculate confidence interval
        conf_interval = self.calculate_confidence_interval(probability, signal)

        # Create forecast
        forecast = Forecast(
            entity=entity,
            probability=probability,
            confidence_interval=conf_interval,
            forecast_type=forecast_type,
            horizon=horizon,
            timestamp=datetime.now(timezone.utc).isoformat(),
            features_used=features,
            model_confidence=model_confidence,
            metadata={
                "signal_strength": signal.get("signal_strength", 0.0),
                "signal_timestamp": signal.get("timestamp", ""),
                "model_type": self.model_type
            }
        )

        self.stats["predictions_made"] += 1
        self.stats["entities_tracked"].add(entity)

        return forecast

    def predict_batch(
        self,
        signals: List[Dict],
        horizon: str = "24h",
        forecast_type: str = "movement"
    ) -> List[Forecast]:
        """
        Generate forecasts for multiple signals

        Args:
            signals: List of signal dictionaries
            horizon: Forecast horizon
            forecast_type: Type of forecast

        Returns:
            List of Forecast objects
        """
        forecasts = []

        for signal in signals:
            try:
                forecast = self.predict_from_signal(signal, horizon, forecast_type)
                forecasts.append(forecast)
            except Exception as exc:
                LOGGER.error(f"Error predicting for {signal.get('entity')}: {exc}")

        LOGGER.info(f"Generated {len(forecasts)} forecasts from {len(signals)} signals")

        return forecasts

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        horizon: str = "24h",
        forecast_type: str = "movement",
        max_forecasts: int = 1000
    ) -> Dict:
        """
        Process signals from JSONL and generate forecasts

        Args:
            input_path: Path to input signals JSONL
            output_path: Path to output forecasts JSONL
            horizon: Forecast horizon
            forecast_type: Type of forecast
            max_forecasts: Maximum forecasts to generate

        Returns:
            Processing statistics
        """
        from utils_io import ensure_parent_dir, safe_write_jsonl

        ensure_parent_dir(output_path)

        signals_processed = 0
        forecasts_written = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if forecasts_written >= max_forecasts:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    signal = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning(f"Line {line_num}: Invalid JSON")
                    continue

                signals_processed += 1

                # Generate forecast
                try:
                    forecast = self.predict_from_signal(signal, horizon, forecast_type)
                    safe_write_jsonl(output_path, forecast.to_dict())
                    forecasts_written += 1
                except Exception as exc:
                    LOGGER.error(f"Line {line_num}: Prediction error: {exc}")

        summary = {
            "signals_processed": signals_processed,
            "forecasts_generated": forecasts_written,
            "unique_entities": len(self.stats["entities_tracked"]),
            "model_type": self.model_type
        }

        LOGGER.info(f"Forecast generation complete: {forecasts_written} forecasts from {signals_processed} signals")

        return summary

    def get_stats(self) -> Dict:
        """Get predictor statistics"""
        stats = self.stats.copy()
        stats["entities_tracked"] = len(self.stats["entities_tracked"])
        return stats


class MovementPredictor(BasePredictor):
    """Predicts entity price/value movement"""

    def __init__(self, model_type: str = "simple"):
        super().__init__(model_type)
        LOGGER.info("MovementPredictor initialized")

    def predict_from_signal(self, signal: Dict, horizon: str = "24h", forecast_type: str = "movement") -> Forecast:
        """Predict movement probability"""
        # Force forecast type to movement
        return super().predict_from_signal(signal, horizon, "movement")


class SentimentPredictor(BasePredictor):
    """Predicts sentiment direction"""

    def __init__(self, model_type: str = "simple"):
        super().__init__(model_type)
        LOGGER.info("SentimentPredictor initialized")

    def predict_probability_simple(self, signal: Dict) -> float:
        """Simple sentiment prediction"""
        features = signal.get("features", {})
        sentiment = features.get("sentiment", 0.0)

        # Probability of positive sentiment
        # Map from [-1, 1] to [0, 1]
        probability = (sentiment + 1) / 2

        return probability

    def predict_from_signal(self, signal: Dict, horizon: str = "24h", forecast_type: str = "sentiment") -> Forecast:
        """Predict sentiment probability"""
        return super().predict_from_signal(signal, horizon, "sentiment")


class VolatilityPredictor(BasePredictor):
    """Predicts volatility/uncertainty"""

    def __init__(self, model_type: str = "simple"):
        super().__init__(model_type)
        LOGGER.info("VolatilityPredictor initialized")

    def predict_probability_simple(self, signal: Dict) -> float:
        """Simple volatility prediction"""
        features = signal.get("features", {})

        # High volatility indicators:
        # - Strong sentiment (positive or negative)
        # - High urgency
        # - Low confidence
        sentiment_abs = abs(features.get("sentiment", 0.0))
        urgency = features.get("urgency", 0.0)
        confidence = features.get("confidence", 0.7)

        # High volatility probability
        volatility = (
            sentiment_abs * 0.4 +
            urgency * 0.4 +
            (1.0 - confidence) * 0.2
        )

        return min(1.0, volatility)

    def predict_from_signal(self, signal: Dict, horizon: str = "24h", forecast_type: str = "volatility") -> Forecast:
        """Predict volatility probability"""
        return super().predict_from_signal(signal, horizon, "volatility")


class MultiPredictor:
    """Combines multiple predictors"""

    def __init__(self, model_type: str = "simple"):
        """Initialize all predictor types"""
        self.movement = MovementPredictor(model_type)
        self.sentiment = SentimentPredictor(model_type)
        self.volatility = VolatilityPredictor(model_type)

        LOGGER.info("MultiPredictor initialized with all forecast types")

    def predict_all(self, signal: Dict, horizon: str = "24h") -> Dict[str, Forecast]:
        """
        Generate all forecast types for a signal

        Returns:
            Dictionary mapping forecast type to Forecast
        """
        return {
            "movement": self.movement.predict_from_signal(signal, horizon),
            "sentiment": self.sentiment.predict_from_signal(signal, horizon),
            "volatility": self.volatility.predict_from_signal(signal, horizon)
        }

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        horizon: str = "24h",
        forecast_types: List[str] = None,
        max_forecasts: int = 1000
    ) -> Dict:
        """
        Process signals and generate all forecast types

        Args:
            input_path: Path to input signals JSONL
            output_path: Path to output forecasts JSONL
            horizon: Forecast horizon
            forecast_types: List of types to generate (default: all)
            max_forecasts: Maximum forecasts per type

        Returns:
            Processing statistics
        """
        from utils_io import ensure_parent_dir, safe_write_jsonl

        if forecast_types is None:
            forecast_types = ["movement", "sentiment", "volatility"]

        ensure_parent_dir(output_path)

        signals_processed = 0
        total_forecasts = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if total_forecasts >= max_forecasts * len(forecast_types):
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    signal = json.loads(line)
                except json.JSONDecodeError:
                    continue

                signals_processed += 1

                # Generate all forecast types
                all_forecasts = self.predict_all(signal, horizon)

                for ftype in forecast_types:
                    if ftype in all_forecasts:
                        forecast = all_forecasts[ftype]
                        safe_write_jsonl(output_path, forecast.to_dict())
                        total_forecasts += 1

        summary = {
            "signals_processed": signals_processed,
            "forecasts_generated": total_forecasts,
            "forecast_types": forecast_types,
            "forecasts_per_type": total_forecasts // len(forecast_types) if forecast_types else 0
        }

        LOGGER.info(f"Multi-forecast generation complete: {total_forecasts} forecasts from {signals_processed} signals")

        return summary


def main():
    """CLI interface for forecasting"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate forecasts from signals")
    parser.add_argument("--input", required=True, help="Input signals JSONL file")
    parser.add_argument("--output", required=True, help="Output forecasts JSONL")
    parser.add_argument("--horizon", default="24h", help="Forecast horizon")
    parser.add_argument("--type", default="movement", choices=["movement", "sentiment", "volatility", "all"])
    parser.add_argument("--model", default="simple", choices=["simple", "logistic", "random_forest"])
    parser.add_argument("--max-forecasts", type=int, default=1000, dest="max_forecasts")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create predictor
    if args.type == "all":
        predictor = MultiPredictor(model_type=args.model)
    elif args.type == "movement":
        predictor = MovementPredictor(model_type=args.model)
    elif args.type == "sentiment":
        predictor = SentimentPredictor(model_type=args.model)
    else:  # volatility
        predictor = VolatilityPredictor(model_type=args.model)

    # Process
    summary = predictor.process_jsonl(
        input_path=args.input,
        output_path=args.output,
        horizon=args.horizon,
        max_forecasts=args.max_forecasts
    )

    # Print summary
    print("\n" + "="*60)
    print("Forecast Generation Summary")
    print("="*60)
    print(f"Signals processed:    {summary['signals_processed']}")
    print(f"Forecasts generated:  {summary['forecasts_generated']}")
    if 'forecast_types' in summary:
        print(f"Forecast types:       {', '.join(summary['forecast_types'])}")
        print(f"Forecasts per type:   {summary['forecasts_per_type']}")
    print(f"Model type:           {args.model}")
    print(f"Horizon:              {args.horizon}")


if __name__ == "__main__":
    main()
