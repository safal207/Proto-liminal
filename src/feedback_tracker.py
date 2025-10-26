"""
Module: feedback_tracker.py
Purpose: Record outcomes and compute evaluation metrics over time
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Outcome:
    """Represents an actual outcome for a forecast"""
    entity: str
    outcome: float  # Actual value (0 or 1 for binary, continuous for regression)
    timestamp: str
    forecast_id: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "forecast_id": self.forecast_id,
            "metadata": self.metadata
        }


@dataclass
class EvaluationResult:
    """Results of forecast evaluation"""
    entity: str
    forecast_type: str
    metrics: Dict[str, float]
    sample_size: int
    timestamp: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "forecast_type": self.forecast_type,
            "metrics": self.metrics,
            "sample_size": self.sample_size,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class FeedbackTracker:
    """Track forecast outcomes and compute evaluation metrics"""

    def __init__(self):
        """Initialize feedback tracker"""
        self.forecasts: Dict[str, Dict] = {}  # forecast_id -> forecast
        self.outcomes: Dict[str, Outcome] = {}  # forecast_id -> outcome
        self.metrics_history: List[EvaluationResult] = []

        self.stats = {
            "forecasts_tracked": 0,
            "outcomes_recorded": 0,
            "evaluations_run": 0
        }

        LOGGER.info("FeedbackTracker initialized")

    def add_forecast(self, forecast: Dict, forecast_id: Optional[str] = None) -> str:
        """
        Add forecast for tracking

        Args:
            forecast: Forecast dictionary
            forecast_id: Optional ID (auto-generated if not provided)

        Returns:
            Forecast ID
        """
        if forecast_id is None:
            # Generate ID from entity + timestamp
            entity = forecast.get("entity", "unknown")
            timestamp = forecast.get("timestamp", datetime.now(timezone.utc).isoformat())
            forecast_id = f"{entity}_{timestamp}".replace(" ", "_").replace(":", "")

        self.forecasts[forecast_id] = forecast
        self.stats["forecasts_tracked"] += 1

        LOGGER.debug(f"Added forecast {forecast_id} for {forecast.get('entity')}")

        return forecast_id

    def record_outcome(self, forecast_id: str, outcome: float, timestamp: Optional[str] = None) -> Outcome:
        """
        Record actual outcome for a forecast

        Args:
            forecast_id: ID of the forecast
            outcome: Actual outcome value (0/1 for binary)
            timestamp: Timestamp of outcome (defaults to now)

        Returns:
            Outcome object
        """
        if forecast_id not in self.forecasts:
            raise ValueError(f"Forecast {forecast_id} not found")

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        forecast = self.forecasts[forecast_id]

        outcome_obj = Outcome(
            entity=forecast.get("entity", "unknown"),
            outcome=outcome,
            timestamp=timestamp,
            forecast_id=forecast_id
        )

        self.outcomes[forecast_id] = outcome_obj
        self.stats["outcomes_recorded"] += 1

        LOGGER.debug(f"Recorded outcome for {forecast_id}: {outcome}")

        return outcome_obj

    def calculate_brier_score(self, forecasts: List[Dict], outcomes: List[float]) -> float:
        """
        Calculate Brier Score for probabilistic forecasts

        Lower is better. Perfect score = 0, worst = 1

        Args:
            forecasts: List of forecast dictionaries
            outcomes: List of actual outcomes (0 or 1)

        Returns:
            Brier score
        """
        if not forecasts or not outcomes:
            return float('nan')

        if len(forecasts) != len(outcomes):
            raise ValueError("Forecasts and outcomes must have same length")

        probabilities = np.array([f.get("probability", 0.5) for f in forecasts])
        outcomes_arr = np.array(outcomes)

        # Brier score = mean((probability - outcome)^2)
        brier = np.mean((probabilities - outcomes_arr) ** 2)

        return float(brier)

    def calculate_log_score(self, forecasts: List[Dict], outcomes: List[float]) -> float:
        """
        Calculate logarithmic (log loss) score

        Lower is better

        Args:
            forecasts: List of forecast dictionaries
            outcomes: List of actual outcomes (0 or 1)

        Returns:
            Log score
        """
        if not forecasts or not outcomes:
            return float('nan')

        probabilities = np.array([f.get("probability", 0.5) for f in forecasts])
        outcomes_arr = np.array(outcomes)

        # Clip probabilities to avoid log(0)
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

        # Log loss = -mean(outcome*log(p) + (1-outcome)*log(1-p))
        log_loss = -np.mean(
            outcomes_arr * np.log(probabilities) +
            (1 - outcomes_arr) * np.log(1 - probabilities)
        )

        return float(log_loss)

    def calculate_calibration(
        self,
        forecasts: List[Dict],
        outcomes: List[float],
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate calibration curve

        Args:
            forecasts: List of forecast dictionaries
            outcomes: List of actual outcomes
            n_bins: Number of bins for calibration

        Returns:
            (bin_centers, observed_frequencies, bin_counts)
        """
        if not forecasts or not outcomes:
            return np.array([]), np.array([]), np.array([])

        probabilities = np.array([f.get("probability", 0.5) for f in forecasts])
        outcomes_arr = np.array(outcomes)

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Bin the probabilities
        bin_indices = np.digitize(probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate observed frequency in each bin
        observed_freq = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                observed_freq[i] = outcomes_arr[mask].mean()
                bin_counts[i] = mask.sum()

        return bin_centers, observed_freq, bin_counts

    def calculate_calibration_error(
        self,
        forecasts: List[Dict],
        outcomes: List[float],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE)

        Args:
            forecasts: List of forecast dictionaries
            outcomes: List of actual outcomes
            n_bins: Number of bins

        Returns:
            ECE score (lower is better)
        """
        bin_centers, observed_freq, bin_counts = self.calculate_calibration(
            forecasts, outcomes, n_bins
        )

        if len(bin_centers) == 0:
            return float('nan')

        # Weighted average of |predicted - observed|
        total_count = bin_counts.sum()
        if total_count == 0:
            return float('nan')

        ece = np.sum(
            bin_counts * np.abs(bin_centers - observed_freq)
        ) / total_count

        return float(ece)

    def calculate_precision_recall(
        self,
        forecasts: List[Dict],
        outcomes: List[float],
        threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, F1 score

        Args:
            forecasts: List of forecast dictionaries
            outcomes: List of actual outcomes
            threshold: Probability threshold for positive prediction

        Returns:
            (precision, recall, f1_score)
        """
        if not forecasts or not outcomes:
            return (float('nan'), float('nan'), float('nan'))

        probabilities = np.array([f.get("probability", 0.5) for f in forecasts])
        outcomes_arr = np.array(outcomes)

        # Convert probabilities to binary predictions
        predictions = (probabilities >= threshold).astype(int)

        # Calculate metrics
        tp = np.sum((predictions == 1) & (outcomes_arr == 1))
        fp = np.sum((predictions == 1) & (outcomes_arr == 0))
        fn = np.sum((predictions == 0) & (outcomes_arr == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return (float(precision), float(recall), float(f1))

    def calculate_accuracy(self, forecasts: List[Dict], outcomes: List[float], threshold: float = 0.5) -> float:
        """
        Calculate classification accuracy

        Args:
            forecasts: List of forecast dictionaries
            outcomes: List of actual outcomes
            threshold: Probability threshold

        Returns:
            Accuracy score (0-1)
        """
        if not forecasts or not outcomes:
            return float('nan')

        probabilities = np.array([f.get("probability", 0.5) for f in forecasts])
        outcomes_arr = np.array(outcomes)

        predictions = (probabilities >= threshold).astype(int)
        accuracy = np.mean(predictions == outcomes_arr)

        return float(accuracy)

    def evaluate_entity(
        self,
        entity: str,
        forecast_type: str = "movement"
    ) -> Optional[EvaluationResult]:
        """
        Evaluate all forecasts for a specific entity

        Args:
            entity: Entity name
            forecast_type: Type of forecast to evaluate

        Returns:
            EvaluationResult or None if insufficient data
        """
        # Collect forecasts and outcomes for this entity
        entity_forecasts = []
        entity_outcomes = []

        for forecast_id, forecast in self.forecasts.items():
            if forecast.get("entity") == entity and forecast.get("forecast_type") == forecast_type:
                if forecast_id in self.outcomes:
                    entity_forecasts.append(forecast)
                    entity_outcomes.append(self.outcomes[forecast_id].outcome)

        if len(entity_forecasts) < 2:
            LOGGER.warning(f"Insufficient data for {entity} ({len(entity_forecasts)} samples)")
            return None

        # Calculate metrics
        metrics = {}

        try:
            metrics["brier_score"] = self.calculate_brier_score(entity_forecasts, entity_outcomes)
            metrics["log_score"] = self.calculate_log_score(entity_forecasts, entity_outcomes)
            metrics["calibration_error"] = self.calculate_calibration_error(entity_forecasts, entity_outcomes)

            precision, recall, f1 = self.calculate_precision_recall(entity_forecasts, entity_outcomes)
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1_score"] = f1

            metrics["accuracy"] = self.calculate_accuracy(entity_forecasts, entity_outcomes)

        except Exception as exc:
            LOGGER.error(f"Error calculating metrics for {entity}: {exc}")
            return None

        result = EvaluationResult(
            entity=entity,
            forecast_type=forecast_type,
            metrics=metrics,
            sample_size=len(entity_forecasts),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        self.metrics_history.append(result)
        self.stats["evaluations_run"] += 1

        return result

    def evaluate_all(self, forecast_type: str = "movement") -> List[EvaluationResult]:
        """
        Evaluate all entities

        Args:
            forecast_type: Type of forecast to evaluate

        Returns:
            List of EvaluationResult objects
        """
        # Get unique entities
        entities = set(f.get("entity") for f in self.forecasts.values())

        results = []
        for entity in entities:
            result = self.evaluate_entity(entity, forecast_type)
            if result:
                results.append(result)

        LOGGER.info(f"Evaluated {len(results)} entities")

        return results

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}

        # Aggregate metrics across all entities
        all_metrics = defaultdict(list)

        for result in self.metrics_history:
            for metric_name, value in result.metrics.items():
                if not np.isnan(value):
                    all_metrics[metric_name].append(value)

        # Calculate averages
        summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary[f"avg_{metric_name}"] = float(np.mean(values))
                summary[f"std_{metric_name}"] = float(np.std(values))
                summary[f"min_{metric_name}"] = float(np.min(values))
                summary[f"max_{metric_name}"] = float(np.max(values))

        summary["total_forecasts"] = self.stats["forecasts_tracked"]
        summary["total_outcomes"] = self.stats["outcomes_recorded"]
        summary["total_evaluations"] = self.stats["evaluations_run"]
        summary["entities_evaluated"] = len(set(r.entity for r in self.metrics_history))

        return summary

    def save_results(self, output_path: str):
        """
        Save evaluation results to JSONL

        Args:
            output_path: Path to output file
        """
        from utils_io import ensure_parent_dir, safe_write_jsonl

        ensure_parent_dir(output_path)

        for result in self.metrics_history:
            safe_write_jsonl(output_path, result.to_dict())

        LOGGER.info(f"Saved {len(self.metrics_history)} evaluation results to {output_path}")

    def load_forecasts_from_jsonl(self, forecasts_path: str, max_forecasts: int = 10000):
        """
        Load forecasts from JSONL file

        Args:
            forecasts_path: Path to forecasts JSONL
            max_forecasts: Maximum forecasts to load
        """
        loaded = 0

        with open(forecasts_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if loaded >= max_forecasts:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    forecast = json.loads(line)
                    self.add_forecast(forecast)
                    loaded += 1
                except json.JSONDecodeError:
                    LOGGER.warning(f"Line {line_num}: Invalid JSON")

        LOGGER.info(f"Loaded {loaded} forecasts from {forecasts_path}")


def main():
    """CLI interface for feedback tracking"""
    import argparse

    parser = argparse.ArgumentParser(description="Track forecast outcomes and metrics")
    parser.add_argument("--forecasts", required=True, help="Input forecasts JSONL")
    parser.add_argument("--outcomes", help="Input outcomes JSONL (optional)")
    parser.add_argument("--output", required=True, help="Output evaluation results JSONL")
    parser.add_argument("--forecast-type", default="movement", dest="forecast_type")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create tracker
    tracker = FeedbackTracker()

    # Load forecasts
    tracker.load_forecasts_from_jsonl(args.forecasts)

    # Load outcomes if provided
    if args.outcomes:
        # TODO: Implement outcome loading
        LOGGER.info(f"Outcome loading not yet implemented")

    # Evaluate
    results = tracker.evaluate_all(forecast_type=args.forecast_type)

    # Save results
    tracker.save_results(args.output)

    # Print summary
    summary = tracker.get_summary_stats()

    print("\n" + "="*60)
    print("Feedback Tracking Summary")
    print("="*60)
    print(f"Total forecasts:      {summary.get('total_forecasts', 0)}")
    print(f"Total outcomes:       {summary.get('total_outcomes', 0)}")
    print(f"Entities evaluated:   {summary.get('entities_evaluated', 0)}")
    print(f"Evaluations run:      {summary.get('total_evaluations', 0)}")

    if "avg_brier_score" in summary:
        print(f"\nAverage Metrics:")
        print(f"  Brier Score:        {summary['avg_brier_score']:.4f}")
        print(f"  Log Score:          {summary['avg_log_score']:.4f}")
        print(f"  Calibration Error:  {summary['avg_calibration_error']:.4f}")
        print(f"  Accuracy:           {summary['avg_accuracy']:.4f}")
        print(f"  Precision:          {summary['avg_precision']:.4f}")
        print(f"  Recall:             {summary['avg_recall']:.4f}")
        print(f"  F1 Score:           {summary['avg_f1_score']:.4f}")


if __name__ == "__main__":
    main()
