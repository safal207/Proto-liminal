"""
Module: calibrator.py
Purpose: Adjust forecast probabilities to align with observed calibration metrics
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibratedForecast:
    """Forecast with calibrated probability"""
    entity: str
    original_probability: float
    calibrated_probability: float
    calibration_score: float  # Calibration adjustment factor
    forecast_type: str
    horizon: str
    timestamp: str
    calibration_method: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        # Ensure probabilities are in [0, 1]
        self.original_probability = max(0.0, min(1.0, self.original_probability))
        self.calibrated_probability = max(0.0, min(1.0, self.calibrated_probability))

    def to_dict(self) -> Dict:
        return {
            "entity": self.entity,
            "original_probability": self.original_probability,
            "calibrated_probability": self.calibrated_probability,
            "calibration_score": self.calibration_score,
            "forecast_type": self.forecast_type,
            "horizon": self.horizon,
            "timestamp": self.timestamp,
            "calibration_method": self.calibration_method,
            "metadata": self.metadata
        }


class Calibrator:
    """Calibrate forecast probabilities based on historical performance"""

    def __init__(self, method: str = "platt"):
        """
        Initialize calibrator

        Args:
            method: Calibration method ("platt", "isotonic", "temperature", "beta")
        """
        self.method = method
        self.is_fitted = False

        # Calibration models
        self.platt_model = None
        self.isotonic_model = None
        self.temperature = 1.0
        self.beta_params = {"a": 1.0, "b": 1.0}

        # Historical data for fitting
        self.train_probabilities = []
        self.train_outcomes = []

        self.stats = {
            "forecasts_calibrated": 0,
            "calibrations_fitted": 0
        }

        LOGGER.info(f"Calibrator initialized with method: {method}")

    def fit(self, probabilities: List[float], outcomes: List[float]):
        """
        Fit calibration model on historical data

        Args:
            probabilities: List of predicted probabilities
            outcomes: List of actual outcomes (0 or 1)
        """
        if len(probabilities) != len(outcomes):
            raise ValueError("Probabilities and outcomes must have same length")

        if len(probabilities) < 2:
            LOGGER.warning("Insufficient data for calibration fitting")
            return

        probs_arr = np.array(probabilities).reshape(-1, 1)
        outcomes_arr = np.array(outcomes)

        if self.method == "platt":
            # Platt scaling: logistic regression
            self.platt_model = LogisticRegression()
            self.platt_model.fit(probs_arr, outcomes_arr)

        elif self.method == "isotonic":
            # Isotonic regression
            self.isotonic_model = IsotonicRegression(out_of_bounds="clip")
            self.isotonic_model.fit(probabilities, outcomes)

        elif self.method == "temperature":
            # Temperature scaling
            self.temperature = self._find_optimal_temperature(probabilities, outcomes)

        elif self.method == "beta":
            # Beta calibration (simplified version)
            self.beta_params = self._fit_beta_calibration(probabilities, outcomes)

        self.is_fitted = True
        self.stats["calibrations_fitted"] += 1

        # Store training data
        self.train_probabilities = probabilities
        self.train_outcomes = outcomes

        LOGGER.info(f"Calibration fitted on {len(probabilities)} samples using {self.method}")

    def _find_optimal_temperature(
        self,
        probabilities: List[float],
        outcomes: List[float],
        T_range: Tuple[float, float] = (0.1, 10.0),
        n_steps: int = 100
    ) -> float:
        """
        Find optimal temperature via grid search

        Args:
            probabilities: Predicted probabilities
            outcomes: Actual outcomes
            T_range: (min_temp, max_temp) to search
            n_steps: Number of steps in grid

        Returns:
            Optimal temperature
        """
        probs_arr = np.array(probabilities)
        outcomes_arr = np.array(outcomes)

        temperatures = np.linspace(T_range[0], T_range[1], n_steps)
        best_T = 1.0
        best_loss = float('inf')

        for T in temperatures:
            # Apply temperature scaling
            scaled_probs = self._apply_temperature(probs_arr, T)

            # Calculate negative log likelihood
            scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                outcomes_arr * np.log(scaled_probs) +
                (1 - outcomes_arr) * np.log(1 - scaled_probs)
            )

            if loss < best_loss:
                best_loss = loss
                best_T = T

        return best_T

    def _apply_temperature(self, probabilities: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to probabilities

        Args:
            probabilities: Array of probabilities
            temperature: Temperature parameter

        Returns:
            Scaled probabilities
        """
        # Convert probabilities to logits
        logits = np.log(probabilities / (1 - probabilities + 1e-7) + 1e-7)

        # Scale logits
        scaled_logits = logits / temperature

        # Convert back to probabilities
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))

        return scaled_probs

    def _fit_beta_calibration(
        self,
        probabilities: List[float],
        outcomes: List[float]
    ) -> Dict[str, float]:
        """
        Fit beta calibration parameters (simplified)

        Args:
            probabilities: Predicted probabilities
            outcomes: Actual outcomes

        Returns:
            Dictionary with 'a' and 'b' parameters
        """
        # Simplified beta calibration
        # In full implementation, would use optimization

        probs_arr = np.array(probabilities)
        outcomes_arr = np.array(outcomes)

        # Simple estimation: use mean and variance
        mean_outcome = outcomes_arr.mean()
        mean_prob = probs_arr.mean()

        # Adjust parameters based on over/under-confidence
        if mean_prob > mean_outcome:
            # Overconfident: shift probabilities down
            a = 0.9
            b = 1.1
        elif mean_prob < mean_outcome:
            # Underconfident: shift probabilities up
            a = 1.1
            b = 0.9
        else:
            # Well calibrated
            a = 1.0
            b = 1.0

        return {"a": a, "b": b}

    def calibrate_probability(self, probability: float) -> float:
        """
        Calibrate a single probability

        Args:
            probability: Original probability

        Returns:
            Calibrated probability
        """
        if not self.is_fitted:
            LOGGER.warning("Calibrator not fitted, returning original probability")
            return probability

        prob = max(1e-7, min(1 - 1e-7, probability))  # Clip to valid range

        if self.method == "platt":
            if self.platt_model:
                calibrated = self.platt_model.predict_proba([[prob]])[0][1]
            else:
                calibrated = prob

        elif self.method == "isotonic":
            if self.isotonic_model:
                calibrated = self.isotonic_model.predict([prob])[0]
            else:
                calibrated = prob

        elif self.method == "temperature":
            calibrated = self._apply_temperature(np.array([prob]), self.temperature)[0]

        elif self.method == "beta":
            # Beta calibration: p_cal = a * p^b
            a = self.beta_params["a"]
            b = self.beta_params["b"]
            calibrated = a * (prob ** b)

        else:
            calibrated = prob

        return float(np.clip(calibrated, 0.0, 1.0))

    def calibrate_forecast(self, forecast: Dict) -> CalibratedForecast:
        """
        Calibrate a forecast

        Args:
            forecast: Forecast dictionary

        Returns:
            CalibratedForecast object
        """
        original_prob = forecast.get("probability", 0.5)
        calibrated_prob = self.calibrate_probability(original_prob)

        calibration_score = calibrated_prob / original_prob if original_prob > 0 else 1.0

        calibrated = CalibratedForecast(
            entity=forecast.get("entity", "unknown"),
            original_probability=original_prob,
            calibrated_probability=calibrated_prob,
            calibration_score=calibration_score,
            forecast_type=forecast.get("forecast_type", "movement"),
            horizon=forecast.get("horizon", "24h"),
            timestamp=datetime.now(timezone.utc).isoformat(),
            calibration_method=self.method,
            metadata={
                "original_confidence_interval": forecast.get("confidence_interval"),
                "model_confidence": forecast.get("model_confidence")
            }
        )

        self.stats["forecasts_calibrated"] += 1

        return calibrated

    def calibrate_batch(self, forecasts: List[Dict]) -> List[CalibratedForecast]:
        """
        Calibrate multiple forecasts

        Args:
            forecasts: List of forecast dictionaries

        Returns:
            List of CalibratedForecast objects
        """
        calibrated_forecasts = []

        for forecast in forecasts:
            try:
                calibrated = self.calibrate_forecast(forecast)
                calibrated_forecasts.append(calibrated)
            except Exception as exc:
                LOGGER.error(f"Error calibrating forecast: {exc}")

        LOGGER.info(f"Calibrated {len(calibrated_forecasts)} forecasts")

        return calibrated_forecasts

    def process_jsonl(
        self,
        input_path: str,
        output_path: str,
        max_forecasts: int = 10000
    ) -> Dict:
        """
        Process forecasts from JSONL and calibrate

        Args:
            input_path: Path to input forecasts JSONL
            output_path: Path to output calibrated JSONL
            max_forecasts: Maximum forecasts to process

        Returns:
            Processing statistics
        """
        from utils_io import ensure_parent_dir, safe_write_jsonl

        if not self.is_fitted:
            LOGGER.warning("Calibrator not fitted, probabilities will not be adjusted")

        ensure_parent_dir(output_path)

        forecasts_processed = 0
        calibrated_written = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if forecasts_processed >= max_forecasts:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    forecast = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning(f"Line {line_num}: Invalid JSON")
                    continue

                forecasts_processed += 1

                # Calibrate forecast
                try:
                    calibrated = self.calibrate_forecast(forecast)
                    safe_write_jsonl(output_path, calibrated.to_dict())
                    calibrated_written += 1
                except Exception as exc:
                    LOGGER.error(f"Line {line_num}: Calibration error: {exc}")

        summary = {
            "forecasts_processed": forecasts_processed,
            "calibrated_written": calibrated_written,
            "calibration_method": self.method,
            "is_fitted": self.is_fitted
        }

        LOGGER.info(f"Calibration complete: {calibrated_written} forecasts processed")

        return summary

    def get_stats(self) -> Dict:
        """Get calibrator statistics"""
        return {
            "forecasts_calibrated": self.stats["forecasts_calibrated"],
            "calibrations_fitted": self.stats["calibrations_fitted"],
            "is_fitted": self.is_fitted,
            "method": self.method,
            "temperature": self.temperature if self.method == "temperature" else None,
            "beta_params": self.beta_params if self.method == "beta" else None
        }


def main():
    """CLI interface for calibration"""
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate forecast probabilities")
    parser.add_argument("--input", required=True, help="Input forecasts JSONL")
    parser.add_argument("--output", required=True, help="Output calibrated JSONL")
    parser.add_argument("--method", default="platt", choices=["platt", "isotonic", "temperature", "beta"])
    parser.add_argument("--fit-data", help="Historical data for fitting calibration (optional)", dest="fit_data")
    parser.add_argument("--max-forecasts", type=int, default=10000, dest="max_forecasts")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create calibrator
    calibrator = Calibrator(method=args.method)

    # Fit if historical data provided
    if args.fit_data:
        LOGGER.info(f"Loading fitting data from {args.fit_data}")
        # TODO: Implement loading historical data for fitting
        LOGGER.warning("Fitting from file not yet implemented")

    # Process forecasts
    summary = calibrator.process_jsonl(
        input_path=args.input,
        output_path=args.output,
        max_forecasts=args.max_forecasts
    )

    # Print summary
    print("\n" + "="*60)
    print("Calibration Summary")
    print("="*60)
    print(f"Forecasts processed:  {summary['forecasts_processed']}")
    print(f"Calibrated written:   {summary['calibrated_written']}")
    print(f"Method:               {summary['calibration_method']}")
    print(f"Fitted:               {summary['is_fitted']}")

    stats = calibrator.get_stats()
    if stats.get("temperature"):
        print(f"Temperature:          {stats['temperature']:.4f}")
    if stats.get("beta_params"):
        print(f"Beta params:          a={stats['beta_params']['a']:.4f}, b={stats['beta_params']['b']:.4f}")


if __name__ == "__main__":
    main()
