"""
Module: rinse_agent.py
Purpose: Execute the Reflect-Integrate-Normalize-Simulate-Evolve cycle for self-improvement
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
class Reflection:
    """Represents a reflection on system performance"""
    iteration: int
    timestamp: str
    observations: Dict[str, float]
    insights: List[str]
    reflection_note: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "observations": self.observations,
            "insights": self.insights,
            "reflection_note": self.reflection_note,
            "metadata": self.metadata
        }


@dataclass
class Adjustment:
    """Represents an adjustment to system parameters"""
    target: str  # What to adjust (e.g., "predictor_weights", "calibration_params")
    parameter: str  # Specific parameter name
    old_value: float
    new_value: float
    reason: str
    confidence: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "target": self.target,
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class RINSECycle:
    """Complete RINSE cycle output"""
    iteration: int
    timestamp: str
    reflection: Reflection
    adjustments: List[Adjustment]
    integrated_feedback: Dict[str, float]
    normalized_corrections: Dict[str, float]
    simulation_score: float
    evolution_applied: bool
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "reflection": self.reflection.to_dict(),
            "adjustments": [adj.to_dict() for adj in self.adjustments],
            "integrated_feedback": self.integrated_feedback,
            "normalized_corrections": self.normalized_corrections,
            "simulation_score": self.simulation_score,
            "evolution_applied": self.evolution_applied,
            "metadata": self.metadata
        }


class RINSEAgent:
    """
    RINSE Agent: Reflect-Integrate-Normalize-Simulate-Evolve

    The core self-improvement loop for Proto-liminal consciousness
    """

    def __init__(
        self,
        reflection_threshold: float = 0.1,
        adjustment_magnitude: float = 0.15,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize RINSE agent

        Args:
            reflection_threshold: Minimum metric change to trigger reflection
            adjustment_magnitude: Maximum adjustment per cycle (0-1)
            confidence_threshold: Minimum confidence to apply adjustment
        """
        self.reflection_threshold = reflection_threshold
        self.adjustment_magnitude = adjustment_magnitude
        self.confidence_threshold = confidence_threshold

        # History
        self.cycles: List[RINSECycle] = []
        self.iteration = 0

        # State
        self.baseline_metrics: Dict[str, float] = {}
        self.current_metrics: Dict[str, float] = {}
        self.accumulated_feedback: Dict[str, List[float]] = defaultdict(list)

        # Parameters being optimized
        self.parameters = {
            "signal_strength_weight": 0.5,
            "sentiment_weight": 0.3,
            "urgency_weight": 0.25,
            "relevance_weight": 0.35,
            "calibration_temperature": 1.0,
            "confidence_threshold": 0.3,
            "prediction_horizon_weight": 1.0
        }

        self.stats = {
            "cycles_completed": 0,
            "adjustments_made": 0,
            "adjustments_applied": 0
        }

        LOGGER.info("RINSEAgent initialized")

    def reflect(self, metrics: Dict[str, float]) -> Reflection:
        """
        Phase 1: Reflect - Observe and analyze system performance

        Args:
            metrics: Current performance metrics

        Returns:
            Reflection object
        """
        self.iteration += 1
        self.current_metrics = metrics

        # Calculate changes from baseline
        observations = {}
        insights = []

        if not self.baseline_metrics:
            # First reflection: establish baseline
            self.baseline_metrics = metrics.copy()
            observations = {"status": "baseline_established"}
            insights.append("Established performance baseline")
            reflection_note = "Initial reflection: Baseline metrics captured for future comparison."

        else:
            # Compare with baseline
            for metric_name, current_value in metrics.items():
                if metric_name in self.baseline_metrics:
                    baseline_value = self.baseline_metrics[metric_name]
                    change = current_value - baseline_value
                    pct_change = (change / baseline_value * 100) if baseline_value != 0 else 0

                    observations[f"{metric_name}_change"] = change
                    observations[f"{metric_name}_pct_change"] = pct_change

                    # Generate insights
                    if abs(pct_change) > self.reflection_threshold * 100:
                        if metric_name in ["brier_score", "log_score", "calibration_error"]:
                            # Lower is better for these metrics
                            if change < 0:
                                insights.append(f"✓ {metric_name} improved by {abs(pct_change):.1f}%")
                            else:
                                insights.append(f"⚠ {metric_name} degraded by {pct_change:.1f}%")
                        else:
                            # Higher is better for accuracy, precision, recall
                            if change > 0:
                                insights.append(f"✓ {metric_name} improved by {pct_change:.1f}%")
                            else:
                                insights.append(f"⚠ {metric_name} degraded by {abs(pct_change):.1f}%")

            # Generate reflection note
            if insights:
                reflection_note = f"Iteration {self.iteration}: Observed {len(insights)} significant changes. " + \
                                  "Performance drift detected, adjustments recommended."
            else:
                reflection_note = f"Iteration {self.iteration}: System performance stable. " + \
                                  "No significant deviations from baseline."

        reflection = Reflection(
            iteration=self.iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            observations=observations,
            insights=insights,
            reflection_note=reflection_note
        )

        LOGGER.info(f"Reflection complete: {len(insights)} insights generated")

        return reflection

    def integrate(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Phase 2: Integrate - Accumulate feedback over time

        Args:
            metrics: Current metrics

        Returns:
            Integrated feedback scores
        """
        # Add current metrics to accumulated feedback
        for metric_name, value in metrics.items():
            self.accumulated_feedback[metric_name].append(value)

        # Calculate integrated feedback (moving averages, trends)
        integrated = {}

        for metric_name, values in self.accumulated_feedback.items():
            if len(values) > 0:
                # Moving average (last 5 cycles)
                recent_values = values[-5:]
                integrated[f"{metric_name}_ma"] = float(np.mean(recent_values))

                # Trend (slope)
                if len(values) >= 2:
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    integrated[f"{metric_name}_trend"] = float(slope)

                # Volatility (standard deviation)
                if len(values) >= 2:
                    integrated[f"{metric_name}_volatility"] = float(np.std(values))

        LOGGER.info(f"Integration complete: {len(integrated)} feedback signals")

        return integrated

    def normalize(self, integrated_feedback: Dict[str, float]) -> Dict[str, float]:
        """
        Phase 3: Normalize - Scale and normalize corrections

        Args:
            integrated_feedback: Integrated feedback scores

        Returns:
            Normalized correction values
        """
        corrections = {}

        # Normalize trends to correction magnitudes
        for key, value in integrated_feedback.items():
            if "_trend" in key:
                metric_name = key.replace("_trend", "")

                # Clip to [-adjustment_magnitude, +adjustment_magnitude]
                normalized = np.clip(
                    value * 10,  # Amplify trend signal
                    -self.adjustment_magnitude,
                    self.adjustment_magnitude
                )

                corrections[metric_name] = float(normalized)

        LOGGER.info(f"Normalization complete: {len(corrections)} corrections")

        return corrections

    def simulate(
        self,
        corrections: Dict[str, float],
        current_params: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """
        Phase 4: Simulate - Test potential adjustments

        Args:
            corrections: Normalized corrections
            current_params: Current parameters

        Returns:
            (simulated_params, simulation_score)
        """
        simulated_params = current_params.copy()

        # Apply corrections to parameters
        for metric_name, correction in corrections.items():
            # Map metrics to parameters
            if "brier" in metric_name or "calibration" in metric_name:
                # Calibration issues
                param = "calibration_temperature"
                if param in simulated_params:
                    simulated_params[param] *= (1 + correction)

            elif "accuracy" in metric_name or "precision" in metric_name:
                # Prediction quality issues
                param = "confidence_threshold"
                if param in simulated_params:
                    simulated_params[param] *= (1 - correction * 0.5)

        # Score simulation (heuristic)
        # Better score = more corrections, higher confidence
        num_corrections = len(corrections)
        avg_correction_magnitude = np.mean([abs(c) for c in corrections.values()]) if corrections else 0

        simulation_score = min(1.0, num_corrections * 0.1 + avg_correction_magnitude)

        LOGGER.info(f"Simulation complete: score={simulation_score:.3f}")

        return simulated_params, float(simulation_score)

    def evolve(
        self,
        simulated_params: Dict[str, float],
        simulation_score: float,
        reflection: Reflection
    ) -> Tuple[List[Adjustment], bool]:
        """
        Phase 5: Evolve - Apply adjustments and evolve system

        Args:
            simulated_params: Simulated parameters
            simulation_score: Simulation quality score
            reflection: Current reflection

        Returns:
            (adjustments_list, applied_bool)
        """
        adjustments = []
        applied = False

        # Decide whether to apply based on confidence
        confidence = simulation_score

        if confidence < self.confidence_threshold:
            LOGGER.info(f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}, not applying")
            return adjustments, False

        # Generate adjustments
        for param_name, new_value in simulated_params.items():
            if param_name in self.parameters:
                old_value = self.parameters[param_name]

                if abs(new_value - old_value) > 1e-6:  # Meaningful change
                    # Determine reason from reflection
                    reason = "Performance optimization"
                    for insight in reflection.insights:
                        if "brier" in insight.lower() and "calibration" in param_name:
                            reason = f"Addressing calibration: {insight}"
                        elif "accuracy" in insight.lower() and "threshold" in param_name:
                            reason = f"Addressing accuracy: {insight}"

                    adjustment = Adjustment(
                        target="system_parameters",
                        parameter=param_name,
                        old_value=old_value,
                        new_value=new_value,
                        reason=reason,
                        confidence=confidence
                    )

                    adjustments.append(adjustment)

        # Apply adjustments
        if adjustments:
            for adjustment in adjustments:
                self.parameters[adjustment.parameter] = adjustment.new_value
                self.stats["adjustments_made"] += 1

            applied = True
            self.stats["adjustments_applied"] += len(adjustments)

            LOGGER.info(f"Evolution complete: {len(adjustments)} adjustments applied")
        else:
            LOGGER.info("Evolution complete: No adjustments needed")

        return adjustments, applied

    def run_cycle(self, metrics: Dict[str, float]) -> RINSECycle:
        """
        Run complete RINSE cycle

        Args:
            metrics: Current performance metrics

        Returns:
            RINSECycle object
        """
        LOGGER.info(f"Starting RINSE cycle {self.iteration + 1}")

        # Phase 1: Reflect
        reflection = self.reflect(metrics)

        # Phase 2: Integrate
        integrated_feedback = self.integrate(metrics)

        # Phase 3: Normalize
        normalized_corrections = self.normalize(integrated_feedback)

        # Phase 4: Simulate
        simulated_params, simulation_score = self.simulate(
            normalized_corrections,
            self.parameters
        )

        # Phase 5: Evolve
        adjustments, evolution_applied = self.evolve(
            simulated_params,
            simulation_score,
            reflection
        )

        # Create cycle record
        cycle = RINSECycle(
            iteration=self.iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reflection=reflection,
            adjustments=adjustments,
            integrated_feedback=integrated_feedback,
            normalized_corrections=normalized_corrections,
            simulation_score=simulation_score,
            evolution_applied=evolution_applied,
            metadata={
                "parameters": self.parameters.copy(),
                "baseline_metrics": self.baseline_metrics.copy(),
                "current_metrics": self.current_metrics.copy()
            }
        )

        self.cycles.append(cycle)
        self.stats["cycles_completed"] += 1

        LOGGER.info(f"RINSE cycle {self.iteration} complete: {len(adjustments)} adjustments, applied={evolution_applied}")

        return cycle

    def load_metrics_from_file(self, metrics_path: str) -> Dict[str, float]:
        """
        Load metrics from evaluation results file

        Args:
            metrics_path: Path to metrics JSONL

        Returns:
            Aggregated metrics dictionary
        """
        all_metrics = defaultdict(list)

        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)
                    metrics = result.get("metrics", {})

                    for metric_name, value in metrics.items():
                        if not np.isnan(value):
                            all_metrics[metric_name].append(value)

                except json.JSONDecodeError:
                    continue

        # Average across entities
        averaged = {}
        for metric_name, values in all_metrics.items():
            if values:
                averaged[metric_name] = float(np.mean(values))

        LOGGER.info(f"Loaded {len(averaged)} metrics from {metrics_path}")

        return averaged

    def save_cycle(self, cycle: RINSECycle, output_path: str):
        """
        Save RINSE cycle to file

        Args:
            cycle: RINSECycle object
            output_path: Path to output JSONL
        """
        from utils_io import ensure_parent_dir, safe_write_jsonl

        ensure_parent_dir(output_path)
        safe_write_jsonl(output_path, cycle.to_dict())

        LOGGER.info(f"Saved RINSE cycle {cycle.iteration} to {output_path}")

    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "cycles_completed": self.stats["cycles_completed"],
            "adjustments_made": self.stats["adjustments_made"],
            "adjustments_applied": self.stats["adjustments_applied"],
            "current_iteration": self.iteration,
            "parameters": self.parameters.copy()
        }

    def get_parameter(self, name: str) -> Optional[float]:
        """Get current parameter value"""
        return self.parameters.get(name)

    def set_parameter(self, name: str, value: float):
        """Set parameter value"""
        self.parameters[name] = value
        LOGGER.info(f"Parameter {name} set to {value}")


def main():
    """CLI interface for RINSE agent"""
    import argparse

    parser = argparse.ArgumentParser(description="Run RINSE self-improvement cycle")
    parser.add_argument("--metrics", required=True, help="Input metrics JSONL from feedback_tracker")
    parser.add_argument("--output", required=True, help="Output RINSE cycle JSONL")
    parser.add_argument("--threshold", type=float, default=0.1, help="Reflection threshold")
    parser.add_argument("--magnitude", type=float, default=0.15, help="Adjustment magnitude")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create agent
    agent = RINSEAgent(
        reflection_threshold=args.threshold,
        adjustment_magnitude=args.magnitude
    )

    # Load metrics
    metrics = agent.load_metrics_from_file(args.metrics)

    if not metrics:
        LOGGER.error("No metrics loaded")
        return 1

    # Run cycle
    cycle = agent.run_cycle(metrics)

    # Save cycle
    agent.save_cycle(cycle, args.output)

    # Print summary
    print("\n" + "="*60)
    print(f"RINSE Cycle {cycle.iteration} Summary")
    print("="*60)
    print(f"Timestamp:           {cycle.timestamp}")
    print(f"Insights:            {len(cycle.reflection.insights)}")
    print(f"Adjustments:         {len(cycle.adjustments)}")
    print(f"Simulation score:    {cycle.simulation_score:.3f}")
    print(f"Evolution applied:   {cycle.evolution_applied}")

    if cycle.reflection.insights:
        print("\nKey Insights:")
        for insight in cycle.reflection.insights[:5]:
            print(f"  • {insight}")

    if cycle.adjustments:
        print("\nAdjustments Made:")
        for adj in cycle.adjustments[:5]:
            print(f"  • {adj.parameter}: {adj.old_value:.4f} → {adj.new_value:.4f}")

    print(f"\nReflection: {cycle.reflection.reflection_note}")

    stats = agent.get_stats()
    print(f"\nTotal cycles: {stats['cycles_completed']}")
    print(f"Total adjustments: {stats['adjustments_applied']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
