#!/usr/bin/env python3
"""
Module: learning_pipeline.py
Purpose: Automated learning pipeline - connects all components into continuous learning loop
Part of LIMINAL ProtoConsciousness MVP - Phase 1, Week 1 Final Integration

Philosophy:
"True intelligence emerges from the continuous cycle of prediction,
observation, reflection, and adaptation. This pipeline embodies that cycle."

Pipeline Flow:
1. Load forecasts (from predictor output)
2. Collect outcomes (OutcomeCollector + MarketDataAPI)
3. Calculate metrics (FeedbackTracker)
4. Reflect and adapt (RINSEAgent)
5. Persist state (SQLite)
6. Repeat

This is the heart of Proto-liminal's self-improvement loop.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class LearningPipeline:
    """
    Automated learning pipeline for continuous self-improvement

    Integrates:
    - OutcomeCollector: Get real market outcomes
    - FeedbackTracker: Calculate performance metrics
    - RINSEAgent: Reflect and adapt based on performance
    """

    def __init__(
        self,
        forecasts_path: str,
        outcomes_path: str,
        metrics_path: str,
        db_path: Optional[str] = None,
        cache_dir: str = "data/market_cache"
    ):
        """
        Initialize learning pipeline

        Args:
            forecasts_path: Path to forecasts JSONL
            outcomes_path: Path to outcomes JSONL (will be created/appended)
            metrics_path: Path to metrics JSONL (for FeedbackTracker)
            db_path: Optional SQLite database for RINSE persistence
            cache_dir: Market data cache directory
        """
        self.forecasts_path = Path(forecasts_path)
        self.outcomes_path = Path(outcomes_path)
        self.metrics_path = Path(metrics_path)
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)

        # Components (lazy initialization)
        self.outcome_collector = None
        self.feedback_tracker = None
        self.rinse_agent = None

        # Statistics
        self.stats = {
            "cycles_run": 0,
            "forecasts_processed": 0,
            "outcomes_collected": 0,
            "rinse_iterations": 0,
            "last_run": None
        }

        LOGGER.info("LearningPipeline initialized")

    def _initialize_components(self):
        """Lazy initialization of components"""
        if self.outcome_collector is None:
            from outcome_collector import OutcomeCollector

            self.outcome_collector = OutcomeCollector(
                forecasts_path=str(self.forecasts_path),
                outcomes_path=str(self.outcomes_path),
                cache_dir=str(self.cache_dir)
            )
            LOGGER.info("OutcomeCollector initialized")

        if self.feedback_tracker is None:
            from feedback_tracker import FeedbackTracker

            self.feedback_tracker = FeedbackTracker()
            LOGGER.info("FeedbackTracker initialized")

        if self.rinse_agent is None:
            from rinse_agent import RINSEAgent

            self.rinse_agent = RINSEAgent(
                reflection_threshold=0.1,
                adjustment_magnitude=0.15,
                confidence_threshold=0.6,
                db_path=self.db_path
            )
            LOGGER.info("RINSEAgent initialized")

    def run_cycle(self) -> Dict:
        """
        Run one complete learning cycle

        Returns:
            Cycle results dictionary
        """
        self._initialize_components()

        cycle_start = datetime.now(timezone.utc)
        LOGGER.info(f"Starting learning cycle at {cycle_start.isoformat()}")

        results = {
            "cycle_number": self.stats["cycles_run"] + 1,
            "timestamp": cycle_start.isoformat(),
            "outcomes_collected": 0,
            "metrics_calculated": False,
            "rinse_executed": False,
            "errors": []
        }

        try:
            # Step 1: Load forecasts
            LOGGER.info("Step 1: Loading forecasts...")
            self.outcome_collector.load_forecasts()
            pending = len(self.outcome_collector.pending_forecasts)
            LOGGER.info(f"  Loaded {pending} pending forecasts")

            # Step 2: Collect outcomes for ready forecasts
            LOGGER.info("Step 2: Collecting outcomes...")
            outcomes = self.outcome_collector.collect_all_ready()
            results["outcomes_collected"] = len(outcomes)
            self.stats["outcomes_collected"] += len(outcomes)
            LOGGER.info(f"  Collected {len(outcomes)} outcomes")

            if not outcomes:
                LOGGER.info("  No outcomes ready yet, skipping metrics calculation")
                return results

            # Save outcomes
            self.outcome_collector.save_outcomes()
            LOGGER.info(f"  Saved outcomes to {self.outcomes_path}")

            # Step 3: Feed outcomes into FeedbackTracker
            LOGGER.info("Step 3: Calculating performance metrics...")

            for outcome in outcomes:
                forecast_id = outcome['forecast_id']

                # Create forecast record
                forecast = {
                    "entity": outcome['entity'],
                    "forecast_type": outcome['metadata']['forecast_type'],
                    "probability": outcome['forecast_probability'],
                    "timestamp": outcome['timestamp']
                }

                self.feedback_tracker.add_forecast(forecast, forecast_id=forecast_id)
                self.feedback_tracker.record_outcome(forecast_id, outcome['outcome'])

            # Evaluate all entities
            eval_results = self.feedback_tracker.evaluate_all()
            LOGGER.info(f"  Evaluated {len(eval_results)} entities")

            # Save metrics
            self.feedback_tracker.save_results(str(self.metrics_path))
            LOGGER.info(f"  Saved metrics to {self.metrics_path}")

            results["metrics_calculated"] = True

            # Step 4: Run RINSE cycle
            LOGGER.info("Step 4: Running RINSE self-improvement cycle...")

            # Get aggregated metrics
            summary = self.feedback_tracker.get_summary_stats()

            if summary:
                # Extract key metrics for RINSE
                metrics = {
                    "accuracy": summary.get("avg_accuracy", 0.0),
                    "brier_score": summary.get("avg_brier_score", 0.0),
                    "log_score": summary.get("avg_log_score", 0.0),
                    "calibration_error": summary.get("avg_calibration_error", 0.0),
                    "precision": summary.get("avg_precision", 0.0),
                    "recall": summary.get("avg_recall", 0.0),
                    "f1_score": summary.get("avg_f1_score", 0.0)
                }

                # Run RINSE cycle
                rinse_cycle = self.rinse_agent.run_cycle(metrics)

                LOGGER.info(f"  RINSE cycle {rinse_cycle.iteration} complete:")
                LOGGER.info(f"    Insights: {len(rinse_cycle.reflection.insights)}")
                LOGGER.info(f"    Adjustments: {len(rinse_cycle.adjustments)}")
                LOGGER.info(f"    Evolution applied: {rinse_cycle.evolution_applied}")

                results["rinse_executed"] = True
                results["rinse_iteration"] = rinse_cycle.iteration
                results["rinse_adjustments"] = len(rinse_cycle.adjustments)
                results["evolution_applied"] = rinse_cycle.evolution_applied

                self.stats["rinse_iterations"] += 1

                # Log insights
                if rinse_cycle.reflection.insights:
                    LOGGER.info("  Key insights:")
                    for insight in rinse_cycle.reflection.insights[:3]:
                        LOGGER.info(f"    • {insight}")

            else:
                LOGGER.warning("  No metrics available for RINSE")

        except Exception as e:
            error_msg = f"Error in learning cycle: {e}"
            LOGGER.error(error_msg)
            results["errors"].append(error_msg)

        finally:
            # Update statistics
            self.stats["cycles_run"] += 1
            self.stats["last_run"] = cycle_start.isoformat()

            cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
            results["duration_seconds"] = cycle_duration

            LOGGER.info(f"Learning cycle completed in {cycle_duration:.1f}s")

        return results

    def run_continuous(
        self,
        interval_hours: int = 24,
        max_cycles: Optional[int] = None
    ):
        """
        Run continuous learning loop

        Args:
            interval_hours: Hours between cycles
            max_cycles: Maximum cycles to run (None = infinite)
        """
        LOGGER.info(f"Starting continuous learning loop (interval: {interval_hours}h)")

        cycles_completed = 0

        try:
            while True:
                # Run cycle
                results = self.run_cycle()

                cycles_completed += 1

                # Print summary
                print(f"\n{'='*60}")
                print(f"Cycle {results['cycle_number']} Summary")
                print(f"{'='*60}")
                print(f"Outcomes collected:  {results['outcomes_collected']}")
                print(f"Metrics calculated:  {results['metrics_calculated']}")
                print(f"RINSE executed:      {results['rinse_executed']}")
                if results.get('rinse_adjustments'):
                    print(f"RINSE adjustments:   {results['rinse_adjustments']}")
                if results.get('errors'):
                    print(f"Errors:              {len(results['errors'])}")
                print(f"{'='*60}\n")

                # Check if max cycles reached
                if max_cycles and cycles_completed >= max_cycles:
                    LOGGER.info(f"Reached max cycles ({max_cycles}), stopping")
                    break

                # Wait for next cycle
                wait_seconds = interval_hours * 3600
                LOGGER.info(f"Waiting {interval_hours}h until next cycle...")
                time.sleep(wait_seconds)

        except KeyboardInterrupt:
            LOGGER.info("Continuous loop interrupted by user")

        finally:
            # Close RINSE agent (closes DB connection)
            if self.rinse_agent:
                self.rinse_agent.close()

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "cycles_run": self.stats["cycles_run"],
            "forecasts_processed": self.stats["forecasts_processed"],
            "outcomes_collected": self.stats["outcomes_collected"],
            "rinse_iterations": self.stats["rinse_iterations"],
            "last_run": self.stats["last_run"]
        }


def main():
    """CLI interface for learning pipeline"""
    parser = argparse.ArgumentParser(
        description="Automated learning pipeline for Proto-liminal"
    )
    parser.add_argument(
        "--forecasts",
        required=True,
        help="Input forecasts JSONL"
    )
    parser.add_argument(
        "--outcomes",
        required=True,
        help="Output outcomes JSONL"
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Output metrics JSONL"
    )
    parser.add_argument(
        "--db",
        help="SQLite database for RINSE persistence"
    )
    parser.add_argument(
        "--cache-dir",
        default="data/market_cache",
        help="Market data cache directory"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous learning loop"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Hours between cycles (continuous mode)"
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        help="Maximum cycles to run (continuous mode)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create pipeline
    pipeline = LearningPipeline(
        forecasts_path=args.forecasts,
        outcomes_path=args.outcomes,
        metrics_path=args.metrics,
        db_path=args.db,
        cache_dir=args.cache_dir
    )

    # Run
    if args.continuous:
        pipeline.run_continuous(
            interval_hours=args.interval,
            max_cycles=args.max_cycles
        )
    else:
        # Run single cycle
        results = pipeline.run_cycle()

        # Print summary
        print("\n" + "="*60)
        print("Learning Pipeline Summary")
        print("="*60)
        print(f"Outcomes collected:  {results['outcomes_collected']}")
        print(f"Metrics calculated:  {results['metrics_calculated']}")
        print(f"RINSE executed:      {results['rinse_executed']}")
        if results.get('rinse_adjustments'):
            print(f"RINSE adjustments:   {results['rinse_adjustments']}")
            print(f"Evolution applied:   {results['evolution_applied']}")
        print(f"Duration:            {results['duration_seconds']:.1f}s")
        print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
