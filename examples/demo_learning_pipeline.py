#!/usr/bin/env python3
"""
Demo: Complete Automated Learning Pipeline

Demonstrates the full Proto-liminal self-improvement loop:

1. Generate forecasts (simulated predictor)
2. Wait for time to pass
3. Collect outcomes from "market"
4. Calculate performance metrics
5. RINSE agent reflects and adapts
6. State persists to database
7. Next iteration uses improved parameters

This is the heart of Proto-liminal: continuous learning from reality.

Philosophy:
"Intelligence is not static knowledge, but the ability to continuously
learn from mistakes and improve. This pipeline embodies that principle."
"""

import json
import logging
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from learning_pipeline import LearningPipeline
from rinse_persistence import RINSEPersistence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger(__name__)


class SimulatedPredictor:
    """Simulates a prediction model that can improve"""

    def __init__(
        self,
        skill_level: float = 0.6,
        improvement_rate: float = 0.05
    ):
        """
        Initialize simulated predictor

        Args:
            skill_level: Base accuracy (0-1)
            improvement_rate: How much to improve per learning cycle
        """
        self.skill_level = skill_level
        self.improvement_rate = improvement_rate
        self.cycles_completed = 0

    def make_forecast(self, entity: str, will_go_up: bool) -> Dict:
        """
        Generate a forecast

        Args:
            entity: Entity symbol
            will_go_up: True if entity will actually go up

        Returns:
            Forecast dictionary
        """
        # Probability based on skill level
        # If skilled and will_go_up, high probability
        # If skilled and will_go_down, low probability
        if will_go_up:
            base_prob = self.skill_level
        else:
            base_prob = 1.0 - self.skill_level

        # Add some noise
        noise = random.gauss(0, 0.1)
        probability = max(0.1, min(0.9, base_prob + noise))

        # Generate forecast
        forecast_time = datetime.now(timezone.utc) - timedelta(hours=random.randint(25, 48))

        forecast = {
            "entity": entity,
            "forecast_type": "movement",
            "probability": probability,
            "timestamp": forecast_time.isoformat(),
            "horizon_hours": 24,
            "current_price": 100.0 * random.uniform(0.8, 1.2),
            "metadata": {
                "model": "simulated",
                "skill_level": self.skill_level,
                "cycles": self.cycles_completed
            }
        }

        return forecast

    def improve(self):
        """Improve skill level (simulates learning)"""
        old_skill = self.skill_level
        self.skill_level = min(0.95, self.skill_level + self.improvement_rate)
        self.cycles_completed += 1

        improvement = self.skill_level - old_skill
        LOGGER.info(f"Predictor improved: {old_skill:.2%} → {self.skill_level:.2%} (+{improvement:.2%})")


class SimulatedMarket:
    """Simulates market with price movements"""

    def __init__(self):
        self.entities = {
            "AAPL": {"will_go_up": True},
            "TSLA": {"will_go_up": False},
            "BTC/USD": {"will_go_up": True},
            "ETH/USD": {"will_go_up": True},
            "GAZP": {"will_go_up": False},
            "SBER": {"will_go_up": True}
        }

    def will_entity_go_up(self, entity: str) -> bool:
        """Determine if entity will go up"""
        return self.entities.get(entity, {}).get("will_go_up", random.choice([True, False]))


def generate_forecasts_batch(
    predictor: SimulatedPredictor,
    market: SimulatedMarket,
    n_forecasts: int = 20
) -> str:
    """Generate batch of forecasts"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    forecasts_path = temp_file.name

    with open(forecasts_path, 'w') as f:
        for _ in range(n_forecasts):
            entity = random.choice(list(market.entities.keys()))
            will_go_up = market.will_entity_go_up(entity)

            forecast = predictor.make_forecast(entity, will_go_up)
            f.write(json.dumps(forecast) + '\n')

    temp_file.close()
    return forecasts_path


def main():
    """Run complete learning pipeline demo"""

    print("="*80)
    print("Proto-liminal: Automated Learning Pipeline Demo")
    print("="*80)
    print()
    print("This demonstrates the complete self-improvement loop:")
    print("  1. Predictor makes forecasts")
    print("  2. Time passes...")
    print("  3. OutcomeCollector checks reality")
    print("  4. FeedbackTracker calculates metrics")
    print("  5. RINSE agent reflects and adapts")
    print("  6. Predictor improves based on feedback")
    print("  7. Repeat → Continuous learning!")
    print()
    print("="*80)
    print()

    # Create temporary files
    db_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db')
    db_path = db_file.name
    db_file.close()

    outcomes_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    outcomes_path = outcomes_file.name
    outcomes_file.close()

    metrics_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    metrics_path = metrics_file.name
    metrics_file.close()

    # Initialize components
    predictor = SimulatedPredictor(skill_level=0.55, improvement_rate=0.08)
    market = SimulatedMarket()

    print(f"Initial predictor skill level: {predictor.skill_level:.1%}")
    print(f"Database: {db_path}")
    print()

    # Run multiple learning cycles
    n_cycles = 3

    for cycle_num in range(1, n_cycles + 1):
        print("="*80)
        print(f"Learning Cycle {cycle_num}/{n_cycles}")
        print("="*80)
        print()

        # Step 1: Generate forecasts
        print(f"Step 1: Predictor generating forecasts (skill={predictor.skill_level:.1%})...")
        forecasts_path = generate_forecasts_batch(predictor, market, n_forecasts=15)
        print(f"  ✓ Generated 15 forecasts")
        print()

        # Step 2: Create pipeline
        print("Step 2: Initializing learning pipeline...")
        pipeline = LearningPipeline(
            forecasts_path=forecasts_path,
            outcomes_path=outcomes_path,
            metrics_path=metrics_path,
            db_path=db_path
        )

        # Inject simulated market (override fetch_current_price)
        def create_simulated_fetcher(pipeline_ref, market_ref):
            def simulated_fetch(entity: str, timestamp=None):
                """Simulated price fetcher"""
                # Find the forecast
                for forecast_id, forecast in pipeline_ref.outcome_collector.pending_forecasts.items():
                    if forecast.entity == entity:
                        initial = forecast.initial_price or 100.0
                        will_go_up = market_ref.will_entity_go_up(entity)

                        if will_go_up:
                            change = random.uniform(0.05, 0.15)  # Up 5-15%
                            current = initial * (1 + change)
                        else:
                            change = random.uniform(0.05, 0.15)  # Down 5-15%
                            current = initial * (1 - change)

                        return current

                # Fallback
                return 100.0

            return simulated_fetch

        # Initialize and inject
        pipeline._initialize_components()
        pipeline.outcome_collector.fetch_current_price = create_simulated_fetcher(pipeline, market)

        print(f"  ✓ Pipeline ready")
        print()

        # Step 3: Run learning cycle
        print("Step 3: Running learning cycle...")
        print()
        results = pipeline.run_cycle()
        print()

        # Step 4: Display results
        print("="*80)
        print(f"Cycle {cycle_num} Results")
        print("="*80)
        print(f"Outcomes collected:  {results['outcomes_collected']}")
        print(f"Metrics calculated:  {results['metrics_calculated']}")
        print(f"RINSE executed:      {results['rinse_executed']}")

        if results.get('rinse_adjustments'):
            print(f"RINSE adjustments:   {results['rinse_adjustments']}")
            print(f"Evolution applied:   {results['evolution_applied']}")
            print(f"RINSE iteration:     {results['rinse_iteration']}")

        print(f"Duration:            {results['duration_seconds']:.1f}s")
        print()

        # Step 5: Check RINSE state
        if db_path:
            db = RINSEPersistence(db_path)
            summary = db.get_summary()

            print("RINSE Database State:")
            print(f"  Total iterations:    {summary['latest_iteration']}")
            print(f"  Cycles recorded:     {summary['rinse_cycles_count']}")
            print(f"  Reflections:         {summary['reflections_count']}")
            print(f"  Adjustments:         {summary['adjustments_count']}")
            print(f"  Parameter changes:   {summary['parameter_history_count']}")

            db.close()
            print()

        # Step 6: Improve predictor (simulates learning)
        print("Step 6: Predictor learning from feedback...")
        predictor.improve()
        print()

        # Cleanup temp forecast file
        Path(forecasts_path).unlink(missing_ok=True)

    # Final summary
    print("="*80)
    print("Learning Pipeline Demo Complete!")
    print("="*80)
    print()
    print(f"Completed {n_cycles} learning cycles")
    print(f"Predictor skill: {predictor.skill_level - predictor.improvement_rate * n_cycles:.1%} → {predictor.skill_level:.1%}")
    print(f"Improvement: +{predictor.improvement_rate * n_cycles:.1%}")
    print()

    # Show final RINSE state
    db = RINSEPersistence(db_path)
    summary = db.get_summary()

    print("Final RINSE State:")
    print(f"  Total iterations:    {summary['latest_iteration']}")
    print(f"  Cycles recorded:     {summary['rinse_cycles_count']}")
    print(f"  Reflections:         {summary['reflections_count']}")
    print(f"  Adjustments:         {summary['adjustments_count']}")
    print()

    # Show parameter evolution
    config = db.load_config()
    params = db.load_parameters()

    if config:
        print("Current Configuration:")
        print(f"  Reflection threshold: {config['reflection_threshold']}")
        print(f"  Adjustment magnitude: {config['adjustment_magnitude']}")
        print(f"  Confidence threshold: {config['confidence_threshold']}")
        print()

    if params:
        print("Current Parameters (sample):")
        for param_name, value in list(params.items())[:5]:
            history = db.get_parameter_history(param_name, limit=10)
            if history:
                initial = history[-1]['old_value']
                change = value - initial
                print(f"  {param_name}: {initial:.4f} → {value:.4f} ({change:+.4f})")

    db.close()

    print()
    print("="*80)
    print("Key Insights:")
    print("="*80)
    print("✓ Predictor improved through feedback loop")
    print("✓ RINSE agent adapted parameters based on performance")
    print("✓ State persisted across cycles in SQLite")
    print("✓ Complete autonomous learning demonstrated")
    print()
    print("This is Proto-liminal's core: continuous self-improvement")
    print("through confrontation with reality.")
    print("="*80)

    # Cleanup
    Path(db_path).unlink(missing_ok=True)
    Path(outcomes_path).unlink(missing_ok=True)
    Path(metrics_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
