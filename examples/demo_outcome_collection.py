#!/usr/bin/env python3
"""
Demo: Automated Outcome Collection

Demonstrates the complete outcome collection workflow with simulated data:
1. Generate sample forecasts
2. Simulate market price movements
3. Collect outcomes automatically
4. Feed into RINSE feedback loop

This shows how the system would work in production with real APIs.
"""

import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from outcome_collector import OutcomeCollector, PendingForecast
from feedback_tracker import FeedbackTracker
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
LOGGER = logging.getLogger(__name__)


class SimulatedMarketAPI:
    """Simulated market data for testing"""

    def __init__(self, volatility: float = 0.02):
        """
        Initialize simulated market

        Args:
            volatility: Price movement volatility (default: 2%)
        """
        self.volatility = volatility
        self.prices = {
            "AAPL": 180.0,
            "TSLA": 250.0,
            "BTC/USD": 43000.0,
            "ETH/USD": 2200.0,
            "GAZP": 155.0,
            "SBER": 280.0
        }

    def get_price(self, entity: str, hours_elapsed: int = 0) -> float:
        """
        Get simulated price after time has passed

        Args:
            entity: Entity symbol
            hours_elapsed: Hours since initial price

        Returns:
            Simulated current price
        """
        if entity not in self.prices:
            return 100.0  # Default price

        initial_price = self.prices[entity]

        # Simulate price movement (random walk)
        # More hours = more potential movement
        cumulative_change = 0.0
        for _ in range(hours_elapsed):
            change = random.gauss(0, self.volatility)
            cumulative_change += change

        current_price = initial_price * (1 + cumulative_change)

        return max(0.01, current_price)  # Prevent negative prices


def generate_sample_forecasts(n_forecasts: int = 10) -> str:
    """
    Generate sample forecasts for testing

    Args:
        n_forecasts: Number of forecasts to generate

    Returns:
        Path to forecasts JSONL file
    """
    entities = ["AAPL", "TSLA", "BTC/USD", "ETH/USD", "GAZP", "SBER"]

    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    forecasts_path = temp_file.name

    market = SimulatedMarketAPI()

    with open(forecasts_path, 'w') as f:
        for i in range(n_forecasts):
            entity = random.choice(entities)

            # Forecast made some time ago (1-48 hours)
            hours_ago = random.randint(1, 48)
            forecast_time = datetime.now(timezone.utc) - timedelta(hours=hours_ago)

            # Horizon: how far ahead to predict
            horizon_hours = random.choice([6, 12, 24, 48])

            # Initial price at forecast time
            initial_price = market.get_price(entity)

            # Generate probability (biased slightly up for variety)
            probability = random.uniform(0.3, 0.9)

            forecast = {
                "entity": entity,
                "forecast_type": "movement",
                "probability": probability,
                "timestamp": forecast_time.isoformat(),
                "horizon_hours": horizon_hours,
                "current_price": initial_price,
                "forecast_id": f"{entity}_{i}",
                "metadata": {
                    "source": "demo",
                    "model": "simulated"
                }
            }

            f.write(json.dumps(forecast) + '\n')

    temp_file.close()
    LOGGER.info(f"Generated {n_forecasts} sample forecasts: {forecasts_path}")

    return forecasts_path


def main():
    """Demo outcome collection workflow"""

    print("="*80)
    print("Automated Outcome Collection Demo")
    print("="*80)
    print()

    # Step 1: Generate sample forecasts
    print("Step 1: Generating sample forecasts...")
    forecasts_path = generate_sample_forecasts(n_forecasts=20)
    print(f"  ✓ Generated: {forecasts_path}")
    print()

    # Step 2: Create outcome collector
    print("Step 2: Initializing outcome collector...")

    outcomes_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    outcomes_path = outcomes_file.name
    outcomes_file.close()

    collector = OutcomeCollector(
        forecasts_path=forecasts_path,
        outcomes_path=outcomes_path
    )

    # Inject simulated market API
    simulated_market = SimulatedMarketAPI(volatility=0.03)

    # Override fetch_current_price with simulation
    original_fetch = collector.fetch_current_price

    def simulated_fetch(entity: str, timestamp=None):
        """Simulated price fetcher"""
        # Find the forecast to know how much time passed
        for forecast_id, forecast in collector.pending_forecasts.items():
            if forecast.entity == entity:
                forecast_time = datetime.fromisoformat(forecast.timestamp.replace('Z', '+00:00'))
                current_time = timestamp or datetime.now(timezone.utc)
                elapsed = current_time - forecast_time
                hours_elapsed = int(elapsed.total_seconds() / 3600)

                price = simulated_market.get_price(entity, hours_elapsed)
                LOGGER.info(f"Simulated price for {entity} after {hours_elapsed}h: ${price:.2f}")
                return price

        # Fallback
        return simulated_market.get_price(entity)

    collector.fetch_current_price = simulated_fetch

    print(f"  ✓ Outcome collector ready")
    print()

    # Step 3: Load forecasts
    print("Step 3: Loading forecasts...")
    collector.load_forecasts()
    print(f"  ✓ Loaded {len(collector.pending_forecasts)} pending forecasts")
    print()

    # Step 4: Collect outcomes for ready forecasts
    print("Step 4: Collecting outcomes...")
    print()

    ready_forecasts = collector.get_ready_forecasts()
    print(f"Found {len(ready_forecasts)} forecasts ready for outcome collection")
    print()

    if ready_forecasts:
        print("Sample ready forecasts:")
        for i, forecast in enumerate(ready_forecasts[:5], 1):
            forecast_time = datetime.fromisoformat(forecast.timestamp.replace('Z', '+00:00'))
            age = datetime.now(timezone.utc) - forecast_time
            print(f"  {i}. {forecast.entity}: "
                  f"probability={forecast.probability:.2f}, "
                  f"horizon={forecast.horizon_hours}h, "
                  f"age={age.total_seconds()/3600:.1f}h")
        print()

    # Collect all outcomes
    outcomes = collector.collect_all_ready()

    print(f"\n✅ Collected {len(outcomes)} outcomes!")
    print()

    # Step 5: Display results
    print("="*80)
    print("Outcome Collection Results")
    print("="*80)
    print()

    if outcomes:
        print("Sample outcomes (first 5):")
        print()
        for i, outcome in enumerate(outcomes[:5], 1):
            entity = outcome['entity']
            initial = outcome['initial_price']
            current = outcome['current_price']
            change = outcome['price_change']
            change_pct = (change / initial * 100) if initial else 0
            outcome_val = outcome['outcome']
            prob = outcome['forecast_probability']

            direction = "UP ↑" if change > 0 else "DOWN ↓"
            predicted = "UP ↑" if prob > 0.5 else "DOWN ↓"
            correct = "✓ CORRECT" if outcome_val == 1.0 else "✗ WRONG"

            print(f"{i}. {entity}")
            print(f"   Price:      ${initial:.2f} → ${current:.2f} ({change_pct:+.2f}%)")
            print(f"   Actual:     {direction}")
            print(f"   Predicted:  {predicted} (p={prob:.2f})")
            print(f"   Result:     {correct}")
            print()

    # Step 6: Calculate accuracy
    if outcomes:
        correct_predictions = sum(1 for o in outcomes if o['outcome'] == 1.0)
        accuracy = correct_predictions / len(outcomes)

        print("="*80)
        print("Summary Statistics")
        print("="*80)
        print(f"Total forecasts:     {len(outcomes)}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy:            {accuracy:.2%}")
        print()

        # Distribution by entity
        entity_counts = {}
        for outcome in outcomes:
            entity = outcome['entity']
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        print("Outcomes by entity:")
        for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            entity_correct = sum(1 for o in outcomes if o['entity'] == entity and o['outcome'] == 1.0)
            entity_accuracy = entity_correct / count if count > 0 else 0
            print(f"  {entity:10s} {count:2d} outcomes ({entity_accuracy:.1%} accuracy)")

    # Step 7: Save outcomes
    print()
    print("Step 7: Saving outcomes...")
    collector.save_outcomes()
    print(f"  ✓ Saved to: {outcomes_path}")

    # Step 8: Integration with FeedbackTracker (demonstration)
    print()
    print("Step 8: Feeding into FeedbackTracker...")
    tracker = FeedbackTracker()

    # Add forecasts and outcomes
    for outcome in outcomes:
        forecast_id = outcome['forecast_id']

        # Create forecast record
        forecast = {
            "entity": outcome['entity'],
            "forecast_type": outcome['metadata']['forecast_type'],
            "probability": outcome['forecast_probability'],
            "timestamp": outcome['timestamp']
        }

        tracker.add_forecast(forecast, forecast_id=forecast_id)
        tracker.record_outcome(forecast_id, outcome['outcome'])

    # Evaluate
    results = tracker.evaluate_all()
    summary = tracker.get_summary_stats()

    print(f"  ✓ FeedbackTracker metrics calculated")
    print()

    print("="*80)
    print("FeedbackTracker Metrics")
    print("="*80)
    print(f"Brier Score:        {summary.get('avg_brier_score', 0):.4f}")
    print(f"Calibration Error:  {summary.get('avg_calibration_error', 0):.4f}")
    print(f"Accuracy:           {summary.get('avg_accuracy', 0):.4f}")
    print(f"Precision:          {summary.get('avg_precision', 0):.4f}")
    print(f"Recall:             {summary.get('avg_recall', 0):.4f}")
    print(f"F1 Score:           {summary.get('avg_f1_score', 0):.4f}")
    print("="*80)
    print()

    # Collector stats
    collector_stats = collector.get_summary()
    print("Collector Statistics:")
    print(f"  Forecasts loaded:    {collector_stats['forecasts_loaded']}")
    print(f"  Outcomes collected:  {collector_stats['outcomes_collected']}")
    print(f"  Errors:              {collector_stats['errors']}")
    print()

    print("="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print()
    print("This demonstrates how the automated outcome collector would work")
    print("in production with real market data APIs (Yahoo/Binance/CoinGecko).")
    print()
    print("Next step: Connect to RINSE agent for continuous learning!")

    # Cleanup
    Path(forecasts_path).unlink(missing_ok=True)
    Path(outcomes_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
