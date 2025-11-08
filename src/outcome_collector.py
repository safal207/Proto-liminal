#!/usr/bin/env python3
"""
Module: outcome_collector.py
Purpose: Automatically collect market outcomes for forecast evaluation
Part of LIMINAL ProtoConsciousness MVP - Phase 1, Week 1, Task 2

Philosophy:
"Reality is the ultimate teacher. The system must continuously confront
its predictions with actual outcomes to evolve."

Architecture:
1. Find forecasts without outcomes (pending forecasts)
2. Check if forecast time horizon has passed
3. Fetch actual market prices from APIs
4. Calculate outcome (0/1 for binary movement predictions)
5. Record outcome in FeedbackTracker
6. Trigger RINSE cycle with new metrics
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from market_data_api import MarketDataAPI
except ImportError:
    MarketDataAPI = None

LOGGER = logging.getLogger(__name__)


@dataclass
class PendingForecast:
    """Forecast awaiting outcome collection"""
    forecast_id: str
    entity: str
    forecast_type: str
    probability: float
    timestamp: str
    horizon_hours: int
    initial_price: Optional[float] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def is_ready_for_outcome(self, current_time: datetime) -> bool:
        """Check if forecast time horizon has passed"""
        forecast_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        elapsed = current_time - forecast_time
        return elapsed >= timedelta(hours=self.horizon_hours)


class OutcomeCollector:
    """
    Automated outcome collection for forecast evaluation

    Integrates with:
    - FeedbackTracker (record outcomes)
    - Market data APIs (get actual prices)
    - RINSE agent (trigger learning cycles)
    """

    def __init__(
        self,
        forecasts_path: Optional[str] = None,
        outcomes_path: Optional[str] = None,
        cache_dir: str = "data/market_cache"
    ):
        """
        Initialize outcome collector

        Args:
            forecasts_path: Path to forecasts JSONL
            outcomes_path: Path to outcomes JSONL (for persistence)
            cache_dir: Directory for caching market data
        """
        self.forecasts_path = forecasts_path
        self.outcomes_path = outcomes_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.pending_forecasts: Dict[str, PendingForecast] = {}
        self.collected_outcomes: Dict[str, Dict] = {}

        # Market data API
        if MarketDataAPI:
            self.market_api = MarketDataAPI(cache_dir=cache_dir)
        else:
            self.market_api = None
            LOGGER.warning("MarketDataAPI not available")

        # Market data cache (legacy)
        self.price_cache: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # Statistics
        self.stats = {
            "forecasts_loaded": 0,
            "outcomes_collected": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "errors": 0
        }

        LOGGER.info("OutcomeCollector initialized")

    def load_forecasts(self, forecasts_path: Optional[str] = None):
        """
        Load forecasts from JSONL file

        Args:
            forecasts_path: Path to forecasts JSONL
        """
        path = forecasts_path or self.forecasts_path
        if not path:
            raise ValueError("No forecasts path provided")

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    forecast = json.loads(line)
                    self._add_pending_forecast(forecast)
                except json.JSONDecodeError:
                    LOGGER.warning(f"Line {line_num}: Invalid JSON")
                    self.stats["errors"] += 1

        LOGGER.info(f"Loaded {len(self.pending_forecasts)} pending forecasts")

    def _add_pending_forecast(self, forecast: Dict):
        """Add forecast to pending queue"""
        forecast_id = self._generate_forecast_id(forecast)

        # Extract forecast details
        pending = PendingForecast(
            forecast_id=forecast_id,
            entity=forecast.get("entity", "unknown"),
            forecast_type=forecast.get("forecast_type", "movement"),
            probability=forecast.get("probability", 0.5),
            timestamp=forecast.get("timestamp", datetime.now(timezone.utc).isoformat()),
            horizon_hours=forecast.get("horizon_hours", 24),
            initial_price=forecast.get("current_price"),
            metadata=forecast
        )

        self.pending_forecasts[forecast_id] = pending
        self.stats["forecasts_loaded"] += 1

    def _generate_forecast_id(self, forecast: Dict) -> str:
        """Generate unique forecast ID"""
        entity = forecast.get("entity", "unknown")
        timestamp = forecast.get("timestamp", datetime.now(timezone.utc).isoformat())
        return f"{entity}_{timestamp}".replace(" ", "_").replace(":", "")

    def get_ready_forecasts(self, current_time: Optional[datetime] = None) -> List[PendingForecast]:
        """
        Get forecasts ready for outcome collection

        Args:
            current_time: Current time (defaults to now)

        Returns:
            List of ready forecasts
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        ready = []
        for forecast in self.pending_forecasts.values():
            if forecast.is_ready_for_outcome(current_time):
                ready.append(forecast)

        return ready

    def fetch_current_price(
        self,
        entity: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[float]:
        """
        Fetch current market price for entity

        Args:
            entity: Entity symbol (e.g., "AAPL", "BTC/USD")
            timestamp: Optional historical timestamp

        Returns:
            Current price or None if unavailable
        """
        if not self.market_api:
            LOGGER.error("MarketDataAPI not available")
            return None

        # Fetch price using market API
        try:
            price = self.market_api.fetch_price(entity)

            if price:
                # Update local cache
                if timestamp is None:
                    timestamp = datetime.now(timezone.utc)
                self.price_cache[entity].append((timestamp, price))

                # Update stats
                api_stats = self.market_api.get_stats()
                self.stats["api_calls"] = api_stats["api_calls"]
                self.stats["cache_hits"] = api_stats["cache_hits"]

                return price

        except Exception as e:
            LOGGER.error(f"Error fetching price for {entity}: {e}")
            self.stats["errors"] += 1

        return None

    def calculate_outcome(
        self,
        forecast: PendingForecast,
        current_price: float
    ) -> float:
        """
        Calculate binary outcome (0 or 1) based on price movement

        Args:
            forecast: Pending forecast
            current_price: Actual current price

        Returns:
            Outcome (1 if prediction correct, 0 if wrong)
        """
        if forecast.initial_price is None:
            LOGGER.warning(f"No initial price for {forecast.entity}, cannot calculate outcome")
            return 0.0

        # Calculate price change
        price_change = current_price - forecast.initial_price
        price_direction = 1 if price_change > 0 else 0

        # Forecast predicts "up" if probability > 0.5
        forecast_direction = 1 if forecast.probability > 0.5 else 0

        # Outcome = 1 if prediction matched reality
        outcome = 1.0 if forecast_direction == price_direction else 0.0

        return outcome

    def collect_outcome(
        self,
        forecast: PendingForecast,
        current_time: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Collect outcome for a single forecast

        Args:
            forecast: Pending forecast
            current_time: Current time (defaults to now)

        Returns:
            Outcome dictionary or None if failed
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Check if ready
        if not forecast.is_ready_for_outcome(current_time):
            LOGGER.debug(f"Forecast {forecast.forecast_id} not ready yet")
            return None

        # Fetch current price
        current_price = self.fetch_current_price(forecast.entity, current_time)

        if current_price is None:
            LOGGER.warning(f"Could not fetch price for {forecast.entity}")
            self.stats["errors"] += 1
            return None

        # Calculate outcome
        outcome_value = self.calculate_outcome(forecast, current_price)

        # Create outcome record
        outcome = {
            "forecast_id": forecast.forecast_id,
            "entity": forecast.entity,
            "outcome": outcome_value,
            "timestamp": current_time.isoformat(),
            "current_price": current_price,
            "initial_price": forecast.initial_price,
            "price_change": current_price - forecast.initial_price if forecast.initial_price else None,
            "forecast_probability": forecast.probability,
            "metadata": {
                "forecast_type": forecast.forecast_type,
                "horizon_hours": forecast.horizon_hours
            }
        }

        # Store outcome
        self.collected_outcomes[forecast.forecast_id] = outcome
        self.stats["outcomes_collected"] += 1

        # Remove from pending
        if forecast.forecast_id in self.pending_forecasts:
            del self.pending_forecasts[forecast.forecast_id]

        LOGGER.info(f"Collected outcome for {forecast.entity}: {outcome_value} "
                   f"(price: {forecast.initial_price} → {current_price})")

        return outcome

    def collect_all_ready(self, current_time: Optional[datetime] = None) -> List[Dict]:
        """
        Collect outcomes for all ready forecasts

        Args:
            current_time: Current time (defaults to now)

        Returns:
            List of collected outcomes
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        ready_forecasts = self.get_ready_forecasts(current_time)

        LOGGER.info(f"Found {len(ready_forecasts)} forecasts ready for outcome collection")

        outcomes = []
        for forecast in ready_forecasts:
            outcome = self.collect_outcome(forecast, current_time)
            if outcome:
                outcomes.append(outcome)

        return outcomes

    def save_outcomes(self, output_path: Optional[str] = None):
        """
        Save collected outcomes to JSONL

        Args:
            output_path: Path to output file
        """
        path = output_path or self.outcomes_path
        if not path:
            raise ValueError("No outcomes path provided")

        from utils_io import ensure_parent_dir, safe_write_jsonl

        ensure_parent_dir(path)

        for outcome in self.collected_outcomes.values():
            safe_write_jsonl(path, outcome)

        LOGGER.info(f"Saved {len(self.collected_outcomes)} outcomes to {path}")

    def get_summary(self) -> Dict:
        """Get collector statistics"""
        return {
            "forecasts_loaded": self.stats["forecasts_loaded"],
            "pending_forecasts": len(self.pending_forecasts),
            "outcomes_collected": self.stats["outcomes_collected"],
            "api_calls": self.stats["api_calls"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["api_calls"] + self.stats["cache_hits"])
            ),
            "errors": self.stats["errors"]
        }


def main():
    """CLI interface for outcome collection"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated outcome collection for forecast evaluation"
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
        "--cache-dir",
        default="data/market_cache",
        help="Market data cache directory"
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

    # Create collector
    collector = OutcomeCollector(
        forecasts_path=args.forecasts,
        outcomes_path=args.outcomes,
        cache_dir=args.cache_dir
    )

    # Load forecasts
    collector.load_forecasts()

    # Collect outcomes
    outcomes = collector.collect_all_ready()

    # Save outcomes
    if outcomes:
        collector.save_outcomes()

    # Print summary
    summary = collector.get_summary()

    print("\n" + "="*60)
    print("Outcome Collection Summary")
    print("="*60)
    print(f"Forecasts loaded:     {summary['forecasts_loaded']}")
    print(f"Pending forecasts:    {summary['pending_forecasts']}")
    print(f"Outcomes collected:   {summary['outcomes_collected']}")
    print(f"API calls:            {summary['api_calls']}")
    print(f"Cache hits:           {summary['cache_hits']}")
    print(f"Cache hit rate:       {summary['cache_hit_rate']:.2%}")
    print(f"Errors:               {summary['errors']}")
    print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
