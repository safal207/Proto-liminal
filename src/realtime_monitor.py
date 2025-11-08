#!/usr/bin/env python3
"""
Real-Time Market Monitor with Liminal Detection

Integrates Tradernet WebSocket feed with Proto-liminal's
adaptive finance framework for real-time market analysis.

Features:
- Live quote streaming from Tradernet
- Real-time liminal state detection
- Market regime classification
- Adaptive risk management
- Alert generation for critical states

Usage:
    python src/realtime_monitor.py --symbols AAPL TSLA BTCUSD
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict, deque
from pathlib import Path

from tradernet_client import TradernetClient, TradernetConfig, Quote
from liminal_detector import LiminalDetector, LiminalState
from market_regime import MarketRegimeClassifier, MarketRegime
from risk_manager import AdaptiveRiskManager, RiskParameters

LOGGER = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Complete market state snapshot for a symbol"""
    symbol: str
    timestamp: str
    price: float
    liminal_state: str
    liminal_score: float
    market_regime: str
    regime_confidence: float
    volatility: float
    risk_adjustment: float
    alert_level: str  # none, warning, critical

    def to_dict(self) -> dict:
        return asdict(self)


class RealtimeMonitor:
    """
    Real-time market monitoring with liminal detection

    Processes live market data from Tradernet and applies
    liminal state detection, regime classification, and
    adaptive risk management in real-time.
    """

    def __init__(
        self,
        symbols: List[str],
        output_dir: str = "data",
        log_quotes: bool = True,
        alert_on_critical: bool = True
    ):
        """
        Initialize real-time monitor

        Args:
            symbols: List of symbols to monitor
            output_dir: Directory for output logs
            log_quotes: Whether to log all quotes to file
            alert_on_critical: Whether to alert on critical states
        """
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.log_quotes = log_quotes
        self.alert_on_critical = alert_on_critical

        # Tradernet client
        config = TradernetConfig(symbols=symbols)
        self.client = TradernetClient(config)

        # Per-symbol detectors
        self.liminal_detectors: Dict[str, LiminalDetector] = {}
        self.regime_classifiers: Dict[str, MarketRegimeClassifier] = {}
        self.risk_managers: Dict[str, AdaptiveRiskManager] = {}

        # Initialize detection systems for each symbol
        for symbol in symbols:
            self.liminal_detectors[symbol] = LiminalDetector(
                volatility_window=20,
                sentiment_window=10,
                liminal_threshold=0.6,
                critical_threshold=0.8
            )

            self.regime_classifiers[symbol] = MarketRegimeClassifier(
                price_window=50,
                volatility_window=20
            )

            self.risk_managers[symbol] = AdaptiveRiskManager(
                initial_equity=10000.0,
                params=RiskParameters(
                    max_position_size_pct=0.1,
                    max_portfolio_risk_pct=0.02,
                    kelly_fraction=0.25,
                    max_drawdown_pct=0.2
                )
            )

        # Price history for volatility calculation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Statistics
        self.snapshots_generated = 0
        self.critical_alerts = 0
        self.start_time = None

        # Output files
        self.snapshot_file = self.output_dir / "realtime_snapshots.jsonl"
        self.alert_file = self.output_dir / "realtime_alerts.jsonl"

        LOGGER.info(f"RealtimeMonitor initialized for {len(symbols)} symbols")

    def calculate_volatility(self, symbol: str, window: int = 20) -> float:
        """Calculate rolling volatility from price history"""
        prices = list(self.price_history[symbol])

        if len(prices) < 2:
            return 0.0

        # Simple volatility: std dev of returns
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)

        if not returns:
            return 0.0

        # Standard deviation
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5

        return volatility

    async def process_quote(self, quote: Quote) -> Optional[MarketSnapshot]:
        """
        Process incoming quote and generate market snapshot

        Args:
            quote: Real-time quote from Tradernet

        Returns:
            MarketSnapshot with liminal state and risk assessment
        """
        try:
            symbol = quote.symbol
            price = quote.price

            # Update price history
            self.price_history[symbol].append(price)

            # Need minimum history
            if len(self.price_history[symbol]) < 5:
                return None

            # Calculate volatility
            volatility = self.calculate_volatility(symbol)

            # Liminal detection
            detector = self.liminal_detectors[symbol]
            liminal_state = detector.detect(
                current_volatility=volatility,
                current_sentiment=0.0,  # TODO: integrate sentiment from news
                current_volume=quote.volume if quote.volume else 1.0,
                indicator_signals={}
            )

            # Market regime classification
            classifier = self.regime_classifiers[symbol]
            regime = classifier.classify(
                price=price,
                sentiment=0.0,
                volume=quote.volume if quote.volume else 1.0
            )

            # Risk adjustment
            risk_mgr = self.risk_managers[symbol]
            liminal_adjustment = risk_mgr.calculate_liminal_adjustment(
                liminal_state=liminal_state,
                regime=regime
            )

            # Determine alert level
            alert_level = "none"
            if liminal_state.state == "critical":
                alert_level = "critical"
            elif liminal_state.state == "liminal":
                alert_level = "warning"

            # Create snapshot
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=quote.timestamp,
                price=price,
                liminal_state=liminal_state.state,
                liminal_score=liminal_state.score,
                market_regime=regime.regime_type,
                regime_confidence=regime.confidence,
                volatility=volatility,
                risk_adjustment=liminal_adjustment,
                alert_level=alert_level
            )

            self.snapshots_generated += 1

            # Log snapshot
            if self.log_quotes:
                with open(self.snapshot_file, "a") as f:
                    f.write(json.dumps(snapshot.to_dict()) + "\n")

            # Generate alert if critical
            if alert_level == "critical" and self.alert_on_critical:
                await self.generate_alert(snapshot)

            return snapshot

        except Exception as e:
            LOGGER.error(f"Error processing quote for {quote.symbol}: {e}")
            return None

    async def generate_alert(self, snapshot: MarketSnapshot):
        """Generate and log critical state alert"""
        try:
            alert = {
                "timestamp": snapshot.timestamp,
                "symbol": snapshot.symbol,
                "alert_type": "CRITICAL_LIMINAL_STATE",
                "details": {
                    "liminal_score": snapshot.liminal_score,
                    "regime": snapshot.market_regime,
                    "volatility": snapshot.volatility,
                    "risk_adjustment": snapshot.risk_adjustment,
                    "recommendation": "REDUCE_EXPOSURE"
                }
            }

            # Log alert
            with open(self.alert_file, "a") as f:
                f.write(json.dumps(alert) + "\n")

            self.critical_alerts += 1

            LOGGER.warning(
                f"🚨 CRITICAL ALERT: {snapshot.symbol} "
                f"(liminal={snapshot.liminal_score:.2f}, "
                f"vol={snapshot.volatility:.4f})"
            )

        except Exception as e:
            LOGGER.error(f"Error generating alert: {e}")

    async def print_snapshot(self, snapshot: MarketSnapshot):
        """Print formatted snapshot to console"""
        # Color codes for terminal
        RESET = "\033[0m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        CYAN = "\033[96m"

        # Color based on state
        if snapshot.liminal_state == "critical":
            color = RED
            state_icon = "🔴"
        elif snapshot.liminal_state == "liminal":
            color = YELLOW
            state_icon = "🟡"
        else:
            color = GREEN
            state_icon = "🟢"

        # Regime icon
        regime_icons = {
            "bull": "🐂",
            "bear": "🐻",
            "sideways": "➡️",
            "transition": "🔄"
        }
        regime_icon = regime_icons.get(snapshot.market_regime, "❓")

        print(
            f"{color}{state_icon} {snapshot.symbol:8s}{RESET} "
            f"${snapshot.price:10.2f} | "
            f"{regime_icon} {snapshot.market_regime:10s} | "
            f"Liminal: {snapshot.liminal_score:.2f} | "
            f"Vol: {snapshot.volatility:.4f} | "
            f"Risk Adj: {snapshot.risk_adjustment:.2f}x"
        )

    async def display_stats(self):
        """Display periodic statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

                print("\n" + "="*80)
                print(f"📊 MONITOR STATISTICS (Uptime: {uptime:.0f}s)")
                print("="*80)

                client_stats = self.client.get_stats()
                print(f"Quotes received: {client_stats['quotes_received']}")
                print(f"Snapshots generated: {self.snapshots_generated}")
                print(f"Critical alerts: {self.critical_alerts}")

                # Per-symbol stats
                print("\nPer-Symbol Status:")
                for symbol in self.symbols:
                    detector = self.liminal_detectors[symbol]
                    prices = list(self.price_history[symbol])

                    if prices:
                        latest_price = prices[-1]
                        state = detector.current_state

                        print(f"  {symbol:8s}: ${latest_price:10.2f} - {state}")

                print("="*80 + "\n")

            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"Error displaying stats: {e}")

    async def run(self):
        """
        Start real-time monitoring

        Main entry point for the monitor. Connects to Tradernet,
        streams quotes, and processes them through liminal detection.
        """
        self.start_time = datetime.now(timezone.utc)

        print("🚀 Proto-liminal Real-Time Monitor")
        print(f"📡 Connecting to Tradernet...")
        print(f"📊 Monitoring: {', '.join(self.symbols)}")
        print(f"📁 Output: {self.output_dir}")
        print("Press Ctrl+C to stop\n")

        # Register quote handler
        async def on_quote(quote: Quote):
            snapshot = await self.process_quote(quote)
            if snapshot:
                await self.print_snapshot(snapshot)

        self.client.register_quote_callback(on_quote)

        # Start tasks
        tasks = [
            asyncio.create_task(self.client.run_with_reconnect()),
            asyncio.create_task(self.display_stats())
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("\n⏹  Stopping monitor...")
            for task in tasks:
                task.cancel()
            await self.client.disconnect()

            # Final stats
            print("\n" + "="*80)
            print("📈 FINAL STATISTICS")
            print("="*80)
            print(f"Snapshots: {self.snapshots_generated}")
            print(f"Critical alerts: {self.critical_alerts}")
            print(f"Output files:")
            print(f"  - {self.snapshot_file}")
            print(f"  - {self.alert_file}")
            print("="*80)


async def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Proto-liminal Real-Time Market Monitor"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "TSLA", "BTCUSD"],
        help="Symbols to monitor"
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable quote logging"
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Disable critical alerts"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Create and run monitor
    monitor = RealtimeMonitor(
        symbols=args.symbols,
        output_dir=args.output,
        log_quotes=not args.no_log,
        alert_on_critical=not args.no_alerts
    )

    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
