#!/usr/bin/env python3
"""
Real-Time Market Monitoring Demo (Simulated Data)

Demonstrates Proto-liminal's real-time monitoring capabilities
with simulated market data. Shows how the system would work
with actual live feeds from Tradernet or other providers.

This demo simulates:
- Live quote streaming
- Real-time liminal state detection
- Market regime classification
- Adaptive risk adjustments
- Critical state alerts

Usage:
    python examples/demo_realtime_simulated.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import random
import logging
from datetime import datetime, timezone
from collections import deque, defaultdict

from liminal_detector import LiminalDetector, LiminalState
from market_regime import MarketRegimeClassifier
from risk_manager import AdaptiveRiskManager, RiskParameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


class SimulatedMarketFeed:
    """Simulates realistic market data feed"""

    def __init__(self, symbols: list, base_prices: dict):
        self.symbols = symbols
        self.prices = base_prices.copy()
        self.volatilities = {s: 0.02 for s in symbols}  # 2% volatility
        self.trends = {s: 0.0 for s in symbols}  # Current trend
        self.phase = "stable"  # Market phase
        self.tick = 0

    async def generate_quote(self, symbol: str):
        """Generate realistic price movement"""

        # Change market phases periodically
        if self.tick % 50 == 0:
            phases = ["stable", "trending_up", "trending_down", "volatile", "crash"]
            self.phase = random.choice(phases)

            print(f"\n🔄 Market Phase Change: {self.phase.upper()}")
            print("-" * 80)

        # Adjust based on phase
        if self.phase == "stable":
            drift = 0.0001
            vol = 0.01
        elif self.phase == "trending_up":
            drift = 0.002
            vol = 0.015
            self.trends[symbol] = 0.5
        elif self.phase == "trending_down":
            drift = -0.002
            vol = 0.015
            self.trends[symbol] = -0.5
        elif self.phase == "volatile":
            drift = 0.0
            vol = 0.04  # High volatility
        elif self.phase == "crash":
            drift = -0.01  # Sharp drop
            vol = 0.08  # Very high volatility
        else:
            drift = 0.0
            vol = 0.02

        # Generate price change
        change = drift + random.gauss(0, vol)
        self.prices[symbol] *= (1 + change)
        self.volatilities[symbol] = abs(change)

        self.tick += 1

        return {
            "symbol": symbol,
            "price": self.prices[symbol],
            "volatility": self.volatilities[symbol],
            "volume": random.uniform(1000, 10000),
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        }


class RealtimeMonitorDemo:
    """Real-time monitoring demonstration"""

    def __init__(self, symbols: list, base_prices: dict):
        self.symbols = symbols
        self.feed = SimulatedMarketFeed(symbols, base_prices)

        # Detection systems per symbol
        self.liminal_detectors = {}
        self.regime_classifiers = {}
        self.risk_managers = {}
        self.price_history = defaultdict(lambda: deque(maxlen=100))

        for symbol in symbols:
            self.liminal_detectors[symbol] = LiminalDetector(
                volatility_window=20,
                sentiment_window=10,
                liminal_threshold=0.6,
                critical_threshold=0.8
            )

            self.regime_classifiers[symbol] = MarketRegimeClassifier(
                trend_window=50,
                volatility_window=20
            )

            self.risk_managers[symbol] = AdaptiveRiskManager(
                params=RiskParameters(
                    max_position_size=0.1,
                    max_portfolio_risk=0.02,
                    kelly_fraction=0.25,
                    max_drawdown_limit=0.2
                )
            )
            self.risk_managers[symbol].update_equity(10000.0)

        # Statistics
        self.quotes_processed = 0
        self.critical_alerts = 0
        self.start_time = None

    def get_state_color(self, state: str) -> tuple:
        """Get color and icon for state"""
        if state == "critical":
            return "\033[91m", "🔴"  # Red
        elif state == "liminal":
            return "\033[93m", "🟡"  # Yellow
        else:
            return "\033[92m", "🟢"  # Green

    def get_regime_icon(self, regime: str) -> str:
        """Get icon for regime"""
        icons = {
            "bull": "🐂",
            "bear": "🐻",
            "sideways": "➡️",
            "transition": "🔄"
        }
        return icons.get(regime, "❓")

    async def process_quote(self, quote: dict):
        """Process quote through detection pipeline"""

        symbol = quote["symbol"]
        price = quote["price"]
        volatility = quote["volatility"]

        # Update price history
        self.price_history[symbol].append(price)

        if len(self.price_history[symbol]) < 5:
            return None

        # Liminal detection
        detector = self.liminal_detectors[symbol]
        liminal_state = detector.detect(
            sentiment=0.0,
            volatility=volatility,
            volume=int(quote["volume"]),
            regime="unknown"
        )

        # Market regime classification
        classifier = self.regime_classifiers[symbol]
        regime = classifier.classify(
            price=price,
            sentiment=0.0
        )

        # Risk adjustment
        risk_mgr = self.risk_managers[symbol]
        risk_adj = risk_mgr.calculate_liminal_adjustment(
            liminal_state={
                'state': liminal_state.state,
                'liminal_score': liminal_state.liminal_score
            },
            regime={
                'regime': regime.regime,
                'confidence': regime.confidence
            }
        )

        self.quotes_processed += 1

        # Print formatted output
        color, state_icon = self.get_state_color(liminal_state.state)
        regime_icon = self.get_regime_icon(regime.regime)
        reset = "\033[0m"

        print(
            f"{color}{state_icon} {symbol:8s}{reset} "
            f"${price:10.2f} | "
            f"{regime_icon} {regime.regime:10s} | "
            f"Liminal: {liminal_state.liminal_score:.2f} | "
            f"Vol: {volatility:.4f} | "
            f"Risk Adj: {risk_adj:.2f}x"
        )

        # Generate alert if critical
        if liminal_state.state == "critical":
            self.critical_alerts += 1
            print(f"  {color}⚠️  CRITICAL STATE DETECTED - REDUCE EXPOSURE{reset}")

        return {
            "symbol": symbol,
            "liminal_state": liminal_state.state,
            "liminal_score": liminal_state.liminal_score,
            "regime": regime.regime,
            "risk_adj": risk_adj
        }

    async def run(self, duration: int = 120):
        """
        Run real-time monitoring demo

        Args:
            duration: Duration in seconds
        """
        self.start_time = datetime.now(timezone.utc)

        print("="*80)
        print("🚀 Proto-liminal Real-Time Market Monitor (DEMO)")
        print("="*80)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Duration: {duration}s")
        print(f"Update interval: 0.5s")
        print()
        print("📊 Market Data Feed (Simulated)")
        print("-"*80)

        end_time = datetime.now(timezone.utc).timestamp() + duration
        iteration = 0

        try:
            while datetime.now(timezone.utc).timestamp() < end_time:
                iteration += 1

                # Process each symbol
                for symbol in self.symbols:
                    quote = await self.feed.generate_quote(symbol)
                    await self.process_quote(quote)

                # Status update every 30 seconds
                if iteration % 60 == 0:
                    elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                    print()
                    print("="*80)
                    print(f"📊 Status Update (t={elapsed:.0f}s)")
                    print("-"*80)
                    print(f"  Quotes processed: {self.quotes_processed}")
                    print(f"  Critical alerts: {self.critical_alerts}")
                    print(f"  Current phase: {self.feed.phase}")
                    print("="*80)
                    print()

                await asyncio.sleep(0.5)  # 2 updates per second

        except KeyboardInterrupt:
            print("\n⏹  Stopped by user")

        # Final summary
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        print()
        print("="*80)
        print("📈 FINAL STATISTICS")
        print("="*80)
        print(f"Duration: {elapsed:.1f}s")
        print(f"Quotes processed: {self.quotes_processed}")
        print(f"Critical alerts: {self.critical_alerts}")
        print(f"Quotes per second: {self.quotes_processed / elapsed:.1f}")
        print()

        # Per-symbol status
        print("Final State per Symbol:")
        for symbol in self.symbols:
            detector = self.liminal_detectors[symbol]
            price = self.price_history[symbol][-1] if self.price_history[symbol] else 0

            print(f"  {symbol:8s}: ${price:10.2f} - {detector.current_state}")

        print("="*80)
        print()
        print("💡 To use with real data:")
        print("   1. Get Tradernet API key from tradernet.com")
        print("   2. Update tradernet_socketio_client.py with authentication")
        print("   3. Run: python src/realtime_monitor.py")
        print()


async def main():
    """Main demo entry point"""

    # Demo symbols with realistic base prices
    symbols = ["AAPL", "TSLA", "BTCUSD"]
    base_prices = {
        "AAPL": 178.50,
        "TSLA": 245.00,
        "BTCUSD": 43500.00
    }

    demo = RealtimeMonitorDemo(symbols, base_prices)

    try:
        await demo.run(duration=120)  # 2 minutes
    except KeyboardInterrupt:
        print("\n⏹  Demo stopped")


if __name__ == "__main__":
    print("\n🎬 Starting Real-Time Monitoring Demo...")
    print("Press Ctrl+C to stop\n")
    asyncio.run(main())
