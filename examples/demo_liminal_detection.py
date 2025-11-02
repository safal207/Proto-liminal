"""
Demo: Liminal State Detection
Demonstrates adaptive risk management based on liminal state detection
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import random
from datetime import datetime, timedelta

from liminal_detector import LiminalDetector
from market_regime import MarketRegimeClassifier


def simulate_market_data(days: int = 30):
    """
    Simulate market data with different regimes

    Returns:
        List of (timestamp, price, sentiment, volatility, volume) tuples
    """
    data = []
    base_price = 100.0
    current_price = base_price

    start_date = datetime(2025, 1, 1)

    # Simulate different market phases
    for day in range(days):
        timestamp = start_date + timedelta(hours=day)

        # Phase transitions
        if day < 10:
            # Bull market
            price_change = random.uniform(0.5, 2.0)
            sentiment = random.uniform(0.3, 0.7)
            volatility = random.uniform(0.1, 0.3)
            volume = random.randint(80, 120)

        elif day < 12:
            # TRANSITION: Bull to Bear
            price_change = random.uniform(-5.0, 5.0)  # High volatility
            sentiment = random.uniform(-0.2, 0.2)  # Neutral/confused
            volatility = random.uniform(0.6, 0.9)  # Spike
            volume = random.randint(150, 250)  # Volume surge

        elif day < 22:
            # Bear market
            price_change = random.uniform(-2.0, -0.5)
            sentiment = random.uniform(-0.7, -0.3)
            volatility = random.uniform(0.2, 0.4)
            volume = random.randint(60, 100)

        else:
            # Sideways / Recovery
            price_change = random.uniform(-1.0, 1.0)
            sentiment = random.uniform(-0.1, 0.1)
            volatility = random.uniform(0.1, 0.2)
            volume = random.randint(40, 80)

        current_price += price_change

        data.append((
            timestamp.isoformat(),
            current_price,
            sentiment,
            volatility,
            volume
        ))

    return data


def main():
    """Run liminal detection demo"""
    print("=" * 70)
    print("  LIMINAL STATE DETECTION DEMO")
    print("  Adaptive Finance Framework - Proto-liminal")
    print("=" * 70)
    print()

    # Initialize detectors
    liminal_detector = LiminalDetector(
        volatility_window=10,
        sentiment_window=5,
        liminal_threshold=0.6,
        critical_threshold=0.8
    )

    regime_classifier = MarketRegimeClassifier(
        trend_window=20,
        volatility_window=10
    )

    # Generate market data
    print("📊 Generating simulated market data (30 days)...")
    market_data = simulate_market_data(days=30)

    # Track states for analysis
    liminal_states = []
    regime_classifications = []

    # Process each data point
    print("\n⏱️  Processing market data...\n")

    for i, (timestamp, price, sentiment, volatility, volume) in enumerate(market_data):
        # Detect liminal state
        liminal_state = liminal_detector.detect(
            sentiment=sentiment,
            volatility=volatility,
            volume=volume
        )

        # Classify regime
        regime = regime_classifier.classify(
            price=price,
            sentiment=sentiment
        )

        liminal_states.append(liminal_state)
        regime_classifications.append(regime)

        # Print significant events
        if liminal_state.state in ["liminal", "critical"] or i % 5 == 0:
            day = i // 1 + 1

            # State indicator
            if liminal_state.state == "critical":
                state_icon = "🔴"
            elif liminal_state.state == "liminal":
                state_icon = "🟡"
            else:
                state_icon = "🟢"

            # Regime indicator
            regime_icons = {
                "bull": "📈",
                "bear": "📉",
                "sideways": "↔️",
                "transition": "🔄"
            }
            regime_icon = regime_icons.get(regime.regime, "❓")

            print(f"Day {day:2d} {state_icon} {regime_icon} | "
                  f"Price: ${price:6.2f} | "
                  f"Liminal: {liminal_state.liminal_score:.2f} | "
                  f"Regime: {regime.regime:10s} | "
                  f"Signals: {len(liminal_state.signals)}")

            # Show signals if any
            for signal in liminal_state.signals:
                print(f"        └─ {signal.signal_type}: {signal.description}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("  DETECTION SUMMARY")
    print("=" * 70)

    # Count states
    stable_count = sum(1 for s in liminal_states if s.state == "stable")
    liminal_count = sum(1 for s in liminal_states if s.state == "liminal")
    critical_count = sum(1 for s in liminal_states if s.state == "critical")

    print(f"\n📊 State Distribution:")
    print(f"  🟢 Stable:   {stable_count:2d} ({stable_count/len(liminal_states)*100:.1f}%)")
    print(f"  🟡 Liminal:  {liminal_count:2d} ({liminal_count/len(liminal_states)*100:.1f}%)")
    print(f"  🔴 Critical: {critical_count:2d} ({critical_count/len(liminal_states)*100:.1f}%)")

    # Count regimes
    regime_counts = {}
    for r in regime_classifications:
        regime_counts[r.regime] = regime_counts.get(r.regime, 0) + 1

    print(f"\n📈 Regime Distribution:")
    for regime, count in sorted(regime_counts.items()):
        pct = count / len(regime_classifications) * 100
        icon = {"bull": "📈", "bear": "📉", "sideways": "↔️", "transition": "🔄"}.get(regime, "❓")
        print(f"  {icon} {regime.capitalize():10s}: {count:2d} ({pct:.1f}%)")

    # Risk adjustment recommendation
    print(f"\n💡 Risk Management Insights:")

    avg_liminal_score = sum(s.liminal_score for s in liminal_states) / len(liminal_states)

    if avg_liminal_score > 0.6:
        print(f"  ⚠️  HIGH LIMINAL ACTIVITY (avg score: {avg_liminal_score:.2f})")
        print(f"  → Recommendation: REDUCE position sizes by 50-70%")
        print(f"  → Tighten stop-losses")
        print(f"  → Increase cash reserves")
    elif avg_liminal_score > 0.4:
        print(f"  ⚡ MODERATE LIMINAL ACTIVITY (avg score: {avg_liminal_score:.2f})")
        print(f"  → Recommendation: REDUCE position sizes by 30-50%")
        print(f"  → Monitor closely for regime changes")
    else:
        print(f"  ✅ LOW LIMINAL ACTIVITY (avg score: {avg_liminal_score:.2f})")
        print(f"  → Recommendation: NORMAL risk exposure acceptable")
        print(f"  → Continue standard trading strategies")

    # Export results
    output_file = Path(__file__).parent.parent / "data" / "liminal_detection_demo.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for liminal, regime in zip(liminal_states, regime_classifications):
            record = {
                "liminal_state": liminal.to_dict(),
                "regime_classification": regime.to_dict()
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n💾 Results exported to: {output_file}")

    print("\n" + "=" * 70)
    print("  Demo complete! ✨")
    print("=" * 70)


if __name__ == "__main__":
    main()
