"""
Demo: Adaptive Risk Management
Demonstrates full adaptive finance framework with liminal-aware risk management
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random
from datetime import datetime, timedelta

from liminal_detector import LiminalDetector
from market_regime import MarketRegimeClassifier
from risk_manager import AdaptiveRiskManager, RiskParameters
from portfolio_manager import AdaptivePortfolioManager


def simulate_market_scenario(days: int = 60):
    """
    Simulate market scenario with regime changes

    Returns:
        List of daily data
    """
    data = []
    base_price = 50000.0
    current_price = base_price

    for day in range(days):
        # Determine phase
        if day < 20:
            # Bull market
            price_change_pct = random.uniform(0.5, 2.0)
            sentiment = random.uniform(0.3, 0.7)
            volatility = random.uniform(0.1, 0.3)
            volume = random.randint(100, 150)

        elif day < 25:
            # TRANSITION: Bull to Bear
            price_change_pct = random.uniform(-4.0, 4.0)
            sentiment = random.uniform(-0.3, 0.3)
            volatility = random.uniform(0.6, 0.9)
            volume = random.randint(200, 300)

        elif day < 45:
            # Bear market
            price_change_pct = random.uniform(-2.0, -0.5)
            sentiment = random.uniform(-0.7, -0.3)
            volatility = random.uniform(0.2, 0.5)
            volume = random.randint(80, 120)

        else:
            # Recovery / Sideways
            price_change_pct = random.uniform(-1.0, 1.5)
            sentiment = random.uniform(-0.2, 0.4)
            volatility = random.uniform(0.1, 0.3)
            volume = random.randint(60, 100)

        current_price *= (1 + price_change_pct / 100)

        data.append({
            'day': day + 1,
            'price': current_price,
            'sentiment': sentiment,
            'volatility': volatility,
            'volume': volume,
            'price_change_pct': price_change_pct
        })

    return data


def main():
    """Run adaptive risk management demo"""
    print("=" * 70)
    print("  ADAPTIVE RISK MANAGEMENT DEMO")
    print("  Proto-liminal Finance Framework")
    print("=" * 70)
    print()

    # Initialize components
    print("🚀 Initializing adaptive finance system...")

    liminal_detector = LiminalDetector()
    regime_classifier = MarketRegimeClassifier()

    risk_params = RiskParameters(
        max_risk_per_trade=0.02,
        kelly_fraction=0.25,
        max_drawdown_limit=0.20
    )
    risk_manager = AdaptiveRiskManager(params=risk_params)

    portfolio_manager = AdaptivePortfolioManager(
        initial_cash=10000.0,
        rebalance_threshold=0.05
    )

    print(f"   Starting capital: ${portfolio_manager.portfolio.cash:,.2f}")
    print()

    # Generate market data
    print("📊 Simulating 60-day market scenario...")
    market_data = simulate_market_scenario(days=60)
    print()

    # Track metrics
    portfolio_values = []
    daily_returns = []
    regime_changes = []
    liminal_events = []

    # Run simulation
    print("⏱️  Running adaptive trading simulation...\n")
    print(f"{'Day':>3} | {'State':^8} | {'Regime':^10} | {'Price':>8} | {'Portfolio':>10} | {'Return':>7} | {'Action':^12}")
    print("-" * 90)

    previous_regime = None

    for day_data in market_data:
        day = day_data['day']
        price = day_data['price']
        sentiment = day_data['sentiment']
        volatility = day_data['volatility']
        volume = day_data['volume']

        # Update risk manager equity
        current_value = portfolio_manager.portfolio.total_value
        risk_manager.update_equity(current_value)

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

        # Track regime changes
        if regime.regime != previous_regime and previous_regime is not None:
            regime_changes.append({
                'day': day,
                'from': previous_regime,
                'to': regime.regime
            })
        previous_regime = regime.regime

        # Track liminal events
        if liminal_state.state in ['liminal', 'critical']:
            liminal_events.append({
                'day': day,
                'state': liminal_state.state,
                'score': liminal_state.liminal_score
            })

        # Generate forecast (simplified)
        forecast = {
            'p_calibrated': 0.5 + sentiment * 0.3,  # Sentiment-based probability
            'p_raw': 0.5 + sentiment * 0.3,
            'confidence_band': [0.4, 0.7]
        }

        # Calculate position sizing
        position_sizing = risk_manager.calculate_position_size(
            symbol='BTC',
            entry_price=price,
            direction='long',
            forecast=forecast,
            liminal_state=liminal_state.to_dict(),
            regime=regime.to_dict(),
            volatility=volatility
        )

        # Portfolio rebalancing
        forecasts = {'BTC': forecast}
        prices = {'BTC': price}

        trades = portfolio_manager.rebalance(
            regime=regime.to_dict(),
            liminal_state=liminal_state.to_dict(),
            forecasts=forecasts,
            prices=prices
        )

        # Track portfolio value
        portfolio_value = portfolio_manager.portfolio.total_value
        portfolio_values.append(portfolio_value)

        # Calculate return
        if len(portfolio_values) > 1:
            daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
            daily_returns.append(daily_return)
        else:
            daily_return = 0.0

        # Determine action
        if trades:
            action = f"{trades[0]['action'].upper()} {trades[0]['symbol']}"
        else:
            action = "HOLD"

        # State icon
        state_icons = {'stable': '🟢', 'liminal': '🟡', 'critical': '🔴'}
        state_icon = state_icons.get(liminal_state.state, '⚪')

        # Regime icon
        regime_icons = {'bull': '📈', 'bear': '📉', 'sideways': '↔️', 'transition': '🔄'}
        regime_icon = regime_icons.get(regime.regime, '❓')

        # Print every 3 days or on significant events
        if day % 3 == 0 or liminal_state.state != 'stable' or trades:
            print(f"{day:3d} | {state_icon} {liminal_state.state:6s} | "
                  f"{regime_icon} {regime.regime:8s} | "
                  f"${price:8,.0f} | "
                  f"${portfolio_value:10,.2f} | "
                  f"{daily_return:6.2%} | "
                  f"{action:12s}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("  SIMULATION SUMMARY")
    print("=" * 70)

    final_performance = portfolio_manager.get_performance()
    risk_stats = risk_manager.get_stats()

    print(f"\n💰 Portfolio Performance:")
    print(f"   Starting Value:    ${10000:,.2f}")
    print(f"   Final Value:       ${final_performance['total_value']:,.2f}")
    print(f"   Total Return:      {final_performance['total_return_pct']:.2f}%")
    print(f"   Total P&L:         ${final_performance['total_pnl']:,.2f}")
    print(f"   Sharpe Ratio:      {final_performance['sharpe_ratio']:.2f}")

    print(f"\n📊 Trading Activity:")
    print(f"   Total Trades:      {final_performance['trades_count']}")
    print(f"   Rebalances:        {final_performance['rebalances_count']}")
    print(f"   Active Positions:  {final_performance['positions_count']}")

    print(f"\n⚠️  Risk Metrics:")
    print(f"   Max Drawdown:      {risk_stats['current_drawdown']:.2%}")
    print(f"   Risk Exposure:     {risk_stats['total_risk_exposure']:.2%}")
    print(f"   Win Rate:          {risk_stats['win_rate']:.2%}")

    print(f"\n🔄 Regime Changes:")
    print(f"   Total Changes:     {len(regime_changes)}")
    for change in regime_changes[:5]:  # Show first 5
        print(f"   Day {change['day']:2d}: {change['from']:10s} → {change['to']:10s}")
    if len(regime_changes) > 5:
        print(f"   ... and {len(regime_changes) - 5} more")

    print(f"\n🟡 Liminal Events:")
    print(f"   Total Events:      {len(liminal_events)}")
    critical_events = [e for e in liminal_events if e['state'] == 'critical']
    print(f"   Critical Events:   {len(critical_events)}")

    # Compare with buy-and-hold
    start_price = market_data[0]['price']
    end_price = market_data[-1]['price']
    bh_return = (end_price - start_price) / start_price

    print(f"\n📈 vs Buy-and-Hold:")
    print(f"   Buy-and-Hold Return:  {bh_return:.2%}")
    print(f"   Adaptive Return:      {final_performance['total_return']:.2%}")
    alpha = final_performance['total_return'] - bh_return
    print(f"   Alpha:                {alpha:.2%} {'✅' if alpha > 0 else '❌'}")

    # Holdings summary
    print(f"\n💼 Current Holdings:")
    holdings = portfolio_manager.get_holdings()
    for symbol, data in holdings.items():
        if symbol == 'CASH':
            print(f"   {symbol:5s}: ${data['amount']:10,.2f} ({data['weight']:.1%})")
        else:
            print(f"   {symbol:5s}: {data['quantity']:.4f} @ ${data['current_price']:,.0f} "
                  f"= ${data['market_value']:,.2f} ({data['weight']:.1%}) "
                  f"[P&L: ${data['unrealized_pnl']:+,.2f}]")

    print("\n" + "=" * 70)
    print("  Simulation Complete! ✨")
    print("=" * 70)

    print("\n💡 Key Insights:")
    print("   • Adaptive system adjusts risk in real-time based on market conditions")
    print("   • Liminal detection reduces exposure during transitions")
    print("   • Regime-based allocation optimizes risk/reward")
    print("   • Kelly Criterion ensures optimal position sizing")


if __name__ == "__main__":
    main()
