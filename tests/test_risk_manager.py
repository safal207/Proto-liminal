"""
Tests for risk_manager.py and portfolio_manager.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from risk_manager import (
    AdaptiveRiskManager,
    RiskParameters,
    PositionSizing
)
from portfolio_manager import (
    AdaptivePortfolioManager,
    Asset,
    Portfolio,
    RebalanceRecommendation
)


# ===== RiskManager Tests =====

def test_risk_manager_init():
    """Test risk manager initialization"""
    manager = AdaptiveRiskManager()

    assert manager.equity == 10000.0
    assert manager.params.max_risk_per_trade == 0.02
    assert manager.win_rate == 0.5


def test_update_equity():
    """Test equity update and drawdown calculation"""
    manager = AdaptiveRiskManager()

    # Increase equity
    manager.update_equity(12000.0)
    assert manager.equity == 12000.0
    assert manager.peak_equity == 12000.0
    assert manager.current_drawdown == 0.0

    # Drawdown
    manager.update_equity(10000.0)
    assert manager.current_drawdown == (12000 - 10000) / 12000


def test_calculate_kelly_criterion():
    """Test Kelly Criterion calculation"""
    manager = AdaptiveRiskManager()

    # Profitable system: 60% win rate, 2:1 reward/risk
    kelly = manager.calculate_kelly_criterion(
        win_rate=0.6,
        avg_win=200.0,
        avg_loss=100.0
    )

    assert kelly > 0.0
    assert kelly <= 1.0


def test_calculate_kelly_from_forecast():
    """Test Kelly calculation from forecast probabilities"""
    manager = AdaptiveRiskManager()

    kelly = manager.calculate_kelly_from_forecast(
        probability=0.65,
        confidence_lower=0.55,
        confidence_upper=0.75
    )

    assert 0.0 <= kelly <= 1.0


def test_calculate_liminal_adjustment_stable():
    """Test liminal adjustment in stable state"""
    manager = AdaptiveRiskManager()

    liminal_state = {'state': 'stable', 'liminal_score': 0.1}
    regime = {'regime': 'bull', 'confidence': 0.8}

    adjustment = manager.calculate_liminal_adjustment(liminal_state, regime)

    # Should be close to 1.0 in stable state
    assert adjustment > 0.8
    assert adjustment <= 1.0


def test_calculate_liminal_adjustment_critical():
    """Test liminal adjustment in critical state"""
    manager = AdaptiveRiskManager()

    liminal_state = {'state': 'critical', 'liminal_score': 0.9}
    regime = {'regime': 'transition', 'confidence': 0.5}

    adjustment = manager.calculate_liminal_adjustment(liminal_state, regime)

    # Should be heavily reduced
    assert adjustment < 0.3


def test_calculate_stop_loss_long():
    """Test stop loss calculation for long position"""
    manager = AdaptiveRiskManager()

    entry_price = 100.0
    atr = 2.0

    stop_loss = manager.calculate_stop_loss(entry_price, 'long', atr=atr)

    # Stop should be below entry for long
    assert stop_loss < entry_price
    # Should be approximately 2 * ATR below
    assert abs((entry_price - stop_loss) - (2 * atr)) < 1.0


def test_calculate_stop_loss_short():
    """Test stop loss calculation for short position"""
    manager = AdaptiveRiskManager()

    entry_price = 100.0
    atr = 2.0

    stop_loss = manager.calculate_stop_loss(entry_price, 'short', atr=atr)

    # Stop should be above entry for short
    assert stop_loss > entry_price


def test_calculate_position_size():
    """Test position sizing calculation"""
    manager = AdaptiveRiskManager()

    forecast = {
        'p_calibrated': 0.65,
        'p_raw': 0.6,
        'confidence_band': [0.55, 0.75]
    }

    liminal_state = {'state': 'stable', 'liminal_score': 0.2}
    regime = {'regime': 'bull', 'confidence': 0.8}

    sizing = manager.calculate_position_size(
        symbol='BTC',
        entry_price=50000.0,
        direction='long',
        forecast=forecast,
        liminal_state=liminal_state,
        regime=regime,
        atr=1000.0
    )

    assert isinstance(sizing, PositionSizing)
    assert sizing.recommended_size > 0.0
    assert sizing.stop_loss_price < 50000.0
    assert sizing.take_profit_price > 50000.0
    assert 0.0 <= sizing.position_pct <= 1.0


def test_calculate_position_size_max_drawdown():
    """Test position sizing when max drawdown reached"""
    manager = AdaptiveRiskManager()

    # Set drawdown to limit
    manager.current_drawdown = 0.21  # Over 20% limit

    forecast = {'p_calibrated': 0.7, 'confidence_band': [0.6, 0.8]}
    liminal_state = {'state': 'stable', 'liminal_score': 0.0}
    regime = {'regime': 'bull', 'confidence': 0.9}

    sizing = manager.calculate_position_size(
        symbol='BTC',
        entry_price=50000.0,
        direction='long',
        forecast=forecast,
        liminal_state=liminal_state,
        regime=regime
    )

    # Should halt trading
    assert sizing.recommended_size == 0.0
    assert sizing.metadata.get('halted') is True


def test_check_position_limits():
    """Test position limits checking"""
    manager = AdaptiveRiskManager()

    # Normal state
    allowed, reason = manager.check_position_limits()
    assert allowed is True

    # Max drawdown
    manager.current_drawdown = 0.25
    allowed, reason = manager.check_position_limits()
    assert allowed is False
    assert "drawdown" in reason.lower()


def test_update_metrics():
    """Test metrics update from trade results"""
    manager = AdaptiveRiskManager()

    # Simulate trades
    manager.update_metrics({'pnl': 100.0, 'win': True})
    manager.update_metrics({'pnl': -50.0, 'win': False})
    manager.update_metrics({'pnl': 150.0, 'win': True})

    assert manager.win_rate == 2.0 / 3.0
    assert manager.avg_win > 0.0
    assert manager.avg_loss > 0.0


# ===== PortfolioManager Tests =====

def test_portfolio_manager_init():
    """Test portfolio manager initialization"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    assert manager.portfolio.cash == 10000.0
    assert len(manager.portfolio.assets) == 0
    assert manager.portfolio.total_value == 10000.0


def test_asset_properties():
    """Test Asset dataclass properties"""
    asset = Asset(
        symbol='BTC',
        quantity=0.5,
        avg_cost=40000.0,
        current_price=50000.0
    )

    assert asset.market_value == 25000.0
    assert asset.unrealized_pnl == 5000.0
    assert asset.return_pct == 0.25


def test_portfolio_properties():
    """Test Portfolio properties"""
    portfolio = Portfolio(
        cash=5000.0,
        assets={
            'BTC': Asset('BTC', 0.5, 40000.0, 50000.0)
        },
        timestamp="2025-01-01T00:00:00Z"
    )

    assert portfolio.total_value == 30000.0
    assert portfolio.assets_value == 25000.0
    assert portfolio.total_pnl == 5000.0

    weights = portfolio.get_weights()
    assert abs(weights['BTC'] - 25000/30000) < 0.001
    assert abs(weights['CASH'] - 5000/30000) < 0.001


def test_update_prices():
    """Test price updates"""
    manager = AdaptivePortfolioManager()

    # Add position
    manager.portfolio.assets['BTC'] = Asset('BTC', 1.0, 40000.0, 40000.0)

    # Update prices
    manager.update_prices({'BTC': 45000.0})

    assert manager.portfolio.assets['BTC'].current_price == 45000.0


def test_calculate_target_weights():
    """Test target weight calculation"""
    manager = AdaptivePortfolioManager()

    regime = {'regime': 'bull', 'confidence': 0.8}
    liminal_state = {'liminal_score': 0.2}
    forecasts = {
        'BTC': {'p_calibrated': 0.7, 'confidence': 0.8},
        'ETH': {'p_calibrated': 0.6, 'confidence': 0.7}
    }

    weights = manager.calculate_target_weights(regime, liminal_state, forecasts)

    # Should have BTC, ETH, and CASH
    assert 'BTC' in weights
    assert 'ETH' in weights
    assert 'CASH' in weights

    # Total should be ~1.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01

    # Bull market should have higher equity allocation
    assert weights['CASH'] < 0.5


def test_calculate_target_weights_bear():
    """Test target weights in bear market"""
    manager = AdaptivePortfolioManager()

    regime = {'regime': 'bear', 'confidence': 0.8}
    liminal_state = {'liminal_score': 0.3}
    forecasts = {'BTC': {'p_calibrated': 0.4, 'confidence': 0.6}}

    weights = manager.calculate_target_weights(regime, liminal_state, forecasts)

    # Bear market should have higher cash allocation
    assert weights['CASH'] > 0.5


def test_execute_trade_buy():
    """Test trade execution - buy"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    success = manager.execute_trade('BTC', 'buy', 0.1, 50000.0)

    assert success is True
    assert manager.portfolio.cash == 5000.0
    assert 'BTC' in manager.portfolio.assets
    assert manager.portfolio.assets['BTC'].quantity == 0.1


def test_execute_trade_sell():
    """Test trade execution - sell"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    # Buy first
    manager.execute_trade('BTC', 'buy', 0.1, 50000.0)

    # Then sell
    success = manager.execute_trade('BTC', 'sell', 0.05, 55000.0)

    assert success is True
    assert manager.portfolio.cash == 5000.0 + 0.05 * 55000.0
    assert manager.portfolio.assets['BTC'].quantity == 0.05


def test_execute_trade_insufficient_cash():
    """Test trade with insufficient cash"""
    manager = AdaptivePortfolioManager(initial_cash=1000.0)

    success = manager.execute_trade('BTC', 'buy', 1.0, 50000.0)

    assert success is False
    assert manager.portfolio.cash == 1000.0


def test_check_rebalance_needed():
    """Test rebalance check"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    # Set target weights
    manager.target_weights = {'BTC': 0.5, 'CASH': 0.5}

    # Current: 100% cash, target: 50% cash
    recommendations = manager.check_rebalance_needed()

    # Should recommend buying BTC
    assert len(recommendations) > 0
    btc_rec = [r for r in recommendations if r.symbol == 'BTC'][0]
    assert btc_rec.action == 'buy'


def test_rebalance():
    """Test portfolio rebalancing"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    regime = {'regime': 'bull', 'confidence': 0.8}
    liminal_state = {'liminal_score': 0.1}
    forecasts = {'BTC': {'p_calibrated': 0.7, 'confidence': 0.8}}
    prices = {'BTC': 50000.0}

    trades = manager.rebalance(regime, liminal_state, forecasts, prices)

    # Should have executed some trades
    assert len(manager.portfolio.assets) > 0


def test_get_performance():
    """Test performance metrics"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    # Buy BTC
    manager.execute_trade('BTC', 'buy', 0.2, 50000.0)

    # Update price (profit)
    manager.update_prices({'BTC': 55000.0})

    performance = manager.get_performance()

    assert performance['total_value'] > 10000.0
    assert performance['total_return'] > 0.0
    assert performance['total_pnl'] > 0.0


def test_get_holdings():
    """Test holdings summary"""
    manager = AdaptivePortfolioManager(initial_cash=10000.0)

    # Buy BTC
    manager.execute_trade('BTC', 'buy', 0.1, 50000.0)

    # Update price
    manager.update_prices({'BTC': 55000.0})

    holdings = manager.get_holdings()

    assert 'BTC' in holdings
    assert 'CASH' in holdings
    assert holdings['BTC']['quantity'] == 0.1
    assert holdings['BTC']['unrealized_pnl'] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
