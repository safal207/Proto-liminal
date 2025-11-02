"""
Module: portfolio_manager.py
Purpose: Adaptive portfolio management with regime-based rebalancing
Part of LIMINAL ProtoConsciousness — Adaptive Finance Framework
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Asset:
    """Represents an asset in the portfolio"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def return_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class Portfolio:
    """Represents the current portfolio state"""
    cash: float
    assets: Dict[str, Asset]
    timestamp: str

    @property
    def total_value(self) -> float:
        return self.cash + sum(asset.market_value for asset in self.assets.values())

    @property
    def assets_value(self) -> float:
        return sum(asset.market_value for asset in self.assets.values())

    @property
    def total_pnl(self) -> float:
        return sum(asset.unrealized_pnl for asset in self.assets.values())

    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        total = self.total_value
        if total == 0:
            return {}

        weights = {}
        for symbol, asset in self.assets.items():
            weights[symbol] = asset.market_value / total
        weights['CASH'] = self.cash / total

        return weights


@dataclass
class RebalanceRecommendation:
    """Recommendation for portfolio rebalancing"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    current_weight: float
    target_weight: float
    delta_weight: float
    delta_value: float  # Amount to buy/sell
    reason: str
    confidence: float

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "delta_weight": round(self.delta_weight, 4),
            "delta_value": round(self.delta_value, 2),
            "reason": self.reason,
            "confidence": round(self.confidence, 4)
        }


class AdaptivePortfolioManager:
    """Adaptive portfolio manager with regime-based allocation"""

    def __init__(
        self,
        initial_cash: float = 10000.0,
        rebalance_threshold: float = 0.05,  # 5% deviation triggers rebalance
        max_positions: int = 10
    ):
        """
        Initialize portfolio manager

        Args:
            initial_cash: Starting cash
            rebalance_threshold: Weight deviation threshold for rebalancing
            max_positions: Maximum number of positions
        """
        self.portfolio = Portfolio(
            cash=initial_cash,
            assets={},
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

        self.initial_value = initial_cash
        self.rebalance_threshold = rebalance_threshold
        self.max_positions = max_positions

        # Target weights (updated based on regime)
        self.target_weights = {}

        # Regime-based allocation strategies
        self.regime_strategies = {
            'bull': {'equity': 0.70, 'cash': 0.30},
            'bear': {'equity': 0.30, 'cash': 0.70},
            'sideways': {'equity': 0.50, 'cash': 0.50},
            'transition': {'equity': 0.40, 'cash': 0.60}
        }

        # Performance tracking
        self.trade_history = []
        self.rebalance_history = []

        LOGGER.info(f"AdaptivePortfolioManager initialized (cash: ${initial_cash:.2f})")

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all assets

        Args:
            prices: Dict of symbol -> price
        """
        for symbol, price in prices.items():
            if symbol in self.portfolio.assets:
                self.portfolio.assets[symbol].current_price = price

        self.portfolio.timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    def calculate_target_weights(
        self,
        regime: Dict,
        liminal_state: Dict,
        forecasts: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Calculate target portfolio weights based on regime and forecasts

        Args:
            regime: Regime classification dict
            liminal_state: Liminal state dict
            forecasts: Dict of symbol -> forecast

        Returns:
            Dict of symbol -> target weight
        """
        regime_type = regime.get('regime', 'sideways')
        liminal_score = liminal_state.get('liminal_score', 0.0)

        # Get base strategy for regime
        base_strategy = self.regime_strategies.get(regime_type, self.regime_strategies['sideways'])

        # Adjust for liminal state (increase cash in uncertain times)
        equity_allocation = base_strategy['equity'] * (1.0 - liminal_score * 0.5)
        cash_allocation = 1.0 - equity_allocation

        # Distribute equity allocation among assets based on forecasts
        target_weights = {'CASH': cash_allocation}

        if not forecasts:
            return target_weights

        # Score each asset
        scores = {}
        for symbol, forecast in forecasts.items():
            probability = forecast.get('p_calibrated', forecast.get('p_raw', 0.5))
            confidence = forecast.get('confidence', 0.5)

            # Score = probability * confidence
            scores[symbol] = probability * confidence

        # Normalize scores
        total_score = sum(scores.values())
        if total_score == 0:
            return target_weights

        # Allocate equity portion based on scores
        for symbol, score in scores.items():
            target_weights[symbol] = (score / total_score) * equity_allocation

        return target_weights

    def check_rebalance_needed(self) -> List[RebalanceRecommendation]:
        """
        Check if rebalancing is needed

        Returns:
            List of rebalance recommendations
        """
        if not self.target_weights:
            return []

        current_weights = self.portfolio.get_weights()
        recommendations = []

        total_value = self.portfolio.total_value

        for symbol in set(list(current_weights.keys()) + list(self.target_weights.keys())):
            if symbol == 'CASH':
                continue

            current_weight = current_weights.get(symbol, 0.0)
            target_weight = self.target_weights.get(symbol, 0.0)
            delta_weight = target_weight - current_weight

            # Check if deviation exceeds threshold
            if abs(delta_weight) > self.rebalance_threshold:
                delta_value = delta_weight * total_value

                if delta_weight > 0:
                    action = 'buy'
                    reason = f"Underweight by {abs(delta_weight):.1%}"
                else:
                    action = 'sell'
                    reason = f"Overweight by {abs(delta_weight):.1%}"

                recommendations.append(RebalanceRecommendation(
                    symbol=symbol,
                    action=action,
                    current_weight=current_weight,
                    target_weight=target_weight,
                    delta_weight=delta_weight,
                    delta_value=delta_value,
                    reason=reason,
                    confidence=0.8  # Default confidence
                ))

        return recommendations

    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float
    ) -> bool:
        """
        Execute a trade

        Args:
            symbol: Trading symbol
            action: 'buy' or 'sell'
            quantity: Number of units
            price: Execution price

        Returns:
            True if successful
        """
        cost = quantity * price

        if action.lower() == 'buy':
            # Check cash availability
            if cost > self.portfolio.cash:
                LOGGER.warning(f"Insufficient cash for {symbol} buy: need ${cost:.2f}, have ${self.portfolio.cash:.2f}")
                return False

            # Deduct cash
            self.portfolio.cash -= cost

            # Add or update position
            if symbol in self.portfolio.assets:
                asset = self.portfolio.assets[symbol]
                # Update average cost
                total_cost = asset.avg_cost * asset.quantity + cost
                total_qty = asset.quantity + quantity
                asset.avg_cost = total_cost / total_qty if total_qty > 0 else price
                asset.quantity = total_qty
                asset.current_price = price
            else:
                self.portfolio.assets[symbol] = Asset(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=price,
                    current_price=price
                )

            LOGGER.info(f"BUY {quantity} {symbol} @ ${price:.2f} = ${cost:.2f}")

        elif action.lower() == 'sell':
            # Check position exists
            if symbol not in self.portfolio.assets:
                LOGGER.warning(f"No position in {symbol} to sell")
                return False

            asset = self.portfolio.assets[symbol]

            # Check quantity
            if quantity > asset.quantity:
                LOGGER.warning(f"Insufficient {symbol} quantity: need {quantity}, have {asset.quantity}")
                return False

            # Add cash
            self.portfolio.cash += cost

            # Update or remove position
            asset.quantity -= quantity
            if asset.quantity <= 0:
                del self.portfolio.assets[symbol]

            LOGGER.info(f"SELL {quantity} {symbol} @ ${price:.2f} = ${cost:.2f}")

        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': cost
        })

        return True

    def rebalance(
        self,
        regime: Dict,
        liminal_state: Dict,
        forecasts: Dict[str, Dict],
        prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Rebalance portfolio based on regime and forecasts

        Args:
            regime: Regime classification
            liminal_state: Liminal state
            forecasts: Forecasts for symbols
            prices: Current prices

        Returns:
            List of executed trades
        """
        # Update prices
        self.update_prices(prices)

        # Calculate target weights
        self.target_weights = self.calculate_target_weights(regime, liminal_state, forecasts)

        # Check if rebalancing needed
        recommendations = self.check_rebalance_needed()

        if not recommendations:
            LOGGER.info("No rebalancing needed")
            return []

        LOGGER.info(f"Rebalancing portfolio ({len(recommendations)} changes)")

        # Execute trades
        executed_trades = []

        for rec in recommendations:
            symbol = rec.symbol
            price = prices.get(symbol)

            if price is None:
                LOGGER.warning(f"No price for {symbol}, skipping")
                continue

            quantity = abs(rec.delta_value) / price

            success = self.execute_trade(symbol, rec.action, quantity, price)

            if success:
                executed_trades.append(rec.to_dict())

        # Record rebalance event
        self.rebalance_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'regime': regime.get('regime'),
            'liminal_score': liminal_state.get('liminal_score'),
            'trades': len(executed_trades)
        })

        return executed_trades

    def get_performance(self) -> Dict:
        """Get portfolio performance metrics"""
        total_value = self.portfolio.total_value
        total_return = (total_value - self.initial_value) / self.initial_value

        # Calculate Sharpe-like metric (simplified)
        if len(self.rebalance_history) > 1:
            values = [self.initial_value]
            for i, event in enumerate(self.rebalance_history):
                # Estimate value at each rebalance (simplified)
                values.append(total_value)

            returns = np.diff(values) / values[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            "total_value": round(total_value, 2),
            "cash": round(self.portfolio.cash, 2),
            "assets_value": round(self.portfolio.assets_value, 2),
            "total_return": round(total_return, 4),
            "total_return_pct": round(total_return * 100, 2),
            "total_pnl": round(self.portfolio.total_pnl, 2),
            "sharpe_ratio": round(sharpe, 2),
            "trades_count": len(self.trade_history),
            "rebalances_count": len(self.rebalance_history),
            "positions_count": len(self.portfolio.assets)
        }

    def get_holdings(self) -> Dict:
        """Get current holdings summary"""
        weights = self.portfolio.get_weights()

        holdings = {}
        for symbol, asset in self.portfolio.assets.items():
            holdings[symbol] = {
                "quantity": round(asset.quantity, 4),
                "avg_cost": round(asset.avg_cost, 2),
                "current_price": round(asset.current_price, 2),
                "market_value": round(asset.market_value, 2),
                "unrealized_pnl": round(asset.unrealized_pnl, 2),
                "return_pct": round(asset.return_pct * 100, 2),
                "weight": round(weights.get(symbol, 0.0), 4)
            }

        holdings['CASH'] = {
            "amount": round(self.portfolio.cash, 2),
            "weight": round(weights.get('CASH', 0.0), 4)
        }

        return holdings
