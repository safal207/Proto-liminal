"""
Module: risk_manager.py
Purpose: Adaptive risk management with Kelly Criterion and liminal adjustment
Part of LIMINAL ProtoConsciousness — Adaptive Finance Framework
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_portfolio_risk: float = 0.10  # 10% max total portfolio risk
    max_drawdown_limit: float = 0.20  # 20% max drawdown before halt
    kelly_fraction: float = 0.25  # Use 1/4 Kelly for safety
    min_position_size: float = 0.01  # Minimum 1% position
    max_position_size: float = 0.20  # Maximum 20% position
    stop_loss_atr_multiplier: float = 2.0  # Stop loss at 2x ATR
    trailing_stop_enabled: bool = True


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    symbol: str
    recommended_size: float  # Position size in base currency
    risk_amount: float  # Amount at risk (stop loss distance)
    position_pct: float  # Percentage of portfolio
    stop_loss_price: Optional[float]  # Stop loss level
    take_profit_price: Optional[float]  # Take profit level
    risk_reward_ratio: float  # Risk/reward ratio
    kelly_size: float  # Kelly criterion size
    adjusted_size: float  # After liminal adjustment
    liminal_adjustment: float  # Adjustment factor applied
    confidence: float  # Confidence in recommendation
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "recommended_size": round(self.recommended_size, 2),
            "risk_amount": round(self.risk_amount, 2),
            "position_pct": round(self.position_pct, 4),
            "stop_loss_price": round(self.stop_loss_price, 2) if self.stop_loss_price else None,
            "take_profit_price": round(self.take_profit_price, 2) if self.take_profit_price else None,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "kelly_size": round(self.kelly_size, 2),
            "adjusted_size": round(self.adjusted_size, 2),
            "liminal_adjustment": round(self.liminal_adjustment, 4),
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata
        }


class AdaptiveRiskManager:
    """Adaptive risk management with liminal state awareness"""

    def __init__(self, params: Optional[RiskParameters] = None):
        """
        Initialize risk manager

        Args:
            params: Risk parameters (uses defaults if None)
        """
        self.params = params or RiskParameters()

        # Track portfolio state
        self.equity = 10000.0  # Default starting equity
        self.peak_equity = 10000.0
        self.current_drawdown = 0.0
        self.open_positions = {}
        self.trade_history = []

        # Risk metrics
        self.total_risk_exposure = 0.0
        self.win_rate = 0.5  # Default 50%
        self.avg_win = 0.0
        self.avg_loss = 0.0

        LOGGER.info(f"AdaptiveRiskManager initialized (Kelly fraction: {self.params.kelly_fraction})")

    def update_equity(self, equity: float):
        """
        Update current equity and recalculate metrics

        Args:
            equity: Current portfolio equity
        """
        self.equity = equity

        # Track peak for drawdown calculation
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Calculate current drawdown
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity

        LOGGER.debug(f"Equity updated: ${equity:.2f} (drawdown: {self.current_drawdown:.2%})")

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion for optimal position size

        Formula: f* = (p*b - q) / b
        where:
          p = win rate
          q = loss rate (1-p)
          b = avg_win / avg_loss

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        # Avoid division by zero
        if abs(avg_win) < 1e-6 or abs(avg_loss) < 1e-6:
            return 0.0

        # Kelly formula
        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win / avg_loss)

        kelly = (p * b - q) / b

        # Kelly can be negative (don't trade) or >1 (unrealistic)
        kelly = max(0.0, min(1.0, kelly))

        # Apply fractional Kelly for safety
        fractional_kelly = kelly * self.params.kelly_fraction

        return fractional_kelly

    def calculate_kelly_from_forecast(
        self,
        probability: float,
        confidence_lower: float,
        confidence_upper: float
    ) -> float:
        """
        Calculate Kelly from forecast probabilities

        Args:
            probability: Predicted probability of success
            confidence_lower: Lower bound of confidence interval
            confidence_upper: Upper bound of confidence interval

        Returns:
            Kelly fraction
        """
        # Use probability as win rate
        win_rate = probability

        # Estimate avg win/loss from confidence interval
        # Wider interval = more uncertainty = smaller kelly
        ci_width = confidence_upper - confidence_lower
        avg_win = 1.0 - ci_width  # Narrower CI = higher potential
        avg_loss = 1.0

        kelly = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)

        return kelly

    def calculate_liminal_adjustment(
        self,
        liminal_state: Dict,
        regime: Dict
    ) -> float:
        """
        Calculate position size adjustment based on liminal state

        Args:
            liminal_state: Liminal state dict with 'state' and 'liminal_score'
            regime: Regime dict with 'regime' and 'confidence'

        Returns:
            Adjustment multiplier (0.0 to 1.0)
        """
        adjustment = 1.0

        # Liminal state adjustment
        state = liminal_state.get('state', 'stable')
        liminal_score = liminal_state.get('liminal_score', 0.0)

        if state == 'critical':
            # Critical state: reduce to 20%
            adjustment *= 0.2
        elif state == 'liminal':
            # Liminal state: reduce to 50%
            adjustment *= 0.5
        else:
            # Stable: small reduction based on score
            adjustment *= (1.0 - liminal_score * 0.3)

        # Regime adjustment
        regime_type = regime.get('regime', 'unknown')
        regime_confidence = regime.get('confidence', 0.5)

        if regime_type == 'transition':
            # Transitions are risky
            adjustment *= 0.6
        elif regime_type == 'sideways':
            # Sideways = less opportunity
            adjustment *= 0.8

        # Confidence adjustment
        if regime_confidence < 0.5:
            adjustment *= regime_confidence

        # Ensure reasonable bounds
        adjustment = max(0.1, min(1.0, adjustment))

        return adjustment

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate dynamic stop loss

        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range (optional)
            volatility: Volatility measure (optional)

        Returns:
            Stop loss price
        """
        # Use ATR if available, otherwise use volatility
        if atr is not None:
            distance = atr * self.params.stop_loss_atr_multiplier
        elif volatility is not None:
            # Use volatility as percentage
            distance = entry_price * volatility * 2.0
        else:
            # Default: 2% stop
            distance = entry_price * 0.02

        if direction.lower() == 'long':
            stop_loss = entry_price - distance
        else:  # short
            stop_loss = entry_price + distance

        return stop_loss

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        forecast: Dict,
        liminal_state: Dict,
        regime: Dict,
        atr: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> PositionSizing:
        """
        Calculate optimal position size with adaptive risk management

        Args:
            symbol: Trading symbol
            entry_price: Intended entry price
            direction: 'long' or 'short'
            forecast: Forecast dict with probability and confidence
            liminal_state: Liminal state dict
            regime: Regime classification dict
            atr: Average True Range (optional)
            volatility: Volatility measure (optional)

        Returns:
            PositionSizing recommendation
        """
        # Check drawdown limit
        if self.current_drawdown >= self.params.max_drawdown_limit:
            LOGGER.warning(f"Max drawdown reached: {self.current_drawdown:.2%}")
            return PositionSizing(
                symbol=symbol,
                recommended_size=0.0,
                risk_amount=0.0,
                position_pct=0.0,
                stop_loss_price=None,
                take_profit_price=None,
                risk_reward_ratio=0.0,
                kelly_size=0.0,
                adjusted_size=0.0,
                liminal_adjustment=0.0,
                confidence=0.0,
                metadata={"halted": True, "reason": "max_drawdown"}
            )

        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(entry_price, direction, atr, volatility)
        risk_per_unit = abs(entry_price - stop_loss)

        # Calculate Kelly size
        probability = forecast.get('p_calibrated', forecast.get('p_raw', 0.5))
        ci_lower, ci_upper = forecast.get('confidence_band', [0.3, 0.7])

        kelly_fraction = self.calculate_kelly_from_forecast(probability, ci_lower, ci_upper)

        # Base position size from Kelly
        kelly_size = self.equity * kelly_fraction

        # Liminal adjustment
        liminal_adj = self.calculate_liminal_adjustment(liminal_state, regime)

        # Apply liminal adjustment
        adjusted_size = kelly_size * liminal_adj

        # Apply position limits
        max_size = self.equity * self.params.max_position_size
        min_size = self.equity * self.params.min_position_size

        adjusted_size = max(min_size, min(max_size, adjusted_size))

        # Calculate units
        units = adjusted_size / entry_price if entry_price > 0 else 0

        # Risk amount
        risk_amount = units * risk_per_unit

        # Check risk limits
        risk_pct = risk_amount / self.equity
        if risk_pct > self.params.max_risk_per_trade:
            # Scale down to meet risk limit
            scale = self.params.max_risk_per_trade / risk_pct
            adjusted_size *= scale
            risk_amount *= scale

        # Position percentage
        position_pct = adjusted_size / self.equity

        # Calculate take profit (2:1 risk/reward default)
        risk_reward_ratio = 2.0
        take_profit_distance = risk_per_unit * risk_reward_ratio

        if direction.lower() == 'long':
            take_profit = entry_price + take_profit_distance
        else:
            take_profit = entry_price - take_profit_distance

        # Confidence in recommendation
        confidence = probability * (1.0 - liminal_state.get('liminal_score', 0.0))

        return PositionSizing(
            symbol=symbol,
            recommended_size=adjusted_size,
            risk_amount=risk_amount,
            position_pct=position_pct,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            kelly_size=kelly_size,
            adjusted_size=adjusted_size,
            liminal_adjustment=liminal_adj,
            confidence=confidence,
            metadata={
                "entry_price": entry_price,
                "direction": direction,
                "units": round(units, 4),
                "risk_pct": round(risk_pct, 4),
                "kelly_fraction": round(kelly_fraction, 4),
                "probability": round(probability, 4),
                "liminal_state": liminal_state.get('state', 'unknown'),
                "regime": regime.get('regime', 'unknown')
            }
        )

    def check_position_limits(self) -> Tuple[bool, str]:
        """
        Check if new positions are allowed

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Check drawdown
        if self.current_drawdown >= self.params.max_drawdown_limit:
            return False, f"Max drawdown exceeded: {self.current_drawdown:.2%}"

        # Check total risk exposure
        if self.total_risk_exposure >= self.params.max_portfolio_risk:
            return False, f"Max portfolio risk exceeded: {self.total_risk_exposure:.2%}"

        return True, "OK"

    def update_metrics(self, trade_result: Dict):
        """
        Update risk metrics from completed trade

        Args:
            trade_result: Dict with 'pnl', 'win', etc.
        """
        self.trade_history.append(trade_result)

        # Recalculate win rate
        wins = [t for t in self.trade_history if t.get('win', False)]
        self.win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0.5

        # Recalculate avg win/loss
        win_amounts = [t['pnl'] for t in wins]
        loss_amounts = [abs(t['pnl']) for t in self.trade_history if not t.get('win', False)]

        self.avg_win = np.mean(win_amounts) if win_amounts else 0.0
        self.avg_loss = np.mean(loss_amounts) if loss_amounts else 0.0

        LOGGER.info(f"Metrics updated: win_rate={self.win_rate:.2%}, avg_win=${self.avg_win:.2f}, avg_loss=${self.avg_loss:.2f}")

    def get_stats(self) -> Dict:
        """Get risk management statistics"""
        return {
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "current_drawdown": round(self.current_drawdown, 4),
            "total_risk_exposure": round(self.total_risk_exposure, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "trades_count": len(self.trade_history),
            "open_positions": len(self.open_positions)
        }
