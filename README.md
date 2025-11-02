# LIMINAL ProtoConsciousness

LIMINAL ProtoConsciousness is an experimental platform exploring emergent analytical behaviors and adaptive learning signals.
This repository contains the foundational modules outlined in the MVP specification to begin iterating on proto-conscious processing loops.

LIMINAL ProtoConsciousness — the living analytical seed that learns through reflection.

See the [MVP specification](docs/MVP_SPEC.md) for detailed requirements and roadmap guidance.

## Features

- **RSS Data Collection**: Automated news aggregation from multiple sources
- **Signal Extraction**: Convert raw data into analytical signals
- **RINSE Cycle**: Reflect-Integrate-Normalize-Simulate-Evolve feedback loop
- **LiminalBD Integration**: Connect to living cellular substrate for adaptive processing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/safal207/Proto-liminal.git
cd Proto-liminal

# Install dependencies
pip install -r requirements.txt
```

### Collect news (RSS → JSONL)
```bash
python src/collector.py --feeds configs/feeds.txt --out data/raw/news_$(date +%Y%m%d).jsonl --min-length 40 --max-items 1000
```

### Normalize news (raw → clean)
```bash
python src/normalizer.py --inp 'data/raw/news_*.jsonl' --out data/clean/news_norm_$(date +%Y%m%d).jsonl --allow-lang en,ru --min-length 80 --dedup
```

### Build features (24h window)
```bash
python src/features_builder.py --inp 'data/clean/news_norm_*.jsonl' --out data/features/news_features.parquet --window-hours 24
```

### Predict (baseline)
```bash
python src/predictor.py --features data/features/news_features.parquet --out data/forecast/forecast_$(date +%Y%m%d%H).jsonl --event BTC_UP_24H_GT_2PCT
```

### Detect liminal states (transition detection)
```bash
python examples/demo_liminal_detection.py
```

## Liminal State Detection

Proto-liminal includes **adaptive risk management** through real-time detection of **liminal (transitional) market states**.

**What are liminal states?**
- 🌊 Moments of transition between market regimes (bull → bear, stable → volatile)
- ⚠️ Periods of uncertainty where traditional models break down
- 🎭 Phase transitions that require adaptive strategy adjustment

**How it works:**
```python
from liminal_detector import LiminalDetector
from market_regime import MarketRegimeClassifier

# Initialize detectors
liminal_detector = LiminalDetector()
regime_classifier = MarketRegimeClassifier()

# Detect current state
liminal_state = liminal_detector.detect(
    sentiment=0.3,
    volatility=0.8,  # High volatility
    volume=250       # Volume spike
)

# Classify regime
regime = regime_classifier.classify(price=45000, sentiment=0.3)

# Adaptive risk management
if liminal_state.state == "critical":
    position_size *= 0.2  # Reduce risk by 80%
elif liminal_state.state == "liminal":
    position_size *= 0.5  # Reduce risk by 50%
```

**Signals detected:**
- 📊 **Volatility spikes** — sudden changes in price movement
- 🔄 **Sentiment flips** — rapid reversal in market mood
- 📈 **Volume anomalies** — unusual trading/news activity
- ⚡ **Indicator conflicts** — contradictory signals

**Regimes classified:**
- 📈 **Bull** — upward trending market
- 📉 **Bear** — downward trending market
- ↔️ **Sideways** — range-bound market
- 🔄 **Transition** — regime change in progress

## Adaptive Risk Management

Proto-liminal features **intelligent risk management** that adapts to market conditions in real-time.

**Key Features:**
- 💎 **Kelly Criterion** — optimal position sizing based on probabilities
- 🎯 **Liminal Adjustment** — dynamic risk reduction during transitions
- 🛡️ **Circuit Breakers** — automatic trading halt at max drawdown
- 📊 **Regime-Based Allocation** — portfolio weights adapt to market regime
- ⚡ **Dynamic Stop Loss** — ATR-based stops that adjust to volatility

**How it works:**
```python
from risk_manager import AdaptiveRiskManager, RiskParameters
from portfolio_manager import AdaptivePortfolioManager

# Initialize risk manager
risk_params = RiskParameters(
    max_risk_per_trade=0.02,  # 2% max risk per trade
    kelly_fraction=0.25,       # Use 1/4 Kelly for safety
    max_drawdown_limit=0.20    # Halt at 20% drawdown
)
risk_manager = AdaptiveRiskManager(params=risk_params)

# Calculate position size with liminal adjustment
position_sizing = risk_manager.calculate_position_size(
    symbol='BTC',
    entry_price=50000.0,
    direction='long',
    forecast=forecast,           # From predictor
    liminal_state=liminal_state, # From detector
    regime=regime,               # From classifier
    atr=1000.0
)

print(f"Kelly size: ${position_sizing.kelly_size:,.2f}")
print(f"Adjusted size: ${position_sizing.adjusted_size:,.2f}")
print(f"Liminal adjustment: {position_sizing.liminal_adjustment:.2f}x")
print(f"Stop loss: ${position_sizing.stop_loss_price:,.2f}")

# Portfolio management with regime-based allocation
portfolio = AdaptivePortfolioManager(initial_cash=10000.0)

# Rebalance based on regime
trades = portfolio.rebalance(
    regime=regime,
    liminal_state=liminal_state,
    forecasts={'BTC': forecast, 'ETH': forecast},
    prices={'BTC': 50000.0, 'ETH': 3000.0}
)
```

**Risk Adjustment Matrix:**

| State / Regime | Stable | Liminal | Critical |
|----------------|--------|---------|----------|
| **Bull**       | 100%   | 50%     | 20%      |
| **Sideways**   | 80%    | 40%     | 20%      |
| **Transition** | 60%    | 30%     | 20%      |
| **Bear**       | 50%    | 30%     | 20%      |

**Portfolio Allocation by Regime:**

| Regime       | Equity | Cash |
|--------------|--------|------|
| **Bull**     | 70%    | 30%  |
| **Bear**     | 30%    | 70%  |
| **Sideways** | 50%    | 50%  |
| **Transition** | 40%  | 60%  |

**Run adaptive risk demo:**
```bash
python examples/demo_adaptive_risk.py
```

## Real-Time Market Monitoring

Proto-liminal supports real-time market monitoring through integration with **Tradernet WebSocket API**, enabling live liminal state detection and adaptive risk management.

### Features

- 📡 **Live Quote Streaming**: Real-time market data from Tradernet WebSocket (`wss://wssdev.tradernet.dev`)
- 🔍 **Real-Time Liminal Detection**: Detect market transitions as they happen
- 🎯 **Market Regime Classification**: Continuous bull/bear/sideways/transition detection
- ⚠️ **Critical State Alerts**: Automatic alerts when markets enter critical liminal states
- 📊 **Multi-Symbol Monitoring**: Track multiple assets simultaneously
- 💾 **JSONL Logging**: All snapshots and alerts saved for analysis

### Quick Start

**Test WebSocket connection:**
```bash
python examples/test_tradernet_connection.py
```

**Start real-time monitor:**
```bash
# Monitor default symbols (AAPL, TSLA, BTCUSD)
python src/realtime_monitor.py

# Monitor custom symbols
python src/realtime_monitor.py --symbols AAPL MSFT GOOGL BTCUSD ETHUSD

# Disable quote logging
python src/realtime_monitor.py --no-log

# Verbose mode
python src/realtime_monitor.py --symbols TSLA BTCUSD --verbose
```

### Python API

```python
import asyncio
from tradernet_client import TradernetClient, TradernetConfig

# Configure client
config = TradernetConfig(
    url="wss://wssdev.tradernet.dev",
    symbols=["AAPL", "TSLA", "BTCUSD"]
)

client = TradernetClient(config)

# Callback for quotes
def on_quote(quote):
    print(f"{quote.symbol}: ${quote.price:.2f}")

client.register_quote_callback(on_quote)

# Run with auto-reconnect
await client.run_with_reconnect()
```

### Output Files

Monitor generates two JSONL files in `data/`:

**1. `realtime_snapshots.jsonl`** - All market snapshots:
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-11-02T14:30:00Z",
  "price": 178.45,
  "liminal_state": "liminal",
  "liminal_score": 0.72,
  "market_regime": "transition",
  "regime_confidence": 0.68,
  "volatility": 0.0234,
  "risk_adjustment": 0.35,
  "alert_level": "warning"
}
```

**2. `realtime_alerts.jsonl`** - Critical state alerts:
```json
{
  "timestamp": "2025-11-02T14:32:15Z",
  "symbol": "BTCUSD",
  "alert_type": "CRITICAL_LIMINAL_STATE",
  "details": {
    "liminal_score": 0.87,
    "regime": "transition",
    "volatility": 0.0456,
    "risk_adjustment": 0.20,
    "recommendation": "REDUCE_EXPOSURE"
  }
}
```

### Console Output

Real-time monitor displays colored status for each quote:

```
🟢 AAPL     $  178.45 | 🐂 bull       | Liminal: 0.12 | Vol: 0.0123 | Risk Adj: 1.00x
🟡 TSLA     $  245.67 | 🔄 transition | Liminal: 0.68 | Vol: 0.0345 | Risk Adj: 0.35x
🔴 BTCUSD   $42567.89 | 🐻 bear       | Liminal: 0.89 | Vol: 0.0523 | Risk Adj: 0.20x
```

- 🟢 **Stable state** - Normal market conditions
- 🟡 **Liminal state** - Warning, increased uncertainty
- 🔴 **Critical state** - High risk, major transition likely

### Architecture

```
Tradernet WebSocket (wss://wssdev.tradernet.dev)
           ↓
   TradernetClient (src/tradernet_client.py)
           ↓
   RealtimeMonitor (src/realtime_monitor.py)
           ↓
   ┌──────────────────────────────────────────┐
   │  LiminalDetector   MarketRegime          │
   │  RiskManager       Portfolio             │
   └──────────────────────────────────────────┘
           ↓
   JSONL Logs + Console Output + Alerts
```

### Integration with Existing Pipeline

Real-time data can feed into the existing RINSE cycle:

```python
from realtime_monitor import RealtimeMonitor
from rinse_agent import RinseAgent

# Start real-time monitor
monitor = RealtimeMonitor(symbols=["BTCUSD"])

# Connect to RINSE agent
rinse = RinseAgent()

async def on_snapshot(snapshot):
    # Feed real-time data into RINSE cycle
    if snapshot.liminal_state == "critical":
        await rinse.reflect_on_state(snapshot)

await monitor.run()
```

## LiminalBD Integration

Proto-liminal can integrate with [LiminalBD](https://github.com/safal207/LiminalBD) to leverage living cellular substrate for adaptive signal processing.

### Setup Integration

1. Install LiminalBD and ensure `liminal-cli` is in your PATH
2. Install integration dependencies:
   ```bash
   pip install cbor2 websockets
   ```

3. Test the integration:
   ```bash
   python examples/test_liminalbd_integration.py
   ```

### Send Signals to LiminalBD

```python
from liminal_bridge import LiminalBridge, create_signal_from_news

# Create bridge
bridge = LiminalBridge()

# Create and send signal
signal = create_signal_from_news(
    entity="Bitcoin Price",
    sentiment=0.8,
    relevance=0.9,
    urgency=0.7
)

bridge.send_signal(signal)
```

### Real-time Demo

Stream news signals to LiminalBD:

```bash
python examples/demo_realtime_integration.py --news data/raw/news_latest.jsonl
```

See [Integration Guide](docs/LIMINALBD_INTEGRATION.md) for detailed documentation.

## License

This project is licensed under the LIMINAL ProtoConsciousness License (LPL). See [LICENSE](LICENSE) for details.
