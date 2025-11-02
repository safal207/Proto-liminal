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
