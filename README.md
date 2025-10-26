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
