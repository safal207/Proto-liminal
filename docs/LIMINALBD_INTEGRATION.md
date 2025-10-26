# LiminalBD Integration Guide

## Overview

This document describes the integration between **Proto-liminal** (Python-based proto-consciousness analytics) and **LiminalBD** (Rust-based cellular substrate with living lifecycle management).

## Architecture

```
┌─────────────────────────────────────────────┐
│         Proto-liminal (Python)              │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │Collector │ → │Signal    │ → │Predictor│ │
│  │(RSS)     │   │Extractor │   │         │ │
│  └──────────┘   └──────────┘   └─────────┘ │
│                       ↓                      │
│                  ┌─────────┐                 │
│                  │Liminal  │                 │
│                  │Bridge   │                 │
│                  └─────────┘                 │
└──────────────────────┬──────────────────────┘
                       ↓ CBOR/WebSocket
┌──────────────────────┴──────────────────────┐
│         LiminalBD (Rust)                    │
│  ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │Impulses  │ → │Cell Life │ → │Harmony  │ │
│  │          │   │Cycle     │   │Loop     │ │
│  └──────────┘   └──────────┘   └─────────┘ │
│                       ↑                      │
│                  ┌─────────┐                 │
│                  │  TRS    │                 │
│                  │  State  │                 │
│                  └─────────┘                 │
└─────────────────────────────────────────────┘
```

## Conceptual Alignment

| Proto-liminal | LiminalBD | Integration |
|---------------|-----------|-------------|
| **Signals** (analytical events) | **Impulses** (cellular stimuli) | Signals → Impulses |
| **RINSE Cycle** (Reflect-Integrate-Normalize-Simulate-Evolve) | **Harmony Loop** (TRS feedback control) | Bidirectional sync |
| **Forecasts** (probabilistic predictions) | **Cellular Models** (living entities) | Forecasts as cells |
| **Calibration** (model tuning) | **Metabolism** (cell energy) | Calibration → Metabolism |

## Components

### 1. LiminalBridge (`src/liminal_bridge.py`)

The core integration module providing:

- **Impulse Creation**: Convert Proto-liminal signals to LiminalBD impulses
- **CBOR Communication**: Send impulses via CBOR protocol
- **Event Listening**: Receive LiminalBD events via WebSocket
- **Statistics Tracking**: Monitor integration health

### 2. Signal → Impulse Mapping

```python
Signal(
    entity="Bitcoin Price",
    features={"sentiment": 0.8, "relevance": 0.9},
    signal_strength=0.85
)
↓
Impulse(
    kind=ImpulseKind.AFFECT,
    pattern="bitcoin/price/sentiment",
    strength=0.85,
    ttl_ms=3600000,
    tags=["proto-liminal", "bitcoin", "price", "sentiment"]
)
```

### 3. Pattern Construction

Patterns are built from entity and features:

- Entity: `"Bitcoin Price"` → `"bitcoin/price"`
- Top feature: `"sentiment"` → `"/sentiment"`
- Final pattern: `"bitcoin/price/sentiment"`

This allows LiminalBD cells to respond to specific analytical signals.

## Setup

### Prerequisites

1. **Proto-liminal** installed and configured
2. **LiminalBD** compiled and `liminal-cli` in PATH
3. Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Edit `configs/liminalbd_integration.json`:

```json
{
  "liminalbd": {
    "enabled": true,
    "cli_path": "liminal-cli",
    "mode": "subprocess"
  }
}
```

## Usage

### Basic Example

```python
from liminal_bridge import LiminalBridge, Signal, create_signal_from_news

# Create bridge
bridge = LiminalBridge()

# Create signal from news analysis
signal = create_signal_from_news(
    entity="Bitcoin Price",
    sentiment=0.8,      # Positive sentiment
    relevance=0.9,      # Highly relevant
    urgency=0.7         # Urgent
)

# Send to LiminalBD
success = bridge.send_signal(signal)

print(f"Signal sent: {success}")
```

### Batch Processing

```python
# Process multiple signals
signals = [
    create_signal_from_news("Bitcoin", 0.8, 0.9, 0.7),
    create_signal_from_news("Ethereum", -0.3, 0.6, 0.5),
    create_signal_from_news("Stock Market", 0.2, 0.8, 0.4),
]

results = bridge.send_batch(signals)
print(f"Sent: {results['success']}, Failed: {results['failed']}")
```

### Event Listening (Async)

```python
import asyncio
from liminal_bridge import LiminalBridge

async def handle_event(event):
    """Handle LiminalBD events"""
    if event['type'] == 'awaken':
        print(f"Model awakened: {event.get('model_id')}")
    elif event['type'] == 'harmony':
        print(f"Harmony status: {event.get('status')}")

async def main():
    bridge = LiminalBridge(mode="websocket")
    await bridge.listen_events(callback=handle_event)

asyncio.run(main())
```

## Integration Patterns

### Pattern 1: Real-time News → Impulses

```python
# In collector.py or signal_extractor.py
from liminal_bridge import LiminalBridge, Signal

bridge = LiminalBridge()

# After extracting signals from news
for entity, features in extracted_signals:
    signal = Signal(
        entity=entity,
        features=features,
        signal_strength=calculate_strength(features),
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    bridge.send_signal(signal)
```

### Pattern 2: RINSE ↔ Harmony Sync

```python
# In rinse_agent.py
from liminal_bridge import LiminalBridge

bridge = LiminalBridge()

# After RINSE reflection
if calibration_needed:
    # Send adjustment signal to LiminalBD
    signal = Signal(
        entity="rinse/calibration",
        features={"adjustment": adjustment_value},
        signal_strength=confidence,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    bridge.send_signal(signal)
```

### Pattern 3: Model Lifecycle Management

```python
# Store forecast models as cellular entities
model_signal = Signal(
    entity=f"model/{model_id}",
    features={
        "accuracy": 0.85,
        "last_used": timestamp,
        "importance": 0.9
    },
    signal_strength=0.85,
    timestamp=datetime.now(timezone.utc).isoformat()
)

bridge.send_signal(model_signal)
```

## Communication Modes

### Subprocess Mode (Default)

Sends impulses via `liminal-cli --cbor-pipe`:

```python
bridge = LiminalBridge(mode="subprocess")
```

**Pros:**
- Simple, no network setup
- Reliable

**Cons:**
- One-way communication only
- Higher latency

### WebSocket Mode (Advanced)

Enables bidirectional communication:

```python
bridge = LiminalBridge(
    mode="websocket",
    ws_url="ws://localhost:9001"
)
```

**Pros:**
- Real-time events
- Bidirectional
- Lower latency

**Cons:**
- Requires LiminalBD WebSocket server
- More complex setup

## RINSE ↔ Harmony Loop Integration

### Phase 1: Signal Flow (Current)
- Proto-liminal signals → LiminalBD impulses
- One-way communication

### Phase 2: Event Feedback (Planned)
- LiminalBD events → Proto-liminal RINSE adjustments
- Harmony status → Calibration triggers

### Phase 3: Full Symbiosis (Future)
- RINSE adjustments → TRS state changes
- Cellular metabolism → Forecast prioritization
- Shared "living consciousness" substrate

## Monitoring

### Statistics

```python
stats = bridge.get_stats()
print(stats)
# {
#   "impulses_sent": 42,
#   "events_received": 15,
#   "errors": 0
# }
```

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Troubleshooting

### Issue: "liminal-cli not found"

**Solution:** Ensure LiminalBD is installed and `liminal-cli` is in PATH:
```bash
export PATH=$PATH:/path/to/LiminalBD/target/release
```

### Issue: "cbor2 not installed"

**Solution:** Install dependencies:
```bash
pip install cbor2 websockets
```

### Issue: "Connection refused" (WebSocket)

**Solution:** Start LiminalBD with WebSocket support:
```bash
liminal-cli --ws-port 9001
```

## Future Enhancements

1. **PyO3 Bindings**: Direct Rust ↔ Python integration (zero-copy)
2. **Shared Memory**: Ultra-low latency communication
3. **Model Persistence**: Store forecast models in LiminalBD ResonantModels
4. **TRS Synchronization**: RINSE directly controls TRS parameters
5. **Distributed Proto-consciousness**: Multiple Proto-liminal instances sharing one LiminalBD substrate

## References

- [LiminalBD Protocol](https://github.com/safal207/LiminalBD/blob/main/docs/PROTOCOL.md)
- [Harmony Loop Brief](https://github.com/safal207/LiminalBD/blob/main/docs/HARMONY_LOOP_BRIEF.md)
- [Proto-liminal MVP Spec](docs/MVP_SPEC.md)

## License

This integration is licensed under the LIMINAL ProtoConsciousness License (LPL). See LICENSE file for details.

---

*"Where Python analytics meet Rust substrate, proto-consciousness emerges."*
