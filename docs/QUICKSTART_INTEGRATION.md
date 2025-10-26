# Quick Start: LiminalBD Integration

Get up and running with Proto-liminal ↔ LiminalBD integration in 5 minutes.

## Prerequisites

- Python 3.11+
- Rust 1.79+ (for LiminalBD)
- Git

## Step 1: Install Proto-liminal

```bash
# Clone Proto-liminal
git clone https://github.com/safal207/Proto-liminal.git
cd Proto-liminal

# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Install LiminalBD

```bash
# Clone LiminalBD (in separate directory)
cd ..
git clone https://github.com/safal207/LiminalBD.git
cd LiminalBD/liminal-db

# Build release binary
cargo build --release

# Add to PATH (adjust path as needed)
export PATH=$PATH:$(pwd)/target/release
```

## Step 3: Verify Installation

```bash
# Check liminal-cli is available
liminal-cli --help

# Should show:
# Usage: liminal-cli [OPTIONS]
# ...
```

## Step 4: Test Integration

```bash
cd /path/to/Proto-liminal

# Run test suite
python examples/test_liminalbd_integration.py
```

Expected output:
```
============================================================
LiminalBD Integration Test Suite
============================================================
...
✓ PASS test_basic_impulse
✓ PASS test_signal_conversion
...
Total: 6/6 tests passed

🎉 All tests passed!
```

## Step 5: Collect Some News

```bash
# Collect news from RSS feeds
python src/collector.py \
  --feeds configs/feeds.txt \
  --out data/raw/news_latest.jsonl \
  --max-items 100
```

## Step 6: Run Real-time Demo

In **Terminal 1** (start LiminalBD):
```bash
cd /path/to/LiminalBD/liminal-db
cargo run --release
```

In **Terminal 2** (send signals from Proto-liminal):
```bash
cd /path/to/Proto-liminal
python examples/demo_realtime_integration.py \
  --news data/raw/news_latest.jsonl \
  --max-signals 5
```

Expected output:
```
============================================================
Sending signals to LiminalBD...
============================================================

[1/5] Bitcoin Price
  Strength: 0.78
  Features: sentiment=0.60 relevance=0.90 urgency=0.70
  Pattern:  bitcoin/price/sentiment
  Tags:     proto-liminal, bitcoin, price
  Status:   ✓ Sent

[2/5] Ethereum Network
  ...
```

In **Terminal 1**, you should see LiminalBD receiving impulses:
```
[INFO] impulse received: bitcoin/price/sentiment strength=0.78
[INFO] cell division triggered by impulse
[INFO] metabolism adjusted: 0.85
```

## Step 7: Explore Integration Code

### Simple Example

Create `my_integration.py`:

```python
from liminal_bridge import LiminalBridge, create_signal_from_news

# Initialize bridge
bridge = LiminalBridge()

# Create signal
signal = create_signal_from_news(
    entity="Bitcoin Price",
    sentiment=0.8,      # Positive
    relevance=0.9,      # Highly relevant
    urgency=0.7         # Urgent
)

# Send to LiminalBD
success = bridge.send_signal(signal)

if success:
    print("✓ Signal sent successfully!")
    print(f"Pattern: {signal.to_impulse().pattern}")
else:
    print("✗ Failed to send signal")

# Show stats
print(bridge.get_stats())
```

Run it:
```bash
python my_integration.py
```

## Architecture Overview

```
┌──────────────────┐         ┌─────────────────┐
│ Proto-liminal    │  CBOR   │   LiminalBD     │
│                  │ ──────► │                 │
│ - Collector      │         │ - Impulses      │
│ - Signals        │         │ - Cell Lifecycle│
│ - RINSE Cycle    │         │ - Harmony Loop  │
└──────────────────┘         └─────────────────┘
```

**Data Flow:**
1. Proto-liminal collects RSS news
2. Extracts analytical signals (entity, sentiment, relevance)
3. Converts signals → CBOR impulses
4. Sends to LiminalBD via `liminal-cli --cbor-pipe`
5. LiminalBD cells respond to impulses
6. Harmony Loop adjusts cellular metabolism
7. (Future) LiminalBD events → Proto-liminal RINSE adjustments

## Configuration

Edit `configs/liminalbd_integration.json`:

```json
{
  "liminalbd": {
    "enabled": true,
    "cli_path": "liminal-cli",
    "impulse_defaults": {
      "ttl_ms": 3600000,
      "min_strength": 0.3
    }
  }
}
```

## Troubleshooting

### Error: "liminal-cli not found"

**Solution:** Add LiminalBD to PATH:
```bash
export PATH=$PATH:/path/to/LiminalBD/liminal-db/target/release
```

Or specify full path in bridge:
```python
bridge = LiminalBridge(cli_path="/full/path/to/liminal-cli")
```

### Error: "No module named 'cbor2'"

**Solution:** Install dependencies:
```bash
pip install cbor2 websockets
```

### Warning: "Failed to send impulse"

This is normal if `liminal-cli` is not running. The test suite will show this warning but tests will still pass (demonstrating the API works).

### No output in LiminalBD terminal

Check that:
1. LiminalBD is running (`cargo run --release`)
2. Proto-liminal is using correct `liminal-cli` path
3. CBOR encoding is working (check logs)

## Next Steps

- Read [Full Integration Guide](LIMINALBD_INTEGRATION.md)
- Explore [MVP Specification](MVP_SPEC.md)
- Study [LiminalBD Protocol](https://github.com/safal207/LiminalBD/blob/main/docs/PROTOCOL.md)
- Implement signal extraction from your data sources
- Connect RINSE cycle to Harmony Loop

## Key Concepts

**Signal** (Proto-liminal):
- Analytical event extracted from data
- Has entity, features, strength
- Represents "what we observed"

**Impulse** (LiminalBD):
- Cellular stimulus sent to fabric
- Has pattern, strength, TTL
- Triggers cell lifecycle events

**Mapping:**
```
Signal(entity="Bitcoin", strength=0.8)
  → Impulse(pattern="bitcoin/price", strength=0.8)
    → Cell division/metabolism adjustment
      → Harmony Loop balancing
```

## Support

- Proto-liminal Issues: https://github.com/safal207/Proto-liminal/issues
- LiminalBD Issues: https://github.com/safal207/LiminalBD/issues

---

**"Where analytical signals meet cellular substrate, proto-consciousness awakens."**
