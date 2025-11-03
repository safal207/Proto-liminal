## Running Tradernet Client Locally

The Tradernet WebSocket client requires running **on your local machine**, not in the sandbox environment.

### Why Local?

The sandbox environment has network restrictions that block WebSocket connections to Tradernet, even though the connection works fine locally (as confirmed with `wscat`).

### Quick Start (Local Machine)

**1. Clone the repository:**
```bash
git clone https://github.com/safal207/Proto-liminal.git
cd Proto-liminal
```

**2. Install dependencies:**
```bash
pip install websockets
```

**3. Test connection:**
```bash
python examples/tradernet_live_test.py
```

Expected output:
```
✅ CONNECTED!
Message 1: [userData]
  Mode: demo
  Version: 4.4.0.prod
  Market delay: False

Message 2: [keepAlive]
  💓 KeepAlive

📊 Testing Quote Subscription
Subscribing to: ['BTC/USD', 'ETH/USD', 'GAZP']
✅ Subscription sent

✅ TEST SUCCESSFUL!
```

**4. Run full client:**
```bash
python src/tradernet_demo_client.py
```

### Using with Proto-liminal Monitoring

Once the connection works locally, integrate with real-time monitoring:

```python
from src.tradernet_demo_client import TradernetWebSocketClient, TradernetConfig
from src.liminal_detector import LiminalDetector
from src.market_regime import MarketRegimeClassifier

# Configure client
config = TradernetConfig(
    symbols=["BTC/USD", "ETH/USD", "GAZP", "SBER"]
)

client = TradernetWebSocketClient(config)

# Setup detectors
detector = LiminalDetector()
regime = MarketRegimeClassifier()

# Process quotes
def on_quote(quote):
    # Detect liminal state
    state = detector.detect(
        sentiment=0.0,
        volatility=quote.change_pct / 100 if quote.change_pct else 0,
        volume=int(quote.volume) if quote.volume else 0
    )

    # Classify regime
    regime_info = regime.classify(price=quote.price)

    print(f"{quote.symbol}: ${quote.price:.2f} | {state.state} | {regime_info.regime}")

client.register_quote_callback(on_quote)

# Run
await client.run()
```

### Troubleshooting

#### HTTP 403 Error

If you get `HTTP 403` even locally:

1. **Check firewall:**
   ```bash
   # Allow outgoing WebSocket connections
   sudo ufw allow out 443/tcp
   ```

2. **Try with VPN:**
   - Tradernet may have geo-restrictions
   - Try connecting through VPN

3. **Verify URL:**
   ```bash
   # Test with wscat first
   wscat -c "wss://wss.tradernet.com/"

   # You should see:
   # < ["userData", {...}, "wstm=..."]
   # < ["keepAlive", [], "wstm=..."]
   ```

4. **Check headers:**
   - Origin header should be `https://tradernet.com`
   - User-Agent should look like a browser

#### No Quotes Received

Connection works but no quote messages:

1. **Verify ticker format:**
   - Use `BTC/USD` not `BTCUSD`
   - Use `GAZP` for Russian stocks
   - Check available tickers in Tradernet docs

2. **Market hours:**
   - Russian market: 10:00-18:45 MSK
   - US market: 16:30-23:00 MSK
   - Crypto: 24/7

3. **Check subscription:**
   ```python
   # After connection:
   await client.subscribe_quotes(["BTC/USD", "ETH/USD"])
   ```

### Testing in Sandbox (Alternative)

If you can't run locally, use the **simulated demo**:

```bash
python examples/demo_realtime_simulated.py
```

This works in sandbox and demonstrates the full monitoring system without requiring real connections.

### Architecture

```
Local Machine
    ↓
Tradernet WebSocket (wss://wss.tradernet.com/)
    ↓
TradernetWebSocketClient (src/tradernet_demo_client.py)
    ↓
Quote Callbacks
    ↓
LiminalDetector + MarketRegime + RiskManager
    ↓
Real-Time Monitoring Dashboard
```

### Available Symbols

#### Crypto (24/7)
- `BTC/USD` - Bitcoin
- `ETH/USD` - Ethereum
- `LTC/USD` - Litecoin
- `XRP/USD` - Ripple

#### Russian Stocks
- `GAZP` - Gazprom
- `SBER` - Sberbank
- `LKOH` - Lukoil
- `YNDX` - Yandex
- `ROSN` - Rosneft

#### US Stocks
- `AAPL` - Apple
- `TSLA` - Tesla
- `GOOGL` - Google
- `MSFT` - Microsoft
- `AMZN` - Amazon

### Message Format

Tradernet sends messages in this format:

```json
[messageType, data, "wstm=timestamp"]
```

#### userData (on connect)
```json
[
  "userData",
  {
    "mode": "demo",
    "marketDataDelay": false,
    "connections": {...},
    "version": "4.4.0.prod"
  },
  "wstm=2025-11-03T04:42:49.824Z"
]
```

#### keepAlive (heartbeat)
```json
["keepAlive", [], "wstm=2025-11-03T04:43:49.824Z"]
```

#### quotes (market data)
```json
[
  "q",
  {
    "c": "BTC/USD",
    "ltp": 43250.50,
    "chg": 1250.00,
    "pchg": 2.98,
    "vol": 1250000,
    "bid": 43248.00,
    "ask": 43252.00
  },
  "wstm=..."
]
```

### Next Steps

1. **Test locally:** `python examples/tradernet_live_test.py`
2. **Run client:** `python src/tradernet_demo_client.py`
3. **Integrate:** Connect to real-time monitoring system
4. **Deploy:** Set up as background service

### Support

- **Tradernet Docs:** https://tradernet.com/tradernet-api
- **GitHub Issues:** https://github.com/safal207/Proto-liminal/issues
- **Test Tool:** `wscat -c "wss://wss.tradernet.com/"`

---

**💡 Remember:** Sandbox has network restrictions. Always test WebSocket connections locally first!
