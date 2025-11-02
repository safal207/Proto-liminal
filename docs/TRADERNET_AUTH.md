# Tradernet WebSocket API Authentication

This document explains how to connect to Tradernet's real-time market data feed.

## Overview

Tradernet provides real-time market quotes via Socket.IO WebSocket API. The demo server requires authentication.

## Endpoints

- **Demo Server**: `https://wsbeta.tradernet.ru`
- **Dev Server**: `wss://wssdev.tradernet.dev` (requires API key)

## Connection Requirements

### 1. Socket.IO Client

Tradernet uses Socket.IO protocol (not pure WebSocket). Use `python-socketio`:

```bash
pip install python-socketio[client]
```

### 2. Authentication (Required for Production)

To access real market data, you need:

1. **Register** at [tradernet.com](https://tradernet.com) or [freedom24.com](https://freedom24.com)
2. **Get API Key** from your account settings
3. **Generate Signature** for authentication

### Authentication Pattern

```python
import socketio
import hashlib
import time

sio = socketio.Client()

@sio.event
def connect():
    nonce = int(time.time() * 1000)  # Timestamp in milliseconds

    # Generate signature: MD5(uid + cmd + sorted_params + API_SECRET)
    signature = hashlib.md5(
        f"{uid}{cmd}{nonce}{API_SECRET}".encode()
    ).hexdigest()

    # Authenticate
    sio.emit('auth', {
        'apiKey': PUBLIC_API_KEY,
        'cmd': 'auth',
        'nonce': nonce,
        'sig': signature
    })

sio.connect('https://wsbeta.tradernet.ru')
```

## Subscription Format

After authentication, subscribe to quotes:

```python
@sio.event
def connect():
    # Subscribe to tickers
    sio.emit('notifyQuotes', ['SBER', 'GAZP', 'LKOH', 'BTCUSD'])

@sio.on('q')
def on_quote(data):
    # Receive quote updates
    print(f"Quote: {data}")
```

## Quote Message Format

Tradernet sends quotes in 'q' events:

```json
{
  "c": "SBER",           // ticker symbol
  "ltp": 245.50,         // last traded price
  "chg": 2.30,           // change
  "pchg": 0.95,          // percent change
  "vol": 15000,          // volume
  "bid": 245.45,         // bid price
  "ask": 245.55,         // ask price
  "h": 246.80,           // high
  "l": 244.20,           // low
  "o": 245.00            // open
}
```

**Notes:**
- First quote contains full data
- Subsequent updates contain only changed fields
- Use previous values for unchanged fields

## Demo Mode (No Authentication)

For development/testing without authentication, use the **simulated feed**:

```bash
python examples/demo_realtime_simulated.py
```

This runs a realistic simulation of the real-time system without requiring API keys.

## Troubleshooting

### HTTP 403 Forbidden

```
ConnectionError: Unexpected status code 403
```

**Cause**: Server requires authentication
**Solution**:
1. Register for API key
2. Implement authentication (see above)
3. Or use demo mode

### Connection Timeout

```
ConnectionError: Connection error
```

**Possible causes:**
- Network/firewall blocking WebSocket
- Server unavailable
- Incorrect endpoint URL

**Solution**:
1. Check network connectivity
2. Verify endpoint URL
3. Try alternative transport (polling):
   ```python
   sio.connect(url, transports=['polling', 'websocket'])
   ```

### No Quotes Received

**Possible causes:**
- Authentication failed silently
- Subscription not sent
- Invalid ticker symbols

**Solution**:
1. Enable debug logging:
   ```python
   sio = socketio.Client(logger=True, engineio_logger=True)
   ```
2. Verify ticker symbols are valid
3. Check subscription was sent after connection

## References

- [Tradernet API Documentation](https://tradernet.com/tradernet-api)
- [Freedom24 WebSocket API](https://freedom24.com/tradernet-api/websocket)
- [GitHub: tradernet/tn.api](https://github.com/tradernet/tn.api)
- [Socket.IO Python Client](https://python-socketio.readthedocs.io/)

## Alternative Data Sources

If Tradernet authentication is not available, consider:

- **Alpha Vantage**: Free API with real-time quotes
- **Yahoo Finance**: yfinance Python library
- **IEX Cloud**: Real-time stock data API
- **Binance** (for crypto): Public WebSocket API, no auth required
- **Polygon.io**: Professional market data API

## Next Steps

1. **Get API Key**: Register at tradernet.com
2. **Implement Auth**: Add authentication to `tradernet_socketio_client.py`
3. **Test Connection**: Run `python src/tradernet_socketio_client.py`
4. **Integrate**: Connect to `realtime_monitor.py` for full system

For immediate testing, use the demo:
```bash
python examples/demo_realtime_simulated.py
```
