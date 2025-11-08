# Tradernet Real-Time Data Setup Guide

## Overview

Proto-liminal supports real-time market monitoring through **Tradernet WebSocket API**. This guide explains how to get your credentials and connect to live market data.

## Quick Start (No Auth Required)

For immediate testing without Tradernet credentials, use the **simulated demo**:

```bash
python examples/demo_realtime_simulated.py
```

This runs a realistic simulation showing the full real-time monitoring system.

## Getting Tradernet Access

To connect to **real live market data**, you need Tradernet credentials.

### Option 1: Free Demo Access (Recommended for Testing)

If you have a Tradernet account, you can find your `user_id` in account settings.

**WebSocket Format:**
```
wss://wss.tradernet.com/?user_id=YOUR_USER_ID
```

**Subscription Format:**
```json
["quotes", ["GAZP", "SBER", "AAPL"]]
```

### Option 2: API Key (Production Use)

For production trading systems, use API key authentication:

1. **Register**: Create account at [tradernet.com](https://tradernet.com) or [freedom24.com](https://freedom24.com)
2. **Enable API Access**: Go to account settings → API
3. **Get Credentials**:
   - Public API Key
   - Secret API Key
   - User ID

4. **Store Securely**:
```bash
# Create .env file (never commit this!)
echo "TRADERNET_USER_ID=your_user_id" >> .env
echo "TRADERNET_API_KEY=your_api_key" >> .env
echo "TRADERNET_SECRET=your_secret" >> .env
```

## Testing Your Connection

### Method 1: Python Client

```bash
# Edit src/tradernet_realtime_client.py
# Update user_id in TradernetConfig

python src/tradernet_realtime_client.py
```

### Method 2: wscat (Command Line)

```bash
# Install wscat
npm install -g wscat

# Connect
wscat -c "wss://wss.tradernet.com/?user_id=YOUR_USER_ID"

# Subscribe (type this after connection)
["quotes",["GAZP","SBER","AAPL"]]

# You should see quote messages streaming in
```

### Method 3: Test Script

```bash
# Tests multiple connection variants
python examples/test_tradernet_variants.py
```

## Troubleshooting

### HTTP 403 Forbidden

```
InvalidStatus: server rejected WebSocket connection: HTTP 403
```

**Причины:**
- ❌ user_id недействителен или устарел
- ❌ Требуется дополнительная аутентификация (API key)
- ❌ IP адрес не в whitelist
- ❌ API доступ не активирован в настройках аккаунта

**Решение:**
1. Проверь свой user_id в личном кабинете Tradernet
2. Убедись что API доступ активирован
3. Попробуй с VPN если есть гео-ограничения
4. Используй demo режим: `python examples/demo_realtime_simulated.py`

### No Messages Received

Connection successful but no quotes coming through:

**Проверь:**
- Правильность тикеров (GAZP, SBER для российских акций)
- Рыночное время (биржа может быть закрыта)
- Подписка отправлена после подключения

### Connection Timeout

**Возможные причины:**
- Firewall блокирует WebSocket
- Нестабильное интернет-соединение
- Сервер недоступен

**Решение:**
- Проверь интернет
- Попробуй с другой сети
- Используй demo режим

## Available Tickers

### Russian Stocks
- **GAZP** - Gazprom
- **SBER** - Sberbank
- **LKOH** - Lukoil
- **ROSN** - Rosneft
- **GMKN** - Norilsk Nickel
- **YNDX** - Yandex

### US Stocks
- **AAPL** - Apple
- **TSLA** - Tesla
- **GOOGL** - Google
- **MSFT** - Microsoft
- **AMZN** - Amazon

### Crypto
- **BTCUSD** - Bitcoin
- **ETHUSD** - Ethereum

## Protocol Details

### Connection
```
wss://wss.tradernet.com/?user_id=YOUR_USER_ID
```

### Subscribe to Quotes
```json
["quotes", ["GAZP", "SBER", "AAPL"]]
```

### Subscribe to Order Book
```json
["orderBook", ["GAZP"]]
```

### Unsubscribe
```json
["quotes", []]
```

### Quote Message Format
```json
{
  "c": "SBER",       // ticker
  "ltp": 245.50,     // last price
  "chg": 2.30,       // change
  "pchg": 0.95,      // change %
  "vol": 15000,      // volume
  "bid": 245.45,     // bid
  "ask": 245.55,     // ask
  "h": 246.80,       // high
  "l": 244.20,       // low
  "o": 245.00        // open
}
```

## Alternative Data Sources

Если Tradernet не подходит, попробуй:

- **Binance** (crypto): Бесплатный WebSocket API без авторизации
  ```python
  # wss://stream.binance.com:9443/ws/btcusdt@trade
  ```

- **Alpha Vantage**: Бесплатный API для акций
  ```bash
  pip install alpha-vantage
  ```

- **Yahoo Finance**: Через yfinance библиотеку
  ```bash
  pip install yfinance
  ```

- **IEX Cloud**: Professional market data API

## Integration with Proto-liminal

Once you have working Tradernet credentials:

```python
from tradernet_realtime_client import TradernetWebSocketClient, TradernetConfig

# Configure with YOUR credentials
config = TradernetConfig(
    url="wss://wss.tradernet.com",
    user_id="YOUR_USER_ID",  # <-- Your actual user_id here
    symbols=["GAZP", "SBER", "BTCUSD"]
)

client = TradernetWebSocketClient(config)

def on_quote(quote):
    print(f"{quote.symbol}: ${quote.price:.2f}")

client.register_quote_callback(on_quote)

# Run
await client.run()
```

## Demo Mode (Recommended)

Для разработки и тестирования используй симулированный режим:

```bash
python examples/demo_realtime_simulated.py
```

**Показывает:**
- ✅ Real-time liminal detection
- ✅ Market regime classification
- ✅ Adaptive risk adjustments
- ✅ Critical state alerts
- ✅ Colored console output

Работает **без API ключей**, показывает полную функциональность системы.

## Next Steps

1. **Test Demo**: `python examples/demo_realtime_simulated.py`
2. **Get Credentials**: Register at tradernet.com
3. **Test Connection**: Update user_id and test
4. **Integrate**: Connect to full monitoring pipeline

## Support

- **Tradernet Docs**: https://tradernet.com/tradernet-api
- **Freedom24 API**: https://freedom24.com/tradernet-api
- **GitHub Issues**: https://github.com/safal207/Proto-liminal/issues
- **WebSocket Test Tool**: `wscat -c "wss://..."`

---

**💡 Важно:** Для немедленного тестирования системы используй demo режим. Реальные данные требуют валидный Tradernet аккаунт.
