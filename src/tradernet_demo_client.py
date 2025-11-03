#!/usr/bin/env python3
"""
Tradernet WebSocket Client (Working Demo Mode)

Connects to wss://wss.tradernet.com/ in demo mode (no user_id required!)
Subscribes to real-time quotes using JSON array format.

Protocol:
- Connect: wss://wss.tradernet.com/ (no auth needed for demo)
- Messages: ["messageType", data, "wstm=timestamp"]
- Subscribe: ["quotes", ["BTC/USD", "ETH/USD", "GAZP"]]
- KeepAlive: ["keepAlive", []]
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Callable, Dict, Any

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "websockets"])
    import websockets

LOGGER = logging.getLogger(__name__)


@dataclass
class Quote:
    """Real-time market quote"""
    symbol: str
    price: float
    timestamp: str
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    raw_data: Optional[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradernetConfig:
    """Configuration for Tradernet WebSocket"""
    url: str = "wss://wss.tradernet.com/"
    symbols: List[str] = None
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    ping_interval: int = 30

    def __post_init__(self):
        if self.symbols is None:
            # Demo mode supports both formats
            self.symbols = ["BTC/USD", "ETH/USD", "GAZP", "SBER"]


class TradernetWebSocketClient:
    """
    WebSocket client for Tradernet (Demo Mode)

    Works WITHOUT authentication in demo mode!
    - Connect: wss://wss.tradernet.com/
    - Message format: [type, data, wstm]
    - Subscribe: ["quotes", [symbols]]

    Usage:
        client = TradernetWebSocketClient()

        def on_quote(quote):
            print(f"{quote.symbol}: ${quote.price}")

        client.register_quote_callback(on_quote)
        await client.run()
    """

    def __init__(self, config: Optional[TradernetConfig] = None):
        self.config = config or TradernetConfig()
        self.ws = None
        self.connected = False
        self.subscribed_symbols = set()
        self.reconnect_count = 0

        # Server info
        self.user_data = None
        self.demo_mode = False

        # Statistics
        self.quotes_received = 0
        self.messages_received = 0
        self.keepalive_received = 0
        self.last_quote_time = None
        self.connection_start = None

        # Callbacks
        self.on_quote_callbacks: List[Callable[[Quote], None]] = []
        self.on_connect_callbacks: List[Callable[[], None]] = []
        self.on_disconnect_callbacks: List[Callable[[], None]] = []
        self.on_raw_message_callbacks: List[Callable[[List], None]] = []

        LOGGER.info(f"TradernetWebSocketClient initialized")
        LOGGER.info(f"URL: {self.config.url}")

    async def connect(self) -> bool:
        """Connect to Tradernet WebSocket"""
        try:
            LOGGER.info(f"Connecting to {self.config.url}...")

            # Add browser-like headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Origin": "https://tradernet.com",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8"
            }

            self.ws = await websockets.connect(
                self.config.url,
                additional_headers=headers,
                ping_interval=self.config.ping_interval,
                ping_timeout=10
            )

            self.connected = True
            self.connection_start = datetime.now(timezone.utc)
            self.reconnect_count = 0

            LOGGER.info("✅ Connected to Tradernet WebSocket")

            # Wait for userData message
            await asyncio.sleep(0.5)

            # Trigger callbacks
            for callback in self.on_connect_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()

            # Subscribe to configured symbols
            if self.config.symbols:
                await self.subscribe_quotes(self.config.symbols)

            return True

        except Exception as e:
            LOGGER.error(f"Connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close WebSocket connection"""
        if self.ws:
            try:
                await self.ws.close()
                LOGGER.info("Disconnected from Tradernet")
            except Exception as e:
                LOGGER.warning(f"Error during disconnect: {e}")
            finally:
                self.connected = False
                self.ws = None

                for callback in self.on_disconnect_callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()

    async def subscribe_quotes(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time quotes

        Format: ["quotes", ["BTC/USD", "ETH/USD", "GAZP"]]
        """
        if not self.connected or not self.ws:
            LOGGER.error("Not connected - cannot subscribe")
            return False

        try:
            message = ["quotes", symbols]
            await self.ws.send(json.dumps(message))

            self.subscribed_symbols.update(symbols)
            LOGGER.info(f"📊 Subscribed to: {', '.join(symbols)}")

            return True

        except Exception as e:
            LOGGER.error(f"Subscription failed: {e}")
            return False

    async def unsubscribe_quotes(self) -> bool:
        """Unsubscribe from all quotes"""
        if not self.connected or not self.ws:
            return False

        try:
            message = ["quotes", []]
            await self.ws.send(json.dumps(message))
            self.subscribed_symbols.clear()
            LOGGER.info("Unsubscribed from all quotes")
            return True

        except Exception as e:
            LOGGER.error(f"Unsubscribe failed: {e}")
            return False

    async def parse_message(self, message: List) -> Optional[str]:
        """
        Parse Tradernet message

        Format: [messageType, data, wstm]

        Message types:
        - userData: Server info
        - keepAlive: Ping
        - q: Quote update
        - quotes: Quote data
        """
        try:
            if not isinstance(message, list) or len(message) < 2:
                return None

            msg_type = message[0]
            data = message[1] if len(message) > 1 else None

            # userData - server info
            if msg_type == "userData":
                self.user_data = data
                self.demo_mode = data.get("mode") == "demo"
                LOGGER.info(f"📋 Server mode: {data.get('mode')}")
                LOGGER.info(f"   Version: {data.get('version')}")
                LOGGER.info(f"   Market delay: {data.get('marketDataDelay')}")
                return "userData"

            # keepAlive - ping
            elif msg_type == "keepAlive":
                self.keepalive_received += 1
                LOGGER.debug("💓 KeepAlive")
                return "keepAlive"

            # quotes - quote data
            elif msg_type in ("q", "quotes"):
                quote = self.parse_quote_data(data)
                if quote:
                    self.quotes_received += 1
                    self.last_quote_time = datetime.now(timezone.utc)

                    # Trigger callbacks
                    for callback in self.on_quote_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(quote)
                            else:
                                callback(quote)
                        except Exception as e:
                            LOGGER.error(f"Quote callback error: {e}")

                return "quote"

            else:
                LOGGER.debug(f"Unknown message type: {msg_type}")
                return None

        except Exception as e:
            LOGGER.debug(f"Parse error: {e}")
            return None

    def parse_quote_data(self, data: Any) -> Optional[Quote]:
        """Parse quote data from message"""
        try:
            if isinstance(data, dict):
                # Extract symbol
                symbol = data.get("c") or data.get("ticker") or data.get("symbol") or data.get("s")

                # Extract price
                price = (
                    data.get("ltp") or
                    data.get("last") or
                    data.get("ltr") or
                    data.get("price") or
                    data.get("p")
                )

                if not symbol or price is None:
                    return None

                return Quote(
                    symbol=symbol,
                    price=float(price),
                    timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    volume=data.get("vol") or data.get("volume") or data.get("v"),
                    bid=data.get("bid") or data.get("b"),
                    ask=data.get("ask") or data.get("a"),
                    change=data.get("chg") or data.get("change"),
                    change_pct=data.get("pchg") or data.get("change_pct") or data.get("pcp"),
                    high=data.get("h") or data.get("high"),
                    low=data.get("l") or data.get("low"),
                    open_price=data.get("o") or data.get("open"),
                    raw_data=data
                )

            elif isinstance(data, list):
                # Array of quotes
                quotes = []
                for item in data:
                    quote = self.parse_quote_data(item)
                    if quote:
                        quotes.append(quote)

                # Return first quote for now
                return quotes[0] if quotes else None

            return None

        except Exception as e:
            LOGGER.debug(f"Quote parse error: {e}")
            return None

    async def receive_messages(self):
        """Receive and process messages"""
        while self.connected and self.ws:
            try:
                raw_message = await self.ws.recv()
                self.messages_received += 1

                # Parse JSON
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    LOGGER.warning(f"Invalid JSON: {raw_message[:100]}")
                    continue

                # Trigger raw callbacks
                for callback in self.on_raw_message_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        LOGGER.error(f"Raw callback error: {e}")

                # Parse message
                await self.parse_message(message)

            except websockets.exceptions.ConnectionClosed:
                LOGGER.warning("Connection closed")
                self.connected = False
                break

            except Exception as e:
                LOGGER.error(f"Error receiving message: {e}")
                await asyncio.sleep(0.1)

    async def run_with_reconnect(self):
        """Run with auto-reconnection"""
        while self.reconnect_count < self.config.max_reconnect_attempts:
            try:
                if not self.connected:
                    success = await self.connect()

                    if not success:
                        self.reconnect_count += 1
                        delay = min(self.config.reconnect_delay * (2 ** self.reconnect_count), 60)
                        LOGGER.info(f"Reconnecting in {delay}s... (attempt {self.reconnect_count})")
                        await asyncio.sleep(delay)
                        continue

                await self.receive_messages()

                if self.reconnect_count < self.config.max_reconnect_attempts:
                    LOGGER.info("Connection lost - reconnecting...")
                    self.connected = False
                    await asyncio.sleep(self.config.reconnect_delay)

            except asyncio.CancelledError:
                LOGGER.info("Shutdown requested")
                break

            except Exception as e:
                LOGGER.error(f"Unexpected error: {e}")
                self.reconnect_count += 1
                await asyncio.sleep(self.config.reconnect_delay)

        await self.disconnect()
        LOGGER.warning("Max reconnection attempts reached")

    async def run(self, duration: Optional[int] = None):
        """Run client"""
        if duration:
            try:
                await asyncio.wait_for(self.run_with_reconnect(), timeout=duration)
            except asyncio.TimeoutError:
                LOGGER.info(f"Duration ({duration}s) reached")
                await self.disconnect()
        else:
            await self.run_with_reconnect()

    def register_quote_callback(self, callback: Callable[[Quote], None]):
        """Register quote callback"""
        self.on_quote_callbacks.append(callback)

    def register_raw_message_callback(self, callback: Callable[[List], None]):
        """Register raw message callback"""
        self.on_raw_message_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get statistics"""
        uptime = None
        if self.connection_start:
            uptime = (datetime.now(timezone.utc) - self.connection_start).total_seconds()

        return {
            "connected": self.connected,
            "demo_mode": self.demo_mode,
            "quotes_received": self.quotes_received,
            "messages_received": self.messages_received,
            "keepalive_received": self.keepalive_received,
            "subscribed_symbols": list(self.subscribed_symbols),
            "uptime_seconds": uptime,
            "last_quote_time": self.last_quote_time.isoformat() if self.last_quote_time else None,
        }


async def main():
    """Test client"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("="*80)
    print("🚀 Tradernet WebSocket Client (Demo Mode - No Auth!)")
    print("="*80)
    print()

    config = TradernetConfig(
        symbols=["BTC/USD", "ETH/USD", "GAZP", "SBER", "AAPL"]
    )

    client = TradernetWebSocketClient(config)

    # Quote handler
    def on_quote(quote: Quote):
        change_str = ""
        if quote.change_pct is not None:
            sign = "+" if quote.change_pct >= 0 else ""
            change_str = f"  {sign}{quote.change_pct:.2f}%"

        print(f"📊 {quote.symbol:12s} ${quote.price:12.2f}{change_str}")

    # Raw message handler (debug)
    def on_raw(msg: List):
        if msg[0] not in ("keepAlive",):  # Skip keepAlive spam
            print(f"🔍 {msg[0]}: {str(msg[1])[:100]}")

    client.register_quote_callback(on_quote)
    client.register_raw_message_callback(on_raw)

    try:
        print(f"Connecting to: {config.url}")
        print(f"Symbols: {', '.join(config.symbols)}")
        print("Press Ctrl+C to stop\n")

        await client.run(duration=60)

    except KeyboardInterrupt:
        print("\n⏹  Stopping...")
        await client.disconnect()

    stats = client.get_stats()
    print("\n" + "="*80)
    print("📈 Statistics:")
    print("="*80)
    print(f"Demo mode: {stats['demo_mode']}")
    print(f"Messages: {stats['messages_received']}")
    print(f"Quotes: {stats['quotes_received']}")
    print(f"KeepAlive: {stats['keepalive_received']}")
    if stats['uptime_seconds']:
        print(f"Uptime: {stats['uptime_seconds']:.1f}s")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
