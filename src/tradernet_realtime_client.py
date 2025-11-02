#!/usr/bin/env python3
"""
Tradernet WebSocket Client (Real Format)

Connects to wss://wss.tradernet.com with user_id parameter
and subscribes to real-time quotes using simple JSON array format.

Protocol:
- Connect: wss://wss.tradernet.com/?user_id=YOUR_USER_ID
- Subscribe: ["quotes", ["GAZP", "SBER", "AAPL"]]
- OrderBook: ["orderBook", ["GAZP"]]
- Unsubscribe: ["quotes", []]

This is PURE WebSocket (not Socket.IO!)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Callable, Dict, Any
from collections import deque

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "websockets"])
    import websockets

LOGGER = logging.getLogger(__name__)


@dataclass
class Quote:
    """Real-time market quote from Tradernet"""
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
    url: str = "wss://wss.tradernet.com"
    user_id: str = "3400204"  # User ID parameter
    symbols: List[str] = None
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    ping_interval: int = 30

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["GAZP", "SBER", "AAPL"]

    @property
    def connection_url(self) -> str:
        """Full WebSocket URL with user_id parameter"""
        return f"{self.url}?user_id={self.user_id}"


class TradernetWebSocketClient:
    """
    Pure WebSocket client for Tradernet real-time quotes

    This client uses the ACTUAL Tradernet WebSocket protocol:
    - Pure WebSocket (not Socket.IO)
    - Simple JSON array messages
    - user_id in URL query parameter

    Usage:
        config = TradernetConfig(
            user_id="YOUR_USER_ID",
            symbols=["GAZP", "SBER", "AAPL"]
        )
        client = TradernetWebSocketClient(config)

        def on_quote(quote):
            print(f"{quote.symbol}: ${quote.price}")

        client.register_quote_callback(on_quote)
        await client.run()
    """

    def __init__(self, config: TradernetConfig):
        self.config = config
        self.ws = None
        self.connected = False
        self.subscribed_symbols = set()
        self.reconnect_count = 0

        # Statistics
        self.quotes_received = 0
        self.messages_received = 0
        self.last_quote_time = None
        self.connection_start = None

        # Callbacks
        self.on_quote_callbacks: List[Callable[[Quote], None]] = []
        self.on_connect_callbacks: List[Callable[[], None]] = []
        self.on_disconnect_callbacks: List[Callable[[], None]] = []
        self.on_raw_message_callbacks: List[Callable[[Dict], None]] = []

        LOGGER.info(f"TradernetWebSocketClient initialized")
        LOGGER.info(f"URL: {config.connection_url}")
        LOGGER.info(f"Symbols: {', '.join(config.symbols)}")

    async def connect(self) -> bool:
        """
        Connect to Tradernet WebSocket

        Returns:
            True if connection successful
        """
        try:
            LOGGER.info(f"Connecting to {self.config.connection_url}...")

            self.ws = await websockets.connect(
                self.config.connection_url,
                ping_interval=self.config.ping_interval,
                ping_timeout=10
            )

            self.connected = True
            self.connection_start = datetime.now(timezone.utc)
            self.reconnect_count = 0

            LOGGER.info("✅ Connected to Tradernet WebSocket")

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

                # Trigger callbacks
                for callback in self.on_disconnect_callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()

    async def subscribe_quotes(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time quotes

        Sends: ["quotes", ["GAZP", "SBER", "AAPL"]]

        Args:
            symbols: List of ticker symbols

        Returns:
            True if subscription sent
        """
        if not self.connected or not self.ws:
            LOGGER.error("Not connected - cannot subscribe")
            return False

        try:
            # Tradernet format: ["quotes", [symbols]]
            message = ["quotes", symbols]
            await self.ws.send(json.dumps(message))

            self.subscribed_symbols.update(symbols)
            LOGGER.info(f"📊 Subscribed to quotes: {', '.join(symbols)}")

            return True

        except Exception as e:
            LOGGER.error(f"Subscription failed: {e}")
            return False

    async def unsubscribe_quotes(self) -> bool:
        """
        Unsubscribe from all quotes

        Sends: ["quotes", []]
        """
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

    async def subscribe_order_book(self, symbol: str) -> bool:
        """
        Subscribe to order book (market depth)

        Sends: ["orderBook", ["GAZP"]]

        Args:
            symbol: Ticker symbol
        """
        if not self.connected or not self.ws:
            return False

        try:
            message = ["orderBook", [symbol]]
            await self.ws.send(json.dumps(message))

            LOGGER.info(f"📚 Subscribed to order book: {symbol}")
            return True

        except Exception as e:
            LOGGER.error(f"Order book subscription failed: {e}")
            return False

    def parse_quote_message(self, data: Any) -> Optional[Quote]:
        """
        Parse Tradernet quote message

        Expected format varies, but typically contains:
        - c or ticker: symbol
        - ltp or last or ltr: last price
        - chg: change
        - pchg: percent change
        - vol: volume
        - bid, ask: bid/ask prices
        - h, l, o: high, low, open

        Args:
            data: Message data from WebSocket

        Returns:
            Quote object or None
        """
        try:
            if isinstance(data, dict):
                # Extract symbol
                symbol = data.get("c") or data.get("ticker") or data.get("symbol")

                # Extract price
                price = (
                    data.get("ltp") or
                    data.get("last") or
                    data.get("ltr") or
                    data.get("price")
                )

                if not symbol or price is None:
                    return None

                return Quote(
                    symbol=symbol,
                    price=float(price),
                    timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    volume=data.get("vol") or data.get("volume"),
                    bid=data.get("bid") or data.get("b"),
                    ask=data.get("ask") or data.get("a"),
                    change=data.get("chg") or data.get("change"),
                    change_pct=data.get("pchg") or data.get("change_pct"),
                    high=data.get("h") or data.get("high"),
                    low=data.get("l") or data.get("low"),
                    open_price=data.get("o") or data.get("open"),
                    raw_data=data
                )

            return None

        except Exception as e:
            LOGGER.debug(f"Parse error: {e}")
            return None

    async def receive_messages(self):
        """
        Receive and process messages from WebSocket
        """
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

                # Trigger raw message callbacks
                for callback in self.on_raw_message_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        LOGGER.error(f"Raw callback error: {e}")

                # Try to parse as quote
                quote = self.parse_quote_message(message)

                if quote:
                    self.quotes_received += 1
                    self.last_quote_time = datetime.now(timezone.utc)

                    # Trigger quote callbacks
                    for callback in self.on_quote_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(quote)
                            else:
                                callback(quote)
                        except Exception as e:
                            LOGGER.error(f"Quote callback error: {e}")

            except websockets.exceptions.ConnectionClosed:
                LOGGER.warning("WebSocket connection closed")
                self.connected = False
                break

            except Exception as e:
                LOGGER.error(f"Error receiving message: {e}")
                await asyncio.sleep(0.1)

    async def run_with_reconnect(self):
        """
        Run client with automatic reconnection
        """
        while self.reconnect_count < self.config.max_reconnect_attempts:
            try:
                # Connect
                if not self.connected:
                    success = await self.connect()

                    if not success:
                        self.reconnect_count += 1
                        delay = min(self.config.reconnect_delay * (2 ** self.reconnect_count), 60)
                        LOGGER.info(f"Reconnecting in {delay}s... (attempt {self.reconnect_count})")
                        await asyncio.sleep(delay)
                        continue

                # Receive messages
                await self.receive_messages()

                # Connection closed, reconnect
                if self.reconnect_count < self.config.max_reconnect_attempts:
                    LOGGER.info("Connection lost - reconnecting...")
                    self.connected = False
                    await asyncio.sleep(self.config.reconnect_delay)

            except asyncio.CancelledError:
                LOGGER.info("Client shutdown requested")
                break

            except Exception as e:
                LOGGER.error(f"Unexpected error: {e}")
                self.reconnect_count += 1
                await asyncio.sleep(self.config.reconnect_delay)

        # Cleanup
        await self.disconnect()
        LOGGER.warning("Max reconnection attempts reached")

    async def run(self, duration: Optional[int] = None):
        """
        Run client for specified duration

        Args:
            duration: Duration in seconds (None = run forever)
        """
        if duration:
            # Run for specified time
            try:
                await asyncio.wait_for(
                    self.run_with_reconnect(),
                    timeout=duration
                )
            except asyncio.TimeoutError:
                LOGGER.info(f"Run duration ({duration}s) reached")
                await self.disconnect()
        else:
            # Run forever
            await self.run_with_reconnect()

    def register_quote_callback(self, callback: Callable[[Quote], None]):
        """Register callback for quote updates"""
        self.on_quote_callbacks.append(callback)

    def register_raw_message_callback(self, callback: Callable[[Dict], None]):
        """Register callback for all raw messages (debugging)"""
        self.on_raw_message_callbacks.append(callback)

    def register_connect_callback(self, callback: Callable[[], None]):
        """Register callback for connection event"""
        self.on_connect_callbacks.append(callback)

    def register_disconnect_callback(self, callback: Callable[[], None]):
        """Register callback for disconnection event"""
        self.on_disconnect_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get client statistics"""
        uptime = None
        if self.connection_start:
            uptime = (datetime.now(timezone.utc) - self.connection_start).total_seconds()

        return {
            "connected": self.connected,
            "quotes_received": self.quotes_received,
            "messages_received": self.messages_received,
            "subscribed_symbols": list(self.subscribed_symbols),
            "uptime_seconds": uptime,
            "last_quote_time": self.last_quote_time.isoformat() if self.last_quote_time else None,
            "reconnect_count": self.reconnect_count
        }


async def main():
    """Test client with real Tradernet connection"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("="*80)
    print("🚀 Tradernet WebSocket Client (Real Protocol)")
    print("="*80)
    print()

    # Configure client
    config = TradernetConfig(
        url="wss://wss.tradernet.com",
        user_id="3400204",
        symbols=["GAZP", "SBER", "AAPL"]
    )

    client = TradernetWebSocketClient(config)

    # Quote handler
    def on_quote(quote: Quote):
        change_str = ""
        if quote.change_pct is not None:
            sign = "+" if quote.change_pct >= 0 else ""
            change_str = f"  {sign}{quote.change_pct:.2f}%"

        print(f"📊 {quote.symbol:10s} ${quote.price:12.2f}{change_str}")

    # Raw message handler (for debugging)
    def on_raw_message(msg: Dict):
        print(f"🔍 RAW: {json.dumps(msg)[:200]}")

    client.register_quote_callback(on_quote)
    # client.register_raw_message_callback(on_raw_message)  # Uncomment for debug

    try:
        print(f"Connecting to: {config.connection_url}")
        print(f"Symbols: {', '.join(config.symbols)}")
        print("Press Ctrl+C to stop\n")

        # Run for 60 seconds
        await client.run(duration=60)

    except KeyboardInterrupt:
        print("\n⏹  Stopping...")
        await client.disconnect()

    # Show stats
    stats = client.get_stats()
    print("\n" + "="*80)
    print("📈 Statistics:")
    print("="*80)
    print(f"Messages received: {stats['messages_received']}")
    print(f"Quotes received: {stats['quotes_received']}")
    if stats['uptime_seconds']:
        print(f"Uptime: {stats['uptime_seconds']:.1f}s")
        if stats['quotes_received'] > 0:
            print(f"Quotes per second: {stats['quotes_received'] / stats['uptime_seconds']:.2f}")
    print(f"Reconnections: {stats['reconnect_count']}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
