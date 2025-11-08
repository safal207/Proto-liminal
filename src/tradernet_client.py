#!/usr/bin/env python3
"""
Tradernet WebSocket Client for Real-Time Market Data

Connects to Tradernet WebSocket API and streams real-time quotes
for financial instruments. Integrates with Proto-liminal's adaptive
finance framework for liminal state detection and risk management.

WebSocket URL: wss://wssdev.tradernet.dev
Protocol: JSON-based subscription model

Example subscription: ["quotes", ["AAPL", "TSLA", "BTCUSD"]]
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
    """Real-time market quote"""
    symbol: str
    price: float
    timestamp: str
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradernetConfig:
    """Configuration for Tradernet WebSocket client"""
    url: str = "wss://wssdev.tradernet.dev"
    symbols: List[str] = None
    reconnect_delay: int = 5
    ping_interval: int = 30
    ping_timeout: int = 10
    max_reconnect_attempts: int = 10

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "TSLA", "BTCUSD"]


class TradernetClient:
    """
    Asynchronous WebSocket client for Tradernet API

    Features:
    - Auto-reconnection with exponential backoff
    - Heartbeat monitoring
    - Quote subscription management
    - Integration with liminal detection pipeline

    Usage:
        config = TradernetConfig(symbols=["AAPL", "TSLA"])
        client = TradernetClient(config)
        await client.connect()
        await client.subscribe_quotes(["BTCUSD"])

        async for quote in client.stream_quotes():
            print(f"{quote.symbol}: {quote.price}")
    """

    def __init__(self, config: TradernetConfig):
        self.config = config
        self.ws = None
        self.connected = False
        self.subscribed_symbols = set()
        self.reconnect_count = 0

        # Quote buffer for streaming
        self.quote_queue = asyncio.Queue()

        # Statistics
        self.quotes_received = 0
        self.last_quote_time = None
        self.connection_start = None

        # Callbacks
        self.on_quote_callbacks: List[Callable[[Quote], None]] = []
        self.on_connect_callbacks: List[Callable[[], None]] = []
        self.on_disconnect_callbacks: List[Callable[[], None]] = []

        LOGGER.info(f"TradernetClient initialized for {len(config.symbols)} symbols")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Tradernet

        Returns:
            True if connection successful, False otherwise
        """
        try:
            LOGGER.info(f"Connecting to {self.config.url}...")

            self.ws = await websockets.connect(
                self.config.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout
            )

            self.connected = True
            self.connection_start = datetime.now(timezone.utc)
            self.reconnect_count = 0

            LOGGER.info("✅ Connected to Tradernet WebSocket")

            # Subscribe to configured symbols
            if self.config.symbols:
                await self.subscribe_quotes(self.config.symbols)

            # Trigger callbacks
            for callback in self.on_connect_callbacks:
                callback()

            return True

        except Exception as e:
            LOGGER.error(f"Connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close WebSocket connection gracefully"""
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
                    callback()

    async def subscribe_quotes(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time quotes for given symbols

        Args:
            symbols: List of ticker symbols (e.g., ["AAPL", "TSLA", "BTCUSD"])

        Returns:
            True if subscription successful
        """
        if not self.connected or not self.ws:
            LOGGER.error("Not connected - cannot subscribe")
            return False

        try:
            # Tradernet subscription format: ["quotes", ["SYMBOL1", "SYMBOL2", ...]]
            subscription_msg = ["quotes", symbols]
            await self.ws.send(json.dumps(subscription_msg))

            self.subscribed_symbols.update(symbols)
            LOGGER.info(f"📊 Subscribed to quotes: {', '.join(symbols)}")

            return True

        except Exception as e:
            LOGGER.error(f"Subscription failed: {e}")
            return False

    async def unsubscribe_quotes(self, symbols: List[str]) -> bool:
        """Unsubscribe from quote updates"""
        if not self.connected or not self.ws:
            return False

        try:
            # Assuming unsubscribe format similar to subscribe
            unsubscribe_msg = ["unsubscribe", symbols]
            await self.ws.send(json.dumps(unsubscribe_msg))

            self.subscribed_symbols.difference_update(symbols)
            LOGGER.info(f"Unsubscribed from: {', '.join(symbols)}")

            return True

        except Exception as e:
            LOGGER.error(f"Unsubscribe failed: {e}")
            return False

    def parse_quote_message(self, msg: Dict[str, Any]) -> Optional[Quote]:
        """
        Parse incoming WebSocket message into Quote object

        Args:
            msg: Raw message from Tradernet (JSON parsed)

        Returns:
            Quote object or None if parsing failed
        """
        try:
            # Common quote message fields (adapt based on actual Tradernet format)
            # Example: {"type": "quote", "symbol": "AAPL", "price": 150.25, ...}

            if isinstance(msg, dict):
                # Dictionary format
                symbol = msg.get("symbol") or msg.get("c") or msg.get("ticker")
                price = msg.get("price") or msg.get("ltp") or msg.get("last")

                if not symbol or price is None:
                    return None

                return Quote(
                    symbol=symbol,
                    price=float(price),
                    timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    volume=msg.get("volume") or msg.get("vol"),
                    bid=msg.get("bid") or msg.get("b"),
                    ask=msg.get("ask") or msg.get("a"),
                    change=msg.get("change") or msg.get("chg"),
                    change_pct=msg.get("change_pct") or msg.get("pchg")
                )

            elif isinstance(msg, list) and len(msg) >= 2:
                # Array format: [type, data]
                msg_type = msg[0]

                if msg_type == "q" or msg_type == "quote":
                    data = msg[1]
                    if isinstance(data, dict):
                        return self.parse_quote_message(data)

            return None

        except Exception as e:
            LOGGER.debug(f"Failed to parse quote: {e}")
            return None

    async def receive_messages(self):
        """
        Continuously receive and process WebSocket messages
        (Internal method - runs in background)
        """
        while self.connected and self.ws:
            try:
                raw_msg = await self.ws.recv()
                msg = json.loads(raw_msg)

                # Parse quote
                quote = self.parse_quote_message(msg)

                if quote:
                    self.quotes_received += 1
                    self.last_quote_time = datetime.now(timezone.utc)

                    # Add to queue
                    await self.quote_queue.put(quote)

                    # Trigger callbacks
                    for callback in self.on_quote_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(quote)
                            else:
                                callback(quote)
                        except Exception as e:
                            LOGGER.error(f"Callback error: {e}")

                else:
                    # Log other message types for debugging
                    LOGGER.debug(f"Non-quote message: {msg}")

            except websockets.exceptions.ConnectionClosed:
                LOGGER.warning("WebSocket connection closed")
                self.connected = False
                break

            except json.JSONDecodeError as e:
                LOGGER.warning(f"Invalid JSON: {e}")
                continue

            except Exception as e:
                LOGGER.error(f"Error receiving message: {e}")
                await asyncio.sleep(1)

    async def stream_quotes(self):
        """
        Stream quotes as async generator

        Usage:
            async for quote in client.stream_quotes():
                process(quote)
        """
        while True:
            try:
                quote = await self.quote_queue.get()
                yield quote
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"Stream error: {e}")
                await asyncio.sleep(0.1)

    async def run_with_reconnect(self):
        """
        Run client with automatic reconnection

        Maintains persistent connection with exponential backoff
        on failures. Use this for production deployment.
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

                # If we get here, connection was closed
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
        LOGGER.warning("Max reconnection attempts reached - stopped")

    def register_quote_callback(self, callback: Callable[[Quote], None]):
        """Register callback to be called on each quote"""
        self.on_quote_callbacks.append(callback)

    def get_stats(self) -> dict:
        """Get client statistics"""
        uptime = None
        if self.connection_start:
            uptime = (datetime.now(timezone.utc) - self.connection_start).total_seconds()

        return {
            "connected": self.connected,
            "quotes_received": self.quotes_received,
            "subscribed_symbols": list(self.subscribed_symbols),
            "uptime_seconds": uptime,
            "last_quote_time": self.last_quote_time.isoformat() if self.last_quote_time else None,
            "reconnect_count": self.reconnect_count
        }


async def main():
    """Example usage and testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Configure client
    config = TradernetConfig(
        symbols=["AAPL", "TSLA", "BTCUSD"]
    )

    client = TradernetClient(config)

    # Quote handler
    def on_quote(quote: Quote):
        print(f"📊 {quote.symbol:8s} ${quote.price:10.2f}  {quote.timestamp}")

    client.register_quote_callback(on_quote)

    try:
        print("🚀 Starting Tradernet client...")
        print(f"Symbols: {', '.join(config.symbols)}")
        print("Press Ctrl+C to stop\n")

        # Run with auto-reconnect
        await client.run_with_reconnect()

    except KeyboardInterrupt:
        print("\n⏹  Stopping...")
        await client.disconnect()

        # Show stats
        stats = client.get_stats()
        print(f"\n📈 Statistics:")
        print(f"  Quotes received: {stats['quotes_received']}")
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
