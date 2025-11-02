#!/usr/bin/env python3
"""
Tradernet Socket.IO Client for Real-Time Market Data

Connects to Tradernet Socket.IO API and streams real-time quotes
using the correct Socket.IO protocol.

Server: https://wsbeta.tradernet.ru
Protocol: Socket.IO with 'notifyQuotes' subscription
Quote messages: 'q' event

Example subscription: notifyQuotes(['AAPL', 'TSLA', 'BTCUSD'])
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Callable, Dict, Any
from collections import deque

try:
    import socketio
except ImportError:
    print("Installing python-socketio...")
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "python-socketio[client]"])
    import socketio

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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradernetConfig:
    """Configuration for Tradernet Socket.IO client"""
    url: str = "https://wsbeta.tradernet.ru"
    symbols: List[str] = None
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10

    def __post_init__(self):
        if self.symbols is None:
            # Default Russian stocks and crypto
            self.symbols = ["SBER", "GAZP", "LKOH", "BTCUSD", "ETHUSD"]


class TradernetSocketIOClient:
    """
    Socket.IO client for Tradernet API

    Features:
    - Auto-reconnection with exponential backoff
    - Quote subscription management (notifyQuotes)
    - Real-time quote streaming via 'q' events
    - Integration with liminal detection pipeline

    Usage:
        config = TradernetConfig(symbols=["SBER", "GAZP", "BTCUSD"])
        client = TradernetSocketIOClient(config)

        def on_quote(quote):
            print(f"{quote.symbol}: ${quote.price}")

        client.register_quote_callback(on_quote)
        client.connect()
        client.wait()
    """

    def __init__(self, config: TradernetConfig):
        self.config = config
        self.connected = False
        self.subscribed_symbols = set()
        self.reconnect_count = 0

        # Create Socket.IO client
        self.sio = socketio.Client(
            logger=False,
            engineio_logger=False,
            reconnection=True,
            reconnection_attempts=config.max_reconnect_attempts,
            reconnection_delay=config.reconnect_delay
        )

        # Quote buffer for async streaming
        self.quote_queue = asyncio.Queue() if asyncio.get_event_loop().is_running() else None

        # Statistics
        self.quotes_received = 0
        self.last_quote_time = None
        self.connection_start = None

        # Callbacks
        self.on_quote_callbacks: List[Callable[[Quote], None]] = []
        self.on_connect_callbacks: List[Callable[[], None]] = []
        self.on_disconnect_callbacks: List[Callable[[], None]] = []

        # Register Socket.IO event handlers
        self._register_handlers()

        LOGGER.info(f"TradernetSocketIOClient initialized for {len(config.symbols)} symbols")

    def _register_handlers(self):
        """Register Socket.IO event handlers"""

        @self.sio.event
        def connect():
            """Handle connection event"""
            self.connected = True
            self.connection_start = datetime.now(timezone.utc)
            self.reconnect_count = 0

            LOGGER.info("✅ Connected to Tradernet Socket.IO")

            # Subscribe to configured symbols
            if self.config.symbols:
                self.subscribe_quotes(self.config.symbols)

            # Trigger callbacks
            for callback in self.on_connect_callbacks:
                callback()

        @self.sio.event
        def disconnect():
            """Handle disconnection event"""
            self.connected = False
            LOGGER.info("Disconnected from Tradernet")

            # Trigger callbacks
            for callback in self.on_disconnect_callbacks:
                callback()

        @self.sio.event
        def connect_error(data):
            """Handle connection error"""
            LOGGER.error(f"Connection error: {data}")
            self.reconnect_count += 1

        @self.sio.on('q')
        def on_quote_message(data):
            """
            Handle quote message ('q' event)

            Args:
                data: Quote data from Tradernet
            """
            try:
                quote = self.parse_quote_message(data)

                if quote:
                    self.quotes_received += 1
                    self.last_quote_time = datetime.now(timezone.utc)

                    # Add to async queue if available
                    if self.quote_queue:
                        try:
                            self.quote_queue.put_nowait(quote)
                        except:
                            pass  # Queue full

                    # Trigger callbacks
                    for callback in self.on_quote_callbacks:
                        try:
                            callback(quote)
                        except Exception as e:
                            LOGGER.error(f"Callback error: {e}")
                else:
                    LOGGER.debug(f"Failed to parse quote: {data}")

            except Exception as e:
                LOGGER.error(f"Error processing quote: {e}")

    def parse_quote_message(self, data: Any) -> Optional[Quote]:
        """
        Parse Tradernet quote message into Quote object

        Tradernet 'q' message format varies, but typically contains:
        - c: ticker/symbol
        - ltp or last: last traded price
        - chg: change
        - pchg: percent change
        - vol: volume
        - bid: bid price
        - ask: ask price
        - h: high
        - l: low
        - o: open

        Args:
            data: Quote data from 'q' event

        Returns:
            Quote object or None if parsing failed
        """
        try:
            if isinstance(data, dict):
                # Extract fields (adapt based on actual format)
                symbol = data.get("c") or data.get("ticker") or data.get("symbol")

                # Price can be in different fields
                price = (
                    data.get("ltp") or
                    data.get("last") or
                    data.get("price") or
                    data.get("ltr")
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
                    open_price=data.get("o") or data.get("open")
                )

            elif isinstance(data, list) and len(data) > 0:
                # Array format - try first element
                if isinstance(data[0], dict):
                    return self.parse_quote_message(data[0])

            return None

        except Exception as e:
            LOGGER.debug(f"Parse error: {e}")
            return None

    def connect(self) -> bool:
        """
        Connect to Tradernet Socket.IO server

        Returns:
            True if connection successful
        """
        try:
            LOGGER.info(f"Connecting to {self.config.url}...")

            self.sio.connect(
                self.config.url,
                transports=['websocket']
            )

            return True

        except Exception as e:
            LOGGER.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from server"""
        if self.sio.connected:
            self.sio.disconnect()
            LOGGER.info("Disconnected")

    def subscribe_quotes(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time quotes for given symbols

        Args:
            symbols: List of ticker symbols (e.g., ["SBER", "GAZP", "BTCUSD"])

        Returns:
            True if subscription sent
        """
        if not self.connected:
            LOGGER.error("Not connected - cannot subscribe")
            return False

        try:
            # Tradernet subscription: emit 'notifyQuotes' with symbol list
            self.sio.emit('notifyQuotes', symbols)

            self.subscribed_symbols.update(symbols)
            LOGGER.info(f"📊 Subscribed to quotes: {', '.join(symbols)}")

            return True

        except Exception as e:
            LOGGER.error(f"Subscription failed: {e}")
            return False

    def unsubscribe_quotes(self, symbols: List[str]) -> bool:
        """Unsubscribe from quotes"""
        if not self.connected:
            return False

        try:
            # Assuming unsubscribe format
            self.sio.emit('unsubscribeQuotes', symbols)

            self.subscribed_symbols.difference_update(symbols)
            LOGGER.info(f"Unsubscribed from: {', '.join(symbols)}")

            return True

        except Exception as e:
            LOGGER.error(f"Unsubscribe failed: {e}")
            return False

    def wait(self):
        """Block and wait for events (synchronous mode)"""
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.disconnect()

    async def stream_quotes(self):
        """
        Stream quotes as async generator

        Usage:
            async for quote in client.stream_quotes():
                process(quote)
        """
        if not self.quote_queue:
            self.quote_queue = asyncio.Queue()

        while True:
            try:
                quote = await self.quote_queue.get()
                yield quote
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOGGER.error(f"Stream error: {e}")
                await asyncio.sleep(0.1)

    def register_quote_callback(self, callback: Callable[[Quote], None]):
        """Register callback to be called on each quote"""
        self.on_quote_callbacks.append(callback)

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
            "subscribed_symbols": list(self.subscribed_symbols),
            "uptime_seconds": uptime,
            "last_quote_time": self.last_quote_time.isoformat() if self.last_quote_time else None,
            "reconnect_count": self.reconnect_count
        }


def main():
    """Example usage and testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Configure client
    config = TradernetConfig(
        symbols=["SBER", "GAZP", "LKOH", "BTCUSD", "ETHUSD"]
    )

    client = TradernetSocketIOClient(config)

    # Quote handler
    def on_quote(quote: Quote):
        change_str = ""
        if quote.change_pct is not None:
            sign = "+" if quote.change_pct >= 0 else ""
            change_str = f"  {sign}{quote.change_pct:.2f}%"

        print(f"📊 {quote.symbol:10s} ${quote.price:12.2f}{change_str}")

    client.register_quote_callback(on_quote)

    try:
        print("🚀 Starting Tradernet Socket.IO client...")
        print(f"Server: {config.url}")
        print(f"Symbols: {', '.join(config.symbols)}")
        print("Press Ctrl+C to stop\n")

        # Connect and wait
        if client.connect():
            client.wait()

    except KeyboardInterrupt:
        print("\n⏹  Stopping...")
        client.disconnect()

        # Show stats
        stats = client.get_stats()
        print(f"\n📈 Statistics:")
        print(f"  Quotes received: {stats['quotes_received']}")
        if stats['uptime_seconds']:
            print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"  Reconnections: {stats['reconnect_count']}")


if __name__ == "__main__":
    main()
