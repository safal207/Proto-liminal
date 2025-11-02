#!/usr/bin/env python3
"""
Test Tradernet WebSocket Connection

Quick test script to verify connectivity to Tradernet WebSocket API.
Subscribes to a few symbols and prints incoming quotes.

Usage:
    python examples/test_tradernet_connection.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradernet_client import TradernetClient, TradernetConfig, Quote


async def test_connection():
    """Test basic WebSocket connection and quote streaming"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("="*80)
    print("🧪 Testing Tradernet WebSocket Connection")
    print("="*80)
    print()

    # Configure client for test symbols
    config = TradernetConfig(
        url="wss://wssdev.tradernet.dev",
        symbols=["AAPL", "TSLA", "BTCUSD"],
        reconnect_delay=5,
        ping_interval=30
    )

    client = TradernetClient(config)

    # Quote counter
    quote_count = 0
    quote_by_symbol = {}

    # Callback to track quotes
    def on_quote(quote: Quote):
        nonlocal quote_count
        quote_count += 1

        if quote.symbol not in quote_by_symbol:
            quote_by_symbol[quote.symbol] = 0
        quote_by_symbol[quote.symbol] += 1

        print(f"📊 {quote.symbol:10s} ${quote.price:12.2f}  [{quote.timestamp}]")

    client.register_quote_callback(on_quote)

    try:
        print(f"📡 Connecting to: {config.url}")
        print(f"🎯 Symbols: {', '.join(config.symbols)}")
        print()
        print("Waiting for quotes... (Ctrl+C to stop)")
        print("-"*80)
        print()

        # Connect
        connected = await client.connect()

        if not connected:
            print("❌ Failed to connect")
            return False

        print("✅ Connected successfully!")
        print()

        # Start receiving messages
        receive_task = asyncio.create_task(client.receive_messages())

        # Run for 30 seconds or until interrupted
        try:
            await asyncio.wait_for(receive_task, timeout=30.0)
        except asyncio.TimeoutError:
            print("\n⏱  Test timeout reached (30s)")

    except KeyboardInterrupt:
        print("\n⏹  Test stopped by user")

    finally:
        # Disconnect
        await client.disconnect()

        # Show results
        print()
        print("="*80)
        print("📊 TEST RESULTS")
        print("="*80)
        print(f"Total quotes received: {quote_count}")
        print(f"Unique symbols: {len(quote_by_symbol)}")
        print()

        if quote_by_symbol:
            print("Quotes per symbol:")
            for symbol, count in sorted(quote_by_symbol.items()):
                print(f"  {symbol:10s}: {count:4d}")
        else:
            print("⚠️  No quotes received!")
            print()
            print("Possible issues:")
            print("  - WebSocket endpoint may not be sending data")
            print("  - Message format might be different than expected")
            print("  - Network connectivity issues")
            print()
            print("Try:")
            print("  1. Check if wss://wssdev.tradernet.dev is accessible")
            print("  2. Verify the subscription message format")
            print("  3. Enable debug logging to see raw messages")

        print("="*80)

        return quote_count > 0


async def test_raw_websocket():
    """
    Test raw WebSocket connection without parsing
    (for debugging message format)
    """
    import websockets

    print("\n" + "="*80)
    print("🔍 RAW WEBSOCKET TEST (Debug Mode)")
    print("="*80)
    print()

    url = "wss://wssdev.tradernet.dev"

    try:
        print(f"Connecting to {url}...")
        async with websockets.connect(url, ping_interval=30) as ws:
            print("✅ Connected!")
            print()

            # Send subscription
            subscription = ["quotes", ["AAPL", "TSLA", "BTCUSD"]]
            await ws.send(json.dumps(subscription))
            print(f"📤 Sent: {subscription}")
            print()
            print("📥 Receiving messages (showing first 5):")
            print("-"*80)

            # Receive a few messages to see format
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    print(f"{i+1}. {msg}")
                    print()
                except asyncio.TimeoutError:
                    print(f"⏱  Timeout waiting for message {i+1}")
                    break

            print("-"*80)

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    return True


async def main():
    """Run all tests"""

    # Test 1: High-level client
    print("\n🧪 TEST 1: TradernetClient")
    print("="*80)
    success1 = await test_connection()

    # Test 2: Raw WebSocket (for debugging)
    print("\n🧪 TEST 2: Raw WebSocket Messages")
    success2 = await test_raw_websocket()

    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    print(f"TradernetClient: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Raw WebSocket:   {'✅ PASS' if success2 else '❌ FAIL'}")
    print("="*80)
    print()

    if not success1:
        print("💡 TIP: Run TEST 2 output to see the actual message format")
        print("        Then adjust parse_quote_message() in tradernet_client.py")
        print()


if __name__ == "__main__":
    asyncio.run(main())
