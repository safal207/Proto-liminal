#!/usr/bin/env python3
"""
Detailed Socket.IO connection test

Tests Tradernet Socket.IO connection with verbose logging.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
import socketio
import time

# Enable verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

print("="*80)
print("🔍 Detailed Socket.IO Connection Test")
print("="*80)
print()

# Test different URLs and transports
test_configs = [
    {
        "name": "wsbeta.tradernet.ru (WebSocket)",
        "url": "https://wsbeta.tradernet.ru",
        "transports": ['websocket']
    },
    {
        "name": "wsbeta.tradernet.ru (Polling + WebSocket)",
        "url": "https://wsbeta.tradernet.ru",
        "transports": ['polling', 'websocket']
    },
    {
        "name": "wssdev.tradernet.dev (WebSocket)",
        "url": "https://wssdev.tradernet.dev",
        "transports": ['websocket']
    }
]

for i, config in enumerate(test_configs, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}/{len(test_configs)}: {config['name']}")
    print(f"{'='*80}")
    print(f"URL: {config['url']}")
    print(f"Transports: {config['transports']}")
    print()

    # Create client with logging
    sio = socketio.Client(
        logger=True,
        engineio_logger=True,
        reconnection=False  # Disable for testing
    )

    connected = False
    quotes_received = 0

    @sio.event
    def connect():
        global connected
        connected = True
        print("✅ CONNECTED!")
        print("Subscribing to quotes...")
        sio.emit('notifyQuotes', ['SBER', 'GAZP', 'BTCUSD'])

    @sio.event
    def disconnect():
        print("Disconnected")

    @sio.on('q')
    def on_quote(data):
        global quotes_received
        quotes_received += 1
        print(f"📊 Quote #{quotes_received}: {data}")

    try:
        print("Attempting connection...")
        sio.connect(config['url'], transports=config['transports'])

        if connected:
            print("Waiting for quotes (5 seconds)...")
            time.sleep(5)
            sio.disconnect()

            print(f"\n✅ SUCCESS: Received {quotes_received} quotes")

            if quotes_received == 0:
                print("⚠️  Connected but no quotes received")
                print("   - Subscription might need different format")
                print("   - Or server might not be sending data")

            break  # Stop testing if successful

    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

# Summary
if connected:
    print("\n🎉 Successfully connected to Tradernet Socket.IO!")
    print(f"Working configuration:")
    print(f"  URL: {config['url']}")
    print(f"  Transports: {config['transports']}")
    print(f"  Quotes received: {quotes_received}")
else:
    print("\n❌ All connection attempts failed")
    print("\nPossible issues:")
    print("  1. Server might be down or moved")
    print("  2. Authentication might be required")
    print("  3. Network/firewall blocking connection")
    print("  4. API might have changed endpoints")
    print("\nRecommendations:")
    print("  - Check official Tradernet documentation")
    print("  - Contact Tradernet support for current API details")
    print("  - Try alternative data sources (Alpha Vantage, Yahoo Finance, etc.)")
