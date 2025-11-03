#!/usr/bin/env python3
"""
Tradernet Live Connection Test

This script tests connection to real Tradernet WebSocket.
Run this LOCALLY (not in sandbox) where wscat works.

Usage:
    python examples/tradernet_live_test.py

Expected output:
    ✅ Connected!
    📋 userData message received
    💓 keepAlive messages
    📊 Quote updates (if subscribed)
"""

import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets


async def test_connection():
    """Test Tradernet WebSocket connection"""

    url = "wss://wss.tradernet.com/"

    print("="*80)
    print("🧪 Tradernet Live Connection Test")
    print("="*80)
    print(f"URL: {url}")
    print()

    try:
        print("Connecting...")

        # Connect with browser-like headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://tradernet.com"
        }

        async with websockets.connect(url, additional_headers=headers) as ws:
            print("✅ CONNECTED!\n")

            # Wait for initial messages
            print("Waiting for messages...\n")

            message_count = 0
            user_data_received = False

            # Receive first few messages
            for i in range(5):
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    message_count += 1

                    msg = json.loads(raw_msg)
                    msg_type = msg[0] if isinstance(msg, list) and len(msg) > 0 else "unknown"

                    print(f"Message {message_count}: [{msg_type}]")

                    if msg_type == "userData":
                        user_data_received = True
                        data = msg[1] if len(msg) > 1 else {}
                        print(f"  Mode: {data.get('mode')}")
                        print(f"  Version: {data.get('version')}")
                        print(f"  Market delay: {data.get('marketDataDelay')}")

                    elif msg_type == "keepAlive":
                        print("  💓 KeepAlive")

                    else:
                        print(f"  Data: {str(msg)[:200]}")

                    print()

                except asyncio.TimeoutError:
                    print(f"⏱  Timeout waiting for message {i+1}\n")
                    break

            # Try to subscribe to quotes
            if user_data_received:
                print("="*80)
                print("📊 Testing Quote Subscription")
                print("="*80)

                symbols = ["BTC/USD", "ETH/USD", "GAZP"]
                subscription = ["quotes", symbols]

                print(f"Subscribing to: {symbols}")
                await ws.send(json.dumps(subscription))
                print("✅ Subscription sent\n")

                print("Waiting for quotes (10 seconds)...")

                try:
                    for i in range(10):
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        msg = json.loads(raw_msg)

                        if isinstance(msg, list) and len(msg) > 0:
                            msg_type = msg[0]

                            if msg_type not in ("keepAlive",):
                                print(f"📩 {msg_type}: {str(msg)[:150]}")

                except asyncio.TimeoutError:
                    pass

            # Summary
            print("\n" + "="*80)
            print("✅ TEST SUCCESSFUL!")
            print("="*80)
            print(f"Messages received: {message_count}")
            print(f"userData: {'✅' if user_data_received else '❌'}")
            print()
            print("Connection works! You can now use src/tradernet_demo_client.py")

    except websockets.exceptions.InvalidStatus as e:
        print(f"❌ HTTP {e.status_code}: Connection rejected")
        print()
        print("Possible reasons:")
        print("  - Firewall blocking connection")
        print("  - IP not whitelisted")
        print("  - VPN required")
        print()
        print("This script works in your local environment, not in sandbox.")
        return False

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

    return True


async def main():
    print("\n⚠️  IMPORTANT: Run this script LOCALLY, not in sandbox!")
    print("   (In sandbox, network access may be restricted)\n")

    input("Press Enter to continue...")

    success = await test_connection()

    if success:
        print("\n💡 Next steps:")
        print("   1. Use src/tradernet_demo_client.py for full client")
        print("   2. Integrate with realtime_monitor.py")
        print("   3. Run: python src/tradernet_demo_client.py")
    else:
        print("\n💡 Alternative:")
        print("   Use simulated demo: python examples/demo_realtime_simulated.py")


if __name__ == "__main__":
    asyncio.run(main())
