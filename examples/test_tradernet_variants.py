#!/usr/bin/env python3
"""
Test different Tradernet WebSocket connection variants
"""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)

# Test different connection URLs and parameters
test_configs = [
    {
        "name": "wss.tradernet.com with user_id",
        "url": "wss://wss.tradernet.com/?user_id=3400204",
        "headers": {}
    },
    {
        "name": "wss.tradernet.com without user_id",
        "url": "wss://wss.tradernet.com/",
        "headers": {}
    },
    {
        "name": "wss.tradernet.com with Origin header",
        "url": "wss://wss.tradernet.com/?user_id=3400204",
        "headers": {"Origin": "https://tradernet.com"}
    },
    {
        "name": "wssdev.tradernet.dev",
        "url": "wss://wssdev.tradernet.dev/?user_id=3400204",
        "headers": {}
    },
    {
        "name": "wsbeta.tradernet.ru",
        "url": "wss://wsbeta.tradernet.ru/socket.io/?transport=websocket&EIO=4",
        "headers": {}
    }
]


async def test_connection(config):
    """Test a single connection configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"URL: {config['url']}")
    print(f"Headers: {config['headers']}")
    print("="*80)

    try:
        # Try to connect
        print("Attempting connection...")

        # Build headers
        headers = config['headers'] if config['headers'] else {}

        async with websockets.connect(
            config['url'],
            additional_headers=headers if headers else None,
            ping_interval=30,
            ping_timeout=10
        ) as ws:
            print("✅ CONNECTED!")

            # Try to subscribe
            print("Sending subscription: ['quotes', ['GAZP', 'SBER', 'AAPL']]")
            await ws.send(json.dumps(["quotes", ["GAZP", "SBER", "AAPL"]]))

            # Wait for messages
            print("Waiting for messages (5 seconds)...")
            try:
                for i in range(5):
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    print(f"📩 Message {i+1}: {msg[:200]}")
            except asyncio.TimeoutError:
                print("⏱  No messages received in 5 seconds")

            print(f"\n✅ SUCCESS with: {config['name']}")
            return True

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ HTTP {e.status_code}: {e}")
        return False

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False


async def main():
    print("="*80)
    print("🧪 Tradernet WebSocket Connection Test")
    print("="*80)
    print(f"Testing {len(test_configs)} different configurations...")

    results = []

    for config in test_configs:
        success = await test_connection(config)
        results.append((config['name'], success))
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    print("="*80)

    successful = [name for name, success in results if success]
    if successful:
        print(f"\n🎉 Working configuration(s):")
        for name in successful:
            print(f"  - {name}")
    else:
        print("\n❌ No working configurations found")
        print("\nPossible reasons:")
        print("  1. user_id might be invalid or expired")
        print("  2. Additional authentication might be required")
        print("  3. IP whitelist or VPN required")
        print("  4. API access needs to be enabled in account settings")
        print("\nRecommendations:")
        print("  - Get your own user_id from tradernet.com account")
        print("  - Check if you need to enable API access")
        print("  - Use simulated demo instead: python examples/demo_realtime_simulated.py")


if __name__ == "__main__":
    asyncio.run(main())
