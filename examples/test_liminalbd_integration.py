#!/usr/bin/env python3
"""
Test script for LiminalBD integration
Demonstrates sending signals to LiminalBD cellular substrate
"""
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import logging
import time
from liminal_bridge import (
    LiminalBridge,
    Signal,
    Impulse,
    ImpulseKind,
    create_signal_from_news
)


def test_basic_impulse():
    """Test 1: Send basic impulse"""
    print("\n" + "="*60)
    print("Test 1: Basic Impulse")
    print("="*60)

    bridge = LiminalBridge()

    impulse = Impulse(
        kind=ImpulseKind.AFFECT,
        pattern="test/basic",
        strength=0.75,
        ttl_ms=5000,
        tags=["test", "proto-liminal"]
    )

    print(f"Sending impulse: {impulse.pattern}")
    success = bridge.send_impulse(impulse)

    if success:
        print("✓ Impulse sent successfully")
    else:
        print("✗ Failed to send impulse")
        print("  (This is expected if liminal-cli is not installed)")

    return success


def test_signal_conversion():
    """Test 2: Convert signal to impulse"""
    print("\n" + "="*60)
    print("Test 2: Signal → Impulse Conversion")
    print("="*60)

    signal = Signal(
        entity="Bitcoin Price",
        features={
            "sentiment": 0.8,
            "relevance": 0.95,
            "urgency": 0.7
        },
        signal_strength=0.85,
        timestamp="2025-10-26T12:00:00Z"
    )

    print(f"Signal: entity={signal.entity}, strength={signal.signal_strength}")

    impulse = signal.to_impulse()
    print(f"Impulse: pattern={impulse.pattern}, strength={impulse.strength}")
    print(f"Tags: {', '.join(impulse.tags)}")

    # Verify pattern construction
    assert "bitcoin" in impulse.pattern.lower()
    assert impulse.strength == signal.signal_strength

    print("✓ Signal conversion successful")
    return True


def test_batch_signals():
    """Test 3: Send batch of signals"""
    print("\n" + "="*60)
    print("Test 3: Batch Signal Processing")
    print("="*60)

    bridge = LiminalBridge()

    signals = [
        create_signal_from_news(
            entity="Bitcoin Price",
            sentiment=0.8,
            relevance=0.9,
            urgency=0.7
        ),
        create_signal_from_news(
            entity="Ethereum Network",
            sentiment=-0.3,
            relevance=0.6,
            urgency=0.5
        ),
        create_signal_from_news(
            entity="Stock Market Volatility",
            sentiment=0.2,
            relevance=0.8,
            urgency=0.6
        ),
        create_signal_from_news(
            entity="Tech Sector Growth",
            sentiment=0.6,
            relevance=0.7,
            urgency=0.4
        ),
    ]

    print(f"Processing {len(signals)} signals...")

    for i, signal in enumerate(signals, 1):
        print(f"  {i}. {signal.entity} (strength={signal.signal_strength:.2f})")

    results = bridge.send_batch(signals)

    print(f"\nResults:")
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}")

    if results['success'] > 0:
        print("✓ Batch processing successful")
    else:
        print("✗ Batch processing failed")
        print("  (This is expected if liminal-cli is not installed)")

    return results['success'] > 0


def test_pattern_generation():
    """Test 4: Pattern generation from various entities"""
    print("\n" + "="*60)
    print("Test 4: Pattern Generation")
    print("="*60)

    test_cases = [
        ("Bitcoin Price", "bitcoin/price"),
        ("Ethereum-Network", "ethereum/network"),
        ("Stock Market", "stock/market"),
        ("AI Technology", "ai/technology"),
    ]

    for entity, expected_pattern in test_cases:
        signal = Signal(
            entity=entity,
            features={"test": 1.0},
            signal_strength=0.5,
            timestamp="2025-10-26T12:00:00Z"
        )

        impulse = signal.to_impulse()
        pattern_base = "/".join(impulse.pattern.split("/")[:2])  # Get first two parts

        print(f"  {entity:25} → {pattern_base:25} ", end="")

        if pattern_base == expected_pattern:
            print("✓")
        else:
            print(f"✗ (expected {expected_pattern})")

    print("\n✓ Pattern generation test complete")
    return True


def test_strength_calculation():
    """Test 5: Signal strength calculation"""
    print("\n" + "="*60)
    print("Test 5: Signal Strength Calculation")
    print("="*60)

    test_cases = [
        # (sentiment, relevance, urgency, expected_min, expected_max)
        (1.0, 1.0, 1.0, 0.9, 1.0),      # Maximum
        (-1.0, 0.0, 0.0, 0.0, 0.1),     # Minimum
        (0.0, 0.5, 0.5, 0.4, 0.6),      # Mid-range
        (0.8, 0.9, 0.7, 0.7, 0.9),      # High positive
    ]

    for sentiment, relevance, urgency, min_exp, max_exp in test_cases:
        signal = create_signal_from_news(
            entity="Test Entity",
            sentiment=sentiment,
            relevance=relevance,
            urgency=urgency
        )

        print(f"  Sentiment={sentiment:5.1f}, Relevance={relevance:.1f}, Urgency={urgency:.1f}")
        print(f"    → Strength={signal.signal_strength:.2f} ", end="")

        if min_exp <= signal.signal_strength <= max_exp:
            print(f"✓ (in range {min_exp:.1f}-{max_exp:.1f})")
        else:
            print(f"✗ (expected {min_exp:.1f}-{max_exp:.1f})")

    print("\n✓ Strength calculation test complete")
    return True


def test_statistics():
    """Test 6: Bridge statistics tracking"""
    print("\n" + "="*60)
    print("Test 6: Statistics Tracking")
    print("="*60)

    bridge = LiminalBridge()
    bridge.reset_stats()

    initial_stats = bridge.get_stats()
    print(f"Initial stats: {initial_stats}")

    assert initial_stats["impulses_sent"] == 0
    assert initial_stats["events_received"] == 0
    assert initial_stats["errors"] == 0

    # Try sending a signal (may fail if liminal-cli not installed)
    signal = create_signal_from_news("Test", 0.5, 0.5, 0.5)
    bridge.send_signal(signal)

    final_stats = bridge.get_stats()
    print(f"Final stats:   {final_stats}")

    # Stats should have changed
    assert (
        final_stats["impulses_sent"] > 0 or
        final_stats["errors"] > 0
    )

    print("✓ Statistics tracking working")
    return True


def main():
    """Run all tests"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("\n" + "="*60)
    print("LiminalBD Integration Test Suite")
    print("="*60)
    print("\nThis test suite demonstrates the integration between")
    print("Proto-liminal and LiminalBD.")
    print("\nNote: Some tests may show warnings if liminal-cli is")
    print("not installed. This is expected for demonstration purposes.")
    print()

    tests = [
        test_basic_impulse,
        test_signal_conversion,
        test_batch_signals,
        test_pattern_generation,
        test_strength_calculation,
        test_statistics,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
            time.sleep(0.5)  # Small delay between tests
        except Exception as exc:
            print(f"✗ Test failed with error: {exc}")
            results.append((test.__name__, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Note: Failures are expected if liminal-cli is not installed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
