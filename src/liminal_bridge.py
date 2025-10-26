"""
LiminalBD Bridge for Proto-liminal Integration
Provides CBOR-based communication with LiminalBD cellular substrate
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cbor2
except ImportError:
    cbor2 = None

try:
    import websockets
except ImportError:
    websockets = None


LOGGER = logging.getLogger(__name__)


class ImpulseKind(Enum):
    """LiminalBD impulse types"""
    AFFECT = "Affect"
    QUERY = "Query"
    WRITE = "Write"


@dataclass
class Impulse:
    """Represents an impulse sent to LiminalBD cellular fabric"""
    kind: ImpulseKind
    pattern: str
    strength: float
    ttl_ms: int = 3600000  # 1 hour default
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        # Validate strength
        self.strength = max(0.0, min(1.0, self.strength))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CBOR serialization"""
        return {
            "kind": self.kind.value,
            "pattern": self.pattern,
            "strength": self.strength,
            "ttl_ms": self.ttl_ms,
            "tags": self.tags
        }


@dataclass
class Signal:
    """Proto-liminal signal that can be converted to impulse"""
    entity: str
    features: Dict[str, float]
    signal_strength: float
    timestamp: str
    source: str = "proto-liminal"

    def to_impulse(self, kind: ImpulseKind = ImpulseKind.AFFECT) -> Impulse:
        """Convert signal to LiminalBD impulse"""
        # Create pattern from entity and top features
        pattern = self._build_pattern()

        return Impulse(
            kind=kind,
            pattern=pattern,
            strength=self.signal_strength,
            ttl_ms=3600000,
            tags=self._extract_tags()
        )

    def _build_pattern(self) -> str:
        """Build pattern string from entity and features"""
        # Normalize entity name: "Bitcoin Price" -> "bitcoin/price"
        base = self.entity.lower().replace(" ", "/").replace("-", "/")

        # Add top feature if available
        if self.features:
            top_feature = max(self.features.items(), key=lambda x: abs(x[1]))
            feature_name = top_feature[0].lower().replace(" ", "_")
            return f"{base}/{feature_name}"

        return base

    def _extract_tags(self) -> List[str]:
        """Extract tags from entity and features"""
        tags = [self.source]

        # Add entity parts as tags
        entity_parts = self.entity.lower().split()
        tags.extend(entity_parts[:3])  # Limit to 3 tags

        # Add feature categories
        if self.features:
            feature_tags = list(self.features.keys())[:2]
            tags.extend([f.lower().replace(" ", "_") for f in feature_tags])

        return tags


class LiminalBridge:
    """Bridge for communicating with LiminalBD via CBOR protocol"""

    def __init__(
        self,
        cli_path: str = "liminal-cli",
        mode: str = "subprocess",  # "subprocess" or "websocket"
        ws_url: str = "ws://localhost:9001"
    ):
        """
        Initialize bridge

        Args:
            cli_path: Path to liminal-cli executable
            mode: Communication mode ("subprocess" or "websocket")
            ws_url: WebSocket URL for event streaming
        """
        self.cli_path = cli_path
        self.mode = mode
        self.ws_url = ws_url
        self.stats = {
            "impulses_sent": 0,
            "events_received": 0,
            "errors": 0
        }

        if cbor2 is None:
            LOGGER.warning("cbor2 not installed, CBOR encoding will fail")

    def send_impulse(self, impulse: Impulse) -> bool:
        """
        Send impulse to LiminalBD

        Args:
            impulse: Impulse to send

        Returns:
            True if successful, False otherwise
        """
        if cbor2 is None:
            LOGGER.error("cbor2 not installed, cannot send impulse")
            self.stats["errors"] += 1
            return False

        try:
            # Encode impulse as CBOR
            cbor_data = cbor2.dumps(impulse.to_dict())
            hex_data = cbor_data.hex()

            # Send via subprocess (CBOR pipe mode)
            result = subprocess.run(
                [self.cli_path, "--cbor-pipe"],
                input=hex_data.encode(),
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                self.stats["impulses_sent"] += 1
                LOGGER.info(f"Impulse sent: {impulse.pattern} (strength={impulse.strength:.2f})")
                return True
            else:
                LOGGER.error(f"Failed to send impulse: {result.stderr.decode()}")
                self.stats["errors"] += 1
                return False

        except subprocess.TimeoutExpired:
            LOGGER.error("Timeout sending impulse to LiminalBD")
            self.stats["errors"] += 1
            return False
        except FileNotFoundError:
            LOGGER.error(f"liminal-cli not found at {self.cli_path}")
            self.stats["errors"] += 1
            return False
        except Exception as exc:
            LOGGER.error(f"Error sending impulse: {exc}")
            self.stats["errors"] += 1
            return False

    def send_signal(self, signal: Signal) -> bool:
        """
        Convert signal to impulse and send

        Args:
            signal: Proto-liminal signal

        Returns:
            True if successful, False otherwise
        """
        impulse = signal.to_impulse()
        return self.send_impulse(impulse)

    def send_batch(self, signals: List[Signal]) -> Dict[str, int]:
        """
        Send multiple signals as impulses

        Args:
            signals: List of signals to send

        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0}

        for signal in signals:
            if self.send_signal(signal):
                results["success"] += 1
            else:
                results["failed"] += 1

        LOGGER.info(f"Batch sent: {results['success']} succeeded, {results['failed']} failed")
        return results

    async def listen_events(self, callback=None, event_types: List[str] = None):
        """
        Listen to LiminalBD events via WebSocket

        Args:
            callback: Optional callback function(event_dict)
            event_types: Filter for specific event types (e.g., ['awaken', 'introspect'])
        """
        if websockets is None:
            LOGGER.error("websockets not installed, cannot listen to events")
            return

        if event_types is None:
            event_types = ["awaken", "introspect", "harmony"]

        try:
            async with websockets.connect(self.ws_url) as ws:
                LOGGER.info(f"Connected to LiminalBD WebSocket at {self.ws_url}")

                async for message in ws:
                    try:
                        # Parse event (JSON or CBOR)
                        if isinstance(message, bytes):
                            event = cbor2.loads(message)
                        else:
                            event = json.loads(message)

                        event_type = event.get("type")

                        # Filter events
                        if event_type in event_types:
                            self.stats["events_received"] += 1
                            LOGGER.info(f"Event received: {event_type}")

                            if callback:
                                callback(event)

                    except Exception as exc:
                        LOGGER.error(f"Error processing event: {exc}")
                        self.stats["errors"] += 1

        except Exception as exc:
            LOGGER.error(f"WebSocket connection error: {exc}")
            self.stats["errors"] += 1

    def get_stats(self) -> Dict[str, int]:
        """Get bridge statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "impulses_sent": 0,
            "events_received": 0,
            "errors": 0
        }


def create_signal_from_news(
    entity: str,
    sentiment: float,
    relevance: float,
    urgency: float = 0.5
) -> Signal:
    """
    Helper to create signal from news analysis

    Args:
        entity: Entity name (e.g., "Bitcoin", "Tesla")
        sentiment: Sentiment score (-1 to 1)
        relevance: Relevance score (0 to 1)
        urgency: Urgency score (0 to 1)

    Returns:
        Signal object
    """
    # Normalize sentiment to 0-1 range
    normalized_sentiment = (sentiment + 1) / 2

    # Calculate signal strength as weighted average
    signal_strength = (
        0.4 * relevance +
        0.3 * normalized_sentiment +
        0.3 * urgency
    )

    features = {
        "sentiment": sentiment,
        "relevance": relevance,
        "urgency": urgency,
        "normalized_sentiment": normalized_sentiment
    }

    return Signal(
        entity=entity,
        features=features,
        signal_strength=signal_strength,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="proto-liminal"
    )


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create bridge
    bridge = LiminalBridge()

    # Create example signals
    signals = [
        create_signal_from_news("Bitcoin Price", sentiment=0.8, relevance=0.9, urgency=0.7),
        create_signal_from_news("Ethereum Network", sentiment=-0.3, relevance=0.6, urgency=0.5),
        create_signal_from_news("Stock Market", sentiment=0.2, relevance=0.8, urgency=0.4),
    ]

    # Send signals
    results = bridge.send_batch(signals)

    # Print stats
    print("\nBridge Statistics:")
    print(json.dumps(bridge.get_stats(), indent=2))
