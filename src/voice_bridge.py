"""
Module: voice_bridge.py
Purpose: Stream forecasts and reflections to voice and dashboard interfaces
Part of LIMINAL ProtoConsciousness MVP — see docs/MVP_SPEC.md
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Deque

LOGGER = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message to be delivered through voice bridge"""
    channel: str  # "voice", "dashboard", "log", "all"
    message: str
    timestamp: str
    priority: str = "normal"  # "low", "normal", "high", "urgent"
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "channel": self.channel,
            "message": self.message,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "metadata": self.metadata
        }


class MessageFormatter:
    """Formats forecasts and reflections into human-readable messages"""

    @staticmethod
    def format_forecast(forecast: Dict) -> str:
        """
        Format a forecast into human-readable text

        Args:
            forecast: Forecast dictionary

        Returns:
            Formatted message string
        """
        entity = forecast.get("entity", "Unknown")
        probability = forecast.get("probability", 0.0)
        forecast_type = forecast.get("forecast_type", "movement")
        horizon = forecast.get("horizon", "24h")

        # Determine probability level
        if probability >= 0.75:
            level = "highly likely"
        elif probability >= 0.6:
            level = "probable"
        elif probability >= 0.4:
            level = "uncertain"
        else:
            level = "unlikely"

        # Format confidence interval if available
        ci = forecast.get("confidence_interval")
        ci_text = ""
        if ci and len(ci) == 2:
            ci_text = f" (confidence range: {ci[0]:.1%} to {ci[1]:.1%})"

        # Build message based on forecast type
        if forecast_type == "movement":
            message = f"{entity} shows {level} positive movement in {horizon} with {probability:.1%} probability{ci_text}."
        elif forecast_type == "sentiment":
            message = f"{entity} sentiment is {level} positive over {horizon} with {probability:.1%} probability{ci_text}."
        elif forecast_type == "volatility":
            message = f"{entity} volatility is {level} in {horizon} with {probability:.1%} probability{ci_text}."
        else:
            message = f"{entity} {forecast_type} forecast: {probability:.1%} probability over {horizon}{ci_text}."

        return message

    @staticmethod
    def format_calibrated_forecast(calibrated: Dict) -> str:
        """
        Format a calibrated forecast

        Args:
            calibrated: Calibrated forecast dictionary

        Returns:
            Formatted message string
        """
        entity = calibrated.get("entity", "Unknown")
        original_prob = calibrated.get("original_probability", 0.0)
        calibrated_prob = calibrated.get("calibrated_probability", 0.0)
        method = calibrated.get("calibration_method", "unknown")

        adjustment = calibrated_prob - original_prob
        direction = "increased" if adjustment > 0 else "decreased"

        message = (
            f"{entity} forecast calibrated using {method} method: "
            f"probability {direction} from {original_prob:.1%} to {calibrated_prob:.1%}."
        )

        return message

    @staticmethod
    def format_reflection(reflection: Dict) -> str:
        """
        Format a RINSE reflection

        Args:
            reflection: Reflection dictionary

        Returns:
            Formatted message string
        """
        insights = reflection.get("insights", [])
        trend = reflection.get("trend", "stable")
        performance_delta = reflection.get("performance_delta", {})

        # Build message
        parts = []

        # Trend analysis
        if trend == "improving":
            parts.append("System performance is improving.")
        elif trend == "degrading":
            parts.append("System performance is degrading.")
        else:
            parts.append("System performance is stable.")

        # Key insights
        if insights:
            parts.append("Key insights:")
            for insight in insights[:3]:  # Top 3 insights
                parts.append(f"  - {insight}")

        # Performance changes
        if performance_delta:
            significant_changes = {
                k: v for k, v in performance_delta.items()
                if abs(v) > 0.05  # 5% threshold
            }
            if significant_changes:
                parts.append("Significant performance changes:")
                for metric, delta in significant_changes.items():
                    direction = "improved" if delta > 0 else "declined"
                    parts.append(f"  - {metric}: {direction} by {abs(delta):.1%}")

        return "\n".join(parts)

    @staticmethod
    def format_rinse_cycle(cycle: Dict) -> str:
        """
        Format a complete RINSE cycle summary

        Args:
            cycle: RINSE cycle dictionary

        Returns:
            Formatted message string
        """
        iteration = cycle.get("iteration", 0)
        adjustments = cycle.get("adjustments", {})
        applied = cycle.get("adjustments_applied", False)
        confidence = cycle.get("evolution_confidence", 0.0)

        parts = [f"RINSE Cycle #{iteration} completed."]

        if applied:
            parts.append(f"Adjustments applied with {confidence:.1%} confidence:")
            for param, value in adjustments.items():
                parts.append(f"  - {param}: {value}")
        else:
            parts.append(f"Adjustments simulated but not applied (confidence {confidence:.1%} below threshold).")

        return "\n".join(parts)

    @staticmethod
    def format_evaluation(evaluation: Dict) -> str:
        """
        Format an evaluation result

        Args:
            evaluation: Evaluation result dictionary

        Returns:
            Formatted message string
        """
        entity = evaluation.get("entity", "Unknown")
        metrics = evaluation.get("metrics", {})
        sample_size = evaluation.get("sample_size", 0)

        parts = [f"Evaluation for {entity} ({sample_size} samples):"]

        # Format key metrics
        if "brier_score" in metrics:
            parts.append(f"  - Brier Score: {metrics['brier_score']:.4f} (lower is better)")
        if "accuracy" in metrics:
            parts.append(f"  - Accuracy: {metrics['accuracy']:.1%}")
        if "calibration_error" in metrics:
            parts.append(f"  - Calibration Error: {metrics['calibration_error']:.4f}")
        if "f1_score" in metrics:
            parts.append(f"  - F1 Score: {metrics['f1_score']:.3f}")

        return "\n".join(parts)


class VoiceBridge:
    """Bridge between LIMINAL system and output interfaces"""

    def __init__(
        self,
        channels: List[str] = None,
        history_size: int = 1000,
        enable_persistence: bool = True
    ):
        """
        Initialize voice bridge

        Args:
            channels: List of enabled channels (default: ["dashboard", "log"])
            history_size: Number of messages to keep in memory
            enable_persistence: Whether to persist messages to disk
        """
        self.channels = channels or ["dashboard", "log"]
        self.history: Deque[Message] = deque(maxlen=history_size)
        self.enable_persistence = enable_persistence

        self.formatter = MessageFormatter()

        self.stats = {
            "messages_sent": 0,
            "messages_by_channel": {ch: 0 for ch in self.channels},
            "messages_by_priority": {p: 0 for p in ["low", "normal", "high", "urgent"]}
        }

        LOGGER.info(f"VoiceBridge initialized with channels: {self.channels}")

    def send_message(
        self,
        message: str,
        channel: str = "all",
        priority: str = "normal",
        metadata: Dict = None
    ) -> Message:
        """
        Send a message through the bridge

        Args:
            message: Message text
            channel: Target channel(s)
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Message object
        """
        msg = Message(
            channel=channel,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            priority=priority,
            metadata=metadata or {}
        )

        # Add to history
        self.history.append(msg)

        # Update stats
        self.stats["messages_sent"] += 1

        # Route to channels
        target_channels = self.channels if channel == "all" else [channel]

        for ch in target_channels:
            if ch in self.channels:
                self._route_to_channel(msg, ch)
                if ch in self.stats["messages_by_channel"]:
                    self.stats["messages_by_channel"][ch] += 1

        if priority in self.stats["messages_by_priority"]:
            self.stats["messages_by_priority"][priority] += 1

        LOGGER.debug(f"Message sent to {channel} with priority {priority}")

        return msg

    def _route_to_channel(self, message: Message, channel: str):
        """
        Route message to specific channel

        Args:
            message: Message to route
            channel: Channel name
        """
        if channel == "log":
            # Log to logger
            log_level = {
                "low": logging.DEBUG,
                "normal": logging.INFO,
                "high": logging.WARNING,
                "urgent": logging.ERROR
            }.get(message.priority, logging.INFO)

            LOGGER.log(log_level, f"[{message.channel}] {message.message}")

        elif channel == "dashboard":
            # In MVP, just log to console
            # In production, would send to WebSocket/SSE
            print(f"[DASHBOARD] {message.message}")

        elif channel == "voice":
            # In MVP, just log
            # In production, would use TTS service
            print(f"[VOICE] {message.message}")

        # Persist if enabled
        if self.enable_persistence:
            self._persist_message(message)

    def _persist_message(self, message: Message):
        """
        Persist message to disk

        Args:
            message: Message to persist
        """
        # Would implement proper persistence here
        # For MVP, could log to JSONL file
        pass

    def broadcast_forecast(self, forecast: Dict, channel: str = "all", priority: str = "normal"):
        """
        Broadcast a forecast

        Args:
            forecast: Forecast dictionary
            channel: Target channel(s)
            priority: Message priority
        """
        message_text = self.formatter.format_forecast(forecast)

        metadata = {
            "type": "forecast",
            "entity": forecast.get("entity"),
            "forecast_type": forecast.get("forecast_type"),
            "probability": forecast.get("probability")
        }

        self.send_message(message_text, channel, priority, metadata)

    def broadcast_calibrated_forecast(
        self,
        calibrated: Dict,
        channel: str = "all",
        priority: str = "normal"
    ):
        """
        Broadcast a calibrated forecast

        Args:
            calibrated: Calibrated forecast dictionary
            channel: Target channel(s)
            priority: Message priority
        """
        message_text = self.formatter.format_calibrated_forecast(calibrated)

        metadata = {
            "type": "calibrated_forecast",
            "entity": calibrated.get("entity"),
            "calibrated_probability": calibrated.get("calibrated_probability")
        }

        self.send_message(message_text, channel, priority, metadata)

    def broadcast_reflection(
        self,
        reflection: Dict,
        channel: str = "all",
        priority: str = "high"
    ):
        """
        Broadcast a RINSE reflection

        Args:
            reflection: Reflection dictionary
            channel: Target channel(s)
            priority: Message priority
        """
        message_text = self.formatter.format_reflection(reflection)

        metadata = {
            "type": "reflection",
            "trend": reflection.get("trend"),
            "insights_count": len(reflection.get("insights", []))
        }

        self.send_message(message_text, channel, priority, metadata)

    def broadcast_rinse_cycle(
        self,
        cycle: Dict,
        channel: str = "all",
        priority: str = "high"
    ):
        """
        Broadcast a complete RINSE cycle

        Args:
            cycle: RINSE cycle dictionary
            channel: Target channel(s)
            priority: Message priority
        """
        message_text = self.formatter.format_rinse_cycle(cycle)

        metadata = {
            "type": "rinse_cycle",
            "iteration": cycle.get("iteration"),
            "adjustments_applied": cycle.get("adjustments_applied")
        }

        self.send_message(message_text, channel, priority, metadata)

    def broadcast_evaluation(
        self,
        evaluation: Dict,
        channel: str = "all",
        priority: str = "normal"
    ):
        """
        Broadcast an evaluation result

        Args:
            evaluation: Evaluation result dictionary
            channel: Target channel(s)
            priority: Message priority
        """
        message_text = self.formatter.format_evaluation(evaluation)

        metadata = {
            "type": "evaluation",
            "entity": evaluation.get("entity"),
            "sample_size": evaluation.get("sample_size")
        }

        self.send_message(message_text, channel, priority, metadata)

    def get_recent_messages(self, n: int = 10, channel: Optional[str] = None) -> List[Message]:
        """
        Get recent messages

        Args:
            n: Number of messages to retrieve
            channel: Filter by channel (optional)

        Returns:
            List of Message objects
        """
        messages = list(self.history)

        if channel:
            messages = [m for m in messages if m.channel == channel or m.channel == "all"]

        return messages[-n:]

    def clear_history(self):
        """Clear message history"""
        self.history.clear()
        LOGGER.info("Message history cleared")

    def get_stats(self) -> Dict:
        """Get bridge statistics"""
        return {
            "messages_sent": self.stats["messages_sent"],
            "messages_by_channel": self.stats["messages_by_channel"],
            "messages_by_priority": self.stats["messages_by_priority"],
            "history_size": len(self.history),
            "enabled_channels": self.channels
        }

    def process_jsonl(
        self,
        input_path: str,
        message_type: str = "forecast",
        channel: str = "all",
        max_messages: int = 1000
    ) -> Dict:
        """
        Process messages from JSONL file

        Args:
            input_path: Path to input JSONL
            message_type: Type of messages ("forecast", "calibrated", "evaluation", "reflection", "cycle")
            channel: Target channel(s)
            max_messages: Maximum messages to process

        Returns:
            Processing statistics
        """
        messages_processed = 0
        messages_broadcast = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if messages_processed >= max_messages:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning(f"Line {line_num}: Invalid JSON")
                    continue

                messages_processed += 1

                # Broadcast based on type
                try:
                    if message_type == "forecast":
                        self.broadcast_forecast(data, channel)
                    elif message_type == "calibrated":
                        self.broadcast_calibrated_forecast(data, channel)
                    elif message_type == "evaluation":
                        self.broadcast_evaluation(data, channel)
                    elif message_type == "reflection":
                        self.broadcast_reflection(data, channel)
                    elif message_type == "cycle":
                        self.broadcast_rinse_cycle(data, channel)
                    else:
                        LOGGER.warning(f"Unknown message type: {message_type}")
                        continue

                    messages_broadcast += 1

                except Exception as exc:
                    LOGGER.error(f"Line {line_num}: Broadcast error: {exc}")

        summary = {
            "messages_processed": messages_processed,
            "messages_broadcast": messages_broadcast,
            "message_type": message_type,
            "channel": channel
        }

        LOGGER.info(f"Processed {messages_broadcast} {message_type} messages")

        return summary


def main():
    """CLI interface for voice bridge"""
    import argparse

    parser = argparse.ArgumentParser(description="Voice bridge for LIMINAL system")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument(
        "--type",
        default="forecast",
        choices=["forecast", "calibrated", "evaluation", "reflection", "cycle"],
        help="Message type"
    )
    parser.add_argument("--channel", default="all", help="Target channel")
    parser.add_argument("--max-messages", type=int, default=1000, dest="max_messages")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Create bridge
    bridge = VoiceBridge()

    # Process messages
    summary = bridge.process_jsonl(
        input_path=args.input,
        message_type=args.type,
        channel=args.channel,
        max_messages=args.max_messages
    )

    # Print summary
    print("\n" + "="*60)
    print("Voice Bridge Summary")
    print("="*60)
    print(f"Messages processed:   {summary['messages_processed']}")
    print(f"Messages broadcast:   {summary['messages_broadcast']}")
    print(f"Message type:         {summary['message_type']}")
    print(f"Channel:              {summary['channel']}")

    stats = bridge.get_stats()
    print(f"\nTotal messages sent:  {stats['messages_sent']}")
    print(f"History size:         {stats['history_size']}")


if __name__ == "__main__":
    main()
