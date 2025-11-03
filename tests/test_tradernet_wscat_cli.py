"""Unit tests for the Tradernet wscat-style helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import argparse


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import tradernet_wscat_cli as cli


def parse_args(*extra: str) -> argparse.Namespace:
    """Build a parser and parse the provided command line arguments."""

    parser = cli.build_argument_parser()
    return parser.parse_args(list(extra))


def test_default_quotes_payload() -> None:
    """The helper should emit a quotes subscription by default."""

    args = parse_args("--tickers", "GAZP", "SBER")
    message = cli._load_message(args)

    assert message is not None
    payload = json.loads(message)
    assert payload == ["quotes", ["GAZP", "SBER"]]


def test_default_order_book_payload() -> None:
    """Switching the default command should produce an order book request."""

    args = parse_args("--default-command", "orderBook", "--tickers", "GAZP")
    message = cli._load_message(args)

    assert message is not None
    payload = json.loads(message)
    assert payload == ["orderBook", ["GAZP"]]


def test_no_default_payload() -> None:
    """Disabling the default payload must skip automatic subscriptions."""

    args = parse_args("--no-default-payload")
    message = cli._load_message(args)

    assert message is None

