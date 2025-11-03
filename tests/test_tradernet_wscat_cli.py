"""Unit tests for the Tradernet wscat-style helper."""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
from pathlib import Path
from typing import Awaitable, Callable, Dict, List

import pytest
import websockets


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


def _run_with_local_server(
    port: int,
    handler: Callable[["websockets.asyncio.server.ServerConnection"], Awaitable[None]],
    args: argparse.Namespace,
) -> None:
    """Run the CLI client against a provided local WebSocket server."""

    async def _runner() -> None:
        async with websockets.serve(handler, "127.0.0.1", port):
            await cli._run_client(args)

    asyncio.run(_runner())


def _build_base_args(*extra: str) -> argparse.Namespace:
    """Helper that prepares CLI arguments for local testing."""

    # Always disable stdin and proxy auto-detection for deterministic tests.
    return parse_args("--no-stdin", "--proxy", "none", "--ping-interval", "0", *extra)


def _allocate_port() -> int:
    """Allocate an ephemeral TCP port for localhost tests."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_run_client_fetches_quotes(capsys: pytest.CaptureFixture[str]) -> None:
    """The client should send a quotes subscription and stream pretty JSON data."""

    received_messages: List[str] = []
    handshake_headers: Dict[str, str] = {}

    async def handler(conn: "websockets.asyncio.server.ServerConnection") -> None:
        handshake_headers["Origin"] = conn.request.headers.get("Origin", "")
        handshake_headers["User-Agent"] = conn.request.headers.get("User-Agent", "")
        payload = await conn.recv()
        received_messages.append(payload)
        await conn.send(
            json.dumps(
                {"type": "quote", "ticker": "GAZP", "price": 123.45},
                ensure_ascii=False,
            )
        )
        await conn.close()

    port = _allocate_port()

    args = _build_base_args(
        "--connect",
        f"ws://127.0.0.1:{port}",
        "--tickers",
        "GAZP",
        "--origin",
        "https://example.test",
        "--user-agent",
        "UnitTestAgent/1.0",
        "--pretty-json",
    )

    _run_with_local_server(port, handler, args)

    captured = capsys.readouterr()
    output = captured.out.strip()

    assert json.loads(received_messages[0]) == ["quotes", ["GAZP"]]
    assert json.loads(output) == {"type": "quote", "ticker": "GAZP", "price": 123.45}
    assert handshake_headers == {
        "Origin": "https://example.test",
        "User-Agent": "UnitTestAgent/1.0",
    }


def test_run_client_fetches_order_book(capsys: pytest.CaptureFixture[str]) -> None:
    """The client should request order book updates and stream raw JSON responses."""

    received_messages: List[str] = []

    async def handler(conn: "websockets.asyncio.server.ServerConnection") -> None:
        payload = await conn.recv()
        received_messages.append(payload)
        await conn.send(
            json.dumps(
                {
                    "type": "orderBook",
                    "ticker": "SBER",
                    "bid": [[281.3, 100]],
                    "ask": [[281.5, 80]],
                },
                ensure_ascii=False,
            )
        )
        await conn.close()

    port = _allocate_port()

    args = _build_base_args(
        "--connect",
        f"ws://127.0.0.1:{port}",
        "--tickers",
        "SBER",
        "--default-command",
        "orderBook",
    )

    _run_with_local_server(port, handler, args)

    captured = capsys.readouterr()
    output = captured.out.strip()

    assert json.loads(received_messages[0]) == ["orderBook", ["SBER"]]
    assert json.loads(output) == {
        "type": "orderBook",
        "ticker": "SBER",
        "bid": [[281.3, 100]],
        "ask": [[281.5, 80]],
    }


def test_cli_streams_multiple_messages(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI mirrors the raw JSON stream seen in the wscat screenshot."""

    received_messages: List[str] = []

    responses = list(cli.DEMO_STREAMS["screenshot"]["responses"])

    async def handler(conn: "websockets.asyncio.server.ServerConnection") -> None:
        payload = await conn.recv()
        received_messages.append(payload)
        for response in responses:
            await conn.send(json.dumps(response, ensure_ascii=False))
        await conn.close()

    port = _allocate_port()

    args = _build_base_args(
        "--connect",
        f"ws://127.0.0.1:{port}",
        "--tickers",
        "BTC/USD",
        "ETH/USD",
    )

    _run_with_local_server(port, handler, args)

    captured = capsys.readouterr()
    output_lines = [line for line in captured.out.splitlines() if line.strip()]

    assert json.loads(received_messages[0]) == ["quotes", ["BTC/USD", "ETH/USD"]]
    assert output_lines == [json.dumps(response, ensure_ascii=False) for response in responses]


def test_demo_stream_quotes(capsys: pytest.CaptureFixture[str]) -> None:
    """Running the CLI in demo mode should replay deterministic quote updates."""

    args = _build_base_args("--demo-stream", "quotes")

    asyncio.run(cli._run_demo(args))

    captured = capsys.readouterr()
    output_lines = [line for line in captured.out.splitlines() if line.strip()]

    expected = [
        json.dumps(response, ensure_ascii=False)
        for response in cli.DEMO_STREAMS["quotes"]["responses"]
    ]

    assert output_lines == expected

