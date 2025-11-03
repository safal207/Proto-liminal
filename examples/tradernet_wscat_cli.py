"""Lightweight `wscat`-style helper for the Tradernet WebSocket feed.

The original user request referenced testing the public Tradernet socket using
`wscat`.  Installing the Node.js utility is not always convenient in automated
sandboxes, so this module provides a tiny Python alternative with a nearly
identical command line surface.

Example usage::

    python examples/tradernet_wscat_cli.py -c \
        "wss://wss.tradernet.com/?user_id=3400204" \
        -H "Origin: https://app.tradernet.com" \
        -H "User-Agent: Mozilla/5.0" \
        -x '{"cmd": "subscribeQuotes", "tickers": ["GAZP", "SBER", "AAPL"]}'

Interactive input from stdin is enabled by default so the tool behaves like
`wscat`.  When running inside a non-interactive CI job you can disable stdin via
``--no-stdin``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
from typing import Any, Dict, Iterable, List, TypedDict

import websockets
from websockets.exceptions import InvalidStatus, InvalidURI

DEFAULT_ORIGIN = "https://app.tradernet.com"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; ProtoLiminalBot/1.0)"
DEFAULT_TICKERS = ["GAZP", "SBER", "AAPL"]


class DemoStream(TypedDict):
    tickers: List[str]
    command: str
    responses: List[Dict[str, Any]]


DEMO_STREAMS: Dict[str, DemoStream] = {
    "quotes": {
        "tickers": ["GAZP", "SBER"],
        "command": "quotes",
        "responses": [
            {
                "quotes": {
                    "GAZP": {
                        "Last": 123.45,
                        "Chg": 1.23,
                        "ChgP": 1.01,
                        "Pc": 122.22,
                        "Volume": 1000,
                    }
                }
            },
            {
                "quotes": {
                    "SBER": {
                        "Last": 281.5,
                        "Chg": -0.7,
                        "ChgP": -0.25,
                        "Pc": 282.2,
                        "Volume": 540,
                    }
                }
            },
        ],
    },
    "orderBook": {
        "tickers": ["SBER"],
        "command": "orderBook",
        "responses": [
            {
                "type": "orderBook",
                "ticker": "SBER",
                "bid": [[281.3, 100], [281.2, 80]],
                "ask": [[281.5, 90], [281.6, 70]],
            }
        ],
    },
    "screenshot": {
        "tickers": ["BTC/USD", "ETH/USD"],
        "command": "quotes",
        "responses": [
            {
                "userData": {
                    "mode": "demo",
                    "marketDataDelay": "fast",
                    "subscription": {
                        "model": "rm",
                        "portfolio": "demo",
                        "balance": 12345.67,
                    },
                }
            },
            {
                "quotes": {
                    "BTC/USD": {
                        "Chg": 0.0,
                        "ChgP": 0.0,
                        "Pc": 100.0,
                        "Last": 100.0,
                        "LastBuy": 100.0,
                        "High": 101.0,
                        "Low": 99.0,
                        "Volume": 42,
                    }
                }
            },
            {
                "quotes": {
                    "ETH/USD": {
                        "Chg": -4.08,
                        "ChgP": -3.8,
                        "Pc": 1500.0,
                        "Last": 1443.83,
                        "LastBuy": 1443.83,
                        "High": 1550.0,
                        "Low": 1400.0,
                        "Volume": 314,
                    }
                }
            },
        ],
    },
}


def _parse_header_arguments(values: Iterable[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for raw in values:
        if ":" not in raw:
            raise argparse.ArgumentTypeError(
                f"Invalid header value '{raw}'. Expected KEY: VALUE format."
            )
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise argparse.ArgumentTypeError(
                f"Invalid header '{raw}'. Header name must not be empty."
            )
        headers[key] = value
    return headers


def _load_message(args: argparse.Namespace) -> str | None:
    if args.execute and args.execute_file:
        raise ValueError("Specify only one of --execute/-x or --execute-file/-f.")

    if args.execute_file:
        with open(args.execute_file, "r", encoding="utf8") as file_obj:
            payload = file_obj.read().strip()
            return payload or None

    if args.execute:
        return args.execute

    if not args.no_default_payload and args.tickers:
        payload = [args.default_command, args.tickers]
        return json.dumps(payload, ensure_ascii=False)

    return None


async def _forward_stdin(ws: websockets.WebSocketClientProtocol) -> None:
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            message = line.rstrip("\r\n")
            if not message:
                continue
            await ws.send(message)
    except asyncio.CancelledError:
        pass
    except websockets.ConnectionClosed:
        logging.debug("Connection closed while reading stdin.")


async def _receive_messages(
    ws: websockets.WebSocketClientProtocol, pretty: bool
) -> None:
    async for message in ws:
        if pretty:
            try:
                decoded = json.loads(message)
            except json.JSONDecodeError:
                print(message)
            else:
                print(json.dumps(decoded, ensure_ascii=False, indent=2))
        else:
            print(message)


async def _run_client(args: argparse.Namespace) -> None:
    headers = _parse_header_arguments(args.header)
    headers.setdefault("Origin", args.origin)
    headers.setdefault("User-Agent", args.user_agent)
    if args.cookie:
        headers.setdefault("Cookie", args.cookie)

    try:
        message = _load_message(args)
    except ValueError as exc:
        logging.error("%s", exc)
        return

    proxy_setting: str | bool | None
    if args.proxy == "auto":
        proxy_setting = True
    elif args.proxy == "none":
        proxy_setting = None
    else:
        proxy_setting = args.proxy

    logging.info("Connecting to %s", args.connect)
    try:
        async with websockets.connect(
            args.connect,
            additional_headers=headers,
            ping_interval=args.ping_interval,
            proxy=proxy_setting,
        ) as ws:
            logging.info("Connected. HTTP headers: %s", headers)
            if message:
                await ws.send(message)
                logging.info("Sent payload: %s", message)

            receiver = asyncio.create_task(_receive_messages(ws, args.pretty_json))
            stdin_task = (
                asyncio.create_task(_forward_stdin(ws)) if not args.no_stdin else None
            )

            try:
                await receiver
            finally:
                if stdin_task:
                    stdin_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stdin_task
    except InvalidStatus as exc:  # type: ignore[misc]
        logging.error("Server rejected the connection: %s", exc)
    except InvalidURI as exc:
        logging.error("Invalid URI '%s': %s", args.connect, exc)
    except OSError as exc:
        logging.error("Network error: %s", exc)


def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _run_demo(args: argparse.Namespace) -> None:
    stream = DEMO_STREAMS[args.demo_stream]

    if args.no_default_payload:
        logging.error("Demo streams require an automatic subscription payload.")
        return

    if args.execute or args.execute_file:
        logging.error("Demo streams cannot be combined with explicit --execute payloads.")
        return

    if args.tickers == DEFAULT_TICKERS:
        args.tickers = list(stream["tickers"])
    elif sorted(args.tickers) != sorted(stream["tickers"]):
        logging.warning(
            "Overriding provided tickers %s with demo defaults %s.",
            args.tickers,
            stream["tickers"],
        )
        args.tickers = list(stream["tickers"])

    args.default_command = stream["command"]

    port = _allocate_port()
    args.connect = f"ws://127.0.0.1:{port}"
    if args.proxy == "auto":
        args.proxy = "none"

    responses: List[str] = [
        json.dumps(response, ensure_ascii=False) for response in stream["responses"]
    ]

    async def handler(conn: "websockets.asyncio.server.ServerConnection") -> None:
        payload = await conn.recv()
        logging.info("Demo server received subscription: %s", payload)
        for response in responses:
            await conn.send(response)
        await conn.close()

    logging.info(
        "Starting local demo server for '%s' stream on %s.",
        args.demo_stream,
        args.connect,
    )

    async with websockets.serve(handler, "127.0.0.1", port):
        await _run_client(args)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal `wscat` replacement focused on Tradernet testing.",
    )
    parser.add_argument(
        "-c",
        "--connect",
        default=None,
        help="Full WebSocket URI. Use --user-id as a shortcut for Tradernet.",
    )
    parser.add_argument(
        "--user-id",
        default="3400204",
        help="Tradernet numeric user identifier used to build the default URI.",
    )
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        default=[],
        help="Additional KEY: VALUE headers. Repeat for multiple entries.",
    )
    parser.add_argument(
        "--origin",
        default=DEFAULT_ORIGIN,
        help="Origin header sent during the handshake.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header used for the handshake request.",
    )
    parser.add_argument(
        "--cookie",
        default=None,
        help="Cookie header value. Useful when authentication is required.",
    )
    parser.add_argument(
        "-x",
        "--execute",
        help="Send a single message after connecting (matches wscat -x).",
    )
    parser.add_argument(
        "-f",
        "--execute-file",
        help="Load the message to send from a file (matches wscat -x @file).",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Ticker symbols for the default subscribeQuotes payload.",
    )
    parser.add_argument(
        "--default-command",
        choices=["quotes", "orderBook"],
        default="quotes",
        help=(
            "Command emitted by the automatic payload. Use 'orderBook' to "
            "subscribe to depth updates."
        ),
    )
    parser.add_argument(
        "--no-default-payload",
        action="store_true",
        help="Connect without sending the automatic subscribeQuotes payload.",
    )
    parser.add_argument(
        "--no-stdin",
        action="store_true",
        help="Disable interactive input from stdin.",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Pretty-print JSON replies for readability.",
    )
    parser.add_argument(
        "--ping-interval",
        type=float,
        default=20.0,
        help="Interval between pings. Set to 0 to disable keepalive pings.",
    )
    parser.add_argument(
        "--proxy",
        default="auto",
        help=(
            "Proxy configuration: 'auto' to honour env vars, 'none' to"
            " disable proxies or provide an explicit proxy URL."
        ),
    )
    parser.add_argument(
        "--demo-stream",
        choices=sorted(DEMO_STREAMS.keys()),
        help=(
            "Start a local demo server and replay deterministic Tradernet "
            "payloads to verify the CLI end-to-end."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.connect:
        args.connect = f"wss://wss.tradernet.com/?user_id={args.user_id}"

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    try:
        if args.demo_stream:
            asyncio.run(_run_demo(args))
        else:
            asyncio.run(_run_client(args))
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")


if __name__ == "__main__":
    main()
