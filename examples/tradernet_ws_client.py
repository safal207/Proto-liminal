"""Utility for testing the Tradernet public WebSocket feed.

This module provides a small command line client that mimics the behaviour of
`wscat` but allows us to customise HTTP headers and subscription payloads from
Python.  The script focuses on the feed exposed at
``wss://wss.tradernet.com/?user_id=<id>`` and is primarily intended for manual
exploration of the protocol.

Typical usage::

    python examples/tradernet_ws_client.py --user-id 3400204 \
        --tickers GAZP SBER AAPL

If the remote endpoint requires authentication headers (for example cookies
obtained from the Tradernet web application) they can be supplied with the
``--header`` argument.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Dict, Iterable, List

import websockets
from websockets.exceptions import InvalidStatus, InvalidURI

DEFAULT_TICKERS: List[str] = ["GAZP", "SBER", "AAPL"]
DEFAULT_ORIGIN = "https://app.tradernet.com"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; ProtoLiminalBot/1.0)"


def _parse_header_arguments(values: Iterable[str]) -> Dict[str, str]:
    """Parse ``KEY: VALUE`` pairs from the command line.

    The `wscat` utility allows sending additional HTTP headers via repeated
    ``-H`` arguments.  We emulate the same behaviour by accepting multiple
    ``--header`` flags and converting them into a dictionary that can be passed
    to :func:`websockets.connect`.
    """

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


def _load_payload(payload: str | None, payload_file: str | None,
                  tickers: Iterable[str]) -> str | None:
    """Determine the payload to transmit after the connection is open.

    ``wss.tradernet.com`` responds to JSON commands.  When no explicit payload
    is supplied we emit a best-effort ``subscribeQuotes`` request which is the
    most common call described in third party Tradernet API examples.  The
    payload is returned as a JSON string to avoid double encoding errors.
    """

    if payload and payload_file:
        raise ValueError("Specify only one of --payload or --payload-file.")

    if payload_file:
        with open(payload_file, "r", encoding="utf8") as file_obj:
            payload_data = file_obj.read().strip()
            return payload_data or None

    if payload:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            # Treat it as raw text – this mirrors wscat behaviour where the
            # payload is sent verbatim.
            return payload
        else:
            return json.dumps(data)

    # Default payload based on community documentation.  The server may still
    # require extra authentication headers which can be supplied via --header
    # or --cookie.
    command = {
        "cmd": "subscribeQuotes",
        "tickers": list(tickers),
    }
    return json.dumps(command)


async def _stream(uri: str, headers: Dict[str, str], message: str | None,
                  keep_alive: float, proxy: str | bool | None) -> None:
    """Open the WebSocket connection and print messages to stdout."""

    logging.info("Connecting to %s", uri)
    try:
        async with websockets.connect(
            uri,
            additional_headers=headers,
            ping_interval=keep_alive,
            proxy=proxy,
        ) as ws:
            logging.info("Connected. HTTP headers: %s", headers)
            if message:
                await ws.send(message)
                logging.info("Sent payload: %s", message)
            async for reply in ws:
                print(reply)
    except InvalidStatus as exc:  # type: ignore[misc]
        logging.error("Server rejected the connection: %s", exc)
    except InvalidURI as exc:
        logging.error("Invalid URI '%s': %s", uri, exc)
    except OSError as exc:
        logging.error("Network error: %s", exc)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal Tradernet WebSocket testing client.",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="3400204",
        help="Tradernet numeric user identifier. Defaults to 3400204.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="List of ticker symbols to subscribe to.",
    )
    parser.add_argument(
        "--origin",
        default=DEFAULT_ORIGIN,
        help="Origin header value sent during the WebSocket handshake.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header sent with the handshake request.",
    )
    parser.add_argument(
        "--cookie",
        default=None,
        help="Cookie header to include. Useful when the endpoint requires"
             " authentication.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional KEY: VALUE headers. Repeat the flag to provide"
             " multiple headers.",
    )
    parser.add_argument(
        "--payload",
        default=None,
        help="Explicit JSON payload to send immediately after connecting.",
    )
    parser.add_argument(
        "--payload-file",
        default=None,
        help="Load the payload from a file instead of specifying it inline.",
    )
    parser.add_argument(
        "--keep-alive",
        type=float,
        default=20.0,
        help="Ping interval (seconds). Set to 0 to disable periodic pings.",
    )
    parser.add_argument(
        "--proxy",
        default="auto",
        help=(
            "Proxy configuration passed to websockets.connect. Use 'auto'"
            " to respect environment variables, a full proxy URL to override"
            " or 'none' to disable proxy usage."
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
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    headers = {"Origin": args.origin, "User-Agent": args.user_agent}
    if args.cookie:
        headers["Cookie"] = args.cookie
    headers.update(_parse_header_arguments(args.header))

    payload = _load_payload(args.payload, args.payload_file, args.tickers)
    uri = f"wss://wss.tradernet.com/?user_id={args.user_id}"

    proxy_setting: str | bool | None
    proxy_option = args.proxy.lower()
    if proxy_option == "none":
        proxy_setting = None
    elif proxy_option == "auto":
        proxy_setting = True
    else:
        proxy_setting = args.proxy

    asyncio.run(_stream(uri, headers, payload, args.keep_alive, proxy_setting))


if __name__ == "__main__":
    main()
