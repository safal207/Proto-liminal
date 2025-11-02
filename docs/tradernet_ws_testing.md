# Tradernet WebSocket testing client

This repository now includes `examples/tradernet_ws_client.py`, a small helper
script that mirrors the features of the `wscat` command line utility.  The
script is useful when you need to experiment with the public Tradernet socket
`wss://wss.tradernet.com/?user_id=<ID>` while running inside a restricted
environment where installing Node.js tooling is inconvenient.

## Quick start

```bash
python examples/tradernet_ws_client.py --user-id 3400204 \
    --tickers GAZP SBER AAPL
```

The command attempts to connect, sends a `subscribeQuotes` request for the
supplied tickers and then streams any incoming JSON payloads to stdout.

The server often requires additional cookies or authorisation headers.  You can
mimic the headers normally issued by a browser session with repeated `--header`
flags:

```bash
python examples/tradernet_ws_client.py --user-id 3400204 \
    --header "Cookie: session=<token>" \
    --header "X-Requested-With: XMLHttpRequest"
```

If you already use `wscat`, you can provide identical payloads via the
`--payload` argument or load them from a JSON file with `--payload-file`.

## Handling handshake failures

In the managed execution environment used for automated evaluation the WebSocket
handshake currently fails with HTTP 403 (forbidden).  The helper script surfaces
that status code so that you can adjust the headers and retry locally with valid
credentials.

The `--proxy` option controls how the script deals with network proxies:

- `--proxy auto` (default) honours proxy variables such as `HTTPS_PROXY`.
- `--proxy none` forces a direct connection.
- `--proxy http://proxy:8080` overrides the proxy with an explicit URL.

Adjusting the proxy settings can be necessary if your shell is already routed
through a corporate or sandbox proxy.

## Comparing with wscat

For parity with the original user request, the behaviour roughly corresponds to
running:

```bash
wscat -c "wss://wss.tradernet.com/?user_id=3400204" \
    -H "Origin: https://app.tradernet.com" \
    -H "User-Agent: Mozilla/5.0 (compatible; ProtoLiminalBot/1.0)" \
    -x '{"cmd": "subscribeQuotes", "tickers": ["GAZP", "SBER", "AAPL"]}'
```

The Python implementation provides additional logging and exception handling, so
it is easier to diagnose issues when the upstream server rejects the connection.
