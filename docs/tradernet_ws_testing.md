# Tradernet WebSocket testing helpers

To mirror the original request of "проверить через cli wcat" the repository now
ships with `examples/tradernet_wscat_cli.py`, a lightweight drop-in replacement
for the popular `wscat` tool.  For more verbose Python-based experiments the
previous `examples/tradernet_ws_client.py` module is still available.

## Quick start

```bash
python examples/tradernet_wscat_cli.py -c \
    "wss://wss.tradernet.com/?user_id=3400204" \
    -H "Origin: https://app.tradernet.com" \
    -x '{"cmd": "subscribeQuotes", "tickers": ["GAZP", "SBER", "AAPL"]}' \
    --no-stdin
```

The command connects, sends a `subscribeQuotes` request for the supplied
tickers and then streams any incoming JSON payloads to stdout.  Interactive mode
is enabled by default (just like `wscat`); the `--no-stdin` flag is handy when
running in a non-interactive environment.

The server often requires additional cookies or authorisation headers.  You can
mimic the headers normally issued by a browser session with repeated `-H/--header`
flags:

```bash
python examples/tradernet_wscat_cli.py --user-id 3400204 \
    -H "Cookie: session=<token>" \
    -H "X-Requested-With: XMLHttpRequest"
```

If you already use `wscat`, you can provide identical payloads with the familiar
`-x` option or load them from disk via `-f/--execute-file`.  The helper sends a
`subscribeQuotes` payload by default; use `--no-default-payload` when you want a
pure handshake without any messages being transmitted automatically.

The legacy `tradernet_ws_client.py` helper exposes similar functionality via
explicit `--payload` and `--payload-file` options while also supporting
additional logging controls for debugging.

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
running the real `wscat` utility:

```bash
wscat -c "wss://wss.tradernet.com/?user_id=3400204" \
    -H "Origin: https://app.tradernet.com" \
    -H "User-Agent: Mozilla/5.0 (compatible; ProtoLiminalBot/1.0)" \
    -x '{"cmd": "subscribeQuotes", "tickers": ["GAZP", "SBER", "AAPL"]}'
```

The Python implementations provide additional logging and exception handling, so
it is easier to diagnose issues when the upstream server rejects the connection.
