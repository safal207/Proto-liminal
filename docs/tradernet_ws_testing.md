# Tradernet WebSocket testing helpers

The original request was to "проверить через cli wcat", so this guide now starts
with the canonical Node.js workflow before documenting the Python helpers that
ship with the repository.

## Using the real `wscat`

1. **Install the tool** (requires Node.js):

   ```bash
   npm install -g wscat
   ```

2. **Connect to Tradernet** (replace the `user_id` with your value):

   ```bash
   wscat -c "wss://wss.tradernet.com/?user_id=3400204"
   ```

   Use the test server by swapping the host:

   ```bash
   wscat -c "wss://wssdev.tradernet.dev/?user_id=3400204"
   ```

3. **Subscribe to quotes** once the prompt appears.  Enter the following JSON
   array and press <kbd>Enter</kbd>:

   ```text
   ["quotes", ["GAZP", "SBER", "AAPL"]]
   ```

   For order-book depth data, send:

   ```text
   ["orderBook", ["GAZP"]]
   ```

4. **Watch the stream**.  Incoming messages will be displayed as JSON blobs.

5. **Unsubscribe** from all quotes by sending:

   ```text
   ["quotes", []]
   ```

If the handshake fails, double-check VPN/firewall rules, supply the correct
`user_id`, and ensure any required authentication cookies are passed via repeated
`-H "Cookie: ..."` parameters.

## Python-based alternative (`tradernet_wscat_cli.py`)

The repo still provides a minimal Python reimplementation for sandboxes where
installing Node.js is inconvenient:

```bash
python examples/tradernet_wscat_cli.py -c \
    "wss://wss.tradernet.com/?user_id=3400204" \
    -H "Origin: https://app.tradernet.com" \
    --no-stdin
```

By default the helper sends the same `["quotes", ["..."]]` payload described
above.  Pass `--default-command orderBook` to automatically request depth data
instead, or disable the automatic message entirely with `--no-default-payload`
if you just want to test the handshake.  The CLI mirrors `wscat` flags for
sending custom payloads (`-x`, `-f`) and respects optional headers (`-H`),
cookies, and proxy settings.

The legacy `examples/tradernet_ws_client.py` module is still available for more
verbose experiments with granular logging controls.

### Offline demo mode

When the real Tradernet endpoint is unreachable (for example, during automated
tests that run in a restricted network), use the built-in demo streams to verify
that the CLI still pushes a subscription and renders incoming data:

```bash
python examples/tradernet_wscat_cli.py --demo-stream screenshot --no-stdin
```

The command starts a local WebSocket server, sends the default
`["quotes", ["BTC/USD", "ETH/USD"]]` subscription, and replays the same
responses that appear in the regression screenshot-based test.  Additional demo
streams are available:

- `quotes` – quick sanity check that emits two spot quotes (`GAZP`, `SBER`).
- `orderBook` – single depth update for `SBER`.
- `screenshot` – multi-message session mirroring the documented example output.

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
