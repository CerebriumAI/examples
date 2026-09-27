# Browser agent

A headless Chromium instance behind an HTTP endpoint, so an agent can browse the
web as a tool call. Requests describe what to do as a list of steps - navigate,
click, type, wait, extract, screenshot - and get structured JSON back instead of
raw HTML.

Useful when an LLM needs to read a page that is rendered client-side, log into
something, or drive a flow that has no API.

## How it works

| File | Role |
|------|------|
| `cerebrium.toml` | CPU hardware, scaling, and the ASGI entrypoint |
| `main.py` | FastAPI app: launches Chromium once, runs step lists per request |

The base image is `mcr.microsoft.com/playwright/python:v1.63.0-noble`, which
already ships Chromium and the ~40 system libraries it needs, so there is no
`playwright install` step at build time and nothing to keep in
`[cerebrium.dependencies.apt]`. The `playwright` pip version is pinned to the
same `1.63.0` as the image tag, because a mismatched pip version looks for a
browser build that is not in the image.

Two details are what make Chromium work in a container at all, both in
`CHROMIUM_ARGS`:

- `--no-sandbox` - Chromium refuses to start as root without it.
- `--disable-dev-shm-usage` - the default 64MB `/dev/shm` is too small for
  Chromium's shared memory cache, and the symptom is a browser that dies partway
  through a page load rather than a clear error.

Chromium starts once per replica, during ASGI startup, and each request gets a
fresh browser *context*. Contexts are cheap and isolated, so cookies and storage
never leak between callers. `/ready` reports `503` if the browser process has
gone away - Chromium can be OOM-killed long after startup - which lets Cerebrium
replace the replica instead of leaving it accepting work it cannot do.

`replica_concurrency = 4` is the number of pages one replica runs at a time.
Raise it with `memory`, since each open page costs real RAM.

## Deploy

```bash
git clone https://github.com/CerebriumAI/examples.git
cd examples/8-application-demos/6-browser-agent
cerebrium deploy
```

## Steps

| Action | Required | Optional |
|--------|----------|----------|
| `goto` | `url` (`http://` or `https://` only) | `timeout_ms` |
| `click` | `selector` | `timeout_ms` |
| `fill` | `selector` | `value`, `timeout_ms` |
| `press` | `selector`, `value` (the key) | `timeout_ms` |
| `wait_for` | `selector` | `timeout_ms` |
| `extract` | - | `selector` (default `body`), `attribute`, `many` |
| `screenshot` | - | - |

`extract` returns text by default, or the value of `attribute` if given. Set
`many: true` to get every match as a list instead of the first one.

If a step fails - a selector that never appears, a page that times out - the
response is `200` with `"ok": false` and the error on the step that broke, so an
agent can see how far it got and adapt. Only a dead browser returns `503`.

## Making a request

```bash
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/<your-project-id>/6-browser-agent/run' \
--header 'Authorization: Bearer <your-api-key>' \
--header 'Content-Type: application/json' \
--data '{
    "steps": [
        {"action": "goto", "url": "https://news.ycombinator.com"},
        {"action": "wait_for", "selector": ".titleline"},
        {"action": "extract", "selector": ".titleline > a", "many": true}
    ]
}'
```

Response:

```json
{
  "ok": true,
  "url": "https://news.ycombinator.com/",
  "title": "Hacker News",
  "results": [
    {"step": 0, "action": "goto", "ok": true, "status": 200},
    {"step": 1, "action": "wait_for", "ok": true, "selector": ".titleline"},
    {"step": 2, "action": "extract", "ok": true, "data": ["Sqlite-vec", "..."]}
  ],
  "duration_ms": 964.2
}
```

There is also `/scrape` for the one-page case:

```bash
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/<your-project-id>/6-browser-agent/scrape' \
--header 'Authorization: Bearer <your-api-key>' \
--header 'Content-Type: application/json' \
--data '{"url": "https://www.cerebrium.ai", "selector": "h1"}'
```

Pass `"screenshot": true` to also get a base64 full-page PNG back.

## Notes

- Auth is left on (`disable_auth = false`), so requests need a bearer token. An
  endpoint that fetches arbitrary URLs on your behalf is worth keeping behind
  one; `goto` is restricted to `http://` and `https://` for the same reason, so
  `file://` and `chrome://` are not reachable.
- This runs on CPU. Nothing here needs a GPU, and `min_replicas = 0` means it
  costs nothing while idle - at the price of a cold start that includes the
  Chromium launch.
- Sites with bot protection will still block a datacentre IP. If that matters,
  point Playwright at a proxy via `browser.new_context(proxy=...)`.
