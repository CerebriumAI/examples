# Cerebrium Websockets Example (BETA)

This example demonstrates how to deploy a Cerebrium app that makes use of websockets.

Warning: This functionality is in beta and may change in the future.

## Required changes

```toml
[cerebrium.runtime.custom]
port = 5000
entrypoint = "uvicorn main:app --host 0.0.0.0 --port 5000"
healthcheck_endpoint = "/health"
```

## Things to note

- The custom runtime is required
- Requests need to be made to `wss://...`

## Making a request

```bash
websocat wss://api.aws.us-east-1.cerebrium.ai/v4/<your-project-id>/<your-app-name>/<your-websocket-function-name>
```
