# Cerebrium ASGI Example (BETA)

This example demonstrates how to deploy a Cerebrium app using an ASGI server.

Warning: This functionality is in beta and may change in the future.

## Required changes

```toml
[cerebrium.runtime.custom]
port = 5000
entrypoint = "uvicorn app.main:app --host 0.0.0.0 --port 5000"
healthcheck_endpoint = "/health"
```

## Things to note

- The `port` should be set to the port on which the ASGI server will run. Note that your requests to cerebrium will still be made to port 443.
- The `entrypoint` should be set to the command that starts the ASGI server.
- The code lives in the `/app` directory. So, the `entrypoint` should point to the ASGI server in the `/app/main.py`
  file. For Uvicorn, the entrypoint should be `uvicorn app.main:app ...`

## Making a request

```bash
curl --location 'https://api.cortex.cerebrium.ai/v4/<your-project-id>/1-asgi-fastapi-server/predict' \
--header 'Authorization: Bearer <your-rest-api-key>' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "your value here"
}'
```