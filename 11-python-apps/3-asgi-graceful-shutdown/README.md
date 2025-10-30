# Cerebrium ASGI Graceful Shutdown Example

This example demonstrates how to deploy a Cerebrium app using an ASGI server with proper graceful shutdown handling and readiness checks.

## Overview

This implementation showcases best practices for running ASGI applications on Cerebrium with:
- Graceful shutdown to ensure in-flight requests complete before termination
- Request tracking and concurrency management
- Proper readiness endpoint to prevent routing traffic to shutting down replicas
- Health check endpoint for basic liveness monitoring

## Configuration

### cerebrium.toml
```toml
[cerebrium.runtime.custom]
port = 8000
dockerfile_path = "./Dockerfile"

[cerebrium.scaling]
replica_concurrency = 1  # Should match max_concurrency in AppState
```

### Dockerfile
```dockerfile
FROM python:3.12-bookworm
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Graceful Shutdown Implementation

### Key Components

**1. AppState Tracking (main.py:11-23)**
The `AppState` dataclass tracks:
- `active_requests`: Current number of in-flight requests
- `shutting_down`: Boolean flag indicating shutdown in progress
- `shutdown_event`: AsyncIO event to signal when all requests complete
- `max_concurrency`: Maximum concurrent requests (should match `replica_concurrency` in cerebrium.toml)

**2. Lifespan Management (main.py:25-52)**
Uses FastAPI's lifespan context manager for graceful shutdown:
- Sets `shutting_down` flag when shutdown signal received
- Waits up to 30 seconds for active requests to complete
- Logs shutdown progress and any timeouts

**3. Request Tracking Middleware (main.py:56-86)**
Tracks all incoming requests with the following logic:
- Rejects new requests with 503 status when shutting down
- Increments/decrements active request counter with thread-safe locks
- Signals shutdown event when last request completes during shutdown

**4. Readiness Endpoint (main.py:92-118)**
Critical for preventing traffic to unavailable replicas:
- Returns 503 when shutting down
- Returns 503 when at max concurrency
- Returns 200 with current state when ready

Without a proper `/ready` endpoint, Cerebrium uses TCP ping which only checks if the port is open, potentially routing traffic to replicas that are shutting down or at capacity.

## Endpoints

### POST /hello
Simple hello world endpoint for testing.

### GET /health
Basic health check - returns healthy if the application is running.

### GET /ready
Readiness check that returns:
- 200: Ready to accept requests
- 503: Not ready (shutting down or at max concurrency)

Response includes current state:
```json
{
  "status": "ready",
  "active_requests": 0,
  "max_concurrency": 1
}
```

## How It Works

1. **Normal Operation**: Requests are tracked, incremented on arrival, decremented on completion
2. **Shutdown Initiated**:
   - `shutting_down` flag is set
   - New requests are rejected with 503
   - System waits for active requests to complete (30s timeout)
3. **Shutdown Complete**: Once all requests finish or timeout is reached, the application terminates

## Making a Request

```bash
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/<your-project-id>/3-asgi-graceful-shutdown/hello' \
--header 'Authorization: Bearer <your-rest-api-key>' \
--header 'Content-Type: application/json' \
--data '{}'
```

## Important Notes

- The `max_concurrency` in `AppState` should match `replica_concurrency` in cerebrium.toml
- The 30-second shutdown timeout is configurable in the lifespan function (main.py:46)
- All request tracking uses asyncio locks to ensure thread safety
- The `/ready` endpoint is essential for proper load balancing during scaling events
