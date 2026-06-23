# Ollama on Cerebrium

Run [Ollama](https://ollama.com) as a custom Docker runtime on Cerebrium, with a Hugging Face GGUF model pulled automatically at startup.

This example uses **yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF** (Q4_K_M, ~7.4 GB) — a Gemma 4 12B coding model fine-tuned on Composer 2.5 and Fable 5 chain-of-thought data.

## How it works

| File | Role |
|------|------|
| `cerebrium.toml` | Custom runtime on port `11434`, GPU hardware, scaling |
| `Dockerfile` | `ollama/ollama:latest` + startup script |
| `entrypoint.sh` | Caches the GGUF on persistent storage, registers the model, warms up GPU load, serves on `:11434` |
| `nginx.conf` | Returns **503** on `:11434` until warmup completes, then proxies to Ollama |

Storage uses a hybrid layout:

- **Persistent storage** (`/persistent-storage/ollama/`) — the raw GGUF file (~7 GB) plus a backup of Ollama's registry (`blobs/`, `manifests/`). The GGUF is downloaded once; the registry backup is written after the first successful import.
- **Ephemeral disk** (`/root/.ollama/`) — Ollama's active registry. Network persistent storage is too slow for Ollama's import/verify step, so imports run against a local GGUF copy and write blobs to ephemeral disk.
- **Ephemeral cache** (`/root/.ollama-cache/`) — local copy of the GGUF used during `ollama create`.

On cold start: restore the registry backup from persistent storage if present (~30s copy), otherwise import from the local GGUF copy (~45s). Then a warmup inference loads the model into GPU memory. Port `:11434` stays closed until warmup completes, so the health check returns **503** until the model can actually serve requests.

Note: Cerebrium's deploy CLI may print "App started successfully" as soon as the container is running — that is not the same as the model being loaded. Wait for `GET /` to return `200` before sending chat requests.

You can also pre-seed the GGUF with the Cerebrium CLI (skips the in-container download):

```bash
# Download locally first, then upload
curl -L -o gemma4-coding-Q4_K_M.gguf \
  https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF/resolve/main/gemma4-coding-Q4_K_M.gguf
cerebrium cp gemma4-coding-Q4_K_M.gguf /ollama/gemma4-coding-Q4_K_M.gguf
```

## Deploy

From this directory (with the [Cerebrium CLI](https://cerebrium.ai/docs) logged in):

```bash
cerebrium deploy
```

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL_NAME` | `gemma4-coding` | Model name in `/api/chat` requests |
| `OLLAMA_GGUF_FILE` | `gemma4-coding-Q4_K_M.gguf` | GGUF filename on persistent storage |
| `OLLAMA_GGUF_URL` | Hugging Face URL | Download source if GGUF not cached |
| `OLLAMA_PERSISTENT_DIR` | `/persistent-storage/ollama` | GGUF + registry backup location |
| `OLLAMA_MODELS` | `/root/.ollama` | Active Ollama registry (ephemeral) |
| `OLLAMA_KEEP_ALIVE` | `-1` | Keep model loaded in VRAM indefinitely |
| `OLLAMA_WARMUP_PROMPT` | `Say OK` | Prompt used for startup GPU warmup |

## Base URL

After deploy, the dashboard shows your API base. Pattern:

```text
https://api.cerebrium.ai/v4/<project-id>/1-ollama
```

This example sets `disable_auth = true`, so no bearer token is required.

## Endpoints

Ollama's native API is exposed directly. Common routes:

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/` | Health check — returns **503** while the model loads, then `Ollama is running` |
| `GET` | `/api/tags` | List loaded models |
| `POST` | `/api/chat` | Native Ollama chat |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |

### Examples

```bash
BASE="https://api.cerebrium.ai/v4/<project-id>/1-ollama"
MODEL="gemma4-coding"

# List models
curl -sS "$BASE/api/tags"

# Chat (Ollama API)
curl -sS -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function to reverse a string.\"}],\"stream\":false}" \
  "$BASE/api/chat"

# Chat (OpenAI-compatible)
curl -sS -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function to reverse a string.\"}]}" \
  "$BASE/v1/chat/completions"
```