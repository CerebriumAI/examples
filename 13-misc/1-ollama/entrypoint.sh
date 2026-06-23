#!/bin/sh
set -eu

PERSISTENT_DIR="${OLLAMA_PERSISTENT_DIR:-/persistent-storage/ollama}"
EPHEMERAL_OLLAMA="${OLLAMA_MODELS:-/root/.ollama}"
GGUF_FILE="${OLLAMA_GGUF_FILE:-gemma4-coding-Q4_K_M.gguf}"
GGUF_URL="${OLLAMA_GGUF_URL:-https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF/resolve/main/${GGUF_FILE}}"
MODEL_NAME="${OLLAMA_MODEL_NAME:-gemma4-coding}"
LOCAL_GGUF="/root/.ollama-cache/${GGUF_FILE}"
PERSISTENT_GGUF="${PERSISTENT_DIR}/${GGUF_FILE}"
REGISTRY_MANIFEST="${PERSISTENT_DIR}/manifests/registry.ollama.ai/library/${MODEL_NAME}/latest"
OLLAMA_INTERNAL_HOST="127.0.0.1:11436"
READY_MARKER="/tmp/ollama-ready"
MIN_GGUF_BYTES="${OLLAMA_MIN_GGUF_BYTES:-7000000000}"
WARMUP_PROMPT="${OLLAMA_WARMUP_PROMPT:-Say OK}"

gguf_size() {
  if [ -f "$1" ]; then
    wc -c < "$1" | tr -d ' '
  else
    echo 0
  fi
}

gguf_is_complete() {
  size="$(gguf_size "$1")"
  [ "${size}" -ge "${MIN_GGUF_BYTES}" ]
}

download_gguf() {
  rm -f "${LOCAL_GGUF}"
  echo "Downloading GGUF to ephemeral disk: ${LOCAL_GGUF}" >&2
  curl -fL --retry 5 --retry-delay 5 -o "${LOCAL_GGUF}" "${GGUF_URL}"
  if ! gguf_is_complete "${LOCAL_GGUF}"; then
    echo "Download incomplete: $(gguf_size "${LOCAL_GGUF}") bytes (need >= ${MIN_GGUF_BYTES})" >&2
    exit 1
  fi
}

cache_to_persistent() {
  mkdir -p "${PERSISTENT_DIR}"
  if cp "${LOCAL_GGUF}" "${PERSISTENT_GGUF}"; then
    echo "Cached GGUF on persistent storage: ${PERSISTENT_GGUF}" >&2
    return 0
  fi
  echo "Warning: could not copy GGUF to persistent storage" >&2
  return 1
}

ensure_persistent_gguf() {
  if gguf_is_complete "${PERSISTENT_GGUF}"; then
    echo "GGUF already on persistent storage: ${PERSISTENT_GGUF}" >&2
    return 0
  fi

  rm -f "${PERSISTENT_GGUF}"

  if gguf_is_complete "${LOCAL_GGUF}"; then
    cache_to_persistent || true
    gguf_is_complete "${PERSISTENT_GGUF}"
    return $?
  fi

  download_gguf
  cache_to_persistent || true
  if ! gguf_is_complete "${PERSISTENT_GGUF}"; then
    echo "Persistent cache unavailable; using ephemeral copy at ${LOCAL_GGUF}" >&2
    cp "${LOCAL_GGUF}" "${PERSISTENT_GGUF}" 2>/dev/null || true
  fi

  if gguf_is_complete "${PERSISTENT_GGUF}"; then
    return 0
  fi
  gguf_is_complete "${LOCAL_GGUF}"
}

ensure_local_gguf() {
  if gguf_is_complete "${LOCAL_GGUF}"; then
    return 0
  fi
  if ! gguf_is_complete "${PERSISTENT_GGUF}"; then
    echo "No complete GGUF available to copy locally" >&2
    return 1
  fi
  echo "Copying GGUF to local disk for import: ${LOCAL_GGUF}" >&2
  mkdir -p "$(dirname "${LOCAL_GGUF}")"
  cp "${PERSISTENT_GGUF}" "${LOCAL_GGUF}"
  gguf_is_complete "${LOCAL_GGUF}"
}

registry_backup_complete() {
  if [ ! -f "${REGISTRY_MANIFEST}" ]; then
    return 1
  fi
  if [ ! -d "${PERSISTENT_DIR}/blobs" ]; then
    return 1
  fi
  find "${PERSISTENT_DIR}/blobs" -maxdepth 1 -name 'sha256-*' 2>/dev/null | grep -q .
}

restore_registry_from_persistent() {
  if ! registry_backup_complete; then
    echo "No complete Ollama registry backup on persistent storage" >&2
    return 1
  fi
  echo "Restoring Ollama registry from persistent storage to ${EPHEMERAL_OLLAMA}" >&2
  mkdir -p "${EPHEMERAL_OLLAMA}"
  rm -rf "${EPHEMERAL_OLLAMA}/blobs" "${EPHEMERAL_OLLAMA}/manifests"
  cp -a "${PERSISTENT_DIR}/blobs" "${EPHEMERAL_OLLAMA}/"
  cp -a "${PERSISTENT_DIR}/manifests" "${EPHEMERAL_OLLAMA}/"
}

backup_registry_to_persistent() {
  if [ ! -d "${EPHEMERAL_OLLAMA}/blobs" ] || [ ! -d "${EPHEMERAL_OLLAMA}/manifests" ]; then
    echo "Nothing to back up from ephemeral registry" >&2
    return 1
  fi
  echo "Backing up Ollama registry to persistent storage" >&2
  mkdir -p "${PERSISTENT_DIR}"
  rm -rf "${PERSISTENT_DIR}/blobs" "${PERSISTENT_DIR}/manifests"
  cp -a "${EPHEMERAL_OLLAMA}/blobs" "${PERSISTENT_DIR}/"
  cp -a "${EPHEMERAL_OLLAMA}/manifests" "${PERSISTENT_DIR}/"
}

wait_for_ollama() {
  host="$1"
  timeout="${2:-240}"
  i=0
  while [ "$i" -lt "$timeout" ]; do
    if OLLAMA_HOST="http://${host}" OLLAMA_MODELS="${EPHEMERAL_OLLAMA}" /bin/ollama list >/dev/null 2>&1; then
      return 0
    fi
    i=$((i + 1))
    sleep 1
  done
  return 1
}

model_is_registered() {
  OLLAMA_HOST="http://${OLLAMA_INTERNAL_HOST}" OLLAMA_MODELS="${EPHEMERAL_OLLAMA}" /bin/ollama list 2>/dev/null | grep -F "${MODEL_NAME}" >/dev/null 2>&1
}

model_is_loaded() {
  curl -sf "http://${OLLAMA_INTERNAL_HOST}/api/ps" | grep -F "${MODEL_NAME}" >/dev/null 2>&1
}

warmup_model() {
  echo "Loading ${MODEL_NAME} into GPU memory..." >&2
  if ! curl -sf "http://${OLLAMA_INTERNAL_HOST}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"${WARMUP_PROMPT}\",\"stream\":false,\"keep_alive\":-1,\"options\":{\"num_predict\":8}}"; then
    echo "Warmup generate request failed" >&2
    return 1
  fi

  i=0
  while [ "$i" -lt 240 ]; do
    if model_is_loaded; then
      echo "Model ${MODEL_NAME} loaded in GPU memory" >&2
      return 0
    fi
    i=$((i + 1))
    sleep 1
  done

  echo "Timed out waiting for ${MODEL_NAME} to appear in /api/ps" >&2
  return 1
}

ensure_persistent_gguf
mkdir -p "${EPHEMERAL_OLLAMA}"
rm -f "${READY_MARKER}"

restore_registry_from_persistent || true

echo "Starting Ollama on ${OLLAMA_INTERNAL_HOST} (registry at ${EPHEMERAL_OLLAMA})..." >&2
OLLAMA_HOST="http://${OLLAMA_INTERNAL_HOST}" OLLAMA_MODELS="${EPHEMERAL_OLLAMA}" /bin/ollama serve &
OLLAMA_PID=$!

if ! wait_for_ollama "${OLLAMA_INTERNAL_HOST}" 240; then
  echo "Timed out waiting for Ollama server" >&2
  kill "${OLLAMA_PID}" 2>/dev/null || true
  exit 1
fi

if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
  echo "Ollama server exited unexpectedly during startup" >&2
  wait "${OLLAMA_PID}" 2>/dev/null || true
  exit 1
fi

if model_is_registered; then
  echo "Ollama model already registered: ${MODEL_NAME}" >&2
else
  if ! ensure_local_gguf; then
    kill "${OLLAMA_PID}" 2>/dev/null || true
    exit 1
  fi
  echo "Creating Ollama model ${MODEL_NAME} from ${LOCAL_GGUF}" >&2
  printf 'FROM %s\n' "${LOCAL_GGUF}" > /tmp/Modelfile
  OLLAMA_HOST="http://${OLLAMA_INTERNAL_HOST}" OLLAMA_MODELS="${EPHEMERAL_OLLAMA}" /bin/ollama create "${MODEL_NAME}" -f /tmp/Modelfile
  backup_registry_to_persistent || true
fi

if ! warmup_model; then
  kill "${OLLAMA_PID}" 2>/dev/null || true
  exit 1
fi

touch "${READY_MARKER}"
echo "Starting readiness proxy on :11434..." >&2
nginx -c /etc/ollama/nginx.conf
echo "Ollama is ready to serve ${MODEL_NAME} on :11434" >&2

wait "${OLLAMA_PID}"
nginx -s quit 2>/dev/null || true
