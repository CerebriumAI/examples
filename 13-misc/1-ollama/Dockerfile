FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates nginx \
    && rm -rf /var/lib/apt/lists/*

ENV OLLAMA_HOST=0.0.0.0
ENV OLLAMA_MODELS=/root/.ollama
ENV OLLAMA_PERSISTENT_DIR=/persistent-storage/ollama
ENV OLLAMA_MODEL_NAME=gemma4-coding
ENV OLLAMA_GGUF_FILE=gemma4-coding-Q4_K_M.gguf
ENV OLLAMA_KEEP_ALIVE=-1

COPY entrypoint.sh /entrypoint.sh
COPY nginx.conf /etc/ollama/nginx.conf
RUN chmod +x /entrypoint.sh

EXPOSE 11434
ENTRYPOINT ["/entrypoint.sh"]
