#!/bin/bash
set -e

# Download model if not already present
echo "Checking for model..."
python3 /app/download_model.py

# Start Triton Inference Server
echo "Starting Triton Inference Server..."
exec tritonserver \
    --model-repository=/app/model_repository \
    --http-port=8000 \
    --grpc-port=8001 \
    --metrics-port=8002