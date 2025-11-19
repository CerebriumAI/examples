FROM nvcr.io/nvidia/tritonserver:25.10-trtllm-python-py3

# Environment variables
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:$PYTHONPATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/persistent-storage/models
ENV TORCH_CUDA_ARCH_LIST=8.6

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
RUN pip install --break-system-packages \
    huggingface_hub \
    transformers \
    || true

# Create required directories
RUN mkdir -p \
    /app/model_repository/llama3_2/1 \
    /persistent-storage/models \
    /persistent-storage/engines

# Copy application files
COPY --chmod=755 download_model.py start_triton.sh /app/
COPY model.py /app/model_repository/llama3_2/1/
COPY config.pbtxt /app/model_repository/llama3_2/

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["/app/start_triton.sh"]