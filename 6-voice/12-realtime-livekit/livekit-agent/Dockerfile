FROM debian:bookworm-slim

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /cortex

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Download model files
RUN python3 main.py download-files

# Set environment variables
ENV HF_HOME=/cortex/.cache/

# Expose port
EXPOSE 8600

# Set entrypoint
CMD ["python3", "main.py", "start"]