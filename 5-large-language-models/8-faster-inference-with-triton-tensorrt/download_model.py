#!/usr/bin/env python3
"""
Download HuggingFace model to persistent storage.
Only downloads if model doesn't already exist.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download, login

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = Path("/persistent-storage/models") / MODEL_ID


def download_model():
    """Download model from HuggingFace if not already present."""
    hf_token = os.environ.get("HF_AUTH_TOKEN")
    
    if not hf_token:
        print("WARNING: HF_AUTH_TOKEN not set, model download may fail")
        return
    
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print("✓ Model already exists")
        return
    
    print("Downloading model from HuggingFace...")
    login(token=hf_token)
    snapshot_download(
        MODEL_ID,
        local_dir=str(MODEL_DIR),
        token=hf_token
    )
    print("✓ Model downloaded successfully")


if __name__ == "__main__":
    download_model()