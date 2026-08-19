#!/usr/bin/env python3
"""Upload fused ControlAI model directly from local disk to Hugging Face Model Hub."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "controlai_fused"
REPO_ID = "atakankahya/ControlAI-Agent"

def main():
    if not MODEL_DIR.exists():
        print(f"Error: Model directory not found at {MODEL_DIR}")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        # Prompt user or read from cache
        from huggingface_hub import get_token
        token = get_token()

    if not token:
        print("Error: Hugging Face token not found. Please provide HF_TOKEN environment variable or login via huggingface_hub.")
        sys.exit(1)

    print(f"Uploading local fused model from {MODEL_DIR} (2.26 GB) to https://huggingface.co/{REPO_ID}...")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("\nUpload complete! Your fine-tuned model is now live on Hugging Face Model Hub.")

if __name__ == "__main__":
    main()
