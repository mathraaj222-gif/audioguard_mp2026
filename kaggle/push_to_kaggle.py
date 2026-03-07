"""
push_to_kaggle.py — Push AudioGuard Training Pipeline to Kaggle
==============================================================
Orchestrates pushing the kernel to Kaggle with session selection.
Usage:
  python push_to_kaggle.py --session 1  # Runs TCA + Light SER
  python push_to_kaggle.py --session 2  # Runs Heavy SER
"""

import os
import json
import argparse
import subprocess
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def update_metadata(session_num):
    """Update kernel-metadata.json with the correct script name."""
    metadata_path = Path("kaggle/kernel-metadata.json")
    if not metadata_path.exists():
        logger.error("kernel-metadata.json not found in kaggle/ directory.")
        return False

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if session_num == 1:
        script = "session1_tca_and_ser_light.py"
    elif session_num == 2:
        script = "session2_ser_heavy.py"
    else:
        script = "train_on_kaggle.py"

    metadata["title"] = "AudioGuardMP 2026 Multimodal Training Pipeline"
    metadata["code_file"] = script
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Updated metadata to run {script}")
    return True

def push_kernel(path="kaggle"):
    """Push the kernel using Kaggle CLI."""
    logger.info(f"Pushing kernel to Kaggle from {path}...")
    try:
        result = subprocess.run(["kaggle", "kernels", "push", "-p", path], check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to push kernel (Result Code {e.returncode})")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push AudioGuard to Kaggle")
    parser.add_argument("--session", type=int, choices=[1, 2], help="Session number to push")
    parser.add_argument("--push", action="store_true", default=True, help="Actually push to Kaggle")
    
    args = parser.parse_args()

    # 1. Update metadata
    if not update_metadata(args.session):
        return

    # 2. Push
    if args.push:
        if push_kernel():
            logger.info("Kernel pushed successfully! Monitor progress at https://www.kaggle.com/kernels")
        else:
            logger.error("Failed to push kernel.")

if __name__ == "__main__":
    main()
