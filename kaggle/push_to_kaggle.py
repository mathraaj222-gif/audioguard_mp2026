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
import shutil
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

def bundle_folders(target_dir: Path, folders: list):
    """Copy source folders into the target directory for bundling."""
    root_dir = Path(".").resolve()
    for folder in folders:
        src = root_dir / folder
        dst = target_dir / folder
        if src.exists():
            logger.info(f"Bundling {folder}...")
            # Use dirs_exist_ok=True (Python 3.8+) to handle existing directories
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            logger.warning(f"Source folder {folder} not found, skipping.")

def cleanup_bundle(target_dir: Path, folders: list):
    """Remove bundled folders from the target directory."""
    for folder in folders:
        dst = target_dir / folder
        if dst.exists() and dst.is_dir():
            logger.info(f"Cleaning up {folder}...")
            shutil.rmtree(dst)

def push_kernel(path="kaggle", bundle=True):
    """Push the kernel using Kaggle CLI."""
    folders_to_bundle = ["tca", "ser", "datasets"]
    
    if bundle:
        bundle_folders(Path(path), folders_to_bundle)

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
    finally:
        if bundle:
            cleanup_bundle(Path(path), folders_to_bundle)

def main():
    parser = argparse.ArgumentParser(description="Push AudioGuard to Kaggle")
    parser.add_argument("--session", type=int, choices=[1, 2], help="Session number to push")
    parser.add_argument("--push", action="store_true", default=True, help="Actually push to Kaggle")
    parser.add_argument("--no-bundle", action="store_true", help="Skip bundling tca/ser/datasets folders")
    
    args = parser.parse_args()

    # 1. Update metadata
    if not update_metadata(args.session):
        return

    # 2. Push
    if args.push:
        if push_kernel(bundle=not args.no_bundle):
            logger.info("Kernel pushed successfully! Monitor progress at https://www.kaggle.com/kernels")
        else:
            logger.error("Failed to push kernel.")

if __name__ == "__main__":
    main()
