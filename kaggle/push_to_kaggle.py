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
    script_dir = Path(__file__).parent.resolve()
    metadata_path = script_dir / "kernel-metadata.json"
    
    if not metadata_path.exists():
        logger.error(f"kernel-metadata.json not found at {metadata_path}")
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if session_num == 1:
        script = "session1_tca_and_ser_light.py"
    elif session_num == 2:
        script = "session2_ser_heavy.py"
    else:
        script = "train_on_kaggle.py"

    metadata["title"] = "AudioGuardMP 2026 Multimodal Training Pipeline"
    metadata["code_file"] = script

    # Ensure dataset source is present
    dataset_slug = "mathanraaj/audioguars-mp2026"
    if "dataset_sources" not in metadata:
        metadata["dataset_sources"] = []
    
    if dataset_slug not in metadata["dataset_sources"]:
        metadata["dataset_sources"].append(dataset_slug)
        logger.info(f"Added missing dataset source: {dataset_slug}")
    
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
        if dst.exists():
            logger.info(f"Cleaning up {folder}...")
            try:
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    dst.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not remove {dst}: {e}")

def push_kernel(bundle=True):
    """Push the kernel using Kaggle CLI."""
    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent
    kaggle_dir = script_dir
    
    logger.info(f"DEBUG: script_dir={script_dir}")
    logger.info(f"DEBUG: root_dir={root_dir}")
    
    folders_to_bundle = ["tca", "ser"]
    
    try:
        if bundle:
            for folder in folders_to_bundle:
                src = root_dir / folder
                dst = kaggle_dir / folder
                if src.exists():
                    logger.info(f"Bundling {folder} -> {dst}...")
                    try:
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    except Exception as e:
                        logger.error(f"Failed to bundle {folder}: {e}")
                else:
                    logger.warning(f"Source folder {src} not found, skipping.")

        logger.info(f"Pushing kernel to Kaggle from {kaggle_dir}...")
        # Using shell=True for Windows might help if 'kaggle' is a batch file
        result = subprocess.run(f'kaggle kernels push -p "{kaggle_dir}"', shell=True, check=True, capture_output=True, text=True)
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to push kernel (Result Code {e.returncode})")
        if e.stdout: logger.error(f"STDOUT: {e.stdout}")
        if e.stderr: logger.error(f"STDERR: {e.stderr}")
        return False
    finally:
        if bundle:
            cleanup_bundle(kaggle_dir, folders_to_bundle)

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
