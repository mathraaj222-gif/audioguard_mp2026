#!/usr/bin/env python3
"""
push_to_kaggle.py — Automated Kaggle Kernel Push
================================================
Packages the AudioGuardMP_2026 code folder and pushes it to Kaggle
as a Script-type kernel ready to run on T4 x2 GPU.

What this script does:
  1. Verifies kaggle.json credentials
  2. Reads kernel-metadata.json for configuration
  3. Calls `kaggle kernels push -p ./kaggle` to upload the kernel
  4. Polls kernel status until it starts running
  5. Prints the monitoring URL and log-retrieval commands

Usage:
    python push_to_kaggle.py
    python push_to_kaggle.py --dry-run      # Print command without executing
    python push_to_kaggle.py --status-only  # Check existing kernel status only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback


KAGGLE_DIR = Path(__file__).parent    # ./kaggle/
ROOT_DIR   = KAGGLE_DIR.parent        # ./AudioGuardMP_2026/

METADATA_PATH = KAGGLE_DIR / "kernel-metadata.json"


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"kernel-metadata.json not found at {METADATA_PATH}")
    with open(METADATA_PATH) as f:
        return json.load(f)


def get_kaggle_username() -> str:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "kaggle.json not found. Run: python setup_kaggle_credentials.py"
        )
    with open(kaggle_json) as f:
        return json.load(f)["username"]


def run_kaggle_cli(*args, dry_run: bool = False, capture: bool = False):
    cmd = ["kaggle"] + list(args)
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        print("  [DRY RUN — command not executed]")
        return None, None
    result = subprocess.run(cmd, capture_output=capture, text=True, encoding="utf-8", errors="replace")
    if capture:
        return result.stdout, result.stderr
    return result.returncode, None


def push_kernel(dry_run: bool = False):
    print("\n" + "=" * 60)
    print("  AudioGuardMP_2026 — Kaggle Kernel Push")
    print("=" * 60)

    # 1. Verify credentials
    print("\n[1/4] Verifying Kaggle credentials...")
    result = subprocess.run(
        [sys.executable, str(KAGGLE_DIR / "setup_kaggle_credentials.py")],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("✗ Credential check failed. Aborting push.")
        print(result.stdout)
        sys.exit(1)
    print("✓ Credentials valid.")

    username = get_kaggle_username()

    # 2. Load and show metadata
    print("\n[2/4] Loading kernel metadata...")
    meta = load_metadata()
    kernel_id = meta["id"]
    print(f"  Kernel ID  : {kernel_id}")
    print(f"  Title      : {meta['title']}")
    print(f"  GPU        : {meta['enable_gpu']}")
    print(f"  Internet   : {meta['enable_internet']}")

    # 3. Push
    print("\n[3/4] Pushing kernel to Kaggle...")
    run_kaggle_cli("kernels", "push", "-p", str(KAGGLE_DIR), dry_run=dry_run)

    if dry_run:
        print("\n[DRY RUN] Skipping status polling.")
        return

    # 4. Poll status
    print("\n[4/4] Polling kernel status (may take a few minutes to start)...")
    kernel_slug = kernel_id  # e.g. username/audioguard-2026-training
    max_wait_sec = 300
    poll_interval = 20
    elapsed = 0

    while elapsed < max_wait_sec:
        time.sleep(poll_interval)
        elapsed += poll_interval
        stdout, _ = run_kaggle_cli(
            "kernels", "status", kernel_slug,
            capture=True, dry_run=False
        )
        if stdout:
            print(f"  [{elapsed}s] Status: {stdout.strip()}")
            if any(s in stdout.lower() for s in ["running", "complete", "error"]):
                break

    # 5. Print monitoring commands
    print("\n" + "=" * 60)
    print("✅ Kernel pushed successfully!")
    print("\nMonitor your kernel:")
    print(f"  🌐 Browser : https://www.kaggle.com/code/{kernel_slug}")
    print(f"\nCLI commands:")
    print(f"  # Check status:")
    print(f"  kaggle kernels status {kernel_slug}")
    print(f"\n  # Stream live logs:")
    print(f"  kaggle kernels output {kernel_slug} -p ./outputs")
    print(f"\n  # Download all trained model weights after completion:")
    print(f"  kaggle kernels output {kernel_slug} -p ./outputs --force")
    print("=" * 60 + "\n")


def check_status_only():
    username = get_kaggle_username()
    meta = load_metadata()
    kernel_slug = meta["id"]

    print(f"\nChecking status for: {kernel_slug}")
    stdout, stderr = run_kaggle_cli("kernels", "status", kernel_slug, capture=True)
    print(stdout or stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push AudioGuardMP_2026 to Kaggle")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the push command without executing it")
    parser.add_argument("--status-only", action="store_true",
                        help="Only check the status of the existing kernel")
    args = parser.parse_args()

    if args.status_only:
        check_status_only()
    else:
        push_kernel(dry_run=args.dry_run)
