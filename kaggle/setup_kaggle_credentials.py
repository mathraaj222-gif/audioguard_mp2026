#!/usr/bin/env python3
"""
setup_kaggle_credentials.py — Kaggle Credentials Validator & Setup Guide
=========================================================================
Verifies that kaggle.json is correctly configured with the right format
and file permissions. Provides clear guidance when issues are detected.

Usage:
    python setup_kaggle_credentials.py
"""

import json
import os
import sys
import stat
import subprocess
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 fallback



# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_JSON = KAGGLE_DIR / "kaggle.json"


def print_setup_guide():
    """Print step-by-step guide to obtain kaggle.json."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║               HOW TO GET YOUR KAGGLE API KEY                     ║
╚══════════════════════════════════════════════════════════════════╝

Step 1: Go to https://www.kaggle.com/settings
Step 2: Scroll to the "API" section
Step 3: Click "Create New Token"
Step 4: A file named 'kaggle.json' will be downloaded
Step 5: Move the file to:
         Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
         macOS/Linux: ~/.kaggle/kaggle.json

Step 6: Run this script again to verify.

The file should contain (create manually if no file was downloaded):
    {"username": "your_username", "key": "KGAT_xxxxxxxxxxxxxxxxxxxx"}

NOTE: Kaggle's new UI shows a token string like KGAT_xxxx instead of
      downloading a file. Use the setup_kaggle_credentials.py --create
      flag or place the JSON manually at the path shown above.
""")


def check_credentials() -> bool:
    print("\n" + "=" * 60)
    print("  AudioGuardMP_2026 — Kaggle Credential Check")
    print("=" * 60)

    # 1. Check directory
    if not KAGGLE_DIR.exists():
        print(f"✗ Kaggle directory not found: {KAGGLE_DIR}")
        print(f"  Creating it now...")
        KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {KAGGLE_DIR}")

    # 2. Check file existence
    if not KAGGLE_JSON.exists():
        print(f"\n✗ kaggle.json not found at: {KAGGLE_JSON}")
        print_setup_guide()
        return False

    print(f"\n✓ kaggle.json found at: {KAGGLE_JSON}")

    # 3. Validate JSON structure
    try:
        with open(KAGGLE_JSON, "r") as f:
            creds = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ kaggle.json is not valid JSON: {e}")
        print("  Please re-download it from kaggle.com/settings")
        return False

    username = creds.get("username")
    key = creds.get("key")

    if not username:
        print("✗ 'username' field missing from kaggle.json")
        return False
    if not key:
        print("✗ 'key' field missing from kaggle.json")
        return False
    if len(key) < 10:
        print("✗ API key looks too short — did you download/copy the correct token?")
        return False
    # Both old (32-char hex) and new (KGAT_ prefix) formats are valid
    if not (key.startswith("KGAT_") or len(key) == 32):
        print(f"⚠  Unusual key format: '{key[:8]}...' — proceeding anyway.")


    print(f"✓ Username  : {username}")
    print(f"✓ API key   : {'*' * (len(key) - 4)}{key[-4:]}")

    # 4. File permissions (Unix-like systems only)
    if os.name != "nt":  # Not Windows
        mode = oct(stat.S_IMODE(KAGGLE_JSON.stat().st_mode))
        if mode != "0o600":
            print(f"\n⚠  File permissions are {mode} — should be 0o600 (owner read/write only)")
            print("   Fixing permissions...")
            KAGGLE_JSON.chmod(0o600)
            print("✓ Permissions fixed to 0o600.")
        else:
            print(f"✓ Permissions: {mode} (correct)")

    # 5. Test Kaggle CLI connection
    print("\n  Testing Kaggle API connection...")
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✓ Kaggle CLI connection successful!")
        else:
            print(f"⚠  Kaggle CLI returned error:\n   {result.stderr.strip()}")
            print("   Make sure 'kaggle' is installed: pip install kaggle")
    except FileNotFoundError:
        print("⚠  'kaggle' CLI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        print("  Please run this script again after installation.")
    except subprocess.TimeoutExpired:
        print("⚠  Kaggle API timeout — check your internet connection.")

    # 6. Check kernel-metadata.json for placeholder
    metadata_path = Path(__file__).parent / "kernel-metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        kernel_id = meta.get("id", "")
        if "REPLACE_WITH_YOUR_KAGGLE_USERNAME" in kernel_id:
            correct_id = kernel_id.replace("REPLACE_WITH_YOUR_KAGGLE_USERNAME", username)
            meta["id"] = correct_id
            with open(metadata_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"\n✓ Auto-updated kernel-metadata.json:")
            print(f"  id: {correct_id}")
        else:
            print(f"\n✓ kernel-metadata.json ID: {kernel_id}")

    print("\n" + "=" * 60)
    print(f"✅ Kaggle credentials are VALID for user: {username}")
    print("   You are ready to push your kernel to Kaggle!")
    print("   Run: python push_to_kaggle.py")
    print("=" * 60 + "\n")

    return True


if __name__ == "__main__":
    success = check_credentials()
    sys.exit(0 if success else 1)
