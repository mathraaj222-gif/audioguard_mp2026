"""
session2_ser_heavy.py - Kaggle Session 2 Orchestrator
=====================================================
Models: S2 (Whisper), S3 (W2V-BERT), S4 (W2V2-Large), S5 (WavLM), S7 (Robust)
"""

import subprocess
import logging
import sys
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def _find_code_dir() -> Path:
    """Auto-detect where the source code (tca/ser folders) is located on Kaggle."""
    cwd = Path.cwd()
    if (cwd / "ser").exists():
        return cwd
    
    # Check /kaggle/input recursively
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Optimization: Look for 'ser' folder specifically
        for p in kaggle_input.rglob("ser"):
            if p.is_dir():
                # We need the parent that contains 'ser', 'tca', etc.
                return p.parent
                
        # Fallback to the specific path provided by the user
        user_path = kaggle_input / "datasets/mathanraaj/audioguars-mp2026"
        if user_path.exists() and (user_path / "ser").exists():
            return user_path
                
    return cwd

CODE_DIR = _find_code_dir()
logger.info(f"Using CODE_DIR: {CODE_DIR}")

# Track execution - updated to use absolute paths from CODE_DIR
MODELS_TO_RUN = [
    (CODE_DIR / "ser/train_whisper_ser.py", "S2"),
    (CODE_DIR / "ser/train_wav2vec_bert.py", "S3"),
    (CODE_DIR / "ser/train_wav2vec2_large.py", "S4"),
    (CODE_DIR / "ser/train_wavlm_large.py", "S5"),
    (CODE_DIR / "ser/train_wav2vec2_robust.py", "S7"),
]

def run_script(script_path, model_id):
    if not script_path.exists():
        logger.error(f"[MISSING] {model_id} script not found at {script_path}")
        logger.error("TIP: Ensure you have added your GitHub Repo as a 'Dataset' to this Kaggle Notebook.")
        return

    logger.info(f"[START] Heavy {model_id} ({script_path})...")
    try:
        # Run from project root
        result = subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(CODE_DIR))
        if result.returncode == 0:
            logger.info(f"[OK] {model_id} finished successfully.")
        else:
            logger.error(f"[FAILED] {model_id} failed with return code {result.returncode}.")
    except Exception as e:
        logger.error(f"[ERROR] Critical error running {model_id}: {e}")

def main():
    logger.info("=== STARTING KAGGLE SESSION 2 (HEAVY SER) ===")
    
    Path("/kaggle/working/outputs").mkdir(exist_ok=True, parents=True)

    if not (CODE_DIR / "ser").exists():
        logger.error("!!! CRITICAL: 'ser' folder not found !!!")
        logger.error("Please add your code as a Dataset to the notebook.")
        return

    for script, mid in MODELS_TO_RUN:
        run_script(script, mid)

    # Run SER Evaluation
    logger.info("[EVAL] Running SER Final Evaluation...")
    run_script(CODE_DIR / "ser/evaluate_ser_all.py", "SER_EVAL")

    logger.info("=== KAGGLE SESSION 2 COMPLETE ===")

if __name__ == "__main__":
    main()
