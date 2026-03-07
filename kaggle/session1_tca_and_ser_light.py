"""
session1_tca_and_ser_light.py - Kaggle Session 1 Orchestrator
=============================================================
Execution Order: T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> S1 -> S6
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
    # 1. Check if we pushed everything directly to working dir (unlikely with -p kaggle)
    if (Path.cwd() / "tca").exists():
        return Path.cwd()
    
    # 2. Check Kaggle Input (Dataset method - recommended)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Look for our specific dataset name
        preferred = kaggle_input / "audioguardmp-2026"
        if preferred.exists():
            return preferred
        # Fallback: find any directory with 'tca' in it
        for d in kaggle_input.iterdir():
            if d.is_dir() and (d / "tca").exists():
                return d
                
    return Path.cwd() # Default failure

CODE_DIR = _find_code_dir()
logger.info(f"Using CODE_DIR: {CODE_DIR}")

# Track execution - updated to use absolute paths from CODE_DIR
MODELS_TO_RUN = [
    (CODE_DIR / "tca/train_bert_nli_baseline.py", "T1"),
    (CODE_DIR / "tca/train_hatebert.py", "T2"),
    (CODE_DIR / "tca/train_deberta_large.py", "T3"),
    (CODE_DIR / "tca/train_roberta_dynabench.py", "T4"),
    (CODE_DIR / "tca/train_deberta_small_ce.py", "T5"),
    (CODE_DIR / "tca/train_twitter_roberta.py", "T6"),
    (CODE_DIR / "ser/train_lstm_baseline.py", "S1"),
    (CODE_DIR / "ser/train_hubert_er.py", "S6"),
]

def run_script(script_path, model_id):
    if not script_path.exists():
        logger.error(f"[MISSING] {model_id} script not found at {script_path}")
        logger.error("TIP: Ensure you have added your GitHub Repo as a 'Dataset' to this Kaggle Notebook.")
        return

    logger.info(f"[START] {model_id} ({script_path})...")
    try:
        # Run from project root so relative paths inside scripts work
        result = subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(CODE_DIR))
        if result.returncode == 0:
            logger.info(f"[OK] {model_id} finished successfully.")
        else:
            logger.error(f"[FAILED] {model_id} failed with return code {result.returncode}.")
    except Exception as e:
        logger.error(f"[ERROR] Critical error running {model_id}: {e}")

def main():
    logger.info("=== STARTING KAGGLE SESSION 1 (TCA + LIGHT SER) ===")
    
    # Ensure outputs directory exists in WORKING DIR (not CODE_DIR which is read-only)
    Path("/kaggle/working/outputs").mkdir(exist_ok=True, parents=True)

    if not (CODE_DIR / "tca").exists():
        logger.error("!!! CRITICAL: 'tca' folder not found !!!")
        logger.error("Please add your code as a Dataset to the notebook.")
        return

    for script, mid in MODELS_TO_RUN:
        run_script(script, mid)

    # Run TCA Evaluation
    logger.info("[EVAL] Running TCA Final Evaluation...")
    run_script(CODE_DIR / "tca/evaluate_tca_all.py", "TCA_EVAL")

    logger.info("=== KAGGLE SESSION 1 COMPLETE ===")

if __name__ == "__main__":
    main()
