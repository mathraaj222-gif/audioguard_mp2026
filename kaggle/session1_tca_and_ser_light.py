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
    cwd = Path.cwd()
    logger.info(f"__file__: {__file__}")
    logger.info(f"Searching for 'tca' folder. Starting at CWD: {cwd}")
    
    if (cwd / "tca").exists():
        logger.info(f"FOUND 'tca' in CWD: {cwd}")
        return cwd
    
    # Check /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        logger.warning("/kaggle/input does not exist. Only searching CWD.")
        return cwd

    # 1. Try common explicit paths (fast) - PRIORITIZE USER REPORTED PATH
    possible_paths = [
        Path("/kaggle/input/datasets/mathanraaj/audioguars-mp2026"), # Exact user path
        kaggle_input / "audioguars-mp2026",
        kaggle_input / "audioguardmp-2026",
        kaggle_input / "audioguard-mp2026",
        kaggle_input / "mathanraaj/audioguars-mp2026",
    ]
    
    for p in possible_paths:
        logger.info(f"Checking potential path: {p}")
        if p.exists() and (p / "tca").exists():
            logger.info(f"FOUND 'tca' in: {p}")
            return p

    # 2. Shallow search (medium)
    logger.info("Shallow searching /kaggle/input subdirectories...")
    for d in kaggle_input.iterdir():
        if d.is_dir() and (d / "tca").exists():
            logger.info(f"FOUND 'tca' in shallow search: {d}")
            return d

    # 3. Deep search (slow - fallback)
    logger.info("Deep searching /kaggle/input (this may take a moment)...")
    for p in kaggle_input.rglob("tca"):
        if p.is_dir():
            logger.info(f"FOUND 'tca' via rglob: {p.parent}")
            return p.parent
                
    logger.error("Could not find 'tca' folder anywhere in /kaggle/working or /kaggle/input")
    
    # --- ULTRA DIAGNOSTICS ---
    logger.info("=== DIAGNOSTIC FILE DUMP ===")
    try:
        logger.info(f"Contents of /kaggle/working: {os.listdir('/kaggle/working')}")
        if Path('/kaggle/input').exists():
            logger.info(f"Contents of /kaggle/input: {os.listdir('/kaggle/input')}")
            for d in Path('/kaggle/input').iterdir():
                if d.is_dir():
                    logger.info(f"Contents of {d}: {os.listdir(d)}")
    except Exception as e:
        logger.error(f"Diagnostic dump failed: {e}")
    
    return cwd # Default failure

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
        return

    logger.info(f"[START] {model_id} ({script_path})...")
    
    # 1. Prepare environment with PYTHONPATH
    env = os.environ.copy()
    tca_path = str(CODE_DIR / "tca")
    ser_path = str(CODE_DIR / "ser")
    
    existing_pythonpath = env.get("PYTHONPATH", "")
    new_pythonpath = f"{tca_path}{os.pathsep}{ser_path}"
    if existing_pythonpath:
        new_pythonpath = f"{new_pythonpath}{os.pathsep}{existing_pythonpath}"
    
    env["PYTHONPATH"] = new_pythonpath
    
    try:
        # 2. Run from Writable Working Dir
        # This ensures any relative outputs (like ./outputs/) land in /kaggle/working
        result = subprocess.run(
            [sys.executable, str(script_path)], 
            check=True, 
            cwd="/kaggle/working",
            env=env
        )
        if result.returncode == 0:
            logger.info(f"[OK] {model_id} finished successfully.")
        else:
            logger.error(f"[FAILED] {model_id} failed with return code {result.returncode}.")
    except Exception as e:
        logger.error(f"[ERROR] Critical error running {model_id}: {e}")

def main():
    logger.info("=== STARTING KAGGLE SESSION 1 (TCA + LIGHT SER) ===")
    
    # Ensure outputs directory exists in WORKING DIR
    Path("/kaggle/working/outputs").mkdir(exist_ok=True, parents=True)

    for script, mid in MODELS_TO_RUN:
        run_script(script, mid)

    # Run TCA Evaluation
    logger.info("[EVAL] Running TCA Final Evaluation...")
    run_script(CODE_DIR / "tca/evaluate_tca_all.py", "TCA_EVAL")

    logger.info("=== KAGGLE SESSION 1 COMPLETE ===")

if __name__ == "__main__":
    main()
