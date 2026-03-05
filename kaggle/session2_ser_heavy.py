"""
session2_ser_heavy.py — Kaggle Session 2 Orchestrator
=====================================================
Models: S2 (Whisper), S3 (W2V-BERT), S4 (W2V2-Large), S5 (WavLM), S7 (Robust)
"""

import subprocess
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Track execution
MODELS_TO_RUN = [
    ("ser/train_whisper_ser.py", "S2"),
    ("ser/train_wav2vec_bert.py", "S3"),
    ("ser/train_wav2vec2_large.py", "S4"),
    ("ser/train_wavlm_large.py", "S5"),
    ("ser/train_wav2vec2_robust.py", "S7"),
]

def run_script(script_path, model_id):
    logger.info(f"🚀 Starting Heavy {model_id} ({script_path})...")
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            logger.info(f"✅ {model_id} finished successfully.")
        else:
            logger.error(f"❌ {model_id} failed with return code {result.returncode}.")
    except Exception as e:
        logger.error(f"💥 Critical error running {model_id}: {e}")

def main():
    logger.info("=== STARTING KAGGLE SESSION 2 (HEAVY SER) ===")
    
    Path("./outputs").mkdir(exist_ok=True)

    for script, mid in MODELS_TO_RUN:
        run_script(script, mid)

    # Run SER Evaluation at the end of Session 2
    logger.info("📊 Running SER Final Evaluation...")
    run_script("ser/evaluate_ser_all.py", "SER_EVAL")

    logger.info("=== KAGGLE SESSION 2 COMPLETE ===")

if __name__ == "__main__":
    main()
