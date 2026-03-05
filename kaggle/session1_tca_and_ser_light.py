"""
session1_tca_and_ser_light.py — Kaggle Session 1 Orchestrator
=============================================================
Execution Order: T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> S1 -> S6
"""

import subprocess
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Track execution
MODELS_TO_RUN = [
    ("tca/train_bert_nli_baseline.py", "T1"),
    ("tca/train_hatebert.py", "T2"),
    ("tca/train_deberta_large.py", "T3"),
    ("tca/train_roberta_dynabench.py", "T4"),
    ("tca/train_deberta_small_ce.py", "T5"),
    ("tca/train_twitter_roberta.py", "T6"),
    ("ser/train_lstm_baseline.py", "S1"),
    ("ser/train_hubert_er.py", "S6"),
]

def run_script(script_path, model_id):
    logger.info(f"🚀 Starting {model_id} ({script_path})...")
    try:
        # We assume running from project root
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            logger.info(f"✅ {model_id} finished successfully.")
        else:
            logger.error(f"❌ {model_id} failed with return code {result.returncode}.")
    except Exception as e:
        logger.error(f"💥 Critical error running {model_id}: {e}")

def main():
    logger.info("=== STARTING KAGGLE SESSION 1 (TCA + LIGHT SER) ===")
    
    # Ensure outputs directory exists
    Path("./outputs").mkdir(exist_ok=True)

    for script, mid in MODELS_TO_RUN:
        run_script(script, mid)

    # Run TCA Evaluation at the end of Session 1
    logger.info("📊 Running TCA Final Evaluation...")
    run_script("tca/evaluate_tca_all.py", "TCA_EVAL")

    logger.info("=== KAGGLE SESSION 1 COMPLETE ===")

if __name__ == "__main__":
    main()
