"""
evaluate_tca_all.py — Unified TCA Evaluation and Leaderboard Generation
=======================================================================
1. Loads T1-T6 models from ./outputs/
2. Runs inference on corresponding test sets
3. Produces a leaderboard CSV and formatted table
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from dataset_loader import load_davidson, load_nli_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "T1": {"name": "google-bert/bert-base-uncased", "dataset": "nli"},
    "T2": {"name": "GroNLP/hateBERT", "dataset": "davidson_multi"},
    "T3": {"name": "microsoft/deberta-v3-large", "dataset": "nli"},
    "T4": {"name": "facebook/roberta-hate-speech-dynabench-r4-target", "dataset": "davidson_binary"},
    "T5": {"name": "cross-encoder/nli-deberta-v3-small", "dataset": "nli"},
    "T6": {"name": "cardiffnlp/twitter-roberta-base-hate-latest", "dataset": "davidson_binary"},
}

def run_evaluation():
    outputs_dir = Path("./outputs")
    leaderboard = []

    # Load datasets once
    datasets = {
        "nli": load_nli_csv()["test"],
        "davidson_multi": load_davidson(binary=False)["test"],
        "davidson_binary": load_davidson(binary=True)["test"],
    }

    for model_id, config in MODEL_CONFIGS.items():
        model_path = outputs_dir / f"{model_id}_{config['name'].split('/')[-1]}"
        if not model_path.exists():
            # Try alternative name if rename happened
            model_path = list(outputs_dir.glob(f"{model_id}_*"))
            if not model_path:
                logger.warning(f"Model {model_id} not found in {outputs_dir}. Skipping.")
                continue
            model_path = model_path[0]

        logger.info(f"Evaluating {model_id} from {model_path}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(DEVICE)
            model.eval()
            
            test_data = datasets[config["dataset"]]
            preds, labels = [], []
            
            batch_size = 16
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i : i + batch_size]
                if config["dataset"] == "nli":
                    inputs = tokenizer(batch["premise"], batch["hypothesis"], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                else:
                    inputs = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                    batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    preds.extend(batch_preds)
                    labels.extend(batch["label"])
            
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            
            leaderboard.append({
                "model_id": model_id,
                "model_name": config["name"],
                "track": "TCA",
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "precision_macro": round(precision, 4),
                "recall_macro": round(recall, 4),
                "dataset": config["dataset"]
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_id}: {e}")

    if not leaderboard:
        logger.error("No models were successfully evaluated.")
        return

    df = pd.DataFrame(leaderboard)
    df = df.sort_values(by="f1_macro", ascending=False)
    
    # Save CSV
    csv_path = outputs_dir / "tca_leaderboard.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Leaderboard saved to {csv_path}")

    # Formatted table
    print("\n" + "="*80)
    print(f"{'TCA LEADERBOARD':^80}")
    print("="*80)
    print(df[["model_id", "model_name", "accuracy", "f1_macro", "dataset"]].to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_evaluation()
