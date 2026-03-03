"""
evaluate_tca.py — Unified TCA Model Evaluation
===============================================
Run inference on the saved HateBERT and DeBERTa-v3-Large models
and produce a comprehensive evaluation report.

Usage:
    python evaluate_tca.py
    python evaluate_tca.py --hatebert-dir ./outputs/hatebert_finetuned
                           --deberta-dir  ./outputs/deberta_nli_finetuned
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset_loader import load_davidson, load_nli_csv, DAVIDSON_LABEL_MAP, NLI_LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = 0 if torch.cuda.is_available() else -1


# ─────────────────────────────────────────────────────────────────────────────
# SHARED EVALUATION UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_dir: str, dataset_split, label_map: dict, task_name: str) -> dict:
    model_path = Path(model_dir)
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating model: {model_path.name}  [{task_name}]")
    logger.info(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_labels = [], []

    # Batch inference
    batch_size = 32
    texts, labels, pairs = [], [], []

    for ex in dataset_split:
        all_labels.append(ex["label"])
        if "text" in ex:
            texts.append(ex["text"])
        else:
            pairs.append((ex["premise"], ex["hypothesis"]))

    # Run inference
    with torch.no_grad():
        for i in range(0, len(all_labels), batch_size):
            if texts:
                batch = texts[i : i + batch_size]
                enc = tokenizer(batch, truncation=True, padding=True, max_length=128, return_tensors="pt")
            else:
                batch_pairs = pairs[i : i + batch_size]
                enc = tokenizer(
                    [p[0] for p in batch_pairs],
                    [p[1] for p in batch_pairs],
                    truncation=True, padding=True, max_length=256, return_tensors="pt",
                )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    num_labels = len(label_map)
    target_names = [label_map[i] for i in range(num_labels)]

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=target_names)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    result = {
        "model": str(model_path),
        "task": task_name,
        "test_samples": len(all_labels),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"F1 (macro):  {f1_macro:.4f}")
    logger.info(f"F1 (wgt):    {f1_weighted:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate TCA models")
    parser.add_argument("--hatebert-dir", default="./outputs/hatebert_finetuned")
    parser.add_argument("--deberta-dir", default="./outputs/deberta_nli_finetuned")
    parser.add_argument("--output-report", default="./outputs/tca_evaluation_report.json")
    args = parser.parse_args()

    results = {}

    # 1. Evaluate HateBERT
    davidson = load_davidson()
    hatebert_result = evaluate_model(
        model_dir=args.hatebert_dir,
        dataset_split=davidson["test"],
        label_map=DAVIDSON_LABEL_MAP,
        task_name="Hate Speech Classification (Davidson)",
    )
    results["hatebert"] = hatebert_result

    # 2. Evaluate DeBERTa-v3-Large
    nli = load_nli_csv()
    deberta_result = evaluate_model(
        model_dir=args.deberta_dir,
        dataset_split=nli["test"],
        label_map=NLI_LABEL_MAP,
        task_name="NLI Classification (Ethics CSV)",
    )
    results["deberta"] = deberta_result

    # Save combined report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Evaluation complete. Report saved to: {report_path}")
    logger.info("\n=== SUMMARY ===")
    logger.info(f"HateBERT  → Accuracy: {results['hatebert']['accuracy']:.4f}  F1-macro: {results['hatebert']['f1_macro']:.4f}")
    logger.info(f"DeBERTa   → Accuracy: {results['deberta']['accuracy']:.4f}  F1-macro: {results['deberta']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
