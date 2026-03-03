"""
evaluate_ser.py — Unified SER Model Evaluation
===============================================
Evaluate both Whisper-Large-v3 and Wav2Vec-BERT 2.0 SER models
on the held-out test split and produce a comprehensive report.

Usage:
    python evaluate_ser.py
    python evaluate_ser.py --whisper-dir ./outputs/whisper_ser_finetuned
                           --wav2vec-dir  ./outputs/wav2vec_bert_ser_finetuned
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataset_loader import load_ser_datasets, ID2EMOTION, NUM_EMOTIONS, TARGET_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE WHISPER SER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_whisper(whisper_dir: str, test_dataset) -> dict:
    from transformers import WhisperProcessor
    from train_whisper import WhisperForEmotionClassification, WhisperSERDataCollator

    model_path = Path(whisper_dir)
    logger.info(f"\n{'='*60}\nEvaluating Whisper-Large-v3 SER from {model_path}\n{'='*60}")

    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForEmotionClassification(
        model_name="openai/whisper-large-v3",
        num_classes=NUM_EMOTIONS,
    )
    model.load_state_dict(torch.load(model_path / "model_weights.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    collator = WhisperSERDataCollator(processor=processor)
    return _run_inference(model, test_dataset, collator, "Whisper-Large-v3 SER")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE WAV2VEC-BERT SER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_wav2vec_bert(wav2vec_dir: str, test_dataset) -> dict:
    from transformers import AutoFeatureExtractor
    from train_wav2vec_bert import Wav2VecBertForEmotionClassification, Wav2VecBertDataCollator

    model_path = Path(wav2vec_dir)
    logger.info(f"\n{'='*60}\nEvaluating Wav2Vec-BERT 2.0 SER from {model_path}\n{'='*60}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
    model = Wav2VecBertForEmotionClassification(
        model_name="facebook/w2v-bert-2.0",
        num_classes=NUM_EMOTIONS,
    )
    model.load_state_dict(torch.load(model_path / "model_weights.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    collator = Wav2VecBertDataCollator(feature_extractor=feature_extractor)
    return _run_inference(model, test_dataset, collator, "Wav2Vec-BERT 2.0 SER")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED INFERENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def _run_inference(model, dataset, collator, model_name: str) -> dict:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, collate_fn=collator)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    target_names = [ID2EMOTION[i] for i in range(NUM_EMOTIONS)]

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=target_names)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    logger.info(f"Accuracy:   {acc:.4f}")
    logger.info(f"F1 (macro): {f1_macro:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_score(all_labels, all_preds, average="weighted"), 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper-dir",  default="./outputs/whisper_ser_finetuned")
    parser.add_argument("--wav2vec-dir",  default="./outputs/wav2vec_bert_ser_finetuned")
    parser.add_argument("--output-report", default="./outputs/ser_evaluation_report.json")
    args = parser.parse_args()

    logger.info("Loading SER test dataset...")
    dataset = load_ser_datasets()
    test_set = dataset["test"]

    results = {}

    if Path(args.whisper_dir).exists():
        results["whisper"] = evaluate_whisper(args.whisper_dir, test_set)
    else:
        logger.warning(f"Whisper model not found at: {args.whisper_dir}")

    if Path(args.wav2vec_dir).exists():
        results["wav2vec_bert"] = evaluate_wav2vec_bert(args.wav2vec_dir, test_set)
    else:
        logger.warning(f"Wav2Vec-BERT model not found at: {args.wav2vec_dir}")

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ SER evaluation complete. Report: {report_path}")
    if "whisper" in results:
        logger.info(f"Whisper-v3   → Acc: {results['whisper']['accuracy']:.4f}  F1: {results['whisper']['f1_macro']:.4f}")
    if "wav2vec_bert" in results:
        logger.info(f"Wav2Vec-BERT → Acc: {results['wav2vec_bert']['accuracy']:.4f}  F1: {results['wav2vec_bert']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
