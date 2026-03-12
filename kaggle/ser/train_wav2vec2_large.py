"""
train_wav2vec2_large.py — S4: Wav2Vec2-Large-960h SER
=====================================================
Model : facebook/wav2vec2-large-960h
Task  : 7-class emotion classification
Data  : RAVDESS + TESS
"""

import os
import json
import logging
import time
import random
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from dataset_loader import load_ser_datasets, NUM_EMOTIONS, TARGET_SAMPLE_RATE

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model_id": "S4",
    "model_name": "facebook/wav2vec2-large-960h",
    "track": "SER",
    "epochs": 5,
    "lr": 2e-5,
    "batch_size": 16,
    "warmup_steps": 500,
    "output_dir": "./outputs/S4_wav2vec2_large/",
}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }

def run_training():
    output_path = Path(CONFIG["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting {CONFIG['model_id']} training...")
    
    # 1. Data
    ds = load_ser_datasets()
    feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG["model_name"])

    def preprocess_function(examples):
        audio = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True, max_length=TARGET_SAMPLE_RATE * 10)
        return inputs

    tokenized_ds = ds.map(preprocess_function, batched=True, batch_size=8, remove_columns=["audio", "source"])

    # 2. Model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        CONFIG["model_name"], 
        num_labels=7,
        ignore_mismatched_sizes=True
    )
    # Freeze CNN feature extractor
    model.freeze_feature_extractor()
    model.to(DEVICE)

    # 3. Training Args
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        warmup_steps=CONFIG["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["lr"],
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Train
    start_time = time.time()
    trainer.train()
    train_time = (time.time() - start_time) / 60
    
    # Peak VRAM
    peak_vram = 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    # 6. Save
    trainer.save_model(str(output_path))
    feature_extractor.save_pretrained(str(output_path))
    
    # Unified summary
    summary = {
        "model_id": CONFIG["model_id"],
        "model_name": CONFIG["model_name"],
        "track": CONFIG["track"],
        "accuracy": round(trainer.evaluate(tokenized_ds["test"])["eval_accuracy"], 4),
        "f1_macro": round(trainer.evaluate(tokenized_ds["test"])["eval_f1_macro"], 4),
        "precision_macro": round(trainer.evaluate(tokenized_ds["test"])["eval_precision_macro"], 4),
        "recall_macro": round(trainer.evaluate(tokenized_ds["test"])["eval_recall_macro"], 4),
        "train_time_minutes": round(train_time, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "epochs_trained": CONFIG["epochs"],
        "dataset": "RAVDESS + TESS",
        "saved_model_path": str(output_path)
    }
    
    summary_file = Path("./outputs/training_summary.json")
    all_summaries = []
    if summary_file.exists():
        with open(summary_file, "r") as f:
            all_summaries = json.load(f)
            if not isinstance(all_summaries, list): all_summaries = [all_summaries]
    
    all_summaries.append(summary)
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)

if __name__ == "__main__":
    run_training()
