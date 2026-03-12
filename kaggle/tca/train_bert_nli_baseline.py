"""
train_bert_nli_baseline.py — T1: BERT-Base-Uncased on NLI Ethics Dataset
=======================================================================
Model : google-bert/bert-base-uncased
Task  : 3-class NLI (entailment, contradiction, neutral)
Data  : hate_speech_ethics_dataset_300.csv
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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from dataset_loader import load_nli_csv, NLI_LABEL_MAP, NLI_NUM_LABELS

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model_id": "T1",
    "model_name": "google-bert/bert-base-uncased",
    "track": "TCA",
    "epochs": 3,
    "lr": 2e-5,
    "batch_size": 16,
    "weight_decay": 0.01,
    "max_length": 128,
    "output_dir": "./outputs/T1_bert_nli_baseline/",
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

    logger.info(f"Starting {CONFIG['model_id']} training on {DEVICE}...")
    
    # 1. Load Data
    dataset = load_nli_csv()
    
    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=False,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], 
        num_labels=NLI_NUM_LABELS
    )
    model.to(DEVICE)

    # 4. Training Args
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["lr"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=CONFIG["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_path / "logs"),
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. Train
    start_time = time.time()
    trainer.train()
    train_time = (time.time() - start_time) / 60
    
    # 7. Evaluate
    logger.info("Evaluating on val set...")
    metrics = trainer.evaluate()
    
    # Peak VRAM
    peak_vram = 0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Peak VRAM used: {peak_vram:.2f} GB")

    # 8. Save
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Unified results format
    summary = {
        "model_id": CONFIG["model_id"],
        "model_name": CONFIG["model_name"],
        "track": CONFIG["track"],
        "accuracy": round(metrics["eval_accuracy"], 4),
        "f1_macro": round(metrics["eval_f1_macro"], 4),
        "precision_macro": round(metrics["eval_precision_macro"], 4),
        "recall_macro": round(metrics["eval_recall_macro"], 4),
        "train_time_minutes": round(train_time, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "epochs_trained": CONFIG["epochs"],
        "dataset": "NLI CSV 300 rows",
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

    logger.info(f"✓ {CONFIG['model_id']} training complete.")

if __name__ == "__main__":
    run_training()
