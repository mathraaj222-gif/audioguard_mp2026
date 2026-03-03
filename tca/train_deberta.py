"""
train_deberta.py — Fine-tune DeBERTa-v3-Large on NLI Ethics Dataset
====================================================================
Model  : microsoft/deberta-v3-large
Task   : 3-class NLI classification
           0 = entailment
           1 = neutral
           2 = contradiction
Data   : hate_speech_ethics_dataset_300.csv (Premise/Hypothesis/Label)
Output : ./outputs/deberta_nli_finetuned/

DeBERTa-v3-Large uses a disentangled attention mechanism and is pre-trained
on a large-scale NLI corpus, making it the strongest baseline for NLI tasks.
We apply:
  - Gradient checkpointing   (memory-efficient for large model)
  - Custom AdamW w/ linear warmup
  - Dynamic padding via DataCollatorWithPadding
  - FP16 on GPU

Usage (local):
    python train_deberta.py

Usage (Kaggle / inside train_on_kaggle.py):
    from tca.train_deberta import run_deberta_training
    run_deberta_training(output_dir="/kaggle/working/deberta_nli_finetuned")
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset_loader import load_nli_csv, NLI_LABEL_MAP, NLI_NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-large"

HPARAMS = {
    "max_length": 256,        # NLI needs longer context (premise + hypothesis)
    "batch_size": 8,          # DeBERTa-v3-L is large; small batch w/ gradient accum
    "gradient_accumulation_steps": 4,   # effective batch = 32
    "eval_batch_size": 16,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "fp16": torch.cuda.is_available(),
    "gradient_checkpointing": True,
}

ID2LABEL = NLI_LABEL_MAP          # {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ─────────────────────────────────────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_nli(examples, tokenizer):
    """
    Encode (premise, hypothesis) as a sentence-pair — the standard NLI input format.
    DeBERTa tokenizer handles SEP tokens automatically when two sequences are passed.
    """
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=HPARAMS["max_length"],
        padding=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_deberta_training(
    output_dir: str = "./outputs/deberta_nli_finetuned",
    nli_csv_path: str | None = None,
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DeBERTa-v3-Large Fine-tuning — NLI Ethics Dataset")
    logger.info("=" * 60)

    # 1. Load dataset
    dataset: DatasetDict = load_nli_csv(csv_path=nli_csv_path, seed=seed)

    # 2. Load tokenizer
    logger.info(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Tokenise
    logger.info("Tokenising NLI pairs (premise + hypothesis)...")
    tokenized = dataset.map(
        lambda ex: tokenize_nli(ex, tokenizer),
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
    tokenized.set_format("torch")

    # 4. Load model with DeBERTa-specific config
    logger.info(f"Loading model from: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NLI_NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Enable gradient checkpointing to save VRAM
    if HPARAMS["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=HPARAMS["num_epochs"],
        per_device_train_batch_size=HPARAMS["batch_size"],
        per_device_eval_batch_size=HPARAMS["eval_batch_size"],
        gradient_accumulation_steps=HPARAMS["gradient_accumulation_steps"],
        learning_rate=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
        warmup_ratio=HPARAMS["warmup_ratio"],
        fp16=HPARAMS["fp16"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2,
        push_to_hub=False,
        # DeBERTa v3 uses SentencePiece — disable fast tokenizer parallel warning
        dataloader_pin_memory=True,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7. Train
    logger.info("Starting DeBERTa-v3-Large training...")
    train_result = trainer.train()
    logger.info(f"Training complete. Metrics: {train_result.metrics}")

    # 8. Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized["test"])
    logger.info(f"Test results: {test_results}")

    # 9. Detailed classification report
    preds_output = trainer.predict(tokenized["test"])
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    report = classification_report(
        labels, preds,
        target_names=[NLI_LABEL_MAP[i] for i in range(NLI_NUM_LABELS)],
    )
    logger.info(f"\nClassification Report:\n{report}")

    # 10. Save model + metrics
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    all_metrics = {**train_result.metrics, **test_results, "classification_report": report}
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Model saved to:   {output_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info("✓ DeBERTa-v3-Large NLI training complete.")

    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_deberta_training()
