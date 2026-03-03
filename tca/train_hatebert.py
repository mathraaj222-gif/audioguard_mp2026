"""
train_hatebert.py — Fine-tune HateBERT on Davidson et al. Hate Speech Dataset
==============================================================================
Model  : GroNLP/hateBERT  (BERT pre-trained on abusive Reddit content)
Task   : 3-class text classification
           0 = hate speech
           1 = offensive language
           2 = neither
Data   : Davidson et al. (2017) via HuggingFace datasets hub
Output : ./outputs/hatebert_finetuned/   (model, tokenizer, metrics)

Usage (local):
    python train_hatebert.py

Usage (Kaggle / inside train_on_kaggle.py):
    from tca.train_hatebert import run_hatebert_training
    run_hatebert_training(output_dir="/kaggle/working/hatebert_finetuned")
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

from dataset_loader import load_davidson, DAVIDSON_LABEL_MAP, DAVIDSON_NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "GroNLP/hateBERT"

HPARAMS = {
    "max_length": 128,
    "batch_size": 32,
    "eval_batch_size": 64,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "fp16": torch.cuda.is_available(),
}

ID2LABEL = DAVIDSON_LABEL_MAP          # {0: "hate_speech", 1: "offensive_language", 2: "neither"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ─────────────────────────────────────────────────────────────────────────────
# TOKENISATION
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_davidson(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=HPARAMS["max_length"],
        padding=False,        # dynamic padding via DataCollatorWithPadding
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
def run_hatebert_training(output_dir: str = "./outputs/hatebert_finetuned", seed: int = 42):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("HateBERT Fine-tuning — Davidson Hate Speech Dataset")
    logger.info("=" * 60)

    # 1. Load dataset
    dataset: DatasetDict = load_davidson(seed=seed)

    # 2. Load tokenizer
    logger.info(f"Loading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Tokenise
    logger.info("Tokenising dataset...")
    tokenized = dataset.map(
        lambda ex: tokenize_davidson(ex, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format("torch")

    # 4. Load model
    logger.info(f"Loading model from: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=DAVIDSON_NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=HPARAMS["num_epochs"],
        per_device_train_batch_size=HPARAMS["batch_size"],
        per_device_eval_batch_size=HPARAMS["eval_batch_size"],
        learning_rate=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
        warmup_ratio=HPARAMS["warmup_ratio"],
        fp16=HPARAMS["fp16"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=2,
        push_to_hub=False,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train
    logger.info("Starting HateBERT training...")
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
        target_names=[DAVIDSON_LABEL_MAP[i] for i in range(DAVIDSON_NUM_LABELS)],
    )
    logger.info(f"\nClassification Report:\n{report}")

    # 10. Save model + metrics
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save metrics to JSON
    all_metrics = {**train_result.metrics, **test_results, "classification_report": report}
    metrics_path = output_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Model saved to: {output_path}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info("✓ HateBERT training complete.")

    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_hatebert_training()
