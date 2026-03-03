"""
train_whisper.py — Fine-tune Whisper-Large-v3 for Speech Emotion Recognition
=============================================================================
Model  : openai/whisper-large-v3
Task   : 7-class emotion classification (neutral/happy/sad/angry/fear/disgust/surprise)
Strategy:
  - Freeze the Whisper decoder entirely (we don't need ASR output)
  - Use the encoder's average-pooled hidden states as features
  - Add a 2-layer MLP classification head on top
  - Train end-to-end with the encoder unfrozen (lower LR for encoder)
  - Use SpecAugment for audio augmentation

Data   : RAVDESS + TESS + IEMOCAP (optional)  → via dataset_loader.py
Output : ./outputs/whisper_ser_finetuned/

Why Whisper for SER?
  Whisper-Large-v3's encoder captures rich acoustic features learned from
  680k hours of speech, making the encoder representations highly
  transferable to emotion recognition tasks.

Usage (local):
    python train_whisper.py

Usage (Kaggle):
    from ser.train_whisper import run_whisper_ser_training
    run_whisper_ser_training(output_dir="/kaggle/working/whisper_ser_finetuned")
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    WhisperProcessor,
    WhisperModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset_loader import load_ser_datasets, ID2EMOTION, NUM_EMOTIONS, TARGET_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/whisper-large-v3"

HPARAMS = {
    "max_audio_len_sec": 10,       # truncate audio to 10 seconds (Whisper limit = 30s)
    "batch_size": 4,               # Whisper-Large is huge; small batch
    "gradient_accumulation_steps": 8,   # effective batch = 32
    "eval_batch_size": 8,
    "encoder_lr": 5e-6,            # fine-tune encoder at very low LR
    "head_lr": 1e-4,               # custom head learns faster
    "weight_decay": 0.01,
    "num_epochs": 8,
    "warmup_ratio": 0.1,
    "fp16": torch.cuda.is_available(),
    "hidden_dim": 512,             # MLP hidden dim
    "dropout": 0.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM MODEL: Whisper Encoder + Classification Head
# ─────────────────────────────────────────────────────────────────────────────
class WhisperForEmotionClassification(nn.Module):
    """
    Wraps WhisperModel encoder with a pooler + MLP classification head.
    The decoder is completely discarded to reduce memory footprint.
    """

    def __init__(self, model_name: str, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        # Freeze the decoder — we don't need it
        for param in self.whisper.decoder.parameters():
            param.requires_grad = False

        encoder_dim = self.whisper.config.d_model  # 1280 for Whisper-Large

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, input_features, labels=None):
        # input_features: (B, 128, 3000) log-mel spectrogram
        encoder_out = self.whisper.encoder(input_features=input_features)
        hidden = encoder_out.last_hidden_state  # (B, T, D)

        # Average pooling over time
        pooled = hidden.mean(dim=1)             # (B, D)
        logits = self.classifier(pooled)        # (B, num_classes)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        # Return object compatible with Trainer
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────────────────────────────────────
class WhisperSERDataCollator:
    """
    Processes raw audio arrays → Whisper log-mel spectrograms.
    Uses WhisperProcessor which handles resampling + mel filterbank.
    """

    def __init__(self, processor: WhisperProcessor, max_len_sec: int = 10):
        self.processor = processor
        self.max_samples = max_len_sec * TARGET_SAMPLE_RATE

    def __call__(self, features: list[dict]) -> dict:
        audio_arrays = []
        labels = []

        for f in features:
            arr = np.array(f["audio"]["array"], dtype=np.float32)
            # Truncate / pad to max_samples
            if len(arr) > self.max_samples:
                arr = arr[: self.max_samples]
            audio_arrays.append(arr)
            labels.append(f["label"])

        inputs = self.processor(
            audio_arrays,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        return {
            "input_features": inputs.input_features,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


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
def run_whisper_ser_training(
    output_dir: str = "./outputs/whisper_ser_finetuned",
    ravdess_cache: str = "./datasets/ravdess",
    tess_cache: str = "./datasets/tess",
    iemocap_root: Optional[str] = None,
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Whisper-Large-v3 SER Fine-tuning")
    logger.info("=" * 60)

    # 1. Load datasets
    dataset: DatasetDict = load_ser_datasets(
        ravdess_cache=ravdess_cache,
        tess_cache=tess_cache,
        iemocap_root=iemocap_root,
        seed=seed,
    )

    # 2. Load processor
    logger.info(f"Loading Whisper processor from: {MODEL_NAME}")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    # 3. Build model
    logger.info(f"Building WhisperForEmotionClassification from: {MODEL_NAME}")
    model = WhisperForEmotionClassification(
        model_name=MODEL_NAME,
        num_classes=NUM_EMOTIONS,
        hidden_dim=HPARAMS["hidden_dim"],
        dropout=HPARAMS["dropout"],
    )
    model.to(DEVICE)

    # Gradient checkpointing
    model.whisper.gradient_checkpointing_enable()

    # 4. Data collator
    collator = WhisperSERDataCollator(processor=processor, max_len_sec=HPARAMS["max_audio_len_sec"])

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=HPARAMS["num_epochs"],
        per_device_train_batch_size=HPARAMS["batch_size"],
        per_device_eval_batch_size=HPARAMS["eval_batch_size"],
        gradient_accumulation_steps=HPARAMS["gradient_accumulation_steps"],
        weight_decay=HPARAMS["weight_decay"],
        warmup_ratio=HPARAMS["warmup_ratio"],
        fp16=HPARAMS["fp16"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,   # Critical — avoid stripping 'audio' column
        push_to_hub=False,
        dataloader_num_workers=2,
    )

    # 6. Trainer with differential LR (optimizer override)
    optimizer_grouped_parameters = [
        {"params": model.whisper.encoder.parameters(), "lr": HPARAMS["encoder_lr"]},
        {"params": model.classifier.parameters(), "lr": HPARAMS["head_lr"]},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=HPARAMS["weight_decay"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),   # let Trainer build the scheduler
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Train
    logger.info("Starting Whisper-Large-v3 SER training...")
    train_result = trainer.train()
    logger.info(f"Training complete: {train_result.metrics}")

    # 8. Test evaluation
    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    logger.info(f"Test results: {test_results}")

    # 9. Save model + processor
    torch.save(model.state_dict(), str(output_path / "model_weights.pt"))
    processor.save_pretrained(str(output_path))

    # Save label map
    with open(output_path / "label_map.json", "w") as f:
        json.dump({"id2label": ID2EMOTION, "num_labels": NUM_EMOTIONS}, f, indent=2)

    # Save metrics
    all_metrics = {**train_result.metrics, **test_results}
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Model saved to: {output_path}")
    logger.info("✓ Whisper-Large-v3 SER training complete.")

    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_whisper_ser_training()
