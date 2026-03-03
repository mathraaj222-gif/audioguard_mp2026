"""
train_wav2vec_bert.py — Fine-tune Wav2Vec-BERT 2.0 for Speech Emotion Recognition
==================================================================================
Model  : facebook/w2v-bert-2.0
Task   : 7-class emotion classification (neutral/happy/sad/angry/fear/disgust/surprise)
Strategy:
  - Load Wav2Vec-BERT 2.0 (joint acoustic & semantic pre-training, 600M params)
  - Add weighted average of all hidden layer states (learnable)
  - Pool → MLP classification head
  - SpecAugment mask applied at training time
  - Differential learning rates: feature extractor frozen initially,
    transformer layers at very low LR, head at higher LR

Data   : RAVDESS + TESS + IEMOCAP (optional)  → via dataset_loader.py
Output : ./outputs/wav2vec_bert_ser_finetuned/

Wav2Vec-BERT 2.0 advantages for SER:
  - Jointly trained with a BERT-based masked LM objective and acoustic CTC
  - Superior to older Wav2Vec 2.0 on emotion tasks (MMS backbone)
  - Native 16 kHz input — no mel conversion needed

Usage (local):
    python train_wav2vec_bert.py

Usage (Kaggle):
    from ser.train_wav2vec_bert import run_wav2vec_bert_ser_training
    run_wav2vec_bert_ser_training(output_dir="/kaggle/working/wav2vec_bert_ser_finetuned")
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score

from dataset_loader import load_ser_datasets, ID2EMOTION, NUM_EMOTIONS, TARGET_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/w2v-bert-2.0"

HPARAMS = {
    "max_audio_len_sec": 10,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,   # effective batch = 32
    "eval_batch_size": 16,
    "feature_extractor_lr": 0.0,        # keep CNN feature extractor frozen
    "transformer_lr": 2e-5,
    "head_lr": 1e-4,
    "weight_decay": 0.01,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "fp16": torch.cuda.is_available(),
    "hidden_dim": 256,
    "dropout": 0.25,
    "mask_time_prob": 0.05,             # SpecAugment: time masking
    "mask_feature_prob": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM MODEL: Wav2Vec-BERT 2.0 + Weighted Layer averaging + MLP Head
# ─────────────────────────────────────────────────────────────────────────────
class Wav2VecBertForEmotionClassification(nn.Module):
    """
    Wav2Vec-BERT 2.0 with learnable weighted average of hidden states
    (inspired by the "scalar mix" technique from BERT probing literature).
    """

    def __init__(self, model_name: str, num_classes: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.w2v_bert = Wav2Vec2BertModel.from_pretrained(
            model_name,
            output_hidden_states=True,
            mask_time_prob=HPARAMS["mask_time_prob"],
            mask_feature_prob=HPARAMS["mask_feature_prob"],
        )

        # Freeze CNN feature extractor
        self.w2v_bert.feature_extractor._freeze_parameters()

        num_layers = self.w2v_bert.config.num_hidden_layers + 1   # +1 for embedding layer
        encoder_dim = self.w2v_bert.config.hidden_size             # 1024

        # Learnable scalar weights for layer averaging
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # Attention pooling over time
        self.attention_pool = nn.Linear(encoder_dim, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, input_values, attention_mask=None, labels=None):
        # Wav2Vec-BERT outputs tuple of hidden states from all layers
        outputs = self.w2v_bert(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # list of (B, T, D)

        # Stack and apply softmax-normalised learned layer weights
        stacked = torch.stack(hidden_states, dim=0)       # (num_layers, B, T, D)
        weights = torch.softmax(self.layer_weights, dim=0)  # (num_layers,)
        weighted = (stacked * weights[:, None, None, None]).sum(dim=0)  # (B, T, D)

        # Attention pooling over time dimension
        attn_scores = self.attention_pool(weighted).squeeze(-1)   # (B, T)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (weighted * attn_weights).sum(dim=1)                   # (B, D)

        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)


# ─────────────────────────────────────────────────────────────────────────────
# DATA COLLATOR
# ─────────────────────────────────────────────────────────────────────────────
class Wav2VecBertDataCollator:
    """
    Processes raw audio arrays → padded input_values with attention_mask.
    """

    def __init__(self, feature_extractor, max_len_sec: int = 10):
        self.feature_extractor = feature_extractor
        self.max_samples = max_len_sec * TARGET_SAMPLE_RATE

    def __call__(self, features: list[dict]) -> dict:
        audio_arrays = []
        labels = []

        for f in features:
            arr = np.array(f["audio"]["array"], dtype=np.float32)
            if len(arr) > self.max_samples:
                arr = arr[: self.max_samples]
            audio_arrays.append(arr)
            labels.append(f["label"])

        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_wav2vec_bert_ser_training(
    output_dir: str = "./outputs/wav2vec_bert_ser_finetuned",
    ravdess_cache: str = "./datasets/ravdess",
    tess_cache: str = "./datasets/tess",
    iemocap_root: Optional[str] = None,
    seed: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Wav2Vec-BERT 2.0 SER Fine-tuning")
    logger.info("=" * 60)

    # 1. Load datasets
    dataset: DatasetDict = load_ser_datasets(
        ravdess_cache=ravdess_cache,
        tess_cache=tess_cache,
        iemocap_root=iemocap_root,
        seed=seed,
    )

    # 2. Load feature extractor
    logger.info(f"Loading feature extractor from: {MODEL_NAME}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

    # 3. Build model
    logger.info(f"Building Wav2VecBertForEmotionClassification from: {MODEL_NAME}")
    model = Wav2VecBertForEmotionClassification(
        model_name=MODEL_NAME,
        num_classes=NUM_EMOTIONS,
        hidden_dim=HPARAMS["hidden_dim"],
        dropout=HPARAMS["dropout"],
    )
    model.to(DEVICE)

    # 4. Data collator
    collator = Wav2VecBertDataCollator(feature_extractor=feature_extractor,
                                        max_len_sec=HPARAMS["max_audio_len_sec"])

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
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_num_workers=2,
    )

    # 6. Differential LR optimizer
    optimizer_grouped_parameters = [
        {"params": model.w2v_bert.encoder.parameters(), "lr": HPARAMS["transformer_lr"]},
        {"params": model.layer_weights, "lr": HPARAMS["transformer_lr"]},
        {"params": model.attention_pool.parameters(), "lr": HPARAMS["head_lr"]},
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
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 7. Train
    logger.info("Starting Wav2Vec-BERT 2.0 SER training...")
    train_result = trainer.train()
    logger.info(f"Training complete: {train_result.metrics}")

    # 8. Test evaluation
    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    logger.info(f"Test results: {test_results}")

    # 9. Save
    torch.save(model.state_dict(), str(output_path / "model_weights.pt"))
    feature_extractor.save_pretrained(str(output_path))

    with open(output_path / "label_map.json", "w") as f:
        json.dump({"id2label": ID2EMOTION, "num_labels": NUM_EMOTIONS}, f, indent=2)

    all_metrics = {**train_result.metrics, **test_results}
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Model saved to: {output_path}")
    logger.info("✓ Wav2Vec-BERT 2.0 SER training complete.")

    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_wav2vec_bert_ser_training()
