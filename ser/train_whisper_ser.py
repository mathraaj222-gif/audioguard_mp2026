"""
train_whisper_ser.py — S2: Whisper-Large-v3 SER (Encoder Only)
=============================================================
Model : openai/whisper-large-v3 (Encoder Only)
Task  : 7-class emotion classification
Data  : RAVDESS + TESS
Logic : Attention pooling + specific freezing logic
"""

import os
import json
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import Audio
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
    "model_id": "S2",
    "model_name": "openai/whisper-large-v3",
    "track": "SER",
    "epochs": 5,
    "lr_head": 1e-4,
    "lr_encoder": 2e-5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "output_dir": "./outputs/S2_whisper_ser/",
}

class WhisperSERModel(nn.Module):
    def __init__(self, model_name, num_labels=7):
        super().__init__()
        # Load full model but we'll only use encoder
        self.whisper = WhisperModel.from_pretrained(model_name)
        encoder_dim = self.whisper.config.d_model # 1280
        
        # Head: Linear(1280, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 7)
        self.attention_pooling = nn.Linear(encoder_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        
        # Discard decoder entirely
        del self.whisper.decoder

    def forward(self, input_features, labels=None):
        encoder_outputs = self.whisper.encoder(input_features)
        last_hidden_state = encoder_outputs.last_hidden_state # (B, T, D)
        
        # Attention Pooling
        weights = torch.softmax(self.attention_pooling(last_hidden_state), dim=1)
        pooled = torch.sum(weights * last_hidden_state, dim=1)
        
        logits = self.head(pooled)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": round(float(acc), 2),
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }

class WhisperFreezeCallback(TrainerCallback):
    """Epoch-level callback: freeze encoder for epochs 0-1, unfreeze last 6 layers from epoch 2+."""
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        if current_epoch < 2:
            # Freeze all encoder layers
            for param in model.whisper.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Epoch {current_epoch}: Encoder FROZEN (head only training)")
        else:
            # Freeze all, then unfreeze last 6 layers
            for param in model.whisper.encoder.parameters():
                param.requires_grad = False
            for layer in model.whisper.encoder.layers[-6:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(f"Epoch {current_epoch}: Last 6 encoder layers UNFROZEN")

def run_training():
    output_path = Path(CONFIG["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # VRAM Guard
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
        if vram < 10e9:
            logger.warning(f"⚠️ LOW VRAM WARNING: {vram/(1024**3):.2f} GB detected. Whisper-Large may OOM.")

    logger.info(f"Starting {CONFIG['model_id']} training...")
    
    # 1. Data
    ds = load_ser_datasets()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(CONFIG["model_name"])

    def preprocess_function(examples):
        # Whisper expects 30s mel spectrogram
        audio = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
        examples["input_features"] = inputs.input_features
        return examples

    tokenized_ds = ds.map(preprocess_function, batched=True, batch_size=10, remove_columns=["audio", "source"])

    # 2. Model
    model = WhisperSERModel(CONFIG["model_name"], num_labels=7)
    model.to(DEVICE)

    # 3. Training Args
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": CONFIG["lr_head"]},
        {"params": model.attention_pooling.parameters(), "lr": CONFIG["lr_head"]},
        {"params": model.whisper.encoder.parameters(), "lr": CONFIG["lr_encoder"]},
    ])

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), WhisperFreezeCallback()],
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
    
    # Unified summary — evaluate ONCE, extract all metrics
    test_metrics = trainer.evaluate(tokenized_ds["test"])
    summary = {
        "model_id": CONFIG["model_id"],
        "model_name": CONFIG["model_name"],
        "track": CONFIG["track"],
        "accuracy": round(test_metrics["eval_accuracy"], 4),
        "f1_macro": round(test_metrics["eval_f1_macro"], 4),
        "precision_macro": round(test_metrics["eval_precision_macro"], 4),
        "recall_macro": round(test_metrics["eval_recall_macro"], 4),
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
