"""
train_lstm_baseline.py — S1: Custom LSTM on MFCCs (TESS Only)
=============================================================
Model : LSTM (TensorFlow/Keras)
Data  : TESS only
Labels: 7 emotions
Features: MFCC (40, 216)
"""

import os
import json
import logging
import time
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
from sklearn.model_selection import train_test_split
from dataset_loader import load_tess, TARGET_SAMPLE_RATE, ID2EMOTION

# Seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "model_id": "S1",
    "model_name": "LSTM-Baseline",
    "track": "SER",
    "epochs": 100,
    "batch_size": 32,
    "n_mfcc": 40,           # base MFCCs
    "n_features": 120,       # 40 MFCC + 40 delta + 40 delta-delta
    "max_frames": 216,
    "output_dir": "./outputs/S1_lstm_baseline/",
}

def extract_features(path: str) -> np.ndarray:
    """
    Extract MFCC + delta + delta-delta features, then per-sample normalize.
    Returns shape: (max_frames, n_features) = (216, 120)
    """
    y, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE)
    
    # 1. Base MFCCs: (n_mfcc, frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG["n_mfcc"])
    
    # 2. Delta features (1st order temporal derivative)
    delta = librosa.feature.delta(mfcc)
    
    # 3. Delta-delta features (2nd order temporal derivative)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 4. Concatenate: (120, frames) → transpose → (frames, 120)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T
    
    # 5. Per-sample normalization (zero-mean, unit-variance per feature)
    #    Critical: prevents MFCC scale differences from derailing gradients
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std
    
    # 6. Pad/truncate to fixed length
    if len(features) > CONFIG["max_frames"]:
        features = features[:CONFIG["max_frames"], :]
    else:
        pad_width = CONFIG["max_frames"] - len(features)
        features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
    
    return features

def run_training():
    output_path = Path(CONFIG["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TESS data for S1 baseline...")
    samples = load_tess()
    if not samples:
        logger.error("No TESS samples loaded for S1. Skipping.")
        return

    # Extract features (MFCC + delta + delta-delta + per-sample normalization)
    X, y = [], []
    for s in samples:
        try:
            X.append(extract_features(s["path"]))
            y.append(s["label"])
        except Exception as e:
            logger.warning(f"Failed to process {s['path']}: {e}")

    X = np.array(X)  # shape: (N, 216, 120)
    y = np.array(y)
    logger.info(f"Feature array shape: {X.shape}, labels: {y.shape}")
    logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 7-class to categorical
    y_cat = tf.keras.utils.to_categorical(y, num_classes=7)

    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

    # Model Architecture — with BatchNorm layers to prevent vanishing gradients
    model = models.Sequential([
        layers.Input(shape=(CONFIG["max_frames"], CONFIG["n_features"])),  # (216, 120)
        layers.LSTM(256, return_sequences=True),
        layers.LayerNormalization(),       # Stabilize LSTM output before second LSTM
        layers.Dropout(0.3),
        layers.LSTM(128),
        layers.LayerNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=[early_stop],
        verbose=1
    )
    train_time = (time.time() - start_time) / 60

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    # Simple F1 Calculation (Macro)
    y_pred = model.predict(X_val)
    preds = np.argmax(y_pred, axis=-1)
    true = np.argmax(y_val, axis=-1)
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(true, preds, average="macro")

    # Save
    h5_path = output_path / "lstm_ser_baseline.h5"
    keras_path = output_path / "lstm_ser_baseline.keras"
    model.save(str(h5_path))
    model.save(str(keras_path))

    # Peak VRAM (TF method)
    peak_vram = 0 # Difficult to track exactly like torch in cross-platform without pynvml
    
    # Unified results format
    summary = {
        "model_id": CONFIG["model_id"],
        "model_name": CONFIG["model_name"],
        "track": CONFIG["track"],
        "accuracy": round(float(val_acc), 4),
        "f1_macro": round(float(f1), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 2),
        "train_time_minutes": round(train_time, 2),
        "peak_vram_gb": 0.0,
        "epochs_trained": len(history.history["loss"]),
        "dataset": "TESS only",
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
