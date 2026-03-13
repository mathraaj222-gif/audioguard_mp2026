"""
evaluate_ser_all.py — Unified SER Evaluation and Leaderboard Generation
=======================================================================
1. Loads S1-S7 models from ./outputs/
2. Runs inference on corresponding test sets
3. Produces a leaderboard CSV and formatted table
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import torch
import librosa
import tensorflow as tf
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from dataset_loader import load_ser_datasets, TARGET_SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "S1": {"name": "LSTM-Baseline", "type": "keras"},
    "S2": {"name": "openai/whisper-large-v3", "type": "hf_encoder_only"},
    "S3": {"name": "facebook/w2v-bert-2.0", "type": "hf_custom"},
    "S4": {"name": "facebook/wav2vec2-large-960h", "type": "hf"},
    "S5": {"name": "microsoft/wavlm-large", "type": "hf"},
    "S6": {"name": "superb/hubert-large-superb-er", "type": "hf"},
    "S7": {"name": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", "type": "hf_custom"},
}

def extract_mfcc(path, n_mfcc=40, max_frames=216):
    """Extract MFCC + delta + delta-delta features to match train_lstm_baseline.py's extract_features()."""
    y, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0).T  # (T, 120)
    if len(features) > max_frames: features = features[:max_frames, :]
    else: features = np.pad(features, ((0, max_frames - len(features)), (0, 0)), mode="constant")
    return features

def run_evaluation():
    outputs_dir = Path("./outputs")
    leaderboard = []

    # Load test dataset once
    ds = load_ser_datasets()
    test_data = ds["test"]

    for model_id, config in MODEL_CONFIGS.items():
        model_path = list(outputs_dir.glob(f"{model_id}_*"))
        if not model_path:
            logger.warning(f"Model {model_id} not found. Skipping.")
            continue
        model_path = model_path[0]

        logger.info(f"Evaluating {model_id} from {model_path}...")
        
        try:
            preds, labels = [], []
            
            if config["type"] == "keras":
                # S1 Baseline
                model = tf.keras.models.load_model(str(model_path / "lstm_ser_baseline.keras"))
                for sample in test_data:
                    mfcc = extract_mfcc(sample["path"])
                    p = model.predict(mfcc[np.newaxis, ...], verbose=0)
                    preds.append(np.argmax(p))
                    labels.append(sample["label"])
            
            else:
                # HF Models
                feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_path))
                
                # Import custom classes if needed
                if model_id == "S2":
                    from train_whisper_ser import WhisperSERModel
                    model = WhisperSERModel("openai/whisper-large-v3", num_labels=7)
                    model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location=DEVICE))
                elif model_id == "S3":
                    from train_wav2vec_bert import Wav2VecBertSERModel
                    model = Wav2VecBertSERModel("facebook/w2v-bert-2.0", num_labels=7)
                    model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location=DEVICE))
                elif model_id == "S7":
                    from train_wav2vec2_robust import Wav2Vec2RobustSERModel
                    model = Wav2Vec2RobustSERModel("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", num_labels=7)
                    model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location=DEVICE))
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                
                model.to(DEVICE).eval()
                
                for sample in test_data:
                    audio = sample["audio"]["array"]
                    inputs = feature_extractor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
                    if "input_features" in inputs:
                        input_val = inputs.input_features.to(DEVICE)
                        with torch.no_grad():
                            logits = model(input_features=input_val).logits
                    else:
                        input_val = inputs.input_values.to(DEVICE)
                        with torch.no_grad():
                            logits = model(input_val).logits
                    
                    preds.append(torch.argmax(logits, dim=-1).item())
                    labels.append(sample["label"])

            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            
            leaderboard.append({
                "model_id": model_id,
                "model_name": config["name"],
                "track": "SER",
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "precision_macro": round(precision, 4),
                "recall_macro": round(recall, 4),
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_id}: {e}")

    if not leaderboard:
        logger.error("No models were successfully evaluated.")
        return

    df = pd.DataFrame(leaderboard).sort_values(by="f1_macro", ascending=False)
    csv_path = outputs_dir / "ser_leaderboard.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print(f"{'SER LEADERBOARD':^80}")
    print("="*80)
    print(df[["model_id", "model_name", "accuracy", "f1_macro"]].to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    run_evaluation()
