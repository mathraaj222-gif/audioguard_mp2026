"""
dataset_loader.py — SER (Speech Emotion Recognition) Data Loading
==================================================================
Unifies three datasets into a single HuggingFace DatasetDict:

  ┌─────────────────────┬──────────────┬────────────────────────────────────┐
  │ Dataset             │ Source       │ Classes                            │
  ├─────────────────────┼──────────────┼────────────────────────────────────┤
  │ RAVDESS             │ Auto-download│ neutral/calm/happy/sad/angry/      │
  │                     │              │ fearful/disgust/surprised           │
  ├─────────────────────┼──────────────┼────────────────────────────────────┤
  │ TESS                │ Auto-download│ neutral/happy/sad/angry/fear/      │
  │                     │              │ disgust/surprise                    │
  ├─────────────────────┼──────────────┼────────────────────────────────────┤
  │ IEMOCAP (optional)  │ Local path   │ neutral/happy/sad/angry/           │
  │                     │ via .env     │ frustrated/excited (6 classes)      │
  └─────────────────────┴──────────────┴────────────────────────────────────┘

Unified label set (7 classes):
  0=neutral, 1=happy, 2=sad, 3=angry, 4=fear, 5=disgust, 6=surprise

Audio is resampled to 16 kHz to match Whisper / Wav2Vec-BERT input.

Usage:
    from dataset_loader import load_ser_datasets
    ds = load_ser_datasets()
    print(ds["train"][0])  # {audio: {array, sampling_rate}, label: int}
"""

import os
import re
import logging
import zipfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, Audio
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED EMOTION MAP
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_LABEL_MAP = {
    "neutral":   0, "calm": 0,
    "happy":     1, "happiness": 1, "excited": 1,
    "sad":       2, "sadness": 2,
    "angry":     3, "anger": 3, "frustrated": 3,
    "fear":      4, "fearful": 4,
    "disgust":   5, "disgusted": 5,
    "surprise":  6, "surprised": 6, "ps": 6,
}

ID2EMOTION = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fear", 5: "disgust", 6: "surprise"}
NUM_EMOTIONS = 7
TARGET_SAMPLE_RATE = 16_000

# ─────────────────────────────────────────────────────────────────────────────
# RAVDESS LOADER
# ─────────────────────────────────────────────────────────────────────────────
# RAVDESS filename format: 03-01-01-01-01-01-01.wav
# Position 3 (1-indexed) = emotion: 01=neutral, 02=calm, 03=happy, 04=sad,
#                                    05=angry, 06=fearful, 07=disgust, 08=surprised
_RAVDESS_EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

_RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
_RAVDESS_SENTENCES_URL = "https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip"


def load_ravdess(cache_dir: str = "./datasets/ravdess") -> list[dict]:
    """Download RAVDESS and return list of {path, label, emotion_str} dicts."""
    cache = Path(cache_dir)
    audio_dir = cache / "audio"

    if not audio_dir.exists():
        logger.info("Downloading RAVDESS dataset (~200 MB)...")
        cache.mkdir(parents=True, exist_ok=True)
        zip_path = cache / "ravdess_speech.zip"
        urllib.request.urlretrieve(_RAVDESS_URL, zip_path)
        logger.info("Extracting RAVDESS...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(audio_dir)
        zip_path.unlink()
        logger.info("RAVDESS extracted.")

    samples = []
    for wav in audio_dir.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion_str = _RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion_str is None:
            continue
        label = EMOTION_LABEL_MAP.get(emotion_str)
        if label is None:
            continue
        samples.append({"path": str(wav), "label": label, "source": "ravdess"})

    logger.info(f"RAVDESS: {len(samples)} samples loaded.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# TESS LOADER
# ─────────────────────────────────────────────────────────────────────────────
# TESS filenames: OAF_angry_0001.wav, YAF_disgust_001.wav etc.
# Emotion is the second underscore-separated token.

_TESS_URL = "https://www.kaggle.com/api/v1/datasets/download/ejlok1/toronto-emotional-speech-set-tess"


def load_tess(cache_dir: str = "./datasets/tess") -> list[dict]:
    """
    Load TESS from local cache or attempt Kaggle API download.
    Falls back gracefully if Kaggle API is not configured.
    """
    cache = Path(cache_dir)

    if not cache.exists() or not list(cache.rglob("*.wav")):
        logger.info("TESS not found locally. Attempting to download via Kaggle API...")
        try:
            import subprocess
            cache.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "ejlok1/toronto-emotional-speech-set-tess",
                 "--unzip", "-p", str(cache)],
                check=True,
            )
            logger.info("TESS downloaded via Kaggle CLI.")
        except Exception as e:
            logger.warning(f"Could not download TESS automatically: {e}")
            logger.warning("Please manually download TESS from https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
            logger.warning(f"and extract to: {cache}")
            return []

    samples = []
    for wav in cache.rglob("*.wav"):
        name_lower = wav.stem.lower()
        # Try to match emotion from filename
        for emo_key in EMOTION_LABEL_MAP:
            if emo_key in name_lower:
                label = EMOTION_LABEL_MAP[emo_key]
                samples.append({"path": str(wav), "label": label, "source": "tess"})
                break

    logger.info(f"TESS: {len(samples)} samples loaded.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP LOADER (optional — requires manual dataset access from USC)
# ─────────────────────────────────────────────────────────────────────────────
# IEMOCAP emotions: neutral, happiness, sadness, anger, frustration, excited, other
# We map to the unified set and skip 'other'.

_IEMOCAP_EMOTION_REMAP = {
    "neu": "neutral", "hap": "happy", "exc": "happy",
    "sad": "sad", "ang": "angry", "fru": "angry",
    "fea": "fear", "dis": "disgust", "sur": "surprise",
}


def load_iemocap(iemocap_root: Optional[str] = None) -> list[dict]:
    """
    Load IEMOCAP from a local directory.
    Expects the standard IEMOCAP directory structure:
        <iemocap_root>/Session*/dialog/EmoEvaluation/*.txt
        <iemocap_root>/Session*/sentences/wav/*/*.wav
    """
    root = Path(iemocap_root) if iemocap_root else Path(os.environ.get("IEMOCAP_PATH", ""))

    if not root.exists():
        logger.info("IEMOCAP path not set or not found — skipping IEMOCAP.")
        return []

    samples = []
    for eval_file in root.rglob("EmoEvaluation/*.txt"):
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("["):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                utt_id = parts[1].strip()          # e.g. Ses01F_impro01_F000
                emo_code = parts[2].strip()[:3].lower() if len(parts) > 2 else ""

                emotion_str = _IEMOCAP_EMOTION_REMAP.get(emo_code)
                if emotion_str is None:
                    continue
                label = EMOTION_LABEL_MAP.get(emotion_str)
                if label is None:
                    continue

                # Find the corresponding .wav file
                session = utt_id.split("_")[0][:6]   # e.g. Ses01F
                dialog = "_".join(utt_id.split("_")[:2])  # e.g. Ses01F_impro01
                wav_path = root / f"{session}_*" / "sentences" / "wav" / dialog / f"{utt_id}.wav"

                # Glob to handle session number variation
                matches = list(root.rglob(f"sentences/wav/{dialog}/{utt_id}.wav"))
                if matches:
                    samples.append({"path": str(matches[0]), "label": label, "source": "iemocap"})

    logger.info(f"IEMOCAP: {len(samples)} samples loaded from {root}.")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_ser_datasets(
    ravdess_cache: str = "./datasets/ravdess",
    tess_cache: str = "./datasets/tess",
    iemocap_root: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """
    Load and unify RAVDESS + TESS + (optionally) IEMOCAP into a single
    HuggingFace DatasetDict with Audio column (16 kHz).

    Returns:
        DatasetDict with splits: 'train', 'validation', 'test'
        Each example: {'audio': {'array': np.ndarray, 'sampling_rate': 16000}, 'label': int}
    """
    all_samples = []
    all_samples.extend(load_ravdess(cache_dir=ravdess_cache))
    all_samples.extend(load_tess(cache_dir=tess_cache))
    all_samples.extend(load_iemocap(iemocap_root=iemocap_root))

    if not all_samples:
        raise RuntimeError("No SER dataset samples loaded. Please check dataset paths.")

    from collections import Counter
    label_counts = Counter(s["label"] for s in all_samples)
    logger.info(f"Total SER samples: {len(all_samples)}")
    logger.info(f"Label distribution: { {ID2EMOTION[k]: v for k, v in sorted(label_counts.items())} }")

    paths = [s["path"] for s in all_samples]
    labels = [s["label"] for s in all_samples]
    sources = [s["source"] for s in all_samples]

    # Stratified splits
    indices = list(range(len(paths)))
    idx_train, idx_temp = train_test_split(indices, test_size=(test_size + val_size),
                                           random_state=seed, stratify=labels)
    labels_temp = [labels[i] for i in idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=test_size / (test_size + val_size),
        random_state=seed, stratify=labels_temp
    )

    def make_dataset(idxs):
        return Dataset.from_dict({
            "audio": [paths[i] for i in idxs],
            "label": [labels[i] for i in idxs],
            "source": [sources[i] for i in idxs],
        }).cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    ds = DatasetDict({
        "train": make_dataset(idx_train),
        "validation": make_dataset(idx_val),
        "test": make_dataset(idx_test),
    })

    logger.info(f"SER DatasetDict ready — Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Quick-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = load_ser_datasets()
    print(ds)
    sample = ds["train"][0]
    print("Sample keys:", list(sample.keys()))
    print("Label:", sample["label"], "→", ID2EMOTION[sample["label"]])
    print("Audio shape:", np.array(sample["audio"]["array"]).shape)
