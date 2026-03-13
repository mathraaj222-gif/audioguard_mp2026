"""
dataset_loader.py — TCA Data Loading Utilities
================================================
Loads two data sources for the Text Content Analysis pipeline:

  1. Davidson et al. (2017) Hate Speech & Offensive Language dataset
     → Downloaded automatically via HuggingFace `datasets` library
     → Labels: 0=hate speech, 1=offensive language, 2=neither
     → Used for HateBERT fine-tuning

  2. Custom NLI Ethics CSV (hate_speech_ethics_dataset_300.csv)
     → 300 Premise/Hypothesis pairs with 3-class NLI labels
     → Labels: 0=entailment, 1=contradiction, 2=neutral
     → Used for DeBERTa-v3-Large fine-tuning
"""

import os
import re
import logging
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Davidson et al. Hate Speech Dataset
# ─────────────────────────────────────────────

DAVIDSON_LABEL_MAP = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
DAVIDSON_BINARY_LABEL_MAP = {0: "hate", 1: "not-hate"}
DAVIDSON_NUM_LABELS = 3


def clean_tweet(text: str) -> str:
    """Standard normalization for tweets."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "@user", text)  # normalize mentions
    return text.strip()


def load_davidson(
    test_size: float = 0.10, 
    val_size: float = 0.10, 
    seed: int = 42, 
    binary: bool = False
) -> DatasetDict:
    """
    Download and split the Davidson et al. hate speech dataset.
    If binary=True, remaps labels: 0 -> 0 (hate), (1, 2) -> 1 (not-hate)
    
    CRITICAL: Remapping happens BEFORE splitting to ensure consistency.
    """
    logger.info(f"Loading Davidson dataset (binary={binary})...")
    # Using the correct handle specified in the prompt
    raw = load_dataset("tdavidson/hate_speech_offensive", split="train")

    # Keep only 'tweet' and 'class' columns
    df = pd.DataFrame({"text": raw["tweet"], "label": raw["class"]})

    # 1. Normalization
    df["text"] = df["text"].apply(clean_tweet)

    # 2. Remapping (BEFORE SPLIT)
    if binary:
        # 0=hate, 1=offensive, 2=neither
        # Target: 0=hate, 1=not-hate
        df["label"] = df["label"].map({0: 0, 1: 1, 2: 1})
        logger.info("Remapped labels to binary (0=hate, 1=not-hate)")

    logger.info(f"Total Davidson samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # 3. Stratified splits
    df_train, df_temp = train_test_split(
        df, test_size=(test_size + val_size), random_state=seed, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
        stratify=df_temp["label"],
    )

    logger.info(f"Davidson splits — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    return DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, preserve_index=False),
            "validation": Dataset.from_pandas(df_val, preserve_index=False),
            "test": Dataset.from_pandas(df_test, preserve_index=False),
        }
    )


# ─────────────────────────────────────────────
# Custom NLI Ethics CSV Dataset
# ─────────────────────────────────────────────

NLI_LABEL_MAP = {0: "entailment", 1: "contradiction", 2: "neutral"}
NLI_NUM_LABELS = 3

# Default path relative to this file's location
_DEFAULT_NLI_CSV = Path(__file__).resolve().parent.parent / "datasets" / "hate_speech_ethics_dataset_300.csv"


def load_nli_csv(
    csv_path: str | Path | None = None,
    test_size: float = 0.20,
    seed: int = 42,
) -> DatasetDict:
    """
    Load the custom NLI ethics CSV and return train/validation DatasetDict.
    Labels: 0=entailment, 1=contradiction, 2=neutral
    """
    if csv_path:
        path = Path(csv_path)
    else:
        # Robust path detection for NLI CSV
        filename = "hate_speech_ethics_dataset_300.csv"
        
        # 1. Check relative to this file (Local/Bundle)
        # When bundled: /kaggle/input/dataset-name/tca/dataset_loader.py
        # CSV is at: /kaggle/input/dataset-name/data_bundle/hate_...
        # OR: /kaggle/input/dataset-name/datasets/hate_...
        
        parent_dir = Path(__file__).resolve().parent.parent
        candidates = [
            # 0. Same directory as this script (works when CSV is copied into tca/ on Kaggle)
            Path(__file__).resolve().parent / filename,
            parent_dir / "data_bundle" / filename,
            parent_dir / "datasets" / filename,
            Path("/kaggle/input/audioguars-mp2026/data_bundle") / filename,
            Path("/kaggle/input/audioguars-mp2026/datasets") / filename,
            Path("/kaggle/working/datasets") / filename,
            _DEFAULT_NLI_CSV,
        ]
        
        path = None
        for cand in candidates:
            if cand.exists():
                path = cand
                logger.info(f"Using NLI CSV at: {path}")
                break
        
        if path is None:
            # Final fallback: search recursively for the filename from parent
            logger.warning(f"NLI CSV not found in standard paths. Searching under {parent_dir}...")
            for p in parent_dir.rglob(filename):
                path = p
                logger.info(f"FOUND NLI CSV via rglob: {path}")
                break
                
        if path is None:
            raise FileNotFoundError(f"NLI CSV '{filename}' not found in any expected location.")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Requirement: 300 rows, balanced 100 per class
    # Columns: premise, hypothesis, label
    df = df[["premise", "hypothesis", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    logger.info(f"Loaded NLI CSV from {path} — {len(df)} rows")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # Requirements say 80% train, 20% validation
    df_train, df_val = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )

    logger.info(f"NLI splits — Train: {len(df_train)}, Val: {len(df_val)}")

    return DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, preserve_index=False),
            "validation": Dataset.from_pandas(df_val, preserve_index=False),
            "test": Dataset.from_pandas(df_val, preserve_index=False), 
        }
    )


if __name__ == "__main__":
    print("\n=== Davidson Dataset (Multi) ===")
    d_multi = load_davidson(binary=False)
    print(d_multi)

    print("\n=== Davidson Dataset (Binary) ===")
    d_bin = load_davidson(binary=True)
    print(d_bin)

    print("\n=== NLI CSV Dataset ===")
    nli = load_nli_csv()
    print(nli)
