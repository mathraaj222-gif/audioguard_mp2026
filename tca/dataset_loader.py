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
     → Labels: 0=entailment, 1=neutral, 2=contradiction
     → Used for DeBERTa-v3-Large fine-tuning
"""

import os
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
DAVIDSON_NUM_LABELS = 3


def load_davidson(test_size: float = 0.15, val_size: float = 0.10, seed: int = 42) -> DatasetDict:
    """
    Download and split the Davidson et al. hate speech dataset.

    Returns a DatasetDict with keys: 'train', 'validation', 'test'
    Each example has fields: 'text' (str) and 'label' (int 0/1/2)
    """
    logger.info("Downloading Davidson et al. hate_speech_offensive dataset from HuggingFace...")
    raw = load_dataset("hate_speech_offensive", split="train")  # only one split available

    # Keep only 'tweet' and 'class' columns; rename for clarity
    raw = raw.select_columns(["tweet", "class"])
    raw = raw.rename_column("tweet", "text")
    raw = raw.rename_column("class", "label")

    # Convert to pandas for easy splitting
    df = raw.to_pandas()
    logger.info(f"Total Davidson samples: {len(df)}")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    # Stratified splits
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

NLI_LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
NLI_NUM_LABELS = 3

# Default path relative to this file's location
_DEFAULT_NLI_CSV = Path(__file__).resolve().parent.parent / "datasets" / "hate_speech_ethics_dataset_300.csv"


def load_nli_csv(
    csv_path: str | Path | None = None,
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
) -> DatasetDict:
    """
    Load the custom NLI ethics CSV and return train/val/test DatasetDict.

    Each example has fields:
      - 'premise'    (str): the premise sentence
      - 'hypothesis' (str): the hypothesis sentence
      - 'label'      (int): 0=entailment, 1=neutral, 2=contradiction
    """
    path = Path(csv_path) if csv_path else _DEFAULT_NLI_CSV

    if not path.exists():
        raise FileNotFoundError(
            f"NLI CSV not found at: {path}\n"
            "Please provide the correct path via the `csv_path` argument."
        )

    df = pd.read_csv(path)

    # Normalise column names (case-insensitive)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "premise" in df.columns, "CSV must contain a 'Premise' column"
    assert "hypothesis" in df.columns, "CSV must contain a 'Hypothesis' column"
    assert "label" in df.columns, "CSV must contain a 'Label' column"

    df = df[["premise", "hypothesis", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    logger.info(f"Loaded NLI CSV from {path} — {len(df)} rows")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    df_train, df_temp = train_test_split(
        df, test_size=(test_size + val_size), random_state=seed, stratify=df["label"]
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
        stratify=df_temp["label"],
    )

    logger.info(f"NLI splits — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    return DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, preserve_index=False),
            "validation": Dataset.from_pandas(df_val, preserve_index=False),
            "test": Dataset.from_pandas(df_test, preserve_index=False),
        }
    )


# ─────────────────────────────────────────────
# Quick-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Davidson Dataset ===")
    davidson = load_davidson()
    print(davidson)
    print(davidson["train"][0])

    print("\n=== NLI CSV Dataset ===")
    nli = load_nli_csv()
    print(nli)
    print(nli["train"][0])
