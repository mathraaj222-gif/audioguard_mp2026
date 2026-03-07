#!/usr/bin/env python3
"""
train_on_kaggle.py — Full Orchestration Script for Kaggle T4 x2 GPU Kernel
============================================================================
This is the SINGLE ENTRY POINT that runs inside a Kaggle notebook kernel.
It orchestrates the complete AudioGuardMP_2026 training pipeline:

  Phase 1: Install dependencies
  Phase 2: Download datasets
  Phase 3: TCA Training  → HateBERT + DeBERTa-v3-Large
  Phase 4: SER Training  → Whisper-Large-v3 + Wav2Vec-BERT 2.0
  Phase 5: Package & persist all model artifacts to /kaggle/working/

Kaggle Environment:
  - GPU: T4 x2 (2× NVIDIA T4, 15 GB VRAM each)
  - RAM: 30 GB
  - Disk: 20 GB per session (/kaggle/working/)
  - Python: 3.10+
  - Internet: enabled (required for dataset downloads)

All model outputs are saved to /kaggle/working/ which Kaggle auto-persists
as "kernel output" files that can be downloaded after training completes.

Estimated runtime on T4 x2:
  - HateBERT  (5 epochs, ~21k samples): ~25 min
  - DeBERTa   (10 epochs, ~240 samples): ~15 min
  - Whisper   (8 epochs, ~3k samples): ~90 min
  - Wav2Vec-BERT (10 epochs, ~3k samples): ~60 min
  Total: ~3.5 hours
"""

import os
import sys
import json
import time
import zipfile
import logging
import subprocess
import torch
from pathlib import Path

# -----------------------------------------------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/kaggle/working/training.log"),
    ],
)
logger = logging.getLogger(__name__)

WORKING_DIR = Path("/kaggle/working")
def _find_code_dir() -> Path:
    """Auto-detect the Kaggle input directory for this kernel."""
    kaggle_input = Path("/kaggle/input")
    # Prefer the correct slug name
    preferred = kaggle_input / "audioguardmp-2026-multimodal-training-pipeline"
    if preferred.exists():
        return preferred
    # Fallback: scan for any subdirectory (in case slug changes again)
    if kaggle_input.exists():
        subdirs = [d for d in kaggle_input.iterdir() if d.is_dir()]
        if subdirs:
            return subdirs[0]
    # Last resort: old slug
    return kaggle_input / "audioguard-2026-training"

CODE_DIR = _find_code_dir()   # Kaggle input directory (auto-detected)

# Output directories
TCA_HATEBERT_OUTPUT  = WORKING_DIR / "outputs" / "hatebert_finetuned"
TCA_DEBERTA_OUTPUT   = WORKING_DIR / "outputs" / "deberta_nli_finetuned"
SER_WHISPER_OUTPUT   = WORKING_DIR / "outputs" / "whisper_ser_finetuned"
SER_WAV2VEC_OUTPUT   = WORKING_DIR / "outputs" / "wav2vec_bert_ser_finetuned"

RAVDESS_CACHE = WORKING_DIR / "datasets" / "ravdess"
TESS_CACHE    = WORKING_DIR / "datasets" / "tess"
NLI_CSV_PATH  = CODE_DIR / "datasets" / "hate_speech_ethics_dataset_300.csv"

# -----------------------------------------------------------------------------
# PHASE 0: ENVIRONMENT CHECK
# -----------------------------------------------------------------------------
def env_check():
    logger.info("=" * 70)
    logger.info("PHASE 0: Environment Check")
    logger.info("=" * 70)

    # torch imported at top
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name}  ({props.total_memory // 1024**2} MB VRAM)")
    logger.info(f"Working dir: {WORKING_DIR}")
    logger.info(f"Code dir:    {CODE_DIR}  (exists: {CODE_DIR.exists()})")
    if not CODE_DIR.exists():
        kaggle_input = Path("/kaggle/input")
        available = list(kaggle_input.iterdir()) if kaggle_input.exists() else []
        logger.warning(f"CODE_DIR not found! Available in /kaggle/input: {available}")
    logger.info("[OK] Environment check passed.")


# -----------------------------------------------------------------------------
# PHASE 1: INSTALL DEPENDENCIES
# -----------------------------------------------------------------------------
def install_dependencies():
    logger.info("=" * 70)
    logger.info("PHASE 1: Installing Dependencies")
    logger.info("=" * 70)

    packages = [
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.4.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "audioread>=3.0.0",
        "SentencePiece>=0.1.99",
        "protobuf>=4.25.0",
        "torchaudio",
    ]

    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"pip install failed:\n{result.stderr}")
        raise RuntimeError("Dependency installation failed.")
    logger.info("[OK] Dependencies installed.")


# -----------------------------------------------------------------------------
# PHASE 2: PREPARE DATASETS
# -----------------------------------------------------------------------------
def prepare_datasets():
    logger.info("=" * 70)
    logger.info("PHASE 2: Preparing Datasets")
    logger.info("=" * 70)

    # Add code directories to Python path
    sys.path.insert(0, str(CODE_DIR / "tca"))
    sys.path.insert(0, str(CODE_DIR / "ser"))

    # RAVDESS download (handled inside ser/dataset_loader.py)
    logger.info("Checking RAVDESS cache...")
    RAVDESS_CACHE.mkdir(parents=True, exist_ok=True)

    # TESS download via Kaggle API (already authenticated in Kaggle kernel)
    logger.info("Checking TESS cache...")
    TESS_CACHE.mkdir(parents=True, exist_ok=True)
    if not list(TESS_CACHE.rglob("*.wav")):
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d",
                 "ejlok1/toronto-emotional-speech-set-tess",
                 "--unzip", "-p", str(TESS_CACHE)],
                check=True
            )
            logger.info("[OK] TESS downloaded.")
        except Exception as e:
            logger.warning(f"TESS download via Kaggle CLI failed: {e}. Continuing without TESS.")

    logger.info("[OK] Dataset preparation complete.")


# -----------------------------------------------------------------------------
# PHASE 3: TCA TRAINING
# -----------------------------------------------------------------------------
def run_tca_training():
    logger.info("=" * 70)
    logger.info("PHASE 3: Text Content Analysis Training")
    logger.info("=" * 70)

    sys.path.insert(0, str(CODE_DIR / "tca"))

    # 3a. HateBERT on Davidson dataset
    logger.info("\n--- 3a. HateBERT (Davidson Hate Speech) ---")
    t0 = time.time()
    from train_hatebert import run_training as run_hatebert_training
    run_hatebert_training() # No args, uses CONFIG inside
    logger.info(f"[OK] HateBERT done in {(time.time()-t0)/60:.1f} min.")

    # 3b. DeBERTa-v3-Large on NLI CSV
    logger.info("\n--- 3b. DeBERTa-v3-Large (NLI Ethics CSV) ---")
    t0 = time.time()
    from train_deberta_large import run_training as run_deberta_training
    run_deberta_training()
    logger.info(f"[OK] DeBERTa done in {(time.time()-t0)/60:.1f} min.")

    logger.info("[OK] TCA training complete.")


# -----------------------------------------------------------------------------
# PHASE 4: SER TRAINING
# -----------------------------------------------------------------------------
def run_ser_training():
    logger.info("=" * 70)
    logger.info("PHASE 4: Speech Emotion Recognition Training")
    logger.info("=" * 70)

    sys.path.insert(0, str(CODE_DIR / "ser"))

    # 4a. Whisper-Large-v3
    logger.info("\n--- 4a. Whisper-Large-v3 SER ---")
    t0 = time.time()
    from train_whisper_ser import run_training as run_whisper_ser_training
    run_whisper_ser_training()
    logger.info(f"[OK] Whisper SER done in {(time.time()-t0)/60:.1f} min.")

    # 4b. Wav2Vec-BERT 2.0
    logger.info("\n--- 4b. Wav2Vec-BERT 2.0 SER ---")
    t0 = time.time()
    from train_wav2vec_bert import run_training as run_wav2vec_bert_ser_training
    run_wav2vec_bert_ser_training()
    logger.info(f"[OK] Wav2Vec-BERT SER done in {(time.time()-t0)/60:.1f} min.")

    logger.info("[OK] SER training complete.")


# -----------------------------------------------------------------------------
# PHASE 5: PACKAGE ARTIFACTS
# -----------------------------------------------------------------------------
def package_artifacts():
    logger.info("=" * 70)
    logger.info("PHASE 5: Packaging Model Artifacts")
    logger.info("=" * 70)

    summary = {"models": {}}

    for name, path in [
        ("hatebert",      TCA_HATEBERT_OUTPUT),
        ("deberta_nli",   TCA_DEBERTA_OUTPUT),
        ("whisper_ser",   SER_WHISPER_OUTPUT),
        ("wav2vec_bert",  SER_WAV2VEC_OUTPUT),
    ]:
        metrics_file = path / "training_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            summary["models"][name] = {
                "output_dir": str(path),
                "metrics": {k: v for k, v in metrics.items()
                           if k not in ["classification_report"]},
            }
            logger.info(f"  {name}: {path}")
        else:
            logger.warning(f"  {name}: metrics file not found at {metrics_file}")

    # Write training summary
    summary_path = WORKING_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n[OK] Training summary saved to: {summary_path}")

    # Create a manifest of all output files
    manifest = []
    for p in sorted(WORKING_DIR.rglob("*")):
        if p.is_file():
            size_mb = float(p.stat().st_size) / (1024**2)
            manifest.append({"path": str(p.relative_to(WORKING_DIR)), "size_mb": round(size_mb, 2)})

    with open(WORKING_DIR / "output_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Output manifest saved: {WORKING_DIR}/output_manifest.json")
    logger.info("[OK] All artifacts packaged.")


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_start = time.time()

    logger.info("+------------------------------------------------------------------+")
    logger.info("|       AudioGuardMP_2026 - Multimodal Training Pipeline          |")
    logger.info("|       Kaggle T4 x2 GPU Kernel                                   |")
    logger.info("+------------------------------------------------------------------+")

    try:
        env_check()
        install_dependencies()
        prepare_datasets()
        run_tca_training()
        run_ser_training()
        package_artifacts()

        total_time = (time.time() - total_start) / 60
        logger.info(f"\n{'='*70}")
        logger.info(f"SUCCESS: ALL TRAINING COMPLETE in {total_time:.1f} minutes.")
        logger.info(f"   All models saved under /kaggle/working/outputs/")
        logger.info(f"   Download them via: kaggle kernels output <username>/audioguard-2026-training")
        logger.info(f"{'='*70}")

    except Exception as e:
        logger.error(f"\nERROR: Training pipeline FAILED: {e}", exc_info=True)
        sys.exit(1)
