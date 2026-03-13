# AudioGuardMP_2026 — Multimodal Training Pipeline

A cloud-based, production-grade training pipeline for the **AudioGuard FYP** project.
Combines **Text Content Analysis (TCA)** and **Speech Emotion Recognition (SER)**
fine-tuning, fully automated for deployment on a **Kaggle T4 × 2 GPU** kernel.

---

## Project Structure

```
AudioGuardMP_2026/
├── tca/                         ← Text Content Analysis
│   ├── dataset_loader.py        # Davidson + NLI CSV data loaders
│   ├── train_hatebert.py        # HateBERT fine-tuning (hate speech)
│   ├── train_deberta.py         # DeBERTa-v3-Large fine-tuning (NLI)
│   ├── evaluate_tca.py          # Unified TCA evaluation
│   └── requirements.txt
├── ser/                         ← Speech Emotion Recognition
│   ├── dataset_loader.py        # RAVDESS + TESS + IEMOCAP (optional) loader
│   ├── train_whisper.py         # Whisper-Large-v3 SER fine-tuning
│   ├── train_wav2vec_bert.py    # Wav2Vec-BERT 2.0 SER fine-tuning
│   ├── evaluate_ser.py          # Unified SER evaluation
│   └── requirements.txt
├── kaggle/                      ← Kaggle Automation Layer
│   ├── train_on_kaggle.py       # Main orchestration kernel script
│   ├── kernel-metadata.json     # Kaggle kernel configuration
│   ├── setup_kaggle_credentials.py  # Credential validator + auto-patcher
│   └── push_to_kaggle.py        # One-command push + status poller
├── data/
│   └── hate_speech_ethics_dataset_300.csv  # NLI ethics dataset
├── .env.example                 # Environment variable template
└── README.md
```

---

## Models

### Track 1: Text Content Analysis (TCA)

| Model | Task | Dataset | Labels |
|-------|------|---------|--------|
| `GroNLP/hateBERT` | Hate speech classification | Davidson et al. (~24k tweets) | 0=hate, 1=offensive, 2=neither |
| `microsoft/deberta-v3-large` | NLI ethics classification | Custom 300-row NLI CSV | 0=entailment, 1=neutral, 2=contradiction |

### Track 2: Speech Emotion Recognition (SER)

| Model | Task | Datasets | Labels |
|-------|------|---------|--------|
| `openai/whisper-large-v3` | Emotion from speech | RAVDESS + TESS + IEMOCAP* | 7 emotions |
| `facebook/w2v-bert-2.0` | Emotion from speech | RAVDESS + TESS + IEMOCAP* | 7 emotions |

*IEMOCAP optional — requires USC licence request.

**7 Unified Emotion Classes:** `neutral, happy, sad, angry, fear, disgust, surprise`

---

## Quick Start

### 1. Set Up Kaggle Credentials

```bash
# Get kaggle.json from kaggle.com/settings → API → Create New Token
# Place it at C:\Users\<YourName>\.kaggle\kaggle.json

cd "c:\AudioGuard FYP\AudioGuardMP_2026\kaggle"
python setup_kaggle_credentials.py
```

Expected output:
```
✓ Username  : your_kaggle_username
✓ API key   : ****xxxx
✓ Kaggle CLI connection successful!
✅ Kaggle credentials are VALID for user: your_username
```

### 2. (Optional) Configure IEMOCAP

```bash
# Copy and edit .env
cp .env.example .env
# Edit .env and set:
# IEMOCAP_PATH=C:/path/to/IEMOCAP_full_release
```

### 3. Push to Kaggle

```bash
# Dry run first (see what will be executed):
python kaggle/push_to_kaggle.py --dry-run

# Actual push:
python kaggle/push_to_kaggle.py
```

### 4. Monitor Training

```bash
# Check kernel status
kaggle kernels status <your-username>/audioguard-2026-training

# Stream output logs
kaggle kernels output <your-username>/audioguard-2026-training -p ./outputs
```

Or open the **browser URL** printed by `push_to_kaggle.py` after the push.

### 5. Download Model Weights

```bash
# After training completes (status = "complete"):
kaggle kernels output <your-username>/audioguard-2026-training -p ./outputs --force

# Output tree:
# outputs/
# ├── hatebert_finetuned/          ← HateBERT weights
# ├── deberta_nli_finetuned/       ← DeBERTa weights
# ├── whisper_ser_finetuned/       ← Whisper SER weights
# ├── wav2vec_bert_ser_finetuned/  ← Wav2Vec-BERT weights
# ├── training_summary.json        ← All metrics
# ├── output_manifest.json         ← File listing with sizes
# └── training.log                 ← Full training log
```

---

## Local Development

### Run TCA locally (CPU/GPU)

```bash
cd "c:\AudioGuard FYP\AudioGuardMP_2026"
pip install -r tca/requirements.txt

# Train HateBERT:
cd tca && python train_hatebert.py

# Train DeBERTa:
python train_deberta.py

# Evaluate both:
python evaluate_tca.py
```

### Run SER locally

```bash
pip install -r ser/requirements.txt

cd ser && python train_whisper.py
python train_wav2vec_bert.py
python evaluate_ser.py
```

---

## Architecture Decisions

### TCA
- **HateBERT** pre-trained on abusive Reddit content makes it the strongest baseline
  for hate speech detection vs. vanilla BERT.
- **DeBERTa-v3-Large** with sentence-pair encoding (premise + hypothesis) follows
  the standard NLI fine-tuning protocol from SuperGLUE.
- **Gradient checkpointing** enabled for DeBERTa to fit in 15 GB VRAM.

### SER
- **Whisper encoder** provides rich acoustic representations from 680k hours of
  speech pre-training; we discard the decoder entirely.
- **Wav2Vec-BERT 2.0** uses learnable scalar mixing across all 24 transformer layers
  to capture multi-granularity acoustic features.
- **Differential learning rates**: CNN feature extractor frozen (or very low LR),
  transformer layers at 2e-5, classification head at 1e-4.
- **Attention pooling** replaces simple mean-pooling for variable-length sequences.

### Kaggle Cloud Execution
- The `kaggle kernels push` command bundles the entire `kaggle/` folder.
- The kernel runs `train_on_kaggle.py` which dynamically adds `tca/` and `ser/`
  to the Python path from the Kaggle input directory.
- All outputs auto-persist in `/kaggle/working/` and are downloadable via CLI.

---

## Estimated Training Budget (T4 × 2)

| Stage | Time |
|-------|------|
| HateBERT (5 epochs ~21k samples) | ~25 min |
| DeBERTa-v3-L (10 epochs ~240 samples) | ~15 min |
| Whisper-Large-v3 SER | ~90 min |
| Wav2Vec-BERT 2.0 SER | ~60 min |
| **Total** | **~3.2 hours** |

Kaggle free-tier kernels allow up to **12 hours** per session.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `kaggle: command not found` | `pip install kaggle` |
| `401 Unauthorized` from Kaggle CLI | Re-download kaggle.json from kaggle.com/settings |
| TESS download fails | Manually download from [Kaggle TESS dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and extract to `ser/datasets/tess/` |
| IEMOCAP not loading | Set `IEMOCAP_PATH` in `.env` or pass `iemocap_root=` to `load_ser_datasets()` |
| OOM on Whisper local training | Reduce `batch_size` to 2 or use `--gradient_checkpointing` |
| DeBERTa SentencePiece error | `pip install sentencepiece protobuf` |


## Kaggle Integration & Synchronization

To streamline working with Kaggle, you have three options to keep your code in sync:

### 1. Native GitHub-to-Dataset Sync (Recommended) ⭐
Kaggle can pull your entire GitHub repository automatically into a Dataset.
1.  Go to **Kaggle > Datasets > New Dataset**.
2.  Select **"Remote URL"** or **"GitHub"** (if available).
3.  Enter your GitHub Repo URL: `https://github.com/your-username/AudioGuardMP_2026`.
4.  Set a periodic update (e.g., Daily) or trigger manually.
5.  In your Kaggle Notebook, add this Dataset as input. All folders (`ser/`, `tca/`) will be available at `/kaggle/input/audioguardmp-2026/`.

### 2. Automated Push via GitHub Actions
We've set up a [workflow](file:///.github/workflows/push_to_kaggle.yml) that pushes your code to Kaggle whenever you push to `main`.
- **Setup**:
    1.  Go to your GitHub Repo **Settings > Secrets and variables > Actions**.
    2.  Add `KAGGLE_USERNAME` (your Kaggle username).
    3.  Add `KAGGLE_KEY` (your Kaggle API Key from `kaggle.json`). 
- **Trigger**: Every `git push origin main` will update your Kaggle Kernel.

### 3. Local Push Script
Use the updated [push_to_kaggle.py](file:///c:/AudioGuard%20FYP/AudioGuardMP_2026/kaggle/push_to_kaggle.py) to push manually from your local machine:
```powershell
python kaggle/push_to_kaggle.py --session 1
```

---

*AudioGuardMP_2026 — Built 2026-03-01*2
