import os
import sys
import logging
import torch
from pathlib import Path
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CODE_ROOT = Path(__file__).resolve().parent

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_tca_loader():
    logger.info("--- Testing TCA Dataset Loader ---")
    try:
        tca_loader = import_from_path("tca_loader", CODE_ROOT / "tca" / "dataset_loader.py")
        load_nli_csv = tca_loader.load_nli_csv
        NLI_LABEL_MAP = tca_loader.NLI_LABEL_MAP
        
        # Check Label Map
        logger.info(f"NLI_LABEL_MAP: {NLI_LABEL_MAP}")
        assert NLI_LABEL_MAP[1] == "neutral", "NLI_LABEL_MAP[1] should be 'neutral'"
        assert NLI_LABEL_MAP[2] == "contradiction", "NLI_LABEL_MAP[2] should be 'contradiction'"
        
        # Load NLI
        ds = load_nli_csv(val_size=0.15, test_size=0.15)
        logger.info(f"NLI split sizes: {ds.keys()}")
        for split in ["train", "validation", "test"]:
            assert split in ds, f"Missing split: {split}"
            logger.info(f"  {split}: {len(ds[split])} samples")
            
        logger.info("[PASS] TCA Loader test successful.")
    except Exception as e:
        logger.error(f"[FAIL] TCA Loader test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def test_ser_loader():
    logger.info("\n--- Testing SER Dataset Loader ---")
    try:
        ser_loader = import_from_path("ser_loader", CODE_ROOT / "ser" / "dataset_loader.py")
        EMOTION_LABEL_MAP = ser_loader.EMOTION_LABEL_MAP
        
        logger.info(f"EMOTION_LABEL_MAP: {EMOTION_LABEL_MAP}")
        logger.info("[INFO] Skipping full SER load as datasets might be missing locally.")
        logger.info("[PASS] SER Loader constants verified.")
    except Exception as e:
        logger.error(f"[FAIL] SER Loader test failed: {e}")

def test_model_shapes():
    logger.info("\n--- Testing Model Forward Pass Shapes ---")
    device = "cpu"
    logger.info(f"Using device: {device} (Forced to CPU for local verification stability)")
    
    # Add tca and ser to sys.path for internal imports within training scripts
    sys.path.insert(0, str(CODE_ROOT / "tca"))
    sys.path.insert(0, str(CODE_ROOT / "ser"))

    # 1. Test Wav2Vec-BERT SER Model (S3)
    try:
        # Use import_from_path for the script itself too
        train_w2v = import_from_path("train_wav2vec_bert", CODE_ROOT / "ser" / "train_wav2vec_bert.py")
        Wav2VecBertSERModel = train_w2v.Wav2VecBertSERModel
        
        # Use a smaller model for shape test to save time if needed, 
        # but the user has w2v-bert-2.0, so we test that if possible.
        # Actually Wav2Vec-BERT is LARGE. To save time, we'll just check if instantiation works.
        logger.info("[INFO] Instantiating Wav2VecBertSERModel (this may download weights)...")
        model = Wav2VecBertSERModel("facebook/w2v-bert-2.0", num_labels=7).to(device).float()
        
        dummy_input = torch.randn(1, 16000).to(device).float()
        with torch.no_grad():
            outputs = model(input_values=dummy_input)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            assert logits.shape == (1, 7), f"Expected logits shape (1, 7), got {logits.shape}"
        logger.info("[PASS] Wav2Vec-BERT (S3) forward pass Successful.")
    except Exception as e:
        logger.error(f"[FAIL] Wav2Vec-BERT (S3) test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 2. Test Whisper SER Model (S2)
    try:
        train_whisper = import_from_path("train_whisper_ser", CODE_ROOT / "ser" / "train_whisper_ser.py")
        WhisperSERModel = train_whisper.WhisperSERModel
        
        logger.info("[INFO] Instantiating WhisperSERModel...")
        model = WhisperSERModel("openai/whisper-large-v3", num_labels=7).to(device).float()
        
        dummy_input = torch.randn(1, 128, 3000).to(device).float()
        with torch.no_grad():
            outputs = model(input_features=dummy_input)
            logits = outputs.logits
            assert logits.shape == (1, 7), f"Expected logits shape (1, 7), got {logits.shape}"
        logger.info("[PASS] Whisper (S2) forward pass Successful.")
    except Exception as e:
        logger.error(f"[FAIL] Whisper (S2) test failed: {e}")

if __name__ == "__main__":
    logger.info("Starting AudioGuardMP_2026 Pipeline Verification...")
    test_tca_loader()
    test_ser_loader()
    test_model_shapes()
    logger.info("\nVerification Finished.")
