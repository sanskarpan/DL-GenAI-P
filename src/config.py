"""
Central configuration for Messy Mashup competition.
All paths, hyperparameters, and constants defined here.
"""
import os
import random
from pathlib import Path

import numpy as np

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/sanskar/dev/DL-GenAI-P")
DATA_DIR = ROOT / "messy_mashup"
GENRES_DIR = DATA_DIR / "genres_stems"
MASHUPS_DIR = DATA_DIR / "mashups"
ESC50_DIR = DATA_DIR / "ESC-50-master"
ESC50_AUDIO_DIR = ESC50_DIR / "audio"
ESC50_META_CSV = ESC50_DIR / "meta" / "esc50.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = ROOT / "outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

# Create output dirs on import
for d in [OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Genre Mapping ────────────────────────────────────────────────────────────
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}
NUM_CLASSES = len(GENRES)

# Stem filenames (verified from actual data)
STEM_NAMES = ["drums.wav", "vocals.wav", "bass.wav", "other.wav"]

# ─── Audio Config ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050        # Hz - librosa default
CHUNK_DURATION = 10        # seconds per training chunk
N_MELS = 128               # mel bands
N_FFT = 2048               # FFT window size
HOP_LENGTH = 512           # hop length (controls time resolution)
FMIN = 20                  # min frequency (Hz)
FMAX = 8000                # max frequency (Hz)

# Derived
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION  # 220500 samples
N_FRAMES = 1 + CHUNK_SAMPLES // HOP_LENGTH    # ~431 time frames

# ─── Augmentation Config (Domain Shift Bridging) ──────────────────────────────
# Noise augmentation
NOISE_SNR_MIN = 5     # dB  (heavy noise)
NOISE_SNR_MAX = 25    # dB  (mild noise)

# Tempo augmentation
TEMPO_STRETCH_MIN = 0.85   # 15% slower
TEMPO_STRETCH_MAX = 1.15   # 15% faster

# Volume jitter per stem (dB)
STEM_VOLUME_MIN = -6   # dB
STEM_VOLUME_MAX = 6    # dB

# Number of stems to mix (2-4 from different songs of same genre)
MIX_STEMS_MIN = 2
MIX_STEMS_MAX = 4

# ─── Dataset Generation ───────────────────────────────────────────────────────
VAL_SONGS_PER_GENRE = 15    # hold-out songs per genre for validation
N_SYNTHETIC_TRAIN = 5000    # synthetic mashups for training
N_SYNTHETIC_VAL = 1000      # synthetic mashups for validation
N_TEST_CHUNKS = 5           # chunks to average for test prediction

# ─── Classical ML Config ──────────────────────────────────────────────────────
N_MFCC = 40                 # MFCC coefficients
MFCC_INCLUDE_DELTA = True   # include delta and delta-delta MFCCs

# ─── CNN Training Config ─────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10               # early stopping patience
GRAD_CLIP = 1.0             # gradient clipping
MODEL_NAME = "efficientnet_b0"  # timm model name
PRETRAINED = True
DROPOUT = 0.3
IMG_SIZE = 224              # input size for EfficientNet (resize mel spec)

# ─── AST (Audio Spectrogram Transformer) Config ──────────────────────────────
AST_SAMPLE_RATE = 16000     # AST expects 16kHz input
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# ─── Inference Config ─────────────────────────────────────────────────────────
TTA_CHUNKS = 8              # test-time augmentation: number of chunks to average

# ─── Device ───────────────────────────────────────────────────────────────────
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    except ImportError:
        return "cpu"
