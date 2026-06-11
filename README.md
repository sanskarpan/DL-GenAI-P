# Messy Mashup: Music Genre Classification

Audio genre classification on synthetic mashups — a domain adaptation challenge where models train on clean stems but predict on noisy mashup recordings.

## Problem

- **Training data**: Clean instrument stems (drums, vocals, bass, other) organized by genre
- **Test data**: Noisy mashups of stems mixed with environmental noise and tempo variations
- **Goal**: Classify each mashup into one of 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **Metric**: Macro F1 score

## Models

| Model | Type | Val Macro F1 | Description |
|-------|------|-------------|-------------|
| XGBoost | Classical ML | ~0.45 | MFCC + chroma + spectral features (286-dim) |
| SimpleCNN | From Scratch | ~0.55 | 4-block CNN on mel spectrograms (422K params) |
| AST-XGBoost | Pretrained | ~0.78 | Frozen Audio Spectrogram Transformer embeddings (768-dim) + XGBoost |

## Project Structure

```
DL-GenAI-P/
├── src/                          # Core library (modular pipeline)
│   ├── config.py                 # Central configuration & hyperparameters
│   ├── augment.py                # Synthetic mashup creation + ESC-50 noise
│   ├── features.py               # MFCC, mel spectrogram, chroma extraction
│   ├── models.py                 # SimpleCNN, EfficientNet, AudioResNet
│   ├── dataset.py                # PyTorch Dataset classes
│   ├── train.py                  # Training loop with early stopping
│   ├── predict.py                # Inference with test-time augmentation
│   └── baseline_ml.py            # Classical ML baseline pipeline
├── scripts/                      # Standalone scripts
│   ├── wandb_experiments.py      # W&B experiment tracking (all 3 models)
│   ├── train_cnn.py              # External CNN training (MPS-safe)
│   ├── extract_ast_features.py   # AST embedding extraction
│   ├── generate_spectrograms.py  # Pre-compute mel spectrograms
│   └── gen_kaggle_nb.py          # Kaggle submission notebook generator
├── notebooks/                    # Jupyter notebooks
│   ├── messy_mashup.ipynb        # Main development notebook
│   ├── kaggle_submisssion.ipynb  # Kaggle submission notebook
│   ├── milestone3.ipynb          # Milestone 3 questions
│   └── milestone4.ipynb          # Milestone 4 questions + CRNN
├── requirements.txt              # Python dependencies
└── README.md
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd DL-GenAI-P

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place dataset in messy_mashup/ directory:
#   messy_mashup/genres_stems/   (training stems)
#   messy_mashup/mashups/        (test mashups)
#   messy_mashup/ESC-50-master/  (noise dataset)
#   messy_mashup/test.csv
```

## Running Experiments

### 1. Generate features

```bash
# Mel spectrograms for CNN
venv/bin/python scripts/generate_spectrograms.py

# AST embeddings
venv/bin/python scripts/extract_ast_features.py --no_mfcc
```

### 2. Train models with W&B logging

```bash
# All 3 models with W&B tracking
venv/bin/python scripts/wandb_experiments.py

# Or train CNN separately
venv/bin/python scripts/train_cnn.py --epochs 50 --patience 15
```

### 3. Generate Kaggle submission

```bash
venv/bin/python scripts/gen_kaggle_nb.py
```

## Key Design Decisions

- **Synthetic data augmentation**: Mix 2-4 stems from same genre + ESC-50 noise at 5-25 dB SNR to match test distribution
- **ESC-50 cache**: 300 clips pre-loaded into RAM for fast augmentation (70ms/sample)
- **No SpecAugment for CNN**: Corrupts BatchNorm running statistics, causing train/val distribution mismatch
- **MPS safety**: `torch.mps.synchronize()` before `empty_cache()` to prevent async double-free crashes

## Experiment Tracking

All experiments are tracked with [Weights & Biases](https://wandb.ai/23f3003478-iit-madras/23f3003478-t12026).

## Dependencies

See `requirements.txt`. Key packages:
- `torch`, `torchaudio` — deep learning
- `librosa`, `soundfile` — audio processing
- `transformers` — AST model
- `xgboost`, `scikit-learn` — classical ML
- `wandb` — experiment tracking
