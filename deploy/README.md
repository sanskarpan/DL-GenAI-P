---
title: Audio Genre Classifier
emoji: 🎵
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.25.0"
app_file: app.py
pinned: false
license: mit
---

# 🎵 Audio Genre Classifier

**IIT Madras — BSDA2001P: Introduction to DL and GenAI Project**

Classify music into 10 genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock.

## Model
- **Features**: 288-dim MFCC + Chroma + Spectral (librosa)
- **Classifier**: XGBoost (200 trees, max_depth=6)
- **Best model** (used in competition): Fine-tuned AST Transformer — `MIT/ast-finetuned-audioset-10-10-0.4593` — achieved **0.93 Macro F1** on Kaggle leaderboard

## Pipeline
1. Load audio → resample to 22050 Hz mono
2. Extract 288-dim features (MFCC + Δ + Δ², Chroma, Spectral)
3. StandardScaler normalisation
4. XGBoost prediction
5. Visualise: mel spectrogram, MFCC heatmap, waveform, confidence chart

## Competition Results
| Model | Val Macro F1 |
|---|---|
| Random Baseline | 0.10 |
| XGBoost (MFCC) | 0.57 |
| SimpleCNN | 0.34 |
| AST Fine-Tuned | **0.93** ← submitted |
