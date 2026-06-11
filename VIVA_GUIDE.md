# Messy Mashup: Complete Viva Guide

> **Submitted notebook**: `Submitted_0.93.ipynb` — Final score: **0.93 Macro F1**
> This guide covers every concept, design decision, code detail, and likely implementation change
> question the examiner can ask.

---

## Table of Contents
1. [Problem Statement & Dataset](#1-problem-statement--dataset)
2. [Setup: Configuration & Utilities](#2-setup-configuration--utilities)
3. [Augmentation Pipeline (The Core)](#3-augmentation-pipeline-the-core)
4. [Feature Extraction](#4-feature-extraction)
5. [M1: EDA](#5-m1-exploratory-data-analysis)
6. [M2: MFCC + XGBoost](#6-m2-classical-ml--mfcc--xgboost)
7. [M3: SimpleCNN on Mel Spectrograms](#7-m3-simplecnn-on-mel-spectrograms)
8. [M4: AST Fine-Tuning](#8-m4-ast-fine-tuning)
9. [M4 Fallback: Frozen AST + XGBoost](#9-m4-fallback-frozen-ast--xgboost)
10. [M5: Inference & Submission](#10-m5-inference--submission)
11. [Deep Concept Explanations](#11-deep-concept-explanations)
12. [Implementation Change Scenarios](#12-implementation-change-scenarios)
13. [Numbers to Remember](#13-numbers-to-remember)
14. [Key Design Decisions Q&A](#14-key-design-decisions-qa)

---

## 1. Problem Statement & Dataset

### What is the task?
**10-class audio genre classification**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.

### The Core Challenge: Domain Shift
| | Training Data | Test Data |
|---|---|---|
| **What it is** | Clean instrument stems | Noisy mashups |
| **Format** | 4 separate tracks per song | Single mixed audio file |
| **Noise** | None | ESC-50 environmental noise |
| **Tempo** | Original | ±15% variations |

**If you train naively on clean stems, you get ~0.45 F1 on test mashups.** The train distribution and test distribution are completely different. This is called **domain shift** or **covariate shift**.

**Solution**: Replicate the test generation process during training. Create synthetic mashups from stems, add noise, add tempo variation. This is **domain adaptation via data augmentation**.

### Dataset Structure
```
messy_mashup/
├── genres_stems/
│   ├── blues/
│   │   ├── song0001/  drums.wav  vocals.wav  bass.wav  other.wav
│   │   ├── song0002/  ...
│   │   └── ...
│   ├── classical/  ...
│   └── ... (10 genres total)
├── mashups/
│   ├── 0001.wav   0002.wav   ...   (test files)
├── ESC-50-master/
│   ├── audio/    (2000 environmental sound clips, 5s each, 50 categories)
│   └── meta/     esc50.csv
└── test.csv      (id, filename columns)
```

### Key Dataset Facts
- **4 stems per song**: `drums.wav`, `vocals.wav`, `bass.wav`, `other.wav`
  - ⚠️ There is a naming inconsistency in the raw data — some songs have `others.wav` instead of `other.wav`. The code checks both.
- **Val split**: 15 songs per genre held out → 150 validation songs total
- **Split is song-level** to prevent data leakage (if you split sample-level, train and val samples can come from the same song)
- **Balanced dataset** — roughly equal songs per genre → Macro F1 ≈ accuracy, but we still use F1 as required

### Why Macro F1 (not accuracy)?
`Macro F1 = mean(F1_per_class)`. Each class F1 = 2 × precision × recall / (precision + recall).

- Accuracy rewards majority classes. If model predicts "rock" for everything, accuracy could be 10% but it "tried"
- Macro F1 equally penalizes poor performance on any class
- Also: F1 is the harmonic mean of precision and recall, so it penalizes both false positives and false negatives
- Random baseline: 1/10 = **0.10 Macro F1**

---

## 2. Setup: Configuration & Utilities

### Cell 1 — All Hyperparameters

```python
# --- Audio ---
SAMPLE_RATE = 22050        # Hz — librosa default, standard for music
AST_SAMPLE_RATE = 16000    # Hz — AST was pretrained at 16kHz
CHUNK_DURATION = 10        # seconds per chunk
CHUNK_SAMPLES = 22050 * 10 = 220500  # samples
N_MELS = 128               # mel frequency bands
N_FFT = 2048               # FFT window = 2048/22050 ≈ 93ms
HOP_LENGTH = 512           # step between windows = 512/22050 ≈ 23ms
# → N_FRAMES = 1 + 220500//512 = 431 time frames
FMIN, FMAX = 20, 8000      # frequency range (human music range)

# --- Augmentation ---
NOISE_SNR_MIN, NOISE_SNR_MAX = 5, 25   # dB (5=very noisy, 25=mild noise)
STEM_VOLUME_MIN, STEM_VOLUME_MAX = -6, 6  # dB per stem
MIX_STEMS_MIN, MIX_STEMS_MAX = 2, 4    # songs to mix per mashup
VAL_SONGS_PER_GENRE = 15

# --- AST Fine-tuning ---
AST_MODEL_NAME = 'MIT/ast-finetuned-audioset-10-10-0.4593'
AST_EPOCHS = 12
AST_LR = 5e-5              # encoder learning rate (small, don't destroy weights)
AST_HEAD_LR = 1e-3         # classifier head LR (large, training from scratch)
AST_BATCH_SIZE = 8
AST_GRAD_ACCUM = 2         # effective batch = 8 × 2 = 16
AST_SAMPLES_PER_EPOCH = 4000
AST_VAL_SAMPLES = 1000
N_CROPS_TEST = 10          # test-time augmentation crops
```

**Why N_FFT=2048?** The window size determines frequency resolution. Longer window = finer frequency resolution but coarser time resolution. 2048 samples at 22050Hz = 93ms window, giving ~11 Hz frequency resolution. Good for music where notes last >100ms.

**Why HOP_LENGTH=512?** This controls time resolution. 512/22050 ≈ 23ms per frame. 75% overlap with N_FFT=2048. This is a standard trade-off between time resolution and computation.

**Why FMIN=20, FMAX=8000?** Human hearing range is ~20Hz–20kHz. Most music content is 20–8kHz. Above 8kHz is mostly noise and artifacts. Restricting the range focuses the model on meaningful frequencies and reduces spectrogram size.

### Device Detection
```python
if torch.cuda.is_available():    DEVICE = 'cuda'   # NVIDIA GPU
elif mps.is_available():         DEVICE = 'mps'    # Apple Silicon GPU
else:                            DEVICE = 'cpu'
```

### Cell 2 — Audio Loading

**`load_audio(path, sr, duration, offset)`**
```python
try:
    audio, _ = librosa.load(str(path), sr=sr, mono=True, ...)
except Exception:
    # Fallback: soundfile (handles more formats/edge cases)
    data, orig_sr = sf.read(str(path), always_2d=True)
    audio = data.mean(axis=1)   # stereo → mono
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
```
- `mono=True`: converts stereo/multichannel to mono by averaging channels
- `sr=22050`: resamples to target sample rate if original differs
- Always returns `float32` — standard for PyTorch tensors

**`pad_or_trim(audio, target_len)`**
- Too short → `np.pad(audio, (0, target_len - len(audio)), mode='constant')` — pads END with zeros
- Too long → `audio[:target_len]` — takes first N samples
- Ensures every audio array has exactly `target_len` samples (required for fixed-size CNN input)

**`random_crop(audio, target_len)`**
- If ≤ target_len: pad
- If > target_len: `start = random.randint(0, len(audio) - target_len)` → random slice
- This is a form of data augmentation — different starts give different views of the same song

### Cell 4 — sync_and_clear (Critical for MPS/CUDA)
```python
def sync_and_clear(device):
    if device.type == 'mps':
        torch.mps.synchronize()   # ← MUST come BEFORE empty_cache
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
```
**Why synchronize first?** Apple MPS and CUDA execute operations asynchronously (GPU runs ahead of CPU). If you call `empty_cache()` while GPU is still computing, you free memory that's still in use → **double-free crash (exit code 134)**. `synchronize()` blocks the CPU until all pending GPU ops finish.

---

## 3. Augmentation Pipeline (The Core)

This is the most important part of the project. Everything else is standard; this is what bridges the domain gap.

### SNR-based Noise Addition

```python
def add_noise_at_snr(signal, noise, snr_db):
    sig_pow = np.mean(signal ** 2) + 1e-10    # signal power (+ epsilon for stability)
    noi_pow = np.mean(noise ** 2) + 1e-10     # noise power
    target_pow = sig_pow / (10 ** (snr_db / 10))  # desired noise power
    noise_scaled = noise * np.sqrt(target_pow / noi_pow)
    return signal + noise_scaled
```

**The SNR math:**
- SNR (dB) = 10 · log₁₀(P_signal / P_noise)
- Therefore: P_noise = P_signal / 10^(SNR/10)
- Scale noise: multiply by √(target_power / current_power)
- Why sqrt? Power ∝ amplitude², so amplitude scaling = √(power scaling)

**Example values:**
- SNR = 5 dB → P_noise = P_signal / 3.16 → noise is only 3× quieter than signal (very noisy)
- SNR = 25 dB → P_noise = P_signal / 316 → noise is 316× quieter (barely audible)

**`+ 1e-10` (epsilon)**: Prevents division by zero for silent audio clips.

### ESC-50 Cache

```python
_ESC50_CACHE_SIZE = 300  # clips in RAM
# 300 clips × 5s × 22050Hz × 4 bytes (float32) ≈ 66 MB

def _build_esc50_audio_cache():
    paths = sorted(ESC50_AUDIO_DIR.glob('*.wav'))
    rng = random.Random(SEED)
    selected = rng.sample(paths, min(300, len(paths)))  # random subset, seeded
    cache = [load_audio(p) for p in selected]
    return cache
```

**Performance impact:**
- Without cache: disk read per noise sample → ~26 seconds/training sample
- With cache: RAM lookup → ~70ms/training sample (370× faster)
- Cache is built once on first noise request, reused forever
- `random.Random(SEED)`: deterministic selection → reproducible experiments

### Tempo Stretch (Fast Implementation)

```python
def apply_tempo_stretch_fast(audio, rate_min=0.85, rate_max=1.15):
    rate = random.uniform(0.85, 1.15)
    if abs(rate - 1.0) < 0.02: return audio  # skip if near 1.0
    orig_len = len(audio)
    resampled = librosa.resample(audio, orig_sr=SAMPLE_RATE,
                                  target_sr=int(SAMPLE_RATE * rate))
    # crop or pad back to original length
    if len(resampled) > orig_len:
        return resampled[:orig_len]
    return np.pad(resampled, (0, max(0, orig_len - len(resampled))))
```

**How resampling fakes tempo stretch:**
- If rate=1.1: resample at 110% of normal SR → audio "plays" 10% faster when played at normal SR
- Technically this changes both tempo AND pitch (unlike a true phase-vocoder stretch)
- But it's ~100× faster than `librosa.effects.time_stretch()` and "good enough" for augmentation
- The `src/augment.py` module uses the true phase-vocoder version; the notebook uses this fast version

### Volume Jitter

```python
def apply_volume_jitter(audio, min_db=-6, max_db=6):
    return audio * (10 ** (random.uniform(-6, 6) / 20))
```

**dB to linear amplitude conversion:**
- +6 dB → gain = 10^(6/20) = 10^0.3 = 2.0 (double amplitude)
- -6 dB → gain = 10^(-6/20) = 0.5 (half amplitude)
- 0 dB → gain = 1.0 (unchanged)
- Note: 20 in denominator (not 10) because dB for amplitude = 20·log₁₀(A₂/A₁)

### create_synthetic_mashup — Step by Step

```python
def create_synthetic_mashup(genre, stem_index, ...):
    # Step 1: Pick 2–4 random songs from same genre
    n_mix = random.randint(2, min(4, len(songs)))
    selected = random.sample(songs, n_mix)

    # Step 2: For each song, pick one stem (cycle through drums/vocals/bass/other)
    stem_pool = ['drums.wav', 'vocals.wav', 'bass.wav', 'other.wav']
    random.shuffle(stem_pool)

    mixed = np.zeros(target_len, dtype=np.float32)
    for i, song_dir in enumerate(selected):
        stem_name = stem_pool[i % 4]  # cycle: song0→drums, song1→vocals, etc.

        # Step 3: Load stem (slightly longer for random crop after tempo stretch)
        stem_audio = load_audio(stem_path, duration=10 * 1.3)  # 13s loaded

        # Step 4: Apply tempo stretch (70% chance if apply_tempo=True)
        if apply_tempo and random.random() > 0.3:
            stem_audio = apply_tempo_stretch_fast(stem_audio)

        # Step 5: Random crop to 10s, apply volume jitter
        stem_audio = random_crop(stem_audio, target_len)
        stem_audio = apply_volume_jitter(stem_audio)

        mixed += stem_audio

    # Step 6: Normalize to 0.9 peak (leave headroom)
    mx = np.abs(mixed).max()
    if mx > 0: mixed = mixed / mx * 0.9

    # Step 7: Add ESC-50 noise at random SNR (5–25 dB)
    mixed = add_random_noise(mixed)

    # Step 8: Re-normalize after noise addition
    mx = np.abs(mixed).max()
    if mx > 0: mixed = mixed / mx * 0.9

    return mixed.astype(np.float32), genre
```

**Why mix stems from the SAME genre?**
Because that's exactly how the test mashups are created. Mixing across genres would produce ambiguous training labels.

**Why cycle through stem types?**
Ensures diversity — rather than always mixing 4 drums tracks, we use drums from song A, vocals from song B, bass from song C, etc. This creates a more realistic mashup.

**Why load 1.3× the target duration?**
After tempo stretching (say rate=1.1), a 10s clip becomes 9.09s. If we loaded exactly 10s before stretching and then got 9.09s, we'd have to pad. Instead, load 13s → after stretch still >10s → random crop works cleanly.

**Why normalize to 0.9 (not 1.0)?**
Headroom: normalization to 1.0 makes the peak exactly at the boundary. Floating-point rounding errors could cause clipping. 0.9 leaves 10% safety margin.

---

## 4. Feature Extraction

### Mel Spectrogram (for CNN)

**Pipeline:**
```
audio (220500,)
  → STFT (N_FFT=2048, HOP=512) → complex spectrogram (1025, 431)
  → mel filterbank (128 triangular filters on mel scale) → mel power (128, 431)
  → power_to_db (log scale, top_db=80) → log-mel in dB (128, 431)
  → normalize to [0,1] → add channel → (1, 128, 431)
```

**Why the mel scale?**
The mel scale is a perceptually uniform frequency scale. Humans distinguish pitch differences better at low frequencies than high. Mel scale compresses the high-frequency range (above ~1kHz) logarithmically, allocating more frequency bins where pitch discrimination is most important. This gives the model a representation that aligns with human perception of music.

**Why log (dB scale)?**
Audio amplitude spans many orders of magnitude. Loud sounds can be 1000× louder than quiet ones. Without log, quiet sounds would barely register. The dB scale (logarithmic) compresses this dynamic range to ~80 dB, making all sounds visible in the spectrogram.

**`top_db=80`**: Values more than 80 dB below the maximum are clipped to -80 dB. Prevents extreme outliers.

**`ref=np.max`**: The dB scale is relative to the maximum energy in the clip. So the loudest frequency bin = 0 dB, all others are negative. This makes normalization to [0,1] natural.

**Normalize to [0,1]:**
```python
mel = (mel - mel.min()) / (mel.max() - mel.min())
# If max - min < 1e-6 (silent clip): return zeros
```
CNN benefits from inputs in a consistent range. Raw dB values would range [-80, 0] — normalization speeds up training and prevents gradient issues.

### MFCC Features (for Classical ML)

**What MFCCs capture:** The *spectral envelope* — the overall shape of the spectrum, which encodes timbre (instrument quality). Two notes at the same pitch but played on different instruments have the same fundamental frequency but different spectral envelopes.

**Computation:**
```
mel spectrogram (log scale) → DCT → take first N_MFCC coefficients
```

The DCT (Discrete Cosine Transform) decorrelates the mel filterbank outputs. The first few coefficients capture the "coarse shape" of the spectrum (low-cepstral order = slow spectral variation = overall timbre). Higher order = fine detail.

**Feature vector:**
```python
# 40 MFCCs × 6 statistics (mean, std for MFCC, delta, delta²) = 240 features
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # (40, T)
delta = librosa.feature.delta(mfcc)                      # (40, T) — 1st derivative
delta2 = librosa.feature.delta(mfcc, order=2)            # (40, T) — 2nd derivative
features = [mfcc.mean(1), mfcc.std(1), delta.mean(1), delta.std(1),
            delta2.mean(1), delta2.std(1)]  # 6 × 40 = 240 values
```

**Why delta and delta²?**
- MFCC mean/std captures static spectral shape
- Delta = how the spectrum changes over time (velocity)
- Delta² = rate of change of delta (acceleration)
- Together: captures spectral dynamics, not just average shape. Important for distinguishing genres with characteristic rhythmic patterns.

**Chroma features (24 features):**
```python
chroma = librosa.feature.chroma_stft(...)  # (12, T) — 12 pitch classes
# mean and std across time → 24 features
```
Pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B. Each bin shows energy at that pitch class regardless of octave. Captures harmonic content — useful because different genres have characteristic chord progressions (blues has blues scale, classical has complex harmony, reggae has specific chord patterns).

**Spectral features (24 features):**
```python
spectral_centroid   # (1, T) — "brightness": weighted mean of frequencies
spectral_bandwidth  # (1, T) — spread around centroid
spectral_rolloff    # (1, T) — frequency below which 85% of energy lies
spectral_contrast   # (7, T) — peak-to-valley ratio in each sub-band
rms                 # (1, T) — root-mean-square energy (loudness)
```
- **Centroid**: High centroid = bright/trebly (metal, rock). Low = bass-heavy (hip-hop, reggae)
- **Rolloff**: High = lots of high-frequency content (cymbals in metal/rock). Low = warm sounds (classical, jazz)
- **Contrast**: High contrast = clear peaks (tonal music). Low contrast = noisy/percussive

**Total: 240 + 24 + 24 = 288 features**

---

## 5. M1: Exploratory Data Analysis

### What Cell 6 does
```python
stem_index = build_stem_index()          # {genre: [list of Path objects]}
train_index, val_index = split_songs_train_val(stem_index)  # 15/genre held out
```

**`build_stem_index()`:**
```python
for genre in GENRES:
    genre_dir = GENRES_DIR / genre
    index[genre] = sorted([d for d in genre_dir.iterdir() if d.is_dir()])
# Returns: {'blues': [Path('blues/song0001'), ...], 'rock': [...], ...}
```

**`split_songs_train_val()`:**
```python
rng = random.Random(seed)   # seeded RNG for reproducibility
for genre, songs in stem_index.items():
    s = songs.copy()
    rng.shuffle(s)
    val_idx[genre] = s[:15]      # first 15 after shuffle = val
    train_idx[genre] = s[15:]    # rest = train
```
Using `random.Random(seed)` instead of `random.seed()` ensures this function doesn't pollute global random state.

### What Cell 7 visualizes
1. **Genre distribution bar chart** — confirms balance (roughly equal songs per genre)
2. **Mel spectrograms per genre** — visually shows genre differences:
   - Classical: smooth, harmonic structure, little percussion
   - Metal: wide frequency spread, high energy throughout
   - Hip-hop: strong low-frequency (bass) bands, rhythmic patterns
3. **Waveform comparison**: Clean stem vs synthetic mashup — shows the added noise/mixing

---

## 6. M2: Classical ML — MFCC + XGBoost

### Data Generation
```python
N_ML_TRAIN, N_ML_VAL = 2000, 500

def generate_ml_features(split_index, n_samples, name):
    n_per = n_samples // NUM_CLASSES   # 200 per genre for training
    for genre in sorted(split_index.keys()):
        for _ in range(n_per):
            audio, _ = create_synthetic_mashup(genre, split_index, add_noise=True)
            feats.append(extract_full_features(audio))  # 288-dim vector
            labels.append(GENRE_TO_IDX[genre])
```

**Why generate from train_index/val_index separately?**
The train_index songs are NOT in val_index. So validation samples come from unseen songs — realistic evaluation of generalization.

### Model
```python
scaler_ml = StandardScaler()
X_tr_s = scaler_ml.fit_transform(X_train_ml)  # fit on train only
X_va_s = scaler_ml.transform(X_val_ml)        # transform val with TRAIN statistics

xgb_ml = XGBClassifier(
    n_estimators=500,      # number of trees
    learning_rate=0.05,    # shrinkage per tree (eta)
    max_depth=6,           # max tree depth
    subsample=0.8,         # 80% random rows per tree (prevents overfitting)
    colsample_bytree=0.8,  # 80% random features per tree (like random forests)
    random_state=SEED,
    n_jobs=-1,             # use all CPU cores
    verbosity=0            # silent
)
xgb_ml.fit(X_tr_s, y_train_ml, eval_set=[(X_va_s, y_val_ml)], verbose=False)
```

**Why StandardScaler?**
Standardizes each feature to mean=0, std=1. XGBoost doesn't strictly need it (tree-based), but features like MFCCs range [-50, 50] while RMS ranges [0, 0.01]. Without scaling, the optimizer might put too much importance on high-magnitude features. Also, scaling often speeds up convergence.

**Why fit on train, transform on val?**
If you fit the scaler on both train+val, the validation statistics "leak" into the scaler — the model indirectly sees val distribution during training normalization. Always fit on train only.

**XGBoost internals:**
- Gradient boosting: build M trees sequentially, each fitting the residuals of the previous
- At each step: compute loss gradient → fit a tree to predict negative gradient → scale by `learning_rate` (shrinkage) → add to ensemble
- `max_depth=6`: trees can make up to 2^6 = 64 decisions per sample
- `subsample` + `colsample_bytree`: stochastic gradient boosting — adds regularization

**Result: ~0.45–0.62 Macro F1**

**Limitation:** The 288 hand-crafted features capture statistics (mean, std) of spectral properties but lose ALL temporal structure. The model can't learn "metal starts with a drum riff" or "reggae has offbeat guitar". It only sees "on average, metal has high spectral centroid".

---

## 7. M3: SimpleCNN on Mel Spectrograms

### Architecture (Cell 12)

```
Input:   (B, 1, 128, 431)    # (batch, channel, mel_bands, time_frames)

Block 1: Conv2d(1→32, 3×3, pad=1)  → BN → ReLU → MaxPool(2,2) → Dropout2d(0.1)
Output:  (B, 32, 64, 215)

Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.1)
Output:  (B, 64, 32, 107)

Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.2)
Output:  (B, 128, 16, 53)

Block 4: Conv2d(128→256, 3×3, pad=1) → BN → ReLU → AdaptiveAvgPool(1,1)
Output:  (B, 256, 1, 1)

Classifier:
  Flatten      → (B, 256)
  Dropout(0.3) → Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→10)
```

**Why 3×3 convolutions?**
3×3 kernels are the standard. Two 3×3 convolutions have the same receptive field as one 5×5 but with fewer parameters (2×9 = 18 params vs 25 params). They also introduce more non-linearity (two ReLUs vs one).

**Why progressively double channels (32→64→128→256)?**
Deeper layers have smaller spatial dimensions (due to pooling) but need to represent more complex patterns. More channels compensate for spatial resolution loss by increasing feature dimensionality.

**Why `padding=1` on 3×3 convolutions?**
Without padding, a 3×3 conv shrinks spatial dimensions by 2 (1 pixel from each side). With `padding=1`, output size = input size (before pooling). This preserves spatial resolution between conv and pooling, giving MaxPool full control over downsampling.

**Why AdaptiveAvgPool(1,1) at the end (not MaxPool)?**
- Collapses arbitrary spatial size to 1×1 — makes classifier input-shape-agnostic
- Averages all spatial locations — smooths out spatial noise, gives global representation
- Alternative: GlobalMaxPool would take the strongest activation anywhere; AvgPool gives mean activation

**Why Dropout2d (not regular Dropout)?**
In 2D feature maps, adjacent pixels in the same feature map are highly correlated. Dropping individual pixels doesn't help — the network can reconstruct them from neighbors. Dropout2d drops entire feature maps (all pixels in a channel), forcing the network to use multiple independent features. Better regularization for CNNs.

**Why BatchNorm after Conv (not before)?**
Standard practice (Conv → BN → ReLU). BN normalizes the pre-activation distribution, which helps ReLU work in a consistent range. If placed after ReLU, you normalize already-rectified values, which is less stable.

**Parameter count: ~422K**
```
Block1: 32×1×3×3 + 32 (bias) = 288+32 = 320 → BatchNorm: 64 → ~400 params
Block2: 64×32×3×3 = 18,432 → ~18K params
Block3: 128×64×3×3 = 73,728 → ~74K params
Block4: 256×128×3×3 = 294,912 → ~295K params
Classifier: 256×128 + 128×10 = 32,768 + 1,280 = ~34K params
Total: ~422K params
```

### LabelSmoothingCrossEntropy

```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)   # 10
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        # ε/(K-1) for non-target classes = 0.1/9 = 0.0111 each
        smooth.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
        # 1-ε for true class = 0.9

        # Cross-entropy: -sum(target * log_prob)
        return -(smooth * log_probs).sum(dim=-1).mean()
```

**Standard cross-entropy target: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]** (one-hot)

**Label-smoothed target: [0.011, 0.011, 0.011, 0.9, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011]**

**Why?** With one-hot targets, the model maximizes logit for the true class → probability approaches 1.0. The model becomes overconfident. With smoothing, even perfect prediction yields a slight loss (you can't perfectly fit ε/(K-1) for non-target classes). This prevents overconfidence and improves calibration.

### Training Loop (Cell 13)

```python
optimizer = torch.optim.AdamW(cnn.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
```

**Epoch loop:**
```python
for epoch in range(1, CNN_EPOCHS + 1):
    # --- Training ---
    cnn.train()  # enables dropout and BatchNorm training mode
    for xb, yb in train_dl:
        loss = cnn_criterion(cnn(xb), yb)
        cnn_opt.zero_grad()      # clear accumulated gradients
        loss.backward()          # compute gradients via backpropagation
        torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1.0)  # gradient clipping
        cnn_opt.step()           # update weights
    cnn_sched.step()             # update LR

    # --- Validation ---
    cnn.eval()   # disables dropout, uses BatchNorm eval mode (running stats)
    with torch.no_grad():        # disables gradient computation (saves memory)
        for xb, yb in val_dl:
            logits = cnn(xb)
            preds.extend(logits.argmax(1).cpu().numpy())

    vf1 = f1_score(trues, preds, average='macro')

    sync_and_clear(DEVICE)  # free GPU memory after each epoch
```

**`cnn.train()` vs `cnn.eval()`:**
- `train()`: Dropout drops random units. BatchNorm uses batch statistics (mean/var of current batch)
- `eval()`: Dropout passes all units unchanged. BatchNorm uses running statistics (accumulated from training). Critical to switch before validation — otherwise you're evaluating with dropout noise.

**Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
```
Computes the L2 norm of all gradients concatenated. If > 1.0, scales all gradients down proportionally. Prevents any single gradient update from being too large (exploding gradients), which can destabilize training.

**Result: ~0.55–0.93 Macro F1** (varies significantly with data size and device)

**Why does CNN beat XGBoost?**
- Learns spatial patterns in the 2D spectrogram directly — e.g., learns that bass frequencies have consistent patterns in reggae
- Captures temporal structure — a 3×3 conv looks at time+frequency jointly
- End-to-end learning: features are optimized for the classification task, not hand-designed
- With enough data, deep features are always more expressive than hand-crafted features

---

## 8. M4: AST Fine-Tuning

### What is the Audio Spectrogram Transformer (AST)?

**Pretrained model:** `MIT/ast-finetuned-audioset-10-10-0.4593`
- Pretrained on **AudioSet**: 2 million 10-second YouTube clips, 527 audio event classes
- Architecture: Vision Transformer (ViT) adapted for audio
- **86.2M total parameters**

**Key concept — Vision Transformer applied to audio:**

The core idea: treat audio the same way ViT treats images.

```
ViT:  Image  → patches → embed → transformer → classify
AST:  Audio  → mel spec → patches → embed → transformer → classify
```

**Step-by-step processing:**
1. Input: raw 16kHz audio waveform
2. `ASTFeatureExtractor` converts to mel spectrogram: **[1024 time, 128 mel bands]**
   - Note: this is larger than the CNN mel spec (431 frames) because AST uses finer time resolution
3. The [1024, 128] spectrogram is split into 16×16 patches (similar to how ViT splits images)
   - Number of patches: (1024/16) × (128/16) = 64 × 8 = **512 patches** + 2 learnable tokens [CLS, dist]
4. Each patch (16×16 = 256 values) → **Linear projection to 768-dim embedding**
5. Add **positional encodings** (learned) so the model knows where each patch came from
6. Pass through **12 Transformer encoder layers** (Multi-Head Self-Attention)
7. Take CLS token or mean of all tokens → **768-dim embedding**
8. Classification head: 768 → 527 (AudioSet classes)

**Fine-tuning for our task:** Replace classification head (527 → 10 genres), unfreeze top 4 encoder layers.

### Transformer Encoder Layer (Deep Explanation)

Each of the 12 layers contains:
```
Input: (512, 768)   [512 patches, 768-dim each]

Layer Norm → Multi-Head Self-Attention → residual add
Layer Norm → Feed-Forward Network → residual add

Output: (512, 768)
```

**Multi-Head Self-Attention:**
```python
# For each head h (there are 12 heads, each 64-dim):
Q_h = X @ W_Q_h    # Query: "what am I looking for?"
K_h = X @ W_K_h    # Key: "what do I contain?"
V_h = X @ W_V_h    # Value: "what should I output if attended to?"

Attn_h = softmax(Q_h @ K_h.T / sqrt(64))  # similarity scores
Head_h = Attn_h @ V_h

# Concatenate all heads, project
Output = concat(Head_1, ..., Head_12) @ W_O  # back to (512, 768)
```

**What does self-attention do?** Each patch attends to ALL other patches and weighs how much information to take from each. A "bass guitar" patch at one time point can attend to "bass guitar" patches at other times, allowing the model to capture long-range temporal dependencies that CNNs can't (CNNs only see local neighbors).

**Feed-Forward Network (FFN):**
```python
# Applied independently to each of 512 patches
x = LayerNorm(x)
x = Linear(768 → 3072) → GELU → Linear(3072 → 768)
# 4× expansion then contraction
```

**Residual connections ("+"):** Output of each sub-layer = sub-layer(x) + x. This allows gradients to flow directly backward through the "skip" connection, enabling training of very deep networks.

**Layer Normalization:** Unlike BatchNorm (normalizes across batch), LayerNorm normalizes across the feature dimension for each sample independently. Transformers use LayerNorm because they work with variable-length sequences where batch statistics are unstable.

### Selective Fine-Tuning (Cell 16)

```python
# Step 1: Freeze everything
for p in ast_model.parameters(): p.requires_grad = False

# Step 2: Unfreeze top 4 encoder layers (8, 9, 10, 11)
for i in range(8, 12):
    for p in ast_model.audio_spectrogram_transformer.encoder.layer[i].parameters():
        p.requires_grad = True

# Step 3: Unfreeze classifier + final LayerNorm
for p in ast_model.classifier.parameters(): p.requires_grad = True
if hasattr(ast_model.audio_spectrogram_transformer, 'layernorm'):
    for p in ast_model.audio_spectrogram_transformer.layernorm.parameters():
        p.requires_grad = True
```

**Why this specific selection?**
- Layers 0–7 (early layers): learn low-level features — spectral edges, simple patterns. These are universal across audio tasks. **Freeze** these — they're already perfect, no need to retrain.
- Layers 8–11 (late layers): learn task-specific patterns — musical phrases, rhythmic patterns. **Unfreeze** and adapt to our 10-genre task.
- Classifier: completely new head (original had 527 classes, we need 10). **Must train** from scratch.
- Final LayerNorm: normalizes the representation before the classifier. **Unfreeze** to adapt the normalization to our domain.

**Trainable parameters: ~4 of 86M total**

### Differential Learning Rates (Cell 17)

```python
clf_params = list(ast_model.classifier.parameters())
clf_ids = {id(p) for p in clf_params}
enc_params = [p for p in ast_model.parameters()
              if p.requires_grad and id(p) not in clf_ids]

optimizer = torch.optim.AdamW([
    {'params': enc_params, 'lr': 5e-5},   # fine-tune carefully
    {'params': clf_params, 'lr': 1e-3},   # train aggressively
], weight_decay=0.01)
```

**Why different LRs?**
The encoder layers (8–11) contain pretrained representations that are already good. A large LR would destroy them (called catastrophic forgetting). 5e-5 = gentle nudge.

The classifier head is random-initialized. It needs to learn quickly from scratch. 1e-3 = fast learning.

The `{id(p) for p in clf_params}` trick: `id()` gives the Python object identity (memory address), which uniquely identifies each parameter tensor. Used to separate encoder params from classifier params.

### Gradient Accumulation

```python
AST_GRAD_ACCUM = 2
# Effective batch = 8 × 2 = 16

for bi, (iv, labels) in enumerate(train_dl):
    loss = criterion(outputs, labels) / AST_GRAD_ACCUM   # scale loss
    loss.backward()    # accumulate gradients

    if (bi + 1) % AST_GRAD_ACCUM == 0:   # every 2 batches
        clip_grad_norm_(ast_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

**Why divide loss by GRAD_ACCUM?**
Without division, gradients would be 2× larger than if you used a real batch of 16. By dividing, `(loss_batch1/2).backward() + (loss_batch2/2).backward()` = same total gradient as `(loss_batch1 + loss_batch2)/2` with a single batch of 16. Mathematically equivalent to larger batch training.

**Why gradient accumulation?**
AST is 86M params, requires ~1GB GPU memory. With batch size 8, memory is manageable. Accumulating over 2 batches simulates batch size 16 without needing 2× memory.

### AMP (Automatic Mixed Precision)

```python
use_amp = DEVICE.type == 'cuda'   # only on CUDA (not MPS)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for iv, labels in train_dl:
    with torch.cuda.amp.autocast(enabled=use_amp):
        outputs = ast_model(input_values=iv)   # runs in float16
        loss = criterion(outputs.logits, labels) / AST_GRAD_ACCUM

    scaler.scale(loss).backward()   # scale up gradients before backward

    if (bi + 1) % AST_GRAD_ACCUM == 0:
        scaler.unscale_(optimizer)  # scale gradients back down
        clip_grad_norm_(ast_model.parameters(), 1.0)
        scaler.step(optimizer)      # update weights
        scaler.update()             # adjust scale factor
```

**float16 vs float32:**
- float16: 2 bytes per value, range ~[6e-5, 65504]
- float32: 4 bytes per value, range ~[1e-38, 3.4e38]

**AMP strategy:**
1. Forward pass in float16 → 2× memory, faster on modern GPUs
2. Gradients in float16 can underflow (become 0 if too small for float16 range)
3. GradScaler multiplies loss by a large factor (e.g., 65536) before backward → gradients stay in float16 range
4. Before optimizer step, divide gradients back by scale factor → correct float32 update
5. If gradient overflow detected → skip step and reduce scale factor

**Why only CUDA?** MPS (Apple Silicon) doesn't support `torch.cuda.amp` — it's CUDA-specific. MPS has its own (limited) mixed precision support.

### ASTSyntheticDataset (Cell 16)

```python
class ASTSyntheticDataset(Dataset):
    def __init__(self, stem_idx, n_samples, fe, add_noise=True):
        self.n_samples = n_samples
        self.genres = sorted(stem_idx.keys())

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        genre = self.genres[idx % len(self.genres)]  # balanced: each genre equally
        audio, _ = create_synthetic_mashup(genre, self.stem_idx, ...)
        audio_16k = librosa.resample(audio, orig_sr=22050, target_sr=16000)
        inputs = self.fe(audio_16k, sampling_rate=16000, return_tensors='np')
        return torch.FloatTensor(inputs['input_values'].squeeze(0)), GENRE_TO_IDX[genre]
```

**Online augmentation**: Every call to `__getitem__` generates a **brand new synthetic mashup**. This means in 12 epochs, the model never sees the exact same sample twice. The dataset is effectively infinite.

**Why `idx % len(self.genres)` for balanced sampling?**
Ensures each genre gets equal representation: sample 0→blues, 1→classical, ..., 9→rock, 10→blues, 11→classical, etc.

**Why resample to 16kHz?**
The ASTFeatureExtractor was trained with 16kHz audio. Its internal filterbank parameters are calibrated for this sample rate. Using 22kHz would cause frequency axis misalignment.

---

## 9. M4 Fallback: Frozen AST + XGBoost

Used when internet is unavailable (can't download pretrained model fresh):

```python
if not INTERNET_AVAILABLE:
    ast_base = ASTModel.from_pretrained(AST_MODEL_NAME)  # base model, no classifier
    ast_base.eval()
    for p in ast_base.parameters(): p.requires_grad = False  # frozen

    @torch.no_grad()
    def extract_ast_emb(audio_16k):
        inputs = ast_fe(audio_16k, sampling_rate=16000, return_tensors='pt')
        out = ast_base(**inputs)
        return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        # Shape: (768,) — mean of all patch embeddings
```

**`last_hidden_state.mean(dim=1)`:**
- `last_hidden_state`: (1, 512, 768) — output of final transformer layer for all patches
- `.mean(dim=1)`: average across 512 patch dimension → (1, 768)
- `.squeeze(0)`: remove batch dim → (768,)

This is the "meaning" of the audio clip in AST's representation space.

**Then XGBoost on these embeddings:**
```python
xgb_fb = XGBClassifier(
    n_estimators=1000,    # more trees for higher-dimensional input
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1,        # L1 regularization (lasso-like)
    reg_lambda=1.0,       # L2 regularization (ridge-like)
    min_child_weight=3,   # minimum samples per leaf (prevents overfitting)
)
xgb_fb.fit(X_tr_s_fb, y_tr_ast, eval_set=[(X_va_s_fb, y_va_ast)], verbose=False)
```

**Result: ~0.78 Macro F1**

**Why does 768-dim frozen embedding + XGBoost beat 288 MFCC + XGBoost so much?**
AST was pretrained on 2 million audio clips from YouTube, learning to distinguish 527 types of sound. Its internal representations encode rich acoustic semantics — instruments, rhythm, texture, melody patterns. The 768-dim embedding captures all of this. Our 288 MFCC features were designed decades ago for speech recognition and miss most of this richness.

---

## 10. M5: Inference & Submission

### Test-Time Augmentation (TTA)

```python
N_CROPS_TEST = 10  # 10 overlapping crops per test file

for fi, (id_, fname) in enumerate(zip(ids, filenames)):
    full_audio = load_audio(str(path), sr=SAMPLE_RATE)
    crop_inputs = []

    for ci in range(N_CROPS_TEST):
        # Evenly-spaced start positions
        ms = len(full_audio) - target_len   # max start position
        start = int(ci * ms / max(N_CROPS_TEST - 1, 1))
        chunk = full_audio[start:start + target_len]

        # Resample to 16kHz for AST
        c16 = librosa.resample(chunk, orig_sr=22050, target_sr=16000)
        inp = ast_fe(c16, sampling_rate=16000, return_tensors='pt')
        crop_inputs.append(inp['input_values'])

    # Batch all 10 crops together for efficiency
    batched = torch.cat(crop_inputs, dim=0).to(DEVICE)  # (10, 1024, 128)
    with torch.no_grad():
        out = ast_model(input_values=batched)
        probs = F.softmax(out.logits, dim=-1).cpu().numpy()  # (10, 10)

    test_probs.append(probs.mean(axis=0))  # average → (10,)
```

**Why evenly-spaced (not random)?**
Deterministic TTA: running the same file twice gives the same prediction. Reproducibility is important for submission.

**Why 10 crops?**
More crops = better approximation of the full file's content. 10 is a balance between accuracy and inference speed.

**Why average probabilities (not logits)?**
Probabilities are in [0,1] and sum to 1. Averaging probabilities = Naive Bayes-style ensemble. Averaging logits is also valid but probabilities are more interpretable. In practice, both give similar results.

**Why `F.softmax(out.logits, dim=-1)` and not `F.softmax(out.logits, dim=0)`?**
`dim=-1` applies softmax over the last dimension (class dimension, size 10). `dim=0` would apply over the batch dimension — wrong.

### Submission Format
```python
test_genres = [IDX_TO_GENRE[int(p)] for p in y_test_pred]
submission = pd.DataFrame({'id': ids, 'genre': test_genres})
submission['id'] = submission['id'].astype(str).str.zfill(4)  # zero-pad to 4 digits
submission.to_csv(sub_path, index=False)
```

Output:
```csv
id,genre
0001,blues
0002,rock
...
```

---

## 11. Deep Concept Explanations

### What is BatchNorm?

```python
nn.BatchNorm2d(32)
```

**Problem it solves:** During training, as weights update, the distribution of each layer's inputs keeps shifting (Internal Covariate Shift). This forces later layers to constantly adapt.

**How it works (training):**
```
For each channel c, across all (B, H, W) positions in the batch:
  μ_c = mean of all values in channel c
  σ_c = std of all values in channel c
  x_norm = (x - μ_c) / (σ_c + ε)
  output = γ_c × x_norm + β_c   # learnable scale and shift
```

**Training vs inference:**
- Training: use batch statistics (mean/std of current batch)
- Inference: use **running mean/var** accumulated during training (exponential moving average)
- This is why `model.eval()` is critical — switches from batch stats to running stats

**Benefits:**
- Higher learning rates → faster convergence
- Reduces sensitivity to weight initialization
- Acts as regularization (batch statistics add noise)
- Stabilizes gradient flow in deep networks

### What is Dropout?

```python
nn.Dropout(0.3)      # 30% of neurons zeroed
nn.Dropout2d(0.1)    # 10% of feature maps zeroed
```

**How it works:** During training, randomly set 30% of inputs to 0 (then scale remaining by 1/0.7 to maintain expected value).

**Why it works:** Forces the network to develop redundant representations. No single neuron can be relied upon → the network learns multiple independent ways to solve the problem → better generalization.

**Training vs inference:** `model.train()` applies dropout. `model.eval()` keeps all neurons active (no dropout). Inference uses the full network.

### What is AdamW?

**Standard SGD:** `w = w - lr × gradient`

**Adam (Adaptive Moment Estimation):**
```
m = β₁ × m + (1-β₁) × g      # exponential moving average of gradients (1st moment)
v = β₂ × v + (1-β₂) × g²     # exponential moving average of gradient² (2nd moment)
m̂ = m / (1 - β₁^t)            # bias correction
v̂ = v / (1 - β₂^t)
w = w - lr × m̂ / (√v̂ + ε)    # adaptive per-parameter learning rate
```

Each parameter gets its own effective learning rate based on gradient history. Parameters with large, consistent gradients get smaller updates (already moving in right direction). Parameters with small/noisy gradients get relatively larger updates.

**AdamW fix:** In Adam, weight decay is added to the gradient: `g' = g + λw`, then used in the moment estimates. This means weight decay is scaled by the gradient magnitude (non-uniform regularization). AdamW decouples: update Adam normally, then separately apply weight decay: `w = (1 - lr × λ) × w - lr × m̂/√v̂`. This is the mathematically correct L2 regularization.

### What is Cosine Annealing?

```python
scheduler = CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS, eta_min=1e-7)
```

Learning rate follows: `lr_t = eta_min + (lr_max - eta_min)/2 × (1 + cos(π × t/T_max))`

- t=0: lr = lr_max (full learning rate)
- t=T_max/2: lr = (lr_max + eta_min)/2
- t=T_max: lr ≈ eta_min (near zero)

**Why better than step decay?**
Step decay: LR suddenly drops by 10× at specific epochs. This can destabilize training (the optimizer was calibrated to a certain LR; sudden change requires readjustment). Cosine annealing: smooth monotonic decrease, the optimizer can gradually settle into finer minima as LR decreases.

### How does Macro F1 Work?

```python
f1_score(y_true, y_pred, average='macro')
```

For each class:
```
precision_i = TP_i / (TP_i + FP_i)   # of all predicted "blues", what fraction is correct?
recall_i = TP_i / (TP_i + FN_i)      # of all actual "blues", what fraction did we find?
F1_i = 2 × precision_i × recall_i / (precision_i + recall_i)
```

Macro F1 = mean(F1_i for i in 0..9)

**Why harmonic mean (F1) and not arithmetic mean of precision/recall?**
Harmonic mean penalizes imbalance. If precision=1.0 and recall=0.0, arithmetic mean = 0.5 but harmonic mean = 0. F1 is low unless BOTH precision and recall are reasonable.

### What is Transfer Learning?

Training a deep neural network from scratch requires massive data (millions of examples). Most tasks don't have this.

**Transfer learning:** Use weights pretrained on a large dataset as initialization for a new task.

**Levels:**
1. **Feature extraction (frozen)**: Use pretrained model as fixed feature extractor. Only train a new classifier on top. (Our M4 fallback)
2. **Fine-tuning**: Unfreeze some (or all) pretrained layers and train with a small LR. Adapts representations to new domain. (Our M4 fine-tuning)
3. **Full training**: Train everything from scratch (our M3 SimpleCNN)

**Why does it work?** Early layers of neural networks learn universal features (edges, textures for images; spectral patterns, rhythm for audio) that transfer across tasks. Only later layers need task-specific adaptation.

---

## 12. Implementation Change Scenarios

These are the types of modifications the examiner could ask you to implement live.

---

### Change 1: Add a New Model (ResidualBlock / AudioResNet)

**Q: "Implement a residual connection in your CNN"**

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))   # ← residual connection
```

**Use it in SimpleCNN by replacing a block:**
```python
# Instead of:
nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()

# Use:
ResidualBlock(64)
```

**Why residual connections?** As networks get deeper, gradient vanishes (multiplied many small numbers through chain rule). Residual connection provides a "shortcut" path: gradient can flow directly back through the skip connection without passing through the conv layers. Enables training of 100+ layer networks.

---

### Change 2: Change the Optimizer

**Q: "Try SGD with momentum instead of AdamW"**

```python
# Replace:
cnn_opt = torch.optim.AdamW(cnn.parameters(), lr=3e-4, weight_decay=1e-4)

# With:
cnn_opt = torch.optim.SGD(
    cnn.parameters(),
    lr=0.01,              # SGD needs MUCH higher LR than Adam
    momentum=0.9,         # accumulate gradient direction (0.9 = standard)
    weight_decay=1e-4,    # L2 regularization
    nesterov=True         # look-ahead gradient (usually better than vanilla momentum)
)
```

**Why different LR?** Adam normalizes by gradient magnitude → effective LR is small. SGD uses raw gradient → needs a larger base LR (0.01 vs 3e-4).

**Momentum formula:**
```
v = momentum × v + gradient
w = w - lr × v
```

---

### Change 3: Change the Learning Rate Scheduler

**Q: "Use ReduceLROnPlateau instead of CosineAnnealingLR"**

```python
# Replace:
cnn_sched = CosineAnnealingLR(cnn_opt, T_max=CNN_EPOCHS)
# and: cnn_sched.step()

# With:
cnn_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    cnn_opt,
    mode='max',        # we want val_f1 to increase
    factor=0.5,        # multiply LR by 0.5 when plateau
    patience=3,        # wait 3 epochs before reducing
    min_lr=1e-6,
    verbose=True
)
# and call:
cnn_sched.step(vf1)   # pass the metric (not just step())
```

**When to use:** ReduceLROnPlateau is adaptive — only reduces LR when the metric stops improving. Useful when you don't know how many epochs to train.

---

### Change 4: Add Focal Loss

**Q: "Implement Focal Loss to focus on hard examples"**

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma        # focusing parameter (2.0 is standard)
        self.alpha = alpha        # class weight (None = uniform)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        # ce_loss = -log(pt) where pt = probability of correct class
        pt = torch.exp(-ce_loss)                    # pt = probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # downweight easy examples
        return focal_loss.mean()
```

**Intuition:** If a sample is correctly classified with probability 0.9 (easy), `(1-0.9)^2 = 0.01` → nearly ignored. If misclassified with probability 0.1 (hard), `(1-0.1)^2 = 0.81` → emphasized. Forces model to focus on hard examples.

**Replace in training:**
```python
cnn_criterion = FocalLoss(gamma=2.0)   # instead of LabelSmoothingCrossEntropy
```

---

### Change 5: Add Mixup Augmentation

**Q: "Implement mixup data augmentation"**

```python
def mixup_batch(x, y, alpha=0.4):
    """
    Interpolate between two random samples in the batch.
    x: (B, C, H, W) tensor
    y: (B,) label tensor
    """
    lam = np.random.beta(alpha, alpha)   # sample mixing coefficient
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)   # random permutation

    x_mixed = lam * x + (1 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return x_mixed, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss = weighted combination of both labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

**In the training loop:**
```python
for xb, yb in train_dl:
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    xb_mix, ya, yb_, lam = mixup_batch(xb, yb)
    loss = mixup_criterion(cnn_criterion, cnn(xb_mix), ya, yb_, lam)
    # rest same...
```

---

### Change 6: Add SpecAugment

**Q: "Add SpecAugment to the CNN training" (and explain why it's disabled)**

```python
def spec_augment(spec, time_mask_max=40, freq_mask_max=20, n_time=2, n_freq=2):
    """
    spec: (C, F, T) tensor/array
    Randomly zero out time and frequency bands.
    """
    spec = spec.clone() if isinstance(spec, torch.Tensor) else spec.copy()
    _, n_freq_bins, n_time_bins = spec.shape
    mask_val = spec.min()

    for _ in range(n_time):
        width = np.random.randint(1, min(time_mask_max, n_time_bins))
        start = np.random.randint(0, n_time_bins - width)
        spec[:, :, start:start+width] = mask_val  # zero out time band

    for _ in range(n_freq):
        width = np.random.randint(1, min(freq_mask_max, n_freq_bins))
        start = np.random.randint(0, n_freq_bins - width)
        spec[:, start:start+width, :] = mask_val  # zero out freq band

    return spec
```

**⚠️ Warning — why SpecAugment was DISABLED in this project:**
During training with SpecAugment, BatchNorm computes running_mean and running_var from masked spectrograms (with many zeros). At test time, the spectrograms are NOT masked. The BatchNorm statistics don't match the test distribution → performance degrades. To use SpecAugment safely: apply it ONLY to the input, AFTER BatchNorm has seen the unmasked stats (or use InstanceNorm/LayerNorm instead of BatchNorm).

---

### Change 7: Change Number of Frozen AST Layers

**Q: "Freeze all layers except the last 2 encoder layers"**

```python
# Freeze all
for p in ast_model.parameters(): p.requires_grad = False

# Unfreeze only layers 10 and 11 (instead of 8-11)
for i in range(10, 12):
    for p in ast_model.audio_spectrogram_transformer.encoder.layer[i].parameters():
        p.requires_grad = True

# Still need classifier
for p in ast_model.classifier.parameters(): p.requires_grad = True
```

**Trade-off:** Fewer trainable layers = less risk of overfitting on small data, faster training, but less task adaptation.

---

### Change 8: Change Validation Metric for Early Stopping

**Q: "Stop training based on validation loss instead of F1"**

In the training loop, change:
```python
# Instead of tracking: if vf1 > best_f1: save
# Track: if val_loss < best_val_loss: save

best_val_loss = float('inf')
patience_counter = 0

val_loss = val_loss_sum / vn
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    torch.save(model.state_dict(), 'best_model.pt')
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch}')
        break
```

**Note:** F1-based early stopping is usually better than loss-based for classification tasks because F1 directly measures what you care about.

---

### Change 9: Implement Your Own F1 Score

**Q: "Compute Macro F1 without sklearn"**

```python
def manual_macro_f1(y_true, y_pred, n_classes=10):
    f1_scores = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return sum(f1_scores) / n_classes  # macro average
```

---

### Change 10: Add Weight Initialization

**Q: "Add custom weight initialization to SimpleCNN"**

```python
def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

# Apply in __init__:
self.apply(_init_weights)
```

**Kaiming (He) initialization:** Designed for ReLU networks. `std = sqrt(2 / fan_out)` for `mode='fan_out'`. Ensures that variance of outputs = variance of inputs through a ReLU layer (prevents vanishing/exploding activations at initialization).

**Xavier initialization:** Designed for tanh/sigmoid networks. `std = sqrt(2 / (fan_in + fan_out))`. Maintains variance through linear layers.

---

### Change 11: Add L1 Regularization to XGBoost

**Q: "Add L1 regularization to XGBoost"**

```python
# Replace:
xgb_ml = XGBClassifier(n_estimators=500, ...)

# With:
xgb_ml = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # ← L1 regularization (lasso) - promotes sparsity
    reg_lambda=1.0,     # ← L2 regularization (ridge) - shrinks all weights
    random_state=SEED,
)
```

**L1 vs L2:**
- L1 (reg_alpha): penalizes sum of absolute weights → drives some weights exactly to 0 → feature selection
- L2 (reg_lambda): penalizes sum of squared weights → shrinks all weights proportionally → smoother model

---

### Change 12: Save and Load Model Checkpoint

**Q: "How do you save the best model and reload it?"**

```python
# Save:
if vf1 > cnn_best_f1:
    cnn_best_f1 = vf1
    torch.save({
        'epoch': epoch,
        'model_state_dict': cnn.state_dict(),
        'optimizer_state_dict': cnn_opt.state_dict(),
        'val_f1': vf1,
    }, 'best_cnn.pt')

# Load:
checkpoint = torch.load('best_cnn.pt', map_location=DEVICE)
cnn = SimpleCNN().to(DEVICE)
cnn.load_state_dict(checkpoint['model_state_dict'])
cnn.eval()   # set to eval mode for inference
print(f"Loaded epoch {checkpoint['epoch']}, val F1: {checkpoint['val_f1']:.4f}")
```

**Why `map_location=DEVICE`?** The checkpoint was saved on GPU/MPS. When loading on CPU (or different GPU), `map_location` redirects tensors to the target device. Without it, you'd get a CUDA-not-available error when loading GPU checkpoints on CPU.

---

### Change 13: Different Number of TTA Crops

**Q: "Change TTA to use 5 crops instead of 10"**

In Cell 1 config:
```python
N_CROPS_TEST = 5   # was 10
```

Or modify the inference loop directly:
```python
for ci in range(5):    # instead of N_CROPS_TEST
    ms = len(full_audio) - target_len
    start = int(ci * ms / max(4, 1))   # max(N_CROPS-1, 1) = max(5-1,1) = 4
    ...
```

**Trade-off:** Fewer crops = faster inference, but less averaging → slightly noisier predictions.

---

### Change 14: Add Per-Class Accuracy Reporting

**Q: "Print per-class accuracy alongside F1"**

```python
from sklearn.metrics import classification_report, confusion_matrix

# After collecting preds and trues:
print(classification_report(
    trues, preds,
    target_names=GENRES,
    digits=3
))
# Shows: precision, recall, f1-score, support per class

# Confusion matrix:
cm = confusion_matrix(trues, preds)
# cm[i,j] = number of samples with true class i predicted as class j
# Diagonal = correct predictions
```

---

### Change 15: Change Batch Size

**Q: "What happens if you change batch size from 32 to 64?"**

```python
train_dl = DataLoader(SpecDataset(train_specs, train_labels),
                      batch_size=64,   # was 32
                      shuffle=True)
```

**Effects:**
- Larger batch → more stable gradient estimates → can use larger LR (linear scaling rule: `new_lr = old_lr × (new_bs / old_bs)`)
- Larger batch → uses more GPU memory
- BatchNorm statistics: with batch=64, mean/var estimated over more samples → more accurate → more stable training
- Practical rule: double batch size, double learning rate: `lr = 3e-4 × (64/32) = 6e-4`

---

## 13. Numbers to Remember

| Item | Value |
|---|---|
| **Genres** | 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) |
| **Stems per song** | 4 (drums, vocals, bass, other) |
| **Val songs per genre** | 15 |
| **Sample rate (CNN)** | 22050 Hz |
| **Sample rate (AST)** | 16000 Hz |
| **Chunk duration** | 10 seconds |
| **Chunk samples** | 220,500 |
| **N_MELS** | 128 |
| **N_FFT** | 2048 |
| **HOP_LENGTH** | 512 |
| **N_FRAMES** | 431 |
| **CNN input shape** | (1, 128, 431) |
| **AST input shape** | (1024, 128) |
| **MFCC features** | 240 (40×6: MFCC+delta+delta², mean+std) |
| **Chroma features** | 24 (12×2) |
| **Spectral features** | 24 |
| **Total ML features** | 288 |
| **SimpleCNN params** | ~422K |
| **AST total params** | 86.2M |
| **AST trainable params** | ~4M (layers 8–11 + classifier) |
| **SNR range** | 5–25 dB |
| **Tempo range** | ±15% (0.85–1.15×) |
| **Volume jitter** | ±6 dB |
| **ESC-50 cache size** | 300 clips (~66 MB) |
| **CNN epochs** | 10 |
| **AST epochs** | 12 |
| **AST batch size** | 8 |
| **AST grad accumulation** | 2 (effective batch = 16) |
| **AST samples/epoch** | 4000 |
| **TTA crops** | 10 |
| **Random baseline F1** | ~0.10 |
| **M2 XGBoost F1** | ~0.45–0.62 |
| **M3 CNN F1** | ~0.55–0.93 |
| **M4 Frozen AST+XGB F1** | ~0.78 |
| **M4 Fine-tuned AST F1** | **0.93** |

---

## 14. Key Design Decisions Q&A

**Q: Why synthetic mashup augmentation?**
Training data is clean stems; test data is noisy mashups — different distributions (domain shift). Training on clean stems would cause poor test performance. By synthesizing the same mixing+noise process during training, we bridge this gap. This is domain adaptation via data augmentation.

**Q: Why ESC-50 for noise, not white noise?**
ESC-50 contains 2000 real-world environmental sounds (rain, traffic, animals, etc.) — the same type of noise in real recordings. White/Gaussian noise has uniform spectral content and doesn't represent real conditions well. Real environmental noise has spectral peaks, temporal patterns, and correlations that challenge the model in realistic ways.

**Q: Why song-level (not sample-level) train/val split?**
If split sample-level, training samples can come from the same song as validation samples. The model could memorize song-specific characteristics (particular instrument, recording quality) and appear to generalize but actually be overfitting. Song-level split: validation songs were never seen during training — true generalization test.

**Q: Why disable SpecAugment for CNN?**
SpecAugment zeros out random time/frequency bands during training. BatchNorm's running_mean and running_var accumulate from MASKED spectrograms (with many zeros). At test time, spectrograms are unmasked → different distribution → BatchNorm normalizes incorrectly → degraded performance. This was empirically verified: val F1 dropped with SpecAugment enabled.

**Q: Why freeze layers 0–7 and fine-tune 8–11 in AST?**
Early transformer layers (0–3): detect low-level features (spectral edges, onset patterns) — universal across audio tasks. Mid layers (4–7): detect intermediate patterns — still fairly general. Late layers (8–11): task-specific semantic representations. We fine-tune only where task-specific adaptation is needed. Freezing early layers: (1) prevents catastrophic forgetting of useful generic features, (2) reduces trainable parameters (training faster, less data needed), (3) acts as implicit regularization.

**Q: Why mean pool AST hidden states instead of using CLS token?**
Both are valid. For audio classification, mean pooling over all patch embeddings often works better because the model needs to integrate information across ALL time positions to classify genre (you can't classify music genre from a single 16ms patch). CLS token was designed for BERT where a single "summarization" token is trained explicitly. AudioSet pretraining of AST also used mean pooling.

**Q: Why resample audio to 16kHz for AST?**
The ASTFeatureExtractor's internal mel filterbank was calibrated for 16kHz input. It uses specific FFT sizes and hop lengths tuned for this sample rate. If you input 22050Hz audio without resampling, the frequency axis of the mel spectrogram would be compressed differently than what the model learned during pretraining → misaligned representation → degraded performance.

**Q: Why normalize audio to 0.9 (not 1.0) after mixing?**
After mixing N stems with volume jitter and adding noise, the peak amplitude can reach >1.0. Before normalization: clip the peak to exactly 1.0, but floating-point arithmetic may still push values slightly above 1.0 after further processing. 0.9 provides a 10% safety margin against accidental clipping.

**Q: Why weight_decay=0.01 for AST but 1e-4 for CNN?**
AST has 86M parameters — much higher risk of overfitting with limited training data (~4000 samples/epoch). Stronger weight decay (0.01 vs 0.0001) provides more regularization. CNN has 422K params — less prone to overfitting, weaker regularization sufficient.

**Q: What if a stem file doesn't exist?**
`get_stem_path()` first tries `other.wav`, then falls back to `others.wav`. If neither exists, returns None. `create_synthetic_mashup()` then tries other stem types for that song. If no stems found for a song, it's skipped (n_mixed tracks that). If ALL songs fail, returns silence + genre label (shouldn't happen).

**Q: How does `idx % len(self.genres)` ensure balanced classes in ASTSyntheticDataset?**
With 10 genres and dataset len=4000: idx 0→blues, 1→classical, ..., 9→rock, 10→blues, ... → 400 samples per genre. Perfectly balanced. The DataLoader shuffles indices before passing to `__getitem__`, so batches see mixed genres.

**Q: Why gradient accumulation specifically 2 for AST?**
Memory constraint: AST (86M params) with batch=8 uses ~3-4GB GPU memory. Batch=16 would need ~8GB which exceeds T4 GPU memory on Kaggle free tier (16GB, but need headroom for activations/gradients). Accumulation=2 gives effective batch=16 while keeping peak memory at 8-sample level.

**Q: Why `pin_memory=True` in AST DataLoader?**
```python
ast_train_dl = DataLoader(ast_train_ds, batch_size=8, pin_memory=True)
```
Pinned (page-locked) CPU memory can be transferred to GPU via DMA (direct memory access) asynchronously while the GPU is computing. Speeds up CPU→GPU data transfer by ~2×. Only relevant when using CUDA GPU.

**Q: What does `ignore_mismatched_sizes=True` do in AST loading?**
```python
ASTForAudioClassification.from_pretrained(
    AST_MODEL_NAME, num_labels=10, ignore_mismatched_sizes=True)
```
The pretrained AST classifier head has shape [768 → 527] (AudioSet classes). We want [768 → 10] (our 10 genres). Without this flag, HuggingFace would raise an error when loading the checkpoint (size mismatch). With it: loads all matching weights, re-initializes only the mismatched layers (classifier head) randomly.

---

*This guide covers every cell in `Submitted_0.93.ipynb`, all design decisions, deep conceptual explanations, and 15 implementation change scenarios with complete working code.*
