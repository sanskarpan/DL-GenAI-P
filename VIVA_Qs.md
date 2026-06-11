# Messy Mashup — Viva Questions & Answers

> Every question is based on actual examiner patterns + deep analysis of `Submitted_0.93.ipynb`.
> Each answer includes the **exact code line being questioned**, the explanation, and working code where needed.

---

## Quick Index

| Category | Questions |
|---|---|
| [A. EDA Walkthrough](#a-eda-walkthrough) | Walk me through your EDA, why these lines |
| [B. Audio Fundamentals](#b-audio-fundamentals) | SNR, RMS, audio gain, normalization, tempo |
| [C. Feature Extraction](#c-feature-extraction) | MFCC, delta, delta2, chroma, mel spec, log |
| [D. CNN Architecture](#d-cnn-architecture) | BatchNorm, MaxPool, AdaptiveAvgPool, channels |
| [E. Training Loop](#e-training-loop) | Full loop, gradient clipping, train/eval mode |
| [F. Loss Functions](#f-loss-functions) | Label smoothing, focal loss, CE, why chosen |
| [G. Optimizers & Schedulers](#g-optimizers--schedulers) | AdamW, cosine annealing, LR choices |
| [H. AST Model](#h-ast-model) | Architecture, fine-tuning, why AST, import/load |
| [I. W&B & Experiments](#i-wb--experiments) | Show logs, what was tracked, why |
| [J. Coding Tasks](#j-coding-tasks) | LSTM, CNN from scratch, training loop, numpy |
| [K. General Theory](#k-general-theory) | Dropout, BatchNorm, pooling, regularization |

---

## A. EDA Walkthrough

### Q1: Walk me through your EDA. What did you find?

**Cell 6 code being asked about:**
```python
stem_index = build_stem_index()
train_index, val_index = split_songs_train_val(stem_index)
```

**Answer:**

EDA had 4 components:

**1. Dataset structure check:**
```
10 genres × 100 songs/genre = 1,000 total songs
Each song has 4 stems: drums.wav, vocals.wav, bass.wav, other.wav
Split: 85 train songs + 15 val songs per genre (song-level split)
Test set: 3,020 pre-made mashup files
```

**2. Genre distribution:** Perfectly balanced — 100 songs per genre. This means macro F1 ≈ weighted F1. Random baseline = 1/10 = 0.10.

**3. Mel spectrogram visualization:** Plotted synthetic mashups for all 10 genres side-by-side. Key visual differences:
- Classical: smooth harmonic bands, few percussion bursts
- Metal: dense energy across all frequencies, high upper bands
- Hip-hop: strong low-frequency (bass) bands, rhythmic peaks
- Reggae: offbeat pattern visible as rhythmic gaps in spectral energy

**4. Waveform comparison:** Showed a clean stem vs a synthetic mashup to demonstrate the domain shift — the mashup has visible noise floor and amplitude variation from mixing.

---

### Q2: Why did you use `build_stem_index()`? What does it return?

```python
def build_stem_index():
    index = {}
    for genre in GENRES:
        genre_dir = GENRES_DIR / genre
        index[genre] = sorted([d for d in genre_dir.iterdir() if d.is_dir()])
    return index
# Returns: {'blues': [Path('blues/song0001'), Path('blues/song0002'), ...], ...}
```

**Answer:** It builds a lookup dictionary mapping each genre to a list of song directories. This is cached and reused throughout training — instead of scanning the filesystem every time you need a song, you look it up in O(1). It also ensures consistent ordering (`sorted`) across runs for reproducibility.

---

### Q3: Why is the split song-level (15 songs per genre for validation) and not sample-level?

**Answer:** If you split at the sample/mashup level, training and validation samples can come from the SAME song. The model could memorise song-specific characteristics — recording quality, specific instrument timbre, room acoustics — and appear to generalise but actually be overfitting to the specific recording.

Song-level split: **every mashup in the validation set is derived from songs the model has never seen** — a true generalization test.

```python
def split_songs_train_val(stem_index, val_per_genre=15, seed=42):
    rng = random.Random(seed)   # separate RNG — doesn't affect global state
    for genre, songs in stem_index.items():
        s = songs.copy()
        rng.shuffle(s)              # shuffle once
        val_idx[genre] = s[:15]     # first 15 = val
        train_idx[genre] = s[15:]   # rest = train
```

Why `random.Random(seed)` and not `random.seed()`? A separate `Random` instance doesn't contaminate the global random state. If you used `random.seed()`, subsequent calls to `random.random()` elsewhere in the code would be affected.

---

### Q4: What does `create_synthetic_mashup` actually do? Walk through it line by line.

```python
def create_synthetic_mashup(genre, stem_index, target_duration=10, n_mix=None,
                            add_noise=True, apply_volume=True, apply_tempo=False):
```

**Line-by-line:**

1. **`n_mix = random.randint(2, min(4, len(songs)))`** — pick 2–4 songs randomly. We need at least 2 to make a "mashup". Max 4 because mixing more makes it harder to train.

2. **`selected = random.sample(songs, n_mix)`** — without replacement: can't mix the same song with itself.

3. **`stem_pool = STEM_NAMES[:]; random.shuffle(stem_pool)`** — shuffle `['drums', 'vocals', 'bass', 'other']`. Then `stem_pool[i % 4]` cycles through them: song0 → drums, song1 → vocals, etc. Ensures diversity of stem types.

4. **`load_dur = target_duration * 1.3 if apply_tempo else target_duration`** — load 13s when stretching. After tempo stretch (e.g., rate=0.85), 13s compressed to ~11s, still > 10s. If we loaded exactly 10s, stretch could give <10s and we'd need to pad (worse quality).

5. **`mixed += stem_audio`** — simple summing. Real test mashups are also made by summing stems.

6. **`mixed = mixed / mx * 0.9`** — normalise to 0.9 (not 1.0) to leave headroom. 1.0 could cause clipping after further floating-point ops.

7. **`add_random_noise(mixed)`** — adds ESC-50 noise at random SNR from RAM cache.

8. **Re-normalise after noise** — noise can push the signal above the previous normalisation level.

---

### Q5: What is the random baseline and why is it 0.10?

**Answer:** With 10 classes and a balanced test set, a random classifier assigns each class with probability 1/10. For each class, precision ≈ 0.1, recall ≈ 0.1, so F1 ≈ 0.1. Macro F1 = mean of per-class F1 ≈ 0.10.

This is the theoretical floor — any trained model should beat this.

---

## B. Audio Fundamentals

### Q6: What is RMS? Why are you using it? Why only RMS and not mean?

**Code:**
```python
rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
# RMS = √(mean(signal²)) for each frame
```

**Answer:**

**RMS = Root Mean Square energy.** It measures the average power (loudness) of an audio signal in each time frame.

```
RMS = √( (1/N) × Σ x[i]² )
```

**Why not mean(signal)?** Audio signals oscillate around zero — their mean is approximately 0 regardless of loudness. Squaring first (x²) removes the sign, giving the actual signal energy. Then taking the mean gives average power, and the square root brings it back to amplitude units.

**Why RMS for genre classification?**
- Rock and metal have higher RMS (loud, compressed) vs classical (dynamic range, loud + quiet)
- Hip-hop has consistently high RMS in bass frequencies
- It captures **energy dynamics** — how loudness varies over time

**Why only RMS and not variance, peak, etc.?**
RMS is the standard measure of perceptual loudness in audio engineering. Peak amplitude can be misleading (a single loud transient spike). RMS integrates over time, giving a more stable loudness estimate. We also compute mean + std of RMS frames, so we get the average loudness AND its variation over time.

---

### Q7: What is audio gain? Why are you using it? How does it help the model? (Examiner gives a situation)

**Code:**
```python
def apply_volume_jitter(audio, min_db=-6, max_db=6):
    return audio * (10 ** (random.uniform(-6, 6) / 20))
```

**Answer:**

**Audio gain** = multiplying the audio waveform by a scalar to increase or decrease loudness.

**The formula:** `gain = 10^(dB/20)`. The 20 (not 10) because dB for amplitude = 20·log₁₀(ratio). For power, it's 10·log₁₀.

| dB | Gain multiplier | Effect |
|----|----------------|--------|
| +6 dB | ×2.0 | Twice as loud |
| 0 dB | ×1.0 | Unchanged |
| -6 dB | ×0.5 | Half as loud |

**Why use it as augmentation?**

In the test mashups, different stems are mixed at different volume levels — the drums might be much louder than the bass, or vice versa. By randomly scaling each stem by ±6 dB before mixing, we simulate this real-world variation.

**Situation the examiner might give:** "Imagine you have two mashups — one where the drums are twice as loud as the vocals, and one where vocals dominate. Without gain jitter augmentation, a model that sees only balanced mixes during training might mistake a drums-heavy mix for metal (lots of percussion energy) and a vocals-heavy mix for pop. With gain jitter, the model learns that genre doesn't depend on which stem happens to be loudest — it learns genre-invariant features."

**How it helps the model:** Prevents the model from learning a spurious shortcut — "if energy is concentrated in the drum frequency range, it's metal" — when the real signal is the presence and interplay of ALL instruments.

---

### Q8: What is SNR? Walk through the math.

**Code:**
```python
def add_noise_at_snr(signal, noise, snr_db):
    sig_pow = np.mean(signal ** 2) + 1e-10
    noi_pow = np.mean(noise ** 2) + 1e-10
    target_pow = sig_pow / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_pow / noi_pow)
    return signal + noise_scaled
```

**Answer:**

**SNR (Signal-to-Noise Ratio)** = how much louder the signal is than the noise:
```
SNR (dB) = 10 · log₁₀(P_signal / P_noise)
```

**Deriving the scale factor:**
```
SNR = 10 · log₁₀(P_sig / P_noise_target)
P_noise_target = P_sig / 10^(SNR/10)        ← rearranged

We have noise with power P_noi_raw.
We want noise with power P_noise_target.
Scale factor on amplitude: noise_scaled = noise × α
Power of noise_scaled = α² × P_noi_raw

Set α² × P_noi_raw = P_noise_target
α = √(P_noise_target / P_noi_raw)          ← that's the sqrt line
```

**Why `+ 1e-10`?** Prevents division by zero when the signal or noise is completely silent (all zeros). 1e-10 is much smaller than any real audio power value, so it doesn't affect the result.

**Why SNR range 5–25 dB?**
- 5 dB: noise is 3× weaker than signal — very noisy, hard to hear the music
- 25 dB: noise is 316× weaker — barely audible background noise
- This range matches real-world recording environments (noisy street = ~5 dB, quiet room = ~25 dB)

---

### Q9: What is normalisation (in audio context)? What is 0.9 peak normalisation?

**Code:**
```python
mx = np.abs(mixed).max()
if mx > 0: mixed = mixed / mx * 0.9
```

**Answer:**

**Normalisation** = scaling audio so the peak amplitude reaches a target value. This prevents:
1. **Clipping** — values exceeding [-1.0, 1.0] get clipped (distorted)
2. **Inconsistent loudness** — some genres might naturally have higher/lower average amplitude

**Why 0.9 and not 1.0?**
After normalising to 1.0, any subsequent operation (like adding noise) could push the signal above 1.0:
```
signal_peak = 1.0
noise adds 0.05 at some point
signal + noise = 1.05 → CLIPPED to 1.0 → distorted
```
With 0.9 headroom:
```
signal_peak = 0.9
signal + noise = 0.95 → still within bounds
```

**Why normalise twice** (after mixing AND after adding noise)?
- First normalisation: ensures the mix of stems doesn't clip before noise addition
- Second normalisation: ensures the noise-added signal is also properly scaled

---

## C. Feature Extraction

### Q10: What is MFCC? Explain from scratch.

**Code:**
```python
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
```

**Answer:**

**MFCC = Mel-Frequency Cepstral Coefficients** — compact representation of the spectral shape (timbre) of audio.

**Step-by-step computation:**
```
1. Divide audio into frames (window = N_FFT = 2048 samples = 93ms)
2. For each frame: FFT → power spectrum |X(f)|²
3. Apply mel filterbank (128 triangular filters on mel scale) → mel spectrum
4. Take log(mel spectrum)
5. Apply DCT (Discrete Cosine Transform) → cepstrum
6. Take first 40 coefficients = MFCCs
```

**What do they capture?** The first few MFCCs capture the "coarse shape" of the spectrum — the overall timbre (what makes a piano sound different from a guitar even at the same pitch). Higher-order MFCCs capture finer spectral details.

**Why 40 MFCCs (not 13 as in speech)?** Speech uses 13 because speech only has vocal tract information. Music has richer timbral detail from multiple instruments — 40 captures more of this.

---

### Q11: What is delta? What is delta2? In mathematical terms what is delta?

**Code:**
```python
delta  = librosa.feature.delta(mfcc)           # first derivative
delta2 = librosa.feature.delta(mfcc, order=2)  # second derivative
```

**Mathematical definition:**

**Delta (Δ)** — first-order temporal derivative of MFCC:
```
Δ[t] = (Σ n·(MFCC[t+n] - MFCC[t-n])) / (2 · Σ n²)    for n = 1, 2, ..., N
```
In practice (using a window of 9 frames, n=1..4):
```
Δ[t] ≈ (MFCC[t+1] - MFCC[t-1]) / 2    [simplified]
```
It measures **how the MFCC is changing over time** — like velocity.

**Delta-Delta (Δ²)** — second derivative:
```
Δ²[t] = Δ[Δ[t]]    (derivative of the derivative)
```
Measures the **rate of change of the change** — like acceleration.

**Why are they useful?**
- MFCC alone: static snapshot of spectral shape at each moment
- MFCC + Δ: captures how quickly the spectrum is changing (e.g., a drum hit = rapid change)
- MFCC + Δ + Δ²: captures onset/offset dynamics (does the spectrum change quickly then stabilise, or smoothly transition?)

**Genre relevance:**
- Blues: slow, smooth transitions in Δ (expressive slides)
- Metal: rapid Δ changes (fast pick attacks, drum blasts)
- Classical: very smooth Δ (legato playing)

---

### Q12: What is the mel scale? Why use it?

**Answer:**

The mel scale is a perceptually-motivated frequency scale. Humans perceive pitch logarithmically — the perceived difference between 100Hz and 200Hz is the same as between 1000Hz and 2000Hz (both are one octave apart).

**Conversion formula:**
```
mel = 2595 × log₁₀(1 + f/700)
```

**Why use mel for music?**
- Uniform spacing in mel = uniform perceptual spacing
- A mel spectrogram allocates more bins to low frequencies (where music pitch lives) and fewer to high frequencies (where mostly noise/overtones are)
- CNN convolutions with equal kernel sizes cover perceptually equal frequency ranges

**Comparison:**
```
Linear: 100 bins cover 0–22050 Hz each covers 220 Hz
Mel:    lower bins cover ~5 Hz each  (piano's range)
        upper bins cover ~500 Hz each (high overtones)
```

---

### Q13: What is `power_to_db`? Why use it?

**Code:**
```python
mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
```

**Answer:**

`power_to_db` converts linear power values to decibel scale:
```
mel_db = 10 × log₁₀(mel / ref)
```

**`ref=np.max`**: The reference value is the maximum power in this spectrogram. So the loudest frequency bin = 0 dB, all others are negative (range: [-80, 0]).

**`top_db=80.0`**: Clips values below -80 dB (i.e., values more than 80 dB below the maximum). Values quieter than -80 dB are essentially silence and don't help classification.

**Why log (dB) scale?**
Audio amplitude spans 6 orders of magnitude (a whisper to a shout differs by 1,000,000×). Without log, quiet sounds would have near-zero values and dominate zero-padding. With log:
- The full 80 dB range maps linearly to [-80, 0]
- Both loud and quiet sounds have distinguishable values
- Matches how humans perceive loudness (logarithmically)

---

### Q14: What is chroma? Why extract it?

**Code:**
```python
chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
# Output: (12, T) — 12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
```

**Answer:**

Chroma (chromagram) shows the energy at each of the 12 musical pitch classes, collapsed across all octaves. C4 (middle C) and C5 (one octave higher) both contribute to the "C" bin.

**Why useful for genre?**
Different genres use characteristic chord progressions that activate different chroma patterns:
- Blues: pentatonic scale (5 specific pitch classes dominant)
- Classical: rich harmony, many pitch classes active
- Metal: power chords (root + fifth, 2 pitch classes)
- Reggae: characteristic minor chord progressions

Chroma captures harmonic content independent of the specific octave or instrument playing it.

---

### Q15: What is spectral centroid and what does it tell you?

**Code:**
```python
centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
# Output: (1, T) — weighted mean of frequencies
```

**Answer:**

Spectral centroid = the "centre of mass" of the frequency spectrum:
```
centroid = Σ(f × |X(f)|²) / Σ|X(f)|²    [weighted mean frequency]
```

**Intuition:** Think of the spectrum as a seesaw with frequencies on one side and magnitudes on the other. The centroid is where it balances.

**High centroid** = bright/trebly sound (lots of high-frequency energy) → metal, rock (cymbals, distorted guitar)
**Low centroid** = warm/bass-heavy sound → hip-hop, reggae (bass-dominant)

**Why mean + std of centroid frames?**
Mean captures the average brightness. Std captures how much brightness varies over time (high std = dynamic music with bright and dark moments; low std = consistently bright or dark).

---

## D. CNN Architecture

### Q16: Why did you use BatchNorm2d? What does it do?

**Code:**
```python
nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
```

**Answer:**

**BatchNorm2d normalises each feature map (channel) across the batch and spatial dimensions:**
```
For channel c:
  μ_c = mean of all values in channel c across (batch, H, W)
  σ_c = std of all values in channel c
  x_norm = (x - μ_c) / (σ_c + ε)
  output = γ_c × x_norm + β_c    [γ, β are learned parameters]
```

**Why after Conv, before ReLU?**
- Conv produces activations with arbitrary scale/shift
- BN normalises them → ReLU operates in a consistent range
- If placed after ReLU, you normalise already-rectified values (no negative values) — less stable

**Benefits:**
1. **Faster training**: higher learning rates are safe, stable gradient flow
2. **Regularisation**: batch statistics add noise (similar to dropout)
3. **Reduces internal covariate shift**: later layers don't need to adapt to constantly shifting input distributions

**Training vs Inference:**
- `model.train()`: uses batch statistics (current batch's mean/var)
- `model.eval()`: uses running statistics (accumulated exponential moving average from training)
- **Critical to call `model.eval()` before validation/inference**

---

### Q17: Why start with 32 output channels? Why increase to 64, 128, 256?

**Code:**
```python
# Block 1: 1 → 32 channels
# Block 2: 32 → 64 channels
# Block 3: 64 → 128 channels
# Block 4: 128 → 256 channels
```

**Answer:**

**Why not start with 1 or 16?** 32 is the minimum that provides enough filter diversity. With fewer channels, the network can't learn enough different local patterns.

**Why progressive doubling?** Each MaxPool(2,2) halves both spatial dimensions:
```
Block 1: (1, 128, 431) → (32, 64, 215)    Spatial: 128×431 = 55,168
Block 2: (32, 64, 215) → (64, 32, 107)    Spatial: 32×107 = 3,424
Block 3: (64, 32, 107) → (128, 16, 53)    Spatial: 16×53 = 848
Block 4: (128, 16, 53) → (256, 1, 1)      Spatial: 1×1 = 1
```

As spatial resolution shrinks, we increase channels to maintain the **total information capacity** of each layer. If we didn't increase channels:
- After Block 4 we'd have (32, 1, 1) = 32 numbers
- That's too few to represent 10 genres
- With doubling: (256, 1, 1) = 256 numbers → rich 256-dim feature vector

**The pattern follows VGG**: standard practice since VGG showed that doubling channels when halving spatial size maintains representational capacity.

---

### Q18: What is MaxPool2d? What does it do?

**Code:**
```python
nn.MaxPool2d(2, 2)   # kernel=2, stride=2
```

**Answer:**

**MaxPool2d(2, 2)** slides a 2×2 window over the feature map with stride 2 and takes the maximum value in each window:

```
Input (4×4):          Output (2×2):
┌─┬─┬─┬─┐            ┌──┬──┐
│1│3│2│0│            │3 │ 4│   (max of top-left 2×2 = max(1,3,2,4) = 4 ← wait)
├─┼─┼─┼─┤   ──►      ├──┼──┤
│2│4│1│3│            │5 │ 3│   (max of bottom-left 2×2 = max(2,4,5,0))
├─┼─┼─┼─┤            └──┴──┘
│5│0│3│2│
├─┼─┼─┼─┤
│1│3│0│1│
└─┴─┴─┴─┘
```

**Effect:**
- Halves spatial dimensions (height and width both ÷2)
- Keeps the most prominent feature in each region (maximum = strongest activation)
- Makes the network translation-invariant to small shifts

**Why not Average Pool here?** MaxPool keeps the strongest signal — useful for detecting presence of features anywhere in the 2×2 region. AvgPool would dilute strong signals with surrounding weak ones.

**Why MaxPool at early layers but AdaptiveAvgPool at the end?**
- Early layers: MaxPool detects "is this feature present anywhere in this region?"
- Final layer: AdaptiveAvgPool summarises "on average, across the whole spectrogram, how much of each high-level feature is present?" — more appropriate for global classification.

---

### Q19: What is AdaptiveAvgPool2d? Why make it to a single number? How does it help?

**Code:**
```python
nn.AdaptiveAvgPool2d((1, 1))
# Input: (B, 256, 16, 53) → Output: (B, 256, 1, 1)
```

**Answer:**

**AdaptiveAvgPool2d((H, W))** automatically computes the pooling kernel size to produce exactly (H, W) output. With (1,1): pools the entire spatial dimension into a single value per channel.

```
For each channel c:
  output[c] = mean of ALL values in channel c across spatial dims
  = mean(feature_map_c)    [scalar per channel]
```

**Why reduce to a single number per channel?**

At Block 4, the feature map is (256, 16, 53). The 256 channels represent 256 different learned patterns. The 16×53 spatial grid represents different time-frequency locations.

For genre classification, **location doesn't matter** — metal features (distortion patterns) can appear at any time point in the audio. We want to know: "is this pattern present ANYWHERE in the clip?" Average pooling answers: "how prevalent is this pattern across the ENTIRE clip?"

After AdaptiveAvgPool: we have 256 numbers, one per pattern, representing global prevalence. This is a fixed-size feature vector regardless of input length — then fed to the classifier.

**Why Adaptive (not fixed)?** If input varies in length, a fixed kernel would produce different output sizes. Adaptive ensures output is always (256, 1, 1) regardless of input shape.

**Alternatively:** Global Average Pooling (GAP) = same thing. Used in GoogLeNet, ResNet, MobileNet.

**Why 256 inputs to the Linear layer?** Because AdaptiveAvgPool outputs (B, 256, 1, 1), after Flatten → (B, 256). So `Linear(256, 128)` takes exactly those 256 values.

---

### Q20: In ConvBlock, what are the two parameters of Conv2d?

**Code:**
```python
nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         ↑  ↑   ↑              ↑
#    in_ch out_ch kernel         padding
```

**Answer:**

The two most important parameters:
1. **`in_channels=1`**: number of input channels (1 for grayscale mel spec — single channel)
2. **`out_channels=32`**: number of filters to learn (32 different patterns)

**Other parameters:**
- **`kernel_size=3`**: 3×3 filter (standard — two 3×3 convolutions have same receptive field as one 5×5 but fewer params)
- **`padding=1`**: adds 1 pixel of zeros around the boundary. With a 3×3 kernel and padding=1, output size = input size (before MaxPool). Without padding, each conv would shrink the spatial dims by 2.

**Why 3×3 specifically?** Research (VGG paper) showed that stacking multiple 3×3 convolutions is better than one large (5×5 or 7×7) convolution:
- Same effective receptive field
- Fewer parameters: 2×(3×3) = 18 params vs 1×(5×5) = 25 params
- More non-linearity (two ReLUs vs one)

**Why `padding=1`?** Without padding, each 3×3 conv shrinks spatial size by 2 (loses 1 pixel each side). We want downsampling to ONLY happen at MaxPool (controlled, ×2 each time), not gradually throughout each block.

---

### Q21: Why 3 channels? (Examiner might mean why use 3-channel approach or why 3×3 kernel)

**Answer (if asking about 3×3 kernel):** See Q20 above.

**Answer (if asking why input has 1 channel, not 3):**
```python
nn.Conv2d(1, 32, ...)  # in_channels = 1
```
The mel spectrogram is **grayscale** (one channel) — each pixel represents energy at one time-frequency location. There's no colour information. Unlike RGB images (3 channels for red/green/blue), audio has a single value per time-frequency bin.

If we had 3 channels: we'd need to concatenate 3 different representations (e.g., MFCC + chroma + mel), but we chose to use only mel spectrogram for the CNN.

---

### Q22: Why BatchNorm2d(32) not BatchNorm2d(1)?

**Answer:**
After `Conv2d(1, 32)`, the output has 32 channels (feature maps). BatchNorm normalises each channel independently. So we need `BatchNorm2d(32)` — one set of (γ, β) parameters per channel = 32 × 2 = 64 learnable parameters.

`BatchNorm2d(1)` would mean normalising a 1-channel input — used before any convolution or when you have a single-channel intermediate representation.

---

### Q23: How to import and use ResNet34?

**Answer:**

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet34
resnet34 = models.resnet34(pretrained=True)

# Modify for audio (mel spectrogram is 1 channel, not 3)
# Option 1: Change the first conv layer
resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Option 2: Adapt input (repeat channel 3 times)
# x_rgb = x.repeat(1, 3, 1, 1)   # (B, 1, H, W) → (B, 3, H, W)

# Modify final classifier for 10 genres
num_features = resnet34.fc.in_features   # 512 for ResNet34
resnet34.fc = nn.Linear(num_features, 10)

# Print parameters
total = sum(p.numel() for p in resnet34.parameters())
trainable = sum(p.numel() for p in resnet34.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")
```

**What ResNet34 looks like:** 34-layer ResNet with residual (skip) connections. Each residual block: `output = F(x) + x` where F is two 3×3 convolutions. The skip connection prevents vanishing gradients.

---

## E. Training Loop

### Q24: Write the training loop. (SURE QUESTION)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np

def train_model(model, train_loader, val_loader,
                n_epochs=10, lr=3e-4, device='cpu'):
    """Complete training loop with val evaluation."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(1, n_epochs + 1):
        # ── Training phase ───────────────────────────────────────
        model.train()   # enable dropout + batch stats in BN
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)              # forward pass
            loss = criterion(logits, y_batch)    # compute loss

            optimizer.zero_grad()                # clear old gradients
            loss.backward()                      # backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
            optimizer.step()                     # update weights

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()   # update learning rate

        # ── Validation phase ─────────────────────────────────────
        model.eval()    # disable dropout, use BN running stats
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():   # no gradient computation needed
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val_d = y_val.to(device)

                logits = model(X_val)
                val_loss += criterion(logits, y_val_d).item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_val.numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        avg_train_loss = total_loss / n_batches
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch:2d}/{n_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  → Saved best model")

    return history
```

---

### Q25: What is gradient clipping? Why max_norm=1.0?

**Code:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Answer:**

**The problem: exploding gradients.** During backpropagation, gradients are multiplied through each layer. In deep networks or with certain weight configurations, these products can grow exponentially → very large gradient updates → weights fly off to infinity → loss becomes NaN.

**What clip_grad_norm_ does:**
```
1. Compute the total L2 norm of ALL gradients concatenated:
   total_norm = √(Σ ||grad_i||²) for all parameter tensors

2. If total_norm > max_norm:
   scale = max_norm / total_norm
   for each grad: grad *= scale    [scales ALL down proportionally]
```

**Why 1.0?** The gradient norm is in the same units as parameter scale. A norm of 1.0 means each parameter gets an update of at most order 1 (before learning rate scaling). This is a standard default for CNNs. Transformers often use 1.0 too; recurrent networks sometimes use 5.0.

**Must be called BEFORE `optimizer.step()`** — you clip the gradients, then the optimizer uses the clipped gradients to update weights.

---

### Q26: What is the difference between `model.train()` and `model.eval()`?

**Answer:**

Two layers behave differently in train vs eval mode:

**Dropout:**
- `train()`: randomly zeros out units with probability p
- `eval()`: passes all units unchanged (no zeroing)
- Why: dropout is a training regulariser; at inference we want deterministic predictions

**BatchNorm:**
- `train()`: uses statistics from the **current batch** (mean, var computed from the batch)
- `eval()`: uses **running statistics** accumulated during training (exponential moving average)
- Why: at inference, batch size might be 1 (can't compute meaningful statistics), so we use the stable running stats

**Critical bug if forgotten:**
```python
# WRONG: evaluating with train mode active
model.train()
with torch.no_grad():
    preds = model(X_val)   # dropout drops random neurons → noisy predictions

# CORRECT:
model.eval()
with torch.no_grad():
    preds = model(X_val)   # deterministic, uses running BN stats
```

---

### Q27: Why `optimizer.zero_grad()` before `loss.backward()`?

**Answer:**

PyTorch **accumulates gradients** — each call to `.backward()` adds new gradients to existing `.grad` attribute of parameters. If you don't zero out gradients, they accumulate across batches:

```
Batch 1: grad = ∇L₁         (after backward)
Batch 2: grad = ∇L₁ + ∇L₂  (accumulated if no zero_grad!)
This gives wrong gradient — you'd be optimising a mix of losses
```

`optimizer.zero_grad()` sets all `.grad` to zero before the new backward pass.

**Note:** In gradient accumulation (AST training), we intentionally DON'T call zero_grad every step — we accumulate over multiple batches, then step:
```python
for bi, batch in enumerate(loader):
    loss = compute_loss(batch) / ACCUM_STEPS
    loss.backward()   # accumulates gradients
    if (bi + 1) % ACCUM_STEPS == 0:
        optimizer.step()    # use accumulated gradient
        optimizer.zero_grad()  # then reset
```

---

### Q28: Why `with torch.no_grad()` during validation?

**Answer:**

`torch.no_grad()` disables gradient computation for everything inside the block.

**Why needed for validation?**
- During forward pass, PyTorch builds a computational graph to enable backward pass
- Building this graph consumes memory (stores intermediate activations)
- During validation, we never call `.backward()` — the graph is wasted memory

`torch.no_grad()` skips building the graph → ~50% memory reduction during validation → allows larger batch sizes for validation.

```python
# Without no_grad: PyTorch builds full computation graph
logits = model(X_val)   # graph built, ~2× memory

# With no_grad: no graph built
with torch.no_grad():
    logits = model(X_val)   # ~1× memory
```

---

## F. Loss Functions

### Q29: What loss function did you use? Why?

**Code:**
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
        return -(smooth * log_probs).sum(dim=-1).mean()
```

**Answer:**

We used **Label Smoothing Cross-Entropy** (ε = 0.1).

**Standard CE:** forces the model to output probability = 1.0 for the correct class. This makes the model overconfident — logit for the true class → +∞.

**Label smoothing:** replaces one-hot targets with soft targets:
```
Standard:   [0,    0,    0,    1,    0,    0,    0,    0,    0,    0   ]
Smoothed:   [0.011,0.011,0.011, 0.9, 0.011,0.011,0.011,0.011,0.011,0.011]
            └──── ε/(K-1) = 0.1/9 = 0.011 ────┘      └── 1-ε = 0.9 ──┘
```

**Why ε/(K-1) and not ε/K?** The total probability must sum to 1: (1-ε) + (K-1)×(ε/(K-1)) = (1-ε) + ε = 1 ✓

**Benefits:**
1. Prevents logit from going to +∞ (bounded gradient)
2. Model stays calibrated — P(correct class) ≈ 0.9, not 0.9999
3. Better generalisation on unseen data
4. Reduces overconfidence that leads to poor calibration on domain-shifted test data

**Why ε = 0.1?** Standard value. ε=0.0 = standard CE. ε=0.2 would make targets too soft (class confidence 0.8), potentially hurting convergence.

---

### Q30: What is Focal Loss? How does it differ from your loss?

**Answer:**

**Focal Loss** (introduced by RetinaNet for object detection):
```python
FL(pt) = -α × (1 - pt)^γ × log(pt)

where:
  pt = probability of the correct class (softmax output)
  γ  = focusing parameter (typically 2.0)
  α  = class weight (optional)
```

**Standard CE:** `CE = -log(pt)` — all examples contribute equally to the loss.

**Focal Loss:** `FL = -(1-pt)^γ × log(pt)` — easy examples (high pt) are downweighted by `(1-pt)^γ`:

```
pt = 0.9  → (1-0.9)^2 = 0.01  → nearly ignored (easy, correct prediction)
pt = 0.1  → (1-0.1)^2 = 0.81  → emphasised (hard, wrong prediction)
```

**Use case:** When easy examples (clear genres) dominate and the model ignores hard cases (rock vs blues confusion). Focal Loss forces attention on hard misclassified examples.

**Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)                       # probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # downweight easy examples
        return focal_loss.mean()
```

**Why we chose Label Smoothing over Focal Loss:** The dataset is balanced (equal samples per genre), so there's no class imbalance problem Focal Loss is designed to solve. Label smoothing addresses the overconfidence problem more relevant to our transfer learning setup.

---

## G. Optimizers & Schedulers

### Q31: What is AdamW? Why use it over Adam?

**Answer:**

**Adam:** Adaptive gradient optimizer with momentum.
```
m = β₁m + (1-β₁)g         # 1st moment: gradient mean (momentum)
v = β₂v + (1-β₂)g²        # 2nd moment: gradient variance
ŵ_step = -lr × m̂/√(v̂ + ε) # adaptive per-param step
```

**Adam's bug with weight decay:** L2 regularization in Adam adds λw to the gradient before momentum/variance estimates:
```
g' = g + λw      ← L2 reg term
Adam then computes: step ∝ g'/√(v + ε)
```
This scales weight decay by the gradient variance — some parameters get weaker regularisation than others. **Mathematically incorrect L2 regularisation.**

**AdamW fix:** Apply weight decay directly to weights, decoupled from gradient:
```python
# Adam (buggy weight decay):
g' = g + λw
param -= lr × f(g')     # weight decay coupled with gradient

# AdamW (correct):
param -= lr × f(g)      # gradient update
param -= lr × λ × w     # separate weight decay
```

**Practical result:** AdamW generalises better, especially important for transformers and fine-tuning.

---

### Q32: What is CosineAnnealingLR? Why better than step decay?

**Code:**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS, eta_min=1e-6)
```

**Formula:**
```
lr_t = eta_min + (lr_max - eta_min)/2 × (1 + cos(π × t / T_max))
```

**Learning rate values:**
```
t=0:        lr = lr_max   (full rate)
t=T_max/4:  lr ≈ 0.85 × lr_max
t=T_max/2:  lr = (lr_max + eta_min)/2
t=T_max:    lr ≈ eta_min  (near zero)
```

**Why better than StepLR?**
- StepLR: every N epochs, multiply LR by 0.1 → sudden drop → optimizer "shocked"
- CosineAnnealing: smooth monotonic decrease → optimizer gradually settles into minimum
- The cosine shape spends more time at high LR early (fast convergence) and low LR late (fine-tuning in narrow valley)

**Why `eta_min=1e-6` (not 0)?** Completely zeroing LR stops all learning — the model can't make any progress even if it hasn't converged. 1e-6 allows tiny final corrections.

---

## H. AST Model

### Q33: Why does AST differ from SimpleCNN? What makes it better?

**Answer:**

| Feature | SimpleCNN | AST |
|---------|-----------|-----|
| Architecture | 4 conv blocks | 12-layer transformer |
| Receptive field | Local (3×3 patches) | Global (self-attention over all patches) |
| Pretraining | None (random init) | AudioSet 2M clips |
| Parameters | 423K | 86.2M |
| Feature type | Learned local patterns | Semantic audio representations |
| Long-range context | None | Full sequence self-attention |

**The key advantage of self-attention:**
A 3×3 conv only sees its immediate neighbourhood. Genre recognition often requires **long-range temporal dependencies**: "this rhythm pattern repeats throughout the clip" (reggae), "the overall harmonic structure changes slowly" (classical). Self-attention allows token 1 to directly attend to token 500 — the model can capture the full 10-second structure in a single operation.

**Why does pretraining matter so much?**
AudioSet contains 2M diverse real-world audio clips. AST's layers have already learned to distinguish: drums from piano, electric guitar from acoustic, distortion from reverb. We only need to adapt these representations to our 10-genre task — not learn them from scratch with 5,000 samples.

---

### Q34: How to import and load ASTForAudioClassification?

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTModel
import torch

# Load feature extractor (handles audio → input tensor)
ast_fe = ASTFeatureExtractor.from_pretrained(
    'MIT/ast-finetuned-audioset-10-10-0.4593'
)

# Load model for classification (replaces 527-class head with 10-class head)
ast_model = ASTForAudioClassification.from_pretrained(
    'MIT/ast-finetuned-audioset-10-10-0.4593',
    num_labels=10,                   # our task: 10 genres
    ignore_mismatched_sizes=True     # because we changed num_labels
)
# ignore_mismatched_sizes: allows loading when classifier shape differs
# (pretrained: 527 classes, we want: 10)

# Check parameters
total = sum(p.numel() for p in ast_model.parameters())
trainable = sum(p.numel() for p in ast_model.parameters() if p.requires_grad)
print(f"Total: {total/1e6:.1f}M, Trainable: {trainable/1e6:.1f}M")

# Run inference
audio = torch.randn(16000 * 10)   # 10 seconds at 16kHz
inputs = ast_fe(audio.numpy(), sampling_rate=16000, return_tensors='pt')
with torch.no_grad():
    outputs = ast_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred_class = probs.argmax()
```

**What `ignore_mismatched_sizes=True` does:** When you load a checkpoint, PyTorch checks that every tensor's shape matches. The classifier head in the checkpoint has shape [768, 527], but we instantiated with [768, 10]. Without `ignore_mismatched_sizes=True`, this raises an error. With it, mismatched layers are re-initialised randomly (only the classifier; all other 86M params load correctly).

---

### Q35: Load just the base AST model (for embeddings, like in fallback):

```python
from transformers import ASTFeatureExtractor, ASTModel

# Base model (no classification head) — for embeddings
ast_base = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
ast_base.eval()

# Freeze everything
for p in ast_base.parameters():
    p.requires_grad = False

@torch.no_grad()
def extract_ast_embedding(audio_16k: np.ndarray) -> np.ndarray:
    """Extract 768-dim embedding from 16kHz audio."""
    inputs = ast_fe(audio_16k, sampling_rate=16000, return_tensors='pt')
    outputs = ast_base(**inputs)
    # outputs.last_hidden_state: (1, 514, 768) — 514 tokens, 768 dim each
    embedding = outputs.last_hidden_state.mean(dim=1)  # mean pool → (1, 768)
    return embedding.squeeze(0).numpy()  # → (768,)
```

---

### Q36: What is `ignore_mismatched_sizes=True`? When do you need it?

**Answer:** See Q34. Need it whenever you load a pretrained model and change any layer's shape:
- Different number of output classes (`num_labels`)
- Different input channels
- Different hidden dimension

Without it → `RuntimeError: size mismatch for classifier.dense.weight`

---

## I. W&B (Weights & Biases)

### Q37: Show me your W&B logs. What did you track?

**Answer:**

Project: `wandb.ai/23f3003478-iit-madras/23f3003478-t12026`

4 runs tracked:

| Run | Model | Val F1 | Epochs |
|-----|-------|--------|--------|
| mfcc-xgboost | XGBoost | 0.5710 | 1 |
| simplecnn-melspec | SimpleCNN | 0.3369 | 10 |
| ast-xgboost | Frozen AST + XGB | 0.7825 | 1 |
| ast-finetuned | Fine-tuned AST | 0.8677 | 12 |

**Metrics logged per epoch (AST run):**
```python
wandb.log({
    "epoch": epoch,
    "train/loss": total_loss / len(ast_train_dl),
    "train/f1": train_f1,
    "val/loss": val_loss_ep / vn,
    "val/f1": val_f1,
    "learning_rate": scheduler.get_last_lr()[0],
})
```

**How to initialise W&B:**
```python
import wandb
wandb.init(
    project='23f3003478-t12026',
    entity='23f3003478-iit-madras',
    name='ast-finetuned',
    config={
        'model': 'AST',
        'lr_encoder': 5e-5,
        'lr_head': 1e-3,
        'epochs': 12,
        'batch_size': 8,
        'grad_accum': 2,
    }
)
```

---

### Q38: Why use W&B over just printing metrics?

**Answer:**
1. **Visualisation**: automatic training curve plots, no matplotlib code needed
2. **Comparison**: overlay multiple runs on the same chart to compare hyperparameters
3. **Persistence**: metrics stored permanently, accessible after session ends (Kaggle sessions expire)
4. **Reproducibility**: config logged alongside metrics — you know exactly which hyperparameters produced which results
5. **Alerts**: can set alerts when val F1 improves or diverges

---

## J. Coding Tasks

### Q39: Write an LSTM from scratch. (SURE QUESTION)

**Basic LSTM classifier:**
```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    LSTM for audio sequence classification.
    Input: (B, T, F) — batch, time steps, features per step
    Output: (B, num_classes)
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_classes=10, dropout=0.3, bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,        # features per timestep (e.g., 128 mel bands)
            hidden_size=hidden_size,      # LSTM cell size
            num_layers=num_layers,        # stacked LSTM layers
            batch_first=True,             # input shape: (B, T, F) not (T, B, F)
            dropout=dropout if num_layers > 1 else 0,  # dropout between LSTM layers
            bidirectional=bidirectional   # process sequence forward + backward
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        # h0, c0 default to zeros if not provided
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (B, T, hidden*directions) — output at every timestep
        # h_n:      (num_layers*directions, B, hidden) — final hidden state

        # Use the last timestep's output for classification
        last_out = lstm_out[:, -1, :]    # (B, hidden*directions)
        # Alternatively: mean pooling
        # last_out = lstm_out.mean(dim=1)

        out = self.dropout(last_out)
        return self.classifier(out)     # (B, num_classes)


# Usage:
model = LSTMClassifier(
    input_size=128,    # 128 mel bands per frame
    hidden_size=256,   # LSTM hidden size
    num_layers=2,      # 2 stacked LSTM layers
    num_classes=10,    # 10 genres
    dropout=0.3,
    bidirectional=True
)

# Test with a batch
B, T, F = 8, 431, 128   # batch=8, 431 time frames, 128 mel bands
x = torch.randn(B, T, F)
out = model(x)
print(f"Input: {x.shape}, Output: {out.shape}")  # (8, 10)
```

---

### Q40: What are the input and output dimensions of LSTM? Why dropout?

**Answer:**

**Input dimensions:**
```
(batch_size, seq_len, input_size)
    ↑              ↑         ↑
  B=8          T=431      F=128
(batch)    (time frames) (mel bands per frame)
```

**Output dimensions:**
```
lstm_out: (B, T, hidden_size × directions)  — output at EACH timestep
h_n:      (num_layers × directions, B, hidden_size)  — FINAL hidden state
c_n:      (num_layers × directions, B, hidden_size)  — FINAL cell state
```

For classification: take `lstm_out[:, -1, :]` (last timestep) or `lstm_out.mean(1)` (mean over time).

**Why dropout?**
- LSTM has many parameters (4 weight matrices per layer)
- Without dropout, easily overfits on small datasets
- `dropout=0.3` in `nn.LSTM`: adds dropout **between layers** (not inside a single LSTM layer's recurrent connections)
- Common extra: `nn.Dropout(0.3)` after the LSTM output before classifier

**Why NOT dropout when `num_layers=1`?**
```python
dropout=dropout if num_layers > 1 else 0
```
When there's only 1 LSTM layer, there's no "between layers" to apply dropout. PyTorch will print a warning if you set `dropout > 0` with `num_layers=1`.

---

### Q41: Write a CNN from scratch (simple version the examiner might ask).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    4-block CNN for mel spectrogram classification.
    Input:  (B, 1, 128, 431)
    Output: (B, 10) logits
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction blocks
        self.features = nn.Sequential(
            # Block 1: (B,1,128,431) → (B,32,64,215)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: (B,32,64,215) → (B,64,32,107)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 3: (B,64,32,107) → (B,128,16,53)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 4: (B,128,16,53) → (B,256,1,1)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # any spatial size → 1×1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (B,256,1,1) → (B,256)
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Test
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 1, 128, 431)   # batch of 4 mel spectrograms
logits = model(x)
print(f"Input: {x.shape} → Output: {logits.shape}")  # (4, 10)

# Count parameters
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {params:,}")  # ~423,050
```

---

### Q42: Write an LSTM training loop.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

def train_lstm(lstm_model, X_train, y_train, X_val, y_val,
               epochs=20, lr=1e-3, batch_size=32, device='cpu'):
    """
    X_train: (N, T, F) numpy array — N samples, T timesteps, F features
    y_train: (N,) numpy array — integer class labels
    """
    lstm_model = lstm_model.to(device)

    # Create DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size * 2)

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # Train
        lstm_model.train()
        total_loss = 0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = lstm_model(X_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        lstm_model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                out = lstm_model(X_b.to(device))
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y_b.numpy())

        f1 = f1_score(trues, preds, average='macro')
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_dl):.4f} | Val F1: {f1:.4f}")
```

---

### Q43: Create 2 numpy arrays and stack them. (Examiner asked this)

```python
import numpy as np

# Create two arrays
a = np.array([1, 2, 3, 4, 5])          # shape: (5,)
b = np.array([10, 20, 30, 40, 50])      # shape: (5,)

# Stack horizontally (side by side) → (5, 2)
stacked_h = np.stack([a, b], axis=1)
print("np.stack axis=1:", stacked_h.shape)   # (5, 2)
# [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]

# Stack vertically (on top) → (2, 5)
stacked_v = np.stack([a, b], axis=0)
print("np.stack axis=0:", stacked_v.shape)   # (2, 5)
# [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]

# Other stacking methods:
c = np.vstack([a, b])     # (2, 5) — same as axis=0
d = np.hstack([a, b])     # (10,) — concatenate 1D arrays
e = np.concatenate([a.reshape(1,-1), b.reshape(1,-1)], axis=0)  # (2, 5)

# Practical example: stack feature vectors
feat1 = np.random.rand(768)   # AST embedding for sample 1
feat2 = np.random.rand(768)   # AST embedding for sample 2
X = np.stack([feat1, feat2], axis=0)   # (2, 768) — batch of 2
print("Feature matrix:", X.shape)
```

---

### Q44: Pick a random number in range [100, 101.117]. Write the code.

```python
import random
import numpy as np

# Method 1: random.uniform (most direct)
x = random.uniform(100, 101.117)
print(f"random.uniform: {x:.4f}")

# Method 2: numpy
x = np.random.uniform(100, 101.117)
print(f"np.random.uniform: {x:.4f}")

# Method 3: manual formula
x = 100 + (101.117 - 100) * random.random()
print(f"manual: {x:.4f}")

# Pick N samples from this range
N = 10
samples = [random.uniform(100, 101.117) for _ in range(N)]
print(f"10 samples: {samples}")

# Or with numpy:
samples = np.random.uniform(100, 101.117, size=N)
print(f"np samples: {samples}")
```

---

### Q45: Write code to load a pretrained model and use it for inference.

```python
import torch
import torch.nn as nn

# ── Option 1: Load our SimpleCNN checkpoint ───────────────────────────────────
class SimpleCNN(nn.Module):
    # ... (same as before)
    pass

# Load
model = SimpleCNN(num_classes=10)
checkpoint = torch.load('best_model.pt', map_location='cpu')

# If saved as state_dict:
model.load_state_dict(checkpoint)
# If saved as full checkpoint dict:
# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()    # ALWAYS set to eval mode after loading

# Inference
with torch.no_grad():
    x = torch.randn(1, 1, 128, 431)   # single mel spectrogram
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax().item()
    GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
    print(f"Predicted genre: {GENRES[pred]} ({probs[0,pred]:.2%} confidence)")


# ── Option 2: Load HuggingFace AST ───────────────────────────────────────────
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import numpy as np

fe = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
model = ASTForAudioClassification.from_pretrained(
    'MIT/ast-finetuned-audioset-10-10-0.4593',
    num_labels=10, ignore_mismatched_sizes=True
)
# Load our fine-tuned weights on top
model.load_state_dict(torch.load('ast_finetuned_best.pt', map_location='cpu'))
model.eval()

# Inference on audio
audio = np.random.randn(160000).astype(np.float32)  # 10s at 16kHz
inputs = fe(audio, sampling_rate=16000, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    pred_class = outputs.logits.argmax(-1).item()
    print(f"Genre: {GENRES[pred_class]}")
```

---

## K. General Theory

### Q46: What is Dropout? Why use it? What does `p=0.3` mean?

**Code:**
```python
nn.Dropout(0.3)     # zeros 30% of neurons
nn.Dropout2d(0.1)   # zeros 10% of entire feature maps
```

**Answer:**

During `model.train()`, each neuron is independently set to zero with probability p=0.3, and the remaining neurons are scaled by 1/(1-p) to maintain expected output magnitude.

**During `model.eval()`:** No zeroing — all neurons active. The 1/(1-p) scaling compensates so the expected output is the same.

**Why it works:**
- Prevents co-adaptation: neurons can't rely on specific other neurons being present
- Equivalent to training ~2^N different sub-networks simultaneously (N=number of neurons)
- At test time: ensemble of all these sub-networks → better generalisation

**Why Dropout2d for conv layers?**
Adjacent pixels in a feature map are highly correlated. Regular Dropout(0.1) drops 10% of individual pixels — the network can reconstruct them from neighbours (no real benefit). Dropout2d drops ENTIRE feature maps — forces learning of redundant features across channels.

**Why increase dropout rate in later layers (0.1 → 0.2 → 0.3)?**
Later layers are closer to the output, have fewer parameters, and are more likely to overfit to specific training patterns. Higher dropout provides stronger regularisation where it's most needed.

---

### Q47: What is normalisation? (General definition the examiner asked)

**Answer:**

**Normalisation** = transforming data so it fits a desired statistical range or distribution. Three types used in this project:

1. **Min-max normalisation (mel spec):** Scale to [0, 1]
```python
x_norm = (x - x.min()) / (x.max() - x.min())
```

2. **Z-score / StandardScaler (ML features):** Scale to mean=0, std=1
```python
x_norm = (x - mean) / std    # sklearn's StandardScaler
```

3. **BatchNorm2d (neural network):** Normalise activations per channel per batch
```python
x_norm = (x - μ_batch) / (σ_batch + ε)
x_out = γ × x_norm + β        # learnable scale/shift
```

4. **Peak normalisation (audio):**
```python
x_norm = x / max(|x|) × 0.9   # keep peak at 0.9
```

**Why normalise?**
- **ML features**: XGBoost doesn't need it (tree-based), but it speeds convergence for distance-based methods
- **Mel spec for CNN**: keeps gradients in a consistent scale, prevents one feature from dominating
- **BatchNorm**: addresses internal covariate shift, enables higher learning rates

---

### Q48: What is the difference between `nn.Dropout` and `nn.Dropout2d`?

```python
nn.Dropout(p=0.3)    # zeros individual scalar values
nn.Dropout2d(p=0.1)  # zeros entire 2D feature maps (channels)
```

**`nn.Dropout`** on a tensor of shape (B, 256):
- Each of the 256 values is independently zeroed with prob 0.3
- Result: random sparse vector

**`nn.Dropout2d`** on a tensor of shape (B, 64, 32, 107):
- Each entire (32, 107) feature map is zeroed with prob 0.1
- If channel 5 is zeroed, ALL 32×107 = 3,424 values become 0
- Only 10% of feature maps are zeroed

**Why spatial dropout for CNNs?** Adjacent pixels in a conv feature map are spatially correlated — dropping one pixel doesn't force the network to learn anything because adjacent pixels can fill in the information. Dropping an entire feature map forces the network to learn the same pattern using other channels (redundancy).

---

### Q49: What is `inplace=True` in ReLU?

**Code:**
```python
nn.ReLU(inplace=True)
# vs
nn.ReLU(inplace=False)  # default
```

**Answer:**

`inplace=True` modifies the input tensor directly instead of creating a new tensor:
```python
# inplace=False (default):
out = relu(x)    # allocates new tensor for output; x unchanged

# inplace=True:
relu_(x)         # modifies x directly; no new allocation
```

**Benefit:** Saves memory — no need to allocate a new tensor for the ReLU output. Important when processing large spectrograms (128×431) with batch size 32.

**Risk:** Can cause issues with autograd if the same tensor is needed in multiple backward passes. Generally safe when used as shown in sequential modules.

---

### Q50: What is `pin_memory=True` in DataLoader?

**Code:**
```python
DataLoader(dataset, batch_size=8, pin_memory=True)
```

**Answer:**

`pin_memory=True` allocates batches in **page-locked (pinned) CPU memory** instead of regular pageable memory.

**Why faster for GPU?** GPU DMA (Direct Memory Access) can directly transfer pinned CPU memory to GPU without going through the OS. Regular pageable memory must first be copied to a temporary pinned buffer, then transferred — extra step.

**Result:** Data transfer from CPU to GPU is ~2× faster with pinned memory.

**When to use:** Only beneficial with CUDA GPU. On CPU or MPS, set `pin_memory=False` (default). In the notebook:
```python
ast_train_dl = DataLoader(ast_train_ds, batch_size=8, pin_memory=True, num_workers=0)
```
`pin_memory=True` helps when training on Kaggle's T4 GPU (CUDA device).

---

### Q51: What is `num_workers=0` in DataLoader?

**Code:**
```python
DataLoader(dataset, batch_size=8, num_workers=0, pin_memory=True)
```

**Answer:**

`num_workers` = number of parallel CPU worker processes for data loading.

- `num_workers=0`: data loading happens in the **main process** (synchronous, no multiprocessing)
- `num_workers=4`: 4 separate processes pre-fetch batches in parallel while GPU trains

**Why `num_workers=0` for AST?** Two reasons:
1. **Kaggle environment**: multiprocessing with `fork` can cause issues in Jupyter notebooks (CUDA/librosa state not safely copied to child processes)
2. **Online augmentation with librosa**: `librosa.resample()` inside `__getitem__` is not fork-safe in all environments

**When to use >0:** Pre-computed spectrograms (Cell 13 CNN training) could use `num_workers=2-4` safely because it's just numpy array lookups.

---

### Q52: What is `torch.no_grad()` vs `torch.inference_mode()`?

**Answer:**

Both disable gradient computation. `inference_mode()` (newer) is slightly more aggressive — it also disables tracking of version counters and other overhead:

```python
# Old way (still works, more common in tutorials):
with torch.no_grad():
    outputs = model(x)

# New way (faster, preferred for pure inference):
with torch.inference_mode():
    outputs = model(x)
```

We used `torch.no_grad()` in the notebook because it's more universally compatible.

---

## Additional Quick-Fire Questions

### Q53: What is `torch.nn.utils.clip_grad_norm_` vs `clip_grad_value_`?

```python
# clip_grad_norm_: clips by total L2 norm of all gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# If total_norm > 1.0: ALL gradients scaled by (1.0/total_norm)

# clip_grad_value_: clips each individual gradient value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
# Each gradient clipped to [-0.5, 0.5] independently
```

`clip_grad_norm_` is preferred: maintains gradient direction (only scales magnitude), doesn't distort relative gradient sizes. `clip_grad_value_` clips each independently which can distort the gradient direction.

---

### Q54: What is `logits.argmax(1)`? What does `dim=1` mean?

```python
preds = logits.argmax(dim=1)
# logits: (B, 10) — batch of 10-class logits
# dim=1: take argmax ALONG dimension 1 (the class dimension)
# result: (B,) — index of highest logit for each sample
```

`dim=0`: argmax across batch (which sample has highest logit for each class) — rarely used
`dim=1`: argmax across classes (which class has highest logit for each sample) — what we want

---

### Q55: What is `F.softmax(logits, dim=-1)` vs `dim=0`?

```python
probs = F.softmax(logits, dim=-1)
# logits: (B, 10)
# dim=-1: apply softmax along LAST dimension (class dim, size 10)
# Each row sums to 1.0: sum(probs[i, :]) = 1.0 ✓

# dim=0 would apply softmax along batch dimension (WRONG for classification)
# Each column would sum to 1.0: sum(probs[:, j]) = 1.0 ✗ (not what we want)
```

`dim=-1` is equivalent to `dim=1` for a 2D tensor. Using `-1` (last dimension) is safer — works for any number of dimensions.

---

### Q56: What is the shape of input to AST's ASTFeatureExtractor?

```python
audio_16k = np.array(...)   # shape: (160000,) — 10s × 16000Hz

inputs = ast_fe(audio_16k, sampling_rate=16000, return_tensors='pt')
# inputs['input_values'].shape: (1, 1024, 128)
#                                ↑    ↑     ↑
#                              batch  time  mel_bands
```

The 1D audio (160,000 samples) → 2D mel spectrogram (1024 × 128) via the feature extractor's internal STFT/mel computation. This is then split into 16×16 patches, producing 512 tokens per sequence.

---

### Q57: What does `scatter_` do in LabelSmoothingCrossEntropy?

```python
smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
smooth.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
```

`scatter_(dim, index, value)` writes `value` at positions specified by `index` along `dim`:

```
Before scatter_:
smooth = [[0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011],  ← row 0
          [0.011, 0.011, ...                                                     ]]  ← row 1

targets = [3, 7]  → unsqueeze(-1) → [[3], [7]]

After scatter_(-1, [[3],[7]], 0.9):
smooth = [[0.011, 0.011, 0.011, 0.9,   0.011, 0.011, 0.011, 0.011, 0.011, 0.011],  ← class 3 = 0.9
          [0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.011, 0.9,   0.011, 0.011]]  ← class 7 = 0.9
```

It's a vectorised in-place assignment: for each row, set the target class position to 1-ε.

---

### Q58: What happens if you don't call `model.eval()` before validation?

**Answer:**

Two things go wrong:

1. **Dropout still active:** 30% of neurons randomly zeroed → predictions have random noise → val F1 is artificially low AND non-reproducible (different every run).

2. **BatchNorm uses batch statistics:** For small validation batches (e.g., 1 sample), batch mean/var is just that one sample's stats — not representative. The model sees wildly different normalisation than what it was trained with → wrong activations → wrong predictions.

**Concrete impact:** In our experiments, not calling `model.eval()` typically drops val F1 by 10–20% compared to the true model performance.

---

### Q59: What is `torch.mps.synchronize()` and why is it critical?

**Code:**
```python
def sync_and_clear(device):
    if device.type == 'mps':
        torch.mps.synchronize()   # MUST come first
        torch.mps.empty_cache()
```

**Answer:**

Apple MPS (Metal Performance Shaders) executes GPU operations **asynchronously** — the Python code continues while the GPU is still computing. `empty_cache()` frees memory that the GPU might still be using.

**Without synchronize:**
```
Python: "free this memory"
GPU:    "wait, I'm still computing with that tensor!"
Result: DOUBLE-FREE crash → exit code 134 (Killed: 9)
```

**With synchronize:**
```
Python: "wait until all GPU ops finish" (blocks here)
GPU:    [finishes all pending operations]
Python: "OK, now free the cache safely"
```

Same issue with CUDA (`torch.cuda.synchronize()`), though CUDA is more forgiving.

---

### Q60: What is gradient accumulation? Write the code pattern.

**Code from notebook:**
```python
AST_GRAD_ACCUM = 2
optimizer.zero_grad()

for bi, (iv, labels) in enumerate(train_dl):
    loss = criterion(outputs.logits, labels) / AST_GRAD_ACCUM   # ← divide here
    scaler.scale(loss).backward()

    if (bi + 1) % AST_GRAD_ACCUM == 0:     # every 2 batches
        scaler.unscale_(optimizer)
        clip_grad_norm_(ast_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()               # reset after update
```

**Why divide loss by ACCUM_STEPS?**
Without division: total gradient = ∇L₁ + ∇L₂ (sum of 2 batches)
With division: total gradient = ∇L₁/2 + ∇L₂/2 = (∇L₁ + ∇L₂)/2 (average)
This is equivalent to computing the gradient on a single batch of double size.

**Why use gradient accumulation?** AST (86M params) with batch=8 fits in GPU memory. Batch=16 would require ~2× the activation memory. Accumulation achieves effective batch=16 at the memory cost of batch=8.

---

### Q61: What is AMP (Automatic Mixed Precision)?

**Code:**
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = ast_model(input_values=iv)   # runs in float16
    loss = criterion(outputs.logits, labels)

scaler.scale(loss).backward()   # scale before backward (prevent underflow)
scaler.unscale_(optimizer)      # unscale before clip_grad_norm
clip_grad_norm_(ast_model.parameters(), 1.0)
scaler.step(optimizer)          # update weights
scaler.update()                 # adjust scale factor
```

**float16 vs float32:**
- float16: 2 bytes, range [6×10⁻⁵, 65504]
- float32: 4 bytes, range [1×10⁻³⁸, 3×10³⁸]

**Problem:** Some gradients during backprop are very small (e.g., 1×10⁻⁶). In float16, this rounds to zero (underflow) → gradient disappears → weights don't update.

**GradScaler solution:**
1. Multiply loss by large factor S (e.g., 65536) before backward
2. Gradients are now S× larger → no underflow in float16
3. Before optimizer step: divide all gradients by S → correct update magnitude
4. If overflow detected → skip step, reduce S → adaptive scaling

**Speed benefit:** float16 arithmetic is ~2× faster on modern GPUs (Tensor Cores). Memory: ~2× reduction in activations during forward pass.

---

## Bonus: What if examiner asks to modify something live

### Scenario 1: "Change the number of CNN blocks to 3"

```python
# Remove Block 4, adjust classifier input
self.features = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),   # 128 channels now
)
self.classifier = nn.Sequential(
    nn.Flatten(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(),  # ← 128 not 256
    nn.Linear(64, num_classes),
)
```

### Scenario 2: "Use GlobalMaxPool instead of GlobalAvgPool"

```python
# Replace AdaptiveAvgPool2d with:
nn.AdaptiveMaxPool2d((1, 1))   # takes maximum instead of mean
```

### Scenario 3: "Add a residual connection to SimpleCNN"

```python
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)   # ← skip connection
```

### Scenario 4: "Change the loss to standard CrossEntropyLoss"

```python
# In Cell 13, replace:
cnn_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
# With:
cnn_criterion = nn.CrossEntropyLoss()
# No other changes needed — same interface: criterion(logits, labels)
```

### Scenario 5: "Add early stopping to CNN training"

```python
best_val_f1 = 0.0
patience_counter = 0
PATIENCE = 5

for epoch in range(1, CNN_EPOCHS + 1):
    # ... (train + val as before) ...

    if vf1 > best_val_f1:
        best_val_f1 = vf1
        patience_counter = 0
        torch.save(cnn.state_dict(), 'best_cnn.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# Reload best
cnn.load_state_dict(torch.load('best_cnn.pt'))
```

---

*This file covers every known examiner question pattern. The highest-probability questions are: training loop code (Q24), CNN from scratch (Q41), LSTM code (Q39), BatchNorm explanation (Q16), AdaptiveAvgPool (Q19), gradient clipping (Q25), delta/delta2 (Q11), RMS (Q6), audio gain (Q7), and AST loading (Q34).*
