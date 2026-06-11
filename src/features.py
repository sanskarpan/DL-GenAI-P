"""
Feature extraction for audio classification.
Supports:
  - Mel spectrogram (2D, for CNN input)
  - MFCC statistics (1D, for classical ML)
"""
import numpy as np
import librosa

from . import config


# ─── Mel Spectrogram ──────────────────────────────────────────────────────────

def audio_to_melspec(audio: np.ndarray,
                     sr: int = config.SAMPLE_RATE,
                     n_mels: int = config.N_MELS,
                     n_fft: int = config.N_FFT,
                     hop_length: int = config.HOP_LENGTH,
                     fmin: float = config.FMIN,
                     fmax: float = config.FMAX,
                     top_db: float = 80.0) -> np.ndarray:
    """
    Convert audio to log-mel spectrogram.

    Returns:
        np.ndarray of shape (n_mels, T) — log-amplitude mel spectrogram
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )
    # Convert to dB scale
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    return mel_db.astype(np.float32)


def normalize_melspec(melspec: np.ndarray) -> np.ndarray:
    """
    Normalize mel spectrogram to [0, 1] range.
    Safe division with epsilon to avoid NaN.
    """
    min_val = melspec.min()
    max_val = melspec.max()
    if max_val - min_val < 1e-6:
        return np.zeros_like(melspec)
    return (melspec - min_val) / (max_val - min_val)


def melspec_for_cnn(audio: np.ndarray,
                    target_frames: int = config.N_FRAMES,
                    normalize: bool = True) -> np.ndarray:
    """
    Extract mel spectrogram and prepare for CNN input.
    Pads or trims to fixed width (target_frames).

    Returns:
        np.ndarray of shape (1, N_MELS, target_frames) — channel-first
    """
    mel = audio_to_melspec(audio)

    # Pad or trim time dimension
    if mel.shape[1] < target_frames:
        pad = target_frames - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant',
                     constant_values=mel.min())
    else:
        mel = mel[:, :target_frames]

    if normalize:
        mel = normalize_melspec(mel)

    # Add channel dimension: (1, N_MELS, T)
    return mel[np.newaxis, :, :].astype(np.float32)


# ─── MFCC Features (for Classical ML) ────────────────────────────────────────

def extract_mfcc_features(audio: np.ndarray,
                          sr: int = config.SAMPLE_RATE,
                          n_mfcc: int = config.N_MFCC,
                          include_delta: bool = config.MFCC_INCLUDE_DELTA
                          ) -> np.ndarray:
    """
    Extract MFCC statistics for classical ML.

    Returns:
        1D feature vector:
          - n_mfcc × 2 (mean, std) if include_delta=False
          - n_mfcc × 6 (mean, std for MFCC + delta + delta2) if True
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features = [mfcc.mean(axis=1), mfcc.std(axis=1)]

    if include_delta:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features += [delta.mean(axis=1), delta.std(axis=1),
                     delta2.mean(axis=1), delta2.std(axis=1)]

    return np.concatenate(features).astype(np.float32)


def extract_chroma_features(audio: np.ndarray,
                            sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Extract chroma STFT statistics (12 × 2 = 24 features)."""
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=config.N_FFT,
                                          hop_length=config.HOP_LENGTH)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype(np.float32)


def extract_spectral_features(audio: np.ndarray,
                              sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Extract spectral statistics:
    - Spectral centroid (1×2)
    - Spectral bandwidth (1×2)
    - Spectral rolloff (1×2)
    - Spectral contrast (7×2)
    - RMS energy (1×2)
    Total: 24 features
    """
    hop = config.HOP_LENGTH
    n_fft = config.N_FFT

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=audio, hop_length=hop)

    parts = [centroid, bandwidth, rolloff, contrast, rms]
    features = []
    for p in parts:
        features += [p.mean(axis=1), p.std(axis=1)]
    return np.concatenate(features).astype(np.float32)


def extract_full_features(audio: np.ndarray,
                          sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Extract combined feature vector for classical ML:
    MFCCs + chroma + spectral = comprehensive descriptor.
    """
    mfcc_feat = extract_mfcc_features(audio, sr)
    chroma_feat = extract_chroma_features(audio, sr)
    spectral_feat = extract_spectral_features(audio, sr)
    return np.concatenate([mfcc_feat, chroma_feat, spectral_feat]).astype(np.float32)




# ─── SpecAugment (for CNN training) ──────────────────────────────────────────

def spec_augment(melspec: np.ndarray,
                 time_mask_max: int = 40,
                 freq_mask_max: int = 20,
                 n_time_masks: int = 2,
                 n_freq_masks: int = 2) -> np.ndarray:
    """
    Apply SpecAugment: random masking of time and frequency bands.
    Input: (C, F, T) or (F, T)

    Reference: https://arxiv.org/abs/1904.08779
    """
    spec = melspec.copy()

    # Handle both (C, F, T) and (F, T)
    if spec.ndim == 3:
        _, n_freq, n_time = spec.shape
        mask_val = spec.min()

        for _ in range(n_time_masks):
            t_width = np.random.randint(1, min(time_mask_max, n_time))
            t_start = np.random.randint(0, n_time - t_width)
            spec[:, :, t_start:t_start + t_width] = mask_val

        for _ in range(n_freq_masks):
            f_width = np.random.randint(1, min(freq_mask_max, n_freq))
            f_start = np.random.randint(0, n_freq - f_width)
            spec[:, f_start:f_start + f_width, :] = mask_val
    else:
        n_freq, n_time = spec.shape
        mask_val = spec.min()

        for _ in range(n_time_masks):
            t_width = np.random.randint(1, min(time_mask_max, n_time))
            t_start = np.random.randint(0, n_time - t_width)
            spec[:, t_start:t_start + t_width] = mask_val

        for _ in range(n_freq_masks):
            f_width = np.random.randint(1, min(freq_mask_max, n_freq))
            f_start = np.random.randint(0, n_freq - f_width)
            spec[f_start:f_start + f_width, :] = mask_val

    return spec


def mixup(spec1: np.ndarray, spec2: np.ndarray,
          label1: int, label2: int,
          alpha: float = 0.3) -> tuple:
    """
    Mixup augmentation: interpolate between two spectrograms.
    Returns mixed spectrogram and soft labels.
    """
    lam = np.random.beta(alpha, alpha)
    mixed_spec = lam * spec1 + (1 - lam) * spec2
    return mixed_spec, label1, label2, lam
