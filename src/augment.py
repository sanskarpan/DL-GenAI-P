"""
Audio augmentation pipeline — the CRITICAL component for bridging train/test gap.

The test set is: stems from same genre mixed + ESC-50 noise + tempo variations.
We replicate this process during training to match the test distribution.
"""
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

from . import config


# ─── Audio Loading Utilities ──────────────────────────────────────────────────

def load_audio(path: str | Path, sr: int = config.SAMPLE_RATE,
               duration: Optional[float] = None,
               offset: float = 0.0) -> np.ndarray:
    """Load audio file robustly, returning mono float32 array."""
    try:
        audio, _ = librosa.load(str(path), sr=sr, mono=True,
                                duration=duration, offset=offset)
    except Exception as e:
        # Fallback: soundfile + resample
        data, orig_sr = sf.read(str(path), always_2d=True)
        audio = data.mean(axis=1).astype(np.float32)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim to target_len samples."""
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]
    return audio


def random_crop(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Randomly crop audio to target_len samples. Pads if too short."""
    if len(audio) <= target_len:
        return pad_or_trim(audio, target_len)
    start = random.randint(0, len(audio) - target_len)
    return audio[start:start + target_len]


# ─── ESC-50 Noise Pool ────────────────────────────────────────────────────────
# Pre-load ESC-50 audio into memory to avoid slow per-sample disk reads.
# We cache a random subset of 300 clips (~26 MB at 22050 Hz mono).

_esc50_paths: Optional[List[Path]] = None
_esc50_audio_cache: Optional[List[np.ndarray]] = None
_ESC50_CACHE_SIZE = 300   # clips to keep in memory


def get_esc50_paths() -> List[Path]:
    """Return list of all ESC-50 audio file paths (paths cached once)."""
    global _esc50_paths
    if _esc50_paths is None:
        _esc50_paths = sorted(config.ESC50_AUDIO_DIR.glob("*.wav"))
        if not _esc50_paths:
            raise FileNotFoundError(f"No ESC-50 files found in {config.ESC50_AUDIO_DIR}")
    return _esc50_paths


def _build_esc50_audio_cache() -> List[np.ndarray]:
    """
    Pre-load a random subset of ESC-50 clips into RAM.
    Called once on first noise request; subsequent calls use the cache.
    300 clips × ~5s × 22050 Hz × 4 bytes ≈ 66 MB — fast and memory-safe.
    """
    global _esc50_audio_cache
    if _esc50_audio_cache is not None:
        return _esc50_audio_cache

    paths = get_esc50_paths()
    rng = random.Random(config.SEED)
    selected = rng.sample(paths, min(_ESC50_CACHE_SIZE, len(paths)))

    cache = []
    for p in selected:
        try:
            audio = load_audio(p, sr=config.SAMPLE_RATE)
            cache.append(audio)
        except Exception:
            pass

    _esc50_audio_cache = cache
    return cache


def load_random_noise(target_len: int) -> np.ndarray:
    """Return a random ESC-50 clip from memory cache (no disk I/O after init)."""
    cache = _build_esc50_audio_cache()
    noise = random.choice(cache)
    return random_crop(noise, target_len)


# ─── Noise Addition ───────────────────────────────────────────────────────────

def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray,
                     snr_db: float) -> np.ndarray:
    """
    Add noise to signal at a specific Signal-to-Noise Ratio (dB).
    SNR = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)
    return signal + noise_scaled


def add_random_noise(audio: np.ndarray,
                     snr_min: float = config.NOISE_SNR_MIN,
                     snr_max: float = config.NOISE_SNR_MAX) -> np.ndarray:
    """Add random ESC-50 noise at random SNR."""
    noise = load_random_noise(len(audio))
    snr_db = random.uniform(snr_min, snr_max)
    return add_noise_at_snr(audio, noise, snr_db)


# ─── Stem-Level Augmentations ─────────────────────────────────────────────────

def apply_volume_jitter(audio: np.ndarray,
                        min_db: float = config.STEM_VOLUME_MIN,
                        max_db: float = config.STEM_VOLUME_MAX) -> np.ndarray:
    """Apply random volume change in dB."""
    db_change = random.uniform(min_db, max_db)
    gain = 10 ** (db_change / 20)
    return audio * gain


def apply_tempo_stretch(audio: np.ndarray,
                        rate_min: float = config.TEMPO_STRETCH_MIN,
                        rate_max: float = config.TEMPO_STRETCH_MAX) -> np.ndarray:
    """Apply random time stretching without changing pitch."""
    rate = random.uniform(rate_min, rate_max)
    if abs(rate - 1.0) < 0.01:
        return audio
    try:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched
    except Exception:
        return audio


def apply_pitch_shift(audio: np.ndarray,
                      n_steps_min: float = -2,
                      n_steps_max: float = 2) -> np.ndarray:
    """Apply random pitch shift in semitones."""
    n_steps = random.uniform(n_steps_min, n_steps_max)
    try:
        return librosa.effects.pitch_shift(audio, sr=config.SAMPLE_RATE,
                                           n_steps=n_steps)
    except Exception:
        return audio


# ─── Genre Stem Index ─────────────────────────────────────────────────────────

_stem_index_cache: Optional[dict] = None


def build_stem_index() -> dict:
    """
    Build index: {genre: [list of song dirs]}.
    Only built once and cached.
    """
    global _stem_index_cache
    if _stem_index_cache is not None:
        return _stem_index_cache

    index = {}
    for genre in config.GENRES:
        genre_dir = config.GENRES_DIR / genre
        if not genre_dir.exists():
            continue
        song_dirs = sorted([d for d in genre_dir.iterdir() if d.is_dir()])
        index[genre] = song_dirs

    _stem_index_cache = index
    return index


def get_stem_path(song_dir: Path, stem_name: str) -> Optional[Path]:
    """Get path to a stem file, trying both 'other.wav' and 'others.wav'."""
    p = song_dir / stem_name
    if p.exists():
        return p
    # Fallback for naming inconsistency
    if stem_name == "other.wav":
        alt = song_dir / "others.wav"
        if alt.exists():
            return alt
    return None


# ─── Synthetic Mashup Creation ────────────────────────────────────────────────

def create_synthetic_mashup(
        genre: str,
        stem_index: dict,
        target_duration: float = config.CHUNK_DURATION,
        n_mix: Optional[int] = None,
        add_noise: bool = True,
        apply_tempo: bool = True,
        apply_volume: bool = True,
) -> Tuple[np.ndarray, str]:
    """
    Create a synthetic mashup by mixing stems from different songs of the SAME genre.

    This EXACTLY replicates the test data generation process:
    - Select stems from different songs of same genre
    - Apply tempo stretch for synchronization
    - Mix stems together
    - Add ESC-50 noise at random intensity

    Args:
        genre: Target genre name
        stem_index: Index of song directories per genre
        target_duration: Duration of output audio in seconds
        n_mix: Number of songs to mix (random 2-4 if None)
        add_noise: Whether to add ESC-50 noise
        apply_tempo: Whether to apply tempo augmentation
        apply_volume: Whether to apply volume jitter per stem

    Returns:
        (audio_array, genre_label)
    """
    target_len = int(target_duration * config.SAMPLE_RATE)
    songs = stem_index.get(genre, [])

    if len(songs) < 2:
        raise ValueError(f"Not enough songs for genre '{genre}' (found {len(songs)})")

    # Pick number of songs to mix
    if n_mix is None:
        n_mix = random.randint(config.MIX_STEMS_MIN, min(config.MIX_STEMS_MAX, len(songs)))
    n_mix = min(n_mix, len(songs))

    selected_songs = random.sample(songs, n_mix)

    # Pick which stems to use from each song
    # Strategy: take different stems from different songs to maximize diversity
    stem_pool = config.STEM_NAMES[:]
    random.shuffle(stem_pool)

    mixed = np.zeros(target_len, dtype=np.float32)
    n_mixed = 0

    for i, song_dir in enumerate(selected_songs):
        # Cycle through stem types across songs
        stem_name = stem_pool[i % len(stem_pool)]
        stem_path = get_stem_path(song_dir, stem_name)

        if stem_path is None:
            # Try any available stem
            for sn in config.STEM_NAMES:
                stem_path = get_stem_path(song_dir, sn)
                if stem_path is not None:
                    break

        if stem_path is None:
            continue

        # Load stem (slightly longer for random crop after tempo stretch)
        load_dur = target_duration * 1.3 if apply_tempo else target_duration
        try:
            stem_audio = load_audio(stem_path, sr=config.SAMPLE_RATE,
                                    duration=load_dur)
        except Exception:
            continue

        if len(stem_audio) < 1000:
            continue

        # Apply tempo stretch (simulates synchronization)
        if apply_tempo and random.random() > 0.3:
            stem_audio = apply_tempo_stretch(stem_audio)

        # Random crop to target length
        stem_audio = random_crop(stem_audio, target_len)

        # Apply volume jitter
        if apply_volume:
            stem_audio = apply_volume_jitter(stem_audio)

        mixed = mixed + stem_audio
        n_mixed += 1

    if n_mixed == 0:
        # Fallback: return silence (shouldn't happen)
        return np.zeros(target_len, dtype=np.float32), genre

    # Normalize the mix
    max_val = np.abs(mixed).max()
    if max_val > 0:
        mixed = mixed / max_val * 0.9  # leave headroom

    # Add ESC-50 noise (simulates test conditions)
    if add_noise:
        mixed = add_random_noise(mixed)
        # Re-normalize after noise
        max_val = np.abs(mixed).max()
        if max_val > 0:
            mixed = mixed / max_val * 0.9

    return mixed.astype(np.float32), genre


# ─── Training Data Split ──────────────────────────────────────────────────────

def split_songs_train_val(stem_index: dict,
                          val_per_genre: int = config.VAL_SONGS_PER_GENRE,
                          seed: int = config.SEED
                          ) -> Tuple[dict, dict]:
    """
    Split songs into train/val sets per genre.
    Returns two dicts with same structure as stem_index.
    """
    rng = random.Random(seed)
    train_index = {}
    val_index = {}

    for genre, songs in stem_index.items():
        songs_shuffled = songs.copy()
        rng.shuffle(songs_shuffled)
        val_songs = songs_shuffled[:val_per_genre]
        train_songs = songs_shuffled[val_per_genre:]
        val_index[genre] = val_songs
        train_index[genre] = train_songs

    return train_index, val_index


# ─── Batch Generation ─────────────────────────────────────────────────────────

def generate_synthetic_batch(
        stem_index: dict,
        n_samples: int,
        add_noise: bool = True,
        apply_tempo: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate a batch of synthetic mashups from all genres uniformly.

    Returns:
        (list of audio arrays, list of genre labels)
    """
    genres = list(stem_index.keys())
    n_per_genre = n_samples // len(genres)
    remainder = n_samples % len(genres)

    audios, labels = [], []

    for i, genre in enumerate(genres):
        n = n_per_genre + (1 if i < remainder else 0)
        for _ in range(n):
            try:
                audio, label = create_synthetic_mashup(
                    genre=genre,
                    stem_index=stem_index,
                    add_noise=add_noise,
                    apply_tempo=apply_tempo,
                )
                audios.append(audio)
                labels.append(label)
            except Exception as e:
                print(f"Warning: failed to create mashup for {genre}: {e}")

    return audios, labels
