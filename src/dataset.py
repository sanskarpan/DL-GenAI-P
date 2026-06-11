"""
PyTorch Dataset classes for Messy Mashup competition.

Two datasets:
1. SyntheticMashupDataset — online augmentation during training
2. TestMashupDataset — for inference on test set
"""
import random
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from .augment import (create_synthetic_mashup, load_audio,
                      random_crop, pad_or_trim, build_stem_index,
                      split_songs_train_val)
from .features import melspec_for_cnn, spec_augment


class SyntheticMashupDataset(Dataset):
    """
    Creates synthetic mashup training examples on-the-fly.
    Each call to __getitem__ generates a NEW synthetic mashup.

    This online augmentation means we get different samples each epoch.
    """

    def __init__(self,
                 stem_index: Dict[str, List[Path]],
                 n_samples: int = config.N_SYNTHETIC_TRAIN,
                 add_noise: bool = True,
                 apply_tempo: bool = True,
                 use_spec_augment: bool = True,
                 target_frames: int = config.N_FRAMES,
                 seed: Optional[int] = None):
        """
        Args:
            stem_index: Dict mapping genre -> list of song directories
            n_samples: Virtual dataset size (how many samples per epoch)
            add_noise: Add ESC-50 noise to synthetic mashups
            apply_tempo: Apply tempo stretching
            use_spec_augment: Apply SpecAugment during training
            target_frames: Fixed time frames for mel spec
            seed: Optional seed for reproducibility
        """
        self.stem_index = stem_index
        self.n_samples = n_samples
        self.add_noise = add_noise
        self.apply_tempo = apply_tempo
        self.use_spec_augment = use_spec_augment
        self.target_frames = target_frames
        self.genres = sorted(stem_index.keys())
        self.n_classes = len(self.genres)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Sample genre uniformly (balanced classes)
        genre = self.genres[idx % self.n_classes]

        # Create synthetic mashup
        audio, label = create_synthetic_mashup(
            genre=genre,
            stem_index=self.stem_index,
            target_duration=config.CHUNK_DURATION,
            add_noise=self.add_noise,
            apply_tempo=self.apply_tempo,
        )

        # Extract mel spectrogram: (1, N_MELS, T)
        spec = melspec_for_cnn(audio, target_frames=self.target_frames)

        # Apply SpecAugment (additional augmentation in feature space)
        if self.use_spec_augment:
            spec = spec_augment(spec)

        label_idx = config.GENRE_TO_IDX[label]

        return torch.FloatTensor(spec), label_idx


class TestMashupDataset(Dataset):
    """
    Dataset for test mashup files.
    Each file is loaded and split into chunks; predictions are averaged.
    """

    def __init__(self,
                 file_paths: List[Path],
                 ids: List[str],
                 n_chunks: int = config.TTA_CHUNKS,
                 target_frames: int = config.N_FRAMES):
        """
        Args:
            file_paths: List of paths to test mashup WAV files
            ids: Corresponding test IDs
            n_chunks: Number of random crops to use per file
            target_frames: Fixed time frames for mel spec
        """
        self.file_paths = file_paths
        self.ids = ids
        self.n_chunks = n_chunks
        self.target_frames = target_frames

        # Expand: each file becomes n_chunks entries
        self.expanded_paths = []
        self.expanded_ids = []
        self.chunk_indices = []
        for file_path, id_ in zip(file_paths, ids):
            for chunk_i in range(n_chunks):
                self.expanded_paths.append(file_path)
                self.expanded_ids.append(id_)
                self.chunk_indices.append(chunk_i)

    def __len__(self) -> int:
        return len(self.expanded_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.expanded_paths[idx]
        id_ = self.expanded_ids[idx]
        chunk_i = self.chunk_indices[idx]

        try:
            audio = load_audio(path, sr=config.SAMPLE_RATE)
        except Exception as e:
            print(f"Warning: failed to load {path}: {e}")
            audio = np.zeros(config.CHUNK_SAMPLES, dtype=np.float32)

        # Extract chunk: for chunk_i, use evenly-spaced start positions
        target_len = config.CHUNK_SAMPLES
        if len(audio) <= target_len:
            chunk = pad_or_trim(audio, target_len)
        else:
            # Evenly spaced chunks for deterministic TTA
            max_start = len(audio) - target_len
            if self.n_chunks == 1:
                start = 0
            else:
                start = int(chunk_i * max_start / (self.n_chunks - 1))
            chunk = audio[start:start + target_len]

        # Extract mel spectrogram
        spec = melspec_for_cnn(chunk, target_frames=self.target_frames)

        return torch.FloatTensor(spec), id_


class PrecomputedDataset(Dataset):
    """
    Dataset from pre-computed mel spectrograms (numpy arrays).
    Used when features are pre-extracted to disk for speed.

    target_size: optional (H, W) to resize each spec — use (64, 64) to
    feed EfficientNet with much smaller inputs for faster training.
    """

    def __init__(self,
                 specs: np.ndarray,
                 labels: np.ndarray,
                 use_spec_augment: bool = False,
                 target_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            specs: (N, 1, F, T) mel spectrograms
            labels: (N,) integer class labels
            use_spec_augment: Apply SpecAugment during training
            target_size: (H, W) to resize each spectrogram. None = no resize.
        """
        self.specs = specs
        self.labels = labels
        self.use_spec_augment = use_spec_augment
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spec = self.specs[idx].copy()
        if self.use_spec_augment:
            spec = spec_augment(spec)
        tensor = torch.FloatTensor(spec)
        if self.target_size is not None:
            # Resize (1, H, W) → (1, target_H, target_W) using bilinear interpolation
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=self.target_size,
                mode='bilinear', align_corners=False
            ).squeeze(0)
        return tensor, int(self.labels[idx])
