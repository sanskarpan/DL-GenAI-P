"""
Inference pipeline for test mashup files.

Strategy:
- Load each test mashup
- Split into N overlapping chunks
- Extract mel spectrogram per chunk
- Average softmax predictions across chunks (TTA)
- Generate submission CSV
"""
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .config import get_device
from .dataset import TestMashupDataset


@torch.no_grad()
def predict_test_set(
        model: nn.Module,
        test_csv_path: Path = config.TEST_CSV,
        n_chunks: int = config.TTA_CHUNKS,
        batch_size: int = config.BATCH_SIZE,
        device: Optional[torch.device] = None,
        data_dir: Path = config.DATA_DIR,
) -> pd.DataFrame:
    """
    Run inference on all test mashup files.

    Args:
        model: Trained PyTorch model
        test_csv_path: Path to test.csv
        n_chunks: Number of audio chunks per file for TTA
        batch_size: Batch size for inference
        device: Compute device
        data_dir: Root data directory (test.csv uses relative paths)

    Returns:
        DataFrame with columns [id, genre]
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    # Load test CSV
    test_df = pd.read_csv(test_csv_path)
    ids = test_df['id'].astype(str).str.zfill(4).tolist()
    filenames = test_df['filename'].tolist()

    # Build absolute paths
    file_paths = [data_dir / fname for fname in filenames]

    # Verify files exist
    missing = [p for p in file_paths if not p.exists()]
    if missing:
        print(f"Warning: {len(missing)} test files not found! First 3: {missing[:3]}")

    # Create dataset with TTA (multiple chunks per file)
    dataset = TestMashupDataset(
        file_paths=file_paths,
        ids=ids,
        n_chunks=n_chunks,
        target_frames=config.N_FRAMES,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,        # 0 workers for stability on macOS
        pin_memory=False,
    )

    # Aggregate predictions per ID
    # {id: [list of softmax probability arrays]}
    id_to_probs: Dict[str, List[np.ndarray]] = defaultdict(list)

    print(f"Running inference on {len(file_paths)} files ({len(dataset)} total chunks)...")

    for specs, batch_ids in tqdm(loader, desc="Inference"):
        specs = specs.to(device, non_blocking=True)
        logits = model(specs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for i, id_ in enumerate(batch_ids):
            id_to_probs[id_].append(probs[i])

    # Average probabilities across chunks per file
    results = []
    for id_ in ids:
        if id_ not in id_to_probs:
            # Fallback: uniform prediction
            avg_probs = np.ones(config.NUM_CLASSES) / config.NUM_CLASSES
        else:
            avg_probs = np.mean(id_to_probs[id_], axis=0)

        pred_idx = np.argmax(avg_probs)
        pred_genre = config.IDX_TO_GENRE[pred_idx]
        results.append({"id": id_, "genre": pred_genre})

    submission_df = pd.DataFrame(results)
    return submission_df


def save_submission(df: pd.DataFrame,
                    output_path: Optional[Path] = None,
                    suffix: str = "") -> Path:
    """
    Save submission CSV in competition format.
    Validates format before saving.
    """
    # Validate
    assert "id" in df.columns, "Missing 'id' column"
    assert "genre" in df.columns, "Missing 'genre' column"
    valid_genres = set(config.GENRES)
    invalid = df[~df["genre"].isin(valid_genres)]
    if len(invalid) > 0:
        raise ValueError(f"Invalid genres in submission: {invalid['genre'].unique()}")

    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.SUBMISSIONS_DIR / f"submission_{ts}{suffix}.csv"

    df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Genre distribution:\n{df['genre'].value_counts().to_string()}")

    return output_path


def generate_random_baseline() -> pd.DataFrame:
    """Generate random baseline submission (sanity check)."""
    test_df = pd.read_csv(config.TEST_CSV)
    ids = test_df['id'].astype(str).str.zfill(4).tolist()
    genres = [np.random.choice(config.GENRES) for _ in ids]
    return pd.DataFrame({"id": ids, "genre": genres})


def generate_majority_baseline() -> pd.DataFrame:
    """Generate majority class baseline submission."""
    test_df = pd.read_csv(config.TEST_CSV)
    ids = test_df['id'].astype(str).str.zfill(4).tolist()
    # With balanced classes, any genre gives ~10% F1
    # Use 'rock' as most common in GTZAN-style datasets
    genres = ["rock"] * len(ids)
    return pd.DataFrame({"id": ids, "genre": genres})
