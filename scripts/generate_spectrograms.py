"""
Pre-generate synthetic mashup mel spectrograms for CNN training.

Why: CNN training needs fast data loading. Creating synthetic mashups on-the-fly
(even at 0.07s each) still results in slow epoch times when combined with
mel spec extraction (~0.1s). Pre-generating once (~14 min) enables fast
training from disk (<1ms per sample load).

Usage:
    python scripts/generate_spectrograms.py --n_train 5000 --n_val 1000
"""
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src import config
from src.config import set_seed
from src.augment import build_stem_index, split_songs_train_val, create_synthetic_mashup, _build_esc50_audio_cache
from src.features import melspec_for_cnn


def generate_and_save(stem_index: dict, n_samples: int, save_path: Path,
                      add_noise: bool = True, apply_tempo: bool = False) -> None:
    """
    Generate n_samples synthetic mashups and save mel spectrograms to disk.

    apply_tempo=False by default for speed (tempo stretch is slow).
    The mel spec CNN is robust to tempo variations via SpecAugment anyway.
    """
    genres = sorted(stem_index.keys())
    n_per_genre = n_samples // len(genres)
    extra = n_samples % len(genres)

    specs_list = []
    labels_list = []

    print(f"Generating {n_samples} synthetic mashup spectrograms...")
    t0 = time.time()

    for i, genre in enumerate(genres):
        n = n_per_genre + (1 if i < extra else 0)
        label = config.GENRE_TO_IDX[genre]

        for _ in tqdm(range(n), desc=f"  {genre:12s}", leave=False):
            try:
                audio, _ = create_synthetic_mashup(
                    genre=genre,
                    stem_index=stem_index,
                    target_duration=config.CHUNK_DURATION,
                    add_noise=add_noise,
                    apply_tempo=apply_tempo,
                )
                spec = melspec_for_cnn(audio, target_frames=config.N_FRAMES)
                specs_list.append(spec)
                labels_list.append(label)
            except Exception as e:
                print(f"  Warning: {genre} failed: {e}")

    specs = np.array(specs_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed/len(specs):.2f}s/sample)")
    print(f"Saving: {save_path} | Specs: {specs.shape} | Labels: {labels.shape}")

    np.savez_compressed(save_path, specs=specs, labels=labels)
    size_mb = save_path.stat().st_size / 1e6
    print(f"Saved: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate spectrogram dataset")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--no_noise", action="store_true", help="Disable noise augmentation")
    parser.add_argument("--with_tempo", action="store_true", help="Enable tempo stretch (slow!)")
    args = parser.parse_args()

    set_seed(config.SEED)

    # Pre-load ESC-50 cache
    print("Building ESC-50 audio cache...")
    t0 = time.time()
    cache = _build_esc50_audio_cache()
    print(f"  Loaded {len(cache)} clips in {time.time()-t0:.1f}s")

    # Build stem index and split
    stem_index = build_stem_index()
    train_index, val_index = split_songs_train_val(stem_index)

    add_noise = not args.no_noise
    apply_tempo = args.with_tempo

    suffix = f"_n{args.n_train}_v{args.n_val}"
    if add_noise: suffix += "_noise"
    if apply_tempo: suffix += "_tempo"

    # Generate training spectrograms
    train_path = config.FEATURES_DIR / f"cnn_train{suffix}.npz"
    if not train_path.exists():
        print(f"\n[TRAIN] Generating {args.n_train} samples...")
        generate_and_save(train_index, args.n_train, train_path,
                          add_noise=add_noise, apply_tempo=apply_tempo)
    else:
        print(f"\n[TRAIN] Cache exists: {train_path}")

    # Generate validation spectrograms
    val_path = config.FEATURES_DIR / f"cnn_val{suffix}.npz"
    if not val_path.exists():
        print(f"\n[VAL] Generating {args.n_val} samples...")
        generate_and_save(val_index, args.n_val, val_path,
                          add_noise=add_noise, apply_tempo=False)
    else:
        print(f"\n[VAL] Cache exists: {val_path}")

    print("\n✓ Dataset generation complete!")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print("\nNext: Run CNN training from notebook (Milestone 3) using PrecomputedDataset")


if __name__ == "__main__":
    main()
