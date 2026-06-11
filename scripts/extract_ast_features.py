"""
Extract frozen AST (Audio Spectrogram Transformer) embeddings + aligned MFCC
features for train/val/test data.

AST pretrained on AudioSet (2M clips incl. music genres) produces 768-dim
embeddings that encode rich audio semantics — much stronger than hand-crafted
MFCC statistics for genre classification.

Both AST and MFCC features are extracted from the SAME audio samples in the
SAME order, so they can be concatenated for training.

Usage:
    cd /Users/sanskar/dev/DL-GenAI-P
    venv/bin/python scripts/extract_ast_features.py

Options:
    --n_train 8000      # synthetic training mashups (balanced per genre)
    --n_val 1000        # synthetic validation mashups
    --n_crops_test 5    # crops per test file for averaging
    --cpu               # force CPU inference
    --batch_size 8      # batch size for AST forward pass

Output:
    outputs/features/ast_train_N{n_train}_v{n_val}.npz
    outputs/features/ast_test_{n_crops}crop.npz
"""
import sys
import gc
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import librosa
import torch
from transformers import ASTFeatureExtractor, ASTModel

from src import config
from src.config import set_seed, get_device
from src.augment import (build_stem_index, split_songs_train_val,
                         create_synthetic_mashup, load_audio, pad_or_trim)
from src.features import (extract_mfcc_features, extract_chroma_features,
                          extract_spectral_features)


# ─── Enhanced feature extraction (same as notebook cell 22) ──────────────────

def extract_enhanced_features_fast(audio, sr=config.SAMPLE_RATE):
    """286-dim feature vector: MFCC + chroma + spectral (skips tonnetz + beat).

    Tonnetz (2.0s) and beat tracking (0.7s) are 97% of extraction time.
    The remaining features (MFCC + chroma + spectral) take only 0.07s.
    """
    mfcc_feat = extract_mfcc_features(audio, sr)        # 240
    chroma_feat = extract_chroma_features(audio, sr)     # 24
    spectral_feat = extract_spectral_features(audio, sr) # 22
    return np.concatenate([mfcc_feat, chroma_feat, spectral_feat]).astype(np.float32)


# ─── AST embedding extraction ───────────────────────────────────────────────

@torch.no_grad()
def extract_ast_embedding(audio_16k, feature_extractor, model, device):
    """
    Extract 768-dim AST embedding from 16kHz audio.

    Args:
        audio_16k: 1D numpy array at 16kHz
        feature_extractor: ASTFeatureExtractor
        model: ASTModel (frozen, eval mode)
        device: torch device

    Returns:
        np.ndarray of shape (768,)
    """
    inputs = feature_extractor(
        audio_16k, sampling_rate=16000, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Mean pool over all patch tokens → single 768-dim vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


def sync_and_clear(device):
    """Synchronize MPS/CUDA then clear cache."""
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ─── Train/Val feature extraction ───────────────────────────────────────────

def extract_train_val_features(n_train, n_val, feature_extractor, model,
                                device, args):
    """Generate synthetic mashups and extract AST + optional MFCC features."""
    set_seed(config.SEED)

    stem_index = build_stem_index()
    train_index, val_index = split_songs_train_val(stem_index)

    genres = sorted(train_index.keys())
    n_classes = len(genres)
    do_mfcc = not args.no_mfcc

    def _extract_split(split_index, n_samples, split_name):
        n_per_genre = n_samples // n_classes
        extra = n_samples % n_classes

        ast_feats = []
        mfcc_feats = []
        labels = []
        failed = 0

        total = 0
        for gi, genre in enumerate(genres):
            n = n_per_genre + (1 if gi < extra else 0)
            for si in range(n):
                total += 1
                try:
                    audio, _ = create_synthetic_mashup(
                        genre=genre,
                        stem_index=split_index,
                        add_noise=True,
                        apply_tempo=False,  # too slow (26s/sample)
                    )

                    # AST: resample to 16kHz
                    audio_16k = librosa.resample(
                        audio, orig_sr=config.SAMPLE_RATE,
                        target_sr=config.AST_SAMPLE_RATE,
                    )
                    ast_emb = extract_ast_embedding(
                        audio_16k, feature_extractor, model, device,
                    )
                    ast_feats.append(ast_emb)

                    # MFCC: fast version (skips tonnetz + beat = 97% of time)
                    if do_mfcc:
                        mfcc = extract_enhanced_features_fast(audio)
                        mfcc_feats.append(mfcc)

                    labels.append(config.GENRE_TO_IDX[genre])

                except Exception as e:
                    failed += 1
                    if failed <= 5:
                        print(f"  Warning: {genre} sample failed: {e}")

                if total % 100 == 0:
                    elapsed = time.time() - t0
                    rate = total / elapsed
                    eta = (n_samples - total) / rate if rate > 0 else 0
                    print(f"  {split_name}: {total}/{n_samples} "
                          f"({rate:.1f} samples/s, ETA {eta:.0f}s)")

                # Clear cache periodically
                if total % 500 == 0:
                    sync_and_clear(device)

        if failed > 0:
            print(f"  {split_name}: {failed}/{n_samples} samples failed")

        X_ast = np.array(ast_feats, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        if do_mfcc:
            X_mfcc = np.array(mfcc_feats, dtype=np.float32)
        else:
            X_mfcc = np.zeros((len(ast_feats), 0), dtype=np.float32)
        return X_ast, X_mfcc, y

    mfcc_str = "AST + MFCC" if do_mfcc else "AST only"
    print(f"\n{'='*60}")
    print(f"Extracting train features ({n_train} samples, {mfcc_str})...")
    print(f"{'='*60}")
    t0 = time.time()
    X_train_ast, X_train_mfcc, y_train = _extract_split(
        train_index, n_train, "Train")
    train_time = time.time() - t0
    print(f"Train done in {train_time:.0f}s: AST {X_train_ast.shape}"
          + (f", MFCC {X_train_mfcc.shape}" if do_mfcc else ""))

    print(f"\n{'='*60}")
    print(f"Extracting val features ({n_val} samples, {mfcc_str})...")
    print(f"{'='*60}")
    t0 = time.time()
    X_val_ast, X_val_mfcc, y_val = _extract_split(
        val_index, n_val, "Val")
    val_time = time.time() - t0
    print(f"Val done in {val_time:.0f}s: AST {X_val_ast.shape}"
          + (f", MFCC {X_val_mfcc.shape}" if do_mfcc else ""))

    return {
        "X_train_ast": X_train_ast,
        "X_train_mfcc": X_train_mfcc,
        "y_train": y_train,
        "X_val_ast": X_val_ast,
        "X_val_mfcc": X_val_mfcc,
        "y_val": y_val,
    }


# ─── Test feature extraction ────────────────────────────────────────────────

def extract_test_features(feature_extractor, model, device, n_crops=5,
                          do_mfcc=True):
    """Extract AST + optional MFCC features for test files with multi-crop averaging."""
    import pandas as pd

    test_df = pd.read_csv(config.TEST_CSV)
    ids = test_df['id'].astype(str).str.zfill(4).tolist()
    filenames = test_df['filename'].tolist()

    target_len = int(config.CHUNK_DURATION * config.SAMPLE_RATE)
    mfcc_dim = 286 if do_mfcc else 0

    mfcc_str = "AST + MFCC" if do_mfcc else "AST only"
    print(f"\n{'='*60}")
    print(f"Extracting test features ({len(filenames)} files, "
          f"{n_crops} crops, {mfcc_str})...")
    print(f"{'='*60}")
    t0 = time.time()

    all_ast = []
    all_mfcc = []
    all_ids = []
    failed = 0

    for fi, (id_, fname) in enumerate(zip(ids, filenames)):
        path = config.DATA_DIR / fname
        try:
            full_audio = load_audio(str(path), sr=config.SAMPLE_RATE)

            crop_ast = []
            crop_mfcc = []

            for crop_i in range(n_crops):
                # Evenly-spaced crops (deterministic TTA)
                if len(full_audio) <= target_len:
                    chunk = pad_or_trim(full_audio, target_len)
                else:
                    max_start = len(full_audio) - target_len
                    start = int(crop_i * max_start / max(n_crops - 1, 1))
                    chunk = full_audio[start:start + target_len]

                # AST embedding
                chunk_16k = librosa.resample(
                    chunk, orig_sr=config.SAMPLE_RATE,
                    target_sr=config.AST_SAMPLE_RATE,
                )
                ast_emb = extract_ast_embedding(
                    chunk_16k, feature_extractor, model, device,
                )
                crop_ast.append(ast_emb)

                # MFCC features (fast version)
                if do_mfcc:
                    mfcc = extract_enhanced_features_fast(chunk)
                    crop_mfcc.append(mfcc)

            # Average across crops
            all_ast.append(np.mean(crop_ast, axis=0))
            if do_mfcc:
                all_mfcc.append(np.mean(crop_mfcc, axis=0))
            all_ids.append(id_)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: {fname}: {e}")
            all_ast.append(np.zeros(768, dtype=np.float32))
            if do_mfcc:
                all_mfcc.append(np.zeros(mfcc_dim, dtype=np.float32))
            all_ids.append(id_)

        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / elapsed
            eta = (len(filenames) - fi - 1) / rate if rate > 0 else 0
            print(f"  Test: {fi+1}/{len(filenames)} "
                  f"({rate:.1f} files/s, ETA {eta:.0f}s)")

        if (fi + 1) % 500 == 0:
            sync_and_clear(device)

    if failed > 0:
        print(f"  Test: {failed}/{len(filenames)} files failed")

    X_test_ast = np.array(all_ast, dtype=np.float32)
    test_ids = np.array(all_ids)

    result = {"X_test_ast": X_test_ast, "test_ids": test_ids}
    if do_mfcc:
        X_test_mfcc = np.array(all_mfcc, dtype=np.float32)
        result["X_test_mfcc"] = X_test_mfcc
        print(f"Test done in {time.time()-t0:.0f}s: AST {X_test_ast.shape}, "
              f"MFCC {X_test_mfcc.shape}")
    else:
        print(f"Test done in {time.time()-t0:.0f}s: AST {X_test_ast.shape}")

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract AST + MFCC features for all datasets")
    parser.add_argument("--n_train", type=int, default=8000)
    parser.add_argument("--n_val", type=int, default=1000)
    parser.add_argument("--n_crops_test", type=int, default=5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_mfcc", action="store_true",
                        help="Skip MFCC extraction (AST-only, much faster)")
    parser.add_argument("--test_only", action="store_true",
                        help="Only extract test features (skip train/val)")
    parser.add_argument("--train_only", action="store_true",
                        help="Only extract train/val features (skip test)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()

    print(f"\n{'='*60}")
    print(f"AST Feature Extraction")
    print(f"{'='*60}")
    print(f"Device:     {device}")
    print(f"Model:      {config.AST_MODEL_NAME}")
    print(f"Train/Val:  {args.n_train}/{args.n_val} samples")
    print(f"Test crops: {args.n_crops_test}")
    print(f"MFCC:       {'disabled (AST-only)' if args.no_mfcc else 'enabled (286-dim fast)'}")
    print(f"{'='*60}")

    # Load AST model
    print("\nLoading AST model...")
    t0 = time.time()
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.AST_MODEL_NAME)
    model = ASTModel.from_pretrained(config.AST_MODEL_NAME)
    model = model.to(device).eval()
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    print(f"AST loaded in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # ── Train/Val features ───────────────────────────────────────────────
    train_path = config.FEATURES_DIR / f"ast_train_N{args.n_train}_v{args.n_val}.npz"

    if not args.test_only:
        if train_path.exists():
            print(f"\nTrain/val features already cached: {train_path}")
            print("  Delete to re-extract, or use --test_only to skip.")
        else:
            data = extract_train_val_features(
                args.n_train, args.n_val,
                feature_extractor, model, device, args,
            )
            np.savez(train_path, **data)
            print(f"\nSaved train/val features: {train_path}")
            sync_and_clear(device)

    # ── Test features ────────────────────────────────────────────────────
    test_path = config.FEATURES_DIR / f"ast_test_{args.n_crops_test}crop.npz"

    if not args.train_only:
        if test_path.exists():
            print(f"\nTest features already cached: {test_path}")
            print("  Delete to re-extract, or use --train_only to skip.")
        else:
            test_data = extract_test_features(
                feature_extractor, model, device,
                n_crops=args.n_crops_test,
                do_mfcc=not args.no_mfcc,
            )
            np.savez(test_path, **test_data)
            print(f"\nSaved test features: {test_path}")

    print(f"\n{'='*60}")
    print("Feature extraction complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
