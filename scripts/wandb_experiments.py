"""
Train and evaluate all three models with Weights & Biases logging.

Models:
1. XGBoost (classical ML baseline) — trained on MFCC + spectral features
2. SimpleCNN (built from scratch) — trained on mel spectrograms
3. AST-XGBoost (pretrained AST embeddings + XGBoost) — frozen AST + XGBoost

All models use pre-cached features for fast iteration.
Each model gets its own W&B run for comparison.

Usage:
    cd /Users/sanskar/dev/DL-GenAI-P
    venv/bin/python scripts/wandb_experiments.py
"""
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import wandb

from src import config
from src.config import set_seed, get_device
from src.models import SimpleCNN, count_parameters
from src.dataset import PrecomputedDataset
from src.train import LabelSmoothingCrossEntropy

# ─── W&B Config ──────────────────────────────────────────────────────────────
WANDB_PROJECT = "23f3003478-t12026"
WANDB_ENTITY = "23f3003478-iit-madras"


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ─── Run 1: XGBoost on MFCC features ────────────────────────────────────────

def run_xgboost_mfcc():
    """Train XGBoost on MFCC + chroma + spectral features (286-dim)."""
    print("\n" + "=" * 60)
    print("RUN 1: XGBoost on MFCC Features")
    print("=" * 60)

    # Load cached features
    data = np.load(config.FEATURES_DIR / "ml_enhanced_Tr8000_V1000.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="xgboost-mfcc",
        config={
            "model": "XGBoost",
            "features": "MFCC + Chroma + Spectral (286-dim)",
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "model_type": "classical_ml",
        },
        tags=["classical-ml", "xgboost", "mfcc"],
    )

    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.SEED,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mlogloss",
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - t0

    # Evaluate
    val_preds = model.predict(X_val)
    train_preds = model.predict(X_train)

    val_f1 = macro_f1(y_val, val_preds)
    val_acc = accuracy_score(y_val, val_preds)
    train_f1 = macro_f1(y_train, train_preds)
    train_acc = accuracy_score(y_train, train_preds)

    print(f"\nTrain — F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   — F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
    print(f"Time: {train_time:.1f}s")
    print(classification_report(y_val, val_preds,
                                target_names=config.GENRES, digits=4))

    wandb.log({
        "train/f1": train_f1,
        "train/accuracy": train_acc,
        "val/f1": val_f1,
        "val/accuracy": val_acc,
        "training_time_s": train_time,
    })

    # Log per-class F1
    per_class_f1 = f1_score(y_val, val_preds, average=None)
    for genre, f1_val in zip(config.GENRES, per_class_f1):
        wandb.log({f"val_f1/{genre}": f1_val})

    wandb.finish()
    print(f"Run 1 complete: val F1={val_f1:.4f}, val Acc={val_acc:.4f}")
    return val_f1


# ─── Run 2: SimpleCNN on mel spectrograms ───────────────────────────────────

def run_simplecnn():
    """Train SimpleCNN from scratch on mel spectrograms."""
    print("\n" + "=" * 60)
    print("RUN 2: SimpleCNN (From Scratch)")
    print("=" * 60)

    # CPU to avoid MPS double-free crashes with large spectrogram tensors
    device = torch.device("cpu")

    # Load cached spectrograms — use subset for faster CPU training
    train_data = np.load(config.FEATURES_DIR / "cnn_train_n5000_v1000_noise.npz",
                         mmap_mode='r')
    val_data = np.load(config.FEATURES_DIR / "cnn_val_n5000_v1000_noise.npz",
                       mmap_mode='r')
    n_train_use = 1000
    n_val_use = 200
    train_specs = np.array(train_data["specs"][:n_train_use])
    train_labels = np.array(train_data["labels"][:n_train_use])
    val_specs = np.array(val_data["specs"][:n_val_use])
    val_labels = np.array(val_data["labels"][:n_val_use])
    print(f"Train: {train_specs.shape}, Val: {val_specs.shape}")

    # Hyperparams (CPU-friendly)
    epochs = 10
    batch_size = 16
    lr = 3e-4
    min_lr = 1e-6
    patience = 8

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="simplecnn-melspec",
        config={
            "model": "SimpleCNN",
            "features": "Mel Spectrogram (128x431)",
            "n_train": train_specs.shape[0],
            "n_val": val_specs.shape[0],
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "model_type": "cnn_from_scratch",
        },
        tags=["cnn", "from-scratch", "melspec"],
    )

    train_loader = DataLoader(
        PrecomputedDataset(train_specs, train_labels, use_spec_augment=False),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        PrecomputedDataset(val_specs, val_labels, use_spec_augment=False),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = SimpleCNN(num_classes=config.NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    wandb.log({"model_params": count_parameters(model)})
    print(f"Model params: {count_parameters(model):,}")
    print(f"Device: {device}")

    best_f1 = -1.0
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for specs, labels in train_loader:
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_f1 = macro_f1(np.array(all_labels), np.array(all_preds))
        train_acc = accuracy_score(np.array(all_labels), np.array(all_preds))

        # Validate
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(specs)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                all_preds.extend(logits.argmax(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        val_f1 = macro_f1(np.array(all_labels), np.array(all_preds))
        val_acc = accuracy_score(np.array(all_labels), np.array(all_preds))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s", flush=True)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/f1": train_f1,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/f1": val_f1,
            "val/accuracy": val_acc,
            "learning_rate": current_lr,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    total_time = time.time() - t_start
    wandb.log({"training_time_s": total_time, "best_val_f1": best_f1})

    # Final per-class F1
    per_class_f1 = f1_score(np.array(all_labels), np.array(all_preds), average=None)
    for genre, f1_val in zip(config.GENRES, per_class_f1):
        wandb.log({f"val_f1/{genre}": f1_val})

    wandb.finish()
    print(f"\nRun 2 complete: best val F1={best_f1:.4f}")

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()

    return best_f1


# ─── Run 3: AST-XGBoost (pretrained embeddings) ────────────────────────────

def run_ast_xgboost():
    """Train XGBoost on frozen AST embeddings (768-dim)."""
    print("\n" + "=" * 60)
    print("RUN 3: AST-XGBoost (Pretrained Embeddings)")
    print("=" * 60)

    # Load cached AST features
    data = np.load(config.FEATURES_DIR / "ast_train_N8000_v1000.npz")
    X_train_ast = data["X_train_ast"]
    X_val_ast = data["X_val_ast"]
    y_train = data["y_train"]
    y_val = data["y_val"]

    # Check for MFCC features
    has_mfcc = "X_train_mfcc" in data and data["X_train_mfcc"].shape[1] > 0
    if has_mfcc:
        X_train = np.concatenate([X_train_ast, data["X_train_mfcc"]], axis=1)
        X_val = np.concatenate([X_val_ast, data["X_val_mfcc"]], axis=1)
        feat_desc = f"AST (768) + MFCC ({data['X_train_mfcc'].shape[1]})"
    else:
        X_train = X_train_ast
        X_val = X_val_ast
        feat_desc = "AST (768-dim)"

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features: {feat_desc}")

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="ast-xgboost",
        config={
            "model": "AST + XGBoost",
            "features": feat_desc,
            "ast_model": config.AST_MODEL_NAME,
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "model_type": "pretrained_transformer",
        },
        tags=["pretrained", "ast", "xgboost", "transformer"],
    )

    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.SEED,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mlogloss",
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - t0

    # Evaluate
    val_preds = model.predict(X_val)
    train_preds = model.predict(X_train)

    val_f1 = macro_f1(y_val, val_preds)
    val_acc = accuracy_score(y_val, val_preds)
    train_f1 = macro_f1(y_train, train_preds)
    train_acc = accuracy_score(y_train, train_preds)

    print(f"\nTrain — F1: {train_f1:.4f}, Acc: {train_acc:.4f}")
    print(f"Val   — F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
    print(f"Time: {train_time:.1f}s")
    print(classification_report(y_val, val_preds,
                                target_names=config.GENRES, digits=4))

    wandb.log({
        "train/f1": train_f1,
        "train/accuracy": train_acc,
        "val/f1": val_f1,
        "val/accuracy": val_acc,
        "training_time_s": train_time,
    })

    per_class_f1 = f1_score(y_val, val_preds, average=None)
    for genre, f1_val in zip(config.GENRES, per_class_f1):
        wandb.log({f"val_f1/{genre}": f1_val})

    wandb.finish()
    print(f"Run 3 complete: val F1={val_f1:.4f}, val Acc={val_acc:.4f}")
    return val_f1


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_xgboost", action="store_true",
                        help="Skip XGBoost run (if already logged)")
    args = parser.parse_args()

    set_seed(config.SEED)

    print("=" * 60)
    print("W&B Experiment Suite: Messy Mashup Genre Classification")
    print(f"Project: {WANDB_PROJECT}")
    print(f"Entity:  {WANDB_ENTITY}")
    print("=" * 60)

    results = {}

    # Run 1: XGBoost baseline
    if not args.skip_xgboost:
        results["xgboost_mfcc"] = run_xgboost_mfcc()
    else:
        print("\nSkipping XGBoost (already logged)")
        results["xgboost_mfcc"] = 0.624  # from previous run

    # Run 2: SimpleCNN from scratch
    results["simplecnn"] = run_simplecnn()

    # Run 3: AST-XGBoost (pretrained)
    results["ast_xgboost"] = run_ast_xgboost()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, f1 in results.items():
        print(f"  {name:20s}: val F1 = {f1:.4f}")
    print(f"\nAll 3 runs logged to W&B: {WANDB_ENTITY}/{WANDB_PROJECT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
