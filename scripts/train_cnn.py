"""
External CNN training script — runs outside Jupyter to avoid memory pressure crashes.

Fixes applied vs previous version:
  1. torch.mps.synchronize() before empty_cache() — MPS ops are async; clearing cache
     while ops are in-flight causes "pointer being freed was not allocated" abort.
  2. No per-batch cache clearing — only one clear per epoch (after synchronize).
  3. Model state moved to CPU before torch.save — avoids MPS buffer serialization bugs.
  4. SpecAugment disabled — BatchNorm running stats are computed on masked (zeroed)
     training inputs; in eval mode the unmasked val inputs produce feature maps with
     a different distribution, causing near-0 val F1 despite good test performance.
  5. Larger batch size (32) — safe now without per-batch clearing overhead.

Usage:
    cd /Users/sanskar/dev/DL-GenAI-P
    venv/bin/python scripts/train_cnn.py

Options:
    --epochs 50
    --lr 3e-4
    --batch_size 32
    --patience 15
    --resume               # resume from last checkpoint
    --cpu                  # force CPU (slower but guaranteed stable)

Output: outputs/models/cnn_simplecnn_best.pt
"""
import sys
import gc
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src import config
from src.config import set_seed, get_device
from src.dataset import PrecomputedDataset
from src.models import get_model, count_parameters
from src.train import LabelSmoothingCrossEntropy


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def sync_and_clear(device):
    """
    Synchronize MPS/CUDA then clear cache.
    MUST call synchronize() first — MPS ops are async. Clearing cache while
    ops are still in-flight causes C-level double-free ('pointer being freed
    was not allocated').
    """
    if device.type == "mps":
        torch.mps.synchronize()   # wait for all pending MPS ops to finish
        torch.mps.empty_cache()   # now safe to free the allocator cache
    elif device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def save_checkpoint(path, epoch, model, optimizer, val_f1, train_f1, history,
                    device, extra=None):
    """
    Save checkpoint with model state on CPU.
    Moving state dict to CPU before serialization avoids MPS tensor
    serialization bugs that can corrupt the file or cause double-frees.
    """
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {
        "epoch": epoch,
        "model_state_dict": cpu_state,
        "val_f1": val_f1,
        "train_f1": train_f1,
        "history": history,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(specs)
        loss   = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = macro_f1(np.array(all_labels), np.array(all_preds))
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(specs)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1       = macro_f1(np.array(all_labels), np.array(all_preds))
    return avg_loss, f1


def main():
    parser = argparse.ArgumentParser(description="Train SimpleCNN outside Jupyter")
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--min_lr",          type=float, default=1e-6)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--patience",        type=int,   default=15)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--resume",  action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--cpu",     action="store_true", help="Force CPU training")
    args = parser.parse_args()

    set_seed(config.SEED)

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"\n{'='*60}")
    print(f"SimpleCNN External Training Script")
    print(f"{'='*60}")
    print(f"Device:       {device}")
    print(f"Epochs:       {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"Patience:     {args.patience}  |  SpecAugment: disabled (BatchNorm fix)")
    print(f"{'='*60}\n")

    # ── Load spectrogram dataset ───────────────────────────────────────────────
    suffix     = "_n5000_v1000_noise"
    train_path = config.FEATURES_DIR / f"cnn_train{suffix}.npz"
    val_path   = config.FEATURES_DIR / f"cnn_val{suffix}.npz"

    if not train_path.exists() or not val_path.exists():
        print(f"ERROR: Pre-generated spectrograms not found!")
        print(f"  Run first: venv/bin/python scripts/generate_spectrograms.py")
        sys.exit(1)

    print("Loading spectrograms...")
    # Load without mmap — external process has no Jupyter memory overhead,
    # so loading 1.1 GB into RAM is fine and avoids mmap-related edge cases.
    train_data = np.load(train_path)
    val_data   = np.load(val_path)
    train_specs, train_labels = train_data["specs"], train_data["labels"]
    val_specs,   val_labels   = val_data["specs"],   val_data["labels"]
    print(f"  Train: {train_specs.shape}  Val: {val_specs.shape}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # SpecAugment disabled: it zeroes out time/freq bands, which shifts the
    # BatchNorm running mean/var during training. In eval mode the unmasked val
    # inputs produce feature maps with a different distribution → model outputs
    # garbage on the val set despite learning correctly on test data.
    train_loader = DataLoader(
        PrecomputedDataset(train_specs, train_labels, use_spec_augment=False),
        batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        PrecomputedDataset(val_specs, val_labels, use_spec_augment=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches/epoch")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model     = get_model(model_type="simple_cnn", num_classes=config.NUM_CLASSES).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    print(f"Model params: {count_parameters(model):,}\n")

    save_path      = config.MODELS_DIR / "cnn_simplecnn_best.pt"
    last_ckpt_path = config.MODELS_DIR / "cnn_simplecnn_last.pt"

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_f1     = -1.0
    no_improve  = 0
    history     = {"train_loss": [], "train_f1": [], "val_loss": [], "val_f1": []}

    if args.resume and last_ckpt_path.exists():
        print(f"Resuming from {last_ckpt_path}...")
        ckpt = torch.load(last_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt["best_f1"]
        no_improve  = ckpt["no_improve"]
        history     = ckpt.get("history", history)
        print(f"  Resumed at epoch {start_epoch}, best_f1={best_f1:.4f}\n")
    elif args.resume:
        print("No checkpoint found — starting fresh.\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Starting training on {device}...")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # Synchronize MPS ops then clear cache — one call per epoch only.
        # Per-batch clearing was causing the double-free crash.
        sync_and_clear(device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # Save best model (state on CPU to avoid MPS serialization bugs)
        if val_f1 > best_f1:
            best_f1   = val_f1
            no_improve = 0
            save_checkpoint(save_path, epoch, model, optimizer,
                            val_f1, train_f1, history, device)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # Resumable checkpoint (CPU state, includes scheduler)
        save_checkpoint(last_ckpt_path, epoch, model, optimizer,
                        val_f1, train_f1, history, device,
                        extra={
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_f1":   best_f1,
                            "no_improve": no_improve,
                        })

    print(f"\n{'='*60}")
    print(f"Training complete!  Best val Macro F1: {best_f1:.4f}")
    print(f"Best model: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
