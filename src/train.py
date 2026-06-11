"""
Training loop for CNN audio genre classifier.

Features:
- Online synthetic mashup augmentation (matches test distribution)
- Macro F1 optimization (aligned with competition metric)
- Early stopping with best model checkpoint
- Cosine annealing LR schedule
- Label smoothing for generalization
- Optional Weights & Biases logging
"""
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

from . import config
from .config import set_seed, get_device

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 — the competition metric."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for better calibration.
    Smoothing regularizes predictions, helps with generalization.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    grad_clip: float = config.GRAD_CLIP
                    ) -> Tuple[float, float]:
    """
    Run one training epoch.
    Returns: (mean_loss, macro_f1)
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (specs, labels) in enumerate(loader):
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = macro_f1(np.array(all_labels), np.array(all_preds))
    return avg_loss, f1


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device
             ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a validation set.
    Returns: (mean_loss, macro_f1, all_labels, all_preds)
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(specs)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = macro_f1(np.array(all_labels), np.array(all_preds))
    return avg_loss, f1, np.array(all_labels), np.array(all_preds)


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          save_path: Optional[Path] = None,
          num_epochs: int = config.NUM_EPOCHS,
          lr: float = config.LEARNING_RATE,
          min_lr: float = config.MIN_LR,
          weight_decay: float = config.WEIGHT_DECAY,
          patience: int = config.PATIENCE,
          device: Optional[torch.device] = None,
          label_smoothing: float = 0.1,
          verbose: bool = True,
          use_wandb: bool = False,
          ) -> Dict:
    """
    Full training loop with:
    - AdamW optimizer
    - Cosine annealing LR schedule
    - Early stopping on val macro F1
    - Best model checkpoint

    Returns: dict with training history
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    if save_path is None:
        save_path = config.MODELS_DIR / "best_model.pt"

    best_f1 = -1.0
    no_improve = 0
    history = {
        "train_loss": [], "train_f1": [],
        "val_loss": [], "val_f1": [],
    }

    print(f"Training on device: {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 60)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # LR step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        # Clear MPS cache after every epoch to prevent memory accumulation
        # (MPS allocator leaks across epochs and crashes around epoch 6 without this)
        if device.type == "mps":
            torch.mps.empty_cache()

        elapsed = time.time() - t0
        if verbose:
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        # Log to W&B
        if use_wandb and HAS_WANDB and wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/f1": train_f1,
                "val/loss": val_loss,
                "val/f1": val_f1,
                "learning_rate": current_lr,
            })

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "train_f1": train_f1,
            }, save_path)
            if verbose:
                print(f"  ✓ Saved best model (F1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val macro F1: {best_f1:.4f}")
    history["best_val_f1"] = best_f1

    return history


def load_best_model(model: nn.Module,
                    checkpoint_path: Path,
                    device: Optional[torch.device] = None) -> nn.Module:
    """Load best saved model weights."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"with val F1={checkpoint['val_f1']:.4f}")
    return model.to(device)


def print_classification_report(y_true: np.ndarray,
                                 y_pred: np.ndarray):
    """Print per-class F1 report aligned with competition metric."""
    n = len(config.GENRES)
    # Clip predictions to valid range in case of any label encoding mismatch
    y_pred_clipped = np.clip(np.asarray(y_pred, dtype=int), 0, n - 1)
    y_true_clipped = np.clip(np.asarray(y_true, dtype=int), 0, n - 1)
    print("\nPer-class Classification Report:")
    print(classification_report(y_true_clipped, y_pred_clipped,
                                labels=list(range(n)),
                                target_names=config.GENRES,
                                digits=4))
    print(f"Macro F1: {macro_f1(y_true_clipped, y_pred_clipped):.4f}")
