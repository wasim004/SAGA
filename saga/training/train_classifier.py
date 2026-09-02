"""
Trains the Hybrid BiLSTM classifier (Section 3.4) on pre-computed error
curves, optionally concatenated with off-the-shelf SyncNet scalars.

Checkpoint selection uses validation accuracy only; the test split is
touched exactly once, after training, for the final report — never used
for early stopping or checkpoint selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data.error_curve_dataset import GENERATOR_NAMES, build_loaders
from ..models.error_signature_encoder import HybridBiLSTM
from .losses import BatchHardTripletLoss, FocalLoss


def _sync_tensor(uids: list[str], sync_features: dict | None, device) -> torch.Tensor | None:
    if sync_features is None:
        return None
    return torch.tensor(
        [sync_features.get(u, [0.0, 0.0]) for u in uids], dtype=torch.float32, device=device
    )


def run_epoch(model, loader, optimizer, focal_loss, triplet_loss, triplet_weight,
              sync_features, device, train: bool):
    model.train(train)
    total = ce_total = tri_total = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            sync = _sync_tensor(batch["uid"], sync_features, device)

            out = model(x, mask, sync=sync)
            l_ce = focal_loss(out["logits"], labels)
            l_tri = triplet_loss(out["embed"], labels) if triplet_weight > 0 else torch.tensor(0.0)
            loss = l_ce + triplet_weight * l_tri

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total += loss.item()
            ce_total += l_ce.item()
            tri_total += l_tri.item() if triplet_weight > 0 else 0.0
            all_preds.extend(out["logits"].argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds, labels_arr = np.array(all_preds), np.array(all_labels)
    n = len(loader)
    return {
        "loss": total / n, "ce_loss": ce_total / n, "tri_loss": tri_total / n,
        "acc": accuracy_score(labels_arr, preds),
        "f1_macro": f1_score(labels_arr, preds, average="macro", zero_division=0),
        "preds": preds, "labels": labels_arr,
    }


def train_classifier(
    meta_csv: str,
    cache_dir: str,
    output_dir: str,
    sync_features: dict | None = None,
    d_in: int = 12,
    d_model: int = 128,
    n_layers: int = 3,
    d_embed: int = 128,
    n_classes: int = 4,
    dropout: float = 0.2,
    sync_dim: int = 2,
    max_windows: int = 16,
    tau_k: float = 1.5,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 15,
    epochs: int = 120,
    batch_size: int = 32,
    triplet_margin: float = 0.5,
    triplet_weight: float = 0.5,
    focal_gamma: float = 2.0,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Trains the Hybrid BiLSTM and returns the final one-shot test metrics."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, class_weights = build_loaders(
        meta_csv, cache_dir, batch_size, max_windows=max_windows, tau_k=tau_k,
        d_in=d_in, with_val=True,
    )
    print(f"[train_classifier] train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = HybridBiLSTM(
        d_in=d_in, d_model=d_model, n_layers=n_layers, d_embed=d_embed,
        n_classes=n_classes, dropout=dropout, sync_dim=sync_dim if sync_features else 0,
    ).to(device)
    print(f"[train_classifier] params={model.count_params():,}")

    focal_loss = FocalLoss(gamma=focal_gamma, weight=class_weights)
    triplet_loss = BatchHardTripletLoss(margin=triplet_margin)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc, best_val_f1, history = 0.0, 0.0, []

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * epoch / warmup_epochs

        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, focal_loss, triplet_loss,
                        triplet_weight, sync_features, device, train=True)
        va = run_epoch(model, val_loader, optimizer, focal_loss, triplet_loss,
                        0.0, sync_features, device, train=False)

        if epoch > warmup_epochs:
            scheduler.step()

        print(f"Ep {epoch:3d}/{epochs} tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.4f} "
              f"va_acc={va['acc']:.4f} va_f1={va['f1_macro']:.4f} ({time.time() - t0:.1f}s)")
        history.append({"epoch": epoch, "tr_loss": tr["loss"], "tr_acc": tr["acc"],
                         "va_acc": va["acc"], "va_f1": va["f1_macro"]})

        if va["acc"] > best_val_acc:
            best_val_acc, best_val_f1 = va["acc"], va["f1_macro"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "best_val_acc": best_val_acc, "best_val_f1": best_val_f1},
                       out_dir / "best.pt")

    # Test set touched exactly once, on the val-selected checkpoint.
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te = run_epoch(model, test_loader, None, focal_loss, triplet_loss, 0.0,
                    sync_features, device, train=False)

    cm = confusion_matrix(te["labels"], te["preds"], labels=list(range(n_classes)))
    print("\nConfusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        print(f"  {GENERATOR_NAMES.get(i, i):>10}  " + "  ".join(f"{v:>6}" for v in row))

    results = {
        "best_val_acc": best_val_acc, "best_val_f1": best_val_f1,
        "acc": float(te["acc"]), "f1_macro": float(te["f1_macro"]),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[train_classifier] Done. test acc={results['acc']:.4f} f1={results['f1_macro']:.4f}")
    return results
