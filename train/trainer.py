# train/trainer.py
"""
Training loop for TopoSurface-DTI.

Metrics tracked:
  - Huber loss (robust to outliers in pKd distribution)
  - RMSE (standard binding affinity metric)
  - Pearson R (ranking correlation, critical for virtual screening)

Typical benchmarks for state-of-art DTI models on PDBBind v2020:
  - RMSE ≈ 1.3-1.5 kcal/mol
  - Pearson R ≈ 0.75-0.82
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional

from models.toposurface_dti import TopoSurfaceDTI
from data.dataset import DTIDataset


# ──────────────────────────────────────────────────────────────────────────
# Collate: variable-size molecular graphs → single-sample processing
# ──────────────────────────────────────────────────────────────────────────

def collate_single(batch):
    """Pass a single sample dict through unchanged (batch_size=1 training)."""
    return batch[0]


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred   - pred.mean()
    t = target - target.mean()
    denom = p.norm() * t.norm()
    return float((p * t).sum() / denom.clamp(min=1e-8))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).pow(2).mean().sqrt())


# ──────────────────────────────────────────────────────────────────────────
# Forward step
# ──────────────────────────────────────────────────────────────────────────

def forward_step(model, sample, device, loss_fn) -> tuple:
    affinity = sample['affinity'].to(device)

    pred = model(
        drug_x        = sample['drug_x'].to(device),
        drug_pos      = sample['drug_pos'].to(device),
        drug_edge     = sample['drug_edge'].to(device),
        pocket_x      = sample['pocket_x'].to(device),
        pocket_edge   = sample['pocket_edge'].to(device),
        pocket_angles = sample['pocket_angles'].to(device),
        pocket_trans  = sample['pocket_trans'].to(device),
        drug_tda      = sample['drug_tda'].to(device),
        pocket_tda    = sample['pocket_tda'].to(device),
    )

    loss = loss_fn(pred.unsqueeze(0), affinity.unsqueeze(0))
    return pred, loss


# ──────────────────────────────────────────────────────────────────────────
# Train / validate epochs
# ──────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, loss_fn) -> Dict:
    model.train()
    total_loss, preds, targets = 0.0, [], []

    for sample in loader:
        optimizer.zero_grad()
        pred, loss = forward_step(model, sample, device, loss_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds.append(pred.detach().cpu())
        targets.append(sample['affinity'])

    p = torch.stack(preds)
    t = torch.stack(targets)
    return {'loss': total_loss / len(loader), 'rmse': rmse(p, t), 'r': pearson_r(p, t)}


@torch.no_grad()
def validate(model, loader, device, loss_fn) -> Dict:
    model.eval()
    total_loss, preds, targets = 0.0, [], []

    for sample in loader:
        pred, loss = forward_step(model, sample, device, loss_fn)
        total_loss += loss.item()
        preds.append(pred.cpu())
        targets.append(sample['affinity'])

    p = torch.stack(preds)
    t = torch.stack(targets)
    return {'loss': total_loss / len(loader), 'rmse': rmse(p, t), 'r': pearson_r(p, t)}


# ──────────────────────────────────────────────────────────────────────────
# Main training entry point
# ──────────────────────────────────────────────────────────────────────────

def train(cfg: dict, resume: Optional[str] = None):
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Device: {device}")

    use_syn = cfg.get('use_synthetic', True)
    train_ds = DTIDataset(
        data_dir=cfg.get('data_dir', 'data/PDBBind'),
        split='train',
        tda_resolution=cfg.get('tda_resolution', 20),
        use_synthetic=use_syn,
        n_synthetic=cfg.get('n_train', 80),
    )
    val_ds = DTIDataset(
        data_dir=cfg.get('data_dir', 'data/PDBBind'),
        split='val',
        tda_resolution=cfg.get('tda_resolution', 20),
        use_synthetic=use_syn,
        n_synthetic=cfg.get('n_val', 20),
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=collate_single)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, collate_fn=collate_single)

    model   = TopoSurfaceDTI.from_config(cfg).to(device)
    params  = model.count_parameters()
    print(f"Parameters: drug={params['drug_encoder']:,}  "
          f"pocket={params['pocket_encoder']:,}  "
          f"fusion={params['fusion']:,}  "
          f"total={params['total']:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get('lr', 1e-3),
        weight_decay=cfg.get('weight_decay', 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-5
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    start_epoch  = 0
    best_val_rmse = float('inf')

    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_rmse = ckpt.get('best_val_rmse', best_val_rmse)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(start_epoch, cfg.get('n_epochs', 100)):
        tr = train_epoch(model, train_loader, optimizer, device, loss_fn)
        va = validate(model, val_loader, device, loss_fn)
        scheduler.step(va['loss'])

        print(
            f"Epoch {epoch:03d} | "
            f"Train  loss={tr['loss']:.4f}  RMSE={tr['rmse']:.3f}  R={tr['r']:.3f} | "
            f"Val    loss={va['loss']:.4f}  RMSE={va['rmse']:.3f}  R={va['r']:.3f}"
        )

        if va['rmse'] < best_val_rmse:
            best_val_rmse = va['rmse']
            torch.save({
                'epoch':         epoch,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'best_val_rmse': best_val_rmse,
                'cfg':           cfg,
            }, 'checkpoints/best_model.pt')

    print(f"\nTraining complete. Best Val RMSE: {best_val_rmse:.4f}")
    return model
