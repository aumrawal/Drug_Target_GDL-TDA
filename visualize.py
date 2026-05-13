#!/usr/bin/env python3
"""
Predicted vs actual binding affinity visualisation for TopoSurface-DTI.

Produces a 4-panel diagnostic figure:
  1. Scatter  — predicted vs actual pKd, coloured by absolute error
  2. Residuals — histogram of (pred - actual) with normal fit overlay
  3. Ranking  — compounds sorted by actual pKd; predicted overlaid
                (the view that matters for virtual screening)
  4. CDF      — cumulative % of predictions within ±X kcal/mol

Usage:
    # after training, loads checkpoints/best_model.pt automatically
    python visualize.py

    # specify checkpoint and number of evaluation samples
    python visualize.py --checkpoint checkpoints/best_model.pt --n-samples 200

    # also train a fresh model first if no checkpoint exists
    python visualize.py --train-first
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset           import DTIDataset
from data.tda_features      import tda_to_tensor
from models.toposurface_dti import TopoSurfaceDTI
from train.trainer          import collate_single, forward_step, train


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, n_samples: int = 200, seed: int = 99) -> tuple:
    """
    Run model over a synthetic evaluation set.
    Returns (preds, actuals) as numpy arrays.
    """
    device = next(model.parameters()).device
    model.eval()

    ds     = DTIDataset(use_synthetic=True, n_synthetic=n_samples,
                        tda_resolution=model.tda_resolution)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    loss_fn = torch.nn.HuberLoss(delta=1.0)
    preds, actuals = [], []

    for sample in loader:
        pred, _ = forward_step(model, sample, device, loss_fn)
        preds.append(float(pred.cpu()))
        actuals.append(float(sample['affinity']))

    return np.array(preds), np.array(actuals)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
    residuals = preds - actuals
    pearson_r, _  = stats.pearsonr(preds, actuals)
    spearman_r, _ = stats.spearmanr(preds, actuals)
    return {
        'rmse':      float(np.sqrt(np.mean(residuals ** 2))),
        'mae':       float(np.mean(np.abs(residuals))),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'bias':      float(np.mean(residuals)),
        'std_err':   float(np.std(residuals)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot panels
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(ax, preds, actuals, metrics):
    """Panel 1 — Predicted vs Actual scatter, coloured by |error|."""
    errors = np.abs(preds - actuals)

    sc = ax.scatter(actuals, preds, c=errors, cmap='RdYlGn_r',
                    s=40, alpha=0.75, edgecolors='none',
                    vmin=0, vmax=np.percentile(errors, 95))

    # Identity line
    lo = min(actuals.min(), preds.min()) - 0.3
    hi = max(actuals.max(), preds.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='Perfect prediction')

    # ±1 RMSE band
    rmse = metrics['rmse']
    ax.fill_between([lo, hi], [lo - rmse, hi - rmse], [lo + rmse, hi + rmse],
                    color='steelblue', alpha=0.08, label=f'±RMSE band')

    # Linear regression line
    slope, intercept, *_ = stats.linregress(actuals, preds)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, slope * xs + intercept, 'steelblue', lw=1.5, alpha=0.8, label='Linear fit')

    plt.colorbar(sc, ax=ax, label='|Error| (pKd units)', shrink=0.85)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual pKd', fontsize=11)
    ax.set_ylabel('Predicted pKd', fontsize=11)
    ax.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    # Metric box
    txt = (f"RMSE  = {metrics['rmse']:.3f}\n"
           f"MAE   = {metrics['mae']:.3f}\n"
           f"R     = {metrics['pearson_r']:.3f}\n"
           f"ρ     = {metrics['spearman_r']:.3f}")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#cccccc'))


def plot_residuals(ax, preds, actuals, metrics):
    """Panel 2 — Error histogram with normal fit."""
    residuals = preds - actuals

    n_bins = max(10, len(residuals) // 8)
    ax.hist(residuals, bins=n_bins, color='steelblue', alpha=0.65,
            edgecolor='white', linewidth=0.5, density=True, label='Residuals')

    # Normal fit overlay
    mu, sigma = metrics['bias'], metrics['std_err']
    xs = np.linspace(residuals.min() - 0.5, residuals.max() + 0.5, 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), 'crimson', lw=2,
            label=f'N({mu:.2f}, {sigma:.2f}²)')

    ax.axvline(0,          color='black',  lw=1.2, ls='--', alpha=0.7)
    ax.axvline(mu,         color='crimson', lw=1.2, ls=':',  alpha=0.9, label=f'Bias = {mu:.3f}')
    ax.axvline( sigma, color='grey', lw=0.8, ls=':', alpha=0.6)
    ax.axvline(-sigma, color='grey', lw=0.8, ls=':', alpha=0.6)

    ax.set_xlabel('Predicted − Actual (pKd)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

    # Percentage within ±1
    within_1 = float(np.mean(np.abs(residuals) < 1.0) * 100)
    ax.text(0.97, 0.95, f'{within_1:.1f}% within ±1 pKd',
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85, ec='#cccccc'))


def plot_ranking(ax, preds, actuals):
    """
    Panel 3 — Virtual screening ranking view.
    Compounds sorted by actual pKd; predicted overlaid.
    Good models should track the same trend.
    """
    order   = np.argsort(actuals)
    xs      = np.arange(len(order))
    act_sorted  = actuals[order]
    pred_sorted = preds[order]

    ax.plot(xs, act_sorted,  color='#2c7bb6', lw=2,   label='Actual pKd',    zorder=3)
    ax.plot(xs, pred_sorted, color='#d7191c', lw=1.5, alpha=0.8,
            label='Predicted pKd', zorder=2)
    ax.fill_between(xs, act_sorted, pred_sorted,
                    where=(pred_sorted >= act_sorted),
                    color='#d7191c', alpha=0.12, label='Over-prediction')
    ax.fill_between(xs, act_sorted, pred_sorted,
                    where=(pred_sorted < act_sorted),
                    color='#2c7bb6', alpha=0.12, label='Under-prediction')

    ax.set_xlabel('Compound rank (sorted by actual pKd)', fontsize=11)
    ax.set_ylabel('pKd', fontsize=11)
    ax.set_title('Ranking View (Virtual Screening)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, len(xs) - 1)


def plot_cdf(ax, preds, actuals):
    """
    Panel 4 — Cumulative distribution of absolute errors.
    Shows what fraction of predictions fall within ±X pKd units.
    """
    abs_errors = np.sort(np.abs(preds - actuals))
    cdf        = np.arange(1, len(abs_errors) + 1) / len(abs_errors)

    ax.plot(abs_errors, cdf * 100, color='#1a9641', lw=2.5)
    ax.fill_between(abs_errors, cdf * 100, alpha=0.15, color='#1a9641')

    # Reference lines at clinically meaningful thresholds
    for threshold, label in [(0.5, '0.5'), (1.0, '1.0'), (1.5, '1.5'), (2.0, '2.0')]:
        pct = float(np.mean(abs_errors <= threshold) * 100)
        ax.axvline(threshold, color='grey', lw=0.8, ls='--', alpha=0.6)
        ax.text(threshold + 0.02, 5, f'{pct:.0f}%\n≤{label}', fontsize=7.5,
                color='grey', va='bottom')

    ax.set_xlabel('Absolute error (pKd units)', fontsize=11)
    ax.set_ylabel('Cumulative % of predictions', fontsize=11)
    ax.set_title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.grid(axis='y', alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(preds: np.ndarray, actuals: np.ndarray,
                save_path: str = 'predictions_vs_actual.png'):
    metrics = compute_metrics(preds, actuals)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f'TopoSurface-DTI  ·  Predicted vs Actual Binding Affinity  '
        f'·  n={len(preds)} compounds',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                           left=0.07, right=0.96, top=0.93, bottom=0.07)

    plot_scatter(   fig.add_subplot(gs[0, 0]), preds, actuals, metrics)
    plot_residuals( fig.add_subplot(gs[0, 1]), preds, actuals, metrics)
    plot_ranking(   fig.add_subplot(gs[1, 0]), preds, actuals)
    plot_cdf(       fig.add_subplot(gs[1, 1]), preds, actuals)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Figure saved → {save_path}')
    plt.show()

    # Print summary table
    print('\n── Evaluation Metrics ──────────────────────')
    print(f'  RMSE         : {metrics["rmse"]:.4f} pKd')
    print(f'  MAE          : {metrics["mae"]:.4f} pKd')
    print(f'  Pearson R    : {metrics["pearson_r"]:.4f}')
    print(f'  Spearman ρ   : {metrics["spearman_r"]:.4f}')
    print(f'  Bias (mean Δ): {metrics["bias"]:.4f} pKd')
    print(f'  Error std    : {metrics["std_err"]:.4f} pKd')
    abs_err = np.abs(preds - actuals)
    for t in [0.5, 1.0, 1.5, 2.0]:
        print(f'  Within ±{t:.1f}   : {np.mean(abs_err <= t)*100:.1f}%')
    print('────────────────────────────────────────────')

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  default='checkpoints/best_model.pt')
    parser.add_argument('--n-samples',   type=int, default=200)
    parser.add_argument('--train-first', action='store_true',
                        help='Train a fresh model before visualising')
    parser.add_argument('--output',      default='predictions_vs_actual.png')
    args = parser.parse_args()

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu'
    )

    # ── Load or train model ─────────────────────────────────────────────
    if args.train_first or not os.path.exists(args.checkpoint):
        print('Training model on synthetic data first...')
        import yaml
        with open('configs/base.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg.update({'use_synthetic': True, 'n_train': 200, 'n_val': 40, 'n_epochs': 50})
        model = train(cfg)
    else:
        ckpt  = torch.load(args.checkpoint, map_location=device)
        cfg   = ckpt.get('cfg', {})
        model = TopoSurfaceDTI.from_config(cfg).to(device)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded checkpoint from epoch {ckpt["epoch"]}  '
              f'(val RMSE = {ckpt.get("best_val_rmse", "?"):.4f})')

    # ── Run inference ───────────────────────────────────────────────────
    print(f'\nRunning inference on {args.n_samples} synthetic samples...')
    preds, actuals = run_inference(model, n_samples=args.n_samples)

    # ── Plot ────────────────────────────────────────────────────────────
    make_figure(preds, actuals, save_path=args.output)


if __name__ == '__main__':
    main()
