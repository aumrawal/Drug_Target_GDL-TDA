#!/usr/bin/env python3
"""
TopoSurface-DTI training entry point.

Usage:
    # Smoke-test with synthetic data (no dependencies beyond torch):
    python run.py

    # Full training with real PDBBind data:
    python run.py --config configs/base.yaml --data /path/to/PDBBind --no-synthetic

    # Resume from checkpoint:
    python run.py --resume checkpoints/best_model.pt
"""
import os
import sys
import argparse
import yaml

# Make project root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description='Train TopoSurface-DTI')
    p.add_argument('--config',       default='configs/base.yaml')
    p.add_argument('--data',         default=None,  help='override data_dir')
    p.add_argument('--no-synthetic', action='store_true', help='use real data')
    p.add_argument('--resume',       default=None,  help='checkpoint path')
    p.add_argument('--epochs',       type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data:
        cfg['data_dir']      = args.data
        cfg['use_synthetic'] = False
    if args.no_synthetic:
        cfg['use_synthetic'] = False
    if args.epochs:
        cfg['n_epochs'] = args.epochs

    print("=" * 60)
    print("TopoSurface-DTI")
    print("  Drug encoder:   SE(3)-invariant SchNet")
    print("  Pocket encoder: GEM gauge-equivariant CNN")
    print("  TDA:            Vietoris-Rips persistent homology")
    print("  Task:           binding affinity regression (pKd)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Synthetic: {cfg.get('use_synthetic', True)}")
    print()

    train(cfg, resume=args.resume)


if __name__ == '__main__':
    main()
