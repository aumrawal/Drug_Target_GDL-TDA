# `run.py`

## Overview

`run.py` is the **command-line entry point** for TopoSurface-DTI training. Its job is purely orchestration: parse the user's command-line flags, load a YAML configuration, apply CLI overrides on top, print a banner identifying the model, and hand control to `train.trainer.train`.

It is intentionally short (~65 lines) because all heavy lifting — dataset construction, model assembly, the optimizer loop — happens downstream. The file makes three guarantees to the rest of the codebase:

1. The project root is on `sys.path`, so `from models...` and `from data...` work regardless of where the user invokes Python from.
2. The `cfg` dict passed into `train` already reflects all CLI overrides (`--data`, `--no-synthetic`, `--epochs`).
3. The user sees a self-documenting banner stating which encoders are in play (SchNet, GEM, TDA) and what the task is (pKd regression).

It supports three workflows:

| Command | Purpose |
|---|---|
| `python run.py` | Smoke test with synthetic drug/pocket graphs — no external data needed. |
| `python run.py --config configs/base.yaml --data /path/PDBBind --no-synthetic` | Full PDBBind training. |
| `python run.py --resume checkpoints/best_model.pt` | Resume training from the best checkpoint. |

## Mathematical Foundations

The math here is administrative rather than algorithmic. The interesting equations all live in `train/trainer.py` and the model files. What `run.py` *does* govern, mathematically, is the **hyperparameter vector** $\phi$ that parametrises the training run itself:

$$
\phi \;=\; (\eta,\ \beta_1,\ \beta_2,\ \lambda,\ \delta_{\mathrm{Huber}},\ B,\ T,\ k_{\mathrm{NN}},\ r_{\mathrm{pocket}},\ \mathrm{res}_{\mathrm{TDA}},\ \dots),
$$

where $\eta$ is the Adam learning rate, $\beta_1, \beta_2$ are the moment-decay constants, $\lambda$ is the weight-decay coefficient, $\delta_{\mathrm{Huber}}$ is the Huber transition point, $B$ is batch size (fixed at 1 here), $T$ is the number of epochs, $k_{\mathrm{NN}}$ is the pocket-graph neighbourhood size, $r_{\mathrm{pocket}}$ is the pocket cutoff radius in ångströms, and $\mathrm{res}_{\mathrm{TDA}}$ is the persistence-image grid resolution. `run.py` loads $\phi$ from `configs/base.yaml` (so it can be version-controlled), then **overrides** selected coordinates of $\phi$ from the CLI. Mathematically this is the function composition

$$
\phi_{\mathrm{final}} \;=\; \pi_{\mathrm{CLI}} \circ \pi_{\mathrm{YAML}}(\text{defaults}),
$$

where each $\pi$ is a partial assignment. The `train` function then minimises

$$
\hat{\mathcal{R}}(\theta;\phi_{\mathrm{final}}) \;=\; \frac{1}{N}\sum_{i=1}^{N} \ell_\delta\!\left(f_\theta(x_i),\ y_i\right)
$$

with respect to the ~**250k model parameters** $\theta$ (per `CLAUDE.md`: drug encoder + pocket encoder + fusion ≈ 250k). The roughly-250k figure breaks down as: SE(3)-invariant SchNet drug encoder, GEM-CNN pocket encoder with feature progression $24\rho_0 \to 8\rho_0 \oplus 8\rho_1 \to 16\rho_0 \oplus 16\rho_1 \to 16\rho_0 \oplus 16\rho_1 \to 32\rho_0$, and a cross-attention fusion head that absorbs the two 800-dim persistence-image vectors (1600-dim total TDA input).

## Code Walk-through

### Imports and path bootstrap (lines 1–23)

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train.trainer import train
```

The `sys.path.insert` line is the standard "make project imports work no matter the cwd" idiom. Without it, `python /some/other/dir/run.py` would fail with `ModuleNotFoundError: train`. By taking the directory of `__file__`, the script anchors itself to the project root.

The single downstream import is the `train` function — `run.py` knows nothing about models or datasets directly, only about the trainer's public entry point.

### `parse_args()` (lines 26–33)

Argparse definitions:

| Flag | Default | Effect |
|---|---|---|
| `--config` | `configs/base.yaml` | Path to YAML hyperparameters. |
| `--data` | `None` | Overrides `cfg['data_dir']` and forces `use_synthetic=False`. |
| `--no-synthetic` | False | Forces `use_synthetic=False` without changing the data path. |
| `--resume` | `None` | Path to a `.pt` checkpoint. |
| `--epochs` | `None` | Overrides `cfg['n_epochs']`. |

No type/value validation is done here — that is delegated to `train` and the dataset constructors, which will produce informative errors if e.g. the data directory doesn't exist.

### `main()` (lines 36–61)

1. **Load YAML** (lines 39–40). `yaml.safe_load` parses `configs/base.yaml` into a Python dict. `safe_load` (rather than `load`) is used to avoid YAML-injection arbitrary-object construction.
2. **Apply CLI overrides** (lines 42–48). Three optional overrides in priority order:
   - `--data` sets the data directory **and** flips `use_synthetic` off — if you point at real PDBBind, you obviously want real data.
   - `--no-synthetic` flips only the flag (useful when the YAML already has the right path).
   - `--epochs` replaces `cfg['n_epochs']`, supporting short smoke-test runs (`python run.py --epochs 5`).
3. **Banner print** (lines 50–59). A six-line banner names the three subsystems — *SE(3)-invariant SchNet*, *GEM gauge-equivariant CNN*, *Vietoris-Rips persistent homology* — and the task (*binding affinity regression (pKd)*). This is good operational hygiene: when a long training run scrolls past in the terminal you can scroll back to confirm what was actually executed.
4. **Hand off** (line 61). `train(cfg, resume=args.resume)` is called and blocks until training completes. All checkpointing and metric reporting happens inside.

### `if __name__ == '__main__':` (lines 64–65)

Standard Python script idiom. Lets the module be imported (for the rare case where you want to call `main()` programmatically from a notebook) without auto-executing.

## Biology / Chemistry Context

`run.py` itself doesn't touch chemistry directly. Its role in the drug-discovery loop is to be the reproducible, version-controllable **experiment dispatcher**: a chemist or ML engineer can write a single config YAML, commit it, and re-run `python run.py --config configs/<experiment>.yaml --data /path/PDBBind --no-synthetic` to reproduce a particular training run exactly. This is non-trivial in computational chemistry, where labs often discover that an "improvement" was actually a different random seed or a different pocket cutoff radius.

The choice of PDBBind as the canonical benchmark is itself a biology-context point: PDBBind contains protein–ligand co-crystal structures with curated experimental affinities (pKd, pKi, $-\log\mathrm{IC}_{50}$). It is the *de facto* benchmark for structure-based binding-affinity prediction because (a) the 3D coordinates of both protein and ligand are known (no need to dock), and (b) the affinities are aggregated from primary literature with reasonable consistency. Whenever someone reports an RMSE on "PDBBind v2020 core set" or "CASF-2016", they are reporting a number directly comparable to the ones this script will produce.

## References

- D. P. Kingma and J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. <https://arxiv.org/abs/1412.6980> — referenced because `run.py` passes through to the Adam-based trainer.
- R. Wang, X. Fang, Y. Lu, S. Wang. *The PDBbind Database.* J. Med. Chem. 47(12):2977–2980, 2004.
- Z. Liu et al. *Forging the basis for developing protein–ligand interaction scoring functions.* Acc. Chem. Res. 50(2):302–309, 2017 (CASF-2016 benchmark).
- M. de Haan et al. *Gauge Equivariant Mesh CNNs.* ICLR 2021. <https://arxiv.org/abs/2003.05425> — the GEM-CNN that the pocket encoder builds on.
- K. T. Schütt et al. *SchNet — A deep learning architecture for molecules and materials.* J. Chem. Phys. 148:241722, 2018 — the drug encoder template.
