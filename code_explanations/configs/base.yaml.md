# `configs/base.yaml` — Base Configuration

This file is the single source of truth for hyperparameters in **TopoSurface-DTI**. It is loaded by `run.py` via `yaml.safe_load`, merged with CLI flags, and then passed into `TopoSurfaceDTI.from_config(cfg)` (see `models/toposurface_dti.py`) and the training loop in `train/trainer.py`. Editing this file is the recommended way to change model capacity, dataset size, or optimisation settings without touching code.

The full file is reproduced below for reference; each section is then dissected line by line.

```yaml
# configs/base.yaml
# TopoSurface-DTI base configuration

# ── Data ──────────────────────────────────────────────────────────────────
data_dir:        /kaggle/input/datasets/madukacharles/pdbbind-protein-ligand-binding-affinity-dataset
use_synthetic:   true
n_train:         200
n_val:           40

# ── Model ─────────────────────────────────────────────────────────────────
drug_in_features:    17
pocket_in_features:  25
hidden_dim:          64
tda_resolution:      20
n_drug_interactions: 3

# ── Training ──────────────────────────────────────────────────────────────
n_epochs:      100
lr:            1.0e-3
weight_decay:  0.0
```

## Data block

**`data_dir`** is the filesystem path that the dataset class consumes. When `use_synthetic` is false, `data/dataset.py::DTIDataset._load_real` reads `{data_dir}/{pdb_id}/{pdb_id}_protein.pdb` (and optionally `{pdb_id}_ligand.sdf`) for each entry in `{data_dir}/{split}_split.json`. On Kaggle the default points at the public PDBBind dataset mirror; locally you would change it to wherever the PDBBind tree lives. CLI flag `--data` in `run.py` overrides this.

**`use_synthetic`** is a boolean switch read inside `DTIDataset.__init__`. When `true`, the dataset bypasses BioPython/RDKit entirely and generates random drug graphs (via `data/molecule_graph.py::synthetic_drug_graph`) and random pocket point clouds (via `data/pocket_mesh.py::synthetic_pocket_graph`) on every call. The affinity targets are sampled from a fixed-seed RNG so the smoke test is reproducible. This keeps `python run.py` runnable with zero external data, which is essential for CI and the Kaggle smoke cell.

**`n_train`** and **`n_val`** are only consulted in synthetic mode — they fix the number of synthetic samples manufactured for the training and validation epochs. In real-data mode the split sizes come from the JSON split files instead. Defaults of 200 / 40 give a fast loop (~30 s per epoch on CPU) suitable for sanity checks; raise them when you actually want to fit something.

## Model block

**`drug_in_features: 17`** must equal `N_ATOM_FEAT` in `data/molecule_graph.py`. It is the input dimension to `DrugEncoder` (`models/drug_encoder.py`). Features split as 10 atom-type one-hots, 4 hybridisation one-hots, and 3 scalars (aromaticity, ring, formal-charge). If you add or remove atom features upstream you must update this number in lockstep or the first linear projection will reject the tensor shape.

**`pocket_in_features: 25`** must equal `N_RESIDUE_FEAT` in `data/pocket_mesh.py` and is the input dimension to `PocketEncoder` (`models/pocket_encoder.py`). Composition: 21 amino-acid one-hots (20 canonical + UNK) + 3D position + distance-to-centroid. The first GEM-CNN block lifts this into the `24ρ₀` representation.

**`hidden_dim: 64`** is the encoder width shared by both streams. In `DrugEncoder` it is the channel dimension of every interaction block and the global readout. In `PocketEncoder` the irrep layout is sized so the *total* channel count matches `hidden_dim` at every stage (e.g. `8ρ₀ ⊕ 8ρ₁` has 8 + 2·8 = 24 real channels, scaling proportionally with `hidden_dim/64`). The `FusionModule` cross-attention also uses this as its key/query/value dimension. Larger values multiply parameter count roughly quadratically inside the GEM blocks.

**`tda_resolution: 20`** controls the persistence-image grid (`data/tda_features.py::persistence_image`). A `R × R` image is computed for each of `H₀` and `H₁`, then flattened. With `R = 20` each stream contributes a `(2 · 400,) = (800,)` vector, and the drug-plus-pocket concatenation entering the fusion MLP is `1600`-dimensional. The fusion MLP's first linear layer must match this; `TopoSurfaceDTI.__init__` reads `tda_resolution` and sizes the projection accordingly. Increase `R` for finer topological detail at quadratic cost in TDA feature size.

**`n_drug_interactions: 3`** is the number of SchNet-style interaction blocks in `DrugEncoder`. Each block performs one round of distance-conditioned message passing $h_i \leftarrow h_i + \sum_j \phi(h_j) \odot \psi(\lVert x_i - x_j \rVert)$. Three blocks is enough to propagate information across the diameter of a typical drug molecule.

## Training block

**`n_epochs: 100`** is the maximum number of epochs the trainer (`train/trainer.py::train`) iterates over. The Kaggle notebook overrides this to 50 for wall-clock reasons. There is no early stopping but the best validation RMSE is checkpointed to `checkpoints/best_model.pt`.

**`lr: 1.0e-3`** is the Adam learning rate. The trainer constructs `torch.optim.Adam(model.parameters(), lr=cfg['lr'])` and wraps it in `ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)`.

**`weight_decay: 0.0`** disables Adam's $L_2$ regularisation. With the model at ~250k parameters and dataset sizes typically in the thousands, slight weight decay (1e-5 to 1e-4) often helps; the default is off to keep the baseline loss curve clean.

## Math notes

### TDA dimensionality

Each persistence image is an $R \times R$ raster. Two homology orders (`H₀`, connected components, and `H₁`, loops) are stacked per stream, giving $2R^2$ floats per stream. Drug and pocket each emit one such vector, so the total TDA feature dimension entering the fusion module is

$$
D_{\mathrm{TDA}} = 2 \cdot R^2 \cdot 2 = 1600 \quad \text{at } R = 20,
$$

where the leading $2$ counts $\{H_0, H_1\}$, $R^2$ is the grid size, and the trailing $2$ counts $\{\text{drug}, \text{pocket}\}$.

### Adam update with weight decay

Adam (Kingma & Ba, 2014) computes biased first/second moments $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ and $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, then bias-corrects them as $\hat m_t = m_t / (1 - \beta_1^t)$ and $\hat v_t = v_t / (1 - \beta_2^t)$. The parameter update with decoupled weight decay (`weight_decay = λ`) is

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t \right),
$$

where $\eta$ is `lr`, $g_t = \nabla_\theta \mathcal L(\theta_t)$ is the loss gradient, $\epsilon \approx 10^{-8}$ is a numerical safeguard, and $\lambda$ is `weight_decay`. With $\lambda = 0$ the second term vanishes and the update reduces to vanilla Adam.

### Parameter count budget

At the default `hidden_dim = 64`, `tda_resolution = 20`, `n_drug_interactions = 3` the model breaks down (as printed by `TopoSurfaceDTI.count_parameters()`) approximately as

| Module | Parameters |
|---|---|
| `DrugEncoder` (3 SchNet blocks) | ~30k |
| `PocketEncoder` (4 GEM-CNN blocks, irrep mixing) | ~120k |
| `FusionModule` (cross-attention + TDA MLP) | ~100k |
| **Total `TopoSurfaceDTI`** | **~250k** |

The pocket stream dominates because each GEM-CNN kernel is a learned scalar weight per (input irrep, output irrep, basis function) triple — see `models/irreps.py` and `models/gem_conv.py`.
