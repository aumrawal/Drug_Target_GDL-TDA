# TopoSurface-DTI

A drug-target interaction (binding affinity) model that fuses two geometric deep learning approaches to predict pKd given a drug molecule and a protein binding pocket.

- **GEM-CNN** (Gauge Equivariant Mesh CNN) on the protein binding pocket surface
- **Persistent Homology** (TDA) — Vietoris-Rips filtration over both drug and protein, vectorized as persistence images

## Architecture

```
Drug (17-dim atom feats + 3D pos)
  → DrugEncoder      → per_atom (V×64), global (64)
  → TDA pipeline     → H₀+H₁ persistence images → (800,)
                                    ↓
                             FusionModule
                          cross-attention (drug↔pocket)
                          + TDA feature injection
                          → MLP → pKd scalar
Protein pocket (25-dim residue feats + kNN graph geometry)
  → PocketEncoder    → per_vertex (V×64), global (64)
  → TDA pipeline     → H₀+H₁ persistence images → (800,)
```

### Equivariance

| Component | Property |
|---|---|
| DrugEncoder | SE(3)-invariant (interatomic distances via RBF) |
| PocketEncoder | Gauge-equivariant via GEM-CNN |
| TDA features | SE(3)-invariant and permutation-invariant |
| Overall model | SE(3)-invariant |

### Feature dimensions

| Quantity | Value |
|---|---|
| Atom features (`N_ATOM_FEAT`) | 17 |
| Residue features (`N_RESIDUE_FEAT`) | 25 |
| TDA vector per molecule | 800 (2 images × 20² grid) |
| Total TDA input to fusion | 1600 |
| Total model parameters | ~250k |

## Setup

```bash
pip install torch numpy scipy pyyaml ripser
pip install biopython rdkit-pypi   # only needed for real PDBBind data
```

## Usage

```bash
# Train on synthetic data (no external files required)
python run.py

# Train on real PDBBind data
python run.py --config configs/base.yaml --data /path/to/PDBBind --no-synthetic

# Resume from checkpoint
python run.py --resume checkpoints/best_model.pt

# Visualise predicted vs actual pKd
python visualize.py
python visualize.py --train-first        # train then plot in one shot
python visualize.py --n-samples 500

# Quick forward-pass sanity check (no training)
python -c "
import sys; sys.path.insert(0, '.')
from data.molecule_graph import synthetic_drug_graph
from data.pocket_mesh import synthetic_pocket_graph
from data.tda_features import compute_tda_features, tda_to_tensor
from models.toposurface_dti import TopoSurfaceDTI
drug = synthetic_drug_graph(); pocket = synthetic_pocket_graph()
model = TopoSurfaceDTI()
pred = model(drug['x'], drug['pos'], drug['edge_index'],
             pocket['x'], pocket['edge_index'], pocket['angles'], pocket['transporters'],
             tda_to_tensor(compute_tda_features(drug['pos'].numpy())),
             tda_to_tensor(compute_tda_features(pocket['pos'].numpy(), max_edge_len=16.0)))
print(pred.item())
"
```

## Real Data Format (PDBBind)

```
data_dir/
  {pdbid}/
    {pdbid}_protein.pdb   ← Cα positions; ligand embedded as 'LIG' OR
    {pdbid}_ligand.sdf    ← separate SDF (standard PDBBind layout)
  train_split.json        ← [{"id": "1abc", "affinity": 7.5}, ...]
  val_split.json
  test_split.json
```

Pocket residues are Cα atoms within 10 Å of the ligand centroid, connected as a k-NN graph (k=10).

## Kaggle

Upload the project folder as a dataset named `drug-target-gdl`, then run `kaggle_run.ipynb`. It installs `ripser`/`pyyaml`, trains on synthetic data, and produces a 4-panel visualisation.

## Project Structure

```
Drug_target_GDL/
├── configs/
│   └── base.yaml              # hyperparameters and data paths
├── data/
│   ├── dataset.py             # DTIDataset (synthetic + real PDBBind)
│   ├── mesh_geometry.py       # tangent frame utilities
│   ├── molecule_graph.py      # drug atom featurisation
│   ├── pocket_mesh.py         # pocket extraction, normal estimation
│   └── tda_features.py        # Ripser-backed persistence images
├── models/
│   ├── drug_encoder.py        # SchNet-style SE(3)-invariant encoder
│   ├── pocket_encoder.py      # GEM-CNN gauge-equivariant encoder
│   ├── gem_conv.py            # GEM convolution layer
│   ├── irreps.py              # SO(2) irrep kernel basis
│   ├── fusion.py              # cross-attention + TDA fusion + MLP head
│   └── toposurface_dti.py     # top-level model
├── run.py                     # training entry point
├── visualize.py               # prediction scatter plot
└── kaggle_run.ipynb           # end-to-end Kaggle notebook
```

## References

- de Haan et al. (2020) — *Gauge Equivariant Mesh CNNs*
- Edelsbrunner & Harer — *Computational Topology* (persistent homology)
- Schütt et al. (2017) — *SchNet: A continuous-filter convolutional neural network*
