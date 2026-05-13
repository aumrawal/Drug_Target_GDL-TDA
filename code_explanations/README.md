# TopoSurface-DTI Code Explanations

This directory is a guided tour of the **TopoSurface-DTI** codebase — a drug-target interaction model that predicts binding affinity (pKd) by fusing two complementary geometric deep learning streams: a **Gauge Equivariant Mesh CNN (GEM-CNN)** over the protein binding pocket surface, and **persistent-homology** topological features computed for both drug and protein. A third, SE(3)-invariant SchNet-style encoder processes the drug molecule graph. The three representations are merged in a cross-attention fusion module that outputs a single scalar pKd. Every Python module in the repository has a corresponding explainer in this folder; this README is the index and the recommended starting point.

## Mathematical Overview

TopoSurface-DTI is a dual-stream geometric architecture with a topological side-channel. Let $D = (X_d, E_d, P_d)$ denote a drug molecule with atom features $X_d \in \mathbb R^{V_d \times 17}$, bond edges $E_d$, and 3D coordinates $P_d \in \mathbb R^{V_d \times 3}$. Let $P = (X_p, E_p, P_p)$ denote the protein pocket as a kNN graph over Cα atoms with residue features $X_p \in \mathbb R^{V_p \times 25}$.

**Drug stream — SE(3)-invariant SchNet.** The drug encoder builds invariant features purely from pairwise distances. With $h_i^{(0)}$ the embedded atom features, each of `n_drug_interactions` blocks performs

$$
h_i^{(t+1)} = h_i^{(t)} + \sum_{j \in \mathcal N(i)} \phi\big(h_j^{(t)}\big) \odot \psi\!\big(\lVert x_i - x_j \rVert\big),
$$

where $\mathcal N(i)$ is the neighbour set of atom $i$, $\phi$ is a per-atom MLP, $\psi$ expands the scalar distance in a radial basis followed by an MLP, $\odot$ is elementwise multiplication, and $\lVert \cdot \rVert$ is the Euclidean norm. Because the only geometric input is $\lVert x_i - x_j \rVert$ — invariant under rotations and translations — the output is SE(3)-invariant by construction. See [models/drug_encoder.md](models/drug_encoder.md).

**Pocket stream — Gauge-equivariant GEM-CNN.** The pocket encoder treats the kNN point cloud as a discrete surface. At each vertex $p$ a local tangent frame is estimated by PCA on its neighbours, and features carry an SO(2) representation type. The GEM convolution at vertex $p$ aggregates neighbour features through parallel-transported kernels:

$$
f_p^{\mathrm{out}} = \sum_{q \in \mathcal N(p)} K(\theta_{pq}) \, \rho_{\mathrm{in}}(g_{q \to p}) \, f_q^{\mathrm{in}},
$$

where $\theta_{pq}$ is the angle of the edge to vertex $q$ in $p$'s local frame, $g_{q \to p}$ is the parallel-transport rotation from $q$'s frame to $p$'s, and $\rho_{\mathrm{in}}$, $\rho_{\mathrm{out}}$ are SO(2) irrep representations of the input and output feature types. Gauge equivariance forces every kernel to satisfy the constraint

$$
K(\theta - g) = \rho_{\mathrm{out}}(-g) \, K(\theta) \, \rho_{\mathrm{in}}(g),
$$

solved analytically by a fixed angular basis (Table 1 of de Haan et al., 2020) with learned scalar weights — see [models/irreps.md](models/irreps.md) and [models/gem_conv.md](models/gem_conv.md). The final block emits only scalar (ρ₀) features, so a global mean pool gives a gauge-invariant pocket embedding.

**TDA stream — persistent homology via Vietoris-Rips.** For each point cloud $X \in \{P_d, P_p\}$ we form the Vietoris-Rips filtration

$$
\mathrm{VR}_\epsilon(X) = \big\{\sigma \subseteq X \;\big|\; \mathrm{diam}(\sigma) \le \epsilon\big\}, \qquad \epsilon \in [0, \epsilon_{\max}],
$$

i.e. the family of simplicial complexes whose simplices are point subsets of diameter at most $\epsilon$. As $\epsilon$ grows, homology classes are born and later die; each class is recorded as a point $(b, d) \in \mathbb R^2$ in a persistence diagram, separately for each homology dimension $H_0$ (connected components) and $H_1$ (loops). The diagrams are vectorised by **persistence images**: each point is mapped to $(b, d-b)$ and replaced with a weighted 2D Gaussian, rendered onto an $R \times R$ grid, and flattened. With $R = 20$ each stream yields an 800-dim vector (400 for $H_0$, 400 for $H_1$); concatenating drug and pocket gives the $D_{\mathrm{TDA}} = 1600$ TDA input. See [data/tda_features.md](data/tda_features.md).

**Fusion — cross-attention plus MLP.** The fusion module computes scaled dot-product cross-attention with the drug embedding as queries and the pocket embedding as keys/values (and vice versa), concatenates the two attended representations with the global drug and pocket pools and the TDA vector, and projects through a small MLP to a scalar:

$$
\widehat{pK_d} = \mathrm{MLP}\big( [z_d, z_p, \mathrm{CrossAttn}(z_d, z_p), \mathrm{TDA}_d, \mathrm{TDA}_p] \big).
$$

All three streams produce rotation/translation invariant outputs, so the overall model is SE(3)-invariant — a necessary property for a physical quantity like binding affinity. See [models/fusion.md](models/fusion.md) and [models/toposurface_dti.md](models/toposurface_dti.md).

## Directory Tree

```
code_explanations/
├── README.md                       ← you are here
├── configs/
│   └── base.yaml.md
├── data/
│   ├── molecule_graph.md
│   ├── pocket_mesh.md
│   ├── mesh_geometry.md
│   ├── tda_features.md
│   └── dataset.md
├── models/
│   ├── drug_encoder.md
│   ├── irreps.md
│   ├── gem_conv.md
│   ├── pocket_encoder.md
│   ├── fusion.md
│   └── toposurface_dti.md
├── train/
│   └── trainer.md
├── scripts/
│   └── make_splits.md
├── run.md
├── visualize.md
└── kaggle_run.ipynb.md
```

## Table of Contents

### Configuration

| Doc | What it covers |
|---|---|
| [configs/base.yaml.md](configs/base.yaml.md) | Every hyperparameter in `configs/base.yaml`, how it propagates, and the math for TDA dimensionality, the Adam-with-weight-decay update, and the ~250k parameter budget. |

### Data pipeline (`data/`)

| Doc | What it covers |
|---|---|
| [data/molecule_graph.md](data/molecule_graph.md) | Atom featurisation (17-dim), RDKit `mol_to_graph`, synthetic drug generator. |
| [data/pocket_mesh.md](data/pocket_mesh.md) | Pocket extraction from PDB + SDF, residue featurisation (25-dim), kNN graph, PCA tangent-frame estimation. |
| [data/mesh_geometry.md](data/mesh_geometry.md) | Angle and parallel-transport precomputation on local frames. |
| [data/tda_features.md](data/tda_features.md) | Vietoris-Rips filtration, Ripser backend and GF(2) fallback, persistence-image vectorisation. |
| [data/dataset.md](data/dataset.md) | `DTIDataset` interface, synthetic vs real loading, split JSON format. |

### Models (`models/`)

| Doc | What it covers |
|---|---|
| [models/drug_encoder.md](models/drug_encoder.md) | SE(3)-invariant SchNet drug encoder with RBF distance expansion. |
| [models/irreps.md](models/irreps.md) | SO(2) irreps, the $K(\theta - g) = \rho_{\mathrm{out}}(-g) K(\theta) \rho_{\mathrm{in}}(g)$ constraint, equivariant kernel basis. |
| [models/gem_conv.md](models/gem_conv.md) | `GEMConv` kernel evaluation, parallel transport, neighbour aggregation. |
| [models/pocket_encoder.md](models/pocket_encoder.md) | Four-block GEM-CNN with irrep schedule `24ρ₀ → 8ρ₀⊕8ρ₁ → 16ρ₀⊕16ρ₁ → 16ρ₀⊕16ρ₁ → 32ρ₀`. |
| [models/fusion.md](models/fusion.md) | Cross-attention, TDA injection, final MLP. |
| [models/toposurface_dti.md](models/toposurface_dti.md) | Top-level module wiring, `from_config`, `count_parameters`. |

### Training (`train/`)

| Doc | What it covers |
|---|---|
| [train/trainer.md](train/trainer.md) | `train`, `train_epoch`, `validate`, `forward_step`, Adam + ReduceLROnPlateau, Huber loss, checkpointing. |

### Scripts and entry points

| Doc | What it covers |
|---|---|
| [scripts/make_splits.md](scripts/make_splits.md) | Generates `train/val/test_split.json` from a PDBBind directory tree. |
| [run.md](run.md) | CLI entry point: config loading, CLI flag overrides, train/resume orchestration. |
| [visualize.md](visualize.md) | Four-panel prediction-vs-actual figure (scatter, residuals, ranking, CDF). |
| [kaggle_run.ipynb.md](kaggle_run.ipynb.md) | Kaggle notebook that wraps setup, training, and visualisation into one file. |

## Reading Order

If you are new to the codebase, walking the docs in this sequence will get you from "what does each function take" to "why does the architecture look the way it does":

1. [README.md](README.md) — this file, for the overall picture.
2. [data/molecule_graph.md](data/molecule_graph.md) — how a drug becomes a graph tensor.
3. [data/pocket_mesh.md](data/pocket_mesh.md) — how a PDB pocket becomes a point cloud with tangent frames.
4. [data/tda_features.md](data/tda_features.md) — how persistent homology produces fixed-length vectors.
5. [models/drug_encoder.md](models/drug_encoder.md) — the simpler of the two encoders; warm-up for SchNet-style message passing.
6. [models/irreps.md](models/irreps.md) — the SO(2) algebra that powers GEM-CNN.
7. [models/gem_conv.md](models/gem_conv.md) — the gauge-equivariant convolution itself.
8. [models/pocket_encoder.md](models/pocket_encoder.md) — how `gem_conv` blocks are stacked.
9. [models/fusion.md](models/fusion.md) — how the three streams combine.
10. [models/toposurface_dti.md](models/toposurface_dti.md) — top-level model.
11. [train/trainer.md](train/trainer.md) — optimisation loop.
12. [run.md](run.md) — how everything is launched from the CLI.

## Background Concepts

The table below is a one-line glossary of the heavier terms scattered through the docs. Each links forward to the doc where it is used most concretely.

| Term | Short definition |
|---|---|
| **SE(3) equivariance** | A function $f$ is SE(3)-equivariant if $f(R x + t) = R f(x) + t$ for every rotation $R \in \mathrm{SO}(3)$ and translation $t \in \mathbb R^3$. *Invariance* is the special case $f(R x + t) = f(x)$ — used here because pKd is a scalar physical quantity. |
| **Gauge equivariance** | A function on a manifold is gauge-equivariant if its outputs transform consistently when the local coordinate frame at each point is rotated. For 2-manifolds this reduces to SO(2) equivariance at every vertex; see [models/irreps.md](models/irreps.md). |
| **Persistent homology** | A multi-scale algebraic-topology invariant of a filtration of spaces; tracks when topological features (components, loops, voids) are born and die as the scale parameter increases. |
| **Betti number** | $\beta_k$ counts the rank of the $k$-th homology group: $\beta_0$ = connected components, $\beta_1$ = independent loops, $\beta_2$ = enclosed voids. Persistent homology generalises Betti numbers across a filtration. |
| **Persistence image** | A fixed-length vectorisation of a persistence diagram: each $(b, d)$ point becomes a weighted Gaussian on the $(b, d-b)$ plane, rasterised onto a grid. Differentiable-friendly and L²-stable under bottleneck perturbations. |
| **Parallel transport** | A way of moving a vector along a curve on a manifold while preserving its meaning relative to the local frame. In GEM-CNN, $g_{q \to p}$ is the rotation that transports vertex $q$'s tangent vectors into vertex $p$'s frame so neighbour features can be compared in one common frame. |
| **SO(2) irrep** | An irreducible representation of the 2D rotation group: $\rho_0$ is the trivial 1-dim representation (scalars), and $\rho_n$ for $n \ge 1$ is the 2-dim rotation-by-$n\theta$ representation. Features in GEM-CNN are decomposed into direct sums of $\rho_n$'s. |
| **pKd** | $-\log_{10} K_d$, the negative base-10 logarithm of the dissociation constant in molar units. A pKd of $7$ means $K_d = 10^{-7}\,\mathrm M$. Larger pKd = tighter binding. |
