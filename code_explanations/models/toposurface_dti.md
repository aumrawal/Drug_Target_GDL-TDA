# `models/toposurface_dti.py`

## Overview

This is the top-level model that wires together every component in the project. `TopoSurfaceDTI` is an `nn.Module` that instantiates a `DrugEncoder` for the small-molecule side, a `PocketEncoder` for the protein binding pocket, and a `FusionModule` to combine their outputs with the persistent-homology (TDA) features. Its forward pass consumes the 11 tensors that define a drug-pocket pair — drug atom features, drug 3D positions, drug edges, pocket residue features, pocket kNN edges, pocket edge angles, pocket parallel-transport angles, drug TDA, pocket TDA — and returns a single scalar pKd estimate.

The model is the central object the rest of the pipeline (`train.py`, `dataset.py`, `run.py`, `visualize.py`) interacts with. A factory classmethod `from_config(cfg)` builds the model from a YAML/dict configuration, and `count_parameters()` reports a per-submodule parameter breakdown.

## Mathematical Foundations

### Module composition and overall symmetry

Let $\mathcal{D}$ be the drug-encoder mapping, $\mathcal{P}$ the pocket-encoder mapping, $\mathcal{T}_{\text{drug}}, \mathcal{T}_{\text{pocket}}$ the (precomputed) TDA pipelines, and $\mathcal{F}$ the fusion head. The full prediction is

$$\widehat{pK_d} \;=\; \mathcal{F}\!\Big(\mathcal{D}(\mathbf{x}^d, \mathbf{r}^d, E^d),\; \mathcal{P}(\mathbf{x}^p, E^p, \boldsymbol\theta, \mathbf{g}),\; \mathcal{T}_{\text{drug}}(\mathbf{r}^d) \oplus \mathcal{T}_{\text{pocket}}(\mathbf{r}^p) \Big),$$

where $\oplus$ denotes concatenation of the two persistence-image vectors. The pocket positions $\mathbf{r}^p$ enter only through derived geometric scalars ($E^p, \boldsymbol\theta, \mathbf{g}$, and the pocket TDA distances).

The model is $SE(3)$-invariant overall: the drug encoder consumes only interatomic distances; the pocket encoder consumes only intrinsic surface geometry (kNN graph topology, tangent-plane edge angles, parallel-transport angles) extracted via local PCA, all of which are $SE(3)$-invariant; the TDA pipelines use Vietoris-Rips filtrations of pairwise distances, also invariant. Composition preserves invariance. The pocket encoder is internally gauge-equivariant at the per-vertex level and gauge-invariant at the readout (because its final feature type is pure $\rho_0$).

### Persistent-homology feature dimensions

The TDA pipeline produces, per molecule, two persistence images: one for $H_0$ (connected components) and one for $H_1$ (1-dimensional loops). Each is a `tda_resolution × tda_resolution` grid (default $20 \times 20$). Stacked and flattened the per-molecule TDA vector has $2 \cdot 20^2 = 800$ entries, and the total TDA input to fusion has $2 \cdot 800 = 1600$ entries:

$$C_{\text{tda}}^{\text{total}} \;=\; 2 \text{ molecules} \times 2 \text{ images} \times R^2 \;=\; 4R^2, \qquad R = \mathtt{tda\_resolution}.$$

This dimension is computed in `__init__` as `tda_dim = 2 * 2 * tda_resolution ** 2` and passed to the fusion module.

### Parameter budget

Roughly 250k parameters total (per the CLAUDE.md). The breakdown produced by `count_parameters()`:

- drug encoder: input embed ($17 \to 64$) plus 3 interaction blocks (each $\sim$32k params for the two MLPs) plus norms — order $30$–$40$k.
- pocket encoder: input embed ($25 \to 24$) plus 4 GEM blocks (parameter count per block scales with $\sum_{\text{block-pair}} N_{\text{basis}} m_{\text{in}} m_{\text{out}}$) plus output projection — order $20$–$40$k.
- fusion: $4 + 4 = 8$ linear layers for the two cross-attentions (each $4 \times 64^2 \approx 16$k), TDA compression $1600 \times 128 + 128 \times 64 \approx 213$k, and the final MLP — dominated by TDA compression.

The TDA compression is by far the largest component, motivating its careful design.

## Code Walk-through

### `TopoSurfaceDTI.__init__` (lines 59–89)

Default hyperparameters:

```python
drug_in_features    = 17    # = N_ATOM_FEAT
pocket_in_features  = 25    # = N_RESIDUE_FEAT
hidden_dim          = 64
tda_resolution      = 20
n_drug_interactions = 3
```

Constructs:

```python
self.drug_encoder = DrugEncoder(
    in_features=drug_in_features,
    hidden_dim=hidden_dim,
    n_interactions=n_drug_interactions,
)
self.pocket_encoder = PocketEncoder(
    in_features=pocket_in_features,
    hidden_dim=hidden_dim,
)
tda_dim = 2 * 2 * tda_resolution ** 2   # = 1600 for default
self.fusion = FusionModule(
    drug_dim=hidden_dim,
    pocket_dim=hidden_dim,
    tda_dim=tda_dim,
    hidden_dim=256,
)
```

The fusion module's internal `hidden_dim = 256` is independent of the encoder `hidden_dim`; it controls the MLP-head width.

### `TopoSurfaceDTI.forward` (lines 91–114)

Signature:

| Arg | Shape | Role |
|---|---|---|
| `drug_x` | `(V_d, 17)` | per-atom features |
| `drug_pos` | `(V_d, 3)` | 3D atom coordinates (used only for distances) |
| `drug_edge` | `(2, E_d)` | bond edges |
| `pocket_x` | `(V_p, 25)` | per-residue features |
| `pocket_edge` | `(2, E_p)` | kNN edges |
| `pocket_angles` | `(E_p,)` | $\theta_{pq}$ |
| `pocket_trans` | `(E_p,)` | $g_{q\to p}$ |
| `drug_tda` | `(800,)` | drug $H_0 \oplus H_1$ persistence images |
| `pocket_tda` | `(800,)` | pocket $H_0 \oplus H_1$ persistence images |

The forward pass is a five-line orchestration:

```python
drug_out   = self.drug_encoder(drug_x, drug_pos, drug_edge)
pocket_out = self.pocket_encoder(pocket_x, pocket_edge,
                                 pocket_angles, pocket_trans)
tda = torch.cat([drug_tda, pocket_tda], dim=0)   # (1600,)
return self.fusion(
    drug_per_atom=drug_out['per_atom'],
    drug_global=drug_out['global'],
    pocket_per_res=pocket_out['per_vertex'],
    pocket_global=pocket_out['global'],
    tda_features=tda,
)
```

Each encoder returns a dict with `per_atom`/`per_vertex` (the local features) and `global` (the mean-pooled summary). The TDA vectors are concatenated along axis 0 to form a single 1600-dim vector before being handed to the fusion module.

### `from_config(cfg)` (lines 116–124)

Classmethod that reads the model hyperparameters from a config dict and constructs the model. Used by `run.py` after loading `configs/base.yaml`:

```python
return cls(
    drug_in_features=cfg.get('drug_in_features', 17),
    pocket_in_features=cfg.get('pocket_in_features', 25),
    hidden_dim=cfg.get('hidden_dim', 64),
    tda_resolution=cfg.get('tda_resolution', 20),
    n_drug_interactions=cfg.get('n_drug_interactions', 3),
)
```

### `count_parameters()` (lines 126–133)

Reports a four-key dictionary with the total trainable parameter count and the per-submodule counts. Useful for sanity-checking changes to feature dimensions or block depths.

## Biology / Chemistry Context

The end-to-end model targets binding affinity prediction — specifically pKd, the negative log of the dissociation constant. A drug binds tightly to a pocket when (i) its 3D shape complements the pocket geometry, (ii) its functional groups can form favourable interactions (hydrogen bonds, van der Waals contacts, $\pi$-stacking, salt bridges) with specific residues, and (iii) the overall conformation is not too entropically penalised on binding. The three streams of the model address these in turn:

- **Local chemistry** (drug encoder) captures functional groups and chemical environments — the 17-dim atom features include atom type, hybridisation, and charge, and the SchNet interactions propagate this information over short ranges.
- **Local geometry** (pocket encoder via GEM-CNN) captures pocket shape — the gauge-equivariant convolutions can learn directional patterns like "this hydrophobic patch extends along the principal pocket axis" or "the H-bond donor faces inward".
- **Global topology** (TDA) captures shape features that local message passing misses — the number of aromatic rings in the drug (drug $H_1$ persistence), the number of subpockets in the binding site (pocket $H_0$ persistence at intermediate scales), loop topology of the pocket outline.

The cross-attention in fusion then learns which drug atoms interact with which pocket residues — implicitly inducing a pharmacophore-like compatibility matrix without it being hard-coded.

The full model is $SE(3)$-invariant by construction. Translational and rotational symmetries are physical realities of binding free energy and forcing them in by design (rather than learning them from data) is a well-established advantage of geometric deep learning for molecular property prediction.

## References

- Wang, R., Fang, X., Lu, Y., Wang, S. *The PDBbind database: collection of binding affinities for protein-ligand complexes with known three-dimensional structures.* J. Med. Chem. 47, 2977–2980 (2004). The standard benchmark for pKd prediction.
- de Haan, P., Weiler, M., Cohen, T., Welling, M. *Gauge Equivariant Mesh CNNs.* ICLR 2021. <https://arxiv.org/abs/2003.05425>. Pocket-encoder backbone.
- Schütt, K. T. et al. *SchNet — A deep learning architecture for molecules and materials.* J. Chem. Phys. 148, 241722 (2018). Drug-encoder backbone.
- Edelsbrunner, H., Harer, J. *Computational Topology: An Introduction.* AMS, 2010. Reference for persistent homology.
- Adams, H. et al. *Persistence Images: A Stable Vector Representation of Persistent Homology.* JMLR 18, 1–35 (2017). TDA vectorisation.
- Bronstein, M. M., Bruna, J., Cohen, T., Veličković, P. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* (2021) <https://arxiv.org/abs/2104.13478>. Unified framework for the equivariance design choices made here.
- Jiménez-Luna, J., Grisoni, F., Schneider, G. *Drug discovery with explainable artificial intelligence.* Nature Mach. Intell. 2, 573–584 (2020). Context for cross-attention as an interpretability mechanism in DTI.
