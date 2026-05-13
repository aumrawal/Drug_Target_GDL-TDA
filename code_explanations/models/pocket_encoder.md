# `models/pocket_encoder.py`

## Overview

This module implements the protein binding-pocket encoder, a stack of Gauge Equivariant Mesh CNN blocks (`GEMBlock` from `gem_conv.py`) that operates on a kNN graph of Cα atoms. The input is a per-residue feature tensor of dimension `N_RESIDUE_FEAT = 25` along with per-edge geometry (edge angles $\theta_{pq}$ in the tangent plane and parallel-transport angles $g_{q\to p}$), and the output is a per-residue equivariant embedding of shape $(V, 64)$ plus a gauge-invariant global pocket fingerprint of shape $(64,)$. The local tangent frames at each residue are estimated via local PCA in `data/pocket_mesh.py` (`estimate_normals_pca`), making GEM applicable to point clouds without requiring an explicit triangulated surface.

The encoder is designed around a carefully chosen feature-type progression: scalars at the input and output (so the readout is gauge-invariant), and mixed scalar/vector representations in the middle (so the network can learn directional surface patterns). This is the protein-side counterpart to the drug encoder and produces signals that the fusion module cross-attends against.

## Mathematical Foundations

### Feature-type progression

A *feature type* is a list of `(irrep_order, multiplicity)` pairs that fully specifies a vector space on which $SO(2)$ acts. The encoder threads features through five types:

| Stage | Feature type | Total dim | Role |
|---|---|---|---|
| Input embed | $24\rho_0$ | 24 | Scalar lifting of raw residue features |
| Block 1 | $8\rho_0 \oplus 8\rho_1$ | $8 + 16 = 24$ | Introduce vector channels |
| Block 2 | $16\rho_0 \oplus 16\rho_1$ | $16 + 32 = 48$ | Grow width |
| Block 3 | $16\rho_0 \oplus 16\rho_1$ | 48 | Deepen |
| Block 4 (output) | $32\rho_0$ | 32 | Project back to scalars for invariant readout |

Two design decisions are worth flagging. First, the input embed maps from a regular $\mathbb{R}^{25}$ residue feature to a $24\rho_0$ block: only scalars are introduced at the start because raw residue features (amino-acid identity, hydropathy, etc.) have no intrinsic direction. Second, the final block projects to pure scalars *before* pooling, which guarantees that the global mean is exactly gauge-invariant — invariant under any per-vertex frame rotation $g_p$ — without needing to invoke an approximation argument.

### Equivariance of the per-vertex outputs

For intermediate features of type $\sum_i (n_i, m_i)$, the network is *gauge-equivariant*: a gauge transformation at vertex $p$ with angle $g_p$ acts block-wise as $\rho_{n_i}(g_p)$. The kernels (see `irreps.py`) satisfy the constraint $K(\theta - g) = \rho_{\text{out}}(-g)\,K(\theta)\,\rho_{\text{in}}(g)$, the transporters rotate neighbour features into the centre's frame, and the norm-nonlinearity preserves equivariance because it depends only on $\lVert v \rVert$. So intermediate per-residue features transform covariantly; only after projecting to $32\rho_0$ does the representation become gauge-invariant.

### Invariance of the global pool

After the last block produces an output of type $32\rho_0$ at each vertex — meaning a $32$-dim vector that is unchanged by any gauge transformation — the global mean

$$\mathbf{g} \;=\; \frac{1}{V} \sum_{p=1}^{V} \mathbf{h}_p, \qquad \mathbf{h}_p \in \mathbb{R}^{32}$$

is invariant under both gauge transformations (because each $\mathbf{h}_p$ is) and permutations of the residue ordering (because mean pooling is symmetric). For binding-affinity prediction we ultimately need a scalar that is also $SE(3)$-invariant in the embedding 3D space; SE(3)-invariance follows because the kNN graph topology, edge angles in the tangent plane, and parallel-transport angles are all derived from intrinsic surface quantities (local PCA of neighbour offsets) that are themselves invariant under rigid motion of the protein in space.

### Putting it all together

The full mapping computed by `PocketEncoder` is

$$\text{Embed}(x_p) \;\xrightarrow{\;\text{GEMBlock}_1\;}\; \text{GEMBlock}_2 \;\xrightarrow{}\; \text{GEMBlock}_3 \;\xrightarrow{}\; \text{GEMBlock}_4 \;\xrightarrow{}\; \big(\mathbf{h}_p\big)_{p=1}^V \in \mathbb{R}^{V \times 32} \;\xrightarrow{\text{proj}}\; \mathbb{R}^{V \times 64},$$

with the global vector obtained by mean-pooling the pre-projection $\mathbb{R}^{32}$ features and then applying the same projection.

## Code Walk-through

### `PocketEncoder.__init__` (lines 48–74)

```python
ftype_embed = [(0, 24)]             # 24 ρ_0
ftype_s     = [(0, 8),  (1, 8)]     # 8 ρ_0  ⊕ 8 ρ_1   (dim 24)
ftype_m     = [(0, 16), (1, 16)]    # 16 ρ_0 ⊕ 16 ρ_1  (dim 48)
ftype_out   = [(0, 32)]             # 32 ρ_0
```

- `self.input_embed = nn.Linear(in_features, 24)`: maps the raw 25-dim residue feature to a 24-dim scalar block. Plain linear, no equivariance constraint required because both input and output are scalars under the gauge group.
- `self.blocks`: a `ModuleList` of four `GEMBlock` instances chaining the feature types `embed → s → m → m → out`.
- `self.out_proj`: a `nn.Linear(32, hidden_dim)` if `hidden_dim ≠ 32`, else `nn.Identity()`. Default `hidden_dim = 64` so this is a linear projection from 32 to 64.

### `PocketEncoder.forward` (lines 76–95)

Takes:
- `x: (V, 25)` — per-residue features
- `edge_index: (2, E)` — kNN graph indices
- `angles: (E,)` — edge angles $\theta_{pq}$ in the tangent plane at $p$
- `transporters: (E,)` — parallel-transport angles $g_{q\to p}$

Executes:

```python
h = self.input_embed(x)               # (V, 24), scalar
for block in self.blocks:
    h = block(h, edge_index, angles, transporters)
# h: (V, 32) — pure scalar, gauge-invariant

per_vertex = self.out_proj(h)         # (V, 64)
global_emb = self.out_proj(h.mean(dim=0))   # (64,)
```

Notice that the projection is applied both to the per-vertex features and to the mean — `self.out_proj` is a single shared linear map. This keeps the per-vertex output and the global readout in the same space, which is important because both are consumed by the fusion module: per-vertex features feed cross-attention, the global vector is concatenated with the cross-attention pool and TDA signals.

A subtle point: applying `out_proj` after the mean rather than before is mathematically equivalent (linearity) but slightly cheaper.

## Biology / Chemistry Context

A protein binding pocket is a concave region on the protein's solvent-accessible surface where a drug molecule can sit. In this codebase, the pocket is defined as the set of residues whose Cα atom is within `pocket_cutoff = 10 Å` of the ligand centroid (`data/pocket_mesh.py: extract_pocket_atoms`). Each pocket residue becomes a graph node carrying a 25-dim feature: a 21-dim one-hot of amino-acid identity (20 standard amino acids plus one "unknown" slot), three coordinates of the Cα position relative to a local origin, and a scalar distance from the ligand centroid.

The pocket surface is curved — it must follow the protein's tertiary structure — and the choice of tangent-plane axes at each residue is arbitrary. Gauge equivariance is the right framework for this geometry because it forces the network to make predictions that do not depend on how the local frame was chosen by `estimate_normals_pca`. Vector channels ($\rho_1$) in the intermediate layers can learn directional patterns: for example, "the polar residue lies along the direction of steepest pocket descent" or "a hydrophobic patch extends in a coherent direction across several residues". These directional signals matter for binding because a drug molecule's geometry must complement the pocket's shape, not merely sit nearby.

The kNN graph (typically k=10) connects each Cα to its 10 spatially nearest pocket neighbours; this captures both sequence-adjacent residues and through-space neighbours from distant loops folded into the same binding site. The four-block depth gives a receptive field of four kNN hops, comfortably enough to integrate information across the whole pocket (typical pockets have 30–60 residues).

## References

- de Haan, P., Weiler, M., Cohen, T., Welling, M. *Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs.* ICLR 2021. <https://arxiv.org/abs/2003.05425>. The core method, including the feature-type framework used here.
- Gainza, P., Sverrisson, F., Monti, F., Rodolà, E., Boscaini, D., Bronstein, M. M., Correia, B. E. *Deciphering interaction fingerprints from protein molecular surfaces using geometric deep learning (MaSIF).* Nat. Methods 17, 184–192 (2020). A precursor approach using geodesic CNNs on protein surfaces.
- Sverrisson, F., Feydy, J., Correia, B. E., Bronstein, M. M. *Fast end-to-end learning on protein surfaces (dMaSIF).* CVPR 2021. <https://arxiv.org/abs/2009.14165>. Differentiable point-cloud variant.
- Townshend, R. J. L., Vögele, M., Suriana, P., et al. *ATOM3D: Tasks on Molecules in Three Dimensions.* NeurIPS 2021. <https://arxiv.org/abs/2012.04035>. Benchmark suite that includes binding-affinity prediction tasks.
