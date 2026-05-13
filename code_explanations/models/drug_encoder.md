# `models/drug_encoder.py`

## Overview

This module implements the drug-molecule encoder, an SE(3)-invariant graph neural network in the SchNet family (Schütt et al. 2017, 2018). It maps a small drug molecule — represented as a 3D point cloud of atoms with per-atom feature vectors of dimension `N_ATOM_FEAT = 17` and an edge index over chemical bonds — to two tensors: a per-atom embedding of shape $(V, 64)$ and a global molecular fingerprint of shape $(64,)$ obtained by mean pooling. These outputs feed into the cross-attention fusion module together with the pocket embeddings and the TDA features.

The SchNet design uses *continuous-filter convolutions*: each edge contributes a message whose filter weights are produced by an MLP applied to a radial basis expansion of the interatomic distance $r_{pq} = \lVert \mathbf{r}_p - \mathbf{r}_q \rVert$. Because the network sees only distances — not positions or directions — every operation is automatically invariant under rigid motions of the molecule. This is the appropriate symmetry for predicting binding affinity, which is a rotation- and translation-invariant scalar.

## Mathematical Foundations

### SchNet-style continuous-filter convolutions

Given a molecular graph with vertices $\{1, \dots, V\}$, 3D positions $\mathbf{r}_p \in \mathbb{R}^3$, edge index $E$, and initial atom features $\mathbf{x}_p \in \mathbb{R}^{17}$, the encoder produces hidden features $\mathbf{h}_p \in \mathbb{R}^{C}$ ($C = 64$) by an embedding followed by $T$ interaction blocks. Each block updates

$$\mathbf{h}_p^{(t+1)} \;=\; \mathrm{LayerNorm}\!\left( \mathbf{h}_p^{(t)} \;+\; W_{\text{out}}\!\Big( \sum_{q \in \mathcal{N}(p)} W_{\text{filter}}\big(\mathbf{e}(r_{pq})\big) \odot \mathbf{h}_q^{(t)} \Big) \right),$$

where $\odot$ is elementwise multiplication, $\mathbf{e}(r) \in \mathbb{R}^{K}$ is the radial-basis expansion of distance, $W_{\text{filter}}$ and $W_{\text{out}}$ are two-layer MLPs with SiLU activations, and $\mathcal{N}(p)$ is the bonded neighbourhood (filtered to $r_{pq} < r_{\text{cut}} = 5$ Å).

### Radial basis function expansion

Distances are embedded into $K = 32$ Gaussian basis functions:

$$e_k(r) \;=\; \exp\!\Big(-\gamma\,(r - \mu_k)^2\Big), \qquad \mu_k \in \{0, \tfrac{r_{\text{cut}}}{K-1}, \dots, r_{\text{cut}}\}, \qquad \gamma \;=\; \frac{K}{r_{\text{cut}}^2}.$$

This is a soft one-hot encoding of distance — each basis function fires when $r$ is close to its centre $\mu_k$, with a width controlled by $\gamma$. The width $1/\sqrt{\gamma} = r_{\text{cut}}/\sqrt{K} \approx 0.88\,\text{Å}$ in the default setting is comparable to a covalent-bond-length resolution. The radial basis serves three purposes: (i) it gives the filter MLP a smooth, differentiable input even though distances live on $\mathbb{R}_{\ge 0}$ rather than a vector space; (ii) it provides enough capacity to represent any radial filter the network might need; (iii) it decouples the filter shape from how distances are stored, so the network is not forced to learn $f(r) = \exp(-\cdots)$ priors itself.

### SE(3) invariance argument

The molecule lives in $\mathbb{R}^3$ with the action of $SE(3) = SO(3) \ltimes \mathbb{R}^3$ on positions: $\mathbf{r}_p \mapsto R\mathbf{r}_p + \mathbf{t}$ for $R \in SO(3), \mathbf{t} \in \mathbb{R}^3$. The pairwise distance is invariant:

$$\lVert (R\mathbf{r}_p + \mathbf{t}) - (R\mathbf{r}_q + \mathbf{t}) \rVert \;=\; \lVert R(\mathbf{r}_p - \mathbf{r}_q) \rVert \;=\; \lVert \mathbf{r}_p - \mathbf{r}_q \rVert,$$

because $R$ is orthogonal. The atom features $\mathbf{x}_p$ are encodings of chemical properties (atom type, hybridisation, charge) that are themselves invariant under rigid motions. Since every operation downstream of the input takes only $(\mathbf{x}, r)$, the output is invariant under $SE(3)$. Translation invariance is automatic for the same reason. Mean pooling at the end preserves the property.

This is a deliberate restriction relative to fully $SE(3)$-equivariant networks (Tensor Field Networks, SE(3)-Transformer, EGNN). Equivariant networks let intermediate features rotate with the molecule; invariant networks throw that information away. For a global scalar prediction like pKd this is a reasonable trade-off — directional information about the drug is captured globally through the persistent-homology pipeline, and locally through the pocket-side GEM-CNN.

## Code Walk-through

### `RBFExpansion` (lines 25–40)

Stores `n_basis = 32` centers `μ_k` linearly spaced on $[0, \text{cutoff}]$ as a non-trainable buffer (line 35), and precomputes the inverse width

$$\gamma \;=\; \frac{K}{r_{\text{cut}}^2}.$$

In the forward pass it broadcasts an edge-distance vector `r` of shape `(E,)` against the centers of shape `(K,)` to produce `(E, K)` Gaussian features. Note that this `γ` is fixed (not learnable) — its scale follows from the cutoff and basis count.

### `InteractionBlock` (lines 43–81)

Holds three submodules:

- `filter_net`: $\mathbb{R}^{32} \to \mathbb{R}^{64} \to \mathbb{R}^{64}$ with SiLU activation. Maps the RBF expansion to per-edge filter weights $W_{pq}$.
- `out_proj`: $\mathbb{R}^{64} \to \mathbb{R}^{64} \to \mathbb{R}^{64}$ with SiLU. Post-aggregation projection.
- `norm`: `nn.LayerNorm(hidden_dim)`.

The forward pass (lines 66–81) computes per-edge messages by elementwise gating: `W = filter_net(rbf)` of shape `(E, 64)`, then `msg = W * h[src]` of shape `(E, 64)`. The aggregation is a scatter-add along the target index. The block returns the residual update

```python
return self.norm(h + self.out_proj(agg))
```

implementing the equation above.

### `DrugEncoder` (lines 84–149)

The full encoder.

- `__init__` (lines 100–117) builds an input linear `embed: 17 → 64` (line 111), an `RBFExpansion(cutoff=5.0, n_basis=32)`, and a `ModuleList` of `n_interactions = 3` `InteractionBlock`s.

- `forward` (lines 119–149) takes `x: (V, 17)`, `pos: (V, 3)`, `edge_index: (2, E)` and:
  1. Computes interatomic distances along edges: `r = (pos[src] - pos[tgt]).norm(dim=-1)`. This is the only point where 3D positions enter — afterwards everything is invariant.
  2. Filters edges with `r < self.cutoff` (lines 131–138). The fallback `pass` branch keeps all edges if none pass the cutoff, so isolated atoms still get an embedding.
  3. Expands distances via RBF to `(E, 32)`.
  4. Embeds atoms to `(V, 64)`.
  5. Runs three `InteractionBlock`s sequentially over `(h, edge_index, rbf)`.
  6. Returns a dictionary with `per_atom: (V, 64)` and `global: (64,)` where global is `h.mean(dim=0)`.

The mean-pool is the simplest permutation-invariant readout and is consistent with predicting a scalar molecule-level property. It also makes the global vector well-defined regardless of how many atoms the molecule has.

## Biology / Chemistry Context

The 17-dimensional atom features (built in `data/molecule_graph.py`) decompose as a 10-dim atom-type one-hot, a 4-dim hybridisation one-hot, and 3 scalar properties — typically (degree, aromaticity flag, formal charge or similar). The edge index encodes covalent bonds derived from the SDF or PDB ligand block by RDKit. 3D positions come from the same source, giving distances that reflect the molecule's actual conformer rather than a 2D drawing.

A drug molecule's binding affinity to a protein depends on many local chemical signatures — the presence of an aromatic ring near a hydrophobic pocket lining, an H-bond donor at a specific position, the flexibility of a torsion. SchNet-style continuous-filter convolutions are well suited to learning such signatures because each interaction block can in principle learn a different "distance fingerprint" — bond-length range for $\sigma$-bonds, ring-spacing range for $\pi$-stacking distances, van-der-Waals contact range, and so on. With $T = 3$ interaction blocks the receptive field extends to all atoms within 3 bonds (or 3 × cutoff radii in space), enough to capture functional-group-level chemistry.

What this encoder *cannot* directly see is global molecular topology — for example, whether the molecule has macrocyclic loops, how many aromatic rings it has, or the persistence of contact patterns at various scales. That information is supplied separately by the persistent-homology pipeline (`data/tda_features.py`) and fused later in `FusionModule`.

## References

- Schütt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A., Müller, K.-R. *SchNet — A deep learning architecture for molecules and materials.* J. Chem. Phys. 148, 241722 (2018). <https://arxiv.org/abs/1712.06113>
- Schütt, K. T., Arbabzadah, F., Chmiela, S., Müller, K. R., Tkatchenko, A. *Quantum-chemical insights from deep tensor neural networks.* Nature Comm. 8, 13890 (2017).
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., Dahl, G. E. *Neural Message Passing for Quantum Chemistry.* ICML 2017. <https://arxiv.org/abs/1704.01212>. The general MPNN framework SchNet is a special case of.
- Satorras, V. G., Hoogeboom, E., Welling, M. *E(n) Equivariant Graph Neural Networks.* ICML 2021. <https://arxiv.org/abs/2102.09844>. A more recent design exploring the same invariance regime.
- Schomburg, K. T., Bietz, S., Briem, H., Henzler, A. M., Urbaczek, S., Rarey, M. *Facing the challenges of structure-based target prediction by inverse virtual screening.* J. Chem. Inf. Model. 54(6), 1676–1686 (2014). Background on structure-based DTI.
