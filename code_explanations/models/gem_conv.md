# `models/gem_conv.py`

## Overview

This module implements the Gauge Equivariant Mesh Convolution layer (`GEMConv`), the parallel-transport routine that rotates neighbour features into the centre vertex's local frame, an approximately-equivariant nonlinearity (`RegularNonlinearity`), and a residual block wrapper (`GEMBlock`) that combines convolution, normalisation, nonlinearity, and a skip connection. It is the workhorse layer of the pocket encoder: the protein binding pocket is treated as a curved 2D manifold sampled at Cα positions, and `GEMConv` performs anisotropic message passing that is exactly equivariant to changes of tangent-plane gauge.

The implementation realises Algorithm 1 of de Haan et al. (2020), specialised to the case where the surface is represented as a kNN graph over Cα atoms (rather than a triangulated mesh) — tangent frames and parallel-transport angles are precomputed via local PCA in `data/pocket_mesh.py` and consumed here as `angles` and `transporters` edge attributes.

## Mathematical Foundations

### Message passing on a gauged surface

Let $f \in \mathbb{R}^{V \times C_{\text{in}}}$ be the input feature tensor over $V$ vertices. At each centre vertex $p$ define its neighbours $\mathcal{N}(p)$ from the kNN graph, the edge angle $\theta_{pq}$ of the line $p \to q$ in the local tangent frame at $p$, and the parallel-transport angle $g_{q\to p}$ that maps the tangent frame at $q$ to the tangent frame at $p$ along the geodesic. The Gauge Equivariant Mesh Convolution updates the feature at $p$ as

$$\boxed{\;\; f'_p \;=\; K_{\text{self}} \, f_p \;+\; \sum_{q \in \mathcal{N}(p)} K_{\text{neigh}}(\theta_{pq}) \, \rho_{\text{in}}(g_{q\to p})\, f_q. \;\;}$$

Each kernel is a learned linear combination of the equivariant basis kernels described in `irreps.py`:

$$K_{\text{neigh}}(\theta) \;=\; \sum_b w^{\text{neigh}}_b \, B_b(\theta), \qquad K_{\text{self}} \;=\; \sum_b w^{\text{self}}_b \, B^{\text{self}}_b.$$

### Why parallel transport is required

A feature $f_q$ measured in the tangent frame at $q$ is not directly comparable with features at $p$ because the two frames may be rotated relative to one another along the surface. The parallel transport $\rho_{\text{in}}(g_{q\to p})$ pre-rotates $f_q$ into the frame at $p$ before the kernel acts. Without this rotation, summing neighbour features would mix vectors expressed in different coordinate systems and the network would not be gauge-equivariant.

Concretely, for an input feature block of irrep type $\rho_n$, the transport is the standard 2D rotation

$$f_q^{\text{transported}} \;=\; \begin{pmatrix} \cos(n\,g_{q\to p}) & -\sin(n\,g_{q\to p}) \\ \sin(n\,g_{q\to p}) & \phantom{-}\cos(n\,g_{q\to p}) \end{pmatrix} f_q.$$

Scalars ($n=0$) are not transported because $\rho_0(g) = 1$ — invariant quantities such as residue identity or charge do not need to be re-expressed in a different frame.

### Equivariance of the message

Combining the kernel constraint $K(\theta - g) = \rho_{\text{out}}(-g)\,K(\theta)\,\rho_{\text{in}}(g)$ with the way edge angles and transporters change under a gauge transformation $g_p$ at $p$ (namely $\theta_{pq} \mapsto \theta_{pq} - g_p$ and $g_{q\to p} \mapsto g_{q\to p} - g_p$), the message becomes

$$K_{\text{neigh}}(\theta_{pq} - g_p)\,\rho_{\text{in}}(g_{q\to p} - g_p)\,f_q \;=\; \rho_{\text{out}}(-g_p)\,K_{\text{neigh}}(\theta_{pq})\,\rho_{\text{in}}(g_{q\to p})\,f_q,$$

i.e. the entire message transforms by $\rho_{\text{out}}(-g_p)$, exactly as required for $f'_p$ to itself transform as $\rho_{\text{out}}$.

### Regular nonlinearities

Pointwise nonlinearities such as ReLU or SiLU are not equivariant on vector-valued ($n \ge 1$) blocks because applying a scalar map componentwise commutes with rotation only for scalars. The standard fix is a *norm nonlinearity*: keep the direction and apply the activation only to the magnitude. For a block $v \in \mathbb{R}^2$,

$$\mathrm{NormNonlin}(v) \;=\; \frac{\mathrm{softplus}(\lVert v \rVert)}{\lVert v \rVert} \, v.$$

This is exactly $SO(2)$-equivariant because rotation preserves $\lVert v \rVert$. Scalar blocks are handled with plain `silu`.

## Code Walk-through

### `scatter_add(src, index, dim, dim_size)` (lines 26–31)

A pure-PyTorch scatter-add that aggregates per-edge messages back to vertices. Replaces the dependency on `torch_scatter`. The function expands `index` to match the shape of `src` and calls `scatter_add_` along the vertex axis. Used inside `GEMConv.forward` to compute $\sum_{q \in \mathcal{N}(p)} \text{msg}_{pq}$.

### `apply_parallel_transport(features, transporters, ftype_in)` (lines 46–78)

Takes the per-edge stacked neighbour features `features` of shape `(E, C_in)`, the per-edge transporter angles `transporters` of shape `(E,)`, and the input feature type. It walks block by block:

- For a scalar block ($\rho_0$, lines 68–69) it copies the block unchanged.
- For a vector/tensor block ($\rho_n, n \ge 1$, lines 70–74) it reshapes to `(E, mult, 2)`, evaluates `rho_batch(order, transporters)` to get `R` of shape `(E, 2, 2)`, and applies the rotation via

```python
rotated = torch.einsum('emd,erd->emr', block, R)
```

The einsum contracts the trailing 2-dim of the feature with the trailing 2-dim of `R`, leaving `(E, mult, 2)`. The result is reshaped back into the flat per-edge feature vector.

### `GEMConv` (lines 85–125)

Single layer implementing the formula $f'_p = K_{\text{self}} f_p + \sum_{q} K_{\text{neigh}}(\theta_{pq})\rho_{\text{in}}(g_{q\to p}) f_q$. The constructor takes `ftype_in` and `ftype_out` and instantiates an `EquivariantKernelBasis`. The forward pass:

```python
K_self = self.kernel.eval_self()              # (dim_out, dim_in)
out = x @ K_self.T                            # (V, dim_out)  — self term

f_q = x[src]                                  # (E, dim_in)
f_q_transported = apply_parallel_transport(f_q, transporters, self.ftype_in)
K_neigh = self.kernel.eval_neigh(angles)      # (E, dim_out, dim_in)
msg = torch.bmm(K_neigh, f_q_transported.unsqueeze(-1)).squeeze(-1)
out = out + scatter_add(msg, tgt, dim=0, dim_size=V)
```

The `bmm` line computes per-edge $K_{\text{neigh}}(\theta_{pq}) \cdot f_q^{\text{transported}}$, giving messages of shape `(E, dim_out)`. The scatter then aggregates them at the target vertex. Edge convention: `edge_index[0] = src = q` (neighbour), `edge_index[1] = tgt = p` (centre).

### `RegularNonlinearity` (lines 132–164)

For each irrep block:

- $\rho_0$: applies `F.silu` directly (line 155).
- $\rho_n, n \ge 1$: reshapes to `(N, mult, 2)`, computes per-channel norms with floor `1e-8` for numerical safety (line 158), applies `softplus` to the norms (line 159), and rescales the original direction:

```python
out[:, offset:offset+C] = (block * (new_norms / norms)).reshape(-1, C)
```

Mathematically this is $v \mapsto \mathrm{softplus}(\lVert v \rVert)\,\hat v$, exactly equivariant under $SO(2)$.

### `GEMBlock` (lines 171–209)

A residual block:

$$h \;=\; \mathrm{RegularNonlinearity}\big(\mathrm{LayerNorm}\big(\mathrm{GEMConv}(f)\big)\big) \;+\; \mathrm{Skip}(f).$$

The skip is `nn.Identity()` when input and output dimensions match, otherwise a bias-free linear projection. During training the inner `conv → norm → nonlin` sub-computation is wrapped in `torch.utils.checkpoint.checkpoint` (lines 201–206) — this trades activation recomputation in the backward pass for reduced memory use, important because `eval_neigh(angles)` materialises an `(E, dim_out, dim_in)` tensor that can be large for dense pocket graphs.

A caveat about `LayerNorm` on irrep-typed features: applying a single learned scale and bias across all feature dimensions is not strictly gauge-equivariant on vector blocks (it can mix the two components of a $\rho_n$ block). In practice this is treated as a controlled approximation — the rotation-invariant part of the norm dominates, and exact equivariance is restored at the pocket-level global pool because the final block produces pure scalars.

## Biology / Chemistry Context

In the binding-pocket setting, each vertex is a Cα atom of a residue lining the pocket cavity (within `pocket_cutoff` = 10 Å of the ligand centroid). The kNN graph encodes which residues are spatially adjacent on the pocket surface. The edge angle $\theta_{pq}$ encodes the *direction* from residue $p$ to residue $q$ in the tangent plane at $p$ — equivalently, "is the neighbour to my north or to my east on the local surface chart?" The parallel transport $g_{q\to p}$ corrects for the fact that the tangent plane at $q$ is tilted relative to the tangent plane at $p$ because the pocket surface curves.

Anisotropic, gauge-equivariant convolution is a better inductive bias for binding pockets than ordinary GCN because pockets have *shape*: a residue may be on a concave bowl-shaped patch versus a convex ridge, the orientation of a polar-residue dipole relative to the pocket axis matters for hydrogen bonding, and so on. Plain GCNs are isotropic and would treat all neighbours identically regardless of where they sit relative to the centre residue's frame. The norm-nonlinearity preserves these directional features through the nonlinear layers.

## References

- de Haan, P., Weiler, M., Cohen, T., Welling, M. *Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs.* ICLR 2021. <https://arxiv.org/abs/2003.05425>. Algorithm 1 in the paper is what `GEMConv.forward` implements.
- Weiler, M., Hamprecht, F. A., Storath, M. *Learning Steerable Filters for Rotation Equivariant CNNs.* CVPR 2018. <https://arxiv.org/abs/1711.07289>. Origin of norm-nonlinearities.
- Cohen, T., Welling, M. *Group Equivariant Convolutional Networks.* ICML 2016. <https://arxiv.org/abs/1602.07576>. The foundational paper on equivariant CNNs.
- Bronstein, M. M. et al. *Geometric Deep Learning.* <https://arxiv.org/abs/2104.13478>. Chapter on gauges and bundles.
