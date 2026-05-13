# `data/mesh_geometry.py`

## Overview

This module implements the core discrete differential geometry on which the Gauge Equivariant Mesh CNN (GEM-CNN) of de Haan et al. (2020) is built. Given a 2-manifold sampled either as a triangulated mesh $(V, F)$ or as a point cloud with a $k$-NN connectivity, it produces the geometric quantities that every gauge-equivariant convolution requires: per-vertex unit normals, an orthonormal reference frame $(e_1, e_2)$ in each tangent plane, a polar angle $\theta_{pq}$ for every directed edge $p \to q$, and a discrete Levi-Civita connection $g_{q \to p}$ (the parallel transporter that aligns the gauge at $q$ with the gauge at $p$).

In the TopoSurface-DTI pipeline, the mesh-based code path (`precompute_geometry`) is the canonical implementation used whenever a triangulated surface is available, while the point-cloud variant in `data/pocket_mesh.py` reuses the same primitives (`log_map`, `build_reference_frames`, `compute_neighbour_angles`, `compute_parallel_transporters`) and only swaps out the normal-estimation step for a local-PCA-based one. The geometric tensors returned by this module are then fed verbatim into `models/gem_conv.py`, where each neighbour kernel is evaluated at $\theta_{pq}$, the input feature at $q$ is transported by $g_{q \to p}$, and the result is aggregated at $p$.

## Mathematical Foundations

Let $M \subset \mathbb{R}^3$ be a smooth orientable 2-manifold approximated by a discrete mesh $(V, F)$ where $V \subset \mathbb{R}^3$ is the set of vertices and $F \subset V^3$ is the set of oriented triangles. At every point $p \in M$ the tangent space $T_pM$ is a 2-dimensional linear subspace of $\mathbb{R}^3$ orthogonal to the outward unit normal $n_p \in S^2$.

### Discrete normals

For a mesh face $(v_0, v_1, v_2) \in F$ the unsigned area normal is the cross product

$$
N_f = (v_1 - v_0) \times (v_2 - v_0), \qquad \|N_f\| = 2 \cdot \mathrm{area}(f).
$$

The discrete vertex normal at $v \in V$ is then the area-weighted average over incident faces, normalised to unit length:

$$
n_v = \frac{\sum_{f \ni v} N_f}{\left\|\sum_{f \ni v} N_f\right\|}.
$$

The area weighting (built in to using $N_f$ rather than $N_f / \|N_f\|$) is the standard MeshLab/CGAL choice — it minimises a one-ring Dirichlet energy and is consistent with the smooth Gauss map in the limit of mesh refinement.

### Tangent-plane logarithmic map

The Riemannian logarithm $\log_p \colon M \to T_pM$ is the inverse of the exponential map; it sends a nearby point $q$ to a tangent vector at $p$ whose magnitude equals the geodesic distance $d(p, q)$ and whose direction is the initial velocity of the geodesic from $p$ to $q$. The discrete approximation used here projects the chord $q - p$ onto $T_pM$ and rescales so that its norm equals $\|q - p\|$:

$$
\widetilde{\log}_p(q) \;=\; \|q - p\| \cdot \frac{(q-p) - \big((q-p) \cdot n_p\big)\, n_p}{\big\|(q-p) - \big((q-p) \cdot n_p\big)\, n_p\big\|}.
$$

The first-order projection $(q-p) - ((q-p)\cdot n_p) n_p$ kills the normal component; rescaling restores the original length so that $\|\widetilde{\log}_p(q)\| = \|q - p\| \approx d(p,q)$ when $q$ is close to $p$.

### Local reference frames (gauge choice)

Within $T_pM$, any orthonormal basis $(e_1^p, e_2^p)$ counts as a *gauge*. The construction adopted here picks an arbitrary reference neighbour $q_0$ and sets

$$
e_1^p = \frac{\widetilde{\log}_p(q_0)}{\|\widetilde{\log}_p(q_0)\|}, \qquad e_2^p = n_p \times e_1^p.
$$

This defines a right-handed orthonormal frame $(e_1^p, e_2^p, n_p)$ of $\mathbb{R}^3$.

### Polar angles of neighbours

In the frame at $p$, an edge $p \to q$ has tangent-plane representative $\widetilde{\log}_p(q)$. Its polar angle is

$$
\theta_{pq} = \operatorname{atan2}\!\left(\widetilde{\log}_p(q) \cdot e_2^p,\; \widetilde{\log}_p(q) \cdot e_1^p\right) \in (-\pi, \pi].
$$

A change of gauge $e_1^p \mapsto R(g) e_1^p$ (rotation by $g$) sends $\theta_{pq} \mapsto \theta_{pq} - g$, which is exactly the SO(2) action that the GEM convolution kernels must intertwine.

### Discrete Levi-Civita parallel transport

To aggregate features from $q$'s frame into $p$'s frame on a *curved* surface one must first parallel-transport along the edge. The discrete Levi-Civita connection of GEM-CNN (de Haan et al. 2020, Eq. 6) is constructed in two steps:

1. **Align the normals.** Rotate $T_qM$ rigidly into $T_pM$ by the unique rotation $R_\alpha$ taking $n_q$ to $n_p$ around the axis $a = (n_q \times n_p)/\|n_q \times n_p\|$, with angle $\alpha$ given by $\cos\alpha = n_q \cdot n_p$, $\sin\alpha = \|n_q \times n_p\|$. Rodrigues' formula gives
   $$R_\alpha v = v\cos\alpha + (a \times v)\sin\alpha + a(a\cdot v)(1-\cos\alpha).$$
2. **Measure residual in-plane rotation.** After step 1 the rotated frame $(R_\alpha e_1^q, R_\alpha e_2^q)$ lies in $T_pM$ but is generally not aligned with $(e_1^p, e_2^p)$. The transporter angle is the residual:
   $$g_{q \to p} = \operatorname{atan2}\!\big((R_\alpha e_2^q) \cdot e_1^p,\; (R_\alpha e_1^q) \cdot e_1^p\big).$$

Equivariance of the resulting convolution then takes the form $K(\theta - g) = \rho_\text{out}(-g) K(\theta) \rho_\text{in}(g)$ as discussed in `models/irreps.py`.

## Code Walk-through

### `compute_vertex_normals(vertices, faces) -> (V, 3)` — lines 22–38

Implements the area-weighted normal $n_v = \widehat{\sum_{f\ni v} N_f}$. Line 31 computes $(v_1 - v_0)\times(v_2-v_0) = N_f$ for every face. Lines 33–36 scatter-add each face's contribution to all three of its vertex slots, and line 38 normalises. Output shape `(V, 3)`.

### `log_map(p, q, normal_p) -> (E, 3)` — lines 41–59

Implements $\widetilde{\log}_p(q)$. Line 52 computes the chord $q - p$ with shape `(E, 3)`. Line 55 computes $(q-p)\cdot n_p$ broadcast along the spatial axis; line 56 subtracts the normal component. Line 59 rescales the in-plane vector to match the original chord length, returning shape `(E, 3)`. The `clamp(min=1e-8)` guards against degenerate edges with $p = q$.

### `build_reference_frames(normals, ref_vectors) -> (e1, e2)` — lines 62–75

Implements the gauge choice $e_1 = \widehat{v_\text{ref}}, \; e_2 = n \times e_1$. Returns two `(V, 3)` tensors forming a right-handed frame together with `normals`.

### `compute_neighbour_angles(log_pq, e1_p, e2_p) -> (E,)` — lines 78–89

Implements $\theta_{pq} = \operatorname{atan2}(\widetilde{\log}_p(q)\cdot e_2^p, \widetilde{\log}_p(q)\cdot e_1^p)$ by dotting the log-map vector against each basis vector (lines 87–88) and feeding the components to `torch.atan2` (line 89). Output shape `(E,)`.

### `_rotation_align_normals(n_src, n_tgt) -> (cos α, sin α, axis)` — lines 92–101

Returns the Rodrigues parameters $(\cos\alpha, \sin\alpha, a)$ for the rotation that takes $n_\text{src}$ to $n_\text{tgt}$. The axis is normalised to unit length (line 100), with a clamp protecting against the antipodal/parallel case where $\sin\alpha \approx 0$.

### `rodrigues_rotate(v, axis, cos_alpha, sin_alpha) -> Tensor` — lines 104–113

Applies the Rodrigues formula $Rv = v\cos\alpha + (a\times v)\sin\alpha + a(a\cdot v)(1-\cos\alpha)$ element-wise. Used to lift the source frame $(e_1^q, e_2^q)$ from $T_qM$ to $T_pM$.

### `compute_parallel_transporters(e1_src, e2_src, e1_tgt, n_src, n_tgt) -> (E,)` — lines 116–143

Implements $g_{q\to p}$. Line 131 retrieves Rodrigues parameters; lines 135–136 rotate $e_1^q$ and $e_2^q$; lines 138–139 handle the degenerate $n_q \approx n_p$ case (a flat patch — no rotation needed); lines 141–142 measure the residual in-plane angle by projecting the rotated $e_1^q$ onto $(e_1^p, e_2^p)$; line 143 returns $g_{q\to p} = \operatorname{atan2}(\sin g, \cos g)$.

### `precompute_geometry(vertices, faces, edge_index) -> dict` — lines 146–198

One-shot driver that orchestrates the entire pipeline. Line 158 computes vertex normals. Lines 160–164 evaluate $\widetilde{\log}_{p}(q)$ for every directed edge (note `src = q`, `tgt = p` per the docstring on line 149). Lines 166–174 build, for each vertex $v$, a "reference edge" pointing to its first-listed incident neighbour; this provides the $q_0$ vector used to fix the gauge. Line 176 builds $(e_1, e_2)$ at every vertex. Lines 178–182 compute all $\theta_{pq}$, and lines 184–190 compute all $g_{q\to p}$. The returned dictionary contains:

```
normals       : (V, 3)
e1, e2        : (V, 3)  orthonormal frame
angles        : (E,)    θ_pq
transporters  : (E,)    g_{q→p}
```

All four are exactly the inputs required by every `GEMConv` layer in `models/gem_conv.py`.

## Biology / Chemistry Context

GEM-CNN was originally developed for shape analysis on closed surfaces (FAUST human meshes, ShapeNet classifications). Applying it to protein binding pockets is a deliberate transfer of an idea from computer graphics into structural bioinformatics. The "surface" in question is the molecular surface of the protein near the bound ligand — historically computed as a solvent-excluded surface (Connolly 1983) or molecular skin surface, but here approximated by the $C_\alpha$ point cloud and its local PCA tangent planes (see `pocket_mesh.py`). Each tangent plane $T_pM$ can be interpreted as the local plane of the pocket wall at residue $p$; its normal $n_p$ points outward from the protein body into the solvent / pocket interior. Parallel transport then physically corresponds to consistently "looking around" the curved pocket without arbitrary reorientation as one moves between adjacent residues — which is exactly the inductive bias one wants for a model that must predict binding affinity from a chemically-meaningful description of the cavity. Atoms are individual chemical elements (C, N, O, etc.); residues are the amino-acid building blocks of proteins (around 20 standard varieties, each containing ~10–20 atoms with one $C_\alpha$ each), and operating at the residue level keeps the geometry tractable while preserving the coarse-grained shape of the pocket.

## References

- de Haan, P., Weiler, M., Cohen, T. & Welling, M. *Gauge equivariant mesh CNNs: Anisotropic convolutions on geometric graphs.* ICLR, 2021. (Equations 4–6.)
- Cohen, T., Weiler, M., Kicanaoglu, B. & Welling, M. *Gauge equivariant convolutional networks and the Icosahedral CNN.* ICML, 2019.
- Crane, K. *Discrete Differential Geometry: An Applied Introduction.* Carnegie Mellon University course notes, 2020. (Chapters on discrete normals, tangent vectors, and connections.)
- Rodrigues, O. *Des lois géométriques qui régissent les déplacements d'un système solide.* J. Math. Pures Appl. 5, 380–440, 1840.
- Connolly, M. L. *Solvent-accessible surfaces of proteins and nucleic acids.* Science 221, 709–713, 1983.
