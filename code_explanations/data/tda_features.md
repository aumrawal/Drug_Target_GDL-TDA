# `data/tda_features.py`

## Overview

This module is the topological half of TopoSurface-DTI. Given a 3D point cloud — either the heavy atoms of a drug or the $C_\alpha$ atoms of a binding pocket — it computes a finite-dimensional, differentiable-friendly summary of its persistent homology in degrees 0 and 1: the persistence image. The intent is to give the fusion network a feature that the GEM-CNN/SchNet branches cannot easily reconstruct on their own — a global topological signature that captures the number of connected components, the number of independent loops, and the *scale* at which each appears.

The module has two implementations of the Vietoris-Rips persistent homology algorithm. The primary path (`_compute_tda_ripser`) delegates to Ripser, a C++ library that is 100–1000× faster than naive matrix reduction on point clouds of this size thanks to the *apparent pairs* optimisation, the *clearing lemma*, and a coboundary (dual) representation. The fallback (`_compute_tda_scratch`) implements the textbook standard reduction algorithm (lowest-1 pivot column-reduction over $\mathbb{F}_2$) in pure NumPy — slower by orders of magnitude but free of external dependencies. Both paths emerge as a list of $(birth, death)$ pairs which is then vectorised via the shared `persistence_image` routine into a `(resolution², )` float vector.

## Mathematical Foundations

### Vietoris-Rips filtration

Given a finite point set $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^3$ equipped with the Euclidean metric, the *Vietoris-Rips complex* at scale $\epsilon$ is the abstract simplicial complex

$$
\mathrm{VR}_\epsilon(X) \;=\; \big\{\, \sigma \subseteq X \;:\; \mathrm{diam}(\sigma) \le \epsilon\,\big\}, \qquad \mathrm{diam}(\sigma) = \max_{x,y \in \sigma}\|x-y\|.
$$

Equivalently, a simplex $\sigma$ is in $\mathrm{VR}_\epsilon(X)$ iff every pair of its vertices is at distance $\le \epsilon$. As $\epsilon$ grows, $\mathrm{VR}_\epsilon$ only acquires new simplices, so the family $\{\mathrm{VR}_\epsilon\}_{\epsilon\ge 0}$ is a *filtration*: a nested sequence of complexes

$$
\varnothing = \mathrm{VR}_0(X) \subseteq \mathrm{VR}_{\epsilon_1}(X) \subseteq \mathrm{VR}_{\epsilon_2}(X) \subseteq \cdots
$$

The implementation truncates this filtration at `max_edge_len`, the maximum scale considered.

### Chain complex and boundary operator

Fix the field $\mathbb{F}_2 = \{0,1\}$. The space of $k$-chains is the $\mathbb{F}_2$-vector space $C_k(\mathrm{VR}_\epsilon)$ with basis the $k$-dimensional simplices. The boundary operator $\partial_k : C_k \to C_{k-1}$ sends an oriented $k$-simplex $\sigma = [v_0, \dots, v_k]$ to the alternating sum of its $(k-1)$-faces — over $\mathbb{F}_2$ the signs disappear:

$$
\partial_k [v_0, \dots, v_k] \;=\; \sum_{i=0}^{k} [v_0, \dots, \widehat{v_i}, \dots, v_k] \pmod 2.
$$

Two foundational identities hold:

$$
\partial_{k-1} \circ \partial_k = 0, \qquad \text{i.e.\;} \mathrm{im}\,\partial_{k+1} \subseteq \ker\partial_k.
$$

This is exactly the condition that the sequence

$$
\cdots \to C_{k+1}(\mathrm{VR}_\epsilon) \xrightarrow{\partial_{k+1}} C_k(\mathrm{VR}_\epsilon) \xrightarrow{\partial_k} C_{k-1}(\mathrm{VR}_\epsilon) \to \cdots
$$

is a *chain complex*. The $k$-th *homology group* is the quotient

$$
H_k(\mathrm{VR}_\epsilon;\mathbb{F}_2) \;=\; \ker\partial_k \,/\, \mathrm{im}\,\partial_{k+1}.
$$

Elements of $\ker\partial_k$ are *cycles* (chains with no boundary); elements of $\mathrm{im}\,\partial_{k+1}$ are *boundaries* (cycles that already bound something). The $k$-th *Betti number* is the rank

$$
\beta_k(\epsilon) \;=\; \dim_{\mathbb{F}_2} H_k(\mathrm{VR}_\epsilon).
$$

Concretely, $\beta_0$ counts connected components, $\beta_1$ counts independent 1-dimensional loops, and $\beta_2$ counts enclosed cavities.

### Persistence diagrams

As $\epsilon$ grows from 0 to $\infty$, homology classes appear ("birth") and disappear ("death"). Persistent homology tracks these events: the $k$-th *persistence diagram* is the multiset

$$
\mathrm{Dgm}_k(X) \;=\; \{(b_i, d_i)\}_i \;\subset\; \{(b,d) : 0 \le b \le d \le \infty\}.
$$

Each point corresponds to a homology class born at scale $b_i$ that dies at scale $d_i$; its *persistence* is $\mathrm{pers}_i = d_i - b_i$. Long bars (large persistence) correspond to robust topological features; short bars are typically interpreted as noise. For $H_0$ specifically, all classes are born at $\epsilon = 0$ (every point starts as its own component), so the diagram lives on the line $b = 0$ and a single essential class persists to $\infty$ (the whole point cloud, once it becomes connected).

### Persistence images

The persistence diagram is set-valued and cannot be fed directly into a neural network. The vectorisation used here is the *persistence image* of Adams et al. (2017), defined on the *birth–persistence* coordinate system $(b, p)$ where $p = d - b$. For a diagram $\{(b_i, d_i)\}$ define the persistence surface

$$
\rho(z) \;=\; \sum_i w(d_i - b_i)\,\cdot\, \frac{1}{2\pi\sigma^2}\,\exp\!\left(-\frac{\|z - (b_i,\, d_i - b_i)\|^2}{2\sigma^2}\right),
$$

where:
- $z = (z_b, z_p) \in \mathbb{R}^2$ is the point in the birth-persistence plane being evaluated,
- $\sigma > 0$ is the kernel bandwidth (here `sigma = max_val/resolution * 2`),
- $w : \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ is a non-decreasing weight that vanishes on the diagonal — the implementation uses the simplest such choice, $w(p) = p$, which makes near-diagonal noise contribute negligibly while long-lived classes dominate.

The persistence image is the discretisation $\rho$ to a $r \times r$ grid (with $r$ = `resolution`, default 20) over $[0,\,\text{max\_val}]^2$, then row-flattened to length $r^2$. The full feature vector concatenates the $H_0$ and $H_1$ images, giving $2 r^2 = 800$ dimensions per point cloud.

The persistence image inherits the stability of persistence diagrams: small perturbations of $X$ (under Hausdorff distance) produce small perturbations of the diagram (under bottleneck distance, by the classical stability theorem), which produce small perturbations of $\rho$ (under $L^p$ norms, by direct kernel calculation). It is therefore differentiable in a Lipschitz sense with respect to the input point cloud.

### What Ripser does

Naive computation of persistent homology requires reducing the filtered boundary matrix $\partial$ via column operations (Edelsbrunner-Letscher-Zomorodian's "standard algorithm") — worst case $O(N^3)$ in the number of simplices, which is $\binom{n}{2} + \binom{n}{3} \approx n^3/6$ even for the truncated VR complex. Ripser (Bauer 2021) achieves orders-of-magnitude speedup via three tricks:

1. **Apparent pairs.** A pair $(\sigma, \tau)$ with $\sigma \prec \tau$ and $\tau$ being the youngest cofacet of $\sigma$ with $\sigma$ being the oldest facet of $\tau$ is automatically a birth–death pair in the reduction. These can be detected lazily without any column reduction, eliminating the *majority* of simplices from the matrix.
2. **Clearing lemma.** If column $j$ of $\partial_k$ reduces to zero, the corresponding column of $\partial_{k+1}$ can be skipped entirely — its reduction is determined by duality.
3. **Cohomology (coboundary) representation.** Ripser actually reduces the *coboundary* matrix $\delta = \partial^\top$ rather than $\partial$. By a duality argument the resulting persistence pairs are the same, but the coboundary matrix is much sparser (each simplex has few cofacets versus many faces beneath the cofacet-count threshold), shrinking memory and runtime.

Together these reduce the practical complexity to near-linear in the number of *persistence pairs*, which for typical molecular point clouds (20–200 atoms) is tiny.

## Code Walk-through

### `persistence_image(pairs, resolution, max_val, sigma) -> (r², ) float32` — lines 41–81

Implements the persistence image $\rho$. Lines 59–60 split each $(b, d)$ pair into a birth $b$ and a persistence $p = d - b$. Lines 62–65 choose defaults: `max_val` is the largest observed birth-or-death (so the grid bounds the data), `sigma = max_val/r \cdot 2` is the kernel bandwidth (roughly two pixels wide). Lines 67–68 build the 1D coordinate axes for the $r \times r$ grid. Lines 70–76 are the heart: for every $(b_i, p_i)$, exploit the separability of an isotropic 2D Gaussian
$$\exp\!\left(-\frac{(z_b - b_i)^2 + (z_p - p_i)^2}{2\sigma^2}\right) = \exp\!\left(-\frac{(z_b - b_i)^2}{2\sigma^2}\right) \exp\!\left(-\frac{(z_p - p_i)^2}{2\sigma^2}\right)$$
and accumulate $p_i \cdot (\text{outer product of 1D Gaussians})$ into the image. Diagonal points with $p < 10^{-8}$ are skipped (line 72) — they would contribute nothing anyway since the weight $w(p) = p$ vanishes. Lines 78–79 max-normalise so the output lives in $[0,1]$; line 81 flattens to a $(r^2,)$ float32.

### `_diagram_to_pairs(diagram, max_val) -> List[(b, d)]` — lines 84–95

Converts Ripser's `(N, 2)` numpy diagram to the project's standard `[(b, d), ...]` list. Drops NaN entries; clips infinite deaths to $1.5 \cdot \mathrm{max\_val}$ so essential bars register as long but finite persistence in the image.

### `_compute_tda_ripser(points, max_edge_len, resolution)` — lines 102–125

Wraps Ripser. Line 114 calls `ripser(points, maxdim=1, thresh=max_edge_len)`. The `thresh` parameter truncates the VR filtration at `max_edge_len`, matching the implementation's semantic of an upper scale cut-off. Lines 117–118 extract the $H_0$ and $H_1$ diagrams; lines 121–122 vectorise each to a `(r²,)` persistence image. Returns a dictionary with the two images plus the raw pair lists (for plotting / debugging).

### `_compute_tda_scratch(points, max_edge_len, resolution, max_points, seed)` — lines 132–236

From-scratch implementation. Lines 149–153 subsample if $n > $ `max_points` (default 200) since the $O(n^3)$ triangle enumeration becomes prohibitive. Lines 156–157 form the full pairwise distance matrix. Lines 160–165 build the sorted edge list $E_\epsilon = \{(i,j) : d_{ij} \le \epsilon\}$. Lines 175–188 enumerate triangles $\{i,j,k\}$ all three of whose edges are in $E_\epsilon$ and timestamp each at $\max(d_{ij}, d_{ik}, d_{jk})$ — the scale at which it first appears in the filtration.

Lines 190–206 are the *standard column-reduction algorithm* of Edelsbrunner-Letscher-Zomorodian. For each column $j$ of the boundary matrix $B$ (stored densely with $\mathbb{F}_2$ entries):
1. Find the *low* entry: the index $\mathrm{low}(j) = \max\{i : B[i,j] = 1\}$ (line 193).
2. If another column $j' < j$ already has the same low index, set $B[\cdot, j] \mathrel{{+}{=}} B[\cdot, j'] \pmod 2$ (lines 198–199) and retry — over $\mathbb{F}_2$, addition equals XOR.
3. When low becomes unique, record $\mathrm{lo}[\mathrm{low}(j)] = j$ (line 205); when the column zeros out, it represents a *birth*.

The matching of births and deaths follows: column $j$ becomes a death paired with the birth at row $\mathrm{low}(j)$, giving the diagram pair $(\text{birth time of row } \mathrm{low}(j),\; \text{death time of column } j)$.

Lines 208–214 compute $H_0$ by reducing the edge-boundary matrix $B_{01}$ (boundary $C_1 \to C_0$): for each edge $(i,k)$ at index $j$, set $B_{01}[i,j] = B_{01}[k,j] = 1$. Each reduced column gives a death at the edge's filtration time; the $H_0$ classes that never die before `max_edge_len` (i.e. unmatched rows) are recorded as $(0,\, 1.5 \cdot \mathrm{max\_d})$ — the essential connected component represented as a long bar.

Lines 217–229 compute $H_1$ analogously by reducing the triangle-boundary matrix $B_{12}$: each triangle $(a,b,c)$ contributes 1 in each of its three edges. A reduced column gives an $H_1$ birth at the largest-edge time of its low edge and an $H_1$ death at the triangle's enclosing time.

Lines 231–236 vectorise both diagrams into persistence images.

### `compute_tda_features(points, max_edge_len, resolution, max_points, seed) -> dict` — lines 243–273

Public entry point. Returns zero-images for $|X| < 2$ (line 264). Dispatches to Ripser if available (line 270), otherwise falls back to the scratch path. The subsampling parameter `max_points` is only used by the scratch path (Ripser handles ~200 points trivially).

### `tda_to_tensor(tda) -> Tensor` — lines 276–281

Concatenates the two `(r²,)` numpy arrays into a single `(2r², ) = (800,)` torch tensor — the format consumed by `FusionModule`.

## Biology / Chemistry Context

In a molecule, $\beta_0$ counts the number of disconnected chemical components — usually one for a connected ligand, but in a binding pocket it counts pseudo-clusters of $C_\alpha$ atoms at the relevant scale. As $\epsilon$ grows, components merge: small persistences correspond to nearby atoms quickly fusing, large persistences correspond to spatially isolated subclusters (e.g. distinct rings of an aromatic system or distinct lobes of a pocket).

$\beta_1$ counts independent loops. In a drug, these correspond directly to *rings* — benzene rings, pyridines, fused polycyclic aromatics — chemically and pharmacologically crucial features. A benzene ring appears at $\epsilon \approx 1.4$ Å (C–C bond length) and persists until $\epsilon$ exceeds the ring diameter (~2.8 Å). In a binding pocket, $H_1$ loops correspond to spatial arrangements of residues that enclose a "ring" of empty space — typical of helices, loops, and the rim of the pocket itself.

The choice of `max_edge_len = 8 Å` for drugs and `16 Å` for pockets is chemically motivated. Drugs are small (a typical 25-heavy-atom molecule fits in a 6–8 Å sphere); 8 Å covers their entire spatial extent and lets the filtration close up every ring. Pockets are larger — a 10 Å-radius pocket has a 20 Å diameter — so 16 Å allows the filtration to reach scales where the pocket's overall shape (and cavities) become visible. The 2× ratio is also the choice made in DTI papers that use TDA features (Cang–Wei, Nguyen et al.).

Compared to the GEM-CNN branch (which is local — each convolution sees only $k$-NN neighbours), the TDA features are *global*: a persistence pair born at scale $\epsilon_1$ and dying at $\epsilon_2$ summarises a topological feature on a length scale that no single convolution can probe. This is the key inductive bias that motivates the fusion architecture of TopoSurface-DTI.

The biological motivation for using persistent homology in drug discovery rests on the empirical observation (Cang & Wei 2017; Townsend et al. 2020) that ligand-pocket complementarity is governed not just by local chemistry but by matching *topological* features — protrusions, cavities, and ring systems — at multiple scales. Persistent homology provides exactly that multi-scale topological summary.

## References

- Edelsbrunner, H. & Harer, J. *Computational Topology: An Introduction.* AMS, 2010. (The textbook.)
- Edelsbrunner, H., Letscher, D. & Zomorodian, A. *Topological persistence and simplification.* Discrete & Computational Geometry 28, 511–533, 2002. (Original standard algorithm.)
- Bauer, U. *Ripser: efficient computation of Vietoris-Rips persistence barcodes.* Journal of Applied and Computational Topology 5, 391–423, 2021.
- Adams, H. et al. *Persistence images: a stable vector representation of persistent homology.* JMLR 18(8), 1–35, 2017.
- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. *Stability of persistence diagrams.* Discrete & Computational Geometry 37, 103–120, 2007.
- Cang, Z. & Wei, G.-W. *TopologyNet: topology-based deep convolutional and multi-task neural networks for biomolecular property predictions.* PLOS Computational Biology 13(7), 2017.
- Townsend, J., Micucci, C., Hymel, J., Maroulas, V. & Vogiatzis, K. *Representation of molecular structures with persistent homology for machine learning applications in chemistry.* Nature Communications 11, 3230, 2020.
