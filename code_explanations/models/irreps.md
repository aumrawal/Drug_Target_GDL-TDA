# `models/irreps.py`

## Overview

This module is the algebraic foundation of the Gauge Equivariant Mesh CNN (GEM-CNN) used to encode the protein binding pocket. It implements (i) the irreducible representations of the rotation group $SO(2)$ that act on tangent-plane features at each pocket vertex, (ii) the closed-form angular basis kernels that solve the gauge-equivariance constraint of de Haan et al. (2020), and (iii) a learnable `EquivariantKernelBasis` module that linearly combines those basis kernels with trainable scalar weights to produce the actual convolution kernel $K_{\text{neigh}}(\theta)$ and the self-interaction kernel $K_{\text{self}}$.

Everything else in the GEM stack — `GEMConv`, `GEMBlock`, `PocketEncoder` — depends on these primitives. The design follows Table 1 of de Haan et al.: rather than learning a generic angular function and projecting it onto the equivariant subspace, the module hard-codes the equivariant basis and learns only the scalar coefficients in front. This makes the network exactly (not approximately) gauge-equivariant.

## Mathematical Foundations

### The gauge group and its representations

On a curved surface — such as a protein pocket — there is no canonical choice of tangent-plane axes. A *gauge* is just a choice of orthonormal frame $(e_1^{(p)}, e_2^{(p)})$ at every vertex $p$. The group of frame redefinitions is the rotation group $SO(2)$. We write a gauge transformation at $p$ as a rotation by angle $g \in [0, 2\pi)$.

A *feature type* in this code is a list of pairs $\{(n_i, m_i)\}$ where $n_i \in \mathbb{Z}_{\ge 0}$ is an irrep order and $m_i$ is its multiplicity. The total feature dimension is

$$\dim\big(\{(n_i, m_i)\}\big) \;=\; \sum_i m_i \cdot \big(1 \text{ if } n_i = 0 \text{ else } 2\big).$$

This is computed by `feature_dim`. The irrep $\rho_n$ of $SO(2)$ acts on the feature block as

$$\rho_0(g) = [1], \qquad
\rho_n(g) \;=\; \begin{pmatrix} \cos(ng) & -\sin(ng) \\ \sin(ng) & \phantom{-}\cos(ng) \end{pmatrix} \quad (n \ge 1).$$

Scalars ($\rho_0$) are gauge-invariant — quantities like residue type, partial charge, or hydrophobicity. Vectors ($\rho_1$) transform as ordinary 2D tangent vectors — surface gradients, principal-direction hints. Tensors ($\rho_2$) capture anisotropic shape patterns such as curvature axes. The `rho` and `rho_batch` functions implement $\rho_n(g)$ for scalar and batched angles respectively.

### The gauge-equivariance constraint

A convolution kernel $K_{\text{neigh}}(\theta)\colon V_{n_{\text{in}}} \to V_{n_{\text{out}}}$ maps a feature on a neighbour to a contribution to the centre vertex, where $\theta$ is the angle of the edge in the local tangent frame. Under a gauge change $g$ at the centre, the edge angle shifts as $\theta \mapsto \theta - g$, the input transforms by $\rho_{n_{\text{in}}}(g)$, and the output transforms by $\rho_{n_{\text{out}}}(g)$. For the message to transform consistently the kernel must satisfy

$$K(\theta - g) \;=\; \rho_{\text{out}}(-g)\, K(\theta)\, \rho_{\text{in}}(g) \qquad \forall\, g, \theta \in [0, 2\pi).$$

This is a linear constraint on the matrix-valued function $K(\theta)$. Its solution space is finite-dimensional and admits a closed-form basis (Table 1 of de Haan et al. 2020), summarised below.

### The closed-form angular basis

Let $n_{\text{in}}$ and $n_{\text{out}}$ be the input and output irrep orders. The number of basis kernels $N_{\text{basis}}(n_{\text{in}}, n_{\text{out}})$ and their explicit forms are:

- $n_{\text{in}} = 0, n_{\text{out}} = 0$: one basis kernel, the constant $K(\theta) = 1$. (Returned in `basis_kernels_neigh` lines 115–116.)
- $n_{\text{in}} > 0, n_{\text{out}} = 0$: two basis kernels (mapping $\mathbb{R}^2 \to \mathbb{R}$):

$$K_0(\theta) = \big(\cos(n_{\text{in}}\theta),\; \sin(n_{\text{in}}\theta)\big), \qquad K_1(\theta) = \big(\sin(n_{\text{in}}\theta),\; -\cos(n_{\text{in}}\theta)\big).$$

- $n_{\text{in}} = 0, n_{\text{out}} > 0$: two basis kernels (mapping $\mathbb{R} \to \mathbb{R}^2$):

$$K_0(\theta) = \begin{pmatrix} \cos(n_{\text{out}}\theta) \\ \sin(n_{\text{out}}\theta) \end{pmatrix}, \qquad K_1(\theta) = \begin{pmatrix} \sin(n_{\text{out}}\theta) \\ -\cos(n_{\text{out}}\theta) \end{pmatrix}.$$

- $n_{\text{in}} > 0, n_{\text{out}} > 0$: four basis kernels. Define $p = n_{\text{out}} + n_{\text{in}}$, $q = |n_{\text{out}} - n_{\text{in}}|$, $\sigma = \mathrm{sign}(n_{\text{out}} - n_{\text{in}})$:

$$K_0 = \begin{pmatrix} \cos q\theta & -\sigma\sin q\theta \\ \sigma\sin q\theta & \cos q\theta \end{pmatrix}, \quad
K_1 = \begin{pmatrix} \sin q\theta & \sigma\cos q\theta \\ -\sigma\cos q\theta & \sin q\theta \end{pmatrix},$$

$$K_2 = \begin{pmatrix} \cos p\theta & \sin p\theta \\ \sin p\theta & -\cos p\theta \end{pmatrix}, \quad
K_3 = \begin{pmatrix} -\sin p\theta & \cos p\theta \\ \cos p\theta & \sin p\theta \end{pmatrix}.$$

You can verify these by direct substitution into $K(\theta - g) = \rho_{n_{\text{out}}}(-g)\,K(\theta)\,\rho_{n_{\text{in}}}(g)$ and using the angle-addition identities.

### Self-interaction kernels

The self-kernel $K_{\text{self}}$ is angle-independent (it acts on the centre vertex itself, not on an edge). The constraint reduces to $K_{\text{self}} = \rho_{\text{out}}(-g)\,K_{\text{self}}\,\rho_{\text{in}}(g)$ for all $g$, which by Schur's lemma forces $K_{\text{self}} = 0$ when $n_{\text{in}} \ne n_{\text{out}}$. When $n_{\text{in}} = n_{\text{out}} = 0$ the basis is the scalar $1$. When $n_{\text{in}} = n_{\text{out}} \ge 1$ the basis is two-dimensional and spanned by the identity $I_2$ and the $90^\circ$ rotation matrix $J = \begin{pmatrix}0&1\\-1&0\end{pmatrix}$.

### The learnable parameterisation

The full kernel between input feature type $\{(n_{\text{in},i}, m_{\text{in},i})\}$ and output type $\{(n_{\text{out},j}, m_{\text{out},j})\}$ is block-diagonal across irrep blocks and dense across multiplicities:

$$K_{\text{neigh}}(\theta)_{j,i} \;=\; \sum_{b=1}^{N_{\text{basis}}} w^{\text{neigh}}_{j,i,b}\, B_b(\theta), \qquad K_{\text{self}}{}_{j,i} \;=\; \sum_{b=1}^{N_{\text{basis,self}}} w^{\text{self}}_{j,i,b}\, B^{\text{self}}_b.$$

The trainable parameters are the scalar weights $w^{\text{neigh}}, w^{\text{self}}$.

## Code Walk-through

### Type aliases (lines 28–42)

```python
FeatureType = List[Tuple[int, int]]
```

A list of `(order, multiplicity)` pairs encoding a direct sum of irreps. For example, `[(0,16), (1,16), (2,8)]` denotes $16\rho_0 \oplus 16\rho_1 \oplus 8\rho_2$, with total dimension $16 \cdot 1 + 16 \cdot 2 + 8 \cdot 2 = 64$. `feature_dim` computes this sum. `scalar_type(n_channels)` is the convenience builder `[(0, n_channels)]` for a pure-scalar feature type.

### `rho(order, angle)` and `rho_batch(order, angles)` (lines 54–96)

These return the rotation matrix $\rho_n(g)$ either for a single angle or for a batch of edge angles. For `order = 0` the function returns the constant $1\times1$ matrix `[[1]]`; for `order ≥ 1` it returns the standard 2D rotation by $ng$. The batched version is used inside `apply_parallel_transport` in `gem_conv.py` to rotate every neighbour's feature block by the appropriate edge transporter angle.

### `basis_kernels_neigh(n_in, n_out, angles)` (lines 103–149)

Returns the tensor of all basis kernel evaluations along the edge batch:

| `n_in, n_out` | Output shape | Number of bases |
|---|---|---|
| 0, 0 | `(E, 1, 1, 1)` | 1 |
| $n_{\text{in}} > 0$, 0 | `(E, 1, 2, 2)` | 2 |
| 0, $n_{\text{out}} > 0$ | `(E, 2, 1, 2)` | 2 |
| $n_{\text{in}} > 0$, $n_{\text{out}} > 0$ | `(E, 2, 2, 4)` | 4 |

Lines 144–147 build the four-basis case using the local helper `mat(a,b,c,d)` that stacks two 2-row tensors into a `(E, 2, 2)` matrix. The `sign` variable encodes $\sigma = \mathrm{sign}(n_{\text{out}} - n_{\text{in}})$, defaulting to $+1$ when $n_{\text{out}} \ge n_{\text{in}}$.

### `basis_kernels_self(n_in, n_out, dim)` (lines 152–170)

Returns `None` when $n_{\text{in}} \ne n_{\text{out}}$ (the Schur constraint kills the block). Otherwise returns the constant 1-basis when both orders are 0, and the two-basis $\{I_2, J\}$ when both orders are $\ge 1$, with $J = \begin{pmatrix}0&1\\-1&0\end{pmatrix}$ (line 169).

### Counting helpers (lines 177–197)

`count_parameters_neigh` and `count_parameters_self` walk through every (in-block, out-block) pair, multiply by the number of basis kernels in that block, and accumulate $m_{\text{in}} \cdot m_{\text{out}}$ trainable scalars per basis. `_n_basis_neigh` is the lookup of the basis-count table above.

### `EquivariantKernelBasis` (lines 200–317)

The actual `nn.Module` exposing learnable parameters.

- `__init__` (lines 208–226) instantiates two `nn.Parameter` tensors `w_neigh` and `w_self` of sizes `count_parameters_neigh(...)` and `count_parameters_self(...)`. They are initialised from $\mathcal{N}(0, 2/(d_{\text{in}}+d_{\text{out}}))$ — a Kaiming-style scaling adapted to the equivariant block structure. It also precomputes `_neigh_layout` and `_self_layout`, lists of `(n_in, m_in, n_out, m_out, n_basis, w_start)` tuples that record where each block's weights live inside the flat parameter vectors.

- `eval_neigh(angles)` (lines 249–279) returns the kernel $K_{\text{neigh}}(\theta) \in \mathbb{R}^{E \times d_{\text{out}} \times d_{\text{in}}}$. For each pair of irrep blocks $(n_{\text{in}}, n_{\text{out}})$ it loads the corresponding slice of `w_neigh`, reshapes to $(m_{\text{out}}, m_{\text{in}}, N_{\text{basis}})$, evaluates `basis_kernels_neigh` to get the basis tensor, and contracts via

```python
k_block = torch.einsum('ijb,exyb->eixjy', w_block, basis)
```

This is the elementwise contraction $K_{\text{block}}[e, i, x, j, y] = \sum_b w[i,j,b]\,B[e,x,y,b]$, where $i$ indexes the output multiplicity, $j$ the input multiplicity, $x$ the output irrep dimension, $y$ the input irrep dimension, $b$ the basis index, and $e$ the edge. The result is reshaped to the kernel block of shape `(E, m_out * d_out, m_in * d_in)` and pasted into the global kernel at offsets `out_offset` and `in_offset`.

- `eval_self()` (lines 281–317) is the angle-independent analogue. The contraction is `torch.einsum('ijb,xyb->ixjy', w_block, basis)` — the same shape as above without the edge axis. Off-diagonal blocks ($n_{\text{in}} \ne n_{\text{out}}$) are skipped because of Schur.

## Biology / Chemistry Context

The protein binding pocket is a 2D surface (the solvent-accessible interface) bent inside 3D space. At every residue location, choosing tangent-plane axes — for example "toward the next residue in sequence" and "perpendicular to that" — is arbitrary, but a CNN that treats those axes as if they were absolute coordinates will produce different predictions for chemically identical pockets that happen to have been frame-indexed differently. Gauge equivariance forces every layer to behave covariantly under such frame redefinitions, which is the right physical symmetry for binding affinity: pKd cannot depend on how a human labelled the axes.

The choice of feature types per layer is biologically meaningful. Pure-scalar features ($\rho_0$) carry quantities such as amino-acid identity, electrostatic charge, hydrophobicity index — all manifestly direction-independent. Vector features ($\rho_1$) carry direction-sensitive signals such as the local surface gradient, the principal curvature direction, or anisotropic packing hints toward a neighbour ligand atom. The progression of feature types used in `PocketEncoder` — $24\rho_0 \to 8\rho_0\!\oplus\!8\rho_1 \to 16\rho_0\!\oplus\!16\rho_1 \to 16\rho_0\!\oplus\!16\rho_1 \to 32\rho_0$ — reflects the strategy of first lifting scalar residue features into vector channels (so the network can reason about directional surface patterns) and finally projecting back to scalars for a gauge-invariant readout.

## References

- de Haan, P., Weiler, M., Cohen, T., Welling, M. *Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs.* ICLR 2021. <https://arxiv.org/abs/2003.05425>. Table 1 of this paper enumerates the basis kernels implemented here.
- Cohen, T., Weiler, M., Kicanaoglu, B., Welling, M. *Gauge Equivariant Convolutional Networks and the Icosahedral CNN.* ICML 2019. <https://arxiv.org/abs/1902.04615>. The general theory of gauge-equivariant CNNs on manifolds.
- Weiler, M., Geiger, M., Welling, M., Boomsma, W., Cohen, T. *3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data.* NeurIPS 2018. <https://arxiv.org/abs/1807.02547>. Background on $SO(n)$ irreducible representations and steerable kernels.
- Bronstein, M. M., Bruna, J., Cohen, T., Veličković, P. *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* (2021) <https://arxiv.org/abs/2104.13478>. The "5G" book with a chapter on gauge equivariance.
