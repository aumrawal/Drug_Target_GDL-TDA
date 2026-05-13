# `data/pocket_mesh.py`

## Overview

This module is responsible for turning a protein structure plus a co-crystallised ligand into the input expected by the GEM-CNN-based pocket encoder of TopoSurface-DTI. The crucial design decision is that the protein is represented at the residue level via its $C_\alpha$ point cloud rather than as a triangulated molecular surface: this avoids the dependency on a heavy surface-meshing toolchain (MSMS, EDTSurf, PyMesh) while still providing every geometric quantity GEM-CNN needs, by reusing the primitives in `data/mesh_geometry.py` and replacing the area-weighted mesh normal with a local-PCA estimate.

The module contains three steps that always run together: (i) extracting pocket residues by a radial cut-off around the ligand centroid, (ii) building a $k$-nearest-neighbour graph over those residues, and (iii) computing GEM geometry (normals, tangent frames, edge angles $\theta_{pq}$, parallel transporters $g_{q\to p}$) so that gauge-equivariant convolutions in `models/gem_conv.py` can be applied directly. The result is a dictionary with the same keys as `mesh_geometry.precompute_geometry` plus the residue feature matrix `x` and the connectivity `edge_index`, giving a drop-in substitute that requires no changes anywhere else in the pipeline.

## Mathematical Foundations

### Pocket extraction by radial cut-off

Let $\mathcal{R}$ be the set of residues in a protein, each with a $C_\alpha$ position $x_r \in \mathbb{R}^3$, and let $L \subset \mathbb{R}^3$ be the set of ligand atom positions. The ligand centroid is

$$
c_\text{lig} = \frac{1}{|L|}\sum_{a \in L} a.
$$

The pocket is the set of residues whose $C_\alpha$ lies within a cut-off distance $d_\text{cut}$ (default 10 Å) of the centroid,

$$
\mathcal{R}_{\text{pkt}} = \big\{ r \in \mathcal{R} \;:\; \|x_r - c_\text{lig}\|_2 < d_\text{cut} \big\}.
$$

We let $V = |\mathcal{R}_\text{pkt}|$ and re-index residues as $\{1, \dots, V\}$.

### Residue featurisation

Each residue carries a 25-dimensional feature vector

$$
X[r] = \big[\,\underbrace{\mathbf{1}_{n_r = c}}_{c \in \mathcal{AA}}\,\big\|\,x_r\,\big\|\,\|x_r - c_\text{lig}\|_2\,\big] \in \mathbb{R}^{25},
$$

where $\mathcal{AA} = \{\text{ALA}, \text{ARG}, \dots, \text{VAL}, \text{other}\}$ is the alphabet of 20 standard amino acids plus an "other" bucket, $n_r$ is the three-letter residue name, $x_r \in \mathbb{R}^3$ is the $C_\alpha$ position, and the last scalar is the distance to the ligand centroid. The dimension is $21 + 3 + 1 = 25$.

### $k$-NN graph

The pocket connectivity is the directed $k$-nearest-neighbour graph

$$
E = \big\{\,(j, i) \;:\; j \in \mathcal{N}_k(i),\; i \in \{1,\dots,V\}\,\big\}, \qquad \mathcal{N}_k(i) = \underset{|S|=k,\; i\notin S}{\arg\min}\;\sum_{j\in S}\|x_j - x_i\|.
$$

Each $i$ gathers from $k$ neighbours; the resulting edge index has $V \cdot k$ entries (typically $30 \cdot 10 = 300$).

### Tangent-plane estimation via local PCA

At each vertex $p$ with neighbour set $\mathcal{N}_k(p) = \{q_1, \dots, q_k\}$, centre the neighbours,

$$
y_i = x_{q_i} - x_p,
$$

and form the local covariance

$$
C_p \;=\; \sum_{i=1}^{k} y_i y_i^\top \;\in\; \mathbb{R}^{3\times 3}.
$$

(The implementation uses the un-normalised sum rather than dividing by $k$; eigenvectors are insensitive to this overall scaling.) Spectral decomposition $C_p = U \Lambda U^\top$ with $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$ gives a local geometric interpretation: $u_1, u_2$ span the directions of maximum point-cloud variance (the local tangent plane) and $u_3$ — the eigenvector for the smallest eigenvalue $\lambda_3$ — points orthogonal to that plane. Hence

$$
n_p = u_3, \qquad \text{i.e.\;\;} n_p \in \arg\min_{\|n\|=1} n^\top C_p\, n.
$$

The PCA normal is the same quantity one obtains in the limit of dense sampling on a smooth surface, where $\lambda_3 / (\lambda_1 + \lambda_2 + \lambda_3) \to 0$. For noisy or sparse data, $\lambda_3$ may still be appreciable, which is acceptable here because GEM-CNN only requires *some* gauge — the equivariance machinery cancels any consistent choice.

Once the normal is in hand the rest of the geometric construction is identical to the meshed case: the log-map $\widetilde{\log}_p(q) = \|q-p\| \cdot \pi_{T_pM}(q-p) / \|\pi_{T_pM}(q-p)\|$ projects edges, $(e_1^p, e_2^p)$ are built from a reference edge and the normal, the polar angle is $\theta_{pq} = \operatorname{atan2}(\widetilde{\log}_p(q)\cdot e_2^p, \widetilde{\log}_p(q)\cdot e_1^p)$, and the parallel transporter $g_{q\to p}$ is computed by the Rodrigues-rotation procedure detailed in `mesh_geometry.md`.

## Code Walk-through

### Constants — lines 33–38

`AMINO_ACIDS` is the 21-entry vocabulary (20 standard + `'other'`). `N_RESIDUE_FEAT = 21 + 4 = 25` (21 amino-acid one-hot, 3 position, 1 distance) — note that the docstring on lines 14–17 lists 4 scalar features but the implementation actually has 4 (`pos[0], pos[1], pos[2], dist`), making the total 25 as advertised.

### `extract_pocket_atoms(pdb_path, ligand_resname, cutoff_angstrom, sdf_path) -> (pos, feat)` — lines 41–121

Implements the radial cut-off pocket extraction. Lines 60–65 use BioPython's `PDBParser` to read the structure. Lines 67–77 iterate over every residue: those whose three-letter name matches `ligand_resname` (default `'LIG'`) contribute their atoms to `ligand_coords`; everything else with a $C_\alpha$ is queued as a candidate pocket residue.

Lines 80–97 are the SDF fallback path used by PDBBind, where the ligand lives in a separate `*_ligand.sdf` file rather than being embedded in the PDB. RDKit reads the conformer and the same `ligand_coords` list is populated from its atom positions. Line 102 computes $c_\text{lig}$ as the centroid.

Lines 104–116 are the actual extraction loop: for each candidate residue, compute $\|x_r - c_\text{lig}\|$, skip if it exceeds the cut-off, otherwise build the 25-dim feature vector $[\mathbf{1}_{n_r = c} : c \in \mathcal{AA}] \,\|\, x_r \,\|\, \|x_r - c_\text{lig}\|$ and the position. Returns `pos` of shape $(V, 3)$ and `feat` of shape $(V, 25)$.

### `build_knn_graph(pos, k) -> (2, E) int64` — lines 124–139

Implements the $k$-NN graph. Lines 129–131 compute the full pairwise distance matrix and put $\infty$ on the diagonal so a vertex is never its own neighbour. Lines 133–138 sort each row's $k$ smallest entries and append the directed edge $(j, i)$ where `src=j` (the neighbour) and `dst=i` (the centre). The convention `src → dst` matches `mesh_geometry.py`'s notation `(2, E) long [src=q, tgt=p]`, so the same `precompute_geometry`-style routine works without modification.

### `estimate_normals_pca(pos, edge_index) -> (V, 3)` — lines 142–168

Implements the local-PCA normal estimation $n_p = u_3$ where $u_3$ is the eigenvector of $C_p = \sum_i y_i y_i^\top$ with smallest eigenvalue. Line 152 collects `src = edge_index[0]` and `tgt = edge_index[1]`. For each vertex $p$ (line 154), lines 155–156 gather its neighbours, line 160 centres them on $p$, and line 161 forms $C_p = Y^\top Y$ where $Y \in \mathbb{R}^{k\times 3}$ has the centred neighbours as rows. Lines 162–164 compute the SVD $C_p = U\Sigma V^\top$ and take `Vt[-1]` (the last row of $V^\top$ = the right singular vector for the smallest singular value, which coincides with the eigenvector of the smallest eigenvalue since $C_p$ is symmetric PSD). Lines 165–166 fall back to $\hat z$ on numerical failure. Line 168 normalises to unit length.

### `precompute_pocket_geometry(pos, edge_index) -> dict` — lines 171–211

Driver that orchestrates GEM-CNN geometry for the point cloud. Line 181 estimates normals by PCA. Line 186 computes $\widetilde{\log}_p(q)$ for every edge via `mesh_geometry.log_map`. Lines 188–194 pick a reference edge per vertex (the first incoming edge if any, else $\hat x$) to anchor the gauge. Line 196 builds $(e_1, e_2)$. Line 198 computes $\theta_{pq}$, lines 200–203 compute $g_{q\to p}$. The returned dict has keys `normals, e1, e2, angles, transporters` — exactly matching `mesh_geometry.precompute_geometry`.

### `synthetic_pocket_graph(n_residues, seed) -> dict` — lines 214–236

Synthetic generator for tests. Lines 221–228 sample residue positions on a spherical shell with radii in $[4, 8]$ Å so the layout is concave (mimicking a pocket wall). Line 230 makes random 25-dim residue features. Line 232 builds the $k$-NN edge index ($k=8$). Line 234 runs the full geometry precomputation. The output dictionary is the same shape used by the real loader, so unit tests and synthetic training paths share their downstream code.

## Biology / Chemistry Context

A binding pocket is a concave region on a protein's surface where a small-molecule drug docks via a combination of hydrogen bonds, hydrophobic contact, $\pi$-stacking, electrostatics, and (sometimes) covalent reaction. Pockets are typically lined by 10–30 amino-acid residues; coarse-graining the protein to one $C_\alpha$ per residue is the standard simplification when the *shape* of the pocket matters more than fine sidechain details — exactly the regime where GEM-CNN's surface-aware convolutions add value.

The 20-amino-acid one-hot vocabulary covers the residues encoded by the standard genetic code: nine non-polar (ALA, GLY, ILE, LEU, MET, PHE, PRO, TRP, VAL), six polar uncharged (ASN, CYS, GLN, SER, THR, TYR), three positively charged (ARG, HIS, LYS) and two negatively charged (ASP, GLU). The `'other'` slot catches modified residues such as MSE (selenomethionine, common in X-ray structures), SEP (phosphoserine), and various protonation-state variants.

The 10 Å radial cut-off is a community-standard pocket-extraction radius: it is large enough to contain the full first-shell binding interaction (typical Cα–ligand distances are 4–8 Å) without ballooning the residue count beyond what fits comfortably in GPU memory. The choice of $k=10$ for the $k$-NN graph approximates the natural sidechain-contact connectivity of a protein interior; experimental contact maps and structural ensembles typically see each residue in spatial contact with 8–12 neighbours.

BioPython (`Bio.PDB.PDBParser`) is the standard library for parsing the wwPDB legacy PDB file format. PDBBind, used here, is a curated subset of the PDB containing protein–ligand complexes with experimentally measured binding affinities ($K_d$, $K_i$, $\mathrm{IC}_{50}$) — the gold-standard benchmark for drug-target interaction modelling.

## References

- de Haan, P., Weiler, M., Cohen, T. & Welling, M. *Gauge equivariant mesh CNNs.* ICLR, 2021.
- Hoppe, H., DeRose, T., Duchamp, T., McDonald, J. & Stuetzle, W. *Surface reconstruction from unorganized points.* SIGGRAPH, 1992. (Original local-PCA normal estimation.)
- Mitra, N. J. & Nguyen, A. *Estimating surface normals in noisy point cloud data.* International Journal of Computational Geometry & Applications 14(4–5), 261–276, 2004.
- Wang, R., Fang, X., Lu, Y. & Wang, S. *The PDBbind database: collection of binding affinities for protein–ligand complexes with known three-dimensional structures.* J. Med. Chem. 47, 2977–2980, 2004.
- Cock, P. J. A. et al. *Biopython: freely available Python tools for computational molecular biology and bioinformatics.* Bioinformatics 25(11), 1422–1423, 2009.
