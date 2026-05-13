# `data/molecule_graph.py`

## Overview

This module converts a small-molecule drug, supplied either as an RDKit `Mol` object with a 3D conformer, as a SMILES string (with on-the-fly conformer generation), or as a synthetic random graph, into the dictionary of tensors that the drug branch of TopoSurface-DTI consumes. The dictionary has four entries: `x` (node features), `pos` (3D coordinates), `edge_index` (covalent connectivity), and `edge_attr` (bond features). Each heavy atom becomes a node carrying a 17-dimensional feature vector describing its chemistry; each covalent bond becomes a pair of directed edges (so the resulting graph is symmetric) annotated with bond type and Euclidean distance.

In the data pipeline this module is the chemistry analogue of `pocket_mesh.py`: while the latter turns a protein into a coarse residue-level point cloud, `molecule_graph.py` turns a ligand into an atom-level graph. The output of `mol_to_graph` is consumed downstream by `DrugEncoder` (`models/drug_encoder.py`) for SE(3)-invariant message passing, and by `compute_tda_features` (`data/tda_features.py`) which uses only `pos` to build a Vietoris–Rips filtration on the atomic point cloud.

## Mathematical Foundations

A drug molecule is modelled as an attributed undirected 3D graph

$$
G_M = (V_M,\; E_M,\; X_M,\; P_M,\; A_M),
$$

where $V_M = \{1, \dots, n\}$ indexes the heavy atoms, $E_M \subseteq V_M \times V_M$ is the symmetric set of covalent bond edges, $X_M \in \mathbb{R}^{n \times 17}$ is the node-feature matrix, $P_M \in \mathbb{R}^{n \times 3}$ is the position matrix in Angstroms, and $A_M \in \mathbb{R}^{|E_M| \times 5}$ is the bond-feature matrix.

### Node featurisation

Each node feature $X_M[i] \in \mathbb{R}^{17}$ is the concatenation

$$
X_M[i] = \big[\,\underbrace{\mathbf{1}_{a_i = c}}_{c \in \mathcal{A}}\;\big\|\; \underbrace{\mathbf{1}_{h_i = c}}_{c \in \mathcal{H}}\;\big\|\; q_i \;\big\|\; \mathbf{1}_{\text{aromatic}}(i) \;\big\|\; \mathbf{1}_{\text{ring}}(i)\,\big]
$$

with vocabularies $\mathcal{A} = \{\text{C, N, O, S, F, Cl, Br, I, P, other}\}$ (size 10) and $\mathcal{H} = \{\text{SP, SP}^2, \text{SP}^3, \text{other}\}$ (size 4). Symbols: $a_i$ is the atomic symbol of atom $i$, $h_i$ its hybridisation, $q_i \in \mathbb{Z}$ its formal charge, and the last two entries are 0/1 flags. The unused/unknown bucket "other" makes the encoding total over $\mathcal{A} \cup \mathcal{H}$.

### Bond featurisation and distance graph

Each undirected bond $\{i,j\}$ is recorded as the two directed edges $(i,j)$ and $(j,i)$ — both carrying the same 5-dimensional feature

$$
A_M[(i,j)] = \big[\,\mathbf{1}_{b_{ij} = c}\;:\; c \in \mathcal{B}\,\big] \;\big\|\; \|x_i - x_j\|_2,
$$

where $\mathcal{B} = \{\text{SINGLE, DOUBLE, TRIPLE, AROMATIC}\}$ and $\|\cdot\|_2$ is the Euclidean norm.

For the synthetic generator the bond set is defined by a *distance cut-off graph* rather than chemistry:

$$
E_M^{\text{synth}} = \big\{\, (i,j) \in V_M \times V_M : i \neq j,\; \|x_i - x_j\|_2 < r_{\text{cut}} \,\big\}, \qquad r_\text{cut} = 1.8\,\text{Å}.
$$

This radius reflects the typical first-shell covalent neighbour distance (single C–C is 1.54 Å, double C=C is 1.34 Å, aromatic ring edges 1.39 Å, all < 1.8 Å). A guard ensures every atom has at least one nearest-neighbour edge so that no node is isolated, which would create rank-deficient adjacency matrices in downstream message passing.

### Permutation equivariance

Because the resulting tuple $(X_M, P_M, E_M, A_M)$ depends only on the labelled atom set and the bond multiset, any relabelling $\pi \in S_n$ acts equivariantly: $X_M \to PX_M$, $P_M \to PP_M$, $E_M \to PE_M P^\top$. Downstream models compose this with SE(3)-invariance (via interatomic distances) to give an overall $S_n \times \mathrm{SE}(3)$-equivariant pipeline.

## Code Walk-through

### Module constants — lines 27–32

`ATOM_TYPES`, `HYBRID_TYPES`, `BOND_TYPES` define the vocabularies $\mathcal{A}, \mathcal{H}, \mathcal{B}$. `N_ATOM_FEAT = 10 + 4 + 3 = 17` and `N_BOND_FEAT = 4 + 1 = 5` are the dimensions used by `configs/base.yaml` and verified by the unit test in `CLAUDE.md`.

### `_atom_features(atom) -> (17,) np.ndarray` — lines 35–50

Builds the 17-dim feature vector for one atom. Lines 37–38 produce the 10-dim atom-type one-hot $\mathbf{1}_{a_i = c}$; the trick of indexing `-1` for unknown elements routes them to the `'other'` bucket. Lines 40–42 do the same for the 4-dim hybridisation one-hot. Line 45–49 concatenates with $[q_i, \mathbf{1}_{\text{aromatic}}, \mathbf{1}_{\text{ring}}]$.

### `_bond_features(bond, pos_i, pos_j) -> (5,) np.ndarray` — lines 53–59

Computes $[\mathbf{1}_{b_{ij}=c}: c\in\mathcal{B}] \,\|\, \|x_i - x_j\|_2$. Note that when the bond type is not in $\mathcal{B}$ (rare — e.g. dative bonds) the one-hot is *all zeros*, not a fifth "other" slot.

### `mol_to_graph(mol) -> dict` — lines 62–90

Main entry point. Line 64 grabs the RDKit 3D conformer; line 65–66 collects positions into $P_M \in \mathbb{R}^{n\times 3}$. Lines 68–69 stack atom features into $X_M$ of shape `(n, 17)`. Lines 71–76 iterate over each bond, appending both directions $(i,j)$ and $(j,i)$ with the same feature so the resulting `edge_index` is symmetric; this gives $|E_M| = 2 \cdot |\text{bonds}|$. Lines 78–83 build the standard PyTorch Geometric `(2, E)` index tensor and `(E, 5)` attribute tensor. Lines 85–90 wrap all four arrays as tensors and return the dictionary.

### `smiles_to_graph(smiles) -> dict` — lines 93–104

Converts SMILES to 3D graph. Line 97 parses, line 100 explicitly adds hydrogens (required for correct geometry), line 101 generates a 3D conformer using ETKDGv3 (a knowledge-based distance-geometry embedder), line 102 relaxes to a local minimum with the MMFF94 force field, line 103 strips hydrogens back off (matching the heavy-atom convention used everywhere else), and line 104 hands off to `mol_to_graph`.

### `synthetic_drug_graph(n_atoms=20, seed=0) -> dict` — lines 107–146

Synthetic generator for testing without RDKit. Lines 112–115 sample $n$ atoms uniformly in $[-5,5]^3$ and build a random feature matrix $X_M$ normalised to a row-stochastic shape (so each "atom" looks like a soft simplex over feature dimensions). Lines 117–126 form the distance-cutoff edge set $E_M^{\text{synth}}$ with $r_\text{cut} = 2.0$ Å; each surviving pair is added in both directions with a single-bond feature plus distance. Lines 129–139 enforce connectivity by adding the nearest-neighbour edge for any atom that ended up isolated. Output schema is identical to `mol_to_graph`.

## Biology / Chemistry Context

Drugs targeted by TopoSurface-DTI are *small molecules* — typically 10 to 50 heavy atoms — distinct from biologics like antibodies. Restricting node features to *heavy atoms* (i.e. ignoring hydrogens at featurisation time, after their geometry has been used for conformer generation) is the universal convention in cheminformatics: hydrogens contribute little to binding affinity prediction beyond what is implicit in hybridisation and aromaticity, while almost doubling the node count.

The 10-element vocabulary $\{$C, N, O, S, F, Cl, Br, I, P, other$\}$ covers every element that ever appears in a clinically-approved small-molecule drug; the halogens (F, Cl, Br, I) are given dedicated slots because halogen bonds and lipophilic enrichment effects matter to binding affinity. Hybridisation (SP / SP² / SP³) discriminates the geometric character of carbons (linear / planar / tetrahedral) and is needed because two carbons with the same atomic number but different hybridisation behave very differently chemically. Aromaticity and ring-membership flags capture electron-delocalisation and ring-strain effects that distance alone cannot.

RDKit, used by `smiles_to_graph`, is the de facto open-source cheminformatics toolkit: it parses SMILES (a string syntax for molecular graphs invented by Weininger in 1988), assigns chemistry-aware bond types, computes hybridisation by Lewis-structure analysis, and provides the ETKDGv3 conformer generator and MMFF94 force field used to obtain a chemically realistic 3D pose. The 1.8 Å covalent radius cut-off is a chemistry rule-of-thumb consistent with all C–C, C–N, C–O single, double, triple and aromatic bond lengths. PDBBind ligands typically arrive as SDF files (an ASCII format with explicit atom positions and bond orders) so the synthetic-graph generator is only used in the unit tests and Kaggle pathway.

## References

- Weininger, D. *SMILES, a chemical language and information system.* J. Chem. Inf. Comput. Sci. 28(1), 31–36, 1988.
- Halgren, T. A. *Merck molecular force field.* J. Comput. Chem. 17(5–6), 490–519, 1996.
- Riniker, S. & Landrum, G. A. *Better informed distance geometry: using what we know to improve conformation generation.* J. Chem. Inf. Model. 55(12), 2562–2574, 2015. (ETKDGv3.)
- Landrum, G. *RDKit: Open-source cheminformatics.* https://www.rdkit.org.
- Gilmer, J. et al. *Neural message passing for quantum chemistry.* ICML, 2017. (Atom/bond featurisation conventions.)
