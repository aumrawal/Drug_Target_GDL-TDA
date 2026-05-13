# `data/dataset.py`

## Overview

This module defines `DTIDataset`, the single `torch.utils.data.Dataset` that supplies every batch consumed by training, validation, and visualisation in TopoSurface-DTI. It orchestrates every other module in the `data/` subpackage: it asks `molecule_graph.py` to build the drug graph, asks `pocket_mesh.py` to build the pocket point-cloud-plus-geometry, asks `tda_features.py` to compute persistence images for both, and packages everything into a single dictionary with consistent keys.

A second important responsibility is the *synthetic-versus-real* branching. When `use_synthetic=True` (the default used by `python run.py` with no flags), no PDB/SDF files are needed — the dataset generates pseudo-random graphs and point clouds reproducibly seeded by sample index, which makes it possible to test the whole pipeline end-to-end without installing BioPython or RDKit. When `use_synthetic=False`, the dataset reads PDBBind-formatted directories (`{pdbid}/{pdbid}_protein.pdb`, `{pdbid}/{pdbid}_ligand.sdf`) and JSON split files, with a robust fall-back to synthetic data on any per-sample loader failure so that training never crashes on a single broken PDB.

## Mathematical Foundations

### The dataset as a labelled mapping

Formally, the dataset is a finite indexed family

$$
\mathcal{D} = \{(\mathbf{D}_i,\, \mathbf{P}_i,\, y_i)\}_{i=1}^{N},
$$

where $\mathbf{D}_i$ is the drug graph object (an attributed 3D graph as in `molecule_graph.py`), $\mathbf{P}_i$ is the pocket graph object together with its precomputed GEM-CNN geometry (as in `pocket_mesh.py`), and $y_i \in \mathbb{R}$ is the binding affinity $y_i = pK_d = -\log_{10} K_d$ (with $K_d$ in molar units). For PDBBind, values typically range over $y \in [2, 12]$; the synthetic generator samples uniformly from $[4, 12]$.

Each sample, after feature extraction, is a record

$$
s_i = \big(X_i^D,\; P_i^D,\; E_i^D,\; X_i^P,\; E_i^P,\; \theta_i^P,\; g_i^P,\; \Phi^{H_0}_i \oplus \Phi^{H_1}_i\big|_{\text{drug}},\; \Phi^{H_0}_i \oplus \Phi^{H_1}_i\big|_{\text{pkt}},\; y_i\big),
$$

with shapes

$$
X_i^D \in \mathbb{R}^{n_D \times 17}, \;\; P_i^D \in \mathbb{R}^{n_D \times 3}, \;\; E_i^D \in \mathbb{Z}^{2 \times m_D}, \;\; X_i^P \in \mathbb{R}^{n_P \times 25}, \;\; E_i^P \in \mathbb{Z}^{2 \times m_P}, \;\; \theta_i^P, g_i^P \in \mathbb{R}^{m_P},
$$

and the two TDA feature vectors $\Phi^{H_0}_i \oplus \Phi^{H_1}_i \in \mathbb{R}^{800}$ where $\Phi^{H_k}$ is the persistence image of $H_k$ (`resolution² = 400`).

### TDA filtration radii

For each sample two filtrations are computed with different scale parameters:

$$
\mathrm{VR}_{\epsilon}(P_i^D)\;\text{for }\epsilon \in [0,\, \epsilon_\text{drug}], \qquad \mathrm{VR}_{\epsilon}(P_i^P)\;\text{for }\epsilon \in [0,\, 2\epsilon_\text{drug}],
$$

with default $\epsilon_\text{drug} = 8$ Å. The 2× factor reflects the relative spatial extents: drugs span $\lesssim 8$ Å while pockets span $\lesssim 20$ Å (`tda_max_edge * 2.0` on line 113).

### Reproducibility

When `use_synthetic=True`, the seed of each synthetic sample is the integer parsed from the sample id `'syn_<int>'`, so the random drug and pocket coordinates of sample $i$ are reproducible across runs and across the train/val/test splits. The labels $y_i$ are drawn from a base RNG with seed 42 (line 72) so the dataset itself is also deterministic.

## Code Walk-through

### Imports — lines 23–35

Pulls in three data submodules: `tda_features` (the TDA pipeline), `molecule_graph` (drug graph builders + constants), and `pocket_mesh` (pocket builders + constants). The constants `N_ATOM_FEAT`, `N_BOND_FEAT`, `N_RESIDUE_FEAT` are imported but not used directly in this file — they are re-exported so that `run.py` and `models/toposurface_dti.py` get them indirectly via the dataset namespace.

### `class DTIDataset(Dataset)` — lines 38–163

A standard PyTorch dataset that maps an integer index $i$ to the per-sample dictionary.

### `__init__(self, ...)` — lines 53–78

Stores configuration. Lines 71–76 build the *synthetic samples* list: $n$ records of the form `{'id': 'syn_i', 'affinity': uniform(4, 12)}`. Affinities are drawn from `np.random.default_rng(42)` so the entire synthetic dataset is deterministic. Lines 77–78 load the split JSON for the real-data path.

### `_load_split(self, split)` — lines 80–88

Reads `{data_dir}/{split}_split.json` which must be a list of `{'id': pdbid, 'affinity': pKd}` records. Errors with a helpful message if the file is missing.

### `__len__(self)` / `__getitem__(self, idx)` — lines 90–129

The Dataset protocol. `__getitem__` is the centrepiece:
- Lines 96–101: choose between synthetic and real data branches.
  - Synthetic path: parse `'syn_i' -> i` for the seed, then call `synthetic_drug_graph(n_atoms=20, seed=i)` and `synthetic_pocket_graph(n_residues=30, seed=i)`.
  - Real path: delegate to `_load_real(sample['id'])`.
- Lines 103–108: drug TDA. Pass `drug['pos'].numpy()` (the $(n_D, 3)$ atomic positions) to `compute_tda_features` with `max_edge_len = tda_max_edge` ($= 8$ Å by default). The resulting dict of $H_0$ and $H_1$ images is flattened to a single `(800,)` tensor by `tda_to_tensor`. This implements $\Phi^{H_0\oplus H_1}_\text{drug} \in \mathbb{R}^{2r^2}$.
- Lines 110–115: pocket TDA. Same call but with `max_edge_len = tda_max_edge * 2.0` ($= 16$ Å), reflecting the larger spatial extent of the residue cloud.
- Lines 117–129: assemble the final dictionary. Keys consumed downstream:
  - `drug_x`, `drug_pos`, `drug_edge`: graph for `DrugEncoder`.
  - `pocket_x`, `pocket_edge`, `pocket_angles`, `pocket_trans`: graph + GEM-CNN geometry for `PocketEncoder`.
  - `drug_tda`, `pocket_tda`: 800-dim TDA vectors for `FusionModule`.
  - `affinity`: scalar $y_i$ wrapped as a `float32` tensor.
  - `id`: bookkeeping string.

### `_load_real(self, pdb_id)` — lines 131–163

Real-data loader, conservatively wrapped in two try/except blocks so a corrupt PDB never crashes a training run.

- Lines 135–143 (drug): build path `{pdb_id}/{pdb_id}_ligand.sdf`, parse with RDKit's `SDMolSupplier`, take the first molecule, pass to `mol_to_graph`. On failure (file missing, bad SDF, RDKit not installed), print a warning and substitute a synthetic drug — this lets large datasets with a few broken entries still train cleanly.
- Lines 145–158 (pocket): paths to both the protein PDB and the ligand SDF. `extract_pocket_atoms` is called with `sdf_path` set to the ligand file *if it exists*; this is the mechanism described in `CLAUDE.md` for the standard PDBBind layout where the ligand lives in a separate SDF rather than embedded in the PDB. The pocket positions are then $k$-NN-graphed and run through `precompute_pocket_geometry` to attach normals, frames, angles, and parallel transporters.

The function returns the `(drug, pocket)` tuple consumed by `__getitem__`.

## Biology / Chemistry Context

PDBBind is a curated database derived from the Protein Data Bank that pairs every entry with an experimentally measured binding affinity, drawn from the published literature. Affinity is reported as $pK_d$, $pK_i$, or $\mathrm{pIC}_{50}$ — all on a $-\log_{10}$ scale, so larger means tighter binding. A drug with $pK_d = 9$ ($K_d = 1\,\mathrm{nM}$) is roughly 1000× tighter than one with $pK_d = 6$ ($K_d = 1\,\mu\mathrm{M}$). The numerical range $[2, 12]$ used by the synthetic generator brackets the realistic spread of measured PDBBind affinities.

The directory layout `{pdb_id}/{pdb_id}_protein.pdb` + `{pdb_id}/{pdb_id}_ligand.sdf` is the standard "refined set" / "general set" layout maintained by the PDBBind authors at the University of Michigan. Splitting the protein and ligand into separate files allows the ligand to be re-protonated, refined, or substituted independently of the protein scaffold; it also means the ligand has its own dedicated SDF rather than being embedded with the somewhat awkward `HETATM` records of the PDB format.

The choice to separate $\epsilon_\text{drug} = 8$ Å and $\epsilon_\text{pocket} = 16$ Å in the TDA pipeline reflects the chemical scales of the two objects: a drug is a small molecule with a handful of rings (typical persistence on 1–6 Å scales), while a binding pocket is a 10 Å-radius cavity whose topology unfolds at 4–16 Å scales. Using one filtration radius for both would either truncate pocket topology too early or oversample drug topology with mostly-empty space.

The robust fall-back to synthetic data on per-sample failure (lines 141–143, 159–161) is a pragmatic concession to real-world structural data: PDB files occasionally have missing atoms, mismatched chain identifiers, or non-standard residues that BioPython cannot parse. Rather than aborting training, the code logs a warning and replaces the broken sample with synthetic data; over a dataset of thousands the contamination is negligible.

## References

- Liu, Z., Su, M., Han, L., Liu, J., Yang, Q., Li, Y. & Wang, R. *Forging the basis for developing protein-ligand interaction scoring functions.* Acc. Chem. Res. 50(2), 302–309, 2017. (PDBBind v2017.)
- Wang, R., Fang, X., Lu, Y. & Wang, S. *The PDBbind database: collection of binding affinities for protein-ligand complexes with known three-dimensional structures.* J. Med. Chem. 47, 2977–2980, 2004.
- Berman, H. M. et al. *The Protein Data Bank.* Nucleic Acids Research 28(1), 235–242, 2000.
- Paszke, A. et al. *PyTorch: an imperative style, high-performance deep learning library.* NeurIPS, 2019. (`torch.utils.data.Dataset` API.)
- Edelsbrunner, H. & Harer, J. *Computational Topology: An Introduction.* AMS, 2010.
