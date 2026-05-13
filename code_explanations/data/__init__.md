# `data/__init__.py`

## Overview

This file is the package marker for the `data/` subpackage of TopoSurface-DTI. Although physically empty (zero non-whitespace lines), its presence is what allows the Python import system to treat the directory as a package, so that modules such as `data.molecule_graph`, `data.pocket_mesh`, `data.tda_features`, `data.dataset`, and `data.mesh_geometry` can be imported with the dotted module syntax used throughout the project (e.g. `from data.tda_features import compute_tda_features` in `dataset.py`).

In the broader TopoSurface-DTI architecture, the `data/` subpackage is the entire ingestion and featurization layer: it owns the conversion from raw biochemical inputs (PDB files, SDF files, SMILES strings, or synthetic point clouds) into the tensors that the model in `models/` consumes. Keeping this file empty — rather than re-exporting the public API — is an explicit design choice. Each downstream consumer (`dataset.py`, `run.py`, the unit-test snippets in `CLAUDE.md`) imports only the symbols it needs, which keeps import-time side effects small and avoids accidentally pulling RDKit/BioPython into processes that only need synthetic data.

## Mathematical Foundations

No mathematics is performed in this file. Its role is purely structural at the level of the Python module system. Conceptually, however, the `data/` package can be viewed as a functor

$$
\mathcal{F} \colon \mathrm{Biochem} \longrightarrow \mathrm{Tensor},
$$

mapping objects in the category of biochemical descriptions (a `.pdb` file paired with a `.sdf` ligand, or a SMILES string, or a triple of integers seeding a synthetic example) to objects in the category of PyTorch tensors arranged in the dictionary schema consumed by `TopoSurfaceDTI`. The submodules implement the components of this functor:

- `molecule_graph.py` sends a molecule $M$ to a graph $(X_M, P_M, E_M)$ with node features $X_M \in \mathbb{R}^{|V_M| \times 17}$, positions $P_M \in \mathbb{R}^{|V_M| \times 3}$, and edges $E_M \subset V_M \times V_M$;
- `pocket_mesh.py` sends a protein $P$ together with a ligand centroid $c_\text{lig}$ to a residue point cloud $(X_P, P_P, E_P)$ together with GEM-CNN geometric quantities $(\theta_{pq}, g_{q \to p})$;
- `tda_features.py` sends a point cloud $X \subset \mathbb{R}^3$ to a persistence-image vector $\Phi(X) \in \mathbb{R}^{800}$;
- `dataset.py` composes all of the above into a `torch.utils.data.Dataset`.

## Code Walk-through

The file has no lines of code. After `import data`, Python registers the package and runs the (empty) initialization. No symbols are introduced into the `data` namespace, so consumers must write `from data.<submodule> import <symbol>` explicitly. This is verified in `dataset.py` (lines 30–35), which uses exactly that import style.

## Biology / Chemistry Context

There is no biology or chemistry directly in this file, but as the entry point of the data subpackage it sits at the boundary between two worlds. On one side: the messy, heterogeneous, file-format-laden world of structural biology — PDB files curated by the wwPDB, SDF/MOL2 ligand files from PDBBind, SMILES strings from medicinal chemistry. On the other: the clean, fixed-shape, batchable world of PyTorch tensors that geometric deep learning needs. The decision to make this file empty (rather than re-export) means the package keeps RDKit and BioPython as truly optional dependencies — invocable only when a consumer explicitly imports from `molecule_graph` or `pocket_mesh`, and never imposed on users running on the synthetic data path.

## References

- Ramsundar, B., Eastman, P., Walters, P. & Pande, V. *Deep Learning for the Life Sciences.* O'Reilly, 2019. (Chapter on molecular featurization motivates the data-layer separation used here.)
- Python Software Foundation. *The Python Language Reference, section 5: The Import System.* https://docs.python.org/3/reference/import.html.
