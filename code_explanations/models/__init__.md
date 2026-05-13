# `models/__init__.py`

## Overview

This file marks the `models/` directory as a Python package, allowing the modules inside (`drug_encoder.py`, `pocket_encoder.py`, `gem_conv.py`, `irreps.py`, `fusion.py`, `toposurface_dti.py`) to be imported via dotted paths such as `from models.toposurface_dti import TopoSurfaceDTI`. The file itself is empty — it contains no symbols, no re-exports, and no initialisation logic. Its sole purpose is to satisfy Python's package-discovery mechanism.

In the broader architecture of TopoSurface-DTI, the `models/` package houses every neural-network component: the SchNet-style invariant drug encoder, the Gauge Equivariant Mesh CNN (GEM-CNN) pocket encoder together with its SO(2) irreducible-representation machinery, the cross-attention fusion head, and the top-level `TopoSurfaceDTI` orchestrator. Because nothing is exported from `__init__.py`, downstream code must import each symbol from its own submodule — for example, `from models.gem_conv import GEMBlock`, `from models.irreps import EquivariantKernelBasis`, etc. This choice keeps the public surface deliberately explicit and avoids circular-import pitfalls between the irrep utilities and the layers that consume them.

## Mathematical Foundations

No mathematical content lives in this file. The mathematical apparatus of the project — SO(2) irreducible representations $\rho_n$, the gauge-equivariance constraint

$$K(\theta - g) \;=\; \rho_{\text{out}}(-g)\, K(\theta)\, \rho_{\text{in}}(g),$$

the radial-basis-function (RBF) expansion $e_k(d) = \exp\!\big(-\gamma\,(d - \mu_k)^2\big)$ used inside SchNet, the scaled-dot-product attention $\mathrm{softmax}(QK^\top / \sqrt{d_k})\,V$, and the persistence-image vectorisation of $H_0$ and $H_1$ — is all implemented in the sibling modules. See `irreps.md`, `gem_conv.md`, `drug_encoder.md`, `fusion.md`, and `toposurface_dti.md` for the actual derivations and code.

## Code Walk-through

```python
# models/__init__.py
```

The file is empty. There are no classes, functions, or constants to walk through. Python interprets the presence of this file (regardless of contents) as evidence that `models/` should be treated as a regular package — as opposed to a namespace package or a plain directory.

A common alternative pattern would be to re-export the most-used symbols here so callers could write `from models import TopoSurfaceDTI` directly:

```python
# An alternative — NOT what this file does
from models.toposurface_dti import TopoSurfaceDTI
from models.drug_encoder import DrugEncoder
from models.pocket_encoder import PocketEncoder
```

The current design forgoes that convenience in favour of fully qualified imports throughout the codebase.

## Biology / Chemistry Context

Although this file contains no biological or chemical logic, it is the entry point for the package that does. Conceptually the `models/` namespace separates two distinct geometric inductive biases used in drug-target interaction prediction: an SE(3)-invariant message-passing scheme appropriate for small drug molecules (whose 3D conformation can be rigidly rotated without affecting binding affinity), and a gauge-equivariant scheme appropriate for the protein binding pocket, whose surface is a curved 2D manifold embedded in 3D space.

## References

- de Haan, P., Weiler, M., Cohen, T., Welling, M. *Gauge Equivariant Mesh CNNs: Anisotropic Convolutions on Geometric Graphs.* ICLR 2021. <https://arxiv.org/abs/2003.05425>
- Schütt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A., Müller, K.-R. *SchNet — A deep learning architecture for molecules and materials.* J. Chem. Phys. 148, 241722 (2018).
- Python Software Foundation. *The Python Tutorial — Packages.* <https://docs.python.org/3/tutorial/modules.html#packages>
