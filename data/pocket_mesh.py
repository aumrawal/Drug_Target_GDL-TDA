# data/pocket_mesh.py
"""
Protein binding pocket extraction and GEM geometry precomputation.

Extracts a point cloud of Cα atoms within a radius of the ligand,
builds a k-NN spatial graph, then computes GEM-CNN geometry:
  - Tangent planes via local PCA (replaces mesh normals for point clouds)
  - Levi-Civita parallel transporters g_{q→p}
  - Neighbour angles θ_{pq}

This gives the pocket the same geometric interface as a triangulated
surface mesh, so GEMBlock layers apply directly.

Residue features (25 dims total):
    21 amino acid one-hot  (20 standard AAs + 'other')
    3  Cα position (x, y, z) in Angstroms
    1  distance to ligand centroid (Å)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, Tuple

from data.mesh_geometry import (
    log_map,
    build_reference_frames,
    compute_neighbour_angles,
    compute_parallel_transporters,
)

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL', 'other'
]
N_RESIDUE_FEAT = len(AMINO_ACIDS) + 4   # = 25


def extract_pocket_atoms(
    pdb_path: str,
    ligand_resname: str   = 'LIG',
    cutoff_angstrom: float = 10.0,
    sdf_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Cα positions and residue features for pocket residues.

    The ligand centroid is determined by (in order of preference):
      1. A residue named ``ligand_resname`` embedded in the PDB file.
      2. The SDF file at ``sdf_path`` (pass this for standard PDBBind layout
         where the ligand lives in a separate *_ligand.sdf file).

    Returns:
        pos  : (N, 3) Cα positions
        feat : (N, 25) residue feature vectors
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        raise ImportError("Install BioPython: pip install biopython")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)

    ligand_coords = []
    all_residues  = []
    for model in structure:
        for chain in model:
            for res in chain:
                name = res.get_resname().strip()
                if name == ligand_resname:
                    for atom in res:
                        ligand_coords.append(atom.get_vector().get_array())
                elif 'CA' in res:
                    all_residues.append(res)

    # Fallback: read ligand centroid from the SDF file (standard PDBBind layout)
    if not ligand_coords:
        if sdf_path is None:
            raise ValueError(
                f"Ligand '{ligand_resname}' not found in {pdb_path} "
                f"and no sdf_path provided. Pass sdf_path to use the "
                f"ligand SDF file for pocket centring."
            )
        try:
            from rdkit import Chem, RDLogger
        except ImportError:
            raise ImportError("Install RDKit: pip install rdkit-pypi")
        RDLogger.DisableLog('rdApp.warning')
        mol = Chem.SDMolSupplier(sdf_path, removeHs=True)[0]
        if mol is None:
            mol = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)[0]
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    pass
        RDLogger.EnableLog('rdApp.warning')
        if mol is None:
            raise ValueError(f"Could not parse ligand SDF: {sdf_path}")
        conf = mol.GetConformer()
        ligand_coords = [
            list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
        ]

    if not ligand_coords:
        raise ValueError(f"No ligand coordinates found for {pdb_path}")

    lig_center = np.mean(ligand_coords, axis=0)

    pos_list, feat_list = [], []
    for res in all_residues:
        ca = res['CA'].get_vector().get_array()
        dist = float(np.linalg.norm(ca - lig_center))
        if dist > cutoff_angstrom:
            continue

        name = res.get_resname().strip()
        aa = [0.0] * len(AMINO_ACIDS)
        aa[AMINO_ACIDS.index(name) if name in AMINO_ACIDS else -1] = 1.0

        pos_list.append(ca)
        feat_list.append(np.array(aa + list(ca) + [dist], dtype=np.float32))

    if len(pos_list) == 0:
        raise ValueError(f"No pocket residues found within {cutoff_angstrom}Å of ligand")

    return np.array(pos_list, np.float32), np.array(feat_list, np.float32)


def build_knn_graph(pos: np.ndarray, k: int = 10) -> np.ndarray:
    """Build k-NN directed edge index (both directions) from (N, 3) positions."""
    N = len(pos)
    k = min(k, N - 1)

    diff  = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)

    src, dst = [], []
    for i in range(N):
        for j in np.argsort(dists[i])[:k]:
            src.append(int(j))
            dst.append(i)

    return np.array([src, dst], dtype=np.int64)


def estimate_normals_pca(pos: Tensor, edge_index: Tensor) -> Tensor:
    """
    Estimate per-vertex normals using local PCA on kNN neighbors.

    The normal direction is the eigenvector of the local covariance matrix
    with the smallest eigenvalue (least variance = normal to local surface).
    """
    V = pos.shape[0]
    normals = torch.zeros(V, 3, dtype=pos.dtype, device=pos.device)
    tgt = edge_index[1]
    src = edge_index[0]

    for p in range(V):
        mask = (tgt == p)
        nbrs = pos[src[mask]]   # (k, 3)
        if nbrs.shape[0] < 2:
            normals[p] = torch.tensor([0., 0., 1.], device=pos.device)
            continue
        centered = nbrs - pos[p]
        cov = centered.T @ centered   # (3, 3)
        try:
            _, _, Vt = torch.linalg.svd(cov)
            normals[p] = Vt[-1]
        except Exception:
            normals[p] = torch.tensor([0., 0., 1.], device=pos.device)

    return F.normalize(normals, dim=-1)


def precompute_pocket_geometry(
    pos:        Tensor,   # (V, 3)
    edge_index: Tensor,   # (2, E)
) -> Dict:
    """
    Compute GEM geometry for a pocket point cloud.

    Replaces mesh_geometry.precompute_geometry for point-cloud input.
    Returns dict with: normals, e1, e2, angles, transporters.
    """
    normals = estimate_normals_pca(pos, edge_index)

    src, tgt = edge_index[0], edge_index[1]
    V = pos.shape[0]

    log_ij = log_map(pos[tgt], pos[src], normals[tgt])

    ref_log = torch.zeros(V, 3, dtype=pos.dtype, device=pos.device)
    for v in range(V):
        mask = (tgt == v)
        if mask.any():
            ref_log[v] = log_ij[mask][0]
        else:
            ref_log[v] = torch.tensor([1., 0., 0.], device=pos.device)

    e1, e2 = build_reference_frames(normals, ref_log)

    angles = compute_neighbour_angles(log_ij, e1[tgt], e2[tgt])

    transporters = compute_parallel_transporters(
        e1_src=e1[src], e2_src=e2[src], e1_tgt=e1[tgt],
        n_src=normals[src], n_tgt=normals[tgt],
    )

    return {
        'normals':      normals,
        'e1':           e1,
        'e2':           e2,
        'angles':       angles,
        'transporters': transporters,
    }


def synthetic_pocket_graph(n_residues: int = 30, seed: int = 0) -> Dict:
    """
    Synthetic protein binding pocket for testing.
    Residues arranged on a spherical shell to mimic a concave pocket.
    """
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0, np.pi, n_residues)
    phi   = rng.uniform(0, 2 * np.pi, n_residues)
    r     = rng.uniform(4, 8, n_residues)
    pos   = np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ], axis=-1).astype(np.float32)

    feat = rng.random((n_residues, N_RESIDUE_FEAT)).astype(np.float32)

    ei  = torch.from_numpy(build_knn_graph(pos, k=min(8, n_residues - 1)))
    pos_t = torch.from_numpy(pos)
    geo = precompute_pocket_geometry(pos_t, ei)

    return {'x': torch.from_numpy(feat), 'pos': pos_t, 'edge_index': ei, **geo}
