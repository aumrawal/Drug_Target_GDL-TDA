# data/molecule_graph.py
"""
Drug molecule representation as a 3D graph.

Converts a molecular structure into a PyTorch Geometric-compatible dict:
  - Nodes are heavy atoms with chemical feature vectors
  - Edges are covalent bonds with bond-type and distance features
  - 3D positions used for distance-based message passing

Atom features (17 dims total):
    10 atom-type one-hot  (C, N, O, S, F, Cl, Br, I, P, other)
    4  hybridization      (SP, SP2, SP3, other)
    1  formal charge      (integer, not normalized)
    1  is_aromatic        (0/1)
    1  is_in_ring         (0/1)

Bond features (5 dims total):
    4  bond-type one-hot  (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    1  Euclidean distance (Å)
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict

ATOM_TYPES       = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
HYBRID_TYPES     = ['SP', 'SP2', 'SP3', 'other']
BOND_TYPES       = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

N_ATOM_FEAT = len(ATOM_TYPES) + len(HYBRID_TYPES) + 3   # = 17
N_BOND_FEAT = len(BOND_TYPES) + 1                        # = 5


def _atom_features(atom) -> np.ndarray:
    sym = atom.GetSymbol()
    atype = [0.0] * len(ATOM_TYPES)
    atype[ATOM_TYPES.index(sym) if sym in ATOM_TYPES else -1] = 1.0

    hyb_str = str(atom.GetHybridization()).split('.')[-1]
    hyb = [0.0] * len(HYBRID_TYPES)
    hyb[HYBRID_TYPES.index(hyb_str) if hyb_str in HYBRID_TYPES else -1] = 1.0

    return np.array(
        atype + hyb + [
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ], dtype=np.float32
    )


def _bond_features(bond, pos_i: np.ndarray, pos_j: np.ndarray) -> np.ndarray:
    btype = str(bond.GetBondType()).split('.')[-1]
    bt = [0.0] * len(BOND_TYPES)
    if btype in BOND_TYPES:
        bt[BOND_TYPES.index(btype)] = 1.0
    dist = float(np.linalg.norm(pos_i - pos_j))
    return np.array(bt + [dist], dtype=np.float32)


def mol_to_graph(mol) -> Dict[str, Tensor]:
    """Convert an RDKit molecule to a graph dict. Generates a 3D conformer if none exists."""
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem
        mol = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        from rdkit import Chem
        mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    pos  = np.array([list(conf.GetAtomPosition(i))
                     for i in range(mol.GetNumAtoms())], dtype=np.float32)

    node_feats = [_atom_features(a) for a in mol.GetAtoms()]
    x = np.stack(node_feats, axis=0)

    src_list, dst_list, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = _bond_features(bond, pos[i], pos[j])
        src_list += [i, j]; dst_list += [j, i]
        edge_feats += [ef, ef]

    if edge_feats:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr  = np.stack(edge_feats, axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, N_BOND_FEAT), dtype=np.float32)

    return {
        'x':          torch.from_numpy(x),
        'pos':        torch.from_numpy(pos),
        'edge_index': torch.from_numpy(edge_index),
        'edge_attr':  torch.from_numpy(edge_attr),
    }


def smiles_to_graph(smiles: str) -> Dict[str, Tensor]:
    """Convert SMILES string to 3D graph (requires RDKit)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol_to_graph(mol)


def synthetic_drug_graph(n_atoms: int = 20, seed: int = 0) -> Dict[str, Tensor]:
    """
    Random drug-like graph for testing without RDKit.
    Atoms placed in a 10Å box; bonds drawn between atoms within 1.8Å.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-5, 5, (n_atoms, 3)).astype(np.float32)
    x   = rng.random((n_atoms, N_ATOM_FEAT)).astype(np.float32)
    x   = (x / (x.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    src, dst, ea = [], [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < 2.0:
                ef = np.zeros(N_BOND_FEAT, dtype=np.float32)
                ef[0] = 1.0   # SINGLE
                ef[-1] = d
                src += [i, j]; dst += [j, i]
                ea  += [ef, ef]

    # Guarantee at least one edge per atom by connecting nearest neighbor
    if len(src) < n_atoms:
        dists = np.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        for i in range(n_atoms):
            j = int(np.argmin(dists[i]))
            if i not in src or dst[src.index(i)] != j:
                d = float(dists[i, j])
                ef = np.zeros(N_BOND_FEAT, dtype=np.float32)
                ef[0] = 1.0; ef[-1] = d
                src += [i, j]; dst += [j, i]
                ea  += [ef, ef]

    return {
        'x':          torch.from_numpy(x),
        'pos':        torch.from_numpy(pos),
        'edge_index': torch.tensor([src, dst], dtype=torch.long),
        'edge_attr':  torch.from_numpy(np.stack(ea, axis=0)),
    }
