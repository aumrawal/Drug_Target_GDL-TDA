# data/dataset.py
"""
Drug-Target Interaction dataset.

Supports:
  - Synthetic data (for testing/prototyping — no external files needed)
  - PDBBind-formatted data (real binding affinities, requires PDB + SDF files)

Each sample returns a dict with all tensors needed for a forward pass of
TopoSurfaceDTI, including pre-computed TDA features (expensive to compute
on-the-fly at training time).

PDBBind directory layout expected:
    data_dir/
      {pdbid}/
        {pdbid}_protein.pdb
        {pdbid}_ligand.sdf
      train_split.json   [{"id": "1abc", "affinity": 7.5}, ...]
      val_split.json
      test_split.json
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional

from data.tda_features    import compute_tda_features, tda_to_tensor
from data.molecule_graph  import N_ATOM_FEAT, N_BOND_FEAT, synthetic_drug_graph
from data.pocket_mesh     import (
    N_RESIDUE_FEAT, build_knn_graph, precompute_pocket_geometry,
    synthetic_pocket_graph,
)


class DTIDataset(Dataset):
    """
    Drug-Target Interaction dataset.

    Args:
        data_dir      : root directory (ignored when use_synthetic=True)
        split         : 'train', 'val', or 'test'
        pocket_cutoff : Å radius around ligand centroid for pocket extraction
        knn_k         : k-NN connectivity for pocket graph
        tda_resolution: persistence image grid size (resolution² features per H_k)
        tda_max_edge  : max filtration radius for drug TDA (Å)
        use_synthetic : generate random data instead of loading files
        n_synthetic   : number of synthetic samples
    """

    def __init__(
        self,
        data_dir:       str   = 'data/PDBBind',
        split:          str   = 'train',
        pocket_cutoff:  float = 10.0,
        knn_k:          int   = 10,
        tda_resolution: int   = 20,
        tda_max_edge:   float = 8.0,
        use_synthetic:  bool  = True,
        n_synthetic:    int   = 100,
    ):
        self.data_dir       = data_dir
        self.pocket_cutoff  = pocket_cutoff
        self.knn_k          = knn_k
        self.tda_resolution = tda_resolution
        self.tda_max_edge   = tda_max_edge
        self.use_synthetic  = use_synthetic

        if use_synthetic:
            rng = np.random.default_rng(42)
            self.samples = [
                {'id': f'syn_{i}', 'affinity': float(rng.uniform(4, 12))}
                for i in range(n_synthetic)
            ]
        else:
            self.samples = self._load_split(split)

    def _load_split(self, split: str) -> List[Dict]:
        path = os.path.join(self.data_dir, f'{split}_split.json')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Split file not found: {path}\n"
                f"Format: [{{'id': 'pdbid', 'affinity': 7.5}}, ...]"
            )
        with open(path) as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if self.use_synthetic:
            seed = int(sample['id'].split('_')[1])
            drug   = synthetic_drug_graph(n_atoms=20, seed=seed)
            pocket = synthetic_pocket_graph(n_residues=30, seed=seed)
        else:
            drug, pocket = self._load_real(sample['id'])

        # TDA for drug (small molecule: tight radius, captures rings)
        drug_tda = tda_to_tensor(compute_tda_features(
            drug['pos'].numpy(),
            max_edge_len=self.tda_max_edge,
            resolution=self.tda_resolution,
        ))

        # TDA for pocket (protein: larger radius, captures loops and cavities)
        pocket_tda = tda_to_tensor(compute_tda_features(
            pocket['pos'].numpy(),
            max_edge_len=self.tda_max_edge * 2.0,
            resolution=self.tda_resolution,
        ))

        return {
            'drug_x':          drug['x'],
            'drug_pos':        drug['pos'],
            'drug_edge':       drug['edge_index'],
            'pocket_x':        pocket['x'],
            'pocket_edge':     pocket['edge_index'],
            'pocket_angles':   pocket['angles'],
            'pocket_trans':    pocket['transporters'],
            'drug_tda':        drug_tda,
            'pocket_tda':      pocket_tda,
            'affinity':        torch.tensor(sample['affinity'], dtype=torch.float32),
            'id':              sample['id'],
        }

    def _load_real(self, pdb_id: str):
        from data.molecule_graph  import mol_to_graph
        from data.pocket_mesh     import extract_pocket_atoms

        # ── Drug ──
        try:
            from rdkit import Chem
            sdf = os.path.join(self.data_dir, pdb_id, f'{pdb_id}_ligand.sdf')
            mol = Chem.SDMolSupplier(sdf, removeHs=True)[0]
            drug = mol_to_graph(mol)
        except Exception as e:
            print(f"[warn] Drug load failed for {pdb_id}: {e}. Using synthetic.")
            drug = synthetic_drug_graph()

        # ── Pocket ──
        try:
            pdb = os.path.join(self.data_dir, pdb_id, f'{pdb_id}_protein.pdb')
            sdf = os.path.join(self.data_dir, pdb_id, f'{pdb_id}_ligand.sdf')
            pos_np, feat_np = extract_pocket_atoms(
                pdb,
                cutoff_angstrom=self.pocket_cutoff,
                sdf_path=sdf if os.path.exists(sdf) else None,
            )
            pos_t  = torch.from_numpy(pos_np)
            feat_t = torch.from_numpy(feat_np)
            ei     = torch.from_numpy(build_knn_graph(pos_np, k=self.knn_k))
            geo    = precompute_pocket_geometry(pos_t, ei)
            pocket = {'x': feat_t, 'pos': pos_t, 'edge_index': ei, **geo}
        except Exception as e:
            print(f"[warn] Pocket load failed for {pdb_id}: {e}. Using synthetic.")
            pocket = synthetic_pocket_graph()

        return drug, pocket
