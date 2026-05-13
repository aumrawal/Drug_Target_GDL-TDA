# models/toposurface_dti.py
"""
TopoSurface-DTI: Topology-Aware Gauge Equivariant Drug-Target Interaction Model.

Combines two geometric deep learning ideas that are individually state-of-art
but have never been combined in a DTI context:

  1. GEM-CNN (gauge equivariant mesh convolution) on protein surface geometry
  2. Persistent Homology topological features for global shape characterization

Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Drug stream                                                         │
  │    DrugEncoder (SE(3)-invariant SchNet-like)                        │
  │      atoms × (17 feats + 3D pos) → per-atom (64) + global (64)    │
  │    TDA: Vietoris-Rips → H₀ + H₁ persistence images → (800,)       │
  ├──────────────────────────────────────────────────────────────────────│
  │  Protein pocket stream                                               │
  │    PocketEncoder (GEM gauge-equivariant convolution)               │
  │      Cα atoms × (25 feats + angles + transporters)                 │
  │      → per-residue (64) + global (64)                              │
  │    TDA: Vietoris-Rips → H₀ + H₁ persistence images → (800,)       │
  ├──────────────────────────────────────────────────────────────────────│
  │  FusionModule                                                        │
  │    Bidirectional cross-attention (drug ↔ pocket)                   │
  │    TDA feature compression                                          │
  │    MLP → pKd (binding affinity)                                    │
  └──────────────────────────────────────────────────────────────────────┘

Invariance/equivariance properties:
  - DrugEncoder: SE(3)-invariant (distances only)
  - PocketEncoder: gauge-equivariant → global pool is gauge-invariant
  - TDA features: permutation-invariant, SE(3)-invariant (distances only)
  - Overall: SE(3)-invariant prediction
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from models.drug_encoder   import DrugEncoder
from models.pocket_encoder import PocketEncoder
from models.fusion         import FusionModule


class TopoSurfaceDTI(nn.Module):
    """
    Full TopoSurface-DTI model.

    Args:
        drug_in_features   : drug atom feature dim  (17 default)
        pocket_in_features : residue feature dim    (25 default)
        hidden_dim         : encoder width           (64 default)
        tda_resolution     : persistence image size  (20 default → 800 per mol)
        n_drug_interactions: depth of drug SchNet    (3 default)
    """

    def __init__(
        self,
        drug_in_features:    int   = 17,
        pocket_in_features:  int   = 25,
        hidden_dim:          int   = 64,
        tda_resolution:      int   = 20,
        n_drug_interactions: int   = 3,
    ):
        super().__init__()
        self.tda_resolution = tda_resolution

        self.drug_encoder = DrugEncoder(
            in_features=drug_in_features,
            hidden_dim=hidden_dim,
            n_interactions=n_drug_interactions,
        )

        self.pocket_encoder = PocketEncoder(
            in_features=pocket_in_features,
            hidden_dim=hidden_dim,
        )

        # 2 images (H₀ + H₁) × resolution² per molecule × 2 molecules
        tda_dim = 2 * 2 * tda_resolution ** 2

        self.fusion = FusionModule(
            drug_dim=hidden_dim,
            pocket_dim=hidden_dim,
            tda_dim=tda_dim,
            hidden_dim=256,
        )

    def forward(
        self,
        drug_x:          Tensor,   # (V_d, drug_in_features)
        drug_pos:        Tensor,   # (V_d, 3)
        drug_edge:       Tensor,   # (2, E_d)
        pocket_x:        Tensor,   # (V_p, pocket_in_features)
        pocket_edge:     Tensor,   # (2, E_p)
        pocket_angles:   Tensor,   # (E_p,) θ_pq
        pocket_trans:    Tensor,   # (E_p,) g_{q→p}
        drug_tda:        Tensor,   # (2 * tda_resolution²,)  H₀ + H₁
        pocket_tda:      Tensor,   # (2 * tda_resolution²,)  H₀ + H₁
    ) -> Tensor:
        drug_out   = self.drug_encoder(drug_x, drug_pos, drug_edge)
        pocket_out = self.pocket_encoder(pocket_x, pocket_edge, pocket_angles, pocket_trans)

        tda = torch.cat([drug_tda, pocket_tda], dim=0)

        return self.fusion(
            drug_per_atom   = drug_out['per_atom'],
            drug_global     = drug_out['global'],
            pocket_per_res  = pocket_out['per_vertex'],
            pocket_global   = pocket_out['global'],
            tda_features    = tda,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> 'TopoSurfaceDTI':
        return cls(
            drug_in_features=cfg.get('drug_in_features', 17),
            pocket_in_features=cfg.get('pocket_in_features', 25),
            hidden_dim=cfg.get('hidden_dim', 64),
            tda_resolution=cfg.get('tda_resolution', 20),
            n_drug_interactions=cfg.get('n_drug_interactions', 3),
        )

    def count_parameters(self) -> Dict[str, int]:
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'drug_encoder':   n(self.drug_encoder),
            'pocket_encoder': n(self.pocket_encoder),
            'fusion':         n(self.fusion),
            'total':          n(self),
        }
