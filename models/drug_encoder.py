# models/drug_encoder.py
"""
SE(3)-invariant drug molecule encoder (SchNet-style).

Uses continuous-filter convolution where message functions depend only on
interatomic distances — invariant to rotations and translations of the
molecule. This is appropriate for drug molecules because:

  1. Binding affinity does not change under rigid-body rotation
  2. Bond lengths and angles are the chemically meaningful quantities
  3. Full SO(3)-equivariant encoding (TFN/SE3-Transformer) adds complexity
     without benefit for global scalar predictions like pKd

Architecture per interaction block:
    m_{pq} = filter_net(RBF(|r_p - r_q|)) ⊙ h_q   (edge messages)
    h'_p   = LayerNorm( h_p + out_proj( Σ_q m_{pq} ) )
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict


class RBFExpansion(nn.Module):
    """
    Radial Basis Function expansion of distances.
        e_k(r) = exp(-γ · (r - μ_k)²)
    with K centers μ_k evenly spaced in [0, cutoff].
    """

    def __init__(self, cutoff: float = 5.0, n_basis: int = 32):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_basis)
        self.register_buffer('centers', centers)
        self.gamma = float(n_basis) / (cutoff ** 2)

    def forward(self, r: Tensor) -> Tensor:
        """r: (E,) → (E, n_basis)"""
        return torch.exp(-self.gamma * (r.unsqueeze(-1) - self.centers) ** 2)


class InteractionBlock(nn.Module):
    """
    Single SchNet-like interaction block.

        W = filter_net(rbf(r_ij))         (E, hidden_dim)  filter weights
        msg_p = Σ_{j∈N(p)}  W_{pj} ⊙ h_j                 elementwise gating
        h'_p  = LayerNorm( h_p + out_proj(msg_p) )
    """

    def __init__(self, hidden_dim: int, n_basis: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h:          Tensor,   # (V, hidden_dim)
        edge_index: Tensor,   # (2, E)
        rbf:        Tensor,   # (E, n_basis)
    ) -> Tensor:
        src, tgt = edge_index[0], edge_index[1]
        V = h.shape[0]

        W   = self.filter_net(rbf)                              # (E, hidden_dim)
        msg = W * h[src]                                        # (E, hidden_dim)

        agg = torch.zeros(V, h.shape[-1], device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, tgt.unsqueeze(-1).expand_as(msg), msg)

        return self.norm(h + self.out_proj(agg))


class DrugEncoder(nn.Module):
    """
    Invariant drug molecule encoder.

    Args:
        in_features    : atom feature dimension (N_ATOM_FEAT = 17)
        hidden_dim     : width of all hidden layers
        n_interactions : number of interaction blocks (depth)
        cutoff         : maximum interaction radius (Å)
        n_basis        : number of RBF centers

    Returns dict with:
        per_atom : (V, hidden_dim)   per-atom contextual embeddings
        global   : (hidden_dim,)     global molecular embedding (mean pool)
    """

    def __init__(
        self,
        in_features:    int,
        hidden_dim:     int   = 64,
        n_interactions: int   = 3,
        cutoff:         float = 5.0,
        n_basis:        int   = 32,
    ):
        super().__init__()
        self.cutoff = cutoff

        self.embed = nn.Linear(in_features, hidden_dim)
        self.rbf   = RBFExpansion(cutoff=cutoff, n_basis=n_basis)

        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, n_basis)
            for _ in range(n_interactions)
        ])

    def forward(
        self,
        x:          Tensor,   # (V, in_features)
        pos:        Tensor,   # (V, 3)
        edge_index: Tensor,   # (2, E)
    ) -> Dict[str, Tensor]:
        src, tgt = edge_index[0], edge_index[1]

        # Interatomic distances (invariant)
        r = (pos[src] - pos[tgt]).norm(dim=-1)   # (E,)

        # Filter to cutoff radius
        mask = r < self.cutoff
        if mask.any():
            edge_index = edge_index[:, mask]
            r = r[mask]
            src, tgt = edge_index[0], edge_index[1]
        else:
            # Fallback: include all edges when none within cutoff
            pass

        rbf = self.rbf(r)
        h   = self.embed(x)

        for block in self.interactions:
            h = block(h, edge_index, rbf)

        return {
            'per_atom': h,
            'global':   h.mean(dim=0),
        }
