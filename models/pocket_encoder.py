# models/pocket_encoder.py
"""
Gauge Equivariant pocket encoder (GEM-based).

Applies GEM convolution layers to the protein binding pocket, treating
each Cα atom as a vertex on a curved surface. GEM is the right choice here
because:

  1. The protein surface IS a curved 2D manifold embedded in 3D space
  2. Binding pocket shape (concave/convex geometry) requires anisotropic
     convolution to distinguish radial vs. tangential patterns
  3. Gauge equivariance ensures the pocket encoding is independent of the
     arbitrary local reference frame at each residue — essential for
     rotation-invariant binding affinity prediction

Feature type progression:
    Input    :  scalar residue features
    Embed    :  24ρ₀  (all scalars — embedding layer breaks no equivariance)
    Block 1  :  8ρ₀ ⊕ 8ρ₁   (introduce vector channels)
    Block 2  :  16ρ₀ ⊕ 16ρ₁
    Block 3  :  16ρ₀ ⊕ 16ρ₁
    Block 4  :  32ρ₀          (project back to scalars for invariant readout)
    Pool     :  mean pool → (32,) gauge-invariant pocket fingerprint
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

from models.gem_conv import GEMBlock
from models.irreps import FeatureType, feature_dim


class PocketEncoder(nn.Module):
    """
    GEM-based binding pocket encoder.

    Args:
        in_features : residue feature dimension (N_RESIDUE_FEAT = 25)
        hidden_dim  : output embedding dimension

    Returns dict with:
        per_vertex : (V, hidden_dim)  per-residue equivariant embeddings
        global     : (hidden_dim,)    global pocket embedding (gauge-invariant)
    """

    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()

        # Feature type definitions
        ftype_embed = [(0, 24)]             # 24ρ₀  = 24 dims  (pure scalar)
        ftype_s     = [(0, 8),  (1, 8)]    # 8ρ₀ ⊕ 8ρ₁  = 24 dims
        ftype_m     = [(0, 16), (1, 16)]   # 16ρ₀ ⊕ 16ρ₁ = 48 dims
        ftype_out   = [(0, 32)]            # 32ρ₀          = 32 dims (invariant)

        # Scalar embedding of raw residue features
        self.input_embed = nn.Linear(in_features, feature_dim(ftype_embed))

        # GEM blocks
        self.blocks = nn.ModuleList([
            GEMBlock(ftype_embed, ftype_s),    # scalars → mixed
            GEMBlock(ftype_s,     ftype_m),    # grow width
            GEMBlock(ftype_m,     ftype_m),    # deepen
            GEMBlock(ftype_m,     ftype_out),  # project to invariant scalars
        ])

        # Project to desired hidden_dim
        gem_out_dim = feature_dim(ftype_out)   # = 32
        self.out_proj = (
            nn.Linear(gem_out_dim, hidden_dim)
            if gem_out_dim != hidden_dim
            else nn.Identity()
        )

    def forward(
        self,
        x:           Tensor,   # (V, in_features)
        edge_index:  Tensor,   # (2, E)
        angles:      Tensor,   # (E,) θ_pq
        transporters: Tensor,  # (E,) g_{q→p}
    ) -> Dict[str, Tensor]:
        h = self.input_embed(x)   # (V, 24) — scalar channels only

        for block in self.blocks:
            h = block(h, edge_index, angles, transporters)
        # h: (V, 32) — pure scalar, gauge-invariant

        per_vertex = self.out_proj(h)              # (V, hidden_dim)
        global_emb = self.out_proj(h.mean(dim=0))  # (hidden_dim,)

        return {
            'per_vertex': per_vertex,
            'global':     global_emb,
        }
