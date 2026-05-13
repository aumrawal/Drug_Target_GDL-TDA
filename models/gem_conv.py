# models/gem_conv.py
"""
Gauge Equivariant Mesh Convolution layer.

Implements Algorithm 1 of de Haan et al. (2020):

    f'_p = Σ_i  w_self_i  · K_self_i · f_p
         + Σ_{i,q∈N(p)}  w_neigh_i · K_neigh_i(θ_pq) · ρ_in(g_{q→p}) · f_q

Used here for the protein binding pocket encoder, where the pocket surface
is treated as a curved 2D manifold embedded in 3D space — exactly the
setting GEM was designed for.

Three properties make GEM superior to plain GCN for binding pockets:
  1. Anisotropic kernel K_neigh(θ): distinguishes radial vs. tangential patterns
  2. Parallel transport ρ_in(g_{q→p}): aligns features across curved surfaces
  3. SO(2) irreps: exact equivariance to gauge (reference frame) choices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scatter_add(src, index, dim, dim_size):
    """Pure PyTorch scatter_add over first dimension."""
    out = torch.zeros(dim_size, src.shape[-1], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out


from models.irreps import (
    FeatureType,
    EquivariantKernelBasis,
    feature_dim,
    rho_batch,
)


# ────────────────────────────────────────────────────────────────────────────
# Parallel transport application
# ────────────────────────────────────────────────────────────────────────────

def apply_parallel_transport(
    features:     Tensor,       # (E, C_in)
    transporters: Tensor,       # (E,)  g_{q→p} angles
    ftype_in:     FeatureType,
) -> Tensor:
    """
    Apply ρ_in(g_{q→p}) to each neighbour's feature vector.

    For each SO(2) irrep block of type ρ_n:
        transported = R(n · g_{q→p}) · feature_block

    This is the operation that makes GEM correct on curved surfaces:
    a feature measured at residue q is rotated into the local frame of
    residue p before being aggregated — so information accumulates coherently
    regardless of how the protein surface curves.
    """
    transported = torch.zeros_like(features)
    offset = 0
    for (order, mult) in ftype_in:
        d = 1 if order == 0 else 2
        block = features[:, offset : offset + mult * d]

        if order == 0:
            transported[:, offset : offset + mult * d] = block
        else:
            block = block.reshape(-1, mult, d)
            R     = rho_batch(order, transporters)
            rotated = torch.einsum('emd,erd->emr', block, R)
            transported[:, offset : offset + mult * d] = rotated.reshape(-1, mult*d)

        offset += mult * d

    return transported


# ────────────────────────────────────────────────────────────────────────────
# GEM Convolution Layer
# ────────────────────────────────────────────────────────────────────────────

class GEMConv(nn.Module):
    """
    Single Gauge Equivariant Mesh Convolution layer.

    Args:
        ftype_in  : input feature type  e.g. [(0,8), (1,8)]
        ftype_out : output feature type e.g. [(0,16), (1,16)]
    """

    def __init__(
        self,
        ftype_in:  FeatureType,
        ftype_out: FeatureType,
    ):
        super().__init__()
        self.ftype_in  = ftype_in
        self.ftype_out = ftype_out
        self.dim_in    = feature_dim(ftype_in)
        self.dim_out   = feature_dim(ftype_out)
        self.kernel    = EquivariantKernelBasis(ftype_in, ftype_out)

    def forward(
        self,
        x:            Tensor,   # (V, C_in)
        edge_index:   Tensor,   # (2, E) [src=q, tgt=p]
        angles:       Tensor,   # (E,) θ_pq
        transporters: Tensor,   # (E,) g_{q→p}
    ) -> Tensor:
        src, tgt = edge_index[0], edge_index[1]
        V = x.shape[0]

        K_self = self.kernel.eval_self()
        out = x @ K_self.T

        f_q = x[src]
        f_q_transported = apply_parallel_transport(f_q, transporters, self.ftype_in)
        K_neigh = self.kernel.eval_neigh(angles)
        msg = torch.bmm(K_neigh, f_q_transported.unsqueeze(-1)).squeeze(-1)
        out = out + scatter_add(msg, tgt, dim=0, dim_size=V)

        return out


# ────────────────────────────────────────────────────────────────────────────
# Regular Non-linearity  (Sec 5 of GEM paper)
# ────────────────────────────────────────────────────────────────────────────

class RegularNonlinearity(nn.Module):
    """
    Approximately gauge-equivariant nonlinearity.

    For scalar blocks (ρ₀): plain pointwise activation.
    For vector/tensor blocks (ρ_n, n>0): norm nonlinearity — preserves
    direction while applying softplus to the magnitude.
    """

    def __init__(self, ftype: FeatureType):
        super().__init__()
        self.ftype = ftype

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        offset = 0

        for (order, mult) in self.ftype:
            d = 1 if order == 0 else 2
            C = mult * d
            block = x[:, offset : offset + C]

            if order == 0:
                out[:, offset : offset + C] = F.silu(block)
            else:
                block = block.reshape(-1, mult, d)
                norms = block.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                new_norms = F.softplus(norms)
                out[:, offset : offset + C] = (block * (new_norms / norms)).reshape(-1, C)

            offset += C

        return out


# ────────────────────────────────────────────────────────────────────────────
# GEM-CNN Block: Conv + Norm + Nonlinearity + Skip
# ────────────────────────────────────────────────────────────────────────────

class GEMBlock(nn.Module):
    """
    Residual GEM-CNN block:
        f → GEMConv → LayerNorm → RegularNonlinearity
        output = result + skip(f)
    """

    def __init__(
        self,
        ftype_in:  FeatureType,
        ftype_out: FeatureType,
    ):
        super().__init__()
        self.ftype_in  = ftype_in
        self.ftype_out = ftype_out
        dim_in  = feature_dim(ftype_in)
        dim_out = feature_dim(ftype_out)

        self.conv   = GEMConv(ftype_in, ftype_out)
        self.norm   = nn.LayerNorm(dim_out)
        self.nonlin = RegularNonlinearity(ftype_out)
        self.skip   = nn.Identity() if dim_in == dim_out else nn.Linear(dim_in, dim_out, bias=False)

    def forward(
        self,
        x:            Tensor,
        edge_index:   Tensor,
        angles:       Tensor,
        transporters: Tensor,
    ) -> Tensor:
        if self.training:
            from torch.utils.checkpoint import checkpoint
            def _fwd(x, ei, ang, tr):
                h = self.conv(x, ei, ang, tr)
                return self.nonlin(self.norm(h))
            h = checkpoint(_fwd, x, edge_index, angles, transporters, use_reentrant=False)
        else:
            h = self.nonlin(self.norm(self.conv(x, edge_index, angles, transporters)))
        return h + self.skip(x)
