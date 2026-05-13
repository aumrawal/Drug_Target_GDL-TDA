# models/irreps.py
"""
SO(2) irreducible representations and gauge-equivariant kernel basis.

Implements the kernel constraint solution from Table 1 of:
    de Haan et al., "Gauge Equivariant Mesh CNNs", 2020.

The key insight: any gauge-equivariant kernel K_neigh(θ) mapping from
irrep ρ_n to irrep ρ_m must satisfy:

    K_neigh(θ - g) = ρ_m(-g) · K_neigh(θ) · ρ_n(g)   ∀ g,θ ∈ [0, 2π)

The solution space is spanned by a small set of "basis kernels" — fixed
angular functions of θ multiplied by learned scalar weights.

In the molecular context:
    - Scalar (ρ₀) features: invariant quantities (energy, partial charge)
    - Vector (ρ₁) features: direction-dependent quantities (surface gradients)
    - Tensor (ρ₂) features: higher-order anisotropic patterns
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple


# ────────────────────────────────────────────────────────────────────────────
# Type definitions
# ────────────────────────────────────────────────────────────────────────────

# A "feature type" is a list of (irrep_order, multiplicity) pairs.
# E.g. [(0, 16), (1, 16), (2, 8)] = 16ρ₀ ⊕ 16ρ₁ ⊕ 8ρ₂
FeatureType = List[Tuple[int, int]]


def feature_dim(ftype: FeatureType) -> int:
    """Total dimension of a feature type."""
    return sum(
        mult * (1 if order == 0 else 2)
        for order, mult in ftype
    )


def scalar_type(n_channels: int) -> FeatureType:
    """All-scalar feature type: n_channels × ρ₀."""
    return [(0, n_channels)]


# ────────────────────────────────────────────────────────────────────────────
# SO(2) representation matrices
# ────────────────────────────────────────────────────────────────────────────

def rho(order: int, angle: Tensor) -> Tensor:
    """
    SO(2) irrep ρ_n evaluated at angle g.

    ρ₀(g) = [[1]]
    ρ_n(g) = [[cos(ng), -sin(ng)],
               [sin(ng),  cos(ng)]]

    Args:
        order: irrep order n ≥ 0
        angle: scalar tensor [radians]
    Returns:
        (d, d) rotation matrix where d = 1 if order=0, else 2
    """
    if order == 0:
        return torch.ones(1, 1, dtype=angle.dtype, device=angle.device)
    g = order * angle
    c, s = torch.cos(g), torch.sin(g)
    return torch.stack([
        torch.stack([ c, -s]),
        torch.stack([ s,  c]),
    ])   # (2, 2)


def rho_batch(order: int, angles: Tensor) -> Tensor:
    """
    Batched SO(2) irrep evaluation.

    Args:
        order  : irrep order n ≥ 0
        angles : (E,) edge angles
    Returns:
        (E, d, d) where d = 1 if order=0, else 2
    """
    E = angles.shape[0]
    if order == 0:
        return torch.ones(E, 1, 1, dtype=angles.dtype, device=angles.device)
    g = order * angles         # (E,)
    c = torch.cos(g)           # (E,)
    s = torch.sin(g)           # (E,)
    row0 = torch.stack([ c, -s], dim=-1)   # (E, 2)
    row1 = torch.stack([ s,  c], dim=-1)   # (E, 2)
    return torch.stack([row0, row1], dim=1)  # (E, 2, 2)


# ────────────────────────────────────────────────────────────────────────────
# Basis kernel evaluation  (Table 1 of GEM paper)
# ────────────────────────────────────────────────────────────────────────────

def basis_kernels_neigh(
    n_in:   int,
    n_out:  int,
    angles: Tensor,  # (E,) edge angles θ_pq
) -> Tensor:
    """
    Evaluate all basis kernels for K_neigh mapping ρ_{n_in} → ρ_{n_out}.
    Returns tensor of shape (E, d_out, d_in, num_basis).
    """
    E = angles.shape[0]
    device = angles.device

    if n_in == 0 and n_out == 0:
        return torch.ones(E, 1, 1, 1, device=device)

    elif n_in > 0 and n_out == 0:
        c = torch.cos(n_in * angles)
        s = torch.sin(n_in * angles)
        k0 = torch.stack([ c,  s], dim=-1).reshape(E, 1, 2)
        k1 = torch.stack([ s, -c], dim=-1).reshape(E, 1, 2)
        return torch.stack([k0, k1], dim=-1)   # (E, 1, 2, 2)

    elif n_in == 0 and n_out > 0:
        c = torch.cos(n_out * angles)
        s = torch.sin(n_out * angles)
        k0 = torch.stack([ c,  s], dim=-1).reshape(E, 2, 1)
        k1 = torch.stack([ s, -c], dim=-1).reshape(E, 2, 1)
        return torch.stack([k0, k1], dim=-1)   # (E, 2, 1, 2)

    else:  # n_in > 0 and n_out > 0 — 4 bases
        p = n_out + n_in
        q = abs(n_out - n_in)
        cp = torch.cos(p * angles); sp = torch.sin(p * angles)
        cq = torch.cos(q * angles); sq = torch.sin(q * angles)
        sign = 1.0 if n_out >= n_in else -1.0

        def mat(a, b, c, d):
            row0 = torch.stack([a, b], dim=-1)
            row1 = torch.stack([c, d], dim=-1)
            return torch.stack([row0, row1], dim=1)

        k0 = mat( cq, -sq*sign,  sq*sign,  cq)
        k1 = mat( sq,  cq*sign, -cq*sign,  sq)
        k2 = mat( cp,  sp,       sp,      -cp)
        k3 = mat(-sp,  cp,       cp,       sp)

        return torch.stack([k0, k1, k2, k3], dim=-1)   # (E, 2, 2, 4)


def basis_kernels_self(
    n_in:  int,
    n_out: int,
    dim:   int,
) -> Tensor:
    """
    Basis kernels for K_self (angle-independent).
    Only non-zero when n_in == n_out.
    Returns (d_out, d_in, num_basis) constant tensor.
    """
    if n_in != n_out:
        return None

    if n_in == 0:
        return torch.ones(1, 1, 1)

    I = torch.eye(2).unsqueeze(-1)
    J = torch.tensor([[0., 1.], [-1., 0.]]).unsqueeze(-1)
    return torch.cat([I, J], dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# Full equivariant kernel
# ────────────────────────────────────────────────────────────────────────────

def count_parameters_neigh(ftype_in: FeatureType, ftype_out: FeatureType) -> int:
    total = 0
    for (n_out, m_out) in ftype_out:
        for (n_in, m_in) in ftype_in:
            total += _n_basis_neigh(n_in, n_out) * m_in * m_out
    return total


def count_parameters_self(ftype_in: FeatureType, ftype_out: FeatureType) -> int:
    total = 0
    for (n_out, m_out) in ftype_out:
        for (n_in, m_in) in ftype_in:
            if n_in == n_out:
                total += (1 if n_in == 0 else 2) * m_in * m_out
    return total


def _n_basis_neigh(n_in: int, n_out: int) -> int:
    if n_in == 0 and n_out == 0: return 1
    if n_in == 0 or n_out == 0:  return 2
    return 4


class EquivariantKernelBasis(nn.Module):
    """
    Parameterised gauge-equivariant kernel K_neigh(θ) + K_self.

        K_neigh(θ) = Σ_i  w_neigh_i * BasisKernel_i(θ)
        K_self     = Σ_i  w_self_i  * BasisKernel_i
    """

    def __init__(self, ftype_in: FeatureType, ftype_out: FeatureType):
        super().__init__()
        self.ftype_in  = ftype_in
        self.ftype_out = ftype_out
        self.dim_in    = feature_dim(ftype_in)
        self.dim_out   = feature_dim(ftype_out)

        n_w_neigh = count_parameters_neigh(ftype_in, ftype_out)
        n_w_self  = count_parameters_self(ftype_in, ftype_out)

        self.w_neigh = nn.Parameter(
            torch.randn(n_w_neigh) * (2.0 / (self.dim_in + self.dim_out)) ** 0.5
        )
        self.w_self = nn.Parameter(
            torch.randn(n_w_self) * (2.0 / (self.dim_in + self.dim_out)) ** 0.5
        )

        self._neigh_layout = self._build_neigh_layout()
        self._self_layout  = self._build_self_layout()

    def _build_neigh_layout(self):
        layout = []
        w = 0
        for (n_out, m_out) in self.ftype_out:
            for (n_in, m_in) in self.ftype_in:
                nb = _n_basis_neigh(n_in, n_out)
                layout.append((n_in, m_in, n_out, m_out, nb, w))
                w += nb * m_in * m_out
        return layout

    def _build_self_layout(self):
        layout = []
        w = 0
        for (n_out, m_out) in self.ftype_out:
            for (n_in, m_in) in self.ftype_in:
                if n_in == n_out:
                    nb = 1 if n_in == 0 else 2
                    layout.append((n_in, m_in, n_out, m_out, nb, w))
                    w += nb * m_in * m_out
        return layout

    def eval_neigh(self, angles: Tensor) -> Tensor:
        """Evaluate K_neigh(θ). Returns (E, dim_out, dim_in)."""
        E = angles.shape[0]
        K = torch.zeros(E, self.dim_out, self.dim_in,
                        device=angles.device, dtype=angles.dtype)

        out_offset = 0
        for (n_out, m_out) in self.ftype_out:
            d_out = 1 if n_out == 0 else 2
            in_offset = 0
            for (n_in, m_in) in self.ftype_in:
                d_in = 1 if n_in == 0 else 2
                nb   = _n_basis_neigh(n_in, n_out)

                entry = next(e for e in self._neigh_layout
                             if e[0]==n_in and e[2]==n_out)
                w_start, n_b = entry[5], entry[4]
                w_block = self.w_neigh[w_start : w_start + nb * m_in * m_out]
                w_block = w_block.reshape(m_out, m_in, nb)

                basis = basis_kernels_neigh(n_in, n_out, angles)

                k_block = torch.einsum('ijb,exyb->eixjy', w_block, basis)
                k_block = k_block.reshape(E, m_out*d_out, m_in*d_in)

                K[:, out_offset:out_offset+m_out*d_out,
                     in_offset:in_offset+m_in*d_in] = k_block
                in_offset += m_in * d_in
            out_offset += m_out * d_out

        return K

    def eval_self(self) -> Tensor:
        """Evaluate K_self. Returns (dim_out, dim_in)."""
        K = torch.zeros(self.dim_out, self.dim_in,
                        device=self.w_self.device, dtype=self.w_self.dtype)

        out_offset = 0
        for (n_out, m_out) in self.ftype_out:
            d_out = 1 if n_out == 0 else 2
            in_offset = 0
            for (n_in, m_in) in self.ftype_in:
                d_in = 1 if n_in == 0 else 2
                if n_in != n_out:
                    in_offset += m_in * d_in
                    continue

                nb = 1 if n_in == 0 else 2
                entry = next(e for e in self._self_layout
                             if e[0]==n_in and e[2]==n_out)
                w_start = entry[5]
                w_block = self.w_self[w_start : w_start + nb * m_in * m_out]
                w_block = w_block.reshape(m_out, m_in, nb)

                basis = basis_kernels_self(n_in, n_out, d_in)
                if basis is None:
                    in_offset += m_in * d_in
                    continue

                basis = basis.to(self.w_self.device)
                k_block = torch.einsum('ijb,xyb->ixjy', w_block, basis)
                k_block = k_block.reshape(m_out*d_out, m_in*d_in)

                K[out_offset:out_offset+m_out*d_out,
                  in_offset:in_offset+m_in*d_in] = k_block
                in_offset += m_in * d_in
            out_offset += m_out * d_out

        return K
