# data/mesh_geometry.py
"""
Core differential geometry operations for GEM-CNN on curved surfaces.

Implements (following de Haan et al. 2020, Secs 4.1 & 4.2):
  - Area-weighted vertex normal estimation
  - Discrete Riemannian logarithmic map  (projects edges onto tangent plane)
  - Local reference frame construction   (gauge choice)
  - Neighbour angle computation           (θ_pq)
  - Discrete Levi-Civita parallel transporters (g_{q→p}, Eq. 6)

Used both for triangulated surface meshes (if available) and for
point clouds where tangent planes are estimated via local PCA.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def compute_vertex_normals(
    vertices: Tensor,   # (V, 3)
    faces:    Tensor,   # (F, 3)  long indices
) -> Tensor:
    """Area-weighted average of adjacent face normals. Returns unit (V, 3)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = torch.linalg.cross(v1 - v0, v2 - v0)

    normals = torch.zeros_like(vertices)
    for i in range(3):
        idx = faces[:, i].unsqueeze(1).expand(-1, 3)
        normals.scatter_add_(0, idx, cross)

    return F.normalize(normals, dim=-1)


def log_map(
    p:        Tensor,   # (E, 3)  source vertex positions
    q:        Tensor,   # (E, 3)  neighbour vertex positions
    normal_p: Tensor,   # (E, 3)  unit normals at p
) -> Tensor:
    """
    Project edge vector (q - p) onto the tangent plane at p,
    rescaled to preserve original edge length.

    Returns tangent vectors of shape (E, 3) in TpM ⊂ R³.
    """
    edge = q - p
    edge_len = edge.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    dot  = (edge * normal_p).sum(dim=-1, keepdim=True)
    proj = edge - dot * normal_p
    proj_len = proj.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return edge_len * proj / proj_len


def build_reference_frames(
    normals:     Tensor,   # (V, 3)
    ref_vectors: Tensor,   # (V, 3)  log_p(q0)
) -> Tuple[Tensor, Tensor]:
    """
    Construct right-handed orthonormal frame (e1, e2) of the tangent plane.

        e1 = normalize(ref_vector)
        e2 = normal × e1
    """
    e1 = F.normalize(ref_vectors, dim=-1)
    e2 = torch.linalg.cross(normals, e1)
    e2 = F.normalize(e2, dim=-1)
    return e1, e2


def compute_neighbour_angles(
    log_pq: Tensor,   # (E, 3)
    e1_p:   Tensor,   # (E, 3)
    e2_p:   Tensor,   # (E, 3)
) -> Tensor:
    """
    Polar angle θ_pq = atan2(log_p(q)·e2_p, log_p(q)·e1_p).
    Returns (E,) in (-π, π].
    """
    cos_comp = (log_pq * e1_p).sum(dim=-1)
    sin_comp = (log_pq * e2_p).sum(dim=-1)
    return torch.atan2(sin_comp, cos_comp)


def _rotation_align_normals(
    n_src: Tensor,
    n_tgt: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """SO(3) rotation aligning n_src onto n_tgt via Rodrigues."""
    axis      = torch.linalg.cross(n_src, n_tgt)
    sin_alpha = axis.norm(dim=-1, keepdim=True)
    cos_alpha = (n_src * n_tgt).sum(dim=-1, keepdim=True)
    axis      = axis / sin_alpha.clamp(min=1e-8)
    return cos_alpha, sin_alpha, axis


def rodrigues_rotate(
    v:         Tensor,
    axis:      Tensor,
    cos_alpha: Tensor,
    sin_alpha: Tensor,
) -> Tensor:
    """Rodrigues' rotation formula: rotate v by angle α around axis."""
    dot   = (axis * v).sum(dim=-1, keepdim=True)
    cross = torch.linalg.cross(axis, v)
    return v * cos_alpha + cross * sin_alpha + axis * dot * (1.0 - cos_alpha)


def compute_parallel_transporters(
    e1_src: Tensor,
    e2_src: Tensor,
    e1_tgt: Tensor,
    n_src:  Tensor,
    n_tgt:  Tensor,
) -> Tensor:
    """
    Discrete Levi-Civita connection angle g_{q→p} (Eq. 6 of GEM paper).

    Corrects for the misalignment between tangent frames at adjacent vertices
    on a curved surface — essential for coherent feature aggregation.

    Returns (E,) transporter angles in (-π, π].
    """
    cos_alpha, sin_alpha, axis = _rotation_align_normals(n_src, n_tgt)

    flat = (sin_alpha.squeeze(-1).abs() < 1e-6)

    e1_rot = rodrigues_rotate(e1_src, axis, cos_alpha, sin_alpha)
    e2_rot = rodrigues_rotate(e2_src, axis, cos_alpha, sin_alpha)

    e1_rot[flat] = e1_src[flat]
    e2_rot[flat] = e2_src[flat]

    cos_g = (e1_rot * e1_tgt).sum(dim=-1)
    sin_g = (e2_rot * e1_tgt).sum(dim=-1)
    return torch.atan2(sin_g, cos_g)


def precompute_geometry(
    vertices:   Tensor,   # (V, 3)
    faces:      Tensor,   # (F, 3) long
    edge_index: Tensor,   # (2, E) long  [src=q, tgt=p]
) -> dict:
    """
    One-shot precomputation of all GEM geometric quantities for a mesh.
    Returns dict with: normals, angles, transporters, e1, e2.
    """
    device = vertices.device
    src, tgt = edge_index[0], edge_index[1]

    normals = compute_vertex_normals(vertices, faces)

    log_ij = log_map(
        p=vertices[tgt],
        q=vertices[src],
        normal_p=normals[tgt],
    )

    full_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    full_log        = torch.cat([log_ij, -log_ij], dim=0)
    f_src, f_tgt    = full_edge_index[0], full_edge_index[1]

    V = vertices.shape[0]
    ref_edge = torch.full((V,), f_tgt.shape[0], dtype=torch.long, device=device)
    for i in range(f_tgt.shape[0] - 1, -1, -1):
        ref_edge[f_tgt[i]] = i
    ref_log = full_log[ref_edge]

    e1, e2 = build_reference_frames(normals, ref_log)

    angles = compute_neighbour_angles(
        log_pq=log_ij,
        e1_p=e1[tgt],
        e2_p=e2[tgt],
    )

    transporters = compute_parallel_transporters(
        e1_src=e1[src],
        e2_src=e2[src],
        e1_tgt=e1[tgt],
        n_src=normals[src],
        n_tgt=normals[tgt],
    )

    return {
        'normals':      normals,
        'e1':           e1,
        'e2':           e2,
        'angles':       angles,
        'transporters': transporters,
    }
