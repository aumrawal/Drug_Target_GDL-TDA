# data/tda_features.py
"""
Topological Data Analysis via Persistent Homology for molecular point clouds.

Uses Ripser (C++ compiled) as the primary backend for computing Vietoris-Rips
persistent homology. Ripser is 100–1000× faster than a Python implementation
for the point cloud sizes encountered in molecular contexts (20–200 atoms):

  - "Apparent pairs" optimization: identifies most birth-death pairs without
    any matrix reduction — the key algorithmic advance over standard reduction
  - "Clearing lemma": skips zero-columns entirely
  - Coboundary (dual) representation: much sparser than boundary matrix
  - C++ core: eliminates Python loop overhead entirely

Install: pip install ripser

The from-scratch fallback (boundary matrix reduction, lowest-1 pivot) is
retained in _compute_tda_scratch() for environments where ripser is not
available. It is functionally identical but significantly slower.

Output: persistence image vectors (weighted 2D Gaussians in birth×persistence
space) — a fixed-size differentiable-friendly representation of the diagram.
"""

import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple, Dict

try:
    from ripser import ripser as _ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# Persistence image vectorization  (shared by both backends)
# ────────────────────────────────────────────────────────────────────────────

def persistence_image(
    pairs:      List[Tuple[float, float]],
    resolution: int   = 20,
    max_val:    float = None,
    sigma:      float = None,
) -> np.ndarray:
    """
    Convert a persistence diagram to a fixed-size image vector.

    Each (birth, persistence) point contributes a weighted 2D Gaussian in
    the [0, max_val]² grid of (birth, persistence=death-birth) space.
    Weight = persistence so longer-lived features dominate.

    Returns (resolution²,) float32 array normalized to [0, 1].
    """
    if len(pairs) == 0:
        return np.zeros(resolution * resolution, dtype=np.float32)

    births   = np.array([b for b, d in pairs], dtype=np.float64)
    persists = np.array([d - b for b, d in pairs], dtype=np.float64)

    if max_val is None:
        max_val = float(max(np.max(births + persists), 1e-3))
    if sigma is None:
        sigma = max_val / resolution * 2.0

    gb = np.linspace(0, max_val, resolution)   # birth axis
    gp = np.linspace(0, max_val, resolution)   # persistence axis

    image = np.zeros((resolution, resolution), dtype=np.float64)
    for b, p in zip(births, persists):
        if p < 1e-8:
            continue
        gauss_b = np.exp(-((gb - b) ** 2) / (2 * sigma ** 2))
        gauss_p = np.exp(-((gp - p) ** 2) / (2 * sigma ** 2))
        image  += p * np.outer(gauss_b, gauss_p)

    if image.max() > 0:
        image /= image.max()

    return image.flatten().astype(np.float32)


def _diagram_to_pairs(diagram: np.ndarray, max_val: float) -> List[Tuple[float, float]]:
    """
    Convert a Ripser diagram array (N, 2) to a list of finite (birth, death) pairs.
    Infinite death values (essential bars) are clipped to max_val * 1.5.
    """
    pairs = []
    for b, d in diagram:
        if np.isnan(b) or np.isnan(d):
            continue
        death = float(min(d, max_val * 1.5)) if np.isinf(d) else float(d)
        pairs.append((float(b), death))
    return pairs


# ────────────────────────────────────────────────────────────────────────────
# Ripser backend  (primary — requires: pip install ripser)
# ────────────────────────────────────────────────────────────────────────────

def _compute_tda_ripser(
    points:       np.ndarray,
    max_edge_len: float,
    resolution:   int,
) -> Dict[str, np.ndarray]:
    """
    Compute H₀ and H₁ persistence using Ripser.

    Ripser accepts raw point coordinates and computes Euclidean distances
    internally. The `thresh` parameter truncates the filtration at max_edge_len,
    matching our semantics exactly.
    """
    result = _ripser(points, maxdim=1, thresh=max_edge_len)
    dgms   = result['dgms']

    h0_pairs = _diagram_to_pairs(dgms[0], max_edge_len)
    h1_pairs = _diagram_to_pairs(dgms[1], max_edge_len)

    return {
        'h0_image': persistence_image(h0_pairs, resolution=resolution, max_val=max_edge_len),
        'h1_image': persistence_image(h1_pairs, resolution=resolution, max_val=max_edge_len),
        'h0_pairs': h0_pairs,
        'h1_pairs': h1_pairs,
    }


# ────────────────────────────────────────────────────────────────────────────
# From-scratch fallback  (no external dependencies, but O(n³) Python loops)
# ────────────────────────────────────────────────────────────────────────────

def _compute_tda_scratch(
    points:       np.ndarray,
    max_edge_len: float,
    resolution:   int,
    max_points:   int,
    seed:         int,
) -> Dict[str, np.ndarray]:
    """
    From-scratch Vietoris-Rips + boundary matrix reduction.
    Adapted from TDA_project/main.py (lowest-1 pivot algorithm, GF(2)).

    Significantly slower than Ripser due to:
      - Pure Python O(n³) triangle enumeration
      - Dense boundary matrix storage
      - No apparent-pairs or clearing optimizations
    Subsampling to max_points is applied to keep runtime manageable.
    """
    N = len(points)
    if N > max_points:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(N, max_points, replace=False)]
        N = max_points

    # Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    # Build edges
    edges = sorted(
        [(i, j, float(dist[i, j]))
         for i in range(N) for j in range(i + 1, N)
         if dist[i, j] <= max_edge_len],
        key=lambda e: e[2],
    )
    if not edges:
        d = resolution * resolution
        return {'h0_image': np.zeros(d, np.float32),
                'h1_image': np.zeros(d, np.float32),
                'h0_pairs': [], 'h1_pairs': []}

    max_d    = max(e[2] for e in edges)
    edge_set = {(e[0], e[1]) for e in edges}

    # Build triangles
    triangles = []
    seen = set()
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, N):
                if (i, k) not in edge_set or (j, k) not in edge_set:
                    continue
                if (i, j, k) not in seen:
                    seen.add((i, j, k))
                    triangles.append((i, j, k, float(max(dist[i][j], dist[i][k], dist[j][k]))))
    triangles.sort(key=lambda t: t[3])

    def reduce(B):
        B = B.copy().astype(np.int32)
        lo = {}
        for col in range(B.shape[1]):
            nz = np.where(B[:, col] != 0)[0]
            if not len(nz):
                continue
            piv = int(nz[-1])
            while piv in lo:
                B[:, col] = (B[:, col] + B[:, lo[piv]]) % 2
                nz = np.where(B[:, col] != 0)[0]
                if not len(nz):
                    piv = -1; break
                piv = int(nz[-1])
            if piv >= 0:
                lo[piv] = col
        return lo

    # H₀
    B01 = np.zeros((N, len(edges)), dtype=np.uint8)
    for j, (i, k, _) in enumerate(edges):
        B01[i, j] = B01[k, j] = 1
    lo01 = reduce(B01)
    h0_pairs = [(0.0, edges[col][2]) for _, col in sorted(lo01.items(), key=lambda x: edges[x[1]][2])]
    h0_pairs += [(0.0, max_d * 1.5)] * (N - len(h0_pairs))

    # H₁
    h1_pairs = []
    if triangles:
        eidx = {(e[0], e[1]): idx for idx, e in enumerate(edges)}
        B12  = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
        for j, (a, b, c, _) in enumerate(triangles):
            for u, v in [(a, b), (a, c), (b, c)]:
                key = (min(u, v), max(u, v))
                if key in eidx:
                    B12[eidx[key], j] = 1
        lo12 = reduce(B12)
        h1_pairs = [(edges[row][2], triangles[col][3])
                    for row, col in lo12.items()
                    if triangles[col][3] > edges[row][2] + 1e-8]

    return {
        'h0_image': persistence_image(h0_pairs, resolution=resolution, max_val=max_edge_len),
        'h1_image': persistence_image(h1_pairs, resolution=resolution, max_val=max_edge_len),
        'h0_pairs': h0_pairs,
        'h1_pairs': h1_pairs,
    }


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def compute_tda_features(
    points:       np.ndarray,        # (N, 3) 3D point cloud in Angstroms
    max_edge_len: float       = 8.0, # filtration radius (Å)
    resolution:   int         = 20,  # persistence image grid size
    max_points:   int         = 200, # scratch-fallback only: subsample threshold
    seed:         int         = 0,
) -> Dict[str, np.ndarray]:
    """
    Compute H₀ and H₁ TDA features from a 3D molecular point cloud.

    Uses Ripser when available (pip install ripser), otherwise falls back to
    the from-scratch boundary matrix reduction. The max_points subsampling
    is only applied in the scratch path — Ripser handles 200+ points easily.

    Returns dict with:
        h0_image : (resolution²,) persistence image for H₀
        h1_image : (resolution²,) persistence image for H₁
        h0_pairs : raw (birth, death) list for H₀
        h1_pairs : raw (birth, death) list for H₁
    """
    N = len(points)
    d = resolution * resolution
    if N < 2:
        return {'h0_image': np.zeros(d, np.float32),
                'h1_image': np.zeros(d, np.float32),
                'h0_pairs': [], 'h1_pairs': []}

    if RIPSER_AVAILABLE:
        return _compute_tda_ripser(points, max_edge_len, resolution)
    else:
        return _compute_tda_scratch(points, max_edge_len, resolution, max_points, seed)


def tda_to_tensor(tda: Dict[str, np.ndarray]) -> Tensor:
    """Concatenate H₀ and H₁ images into a single float tensor."""
    return torch.cat([
        torch.from_numpy(tda['h0_image']),
        torch.from_numpy(tda['h1_image']),
    ], dim=0)
