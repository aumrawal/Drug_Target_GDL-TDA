# models/fusion.py
"""
Bidirectional cross-attention fusion module.

Fuses drug and pocket representations with topological features to
predict binding affinity. Three information streams are combined:

  1. LOCAL CHEMISTRY (drug encoder): per-atom SchNet embeddings
  2. LOCAL GEOMETRY  (pocket encoder): per-residue GEM embeddings
  3. GLOBAL TOPOLOGY (TDA): persistence image vectors for both molecules

The cross-attention lets individual drug atoms "look at" pocket residues
and vice versa — identifying which parts of the drug interact most strongly
with which parts of the pocket. This is more expressive than simply
concatenating two global mean-pools.

TDA features inject shape information that message-passing cannot see:
  - Drug H₁: aromatic ring count and geometry
  - Drug H₁: macrocycle topology
  - Pocket H₀: number of distinct residue clusters
  - Pocket H₁: loop topology of the binding site

Fusion input dimension:
    common_dim × 4  (drug_global + pocket_global + drug_attended + pocket_attended)
  + tda_compressed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: query from source, key/value from context.

    query  : (Q, dim)
    context: (K, dim)
    output : (Q, dim)  — same shape as query
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(dim, dim)
        self.k_proj   = nn.Linear(dim, dim)
        self.v_proj   = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm     = nn.LayerNorm(dim)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        Q = query.shape[0]

        def split(t):
            L = t.shape[0]
            return t.view(L, self.n_heads, self.head_dim).transpose(0, 1)

        q = split(self.q_proj(query))      # (H, Q, d)
        k = split(self.k_proj(context))    # (H, K, d)
        v = split(self.v_proj(context))    # (H, K, d)

        attn = F.softmax(torch.bmm(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = torch.bmm(attn, v)                                   # (H, Q, d)
        out  = out.transpose(0, 1).contiguous().view(Q, -1)         # (Q, dim)
        return self.norm(query + self.out_proj(out))


class FusionModule(nn.Module):
    """
    Drug-Pocket interaction head with TDA injection.

    Args:
        drug_dim   : drug encoder output dim
        pocket_dim : pocket encoder output dim
        tda_dim    : total TDA feature dim (drug + pocket concatenated)
        hidden_dim : MLP hidden width
        n_heads    : number of cross-attention heads
    """

    def __init__(
        self,
        drug_dim:   int,
        pocket_dim: int,
        tda_dim:    int,
        hidden_dim: int = 256,
        n_heads:    int = 4,
    ):
        super().__init__()
        common = max(drug_dim, pocket_dim)
        # Round to be divisible by n_heads
        common = ((common + n_heads - 1) // n_heads) * n_heads

        self.drug_proj   = nn.Linear(drug_dim,   common)
        self.pocket_proj = nn.Linear(pocket_dim, common)

        self.drug_to_pocket  = CrossAttention(common, n_heads)
        self.pocket_to_drug  = CrossAttention(common, n_heads)

        # Compress TDA features
        tda_hidden = max(hidden_dim // 4, 32)
        self.tda_compress = nn.Sequential(
            nn.Linear(tda_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, tda_hidden),
        )

        # Final predictor
        # inputs: 4 × common + tda_hidden
        predictor_in = 4 * common + tda_hidden
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Scale final layer for stable initialization
        with torch.no_grad():
            self.predictor[-1].weight.mul_(0.1)

    def forward(
        self,
        drug_per_atom:  Tensor,   # (N_d, drug_dim)
        drug_global:    Tensor,   # (drug_dim,)
        pocket_per_res: Tensor,   # (N_p, pocket_dim)
        pocket_global:  Tensor,   # (pocket_dim,)
        tda_features:   Tensor,   # (tda_dim,)
    ) -> Tensor:
        # Project to common dimension
        d_atoms  = self.drug_proj(drug_per_atom)       # (N_d, C)
        p_res    = self.pocket_proj(pocket_per_res)    # (N_p, C)
        d_global = self.drug_proj(drug_global)         # (C,)
        p_global = self.pocket_proj(pocket_global)     # (C,)

        # Cross-attention
        d_attended = self.drug_to_pocket(d_atoms, p_res)    # drug attending pocket
        p_attended = self.pocket_to_drug(p_res, d_atoms)    # pocket attending drug

        # Mean-pool attended representations
        d_attn_mean = d_attended.mean(0)    # (C,)
        p_attn_mean = p_attended.mean(0)    # (C,)

        # Compress TDA
        tda = self.tda_compress(tda_features)   # (tda_hidden,)

        # Concatenate all signals
        fused = torch.cat([d_global, p_global, d_attn_mean, p_attn_mean, tda], dim=0)

        return self.predictor(fused).squeeze(-1)   # scalar pKd estimate
