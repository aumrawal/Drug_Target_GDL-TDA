# `models/fusion.py`

## Overview

This module combines the three streams of information produced upstream — the SchNet-style drug embeddings, the GEM-CNN pocket embeddings, and the persistent-homology (TDA) feature vectors — into a single scalar pKd prediction. The two heavy components are `CrossAttention`, a multi-head scaled-dot-product attention layer used bidirectionally between drug atoms and pocket residues, and `FusionModule`, which orchestrates the projections, the cross-attentions, the TDA compression, and the final MLP head.

The cross-attention design is the key idea: rather than concatenating two mean-pooled global vectors and asking an MLP to figure out which drug-pocket pairs interact, the model lets each drug atom *query* the set of pocket residues for relevant context, and symmetrically lets each pocket residue query the drug. This biases the network toward learning local interaction patterns — exactly the regime where binding affinity is determined.

## Mathematical Foundations

### Scaled dot-product attention

Given $Q$ query vectors of dimension $d$ stacked into $Q \in \mathbb{R}^{N_q \times d}$, $K$ key vectors and $V$ value vectors stacked into $K, V \in \mathbb{R}^{N_k \times d}$, the standard scaled-dot-product attention output is

$$\mathrm{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V \;\in\; \mathbb{R}^{N_q \times d}.$$

The softmax row $\mathrm{softmax}(q_i K^\top / \sqrt{d})_j$ is the (probability) weight that query $i$ assigns to key $j$. The $1/\sqrt{d}$ scaling prevents the inner products from growing as $d$ increases and pushing the softmax into saturation.

### Multi-head attention

To let the model attend along several semantic axes simultaneously, the heads split the embedding dim $D$ into $H$ heads of dim $d = D/H$, attend independently per head, and concatenate:

$$\mathrm{MHA}(Q, K, V) \;=\; \mathrm{Concat}_{h=1}^{H} \mathrm{Attention}(QW_h^Q,\; KW_h^K,\; VW_h^V)\, W^O,$$

where $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{D \times d}$ are per-head projections and $W^O \in \mathbb{R}^{D \times D}$ is an output projection. In this implementation $H = 4$ heads. The query/key/value projections are implemented as single linear layers acting on the full $D$-dim input and then reshaped — equivalent to the per-head projections concatenated along the last axis.

### Bidirectional cross-attention in the DTI setting

Two cross-attention layers are used: drug-to-pocket and pocket-to-drug.

$$\widetilde{D} \;=\; \mathrm{MHA}\big(\,D,\; P,\; P\,\big), \qquad \widetilde{P} \;=\; \mathrm{MHA}\big(\,P,\; D,\; D\,\big),$$

where $D \in \mathbb{R}^{N_d \times C}$ are projected drug-atom embeddings and $P \in \mathbb{R}^{N_p \times C}$ are projected pocket-residue embeddings. Each atom now has a contextual representation that aggregates relevant pocket residues, and each residue has one that aggregates relevant atoms. Mean pooling over atoms and residues yields fixed-size summaries $\bar{\widetilde{D}}, \bar{\widetilde{P}} \in \mathbb{R}^C$.

### Final pooling and prediction

The model concatenates four $C$-dim vectors — the global drug embedding $d_g$, the global pocket embedding $p_g$, the cross-attended drug mean $\bar{\widetilde{D}}$, and the cross-attended pocket mean $\bar{\widetilde{P}}$ — together with a compressed TDA vector $\mathbf{t} \in \mathbb{R}^{C_{\text{tda}}}$ obtained by a two-layer MLP $\mathbb{R}^{1600} \to \mathbb{R}^{128} \to \mathbb{R}^{C_{\text{tda}}}$ where $C_{\text{tda}} = \max(64, 32)$:

$$\mathrm{fused} \;=\; \big[\, d_g \,\|\, p_g \,\|\, \bar{\widetilde{D}} \,\|\, \bar{\widetilde{P}} \,\|\, \mathbf{t} \,\big] \in \mathbb{R}^{4C + C_{\text{tda}}},$$

and a three-layer MLP with SiLU activations and a dropout of 0.1 produces the scalar pKd.

### Residual + LayerNorm inside the attention block

After the multi-head output projection, the attention layer adds the residual `query` and applies `LayerNorm`:

$$\mathrm{out} \;=\; \mathrm{LayerNorm}\!\big(Q + W^O \cdot \mathrm{MHA}(Q,K,V)\big).$$

This is the standard Pre/Post-LN transformer pattern (here post-LN). It stabilises training when the cross-attention update is large relative to the query.

## Code Walk-through

### `CrossAttention` (lines 34–70)

The constructor takes `dim` (the common embedding dim) and `n_heads` (default 4), asserts divisibility, and stores `head_dim = dim // n_heads` along with the scale factor `head_dim ** -0.5 = 1/\sqrt{d}`. It instantiates four linear layers `q_proj`, `k_proj`, `v_proj`, `out_proj`, each of shape `(dim, dim)`, and a `LayerNorm(dim)`.

The forward pass (lines 56–70) follows:

```python
def split(t):  # (L, dim) → (H, L, head_dim)
    L = t.shape[0]
    return t.view(L, self.n_heads, self.head_dim).transpose(0, 1)

q = split(self.q_proj(query))      # (H, Q, d)
k = split(self.k_proj(context))    # (H, K, d)
v = split(self.v_proj(context))    # (H, K, d)

attn = F.softmax(torch.bmm(q, k.transpose(-2, -1)) * self.scale, dim=-1)
out  = torch.bmm(attn, v)                               # (H, Q, d)
out  = out.transpose(0, 1).contiguous().view(Q, -1)     # (Q, dim)
return self.norm(query + self.out_proj(out))
```

The two `torch.bmm` calls implement $QK^\top$ and the attention-weighted sum. The transpose+view at the end concatenates the heads.

Note this is *single-sample* attention: it operates on unbatched 2D tensors `(L, dim)` rather than batched `(B, L, dim)`. This is consistent with the rest of the codebase, which processes one drug-pocket pair per forward pass.

### `FusionModule` (lines 73–156)

The constructor (lines 85–126):

- Computes `common = max(drug_dim, pocket_dim)` and rounds it up to the nearest multiple of `n_heads` so that the multi-head splitting divides evenly. With default `hidden_dim = 64` and `n_heads = 4`, `common = 64`.
- `drug_proj` and `pocket_proj` are linear maps to `common`. With matching default dims these become essentially identity-shaped re-projections.
- `drug_to_pocket` and `pocket_to_drug` are two `CrossAttention(common, n_heads)` modules.
- `tda_compress` is a small two-layer MLP $\mathbb{R}^{\text{tda\_dim}} \to \mathbb{R}^{\text{hidden\_dim}/2} \to \mathbb{R}^{\text{tda\_hidden}}$ with SiLU between, where `tda_hidden = max(hidden_dim // 4, 32) = max(64, 32) = 64` (using the predictor's `hidden_dim=256`, so `hidden_dim//4 = 64`). With the default `tda_dim = 2 \times 2 \times 20^2 = 1600` and `hidden_dim = 256`, this compresses $1600 \to 128 \to 64$.
- `predictor` is a three-layer MLP $\mathbb{R}^{4C + 64} \to \mathbb{R}^{256} \to \mathbb{R}^{128} \to \mathbb{R}^{1}$ with SiLU activations and a `Dropout(0.1)` between the first two layers.
- The final layer's weight is scaled down by `0.1` (lines 125–126) for stable initialisation — at the start of training, pKd predictions hover near zero rather than oscillating wildly.

The forward pass (lines 128–156):

```python
d_atoms  = self.drug_proj(drug_per_atom)         # (N_d, C)
p_res    = self.pocket_proj(pocket_per_res)      # (N_p, C)
d_global = self.drug_proj(drug_global)           # (C,)
p_global = self.pocket_proj(pocket_global)       # (C,)

d_attended = self.drug_to_pocket(d_atoms, p_res)    # (N_d, C) — atoms attend residues
p_attended = self.pocket_to_drug(p_res, d_atoms)    # (N_p, C) — residues attend atoms

d_attn_mean = d_attended.mean(0)                    # (C,)
p_attn_mean = p_attended.mean(0)                    # (C,)

tda = self.tda_compress(tda_features)               # (tda_hidden,)

fused = torch.cat([d_global, p_global, d_attn_mean, p_attn_mean, tda], dim=0)
return self.predictor(fused).squeeze(-1)            # scalar
```

The concatenation order is fixed: drug global, pocket global, drug attended pool, pocket attended pool, TDA compressed.

## Biology / Chemistry Context

Cross-attention is a natural inductive bias for drug-target interaction because binding is intrinsically *local-to-local*: a specific drug atom (the carbonyl oxygen of an amide, say) interacts with a specific pocket residue (an arginine side chain donating a hydrogen bond). Globally pooled fingerprints lose that local specificity. By letting each drug atom attend to all pocket residues, the model can recover atom-residue compatibility scores — implicitly, a learned analogue of a pharmacophore fingerprint. The symmetric pocket-to-drug direction gives residues a way to summarise which atoms they "see" most relevant.

The TDA features inject coarse-grained shape information that the message-passing layers cannot easily see in three message-passing hops: drug $H_1$ (1D persistent homology) responds to aromatic ring count and macrocycle topology; pocket $H_0$ encodes the number of residue clusters at various scales (e.g. whether the pocket has subpockets); pocket $H_1$ encodes loop topology of the binding-site outline. Compressing the 1600-dim raw persistence images down to 64 with an MLP filters out grid-discretisation noise.

The final MLP layer's small initial scale (multiplied by 0.1) is a practical safeguard. pKd values are typically in the range $4$–$12$ but the unscaled `predictor` initialization would produce predictions of unbounded magnitude at step 0 and explode the regression loss.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. *Attention Is All You Need.* NeurIPS 2017. <https://arxiv.org/abs/1706.03762>. Origin of scaled-dot-product attention.
- Bahdanau, D., Cho, K., Bengio, Y. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. <https://arxiv.org/abs/1409.0473>. Origin of cross-attention.
- Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., Teh, Y. W. *Set Transformer.* ICML 2019. <https://arxiv.org/abs/1810.00825>. Attention over unordered sets, conceptually similar to cross-attending atoms and residues.
- Jiang, M., Li, Z., Bian, Y., et al. *Drug–target affinity prediction using graph neural network and contact maps.* RSC Adv. 10, 20701 (2020). A representative cross-modal DTI architecture.
- Adams, H., Emerson, T., Kirby, M., et al. *Persistence Images: A Stable Vector Representation of Persistent Homology.* JMLR 18, 1–35 (2017). The vectorisation used to feed TDA into the MLP.
