# `train/trainer.py`

## Overview

`trainer.py` is the engine room of TopoSurface-DTI. It defines the data-iteration logic, the per-sample forward pass, the per-epoch train and validate loops, and the top-level `train(cfg, resume)` function that wires together the dataset, the model, the optimizer, and a learning-rate scheduler into a complete training run.

Concretely it:

1. Constructs `DTIDataset` objects for the `train` and `val` splits — either synthetic placeholders or real PDBBind complexes — and wraps them in `DataLoader`s with `batch_size=1` because each drug–pocket pair has a different number of atoms / residues.
2. Builds a `TopoSurfaceDTI` model via `TopoSurfaceDTI.from_config(cfg)` and moves it to the best available device (CUDA → MPS → CPU).
3. Optimizes the model parameters $\theta$ with **Adam** under a **Huber** regression loss against the pKd target, monitoring **RMSE** and **Pearson $R$** every epoch.
4. Halves the learning rate when validation loss plateaus (`ReduceLROnPlateau`, patience 10), clips gradients to unit norm, and checkpoints the model that achieves the lowest validation RMSE so far.

The file is the contract that `run.py` (training entry point) and `visualize.py` (re-uses `forward_step` and `collate_single` for inference) both depend on.

## Mathematical Foundations

### The regression target — pKd

The model predicts the binding affinity of a drug for its target on the **pKd scale**:

$$
\text{pKd} \;=\; -\log_{10} K_d,
$$

where $K_d$ is the equilibrium dissociation constant of the protein–ligand complex measured in moles per litre (M). pKd is meaningful as a regression target because of the underlying thermodynamics. At equilibrium and constant temperature, the Gibbs free energy of binding is

$$
\Delta G \;=\; -RT\ln K_d,
$$

with $R = 8.314\ \mathrm{J\,K^{-1}mol^{-1}}$ and $T$ the absolute temperature. Combining,

$$
\text{pKd} \;=\; -\log_{10} K_d \;=\; \frac{-\Delta G}{RT\ln 10} \;=\; \frac{|\Delta G|}{RT\ln 10}.
$$

So pKd is **linear in free energy**. A model that is unbiased in pKd is unbiased in $\Delta G$, which is the chemically additive quantity (free-energy contributions from individual interactions sum). On the linear $K_d$ scale, a typical PDBBind range of $10^{-12}$ M (sub-pM) to $10^{-2}$ M (10 mM) spans ten orders of magnitude — predicting it directly would be numerically pathological and would heavily over-weight strong binders. Predicting pKd ($\approx$ 2–12) gives a well-conditioned regression problem.

### The Huber loss

`trainer.py` uses `nn.HuberLoss(delta=1.0)` rather than pure MSE. For prediction $\hat y$ and target $y$ with $\delta = 1$,

$$
\ell_\delta(\hat y, y) \;=\;
\begin{cases}
\tfrac{1}{2}(\hat y - y)^2 & \text{if } |\hat y - y| \le \delta, \\[4pt]
\delta\,|\hat y - y| - \tfrac{1}{2}\delta^{2} & \text{otherwise}.
\end{cases}
$$

Inside the quadratic region (within $\pm 1$ pKd of truth) Huber agrees with the **mean squared error**

$$
\mathcal{L}_{\mathrm{MSE}} \;=\; \frac{1}{N}\sum_{i=1}^{N}(\hat y_i - y_i)^2,
$$

so locally the gradient is the familiar $\partial \ell / \partial \hat y = \hat y - y$. Outside the region the loss is linear, giving the **mean absolute error** gradient $\pm\delta$. This is the standard recipe for robust regression: PDBBind affinity measurements are aggregated across many assays and units, so a handful of outlier labels with several-pKd error should not dominate the gradient. Empirically Huber tracks MSE in performance but is far less sensitive to label noise.

The total empirical risk minimised by `train_epoch` is therefore

$$
\hat{\mathcal{R}}(\theta) \;=\; \frac{1}{N}\sum_{i=1}^{N} \ell_\delta\!\left(f_\theta(x_i),\, y_i\right),
$$

with $f_\theta$ the TopoSurface-DTI model and $x_i$ the bundle of tensors `(drug_x, drug_pos, drug_edge, pocket_x, pocket_edge, pocket_angles, pocket_trans, drug_tda, pocket_tda)`.

### Adam optimizer

Adam (Kingma & Ba, 2014) maintains exponential moving averages of the first and second gradient moments. With $g_t = \nabla_\theta \hat{\mathcal{R}}(\theta_{t-1})$, default hyperparameters $\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$, and learning rate $\eta = 10^{-3}$ (the default `cfg['lr']`):

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t, \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t \odot g_t, \\
\hat m_t &= \frac{m_t}{1 - \beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^{\,t}}, \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
\end{aligned}
$$

The bias-correction terms $1 - \beta_k^{\,t}$ compensate for the fact that $m_0 = v_0 = 0$, which would otherwise bias $m_t$ and $v_t$ towards zero in early steps. The per-parameter adaptive step $\eta / (\sqrt{\hat v_t} + \epsilon)$ is what makes Adam robust to the wildly different gradient scales across the SchNet drug encoder, the gauge-equivariant pocket encoder, the TDA MLP, and the cross-attention fusion head.

### Gradient clipping

Before each `optimizer.step()` the code applies

$$
g \;\leftarrow\; g \cdot \min\!\left(1,\, \frac{c}{\lVert g \rVert_2}\right), \qquad c = 1.0,
$$

via `nn.utils.clip_grad_norm_`. This bounds the global $\ell_2$ norm of the parameter gradient at $c$, which prevents the rare exploding step in early epochs (especially common when persistence-image features dominate the input scale).

### Learning-rate schedule

`ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-5)` monitors the validation loss. After 10 consecutive epochs of no improvement,

$$
\eta \;\leftarrow\; \max(\eta / 2,\ \eta_{\min}), \qquad \eta_{\min} = 10^{-5},
$$

which effectively gives a piecewise-constant decay schedule keyed to the validation curve rather than a fixed epoch count.

### Evaluation metrics

For per-epoch reporting the trainer computes

- **RMSE**:
  $$\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat y_i - y_i)^2}.$$
- **Pearson correlation $R$**:
  $$R \;=\; \frac{\sum_i (\hat y_i - \bar{\hat y})(y_i - \bar y)}{\sqrt{\sum_i (\hat y_i - \bar{\hat y})^2}\,\sqrt{\sum_i (y_i - \bar y)^2}},$$
  the cosine similarity of the mean-centred prediction and target vectors. Pearson $R$ is the most informative single number for **virtual screening**: it measures how well the model ranks compounds, which is what dictates whether the right molecules get prioritized for wet-lab follow-up.

A clamp of $10^{-8}$ on the denominator in `pearson_r` (line 43) prevents division by zero on degenerate (constant) predictions.

The complementary metrics tracked in `visualize.py` — MAE $= \tfrac{1}{N}\sum|\hat y_i - y_i|$, Spearman $\rho$ (Pearson $R$ on the ranks), and the concordance index $\mathrm{CI} = \Pr[\hat y_i > \hat y_j \mid y_i > y_j]$ — round out the regression diagnostic. CI is closest in spirit to "how often does the model agree with the experimentalist about which of two compounds binds harder?".

### Train/val split and "early stopping"

There is no formal `EarlyStopping` callback. Instead the trainer implements the equivalent **best-checkpoint-on-validation-RMSE** policy (lines 186–195): whenever $\mathrm{RMSE}_{\mathrm{val}}^{(t)} < \min_{\tau < t} \mathrm{RMSE}_{\mathrm{val}}^{(\tau)}$, the current weights are saved to `checkpoints/best_model.pt`. The training loop itself always runs for the full `n_epochs` (default 100), but at any point you can stop and load the best checkpoint — it is exactly the parameter set that minimised validation RMSE.

## Code Walk-through

### Imports and metadata (lines 1–23)

Module docstring records the benchmark expectations for PDBBind v2020:

- RMSE $\approx 1.3$–$1.5$ kcal/mol,
- Pearson $R \approx 0.75$–$0.82$.

(Note: although the docstring says "kcal/mol", the actual unit is pKd; the conversion is $\Delta G = -RT\ln 10 \cdot \text{pKd} \approx -1.364\ \text{kcal/mol} \cdot \text{pKd}$ at 298 K — so RMSE $= 1.3$ in pKd corresponds to about $1.8$ kcal/mol on the free-energy scale.)

### `collate_single(batch)` — lines 30–32

```python
def collate_single(batch):
    return batch[0]
```

PyTorch's default collate stacks tensors along a new batch axis, but drug graphs and pocket graphs have **variable $V$** (number of atoms / residues). Stacking is impossible without padding, and the GEM layers and TDA pipeline both assume per-graph dimensionality. So `batch_size=1` is used and the collate function just unwraps the singleton list. Shape: input is a Python list of length 1 containing a sample dict; output is that dict unchanged.

### `pearson_r` / `rmse` — lines 39–47

Pure-PyTorch implementations operating on stacked prediction/target tensors of shape `(N,)`. `pearson_r` centres both vectors then computes their cosine similarity via the dot-product / norm-product form. The `clamp(min=1e-8)` on the denominator handles the zero-variance degenerate case. Both return Python floats.

### `forward_step(model, sample, device, loss_fn)` — lines 54–70

Single source of truth for the model call signature. It moves every tensor in the sample dict onto `device` and invokes the model with named keyword arguments:

| Argument        | Tensor                | Shape          |
|-----------------|----------------------|----------------|
| `drug_x`        | atom features         | `(V_d, 17)`    |
| `drug_pos`      | 3D coordinates        | `(V_d, 3)`     |
| `drug_edge`     | bond list             | `(2, E_d)`     |
| `pocket_x`      | residue features      | `(V_p, 25)`    |
| `pocket_edge`   | kNN edges             | `(2, E_p)`     |
| `pocket_angles` | local-frame edge angles | `(E_p,)`     |
| `pocket_trans`  | parallel-transport rotations | `(E_p,)` |
| `drug_tda`      | drug persistence images | `(1600,)` (2 × 20²) |
| `pocket_tda`    | pocket persistence images | `(1600,)`   |

The scalar prediction `pred` (shape `()`) is paired with the scalar target `affinity` (shape `()`) — each unsqueezed to `(1,)` before being passed to `loss_fn` because `HuberLoss` expects at least one batch dimension. Returns `(pred, loss)`.

### `train_epoch` — lines 77–94

The optimization inner loop. For each sample:

1. `optimizer.zero_grad()` — zero the accumulated gradient buffer for $\theta$.
2. `pred, loss = forward_step(...)` — compute $\hat y$ and $\ell_\delta(\hat y, y)$.
3. `loss.backward()` — populate `param.grad` with $\partial \ell / \partial \theta$ via reverse-mode autodiff.
4. `clip_grad_norm_(..., 1.0)` — enforce $\lVert \nabla_\theta \ell \rVert_2 \le 1$.
5. `optimizer.step()` — apply the Adam update equations from above; in particular the parameter update is
   $$\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

After the loop it stacks predictions (shape `(N,)`) and targets (shape `(N,)`) and returns `{'loss', 'rmse', 'r'}`.

### `validate` — lines 97–110

Identical structure but wrapped in `@torch.no_grad()` and `model.eval()`. Crucially, no `loss.backward()` or `optimizer.step()` — only the metrics are computed. `model.eval()` switches any dropout / batch-norm to inference mode (the current model has neither but the convention is important for future extensions).

### `train(cfg, resume)` — lines 117–198

The end-to-end driver.

1. **Device selection (lines 118–123).** Cascade CUDA → MPS → CPU. MPS is Apple-silicon's Metal backend; PDBBind-scale runs are feasible on an M-series chip.
2. **Datasets and loaders (lines 125–142).** Creates `DTIDataset` for `train` and `val`. `use_synthetic=True` returns 80 random graphs + persistence images per epoch; `use_synthetic=False` reads real PDBBind structures. Loaders use `batch_size=1` and the `collate_single` function from above. `shuffle=True` on train only — standard practice to decorrelate gradient updates.
3. **Model construction (lines 144–149).** `TopoSurfaceDTI.from_config(cfg)` builds the full ~250k-parameter network. `model.count_parameters()` reports the per-submodule split (drug, pocket, fusion, total) and is printed at startup — useful for sanity-checking config edits.
4. **Optimizer, scheduler, loss (lines 151–159).** Adam with `lr=1e-3, weight_decay=0`; `ReduceLROnPlateau` with factor 0.5, patience 10, floor $10^{-5}$; `HuberLoss(delta=1.0)`.
5. **Resume logic (lines 164–171).** If a checkpoint path is supplied and exists, restore weights, optimizer state, scheduler state, the epoch counter, and the best-so-far validation RMSE. This makes training **idempotent**: pausing and resuming yields the same trajectory as a single uninterrupted run.
6. **Epoch loop (lines 175–195).** Calls `train_epoch`, then `validate`, then `scheduler.step(va['loss'])` so the LR halving is keyed to validation loss. The print line formats loss, RMSE, and Pearson $R$ for both splits — at a glance you can see whether the model is overfitting (training $R$ rises while validation $R$ plateaus or drops). The best-checkpoint save bundles `epoch`, `model`, `optimizer`, `scheduler`, `best_val_rmse`, and the full `cfg` — enough to fully reproduce the run.

## Biology / Chemistry Context

Binding affinity quantifies how tightly a small-molecule drug binds to its protein target. The most fundamental measurement is the **dissociation constant $K_d$**, defined operationally for the equilibrium

$$
\mathrm{P} + \mathrm{L} \;\rightleftharpoons\; \mathrm{PL}, \qquad K_d = \frac{[\mathrm{P}][\mathrm{L}]}{[\mathrm{PL}]},
$$

with units of molarity (M). Strong binders (tight, "drug-like") sit at $K_d$ from nanomolar ($10^{-9}$ M) to picomolar ($10^{-12}$ M); weak binders are micromolar ($10^{-6}$ M) or worse. PDBBind annotations span roughly **$\text{pKd} \in [2, 12]$**, i.e. millimolar to picomolar.

You will also encounter:

- **$K_i$** — the *inhibition constant* from competition assays. Under standard Cheng–Prusoff conditions, $K_i \approx K_d$ for purely competitive inhibitors.
- **$\mathrm{IC}_{50}$** — the concentration that produces 50% inhibition. It depends on the substrate concentration; the Cheng–Prusoff conversion gives $K_i = \mathrm{IC}_{50} / (1 + [\mathrm{S}]/K_m)$ for Michaelis–Menten kinetics.

PDBBind pools all three (with the relations above) onto a single $-\log_{10}$ axis labelled "pKd/pKi". This is acceptable because, after log-transforming, the unit-conversion bias is a constant shift smaller than the experimental noise floor — well within the Huber-loss inlier region.

The crucial chemical insight that motivates pKd-space regression is that **free energies are additive** while $K_d$s are multiplicative. The Gibbs–Helmholtz relation $\Delta G = -RT\ln K_d$ means that adding a methyl group that contributes $-0.5\ \mathrm{kcal/mol}$ of binding free energy *multiplies* $K_d$ by $\exp(-0.5/RT) \approx 0.43$ at 298 K, i.e. it adds approximately $0.37$ to pKd. The pKd scale is therefore the one on which structure–activity relationships are linear — which is precisely the scale our neural network should regress on.

The reason `trainer.py` reports Pearson $R$ so prominently is that **virtual screening uses rankings, not absolute affinities**. A drug-discovery campaign typically screens millions of candidates and follows up on the top few hundred. What matters is that the model's ranking agrees with reality near the top — and that is exactly what Pearson $R$ (and Spearman $\rho$ in `visualize.py`) measure.

## References

- D. P. Kingma and J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. <https://arxiv.org/abs/1412.6980>
- P. J. Huber. *Robust Estimation of a Location Parameter.* Annals of Mathematical Statistics, 35(1):73–101, 1964.
- R. Wang, X. Fang, Y. Lu, S. Wang. *The PDBbind Database: Collection of Binding Affinities for Protein–Ligand Complexes with Known Three-Dimensional Structures.* J. Med. Chem., 47(12):2977–2980, 2004.
- T. Liu et al. *PDBbind v2020: An Updated Set of Binding Affinity Data for the Comparative Assessment of Scoring Functions.* Acc. Chem. Res. 2017.
- A. Fersht. *Structure and Mechanism in Protein Science* (W. H. Freeman, 1999) — derivation of $\Delta G = -RT\ln K_d$ and binding thermodynamics.
- Y. Cheng and W. H. Prusoff. *Relationship between the inhibition constant ($K_i$) and the concentration of inhibitor which causes 50 per cent inhibition ($\mathrm{IC}_{50}$) of an enzymatic reaction.* Biochem. Pharmacol. 22(23):3099–3108, 1973.
