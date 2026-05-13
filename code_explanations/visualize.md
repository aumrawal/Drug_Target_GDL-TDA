# `visualize.py`

## Overview

`visualize.py` is the **post-training diagnostic plotter** for TopoSurface-DTI. After a model has been trained (or, with `--train-first`, in the same invocation) it runs inference on a synthetic evaluation set and produces a 4-panel figure that summarises model quality from four complementary angles:

1. **Scatter** — predicted vs actual pKd, points coloured by absolute error, with the identity line $y=x$, an $\pm$RMSE band, and a least-squares fit.
2. **Residuals** — histogram of $r_i = \hat y_i - y_i$ with a normal fit $\mathcal{N}(\mu, \sigma^2)$ overlay.
3. **Ranking** — compounds sorted by actual pKd, with the predicted curve overlaid (the view that matters for virtual screening).
4. **CDF** — empirical cumulative distribution of $|r_i|$, with the fraction of predictions within $\pm 0.5,\pm 1.0,\pm 1.5,\pm 2.0$ pKd annotated.

The script re-uses `collate_single` and `forward_step` from `train.trainer`, which keeps inference numerically identical to validation and avoids subtle bugs from independently reimplementing the model call.

Outputs a PNG (`predictions_vs_actual.png` by default) at 150 DPI and prints a tabular metrics summary to stdout.

## Mathematical Foundations

Let the evaluation set be $\{(x_i, y_i)\}_{i=1}^{N}$ with $N$ set by `--n-samples` (default 200). The model produces predictions $\hat y_i = f_\theta(x_i)$ with $\theta$ loaded from `checkpoints/best_model.pt`.

### Residuals and bias

$$
r_i \;=\; \hat y_i - y_i, \qquad
\mu \;=\; \bar r \;=\; \frac{1}{N}\sum_i r_i, \qquad
\sigma \;=\; \sqrt{\frac{1}{N}\sum_i (r_i - \mu)^2}.
$$

`compute_metrics` (lines 72–83) computes $\mu$ (`bias`) and $\sigma$ (`std_err`). A non-zero $\mu$ means the model is *systematically* over- or under-predicting affinity; a small $\sigma$ relative to the pKd dynamic range indicates a sharp residual distribution.

### Standard regression metrics

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i r_i^2}, \qquad
\mathrm{MAE} = \frac{1}{N}\sum_i |r_i|.
$$

Both are reported in pKd units. RMSE penalises large errors quadratically; MAE is the median-robust complement.

### Coefficient of determination $R^2$

Although not explicitly printed, the linear regression line (line 109) computes the same quantity via `scipy.stats.linregress`. The coefficient of determination is defined as

$$
R^2 \;=\; 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}}, \qquad
\mathrm{SS}_{\mathrm{res}} = \sum_i (y_i - \hat y_i)^2, \qquad
\mathrm{SS}_{\mathrm{tot}} = \sum_i (y_i - \bar y)^2,
$$

where $\bar y$ is the mean of the targets. It quantifies the fraction of target variance the model explains. For an unbiased linear regression of $\hat y$ on $y$, $R^2$ equals the squared Pearson correlation. A perfect model achieves $R^2 = 1$; the constant-mean baseline achieves $R^2 = 0$; arbitrarily bad models can achieve $R^2 < 0$.

### Pearson and Spearman correlations

$$
R \;=\; \frac{\sum_i (\hat y_i - \bar{\hat y})(y_i - \bar y)}{\sqrt{\sum_i (\hat y_i - \bar{\hat y})^2}\sqrt{\sum_i (y_i - \bar y)^2}}.
$$

Spearman $\rho$ is Pearson $R$ computed on the ranks $\operatorname{rank}(\hat y_i)$ and $\operatorname{rank}(y_i)$ — invariant to any monotone transformation, so it isolates ranking quality from absolute calibration. In the scatter panel both are shown in a corner box.

### The identity line and the $\pm$RMSE band

The reference geometry plotted on Panel 1:

- **Identity line** $\hat y = y$: a perfect model lies on this line. Distance from a point to this line equals $|r_i|/\sqrt{2}$.
- **$\pm$RMSE band** $\{(\hat y, y) : |\hat y - y| \le \mathrm{RMSE}\}$: a translucent strip of width $2\,\mathrm{RMSE}$ along the identity. Under a Gaussian residual model with $\mu = 0$, this band contains approximately $\Pr(|r| \le \sigma) \approx 68\%$ of the points.
- **Linear fit** $\hat y = a y + b$ from `scipy.stats.linregress`: deviation of the slope $a$ from 1 quantifies *regression dilution* — the well-known shrinkage of predictions toward the mean caused by noisy features.

### Normal fit on residuals

Panel 2 overlays the maximum-likelihood Gaussian density

$$
p(r;\mu,\sigma) \;=\; \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(r-\mu)^2}{2\sigma^2}\right)
$$

on top of the histogram. Visible departures from Gaussianity — heavy tails, skew, bimodality — flag systematic failure modes that the scalar metrics hide. The histogram is plotted with `density=True` so it integrates to 1 and can be directly compared to the analytical PDF. There is no kernel density estimation; the implementation uses a vanilla histogram (bin count $\max(10, N/8)$, line 136) with a parametric Gaussian overlay.

### Empirical CDF of absolute errors

Panel 4 plots the **empirical cumulative distribution function** of $|r|$:

$$
\hat F_{|r|}(t) \;=\; \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[|r_i| \le t\right].
$$

`np.sort(abs_errors)` produces the order statistics $|r|_{(1)} \le |r|_{(2)} \le \cdots \le |r|_{(N)}$, and the CDF at $|r|_{(k)}$ is $k/N$ — exactly the construction in lines 196–197. Vertical reference lines mark $t \in \{0.5, 1.0, 1.5, 2.0\}$, the thresholds that medicinal chemists tend to care about (a $\pm 1$ pKd error band is roughly the experimental reproducibility floor between independent assays of the same complex).

### Ranking view

Panel 3 sorts compounds by actual pKd, $\pi$ being the permutation that achieves $y_{\pi(1)} \le y_{\pi(2)} \le \cdots \le y_{\pi(N)}$. It then plots $i \mapsto y_{\pi(i)}$ and $i \mapsto \hat y_{\pi(i)}$ on the same axis. A perfectly-ranking model produces a monotone-increasing predicted curve. The shaded fill colours the regions where $\hat y > y$ (over-prediction) versus $\hat y < y$ (under-prediction), making asymmetric bias visually obvious. The implicit metric being visualised is closely related to the concordance index

$$
\mathrm{CI} \;=\; \frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}} \mathbf{1}\!\left[\hat y_i > \hat y_j\right], \qquad
\mathcal{P} = \{(i,j) : y_i > y_j\},
$$

i.e., the empirical probability that for two randomly chosen compounds the model agrees with the experiment about which is the tighter binder.

## Code Walk-through

### Imports and path bootstrap (lines 23–37)

Standard `sys.path.insert` plus imports of `DTIDataset`, `tda_to_tensor`, `TopoSurfaceDTI`, and the three reused trainer helpers (`collate_single`, `forward_step`, `train`).

### `run_inference(model, n_samples, seed)` (lines 44–65)

Wrapped in `@torch.no_grad()` so no autograd graphs are constructed (faster, less memory). Builds a fresh `DTIDataset(use_synthetic=True, n_synthetic=n_samples, tda_resolution=model.tda_resolution)` so the persistence-image dimensionality matches what the trained model expects. Note that `model.tda_resolution` is the field the loaded model exposes — reading it back from the model rather than from the YAML defends against config drift.

The loop iterates once per sample (batch size 1), calls `forward_step`, and accumulates scalar predictions and targets into two numpy arrays of shape `(N,)`. `loss_fn = HuberLoss(delta=1.0)` is constructed only because `forward_step` requires it; the returned `_` loss is discarded.

### `compute_metrics(preds, actuals)` (lines 72–83)

Returns a dict with keys `rmse, mae, pearson_r, spearman_r, bias, std_err`. Uses `scipy.stats.pearsonr` and `scipy.stats.spearmanr` (which discard the $p$-value via tuple unpacking). All values are cast to `float` for clean printing.

### `plot_scatter(ax, preds, actuals, metrics)` (lines 90–129)

Builds Panel 1. Notable details:

- **Colourmap**: `'RdYlGn_r'` (red-yellow-green reversed) — low absolute error is green, high is red. `vmin=0, vmax=np.percentile(errors, 95)` clips the top 5% so a single outlier doesn't desaturate the rest.
- **Axes**: `(lo, hi)` extend the observed range by 0.3 in each direction so points don't sit on the spine. The identity line is dashed black; the linear fit (slope $a$, intercept $b$ from `scipy.stats.linregress`) is solid steelblue.
- **Metric box**: anchored to the lower-right corner via `ax.transAxes`, showing RMSE, MAE, Pearson $R$, Spearman $\rho$.

### `plot_residuals(ax, preds, actuals, metrics)` (lines 132–160)

Panel 2. Histogram with `density=True` plus the analytical Gaussian PDF $\mathcal{N}(\mu, \sigma^2)$ overlaid, dashed vertical at $r=0$ (no-bias reference), dotted vertical at $r=\mu$ (actual bias). A text annotation reports the fraction of residuals within $\pm 1$ pKd — a chemistry-flavoured pass/fail summary.

### `plot_ranking(ax, preds, actuals)` (lines 163–188)

Panel 3. Sorts by actual pKd and overlays predicted. `where` arguments on `fill_between` partition the gap into over- and under-prediction regions.

### `plot_cdf(ax, preds, actuals)` (lines 191–215)

Panel 4. Empirical CDF as defined above. The for-loop at line 203 walks the four thresholds and annotates each with its fraction. Y-axis formatted as percent via `FuncFormatter`.

### `make_figure(preds, actuals, save_path)` (lines 222–259)

Builds the `2x2` grid via `GridSpec` (4 panels, `hspace=0.38, wspace=0.32`), titles the figure with the sample count, calls the four panel functions, saves the PNG at 150 DPI, and prints the metric table to stdout — including `bias`, `std_err`, and the four within-band percentages.

### `main()` (lines 266–303)

1. **Argparse**: `--checkpoint`, `--n-samples`, `--train-first`, `--output`.
2. **Device cascade**: CUDA → MPS → CPU (mirror of `trainer.py`).
3. **Load or train**: if `--train-first` or the checkpoint file doesn't exist, run a short 50-epoch synthetic training using `train(cfg)`; otherwise `torch.load` the checkpoint, rebuild the model with `TopoSurfaceDTI.from_config(ckpt['cfg'])`, and load its state dict.
4. **Inference**: `run_inference(model, n_samples=...)`.
5. **Plot**: `make_figure(preds, actuals, save_path=args.output)`.

## Biology / Chemistry Context

The four-panel figure is the *standard* way to report a DTI model's quality. Each panel speaks to a different question a medicinal chemist or a reviewer will ask:

- **Scatter (Panel 1)** answers *"Is the model calibrated?"* — the regression slope should be close to 1 and the intercept close to 0. If the slope is 0.6 with a positive intercept, the model is **regressing to the mean**: tight binders are predicted too weak and weak binders predicted too strong. This is a common pathology for noisy features.
- **Residuals (Panel 2)** answers *"What is the error scale, and is it symmetric?"* If $\mu \ne 0$, the model is systematically biased — perhaps it was trained on a different pKd distribution than the evaluation set. If the residual distribution is heavy-tailed compared to the Gaussian overlay, a few outliers may be driving the RMSE.
- **Ranking (Panel 3)** answers *"Will this model triage a screening library correctly?"* In virtual screening the only operational question is whether the top-$k$ predicted compounds are enriched in true tight binders. The ranking view shows at a glance whether the model captures the high-affinity tail.
- **CDF (Panel 4)** answers *"What fraction of predictions are within an actionable error of the truth?"* A useful rule of thumb in PDBBind-class benchmarks: a state-of-the-art model gets roughly 50–60% of test compounds within $\pm 1$ pKd (about 1.36 kcal/mol — comparable to the inter-assay reproducibility floor).

A subtle but important biology point about the evaluation: this script runs on **synthetic** drug/pocket data by default (line 53). Synthetic data is great for plumbing tests but not for biological conclusions. When you swap in real PDBBind, evaluation also has to defend against **leakage**: similar ligands in train and test, or close protein homologues across splits, inflate metrics in ways that don't reflect real-world generalisation. `scripts/make_splits.py` makes only a random split, so additional clustering-based splits (sequence/structure similarity for proteins, Tanimoto for ligands) should be applied for publishing-grade evaluation.

## References

- F. R. S. Pearson. *Notes on regression and inheritance in the case of two parents.* Proc. Roy. Soc. London, 58:240–242, 1895 — origin of $R$.
- C. Spearman. *The proof and measurement of association between two things.* Am. J. Psychol., 15(1):72–101, 1904.
- D. C. Montgomery, E. A. Peck, G. G. Vining. *Introduction to Linear Regression Analysis,* 5th ed., Wiley, 2012 — derivation of $R^2$ and regression dilution.
- M. Su et al. *Comparative Assessment of Scoring Functions: The CASF-2016 Update.* J. Chem. Inf. Model. 59(2):895–913, 2019 — the standard reporting framework that the four panels above mirror.
- M. K. Gilson, J. A. Given, B. L. Bush, J. A. McCammon. *The statistical-thermodynamic basis for computation of binding affinities.* Biophys. J. 72:1047–1069, 1997 — why pKd, not $K_d$, is the right axis.
