# `kaggle_run.ipynb` — Kaggle Training Notebook

`kaggle_run.ipynb` is the single-notebook entry point for training TopoSurface-DTI on real PDBBind data inside a Kaggle kernel. It replicates the work that `run.py` does on a local machine but adds Kaggle-specific bootstrap: pip-installing optional dependencies, generating split JSONs on the writable `/kaggle/working` filesystem, and producing the four-panel evaluation figure that `visualize.py` would otherwise generate offline. The notebook is organised into ten cells that fall into three logical phases: setup, training, and visualisation.

## Setup (cells 0 to 6)

**Cell 0 (markdown)** is the cover. It documents the two Kaggle datasets that must be attached before the notebook will run: the project source itself (`drug-target-gdl`) and the PDBBind binding-affinity dataset by user `madukacharles`. The notebook hard-codes these mount paths in later cells, so attaching them with different slugs would require editing those constants.

**Cell 1** pip-installs `ripser` and `pyyaml`. `ripser` is the fast C++ persistent homology backend consumed by `data/tda_features.py`; without it the code falls back to its from-scratch GF(2) boundary-matrix reducer (much slower but produces the same persistence diagrams). `pyyaml` is needed to parse `configs/base.yaml`. Both installs are quiet (`-q`).

**Cell 2** pip-installs the real-data dependencies `biopython` and `rdkit-pypi`. BioPython is used by `data/pocket_mesh.py::extract_pocket_atoms` to read Cα atoms from PDB files; RDKit is used by `data/molecule_graph.py::mol_to_graph` to parse SDF ligand files into atom-feature graphs. Both are only needed when `use_synthetic = False`.

**Cell 3** prepends the project root to `sys.path` and `os.chdir`s into it. The first guess `/kaggle/input/drug-target-gdl/Drug_target_GDL` matches the layout Kaggle creates when you upload a folder; if that path does not exist it falls back to `/kaggle/input/drug-target-gdl`. After this cell, every subsequent `from data...` or `from models...` import resolves against the project's own modules.

**Cell 4** runs `scripts/make_splits.py` as a subprocess. That script walks `DATA_DIR`, finds every directory containing both a `*_protein.pdb` and a `*_ligand.sdf`, looks up the binding-affinity label from PDBBind's index file, and writes `train_split.json`, `val_split.json`, `test_split.json` into `/kaggle/working`. The notebook reads the subprocess's stdout/stderr and raises if the return code is non-zero, so split-generation failures are caught early before model training starts.

**Cell 5** is an environment sanity check: it prints the PyTorch and NumPy versions, reports the GPU name (or "none (CPU run)"), and runs Ripser against a 10-point random cloud to confirm the C++ extension actually loaded. If this cell errors, every later TDA call would silently fall back to the Python implementation; printing `H0=10` and `H1=...` bar counts proves the fast path is live.

**Cell 6** is the forward-pass smoke test, identical in spirit to the one-liner in `CLAUDE.md`. It builds synthetic drug and pocket graphs, runs the TDA pipeline on their point clouds, constructs a fresh `TopoSurfaceDTI`, prints per-submodule parameter counts via `model.count_parameters()`, and emits one scalar pKd prediction. This catches shape mismatches and unboxed CUDA errors before the expensive training loop starts.

## Training (cell 7)

**Cell 7** is the workhorse. It loads `configs/base.yaml`, overrides `use_synthetic = False`, points `data_dir` at the writable `/kaggle/working` directory (where the split JSONs were written in cell 4), and reduces `n_epochs` to 50.

It then subclasses `DTIDataset` as `PDBBindDataset` to handle the Kaggle quirk that **structure files live in `/kaggle/input/...` (read-only) but split JSONs live in `/kaggle/working`** — the standard `DTIDataset` assumes both live in the same `data_dir`. The override only reimplements `_load_real(pdb_id)`: it reads the SDF with RDKit, calls `extract_pocket_atoms` on the PDB with the ligand SDF as the centroid hint, builds the kNN edge index via `build_knn_graph`, and precomputes tangent-frame transport via `precompute_pocket_geometry`. Any exception in either branch falls back to a synthetic placeholder so a single corrupt entry does not crash a 14-hour training run.

After dataset construction the cell wires the standard PyTorch loop: `DataLoader(batch_size=1, collate_fn=collate_single)` (TopoSurface-DTI processes one drug-pocket pair at a time because their sizes vary), `TopoSurfaceDTI.from_config(cfg).to(device)`, Adam with `ReduceLROnPlateau` scheduling, and Huber loss (`delta=1.0`) which is robust to outlier affinities. For each epoch it calls `train_epoch` and `validate` (both from `train/trainer.py`), prints loss, RMSE, and Pearson R for both splits, steps the scheduler on validation loss, and checkpoints `model.state_dict()` plus the config to `checkpoints/best_model.pt` whenever validation RMSE improves. This is the same training contract that `run.py` implements outside the notebook — the cell exists in the notebook so the entire pipeline stays self-contained on Kaggle.

## Visualisation (cells 8 and 9)

**Cell 8** runs inference over the validation set and aggregates predictions vs targets. `model.eval()` is set, gradient tracking is disabled, and `forward_step` (from `train/trainer.py`) is called once per sample. Predictions and actuals are stacked into NumPy arrays. Six summary statistics are computed: RMSE, MAE, Pearson R, Spearman ρ, mean bias, and standard deviation of residuals.

**Cell 9** renders the four-panel figure (the same layout `visualize.py` produces):

1. **Predicted vs Actual scatter** — points coloured by absolute error, with the 45° identity line, a ±RMSE band, and a linear regression fit. A text box reports RMSE / MAE / R / ρ.
2. **Residuals histogram** — densities overlaid with a Gaussian fit using the empirical bias and std; vertical lines mark zero, the bias, and ±σ. An inset reports the fraction of predictions within ±1 pKd.
3. **Ranking view** — compounds sorted by actual pKd along the x-axis with actual and predicted pKd plotted as line series. Fill colours distinguish over- and under-predictions. This is the most relevant view for virtual screening since it shows whether the model preserves the *order* of binders.
4. **Cumulative error CDF** — fraction of predictions whose absolute error is ≤ τ, with vertical reference lines at τ ∈ {0.5, 1.0, 1.5, 2.0}.

The figure is saved to `/kaggle/working/predictions_vs_actual.png` at 150 dpi and a compact metrics table is printed to stdout for easy copy-paste into reports.

## Cross-references

This notebook is the Kaggle-tailored counterpart to two scripts in the repository root: `run.py` (training entry point — see the corresponding doc for the `--config`, `--data`, `--resume`, and `--no-synthetic` CLI flags) and `visualize.py` (figure generator — same plotting code as cells 8 and 9, but operating on `checkpoints/best_model.pt` rather than the in-memory model). Reading those two scripts alongside this notebook makes the Kaggle / local symmetry obvious.
