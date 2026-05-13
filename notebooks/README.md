# Notebooks

### `01_TDA_persistent_homology.ipynb` (9 sections)

| Section | What you do interactively |
|---|---|
| 1 | Plot a ring vs chain — same atoms, different H₁ |
| 2 | Build simplicial complexes by hand, watch the ∂₁∘∂₂=0 identity |
| 3 | Animate the Vietoris-Rips filtration at 5 ε values |
| 4 | Construct ∂₁ and ∂₂ matrices explicitly, verify the boundary-of-boundary law |
| 5 | Run the lowest-1 pivot reduction algorithm step-by-step with printed pivots |
| 6 | Time Ripser vs the scratch fallback for N=20,40,80 — see the actual speedup |
| 7 | Plot barcodes + persistence diagram for a synthetic drug molecule |
| 8 | Vary σ across 4 values — see how Gaussian smoothing changes the persistence image |
| 9 | Produce all 4 TDA images (drug H₀/H₁, pocket H₀/H₁) and run a forward pass |

---

### `02_GDL_GEM_gauge_theory.ipynb` (9 sections)

| Section | What you do interactively |
|---|---|
| 1 | 3D-plot a pocket before/after 47° rotation — motivate the problem |
| 2 | Plot ρ₀, ρ₁, ρ₂ matrix entries as g varies; apply a concrete gauge transform |
| 3 | Numerically verify K(θ−g) = ρ_out(−g)·K(θ)·ρ_in(g) for all 6 irrep pairs |
| 4 | Plot all basis kernels as θ sweeps 0→360°; count GEM vs unconstrained params |
| 5 | Fit local PCA normals and reference frames on a paraboloid surface, visualize in 3D |
| 6 | Parallel transport on a sphere — demonstrate holonomy; plot pocket transporter distribution |
| 7 | Manual step-by-step GEMConv forward pass with shape annotations at each step |
| 8 | Gauge equivariance verification at 7 rotation angles — errors at float precision |
| 9 | Run PocketEncoder, PCA-colour the 64-dim per-vertex features on a 3D pocket |
