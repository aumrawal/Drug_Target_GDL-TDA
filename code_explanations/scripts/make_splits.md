# `scripts/make_splits.py`

## Overview

`make_splits.py` is a one-shot data-preparation utility. Its job is to take a raw PDBBind dataset directory and emit the three JSON files (`train_split.json`, `val_split.json`, `test_split.json`) that `data/dataset.py` consumes at training time.

Concretely it:

1. **Finds an index/affinity file** in the user-supplied data directory by globbing a list of candidate filename patterns (`INDEX_*`, `*.csv`, `*affinity*`, …).
2. **Parses it** as either a CSV (with auto-detection of the PDB-ID and affinity columns by header name) or as the classic whitespace-delimited PDBBind INDEX format.
3. **Filters** by file existence — only keeps entries for which both `{pdbid}_protein.pdb` and `{pdbid}_ligand.sdf` are present under `data_dir/{pdbid}/`.
4. **Splits** the surviving entries 80% / 10% / 10% with a fixed random seed (`seed=42`).
5. **Writes** three JSON files to the chosen output directory.

It is designed to be tolerant: the PDBBind ecosystem has shipped many slightly different index-file formats over the years (different column names, different separators, different orderings), and the auto-detection here covers the common cases without manual edits.

## Mathematical Foundations

### Partitioning a labeled dataset

The script operates on a labeled dataset

$$
\mathcal{D} \;=\; \{(x_i, y_i)\}_{i=1}^{N},
$$

where $x_i$ is the PDB ID of a protein–ligand complex (a key from which the structural inputs are loaded later) and $y_i \in \mathbb{R}$ is the experimental pKd. The goal of splitting is to produce a three-way partition

$$
\mathcal{D} \;=\; \mathcal{D}_{\mathrm{train}}\,\sqcup\,\mathcal{D}_{\mathrm{val}}\,\sqcup\,\mathcal{D}_{\mathrm{test}}, \qquad
\mathcal{D}_a \cap \mathcal{D}_b = \emptyset \text{ for } a \ne b,
$$

with sizes approximately satisfying

$$
|\mathcal{D}_{\mathrm{train}}| : |\mathcal{D}_{\mathrm{val}}| : |\mathcal{D}_{\mathrm{test}}| \;\approx\; 0.80 : 0.10 : 0.10.
$$

`split_entries` (lines 159–173) implements this via a uniform random shuffle followed by index slicing. The seeded RNG (`random.Random(42)`) makes the split **deterministic and reproducible**: the same input dataset always yields the same three JSON files.

### Random vs stratified splitting

The implementation here is a **uniform random split**, i.e. each complex is assigned to a partition independently and with probabilities $(p_{\mathrm{train}}, p_{\mathrm{val}}, p_{\mathrm{test}}) = (0.8, 0.1, 0.1)$. Formally, with $\pi$ a uniformly random permutation of $\{1,\dots,N\}$ and $n_{\mathrm{val}} = \lceil 0.1 N \rceil$, $n_{\mathrm{test}} = \lceil 0.1 N \rceil$, $n_{\mathrm{train}} = N - n_{\mathrm{val}} - n_{\mathrm{test}}$,

$$
\mathcal{D}_{\mathrm{train}} = \{(x_{\pi(i)}, y_{\pi(i)})\}_{i=1}^{n_{\mathrm{train}}},
$$

and similarly for the val/test prefixes of the permuted sequence.

A **stratified** split would instead bin the pKd axis into $K$ quantile bins $\{B_1, \ldots, B_K\}$ and split *within each bin* with the same probabilities, guaranteeing that each partition contains the same pKd histogram up to discretisation. Formally one wants the per-bin proportions

$$
\frac{|\mathcal{D}_{\mathrm{train}} \cap B_k|}{|\mathcal{D}_{\mathrm{train}}|} \;\approx\; \frac{|\mathcal{D}_{\mathrm{val}} \cap B_k|}{|\mathcal{D}_{\mathrm{val}}|} \;\approx\; \frac{|\mathcal{D}_{\mathrm{test}} \cap B_k|}{|\mathcal{D}_{\mathrm{test}}|} \;\approx\; \frac{|B_k|}{N} \qquad \forall k.
$$

Stratification is particularly valuable when the label distribution is **skewed** — and pKd in PDBBind is moderately skewed, with a long tail of weak binders ($\text{pKd} \lesssim 4$) and few ultra-tight binders ($\text{pKd} \gtrsim 11$). With a plain random split and a small validation set, you can by chance end up with no examples in either tail. The current script accepts that risk in exchange for code simplicity and reproducibility; for production-grade benchmarking one would substitute a stratified or scaffold-based splitter.

### Why the 80/10/10 ratio

Variance of an evaluation metric estimated on a held-out set scales as $1/n_{\mathrm{eval}}$. For PDBBind-scale datasets ($N$ in the low thousands), $n_{\mathrm{val}} = n_{\mathrm{test}} \approx 0.1 N$ gives a few hundred test compounds, enough to estimate RMSE to roughly $\pm 0.05$ pKd standard error — i.e., well below the model's expected RMSE of $\sim 1.3$. The remaining 80% maximises the training signal.

### Why a fixed seed

Reproducibility. A fixed seed makes the split a deterministic function of $\mathcal{D}$, so two researchers running `python make_splits.py` on the same PDBBind release will get bitwise-identical JSON files. Any subsequent metric differences then unambiguously come from the model, not from the split.

## Code Walk-through

### `find_index_file(data_dir)` (lines 17–35)

Globs `data_dir` against a priority-ordered list of patterns (`*.csv`, `INDEX_*.txt`, `INDEX_*`, `*.txt`, `*index*`, `*INDEX*`, `*affinity*`, `*binding*`, `*data*`), returning the first hit. Returns `None` if nothing matches.

### `_find_affinity_col` and `_find_id_col` (lines 38–57)

Column-name heuristics for CSV headers. Lowercases the header, strips spaces (or replaces them with underscores), and checks for any keyword in a hand-curated list:

- Affinity keywords: `pkd, pki, affinity, -logkd, -logki, logkd, logki, binding_affinity, log_affinity, neg_log, pchembl, -log(kd/ki), -log(kd, log(1/kd`.
- ID keywords: `pdb, pdbid, pdb_id, code, id, complex`.

The first matching column index is returned.

### `parse_index_file(index_path)` (lines 60–144)

The heart of the file. Reads the whole index file into memory, then branches on whether it is CSV or whitespace-delimited (detected by file extension or presence of a comma in the first line).

**CSV branch** (lines 82–121):

- Uses `csv.DictReader`.
- Calls the two header detectors. If either fails, falls back to "column 0 = ID, first numeric column = affinity" with a printed warning.
- For each successfully-parsed row, appends `{"id": pdbid.lower(), "affinity": float(pkd)}` to `entries`. Rows that fail to parse are counted in `skipped`.

**Whitespace branch** (lines 124–139):

- Splits each line on whitespace and assumes column 0 is the PDB ID and **column 3** is the float pKd — this matches the canonical PDBBind INDEX_general_PL.year format
  ```
  1a1e    2.00  2003  6.92  Kd=1.20nM  ...
  ```
- Lines starting with `#` or with fewer than 4 columns are skipped.

Both branches print a count of skipped lines to help diagnose format issues.

### `verify_files(entries, data_dir)` (lines 147–156)

Walks `entries`, retaining only those for which both `data_dir/{pid}/{pid}_protein.pdb` and `data_dir/{pid}/{pid}_ligand.sdf` exist on disk. This is the "matching file set" filter — the index file may reference more complexes than were actually downloaded.

### `split_entries(entries, seed=42)` (lines 159–173)

```python
rng = random.Random(seed)
rng.shuffle(shuffled)

n = len(shuffled)
n_val  = max(1, int(round(n * 0.10)))
n_test = max(1, int(round(n * 0.10)))
n_train = n - n_val - n_test
```

`max(1, ...)` guards against tiny datasets (so val/test never collapse to zero). The split is then three contiguous slices of the shuffled list, in the order train → val → test. Returns the three lists.

### `save_split(entries, path)` (lines 176–178)

Two-line `json.dump` with `indent=2` so the output files are human-readable diffs.

### `main()` (lines 181–319)

CLI orchestration:

1. Parse `--data_dir`, `--index_file`, `--out_dir` (default Kaggle-friendly paths).
2. Locate index file (use `--index_file` if given, else search).
3. If no index file is found, print a helpful diagnostic listing the top-level contents of `data_dir` and exit.
4. Echo the first three non-empty lines of the index file so the user can sanity-check the parser will see what they expect.
5. Parse entries; if empty, abort with a usage hint.
6. Filter by file existence; print dropped count.
7. Split and save the three JSON files.
8. Print a final summary table with counts and output paths.

The exit-on-error pattern uses `sys.exit(1)` consistently so the script behaves correctly under shell pipelines.

## Biology / Chemistry Context

### What the index file contains

The PDBBind INDEX files annotate each entry with the PDB ID, the resolution, the year, the negative log of the affinity, and a free-text affinity record (e.g. `Kd=1.20nM`). The 4th column is what the model trains against — a $-\log_{10}$ transform of the binding constant. Whether the source measurement is $K_d$, $K_i$, or $\mathrm{IC}_{50}$ is encoded in the trailing string but not used downstream; all are pooled onto the pKd axis.

### Why filter by file existence

PDBBind ships affinity annotations for thousands more complexes than any single downloader retrieves. Some entries are flagged "general set" but only the "refined" or "core" subsets ship structural files. Filtering eliminates the silent failure mode where the dataset loader hits a missing PDB at training time.

### Evaluation fairness in drug discovery

The basic biology concern with random splits in DTI is **data leakage** by chemical or biological similarity. Two kinds:

- **Ligand similarity leakage.** PDBBind contains many congeneric series — molecules differing by a single substituent. If one analogue lands in train and a near-identical analogue in test, the test score reflects interpolation, not generalisation. Industry-standard remedy: split by Murcko **scaffold** so an entire scaffold class is in exactly one partition. Equivalent stricter remedy: Tanimoto-based clustering with similarity threshold $\le 0.4$.
- **Protein similarity leakage.** If close homologues (>30% sequence identity) appear in both train and test, the model can memorise the binding pocket without learning the chemistry. Remedy: split by sequence identity clustering (e.g. with CD-HIT at 30% identity).

The current `make_splits.py` performs neither; it is a vanilla random split. Re-running with a scaffold or sequence-cluster splitter is the recommended next step before reporting publishable numbers. For internal R&D and synthetic-data smoke tests, the random split is fine.

### Unit considerations

The fourth INDEX column is dimensionless ($-\log_{10}$ of a quantity in M), but it is implicitly defined with $K$ in molar. So pKd = 6 means $K = 10^{-6}$ M = 1 μM (a typical fragment hit); pKd = 9 means $K = 10^{-9}$ M = 1 nM (a drug-like lead).

## References

- R. Wang, X. Fang, Y. Lu, S. Wang. *The PDBbind Database: Collection of Binding Affinities for Protein–Ligand Complexes with Known Three-Dimensional Structures.* J. Med. Chem. 47(12):2977–2980, 2004.
- Z. Liu et al. *Forging the basis for developing protein–ligand interaction scoring functions: the PDBbind database.* Acc. Chem. Res. 50(2):302–309, 2017.
- G. W. Bemis and M. A. Murcko. *The properties of known drugs. 1. Molecular frameworks.* J. Med. Chem. 39(15):2887–2893, 1996 — the scaffold-based splitting approach.
- W. Li and A. Godzik. *Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences.* Bioinformatics 22(13):1658–1659, 2006.
- C. Bishop. *Pattern Recognition and Machine Learning,* Springer, 2006 — train/val/test partitioning, generalisation theory.
- S. Kearnes et al. *Molecular Graph Convolutions: Moving Beyond Fingerprints.* J. Comput. Aided Mol. Des. 30:595–608, 2016 — the leakage discussion for ligand-based DTI.
