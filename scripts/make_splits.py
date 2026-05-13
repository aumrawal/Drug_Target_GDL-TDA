"""
make_splits.py — Generate train/val/test split JSON files for the PDBBind dataset.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --data_dir /path/to/pdbbind --out_dir /kaggle/working/
"""

import argparse
import glob
import json
import os
import random
import sys


def find_index_file(data_dir: str) -> str | None:
    """Search for the PDBBind index file in data_dir using several glob patterns."""
    patterns = [
        "*.csv",
        "INDEX_*.txt",
        "INDEX_*",
        "*.txt",
        "*index*",
        "*INDEX*",
        "*affinity*",
        "*binding*",
        "*data*",
    ]
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(data_dir, pat)))
        matches = [m for m in matches if os.path.isfile(m)]
        if matches:
            return matches[0]
    return None


def _find_affinity_col(headers: list[str]) -> int | None:
    """Return the index of the affinity column given a list of CSV header names."""
    affinity_keywords = [
        "pkd", "pki", "affinity", "-logkd", "-logki", "logkd", "logki",
        "binding_affinity", "log_affinity", "neg_log", "pchembl",
        "-log(kd/ki)", "-log(kd", "log(1/kd",
    ]
    for i, h in enumerate(headers):
        if any(kw in h.lower().replace(" ", "") for kw in affinity_keywords):
            return i
    return None


def _find_id_col(headers: list[str]) -> int | None:
    """Return the index of the PDB ID column given a list of CSV header names."""
    id_keywords = ["pdb", "pdbid", "pdb_id", "code", "id", "complex"]
    for i, h in enumerate(headers):
        if any(kw in h.lower().replace(" ", "_") for kw in id_keywords):
            return i
    return None


def parse_index_file(index_path: str) -> list[dict]:
    """
    Parse a PDBBind index/affinity file — handles both CSV and whitespace formats.

    CSV format (with headers):
        PDB ID column is auto-detected by name (pdb, pdbid, code, id, ...)
        Affinity column is auto-detected by name (pkd, affinity, -logKd, ...)

    Classic whitespace format:
        1a1e    2.00  2003  6.92  Kd=1.20nM  ...
        col 0 = PDB ID,  col 3 = pKd float
    """
    import csv as csv_mod

    entries = []
    skipped = 0

    with open(index_path, "r") as fh:
        raw = fh.read()

    is_csv = index_path.endswith(".csv") or ("," in raw.split("\n")[0])

    if is_csv:
        reader = csv_mod.DictReader(raw.splitlines())
        headers = reader.fieldnames or []
        print(f"  CSV headers detected: {list(headers)}")

        id_col  = _find_id_col(list(headers))
        aff_col = _find_affinity_col(list(headers))

        if id_col is None or aff_col is None:
            # Fall back: col 0 = id, try every column for a float affinity
            print("  [warn] Could not auto-detect id/affinity columns by name.")
            print("  Falling back: column 0 = PDB ID, scanning for first numeric column.")
            for row in csv_mod.reader(raw.splitlines()):
                if not row or row[0].startswith("#"):
                    continue
                pdbid = row[0].strip().lower()
                # find first numeric value after col 0
                pkd = None
                for val in row[1:]:
                    try:
                        pkd = float(val.strip())
                        break
                    except ValueError:
                        continue
                if pkd is None:
                    skipped += 1
                    continue
                entries.append({"id": pdbid, "affinity": pkd})
        else:
            id_name  = list(headers)[id_col]
            aff_name = list(headers)[aff_col]
            print(f"  Using id column='{id_name}', affinity column='{aff_name}'")
            for row in reader:
                pdbid = row[id_name].strip().lower()
                try:
                    pkd = float(row[aff_name].strip())
                except (ValueError, KeyError):
                    skipped += 1
                    continue
                entries.append({"id": pdbid, "affinity": pkd})

    else:
        # Classic whitespace-separated INDEX file
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 4:
                skipped += 1
                continue
            pdbid = cols[0].lower()
            try:
                pkd = float(cols[3])
            except ValueError:
                skipped += 1
                continue
            entries.append({"id": pdbid, "affinity": pkd})

    if skipped:
        print(f"  [info] Skipped {skipped} malformed/unparseable lines.")

    return entries


def verify_files(entries: list[dict], data_dir: str) -> list[dict]:
    """Keep only entries where both _protein.pdb and _ligand.sdf exist."""
    valid = []
    for e in entries:
        pid = e["id"]
        pdb_path = os.path.join(data_dir, pid, f"{pid}_protein.pdb")
        sdf_path = os.path.join(data_dir, pid, f"{pid}_ligand.sdf")
        if os.path.isfile(pdb_path) and os.path.isfile(sdf_path):
            valid.append(e)
    return valid


def split_entries(entries: list[dict], seed: int = 42) -> tuple[list, list, list]:
    """Split into 80% train / 10% val / 10% test with a fixed random seed."""
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(round(n * 0.10)))
    n_test = max(1, int(round(n * 0.10)))
    n_train = n - n_val - n_test

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def save_split(entries: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(entries, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDBBind train/val/test split JSON files."
    )
    parser.add_argument(
        "--data_dir",
        default=(
            "/kaggle/input/datasets/madukacharles/"
            "pdbbind-protein-ligand-binding-affinity-dataset"
        ),
        help="Path to the PDBBind dataset directory (contains complex subdirs + index file).",
    )
    parser.add_argument(
        "--index_file",
        default=None,
        help=(
            "Explicit path to the PDBBind index/affinity file. "
            "If not given, the script searches data_dir automatically."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="/kaggle/working/",
        help=(
            "Directory where train_split.json, val_split.json, test_split.json are saved. "
            "On Kaggle, the input dir is read-only — use /kaggle/working/ (the default)."
        ),
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)

    print(f"data_dir : {data_dir}")
    print(f"out_dir  : {out_dir}")
    print()

    # -- Locate index file -------------------------------------------------
    index_path = args.index_file if args.index_file else find_index_file(data_dir)
    if index_path is None:
        print("ERROR: No index file found in data_dir.\n")
        print("Files and directories found at that path:")
        try:
            all_entries = sorted(os.listdir(data_dir))
            files = [e for e in all_entries if os.path.isfile(os.path.join(data_dir, e))]
            dirs  = [e for e in all_entries if os.path.isdir(os.path.join(data_dir, e))]
            if files:
                print("  Files:")
                for f in files:
                    print(f"    {f}")
            else:
                print("  (no files at top level — only subdirectories)")
            print(f"  Directories ({len(dirs)} total, first 10 shown):")
            for d in dirs[:10]:
                print(f"    {d}/")
        except Exception as exc:
            print(f"  (could not list directory: {exc})")
        print(
            "\nRe-run with --index_file pointing directly to the index/affinity file, e.g.:\n"
            "  python scripts/make_splits.py --index_file /path/to/affinity_data.csv"
        )
        sys.exit(1)

    print(f"Found index file: {index_path}")

    # Print the first 3 non-empty lines (including comments) so the user can
    # verify we picked the right file.
    print("\nFirst 3 lines of index file:")
    with open(index_path, "r") as fh:
        count = 0
        for line in fh:
            stripped = line.rstrip()
            if stripped:
                print(f"  {stripped}")
                count += 1
                if count >= 3:
                    break
    print()

    # -- Parse index -------------------------------------------------------
    print("Parsing index file...")
    entries = parse_index_file(index_path)
    print(f"  Parsed {len(entries)} entries with valid pKd values.")

    if not entries:
        print(
            "ERROR: No valid entries parsed. "
            "Check that column 3 is the numeric pKd value.\n"
            "Inspect the first non-comment line above and compare with the expected format:\n"
            "  1a1e    2.00  2003  6.92  Kd=1.20nM  ..."
        )
        sys.exit(1)

    # -- Filter by file existence ------------------------------------------
    print("Checking for _protein.pdb and _ligand.sdf files...")
    valid = verify_files(entries, data_dir)
    missing = len(entries) - len(valid)
    print(f"  Valid (both files present): {len(valid)}")
    if missing:
        print(f"  Dropped {missing} entries with missing structure files.")

    if not valid:
        print(
            "ERROR: No entries have both _protein.pdb and _ligand.sdf files. "
            "Ensure data_dir contains subdirectories named by PDB ID, e.g.:\n"
            f"  {data_dir}/1a1e/1a1e_protein.pdb\n"
            f"  {data_dir}/1a1e/1a1e_ligand.sdf"
        )
        sys.exit(1)

    # -- Split -------------------------------------------------------------
    train, val, test = split_entries(valid, seed=42)

    # -- Save --------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train_split.json")
    val_path = os.path.join(out_dir, "val_split.json")
    test_path = os.path.join(out_dir, "test_split.json")

    save_split(train, train_path)
    save_split(val, val_path)
    save_split(test, test_path)

    # -- Summary -----------------------------------------------------------
    print()
    print("=" * 50)
    print("Split summary")
    print("=" * 50)
    print(f"  Total parsed      : {len(entries)}")
    print(f"  Valid (files ok)  : {len(valid)}")
    print(f"  Train             : {len(train)}  → {train_path}")
    print(f"  Val               : {len(val)}   → {val_path}")
    print(f"  Test              : {len(test)}  → {test_path}")
    print()
    print("Done. Pass --out_dir to run.py (or copy the files to data_dir) so that")
    print("dataset._load_split() can find them at:")
    print(f"  {out_dir}/{{train,val,test}}_split.json")


if __name__ == "__main__":
    main()
