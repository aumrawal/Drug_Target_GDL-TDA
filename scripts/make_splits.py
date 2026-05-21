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
        "INDEX_*.lst",
        "INDEX_*.txt",
        "INDEX_*",
        "*.lst",
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


def _pkd_after_year(cols: list[str]) -> "float | None":
    """
    Scan a list of string tokens and return the first float that comes
    after a 4-digit release-year token (1970-2030).
    Falls back to returning the first float in the list if no year is found.
    """
    year_found = False
    first_float = None
    for val in cols:
        try:
            v = float(val)
            if first_float is None:
                first_float = v
            if year_found:
                return v          # first float after the year → pKd
        except ValueError:
            pass
        if not year_found:
            try:
                yr = int(val)
                if 1970 <= yr <= 2030:
                    year_found = True
            except ValueError:
                pass
    return first_float            # best-effort if no year column found


def parse_index_file(index_path: str) -> list[dict]:
    """
    Parse a PDBBind index/affinity file.

    Supports:
      • Classic PDBbind whitespace format  (col0=PDB, col2=year, col3=pKd)
      • Tab-separated without headers
      • CSV with named headers (pdb/id col + pkd/affinity col auto-detected)
      • CSV without headers (col0=PDB ID, pKd found after year column)

    Format is detected from the FIRST NON-COMMENT data line, not from the
    file extension or the first line (which is usually a comment block).
    """
    import csv as csv_mod

    entries = []
    skipped = 0

    with open(index_path, "r", errors="replace") as fh:
        raw = fh.read()

    # Find first non-comment data line — use THIS to detect the delimiter.
    first_data = next(
        (l.strip() for l in raw.splitlines()
         if l.strip() and not l.strip().startswith("#")),
        ""
    )
    print(f"  First data line: {first_data[:120]}")

    has_comma = "," in first_data
    has_tab   = "\t" in first_data

    # ── CSV / TSV path ────────────────────────────────────────────────────
    if has_comma or has_tab or index_path.endswith(".csv"):
        delim = "," if (has_comma or index_path.endswith(".csv")) else "\t"

        # Strip comment lines, then try DictReader to detect named headers.
        data_lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
        reader  = csv_mod.DictReader(data_lines, delimiter=delim)
        headers = list(reader.fieldnames or [])
        print(f"  Delimiter={delim!r}, candidate headers: {headers}")

        id_col  = _find_id_col(headers)
        aff_col = _find_affinity_col(headers)

        if id_col is not None and aff_col is not None:
            id_name  = headers[id_col]
            aff_name = headers[aff_col]
            print(f"  Using id='{id_name}', affinity='{aff_name}'")
            for row in reader:
                pdbid = row[id_name].strip().lower()
                try:
                    pkd = float(row[aff_name].strip())
                except (ValueError, KeyError):
                    skipped += 1
                    continue
                entries.append({"id": pdbid, "affinity": pkd})
        else:
            # No recognisable headers — parse manually:
            # col 0 = PDB ID, pKd found after a year token.
            print("  No named headers found; parsing col[0]=PDB ID, pKd by year column.")
            for line in data_lines:
                line = line.strip()
                if not line:
                    continue
                cols = [c.strip() for c in line.split(delim)]
                if not cols or len(cols[0]) != 4:
                    skipped += 1
                    continue
                pkd = _pkd_after_year(cols[1:])
                if pkd is None:
                    skipped += 1
                    continue
                entries.append({"id": cols[0].lower(), "affinity": pkd})

    # ── Whitespace path ───────────────────────────────────────────────────
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 2 or len(cols[0]) != 4:
                skipped += 1
                continue
            pkd = _pkd_after_year(cols[1:])
            if pkd is None:
                skipped += 1
                continue
            entries.append({"id": cols[0].lower(), "affinity": pkd})

    if skipped:
        print(f"  [info] Skipped {skipped} malformed/unparseable lines.")

    # ── Sanity check: pKd values must be in [0, 20]. ─────────────────────
    # If the mean is >> 20 the parser grabbed raw Kd/Ki values (nM, pM, …)
    # instead of −log₁₀(Kd). Try known unit offsets until the mean lands in
    # the chemically reasonable pKd window [4, 12].
    if entries:
        import math
        mean_aff = sum(e["affinity"] for e in entries) / len(entries)
        print(f"  Parsed affinity mean = {mean_aff:.4f}")

        if mean_aff > 20:
            print(f"  WARNING: mean {mean_aff:.1f} >> 20 — looks like raw Kd/Ki, not pKd.")
            print("  Trying unit conversions  (pKd = factor − log10(value)):")
            # pKd = −log10(Kd_M) = factor − log10(Kd_unit)
            # fM→15, pM→12, nM→9, μM→6, mM→3
            conversions = [("fM", 15), ("pM", 12), ("nM", 9), ("µM", 6), ("mM", 3)]
            best = None
            for unit, factor in conversions:
                converted_mean = factor - math.log10(max(mean_aff, 1e-9))
                print(f"    {unit}: mean pKd = {converted_mean:.2f}")
                if best is None and 4.0 <= converted_mean <= 12.0:
                    best = (unit, factor, converted_mean)
            if best:
                unit, factor, new_mean = best
                print(f"  Applying: pKd = {factor} − log10(value)  [assuming {unit}]")
                for e in entries:
                    raw = max(e["affinity"], 1e-9)
                    e["affinity"] = round(factor - math.log10(raw), 4)
                print(f"  Converted mean pKd = {new_mean:.2f}")
            else:
                print("  Could not find a unit that gives a sane pKd — keeping raw values.")
                print("  Pass --index_file pointing to a file with −logKd/Ki values.")

    return entries


def verify_files(entries: list[dict], data_dir: str) -> list[dict]:
    """Keep only entries where _protein.pdb and a ligand file (.sdf or .mol2) exist."""
    valid = []
    for e in entries:
        pid = e["id"]
        pdb_path  = os.path.join(data_dir, pid, f"{pid}_protein.pdb")
        sdf_path  = os.path.join(data_dir, pid, f"{pid}_ligand.sdf")
        mol2_path = os.path.join(data_dir, pid, f"{pid}_ligand.mol2")
        if os.path.isfile(pdb_path) and (os.path.isfile(sdf_path) or os.path.isfile(mol2_path)):
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


def _entries_from_dirs(data_dir: str) -> list[dict]:
    """Fallback: build an entry list from PDB ID sub-directories with affinity=0.0."""
    entries = []
    try:
        for name in sorted(os.listdir(data_dir)):
            if len(name) == 4 and name.isalnum():
                entries.append({"id": name.lower(), "affinity": 0.0})
    except OSError:
        pass
    return entries


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
        print("WARNING: No affinity index file found — falling back to directory scan.")
        print("         Affinities will be set to 0.0. Metrics won't be meaningful")
        print("         until real pKd values are supplied via --index_file.\n")
        entries = _entries_from_dirs(data_dir)
        if not entries:
            print(
                "ERROR: data_dir contains no 4-char alphanumeric subdirectories "
                "that look like PDB IDs. Check --data_dir."
            )
            sys.exit(1)
        print(f"  Found {len(entries)} PDB ID directories in data_dir.")
        valid = verify_files(entries, data_dir)
        missing = len(entries) - len(valid)
        print(f"  Valid (protein.pdb + ligand file present): {len(valid)}")
        if missing:
            print(f"  Dropped {missing} entries with missing structure files.")
        if not valid:
            print("ERROR: No entries have _protein.pdb + _ligand.sdf/.mol2. Check data_dir.")
            sys.exit(1)
        train, val, test = split_entries(valid, seed=42)
        os.makedirs(out_dir, exist_ok=True)
        save_split(train, os.path.join(out_dir, "train_split.json"))
        save_split(val,   os.path.join(out_dir, "val_split.json"))
        save_split(test,  os.path.join(out_dir, "test_split.json"))
        print(f"\n  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
        print("Done (no affinity index — placeholder affinities used).")
        sys.exit(0)

    print(f"Found index file: {index_path}")

    # Print the first comment block AND the first 3 actual data lines.
    print("\nFirst data lines of index file:")
    with open(index_path, "r") as fh:
        data_count = 0
        for line in fh:
            stripped = line.rstrip()
            if not stripped:
                continue
            is_comment = stripped.lstrip().startswith("#")
            print(f"  {'#' if is_comment else '>'} {stripped}")
            if not is_comment:
                data_count += 1
            if data_count >= 3:
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
