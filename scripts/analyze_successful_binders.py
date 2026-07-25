#!/usr/bin/env python3
"""
Analyze successful binders from Proteina/BindProteina runs.

Features:
- Walk an input directory (or explicit CSV) and collect all scoring CSV files.
- Extract ipTM and pLDDT (or similarly named) columns.
- Print summary statistics (mean, median, std, min, max, percentiles).
- Select the top N binders (default 20) sorted by ipTM then pLDDT.
- Copy (or move) the corresponding PDB files into an output folder.
- Write ``binder_stats.csv`` (all samples) and ``top_binders.csv`` (top N only).
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_column(df: pd.DataFrame, keywords: List[str]) -> str:
    """Return the first column name that contains any of the *keywords* (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for kw in keywords:
        kw_l = kw.lower()
        for lc, orig in lower_cols.items():
            if kw_l in lc:
                return orig
    raise KeyError(f"No column matching {keywords!r} found in {list(df.columns)}")


def _load_csv(csv_path: Path) -> pd.DataFrame:
    """Load a CSV with utf-8 fallback to latin1 if needed."""
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin1")


def _gather_csv_files(root: Path) -> List[Path]:
    """Return a list of scoring summary CSV files under *root* (recursively)."""
    patterns = ["*scoring_summary.csv", "*_prescoring.csv"]
    files: List[Path] = []
    seen = set()
    for pat in patterns:
        for f in root.rglob(pat):
            # Skip macOS resource forks and duplicates
            if f.name.startswith("._"):
                continue
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(f)
    return sorted(files)


def _is_truthy(val) -> bool:
    """Interpret a CSV cell as a boolean pass/fail flag.

    Handles native bools, the strings written by ``DataFrame.to_csv`` ("True"/
    "False"), numeric flags, and PASS/FAIL strings. Missing/blank values are
    treated as *not passing*.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    return s in {"true", "1", "1.0", "yes", "pass", "passed", "y", "t"}


def _get_sample_name(row: pd.Series) -> Optional[str]:
    """Extract the per-sample name from a CSV row.

    Checks common column names: design_name, design, binder, name, id.
    Returns None if no suitable column is found.
    """
    for col in ["design_name", "design", "binder", "name", "id"]:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                return val
    return None


def _find_pdb(sample_name: str, csv_dir: Path) -> Optional[Path]:
    """Find the PDB for *sample_name* inside *csv_dir* (recursively).

    Tries exact name matches first (``<sample>_relaxed.pdb``, ``<sample>.pdb``),
    then falls back to substring matching.
    """
    # Exact name patterns (most specific first)
    for suffix in [f"{sample_name}_relaxed.pdb", f"{sample_name}.pdb", f"{sample_name}.cif"]:
        candidates = list(csv_dir.rglob(suffix))
        if candidates:
            return candidates[0]
    # Substring fallback: any PDB whose stem starts with sample_name
    for p in csv_dir.rglob("*.pdb"):
        if p.stem.startswith(sample_name):
            return p
    return None


# ---------------------------------------------------------------------------
# Core analysis workflow
# ---------------------------------------------------------------------------

def analyze(
    input_root: Path,
    output_dir: Path,
    top_n: int = 20,
    copy: bool = True,
    explicit_csv: Optional[Path] = None,
    include_failed: bool = False,
) -> None:
    """Execute the full analysis pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Gather CSVs -----
    csv_files: List[Path]
    if explicit_csv:
        csv_files = [explicit_csv]
    else:
        csv_files = _gather_csv_files(input_root)

    if not csv_files:
        print("[ERROR] No scoring CSV files found.")
        sys.exit(1)

    print(f"Found {len(csv_files)} scoring CSV files.")

    # ----- Parse all CSVs into a flat record list -----
    records: list[dict] = []

    for csv_path in tqdm(csv_files, desc="Loading CSVs"):
        df = _load_csv(csv_path)
        try:
            iptm_col = _detect_column(df, ["iptm", "ipTM", "ip_tm", "i_pTM", "i_ptm"])
            plddt_col = _detect_column(df, ["plddt", "pLDDT", "plddt_score"])
        except KeyError as exc:
            print(f"[WARN] Skipping {csv_path}: {exc}")
            continue
        
        # Detect the PyRosetta pass/fail column (per-design boolean).
        pyrosetta_col = None
        try:
            pyrosetta_col = _detect_column(df, ["pyrosetta_filter_pass"])
        except KeyError:
            pass

        # Try to detect an aggregate successful-samples count column (optional).
        successful_col = None
        try:
            successful_col = _detect_column(df, ["num_successful", "num_relaxed"])
        except KeyError:
            pass

        for _, row in df.iterrows():
            iptm = float(row[iptm_col]) if pd.notna(row[iptm_col]) else None
            plddt = float(row[plddt_col]) if pd.notna(row[plddt_col]) else None
            if iptm is None or plddt is None:
                continue

            sample_name = _get_sample_name(row)
            if sample_name is None:
                # Last resort: use CSV parent dir name + row index
                sample_name = csv_path.parent.name
            
            # Build a globally unique ID: sweep_dir / run_dir / sample_name
            # e.g.  PDGFR_sweep_sw_hs3_c75az0_s0/initial_20260609_0528/linked_sample_002
            csv_dir = csv_path.parent
            uid = f"{csv_dir.parent.name}/{csv_dir.name}/{sample_name}"

            # PyRosetta filter outcome. If the column is absent we cannot verify
            # a design passed, so treat it as not-passing (excluded by default).
            pyrosetta_pass = (
                _is_truthy(row[pyrosetta_col]) if pyrosetta_col is not None else False
            )

            # Start from every column in the scoring CSV so all metrics are
            # carried through, then layer on the derived / normalized fields.
            rec = row.to_dict()
            rec.update({
                "uid": uid,
                "sample_name": sample_name,
                "iptm": iptm,
                "plddt": plddt,
                "pyrosetta_filter_pass": pyrosetta_pass,
                "source_csv": str(csv_path),
                "csv_dir": str(csv_dir),
            })

            # Add successful samples count if available
            if successful_col and pd.notna(row[successful_col]):
                rec["successful_samples"] = int(row[successful_col])

            records.append(rec)

    if not records:
        print("[ERROR] No valid records extracted from CSVs.")
        sys.exit(1)

    df_all = pd.DataFrame.from_records(records)

    # ----- Summary statistics -----
    def _print_stats(label: str, series: pd.Series, fmt: str = ".3f") -> None:
        print(f"--- {label} ---")
        print(f"  mean   : {series.mean():{fmt}}")
        print(f"  median : {series.median():{fmt}}")
        print(f"  std    : {series.std():{fmt}}")
        print(f"  min    : {series.min():{fmt}}")
        print(f"  max    : {series.max():{fmt}}")
        print(f"  25th   : {series.quantile(0.25):{fmt}}")
        print(f"  75th   : {series.quantile(0.75):{fmt}}")

    n_pass = int(df_all["pyrosetta_filter_pass"].sum())
    print(f"\n=== Binder Success Summary ===")
    print(f"Total samples             : {len(df_all)}")
    print(f"PyRosetta-passing samples : {n_pass}")
    _print_stats("ipTM", df_all["iptm"], ".3f")
    _print_stats("pLDDT", df_all["plddt"], ".2f")

    # Print total successful samples count if available
    if "successful_samples" in df_all.columns:
        total_successful = df_all["successful_samples"].sum()
        print(f"Successful samples (PyRosetta): {int(total_successful)}")

    # ----- Restrict top-N selection to PyRosetta-passing designs -----
    if include_failed:
        df_pass = df_all
    else:
        df_pass = df_all[df_all["pyrosetta_filter_pass"]].copy()
        if df_pass.empty:
            print(
                "[ERROR] No PyRosetta-passing designs found (checked column "
                "'pyrosetta_filter_pass'). If your CSVs do not record this "
                "column, re-run with --include-failed to keep the old behaviour."
            )
            sys.exit(1)

    # ----- Top-N selection -----
    df_sorted = df_pass.sort_values(["iptm", "plddt"], ascending=False).reset_index(drop=True)

    copied = []  # list of dicts for top binder rows
    seen_uids: set[str] = set()

    for _, row in df_sorted.iterrows():
        if len(copied) >= top_n:
            break
        uid = row["uid"]
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        csv_dir = Path(row["csv_dir"])
        pdb_path = _find_pdb(row["sample_name"], csv_dir)

        if pdb_path is None:
            print(f"[WARN] PDB for {uid} not found in {csv_dir} – skipping.")
            continue

        # Copy/move to output
        dest = output_dir / pdb_path.name
        # Handle name collisions by prefixing with parent dirs
        if dest.exists():
            prefix = f"{csv_dir.parent.name}__{csv_dir.name}__"
            dest = output_dir / (prefix + pdb_path.name)
        try:
            if copy:
                shutil.copy2(pdb_path, dest)
            else:
                shutil.move(str(pdb_path), str(dest))
        except Exception as exc:
            print(f"[ERROR] Failed to {'copy' if copy else 'move'} {pdb_path}: {exc}")
            continue

        # Carry through every scoring metric for this design, then append the
        # resolved PDB source/destination paths.
        rec = row.to_dict()
        rec["pdb_source"] = str(pdb_path)
        rec["pdb_dest"] = str(dest)
        copied.append(rec)

    # ----- Write CSVs -----
    # Full stats
    stats_csv_path = output_dir / "binder_stats.csv"
    df_all.to_csv(stats_csv_path, index=False)

    # Top binders
    df_top = pd.DataFrame.from_records(copied)
    top_csv_path = output_dir / "top_binders.csv"
    df_top.to_csv(top_csv_path, index=False)

    print(f"\nFull statistics written to : {stats_csv_path}  ({len(df_all)} rows)")
    print(f"Top binders CSV written to: {top_csv_path}  ({len(df_top)} rows)")
    print(f"Top {len(df_top)} PDB files {'copied' if copy else 'moved'} to: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze successful binders, print stats, and collect top samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Root directory containing results.")
    parser.add_argument("--output", "-o", type=Path, default="top_binders",
                        help="Folder where top-binder PDBs and CSVs will be stored.")
    parser.add_argument("--csv", type=Path,
                        help="Explicit CSV file to analyse (overrides directory scan).")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top binders to collect.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy PDB files instead of moving them.")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include designs that did NOT pass the PyRosetta "
                             "filters (default: only passing designs are collected).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        input_root=args.input,
        output_dir=args.output,
        top_n=args.top_n,
        copy=args.copy,
        explicit_csv=args.csv,
        include_failed=args.include_failed,
    )
