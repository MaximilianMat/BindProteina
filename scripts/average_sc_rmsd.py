#!/usr/bin/env python3
"""
Average SC_RMSD from Protenix prescoring CSVs
-------------------------------------------------
Recursively searches a directory for `protenix_prescoring.csv` files,
optionally filters them by a hotspot‑residue substring (e.g. "hs60"),
extracts the `sc_rmsd` column, and prints the overall average.

Usage:
    python3 scripts/average_sc_rmsd_from_prescoring.py /path/to/root [--residue hs60]

The script only outputs a single numeric summary – no plots are generated.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm


def find_prescoring_files(root_dir):
    """Return a list of all `protenix_prescoring.csv` files under `root_dir`."""
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == "protenix_prescoring_results.csv":
                matches.append(os.path.join(dirpath, f))
    return matches


def collect_sc_rmsd(csv_files):
    """Read the `sc_rmsd` column from each CSV and return a flat list of values."""
    rmsd_values = []
    for csv_path in tqdm(csv_files, desc="Reading CSVs"):
        try:
            df = pd.read_csv(csv_path)
            if "sc_rmsd" in df.columns:
                # Drop NaNs and extend the list
                rmsd_values.extend(df["sc_rmsd"].dropna().astype(float).tolist())
        except Exception:
            # Skip unreadable or malformed files silently
            continue
    return rmsd_values


def main():
    parser = argparse.ArgumentParser(
        description="Average SC_RMSD from Protenix prescoring CSV files"
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing subfolders with `protenix_prescoring.csv` files",
    )
    parser.add_argument(
        "--residue",
        help="Optional hotspot residue substring filter (e.g., 'hs60')",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Directory not found: {args.root_dir}")
        return

    # Locate all CSV files
    csv_files = find_prescoring_files(args.root_dir)

    # Apply optional residue filter on the full path
    if args.residue:
        csv_files = [p for p in csv_files if args.residue in p]

    if not csv_files:
        print("No `protenix_prescoring.csv` files matched the criteria.")
        return

    # Extract RMSD values
    rmsd_vals = collect_sc_rmsd(csv_files)

    if not rmsd_vals:
        print("No `sc_rmsd` values found in the selected CSV files.")
        return

    avg_rmsd = sum(rmsd_vals) / len(rmsd_vals)
    print("\n=== SC_RMSD FROM PRESCORING SUMMARY ===")
    print(f"Processed {len(csv_files)} CSV files.")
    print(f"Total RMSD entries: {len(rmsd_vals)}")
    print(f"Average SC_RMSD: {avg_rmsd:.3f} Å")


if __name__ == "__main__":
    main()
