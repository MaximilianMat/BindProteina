#!/usr/bin/env python3
"""
Collect final samples from Proteina Complexa reward CSV.

Reads a rewards CSV from a Proteina Complexa beam search run, identifies the
final design samples, and moves them to a subfolder for easy access.

Usage:
    python scripts/collect_protenix_complexa_samples.py \
        --csv /nfs/path/to/rewards_search_binder_local_pipeline_0.csv \
        [--output_subdir final_samples] \
        [--extensions pdb,cif,pae] \
        [--copy]

This script will:
  1. Parse the rewards CSV to extract design names
  2. Find corresponding structure files in the same directory as the CSV
  3. Move (or copy with --copy) them to a subfolder
  4. Print a summary
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd
from tqdm import tqdm


def find_design_files(
    design_name: str,
    search_dir: Path,
    extensions: List[str],
) -> List[Path]:
    """Find all files matching a design name with given extensions."""
    matches = []
    for ext in extensions:
        # Match files like design_name.pdb, design_name_sample_0.pdb, etc.
        pattern1 = f"{design_name}.{ext}"
        pattern2 = f"{design_name}_*{ext}"

        for match in search_dir.glob(pattern1):
            if match.is_file():
                matches.append(match)

        for match in search_dir.glob(pattern2):
            if match.is_file():
                matches.append(match)

    return list(set(matches))  # Deduplicate


def extract_design_names(csv_path: Path) -> Set[str]:
    """Extract unique design names from the rewards CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"ERROR: Failed to read CSV {csv_path}: {e}")
        sys.exit(1)

    # Try common column names for design identifiers
    design_cols = [
        "design_name",
        "design",
        "name",
        "binder_name",
        "id",
        "sample_id",
    ]

    design_col = None
    for col in design_cols:
        if col in df.columns:
            design_col = col
            break

    if design_col is None:
        print(f"ERROR: Could not identify design name column in {csv_path}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    return set(df[design_col].dropna().astype(str).unique())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        "-c",
        type=Path,
        required=True,
        help="Path to rewards CSV file from Proteina Complexa run",
    )
    parser.add_argument(
        "--output_subdir",
        "-o",
        default="final_samples",
        help="Name of subfolder to collect samples into (default: final_samples)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        default="pdb,cif,pae.json,json",
        help="Comma-separated file extensions to collect (default: pdb,cif,pae.json,json)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    search_dir = csv_path.parent
    output_dir = search_dir / args.output_subdir
    extensions = [e.strip() for e in args.extensions.split(",")]

    print(f"Reading rewards CSV: {csv_path}")
    design_names = extract_design_names(csv_path)
    print(f"Found {len(design_names)} unique designs")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    total_moved = 0
    designs_with_files = 0

    for design_name in tqdm(sorted(design_names), desc="Processing designs"):
        files = find_design_files(design_name, search_dir, extensions)

        if files:
            designs_with_files += 1
            for src_file in files:
                dst_file = output_dir / src_file.name
                try:
                    if args.copy:
                        shutil.copy2(src_file, dst_file)
                    else:
                        shutil.move(str(src_file), str(dst_file))
                    total_moved += 1
                except Exception as e:
                    print(f"  ERROR moving {src_file}: {e}")

    operation = "copied" if args.copy else "moved"
    print(f"\n=== SUMMARY ===")
    print(f"Total designs: {len(design_names)}")
    print(f"Designs with files: {designs_with_files}")
    print(f"Total files {operation}: {total_moved}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
