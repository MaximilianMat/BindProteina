#!/usr/bin/env python3
"""
Secondary Structure Analysis for Ablation Study
=================================================

Scans an ablation folder for all successful binders (relaxed/*.pdb),
computes per-residue secondary structure using DSSP (via mdtraj),
and produces a scatter plot of helix fraction vs. sheet fraction
for every successful sample, colored by experimental condition.

Usage:
    # Single folder – auto-splits by fragment type:
    python scripts/ablation_secondary_structure.py \
        --input /path/to/ablation_folder \
        --output /path/to/output_folder \
        --chain B

    # Multiple folders – auto-splits by source folder:
    python scripts/ablation_secondary_structure.py \
        --input /path/to/folder_A /path/to/folder_B \
        --output /path/to/output_folder

    # Explicit split control:
    python scripts/ablation_secondary_structure.py \
        --input /path/to/ablation_folder \
        --output /path/to/output_folder \
        --split-by fragment
"""

import os
import re
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mdtraj as md
from tqdm import tqdm


# ============================================================
# CONDITION PARSING (reused from generate_ablation_results.py)
# ============================================================

# Fragment codes – ORDER MATTERS: longer/more-specific codes must come first
# to prevent substring matches (e.g. '3hxc' must beat '3hx').
_FRAGMENT_CODES = [
    ("3hxc", "3_res_helix_custom"),
    ("3btc", "3_res_beta_custom"),
    ("3hx",  "3_res_helix"),
    ("3bs",  "3_res_beta"),
    ("3bt",  "3_res_beta"),
    ("1gly", "single_gly"),
    ("gly",  "single_gly"),
    ("5hx",  "5_res_helix"),
    ("alpha", "cath_alpha"),
    ("beta",  "cath_beta"),
    ("mixed", "cath_mixed"),
    ("default", "cath_default"),
]

# Regex for flexible distance: supports '10A', '6_5A', '6-5A'
_DIST_RE = re.compile(r"(\d+(?:[_-]\d+)?)A(?:_|/|$)")


def parse_condition_from_path(path_str):
    """
    Infer experimental conditions from the directory path.
    Naming conventions:
      - sni -> inward, sno -> outward
      - 8A, 10A, 12A, 14A (or fractional like 6_5A) -> distance
      - longer fragment codes checked before shorter ones to avoid false substring matches
      - _rotXX -> angle XX
    """
    orientation = "unknown"
    if "sno" in path_str:
        orientation = "outward"
    elif "sni" in path_str:
        orientation = "inward"

    # Distance – supports fractional notation like 6_5A (= 6.5 Å)
    distance = 0
    m = _DIST_RE.search(path_str)
    if m:
        raw = m.group(1)
        try:
            distance = float(raw.replace("_", ".").replace("-", "."))
            if distance == int(distance):
                distance = int(distance)
        except ValueError:
            pass

    # Fragment – check longer codes first to avoid premature substring matches
    fragment = "unknown"
    for code, label in _FRAGMENT_CODES:
        if code in path_str:
            fragment = label
            break

    # Rotation angle from _rotXX path components or inline rot\d+ token
    angle = 0
    rot_m = re.search(r"_?rot(\d+)", path_str)
    if rot_m:
        try:
            angle = int(rot_m.group(1))
        except ValueError:
            pass

    return orientation, angle, distance, fragment


# ============================================================
# SCORING LOOKUP (for --select)
# ============================================================
# NOTE: kept byte-for-byte equivalent to the same helpers in
# scripts/seed_ablation_clusters.py so the two analyses select identical sets.
# If you change one, change the other (or factor both into functions/).

_TRUE_STRINGS = {"true", "1", "1.0", "yes", "pass", "passed", "y", "t"}
_SCORING_CSV = "scoring_summary.csv"


def build_scoring_index(roots):
    """Map ``(run_dir, design_name) -> passed_pyrosetta_filters``.

    IMPORTANT: the key includes the run directory. ``design_name`` is only
    unique *within* a run -- seed-search runs all emit ``linked_sample_NNN``, so
    a global name set would match a design that passed in one condition against
    identically-named PDBs in every other condition (5x over-count on
    seed_search_cd19: 148 instead of 30).

    A run whose scoring_summary.csv is missing or empty contributes nothing, so
    its structures are absent from the index and get dropped by any --select
    mode other than 'all'.
    """
    index = {}
    n_csv = n_empty = 0
    for root in roots:
        for csv_path in root.rglob(_SCORING_CSV):
            if csv_path.name.startswith("._"):  # macOS resource fork
                continue
            n_csv += 1
            try:
                if csv_path.stat().st_size == 0:
                    n_empty += 1
                    continue
                df = pd.read_csv(csv_path)
            except Exception:
                n_empty += 1
                continue
            if df.empty or "design_name" not in df.columns:
                n_empty += 1
                continue

            if "pyrosetta_filter_pass" in df.columns:
                ok = (df["pyrosetta_filter_pass"].astype(str)
                        .str.strip().str.lower().isin(_TRUE_STRINGS))
            else:
                # Cannot verify a pass without the column -> treat as not passing,
                # consistent with analyze_successful_binders.py.
                ok = pd.Series(False, index=df.index)

            run_dir = str(csv_path.parent)
            for name, passed in zip(df["design_name"].astype(str), ok):
                index[(run_dir, name)] = bool(passed)

    print(f"  scoring index: {len(index)} designs from {n_csv} "
          f"{_SCORING_CSV} files ({n_empty} empty/unusable)")
    return index


def scoring_key(pdb_file):
    """Key a PDB into the scoring index.

    The scoring CSV sits beside the PDB directory (``<run_dir>/<pdb_dir>/x.pdb``
    next to ``<run_dir>/scoring_summary.csv``) in all three layouts -- complexa
    ``shard_XXX/``, seed-search ``initial_*/`` and reseed ``genN/_scored/``.
    Relaxed PDBs carry a ``_relaxed`` suffix that design_name does not.
    """
    stem = pdb_file.stem
    if stem.endswith("_relaxed"):
        stem = stem[: -len("_relaxed")]
    return (str(pdb_file.parent.parent), stem)


# ============================================================
# DSSP ANALYSIS
# ============================================================

def compute_ss_fractions(pdb_path, chain_id="B"):
    """
    Load a PDB, run DSSP on the specified chain, and return
    (helix_fraction, sheet_fraction, loop_fraction) or None on failure.
    """
    try:
        traj = md.load(pdb_path)
        topology = traj.topology

        selected_residues = []
        for chain in topology.chains:
            if chain.chain_id == chain_id:
                for residue in chain.residues:
                    selected_residues.append(residue.index)

        if len(selected_residues) == 0:
            return None

        ss = md.compute_dssp(traj, simplified=True)[0]

        counts = Counter()
        for residue_index in selected_residues:
            state = ss[residue_index]
            if state == "H":
                counts["helix"] += 1
            elif state == "E":
                counts["sheet"] += 1
            else:
                counts["loop"] += 1

        total = sum(counts.values())
        if total == 0:
            return None

        return (
            counts["helix"] / total,
            counts["sheet"] / total,
            counts["loop"] / total,
        )
    except Exception as e:
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Secondary structure scatter plot for ablation study (relaxed PDBs only)."
    )
    parser.add_argument("--input", required=True, nargs='+', help="One or more input folders (folder basename = source label)")
    parser.add_argument(
        "--labels",
        nargs='+',
        help=(
            "Optional display label per --input folder (same count and order). "
            "Repeat a label to pool several folders into one group, e.g. "
            "--labels 'Proteina Complexa' 'Proteina Complexa' 'LaProteina' 'LaProteina'"
        ),
    )
    parser.add_argument("--output", required=True, help="Output folder for plots and CSV")
    parser.add_argument("--chain", default="B", help="Chain ID to analyze (default: B)")
    parser.add_argument("--residue", help="Optional hotspot residue filter (e.g. '60' for hs60)")
    parser.add_argument(
        "--pdb-dir",
        dest="pdb_dir",
        default="relaxed",
        help="Name of the leaf directory containing the PDBs (default: relaxed)",
    )
    parser.add_argument(
        "--include-path",
        dest="include_path",
        help=(
            "Only keep PDBs whose full path matches this regex. Use it to pin a target "
            "when an input root mixes several, e.g. --include-path '/CD19/'"
        ),
    )
    parser.add_argument(
        "--exclude-path",
        dest="exclude_path",
        help="Drop PDBs whose full path matches this regex. Applied after --include-path.",
    )
    parser.add_argument(
        "--select",
        choices=["all", "scored", "pyrosetta_pass"],
        default="all",
        help=(
            "Which structures to analyse. 'all' (default) = every PDB on disk, the "
            "historical behaviour. 'scored' = only those with a row in the run's "
            "scoring_summary.csv. 'pyrosetta_pass' = only those whose row has "
            "pyrosetta_filter_pass true, matching "
            "scripts/analyze_successful_binders.py and seed_ablation_clusters.py."
        ),
    )
    parser.add_argument(
        "--split-by",
        dest="split_by",
        choices=["source", "fragment"],
        default=None,
        help=(
            "Column to color and split plots by. "
            "Defaults to 'source' when multiple input folders are provided, "
            "otherwise 'fragment'."
        ),
    )

    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input]

    if args.labels:
        if len(args.labels) != len(input_dirs):
            parser.error(
                f"--labels takes exactly one label per --input folder "
                f"({len(input_dirs)} folders, {len(args.labels)} labels given)"
            )
        source_labels = list(args.labels)
    else:
        source_labels = [d.name for d in input_dirs]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Discover all relaxed PDB files
    # --------------------------------------------------------
    print(f"Scanning {[str(d) for d in input_dirs]} for {args.pdb_dir}/*.pdb ...")
    # pdb_files is a list of (path, source_label) tuples
    pdb_files = []
    for input_dir, source in zip(input_dirs, source_labels):
        for p in sorted(input_dir.rglob(f"{args.pdb_dir}/*.pdb")):
            if p.is_file() and p.parent.name == args.pdb_dir:
                pdb_files.append((p, source))

    if args.residue:
        pdb_files = [(p, s) for p, s in pdb_files if f"hs{args.residue}" in str(p)]
        print(f"  Filtered to hotspot residue hs{args.residue}")

    if args.include_path:
        rx = re.compile(args.include_path)
        before = len(pdb_files)
        pdb_files = [(p, s) for p, s in pdb_files if rx.search(str(p))]
        print(f"  --include-path {args.include_path!r}: kept {len(pdb_files)}/{before}")

    if args.exclude_path:
        rx = re.compile(args.exclude_path)
        before = len(pdb_files)
        pdb_files = [(p, s) for p, s in pdb_files if not rx.search(str(p))]
        print(f"  --exclude-path {args.exclude_path!r}: kept {len(pdb_files)}/{before}")

    if args.select != "all":
        scoring_index = build_scoring_index(input_dirs)
        before = len(pdb_files)
        if args.select == "scored":
            pdb_files = [(p, s) for p, s in pdb_files
                         if scoring_key(p) in scoring_index]
        else:  # pyrosetta_pass
            pdb_files = [(p, s) for p, s in pdb_files
                         if scoring_index.get(scoring_key(p), False)]
        print(f"  --select {args.select}: kept {len(pdb_files)}/{before}")
        if not pdb_files:
            print("No structures left after --select filtering. Exiting.")
            return

    print(f"  Found {len(pdb_files)} successful PDB files.")

    # Per-source counts up front: makes a mislabelled or contaminated input obvious
    # before spending minutes on DSSP.
    per_source = Counter(s for _, s in pdb_files)
    for lbl in dict.fromkeys(source_labels):
        print(f"    {lbl}: {per_source.get(lbl, 0)}")
    print()

    if len(pdb_files) == 0:
        print("No PDB files found. Exiting.")
        return

    # --------------------------------------------------------
    # 2. Compute secondary structure for each file
    # --------------------------------------------------------
    records = []

    for pdb_path, source in tqdm(pdb_files, desc="Computing DSSP"):
        result = compute_ss_fractions(str(pdb_path), chain_id=args.chain)
        if result is None:
            continue

        helix_frac, sheet_frac, loop_frac = result
        path_str = str(pdb_path)

        orientation, angle, distance, fragment = parse_condition_from_path(path_str)

        records.append({
            "pdb_file": pdb_path.name,
            "pdb_path": path_str,
            "source": source,
            "helix_frac": helix_frac,
            "sheet_frac": sheet_frac,
            "loop_frac": loop_frac,
            "orientation": orientation,
            "angle": angle,
            "distance": distance,
            "fragment": fragment,
            "condition": f"{fragment}_{orientation}_{distance}A",
        })

    df = pd.DataFrame(records)

    if df.empty:
        print("No valid DSSP results. Exiting.")
        return

    print(f"\nSuccessfully analyzed {len(df)} structures.")

    # Save raw data
    csv_path = output_dir / "secondary_structure_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # --------------------------------------------------------
    # Determine split column (source vs fragment)
    # --------------------------------------------------------
    split_col = args.split_by
    if split_col is None:
        split_col = "source" if df["source"].nunique() > 1 else "fragment"
    print(f"Splitting plots by: '{split_col}'")

    # --------------------------------------------------------
    # 3. Scatter plot: Helix % vs Sheet %, colored by split_col
    # --------------------------------------------------------
    # Keep the order the user passed on the command line for 'source';
    # anything else has no meaningful input order, so sort it.
    if split_col == "source":
        seen = dict.fromkeys(source_labels)
        present = set(df["source"].unique())
        groups = [g for g in seen if g in present]
    else:
        groups = sorted(df[split_col].unique())

    palette = dict(zip(groups, sns.color_palette("Set2", n_colors=len(groups))))

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x="helix_frac",
        y="sheet_frac",
        hue=split_col,
        hue_order=groups,
        palette=palette,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
        s=50,
        ax=ax,
    )

    # Diagonal reference lines showing constant loop fraction
    for loop_pct in [0.2, 0.4, 0.6, 0.8]:
        x_vals = np.linspace(0, 1 - loop_pct, 50)
        y_vals = 1 - loop_pct - x_vals
        ax.plot(x_vals, y_vals, "--", color="gray", alpha=0.3, linewidth=0.8)
        ax.text(
            0.01, 1 - loop_pct - 0.02,
            f"{int(loop_pct*100)}% loop",
            fontsize=7, color="gray", alpha=0.5,
        )

    ax.set_xlabel("Helix Fraction", fontsize=12)
    ax.set_ylabel("Sheet Fraction", fontsize=12)
    ax.set_title("Secondary Structure of Successful Binders", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    legend_title = split_col.replace("_", " ").title()
    ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    scatter_path = output_dir / f"ss_helix_vs_sheet_by_{split_col}.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

    # --------------------------------------------------------
    # 4. Faceted scatter: one panel per group in split_col
    # --------------------------------------------------------
    if len(groups) > 1:
        ncols = min(4, len(groups))
        nrows = (len(groups) + ncols - 1) // ncols
        fig, axes_arr = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            sharey=True, sharex=True,
            constrained_layout=True,
        )
        # Flatten to 1-D for uniform iteration
        axes_flat = np.array(axes_arr).flatten() if len(groups) > 1 else [axes_arr]

        for ax, grp in zip(axes_flat, groups):
            subset = df[df[split_col] == grp]
            ax.scatter(
                subset["helix_frac"],
                subset["sheet_frac"],
                c=[palette[grp]],
                alpha=0.6,
                edgecolor="white",
                linewidth=0.3,
                s=40,
            )
            ax.set_title(f"{grp}\n(n={len(subset)})")
            ax.set_xlabel("Helix Fraction")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

        # Hide any unused axes
        for ax in axes_flat[len(groups):]:
            ax.set_visible(False)

        for ax in axes_flat[::ncols]:
            ax.set_ylabel("Sheet Fraction")

        plt.suptitle(f"Secondary Structure by {legend_title}", fontsize=14)

        facet_path = output_dir / f"ss_faceted_by_{split_col}.png"
        fig.savefig(facet_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Faceted plot saved to {facet_path}")

    # --------------------------------------------------------
    # 5. Summary statistics per split_col group
    # --------------------------------------------------------
    print(f"\n=== PER-{split_col.upper()} SUMMARY ===")
    summary = df.groupby(split_col).agg(
        n_samples=("helix_frac", "count"),
        mean_helix=("helix_frac", "mean"),
        std_helix=("helix_frac", "std"),
        mean_sheet=("sheet_frac", "mean"),
        std_sheet=("sheet_frac", "std"),
        mean_loop=("loop_frac", "mean"),
        std_loop=("loop_frac", "std"),
    ).round(3)
    print(summary)

    summary_path = output_dir / f"ss_summary_by_{split_col}.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
