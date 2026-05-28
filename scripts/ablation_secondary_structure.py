#!/usr/bin/env python3
"""
Secondary Structure Analysis for Ablation Study
=================================================

Scans an ablation folder for all successful binders (relaxed/*.pdb),
computes per-residue secondary structure using DSSP (via mdtraj),
and produces a scatter plot of helix fraction vs. sheet fraction
for every successful sample, colored by experimental condition.

Usage:
    python scripts/ablation_secondary_structure.py \
        --input /path/to/ablation_folder \
        --output /path/to/output_folder \
        --chain B
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

def parse_condition_from_path(path_str):
    """
    Infer experimental conditions from the directory path.
    Naming conventions:
      - sni -> inward, sno -> outward
      - 8A, 10A, 12A, 14A -> distance
      - gly -> single_gly, 3hx -> 3_res_helix, 3bt/3bs -> 3_res_beta, 5hx -> 5_res_helix
      - _rotXX -> angle XX
    """
    orientation = "unknown"
    if "sno" in path_str:
        orientation = "outward"
    elif "sni" in path_str:
        orientation = "inward"

    distance = 0
    for d in [8, 10, 12, 14]:
        if f"{d}A" in path_str:
            distance = d
            break

    fragment = "unknown"
    if "gly" in path_str:
        fragment = "single_gly"
    elif "3hx" in path_str:
        fragment = "3_res_helix"
    elif "3bt" in path_str or "3bs" in path_str:
        fragment = "3_res_beta"
    elif "5hx" in path_str:
        fragment = "5_res_helix"

    angle = 0
    for part in Path(path_str).parts:
        if part.startswith("_rot"):
            try:
                angle = int(part.replace("_rot", ""))
            except ValueError:
                pass

    return orientation, angle, distance, fragment


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
    parser.add_argument("--input", required=True, help="Base directory of ablation study")
    parser.add_argument("--output", required=True, help="Output folder for plots and CSV")
    parser.add_argument("--chain", default="B", help="Chain ID to analyze (default: B)")
    parser.add_argument("--residue", help="Optional hotspot residue filter (e.g. '60' for hs60)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Discover all relaxed PDB files
    # --------------------------------------------------------
    print(f"Scanning {input_dir} for relaxed/*.pdb ...")
    pdb_files = sorted(input_dir.rglob("relaxed/*.pdb"))
    pdb_files = [p for p in pdb_files if p.is_file() and p.parent.name == "relaxed"]

    if args.residue:
        pdb_files = [p for p in pdb_files if f"hs{args.residue}" in str(p)]
        print(f"  Filtered to hotspot residue hs{args.residue}")

    print(f"  Found {len(pdb_files)} successful PDB files.\n")

    if len(pdb_files) == 0:
        print("No PDB files found. Exiting.")
        return

    # --------------------------------------------------------
    # 2. Compute secondary structure for each file
    # --------------------------------------------------------
    records = []

    for pdb_path in tqdm(pdb_files, desc="Computing DSSP"):
        result = compute_ss_fractions(str(pdb_path), chain_id=args.chain)
        if result is None:
            continue

        helix_frac, sheet_frac, loop_frac = result
        path_str = str(pdb_path)

        orientation, angle, distance, fragment = parse_condition_from_path(path_str)

        records.append({
            "pdb_file": pdb_path.name,
            "pdb_path": path_str,
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
    # 3. Scatter plot: Helix % vs Sheet %
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by fragment type for the clearest biological signal
    palette = {
        "single_gly": "#e74c3c",
        "3_res_helix": "#3498db",
        "5_res_helix": "#2ecc71",
        "3_res_beta": "#f39c12",
    }
    # Filter palette to only include fragments in the data
    used_fragments = df["fragment"].unique()
    palette = {k: v for k, v in palette.items() if k in used_fragments}

    sns.scatterplot(
        data=df,
        x="helix_frac",
        y="sheet_frac",
        hue="fragment",
        palette=palette,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.3,
        s=50,
        ax=ax,
    )

    # Add diagonal reference lines showing constant loop fraction
    for loop_pct in [0.2, 0.4, 0.6, 0.8]:
        x_vals = np.linspace(0, 1 - loop_pct, 50)
        y_vals = 1 - loop_pct - x_vals
        ax.plot(x_vals, y_vals, "--", color="gray", alpha=0.3, linewidth=0.8)
        # Label the line
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
    ax.legend(title="Fragment Type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    scatter_path = output_dir / "ss_helix_vs_sheet.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

    # --------------------------------------------------------
    # 4. Faceted by fragment type
    # --------------------------------------------------------
    n_fragments = len(used_fragments)
    if n_fragments > 1:
        fig, axes = plt.subplots(
            1, n_fragments,
            figsize=(5 * n_fragments, 5),
            sharey=True, sharex=True,
            constrained_layout=True,
        )
        if n_fragments == 1:
            axes = [axes]

        for ax, frag in zip(axes, sorted(used_fragments)):
            subset = df[df["fragment"] == frag]
            ax.scatter(
                subset["helix_frac"],
                subset["sheet_frac"],
                c=palette.get(frag, "gray"),
                alpha=0.6,
                edgecolor="white",
                linewidth=0.3,
                s=40,
            )
            ax.set_title(f"{frag} (n={len(subset)})")
            ax.set_xlabel("Helix Fraction")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)

        axes[0].set_ylabel("Sheet Fraction")
        plt.suptitle("Secondary Structure by Fragment Type", fontsize=14)

        facet_path = output_dir / "ss_faceted_by_fragment.png"
        fig.savefig(facet_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Faceted plot saved to {facet_path}")

    # --------------------------------------------------------
    # 5. Summary statistics
    # --------------------------------------------------------
    print("\n=== PER-FRAGMENT SUMMARY ===")
    summary = df.groupby("fragment").agg(
        n_samples=("helix_frac", "count"),
        mean_helix=("helix_frac", "mean"),
        std_helix=("helix_frac", "std"),
        mean_sheet=("sheet_frac", "mean"),
        std_sheet=("sheet_frac", "std"),
        mean_loop=("loop_frac", "mean"),
        std_loop=("loop_frac", "std"),
    ).round(3)
    print(summary)

    summary_path = output_dir / "ss_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
