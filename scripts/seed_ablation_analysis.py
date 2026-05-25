"""
Protein Binder Ablation Study Visualization Script
==================================================

Loads ablation results from a JSON file and generates:
1. Faceted heatmaps
2. Marginal effect plots (Distance, Angle, Orientation, Fragment)
3. Cluster composition plots (optional)
4. Statistical summaries
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot seed ablation results.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file (e.g., pdl1_ablation_results.json)")
    parser.add_argument("--output", required=True, help="Path to the output folder for plots")
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        results = json.load(f)
        
    df = pd.DataFrame(results)

    if len(df) == 0:
        raise ValueError("No data found in JSON.")

    fragment_order = [
        "single_gly",
        "3_res_helix",
        "5_res_helix",
        "3_res_beta"
    ]
    # Keep only fragments that actually exist in the data to avoid empty rows
    fragment_order = [f for f in fragment_order if f in df["fragment"].unique()]
    
    orientation_order = ["outward", "inward"]

    # ============================================================
    # 2. FACETED HEATMAPS
    # ============================================================
    print("Generating heatmaps...")
    fig, axes = plt.subplots(
        nrows=len(fragment_order),
        ncols=len(orientation_order),
        figsize=(12, 4 * len(fragment_order)),
        constrained_layout=True
    )
    
    # Ensure axes is a 2D array even if len(fragment_order) == 1
    if len(fragment_order) == 1:
        axes = np.array([axes])

    vmin = df["success_rate"].min()
    vmax = df["success_rate"].max()

    for i, fragment in enumerate(fragment_order):
        for j, orientation in enumerate(orientation_order):
            ax = axes[i, j]
            
            subset = df[
                (df["fragment"] == fragment) &
                (df["orientation"] == orientation)
            ]
            
            if subset.empty:
                ax.set_title(f"{fragment} | {orientation}\n(No Data)")
                ax.axis('off')
                continue

            pivot = subset.pivot(
                index="distance",
                columns="angle",
                values="success_rate"
            )

            sns.heatmap(
                pivot,
                ax=ax,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                vmin=vmin,
                vmax=vmax,
                cbar=(i == 0 and j == 1)
            )

            ax.set_title(f"{fragment} | {orientation}")
            ax.set_xlabel("Angle")
            ax.set_ylabel("Distance (Å)")

    plt.suptitle("Success Rate Heatmaps", fontsize=18)
    heatmap_path = output_dir / "heatmaps.png"
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()


    # ============================================================
    # 3. MARGINAL EFFECTS (COMBINED 1x4 PLOT)
    # ============================================================
    print("Generating marginal effect plots...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

    # 1. Distance effect
    distance_stats = df.groupby("distance")["success_rate"].agg(["mean", "std", "count"]).reset_index()
    distance_stats["se"] = distance_stats["std"] / np.sqrt(distance_stats["count"])
    axes[0].errorbar(distance_stats["distance"], distance_stats["mean"], yerr=distance_stats["se"], marker="o")
    axes[0].set_title("Distance Effect")
    axes[0].set_xlabel("Distance (Å)")
    axes[0].set_ylabel("Mean Success Rate")

    # 2. Angle effect
    angle_stats = df.groupby("angle")["success_rate"].agg(["mean", "std", "count"]).reset_index()
    angle_stats["se"] = angle_stats["std"] / np.sqrt(angle_stats["count"])
    axes[1].errorbar(angle_stats["angle"], angle_stats["mean"], yerr=angle_stats["se"], marker="o")
    axes[1].set_title("Angle Effect")
    axes[1].set_xlabel("Angle")
    axes[1].set_ylabel("Mean Success Rate")

    # 3. Orientation effect
    orientation_stats = df.groupby("orientation")["success_rate"].agg(["mean", "std", "count"]).reset_index()
    orientation_stats["se"] = orientation_stats["std"] / np.sqrt(orientation_stats["count"])
    axes[2].bar(orientation_stats["orientation"], orientation_stats["mean"], yerr=orientation_stats["se"])
    axes[2].set_title("Orientation Effect")
    axes[2].set_xlabel("Orientation")
    axes[2].set_ylabel("Mean Success Rate")

    # 4. Fragment effect
    fragment_stats = df.groupby("fragment")["success_rate"].agg(["mean", "std", "count"]).reset_index()
    fragment_stats["se"] = fragment_stats["std"] / np.sqrt(fragment_stats["count"])
    axes[3].bar(fragment_stats["fragment"], fragment_stats["mean"], yerr=fragment_stats["se"])
    axes[3].set_title("Fragment Effect")
    axes[3].set_xlabel("Fragment Type")
    axes[3].set_ylabel("Mean Success Rate")
    axes[3].tick_params(axis='x', rotation=20)

    plt.suptitle("Marginal Effects on Success Rate", fontsize=18)
    effects_path = output_dir / "marginal_effects.png"
    fig.savefig(effects_path, dpi=300, bbox_inches="tight")
    plt.close()


    # ============================================================
    # 4. SUMMARY TABLE
    # ============================================================
    summary = df.groupby(["fragment", "orientation", "distance", "angle"])["success_rate"].mean().reset_index()
    summary_path = output_dir / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== SUMMARY ===")
    print(f"Heatmaps saved to: {heatmap_path}")
    print(f"Marginal effects saved to: {effects_path}")
    print(f"Summary table saved to: {summary_path}")

if __name__ == "__main__":
    main()