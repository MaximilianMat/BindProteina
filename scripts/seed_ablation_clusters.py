import os
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(description="Cluster generated binders with Foldseek and evaluate diversity per input seed.")

parser.add_argument(
    "--input",
    required=True,
    nargs='+',
    help="One or more input folders to compare (source label = folder basename)"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output folder"
)

parser.add_argument(
    "--pdb-dir",
    default="relaxed",
    help="Name of the leaf directory containing the PDBs (e.g., 'relaxed' or 'protenix_batch')"
)

parser.add_argument(
    "--tmp",
    default="tmp_foldseek",
    help="Temporary Foldseek directory"
)

parser.add_argument(
    "--alignment-type",
    default="1",
    help="Foldseek alignment type (1 = TM-align)"
)

parser.add_argument(
    "--tmscore-threshold",
    default="0.5",
    help="TM-score threshold"
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

INPUT_DIRS = [Path(p) for p in args.input]
OUTPUT_DIR = Path(args.output)
TMP_DIR = Path(args.tmp)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISCOVER PDB FILES & BUILD METADATA
# ============================================================

print(f"Discovering PDB files (STRICT '{args.pdb_dir}' leaf-level only)...")

# Collect (pdb_file, source_label, source_dir) from all input dirs
all_pdb_files = []
for input_dir in INPUT_DIRS:
    for pdb_file in sorted(input_dir.rglob(f"{args.pdb_dir}/*.pdb")):
        all_pdb_files.append((pdb_file, input_dir.name, input_dir))
all_pdb_files.sort(key=lambda x: (x[1], str(x[0])))

samples = []
for i, (pdb_file, source, source_dir) in enumerate(all_pdb_files):

    if not pdb_file.is_file():
        continue

    if pdb_file.parent.name != args.pdb_dir:
        continue

    rel_parts = pdb_file.relative_to(source_dir).parts
    seed = next((p for p in rel_parts if p.startswith("initial_")), "unknown_seed")
    seed_idx = list(rel_parts).index(seed) if seed != "unknown_seed" else len(rel_parts) - 2
    # experiment = path between source root and initial_* dir; works for any sub-folder depth
    experiment = "/".join(rel_parts[:seed_idx]) if seed_idx > 0 else "default"

    extracted_filename = f"struct_{i:06d}.pdb"

    samples.append({
        "extracted_filename": extracted_filename,
        "original_path": str(pdb_file.resolve()),
        "source": source,
        "experiment": experiment,
        "seed": seed,
        "sample": pdb_file.stem,
        "group_id": f"{source}/{experiment}/{seed}",
    })

samples_df = pd.DataFrame(samples)

print(f"Found {len(samples_df)} valid structures.")
if not samples_df.empty:
    print(samples_df[["group_id", "sample", "extracted_filename"]].head())
else:
    print("No structures found. Exiting.")
    exit(0)

# Save the metadata mapping BEFORE clustering (crucial design choice)
mapping_path = OUTPUT_DIR / "file_metadata_mapping.csv"
samples_df.to_csv(mapping_path, index=False)
print(f"Saved file metadata mapping to {mapping_path}")


# ============================================================
# EXTRACT CHAIN B STRUCTURES
# ============================================================

def extract_chain_b(input_pdb, output_pdb):
    kept_lines = []
    found = False

    try:
        with open(input_pdb, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain = line[21].strip()
                    # EXPLICIT STRUCTURAL KNOWLEDGE: CHAIN B IS THE BINDER
                    if chain == "B":
                        kept_lines.append(line)
                        found = True
                elif line.startswith("TER") and found:
                    kept_lines.append(line)
    except Exception as e:
        print(f"Error reading {input_pdb}: {e}")
        return False

    if not found:
        return False

    try:
        with open(output_pdb, "w") as f:
            f.writelines(kept_lines)
    except Exception as e:
        print(f"Error writing {output_pdb}: {e}")
        return False

    return True

print("Extracting chain B only (validating structures)...")

ALL_STRUCTURES = OUTPUT_DIR / "all_structures"
ALL_STRUCTURES.mkdir(exist_ok=True)

num_ok = 0
num_fail = 0

valid_extracted_files = []

for _, row in samples_df.iterrows():
    input_pdb = row["original_path"]
    output_pdb = ALL_STRUCTURES / row["extracted_filename"]

    ok = extract_chain_b(input_pdb, output_pdb)

    # Fail fast & validate early: ensure the PDB is actually populated
    if ok and output_pdb.stat().st_size > 50:
        num_ok += 1
        valid_extracted_files.append(row["extracted_filename"])
    else:
        num_fail += 1
        if output_pdb.exists():
            output_pdb.unlink()

print(f"Chain B successfully extracted: {num_ok}")
print(f"Extraction failed (empty or no chain B): {num_fail}")

if num_ok == 0:
    print("No valid Chain B structures to cluster. Exiting.")
    exit(0)

# Filter our metadata to only include successfully extracted files
samples_df = samples_df[samples_df["extracted_filename"].isin(valid_extracted_files)].copy()


# ============================================================
# RUN FOLDSEEK CLUSTERING
# ============================================================

print("Running Foldseek clustering...")

cluster_output = OUTPUT_DIR / "cluster_result"

cmd = [
    "./bin/foldseek",
    "easy-cluster",
    str(ALL_STRUCTURES),
    str(cluster_output),
    str(TMP_DIR),
    "--alignment-type",
    args.alignment_type,
    "--tmscore-threshold",
    args.tmscore_threshold,
]

print("COMMAND:")
print(" ".join(cmd))

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"Foldseek clustering failed: {e}")
    exit(1)


# ============================================================
# LOAD CLUSTER RESULTS & JOIN METADATA
# ============================================================

cluster_tsv = Path(str(cluster_output) + "_cluster.tsv")

if not cluster_tsv.exists():
    raise ValueError(f"Missing cluster file: {cluster_tsv}")

print("Loading cluster assignments and joining with metadata...")

clusters = pd.read_csv(
    cluster_tsv,
    sep="\t",
    header=None,
    names=["representative", "member"]
)

# Foldseek might keep or strip the .pdb extension depending on version/flags.
# We normalize by stripping it before joining.
clusters["representative_key"] = clusters["representative"].str.replace(".pdb", "", regex=False)
clusters["member_key"] = clusters["member"].str.replace(".pdb", "", regex=False)

samples_df["match_key"] = samples_df["extracted_filename"].str.replace(".pdb", "", regex=False)

# Merge clustering results with our pristine metadata map
final_df = clusters.merge(
    samples_df, 
    left_on="member_key", 
    right_on="match_key", 
    how="inner"
)

# Assign a numeric ID to each cluster based on its representative
rep_to_cluster = {
    rep: i
    for i, rep in enumerate(final_df["representative_key"].unique())
}
final_df["cluster_id"] = final_df["representative_key"].map(rep_to_cluster)

# Save the unified dataset
final_df.to_csv(OUTPUT_DIR / "cluster_assignments_with_metadata.csv", index=False)


# ============================================================
# COMPUTE DIVERSITY STATISTICS
# ============================================================

print("Computing diversity statistics...")

# Compute global cluster sizes and tag singletons
cluster_sizes = final_df["cluster_id"].value_counts().reset_index()
cluster_sizes.columns = ["cluster_id", "cluster_size"]
final_df = final_df.merge(cluster_sizes, on="cluster_id", how="left")
final_df["is_singleton"] = final_df["cluster_size"] == 1

# Seed-level stats (one row per initial_* dir per source)
seed_stats = final_df.groupby("group_id").agg(
    source=("source", "first"),
    experiment=("experiment", "first"),
    seed=("seed", "first"),
    num_samples=("member", "count"),
    num_clusters=("cluster_id", "nunique"),
    num_singletons=("is_singleton", "sum"),
).reset_index()
seed_stats["cluster_hit_fraction"] = seed_stats["num_clusters"] / seed_stats["num_samples"]
seed_stats["singleton_fraction"]   = seed_stats["num_singletons"] / seed_stats["num_samples"]
seed_stats.to_csv(OUTPUT_DIR / "seed_diversity_statistics.csv", index=False)

# Experiment-level stats (aggregated across seeds)
exp_stats = final_df.groupby(["source", "experiment"]).agg(
    num_seeds=("seed", "nunique"),
    num_samples=("member", "count"),
    num_clusters=("cluster_id", "nunique"),
    num_singletons=("is_singleton", "sum"),
).reset_index()
exp_stats["cluster_hit_fraction"] = exp_stats["num_clusters"] / exp_stats["num_samples"]
exp_stats["singleton_fraction"]   = exp_stats["num_singletons"] / exp_stats["num_samples"]
exp_stats.to_csv(OUTPUT_DIR / "experiment_statistics.csv", index=False)

# Source-level stats (aggregated across all experiments and seeds)
source_stats = final_df.groupby("source").agg(
    num_samples=("member", "count"),
    num_clusters=("cluster_id", "nunique"),
    num_singletons=("is_singleton", "sum"),
).reset_index()
source_stats["cluster_hit_fraction"] = source_stats["num_clusters"] / source_stats["num_samples"]
source_stats["singleton_fraction"]   = source_stats["num_singletons"] / source_stats["num_samples"]
source_stats.to_csv(OUTPUT_DIR / "source_statistics.csv", index=False)


# ============================================================
# PLOTS
# ============================================================

sns.set_style("whitegrid")

# ── Plot 1: Absolute counts per source ───────────────────────
# Raw numbers: how many samples, unique clusters, and singletons each source produced.
abs_melted = source_stats.melt(
    id_vars="source",
    value_vars=["num_samples", "num_clusters", "num_singletons"],
    var_name="metric", value_name="count"
)
abs_melted["metric"] = abs_melted["metric"].map({
    "num_samples":    "Samples",
    "num_clusters":   "Clusters Hit",
    "num_singletons": "Singletons",
})
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=abs_melted, x="metric", y="count", hue="source", ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)
ax.set_title("Absolute Counts per Source")
ax.set_xlabel("")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot1_absolute_counts.png", dpi=300)
plt.close()

# ── Plot 2: Diversity fractions per source ────────────────────
# Normalised fractions make sources comparable regardless of sample count.
frac_melted = source_stats.melt(
    id_vars="source",
    value_vars=["cluster_hit_fraction", "singleton_fraction"],
    var_name="metric", value_name="fraction"
)
frac_melted["metric"] = frac_melted["metric"].map({
    "cluster_hit_fraction": "Cluster Hit Fraction\n(diversity)",
    "singleton_fraction":   "Singleton Fraction\n(novelty)",
})
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=frac_melted, x="metric", y="fraction", hue="source", ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
ax.set_title("Diversity Fractions per Source")
ax.set_ylabel("Fraction")
ax.set_xlabel("")
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot2_diversity_fractions.png", dpi=300)
plt.close()

# ── Plot 3: Seed-level diversity distribution (violin) ────────
# Shows consistency: is every seed diverse, or only some?
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, label in zip(
    axes,
    ["cluster_hit_fraction", "singleton_fraction"],
    ["Cluster Hit Fraction", "Singleton Fraction"],
):
    sns.violinplot(data=seed_stats, x="source", y=metric, ax=ax, inner="box", cut=0)
    ax.set_title(f"{label}")
    ax.set_ylabel(label)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
plt.suptitle("Seed-Level Diversity Distribution (per source)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot3_seed_diversity_distribution.png", dpi=300)
plt.close()

# ── Plot 4: Cluster size CDF per source ───────────────────────
# Left-shifted CDF = more unique structures (smaller clusters on average).
fig, ax = plt.subplots(figsize=(8, 5))
for src in sorted(final_df["source"].unique()):
    sizes = final_df[final_df["source"] == src]["cluster_size"].sort_values()
    cdf = np.arange(1, len(sizes) + 1) / len(sizes)
    ax.plot(sizes.values, cdf, label=src)
ax.set_xlabel("Cluster Size")
ax.set_ylabel("CDF (fraction of structures)")
ax.set_title("Cluster Size CDF by Source\n(steeper rise left = more unique structures)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot4_cluster_size_cdf.png", dpi=300)
plt.close()

# ── Plot 5: Cross-source cluster overlap ─────────────────────
# How much structural space do the two generators share vs explore exclusively?
sources = sorted(final_df["source"].unique())
if len(sources) == 2:
    s1, s2 = sources
    cluster_src_sets = final_df.groupby("cluster_id")["source"].apply(lambda x: frozenset(x.unique()))
    only_s1 = (cluster_src_sets == frozenset([s1])).sum()
    only_s2 = (cluster_src_sets == frozenset([s2])).sum()
    shared   = (cluster_src_sets == frozenset([s1, s2])).sum()
    labels = [f"Only\n{s1}", "Shared", f"Only\n{s2}"]
    values = [int(only_s1), int(shared), int(only_s2)]
    colors = ["#4e79a7", "#76b7b2", "#f28e2b"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val), ha='center', va='bottom')
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Cross-Source Cluster Overlap\n(shared = structural motif found by both generators)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot5_cross_source_overlap.png", dpi=300)
    plt.close()

# ── Plot 6: Per-experiment comparison (if multiple experiments) ─
# Direct head-to-head for each experimental condition.
if exp_stats["experiment"].nunique() > 1:
    n_exp = len(exp_stats)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_exp * 1.2), 5))
    for ax, metric, label in zip(
        axes,
        ["cluster_hit_fraction", "singleton_fraction"],
        ["Cluster Hit Fraction", "Singleton Fraction"],
    ):
        sns.barplot(data=exp_stats, x="experiment", y=metric, hue="source", ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
        ax.set_title(f"{label} per Experiment")
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.1)
    plt.suptitle("Per-Experiment Diversity Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot6_per_experiment_comparison.png", dpi=300)
    plt.close()

# ── Plot 7: Global cluster size distribution ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(cluster_sizes["cluster_size"], bins=30, edgecolor="black")
ax.set_xlabel("Cluster Size")
ax.set_ylabel("Count")
ax.set_title("Global Cluster Size Distribution (all sources combined)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "plot7_cluster_size_distribution.png", dpi=300)
plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n=== SUMMARY (Per Source) ===")
print(source_stats.to_string())

print("\n=== SUMMARY (Per Experiment) ===")
print(exp_stats.to_string())

print("\nDone.")
print(f"Results written to: {OUTPUT_DIR}")
