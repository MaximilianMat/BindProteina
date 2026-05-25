import os
import argparse
import subprocess
from pathlib import Path

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
    help="Folder containing experimental runs"
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

INPUT_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output)
TMP_DIR = Path(args.tmp)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISCOVER PDB FILES & BUILD METADATA
# ============================================================

print(f"Discovering PDB files (STRICT '{args.pdb_dir}' leaf-level only)...")

samples = []
# Ensure stable sorting for reproducibility
for i, pdb_file in enumerate(sorted(INPUT_DIR.rglob(f"{args.pdb_dir}/*.pdb"))):

    if not pdb_file.is_file():
        continue

    if pdb_file.parent.name != args.pdb_dir:
        continue

    parts = pdb_file.parts

    # Parse metadata from path
    seed = next((p for p in reversed(parts) if p.startswith("initial_")), "unknown_seed")
    rotation = next((p for p in reversed(parts) if p.startswith("_rot")), "unknown_rot")
    
    base_run = "unknown_base"
    if rotation != "unknown_rot":
        try:
            rot_idx = parts.index(rotation)
            if rot_idx > 0:
                base_run = parts[rot_idx - 1]
        except ValueError:
            pass

    sample_name = pdb_file.stem
    
    # We assign a unique ID to each file to prevent any name collision or parsing issues later
    extracted_filename = f"struct_{i:06d}.pdb"
    
    samples.append({
        "extracted_filename": extracted_filename,
        "original_path": str(pdb_file.resolve()),
        "base_run": base_run,
        "rotation": rotation,
        "seed": seed,
        "sample": sample_name,
        "experiment": f"{base_run}__{rotation}",
        "group_id": f"{base_run}__{rotation}__{seed}" # unique identifier for the input seed
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
# COMPUTE DIVERSITY PER INPUT SEED
# ============================================================

print("Computing statistics per input seed...")

# We want to see how diverse the 100 samples were *for each input seed*
seed_stats = final_df.groupby("group_id").agg(
    experiment=("experiment", "first"),
    base_run=("base_run", "first"),
    rotation=("rotation", "first"),
    seed=("seed", "first"),
    num_samples=("member", "count"),
    num_clusters=("cluster_id", "nunique")
).reset_index()

# Fraction of clusters hit by the samples of this seed
seed_stats["cluster_hit_fraction"] = seed_stats["num_clusters"] / seed_stats["num_samples"]

# Compute global cluster sizes
cluster_sizes = final_df["cluster_id"].value_counts().reset_index()
cluster_sizes.columns = ["cluster_id", "cluster_size"]

final_df = final_df.merge(cluster_sizes, on="cluster_id", how="left")
final_df["is_singleton"] = final_df["cluster_size"] == 1

# Calculate proportion of samples per seed that ended up as singletons globally
seed_singletons = final_df.groupby("group_id")["is_singleton"].mean().reset_index(name="singleton_fraction")
seed_stats = seed_stats.merge(seed_singletons, on="group_id")

seed_stats.to_csv(OUTPUT_DIR / "seed_diversity_statistics.csv", index=False)


# ============================================================
# COMPUTE EXPERIMENT LEVEL STATS
# ============================================================

print("Computing statistics per experiment (base_run + rotation)...")

exp_stats = final_df.groupby("experiment").agg(
    base_run=("base_run", "first"),
    rotation=("rotation", "first"),
    num_seeds=("seed", "nunique"),
    num_samples=("member", "count"),
    num_clusters=("cluster_id", "nunique")
).reset_index()

exp_singletons = final_df.groupby("experiment")["is_singleton"].mean().reset_index(name="singleton_fraction")
exp_stats = exp_stats.merge(exp_singletons, on="experiment")

exp_stats.to_csv(OUTPUT_DIR / "experiment_statistics.csv", index=False)


# ============================================================
# PLOTS
# ============================================================

sns.set_style("whitegrid")

# 1. Diversity (Clusters Hit) per Seed
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=seed_stats,
    x="experiment",
    y="cluster_hit_fraction",
    color="lightgray",
    showfliers=False
)
sns.stripplot(
    data=seed_stats,
    x="experiment",
    y="cluster_hit_fraction",
    hue="num_samples",
    palette="viridis",
    alpha=0.8,
    jitter=True
)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Diversity Fraction\n(Unique Clusters / Successful Samples)")
plt.title("Normalized Structural Diversity Per Input Seed\n(Dots colored by number of successful samples)")

# Fix legend for continuous hue if needed
plt.legend(title="Successes\n(Num Samples)", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "diversity_fraction_per_seed.png", dpi=300)
plt.close()

# 2. Singleton Fraction per Experiment
plt.figure(figsize=(10, 6))
sns.barplot(
    data=exp_stats,
    x="experiment",
    y="singleton_fraction"
)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Singleton Fraction")
plt.title("Novelty / Uniqueness Per Experiment")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "singleton_fraction_per_experiment.png", dpi=300)
plt.close()

# 3. Global Cluster Size Distribution
plt.figure(figsize=(8, 6))
plt.hist(
    cluster_sizes["cluster_size"],
    bins=30,
    edgecolor="black"
)
plt.xlabel("Cluster Size")
plt.ylabel("Count")
plt.title("Global Cluster Size Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cluster_size_distribution.png", dpi=300)
plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n=== SUMMARY (Per Experiment) ===")
print(exp_stats.to_string())

print("\nDone.")
print(f"Results written to: {OUTPUT_DIR}")
