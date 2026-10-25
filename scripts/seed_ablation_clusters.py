import os
import re
import shutil
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
    "--labels",
    nargs='+',
    help=(
        "Optional display label per --input folder (same count and order). "
        "Repeat a label to pool several folders into one source group, e.g. "
        "--labels 'Proteina Complexa' 'Proteina Complexa' 'LaProteina' 'LaProteina'"
    )
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
    "--include-path",
    dest="include_path",
    help=(
        "Only keep PDBs whose full path matches this regex. Use it to pin a target "
        "when an input root mixes several, e.g. --include-path '/CD19/'"
    )
)

parser.add_argument(
    "--exclude-path",
    dest="exclude_path",
    help="Drop PDBs whose full path matches this regex. Applied after --include-path."
)

parser.add_argument(
    "--select",
    choices=["all", "scored", "pyrosetta_pass"],
    default="all",
    help=(
        "Which structures to cluster. 'all' (default) = every PDB on disk, the "
        "historical behaviour. 'scored' = only those with a row in the run's "
        "scoring_summary.csv. 'pyrosetta_pass' = only those whose row has "
        "pyrosetta_filter_pass true, matching scripts/analyze_successful_binders.py."
    )
)

parser.add_argument(
    "--seed-marker",
    dest="seed_marker",
    nargs='+',
    default=["initial_"],
    help=(
        "Path-component prefix(es) identifying the seed directory (default: 'initial_'). "
        "Accepts several so input families with different layouts can be mixed in one "
        "run, e.g. --seed-marker initial_ gen (seed-search uses initial_*, reseed uses gen*). "
        "The first path component matching any marker wins."
    )
)

parser.add_argument(
    "--max-experiments-plot",
    dest="max_experiments_plot",
    type=int,
    default=40,
    help="Skip the per-experiment plot above this many experiments (default: 40)"
)

parser.add_argument(
    "--foldseek",
    default=None,
    help=(
        "Path to the foldseek binary. Defaults to ./bin/foldseek if present, "
        "otherwise the first 'foldseek' found on PATH."
    )
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

if args.labels:
    if len(args.labels) != len(INPUT_DIRS):
        parser.error(
            f"--labels takes exactly one label per --input folder "
            f"({len(INPUT_DIRS)} folders, {len(args.labels)} labels given)"
        )
    SOURCE_LABELS = list(args.labels)
else:
    SOURCE_LABELS = [d.name for d in INPUT_DIRS]

# Plot/legend order follows the command line, not alphabetical order.
SOURCE_ORDER = list(dict.fromkeys(SOURCE_LABELS))

# A label is "pooled" when it covers more than one input folder. Experiment names
# are only unique within a folder (e.g. every run has a 'shard_000'), so pooled
# labels need the folder name prefixed to keep group_id collision-free.
_label_folder_count = {lbl: SOURCE_LABELS.count(lbl) for lbl in SOURCE_ORDER}

OUTPUT_DIR = Path(args.output)
TMP_DIR = Path(args.tmp)

# Resolve the foldseek binary up front so we fail before doing any expensive work.
if args.foldseek:
    FOLDSEEK_BIN = args.foldseek
    if not (Path(FOLDSEEK_BIN).is_file() and os.access(FOLDSEEK_BIN, os.X_OK)):
        parser.error(f"--foldseek is not an executable file: {FOLDSEEK_BIN}")
elif Path("./bin/foldseek").is_file():
    FOLDSEEK_BIN = "./bin/foldseek"
else:
    FOLDSEEK_BIN = shutil.which("foldseek")
    if FOLDSEEK_BIN is None:
        parser.error(
            "Could not find the foldseek binary: ./bin/foldseek does not exist and "
            "'foldseek' is not on PATH. Pass --foldseek /path/to/foldseek."
        )
print(f"Using foldseek binary: {FOLDSEEK_BIN}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SCORING LOOKUP (for --select)
# ============================================================

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
# DISCOVER PDB FILES & BUILD METADATA
# ============================================================

print(f"Discovering PDB files (STRICT '{args.pdb_dir}' leaf-level only)...")

# Collect (pdb_file, source_label, source_dir) from all input dirs
all_pdb_files = []
for input_dir, source_label in zip(INPUT_DIRS, SOURCE_LABELS):
    for pdb_file in sorted(input_dir.rglob(f"{args.pdb_dir}/*.pdb")):
        all_pdb_files.append((pdb_file, source_label, input_dir))

if args.include_path:
    _rx = re.compile(args.include_path)
    _before = len(all_pdb_files)
    all_pdb_files = [t for t in all_pdb_files if _rx.search(str(t[0]))]
    print(f"  --include-path {args.include_path!r}: kept {len(all_pdb_files)}/{_before}")

if args.exclude_path:
    _rx = re.compile(args.exclude_path)
    _before = len(all_pdb_files)
    all_pdb_files = [t for t in all_pdb_files if not _rx.search(str(t[0]))]
    print(f"  --exclude-path {args.exclude_path!r}: kept {len(all_pdb_files)}/{_before}")

if args.select != "all":
    scoring_index = build_scoring_index(INPUT_DIRS)
    _before = len(all_pdb_files)
    if args.select == "scored":
        all_pdb_files = [t for t in all_pdb_files if scoring_key(t[0]) in scoring_index]
    else:  # pyrosetta_pass
        all_pdb_files = [t for t in all_pdb_files
                         if scoring_index.get(scoring_key(t[0]), False)]
    print(f"  --select {args.select}: kept {len(all_pdb_files)}/{_before}")
    if not all_pdb_files:
        print("No structures left after --select filtering. Exiting.")
        exit(0)

all_pdb_files.sort(key=lambda x: (x[1], str(x[0])))

samples = []
for i, (pdb_file, source, source_dir) in enumerate(all_pdb_files):

    if not pdb_file.is_file():
        continue

    if pdb_file.parent.name != args.pdb_dir:
        continue

    rel_parts = pdb_file.relative_to(source_dir).parts
    seed = next(
        (p for p in rel_parts if any(p.startswith(m) for m in args.seed_marker)),
        "unknown_seed",
    )
    seed_idx = list(rel_parts).index(seed) if seed != "unknown_seed" else len(rel_parts) - 2
    # experiment = path between source root and the seed dir; works for any sub-folder depth
    experiment = "/".join(rel_parts[:seed_idx]) if seed_idx > 0 else "default"

    # Disambiguate identically-named experiments coming from different folders
    # that were pooled under the same label (e.g. two runs each with 'shard_000').
    folder = source_dir.name
    if _label_folder_count[source] > 1:
        experiment = f"{folder}/{experiment}"

    extracted_filename = f"struct_{i:06d}.pdb"

    samples.append({
        "extracted_filename": extracted_filename,
        "original_path": str(pdb_file.resolve()),
        "source": source,
        "folder": folder,
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

# Per-source / per-folder breakdown up front: a mislabelled --labels order or a
# contaminated input root (e.g. a reseed tree holding several targets) shows up
# here, before Foldseek spends time clustering the wrong thing.
print("\nStructures per source:")
for lbl in SOURCE_ORDER:
    sub = samples_df[samples_df["source"] == lbl]
    print(f"  {lbl}: {len(sub)}")
    for folder, n in sub["folder"].value_counts().items():
        print(f"      {folder}: {n}")
print()

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

# Clear leftovers from a previous run into the same --output. Extracted names
# (struct_NNNNNN.pdb) are positional, so a smaller re-run (e.g. switching
# --select from 'all' to 'pyrosetta_pass') would otherwise leave stale
# structures behind and Foldseek would silently cluster those too.
_stale = list(ALL_STRUCTURES.glob("struct_*.pdb"))
if _stale:
    print(f"Clearing {len(_stale)} structures from a previous run in {ALL_STRUCTURES}")
    for _f in _stale:
        _f.unlink()

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
    FOLDSEEK_BIN,
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

# Present sources in command-line order rather than alphabetically.
PRESENT_SOURCES = [s for s in SOURCE_ORDER if s in set(final_df["source"].unique())]
source_stats["source"] = pd.Categorical(source_stats["source"], categories=PRESENT_SOURCES, ordered=True)
source_stats = source_stats.sort_values("source").reset_index(drop=True)

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
sns.barplot(data=abs_melted, x="metric", y="count", hue="source", hue_order=PRESENT_SOURCES, ax=ax)
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
sns.barplot(data=frac_melted, x="metric", y="fraction", hue="source", hue_order=PRESENT_SOURCES, ax=ax)
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
    sns.violinplot(data=seed_stats, x="source", y=metric, order=PRESENT_SOURCES, ax=ax, inner="box", cut=0)
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
for src in PRESENT_SOURCES:
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
sources = PRESENT_SOURCES
if len(sources) >= 2:
    # Which set of sources does each cluster contain?
    cluster_src_sets = final_df.groupby("cluster_id")["source"].apply(lambda x: frozenset(x.unique()))

    # Every combination that actually occurs, ordered: all exclusives first (in
    # command-line order), then shared combinations by increasing breadth. This
    # degenerates to the original Only/Shared/Only bar chart for two sources.
    combo_counts = cluster_src_sets.value_counts()
    exclusives = [frozenset([s]) for s in sources if frozenset([s]) in combo_counts.index]
    shared_combos = sorted(
        (c for c in combo_counts.index if len(c) > 1),
        key=lambda c: (len(c), [sources.index(s) for s in sorted(c, key=sources.index)]),
    )
    ordered_combos = exclusives + shared_combos

    def _combo_label(combo):
        if len(combo) == 1:
            return f"Only\n{next(iter(combo))}"
        ordered = sorted(combo, key=sources.index)
        if len(combo) == len(sources):
            return "Shared by all\n(" + " + ".join(ordered) + ")"
        return "Shared\n" + " + ".join(ordered)

    labels = [_combo_label(c) for c in ordered_combos]
    values = [int(combo_counts[c]) for c in ordered_combos]
    # Exclusive bars get distinct colours; shared bars use a common teal.
    base_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948"]
    colors = [
        base_colors[sources.index(next(iter(c))) % len(base_colors)] if len(c) == 1 else "#76b7b2"
        for c in ordered_combos
    ]

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val), ha='center', va='bottom')
    ax.set_ylabel("Number of Clusters")
    ax.set_title("Cross-Source Cluster Overlap\n(shared = structural motif found by more than one generator)")
    ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot5_cross_source_overlap.png", dpi=300)
    plt.close()

    # The same numbers as a table, which is easier to read than the bars at N>=3.
    overlap_df = pd.DataFrame({
        "sources": [" + ".join(sorted(c, key=sources.index)) for c in ordered_combos],
        "n_sources": [len(c) for c in ordered_combos],
        "num_clusters": values,
    })
    overlap_df.to_csv(OUTPUT_DIR / "cross_source_overlap.csv", index=False)

# ── Plot 6: Per-experiment comparison (if multiple experiments) ─
# Direct head-to-head for each experimental condition.
_n_unique_exp = exp_stats["experiment"].nunique()
if _n_unique_exp > args.max_experiments_plot:
    print(
        f"Skipping plot 6: {_n_unique_exp} experiments exceeds "
        f"--max-experiments-plot ({args.max_experiments_plot}); "
        f"see experiment_statistics.csv instead."
    )
elif _n_unique_exp > 1:
    n_exp = len(exp_stats)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_exp * 1.2), 5))
    for ax, metric, label in zip(
        axes,
        ["cluster_hit_fraction", "singleton_fraction"],
        ["Cluster Hit Fraction", "Singleton Fraction"],
    ):
        sns.barplot(data=exp_stats, x="experiment", y=metric, hue="source", hue_order=PRESENT_SOURCES, ax=ax)
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
