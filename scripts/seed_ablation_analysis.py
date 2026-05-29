"""
Protein Binder Ablation Study Visualization Script
==================================================

Loads ablation results from a JSON file and generates:
1. Faceted heatmaps (auto-discovered 2D grids, faceted by remaining dims)
2. Marginal effect plots (one per discovered dimension)
3. Statistical summaries (CSV)

Dimensions are treated as *sparse* – different subtrees may have different
sets of parameters.  The script auto-detects which dimensions exist in the
data and adapts the visualizations accordingly.
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations


# Dimensions that carry ablation semantics (used for marginal effects & facets)
# Anything in the JSON *except* these keys is treated as a dimension.
_META_KEYS = {"success_rate", "n_samples", "n_success", "condition_path", "target"}


def _discover_dimensions(df: pd.DataFrame) -> list[str]:
    """Return the list of dimensions present in the DataFrame, ordered by
    number of unique values (fewest first, so categorical dims come first)."""
    dims = [c for c in df.columns if c not in _META_KEYS]
    # Sort: fewer unique values first (good for faceting), ties by name
    dims.sort(key=lambda d: (df[d].nunique(), d))
    return dims


def _is_numeric_dim(series: pd.Series) -> bool:
    """Check if a dimension is numeric (for line plots vs bar plots)."""
    try:
        pd.to_numeric(series)
        return True
    except (ValueError, TypeError):
        return False


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Try to convert to numeric; return as-is if not possible."""
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


# ============================================================
# HEATMAPS: pick two numeric-ish dimensions as x/y, facet rest
# ============================================================

def _pick_heatmap_axes(dims: list[str], df: pd.DataFrame):
    """Pick the best two dimensions for the heatmap grid.
    Prefer numeric dimensions with more than one unique value.
    Returns (row_dim, col_dim) or None if not enough dims."""
    numeric_dims = [d for d in dims if _is_numeric_dim(df[d]) and df[d].nunique() > 1]
    if len(numeric_dims) >= 2:
        # Prefer distance on y-axis, rotation/angle on x-axis
        priority = {"distance": 0, "rotation": 1, "angle": 1, "tilt": 2}
        numeric_dims.sort(key=lambda d: (priority.get(d, 99), d))
        return numeric_dims[0], numeric_dims[1]
    # If only one numeric, pair with the first categorical
    cat_dims = [d for d in dims if not _is_numeric_dim(df[d]) and df[d].nunique() > 1]
    if numeric_dims and cat_dims:
        return cat_dims[0], numeric_dims[0]
    if len(cat_dims) >= 2:
        return cat_dims[0], cat_dims[1]
    return None


def plot_heatmaps(df: pd.DataFrame, dims: list[str], output_dir: Path):
    """Generate faceted heatmaps.  Auto-selects two dims for grid axes and
    facets along all remaining dimensions."""
    axes_pair = _pick_heatmap_axes(dims, df)
    if axes_pair is None:
        print("  ⏭ Skipping heatmaps: need at least 2 dimensions with >1 value.")
        return

    row_dim, col_dim = axes_pair
    facet_dims = [d for d in dims if d not in (row_dim, col_dim) and df[d].nunique() > 1]

    if facet_dims:
        # Build facet groups from the Cartesian product of remaining dims
        facet_groups = df.groupby(facet_dims, dropna=False)
    else:
        facet_groups = [(None, df)]

    facet_list = list(facet_groups)
    n_facets = len(facet_list)

    if n_facets == 0:
        return

    ncols = min(3, n_facets)
    nrows = (n_facets + ncols - 1) // ncols

    fig, axes_arr = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    vmin = df["success_rate"].min()
    vmax = df["success_rate"].max()

    for idx, (facet_key, subset) in enumerate(facet_list):
        ax = axes_arr[idx // ncols, idx % ncols]

        # Coerce to numeric for pivot
        subset = subset.copy()
        subset[row_dim] = _coerce_numeric(subset[row_dim])
        subset[col_dim] = _coerce_numeric(subset[col_dim])

        try:
            pivot = subset.pivot_table(
                index=row_dim, columns=col_dim,
                values="success_rate", aggfunc="mean",
            )
        except Exception:
            ax.set_title("(pivot failed)")
            ax.axis("off")
            continue

        # Sort axes numerically if possible
        try:
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot[sorted(pivot.columns, key=lambda x: float(x))]
        except (ValueError, TypeError):
            pass

        sns.heatmap(
            pivot, ax=ax, cmap="viridis",
            annot=True, fmt=".2f",
            vmin=vmin, vmax=vmax,
            cbar=(idx == 0),
        )

        if facet_key is not None:
            if isinstance(facet_key, tuple):
                title = " | ".join(f"{d}={v}" for d, v in zip(facet_dims, facet_key))
            else:
                title = f"{facet_dims[0]}={facet_key}"
        else:
            title = "All data"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(col_dim)
        ax.set_ylabel(row_dim)

    # Blank out unused axes
    for idx in range(n_facets, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")

    plt.suptitle(f"Success Rate: {row_dim} × {col_dim}", fontsize=16)
    path = output_dir / "heatmaps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Heatmaps → {path}")


# ============================================================
# MARGINAL EFFECTS: one subplot per dimension
# ============================================================

def plot_marginal_effects(df: pd.DataFrame, dims: list[str], output_dir: Path):
    """Generate a single figure with one subplot per dimension showing
    the marginal effect on success rate."""
    # Only plot dimensions with >1 unique value
    plot_dims = [d for d in dims if df[d].nunique() > 1]
    if not plot_dims:
        print("  ⏭ Skipping marginal effects: no varying dimensions.")
        return

    n = len(plot_dims)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, dim in zip(axes, plot_dims):
        stats = df.groupby(dim)["success_rate"].agg(["mean", "std", "count"]).reset_index()
        stats["se"] = stats["std"] / np.sqrt(stats["count"])
        stats[dim] = _coerce_numeric(stats[dim])

        # Sort
        try:
            stats = stats.sort_values(dim, key=lambda s: pd.to_numeric(s, errors="coerce"))
        except Exception:
            pass

        if _is_numeric_dim(df[dim]) and df[dim].nunique() > 2:
            # Line plot for numeric dims
            ax.errorbar(
                stats[dim].astype(float), stats["mean"],
                yerr=stats["se"], marker="o", capsize=3,
            )
        else:
            # Bar plot for categorical / low-cardinality dims
            x = range(len(stats))
            ax.bar(x, stats["mean"], yerr=stats["se"], capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(stats[dim].astype(str), rotation=25, ha="right")

        ax.set_title(f"{dim} effect")
        ax.set_xlabel(dim)
        ax.set_ylabel("Mean Success Rate")
        ax.set_ylim(bottom=0)

    plt.suptitle("Marginal Effects on Success Rate", fontsize=16)
    path = output_dir / "marginal_effects.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Marginal effects → {path}")


# ============================================================
# PAIRWISE INTERACTION HEATMAPS (optional, if ≥2 dims)
# ============================================================

def plot_pairwise_interactions(df: pd.DataFrame, dims: list[str], output_dir: Path):
    """For each pair of dimensions, produce a small heatmap of mean success rate."""
    varying = [d for d in dims if df[d].nunique() > 1]
    pairs = list(combinations(varying, 2))
    if len(pairs) == 0:
        return

    # Limit to at most 10 pairs
    pairs = pairs[:10]
    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes_arr = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows),
        squeeze=False, constrained_layout=True,
    )

    for idx, (d1, d2) in enumerate(pairs):
        ax = axes_arr[idx // ncols, idx % ncols]
        pivot = df.pivot_table(index=d1, columns=d2, values="success_rate", aggfunc="mean")
        try:
            pivot = pivot.sort_index()
            pivot = pivot[sorted(pivot.columns, key=lambda x: (float(x) if str(x).replace(".", "").replace("-", "").isdigit() else x))]
        except Exception:
            pass
        sns.heatmap(pivot, ax=ax, cmap="viridis", annot=True, fmt=".2f", cbar=False)
        ax.set_title(f"{d1} × {d2}", fontsize=10)

    for idx in range(n, nrows * ncols):
        axes_arr[idx // ncols, idx % ncols].axis("off")

    plt.suptitle("Pairwise Interaction Heatmaps", fontsize=16)
    path = output_dir / "pairwise_interactions.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Pairwise interactions → {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot seed ablation results.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    parser.add_argument("--output", required=True, help="Path to the output folder for plots")
    parser.add_argument(
        "--residue",
        help="Optional residue filter (substring match on condition_path, e.g. 'hs60')",
    )
    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. LOAD DATA ─────────────────────────────────────────────
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        results = json.load(f)

    df = pd.DataFrame(results)
    if len(df) == 0:
        raise ValueError("No data found in JSON.")

    # Apply residue filter (simple substring match on condition_path)
    if args.residue:
        mask = df["condition_path"].str.contains(args.residue, case=False, na=False)
        df = df[mask].reset_index(drop=True)
        print(f"  Filtered to {len(df)} records matching '{args.residue}'")
        if len(df) == 0:
            raise ValueError(f"No records match residue filter '{args.residue}'")

    # ── 2. DISCOVER DIMENSIONS ───────────────────────────────────
    dims = _discover_dimensions(df)
    print(f"Discovered dimensions ({len(dims)}): {dims}")
    for d in dims:
        vals = sorted(df[d].dropna().unique(), key=lambda v: (float(v) if str(v).replace(".", "").replace("-", "").isdigit() else str(v)))
        print(f"  {d}: {vals}")

    # ── 3. HEATMAPS ──────────────────────────────────────────────
    print("\nGenerating heatmaps...")
    plot_heatmaps(df, dims, output_dir)

    # ── 4. MARGINAL EFFECTS ──────────────────────────────────────
    print("\nGenerating marginal effect plots...")
    plot_marginal_effects(df, dims, output_dir)

    # ── 5. PAIRWISE INTERACTIONS ─────────────────────────────────
    print("\nGenerating pairwise interaction plots...")
    plot_pairwise_interactions(df, dims, output_dir)

    # ── 6. SUMMARY TABLE ─────────────────────────────────────────
    group_cols = [d for d in dims if d in df.columns]
    summary = df.groupby(group_cols, dropna=False)["success_rate"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.columns = group_cols + ["mean_success_rate", "std_success_rate", "n_conditions"]
    summary_path = output_dir / "summary_stats.csv"
    summary.to_csv(summary_path, index=False)

    # ── 7. TOP / BOTTOM CONDITIONS ───────────────────────────────
    df_sorted = df.sort_values("success_rate", ascending=False)
    top_path = output_dir / "top_conditions.csv"
    df_sorted.head(20).to_csv(top_path, index=False)

    print(f"\n=== SUMMARY ===")
    print(f"  Total records: {len(df)}")
    print(f"  Dimensions: {dims}")
    print(f"  Summary table → {summary_path}")
    print(f"  Top conditions → {top_path}")


if __name__ == "__main__":
    main()