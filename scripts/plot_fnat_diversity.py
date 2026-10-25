#!/usr/bin/env python3
"""
Slide figures for interface_fnat_diversity.py output.

Reads an existing ``analysis/fnat/`` directory and renders three presentation
figures that answer, in order:

    1. epitope_novelty.png    Are new epitopes discovered?
    2. mechanism.png          Same epitope -- same mechanism?
    3. fold_vs_epitope.png    Fold diversity is not epitope diversity

Deliberately spare: title + short axis labels only, no captions, no annotations,
no stat boxes. Colour-blind safe palette. Everything else the speaker says aloud.

Consumes only files interface_fnat_diversity.py already writes:
    designs.csv               generation + epitope residues per design
    pairwise.csv.gz           every pair (fnat, epitope Jaccard, restype agreement)
    fold_cluster_epitope.csv  optional, for figure 3

Usage
-----
    python scripts/plot_fnat_diversity.py \\
        --in-dir /nfs/students/matscher/reseed_output/cd19_v4/merged/runs/CD19/analysis/fnat

    # override the epitope threshold used for clustering (must match the run)
    python scripts/plot_fnat_diversity.py --in-dir ... --epitope-thresh 0.5
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# colour-blind safe
C_OLD = "#4C72B0"    # existing / conserved
C_NEW = "#C44E52"    # new / divergent
C_CHEM = "#DD8452"   # chemistry series
C_GREY = "#8C8C8C"


def _style(ax, xlabel: str, ylabel: str) -> None:
    """Axis labels only -- figures carry no title; the slide supplies it."""
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def parse_set(s) -> frozenset:
    if not isinstance(s, str) or not s.strip():
        return frozenset()
    return frozenset(int(x) for x in s.split(",") if x.strip())


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def greedy_clusters(fps: List[frozenset], thresh: float) -> List[int]:
    reps: List[frozenset] = []
    assign: List[int] = []
    for fp in fps:
        for ci, rep in enumerate(reps):
            if jaccard(fp, rep) >= thresh:
                assign.append(ci)
                break
        else:
            reps.append(fp)
            assign.append(len(reps) - 1)
    return assign


# --------------------------------------------------------------------------------------
# figure 1 -- are new epitopes discovered?
# --------------------------------------------------------------------------------------

def fig_epitope_novelty(designs: pd.DataFrame, thresh: float,
                        out_png: Path, dpi: int) -> int:
    fps = [parse_set(s) for s in designs["epitope_residues"]]
    assign = greedy_clusters(fps, thresh)
    designs = designs.assign(epi=assign)
    founder_epis = set(designs[designs["generation"] == 0]["epi"])

    gens = sorted(designs["generation"].unique())
    old, new = [], []
    for g in gens:
        sub = designs[designs["generation"] == g]
        old.append(int((sub["epi"].isin(founder_epis)).sum()))
        new.append(int((~sub["epi"].isin(founder_epis)).sum()))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(gens))
    ax.bar(x, old, color=C_OLD, width=0.68, label="known epitope")
    ax.bar(x, new, bottom=old, color=C_NEW, width=0.68, label="new epitope")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(g)) for g in gens])
    _style(ax, "generation", "designs")
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  {out_png.name}: known={sum(old)} new={sum(new)} "
          f"({len(set(assign))} epitope clusters, {len(founder_epis)} in founders)")
    return len(set(assign))


# --------------------------------------------------------------------------------------
# figure 2 -- same epitope, same mechanism?
# --------------------------------------------------------------------------------------

def fig_mechanism(pw: pd.DataFrame, designs: pd.DataFrame, fnat_thresh: float,
                  out_png: Path, dpi: int) -> None:
    """
    Boxes: fnat of each design against its own lineage founder, index mode only
    (true CAPRI fnat -- binder numbering corresponds within a lineage).
    Line: fraction of shared epitope residues engaged by the SAME amino acid.

    High boxes + low line = same geometry, different chemistry.
    """
    gen0 = set(designs[designs["generation"] == 0]["sample_id"])
    vs_f = pw[(pw["same_lineage"].astype(str).str.lower() == "true")
              & (pw["sample_i"].isin(gen0) ^ pw["sample_j"].isin(gen0))].copy()
    if not len(vs_f):
        print("  mechanism.png: no founder pairs found -- skipped")
        return
    # generation of the non-founder side
    vs_f["generation"] = np.where(vs_f["sample_i"].isin(gen0),
                                 vs_f["gen_j"], vs_f["gen_i"])
    # Prefer whichever single engine produced the run. "dockq" is the reference
    # implementation; "index" is the internal positional fnat. Never mix modes --
    # they are on different scales (and "spatial" is not a reliable discriminator).
    use, mode = None, None
    for cand in ("dockq", "index"):
        m = vs_f[vs_f["fnat_mode"] == cand]
        if len(m):
            use, mode = m, cand
            break
    if use is None:
        use, mode = vs_f, "spatial"

    gens = sorted(int(g) for g in use["generation"].unique() if g > 0)
    data = [use[use["generation"] == g]["fnat_shared"].dropna().to_numpy() for g in gens]
    chem = [use[use["generation"] == g]["restype_agreement"].mean() for g in gens]
    keep = [i for i, a in enumerate(data) if len(a)]
    if not keep:
        print("  mechanism.png: no fnat values -- skipped")
        return
    gens = [gens[i] for i in keep]
    data = [data[i] for i in keep]
    chem = [chem[i] for i in keep]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(gens))
    bp = ax.boxplot(data, positions=x, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=2))
    for b in bp["boxes"]:
        b.set(facecolor=C_OLD, edgecolor=C_OLD, alpha=0.95)
    for k in ("whiskers", "caps"):
        for a in bp[k]:
            a.set(color=C_OLD)
    ax.plot(x, chem, "o-", color=C_CHEM, lw=2.5, ms=8, label="same amino acid")
    ax.axhline(fnat_thresh, ls="--", lw=1.2, color=C_GREY)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gens])
    # "fraction", not "fnat": the axis carries both the fnat boxes and the
    # amino-acid-agreement line, and both are 0-1 fractions.
    _style(ax, "generation", "fraction")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  {out_png.name}: fnat mode={mode}, "
          f"medians={[round(float(np.median(a)),3) for a in data]}, "
          f"chem={[round(float(c),3) for c in chem]}")


# --------------------------------------------------------------------------------------
# figure 3 -- fold diversity is not epitope diversity
# --------------------------------------------------------------------------------------

def fig_fold_vs_epitope(n_designs: int, n_fold: Optional[int], n_epi: int,
                        out_png: Path, dpi: int) -> None:
    labels, vals, colors = ["designs"], [n_designs], [C_GREY]
    if n_fold is not None:
        labels.append("folds")
        vals.append(n_fold)
        colors.append(C_OLD)
    labels.append("epitopes")
    vals.append(n_epi)
    colors.append(C_NEW)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.12, str(v), ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(vals) * 3)
    # NB: title must hold for BOTH runs. PDL1 has 22 folds -> 3 epitopes (folds
    # overstate diversity), but CD19 has 3 folds -> 4 epitopes (they do not). The
    # claim that generalises is the collapse of many designs onto few epitopes.
    _style(ax, "", "count")
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  {out_png.name}: designs={n_designs} folds={n_fold} epitopes={n_epi}")


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Slide figures for interface_fnat_diversity.py output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-dir", required=True,
                    help="an analysis/fnat directory written by interface_fnat_diversity.py")
    ap.add_argument("--out-dir", default=None, help="default = --in-dir")
    ap.add_argument("--epitope-thresh", type=float, default=0.5,
                    help="must match the value used for the run")
    ap.add_argument("--fnat-thresh", type=float, default=0.5,
                    help="reference line on the mechanism figure")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    ind = Path(args.in_dir).expanduser()
    out = Path(args.out_dir).expanduser() if args.out_dir else ind
    out.mkdir(parents=True, exist_ok=True)

    dpath = ind / "designs.csv"
    ppath = ind / "pairwise.csv.gz"
    if not dpath.exists():
        print(f"ERROR: {dpath} missing", file=sys.stderr)
        return 2
    designs = pd.read_csv(dpath)
    print(f"designs: {len(designs)}")

    n_epi = fig_epitope_novelty(designs, args.epitope_thresh,
                                out / "epitope_novelty.png", args.dpi)

    if ppath.exists():
        with gzip.open(ppath, "rt") as fh:
            pw = pd.read_csv(fh)
        fig_mechanism(pw, designs, args.fnat_thresh,
                      out / "mechanism.png", args.dpi)
    else:
        print(f"  mechanism.png: {ppath} missing (rerun without --no-pairwise-dump)")

    n_fold = None
    fpath = ind / "fold_cluster_epitope.csv"
    if fpath.exists():
        n_fold = int(pd.read_csv(fpath)["fold_cluster_id"].nunique())
    fig_fold_vs_epitope(len(designs), n_fold, n_epi,
                        out / "fold_vs_epitope.png", args.dpi)

    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
