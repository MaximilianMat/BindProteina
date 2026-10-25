"""
Slide figures for the re-seeding ablation (robust / aggregate across targets).

Reads one or two run trees (Branch A = random-seed, Branch B = interface-seed) laid out
as ``<branch>/runs/<target>/{records.parquet|csv, lineage.json, analysis/*.csv, gen*/}``
and writes the slide figures to ``--out``.

Figures
  fig_rq1_efficiency   per-target stacked bar of successes added per generation, with
                       total-samples-generated annotation; Branch-B targets get A vs B bars.
  fig_rq2_ss_inheritance   input(parent) SS -> output(progeny) SS, per-input then aggregated,
                       + progeny SS grouped by the seed's SS class.
  fig_rq1_novelty      cross-target cumulative new clusters vs generation (needs cluster_novelty.csv).
  fig_extinction       strains alive vs round per target (needs lineage.json).
  fig_outcome          per-target reseed success rate (the conditional-win conclusion figure).

Usage
  python scripts/reseed_slide_figures.py \
      --branch_a /nfs/students/matscher/reseed_output_redis \
      --branch_b /nfs/students/matscher/reseed_output_iface \
      --branch_b_targets NIPAH PDGFR \
      --out reseed_slides_figs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# make the repo root importable so `from pipeline.reseed...` works when the script is
# launched as `python scripts/reseed_slide_figures.py` (baseline stats need it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# easy -> hard reading order + difficulty labels for the slides
TARGET_ORDER = ["NPR3", "MUC1", "PDGFR", "HLADPA1", "ACTRIIA", "NIPAH", "CD164"]
DIFFICULTY = {"NPR3": "easy", "MUC1": "medium", "PDGFR": "medium", "HLADPA1": "medium",
              "ACTRIIA": "hard", "NIPAH": "hard", "CD164": "very hard"}

# Total seed-search samples generated per target (the baseline denominator). The scan
# only sees the fully-scored conditions, so these are supplied manually to get the true
# baseline success rate = PyRosetta successes / total generated.
REFERENCE_TOTALS = {
    "MUC1": 17600, "NIPAH": 17400, "PDGFR": 17400, "ACTRIIA": 17400,
    "NPR3": 14000, "CD164": 19500, "HLADPA1": 14600,
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "pass", "passed", "yes")


def target_dir(branch: str, target: str) -> Path:
    return Path(branch) / "runs" / target


def load_records(branch: str, target: str) -> Optional[pd.DataFrame]:
    d = target_dir(branch, target)
    for name in ("records.parquet", "records.csv"):
        p = d / name
        if p.exists():
            try:
                df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                df["_pass"] = df["passed_filter"].apply(_truthy)
                return df
            except Exception:  # noqa: BLE001
                continue
    return None


def generated_per_gen(branch: str, target: str, gens: List[int]) -> Dict[int, int]:
    """Total samples generated per generation (prescoring rows; fast, one CSV/gen)."""
    d = target_dir(branch, target)
    out: Dict[int, int] = {}
    for g in gens:
        if g == 0:
            continue
        pres = sorted((d / f"gen{g}" / "_scored").glob("*_prescoring_results.csv"))
        n = None
        if pres:
            try:
                n = len(pd.read_csv(pres[0]))
            except Exception:  # noqa: BLE001
                n = None
        if n is None:
            si = d / f"gen{g}" / "_scoring_input"
            n = len(list(si.glob("*.pdb"))) if si.is_dir() else 0
        out[g] = n
    return out


def analysis_csv(branch: str, target: str, name: str) -> Optional[pd.DataFrame]:
    p = target_dir(branch, target) / "analysis" / f"{name}.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:  # noqa: BLE001
            return None
    return None


def lineage(branch: str, target: str) -> dict:
    p = target_dir(branch, target) / "lineage.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def resolve_branch(branch_a, branch_b, branch_b_targets, target: str) -> str:
    """Which run tree to read for a target.

    The interface targets (NIPAH, PDGFR) are only interesting in their Branch-B
    (interface-restricted) arm — their random-seed Branch-A lineages went extinct and are
    not shown. Every figure reads Branch B for those targets and Branch A for the rest.
    """
    if branch_b and target in (branch_b_targets or []):
        return branch_b
    return branch_a


def branch_tag(branch_b, branch_b_targets, target: str) -> str:
    """Short suffix noting a target is shown via its interface arm."""
    return " · interface" if branch_b and target in (branch_b_targets or []) else ""


# --------------------------------------------------------------------------------------
# RQ1 — efficiency stacked bar
# --------------------------------------------------------------------------------------

def _succ_by_gen(df: pd.DataFrame) -> Dict[int, int]:
    return {int(g): int(sub["_pass"].sum())
            for g, sub in df.groupby("generation")}


_BASE_CACHE: Dict = {}


def baseline_stats(branch: str, target: str):
    """Seed-search baseline success rate, comparable to the ablation.

    Returns (pyrosetta_successes, total_generated, rate%) over the baseline's fully-scored
    conditions, where the denominator is the TOTAL GENERATED (prescoring rows), NOT the
    folding-passing subset in scoring_summary.csv — so the rate matches how the ablation
    rate is defined (pyrosetta pass / generated). Reads baseline_run_dir from the run's
    config.json. Best-effort; heavy one-time scan.
    """
    key = (branch, target)
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]
    res = None
    cfgp = target_dir(branch, target) / "config.json"
    if cfgp.exists():
        try:
            bdir = json.loads(cfgp.read_text()).get("baseline_run_dir")
            if bdir:
                succ = gen = 0
                for ss in Path(bdir).rglob("scoring_summary.csv"):
                    try:
                        df = pd.read_csv(ss)
                    except Exception:  # noqa: BLE001
                        continue
                    if "pyrosetta_filter_pass" not in df.columns:
                        continue
                    succ += int(df["pyrosetta_filter_pass"].apply(_truthy).sum())
                    # total generated for this condition = folded samples (prescoring rows)
                    pres = sorted(ss.parent.glob("*_prescoring_results.csv"))
                    if pres:
                        try:
                            gen += len(pd.read_csv(pres[0]))
                        except Exception:  # noqa: BLE001
                            gen += len(df)
                    else:
                        gen += len(df)
                # override the scanned denominator with the true total budget
                if target in REFERENCE_TOTALS:
                    gen = REFERENCE_TOTALS[target]
                if gen > 0:
                    res = (succ, gen, 100 * succ / gen)
        except Exception as e:  # noqa: BLE001
            print(f"  baseline unavailable for {target}: {e}")
    _BASE_CACHE[key] = res
    return res


def fig_rq1_efficiency(branch_a, branch_b, targets, branch_b_targets, out: Path,
                       with_baseline: bool = True):
    cmap = plt.get_cmap("viridis")
    # ---- gather per target a list of side-by-side entries ----
    # entry = dict(label, kind, sbg, gen_tot, tot, rate)  (kind: 'baseline'|'reseed')
    entries_by_t: Dict[str, list] = {}
    max_gen, max_total = 0, 1
    for t in targets:
        br = resolve_branch(branch_a, branch_b, branch_b_targets, t)
        reseed_ents, r_tot = [], None
        df = load_records(br, t)
        if df is not None and len(df):
            sbg = _succ_by_gen(df)
            gens = sorted(sbg)
            gen_tot = generated_per_gen(br, t, gens)
            tot = sum(gen_tot.values())
            succ_prog = sum(v for g, v in sbg.items() if g > 0)
            reseed_ents.append(dict(label="", kind="reseed", sbg=sbg, gen_tot=gen_tot,
                                    tot=tot, rate=100 * succ_prog / tot if tot else 0.0))
            r_tot = tot
            max_gen = max(max_gen, max(gens) if gens else 0)
            max_total = max(max_total, sum(sbg.values()))
        ents = []
        if with_baseline:
            b = baseline_stats(br, t)
            if b:
                succ, samp, rate = b
                # show the baseline at MATCHED compute (its rate applied to the re-seed
                # sample budget) so bar heights are directly comparable -> the efficiency
                # gain is visual; raw numbers kept in the annotation.
                height = (rate / 100.0) * r_tot if r_tot else succ
                ents.append(dict(label="seed-search", kind="baseline",
                                 height=height, raw=(succ, samp, rate)))
                max_total = max(max_total, height)
        ents += reseed_ents
        entries_by_t[t] = ents

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(targets)), 6))
    seg_label_thresh = 0.035 * max_total
    xticks, xlabels = [], []
    # keep a target's bars tight together (< the gap to the next target) so it's clear
    # which bars belong to which target.
    group_w, x = 0.75, 0.0
    for t in targets:
        ents = entries_by_t[t]
        n = len(ents)
        offs = np.linspace(-group_w / 2, group_w / 2, n) if n > 1 else [0.0]
        bw = (group_w / max(n, 1)) * 0.92 if n > 1 else 0.7
        for e, dx in zip(ents, offs):
            cx = x + dx
            if e["kind"] == "baseline":
                succ, samp, rate = e["raw"]
                ax.bar(cx, e["height"], width=bw, color="0.6", hatch="//",
                       edgecolor="white")
                ax.text(cx, e["height"], f"seed-search\n{rate:.2f}%\n({succ}/{samp:,})",
                        ha="center", va="bottom", fontsize=6.5, color="0.3")
                continue
            bottom = 0.0
            for g in sorted(e["sbg"]):
                seg = e["sbg"][g]
                color = "0.8" if g == 0 else cmap(g / max(1, max_gen))
                ax.bar(cx, seg, width=bw, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                # per-round success percentage inside the box
                if g > 0 and e["gen_tot"].get(g) and seg >= seg_label_thresh:
                    rr = 100 * seg / e["gen_tot"][g]
                    ax.text(cx, bottom + seg / 2, f"{rr:.1f}%", ha="center",
                            va="center", fontsize=6.5, color="white", fontweight="bold")
                bottom += seg
            pref = f"{e['label']}  " if e["label"] else ""
            ax.text(cx, bottom, f"{pref}{e['tot']:,}·{e['rate']:.1f}%",
                    ha="center", va="bottom", fontsize=7)
        xticks.append(x)
        tag = branch_tag(branch_b, branch_b_targets, t)
        xlabels.append(f"{t}\n({DIFFICULTY.get(t, '?')}{tag})")
        x += 1.2

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("successful binders")
    ax.set_title("RQ1 — successes per generation vs seed-search baseline (at matched compute)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.6", hatch="//"),
               plt.Rectangle((0, 0), 1, 1, color="0.8")]
    labels = ["seed-search baseline", "gen 0 (founders, free)"]
    for g in range(1, max_gen + 1):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(g / max(1, max_gen))))
        labels.append(f"gen {g}")
    ax.legend(handles, labels, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.005, 1.0), frameon=True)
    fig.tight_layout()
    fig.savefig(out / "fig_rq1_efficiency.png", dpi=200)
    plt.close(fig)
    print("wrote fig_rq1_efficiency.png")


# --------------------------------------------------------------------------------------
# RQ2 — SS improvement over generations (two per-target grid options + a trend line)
# --------------------------------------------------------------------------------------

def _grid_axes(n, fig_w=3.2, fig_h=2.9, ncols=4):
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w * ncols, fig_h * nrows),
                             squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat


def _ss_success(df):
    return df[df["_pass"]].dropna(subset=["ss_helix_frac", "ss_sheet_frac", "ss_loop_frac"])


def fig_rq2_ss_stacked_grid(branch_a, branch_b, branch_b_targets, targets, out: Path):
    """Per-target grid of stacked helix/sheet/loop bars over generations."""
    def _br(t):
        return resolve_branch(branch_a, branch_b, branch_b_targets, t)
    have = [t for t in targets if load_records(_br(t), t) is not None]
    if not have:
        return
    fig, axes = _grid_axes(len(have))
    cols = {"ss_helix_frac": ("#4477AA", "helix"),
            "ss_sheet_frac": ("#EE6677", "sheet"),
            "ss_loop_frac": ("#BBBBBB", "loop")}
    for ax, t in zip(axes, have):
        succ = _ss_success(load_records(_br(t), t))
        tag = branch_tag(branch_b, branch_b_targets, t)
        if not len(succ):
            ax.set_title(f"{t} (no data)", fontsize=9)
            continue
        m = succ.groupby("generation")[list(cols)].mean()
        bottom = np.zeros(len(m))
        for c, (color, _) in cols.items():
            ax.bar(m.index.astype(int), m[c], bottom=bottom, color=color, width=0.8)
            bottom += m[c].values
        ax.set_ylim(0, 1)
        ax.set_xticks(m.index.astype(int))
        ax.set_title(f"{t} ({DIFFICULTY.get(t, '?')}{tag})", fontsize=9)
        ax.set_xlabel("generation", fontsize=8)
    axes[0].set_ylabel("binder SS fraction", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in cols.values()]
    fig.legend(handles, [lab for _, lab in cols.values()], loc="upper right",
               ncol=3, fontsize=9)
    fig.suptitle("RQ2 — secondary-structure composition over generations (per target)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "fig_rq2_ss_stacked_grid.png", dpi=200)
    plt.close(fig)
    print("wrote fig_rq2_ss_stacked_grid.png")


def fig_rq2_ss_scatter_grid(branch_a, branch_b, branch_b_targets, targets, out: Path):
    """Per-target grid of helix×sheet scatter, points shaded by generation."""
    def _br(t):
        return resolve_branch(branch_a, branch_b, branch_b_targets, t)
    have = [t for t in targets if load_records(_br(t), t) is not None]
    if not have:
        return
    fig, axes = _grid_axes(len(have))
    cmap = plt.get_cmap("viridis")
    sc = None
    for ax, t in zip(axes, have):
        succ = _ss_success(load_records(_br(t), t))
        tag = branch_tag(branch_b, branch_b_targets, t)
        if not len(succ):
            ax.set_title(f"{t} (no data)", fontsize=9)
            continue
        base = succ[succ["generation"] == 0]
        prog = succ[succ["generation"] > 0]
        mg = int(succ["generation"].max())
        if len(base):
            ax.scatter(base["ss_helix_frac"], base["ss_sheet_frac"], c="lightgrey",
                       s=12, edgecolors="none")
        if len(prog):
            sc = ax.scatter(prog["ss_helix_frac"], prog["ss_sheet_frac"],
                            c=prog["generation"], cmap=cmap, vmin=1, vmax=max(2, mg),
                            s=16, edgecolors="k", linewidths=0.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{t} ({DIFFICULTY.get(t, '?')}{tag})", fontsize=9)
        ax.set_xlabel("helix", fontsize=8); ax.set_ylabel("sheet", fontsize=8)
    # leave room on the right for a dedicated colorbar axis and between rows for titles
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.90,
                        hspace=0.55, wspace=0.30)
    if sc is not None:
        cax = fig.add_axes([0.925, 0.15, 0.015, 0.7])
        fig.colorbar(sc, cax=cax, label="generation")
    fig.suptitle("RQ2 — SS drift over generations (grey = gen-0 founders)", fontsize=12,
                 y=0.965)
    fig.savefig(out / "fig_rq2_ss_scatter_grid.png", dpi=200)
    plt.close(fig)
    print("wrote fig_rq2_ss_scatter_grid.png")


def fig_rq2_ss_beta_trend(branch_a, branch_b, branch_b_targets, targets, out: Path):
    """Compact alternative: mean β-sheet fraction vs generation, one line per target."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    any_data = False
    for i, t in enumerate(targets):
        br = resolve_branch(branch_a, branch_b, branch_b_targets, t)
        df = load_records(br, t)
        if df is None:
            continue
        succ = _ss_success(df)
        if not len(succ):
            continue
        any_data = True
        tag = branch_tag(branch_b, branch_b_targets, t)
        m = succ.groupby("generation")["ss_sheet_frac"].mean()
        ax.plot(m.index, m.values, marker="o", color=cmap(i % 10),
                label=f"{t} ({DIFFICULTY.get(t, '?')}{tag})")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("generation"); ax.set_ylabel("mean binder β-sheet fraction")
    ax.set_ylim(0, None)
    ax.set_title("RQ2 — β-sheet content over generations (per target)")
    ax.legend(fontsize=8)
    if any_data:
        fig.tight_layout(); fig.savefig(out / "fig_rq2_ss_beta_trend.png", dpi=200)
        print("wrote fig_rq2_ss_beta_trend.png")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# RQ2 — input SS -> output SS
# --------------------------------------------------------------------------------------

def _dominant(ss_json) -> Optional[str]:
    try:
        arr = json.loads(ss_json) if isinstance(ss_json, str) else ss_json
    except (TypeError, json.JSONDecodeError):
        return None
    if not arr:
        return None
    return Counter(arr).most_common(1)[0][0]


def fig_rq2_ss_inheritance(branch_a, branch_b, branch_b_targets, targets, out: Path):
    # pool progeny across all targets; join each progeny to its parent's SS.
    pairs = []       # (parent_sheet, progeny_sheet, progeny_helix)
    seedcls = []     # (dominant seed SS class, progeny_sheet, progeny_helix)
    for t in targets:
        br = resolve_branch(branch_a, branch_b, branch_b_targets, t)
        df = load_records(br, t)
        if df is None or not len(df):
            continue
        df = df.dropna(subset=["ss_sheet_frac", "ss_helix_frac"])
        parent_sheet = dict(zip(df["sample_id"], df["ss_sheet_frac"]))
        prog = df[(df["generation"] > 0) & (df["_pass"])]
        for _, r in prog.iterrows():
            ps = parent_sheet.get(r.get("parent_id"))
            if ps is not None and pd.notna(ps):
                pairs.append((float(ps), float(r["ss_sheet_frac"]), float(r["ss_helix_frac"])))
            dc = _dominant(r.get("seed_res_ss"))
            if dc:
                seedcls.append((dc, float(r["ss_sheet_frac"]), float(r["ss_helix_frac"])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left: per-input-sample inheritance (aggregate progeny per parent)
    if pairs:
        P = pd.DataFrame(pairs, columns=["in_sheet", "out_sheet", "out_helix"])
        by_parent = P.groupby("in_sheet")["out_sheet"].mean().reset_index()
        ax1.scatter(P["in_sheet"], P["out_sheet"], s=8, alpha=0.15, color="tab:blue")
        # binned mean
        bins = np.linspace(0, 1, 11)
        P["bin"] = pd.cut(P["in_sheet"], bins)
        bm = P.groupby("bin", observed=True)["out_sheet"].mean()
        centers = [iv.mid for iv in bm.index]
        ax1.plot(centers, bm.values, "o-", color="tab:red", lw=2, label="binned mean")
        ax1.plot([0, 1], [0, 1], "--", color="0.5", label="y = x")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_aspect("equal")
    ax1.set_xlabel("parent β-sheet fraction")
    ax1.set_ylabel("child β-sheet fraction")
    ax1.set_title("SS inheritance: parent → child (pooled)")
    ax1.legend(fontsize=8)

    # right: progeny sheet fraction grouped by seed SS class
    if seedcls:
        S = pd.DataFrame(seedcls, columns=["seed_ss", "out_sheet", "out_helix"])
        order = [c for c in ("E", "H", "L") if c in set(S["seed_ss"])]
        data = [S[S["seed_ss"] == c]["out_sheet"].values for c in order]
        ax2.violinplot(data, showmeans=True)
        ax2.set_xticks(range(1, len(order) + 1))
        ax2.set_xticklabels([{"E": "sheet-cut", "H": "helix-cut", "L": "loop-cut"}[c]
                             for c in order])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("child β-sheet fraction")
    ax2.set_title("Child SS by seed's SS class")
    fig.tight_layout()
    fig.savefig(out / "fig_rq2_ss_inheritance.png", dpi=200)
    plt.close(fig)
    print("wrote fig_rq2_ss_inheritance.png")


# --------------------------------------------------------------------------------------
# Completing figures: novelty, extinction, outcome
# --------------------------------------------------------------------------------------

def fig_rq1_novelty(branch_a, branch_b, targets, branch_b_targets, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    all_curves, present, missing = [], [], []
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(targets):
        br = resolve_branch(branch_a, branch_b, branch_b_targets, t)
        nv = analysis_csv(br, t, "cluster_novelty")
        c = cmap(i % 10)
        tag = branch_tag(branch_b, branch_b_targets, t)
        if nv is not None and "cumulative_novel_clusters" in nv.columns and len(nv):
            present.append(t)
            ax.plot(nv["generation"], nv["cumulative_novel_clusters"], color=c, lw=1.6,
                    marker="o", ms=4, alpha=0.9,
                    label=f"{t} ({DIFFICULTY.get(t, '?')}{tag})")
            all_curves.append(nv[["generation", "cumulative_novel_clusters"]])
        else:
            missing.append(t)
    if all_curves:
        allc = pd.concat(all_curves)
        mean = allc.groupby("generation")["cumulative_novel_clusters"].mean()
        ax.plot(mean.index, mean.values, color="black", lw=3, marker="o",
                label="mean across targets", zorder=5)
    ax.set_xlabel("generation")
    ax.set_ylabel("cumulative NEW clusters\n(new clusters compared to the full seed-search pool)")
    ax.set_title("RQ1 — novelty: newly discovered clusters not present in the seed-search set")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole-number generations only
    ax.legend(fontsize=8, loc="upper left")
    if present:
        fig.tight_layout(); fig.savefig(out / "fig_rq1_novelty.png", dpi=200)
        print(f"wrote fig_rq1_novelty.png (A targets: {present})")
    if missing:
        print(f"  NOTE novelty missing for {missing} — run cluster novelty first: "
              f"`python -m pipeline.reseed.analysis --run_dir <branch>/runs/<T> "
              f"--settings settings_reseed/<T>.json --foldseek ./bin/foldseek`")
    plt.close(fig)


def fig_extinction(branch_a, branch_b, targets, branch_b_targets, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in targets:
        br = resolve_branch(branch_a, branch_b, branch_b_targets, t)
        lin = lineage(br, t)
        if not lin:
            continue
        gens = sorted({int(k) for s in lin.values() for k in s.get("rounds", {})})
        alive = [sum(1 for s in lin.values() if s.get("rounds", {}).get(str(g)) is not None)
                 for g in gens]
        tag = branch_tag(branch_b, branch_b_targets, t)
        ax.plot([0] + gens, [len(lin)] + alive, marker="o",
                label=f"{t} ({DIFFICULTY.get(t,'?')}{tag})")
    ax.set_xlabel("round")
    ax.set_ylabel("strains alive")
    ax.set_title("Lineage extinction per target (mechanism behind the conditional result)")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole-number rounds only
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_extinction.png", dpi=200)
    plt.close(fig)
    print("wrote fig_extinction.png")


def fig_outcome(branch_a, branch_b, branch_b_targets, targets, out: Path):
    rows = []
    for t in targets:
        br = resolve_branch(branch_a, branch_b, branch_b_targets, t)
        df = load_records(br, t)
        if df is None:
            continue
        gens = sorted(int(g) for g in df["generation"].unique())
        gt = generated_per_gen(br, t, gens)
        tot = sum(gt.values())
        succ = int(df[(df["generation"] > 0) & df["_pass"]]["_pass"].sum())
        rows.append({"target": t, "rate": 100 * succ / tot if tot else 0.0,
                     "succ": succ, "difficulty": DIFFICULTY.get(t, "?")})
    if not rows:
        return
    R = pd.DataFrame(rows)
    colors = {"easy": "#2a9d8f", "medium": "#8ab17d", "hard": "#e9c46a", "very hard": "#e76f51"}
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(R["target"], R["rate"], color=[colors.get(d, "0.6") for d in R["difficulty"]])
    for i, r in R.iterrows():
        ax.text(i, r["rate"], f"{r['rate']:.2f}%\n({r['succ']})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("re-seed success rate (progeny)")
    ax.set_title("Per-target outcome — the conditional win (colour = difficulty)")
    fig.tight_layout(); fig.savefig(out / "fig_outcome.png", dpi=200)
    plt.close(fig)
    print("wrote fig_outcome.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch_a", required=True, help="random-seed run tree (has runs/<target>/)")
    ap.add_argument("--branch_b", default=None, help="interface-seed run tree")
    ap.add_argument("--branch_b_targets", nargs="*", default=["NIPAH", "PDGFR"])
    ap.add_argument("--targets", nargs="*", default=None, help="default: all present in branch_a")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_baseline", action="store_true",
                    help="skip the seed-search baseline bar (avoids the heavy baseline scan)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.targets:
        targets = args.targets
    else:
        present = {p.name for p in (Path(args.branch_a) / "runs").glob("*") if p.is_dir()}
        targets = [t for t in TARGET_ORDER if t in present] + \
                  [t for t in sorted(present) if t not in TARGET_ORDER]

    print(f"Targets: {targets}")
    fig_rq1_efficiency(args.branch_a, args.branch_b, targets, args.branch_b_targets, out,
                       with_baseline=not args.no_baseline)
    fig_rq2_ss_inheritance(args.branch_a, args.branch_b, args.branch_b_targets, targets, out)
    fig_rq1_novelty(args.branch_a, args.branch_b, targets, args.branch_b_targets, out)
    fig_extinction(args.branch_a, args.branch_b, targets, args.branch_b_targets, out)
    fig_outcome(args.branch_a, args.branch_b, args.branch_b_targets, targets, out)
    fig_rq2_ss_stacked_grid(args.branch_a, args.branch_b, args.branch_b_targets, targets, out)
    fig_rq2_ss_scatter_grid(args.branch_a, args.branch_b, args.branch_b_targets, targets, out)
    fig_rq2_ss_beta_trend(args.branch_a, args.branch_b, args.branch_b_targets, targets, out)
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
