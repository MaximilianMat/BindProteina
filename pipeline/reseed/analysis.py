"""
Per-generation analysis for the re-seeding ablation (spec §9.2-9.4, RQ1/RQ2).

Consumes ``runs/<target>/records.parquet`` (+ the archived per-sample PDBs) and produces
CSVs and PNGs under ``runs/<target>/analysis/``:

  RQ1  cumulative successes vs cumulative compute (re-seed vs baseline);
       structural diversity per generation (FoldSeek clustering + mean pairwise TM);
       sequence diversity per generation (lightweight pairwise identity);
       extinction rate + near-copy (novelty) fraction per generation.
  RQ2  secondary-structure distribution per generation, its shift (KS / earth-mover),
       and progeny SS-delta as a function of the cut's SS class (seed_res_ss).

FoldSeek is invoked as a subprocess; its binary path is configurable (default resolves
``./bin/foldseek`` then ``foldseek`` on PATH). Runs headless (Agg backend).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_records(run_dir: str) -> pd.DataFrame:
    run = Path(run_dir)
    for name in ("records.parquet", "records.csv"):
        p = run / name
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"No records.parquet/csv in {run}")


def _sample_pdb_path(run_dir: Path, row) -> Optional[Path]:
    """Archived complex PDB for a scored/founder sample."""
    gen = int(row["generation"])
    if gen == 0:
        # founders live directly under gen0/
        cands = list((run_dir / "gen0").glob(f"strain{int(row['strain_id']):02d}_*.pdb"))
        return cands[0] if cands else None
    cut = row.get("cut_id")
    m = re.search(r"_sample_(\d+)", str(row["sample_id"]))
    idx = m.group(1) if m else "000"
    p = run_dir / f"gen{gen}" / f"strain{int(row['strain_id']):02d}" / f"cut{cut}" / f"sample{idx}.pdb"
    return p if p.exists() else None


# --------------------------------------------------------------------------------------
# FoldSeek structural diversity (spec §9.2)
# --------------------------------------------------------------------------------------

def _resolve_foldseek(foldseek_bin: Optional[str]) -> Optional[str]:
    if foldseek_bin and Path(foldseek_bin).exists():
        return foldseek_bin
    for cand in ("./bin/foldseek", shutil.which("foldseek")):
        if cand and Path(cand).exists():
            return cand
    return shutil.which("foldseek")


def _safe_name(sample_id: str) -> str:
    """Filesystem-safe stem for a sample/founder id (founder ids contain '/')."""
    return str(sample_id).replace("/", "__").replace("\\", "__")


def _extract_chain_b(input_pdb: str, output_pdb: str) -> bool:
    """Write a chain-B-only PDB (binder). Pure text; mirrors seed_ablation_clusters.py."""
    kept, found = [], False
    try:
        with open(input_pdb) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")) and line[21].strip() == "B":
                    kept.append(line)
                    found = True
                elif line.startswith("TER") and found:
                    kept.append(line)
    except OSError:
        return False
    if not found:
        return False
    with open(output_pdb, "w") as f:
        f.writelines(kept)
    return True


def _foldseek_cluster(pdb_dir: Path, work: Path, foldseek: str,
                      tmscore: float = 0.5) -> Optional[pd.DataFrame]:
    out = work / "cluster_result"
    try:
        subprocess.run(
            [foldseek, "easy-cluster", str(pdb_dir), str(out), str(work / "tmp"),
             "--alignment-type", "1", "--tmscore-threshold", str(tmscore)],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"foldseek cluster failed: {e}")
        return None
    tsv = Path(str(out) + "_cluster.tsv")
    if not tsv.exists():
        return None
    return pd.read_csv(tsv, sep="\t", names=["representative", "member"])


def _foldseek_mean_tm(pdb_dir: Path, work: Path, foldseek: str) -> float:
    """Mean off-diagonal all-vs-all TM-score via foldseek easy-search."""
    aln = work / "aln.tsv"
    try:
        subprocess.run(
            [foldseek, "easy-search", str(pdb_dir), str(pdb_dir), str(aln),
             str(work / "tmp_search"), "--alignment-type", "1",
             "--format-output", "query,target,alntmscore", "-e", "inf"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"foldseek search failed: {e}")
        return float("nan")
    if not aln.exists():
        return float("nan")
    df = pd.read_csv(aln, sep="\t", names=["query", "target", "tm"])
    off = df[df["query"] != df["target"]]
    return float(off["tm"].mean()) if len(off) else float("nan")


def structural_diversity_by_generation(df: pd.DataFrame, run_dir: str,
                                       foldseek_bin: Optional[str] = None) -> pd.DataFrame:
    run = Path(run_dir)
    foldseek = _resolve_foldseek(foldseek_bin)
    rows = []
    for gen, sub in df[df["passed_filter"] == True].groupby("generation"):  # noqa: E712
        n_success = len(sub)
        work = Path(tempfile.mkdtemp(prefix=f"reseed_fs_g{gen}_"))
        pdb_dir = work / "chainB"
        pdb_dir.mkdir()
        n_ok = 0
        for _, r in sub.iterrows():
            src = _sample_pdb_path(run, r)
            if src is None:
                continue
            if _extract_chain_b(str(src), str(pdb_dir / f"{_safe_name(r['sample_id'])}.pdb")):
                n_ok += 1
        n_clusters = float("nan")
        mean_tm = float("nan")
        singleton_frac = float("nan")
        cluster_hit_fraction = float("nan")
        if foldseek and n_ok >= 2:
            clusters = _foldseek_cluster(pdb_dir, work, foldseek)
            if clusters is not None and len(clusters):
                sizes = clusters.groupby("representative").size()
                n_clusters = int(clusters["representative"].nunique())
                singleton_frac = float((sizes == 1).mean())
                cluster_hit_fraction = n_clusters / n_ok
            mean_tm = _foldseek_mean_tm(pdb_dir, work, foldseek)
        rows.append({
            "generation": int(gen), "n_success": n_success, "n_extracted": n_ok,
            "n_clusters": n_clusters, "cluster_hit_fraction": cluster_hit_fraction,
            "singleton_fraction": singleton_frac, "mean_pairwise_tm": mean_tm,
        })
        shutil.rmtree(work, ignore_errors=True)
    return pd.DataFrame(rows).sort_values("generation")


# --------------------------------------------------------------------------------------
# Cluster novelty vs the full prior success pool (did we hit NEW clusters?)
# --------------------------------------------------------------------------------------

def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "pass", "passed", "yes")


def cluster_novelty_by_generation(df: pd.DataFrame, run_dir: str, baseline_run_dir: str,
                                  foldseek_bin: Optional[str] = None,
                                  filter_stage: str = "auto",
                                  max_reference: Optional[int] = None) -> pd.DataFrame:
    """Cluster progeny successes TOGETHER with the FULL prior success pool.

    Answers: did re-seeding reach NEW structural clusters, or just re-generate binders
    that already existed in the seed-search success set? Every progeny success (gen>=1) is
    FoldSeek-clustered jointly with **all** prior successful binders — the complete
    ``load_success_set(baseline_run_dir)`` pool, including the gen-0 founders that were
    NOT selected as strains. A progeny is 'novel' if its cluster contains no reference
    (prior-pool) member, 'existing' otherwise.

    Per-generation columns: n_success, n_novel, n_existing, novel_fraction,
    n_novel_clusters (distinct progeny-only clusters that generation hit),
    cumulative_novel_clusters, plus context n_reference / n_reference_clusters.
    """
    run = Path(run_dir)
    foldseek = _resolve_foldseek(foldseek_bin)
    if not foldseek:
        print("foldseek not found; cannot compute cluster novelty")
        return pd.DataFrame()

    from pipeline.reseed import founders as founders_mod
    work = Path(tempfile.mkdtemp(prefix="reseed_novelty_"))
    pdb_dir = work / "all"
    pdb_dir.mkdir()

    # 1. reference = FULL prior success pool (not just the 20 selected founders).
    refs = founders_mod.load_success_set(baseline_run_dir, filter_stage=filter_stage)
    if max_reference and len(refs) > max_reference:
        idx = np.linspace(0, len(refs) - 1, max_reference).astype(int)
        refs = [refs[i] for i in idx]
    n_ref = 0
    for i, fr in enumerate(refs):
        if _extract_chain_b(fr.pdb_path, str(pdb_dir / f"ref__{i:06d}.pdb")):
            n_ref += 1

    # 2. progeny = ablation successes at gen >= 1 (gen-0 founders are reference members).
    prog = df[df["passed_filter"].apply(_truthy) & (df["generation"] > 0)]
    sample_gen: Dict[str, int] = {}
    for _, r in prog.iterrows():
        src = _sample_pdb_path(run, r)
        if src is None:
            continue
        safe = _safe_name(r["sample_id"])
        if _extract_chain_b(str(src), str(pdb_dir / f"{safe}.pdb")):
            sample_gen[safe] = int(r["generation"])

    if n_ref == 0 or not sample_gen:
        print(f"cluster novelty: n_reference={n_ref}, n_progeny={len(sample_gen)} — skipping")
        shutil.rmtree(work, ignore_errors=True)
        return pd.DataFrame()

    # 3. cluster everything together (one FoldSeek run).
    clusters = _foldseek_cluster(pdb_dir, work, foldseek)
    if clusters is None or not len(clusters):
        shutil.rmtree(work, ignore_errors=True)
        return pd.DataFrame()
    rep = clusters["representative"].str.replace(".pdb", "", regex=False)
    mem = clusters["member"].str.replace(".pdb", "", regex=False)
    ref_clusters = set(rep[mem.str.startswith("ref__")].unique())   # clusters w/ a reference
    cluster_of = dict(zip(mem, rep))
    n_ref_clusters = len(ref_clusters)

    # 4. classify progeny, aggregate per generation (+ cumulative novel clusters).
    per_gen: Dict[int, dict] = {}
    for sid, gen in sample_gen.items():
        crep = cluster_of.get(sid)
        novel = crep is not None and crep not in ref_clusters
        d = per_gen.setdefault(gen, {"n": 0, "novel": 0, "novel_clusters": set()})
        d["n"] += 1
        if novel:
            d["novel"] += 1
            d["novel_clusters"].add(crep)

    rows, seen_novel = [], set()
    for gen in sorted(per_gen):
        d = per_gen[gen]
        seen_novel |= d["novel_clusters"]
        rows.append({
            "generation": gen, "n_success": d["n"], "n_novel": d["novel"],
            "n_existing": d["n"] - d["novel"],
            "novel_fraction": d["novel"] / d["n"] if d["n"] else float("nan"),
            "n_novel_clusters": len(d["novel_clusters"]),
            "cumulative_novel_clusters": len(seen_novel),
            "n_reference": n_ref, "n_reference_clusters": n_ref_clusters,
        })
    shutil.rmtree(work, ignore_errors=True)
    return pd.DataFrame(rows).sort_values("generation")


def _plot_novelty(nov_df, out_png):
    if nov_df is None or not len(nov_df):
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(nov_df["generation"], nov_df["novel_fraction"], color="tab:green", alpha=0.6,
            label="novel fraction")
    ax1.set_xlabel("generation")
    ax1.set_ylabel("fraction of progeny in NEW clusters")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(nov_df["generation"], nov_df["cumulative_novel_clusters"], "o-",
             color="tab:red", label="cumulative new clusters")
    ax2.set_ylabel("cumulative new clusters")
    ax1.set_title("Cluster novelty vs seed-search pool")
    ax1.set_xticks([int(g) for g in nov_df["generation"]])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Sequence diversity (lightweight, spec §9.2 — no MMseqs2)
# --------------------------------------------------------------------------------------

def _mean_pairwise_identity(seqs: List[str]) -> float:
    seqs = [s for s in seqs if isinstance(s, str) and s]
    if len(seqs) < 2:
        return float("nan")
    ids = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            a, b = seqs[i], seqs[j]
            n = min(len(a), len(b))
            if n == 0:
                continue
            ids.append(sum(x == y for x, y in zip(a[:n], b[:n])) / n)
    return float(np.mean(ids)) if ids else float("nan")


def sequence_diversity_by_generation(df: pd.DataFrame) -> pd.DataFrame:
    if "sequence" not in df.columns:
        return pd.DataFrame(columns=["generation", "mean_pairwise_identity", "n_success"])
    rows = []
    for gen, sub in df[df["passed_filter"] == True].groupby("generation"):  # noqa: E712
        rows.append({"generation": int(gen),
                     "mean_pairwise_identity": _mean_pairwise_identity(list(sub["sequence"])),
                     "n_success": len(sub)})
    return pd.DataFrame(rows).sort_values("generation")


# --------------------------------------------------------------------------------------
# Secondary structure (RQ2, spec §9.4)
# --------------------------------------------------------------------------------------

def ss_by_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Mean binder-relative SS per generation, with KS / earth-mover shift vs previous."""
    from scipy.stats import ks_2samp, wasserstein_distance
    succ = df[df["passed_filter"] == True].dropna(subset=["ss_sheet_frac"])  # noqa: E712
    gens = sorted(succ["generation"].unique())
    rows = []
    prev_sheet = None
    for gen in gens:
        sub = succ[succ["generation"] == gen]
        sheet = sub["ss_sheet_frac"].to_numpy()
        rec = {
            "generation": int(gen), "n_success": len(sub),
            "helix_mean": float(sub["ss_helix_frac"].mean()),
            "sheet_mean": float(sub["ss_sheet_frac"].mean()),
            "loop_mean": float(sub["ss_loop_frac"].mean()),
            "ks_vs_prev_sheet": float("nan"), "emd_vs_prev_sheet": float("nan"),
        }
        if prev_sheet is not None and len(sheet) and len(prev_sheet):
            rec["ks_vs_prev_sheet"] = float(ks_2samp(sheet, prev_sheet).statistic)
            rec["emd_vs_prev_sheet"] = float(wasserstein_distance(sheet, prev_sheet))
        rows.append(rec)
        prev_sheet = sheet
    return pd.DataFrame(rows)


def ss_shift_by_seed_ss(df: pd.DataFrame) -> pd.DataFrame:
    """Progeny sheet-fraction grouped by the dominant SS class of the cut (RQ2b)."""
    succ = df[(df["generation"] > 0) & (df["passed_filter"] == True)].copy()  # noqa: E712

    def dominant(ss_json):
        try:
            arr = json.loads(ss_json) if isinstance(ss_json, str) else ss_json
        except (TypeError, json.JSONDecodeError):
            return None
        if not arr:
            return None
        from collections import Counter
        return Counter(arr).most_common(1)[0][0]

    succ["seed_dominant_ss"] = succ["seed_res_ss"].apply(dominant)
    rows = []
    for ss_class, sub in succ.dropna(subset=["seed_dominant_ss"]).groupby("seed_dominant_ss"):
        rows.append({"seed_dominant_ss": ss_class, "n": len(sub),
                     "progeny_sheet_mean": float(sub["ss_sheet_frac"].mean()),
                     "progeny_helix_mean": float(sub["ss_helix_frac"].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Novelty / extinction / RQ1 curve
# --------------------------------------------------------------------------------------

def novelty_by_generation(df: pd.DataFrame, rmsd_thresh: float = 1.0,
                          seqid_thresh: float = 0.8) -> pd.DataFrame:
    rows = []
    for gen, sub in df[df["generation"] > 0].groupby("generation"):
        near = sub[(sub["rmsd_to_parent"] <= rmsd_thresh) &
                   (sub["seqid_to_parent"] >= seqid_thresh)]
        rows.append({"generation": int(gen), "n": len(sub),
                     "near_copy_fraction": (len(near) / len(sub)) if len(sub) else float("nan")})
    return pd.DataFrame(rows).sort_values("generation")


def extinction_by_generation(run_dir: str, n_strains_start: int) -> pd.DataFrame:
    lineage_path = Path(run_dir) / "lineage.json"
    if not lineage_path.exists():
        return pd.DataFrame(columns=["generation", "alive", "extinct_fraction"])
    lineage = json.loads(lineage_path.read_text())
    gens = set()
    for s in lineage.values():
        gens.update(int(k) for k in s.get("rounds", {}).keys())
    rows = []
    for gen in sorted(gens):
        alive = sum(1 for s in lineage.values()
                    if s.get("rounds", {}).get(str(gen)) is not None)
        rows.append({"generation": gen, "alive": alive,
                     "extinct_fraction": 1 - alive / max(1, n_strains_start)})
    return pd.DataFrame(rows)


def rq1_cumulative(df: pd.DataFrame, baseline_run_dir: Optional[str] = None
                   ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Re-seed cumulative successes vs cumulative samples; optional baseline overlay."""
    gen = df[df["generation"] > 0].sort_values(["generation", "strain_id", "sample_id"])
    gen = gen.reset_index(drop=True)
    gen["sample_index"] = gen.index + 1
    gen["cumulative_successes"] = gen["passed_filter"].astype(int).cumsum()
    reseed = gen[["sample_index", "cumulative_successes", "generation"]]
    baseline = None
    if baseline_run_dir:
        from pipeline.reseed.founders import baseline_cumulative
        try:
            baseline = baseline_cumulative(baseline_run_dir)
        except Exception as e:  # noqa: BLE001
            print(f"baseline load failed: {e}")
    return reseed, baseline


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def _plot_rq1(reseed, baseline, out_png):
    plt.figure(figsize=(7, 5))
    plt.plot(reseed["sample_index"], reseed["cumulative_successes"],
             label="re-seed", lw=2)
    if baseline is not None and len(baseline):
        plt.plot(baseline["sample_index"], baseline["cumulative_successes"],
                 label="blind baseline", ls="--", lw=2)
    plt.xlabel("cumulative samples (fold calls)")
    plt.ylabel("cumulative successes")
    plt.title("RQ1: successes vs compute")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_ss_overlay(df, out_png):
    succ = df[df["passed_filter"] == True].dropna(subset=["ss_sheet_frac"])  # noqa: E712
    gens = sorted(succ["generation"].unique())
    if not gens:
        return
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(7, 5))
    for i, gen in enumerate(gens):
        sub = succ[succ["generation"] == gen]
        color = cmap(i / max(1, len(gens) - 1))
        plt.scatter(sub["ss_helix_frac"], sub["ss_sheet_frac"], s=14, alpha=0.6,
                    color=color, label=f"gen {gen}")
    plt.xlabel("helix fraction")
    plt.ylabel("sheet fraction")
    plt.title("RQ2: binder SS by generation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_diversity(struct_df, seq_df, out_png):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    if len(struct_df):
        ax[0].plot(struct_df["generation"], struct_df["cluster_hit_fraction"], "o-")
        ax[0].set_title("clusters / successes"); ax[0].set_xlabel("generation")
        ax[1].plot(struct_df["generation"], struct_df["mean_pairwise_tm"], "o-")
        ax[1].set_title("mean pairwise TM"); ax[1].set_xlabel("generation")
    if len(seq_df):
        ax[2].plot(seq_df["generation"], seq_df["mean_pairwise_identity"], "o-")
        ax[2].set_title("mean pairwise seq id"); ax[2].set_xlabel("generation")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run_analysis(run_dir: str, baseline_run_dir: Optional[str] = None,
                 foldseek_bin: Optional[str] = None, n_strains_start: int = 20,
                 filter_stage: str = "auto") -> Path:
    run = Path(run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    df = load_records(run_dir)

    struct_df = structural_diversity_by_generation(df, run_dir, foldseek_bin)
    seq_df = sequence_diversity_by_generation(df)
    if baseline_run_dir:
        nov_clusters_df = cluster_novelty_by_generation(
            df, run_dir, baseline_run_dir, foldseek_bin, filter_stage)
        if len(nov_clusters_df):
            nov_clusters_df.to_csv(out / "cluster_novelty.csv", index=False)
            _plot_novelty(nov_clusters_df, out / "cluster_novelty.png")
    ss_df = ss_by_generation(df)
    ss_seed_df = ss_shift_by_seed_ss(df)
    nov_df = novelty_by_generation(df)
    ext_df = extinction_by_generation(run_dir, n_strains_start)
    reseed, baseline = rq1_cumulative(df, baseline_run_dir)

    for name, d in [("structural_diversity", struct_df), ("sequence_diversity", seq_df),
                    ("secondary_structure", ss_df), ("ss_by_seed_ss", ss_seed_df),
                    ("novelty", nov_df), ("extinction", ext_df),
                    ("rq1_cumulative_reseed", reseed)]:
        d.to_csv(out / f"{name}.csv", index=False)
    if baseline is not None:
        baseline.to_csv(out / "rq1_cumulative_baseline.csv", index=False)

    _plot_rq1(reseed, baseline, out / "rq1_successes_vs_compute.png")
    _plot_ss_overlay(df, out / "rq2_ss_by_generation.png")
    _plot_diversity(struct_df, seq_df, out / "diversity_by_generation.png")

    print(f"Analysis written to {out}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Re-seeding ablation analysis")
    ap.add_argument("--run_dir", required=True, help="runs/<target> directory")
    ap.add_argument("--baseline_run_dir", default=None, help="prior blind run for RQ1 baseline")
    ap.add_argument("--settings", default=None, help="settings_reseed json (for baseline_run_dir + n_strains)")
    ap.add_argument("--foldseek", default=None, help="path to foldseek binary")
    ap.add_argument("--n_strains", type=int, default=20)
    args = ap.parse_args()

    baseline = args.baseline_run_dir
    n_strains = args.n_strains
    filter_stage = "auto"
    if args.settings:
        from pipeline.reseed.config import load_config
        cfg = load_config(args.settings)
        baseline = baseline or cfg.baseline_run_dir
        n_strains = cfg.n_strains
        filter_stage = cfg.founder_filter_stage
    run_analysis(args.run_dir, baseline, args.foldseek, n_strains, filter_stage)


if __name__ == "__main__":
    main()
