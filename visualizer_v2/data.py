"""
Data access for the re-seeding ablation viewer.

Loads ``runs/<target>/records.parquet`` (or .csv), ``lineage.json``, and any precomputed
``analysis/*.csv``, and resolves per-sample structure paths in the spec §8 layout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SAMPLE_IDX_RE = re.compile(r"_sample_(\d+)")


@dataclass
class RunData:
    run_dir: Path
    target: str
    records: pd.DataFrame
    lineage: Dict[str, dict] = field(default_factory=dict)
    analysis: Dict[str, pd.DataFrame] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)   # runs/<target>/config.json

    @property
    def generations(self) -> List[int]:
        return sorted(int(g) for g in self.records["generation"].unique())


def find_runs(base_dir: str) -> List[Path]:
    """Every directory under ``base_dir`` that holds a records file (recursively)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = set()
    for name in ("records.parquet", "records.csv"):
        for p in base.rglob(name):
            runs.add(p.parent)
    return sorted(runs)


def load_run(run_dir: str) -> RunData:
    run = Path(run_dir)
    records = None
    for name in ("records.parquet", "records.csv"):
        p = run / name
        if not p.exists():
            continue
        try:
            records = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            break
        except Exception:  # noqa: BLE001 - e.g. parquet present but pyarrow missing; try csv
            continue
    if records is None:
        raise FileNotFoundError(f"No readable records in {run}")

    lineage = {}
    lp = run / "lineage.json"
    if lp.exists():
        lineage = json.loads(lp.read_text())

    analysis: Dict[str, pd.DataFrame] = {}
    adir = run / "analysis"
    if adir.is_dir():
        for csv in adir.glob("*.csv"):
            try:
                analysis[csv.stem] = pd.read_csv(csv)
            except Exception:  # noqa: BLE001
                pass

    config = {}
    cp = run / "config.json"
    if cp.exists():
        try:
            config = json.loads(cp.read_text())
        except Exception:  # noqa: BLE001
            pass

    target = str(records["target"].iloc[0]) if len(records) else run.name
    return RunData(run_dir=run, target=target, records=records,
                   lineage=lineage, analysis=analysis, config=config)


def sample_pdb_path(run: RunData, row) -> Optional[Path]:
    """Archived complex PDB (chain A/B) for a records row, spec §8 layout."""
    run_dir = run.run_dir
    gen = int(row["generation"])
    if gen == 0:
        cands = list((run_dir / "gen0").glob(f"strain{int(row['strain_id']):02d}_*.pdb"))
        return cands[0] if cands else None
    cut = row.get("cut_id")
    m = _SAMPLE_IDX_RE.search(str(row["sample_id"]))
    idx = m.group(1) if m else "000"
    p = run_dir / f"gen{gen}" / f"strain{int(row['strain_id']):02d}" / f"cut{cut}" / f"sample{idx}.pdb"
    return p if p.exists() else None


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("true", "1", "pass", "passed", "yes")


def generation_totals(run: RunData) -> pd.DataFrame:
    """Per-generation counts with the CORRECT denominator.

    ``records`` only holds the samples that passed the refolding prefilter and were
    PyRosetta-scored, so it undercounts. The true number generated per round is the
    number of chain-split complexes fed to scoring (``gen<k>/_scoring_input/*.pdb``);
    the folding-prefilter pass count comes from ``gen<k>/_scored/*_prescoring_results.csv``.

    Columns: generation, n_generated (all generated), n_folding_pass (passed refold
    prefilter), n_scored (reached PyRosetta = records rows), n_success (PyRosetta pass),
    success_rate = n_success / n_generated.
    """
    df = run.records
    rows = []
    for gen in run.generations:
        sub = df[df["generation"] == gen]
        n_scored = len(sub)
        n_success = int(sub["passed_filter"].apply(_truthy).sum())

        if gen == 0:  # founders — the seed set, all "successes" by construction
            n_generated = n_folding_pass = n_scored
        else:
            gdir = run.run_dir / f"gen{gen}"
            si = gdir / "_scoring_input"
            n_generated = len(list(si.glob("*.pdb"))) if si.is_dir() else 0
            n_folding_pass = None
            sdir = gdir / "_scored"
            if sdir.is_dir():
                pres = sorted(sdir.glob("*_prescoring_results.csv"))
                if pres:
                    try:
                        pdf = pd.read_csv(pres[0])
                        if not n_generated:            # scoring_input cleaned -> use prescoring
                            n_generated = len(pdf)
                        if "folding_filter_pass" in pdf.columns:
                            n_folding_pass = int(pdf["folding_filter_pass"].apply(_truthy).sum())
                    except Exception:  # noqa: BLE001
                        pass
            if not n_generated:                        # last resort
                n_generated = n_scored

        rows.append({
            "generation": gen, "n_generated": n_generated,
            "n_folding_pass": n_folding_pass, "n_scored": n_scored,
            "n_success": n_success,
            "success_rate": (n_success / n_generated) if n_generated else 0.0,
        })
    return pd.DataFrame(rows)


def generation_summary(run: RunData) -> pd.DataFrame:
    """generation_totals plus the per-generation count of still-alive strains."""
    tot = generation_totals(run)
    alive = []
    for gen in tot["generation"]:
        a = None
        if run.lineage and gen > 0:
            a = sum(1 for s in run.lineage.values()
                    if s.get("rounds", {}).get(str(gen)) is not None)
        alive.append(a)
    tot["alive_strains"] = alive
    return tot


# Identity / lineage columns that are not metrics to plot.
_NON_METRIC_COLS = {
    "generation", "strain_id", "founder_id", "parent_id", "target", "sample_id",
    "cut_id", "cut_pattern", "seed_res_indices", "seed_res_ss", "passed_filter",
    "sequence", "status", "folding_model", "binder_chains", "design_name", "model",
    "length", "num_recycles",
}


def numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    """Records columns that hold a plottable numeric metric."""
    cols = []
    for c in df.columns:
        if c in _NON_METRIC_COLS:
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            cols.append(c)
    return sorted(cols)


def metric_evolution(run: RunData, metric: str, only_success: bool = True) -> pd.DataFrame:
    """Per-generation count / mean / min / max / median / std of one metric."""
    df = run.records
    if only_success:
        df = df[df["passed_filter"].apply(_truthy)]
    rows = []
    for gen in sorted(int(g) for g in df["generation"].unique()):
        vals = pd.to_numeric(df[df["generation"] == gen][metric], errors="coerce").dropna()
        if not len(vals):
            rows.append({"generation": gen, "count": 0, "mean": None, "min": None,
                         "max": None, "median": None, "std": None})
            continue
        rows.append({"generation": gen, "count": int(len(vals)),
                     "mean": float(vals.mean()), "min": float(vals.min()),
                     "max": float(vals.max()), "median": float(vals.median()),
                     "std": float(vals.std())})
    return pd.DataFrame(rows)


def metric_means_by_generation(run: RunData, metrics: List[str],
                               only_success: bool = True) -> pd.DataFrame:
    """Mean of every metric per generation (metrics as rows, generations as columns)."""
    df = run.records
    if only_success:
        df = df[df["passed_filter"].apply(_truthy)]
    num = df[metrics].apply(pd.to_numeric, errors="coerce")
    num = num.assign(generation=df["generation"].values)
    return num.groupby("generation").mean().T


def _mean_pairwise_dist(points: "np.ndarray") -> float:
    """Mean pairwise Euclidean distance among rows of ``points`` (a spread measure)."""
    n = len(points)
    if n < 2:
        return float("nan")
    if n > 1200:  # cap the O(n^2) cost; sample deterministically
        idx = np.linspace(0, n - 1, 1200).astype(int)
        points = points[idx]
        n = len(points)
    diff = points[:, None, :] - points[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(n, 1)
    return float(d[iu].mean())


def ss_diversity_scalar(df: pd.DataFrame) -> float:
    """SS diversity of a sample set = mean pairwise distance in (helix, sheet) space."""
    sub = df.dropna(subset=["ss_helix_frac", "ss_sheet_frac"])
    if len(sub) < 2:
        return float("nan")
    pts = sub[["ss_helix_frac", "ss_sheet_frac"]].to_numpy(dtype=float)
    return _mean_pairwise_dist(pts)


def global_trends(base_dir: str) -> Dict[str, pd.DataFrame]:
    """Aggregate every run under ``base_dir`` into cross-target trend tables.

    Returns:
      - ``metrics``: long df [target, generation, metric, mean] (success set).
      - ``ss_cluster``: [target, generation, n_success, ss_diversity,
        cluster_hit_fraction, mean_pairwise_tm, singleton_fraction]. Cluster columns
        come from each run's precomputed ``analysis/structural_diversity.csv`` (FoldSeek),
        NaN where not computed yet.
    """
    metrics_rows: List[dict] = []
    sc_rows: List[dict] = []
    for rd in find_runs(base_dir):
        try:
            run = load_run(str(rd))
        except Exception:  # noqa: BLE001
            continue
        recs = run.records
        if "passed_filter" not in recs.columns or "generation" not in recs.columns:
            continue
        succ = recs[recs["passed_filter"].apply(_truthy)]

        mcols = numeric_metric_columns(recs)
        if mcols and len(succ):
            num = succ[mcols].apply(pd.to_numeric, errors="coerce")
            num = num.assign(generation=succ["generation"].values)
            for gen, row in num.groupby("generation").mean().iterrows():
                for m in mcols:
                    if pd.notna(row[m]):
                        metrics_rows.append({"target": run.target, "generation": int(gen),
                                             "metric": m, "mean": float(row[m])})

        clust = run.analysis.get("structural_diversity")
        clust_idx = {}
        if clust is not None and "generation" in clust.columns:
            clust_idx = {int(r["generation"]): r for _, r in clust.iterrows()}

        for gen in sorted(int(g) for g in recs["generation"].unique()):
            sgen = succ[succ["generation"] == gen]
            row = {"target": run.target, "generation": gen, "n_success": int(len(sgen)),
                   "ss_diversity": ss_diversity_scalar(sgen),
                   "cluster_hit_fraction": float("nan"), "mean_pairwise_tm": float("nan"),
                   "singleton_fraction": float("nan")}
            c = clust_idx.get(gen)
            if c is not None:
                for k in ("cluster_hit_fraction", "mean_pairwise_tm", "singleton_fraction"):
                    if k in c and pd.notna(c[k]):
                        row[k] = float(c[k])
            sc_rows.append(row)

    return {"metrics": pd.DataFrame(metrics_rows), "ss_cluster": pd.DataFrame(sc_rows)}


def seed_global_ids(row, target_len: int) -> List[int]:
    """Global residue ids (chain B, offset by target_len) of a sample's seed residues."""
    idx = row.get("seed_res_indices")
    if idx is None or (isinstance(idx, float)):
        return []
    try:
        local = json.loads(idx) if isinstance(idx, str) else list(idx)
    except (TypeError, json.JSONDecodeError):
        return []
    return [int(i) for i in local]
