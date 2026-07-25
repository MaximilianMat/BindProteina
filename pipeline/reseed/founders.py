"""
Founder (generation-0) loading and strain selection, plus the RQ1 equal-compute baseline.

Founders are the post-filter successful binders from a target's prior blind
position/orientation run. Per the user's decision, the same prior run supplies:
  - the founder set (generation 0), and
  - the RQ1 equal-compute baseline (cumulative successes vs cumulative samples).

The prior run is a ``generate.py`` output directory, i.e. it contains a
``scored/scoring_summary.csv`` (recursively). We keep only rows whose existing
pipeline filter passed (``pyrosetta_filter_pass``), resolve each design's refolded
complex PDB (chain A = target, chain B = binder), and hand those to the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger


@dataclass
class FounderRecord:
    """A single post-filter successful binder reused as a generation-0 founder."""

    founder_id: str
    pdb_path: str          # complex PDB, chain A = target, chain B = binder
    binder_len: int
    metrics: Dict[str, object] = field(default_factory=dict)
    ss: Optional[tuple] = None   # (helix_frac, sheet_frac, loop_frac), filled lazily


# Known complex-PDB locations under a scored/initial dir, in preference order.
# Prior blind runs vary: fully-scored conditions have relaxed/ (PyRosetta) and
# protenix_predictions/protenix_batch/ (refolds); folding-only conditions have
# scoring_input/ (the generated design) and the same protenix_batch/ refolds.
def _complex_pdb_candidates(scored_dir: Path, design_name: str) -> List[Path]:
    d = scored_dir
    return [
        d / "relaxed" / f"{design_name}_relaxed.pdb",                     # PyRosetta complex
        d / "protenix_predictions" / "protenix_batch" / f"{design_name}_protenix.pdb",
        d / "af2_predictions" / f"{design_name}_model1.pdb",
        d / "scoring_input" / f"{design_name}.pdb",                       # generated design
        d / "protenix_predictions" / f"{design_name}_protenix.pdb",
    ]


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "pass", "passed", "yes")


def _resolve_complex_pdb(scored_dir: Path, design_name: str) -> Optional[Path]:
    """Find the complex PDB (chain A target + chain B binder) for a design."""
    for c in _complex_pdb_candidates(scored_dir, design_name):
        if c.exists():
            return c
    # Fuzzy fallback for odd naming (bounded to the known subdirs).
    for sub in ("relaxed", "protenix_predictions/protenix_batch",
                "af2_predictions", "scoring_input"):
        base = scored_dir / sub
        if base.is_dir():
            hits = sorted(base.glob(f"{design_name}*.pdb"))
            if hits:
                return hits[0]
    return None


def _rows_to_founders(df, scored_dir: Path, pass_col: str, seen: set,
                      stage: str) -> List[FounderRecord]:
    out: List[FounderRecord] = []
    passed = df[df[pass_col].apply(_truthy)]
    for _, row in passed.iterrows():
        design_name = str(row["design_name"])
        pdb = _resolve_complex_pdb(scored_dir, design_name)
        if pdb is None:
            continue
        fid = f"{scored_dir.parent.name}/{design_name}"
        n = 1
        while fid in seen:
            n += 1
            fid = f"{scored_dir.parent.name}/{design_name}__{n}"
        seen.add(fid)
        try:
            binder_len = int(row.get("binder_length"))
        except (TypeError, ValueError):
            binder_len = -1
        metrics = row.to_dict()
        metrics["_founder_stage"] = stage
        out.append(FounderRecord(founder_id=fid, pdb_path=str(pdb),
                                 binder_len=binder_len, metrics=metrics))
    return out


def load_success_set(run_dir: str, filter_stage: str = "auto",
                     min_needed: Optional[int] = None) -> List[FounderRecord]:
    """Collect founders from a prior blind run directory.

    ``filter_stage``:
      - "pyrosetta": only PyRosetta-passing founders (scoring_summary.csv).
      - "folding":   only folding-passing founders (protenix_prescoring_results.csv).
      - "auto":      PyRosetta successes, supplemented with folding-passing designs if
                     fewer than ``min_needed`` were found (some prior runs only ran the
                     folding prefilter).
    Founders are returned PyRosetta-first.
    """
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"baseline_run_dir not found: {root}")

    seen: set = set()
    pyros: List[FounderRecord] = []
    if filter_stage in ("auto", "pyrosetta"):
        for csv in sorted(root.rglob("scoring_summary.csv")):
            try:
                df = pd.read_csv(csv)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not read {csv}: {e}")
                continue
            if "pyrosetta_filter_pass" not in df.columns:
                continue
            pyros += _rows_to_founders(df, csv.parent, "pyrosetta_filter_pass",
                                       seen, stage="pyrosetta")
        logger.info(f"Found {len(pyros)} PyRosetta-passing founders in {root}")

    need_folding = filter_stage == "folding" or (
        filter_stage == "auto" and min_needed is not None and len(pyros) < min_needed)

    folding: List[FounderRecord] = []
    if need_folding:
        for csv in sorted(root.rglob("protenix_prescoring_results.csv")):
            try:
                df = pd.read_csv(csv)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not read {csv}: {e}")
                continue
            if "folding_filter_pass" not in df.columns:
                continue
            folding += _rows_to_founders(df, csv.parent, "folding_filter_pass",
                                         seen, stage="folding")
        logger.info(f"Found {len(folding)} folding-passing founders in {root} "
                    f"(supplementing {len(pyros)} PyRosetta founders)")

    founders = pyros + folding
    if not founders:
        raise RuntimeError(
            f"No founders resolvable in {root} (filter_stage={filter_stage}). "
            f"Try filter_stage='folding' or run pipeline.reseed.prepare_founders.")
    logger.info(f"Loaded {len(founders)} founders from {root} (stage={filter_stage})")
    return founders


def select_strain_founders(founders: List[FounderRecord], S: int = 20,
                           chain_binder: str = "B", max_ss_eval: int = 300,
                           seed: int = 0) -> List[FounderRecord]:
    """Choose S founders, spanning secondary-structure space when there are more than S.

    Spec §3: with >S founders, diversity-select S that span SS space (matters for RQ2
    coverage); the remainder are discarded (but still recorded by the caller). With <=S
    founders, keep all (round budget then falls to 100·k).

    To bound gen-0 cost when a folding-supplemented pool is huge, at most ``max_ss_eval``
    founders are DSSP-evaluated (PyRosetta founders are kept preferentially, the rest
    randomly subsampled).
    """
    if len(founders) <= S:
        return list(founders)

    # Lazy import: mdtraj is only available in the cluster env.
    from pipeline.reseed.metrics_extra import ss_fractions
    import numpy as np

    pool = founders
    if len(pool) > max_ss_eval:
        pyros = [f for f in pool if f.metrics.get("_founder_stage") == "pyrosetta"]
        rest = [f for f in pool if f.metrics.get("_founder_stage") != "pyrosetta"]
        rng = np.random.default_rng(seed)
        keep_rest = max_ss_eval - len(pyros)
        if keep_rest > 0 and rest:
            idx = rng.choice(len(rest), size=min(keep_rest, len(rest)), replace=False)
            rest = [rest[i] for i in idx]
        else:
            rest = []
        pool = (pyros[:max_ss_eval] + rest)
        logger.info(f"Subsampled founder pool {len(founders)} -> {len(pool)} for SS eval")

    pts = []
    valid = []
    for f in pool:
        ss = ss_fractions(f.pdb_path, chain_id=chain_binder)
        if ss is None:
            continue
        f.ss = ss
        pts.append([ss[0], ss[1]])  # (helix_frac, sheet_frac)
        valid.append(f)

    if len(valid) <= S:
        logger.warning(f"Only {len(valid)} founders had computable SS; keeping all of them")
        return valid

    pts = np.asarray(pts)
    # Greedy farthest-point sampling in (helix, sheet) space; seed with the most
    # sheet-rich founder so beta coverage is guaranteed.
    selected = [int(np.argmax(pts[:, 1]))]
    while len(selected) < S:
        sel_pts = pts[selected]
        d = np.min(np.linalg.norm(pts[:, None, :] - sel_pts[None, :, :], axis=2), axis=1)
        d[selected] = -1.0
        selected.append(int(np.argmax(d)))

    chosen = [valid[i] for i in selected]
    logger.info(f"Diversity-selected {len(chosen)} of {len(valid)} founders spanning SS space")
    return chosen


def baseline_cumulative(run_dir: str) -> "pd.DataFrame":
    """RQ1 equal-compute baseline: cumulative successes vs cumulative samples.

    Reads every scored sample (passing or not) from the prior run and returns a
    DataFrame with columns ``[sample_index, passed, cumulative_successes]`` — the
    reference curve the re-seeding cumulative-successes-vs-compute plot is drawn against.
    """
    root = Path(run_dir)
    rows = []
    summaries = sorted(root.rglob("scoring_summary.csv"))
    if summaries:
        csvs, col = summaries, "pyrosetta_filter_pass"
    else:  # folding-only prior run
        csvs, col = sorted(root.rglob("protenix_prescoring_results.csv")), "folding_filter_pass"
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:  # noqa: BLE001
            continue
        if col not in df.columns:
            continue
        for _, r in df.iterrows():
            rows.append({"design_name": r.get("design_name"),
                         "passed": _truthy(r.get(col))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.reset_index(drop=True)
    out["sample_index"] = out.index + 1
    out["cumulative_successes"] = out["passed"].cumsum()
    return out
