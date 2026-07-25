"""
Founder preparation for prior runs that only completed the folding prefilter.

Some blind sweeps (e.g. ACTRIIA / AVRA2A) were only scored through the Protenix folding
prefilter, so they contain very few PyRosetta-passing designs — not enough to seed the
ablation's strains. This script materialises a proper PyRosetta-scored founder set:

  1. collect the folding-passing designs from ``baseline_run_dir``
     (protenix_prescoring_results.csv, folding_filter_pass), resolve their complex PDBs;
  2. rank by i_pTM and keep the top ``--max_candidates``;
  3. run the existing two-step scorer (fold + PyRosetta + filter) on them, producing
     ``<out>/scored/scoring_summary.csv`` with ``pyrosetta_filter_pass``.

Point the ablation's ``baseline_run_dir`` at ``<out>`` and set
``founder_filter_stage="pyrosetta"``.

Usage:
    python -m pipeline.reseed.prepare_founders --settings settings_reseed/ACTRIIA.json \
        --out /nfs/.../reseed_output/actriia/founders_prep --max_candidates 300
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pipeline.reseed.config import load_config
from pipeline.reseed import founders as founders_mod


def main():
    ap = argparse.ArgumentParser(description="Materialise PyRosetta-scored founders")
    ap.add_argument("--settings", required=True, help="settings_reseed/<target>.json")
    ap.add_argument("--out", required=True, help="output dir (becomes baseline_run_dir)")
    ap.add_argument("--max_candidates", type=int, default=300,
                    help="top-N folding-passing complexes to PyRosetta-score")
    ap.add_argument("--rank_by", default="i_pTM", help="metric to rank candidates by")
    args = ap.parse_args()

    cfg = load_config(args.settings)
    out = Path(args.out)
    staged = out / "staged"
    scored = out / "scored"
    staged.mkdir(parents=True, exist_ok=True)

    # 1-2. folding-passing complexes, ranked by confidence.
    cands = founders_mod.load_success_set(cfg.baseline_run_dir, filter_stage="folding")
    logger.info(f"{len(cands)} folding-passing candidates before ranking")

    def key(fr):
        try:
            return float(fr.metrics.get(args.rank_by, float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    cands = sorted(cands, key=key, reverse=True)[: args.max_candidates]
    logger.info(f"Staging top {len(cands)} candidates by {args.rank_by}")

    for i, fr in enumerate(cands):
        dst = staged / f"cand{i:04d}_{Path(fr.pdb_path).stem}.pdb"
        try:
            shutil.copy(fr.pdb_path, dst)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"copy failed {fr.pdb_path}: {e}")

    n = len(list(staged.glob("*.pdb")))
    if n == 0:
        raise RuntimeError("No candidate complexes staged; nothing to score.")
    logger.info(f"Scoring {n} staged complexes with the two-step scorer ...")

    # 3. full two-step scoring -> scored/scoring_summary.csv (pyrosetta_filter_pass).
    filters_path = cfg.filter_config
    if filters_path and not Path(filters_path).is_absolute():
        filters_path = str(Path(__file__).resolve().parents[2] / filters_path)

    from pipeline.scoring.scorer import score_complexes
    score_complexes(
        input_dir=str(staged),
        output_dir=str(scored),
        filters_path=filters_path,
        folding_model=cfg.folding_model,
        protenix_model_name=cfg.protenix_model,
        target_hotspot_residues=cfg.target_hotspot_residues,
        target_reference_pdb=cfg.target_pdb,
    )

    summary = scored / "scoring_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        n_pass = int(df["pyrosetta_filter_pass"].apply(founders_mod._truthy).sum()) \
            if "pyrosetta_filter_pass" in df.columns else 0
        logger.info(f"Done. {n_pass} PyRosetta-passing founders in {summary}")
    logger.info(f"Set baseline_run_dir='{out}' and founder_filter_stage='pyrosetta' "
                f"in {args.settings}")


if __name__ == "__main__":
    main()
