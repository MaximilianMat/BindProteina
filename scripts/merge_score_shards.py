"""Merge the per-shard outputs of an array run of laproteina_score_existing.slurm.

Each array task writes into <output_dir>/shard_XXX/. This collects those into
single tables at the top level:

    <output_dir>/scoring_summary_all.csv
    <output_dir>/<folding_model>_prescoring_results_all.csv

The "_all" suffix is deliberate: scripts/analyze_successful_binders.py globs for
"*scoring_summary.csv" recursively, so the merged file must not match that
pattern or every design would be counted twice. Point that script at the
output dir as usual (it walks the shards), or at --csv scoring_summary_all.csv.

Usage:
    python scripts/merge_score_shards.py --output_dir /path/to/scored_output
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _concat(paths: List[Path]) -> Optional[pd.DataFrame]:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:  # empty or half-written CSV
            print(f"  WARNING: could not read {p}: {e}")
            continue
        if len(df):
            df.insert(0, "shard", p.parent.name)
            frames.append(df)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    if "design_name" in merged.columns:
        merged = merged.drop_duplicates(subset="design_name", keep="first")
        merged = merged.sort_values("design_name").reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", "-o", type=Path, required=True,
                        help="The --output_dir the array job was submitted with")
    args = parser.parse_args()

    out_dir = args.output_dir
    shards = sorted(p for p in out_dir.glob("shard_*") if p.is_dir())
    if not shards:
        raise SystemExit(f"No shard_* directories under {out_dir} — was this an array run?")

    print(f"Found {len(shards)} shard(s) in {out_dir}")

    summaries = []
    for shard in shards:
        summary = shard / "scoring_summary.csv"
        if summary.exists():
            summaries.append(summary)
        elif (shard / "scoring_summary_partial.csv").exists():
            print(f"  {shard.name}: still running or interrupted (partial results only)")
            summaries.append(shard / "scoring_summary_partial.csv")
        else:
            print(f"  {shard.name}: no scoring output yet")

    merged = _concat(summaries)
    if merged is None:
        raise SystemExit("No scoring results to merge.")

    merged_path = out_dir / "scoring_summary_all.csv"
    merged.to_csv(merged_path, index=False)
    print(f"\nMerged {len(merged)} designs -> {merged_path}")
    if "pyrosetta_filter_pass" in merged.columns:
        n_pass = int(merged["pyrosetta_filter_pass"].fillna(False).astype(bool).sum())
        print(f"Passing all PyRosetta filters: {n_pass} ({n_pass / len(merged) * 100:.1f}%)")

    for model in ("af2", "protenix"):
        pre = [s / f"{model}_prescoring_results.csv" for s in shards]
        pre = [p for p in pre if p.exists()]
        if not pre:
            continue
        pre_merged = _concat(pre)
        if pre_merged is None:
            continue
        pre_path = out_dir / f"{model}_prescoring_results_all.csv"
        pre_merged.to_csv(pre_path, index=False)
        print(f"Merged {len(pre_merged)} {model} prescoring rows -> {pre_path}")


if __name__ == "__main__":
    main()
