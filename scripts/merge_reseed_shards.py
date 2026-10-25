#!/usr/bin/env python3
"""Merge sharded re-seeding output into one run dir for analysis.

``reseed_shard.slurm`` gives each strain its own ``output_base_dir`` so the shards never
contend for ``records.parquet`` / ``lineage.json`` / the checkpoint. Strain ids are
global, so merging is a union: concatenate the record tables, merge the lineages, and
symlink each shard's ``gen<k>/strain<ss>`` directories into one tree.

The result is a normal ``runs/<target>/`` layout that
``python -m pipeline.reseed.analysis --run_dir <merged>`` accepts unchanged.

Usage:
    python scripts/merge_reseed_shards.py \
        --base_out /nfs/students/matscher/reseed_output/cd19 \
        --target CD19
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_out", type=Path, required=True,
                    help="Directory holding shard00/, shard01/, ...")
    ap.add_argument("--target", required=True, help="Target name, e.g. CD19")
    ap.add_argument("--out", type=Path, default=None,
                    help="Merged run dir (default <base_out>/merged/runs/<target>)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy generation dirs instead of symlinking them")
    args = ap.parse_args()

    shard_runs = sorted(args.base_out.glob(f"shard*/runs/{args.target}"))
    if not shard_runs:
        ap.error(f"no shard run dirs under {args.base_out}/shard*/runs/{args.target}")

    out = args.out or (args.base_out / "merged" / "runs" / args.target)
    out.mkdir(parents=True, exist_ok=True)

    frames, lineage, n_gen_dirs = [], {}, 0

    for run in shard_runs:
        for name in ("records.parquet", "records.csv"):
            p = run / name
            if p.exists():
                frames.append(pd.read_parquet(p) if p.suffix == ".parquet"
                              else pd.read_csv(p))
                break
        else:
            print(f"WARNING: no records table in {run}")

        lin = run / "lineage.json"
        if lin.exists():
            for sid, entry in json.loads(lin.read_text()).items():
                if sid in lineage:
                    print(f"WARNING: strain {sid} present in more than one shard; "
                          f"keeping the first")
                    continue
                lineage[sid] = entry

        # gen0/ holds per-strain founder PDBs; gen<k>/ holds strain<ss>/cut<Pn>/.
        for gen_dir in sorted(run.glob("gen*")):
            dst_gen = out / gen_dir.name
            dst_gen.mkdir(exist_ok=True)
            for item in sorted(gen_dir.iterdir()):
                if item.name.startswith("_"):      # _motifs, _scored, _scoring_input
                    continue
                dst = dst_gen / item.name
                if dst.exists() or dst.is_symlink():
                    continue
                if args.copy and item.is_dir():
                    shutil.copytree(item, dst)
                elif args.copy:
                    shutil.copy2(item, dst)
                else:
                    dst.symlink_to(item.resolve())
                n_gen_dirs += 1

    if not frames:
        ap.error("no records found in any shard")

    records = pd.concat(frames, ignore_index=True)
    dupes = records.duplicated(subset=["sample_id"]).sum()
    if dupes:
        print(f"WARNING: {dupes} duplicate sample_id rows across shards; dropping")
        records = records.drop_duplicates(subset=["sample_id"])
    records = records.sort_values(["generation", "strain_id"], kind="stable")

    try:
        records.to_parquet(out / "records.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"parquet write failed ({e}); CSV only")
    records.to_csv(out / "records.csv", index=False)
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2))

    cfgs = sorted(args.base_out.glob(f"shard*/runs/{args.target}/config.json"))
    if cfgs:
        cfg = json.loads(cfgs[0].read_text())
        cfg["_merged_from_shards"] = len(shard_runs)
        cfg["n_shards"], cfg["shard_index"] = 1, 0   # merged tree is not a shard
        (out / "config.json").write_text(json.dumps(cfg, indent=2))

    strains = sorted(records["strain_id"].unique())
    print(f"shards merged:   {len(shard_runs)}")
    print(f"records:         {len(records)} rows")
    print(f"strains:         {len(strains)}  (ids {strains[0]}..{strains[-1]})")
    print(f"generations:     {sorted(records['generation'].unique())}")
    print(f"linked gen dirs: {n_gen_dirs}")
    print(f"merged run dir:  {out}")
    print()
    print("Analyse with:")
    print(f"  python -m pipeline.reseed.analysis --run_dir {out} \\")
    print(f"      --settings settings_reseed/{args.target}.json --foldseek ./bin/foldseek")


if __name__ == "__main__":
    main()
