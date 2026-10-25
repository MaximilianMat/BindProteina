#!/usr/bin/env python3
"""Build settings + batch manifest to expand the best seeds from a seed search.

A seed search runs N placements x 100 samples and most placements yield nothing.
This script reads the search's ``binder_stats.csv``, ranks placements by ipTM,
and re-runs the best ones with a larger sample budget split into independent
replicate jobs.

Two independence guarantees, both required - replicates would otherwise be exact
duplicates rather than additional sampling:

1. **Output directories.**  ``pipeline/generate.py`` derives its run folder as
   ``<output_dir>/initial_<YYYYMMDD_HHMM>`` with ``exist_ok=True``, so two array
   tasks sharing an ``output_dir`` and starting in the same minute would write
   into the same folder and overwrite each other's samples.  Each replicate gets
   its own ``output_dir``.

2. **RNG seed.**  LaProteina generation is deterministic:
   ``laproteina_generator.py`` calls ``L.seed_everything(gen_cfg.seed)`` and the
   seed defaults to 5.  Two replicates of one settings file at the same seed
   produce *identical* structures.  Each replicate gets a distinct
   ``laproteina_settings.seed``, consumed by ``generate.py`` and recorded in the
   run's ``settings_used.json``.

Ranking uses the ``iptm`` column, which is what the pipeline itself sorts on
(``analyze_successful_binders.py --sort-by iptm``; ``top_binders.csv`` records
``sorted_by: iptm``).  Note this is a max-statistic over the placement's 100
samples, so placements with a single hit rest on one draw.

Usage:
    python scripts/expand_seed_search.py --target cd19
    python scripts/expand_seed_search.py --target pdl1
    python scripts/expand_seed_search.py --target pdl1 --n-seeds 10 --n-reps 4
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

# Per-target layout. ``settings_prefix`` + the placement key (``sw_...``) +
# "_settings.json" is the filename in the search's meta/ directory; the naming
# convention differs between searches.
TARGETS = {
    "cd19": {
        "root": Path("/nfs/students/matscher/samples/seed_search_cd19"),
        "stats_csv": "top_binders_2/binder_stats.csv",
        "settings_prefix": "CD19_sweep_",
        "out_base": Path("/nfs/students/matscher/samples/seed_search_cd19_expand10k"),
    },
    "pdl1": {
        "root": Path("/nfs/students/matscher/samples/seed_search_pdl1"),
        "stats_csv": "top_binders/binder_stats.csv",
        "settings_prefix": "pdl1_sw_",
        "out_base": Path("/nfs/students/matscher/samples/seed_search_pdl1_expand10k"),
    },
}

# Base for replicate RNG seeds, kept well clear of the default (5) used by the
# original sweeps so an expansion never re-samples the search it came from.
SEED_BASE = 1000


def rank_seeds(stats_csv: Path, target: str):
    """Return [(placement_key, max_iptm, n_hits)] ranked by max ipTM descending."""
    by_seed = defaultdict(list)
    with open(stats_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            uid, iptm = row.get("uid"), row.get("iptm")
            if not uid or not iptm:
                continue
            run_dir = uid.split("/")[0]                       # seed_search_<target>_sw_...
            key = run_dir[len(f"seed_search_{target}_"):]     # sw_...
            by_seed[key].append(float(iptm))

    ranked = [(k, max(v), len(v)) for k, v in by_seed.items()]
    ranked.sort(key=lambda r: (-r[1], r[0]))
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="How many of the top placements to expand (default: 20). "
                             "Clamped to the number that actually produced hits.")
    parser.add_argument("--n-reps", type=int, default=3,
                        help="Replicate jobs per placement (default: 3)")
    parser.add_argument("--total-samples", type=int, default=10000,
                        help="Total samples across the whole expansion (default: 10000). "
                             "Per-job count is derived from this; --n-samples overrides.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Samples per replicate job. Overrides --total-samples. "
                             "Size this to the wall clock: the original searches ran "
                             "0.45-2.34 h per 100 samples depending on target and load.")
    parser.add_argument("--stats-csv", type=Path, default=None)
    parser.add_argument("--out-base", type=Path, default=None)
    parser.add_argument("--write-prefix", type=str, default="",
                        help="Prefix prepended when physically reading/writing files, e.g. "
                             "a local NFS mount root. Paths *embedded* in the settings and "
                             "manifest are always the unprefixed cluster paths, so output "
                             "generated over a mount still runs correctly on the cluster.")
    args = parser.parse_args()

    cfg_t = TARGETS[args.target]
    root = cfg_t["root"]
    src_meta = root / "meta"
    stats_csv = args.stats_csv or (root / cfg_t["stats_csv"])
    out_base = args.out_base or cfg_t["out_base"]

    def phys(p: Path) -> Path:
        """Physical path to touch on disk; logical paths stay cluster-absolute."""
        return Path(args.write_prefix + str(p)) if args.write_prefix else p

    ranked = rank_seeds(phys(stats_csv), args.target)
    if not ranked:
        parser.error(f"no ranked placements found in {stats_csv}")

    n_seeds = min(args.n_seeds, len(ranked))
    if n_seeds < args.n_seeds:
        print(f"NOTE: only {len(ranked)} placements produced hits; "
              f"expanding all {n_seeds} instead of {args.n_seeds}.")
    seeds = ranked[:n_seeds]

    n_jobs = n_seeds * args.n_reps
    n_samples = args.n_samples or math.ceil(args.total_samples / n_jobs)

    settings_dir = out_base / "meta"
    phys(settings_dir).mkdir(parents=True, exist_ok=True)

    manifest, seed_log = [], []
    rng_seed = SEED_BASE

    for key, max_iptm, n_hits in seeds:
        src = src_meta / f"{cfg_t['settings_prefix']}{key}_settings.json"
        if not phys(src).exists():
            raise FileNotFoundError(phys(src))
        base_cfg = json.loads(phys(src).read_text())

        for rep in range(1, args.n_reps + 1):
            rng_seed += 1  # globally unique across the whole expansion
            cfg = json.loads(json.dumps(base_cfg))  # deep copy

            cfg["output_dir"] = str(out_base / f"seed_search_{args.target}_{key}" / f"rep{rep:02d}")
            cfg["laproteina_settings"]["n_samples"] = n_samples
            cfg["laproteina_settings"]["seed"] = rng_seed

            dst = settings_dir / f"{args.target}_expand_{key}_rep{rep:02d}_settings.json"
            phys(dst).write_text(json.dumps(cfg, indent=4) + "\n")
            manifest.append(str(dst))
            seed_log.append((dst.name, cfg["output_dir"], rng_seed, key, max_iptm, n_hits))

    # Fail loudly rather than silently shipping duplicate work.
    out_dirs = [r[1] for r in seed_log]
    rng_seeds = [r[2] for r in seed_log]
    assert len(set(out_dirs)) == len(out_dirs), "duplicate output_dir - runs would collide"
    assert len(set(rng_seeds)) == len(rng_seeds), "duplicate RNG seed - runs would be identical"

    manifest_path = settings_dir / f"batch_manifest_{args.target}_expand.txt"
    phys(manifest_path).write_text("\n".join(manifest) + "\n")

    seed_table = settings_dir / "replicate_seeds.tsv"
    phys(seed_table).write_text(
        "settings_file\toutput_dir\trng_seed\tplacement\tmax_iptm\tn_hits\n"
        + "\n".join(f"{n}\t{d}\t{s}\t{k}\t{m:.6f}\t{h}" for n, d, s, k, m, h in seed_log) + "\n"
    )

    print(f"target:           {args.target}")
    print(f"source stats:     {stats_csv}")
    print(f"placements:       {n_seeds} (of {len(ranked)} with hits)")
    print(f"replicates/seed:  {args.n_reps} x {n_samples} samples = "
          f"{args.n_reps * n_samples} per placement")
    print(f"array tasks:      {n_jobs}")
    print(f"total samples:    {n_jobs * n_samples}")
    print(f"rng seeds:        {SEED_BASE + 1}..{rng_seed} (all distinct)")
    print(f"manifest:         {manifest_path}")
    print(f"seed table:       {seed_table}")
    print()
    print("Submit with:")
    print(f"  sbatch --array=0-{n_jobs - 1}%15 --time=04:00:00 \\")
    print(f"    --partition=gpu_a100 \\")
    print(f"    laproteina_generate_batch.slurm --batch {manifest_path}")


if __name__ == "__main__":
    main()
