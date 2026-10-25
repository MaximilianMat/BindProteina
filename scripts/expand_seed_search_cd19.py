#!/usr/bin/env python3
"""Build settings + batch manifest for the 10k expansion over the best CD19 seeds.

The seed search (250 placements x 100 samples) produced 163 filtered hits across
66 seeds.  This script takes the top seeds by ipTM and re-runs each of them with
a larger sample budget, split into independent replicate jobs.

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
   produce *identical* structures.  Each replicate therefore gets a distinct
   ``laproteina_settings.seed``, consumed by ``generate.py`` and recorded in the
   run's ``settings_used.json``.

Usage:
    python scripts/expand_seed_search_cd19.py
    python scripts/expand_seed_search_cd19.py --n-seeds 10 --n-reps 4
"""

import argparse
import json
from pathlib import Path

# Top seeds from top_binders_2/binder_stats.csv, ordered by max ipTM (descending).
TOP_SEEDS = [
    "sw_hs5_c20az90_s72_tilt90",
    "sw_hs4_c20az180_s0_tilt0",
    "sw_hs5_c20az90_s0_tilt0",
    "sw_hs4_c20az90_s144_tilt180",
    "sw_hs4_c20az0_s72_tilt180",
    "sw_cen_c60az270_s216_tilt90",
    "sw_cen_c60az0_s72_tilt90",
    "sw_hs5_c20az0_s72_tilt90",
    "sw_hs2_c60az0_s72_tilt0",
    "sw_hs2_c60az0_s144_tilt90",
    "sw_hs5_c20az0_s0_tilt90",
    "sw_cen_c20az0_s72_tilt0",
    "sw_cen_c60az0_s0_tilt0",
    "sw_hs4_c20az90_s0_tilt0",
    "sw_hs5_c20az180_s144_tilt90",
    "sw_hs5_c20az90_s144_tilt90",
    "sw_hs4_c20az180_s72_tilt90",
    "sw_hs1_c60az270_s144_tilt90",
    "sw_hs5_c20az0_s288_tilt0",
    "sw_hs1_c60az0_s72_tilt0",
]

SRC_META = Path("/nfs/students/matscher/samples/seed_search_cd19/meta")
OUT_BASE = Path("/nfs/students/matscher/samples/seed_search_cd19_expand10k")

# Base for replicate RNG seeds. Kept well clear of the default (5) used by the
# original 250-placement sweep, so the expansion never re-samples that run.
SEED_BASE = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="How many of the top seeds to expand (default: 20)")
    parser.add_argument("--n-reps", type=int, default=3,
                        help="Replicate jobs per seed (default: 3)")
    parser.add_argument("--n-samples", type=int, default=167,
                        help="Samples per replicate job. Default 167 sizes each job to fit "
                             "a 4 h wall: the 20 selected seeds ran 0.72-2.21 h per 100 "
                             "samples in the original sweep, so 167 projects to ~2.0 h "
                             "median and ~3.7 h for the slowest seed.")
    parser.add_argument("--src-meta", type=Path, default=SRC_META)
    parser.add_argument("--out-base", type=Path, default=OUT_BASE)
    parser.add_argument("--write-prefix", type=str, default="",
                        help="Prefix prepended when physically reading/writing files, e.g. "
                             "a local NFS mount root. Paths *embedded* in the settings and "
                             "manifest are always the unprefixed cluster paths, so output "
                             "generated over a mount still runs correctly on the cluster.")
    args = parser.parse_args()

    if args.n_seeds > len(TOP_SEEDS):
        parser.error(f"only {len(TOP_SEEDS)} seeds are listed, requested {args.n_seeds}")
    seeds = TOP_SEEDS[:args.n_seeds]

    def phys(p: Path) -> Path:
        """Physical path to touch on disk; args.*  stay logical (cluster) paths."""
        return Path(args.write_prefix + str(p)) if args.write_prefix else p

    settings_dir = args.out_base / "meta"          # logical, goes into the manifest
    phys(settings_dir).mkdir(parents=True, exist_ok=True)

    manifest = []
    seed_log = []
    rng_seed = SEED_BASE

    for seed in seeds:
        src = args.src_meta / f"CD19_sweep_{seed}_settings.json"
        if not phys(src).exists():
            raise FileNotFoundError(phys(src))
        base_cfg = json.loads(phys(src).read_text())

        for rep in range(1, args.n_reps + 1):
            rng_seed += 1  # globally unique across the whole expansion
            cfg = json.loads(json.dumps(base_cfg))  # deep copy

            cfg["output_dir"] = str(args.out_base / f"seed_search_cd19_{seed}" / f"rep{rep:02d}")
            cfg["laproteina_settings"]["n_samples"] = args.n_samples
            cfg["laproteina_settings"]["seed"] = rng_seed

            dst = settings_dir / f"CD19_expand_{seed}_rep{rep:02d}_settings.json"
            phys(dst).write_text(json.dumps(cfg, indent=4) + "\n")
            manifest.append(str(dst))
            seed_log.append((dst.name, cfg["output_dir"], rng_seed))

    # Fail loudly rather than silently shipping duplicate work.
    out_dirs = [d for _, d, _ in seed_log]
    rng_seeds = [s for _, _, s in seed_log]
    assert len(set(out_dirs)) == len(out_dirs), "duplicate output_dir - runs would collide"
    assert len(set(rng_seeds)) == len(rng_seeds), "duplicate RNG seed - runs would be identical"

    manifest_path = settings_dir / "batch_manifest_expand10k.txt"
    phys(manifest_path).write_text("\n".join(manifest) + "\n")

    seed_table = settings_dir / "replicate_seeds.tsv"
    phys(seed_table).write_text(
        "settings_file\toutput_dir\trng_seed\n"
        + "\n".join(f"{n}\t{d}\t{s}" for n, d, s in seed_log) + "\n"
    )

    total = len(manifest) * args.n_samples
    print(f"seeds:            {len(seeds)}")
    print(f"replicates/seed:  {args.n_reps} x {args.n_samples} samples "
          f"= {args.n_reps * args.n_samples} per seed")
    print(f"array tasks:      {len(manifest)}")
    print(f"total samples:    {total}")
    print(f"rng seeds:        {SEED_BASE + 1}..{rng_seed} (all distinct)")
    print(f"manifest:         {manifest_path}")
    print(f"seed table:       {seed_table}")
    print()
    print("Submit with:")
    print(f"  sbatch --array=0-{len(manifest) - 1}%15 --time=04:00:00 \\")
    print(f"    --partition=gpu_a100 \\")
    print(f"    laproteina_generate_batch.slurm --batch {manifest_path}")


if __name__ == "__main__":
    main()
