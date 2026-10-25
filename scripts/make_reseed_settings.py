#!/usr/bin/env python3
"""Write a re-seeding settings JSON that pools a seed search with its expansion.

Founders (generation 0) and the RQ1 baseline both come from ``baseline_run_dir``.
The initial seed search and its follow-up expansion live in sibling directories, so
this writes a comma-separated ``baseline_run_dir`` - ``founders.load_success_set``
splits it and pools matches from every root.

Budget: the nominal sample count of a run is

    n_strains * n_cuts * samples_per_cut * rounds

With the ablation's 5 seed fragments (P1..P5) fixed and rounds given, this solves for
``samples_per_cut`` to hit ``--budget`` and reports the per-job cost.

NOTE the nominal budget is an upper bound: strains that go extinct stop generating, so
realized samples land below it. ``redistribute_budget`` normally reclaims that, but it
is a no-op under one-strain-per-shard execution (a job only sees its own strain), so a
sharded run always decays with extinction.

Usage:
    python scripts/make_reseed_settings.py --target cd19
    python scripts/make_reseed_settings.py --target pdl1 --budget 15000 --rounds 5
"""

import argparse
import json
from pathlib import Path

SAMPLES = Path("/nfs/students/matscher/samples")
TARGETS_DIR = Path("/nfs/homedirs/matscher/targets")
CKPT = Path("/nfs/students/matscher/laproteina_checkpoints")
# Cluster-absolute, NOT derived from __file__: under --write-prefix this script runs
# from a mounted copy whose __file__ already carries the prefix.
REPO = Path("/nfs/homedirs/matscher/repos/BindProteina")

TARGETS = {
    "cd19": {
        "target": "CD19",
        "target_pdb": TARGETS_DIR / "cd19" / "cd19.pdb",
        "target_len": 272,
        # Initial seed search + the 10k expansion built by expand_seed_search.py.
        "baseline_dirs": [
            SAMPLES / "seed_search_cd19",
            SAMPLES / "seed_search_cd19_expand10k",
        ],
        "hotspots": "14,16,20,64,163,176,178,221,265,271",
        "output_base_dir": Path("/nfs/students/matscher/reseed_output/cd19"),
    },
    "pdl1": {
        "target": "PDL1",
        "target_pdb": TARGETS_DIR / "pdl1" / "pdl1_cropped.pdb",
        "target_len": 209,
        "baseline_dirs": [
            SAMPLES / "seed_search_pdl1",
            SAMPLES / "seed_search_pdl1_expand10k",
        ],
        "hotspots": "38,40,43,95,97,99,103,104,105,107",
        "output_base_dir": Path("/nfs/students/matscher/reseed_output/pdl1"),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--budget", type=int, default=15000,
                    help="Total nominal samples across the whole run (default: 15000)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n-strains", type=int, default=20,
                    help="S. Also the number of shards to run one strain per GPU.")
    ap.add_argument("--n-cuts", type=int, default=5,
                    help="Seed fragments per parent (P1..P5 - the ablation's sweep)")
    ap.add_argument("--seed-n-res", type=int, default=5)
    ap.add_argument("--champion-selection", default="ss_stratified",
                    choices=["uniform", "ss_stratified"],
                    help="Which winner seeds a strain's next round. Defaults to "
                         "ss_stratified to match every configured target in "
                         "settings_reseed/; 'uniform' is the spec's no-pressure "
                         "reference arm (run it into a separate output tree).")
    ap.add_argument("--redistribute-budget", action="store_true",
                    help="Spread an extinct strain's freed budget over the survivors, "
                         "keeping folds/round ~constant. Matches the other configured "
                         "targets, but is a NO-OP when the run is sharded one strain "
                         "per job (a job only sees its own strain) - use it with a "
                         "small --n_shards, or unsharded.")
    ap.add_argument("--seed-selection", default="random", choices=["random", "interface"],
                    help="'random' (default, spec's P1..P5 sweep) or 'interface' "
                         "(anchors drawn only from binder residues contacting the "
                         "target hotspots - carries real binding geometry instead of "
                         "arbitrary surface positions; the README's suggested rescue "
                         "for low-yield targets).")
    ap.add_argument("--seed-free-sequence", action="store_true",
                    help="Free the seed's amino acid identity, fixing only backbone "
                         "position (spec sec 11-c). Motif-conditioned generation "
                         "normally fixes BOTH sequence and backbone at the anchors; "
                         "letting LaProteina redesign the anchor residues in their new "
                         "context can reduce clashes/awkward packing versus keeping "
                         "residue identities pinned from a structurally unrelated "
                         "generation. Use when progeny quality (not just count) is the "
                         "problem, i.e. a low folding-filter pass rate.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Settings path (default settings_reseed/<TARGET>.json)")
    ap.add_argument("--write-prefix", type=str, default="",
                    help="Prefix for physical writes (e.g. a local NFS mount root); "
                         "embedded paths stay cluster-absolute.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Write the settings even if a baseline dir does not exist yet "
                         "(the expansion may not have run).")
    args = ap.parse_args()

    t = TARGETS[args.target]

    def phys(p: Path) -> Path:
        return Path(args.write_prefix + str(p)) if args.write_prefix else p

    per_round = args.n_strains * args.n_cuts
    samples_per_cut, rem = divmod(args.budget, per_round * args.rounds)
    if samples_per_cut < 1:
        ap.error(f"budget {args.budget} too small for "
                 f"{args.n_strains} strains x {args.n_cuts} cuts x {args.rounds} rounds")
    nominal = samples_per_cut * per_round * args.rounds

    missing = [d for d in t["baseline_dirs"] if not phys(d).exists()]
    if missing:
        msg = "baseline dir(s) not present yet: " + ", ".join(str(m) for m in missing)
        if not args.allow_missing:
            ap.error(msg + "  (re-run with --allow-missing to write anyway)")
        print(f"WARNING: {msg}")

    cfg = {
        "_comment": (
            "Re-seeding ablation over the pooled initial seed search + expansion. "
            "baseline_run_dir is comma-separated: founders and the RQ1 baseline are "
            "drawn from every listed root."),
        "target": t["target"],
        "target_pdb": str(t["target_pdb"]),
        "target_len": t["target_len"],
        "baseline_run_dir": ",".join(str(d) for d in t["baseline_dirs"]),
        "founder_filter_stage": "auto",
        "checkpoint": str(CKPT / "LD4_motif_idx_aa.ckpt"),
        "autoencoder_path": str(CKPT / "AE3_motif.ckpt"),
        "output_base_dir": str(t["output_base_dir"]),
        "laproteina_mode": "all_atom",
        "filter_config": "settings_filters/laProteina_filters.json",
        "folding_model": "protenix",
        "protenix_model": "protenix_base_20250630_v1.0.0",
        "device": "cuda",
        "n_steps": 400,
        "max_batch_size": 2,
        "n_strains": args.n_strains,
        "samples_per_cut": samples_per_cut,
        "n_cuts": args.n_cuts,
        "seed_n_res": args.seed_n_res,
        "seed_selection": args.seed_selection,
        "rounds": args.rounds,
        "champion_selection": args.champion_selection,
        "champion_ss_grid": 3,
        # Only meaningful when a job can see several strains: under one-strain-per-shard
        # execution this is a no-op, so leaving it false keeps the config honest about
        # what the sharded run actually does.
        "redistribute_budget": bool(args.redistribute_budget),
        "target_hotspot_residues": t["hotspots"],
        "seed_free_sequence": bool(args.seed_free_sequence),
        "ss_bias_target": False,
        "beta_heavy_cutoff": 0.30,
        "master_seed": 0,
    }

    out = args.out or Path(f"settings_reseed/{t['target']}.json")
    if not out.is_absolute():
        out = REPO / out
    phys(out).parent.mkdir(parents=True, exist_ok=True)
    phys(out).write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"target:            {t['target']}")
    print(f"baseline roots:    {len(t['baseline_dirs'])}")
    for d in t["baseline_dirs"]:
        print(f"                     {d}{'' if phys(d).exists() else '   [MISSING]'}")
    print(f"strains (S):       {args.n_strains}")
    print(f"cuts:              {args.n_cuts}  (P1..P5, seed_n_res={args.seed_n_res})")
    print(f"rounds:            {args.rounds}")
    print(f"samples_per_cut:   {samples_per_cut}")
    print(f"nominal budget:    {nominal}"
          + (f"  ({args.budget - nominal} short of {args.budget}: "
             f"not divisible by {per_round * args.rounds})" if rem else ""))
    print(f"per round:         {per_round * samples_per_cut} samples "
          f"({args.n_strains} strains x {args.n_cuts} cuts x {samples_per_cut})")
    print(f"per shard-job:     {args.n_cuts * samples_per_cut} samples "
          f"(1 strain x 1 round)")
    print(f"champion:          {args.champion_selection}")
    print(f"seed_selection:    {args.seed_selection}"
          + ("  (P1..P5 fragmentation sweep)" if args.seed_selection == "random"
             else f"  (cutoff 8.0A around hotspots={t['hotspots']})"))
    print(f"seed_free_sequence:{args.seed_free_sequence}")
    print(f"redistribute:      {bool(args.redistribute_budget)}"
          + ("  <- NO-OP if run one strain per shard"
             if args.redistribute_budget else ""))
    print(f"settings written:  {out}")


if __name__ == "__main__":
    main()
