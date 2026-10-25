# BindProteina

Code for the master's thesis ***Designing High Affinity Binders across Targets
with Generative ML***
(Maximilian Matscher, M.Sc., Chair of Data Analytics and Machine Learning,
School of Computation, Information and Technology, TUM).

La-Proteina is an atomistic generator of protein **monomers**. BindProteina
turns it into a **binder designer without retraining it**: a small structural
fragment — a *seed* — is placed next to a chosen epitope on the target, the
target-plus-seed is handed to La-Proteina as a single-chain motif-scaffolding
problem, and the generated chain is split back into a target and a binder chain
for scoring. All target information enters as geometry, before generation, and
is checked again afterwards by scoring. No model weights are trained.

> **Attribution.** This repository vendors two upstream projects. `laproteina/`
> is NVIDIA's **La-Proteina** (the generative model) and `functions/` is
> **BindCraft** (the supportive scoring framework). What is mine, what is
> theirs, and exactly which vendored files were modified is set out file by file
> in **[PROVENANCE.md](PROVENANCE.md)**. Upstream licenses are in `licenses/`.

---

## Method at a glance

```
target PDB
   │
   ├─ 1. target preparation            pipeline/target_preparer.py
   │     clean target, pick epitope (PeSTo-nominated, top-10 residues)
   │
   ├─ 2. seed placement                designer/  ·  pipeline/config_builder.py
   │     place a fragment near the epitope at a chosen position, orientation
   │     and distance; or expand a full grid over those parameters
   │
   ├─ 3. generation                    pipeline/laproteina_generator.py
   │     La-Proteina scaffolds a binder around the seed, as one chain;
   │     target = leading motif, binder = trailing scaffold segment
   │
   ├─ 4. chain split + folding         pipeline/scoring/folding_protenix.py
   │     reindex into target (A) + binder (B); re-fold the complex with
   │     Protenix (default) or AlphaFold2 initial-guess (optional)
   │     → pTM, i_pTM, binder pTM/ipTM, pLDDT, pAE, i_pAE, ipSAE, sc_RMSD
   │
   └─ 5. filter + interface scoring    pipeline/scoring/scorer.py
         designs passing the folding filter are relaxed and analysed with
         PyRosetta (dG, dSASA, shape complementarity, hotspot contacts);
         designs passing both stages count as in-silico successes
```

Scoring is deliberately two-stage: every design is re-folded, but only the
designs that survive the folding filter are relaxed, which is what makes large
sweeps affordable.

**Re-seeding** (`pipeline/reseed/`) iterates the loop. Founders are post-filter
successes of a prior blind run; each round fragments a strain's champion under
five cut patterns (P1–P5), scaffolds progeny around each cut, scores them, and
advances a **uniformly-random** survivor rather than the best one.

---

## Installation

A Linux machine with a CUDA GPU is required (A100 for high-throughput scoring).

```bash
git clone git@github.com:MaximilianMat/BindProteina.git
cd BindProteina
bash install.sh --cuda '12.6' --pkg_manager 'conda'
```

`install.sh` creates the main environment (PyRosetta, biotite, BioPython,
PyTorch, Lightning, Hydra) and a **separate** `protenix_env`, because Protenix
pins conflicting dependencies. See also `environment.yml` and
`requirements.txt`.

Not installed by the script, and needed for the full workflow:

| Dependency | Needed for |
|---|---|
| La-Proteina checkpoints (`LD4_motif_idx_aa.ckpt`, `LD5_motif_idx_tip.ckpt`, `AE3_motif.ckpt`) | generation |
| AlphaFold2 weights in `params/` + ColabDesign | the optional AF2 oracle |
| FoldSeek binary | re-seeding diversity and novelty analysis |
| PeSTo | epitope nomination (run separately; its output feeds the Seed Designer) |

---

## Running

### Generation

Per-target configuration lives in `settings_target/`; copy one and edit the
paths.

```bash
# directly
python pipeline/generate.py --settings settings_target/PDL1_direct_binder.json

# via SLURM
sbatch laproteina_generate_batch.slurm --settings settings_target/NPT2B.json
```

### Scoring an existing set of complexes

```bash
python pipeline/score_existing_two_step.py \
    --input_dir  ./my_designs \
    --output_dir ./scoring_results \
    --folding_model protenix \
    --filters settings_filters/laProteina_filters.json
```

Large runs shard across a SLURM array; each task scores a disjoint slice into
its own shard directory, and the shards are merged afterwards:

```bash
sbatch --array=0-9%10 laproteina_score_existing.slurm \
    --input_dir ./my_designs --output_dir ./scored_output --folding_model protenix
python scripts/merge_score_shards.py --output_dir ./scored_output
```

The job also requeues itself shortly before its wall-clock limit and resumes
from its own CSV, so short backfill slots can be used.

Two further scoring entry points:

```bash
# score from sequences only, via a Protenix-style input JSON
python pipeline/score_json.py --json_path input.json --output_dir ./out

# re-apply changed filters to a finished run without re-folding anything
python pipeline/refilter_run.py --run_dir ./initial_20260723_0505 --dry_run
```

### Re-seeding ablation

```bash
cp settings_reseed/EXAMPLE.json settings_reseed/MYTARGET.json   # fill in paths

# pilot, end-to-end on one strain
sbatch reseed_ablation.slurm --settings settings_reseed/MYTARGET.json \
       --n_strains 1 --samples_per_cut 2 --rounds 1

# full run: 20 strains x 5 cuts x 20 samples x 4 rounds
sbatch reseed_ablation.slurm --settings settings_reseed/MYTARGET.json

# per-generation analysis (clustering, novelty, SS shift)
python -m pipeline.reseed.analysis --run_dir runs/<target> \
       --settings settings_reseed/<target>.json --foldseek ./bin/foldseek
```

See `pipeline/reseed/README.md` for the module-level description.

### The two tools

Both are described in thesis Section 4 and Appendix A.

```bash
# Seed Designer: interactive seed placement, grid expansion, config export
pip install -r designer/requirements.txt
streamlit run designer/app.py

# Evolution Viewer: browse re-seeding runs per generation
pip install -r visualizer_v2/requirements.txt
streamlit run visualizer_v2/app.py -- --base_dir /path/to/reseed_output
```

---

## Repository layout

```
pipeline/            generation and scoring (own work)
  target_preparer.py     target cleaning, epitope handling, chain splitting
  config_builder.py      generation config from a target settings file
  laproteina_generator.py  embeds La-Proteina as a library
  generate.py            generation entry point
  score_*.py             scoring entry points
  refilter_run.py        re-apply filters to a finished run
  reseed/                the re-seeding ablation (thesis Sec. 5.2.5)
  scoring/               folding oracles, PyRosetta scoring, ipSAE, RMSD, filters
designer/            Seed Designer (own work)
visualizer_v2/       Evolution Viewer (own work)
scripts/             run orchestration and the analysis behind Chapter 5
tests/               re-seeding selection / seed extraction, PyRosetta threading
settings_target/     per-target generation configs
settings_reseed/     per-target re-seeding configs (12 targets)
settings_filters/    filter sets; laProteina_filters.json is the one used
*.slurm              cluster entry points
laproteina/          vendored La-Proteina  (NVIDIA, Apache-2.0 — see PROVENANCE.md)
functions/           vendored BindCraft    (MIT         — see PROVENANCE.md)
licenses/            upstream license texts
PROVENANCE.md        what is adapted, what is mine, file by file
```

---

## Credits

- **La-Proteina** — Geffner et al., NVIDIA. The generative model this work
  builds on; vendored in `laproteina/`.
- **BindCraft** — Pacesa et al. The binder-design framework whose PyRosetta
  interface scoring and filter conventions are used here; vendored in
  `functions/`.
- **ipSAE** — Dunbrack lab. Interface confidence score; re-implemented in
  `pipeline/scoring/ipsae.py`.
- **Protenix** — ByteDance. Open reproduction of AlphaFold3; the default
  scoring oracle.
- **PeSTo** — Krapp et al. Geometric interface prediction, used for epitope
  nomination.
- **ColabDesign** (Ovchinnikov) and **AlphaFold2** (Jumper et al.) — the
  optional AF2 initial-guess oracle.
- **ProteinMPNN** — Dauparas et al. Sequence design, used in comparisons.
- **Proteina-Complexa** — Didi, Zhang, Zhou, Reidenbach et al., NVIDIA
  (ICLR 2026). Used as a comparison model; cloned separately.
- **PyRosetta** — interface analysis and relaxation.
