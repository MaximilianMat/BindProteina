# Re-seeding Ablation (`pipeline/reseed/`)

Iterative, self-contained-strain re-seeding of LaProteina binders (spec `reseed_ablation_spec.md`).
Founders = post-filter successes from a prior blind run; each round fragments a strain's champion
into five seed-cuts (P1..P5), scaffolds progeny around each, refolds/scores them with the existing
pipeline, and advances a **uniformly-random** winner. Does not modify `evolve_v2.py`, `generate.py`,
or the scoring stack.

## Modules
| File | Role |
|---|---|
| `config.py` | `ReseedConfig` + `load_config`; `CUT_PATTERNS` (P1..P5). |
| `founders.py` | Load gen-0 founders + RQ1 baseline from a prior run; SS-diversity strain selection. |
| `seed_extract.py` | **Core.** Draw 5 residues per pattern, build fixed-length contig + motif PDB. |
| `metrics_extra.py` | Binder SS (mdtraj DSSP), RMSD-to-parent, seq-id-to-parent. |
| `reseed_ablation.py` | Driver: strain loop → generate → score → champion → `records.parquet` + `lineage.json`. |
| `analysis.py` | Per-generation FoldSeek clustering, mean-TM, seq diversity, SS shift, novelty, RQ1 curve. |

## Run

```bash
# 1. configure  (fill baseline_run_dir with a prior blind generate.py run dir)
cp settings_reseed/EXAMPLE.json settings_reseed/MYTARGET.json && $EDITOR settings_reseed/MYTARGET.json

# 2. pilot (fast, 1 strain / 2 samples / 1 round) to validate end-to-end
sbatch reseed_ablation.slurm --settings settings_reseed/MYTARGET.json \
       --n_strains 1 --samples_per_cut 2 --rounds 1

# 3. full run (canonical 20 strains x 5 cuts x 20 samples x 4 rounds)
sbatch reseed_ablation.slurm --settings settings_reseed/MYTARGET.json
# resume after a checkpointed round:
sbatch reseed_ablation.slurm --settings settings_reseed/MYTARGET.json --resume

# 4. analysis (needs foldseek binary)
python -m pipeline.reseed.analysis --run_dir <output_base_dir>/runs/MYTARGET \
       --settings settings_reseed/MYTARGET.json --foldseek ./bin/foldseek

# 5. viewer
streamlit run visualizer_v2/app.py -- --base_dir <output_base_dir>
```

## Configured targets (`settings_reseed/`)

Founders + RQ1 baseline are drawn from each target's prior blind sweep under
`/nfs/students/matscher/samples/seed_search_*`. Those sweeps vary in how far they were
scored, so `founder_filter_stage` (in the settings) controls the founder source:

| Target | baseline_run_dir | PyRosetta founders | founder source |
|---|---|---|---|
| PDGFR | `seed_search_PDGFR` | ~320 | PyRosetta (auto) — directly usable |
| MUC1 | `seed_search_MUC1` | ~196 | PyRosetta (auto) — directly usable |
| NIPAH | `seed_search_nipah` | ~29 | PyRosetta (auto), folding-supplemented if needed |
| ACTRIIA | `seed_search_AVRA2A` | ~1 | auto → **folding-passing** (few conditions were PyRosetta-scored) |

`founder_filter_stage`: `auto` (PyRosetta successes, supplemented with folding-passing if
fewer than `n_strains`), `pyrosetta` (strict), or `folding` (folding-passing only).
Progeny are always scored with the full two-stage filter regardless.

### Champion selection (`champion_selection`)
Which winner seeds a strain's next round:
- `uniform` (default) — uniform random among the strain's winners (spec §7, **no imposed pressure**).
- `ss_stratified` — bin the winners into a `champion_ss_grid`×`champion_ss_grid` helix×sheet grid over
  [0,1]², pick a **non-empty cell uniformly**, then a winner uniformly within it. This gives uniform
  coverage of SS space and steers lineages into under-sampled β-heavy regions (a lone β winner gets
  ≈1/`n_cells` weight instead of 1/`n_winners`). Falls back to uniform if winners have no SS.

**Interpretation caveat:** under `ss_stratified` the RQ2 SS drift partly reflects this *imposed*
SS-coverage pressure, not re-seeding alone — keep `uniform` as the no-pressure reference. The two arms
are run into separate output trees (e.g. `reseed_output` vs `reseed_output_eq`) for a clean A/B; point the
viewer's `--base_dir` at either.

### Seed residue selection (`seed_selection`)
Which binder residues become the seed:
- `random` (default) — up to `seed_n_res` residues anywhere on the binder (the P1–P5
  fragmentation sweep).
- `interface` — up to `seed_n_res` residues drawn **only from binder residues that contact
  the target hotspots** (within `seed_interface_cutoff` Å; hotspots from
  `seed_interface_hotspots`, else `target_hotspot_residues`; empty ⇒ the whole binding
  interface). Consecutive picks merge into fragments (cut ids `I1..I5`). If a parent has
  fewer than `seed_n_res_min` interface residues that strain is skipped. Meant for
  low-yield targets (e.g. NIPAH) so the seed carries the actual binding geometry rather
  than arbitrary surface residues.

**Caveats:** (1) hotspot ids are matched against the parent complex's **chain-A numbering**
(1-based for refolded complexes) — supply them in that numbering, not necessarily author
numbering. (2) With **fixed** hotspots, founders that bound a *different* epitope have no
interface residues near them and are skipped → this can *increase* extinction; leaving
hotspots empty (whole-interface mode) works for every founder and is usually the safer
first try. (3) The summary CSV only carries **target-side** interface residues
(`target_interface_residues`); the binder-side residues used here are computed
geometrically from the parent complex.

### Budget redistribution (`redistribute_budget`)
When a strain goes extinct its per-round generation budget is normally lost (realized folds/round decay
below `n_strains×n_cuts×samples_per_cut`). With `redistribute_budget: true`, that freed budget is spread
**evenly over the surviving strains** each round: every survivor generates
`round(samples_per_cut × n_strains / n_alive)` samples per cut (e.g. base 20 → 40 when half the strains
have died, → 400 for a lone survivor), holding folds/round ≈ constant. This **deepens the real lineages**
— it does *not* respawn founders, so extinction is still recorded and per-generation semantics stay clean.
It also keeps the RQ1 compute axis full without idle GPU slots. Default `false` (budget decays, the
original spec §7 behaviour).

**ACTRIIA — optional strict founders.** To seed ACTRIIA from genuine PyRosetta founders
instead of folding-passing ones, materialise a scored founder set first:

```bash
python -m pipeline.reseed.prepare_founders --settings settings_reseed/ACTRIIA.json \
       --out /nfs/students/matscher/reseed_output/actriia_founders --max_candidates 300
# then set baseline_run_dir to that --out and founder_filter_stage="pyrosetta"
```

### Run all four (SLURM)
```bash
for T in PDGFR MUC1 NIPAH ACTRIIA; do
  sbatch reseed_ablation.slurm --settings settings_reseed/$T.json
done
```

## Outputs (`<output_base_dir>/runs/<target>/`)
```
gen0/                             founder complexes (generation 0)
gen<k>/strain<ss>/cut<Pn>/sample<i>.{pdb,json}
records.parquet + records.csv     one row/sample, spec §8 schema
lineage.json                      strain -> round -> champion
config.json                       resolved run config (baseline_run_dir, filter stage, …)
checkpoints/<target>_state.json   resume state
analysis/                         per-generation CSVs + PNGs (after step 4)
```

### Cluster novelty vs the seed-search pool
`analysis.cluster_novelty_by_generation` clusters every progeny success **together with the
full prior success set** (all `load_success_set(baseline_run_dir)` binders — including the
gen-0 founders not selected as strains), so each progeny is labelled **novel** (its FoldSeek
cluster contains no prior-pool member) or **existing** (re-generated a seed-search binder).
It answers "did re-seeding reach new structural space or just echo the seeds?" → per
generation: `novel_fraction`, `n_novel_clusters`, `cumulative_novel_clusters`. Written to
`analysis/cluster_novelty.csv` (+ `.png`) by `run_analysis` when a baseline is given, or
on-demand from the viewer's **Diversity → Cluster novelty** panel (baseline auto-filled from
`config.json`). The `analysis --settings <target>.json` CLI supplies the baseline + stage.

## Tests
`python -m pytest tests/test_reseed_seed_extract.py -v` — pure fragment/contig logic runs anywhere;
the motif-PDB test runs where biotite + `top_binders/*.pdb` are available.
