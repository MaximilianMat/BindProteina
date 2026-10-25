# Provenance and attribution

This repository accompanies the master's thesis *Designing High Affinity Binders
across Targets with Generative ML* (Maximilian Matscher, M.Sc., Chair of Data
Analytics and Machine Learning, School of Computation, Information and
Technology, TUM). It mixes three kinds of code, and this document states exactly
which is which, file by file.

The short version:

| Component | Origin | Status |
|---|---|---|
| `pipeline/` (except the ports noted below) | own work | the method of the thesis |
| `designer/` | own work | Seed Designer (thesis Sec. 4, App. A) |
| `visualizer_v2/` | own work | Evolution Viewer (thesis Sec. 4, App. A) |
| `scripts/`, `tests/`, `*.slurm`, `settings_{target,reseed,filters}/` | own work | runs, analysis, configuration |
| `laproteina/` | NVIDIA **La-Proteina** | vendored; 110 files verbatim, 22 modified, 11 added |
| `functions/` | **BindCraft** | vendored; 2 files verbatim, 2 modified |
| `pipeline/scoring/ipsae.py` | Dunbrack lab **ipSAE** | own re-implementation of the published algorithm |

Upstream versions this repository was taken from and diffed against:

- **La-Proteina** — <https://github.com/NVIDIA-Digital-Bio/la-proteina>,
  commit `cde5de3` (2025-09-23). Code under Apache-2.0; released weights under
  the NVIDIA Open Model License. See `licenses/LICENSE-la-proteina-code.txt` and
  `licenses/LICENSE-la-proteina-weights.txt`.
- **BindCraft** — <https://github.com/martinpacesa/BindCraft>, commit `0366085`
  (2025-08-12). MIT, © 2024 Martin Pacesa. See `licenses/LICENSE-bindcraft.txt`.
- **ipSAE** — <https://github.com/DunbrackLab/IPSAE>, commit `b0a5493`
  (2025-04-06). MIT, © 2025 Lab of Dr. Roland Dunbrack. See
  `licenses/LICENSE-ipsae.txt`.

Every "verbatim / modified / added" count below was produced by diffing this
tree against those upstream commits, not by recollection.

---

## 1. What is my own claim

The contribution of the thesis is a *method*, not a model: La-Proteina is a
monomer generator, and this repository is the machinery that turns it into a
binder designer without retraining it. Concretely, the following is my own work.

### `pipeline/` — generation and scoring

| File | Role |
|---|---|
| `target_preparer.py` | target cleaning, epitope handling, seed/linker geometry, chain splitting of La-Proteina output |
| `laproteina_generator.py` | embeds La-Proteina as a library: Hydra composition, checkpoint loading, batched sampling, contig construction |
| `config_builder.py` | builds the generation configuration from a target settings file |
| `generate.py` | the generation entry point: placement grid → generation → pre-scoring |
| `score_existing_two_step.py`, `score_json.py`, `refilter_run.py` | batch scoring entry points over an existing design directory, a Protenix JSON, or a completed run |
| `scoring/scorer.py` | the two-stage scoring orchestration (fold → filter → relax → filter) and the filter evaluation |
| `scoring/folding_protenix.py` | Protenix oracle: input construction, MSA handling, confidence extraction |
| `scoring/folding.py` | AlphaFold2 oracle with the initial-guess protocol (optional alternative) |
| `scoring/pyrosetta_scorer.py` | interface analysis driver on top of the BindCraft PyRosetta helpers |
| `scoring/rmsd.py`, `scoring/utils.py` | self-consistency RMSD and shared helpers |
| `reseed/` (8 files) | the re-seeding ablation of thesis Sec. 5.2.5: founder selection, the five cut patterns P1–P5, seed extraction, the strain loop, and the per-generation analysis (FoldSeek clustering, novelty, secondary-structure shift) |

### `designer/` — Seed Designer

Browser application for interactive seed placement: PeSTo confidence on the
target surface, manual positioning and rotation, clash detection, parameter-grid
expansion, and export of a generation configuration. This is the tool that
produced the sweeps of thesis Sec. 5.2.3.

### `visualizer_v2/` — Evolution Viewer

Streamlit application that browses re-seeding runs by generation: success
counts, the RQ1 compute curve, structural and sequence diversity,
secondary-structure shift, a structure inspector, and the strain → round →
champion lineage view. This is the Evolution Viewer of thesis Sec. 4 and
Appendix A.

> An earlier, greedy evolution driver (`pipeline/evolve_v2.py`) and its NGL-based
> viewer (`visualizer/`) were removed during cleanup for the thesis submission.
> They are first-generation work that the thesis does not report; see the commit
> *"Remove superseded first-generation evolution stack"* to recover them.

### `scripts/`, `tests/`, `*.slurm`, `settings_*/`

Run orchestration (SLURM array sharding and shard merging), the analysis behind
Chapter 5 (seed ablations, parameter sensitivity, secondary-structure and
cluster analyses, interface diversity), the Proteina-Complexa comparison glue,
the wet-lab handoff, and the per-target configuration actually used for the runs.

### `pipeline/scoring/ipsae.py` — a port, clearly marked

A self-contained re-implementation of the *core* ipSAE algorithm of Dunbrack
et al. The reference implementation is a standalone CLI that writes side-files
and loops over all chain pairs; this version is a single pure function for the
target↔binder interface that returns a dict and never touches disk. The
definitions (`ptm_func`, `calc_d0`, the per-residue d0 and the max/min over
directions) are kept identical to the reference, and the file's own header says
so. The algorithm is not my claim; this implementation of it is.

---

## 2. `laproteina/` — vendored La-Proteina

A partial vendoring of the upstream tree: `proteinfoundation/` (the model),
`openfold/` (which upstream itself vendors from OpenFold), `configs/`, and
`script_utils/`. Upstream's own entry points (`train.py`,
`proteinfoundation/generate.py`, `merge_lora_checkpoint.py`) are **not** vendored
— generation is driven from `pipeline/laproteina_generator.py` instead.

**110 files are byte-identical to upstream.** The rest:

### 2.1 Modified upstream files (22)

Counts are added/removed lines against commit `cde5de3`.

| File | +/- | What the change does |
|---|---|---|
| `proteinfoundation/datasets/gen_dataset.py` | +189/−15 | **Direct binder mode.** Adds `direct_binder_mode`, `binder_chain_gap`, `binder_target_nres`: the target enters as the leading motif and the binder is the trailing scaffold segment, with a residue-index gap marking the future chain break. |
| `proteinfoundation/proteina.py` | +42/−5 | **Motif pinning and two-chain output.** Holds motif residue identities and motif atom coordinates fixed in the decoded sample, and threads `chains`, `residue_pdb_idx` and `atom_mask` out of the sampling loop so the result can be split into target and binder chains. Also adds the training-only `loss_on_non_motif_only` branch. |
| `proteinfoundation/utils/pdb_utils.py` | +31/−4 | Chain- and residue-index-aware PDB writing, so a generated sample can be written as a two-chain complex. |
| `proteinfoundation/utils/motif_utils.py` | +26/−1 | Adds `infer_leading_motif_length`, which derives the target length from a contig string. |
| `proteinfoundation/nn/modules/attn_n_transition.py` | +38/−7 | **Optional** relative-position attention bias (`use_relpos_bias`, `relpos_max`) via a learned embedding of clamped residue-index differences. |
| `proteinfoundation/nn/modules/pair_bias_attn.py` | +13/−5 | Accepts that bias as `extra_bias` and adds it to the pair bias. |
| `proteinfoundation/nn/local_latents_transformer.py` | +10/−1 | Threads `residue_pdb_idx` and the bias flags through the indexed transformer. |
| `proteinfoundation/nn/local_latents_transformer_unindexed.py` | +28/−1 | The same for the unindexed variant. |
| `proteinfoundation/nn/feature_factory.py` | +16/−12 | Handles the `all_atom` / `tip_atom` motif modes used by the two checkpoints. |
| `proteinfoundation/flow_matching/product_space_flow_matcher.py` | +22/−1 | Training-only: honours a `loss_mask` so loss can be restricted to non-motif residues. |
| `proteinfoundation/datasets/pdb_data.py` | +42/−1 | Training-only: multimer-aware dataset columns and splits for the fine-tuning attempt. |
| `proteinfoundation/datasets/transforms.py` | +17/0 | Transform plumbing for the above. |
| `proteinfoundation/__init__.py` | +18/−3 | Package docstring describing the vendoring. |
| `configs/generation/motif_dict.yaml` | +18/0 | Adds the `TARGET_BINDER_AA` / `TARGET_BINDER_TIP` motif tasks. |
| `configs/generation/motif.yaml` | +2/−2 | Sample-count defaults. |
| `configs/inference_motif_idx_tip.yaml`, `configs/inference_motif_uidx_aa.yaml` | +1/−1 each | Point at the binder motif task. |
| `configs/nn/local_latents_score_nn_160M*.yaml` (5 files) | +2–3/0 each | Declare `use_relpos_bias: False` and `relpos_max`, i.e. the bias is **off** in every released-checkpoint config. |

### 2.2 Files added inside the vendored tree (11)

| File | Role |
|---|---|
| `configs/generation/motif_binder.yaml` | the binder generation task |
| `configs/inference_motif_idx_aa_binder.yaml` | inference config for the all-atom (LD4) binder path |
| `configs/inference_motif_idx_tip_binder.yaml` | inference config for the tip-atom (LD5) binder path |
| `proteinfoundation/utils/interface_utils.py` | interface geometry helpers used during generation |
| `proteinfoundation/utils/pdb_slicer.py` | target slicing for motif construction |
| `__init__.py` | makes the vendored tree importable |
| `configs/nn/local_latents_score_nn_160M_motif_idx_aa_binder.yaml`, `configs/training_local_latents_binder_ft.yaml`, `configs/dataset/pdb/pdb_train_motif_aa_multimer.yaml`, `proteinfoundation/optimizer.py`, `script_utils/analyze_binder_outputs.py` | the abandoned fine-tuning attempt (Sec. 2.3) |

### 2.3 On the weights, and on what "no changes to inference code" means

**No La-Proteina weights were trained or modified for the reported results.**
The runs use the released `LD4_motif_idx_aa.ckpt` and `LD5_motif_idx_tip.ckpt`
checkpoints, with `AE3_motif.ckpt` as autoencoder, loaded explicitly by
`pipeline/laproteina_generator.py` from the path in the target settings file.

Two points of precision, since the thesis states that no direct changes were
made to La-Proteina's inference code:

1. **Some inference code *was* modified**, as the table in Sec. 2.1 shows. The
   changes are I/O and conditioning plumbing — direct-binder dataset mode,
   motif pinning, chain-aware output — not changes to the generative model or
   its learned behaviour. No layer, loss or sampling rule of the released model
   is altered on that path.
2. **The relative-position attention bias is architectural, and it is off.** It
   adds a learned embedding to attention, so enabling it with released weights
   would mean untrained parameters. It is `False` in all five released-checkpoint
   `configs/nn/` files, and the one config that sets it `True`
   (`local_latents_score_nn_160M_motif_idx_aa_binder.yaml`) is referenced *only*
   by the fine-tuning training config, never by the inference chain. The
   inference configs declare no `nn:` default at all — the architecture comes
   from the checkpoint's stored hyperparameters. So on the reported path the bias
   is dormant.

One leftover is worth knowing: `configs/inference_motif_idx_aa_binder.yaml`
still carries `run_name_: laproteina_ft_reference` and a `ckpt_path` pointing at
a fine-tuned checkpoint directory. Both are overridden at load time by
`laproteina_generator.py`, which loads the checkpoint named in the settings
file, so this does not affect the reported runs.

The fine-tuning attempt itself (LoRA on La-Proteina) was explored and cut. It
backs no number in the thesis — it survives there only in commented-out
conclusion text. Its training-side files are kept (listed in Sec. 2.2); its
launchers and one-off audit scripts were removed during cleanup.

---

## 3. `functions/` — vendored BindCraft

BindCraft is the supportive framework this work borrows its interface-scoring
conventions from. Only four pieces of it are actually reached from this
pipeline: `pr_relax`, `score_interface` and `get_pyrosetta_init_cmd` from
`pyrosetta_utils.py`, `three_to_one_map` from `biopython_utils.py`, `clean_pdb`
from `generic_utils.py`, and the `DAlphaBall.gcc` binary used by PyRosetta's
shape-complementarity calculation.

| File | Status vs. BindCraft `0366085` |
|---|---|
| `biopython_utils.py` | **verbatim** |
| `generic_utils.py` | 2 lines changed |
| `pyrosetta_utils.py` | 123 lines changed — the interface-analysis entry points were adapted to this pipeline's chain convention and threading model |
| `__init__.py` | one star-import line removed during cleanup |
| `DAlphaBall.gcc` | upstream binary, unmodified |

Two honest caveats:

- `generic_utils.py` is kept verbatim apart from those 2 lines, which means it
  still contains BindCraft CLI helpers (`perform_input_check`,
  `perform_advanced_settings_check`) that are never called from here and whose
  default paths now point at files removed during cleanup
  (`settings_advanced/`, `functions/dssp`). Trimming vendored code would make
  the upstream diff harder to verify, so it was left as it is.
- `settings_filters/` is derived from BindCraft's filter files — five of them
  started as upstream copies and carry roughly 26 added lines each (the
  binder-level pTM/ipTM entries). `laProteina_filters.json` is the filter set
  actually used for the reported results and is my own.

Other upstream projects are **used as dependencies, not vendored**: ColabDesign
(imported by the AF2 oracle), Protenix (run as a separate conda environment),
PyRosetta, PeSTo, FoldSeek and Proteina-Complexa (the ICLR 2026 NVIDIA model
used as a comparison, cloned separately).

---

## 4. Reproducing these counts

Nothing here needs to be taken on trust:

```bash
git clone https://github.com/NVIDIA-Digital-Bio/la-proteina /tmp/la-proteina
git -C /tmp/la-proteina checkout cde5de3
for f in $(git ls-files laproteina | sed 's|^laproteina/||'); do
    diff -q "laproteina/$f" "/tmp/la-proteina/$f" 2>/dev/null \
      || echo "differs or added: $f"
done

git clone https://github.com/martinpacesa/BindCraft /tmp/bindcraft
git -C /tmp/bindcraft checkout 0366085
diff functions/pyrosetta_utils.py /tmp/bindcraft/functions/pyrosetta_utils.py
```
