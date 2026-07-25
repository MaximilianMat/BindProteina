"""
Re-seeding ablation package.

Implements the progeny / re-seeding evolutionary ablation described in
`reseed_ablation_spec.md`:

  - Founders (generation 0) = post-filter successful binders from a prior blind run.
  - Each founder seeds an independent *strain*.
  - Per round, a *parent* (the strain's current champion) is fragmented into five
    seed-cuts (P1..P5); each cut is scaffolded by LaProteina into 20 progeny.
  - Progeny are refolded + scored with the existing pipeline; the champion for the
    next round is drawn *uniformly at random* from that strain's passing samples.

The package deliberately does not modify `pipeline/evolve_v2.py`, `pipeline/generate.py`,
or the scoring stack — it only reuses them.
"""
