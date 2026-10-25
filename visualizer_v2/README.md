# Re-seeding Ablation Viewer (`visualizer_v2`)

Streamlit app for inspecting a re-seeding ablation run (`runs/<target>/`) **by generation**.

## Run

```bash
pip install -r visualizer_v2/requirements.txt
streamlit run visualizer_v2/app.py -- --base_dir /nfs/.../reseed_output
```

`--base_dir` is scanned recursively for `records.parquet` (or `records.csv`); pick a run
and a generation in the sidebar.

## Tabs
- **Overview** — per-generation success counts, RQ1 cumulative-successes-vs-compute
  (re-seed vs blind baseline, if analysis was run), extinction + near-copy fractions.
- **Diversity** — FoldSeek structural diversity (cluster ratio, mean pairwise TM) and
  lightweight pairwise sequence identity per generation.
- **Secondary structure** — binder-relative helix/sheet/loop by generation (RQ2), the
  per-generation distribution, progeny SS by cut SS class (RQ2b), and KS / earth-mover shift.
- **Structure inspector** — py3Dmol view of any sample's complex (target grey, binder
  spectrum, **seed residues highlighted**) with its metrics.
- **Lineage** — strain → round → champion chain (extinct strains marked).

The Diversity / RQ1 / SS-shift panels read precomputed CSVs from
`runs/<target>/analysis/`; generate them first with:

```bash
python -m pipeline.reseed.analysis --run_dir runs/<target> --settings settings_reseed/<target>.json --foldseek ./bin/foldseek
```
