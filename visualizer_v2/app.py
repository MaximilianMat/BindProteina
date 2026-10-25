"""
Streamlit viewer for the re-seeding ablation.

Run:
    streamlit run visualizer_v2/app.py -- --base_dir /nfs/.../reseed_output

Lets you pick a run and a generation, then inspect (per generation): success counts and
the RQ1 cumulative-vs-compute curve, structural + sequence diversity, secondary-structure
distributions (RQ2), individual structures with their seed residues highlighted, and the
strain -> round -> champion lineage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualizer_v2 import data as D

st.set_page_config(page_title="Re-seeding Ablation Viewer", layout="wide")


# --------------------------------------------------------------------------------------
# Structure rendering (py3Dmol)
# --------------------------------------------------------------------------------------

def render_complex(pdb_text: str, seed_local_indices=None, height: int = 480) -> str:
    import py3Dmol
    view = py3Dmol.view(width="100%", height=height)
    view.addModel(pdb_text, "pdb")
    # target grey, binder spectrum
    view.setStyle({"chain": "A"}, {"cartoon": {"color": "lightgray", "opacity": 0.8}})
    view.setStyle({"chain": "B"}, {"cartoon": {"color": "spectrum"}})
    if seed_local_indices:
        view.addStyle({"chain": "B", "resi": [int(i) for i in seed_local_indices]},
                      {"stick": {"colorscheme": "orangeCarbon", "radius": 0.3}})
        view.addStyle({"chain": "B", "resi": [int(i) for i in seed_local_indices]},
                      {"sphere": {"colorscheme": "orangeCarbon", "scale": 0.3}})
    view.zoomTo()
    return view._make_html()


# --------------------------------------------------------------------------------------
# Sidebar: run + generation selection
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading all runs …")
def load_global_cached(base_dir: str, nonce: int):
    return D.global_trends(base_dir)


def _base_dir_default() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--base_dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "."


st.sidebar.title("Re-seeding Ablation")
base_dir = st.sidebar.text_input("Base output dir", value=_base_dir_default())
runs = D.find_runs(base_dir)
if not runs:
    st.warning(f"No runs (records.parquet/csv) found under `{base_dir}`.")
    st.stop()

run_labels = {str(p.relative_to(base_dir)) if Path(base_dir) in p.parents or str(p).startswith(base_dir) else str(p): p
              for p in runs}
choice = st.sidebar.selectbox("Run", list(run_labels.keys()))
run = D.load_run(str(run_labels[choice]))

gens = run.generations
gen = st.sidebar.select_slider("Generation", options=gens, value=gens[0])

st.sidebar.markdown(f"**Target:** {run.target}")
st.sidebar.markdown(f"**Generations:** {gens}")

summary = D.generation_summary(run)

(tab_overview, tab_metrics, tab_global, tab_div, tab_ss, tab_struct,
 tab_lineage) = st.tabs(
    ["Overview", "Metrics", "Global trends", "Diversity", "Secondary structure",
     "Structure inspector", "Lineage"])


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
with tab_overview:
    st.header(f"{run.target} — overview")
    c1, c2, c3, c4 = st.columns(4)
    grow = summary[summary["generation"] == gen]
    if len(grow):
        r = grow.iloc[0]
        c1.metric("Generated", int(r["n_generated"]))
        c2.metric("Folding-pass",
                  "-" if pd.isna(r["n_folding_pass"]) else int(r["n_folding_pass"]))
        c3.metric("Successes", int(r["n_success"]))
        c4.metric("Success rate", f"{r['success_rate']*100:.2f}%")
    st.caption(
        "**Generated** = all samples produced this generation (chain-split complexes fed "
        "to scoring). **Folding-pass** cleared the refolding prefilter. **Successes** passed "
        "the full PyRosetta filter. Success rate = Successes / Generated (not / Folding-pass)."
    )
    st.subheader("Per-generation summary")
    st.dataframe(summary, use_container_width=True)

    if "rq1_cumulative_reseed" in run.analysis:
        st.subheader("RQ1 — cumulative successes vs compute")
        reseed = run.analysis["rq1_cumulative_reseed"]
        chart_df = reseed[["sample_index", "cumulative_successes"]].set_index("sample_index")
        if "rq1_cumulative_baseline" in run.analysis:
            base = run.analysis["rq1_cumulative_baseline"][["sample_index", "cumulative_successes"]]
            base = base.rename(columns={"cumulative_successes": "baseline"}).set_index("sample_index")
            chart_df = chart_df.rename(columns={"cumulative_successes": "re-seed"}).join(base, how="outer")
        st.line_chart(chart_df)
    if "extinction" in run.analysis:
        st.subheader("Extinction rate per generation")
        st.dataframe(run.analysis["extinction"], use_container_width=True)
    if "novelty" in run.analysis:
        st.subheader("Near-copy (novelty) fraction per generation")
        st.dataframe(run.analysis["novelty"], use_container_width=True)


# --------------------------------------------------------------------------------------
# Metrics — mean / min / max evolution over generations
# --------------------------------------------------------------------------------------
with tab_metrics:
    st.header("Metric evolution over generations")
    df = run.records
    metric_cols = D.numeric_metric_columns(df)
    if not metric_cols:
        st.info("No numeric metrics in records yet.")
    else:
        default = "i_pTM" if "i_pTM" in metric_cols else metric_cols[0]
        c1, c2 = st.columns([2, 1])
        metric = c1.selectbox("Metric", metric_cols, index=metric_cols.index(default),
                              key="metrics_tab_metric")
        only_success = c2.checkbox("Only successful samples", value=True, key="metric_succ")

        ev = D.metric_evolution(run, metric, only_success)
        ev_valid = ev.dropna(subset=["mean"])
        if len(ev_valid):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.fill_between(ev_valid["generation"], ev_valid["min"], ev_valid["max"],
                            color="tab:blue", alpha=0.15, label="min–max range")
            ax.plot(ev_valid["generation"], ev_valid["max"], color="tab:blue", ls=":", lw=1)
            ax.plot(ev_valid["generation"], ev_valid["min"], color="tab:blue", ls=":", lw=1)
            ax.plot(ev_valid["generation"], ev_valid["mean"], color="tab:blue",
                    marker="o", lw=2, label="mean")
            ax.plot(ev_valid["generation"], ev_valid["median"], color="tab:orange",
                    marker="s", lw=1, ls="--", label="median")
            ax.set_xlabel("generation")
            ax.set_ylabel(metric)
            ax.set_title(f"{run.target} — {metric} over generations")
            ax.set_xticks([int(g) for g in ev_valid["generation"]])
            ax.legend(fontsize=8)
            st.pyplot(fig)
        else:
            st.info(f"No numeric values for {metric}.")

        st.subheader(f"{metric} — per-generation statistics")
        st.dataframe(ev, use_container_width=True)

        with st.expander("All metrics — mean by generation"):
            means = D.metric_means_by_generation(run, metric_cols, only_success)
            st.dataframe(means, use_container_width=True)


# --------------------------------------------------------------------------------------
# Global trends across all targets
# --------------------------------------------------------------------------------------
def _plot_target_trends(d, ycol, ylabel, title, delta):
    """Thin per-target lines + bold global mean vs generation. Optional Δ-vs-gen0."""
    import matplotlib.pyplot as plt
    d = d.dropna(subset=[ycol]).copy()
    if not len(d):
        st.info(f"No data for {ylabel}.")
        return
    if delta:
        base = d[d["generation"] == 0].set_index("target")[ycol]
        d[ycol] = d.apply(lambda r: r[ycol] - base.get(r["target"], float("nan")), axis=1)
        d = d.dropna(subset=[ycol])
        if not len(d):
            st.info(f"No Δ data for {ylabel} (missing generation-0 baseline).")
            return
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in sorted(d["target"].unique()):
        s = d[d["target"] == t].sort_values("generation")
        ax.plot(s["generation"], s[ycol], color="lightgray", lw=1, marker="o", ms=3, zorder=1)
    g = d.groupby("generation")[ycol].mean()
    ax.plot(g.index, g.values, color="tab:red", lw=2.5, marker="o",
            label="global mean", zorder=3)
    ax.set_xlabel("generation")
    ax.set_ylabel(("Δ " if delta else "") + ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(int(x) for x in d["generation"].unique()))
    if delta:
        ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    st.pyplot(fig)


with tab_global:
    st.header("Global trends across all targets")
    st.caption("Aggregated over every run under the base dir (success set per "
               "target/generation). Grey = individual targets, red = global mean.")
    if st.button("Reload all runs", key="global_reload"):
        st.session_state["global_nonce"] = st.session_state.get("global_nonce", 0) + 1
        st.rerun()

    G = load_global_cached(base_dir, st.session_state.get("global_nonce", 0))
    mdf, scdf = G["metrics"], G["ss_cluster"]
    if scdf.empty:
        st.info("No runs with data found under the base dir.")
    else:
        delta = st.checkbox("Show as change vs generation 0 (Δ)", value=True,
                            key="global_delta")
        st.caption(f"Targets: {', '.join(sorted(scdf['target'].unique()))}")

        st.subheader("Secondary-structure diversity increase")
        _plot_target_trends(scdf, "ss_diversity",
                            "SS diversity (mean pairwise dist, helix/sheet)",
                            "SS diversity over generations", delta)

        st.subheader("Cluster diversity increase (FoldSeek)")
        if scdf["cluster_hit_fraction"].notna().any():
            _plot_target_trends(scdf, "cluster_hit_fraction",
                                "clusters / successes", "Structural cluster diversity", delta)
            if scdf["mean_pairwise_tm"].notna().any():
                _plot_target_trends(scdf, "mean_pairwise_tm",
                                    "mean pairwise TM (lower = more diverse)",
                                    "Structural similarity", delta)
        else:
            st.info("No cluster data yet — run each target's Diversity → 'Compute "
                    "diversity' (needs FoldSeek); results are then pooled here.")

        st.subheader("Metric trend across targets")
        if not mdf.empty:
            allm = sorted(mdf["metric"].unique())
            dm = allm.index("i_pTM") if "i_pTM" in allm else 0
            met = st.selectbox("Metric", allm, index=dm, key="global_tab_metric")
            _plot_target_trends(mdf[mdf["metric"] == met].rename(columns={"mean": met}),
                                met, met, f"{met} across targets", delta)

        st.subheader("Global summary (mean across targets per generation)")
        gsum = scdf.groupby("generation")[
            ["n_success", "ss_diversity", "cluster_hit_fraction",
             "mean_pairwise_tm", "singleton_fraction"]].mean()
        st.dataframe(gsum, use_container_width=True)


# --------------------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------------------
with tab_div:
    st.header("Structural & sequence diversity")

    # On-demand computation (reuses pipeline.reseed.analysis). FoldSeek clustering of a
    # few hundred small chain-B PDBs per generation is fast (seconds/gen); results are
    # cached to runs/<target>/analysis/ so it only runs when asked.
    with st.expander("Compute / refresh diversity now", expanded="structural_diversity" not in run.analysis):
        foldseek_bin = st.text_input("FoldSeek binary (leave for PATH / ./bin/foldseek)",
                                     value="./bin/foldseek", key="div_foldseek_bin")
        st.caption("Needs the FoldSeek binary on this host. Uses the archived per-sample "
                   "structures under runs/<target>/gen*/ — safe to run while the ablation runs.")
        if st.button("Compute diversity (FoldSeek + sequence)", key="div_compute_btn"):
            with st.spinner("FoldSeek clustering + sequence identity ..."):
                try:
                    from pipeline.reseed import analysis as A
                    adir = run.run_dir / "analysis"
                    adir.mkdir(exist_ok=True)
                    sd = A.structural_diversity_by_generation(
                        run.records, str(run.run_dir), foldseek_bin or None)
                    sq = A.sequence_diversity_by_generation(run.records)
                    sd.to_csv(adir / "structural_diversity.csv", index=False)
                    sq.to_csv(adir / "sequence_diversity.csv", index=False)
                    st.success("Done — reloading.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Diversity computation failed: {e}")

    if "structural_diversity" in run.analysis:
        sd = run.analysis["structural_diversity"]
        st.subheader("Structural (FoldSeek)")
        st.dataframe(sd, use_container_width=True)
        cols = [c for c in ("mean_pairwise_tm", "cluster_hit_fraction", "singleton_fraction")
                if c in sd.columns]
        if cols:
            st.line_chart(sd.set_index("generation")[cols])
    else:
        st.info("No structural-diversity CSV yet — use *Compute diversity* above "
                "(or run `python -m pipeline.reseed.analysis`).")
    if "sequence_diversity" in run.analysis:
        st.subheader("Sequence (pairwise identity)")
        sq = run.analysis["sequence_diversity"]
        st.line_chart(sq.set_index("generation")[["mean_pairwise_identity"]])

    # ---- Cluster novelty vs the full prior success pool -----------------------------
    st.markdown("---")
    st.subheader("Cluster novelty vs seed-search pool")
    st.caption("Clusters every progeny success TOGETHER with the FULL prior success set "
               "(all seed-search successes, incl. gen-0 founders not used as strains) to "
               "tell NEW clusters from re-generated existing binders.")
    with st.expander("Compute cluster novelty",
                     expanded="cluster_novelty" not in run.analysis):
        base_default = str(run.config.get("baseline_run_dir", "")) if run.config else ""
        base_dir_in = st.text_input("baseline_run_dir (prior seed-search success pool)",
                                    value=base_default, key="nov_baseline")
        stage_in = st.text_input("founder_filter_stage",
                                 value=str(run.config.get("founder_filter_stage", "auto")),
                                 key="nov_stage")
        st.caption("Auto-filled from runs/<target>/config.json when present. Heavier than "
                   "per-generation diversity (extracts the whole reference pool once).")
        if st.button("Compute cluster novelty (FoldSeek)", key="nov_compute_btn"):
            if not base_dir_in:
                st.error("Provide baseline_run_dir (the prior seed-search run).")
            else:
                with st.spinner("Clustering progeny + full reference pool ..."):
                    try:
                        from pipeline.reseed import analysis as A
                        adir = run.run_dir / "analysis"
                        adir.mkdir(exist_ok=True)
                        nv = A.cluster_novelty_by_generation(
                            run.records, str(run.run_dir), base_dir_in,
                            foldseek_bin=st.session_state.get("div_foldseek_bin") or None,
                            filter_stage=stage_in or "auto")
                        if len(nv):
                            nv.to_csv(adir / "cluster_novelty.csv", index=False)
                            st.success("Done — reloading.")
                            st.rerun()
                        else:
                            st.warning("No novelty result (no reference pool or no progeny "
                                       "resolved). Check baseline_run_dir / FoldSeek.")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Cluster-novelty computation failed: {e}")

    if "cluster_novelty" in run.analysis:
        nv = run.analysis["cluster_novelty"]
        if len(nv):
            r0 = nv.iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Reference successes", int(r0.get("n_reference", 0)))
            c2.metric("Reference clusters", int(r0.get("n_reference_clusters", 0)))
            st.dataframe(nv, use_container_width=True)
            plot_cols = [c for c in ("novel_fraction",) if c in nv.columns]
            if plot_cols:
                st.bar_chart(nv.set_index("generation")[plot_cols])
            if "cumulative_novel_clusters" in nv.columns:
                st.line_chart(nv.set_index("generation")[["cumulative_novel_clusters"]])


# --------------------------------------------------------------------------------------
# Secondary structure (RQ2)
# --------------------------------------------------------------------------------------
with tab_ss:
    st.header("Secondary structure by generation (RQ2)")
    df = run.records
    ss_all = df.dropna(subset=["ss_helix_frac", "ss_sheet_frac"]).copy()
    if not len(ss_all):
        st.info("No secondary-structure data in records yet.")
    else:
        cola, colb = st.columns([1, 1])
        only_success = cola.checkbox("Only successful samples", value=True, key="ss_succ")
        pool = ss_all[ss_all["passed_filter"].apply(bool)] if only_success else ss_all
        strains = sorted(int(s) for s in pool["strain_id"].dropna().unique())
        sel = colb.selectbox("Lineage (strain)",
                             ["All lineages"] + [f"strain {s}" for s in strains],
                             key="ss_lineage")

        # Grey baseline = the starting samples (generation-0 founders, all strains).
        base = ss_all[ss_all["generation"] == 0]
        if sel == "All lineages":
            prog = pool[pool["generation"] > 0]
        else:
            sid = int(sel.split()[-1])
            prog = pool[(pool["strain_id"] == sid) & (pool["generation"] > 0)]
        max_gen = int(pool["generation"].max()) if len(pool) else 1

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        if len(base):
            ax.scatter(base["ss_helix_frac"], base["ss_sheet_frac"], c="lightgrey",
                       s=28, edgecolors="none", label="baseline (gen 0)", zorder=1)
        if len(prog):
            sc = ax.scatter(prog["ss_helix_frac"], prog["ss_sheet_frac"],
                            c=prog["generation"], cmap="viridis",
                            vmin=1, vmax=max(2, max_gen), s=40,
                            edgecolors="black", linewidths=0.3, zorder=2)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("generation")
            if max_gen <= 12:
                cb.set_ticks(list(range(1, max_gen + 1)))
        # Fixed [0,1] axes on both — no min/max normalisation, so shifts are comparable.
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("helix fraction")
        ax.set_ylabel("sheet fraction")
        ax.set_title(f"{run.target} — SS shift ({sel})")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig)
        st.caption("Grey = starting samples (generation-0 founders). Coloured points = the "
                   "selected lineage's samples, shaded light→dark by generation, so the SS "
                   "shift is visible against the starting distribution. Axes fixed to [0,1].")

        means = pool.groupby("generation")[
            ["ss_helix_frac", "ss_sheet_frac", "ss_loop_frac"]].mean()
        st.subheader("Mean SS fractions by generation")
        st.bar_chart(means)
        if "ss_by_seed_ss" in run.analysis:
            st.subheader("Progeny SS by cut SS class (RQ2b)")
            st.dataframe(run.analysis["ss_by_seed_ss"], use_container_width=True)
        if "secondary_structure" in run.analysis:
            st.subheader("SS shift (KS / earth-mover vs previous generation)")
            st.dataframe(run.analysis["secondary_structure"], use_container_width=True)


# --------------------------------------------------------------------------------------
# Structure inspector
# --------------------------------------------------------------------------------------
with tab_struct:
    st.header(f"Structure inspector — generation {gen}")
    import streamlit.components.v1 as components
    df = run.records
    gsub = df[df["generation"] == gen].copy()
    only_success = st.checkbox("Only successful samples", value=True, key="struct_succ")
    if only_success:
        gsub = gsub[gsub["passed_filter"].astype(bool)]
    if not len(gsub):
        st.info("No samples for this generation / filter.")
    else:
        sample_id = st.selectbox("Sample", list(gsub["sample_id"]), key="struct_sample")
        row = gsub[gsub["sample_id"] == sample_id].iloc[0]
        pdb = D.sample_pdb_path(run, row)
        cols = st.columns([2, 1])
        with cols[0]:
            if pdb and pdb.exists():
                seed_ids = D.seed_global_ids(row, target_len=0)  # chain B renumbered 1..
                components.html(render_complex(pdb.read_text(), seed_ids), height=500)
            else:
                st.warning("Structure file not found on disk.")
        with cols[1]:
            st.markdown(f"**strain** {row['strain_id']} · **cut** {row.get('cut_id')} "
                        f"({row.get('cut_pattern')})")
            st.markdown(f"**parent** `{row.get('parent_id')}`")
            for key in ("i_pTM", "pTM", "pLDDT", "ipsae", "ipsae_min", "interface_dG",
                        "interface_sc", "rmsd_to_parent", "seqid_to_parent",
                        "ss_helix_frac", "ss_sheet_frac", "ss_loop_frac"):
                if key in row and pd.notna(row[key]):
                    st.text(f"{key}: {row[key]}")


# --------------------------------------------------------------------------------------
# Lineage
# --------------------------------------------------------------------------------------
with tab_lineage:
    st.header("Strain → round → champion lineage")
    if not run.lineage:
        st.info("No lineage.json found.")
    else:
        rows = []
        for sid, info in sorted(run.lineage.items(), key=lambda kv: int(kv[0])):
            rec = {"strain": int(sid), "founder": info.get("founder_id")}
            for k, champ in sorted(info.get("rounds", {}).items(), key=lambda kv: int(kv[0])):
                rec[f"gen{k}"] = champ if champ is not None else "† extinct"
            rows.append(rec)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
