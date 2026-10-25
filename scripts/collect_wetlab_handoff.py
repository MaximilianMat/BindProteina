#!/usr/bin/env python3
"""
Build a wet-lab handoff package from scored BindProteina runs.

This is a *packaging* step, not a new analysis. It sits on top of two things
that already exist and deliberately does not recompute either of them:

- scoring metrics come from ``scripts/analyze_successful_binders.py`` -- its
  helpers are imported, so ipSAE_min / dG-dSASA / meta_interaction are computed
  by the exact same code path and cannot drift from the numbers in that script;
- structural diversity comes from the Foldseek run's
  ``cluster_assignments_with_metadata.csv``, so the constructs you hand over are
  the same clusters as the figures in your presentation.

What it writes into ``--output``:

    binder_stats.csv    every design that passed the PyRosetta filters, with
                        all scoring columns plus cluster_id -- the evidence set
    top_binders.csv     the selected constructs, wet-lab column order
    <construct_id>.pdb  one relaxed PDB per selected construct

Selection
---------
Ranking is by ``meta_interaction`` (ipSAE_min x dG/dSASA, ascending -- more
negative is better; Overath et al. bioRxiv 2025.08.14.670059, max-F1 threshold
-1.5).

Picking the global top-N by that score alone is the wrong deliverable: seed
search emits many near-identical ``linked_sample_*`` from one seed, so a naive
top-N is largely one structure repeated. Instead this walks **round robin**:

    round 1: best unused cluster for method A, then for method B, then A, ...
    round 2: same, once every cluster has contributed a construct
    ...

Each round takes at most one design per cluster, so the first N picks are N
distinct folds. With ``--balance-by source`` (default) the next construct always
goes to the method holding fewest so far, giving an even split -- a top-N drawn
purely by score can come out 20:4 and then the wet lab cannot compare the two
methods at all. Use ``--balance-by none`` if you only care about the best
binders regardless of where they came from.

Diversity outranks balance: if one method simply does not span enough clusters,
it contributes fewer constructs rather than the selection falling back on
near-duplicates to even the count. Check the per-method line in the summary if
the split matters to you.

PDBs are always **copied**, never moved.

Example
-------
    python scripts/collect_wetlab_handoff.py \\
        --input $S/CD19_complexa_score_protenix_beam-search \\
                $S/CD19_complexa_score_protenix_best-of-n \\
                $S/seed_search_cd19 $S/seed_search_cd19_expand10k \\
                $R/cd19_v2 $R/cd19_v4 \\
        --labels "Proteina Complexa" "Proteina Complexa" \\
                 "LaProteina" "LaProteina" "LaProteina" "LaProteina" \\
        --clusters $S/foldseek_clusters_CD19_complexa_vs_laproteina+/cluster_assignments_with_metadata.csv \\
        --ss-csv $S/ss_analysis_CD19_complexa_vs_laproteina+/secondary_structure_data.csv \\
        --target CD19 \\
        --top-n 24 \\
        --output $S/cd19_wetlab_handoff
"""

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Reuse the scoring logic verbatim rather than reimplementing it. The module is
# main-guarded and has no import-time side effects, so this cannot perturb it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_successful_binders import (  # noqa: E402
    _detect_column,
    _detect_optional_column,
    _find_pdb,
    _gather_csv_files,
    _get_sample_name,
    _is_truthy,
    _load_csv,
    _safe_float,
)

# Wet-lab column order. The operator orders DNA, so sequence comes first after
# the ID; everything else is justification.
HANDOFF_COLUMNS = [
    "construct_id",
    "sequence",
    "binder_length",
    "method",
    "cluster_id",
    "tier",
    "ipsae_min",
    "dg_sasa_ratio",
    "meta_interaction",
    "i_ptm",
    "binder_plddt",
    "passes_meta_threshold",
    "helix_frac",
    "sheet_frac",
    "original_path",
]


# ============================================================
# SCORING RECORDS (mirrors analyze_successful_binders.analyze)
# ============================================================

def build_records(roots: List[Path],
                  labels: Optional[List[str]],
                  dg_sasa_scale: float = 1.0) -> pd.DataFrame:
    """Collect every scored design under *roots* into one DataFrame.

    Same traversal, column detection and derived fields as
    ``analyze_successful_binders.analyze`` so the two scripts agree design for
    design. Designs whose ipTM or pLDDT cannot be read are skipped, matching
    that script.
    """
    csv_files = _gather_csv_files(roots, labels)
    if not csv_files:
        sys.exit("[ERROR] No scoring CSVs found under the given --input roots.")
    print(f"Reading {len(csv_files)} scoring CSVs ...")

    records: List[dict] = []
    for csv_path, csv_root, csv_label in csv_files:
        df = _load_csv(csv_path)
        if df is None or df.empty:
            # Worth saying out loud: an empty scoring_summary.csv silently removes
            # a whole run from the candidate pool.
            print(f"[WARN] Skipping {csv_path}: empty CSV (no data rows)")
            continue
        try:
            iptm_col = _detect_column(df, ["iptm", "ipTM", "ip_tm", "i_pTM", "i_ptm"])
            plddt_col = _detect_column(df, ["plddt", "pLDDT", "plddt_score"])
        except KeyError as exc:
            print(f"[WARN] Skipping {csv_path}: {exc}")
            continue

        pyrosetta_col = _detect_optional_column(df, ["pyrosetta_filter_pass"])
        ipsae_col = _detect_optional_column(df, ["ipSAE_min", "ipsae_min"])
        dgsasa_col = _detect_optional_column(
            df, ["dG/dSASA", "dG_SASA_ratio", "dg_dsasa_ratio"])
        binder_plddt_col = _detect_optional_column(df, ["binder_pLDDT", "binder_plddt"])
        seq_col = _detect_optional_column(df, ["sequence"])
        length_col = _detect_optional_column(df, ["binder_length"])

        for _, row in df.iterrows():
            if pd.isna(row[iptm_col]) or pd.isna(row[plddt_col]):
                continue
            sample_name = _get_sample_name(row) or csv_path.parent.name

            rec = row.to_dict()
            rec.update({
                "sample_name": sample_name,
                "i_ptm": float(row[iptm_col]),
                "plddt": float(row[plddt_col]),
                "pyrosetta_filter_pass": (
                    _is_truthy(row[pyrosetta_col]) if pyrosetta_col is not None else False
                ),
                "source": csv_label,
                "source_root": str(csv_root),
                "source_csv": str(csv_path),
                "csv_dir": str(csv_path.parent),
                "ipsae_min": _safe_float(row.get(ipsae_col)) if ipsae_col else float("nan"),
                "binder_plddt": (
                    _safe_float(row.get(binder_plddt_col)) if binder_plddt_col else float("nan")
                ),
                "sequence": str(row.get(seq_col, "")) if seq_col else "",
                "binder_length": (
                    _safe_float(row.get(length_col)) if length_col else float("nan")
                ),
            })
            dg_raw = _safe_float(row.get(dgsasa_col)) if dgsasa_col else float("nan")
            rec["dg_sasa_ratio"] = dg_raw * dg_sasa_scale
            records.append(rec)

    if not records:
        sys.exit("[ERROR] No valid scoring records extracted.")

    df_all = pd.DataFrame.from_records(records)
    df_all["meta_interaction"] = df_all["ipsae_min"] * df_all["dg_sasa_ratio"]
    return df_all


# ============================================================
# PDB RESOLUTION
# ============================================================

def resolve_pdbs(df: pd.DataFrame, pdb_dir: str) -> pd.Series:
    """Resolve each design to its relaxed PDB on disk.

    ``_find_pdb`` rglobs the whole run directory per design, which is minutes of
    NFS traffic for a few hundred designs. Almost every design lives in
    ``<csv_dir>/<pdb_dir>/<name>[_relaxed].pdb``, so list that directory once per
    run and serve lookups from the cache; only true misses fall through to the
    original rglob.
    """
    cache: Dict[str, Dict[str, str]] = {}
    out: List[Optional[str]] = []

    for csv_dir, name in zip(df["csv_dir"], df["sample_name"]):
        index = cache.get(csv_dir)
        if index is None:
            index = {}
            leaf = Path(csv_dir) / pdb_dir
            if leaf.is_dir():
                for p in leaf.glob("*.pdb"):
                    if p.name.startswith("._"):
                        continue
                    stem = p.stem
                    index[stem] = str(p)
                    if stem.endswith("_relaxed"):
                        index[stem[: -len("_relaxed")]] = str(p)
            cache[csv_dir] = index

        hit = index.get(name) or index.get(f"{name}_relaxed")
        if hit is None:
            found = _find_pdb(name, Path(csv_dir))
            hit = str(found) if found else None
        out.append(hit)

    return pd.Series(out, index=df.index, dtype="object")


# ============================================================
# CLUSTER JOIN
# ============================================================

def _norm(path: str, strip: Optional[str]) -> str:
    """Normalize a path for joining. Pure string work -- no stat() on NFS."""
    p = os.path.normpath(str(path))
    if strip and p.startswith(strip):
        p = p[len(strip):]
        if not p.startswith("/"):
            p = "/" + p
    return p


def attach_clusters(df: pd.DataFrame,
                    clusters_csv: Path,
                    local_strip: Optional[str],
                    cluster_strip: Optional[str]) -> pd.DataFrame:
    """Join cluster_id onto *df* from the Foldseek run's metadata CSV."""
    clusters = pd.read_csv(clusters_csv)
    for col in ("original_path", "cluster_id"):
        if col not in clusters.columns:
            sys.exit(f"[ERROR] {clusters_csv} has no '{col}' column -- is this the "
                     f"cluster_assignments_with_metadata.csv from the Foldseek run?")

    lookup = {
        _norm(p, cluster_strip): c
        for p, c in zip(clusters["original_path"], clusters["cluster_id"])
    }
    keys = df["pdb_path"].map(lambda p: _norm(p, local_strip) if p else "")
    df = df.copy()
    df["cluster_id"] = keys.map(lookup)

    n_hit = int(df["cluster_id"].notna().sum())
    print(f"  cluster join: {n_hit}/{len(df)} accepted designs matched "
          f"({clusters['cluster_id'].nunique()} clusters in {clusters_csv.name})")
    if n_hit == 0 and len(df):
        # Almost always a mount-prefix mismatch (/nfs/... vs a local mirror).
        example_local = next((p for p in df["pdb_path"] if p), "<none>")
        example_clust = str(clusters["original_path"].iloc[0])
        print("[ERROR] Nothing matched. The two path spaces look different:\n"
              f"        local   : {example_local}\n"
              f"        clusters: {example_clust}\n"
              "        Line them up with --local-path-prefix / --cluster-path-prefix.")
        sys.exit(1)
    return df


# ============================================================
# ROUND-ROBIN SELECTION
# ============================================================

def round_robin_select(df: pd.DataFrame,
                       n: int,
                       groups: Optional[List[str]],
                       quota: Optional[Dict[str, int]] = None) -> List[int]:
    """Pick *n* designs, cycling groups and never reusing a cluster in a round.

    Returns row labels of *df* in selection order.

    ``groups`` is the list of source labels to alternate between, or ``None`` to
    treat everything as one pool. Within a group, designs are consumed best
    (most negative ``meta_interaction``) first.

    ``max_per_cluster`` starts at 1 and only rises when a full round produces
    nothing, i.e. once every cluster already holds a construct. So the first
    ``n_clusters`` picks are guaranteed structurally distinct, and asking for
    more than that degrades gracefully instead of failing.

    ``quota`` caps how many constructs each group may take. With a quota the
    cluster limit is what gives way when a group runs short -- a group at its cap
    stops competing, so the shortfall cannot be silently handed to the other
    group. Without one, group counts float and cluster distinctness is preserved
    instead.
    """
    ranked = df.sort_values("meta_interaction", ascending=True, kind="mergesort")
    if groups is None:
        queues = {"__all__": list(ranked.index)}
        order = ["__all__"]
    else:
        queues = {g: list(ranked.index[ranked["source"] == g]) for g in groups}
        order = [g for g in groups if queues[g]]
    if not order:
        return []

    cluster_of = df["cluster_id"].to_dict()
    cluster_use: Counter = Counter()
    per_group: Counter = Counter({g: 0 for g in order})
    picked: set = set()
    picks: List[int] = []
    cursor = {g: 0 for g in order}
    max_per_cluster = 1

    def next_available(g: str) -> Optional[int]:
        """Best unpicked design for *g* whose cluster still has room, or None.

        The cursor may only advance over entries that are already picked -- those
        can never come back. An entry merely blocked by ``max_per_cluster`` has to
        stay reachable, or raising the limit later would never revisit it, so the
        forward scan position is deliberately not persisted.
        """
        q = queues[g]
        i = cursor[g]
        while i < len(q) and q[i] in picked:
            i += 1
        cursor[g] = i
        for j in range(i, len(q)):
            idx = q[j]
            if idx not in picked and cluster_use[cluster_of[idx]] < max_per_cluster:
                return idx
        return None

    # Past this, raising the cluster limit cannot unlock anything new -- without
    # it a group that is short of its quota would spin forever.
    largest_cluster = int(df["cluster_id"].value_counts().max())

    while len(picks) < n:
        avail = {
            g: idx for g in order
            if not (quota and per_group[g] >= quota.get(g, 0))
            and (idx := next_available(g)) is not None
        }
        if not avail:
            if all(cursor[g] >= len(queues[g]) for g in order):
                break  # every design consumed
            if max_per_cluster >= largest_cluster:
                break  # no cluster can give more
            max_per_cluster += 1
            print(f"  [note] every cluster used -- allowing up to "
                  f"{max_per_cluster} constructs per cluster")
            continue

        # Least-served group wins, ties broken by the order given on the command
        # line. Taking a fixed group order instead would hand every short round's
        # leftover to the same method and skew the split.
        g = min(avail, key=lambda g: (per_group[g], order.index(g)))
        idx = avail[g]
        picks.append(idx)
        picked.add(idx)
        per_group[g] += 1
        cluster_use[cluster_of[idx]] += 1

    return picks


# ============================================================
# CONSTRUCT IDS
# ============================================================

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def method_code(label: str) -> str:
    """Derive a short code from a method label: 'Proteina Complexa' -> 'PC'.

    Splits on whitespace/underscores and on camelCase boundaries, so the
    single-token 'LaProteina' still yields 'LP' rather than 'LA'.
    """
    words = [w for w in _CAMEL.sub(" ", label.replace("_", " ")).split() if w]
    if len(words) >= 2:
        return "".join(w[0] for w in words[:3]).upper()
    return (words[0][:2].upper() if words else "XX")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package top binders for a wet-lab handoff "
                    "(cluster-aware round-robin selection).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, nargs="+",
                        help="Run roots to scan for scoring_summary.csv.")
    parser.add_argument("--labels", nargs="+",
                        help="Method label per --input root (same count/order). "
                             "Repeat a label to pool roots into one method.")
    parser.add_argument("--clusters", type=Path, required=True,
                        help="cluster_assignments_with_metadata.csv from the "
                             "Foldseek run. Reused so the handoff matches the "
                             "clustering figure in your presentation.")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output folder (created if missing).")
    parser.add_argument("--target", default="BINDER",
                        help="Target name used in construct IDs (e.g. CD19).")
    parser.add_argument("--top-n", type=int, default=24,
                        help="Number of constructs to select (default: 24).")
    parser.add_argument("--backups", type=int, default=0,
                        help="Extra tier-2 constructs selected after the top-N "
                             "and marked tier=2 (default: 0).")
    parser.add_argument("--balance-by", choices=["source", "none"], default="source",
                        help="'source' (default) alternates methods so the split "
                             "is even; 'none' ranks all methods in one pool.")
    parser.add_argument("--exact-split", action="store_true",
                        help="Force an identical construct count per method "
                             "(--top-n and --backups must divide evenly by the "
                             "number of methods). A method short of distinct "
                             "clusters takes a second construct from a cluster it "
                             "already used rather than under-filling its share. "
                             "Requires --balance-by source.")
    parser.add_argument("--pdb-dir", default="relaxed",
                        help="Leaf directory holding the PDBs (default: relaxed).")
    parser.add_argument("--ss-csv", type=Path,
                        help="secondary_structure_data.csv, to carry helix_frac / "
                             "sheet_frac into the handoff table.")
    parser.add_argument("--meta-threshold", type=float, default=-1.5,
                        help="Overath et al. max-F1 threshold for the product; "
                             "reported per construct, never enforced (default: -1.5).")
    parser.add_argument("--dg-sasa-scale", type=float, default=1.0,
                        help="Multiplier for dG/dSASA before forming the product. "
                             "Keep 1.0 for CSVs from this repo.")
    parser.add_argument("--method-codes", nargs="+",
                        help="Override the auto-derived construct-ID codes, one "
                             "per distinct label in --labels order.")
    parser.add_argument("--local-path-prefix",
                        help="Prefix stripped from resolved PDB paths before the "
                             "cluster join (for running off a mirrored mount).")
    parser.add_argument("--cluster-path-prefix",
                        help="Prefix stripped from original_path in --clusters.")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.input):
        sys.exit(f"[ERROR] --labels takes one label per --input root "
                 f"({len(args.input)} roots, {len(args.labels)} labels given).")
    if args.top_n < 1:
        sys.exit("[ERROR] --top-n must be at least 1.")
    if args.exact_split and args.balance_by != "source":
        sys.exit("[ERROR] --exact-split requires --balance-by source "
                 "(there are no per-method shares to equalise otherwise).")
    if args.exact_split and args.labels:
        # Checkable from the arguments alone; the same guard runs again later for
        # the label-less case, but scanning the run trees first would waste minutes.
        n_methods = len(set(args.labels))
        for name, value in (("--top-n", args.top_n), ("--backups", args.backups)):
            if value % n_methods:
                sys.exit(f"[ERROR] --exact-split needs {name} ({value}) to divide "
                         f"evenly by the {n_methods} methods in --labels.")

    # Resolve construct-ID codes up front. Scanning the run trees takes minutes
    # on NFS, so every check that can be made from the arguments alone is made
    # before that, not after.
    label_order = list(dict.fromkeys(args.labels)) if args.labels else None
    codes: Optional[Dict[str, str]] = None
    if label_order:
        if args.method_codes:
            if len(args.method_codes) != len(label_order):
                sys.exit(f"[ERROR] --method-codes takes one code per distinct label "
                         f"({len(label_order)} labels, {len(args.method_codes)} given).")
            codes = dict(zip(label_order, (c.upper() for c in args.method_codes)))
        else:
            codes = {lbl: method_code(lbl) for lbl in label_order}
            if len(set(codes.values())) != len(codes):
                sys.exit(f"[ERROR] Auto-derived method codes collide: {codes}. "
                         f"Pass --method-codes to disambiguate.")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. every scored design -------------------------------------------
    df_all = build_records(args.input, args.labels, args.dg_sasa_scale)
    print(f"  {len(df_all)} scored designs")

    # ---- 2. second filtering step: PyRosetta filters ----------------------
    accepted = df_all[df_all["pyrosetta_filter_pass"]].copy()
    print(f"  {len(accepted)} passed the PyRosetta filters")
    if accepted.empty:
        sys.exit("[ERROR] No design passed the PyRosetta filters.")

    # ---- 3. locate PDBs ---------------------------------------------------
    print("Resolving PDB files ...")
    accepted["pdb_path"] = resolve_pdbs(accepted, args.pdb_dir)
    n_missing = int(accepted["pdb_path"].isna().sum())
    if n_missing:
        print(f"  [WARN] {n_missing} accepted designs have no PDB on disk; "
              f"they stay in binder_stats.csv but cannot be selected.")

    # ---- 4. clusters ------------------------------------------------------
    print("Attaching cluster IDs ...")
    accepted = attach_clusters(accepted, args.clusters,
                               args.local_path_prefix, args.cluster_path_prefix)

    # ---- 5. secondary structure (optional) --------------------------------
    if args.ss_csv:
        ss = pd.read_csv(args.ss_csv)
        ss_map = {
            _norm(p, args.cluster_path_prefix): (h, s)
            for p, h, s in zip(ss["pdb_path"], ss["helix_frac"], ss["sheet_frac"])
        }
        keys = accepted["pdb_path"].map(
            lambda p: _norm(p, args.local_path_prefix) if p else "")
        accepted["helix_frac"] = keys.map(lambda k: ss_map.get(k, (None, None))[0])
        accepted["sheet_frac"] = keys.map(lambda k: ss_map.get(k, (None, None))[1])
        print(f"  secondary structure: {int(accepted['helix_frac'].notna().sum())}"
              f"/{len(accepted)} matched")
    else:
        accepted["helix_frac"] = pd.NA
        accepted["sheet_frac"] = pd.NA

    # binder_stats.csv is the full evidence set -- written before selection so it
    # is complete even if selection finds fewer than --top-n candidates.
    stats_path = output_dir / "binder_stats.csv"
    accepted.to_csv(stats_path, index=False)
    print(f"\nbinder_stats.csv written: {stats_path} ({len(accepted)} accepted designs)")

    # ---- 6. selection -----------------------------------------------------
    eligible = accepted[
        accepted["pdb_path"].notna()
        & accepted["cluster_id"].notna()
        & accepted["meta_interaction"].notna()
    ]
    n_unrankable = len(accepted) - len(eligible)
    if n_unrankable:
        print(f"  {n_unrankable} accepted designs are not selectable "
              f"(missing PDB, cluster or meta_interaction)")
    if eligible.empty:
        sys.exit("[ERROR] No accepted design is selectable.")

    if args.balance_by == "source":
        groups = list(dict.fromkeys(args.labels)) if args.labels \
            else sorted(eligible["source"].unique())
        groups = [g for g in groups if (eligible["source"] == g).any()]
    else:
        groups = None

    total = args.top_n + args.backups
    quota = None
    if args.exact_split:
        n_g = len(groups)
        for name, value in (("--top-n", args.top_n), ("--backups", args.backups)):
            if value % n_g:
                sys.exit(f"[ERROR] --exact-split needs {name} ({value}) to divide "
                         f"evenly by the {n_g} methods present: {groups}.")
        quota = {g: total // n_g for g in groups}
        print(f"\n--exact-split: {args.top_n // n_g} tier-1 + "
              f"{args.backups // n_g} tier-2 constructs per method")

    print(f"\nSelecting {total} constructs "
          f"({args.top_n} tier 1 + {args.backups} tier 2), "
          f"round robin over {eligible['cluster_id'].nunique()} clusters"
          + (f", alternating {len(groups)} methods" if groups else ", single pool"))
    picks = round_robin_select(eligible, total, groups, quota)
    if len(picks) < total:
        print(f"  [WARN] only {len(picks)} constructs available (asked for {total}).")

    sel = eligible.loc[picks].copy()
    if quota:
        # Tier per method, so both tiers are split evenly too. Tiering by global
        # pick order would give tier 1 to whichever method scores better overall.
        per_method_t1 = args.top_n // len(groups)
        rank_in_method = (
            sel.sort_values("meta_interaction", ascending=True, kind="mergesort")
               .groupby("source", sort=False)
               .cumcount()
        )
        sel["tier"] = (rank_in_method.reindex(sel.index) >= per_method_t1).map(
            {False: 1, True: 2})
    else:
        sel["tier"] = [1 if i < args.top_n else 2 for i in range(len(sel))]

    # ---- 7. construct IDs -------------------------------------------------
    # Without --labels the method names are the root basenames, only knowable
    # once the CSVs have been read.
    if codes is None:
        codes = {lbl: method_code(lbl) for lbl in sorted(sel["source"].unique())}
        if len(set(codes.values())) != len(codes):
            sys.exit(f"[ERROR] Auto-derived method codes collide: {codes}. "
                     f"Pass --labels/--method-codes to disambiguate.")

    # Best-first within each method, so PC-01 really is that method's best.
    sel = sel.sort_values(["tier", "meta_interaction"], ascending=[True, True],
                          kind="mergesort")
    counters: Counter = Counter()
    ids = []
    for src in sel["source"]:
        counters[src] += 1
        ids.append(f"{args.target}-{codes[src]}-{counters[src]:02d}")
    sel["construct_id"] = ids
    sel["method"] = sel["source"]
    sel["passes_meta_threshold"] = sel["meta_interaction"] < args.meta_threshold
    sel["original_path"] = sel["pdb_path"]

    # ---- 8. copy PDBs -----------------------------------------------------
    print("\nCopying PDBs ...")
    n_copied = 0
    for cid, src_pdb in zip(sel["construct_id"], sel["pdb_path"]):
        try:
            shutil.copy2(src_pdb, output_dir / f"{cid}.pdb")
            n_copied += 1
        except OSError as exc:
            print(f"  [ERROR] {cid}: {exc}")
    print(f"  {n_copied}/{len(sel)} copied into {output_dir}")

    # ---- 9. handoff table -------------------------------------------------
    for col in HANDOFF_COLUMNS:
        if col not in sel.columns:
            sel[col] = pd.NA
    top_path = output_dir / "top_binders.csv"
    sel[HANDOFF_COLUMNS].to_csv(top_path, index=False)
    print(f"top_binders.csv written: {top_path} ({len(sel)} constructs)")

    # ---- 10. summary ------------------------------------------------------
    print("\n" + "=" * 60)
    print("HANDOFF SUMMARY")
    print("=" * 60)
    print(f"  accepted (PyRosetta pass) : {len(accepted)}")
    print(f"  selected                  : {len(sel)} "
          f"(tier 1: {int((sel['tier'] == 1).sum())}, "
          f"tier 2: {int((sel['tier'] == 2).sum())})")
    print(f"  distinct clusters         : {sel['cluster_id'].nunique()}")
    print(f"  meta_interaction < {args.meta_threshold} : "
          f"{int(sel['passes_meta_threshold'].sum())}/{len(sel)}")
    print("  by method:")
    for src, n in sel["method"].value_counts().items():
        t1 = int(((sel["method"] == src) & (sel["tier"] == 1)).sum())
        print(f"    {src:24s} {n:3d}  (tier 1: {t1})")
    if quota:
        counts = {g: int((sel["method"] == g).sum()) for g in groups}
        short = {g: quota[g] - c for g, c in counts.items() if c < quota[g]}
        if short:
            # Running out of designs is not the same as an uneven split: both
            # methods can come up short by the same amount and stay balanced.
            even = len(set(counts.values())) == 1
            print(f"  [WARN] --exact-split quota ({quota[groups[0]]} per method) not "
                  f"filled -- these methods ran out of designs: {short}.")
            print(f"         Final split is {'still even' if even else 'UNEVEN'}: "
                  f"{counts}")
    if sel["sequence"].eq("").any() or sel["sequence"].isna().any():
        n_noseq = int(sel["sequence"].eq("").sum() + sel["sequence"].isna().sum())
        print(f"  [WARN] {n_noseq} constructs have no sequence column -- the wet "
              f"lab cannot order those without it.")


if __name__ == "__main__":
    main()
