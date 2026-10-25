#!/usr/bin/env python3
"""
Analyze successful binders from Proteina/BindProteina runs.

Features:
- Walk one or more input directories (or an explicit CSV) and collect all
  scoring CSV files. Designs from every root compete in a single global
  ranking, so ``--top-n`` is drawn across folders -- e.g. the best binders out
  of a seed-search tree and several reseed trees combined. Tag roots with
  ``--labels`` to see which folder each winner came from (``source`` column).
- Extract ipTM and pLDDT (or similarly named) columns.
- Print summary statistics (mean, median, std, min, max, percentiles).
- Select the top N binders (default 20) by the chosen ranking metric.
- Copy (or move) the corresponding PDB files into an output folder.
- Write ``binder_stats.csv`` (all samples) and ``top_binders.csv`` (top N only).

By default only designs that passed the PyRosetta filters
(``pyrosetta_filter_pass``) are eligible for top-N collection; pass
``--include-failed`` to consider every scored design.  ``binder_stats.csv``
always contains every parsed row, regardless of filtering.

Ranking modes (``--sort-by``)
----------------------------
``iptm`` (default)
    Legacy behaviour: sort by ipTM, then pLDDT (both descending).

``ipsae_min``
    Sort by ipSAE_min descending. Overath et al. (bioRxiv 2025.08.14.670059)
    found AF3 ipSAE_min to be the single best predictor of in vitro binding
    across 3,766 experimentally characterised binders (1.4-fold higher average
    precision than ipAE).

``meta_interaction``
    Sort by the meta-analysis interaction term

        ipSAE_min  x  interface_dG/dSASA

    ascending (more negative = more likely to bind). The same paper reports a
    max-F1 threshold of **-1.5** for this product, applied here as an optional
    hard cut via ``--meta-threshold`` / ``--apply-meta-threshold``.

    Unit note: the interaction term is a *raw* product of the two features
    (Interaction(fi, fj) = fi x fj); the z-score standardisation described in
    that paper applies only to its logistic-regression / XGBoost classifiers,
    not to this threshold. The paper's feature is ``rosetta_interface_dG_SASA_
    ratio``, which is the same InterfaceAnalyzer quantity this repo reports in
    the ``dG/dSASA`` column -- but note ``functions/pyrosetta_utils.py`` stores
    it as ``dG_dSASA_ratio * 100``. That x100 scaling is what puts the product
    on the scale where -1.5 is meaningful; if you feed in unscaled ratios the
    product lands near -0.01 and can never cross the threshold. Use
    ``--dg-sasa-scale`` if your CSVs were produced without the x100 factor.
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_column(df: pd.DataFrame, keywords: List[str]) -> str:
    """Return the first column name that contains any of the *keywords* (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for kw in keywords:
        kw_l = kw.lower()
        for lc, orig in lower_cols.items():
            if kw_l in lc:
                return orig
    raise KeyError(f"No column matching {keywords!r} found in {list(df.columns)}")


def _detect_optional_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Like :func:`_detect_column` but returns ``None`` instead of raising.

    Prefers ``Average_`` variants (the scorer emits ``Average_<metric>`` plus
    per-model ``1_<metric>``..``5_<metric>``); an exact case-insensitive match
    wins over a substring match so that e.g. ``ipSAE_min`` is not shadowed by
    ``ipSAE_min_something``.
    """
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    for kw in keywords:
        kw_l = kw.lower()
        # 1) Average_ variant, exact
        if f"average_{kw_l}" in lower:
            return lower[f"average_{kw_l}"]
        # 2) exact match
        if kw_l in lower:
            return lower[kw_l]
    # 3) substring fallback, Average_ preferred
    for kw in keywords:
        kw_l = kw.lower()
        subs = [c for c in cols if kw_l in c.lower()]
        if subs:
            avg = [c for c in subs if c.lower().startswith("average_")]
            return (avg or subs)[0]
    return None


def _load_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV with utf-8 fallback to latin1.

    Returns None for a file with no parseable header (an aborted run leaves a
    zero-byte CSV behind), so one bad folder cannot abort a whole sweep.
    """
    try:
        if csv_path.stat().st_size == 0:
            return None
    except OSError:
        return None
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    except UnicodeDecodeError:
        try:
            return pd.read_csv(csv_path, encoding="latin1")
        except pd.errors.EmptyDataError:
            return None


def _gather_csv_files(
    roots: List[Path],
    labels: Optional[List[str]] = None,
) -> List[Tuple[Path, Path, str]]:
    """Collect scoring summary CSVs under each root (recursively).

    Returns ``(csv_path, root, label)`` triples. Deduplication is global across
    roots by resolved path, so nested or repeated roots cannot double-count a
    CSV; the first root that claims a file keeps it.
    """
    patterns = ["*scoring_summary.csv", "*_prescoring.csv"]
    if labels is None:
        labels = [r.name for r in roots]

    out: List[Tuple[Path, Path, str]] = []
    seen = set()
    for root, label in zip(roots, labels):
        for pat in patterns:
            for f in root.rglob(pat):
                # Skip macOS resource forks and duplicates
                if f.name.startswith("._"):
                    continue
                resolved = f.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    out.append((f, root, label))
    return sorted(out, key=lambda t: (t[2], str(t[0])))


def _is_truthy(val) -> bool:
    """Interpret a CSV cell as a boolean pass/fail flag.

    Handles native bools, the strings written by ``DataFrame.to_csv`` ("True"/
    "False"), numeric flags, and PASS/FAIL strings. Missing/blank values are
    treated as *not passing*.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    return s in {"true", "1", "1.0", "yes", "pass", "passed", "y", "t"}


def _safe_float(val) -> float:
    """Coerce a CSV cell to float, returning NaN for blank / non-numeric values.

    Keeps a single malformed cell from aborting an entire sweep.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _get_sample_name(row: pd.Series) -> Optional[str]:
    """Extract the per-sample name from a CSV row.

    Checks common column names: design_name, design, binder, name, id.
    Returns None if no suitable column is found.
    """
    for col in ["design_name", "design", "binder", "name", "id"]:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                return val
    return None


def _find_pdb(sample_name: str, csv_dir: Path) -> Optional[Path]:
    """Find the PDB for *sample_name* inside *csv_dir* (recursively).

    Tries exact name matches first (``<sample>_relaxed.pdb``, ``<sample>.pdb``),
    then falls back to substring matching.
    """
    # Exact name patterns (most specific first)
    for suffix in [f"{sample_name}_relaxed.pdb", f"{sample_name}.pdb", f"{sample_name}.cif"]:
        candidates = list(csv_dir.rglob(suffix))
        if candidates:
            return candidates[0]
    # Substring fallback: any PDB whose stem starts with sample_name
    for p in csv_dir.rglob("*.pdb"):
        if p.stem.startswith(sample_name):
            return p
    return None


# ---------------------------------------------------------------------------
# Core analysis workflow
# ---------------------------------------------------------------------------

def analyze(
    input_roots: List[Path],
    output_dir: Path,
    top_n: int = 20,
    copy: bool = True,
    labels: Optional[List[str]] = None,
    explicit_csv: Optional[Path] = None,
    include_failed: bool = False,
    sort_by: str = "iptm",
    meta_threshold: float = -1.5,
    apply_meta_threshold: bool = False,
    dg_sasa_scale: float = 1.0,
) -> None:
    """Execute the full analysis pipeline.

    Parameters
    ----------
    input_roots
        One or more root directories to scan. Designs from all roots compete in
        a single global ranking, so the top-N is drawn across folders (e.g. a
        seed-search tree and several reseed trees together).
    labels
        Optional display label per root, same count and order. Repeat a label to
        pool several roots under one name. Defaults to each root's basename.
        Recorded in the ``source`` column of both output CSVs.
    include_failed
        If True, consider designs that did NOT pass the PyRosetta filters.
        By default only ``pyrosetta_filter_pass`` designs are eligible.
    sort_by
        ``"iptm"`` (legacy), ``"ipsae_min"`` or ``"meta_interaction"``.
    meta_threshold
        Max-F1 cut for the ``ipSAE_min x dG/dSASA`` product (Overath et al.:
        -1.5). Only used when ``sort_by="meta_interaction"``.
    apply_meta_threshold
        If True, drop designs failing ``product < meta_threshold`` before
        top-N selection. If False (default) the product is used for ranking
        only and the threshold is reported but not enforced -- consistent with
        the paper's own finding that optimal cutoffs are target-dependent.
    dg_sasa_scale
        Multiplier applied to the ``dG/dSASA`` column before forming the
        product. Leave at 1.0 for CSVs from this repo (``pyrosetta_utils.py``
        already applies x100); set to 100 for unscaled ratios.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Gather CSVs -----
    # Each entry is (csv_path, root, label); the root lets us build a globally
    # unique per-design id, the label groups roots for reporting.
    csv_files: List[Tuple[Path, Path, str]]
    if explicit_csv:
        csv_files = [(explicit_csv, explicit_csv.parent, explicit_csv.parent.name)]
    else:
        csv_files = _gather_csv_files(input_roots, labels)

    if not csv_files:
        print("[ERROR] No scoring CSV files found.")
        sys.exit(1)

    print(f"Found {len(csv_files)} scoring CSV files.")
    if len(input_roots) > 1 and not explicit_csv:
        per_root = Counter(lbl for _, _, lbl in csv_files)
        for lbl in dict.fromkeys(labels or [r.name for r in input_roots]):
            print(f"    {lbl}: {per_root.get(lbl, 0)} CSVs")

    # ----- Parse all CSVs into a flat record list -----
    records: list[dict] = []

    for csv_path, csv_root, csv_label in tqdm(csv_files, desc="Loading CSVs"):
        df = _load_csv(csv_path)
        if df is None or df.empty:
            print(f"[WARN] Skipping {csv_path}: empty CSV (no data rows)")
            continue
        try:
            iptm_col = _detect_column(df, ["iptm", "ipTM", "ip_tm", "i_pTM", "i_ptm"])
            plddt_col = _detect_column(df, ["plddt", "pLDDT", "plddt_score"])
        except KeyError as exc:
            print(f"[WARN] Skipping {csv_path}: {exc}")
            continue

        # Detect the PyRosetta pass/fail column (per-design boolean).
        pyrosetta_col = _detect_optional_column(df, ["pyrosetta_filter_pass"])

        # Try to detect an aggregate successful-samples count column (optional).
        # Deliberately narrow: a broader keyword list resolves to
        # pyrosetta_filter_pass, which is a per-design bool, not a count.
        successful_col = _detect_optional_column(df, ["num_successful", "num_relaxed"])

        # Optional metrics for the ipSAE / meta-analysis ranking modes.
        # Column names follow scorer.py's mapping (ipsae_min -> "ipSAE_min")
        # and generate_dataframe_labels() (dG/dSASA), incl. Average_ variants.
        ipsae_col = _detect_optional_column(df, ["ipSAE_min", "ipsae_min"])
        dgsasa_col = _detect_optional_column(df, ["dG/dSASA", "dG_SASA_ratio", "dg_dsasa_ratio"])

        for _, row in df.iterrows():
            iptm = float(row[iptm_col]) if pd.notna(row[iptm_col]) else None
            plddt = float(row[plddt_col]) if pd.notna(row[plddt_col]) else None
            if iptm is None or plddt is None:
                continue

            sample_name = _get_sample_name(row)
            if sample_name is None:
                # Last resort: use CSV parent dir name + row index
                sample_name = csv_path.parent.name

            # Build a globally unique ID from the CSV's path relative to its
            # root, prefixed by the root's own name.
            #
            # The last two path components are NOT enough once several roots are
            # scanned: reseed trees bottom out in <root>/shardNN/runs/<TGT>/genN/
            # _scored, so cd19_v3 and cd19_v4 would both yield "genN/_scored/
            # CD19_gN_sSS_IC_sample_XXX" for the same strain/cut/index and the
            # second one would be silently dropped as a duplicate.
            csv_dir = csv_path.parent
            try:
                rel_dir = csv_dir.relative_to(csv_root)
            except ValueError:
                rel_dir = Path(csv_dir.name)
            rel_str = "" if str(rel_dir) == "." else f"{rel_dir}/"
            uid = f"{csv_root.name}/{rel_str}{sample_name}"

            # PyRosetta filter outcome. If the column is absent we cannot verify
            # a design passed, so treat it as not-passing (excluded by default).
            pyrosetta_pass = (
                _is_truthy(row[pyrosetta_col]) if pyrosetta_col is not None else False
            )

            # Start from every column in the scoring CSV so all metrics are
            # carried through, then layer on the derived / normalized fields.
            rec = row.to_dict()
            rec.update({
                "uid": uid,
                "sample_name": sample_name,
                "iptm": iptm,
                "plddt": plddt,
                "pyrosetta_filter_pass": pyrosetta_pass,
                "source": csv_label,
                "source_root": str(csv_root),
                "source_csv": str(csv_path),
                "csv_dir": str(csv_dir),
            })

            # Add successful samples count if available
            if successful_col and pd.notna(row[successful_col]):
                rec["successful_samples"] = int(row[successful_col])

            # Optional ranking metrics (NaN when the column is absent, so a
            # mixed set of CSVs does not silently drop designs).
            rec["ipsae_min"] = (
                _safe_float(row.get(ipsae_col)) if ipsae_col else float("nan")
            )
            dg_raw = _safe_float(row.get(dgsasa_col)) if dgsasa_col else float("nan")
            rec["dg_sasa_ratio"] = dg_raw * dg_sasa_scale

            records.append(rec)

    if not records:
        print("[ERROR] No valid records extracted from CSVs.")
        sys.exit(1)

    df_all = pd.DataFrame.from_records(records)

    # meta_interaction: Overath et al. product of ipSAE_min and dG/dSASA.
    df_all["meta_interaction"] = df_all["ipsae_min"] * df_all["dg_sasa_ratio"]

    # ----- Summary statistics -----
    def _print_stats(label: str, series: pd.Series, fmt: str = ".3f") -> None:
        print(f"--- {label} ---")
        print(f"  mean   : {series.mean():{fmt}}")
        print(f"  median : {series.median():{fmt}}")
        print(f"  std    : {series.std():{fmt}}")
        print(f"  min    : {series.min():{fmt}}")
        print(f"  max    : {series.max():{fmt}}")
        print(f"  25th   : {series.quantile(0.25):{fmt}}")
        print(f"  75th   : {series.quantile(0.75):{fmt}}")

    n_pyrosetta_pass = int(df_all["pyrosetta_filter_pass"].sum())
    print(f"\n=== Binder Success Summary ===")
    print(f"Total samples             : {len(df_all)}")
    print(f"PyRosetta-passing samples : {n_pyrosetta_pass}")
    if df_all["source"].nunique() > 1:
        print("Per source (total / passing):")
        for src, grp in df_all.groupby("source", sort=False):
            print(f"  {src}: {len(grp)} / {int(grp['pyrosetta_filter_pass'].sum())}")
    _print_stats("ipTM", df_all["iptm"], ".3f")
    _print_stats("pLDDT", df_all["plddt"], ".2f")

    # Print total successful samples count if available
    if "successful_samples" in df_all.columns:
        total_successful = df_all["successful_samples"].sum()
        print(f"Successful samples (PyRosetta): {int(total_successful)}")

    if df_all["ipsae_min"].notna().any():
        _print_stats("ipSAE_min", df_all["ipsae_min"].dropna(), ".3f")
    if df_all["dg_sasa_ratio"].notna().any():
        _print_stats("dG/dSASA", df_all["dg_sasa_ratio"].dropna(), ".3f")

    # ----- Restrict candidate pool to PyRosetta-passing designs -----
    # binder_stats.csv keeps every row; only the top-N pool is filtered.
    if include_failed:
        df_rank = df_all.copy()
    else:
        df_rank = df_all[df_all["pyrosetta_filter_pass"]].copy()
        if df_rank.empty:
            print(
                "[ERROR] No PyRosetta-passing designs found (checked column "
                "'pyrosetta_filter_pass'). If your CSVs do not record this "
                "column, re-run with --include-failed to keep the old behaviour."
            )
            sys.exit(1)

    # ----- Ranking -----
    n_ipsae = int(df_rank["ipsae_min"].notna().sum())
    n_dgsasa = int(df_rank["dg_sasa_ratio"].notna().sum())
    n_meta = int(df_rank["meta_interaction"].notna().sum())

    if sort_by == "iptm":
        df_sorted = df_rank.sort_values(["iptm", "plddt"], ascending=False)

    elif sort_by == "ipsae_min":
        if n_ipsae == 0:
            print("[ERROR] --sort-by ipsae_min requested but no ipSAE_min column was "
                  "found in any CSV.\n"
                  "        ipSAE is computed in the Protenix path "
                  "(pipeline/scoring/folding_protenix.py); re-score with Protenix "
                  "folding enabled, or use --sort-by iptm.")
            sys.exit(1)
        print(f"\nRanking by ipSAE_min (Overath et al. best single predictor) "
              f"-- available for {n_ipsae}/{len(df_rank)} candidate designs.")
        # NaNs sort last regardless of direction
        df_sorted = df_rank.sort_values("ipsae_min", ascending=False, na_position="last")

    elif sort_by == "meta_interaction":
        if n_meta == 0:
            missing = []
            if n_ipsae == 0:
                missing.append("ipSAE_min (needs Protenix folding: "
                               "pipeline/scoring/folding_protenix.py)")
            if n_dgsasa == 0:
                missing.append("dG/dSASA (needs pyRosetta interface scoring)")
            print("[ERROR] --sort-by meta_interaction requires both ipSAE_min and "
                  "dG/dSASA.\n        Missing: " + "; ".join(missing))
            sys.exit(1)

        print(f"\nRanking by meta-analysis interaction term "
              f"ipSAE_min x dG/dSASA (ascending; more negative = better) -- "
              f"available for {n_meta}/{len(df_rank)} candidate designs "
              f"[ipSAE_min: {n_ipsae}, dG/dSASA: {n_dgsasa}].")

        n_meta_pass = int((df_rank["meta_interaction"] < meta_threshold).sum())
        print(f"  Designs passing product < {meta_threshold}: {n_meta_pass}/{n_meta}"
              f"  ({100.0 * n_meta_pass / n_meta:.1f}% of scored)")

        if apply_meta_threshold:
            before = len(df_rank)
            df_rank = df_rank[df_rank["meta_interaction"] < meta_threshold].copy()
            print(f"  --apply-meta-threshold: kept {len(df_rank)}/{before} designs.")
            if df_rank.empty:
                print("[ERROR] No designs passed the threshold; nothing to collect.\n"
                      "        Re-run without --apply-meta-threshold to rank only, "
                      "or loosen --meta-threshold.")
                sys.exit(1)
        else:
            print("  (ranking only -- pass --apply-meta-threshold to enforce it as a "
                  "hard cut)")

        # More negative is better -> ascending. NaNs last.
        df_sorted = df_rank.sort_values("meta_interaction", ascending=True,
                                        na_position="last")

    else:  # argparse choices should prevent this
        print(f"[ERROR] Unknown --sort-by value: {sort_by!r}")
        sys.exit(1)

    df_sorted = df_sorted.reset_index(drop=True)

    copied = []  # list of dicts for top binder rows
    seen_uids: set[str] = set()

    for _, row in df_sorted.iterrows():
        if len(copied) >= top_n:
            break
        uid = row["uid"]
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        csv_dir = Path(row["csv_dir"])
        pdb_path = _find_pdb(row["sample_name"], csv_dir)

        if pdb_path is None:
            print(f"[WARN] PDB for {uid} not found in {csv_dir} – skipping.")
            continue

        # Copy/move to output
        dest = output_dir / pdb_path.name
        # Handle name collisions by prefixing with the source root and parent
        # dirs; identical relative layouts across roots (e.g. several reseed
        # versions) make the parent dirs alone insufficient.
        if dest.exists():
            prefix = f"{Path(row['source_root']).name}__{csv_dir.parent.name}__{csv_dir.name}__"
            dest = output_dir / (prefix + pdb_path.name)
            # Last resort: never silently overwrite a previously collected binder.
            n = 2
            while dest.exists():
                dest = output_dir / f"{prefix}{pdb_path.stem}_{n}{pdb_path.suffix}"
                n += 1
        try:
            if copy:
                shutil.copy2(pdb_path, dest)
            else:
                shutil.move(str(pdb_path), str(dest))
        except Exception as exc:
            print(f"[ERROR] Failed to {'copy' if copy else 'move'} {pdb_path}: {exc}")
            continue

        # Carry through every scoring metric for this design, then append the
        # rank / resolved PDB paths.
        rec = row.to_dict()
        rec["rank"] = len(copied) + 1
        rec["sorted_by"] = sort_by
        rec["pdb_source"] = str(pdb_path)
        rec["pdb_dest"] = str(dest)
        copied.append(rec)

    # ----- Write CSVs -----
    # Full stats
    stats_csv_path = output_dir / "binder_stats.csv"
    df_all.to_csv(stats_csv_path, index=False)

    # Top binders
    df_top = pd.DataFrame.from_records(copied)
    top_csv_path = output_dir / "top_binders.csv"
    df_top.to_csv(top_csv_path, index=False)

    print(f"\nFull statistics written to : {stats_csv_path}  ({len(df_all)} rows)")
    print(f"Top binders CSV written to: {top_csv_path}  ({len(df_top)} rows)")
    print(f"Top {len(df_top)} PDB files {'copied' if copy else 'moved'} to: {output_dir}")

    # Which folders actually won slots -- the point of scanning several roots.
    if not df_top.empty and "source" in df_top.columns and df_all["source"].nunique() > 1:
        print("\nTop-N composition by source:")
        for src, n in df_top["source"].value_counts().items():
            print(f"  {src}: {n}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze successful binders, print stats, and collect top samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True, nargs="+",
                        help="One or more root directories containing results. "
                             "Designs from all roots compete in a single global "
                             "ranking, so the top-N is drawn across folders.")
    parser.add_argument("--labels", nargs="+",
                        help="Optional display label per --input root (same count "
                             "and order). Repeat a label to pool several roots "
                             "under one name, e.g. --labels seed_search reseed "
                             "reseed reseed. Defaults to each root's basename; "
                             "recorded in the 'source' column.")
    parser.add_argument("--output", "-o", type=Path, default="top_binders",
                        help="Folder where top-binder PDBs and CSVs will be stored.")
    parser.add_argument("--csv", type=Path,
                        help="Explicit CSV file to analyse (overrides directory scan).")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top binders to collect.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy PDB files instead of moving them.")
    parser.add_argument("--include-failed", action="store_true",
                        help="Include designs that did NOT pass the PyRosetta "
                             "filters (default: only passing designs are collected).")
    parser.add_argument("--sort-by", choices=["iptm", "ipsae_min", "meta_interaction"],
                        default="iptm",
                        help="Ranking metric. 'iptm' = ipTM then pLDDT (legacy). "
                             "'ipsae_min' = Overath et al. best single predictor. "
                             "'meta_interaction' = ipSAE_min x dG/dSASA, ascending "
                             "(more negative = better).")
    parser.add_argument("--meta-threshold", type=float, default=-1.5,
                        help="Max-F1 threshold for the ipSAE_min x dG/dSASA product "
                             "(Overath et al. 2025). Reported always; enforced only "
                             "with --apply-meta-threshold.")
    parser.add_argument("--apply-meta-threshold", action="store_true",
                        help="Enforce product < --meta-threshold as a hard filter "
                             "before top-N selection, instead of ranking only.")
    parser.add_argument("--dg-sasa-scale", type=float, default=1.0,
                        help="Multiplier for the dG/dSASA column before forming the "
                             "product. Keep 1.0 for CSVs from this repo (already "
                             "x100 in pyrosetta_utils.py); use 100 for raw ratios.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.labels and len(args.labels) != len(args.input):
        print(f"[ERROR] --labels takes exactly one label per --input root "
              f"({len(args.input)} roots, {len(args.labels)} labels given).")
        sys.exit(1)
    analyze(
        input_roots=args.input,
        output_dir=args.output,
        top_n=args.top_n,
        copy=args.copy,
        labels=args.labels,
        explicit_csv=args.csv,
        include_failed=args.include_failed,
        sort_by=args.sort_by,
        meta_threshold=args.meta_threshold,
        apply_meta_threshold=args.apply_meta_threshold,
        dg_sasa_scale=args.dg_sasa_scale,
    )
