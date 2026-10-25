#!/usr/bin/env python3
"""
Re-apply the current filters to an existing scoring run folder, in place.

The folding stage is never re-run: the per-design metrics are read straight from
the ``*_prescoring_results.csv`` that is already in the folder.  What this does:

  1. Re-evaluates the folding filters against the current filters JSON.
  2. Re-evaluates the PyRosetta filters for designs that were already scored,
     reusing the interface metrics stored in ``scoring_summary.csv`` (so no
     relax / interface re-computation is needed).
  3. Runs PyRosetta only for designs that newly pass the folding filter and have
     never been scored.
  4. Drops designs from ``scoring_summary.csv`` that no longer pass the folding
     filter, so the folder ends up exactly as a normal run with these filters
     would have left it.

``--dry_run`` reports the diff and writes nothing.

Usage:
    python pipeline/refilter_run.py --run_dir /path/to/initial_20260723_0505 --dry_run
    python pipeline/refilter_run.py --run_dir /path/to/initial_20260723_0505
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Ensure imports work when running as a script path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.scoring.ipsae import BUNDLE_KEYS as IPSAE_BUNDLE_KEYS
from pipeline.scoring.scorer import (
    _hotspot_contact_metrics,
    _pick_filter_spec,
    _resolve_hotspot_positions,
    _resolve_min_hotspot_contacts,
)

# Must stay identical to the metric_to_suffix maps in scorer.py (score_complexes
# and score_from_protenix_json) so the rewritten CSVs match a normal run.
FOLDING_METRIC_TO_SUFFIX = {
    "pLDDT": "pLDDT",
    "binder_pLDDT": "Binder_pLDDT",
    "pTM": "pTM",
    "binder_pTM": "Binder_pTM",
    "i_pTM": "i_pTM",
    "binder_iptm": "Binder_ipTM",
    "pAE": "pAE",
    "i_pAE": "i_pAE",
    "sc_rmsd": "Binder_RMSD",
    "ipsae": "ipSAE",
    "ipsae_min": "ipSAE_min",
}

PYROSETTA_METRIC_TO_SUFFIX = {
    "binder_score": "Binder_Energy_Score",
    "surface_hydrophobicity": "Surface_Hydrophobicity",
    "interface_sc": "ShapeComplementarity",
    "interface_packstat": "PackStat",
    "interface_dG": "dG",
    "interface_dSASA": "dSASA",
    "interface_dG_SASA_ratio": "dG/dSASA",
    "interface_hydrophobicity": "Interface_Hydrophobicity",
    "interface_nres": "n_InterfaceResidues",
    "interface_interface_hbonds": "n_InterfaceHbonds",
    "interface_hbond_percentage": "InterfaceHbondsPercentage",
    "interface_delta_unsat_hbonds": "n_InterfaceUnsatHbonds",
    "interface_delta_unsat_hbonds_percentage": "InterfaceUnsatHbondsPercentage",
}

# Leading columns of scoring_summary.csv, in the order score_complexes writes them.
SUMMARY_HEAD_KEYS = (
    "design_name", "status", "folding_model",
    "pLDDT", "binder_pLDDT", "pTM", "i_pTM", "pAE", "i_pAE",
    "sc_rmsd", "binder_length",
) + tuple(IPSAE_BUNDLE_KEYS)


# ──────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(value):
    """NaN / empty-string -> None, so filter checks behave like a fresh run."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _is_empty_csv(csv_path: Path) -> bool:
    """True when the file has no parseable header (0 bytes / whitespace only)."""
    try:
        if csv_path.stat().st_size == 0:
            return True
    except OSError:
        return True
    try:
        with open(csv_path) as fh:
            for line in fh:
                if line.strip():
                    return False
    except OSError:
        return True
    return True


def _load_rows(csv_path: Path) -> List[Dict]:
    """Load a CSV to row dicts; an empty / header-less file yields no rows."""
    if _is_empty_csv(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return []
    return [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict("records")]


def _as_number(value):
    """Coerce a CSV value to float for threshold comparison; None when not numeric."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _find_prescoring_csv(run_dir: Path, folding_model: Optional[str]) -> Tuple[Path, str]:
    if folding_model:
        candidate = run_dir / f"{folding_model}_prescoring_results.csv"
        if not candidate.exists():
            raise SystemExit(f"No pre-scoring CSV at {candidate}")
        if _is_empty_csv(candidate):
            raise SystemExit(
                f"Pre-scoring CSV is empty: {candidate}\n"
                "The folding stage never wrote results for this folder, and this tool "
                "does not refold. Re-run the normal scoring pipeline on it instead."
            )
        return candidate, folding_model
    found = []
    empty = []
    for model in ("protenix", "af2"):
        candidate = run_dir / f"{model}_prescoring_results.csv"
        if candidate.exists():
            if _is_empty_csv(candidate):
                empty.append(candidate)
            else:
                found.append((candidate, model))
    if not found and empty:
        listed = ", ".join(p.name for p in empty)
        raise SystemExit(
            f"Pre-scoring CSV present but empty in {run_dir}: {listed}\n"
            "The folding stage never wrote results for this folder, and this tool "
            "does not refold. Re-run the normal scoring pipeline on it instead."
        )
    if not found:
        raise SystemExit(
            f"No *_prescoring_results.csv in {run_dir}. "
            "This tool needs an existing folding stage; it never refolds."
        )
    if len(found) > 1:
        names = ", ".join(m for _, m in found)
        raise SystemExit(f"Multiple pre-scoring CSVs ({names}); pass --folding_model.")
    return found[0]


def _index_folded_pdbs(predictions_dir: Path) -> Dict[str, str]:
    """Map design_name -> folded PDB, same naming rules score_complexes uses."""
    mapping: Dict[str, str] = {}
    if not predictions_dir.is_dir():
        return mapping
    for pdb in predictions_dir.rglob("*.pdb"):
        if pdb.name.endswith("_protenix.pdb"):
            mapping[pdb.name[: -len("_protenix.pdb")]] = str(pdb)
        elif "_model" in pdb.name:
            mapping[pdb.name.rsplit("_model", 1)[0]] = str(pdb)
    return mapping


def _resolve_hotspots_from_run(run_dir: Path) -> Optional[str]:
    """Prefer the 1-based positions the designer wrote into settings_used.json."""
    settings_path = run_dir / "settings_used.json"
    if not settings_path.exists():
        return None
    try:
        with open(settings_path) as fh:
            settings = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return settings.get("target_hotspot_positions") or settings.get("target_hotspot_residues") or None


# ──────────────────────────────────────────────────────────────────────────────
# Filter evaluation (mirrors score_complexes exactly)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_folding_filters(
    rows: List[Dict],
    filters: Dict,
    model_num: int,
    force_rosetta: bool,
) -> List[Dict]:
    """Return fresh rows with recomputed filter_* / folding_filter_pass columns."""
    out: List[Dict] = []
    for row in rows:
        base = {
            k: v for k, v in row.items()
            if not k.startswith("filter_") and k != "folding_filter_pass"
        }
        if row.get("status") != "predicted":
            base["folding_filter_pass"] = False
            out.append(base)
            continue
        if force_rosetta:
            base["folding_filter_pass"] = True
            out.append(base)
            continue

        passed = True
        for metric, suffix in FOLDING_METRIC_TO_SUFFIX.items():
            key, spec = _pick_filter_spec(filters, suffix, model_num)
            if not key or not spec:
                continue
            raw = base.get(metric, base.get(metric.lower()))
            value = _as_number(raw)
            thr = _as_number(spec.get("threshold"))
            if value is None or thr is None:
                base[f"filter_{metric}"] = "N/A"
                continue
            higher = spec.get("higher", True)
            if (higher and value < thr) or ((not higher) and value > thr):
                base[f"filter_{metric}"] = "FAIL"
                passed = False
            else:
                base[f"filter_{metric}"] = "PASS"
        base["folding_filter_pass"] = passed
        out.append(base)
    return out


def _missing_metric_columns(
    rows: List[Dict],
    filters: Dict,
    model_num: int,
) -> List[Tuple[str, str, object]]:
    """
    Filters that have a live threshold but no data in this CSV.

    A run made before a metric existed cannot be filtered on it: the value is
    None, so the check records "N/A" and lets the design through.  Such a folder
    can never be made equivalent to a fresh run without refolding, so the
    mismatch is reported rather than left silent.
    """
    missing = []
    for metric, suffix in FOLDING_METRIC_TO_SUFFIX.items():
        key, spec = _pick_filter_spec(filters, suffix, model_num)
        if not key or not spec or spec.get("threshold") is None:
            continue
        if not any(_as_number(r.get(metric, r.get(metric.lower()))) is not None for r in rows):
            missing.append((metric, key, spec.get("threshold")))
    return missing


def _apply_pyrosetta_filters(
    metrics: Dict,
    filters: Dict,
    model_num: int,
    hotspot_ids: Set[int],
    min_hotspot_contacts: int,
) -> Tuple[Dict, bool]:
    """Evaluate the PyRosetta filters (plus hotspot contacts) over ``metrics``."""
    filter_results: Dict[str, str] = {}
    passed = True
    for metric, suffix in PYROSETTA_METRIC_TO_SUFFIX.items():
        key, spec = _pick_filter_spec(filters, suffix, model_num)
        if not key or not spec:
            continue
        value = _as_number(metrics.get(metric))
        thr = _as_number(spec.get("threshold"))
        if value is None or thr is None:
            filter_results[f"pyrosetta_filter_{metric}"] = "N/A"
            continue
        higher = spec.get("higher", True)
        if (higher and value < thr) or ((not higher) and value > thr):
            filter_results[f"pyrosetta_filter_{metric}"] = "FAIL"
            passed = False
        else:
            filter_results[f"pyrosetta_filter_{metric}"] = "PASS"

    if hotspot_ids:
        hs_metrics, hs_passed = _hotspot_contact_metrics(
            metrics, hotspot_ids, min_hotspot_contacts
        )
        metrics.update(hs_metrics)
        filter_results["pyrosetta_filter_hotspot_contacts"] = "PASS" if hs_passed else "FAIL"
        if not hs_passed:
            passed = False

    return filter_results, passed


def _build_summary_row(
    design_name: str,
    folding_row: Dict,
    metrics: Dict,
    filter_results: Dict,
    passed: bool,
) -> Dict:
    """Assemble one scoring_summary.csv row in score_complexes' column order."""
    row: Dict = {"design_name": design_name, "status": "scored"}
    row["folding_model"] = folding_row.get("folding_model")
    for key in SUMMARY_HEAD_KEYS[3:]:
        row[key] = folding_row.get(key)

    skip = set(row) | {"sequence", "pyrosetta_filter_pass", "reason"}
    for key, value in metrics.items():
        if key in skip or key.startswith("pyrosetta_filter_"):
            continue
        row[key] = value

    row.update(filter_results)
    row["pyrosetta_filter_pass"] = passed
    row["sequence"] = folding_row.get("sequence")
    return row


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-apply filters to an existing run folder without refolding."
    )
    parser.add_argument("--run_dir", "-d", required=True,
                        help="Run folder containing *_prescoring_results.csv")
    parser.add_argument("--filters", "-f", default=None,
                        help="Filters JSON (default: settings_filters/laProteina_filters.json)")
    parser.add_argument("--folding_model", default=None, choices=["protenix", "af2"],
                        help="Which pre-scoring CSV to read (default: auto-detect)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change and write nothing.")
    parser.add_argument("--af2_model", type=int, default=None,
                        help="Model number for per-model filter keys (default: 1)")
    parser.add_argument("--force_rosetta", action="store_true",
                        help="Treat every predicted design as folding-passing.")
    parser.add_argument("--hotspot_residues", default=None,
                        help="Override hotspots (default: from settings_used.json in the run dir)")
    parser.add_argument("--target_pdb", default=None,
                        help="Reference PDB with author numbering for --hotspot_residues")
    parser.add_argument("--target_chain", default="A",
                        help="Chain in --target_pdb to map hotspots through (default: A)")
    parser.add_argument("--no_hotspot", action="store_true",
                        help="Disable the hotspot-contact filter entirely.")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Re-attempt PyRosetta for designs stored as rosetta_failed.")
    parser.add_argument("--no_backup", action="store_true",
                        help="Do not keep timestamped .bak copies of the rewritten CSVs.")
    parser.add_argument("--dalphaball", default=None, help="Path to DAlphaBall.gcc")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    repo_root = Path(__file__).resolve().parent.parent

    filters_path = args.filters or str(repo_root / "settings_filters" / "laProteina_filters.json")
    if not os.path.exists(filters_path):
        raise SystemExit(f"Filters JSON not found: {filters_path}")
    with open(filters_path) as fh:
        filters = json.load(fh)

    model_num = args.af2_model or 1
    folding_csv, folding_model = _find_prescoring_csv(run_dir, args.folding_model)
    summary_csv = run_dir / "scoring_summary.csv"

    print(f"Run dir:       {run_dir}")
    print(f"Filters:       {filters_path}")
    print(f"Pre-scoring:   {folding_csv.name} (folding stage NOT re-run)")
    print(f"Mode:          {'DRY RUN (no writes)' if args.dry_run else 'IN-PLACE REWRITE'}")

    # ── Hotspots ─────────────────────────────────────────────────────────
    hotspot_spec = None if args.no_hotspot else (
        args.hotspot_residues or _resolve_hotspots_from_run(run_dir)
    )
    if hotspot_spec and not args.hotspot_residues:
        print(f"Hotspots:      from settings_used.json -> {hotspot_spec}")
    hotspot_ids = _resolve_hotspot_positions(hotspot_spec, args.target_pdb, args.target_chain)
    min_hotspot_contacts = _resolve_min_hotspot_contacts(filters, model_num)
    if hotspot_ids:
        print(f"Hotspot check: {len(hotspot_ids)} position(s), min {min_hotspot_contacts} contact(s)")
    else:
        print("Hotspot check: DISABLED (no hotspots resolved) — "
              "designs will not be filtered on hotspot contact")

    # ── Stage 1: folding filters ─────────────────────────────────────────
    old_rows = _load_rows(folding_csv)
    if not old_rows:
        raise SystemExit(
            f"{folding_csv} has a header but no data rows — nothing to re-filter."
        )
    old_rows = [r for r in old_rows if r.get("design_name")]
    if not old_rows:
        raise SystemExit(f"{folding_csv} has no usable design_name column.")
    new_rows = _apply_folding_filters(old_rows, filters, model_num, args.force_rosetta)

    old_pass = {r["design_name"]: bool(r.get("folding_filter_pass")) for r in old_rows}
    new_pass = {r["design_name"]: bool(r.get("folding_filter_pass")) for r in new_rows}
    gained = sorted(d for d in new_pass if new_pass[d] and not old_pass.get(d))
    lost = sorted(d for d in new_pass if not new_pass[d] and old_pass.get(d))

    n_predicted = sum(1 for r in new_rows if r.get("status") == "predicted")
    print("\n=== Folding filter ===")
    print(f"  designs: {len(new_rows)} ({n_predicted} predicted)")
    print(f"  pass: {sum(old_pass.values())} -> {sum(new_pass.values())} "
          f"(+{len(gained)} gained, -{len(lost)} lost)")
    for metric in FOLDING_METRIC_TO_SUFFIX:
        col = f"filter_{metric}"
        before = sum(1 for r in old_rows if r.get(col) == "FAIL")
        after = sum(1 for r in new_rows if r.get(col) == "FAIL")
        present = any(col in r for r in new_rows) or any(col in r for r in old_rows)
        if present and before != after:
            key, spec = _pick_filter_spec(filters, FOLDING_METRIC_TO_SUFFIX[metric], model_num)
            thr = spec.get("threshold") if spec else None
            print(f"    {col:<26} FAIL {before:>5} -> {after:<5} (now {key}={thr})")

    missing = _missing_metric_columns(new_rows, filters, model_num)
    if missing:
        print(f"  WARNING: {len(missing)} filter(s) have a threshold but no data in this CSV;")
        print("           they are recorded as N/A and NOT enforced:")
        for metric, key, thr in missing:
            print(f"             {metric} ({key}={thr})")
        print("           Either the run predates the metric (only refolding can supply")
        print("           it, so this folder cannot be made identical to a fresh run), or")
        print("           the folding model never emits it (e.g. Protenix leaves pAE/i_pAE")
        print("           blank) — in which case that threshold is dead for this model.")

    # ── Stage 2: PyRosetta filters over already-stored metrics ───────────
    folding_by_name = {r["design_name"]: r for r in new_rows}
    # An empty / absent summary is a legitimate state: folding finished but
    # PyRosetta never ran, so every folding-passing design is simply unvalidated.
    stored: Dict[str, Dict] = {}
    if summary_csv.exists():
        summary_rows = _load_rows(summary_csv)
        if not summary_rows:
            print(f"Note: {summary_csv.name} is empty — treating all folding-passing "
                  "designs as unvalidated.")
        for rec in summary_rows:
            design_name = rec.get("design_name")
            if design_name:
                stored[design_name] = rec

    scored_before = {d for d, r in stored.items() if r.get("status") == "scored"}
    failed_before = {d for d, r in stored.items() if r.get("status") == "rosetta_failed"}

    keep_rows: Dict[str, Dict] = {}
    pyro_flips: List[str] = []
    for design_name in scored_before:
        if not new_pass.get(design_name):
            continue  # dropped below
        stored_row = stored[design_name]
        metrics = {
            k: v for k, v in stored_row.items()
            if k not in SUMMARY_HEAD_KEYS
            and not k.startswith("pyrosetta_filter_")
            and k not in ("sequence", "reason")
        }
        filter_results, passed = _apply_pyrosetta_filters(
            metrics, filters, model_num, hotspot_ids, min_hotspot_contacts
        )
        was = bool(stored_row.get("pyrosetta_filter_pass"))
        if was != passed:
            pyro_flips.append(f"{design_name}: {was} -> {passed}")
        keep_rows[design_name] = _build_summary_row(
            design_name, folding_by_name.get(design_name, {}), metrics, filter_results, passed
        )

    retry = sorted(failed_before & {d for d in new_pass if new_pass[d]}) if args.retry_failed else []
    to_drop = sorted(d for d in stored if not new_pass.get(d))
    to_score = sorted(
        d for d, ok in new_pass.items()
        if ok and d not in stored
    ) + retry

    print(f"\n=== PyRosetta filters (re-evaluated from stored metrics, no re-run) ===")
    print(f"  already scored and still folding-passing: {len(keep_rows)}")
    print(f"  pyrosetta_filter_pass flips: {len(pyro_flips)}")
    for line in pyro_flips[:15]:
        print(f"    {line}")
    if len(pyro_flips) > 15:
        print(f"    ... and {len(pyro_flips) - 15} more")

    folded = _index_folded_pdbs(run_dir / f"{folding_model}_predictions")
    missing_pdb = [d for d in to_score if d not in folded]
    runnable = [d for d in to_score if d in folded]

    print(f"\n=== PyRosetta runs needed ===")
    print(f"  new / unvalidated designs to score: {len(to_score)}")
    if retry:
        print(f"    (incl. {len(retry)} rosetta_failed retries)")
    for d in runnable[:15]:
        print(f"    + {d}")
    if len(runnable) > 15:
        print(f"    ... and {len(runnable) - 15} more")
    if missing_pdb:
        print(f"  WARNING: no folded PDB found for {len(missing_pdb)} design(s); "
              f"they will be skipped (same as a normal run): {', '.join(missing_pdb[:5])}"
              f"{' ...' if len(missing_pdb) > 5 else ''}")

    print(f"\n=== Rows dropped from scoring_summary.csv ===")
    print(f"  no longer folding-passing: {len(to_drop)}")
    for d in to_drop[:15]:
        print(f"    - {d}")
    if len(to_drop) > 15:
        print(f"    ... and {len(to_drop) - 15} more")

    projected = sum(1 for r in keep_rows.values() if r.get("pyrosetta_filter_pass"))
    print(f"\n=== Projected result ===")
    print(f"  scoring_summary rows: {len(stored)} -> "
          f"{len(keep_rows) + len(runnable)} (+{len(runnable)} new, -{len(to_drop)} dropped)")
    print(f"  pyrosetta_filter_pass among re-evaluated rows: {projected}/{len(keep_rows)}"
          f"  (+ up to {len(runnable)} from new PyRosetta runs)")

    if args.dry_run:
        print("\nDRY RUN — nothing written. Re-run without --dry_run to apply.")
        return

    # ── Stage 3: PyRosetta for genuinely new designs ─────────────────────
    dalphaball = args.dalphaball or str(repo_root / "functions" / "DAlphaBall.gcc")
    if runnable and not os.path.exists(dalphaball):
        print(f"WARNING: dalphaball not found at {dalphaball}")

    if runnable:
        from tqdm import tqdm
        from pipeline.scoring.pyrosetta_scorer import run_pyrosetta_scoring

        start = time.time()
        for design_name in tqdm(runnable, desc="PyRosetta (new designs)"):
            try:
                metrics = run_pyrosetta_scoring(
                    af2_pdb=folded[design_name],
                    output_dir=run_dir,
                    design_name=design_name,
                    dalphaball_path=dalphaball,
                )
                filter_results, passed = _apply_pyrosetta_filters(
                    metrics, filters, model_num, hotspot_ids, min_hotspot_contacts
                )
                keep_rows[design_name] = _build_summary_row(
                    design_name, folding_by_name.get(design_name, {}),
                    metrics, filter_results, passed,
                )
            except Exception as exc:  # same shape a normal run records
                keep_rows[design_name] = {
                    "design_name": design_name,
                    "status": "rosetta_failed",
                    "reason": str(exc),
                    "pyrosetta_filter_pass": False,
                }
        print(f"PyRosetta stage: {time.time() - start:.1f}s for {len(runnable)} design(s)")

    # Keep rosetta_failed rows we did not retry, so nothing is silently lost.
    for design_name in sorted(failed_before):
        if new_pass.get(design_name) and design_name not in keep_rows:
            keep_rows[design_name] = stored[design_name]

    # ── Stage 4: write ───────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (folding_csv, summary_csv):
        if path.exists() and not args.no_backup:
            shutil.copy2(path, path.with_suffix(f".csv.bak_{stamp}"))

    pd.DataFrame(new_rows).to_csv(folding_csv, index=False)

    # Row order follows the pre-scoring CSV, exactly like a normal run.
    ordered = [keep_rows[r["design_name"]] for r in new_rows if r["design_name"] in keep_rows]
    pd.DataFrame(ordered).to_csv(summary_csv, index=False)

    partial = run_dir / "scoring_summary_partial.csv"
    if partial.exists():
        partial.unlink()

    n_pass = sum(1 for r in ordered if r.get("pyrosetta_filter_pass"))
    print(f"\nWrote {folding_csv.name} ({len(new_rows)} rows) and "
          f"{summary_csv.name} ({len(ordered)} rows, {n_pass} passing).")
    if not args.no_backup:
        print(f"Backups kept with suffix .bak_{stamp}")


if __name__ == "__main__":
    main()
