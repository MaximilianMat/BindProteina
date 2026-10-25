"""
Reclaim disk by deleting intermediate scoring inputs from completed generation runs.

Each generation run (pipeline/generate.py) writes, per run directory:

  KEEP  generated/              raw LaProteina designs
  KEEP  protenix_predictions/   (or af2_predictions/) refolded complexes
  KEEP  relaxed/                PyRosetta-relaxed final structures
  KEEP  *.csv, *.log, *.json    scoring summaries / metadata at the run root

  DROP  scoring_input/          folding-stage input PDBs; only re-read if you
                                re-run scoring from scratch
  DROP  pyrosetta_input/        created but never consumed downstream (PyRosetta
                                relaxes the refold in protenix_predictions/)

`pyrosetta_input/` is always safe to remove.  `scoring_input/` is removed only when
its run looks finished — a `scoring_summary.csv` exists somewhere under the run
directory — unless you pass --force.  Use --dry-run to preview.

Companion to clean_past_chunks.py (which removes leftover Protenix chunk_* dirs).

Usage:
    python clean_intermediate_inputs.py --dir /nfs/students/matscher/samples
    python clean_intermediate_inputs.py --dir . --dry-run
    python clean_intermediate_inputs.py --dir . --force          # skip completion check
    python clean_intermediate_inputs.py --dir . --targets scoring_input
"""

import os
import shutil
from pathlib import Path

# Folders that are always always safe to delete (never read after creation).
ALWAYS_SAFE = {"pyrosetta_input"}
# Folders that are safe only once the run has produced a scoring summary.
GATED = {"scoring_input"}
DEFAULT_TARGETS = ["scoring_input", "pyrosetta_input"]

COMPLETION_MARKER = "scoring_summary.csv"


def get_dir_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total


def _run_is_complete(run_dir: Path) -> bool:
    """A run counts as complete if a scoring_summary.csv exists under its run dir."""
    marker = run_dir / COMPLETION_MARKER
    if marker.exists():
        return True
    # evolve_v2 writes it under a 'scored/' subdir; search shallowly to be safe.
    return any(run_dir.rglob(COMPLETION_MARKER))


def clean_intermediate_inputs(base_dir: str, targets, dry_run: bool = False, force: bool = False):
    base = Path(base_dir).resolve()
    action = "Would delete" if dry_run else "Deleting"
    print(f"Scanning {base} for intermediate folders: {', '.join(targets)}")
    if dry_run:
        print("(dry-run: nothing will be removed)\n")

    deleted_count = 0
    skipped_count = 0
    freed_size = 0
    target_set = set(targets)

    # Recurse every sub-folder with os.walk (top-down).  Unlike Path.rglob, os.walk
    # silently skips branches it can't read instead of aborting the whole scan, which
    # matters on flaky NFS mounts.
    for dirpath, dirnames, _ in os.walk(base):
        for name in sorted(set(dirnames) & target_set):
            tdir = Path(dirpath) / name
            # Don't descend into a folder we're about to remove / report.
            dirnames.remove(name)

            run_dir = tdir.parent
            gated = name in GATED and name not in ALWAYS_SAFE
            if gated and not force and not _run_is_complete(run_dir):
                print(f"SKIP  {tdir}  (run not complete — no {COMPLETION_MARKER}; use --force to override)")
                skipped_count += 1
                continue

            size = get_dir_size(tdir)
            print(f"{action}  {tdir}  ({size / (1024**3):.2f} GB)")
            if dry_run:
                deleted_count += 1
                freed_size += size
                continue
            try:
                shutil.rmtree(tdir)
                deleted_count += 1
                freed_size += size
                print(f"  -> done")
            except Exception as e:
                print(f"  -> FAILED: {e}")

    verb = "Would free" if dry_run else "Freed"
    print(f"\nCleanup {'preview' if dry_run else 'complete'}!")
    print(f"{'Would remove' if dry_run else 'Removed'} {deleted_count} folder(s); {skipped_count} skipped.")
    print(f"{verb} {freed_size / (1024**3):.2f} GB of disk space.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Delete intermediate scoring_input/pyrosetta_input folders from completed runs."
    )
    parser.add_argument("--dir", type=str, default=".",
                        help="Base directory to scan (e.g. /nfs/students/matscher/samples)")
    parser.add_argument("--targets", type=str, default=",".join(DEFAULT_TARGETS),
                        help=f"Comma-separated folder names to remove (default: {','.join(DEFAULT_TARGETS)})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be deleted without removing anything.")
    parser.add_argument("--force", action="store_true",
                        help="Delete scoring_input even if the run has no scoring_summary.csv.")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    clean_intermediate_inputs(args.dir, targets, dry_run=args.dry_run, force=args.force)
