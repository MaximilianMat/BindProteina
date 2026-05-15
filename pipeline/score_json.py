"""
Score complexes from a Protenix-style input JSON (sequences only, no PDB required).

Usage:
    python pipeline/score_json.py \\
        --json_path input.json \\
        --output_dir /path/to/output \\
        [--filters settings_filters/laProteina_filters.json] \\
        [--target_msa /path/to/target_msa_dir_or.a3m] \\
        [--target_template /path/to/target.pdb] \\
        [--protenix_model protenix-v2] \\
        [--seed 42] \\
        [--chunk_size 50]

Convention: first chain in each JSON entry = target; all remaining chains = binder.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.scoring.scorer import score_from_protenix_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score protein complexes from a Protenix input JSON (MSA → Protenix → PyRosetta)."
    )
    parser.add_argument("--json_path", "-j", required=True,
                        help="Path to Protenix-style input JSON (list of named sequence entries).")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory for all scoring outputs.")
    parser.add_argument("--filters", "-f", default=None,
                        help="Path to filter JSON (optional; defaults to settings_filters/laProteina_filters.json).")
    parser.add_argument("--target_msa", default=None,
                        help="Pre-computed MSA directory or .a3m file for the target chain (optional; "
                             "auto-generated via protenix msa if omitted).")
    parser.add_argument("--target_template", default=None,
                        help="Structural template for the target: .pdb/.cif (runs protenix mt) or "
                             ".a3m/.hhr (used directly).")
    parser.add_argument("--protenix_model", default="protenix_base_20250630_v1.0.0",
                        help="Protenix model name (default: protenix_base_20250630_v1.0.0).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Protenix (default: 42).")
    parser.add_argument("--chunk_size", type=int, default=25,
                        help="Designs per Protenix subprocess chunk (default: 50).")
    parser.add_argument("--force_rosetta", action="store_true",
                        help="Run PyRosetta on all folded designs regardless of folding filter results.")
    parser.add_argument("--fetch_all_msas", action="store_true",
                        help="Fetch MSAs for all chains in the complex (e.g. for antibody folding).")
    args = parser.parse_args()

    filter_path = args.filters
    if not filter_path:
        repo_root = Path(__file__).resolve().parent.parent
        default = repo_root / "settings_filters" / "laProteina_filters.json"
        if default.exists():
            filter_path = str(default)

    score_from_protenix_json(
        json_path=args.json_path,
        output_dir=args.output_dir,
        filters_path=filter_path,
        protenix_model_name=args.protenix_model,
        target_msa=args.target_msa,
        target_template=args.target_template,
        chunk_size=args.chunk_size,
        seed=args.seed,
        force_rosetta=args.force_rosetta,
        fetch_all_msas=args.fetch_all_msas,
    )


if __name__ == "__main__":
    main()
