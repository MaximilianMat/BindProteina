import argparse
import json
import os
import sys
from pathlib import Path

# Ensure imports work when running as a script path:
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pipeline.scoring.scorer import score_complexes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-step scoring for existing complexes: Folding prefilter -> PyRosetta scoring"
    )
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory with input complex PDBs")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory for scoring outputs")
    parser.add_argument("--filters", "-f", type=str, default=None, help="Filter JSON path")
    parser.add_argument("--af2_model", type=int, default=None, help="AF2 model index 1-5 (default: random)")
    parser.add_argument("--folding_model", type=str, default="af2", choices=["af2", "protenix"],
                        help="Folding model to use for refolding (default: af2)")
    parser.add_argument("--protenix_model", type=str, default="protenix-v2",
                        help="Protenix model name (default: protenix-v2)")
    args = parser.parse_args()

    filter_path = args.filters
    if not filter_path:
        repo_root = Path(__file__).parent.parent
        default_filters = repo_root / "settings_filters" / "laProteina_filters.json"
        if default_filters.exists():
            filter_path = str(default_filters)

    score_complexes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filters_path=filter_path,
        af2_model_idx=args.af2_model,
        folding_model=args.folding_model,
        protenix_model_name=args.protenix_model,
    )

if __name__ == "__main__":
    main()
