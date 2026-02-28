import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure imports work when running as a script path:
# python laproteina/proteinfoundation/score_existing_two_step.py
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


def _pick_filter_spec(
    filters: Dict,
    metric_suffix: str,
    model_num: int,
) -> Tuple[Optional[str], Optional[Dict]]:
    """Pick model-specific filter key, falling back to model 1 then unsuffixed."""
    candidates = [f"{model_num}_{metric_suffix}", f"1_{metric_suffix}", metric_suffix]
    for key in candidates:
        spec = filters.get(key)
        if isinstance(spec, dict) and spec.get("threshold") is not None:
            return key, spec
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-step scoring for existing complexes: AF2 prefilter -> PyRosetta scoring"
    )
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory with input complex PDBs")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory for scoring outputs")
    parser.add_argument("--filters", "-f", type=str, default=None, help="Filter JSON path")
    parser.add_argument("--af2_model", type=int, default=None, help="AF2 model index 1-5 (default: random)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filters = {}
    filter_path = args.filters
    if not filter_path:
        repo_root = Path(__file__).parent.parent.parent
        default_filters = repo_root / "settings_filters" / "laProteina_filters.json"
        if default_filters.exists():
            filter_path = str(default_filters)
    if filter_path and os.path.exists(filter_path):
        with open(filter_path, "r") as f:
            filters = json.load(f)
        print(f"Using filters: {filter_path}")
    else:
        print("No filter file provided/found; AF2 filter stage will keep all designs.")

    pdbs = sorted(input_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No PDBs found in {input_dir}")
    print(f"Found {len(pdbs)} PDB files")

    from colabdesign import mk_afdesign_model, clear_mem
    from colabdesign.shared.utils import copy_dict
    from laproteina_scoring.utils import extract_sequence_from_pdb

    af2_predictions_dir = output_dir / "af2_predictions"
    af2_predictions_dir.mkdir(parents=True, exist_ok=True)

    clear_mem()
    complex_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=1,
        use_multimer=True,
        use_initial_guess=False,
        use_initial_atom_pos=False,
    )

    if args.af2_model is None:
        import random
        af2_model_num = random.randint(0, 4)
    else:
        af2_model_num = max(0, min(4, int(args.af2_model) - 1))
    print(f"AF2 model: {af2_model_num + 1}")

    af2_rows = []
    af2_outputs = []

    for pdb in tqdm(pdbs, desc="AF2 Pre-scoring"):
        design_name = pdb.stem
        try:
            binder_seq = extract_sequence_from_pdb(str(pdb), "B")
            if not binder_seq:
                af2_rows.append({"design_name": design_name, "status": "failed", "reason": "No binder sequence found (Chain B missing?)"})
                continue
            binder_seq = "".join([c for c in binder_seq.upper() if "A" <= c <= "Z"])
            binder_len = len(binder_seq)
            if binder_len == 0:
                af2_rows.append({"design_name": design_name, "status": "failed", "reason": "Empty binder sequence"})
                continue

            complex_model.prep_inputs(
                pdb_filename=str(pdb),
                chain="A",
                binder_len=binder_len,
                rm_target_seq=True,
                rm_target_sc=True,
            )
            complex_model.predict(
                seq=binder_seq,
                models=[af2_model_num],
                num_recycles=1,
                verbose=False,
            )
            af2_pdb_path = af2_predictions_dir / f"{design_name}_model{af2_model_num + 1}.pdb"
            complex_model.save_pdb(str(af2_pdb_path))

            metrics = copy_dict(complex_model.aux["log"])
            residue_plddt = complex_model.aux.get("plddt")
            binder_plddt = None
            if residue_plddt is not None:
                residue_plddt = np.asarray(residue_plddt).reshape(-1)
                if residue_plddt.size >= binder_len:
                    binder_plddt = round(float(np.mean(residue_plddt[-binder_len:])), 3)

            row = {
                "design_name": design_name,
                "status": "predicted",
                "sequence": binder_seq,
                "binder_length": binder_len,
                "af2_model": af2_model_num + 1,
                "pLDDT": round(metrics["plddt"], 3),
                "binder_pLDDT": binder_plddt,
                "pTM": round(metrics["ptm"], 3),
                "i_pTM": round(metrics["i_ptm"], 3),
                "pAE": round(metrics["pae"], 3),
                "i_pAE": round(metrics["i_pae"], 3),
            }
            af2_rows.append(row)
            af2_outputs.append((design_name, str(af2_pdb_path)))
        except Exception as e:
            af2_rows.append({"design_name": design_name, "status": "failed", "reason": str(e)})

    af2_df = pd.DataFrame(af2_rows)
    af2_csv = output_dir / "af2_prescoring_results.csv"
    af2_df.to_csv(af2_csv, index=False)
    print(f"Saved AF2 pre-scoring to {af2_csv}")

    metric_to_suffix = {
        "pLDDT": "pLDDT",
        "binder_pLDDT": "Binder_pLDDT",
        "pTM": "pTM",
        "i_pTM": "i_pTM",
        "pAE": "pAE",
        "i_pAE": "i_pAE",
    }

    passing = []
    for design_name, af2_pdb in af2_outputs:
        row = next((r for r in af2_rows if r.get("design_name") == design_name and r.get("status") == "predicted"), None)
        if row is None:
            continue
        passed = True
        for metric, suffix in metric_to_suffix.items():
            key, spec = _pick_filter_spec(filters, suffix, af2_model_num + 1)
            if not key or not spec:
                continue
            value = row.get(metric)
            thr = spec.get("threshold")
            if value is None or thr is None:
                continue
            higher = spec.get("higher", True)
            if (higher and value < thr) or ((not higher) and value > thr):
                passed = False
                break
        if passed:
            passing.append((design_name, af2_pdb))

    print(f"AF2 filter pass: {len(passing)}/{len(af2_outputs)}")
    if not passing:
        print("No AF2-passing designs; stopping before PyRosetta.")
        return

    import pyrosetta as pr
    from functions.pyrosetta_utils import get_pyrosetta_init_cmd, pr_relax, score_interface

    repo_root = Path(__file__).parent.parent.parent
    dalphaball_path = str(repo_root / "functions" / "DAlphaBall.gcc")
    try:
        pr.init(get_pyrosetta_init_cmd(dalphaball_path))
    except Exception:
        pass

    relaxed_dir = output_dir / "relaxed"
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    scored_dir = output_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)

    final_rows = []
    for design_name, af2_pdb in tqdm(passing, desc="PyRosetta Scoring"):
        try:
            relaxed_pdb = relaxed_dir / f"{design_name}_relaxed.pdb"
            pr_relax(af2_pdb, str(relaxed_pdb))
            interface_scores, _, _ = score_interface(str(relaxed_pdb), binder_chain="B")
            af2_row = next((r for r in af2_rows if r.get("design_name") == design_name), {})
            final_rows.append(
                {
                    "design_name": design_name,
                    "status": "scored",
                    "pLDDT": af2_row.get("pLDDT"),
                    "binder_pLDDT": af2_row.get("binder_pLDDT"),
                    "pTM": af2_row.get("pTM"),
                    "i_pTM": af2_row.get("i_pTM"),
                    "pAE": af2_row.get("pAE"),
                    "i_pAE": af2_row.get("i_pAE"),
                    "af2_model": af2_row.get("af2_model"),
                    "binder_length": af2_row.get("binder_length"),
                    **interface_scores,
                    "sequence": af2_row.get("sequence"),
                }
            )
        except Exception as e:
            final_rows.append({"design_name": design_name, "status": "rosetta_failed", "reason": str(e)})

    final_df = pd.DataFrame(final_rows)
    summary_csv = scored_dir / "scoring_summary.csv"
    final_df.to_csv(summary_csv, index=False)
    print(f"Saved final scoring summary to {summary_csv}")


if __name__ == "__main__":
    main()
