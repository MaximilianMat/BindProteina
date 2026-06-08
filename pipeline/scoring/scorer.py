import os
import json
import random
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

from pipeline.scoring.pyrosetta_scorer import run_pyrosetta_scoring

def _pick_filter_spec(filters: Dict, metric_suffix: str, model_num: int):
    candidates = [f"{model_num}_{metric_suffix}", f"1_{metric_suffix}", metric_suffix]
    for key in candidates:
        spec = filters.get(key)
        if isinstance(spec, dict) and spec.get("threshold") is not None:
            return key, spec
    return None, None

def score_complexes(
    input_dir: str,
    output_dir: str,
    filters_path: Optional[str] = None,
    af2_model_idx: Optional[int] = None,
    dalphaball_path: Optional[str] = None,
    folding_model: str = "af2",
    protenix_model_name: str = "protenix-v2",
    target_msa: Optional[str] = None,
    msa_db_path: Optional[str] = None,
    target_template: Optional[str] = None,
    force_rosetta: bool = False,
    fetch_all_msas: bool = False,
    n_diffusion_samples: int = 5,
):
    """
    Universal 2-stage scoring orchestrator.

    Stage 1: Re-fold each complex with the chosen folding model (AF2 or Protenix).
    Stage 2: Score AF2/Protenix-passing designs with PyRosetta.

    Args:
        input_dir: Directory of input complex PDBs (chain A = target, B = binder).
        output_dir: Directory for all scoring outputs.
        filters_path: Optional JSON file with AF2/Protenix metric thresholds.
        af2_model_idx: AF2 model index 1-5 (only used when folding_model="af2").
        dalphaball_path: Path to DAlphaBall executable for PyRosetta.
        folding_model: Which folding model to use: "af2" or "protenix".
        protenix_model_name: Protenix model name (only used when folding_model="protenix").
        target_msa: Optional path to a pre-computed MSA directory (containing pairing.a3m +
            non_pairing.a3m) for the target protein (Chain A).  A single .a3m file is also
            accepted for backward compatibility.
        target_template: Optional path to a structural template for the target chain.
            Must be a .a3m or .hhr file (Protenix templatesPath format). Generate it
            with `protenix mt -i <input.json> -o <out_dir> -m protenix` on a node with
            internet access; the result is at <out_dir>/target/msa/hmmsearch.a3m.
            When provided, --use_template true is passed to protenix pred.
        fetch_all_msas: Whether to fetch MSAs for all chains.
    """
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    start_time_total = time.time()

    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filters = {}
    if filters_path and os.path.exists(filters_path):
        with open(filters_path, "r") as f:
            filters = json.load(f)
        print(f"Using filters: {filters_path}")
    else:
        print("No filter file provided; folding filter stage will keep all designs.")

    pdbs = sorted(input_path.rglob("*.pdb"))
    if not pdbs:
        print(f"No PDBs found in {input_path}")
        return

    # ── Resolve / generate Target MSA directory ──────────────────────
    if folding_model == "protenix":
        from pipeline.scoring.folding_protenix import generate_target_msa_dir
        import shutil as _shutil
        msa_out_dir = out_path / "target_msa"
        start_time_msa = time.time()
        try:
            if target_msa and Path(target_msa).is_file():
                # User supplied a custom non-pairing .a3m:
                # 1. Generate the MSA directory online to obtain pairing.a3m
                # 2. Override the generated non_pairing.a3m with the supplied file
                from pipeline.scoring.utils import extract_sequence_from_pdb
                target_seq = "".join(
                    c for c in extract_sequence_from_pdb(str(pdbs[0]), "A").upper()
                    if "A" <= c <= "Z"
                )
                supplied_a3m = Path(target_msa).absolute()
                print(f"Generating target pairing MSA; will override non_pairing.a3m with: {supplied_a3m.name}")
                target_msa = generate_target_msa_dir(target_seq=target_seq, output_dir=msa_out_dir)
                non_pairing_dst = Path(target_msa) / "non_pairing.a3m"
                if non_pairing_dst.exists():
                    non_pairing_dst.unlink()
                _shutil.copy2(supplied_a3m, non_pairing_dst)
                print(f"  Overrode non_pairing.a3m with supplied file.")
            elif target_msa and Path(target_msa).is_dir():
                pass  # already the right format; use as-is
            else:
                # Nothing supplied — generate via the Protenix online MSA server
                from pipeline.scoring.utils import extract_sequence_from_pdb
                target_seq = "".join(
                    c for c in extract_sequence_from_pdb(str(pdbs[0]), "A").upper()
                    if "A" <= c <= "Z"
                )
                print("Auto-generating target MSA (protenix online server)...")
                target_msa = generate_target_msa_dir(
                    target_seq=target_seq,
                    output_dir=msa_out_dir,
                )
            elapsed_msa = time.time() - start_time_msa
            print(f"Target MSA directory: {target_msa} ({elapsed_msa:.1f}s)")
        except Exception as e:
            print(f"WARNING: MSA setup failed: {e}. Proceeding without MSA.")
            target_msa = None

    # ── Resolve template ─────────────────────────────────────────────
    # Pass the file directly to Protenix as templatesPath.
    # .pdb/.cif files are used as-is; .a3m/.hhr files are passed through unchanged.
    # (We do NOT run protenix mt here — that requires internet access and searches
    # for homologous structures, which is wrong when the user has already supplied
    # the exact crystal structure they want to use as a template.)
    resolved_template: Optional[str] = target_template if target_template and Path(target_template).exists() else None
    if target_template and not resolved_template:
        print(f"WARNING: Template file not found: {target_template}. Proceeding without template.")
    if resolved_template and Path(resolved_template).suffix.lower() not in (".a3m", ".hhr"):
        print(f"WARNING: Template file {resolved_template} is not .a3m/.hhr — Protenix only supports those formats. Proceeding without template.")
        resolved_template = None

    # ── Stage 1: Folding ─────────────────────────────────────────────
    start_time_folding = time.time()
    predictions_dir = out_path / f"{folding_model}_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    folding_rows = []
    folding_outputs = []

    folding_csv = out_path / f"{folding_model}_prescoring_results.csv"
    skip_folding = False
    
    if folding_csv.exists():
        existing_df = pd.read_csv(folding_csv)
        if len(existing_df) == len(pdbs) and (existing_df["status"] == "predicted").all():
            print(f"Found complete pre-scoring CSV with {len(existing_df)} designs, skipping folding stage.")
            folding_rows = existing_df.to_dict("records")
            for row in folding_rows:
                if row.get("status") == "predicted":
                    design_name = row["design_name"]
                    cands = list(predictions_dir.rglob(f"{design_name}_*.pdb"))
                    if cands:
                        folding_outputs.append((design_name, str(cands[0])))
            skip_folding = True

    if not skip_folding:
        if folding_model == "af2":
            from colabdesign import mk_afdesign_model, clear_mem
            from pipeline.scoring.folding import run_af2_refolding
    
            clear_mem()
            complex_model = mk_afdesign_model(
                protocol="binder",
                num_recycles=1,
                use_multimer=True,
                use_initial_guess=False,
                use_initial_atom_pos=False,
            )
    
            if af2_model_idx is None:
                af2_model_num = random.randint(1, 5)
            else:
                af2_model_num = max(1, min(5, int(af2_model_idx)))
            print(f"Folding model: AF2 (model {af2_model_num})")
    
            for pdb in tqdm(pdbs, desc="AF2 Re-folding"):
                design_name = pdb.stem
                try:
                    folded_pdb, metrics = run_af2_refolding(
                        pdb_path=str(pdb),
                        design_name=design_name,
                        output_dir=predictions_dir,
                        complex_model=complex_model,
                        af2_model_num=af2_model_num,
                    )
                    if folded_pdb is None:
                        folding_rows.append({"design_name": design_name, "status": "failed", "reason": metrics.get("error", "Unknown error")})
                        continue
    
                    metrics["folding_model"] = f"af2_model{af2_model_num}"
                    _add_sc_rmsd(str(pdb), folded_pdb, metrics)
    
                    row = {"design_name": design_name, "status": "predicted", **metrics}
                    folding_rows.append(row)
                    folding_outputs.append((design_name, folded_pdb))
                except Exception as e:
                    folding_rows.append({"design_name": design_name, "status": "failed", "reason": str(e)})
    
        elif folding_model == "protenix":
            from pipeline.scoring.folding_protenix import run_protenix_refolding_batch
    
            print(f"Folding model: Protenix ({protenix_model_name}) - Batch Mode")
            
            pdb_paths_str = [str(p) for p in pdbs]
            try:
                batch_results = run_protenix_refolding_batch(
                    pdb_paths=pdb_paths_str,
                    output_dir=predictions_dir,
                    model_name=protenix_model_name,
                    target_msa_path=target_msa,
                    target_template_path=resolved_template,
                    fetch_all_msas=fetch_all_msas,
                    n_diffusion_samples=n_diffusion_samples,
                )
                
                for design_name, folded_pdb, metrics in batch_results:
                    if folded_pdb is None:
                        folding_rows.append({"design_name": design_name, "status": "failed", "reason": metrics.get("error", "Unknown error")})
                        continue
                    
                    orig_pdb = next((p for p in pdbs if p.stem == design_name), None)
                    if orig_pdb:
                        _add_sc_rmsd(str(orig_pdb), folded_pdb, metrics)
                    
                    row = {"design_name": design_name, "status": "predicted", **metrics}
                    folding_rows.append(row)
                    folding_outputs.append((design_name, folded_pdb))
                    
            except Exception as e:
                print(f"Batch prediction exception: {e}")
                for p in pdbs:
                    folding_rows.append({"design_name": p.stem, "status": "failed", "reason": str(e)})
        else:
            raise ValueError(f"Unknown folding_model: {folding_model!r}. Use 'af2' or 'protenix'.")

    folding_df = pd.DataFrame(folding_rows)
    folding_df.to_csv(folding_csv, index=False)
    print(f"Saved {folding_model} pre-scoring to {folding_csv}")

    elapsed_folding = time.time() - start_time_folding

    # ── Filter ────────────────────────────────────────────────────────
    # Map from our internal metric name -> filter JSON suffix
    metric_to_suffix = {
        "pLDDT": "pLDDT",
        "binder_pLDDT": "Binder_pLDDT",
        "pTM": "pTM",
        "i_pTM": "i_pTM",
        "pAE": "pAE",
        "i_pAE": "i_pAE",
        "sc_rmsd": "Binder_RMSD",
    }

    af2_model_num_for_filter = af2_model_idx or 1

    if force_rosetta:
        print("--force_rosetta: skipping folding filters, sending all predicted designs to PyRosetta.")

    passing = []
    for row in folding_rows:
        if row.get("status") != "predicted":
            row["folding_filter_pass"] = False
            continue

        design_name = row["design_name"]
        folded_pdb = next((fp for dn, fp in folding_outputs if dn == design_name), None)

        if force_rosetta:
            row["folding_filter_pass"] = True
            if folded_pdb:
                passing.append((design_name, folded_pdb))
            continue

        passed = True
        for metric, suffix in metric_to_suffix.items():
            key, spec = _pick_filter_spec(filters, suffix, af2_model_num_for_filter)
            if not key or not spec:
                continue

            # Check both capitalized and lowercase for robustness
            value = row.get(metric, row.get(metric.lower()))
            thr = spec.get("threshold")

            if value is None or thr is None:
                row[f"filter_{metric}"] = "N/A"
                continue

            higher = spec.get("higher", True)
            if (higher and value < thr) or ((not higher) and value > thr):
                row[f"filter_{metric}"] = "FAIL"
                passed = False
            else:
                row[f"filter_{metric}"] = "PASS"

        row["folding_filter_pass"] = passed
        if passed and folded_pdb:
            passing.append((design_name, folded_pdb))

    # Re-save the pre-scoring CSV with per-filter and overall pass columns
    folding_df = pd.DataFrame(folding_rows)
    folding_df.to_csv(folding_csv, index=False)

    # Print statistical summary for pre-scoring
    total_folding = len(folding_rows)
    if total_folding > 0:
        print(f"\n--- Folding Filter Statistics ---")
        print(f"Total designs: {total_folding}")
        print(f"Passing designs: {len(passing)} ({(len(passing)/total_folding)*100:.1f}%)")
        print("Failing rates per filter:")
        for metric in metric_to_suffix.keys():
            col = f"filter_{metric}"
            if col in folding_df.columns:
                n_fail = (folding_df[col] == "FAIL").sum()
                print(f"  {metric}: {(n_fail/total_folding)*100:.1f}%")
        print("---------------------------------\n")

    if not passing:
        print("No folding-passing designs; stopping before PyRosetta.")
        elapsed_total = time.time() - start_time_total
        print("\n=== Scoring Timing Summary ===")
        if 'elapsed_msa' in locals():
            print(f"MSA Generation:   {elapsed_msa:.2f}s")
        print(f"Folding Stage:    {elapsed_folding:.2f}s")
        print(f"Total Time:       {elapsed_total:.2f}s")
        print("==============================\n")
        return

    # ── Stage 2: PyRosetta Scoring (only passing designs) ─────────────
    start_time_pyrosetta = time.time()
    if dalphaball_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        dalphaball_path = str(repo_root / "functions" / "DAlphaBall.gcc")
        if not os.path.exists(dalphaball_path):
            print(f"WARNING: dalphaball default path not found at {dalphaball_path}.")

    pyrosetta_metric_to_suffix = {
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
        "interface_delta_unsat_hbonds_percentage": "InterfaceUnsatHbondsPercentage"
    }

    final_rows = []
    pyrosetta_passing = 0
    
    partial_csv = out_path / "scoring_summary_partial.csv"
    existing_scored = set()
    if partial_csv.exists():
        try:
            existing_df = pd.read_csv(partial_csv)
            existing_scored = set(existing_df["design_name"])
            final_rows = existing_df.to_dict("records")
            for row in final_rows:
                if row.get("pyrosetta_filter_pass", False):
                    pyrosetta_passing += 1
            print(f"Resuming PyRosetta: {len(existing_scored)} designs already scored")
        except Exception as e:
            print(f"Warning: could not read partial CSV: {e}")

    for design_name, folded_pdb in tqdm(passing, desc="PyRosetta Scoring"):
        if design_name in existing_scored:
            continue
        try:
            interface_scores = run_pyrosetta_scoring(
                af2_pdb=folded_pdb,
                output_dir=out_path,
                design_name=design_name,
                dalphaball_path=dalphaball_path
            )
            folding_row = next((r for r in folding_rows if r.get("design_name") == design_name), {})
            
            # Apply PyRosetta filters
            pyrosetta_passed = True
            filter_results = {}
            for metric, suffix in pyrosetta_metric_to_suffix.items():
                key, spec = _pick_filter_spec(filters, suffix, af2_model_num_for_filter)
                if not key or not spec:
                    continue
                
                value = interface_scores.get(metric)
                thr = spec.get("threshold")
                
                if value is None or thr is None:
                    filter_results[f"pyrosetta_filter_{metric}"] = "N/A"
                    continue
                
                higher = spec.get("higher", True)
                if (higher and value < thr) or ((not higher) and value > thr):
                    filter_results[f"pyrosetta_filter_{metric}"] = "FAIL"
                    pyrosetta_passed = False
                else:
                    filter_results[f"pyrosetta_filter_{metric}"] = "PASS"
            
            if pyrosetta_passed:
                pyrosetta_passing += 1
                
            final_rows.append(
                {
                    "design_name": design_name,
                    "status": "scored",
                    "folding_model": folding_row.get("folding_model"),
                    "pLDDT": folding_row.get("pLDDT"),
                    "binder_pLDDT": folding_row.get("binder_pLDDT"),
                    "pTM": folding_row.get("pTM"),
                    "i_pTM": folding_row.get("i_pTM"),
                    "pAE": folding_row.get("pAE"),
                    "i_pAE": folding_row.get("i_pAE"),
                    "sc_rmsd": folding_row.get("sc_rmsd"),
                    "binder_length": folding_row.get("binder_length"),
                    **interface_scores,
                    **filter_results,
                    "pyrosetta_filter_pass": pyrosetta_passed,
                    "sequence": folding_row.get("sequence"),
                }
            )
        except Exception as e:
            final_rows.append({"design_name": design_name, "status": "rosetta_failed", "reason": str(e), "pyrosetta_filter_pass": False})
            
        if len(final_rows) % 10 == 0:
            pd.DataFrame(final_rows).to_csv(partial_csv, index=False)

    final_df = pd.DataFrame(final_rows)
    summary_csv = out_path / "scoring_summary.csv"
    final_df.to_csv(summary_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()
    
    # Print statistical summary for PyRosetta scoring
    total_pyrosetta = len(final_rows) # those that failed rosetta will be here with status rosetta_failed
    # Actually, we should only count scored designs or just use len(final_rows)
    if total_pyrosetta > 0:
        print(f"\n--- PyRosetta Filter Statistics ---")
        print(f"Total scored designs: {total_pyrosetta}")
        print(f"Passing designs: {pyrosetta_passing} ({(pyrosetta_passing/total_pyrosetta)*100:.1f}%)")
        print("Failing rates per filter:")
        for metric in pyrosetta_metric_to_suffix.keys():
            col = f"pyrosetta_filter_{metric}"
            if col in final_df.columns:
                n_fail = (final_df[col] == "FAIL").sum()
                print(f"  {metric}: {(n_fail/total_pyrosetta)*100:.1f}%")
        print("-----------------------------------\n")

    elapsed_pyrosetta = time.time() - start_time_pyrosetta
    elapsed_total = time.time() - start_time_total

    print("\n=== Scoring Timing Summary ===")
    if 'elapsed_msa' in locals():
        print(f"MSA Generation:   {elapsed_msa:.2f}s")
    print(f"Folding Stage:    {elapsed_folding:.2f}s")
    print(f"PyRosetta Stage:  {elapsed_pyrosetta:.2f}s")
    print(f"Total Time:       {elapsed_total:.2f}s")
    print("==============================\n")

    print(f"Saved final scoring summary to {summary_csv}")


def score_from_protenix_json(
    json_path: str,
    output_dir: str,
    filters_path: Optional[str] = None,
    protenix_model_name: str = "protenix-v2",
    target_msa: Optional[str] = None,
    target_template: Optional[str] = None,
    dalphaball_path: Optional[str] = None,
    chunk_size: int = 50,
    seed: int = 42,
    force_rosetta: bool = False,
    fetch_all_msas: bool = False,
):
    """
    Full scoring pipeline starting from a Protenix-style input JSON.

    No PDB files required — sequences are taken directly from the JSON.
    Convention: first chain in each entry = target; all remaining chains = binder.

    Stages:
      1. Generate / resolve target MSA (first chain of first entry used for sequence).
      2. Run Protenix to fold all complexes in the JSON.
      3. Apply folding filters (if filters_path supplied).
      4. Run PyRosetta on passing designs.
      5. Write protenix_prescoring_results.csv and scoring_summary.csv.
    """
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    start_total = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(json_path) as fh:
        entries = json.load(fh)

    if not entries:
        print("Empty JSON — nothing to score.")
        return

    filters: dict = {}
    if filters_path and os.path.exists(filters_path):
        with open(filters_path) as fh:
            filters = json.load(fh)
        print(f"Using filters: {filters_path}")
    else:
        print("No filter file provided; folding filter stage will keep all designs.")

    # ── Resolve / generate target MSA ─────────────────────────────────
    from pipeline.scoring.folding_protenix import generate_target_msa_dir
    import shutil as _shutil

    msa_out_dir = out_path / "target_msa"
    start_msa = time.time()
    try:
        first_chain_seq = "".join(
            c for c in entries[0]["sequences"][0]["proteinChain"]["sequence"].upper()
            if "A" <= c <= "Z"
        )
        if target_msa and Path(target_msa).is_file():
            supplied_a3m = Path(target_msa).absolute()
            print(f"Generating target pairing MSA; will override non_pairing.a3m with: {supplied_a3m.name}")
            target_msa = generate_target_msa_dir(target_seq=first_chain_seq, output_dir=msa_out_dir)
            non_pairing_dst = Path(target_msa) / "non_pairing.a3m"
            if non_pairing_dst.exists():
                non_pairing_dst.unlink()
            _shutil.copy2(supplied_a3m, non_pairing_dst)
        elif target_msa and Path(target_msa).is_dir():
            pass  # use as-is
        else:
            print("Auto-generating target MSA (protenix online server)...")
            target_msa = generate_target_msa_dir(target_seq=first_chain_seq, output_dir=msa_out_dir)
        elapsed_msa = time.time() - start_msa
        print(f"Target MSA directory: {target_msa} ({elapsed_msa:.1f}s)")
    except Exception as e:
        print(f"WARNING: MSA setup failed: {e}. Proceeding without MSA.")
        target_msa = None
        elapsed_msa = time.time() - start_msa

    # ── Resolve template ───────────────────────────────────────────────
    resolved_template: Optional[str] = target_template if target_template and Path(target_template).exists() else None
    if target_template and not resolved_template:
        print(f"WARNING: Template file not found: {target_template}. Proceeding without template.")
    if resolved_template and Path(resolved_template).suffix.lower() not in (".a3m", ".hhr"):
        print(f"WARNING: Template file {resolved_template} is not .a3m/.hhr — Protenix only supports those formats. Proceeding without template.")
        resolved_template = None

    # ── Stage 1: Protenix folding ──────────────────────────────────────
    from pipeline.scoring.folding_protenix import run_protenix_from_json_payload

    predictions_dir = out_path / "protenix_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    start_folding = time.time()

    folding_csv = out_path / "protenix_prescoring_results.csv"
    folding_rows = []
    folding_outputs = []
    skip_folding = False
    
    if folding_csv.exists():
        existing_df = pd.read_csv(folding_csv)
        if len(existing_df) == len(entries) and (existing_df["status"] == "predicted").all():
            print(f"Found complete pre-scoring CSV with {len(existing_df)} complexes, skipping folding stage.")
            folding_rows = existing_df.to_dict("records")
            for row in folding_rows:
                if row.get("status") == "predicted":
                    design_name = row["design_name"]
                    binder_chains = row.get("binder_chains", "B")
                    cands = list(predictions_dir.rglob(f"{design_name}_protenix.pdb"))
                    if cands:
                        folding_outputs.append((design_name, str(cands[0]), binder_chains))
            skip_folding = True

    if not skip_folding:
        print(f"Folding model: Protenix ({protenix_model_name}) — JSON Batch Mode")
        batch_results = run_protenix_from_json_payload(
        entries=entries,
        output_dir=predictions_dir,
        model_name=protenix_model_name,
        seed=seed,
        target_msa_path=target_msa,
        target_template_path=resolved_template,
        chunk_size=chunk_size,
        fetch_all_msas=fetch_all_msas,
    )

        for design_name, folded_pdb, metrics in batch_results:
            if folded_pdb is None:
                folding_rows.append({
                    "design_name": design_name,
                    "status": "failed",
                    "reason": metrics.get("error", "Unknown error"),
                })
            else:
                folding_rows.append({"design_name": design_name, "status": "predicted", **metrics})
                folding_outputs.append((design_name, folded_pdb, metrics.get("binder_chains", "B")))
    
        folding_df = pd.DataFrame(folding_rows)
        folding_df.to_csv(folding_csv, index=False)
        print(f"Saved Protenix pre-scoring to {folding_csv}")
    elapsed_folding = time.time() - start_folding

    # ── Folding filters ────────────────────────────────────────────────
    if force_rosetta:
        print("--force_rosetta: skipping folding filters, sending all predicted designs to PyRosetta.")

    metric_to_suffix = {
        "pLDDT": "pLDDT", "binder_pLDDT": "Binder_pLDDT",
        "pTM": "pTM", "i_pTM": "i_pTM", "pAE": "pAE", "i_pAE": "i_pAE",
    }
    passing = []
    for row in folding_rows:
        if row.get("status") != "predicted":
            row["folding_filter_pass"] = False
            continue

        if force_rosetta:
            row["folding_filter_pass"] = True
            fo = next((t for t in folding_outputs if t[0] == row["design_name"]), None)
            if fo:
                passing.append(fo)
            continue

        passed = True
        for metric, suffix in metric_to_suffix.items():
            key, spec = _pick_filter_spec(filters, suffix, 1)
            if not key or not spec:
                continue
            value = row.get(metric)
            thr = spec.get("threshold")
            if value is None or thr is None:
                row[f"filter_{metric}"] = "N/A"
                continue
            higher = spec.get("higher", True)
            if (higher and value < thr) or (not higher and value > thr):
                row[f"filter_{metric}"] = "FAIL"
                passed = False
            else:
                row[f"filter_{metric}"] = "PASS"
        row["folding_filter_pass"] = passed
        if passed:
            fo = next((t for t in folding_outputs if t[0] == row["design_name"]), None)
            if fo:
                passing.append(fo)

    folding_df = pd.DataFrame(folding_rows)
    folding_df.to_csv(folding_csv, index=False)

    total_folding = len(folding_rows)
    if total_folding > 0:
        print(f"\n--- Folding Filter Statistics ---")
        print(f"Total: {total_folding}  Passing: {len(passing)} ({len(passing)/total_folding*100:.1f}%)")
        print("---------------------------------\n")

    if not passing:
        print("No folding-passing designs; stopping before PyRosetta.")
        print(f"\n=== Timing ===  MSA: {elapsed_msa:.1f}s  Folding: {elapsed_folding:.1f}s  "
              f"Total: {time.time()-start_total:.1f}s\n")
        return

    # ── Stage 2: PyRosetta ─────────────────────────────────────────────
    from pipeline.scoring.pyrosetta_scorer import run_pyrosetta_scoring

    if dalphaball_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        dalphaball_path = str(repo_root / "functions" / "DAlphaBall.gcc")
        if not os.path.exists(dalphaball_path):
            print(f"WARNING: dalphaball not found at {dalphaball_path}.")

    pyrosetta_metric_to_suffix = {
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

    start_pyrosetta = time.time()
    final_rows = []
    pyrosetta_passing = 0

    partial_csv = out_path / "scoring_summary_partial.csv"
    existing_scored = set()
    if partial_csv.exists():
        try:
            existing_df = pd.read_csv(partial_csv)
            existing_scored = set(existing_df["design_name"])
            final_rows = existing_df.to_dict("records")
            for row in final_rows:
                if row.get("pyrosetta_filter_pass", False):
                    pyrosetta_passing += 1
            print(f"Resuming PyRosetta: {len(existing_scored)} complexes already scored")
        except Exception as e:
            print(f"Warning: could not read partial CSV: {e}")

    for design_name, folded_pdb, binder_chains in tqdm(passing, desc="PyRosetta Scoring"):
        if design_name in existing_scored:
            continue
        try:
            interface_scores = run_pyrosetta_scoring(
                af2_pdb=folded_pdb,
                output_dir=out_path,
                design_name=design_name,
                dalphaball_path=dalphaball_path,
                target_chains="A",
                binder_chains=binder_chains,
            )
            folding_row = next((r for r in folding_rows if r.get("design_name") == design_name), {})
            pyrosetta_passed = True
            filter_results = {}
            for metric, suffix in pyrosetta_metric_to_suffix.items():
                key, spec = _pick_filter_spec(filters, suffix, 1)
                if not key or not spec:
                    continue
                value = interface_scores.get(metric)
                thr = spec.get("threshold")
                if value is None or thr is None:
                    filter_results[f"pyrosetta_filter_{metric}"] = "N/A"
                    continue
                higher = spec.get("higher", True)
                if (higher and value < thr) or (not higher and value > thr):
                    filter_results[f"pyrosetta_filter_{metric}"] = "FAIL"
                    pyrosetta_passed = False
                else:
                    filter_results[f"pyrosetta_filter_{metric}"] = "PASS"
            if pyrosetta_passed:
                pyrosetta_passing += 1
            final_rows.append({
                "design_name": design_name,
                "status": "scored",
                "folding_model": folding_row.get("folding_model"),
                "binder_chains": binder_chains,
                "pLDDT": folding_row.get("pLDDT"),
                "binder_pLDDT": folding_row.get("binder_pLDDT"),
                "pTM": folding_row.get("pTM"),
                "i_pTM": folding_row.get("i_pTM"),
                "ranking_score": folding_row.get("ranking_score"),
                "binder_length": folding_row.get("binder_length"),
                **interface_scores,
                **filter_results,
                "pyrosetta_filter_pass": pyrosetta_passed,
                "sequence": folding_row.get("sequence"),
            })
        except Exception as e:
            final_rows.append({
                "design_name": design_name,
                "status": "rosetta_failed",
                "reason": str(e),
                "pyrosetta_filter_pass": False,
            })
            
        if len(final_rows) % 10 == 0:
            pd.DataFrame(final_rows).to_csv(partial_csv, index=False)

    final_df = pd.DataFrame(final_rows)
    summary_csv = out_path / "scoring_summary.csv"
    final_df.to_csv(summary_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()

    elapsed_pyrosetta = time.time() - start_pyrosetta
    elapsed_total = time.time() - start_total
    print(f"\n=== Scoring Timing ===")
    print(f"MSA:       {elapsed_msa:.1f}s")
    print(f"Folding:   {elapsed_folding:.1f}s")
    print(f"PyRosetta: {elapsed_pyrosetta:.1f}s")
    print(f"Total:     {elapsed_total:.1f}s")
    print(f"======================\n")
    print(f"Saved final scoring summary to {summary_csv}")


def _add_sc_rmsd(original_pdb: str, folded_pdb: str, metrics: Dict):
    """Compute self-consistency RMSD and add it to the metrics dict."""
    try:
        from pipeline.scoring.rmsd import compute_binder_rmsd
        rmsd_val = compute_binder_rmsd(original_pdb, folded_pdb)
        metrics["sc_rmsd"] = round(rmsd_val, 3) if not np.isnan(rmsd_val) else None
    except Exception:
        metrics["sc_rmsd"] = None
