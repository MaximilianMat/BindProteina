"""
Protenix folding adapter.

Runs Protenix structure prediction via its CLI (`protenix pred`),
then converts the CIF output to PDB and extracts confidence metrics
so that the downstream scoring pipeline sees the same interface as AF2.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger


def _build_protenix_input_json(
    target_seq: str, binder_seq: str, job_name: str, out_path: Path, target_msa_path: Optional[str] = None
) -> str:
    """
    Build a Protenix input JSON with two protein chains.
    If target_msa_path is provided, it is linked to the target chain.
    """
    target_entry = {"proteinChain": {"count": 1, "sequence": target_seq, "modifications": []}}
    if target_msa_path and os.path.exists(target_msa_path):
        # Protenix v2 uses these specifically
        target_entry["proteinChain"]["unpairedMsaPath"] = str(Path(target_msa_path).absolute())
        # We can also add an empty pairedMsaPath if needed, but omitting it is usually fine

    payload = [
        {
            "name": job_name,
            "covalent_bonds": [],
            "sequences": [
                target_entry,
                {"proteinChain": {"count": 1, "sequence": binder_seq, "modifications": []}},
            ],
        }
    ]
    json_path = out_path / f"{job_name}_input.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(json_path)


def _cif_to_pdb(cif_path: str, pdb_path: str) -> str:
    """Convert a CIF file to PDB format using biotite."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc

    pdbx_file = pdbx.CIFFile.read(cif_path)
    # Take the first block
    block = list(pdbx_file.values())[0]
    atoms = pdbx.get_structure(block, model=1)

    # Filter to standard amino acid atoms only
    atoms = atoms[struc.filter_amino_acids(atoms)]

    pdb_file = pdb_io.PDBFile()
    pdb_file.set_structure(atoms)
    pdb_file.write(pdb_path)
    return pdb_path


def _parse_protenix_confidence(json_path: str) -> Dict[str, Any]:
    """
    Parse the Protenix summary_confidence JSON and return metrics
    in a format compatible with the AF2 metrics dict.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Protenix outputs: plddt, ptm, iptm, ranking_score, chain_plddt, etc.
    metrics = {
        "pLDDT": round(float(data.get("plddt", 0.0)), 3),
        "pTM": round(float(data.get("ptm", 0.0)), 3),
        "i_pTM": round(float(data.get("iptm", 0.0)), 3),
        "ranking_score": round(float(data.get("ranking_score", 0.0)), 3),
        "has_clash": data.get("has_clash", None),
    }

    # Extract per-chain pLDDT for binder (chain index 1 = entity 2)
    chain_plddt = data.get("chain_plddt")
    if chain_plddt and len(chain_plddt) >= 2:
        metrics["binder_pLDDT"] = round(float(chain_plddt[1]), 3)
    else:
        metrics["binder_pLDDT"] = None

    # Extract per-chain pTM
    chain_ptm = data.get("chain_ptm")
    if chain_ptm and len(chain_ptm) >= 2:
        metrics["binder_pTM"] = round(float(chain_ptm[1]), 3)

    # ipTM between chains (target-binder)
    chain_pair_iptm = data.get("chain_pair_iptm")
    if chain_pair_iptm and len(chain_pair_iptm) >= 2 and len(chain_pair_iptm[0]) >= 2:
        metrics["chain_pair_i_pTM"] = round(float(chain_pair_iptm[0][1]), 3)

    return metrics


def run_protenix_refolding(
    pdb_path: str,
    design_name: str,
    output_dir: Path,
    model_name: str = "protenix-v2",
    seed: int = 42,
    target_msa_path: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Run Protenix structure prediction on a complex PDB.

    1. Extract target (chain A) and binder (chain B) sequences from the input PDB.
    2. Build a Protenix input JSON.
    3. Run `protenix pred` via subprocess.
    4. Convert the CIF output to PDB.
    5. Parse confidence metrics.

    Returns: (path_to_protenix_pdb, metrics_dict)
    """
    from pipeline.scoring.utils import extract_sequence_from_pdb

    target_seq = extract_sequence_from_pdb(pdb_path, "A")
    binder_seq = extract_sequence_from_pdb(pdb_path, "B")

    if not binder_seq:
        return None, {"error": "No binder sequence found (Chain B missing?)"}
    if not target_seq:
        return None, {"error": "No target sequence found (Chain A missing?)"}

    binder_seq = "".join([c for c in binder_seq.upper() if "A" <= c <= "Z"])
    target_seq = "".join([c for c in target_seq.upper() if "A" <= c <= "Z"])
    binder_len = len(binder_seq)

    if binder_len == 0:
        return None, {"error": "Empty binder sequence"}

    # Create a working directory for this design
    work_dir = output_dir / f"{design_name}_protenix"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build input JSON
    input_json = _build_protenix_input_json(
        target_seq=target_seq,
        binder_seq=binder_seq,
        job_name=design_name,
        out_path=work_dir,
        target_msa_path=target_msa_path,
    )

    import sys

    # Resolve protenix command.
    # We prefer the 'protenix_env' if it exists to avoid NumPy version conflicts.
    protenix_cmd = "protenix"
    try:
        conda_base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        protenix_bin = os.path.join(conda_base, "envs", "protenix_env", "bin", "protenix")
        if os.path.exists(protenix_bin):
            protenix_cmd = protenix_bin
    except Exception:
        pass

    # Run protenix pred
    # We only disable msa/template if a target_msa is NOT provided. 
    # If a target_msa is provided, we let protenix use the linked file.
    use_msa_flag = "false" if not target_msa_path else "true"

    cmd = [
        protenix_cmd, "pred",
        "-i", input_json,
        "-o", str(work_dir),
        "-n", model_name,
        "--seeds", str(seed),
        "--use_msa", use_msa_flag,
        "--use_template", "false",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per design
        )
        if result.returncode != 0:
            return None, {"error": f"Protenix failed: {result.stderr[:500]}"}
    except subprocess.TimeoutExpired:
        return None, {"error": "Protenix prediction timed out (600s)"}
    except FileNotFoundError:
        return None, {"error": "protenix CLI not found. Is protenix installed?"}

    # 4. Locate the best output CIF and confidence JSON
    conf_files = list(work_dir.rglob("*_summary_confidence_sample_*.json"))
    
    best_cif = None
    best_scores = {"plddt": 0.0, "ptm": 0.0, "iptm": 0.0}
    best_data = {}  # Initialize in case no conf files found
    
    if conf_files:
        # Load all confidence scores
        sample_confidences = []
        for cf in conf_files:
            try:
                with open(cf, 'r') as f:
                    data = json.load(f)
                    # Higher is better
                    score = data.get("ranking_score", data.get("iptm", 0.0))
                    sample_confidences.append((cf, score, data))
            except Exception:
                continue
        
        if sample_confidences:
            # Pick the best sample based on ranking_score
            sample_confidences.sort(key=lambda x: x[1], reverse=True)
            best_conf_file, _, best_data = sample_confidences[0]
            
            # Map back to CIF: {name}_summary_confidence_sample_{N}.json -> {name}_sample_{N}.cif
            sample_suffix = best_conf_file.name.replace("_summary_confidence_", "_").replace(".json", ".cif")
            potential_cif = best_conf_file.parent / sample_suffix
            if potential_cif.exists():
                best_cif = potential_cif
            
            best_scores = {
                "plddt": best_data.get("plddt", 0.0),
                "ptm": best_data.get("ptm", 0.0),
                "iptm": best_data.get("iptm", 0.0)
            }
            
    if not best_cif:
        # Fallback to any cif if JSON failed
        cif_files = list(work_dir.rglob("*.cif"))
        if not cif_files:
            return None, {"error": f"No Protenix output (.cif) found in {work_dir}"}
        best_cif = cif_files[0]

    # Convert CIF to PDB for visual check/other tools
    protenix_pdb_path = str(work_dir / f"{design_name}_protenix.pdb")
    try:
        _cif_to_pdb(str(best_cif), protenix_pdb_path)
    except Exception as e:
        # Not fatal, but worth noting
        pass

    # Build the metrics dictionary expected by scorer.py
    # Standardize keys to match AF2 output in scorer.py
    plddt_val = best_scores["plddt"]
    # Ensure plddt is on 0-100 scale (protenix global plddt is already, but just in case)
    if plddt_val < 1.0:
        plddt_val *= 100.0
        
    # Extract binder pLDDT if available (usually chain 1 in Protenix if A=target, B=binder)
    binder_plddt = plddt_val # fallback
    if "chain_plddt" in best_data and len(best_data["chain_plddt"]) > 1:
        # protenix chain_plddt seems to be 0-1 scale in JSON? 
        # let's check: [0.92, 0.74] -> yes.
        binder_plddt = best_data["chain_plddt"][1] * 100.0

    metrics = {
        "pLDDT": plddt_val,
        "binder_pLDDT": binder_plddt,
        "pTM": best_data.get("ptm", 0.0),
        "i_pTM": best_data.get("iptm", 0.0),
        "pAE": None,   # For compatibility
        "i_pAE": None, # For compatibility
        "sequence": binder_seq,
        "binder_length": binder_len,
        "folding_model": model_name,
    }

    return protenix_pdb_path, metrics


def run_protenix_refolding_batch(
    pdb_paths: List[str],
    output_dir: Path,
    model_name: str = "protenix-v2",
    seed: int = 42,
    target_msa_path: Optional[str] = None,
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """
    Run Protenix structure prediction on multiple complex PDBs in a single batch.
    Returns: List of (design_name, path_to_protenix_pdb, metrics_dict)
    """
    from pipeline.scoring.utils import extract_sequence_from_pdb

    payload = []
    job_info = {}

    work_dir = output_dir / "protenix_batch"
    work_dir.mkdir(parents=True, exist_ok=True)

    for pdb_path in pdb_paths:
        design_name = Path(pdb_path).stem
        target_seq = extract_sequence_from_pdb(pdb_path, "A")
        binder_seq = extract_sequence_from_pdb(pdb_path, "B")

        if not binder_seq:
            job_info[design_name] = {"error": "No binder sequence found (Chain B missing?)", "seq": ""}
            continue
        if not target_seq:
            job_info[design_name] = {"error": "No target sequence found (Chain A missing?)", "seq": ""}
            continue

        binder_seq = "".join([c for c in binder_seq.upper() if "A" <= c <= "Z"])
        target_seq = "".join([c for c in target_seq.upper() if "A" <= c <= "Z"])
        binder_len = len(binder_seq)

        if binder_len == 0:
            job_info[design_name] = {"error": "Empty binder sequence", "seq": ""}
            continue

        target_entry = {"proteinChain": {"count": 1, "sequence": target_seq, "modifications": []}}
        if target_msa_path and os.path.exists(target_msa_path):
            target_entry["proteinChain"]["unpairedMsaPath"] = str(Path(target_msa_path).absolute())

        payload.append({
            "name": design_name,
            "covalent_bonds": [],
            "sequences": [
                target_entry,
                {"proteinChain": {"count": 1, "sequence": binder_seq, "modifications": []}},
            ],
        })
        job_info[design_name] = {
            "binder_seq": binder_seq,
            "binder_len": binder_len
        }

    results = []
    if not payload:
        for p in pdb_paths:
            dnn = Path(p).stem
            err = job_info.get(dnn, {}).get("error", "Failed precondition")
            results.append((dnn, None, {"error": err}))
        return results

    json_path = work_dir / "batch_input.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    protenix_cmd = "protenix"
    try:
        conda_base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        protenix_bin = os.path.join(conda_base, "envs", "protenix_env", "bin", "protenix")
        if os.path.exists(protenix_bin):
            protenix_cmd = protenix_bin
    except Exception:
        pass

    use_msa_flag = "false" if not target_msa_path else "true"
    cmd = [
        protenix_cmd, "pred",
        "-i", str(json_path),
        "-o", str(work_dir),
        "-n", model_name,
        "--seeds", str(seed),
        "--use_msa", use_msa_flag,
        "--use_template", "false",
    ]

    import time
    from tqdm import tqdm
    
    try:
        timeout_val = len(payload) * 600
        logger.info(f"Running Protenix Batch with {len(payload)} designs")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        completed = set()
        pbar = tqdm(total=len(payload), desc="Protenix Batch Prediction")
        
        start_time = time.time()
        retcode = None
        
        while True:
            retcode = process.poll()
            
            current_cifs = list(work_dir.rglob("*.cif"))
            
            new_completed = set()
            for p in payload:
                design_name = p["name"]
                if design_name in completed:
                    new_completed.add(design_name)
                    continue
                    
                has_output = any(c.name.startswith(f"{design_name}_") or c.parent.name == design_name for c in current_cifs)
                if has_output:
                    new_completed.add(design_name)
            
            newly_done = len(new_completed) - len(completed)
            if newly_done > 0:
                pbar.update(newly_done)
                completed = new_completed
            
            if retcode is not None:
                break
                
            if time.time() - start_time > timeout_val:
                process.terminate()
                pbar.close()
                err = f"Protenix batch timed out ({timeout_val}s)"
                for p in payload:
                    results.append((p["name"], None, {"error": err}))
                return results

            time.sleep(2.0)

        remaining_update = len(payload) - len(completed)
        if remaining_update > 0 and retcode == 0:
            pbar.update(remaining_update)
            
        pbar.close()
        stdout, stderr = process.communicate()
        
        if retcode != 0:
            logger.error(f"Protenix batch process exited with code {retcode}. Stderr: {stderr[:500]}")

    except FileNotFoundError:
        err = "protenix CLI not found"
        for p in payload:
            results.append((p["name"], None, {"error": err}))
        return results
    except Exception as e:
        err = f"Protenix batch exception: {str(e)}"
        for p in payload:
            results.append((p["name"], None, {"error": err}))
        return results

    # Parse output for each design
    for p in payload:
        design_name = p["name"]
        jinfo = job_info[design_name]
        
        # Try both the prefix pattern and exact subdirectory if it made one
        subset_files = [
            cf for cf in work_dir.rglob("*_summary_confidence_sample_*.json") 
            if cf.name.startswith(f"{design_name}_summary_confidence") or cf.parent.name == design_name
        ]
        
        best_cif = None
        best_scores = {"plddt": 0.0, "ptm": 0.0, "iptm": 0.0}
        best_data = {}
        
        if subset_files:
            sample_confidences = []
            for cf in subset_files:
                try:
                    with open(cf, 'r') as f:
                        data = json.load(f)
                        score = data.get("ranking_score", data.get("iptm", 0.0))
                        sample_confidences.append((cf, score, data))
                except Exception:
                    continue
            
            if sample_confidences:
                sample_confidences.sort(key=lambda x: x[1], reverse=True)
                best_conf_file, _, best_data = sample_confidences[0]
                
                sample_suffix = best_conf_file.name.replace("_summary_confidence_", "_").replace(".json", ".cif")
                potential_cif = best_conf_file.parent / sample_suffix
                if potential_cif.exists():
                    best_cif = potential_cif
                
        if not best_cif:
            cifs = [c for c in work_dir.rglob("*.cif") if c.name.startswith(f"{design_name}_") or c.parent.name == design_name]
            if cifs:
                best_cif = cifs[0]
                
        if not best_cif:
            results.append((design_name, None, {"error": "No output cif found for this design"}))
            continue
            
        # Optional: Save a standard pdb next to it
        protenix_pdb_path = str(work_dir / f"{design_name}_protenix.pdb")
        try:
            _cif_to_pdb(str(best_cif), protenix_pdb_path)
        except Exception:
            pass

        plddt_val = best_scores["plddt"]
        if plddt_val < 1.0:
            plddt_val *= 100.0
            
        binder_plddt = plddt_val
        if "chain_plddt" in best_data and len(best_data["chain_plddt"]) > 1:
            binder_plddt = best_data["chain_plddt"][1] * 100.0

        metrics = {
            "pLDDT": plddt_val,
            "binder_pLDDT": binder_plddt,
            "pTM": best_data.get("ptm", 0.0),
            "i_pTM": best_data.get("iptm", 0.0),
            "pAE": None,
            "i_pAE": None,
            "sequence": jinfo["binder_seq"],
            "binder_length": jinfo["binder_len"],
            "folding_model": model_name,
        }
        results.append((design_name, protenix_pdb_path, metrics))

    return results


def auto_generate_msa(
    pdb_path: str,
    output_dir: Path,
    db_path: str,
) -> str:
    """
    Generate a target MSA using 'protenix prep' once.
    Returns the path to the generated .a3m file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract target sequence
    try:
        from pipeline.scoring.utils import extract_sequence_from_pdb
        target_seq = extract_sequence_from_pdb(pdb_path, "A")
    except Exception:
        # Fallback
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("tmp", pdb_path)
        residues = [r for r in struct.get_residues() if r.get_parent().id == "A"]
        from Bio.SeqUtils import seq1
        target_seq = "".join([seq1(r.get_resname()) for r in residues])

    # 2. Build minimal prep input
    prep_input = [
        {
            "name": "target_ref",
            "sequences": [
                {"proteinChain": {"count": 1, "sequence": target_seq}}
            ]
        }
    ]
    json_path = output_dir / "prep_input.json"
    with open(json_path, "w") as f:
        json.dump(prep_input, f, indent=2)

    # 3. Resolve protenix command
    protenix_cmd = "protenix"
    try:
        conda_base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        protenix_bin = os.path.join(conda_base, "envs", "protenix_env", "bin", "protenix")
        if os.path.exists(protenix_bin):
            protenix_cmd = protenix_bin
    except Exception:
        pass

    # 4. Run protenix prep
    cmd = [
        protenix_cmd, "prep",
        "-i", str(json_path),
        "-o", str(output_dir),
        "--database_dir", db_path
    ]
    
    logger.info(f"Running Protenix MSA prep: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Protenix prep failed: {result.stderr}")

    # 5. Extract the generated .a3m path
    msa_path = output_dir / "target_ref" / "msa" / "0" / "non_pairing.a3m"
    if not msa_path.exists():
        candidates = list(output_dir.rglob("*.a3m"))
        if candidates:
            msa_path = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find generated MSA (.a3m) in {output_dir}")

    return str(msa_path.absolute())
