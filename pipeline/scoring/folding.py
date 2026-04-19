import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def run_af2_refolding(
    pdb_path: str,
    design_name: str,
    output_dir: Path,
    complex_model,
    af2_model_num: int = 1,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Run AF2 re-folding natively (without relaxation) to extract pLDDT, pTM, i_pTM etc.
    Returns: (path_to_af2_pdb, metrics_dict)
    """
    from pipeline.scoring.utils import extract_sequence_from_pdb
    from colabdesign.shared.utils import copy_dict
    
    binder_seq = extract_sequence_from_pdb(str(pdb_path), "B")
    if not binder_seq:
        return None, {"error": "No binder sequence found (Chain B missing?)"}
        
    binder_seq = "".join([c for c in binder_seq.upper() if "A" <= c <= "Z"])
    binder_len = len(binder_seq)
    
    if binder_len == 0:
        return None, {"error": "Empty binder sequence"}
        
    complex_model.prep_inputs(
        pdb_filename=str(pdb_path),
        chain="A",
        binder_len=binder_len,
        rm_target_seq=True,
        rm_target_sc=True,
    )
    
    complex_model.predict(
        seq=binder_seq,
        models=[af2_model_num - 1],
        num_recycles=1,
        verbose=False,
    )
    
    af2_pdb_path = output_dir / f"{design_name}_model{af2_model_num}.pdb"
    complex_model.save_pdb(str(af2_pdb_path))
    
    metrics = copy_dict(complex_model.aux["log"])
    residue_plddt = complex_model.aux.get("plddt")
    binder_plddt = None
    if residue_plddt is not None:
        residue_plddt = np.asarray(residue_plddt).reshape(-1)
        if residue_plddt.size >= binder_len:
            binder_plddt = round(float(np.mean(residue_plddt[-binder_len:])), 3)
            
    return str(af2_pdb_path), {
        "sequence": binder_seq,
        "binder_length": binder_len,
        "af2_model": af2_model_num,
        "pLDDT": round(metrics["plddt"], 3),
        "binder_pLDDT": binder_plddt,
        "pTM": round(metrics["ptm"], 3),
        "i_pTM": round(metrics["i_ptm"], 3),
        "pAE": round(metrics["pae"], 3),
        "i_pAE": round(metrics["i_pae"], 3),
    }
