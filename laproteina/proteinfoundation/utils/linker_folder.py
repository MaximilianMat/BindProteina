import os
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger

def fold_linked_pdb(
    target_pdb: str,
    linker_sequence: str,
    output_pdb: str,
    af_params_dir: Optional[str] = None
) -> str:
    """
    Use AlphaFold2 (via ColabDesign) to fold a linker onto a target PDB,
    then seamlessly "stitch" it to the original native target coordinates.
    """
    try:
        from colabdesign import mk_afdesign_model, clear_mem
        import biotite.structure.io as strucio
        import biotite.structure as struc
        from openfold.np.residue_constants import restype_3to1
    except ImportError as e:
        logger.error(f"Failed to import ColabDesign or dependencies: {e}")
        raise

    # 1. Load native target and extract sequence
    native_target = strucio.load_structure(target_pdb)
    # Filter for CA atoms to get sequence reliably
    target_ca = native_target[native_target.atom_name == "CA"]
    if len(target_ca) == 0:
        target_ca = native_target
        
    target_seq = "".join([restype_3to1.get(res.res_name, "A") for res in target_ca])
    full_sequence = target_seq + linker_sequence
    
    logger.info(f"Folding + Stitching linker (total_len={len(full_sequence)}) onto native {Path(target_pdb).name}")

    # 2. Run AF2 Folding with Template
    clear_mem()
    
    if not af_params_dir:
        bindcraft_root = Path(__file__).parents[3]
        af_params_dir = str(bindcraft_root / "params")
    
    if not os.path.exists(af_params_dir):
        af_params_dir = "./params"

    af_model = mk_afdesign_model(protocol="hallucination", use_templates=True, data_dir=af_params_dir)
    af_model.prep_inputs(length=len(full_sequence), pdb_filename=target_pdb, chain="A")
    af_model.predict(seq=full_sequence, num_recycles=12, verbose=False)
    
    af2_pdb = output_pdb + ".af2.pdb"
    af_model.save_pdb(af2_pdb)
    
    # 3. Seamless Stitching Logic
    # Load AF2 folded complex
    folded_complex = strucio.load_structure(af2_pdb)
    
    # Identify target part in folded_complex (first N residues)
    target_len = len(target_seq)
    # Mask for residues 1 to target_len in folded_complex
    # Residue IDs in folded_complex (hallucination) might be 1..N+L
    unique_res_ids = np.unique(folded_complex.res_id)
    target_res_ids = unique_res_ids[:target_len]
    linker_res_ids = unique_res_ids[target_len:]
    
    folded_target = folded_complex[np.isin(folded_complex.res_id, target_res_ids)]
    folded_linker = folded_complex[np.isin(folded_complex.res_id, linker_res_ids)]
    
    # Align folded_target to native_target using backbone atoms
    # We need matching subsets. 
    native_bb = native_target[(native_target.atom_name == "N") | (native_target.atom_name == "CA") | (native_target.atom_name == "C")]
    folded_bb = folded_target[(folded_target.atom_name == "N") | (folded_target.atom_name == "CA") | (folded_target.atom_name == "C")]
    
    # Ensure equal atom counts for alignment (might differ if templates had missing atoms)
    # Biotite superimpose needs same number of atoms
    # Find matching residues between native and folded target
    native_res_ids = np.unique(native_bb.res_id)
    folded_res_ids = np.unique(folded_bb.res_id)
    common_res_ids = np.intersect1d(native_res_ids, folded_res_ids)
    
    if len(common_res_ids) == 0:
        logger.warning("No common residues found for alignment. Using unaligned linker.")
        aligned_linker = folded_linker
    else:
        # Filter to common residues only
        native_bb_common = native_bb[np.isin(native_bb.res_id, common_res_ids)]
        folded_bb_common = folded_bb[np.isin(folded_bb.res_id, common_res_ids)]
        
        # Ensure same number of atoms (same residues, same atom types)
        # Sort by residue ID and atom name to ensure matching order
        native_bb_common = native_bb_common[np.lexsort((native_bb_common.atom_name, native_bb_common.res_id))]
        folded_bb_common = folded_bb_common[np.lexsort((folded_bb_common.atom_name, folded_bb_common.res_id))]
        
        # Check if we have matching atom counts
        if len(native_bb_common) != len(folded_bb_common):
            logger.warning(f"Atom count mismatch ({len(native_bb_common)} vs {len(folded_bb_common)}). Using unaligned linker.")
            aligned_linker = folded_linker
        else:
            try:
                # Get coordinates
                native_coords = native_bb_common.coord
                folded_coords = folded_bb_common.coord
                
                # Calculate transformation matrix using Kabsch algorithm
                # Center coordinates
                native_center = np.mean(native_coords, axis=0)
                folded_center = np.mean(folded_coords, axis=0)
                native_centered = native_coords - native_center
                folded_centered = folded_coords - folded_center
                
                # Calculate covariance matrix
                H = folded_centered.T @ native_centered
                
                # SVD
                U, S, Vt = np.linalg.svd(H)
                
                # Rotation matrix
                R = Vt.T @ U.T
                
                # Ensure proper rotation (det(R) = 1)
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                
                # Translation vector
                t = native_center - R @ folded_center
                
                # Apply transformation to entire folded complex
                folded_complex_coords = folded_complex.coord
                transformed_coords = (folded_complex_coords @ R.T) + t
                
                # Create aligned structure
                folded_complex_aligned = folded_complex.copy()
                folded_complex_aligned.coord = transformed_coords
                
                # Re-extract aligned linker
                aligned_linker = folded_complex_aligned[np.isin(folded_complex_aligned.res_id, linker_res_ids)]
                
                logger.info(f"Successfully aligned linker using {len(common_res_ids)} common residues")
            except Exception as e:
                logger.warning(f"Structural alignment failed: {e}. Falling back to unaligned prediction.")
                aligned_linker = folded_linker

    # 4. Construct Hybrid PDB
    # Re-number aligned linker to follow native target
    last_res_id = np.max(native_target.res_id)
    aligned_linker.res_id[:] = aligned_linker.res_id - np.min(aligned_linker.res_id) + last_res_id + 1
    
    # Force single chain
    native_target.chain_id[:] = "A"
    aligned_linker.chain_id[:] = "A"
    
    # Merge structures
    stitched_complex = native_target + aligned_linker
    
    # Ensure everything is 1..N indexed for LaProteina
    final_res_ids = np.unique(stitched_complex.res_id)
    res_map = {old: new for new, old in enumerate(sorted(final_res_ids), 1)}
    for old, new in res_map.items():
        stitched_complex.res_id[stitched_complex.res_id == old] = new
        
    strucio.save_structure(output_pdb, stitched_complex)
    
    # Cleanup
    if os.path.exists(af2_pdb):
        os.remove(af2_pdb)
    
    # Clear AF2 model from GPU memory
    del af_model
    clear_mem()
    
    # Also clear PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU memory after linker folding")
    except ImportError:
        pass
        
    logger.info(f"Successfully stitched linker to native target -> {output_pdb}")
    return output_pdb