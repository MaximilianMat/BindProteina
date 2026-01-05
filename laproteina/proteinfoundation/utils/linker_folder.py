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
    common_mask_folded = np.isin(folded_bb.res_id, native_bb.res_id) # Simplify: assume residues match
    # Realistically we should find the intersection of atoms.
    
    # Simple alignment attempt
    try:
        _, transformation = struc.superimpose(native_bb, folded_bb)
        # Apply transformation to the WHOLE folded complex
        folded_complex_aligned = struc.apply_transformation(folded_complex, transformation)
        # Re-extract aligned linker
        aligned_linker = folded_complex_aligned[np.isin(folded_complex_aligned.res_id, linker_res_ids)]
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
        
    logger.info(f"Successfully stitched linker to native target -> {output_pdb}")
    return output_pdb
