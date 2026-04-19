"""
Target Preparer for LaProteina V2 Pipeline

Prepares target proteins with linkers for binder generation.

Process:
1. Extract a LARGE focused region around the hotspot (max context for robust alignment)
2. Fold the linker onto the focused region using AF2
3. Align the AF2 result back to the full target coordinates
4. If alignment RMSD > threshold, fall back to direct coordinate positioning
5. Append positioned linker to the FULL original target
6. LaProteina sees: full target + floating linker in correct 3D position
7. Post-processing: Remove linker from output for scoring
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Sequence, Dict
from loguru import logger

import biotite.structure as struc
import biotite.structure.io as strucio


# RMSD threshold (Å). If alignment RMSD exceeds this, switch to direct
# coordinate positioning instead of alignment-based placement.
DEFAULT_RMSD_THRESHOLD = 3.0


# TargetPreparer class and PreparedTarget removed as synthetic linker generation is deprecated.


def get_target_length(pdb_path: str, chain_id: str = "A") -> int:
    """
    Utility function to get the number of residues in a PDB chain.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to analyze
        
    Returns:
        Number of residues in the chain
    """
    structure = strucio.load_structure(pdb_path)
    if chain_id:
        structure = structure[structure.chain_id == chain_id]
    
    unique_res_ids = np.unique(structure.res_id)
    return len(unique_res_ids)


def remove_linker_from_output(
    pdb_path: str,
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_path: str
) -> str:
    """
    Remove the linker from LaProteina output to get clean target + binder.
    
    LaProteina output has: target (1..target_len) + linker (target_len+1..target_len+linker_len) + binder.
    We output: target (chain A) + (kept linker tail + binder) (chain B).
    
    Args:
        pdb_path: Path to LaProteina output PDB
        target_length: Length of original target
        linker_length: Length of linker to remove
        keep_linker_end_n: Number of residues to keep from linker C-terminus as
            binder seed (0 means remove full linker)
        output_path: Path to save cleaned PDB
        
    Returns:
        Path to cleaned PDB file
    """
    structure = strucio.load_structure(pdb_path)

    if keep_linker_end_n < 0:
        raise ValueError(f"keep_linker_end_n must be >= 0, got {keep_linker_end_n}")
    if keep_linker_end_n > linker_length:
        raise ValueError(
            f"keep_linker_end_n ({keep_linker_end_n}) cannot exceed linker_length ({linker_length})"
        )
    
    # Get all residue IDs
    all_res_ids = sorted(np.unique(structure.res_id))
    
    # Identify residue ranges
    linker_start = target_length + 1
    linker_end = target_length + linker_length
    
    # Target residues: 1 to target_length
    target_res_ids = [r for r in all_res_ids if r <= target_length]
    
    # Linker residues: target_length+1 to target_length+linker_length
    linker_res_ids = [r for r in all_res_ids if linker_start <= r <= linker_end]

    # Keep only the final N linker residues as binder seed (if requested)
    linker_keep_ids = linker_res_ids[-keep_linker_end_n:] if keep_linker_end_n > 0 else []
    linker_removed_ids = linker_res_ids[:-keep_linker_end_n] if keep_linker_end_n > 0 else linker_res_ids
    
    # Binder residues: everything after linker
    binder_res_ids = [r for r in all_res_ids if r > linker_end]
    
    if not binder_res_ids:
        logger.warning(f"No binder residues found in {pdb_path}")
        return pdb_path
    
    logger.info(
        f"Removing linker: target={len(target_res_ids)} res, "
        f"linker_removed={len(linker_removed_ids)} res, "
        f"linker_kept={len(linker_keep_ids)} res, "
        f"binder={len(binder_res_ids)} res"
    )
    
    # Extract target and chain-B residues (kept linker tail + binder)
    chain_b_res_ids = sorted(linker_keep_ids + binder_res_ids)
    target_mask = np.isin(structure.res_id, target_res_ids)
    binder_mask = np.isin(structure.res_id, chain_b_res_ids)
    
    target_atoms = structure[target_mask].copy()
    binder_atoms = structure[binder_mask].copy()
    
    # Assign chains
    target_atoms.chain_id[:] = "A"
    binder_atoms.chain_id[:] = "B"
    
    # Renumber target 1..N
    for new_id, old_id in enumerate(sorted(target_res_ids), 1):
        target_atoms.res_id[target_atoms.res_id == old_id] = new_id + 100000
    target_atoms.res_id = target_atoms.res_id - 100000
    
    # Renumber chain B 1..M (kept linker tail first, then binder)
    for new_id, old_id in enumerate(chain_b_res_ids, 1):
        binder_atoms.res_id[binder_atoms.res_id == old_id] = new_id + 100000
    binder_atoms.res_id = binder_atoms.res_id - 100000
    
    # Combine
    cleaned = target_atoms + binder_atoms
    
    # Save
    strucio.save_structure(output_path, cleaned)
    
    logger.info(f"Saved cleaned structure (no linker): {output_path}")
    return output_path


def batch_remove_linker(
    pdb_paths: List[str],
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_dir: str
) -> List[str]:
    """
    Remove linker from multiple LaProteina output files.
    
    Args:
        pdb_paths: List of LaProteina output PDB paths
        target_length: Length of original target
        linker_length: Length of linker to remove
        output_dir: Directory to save cleaned PDBs
        
    Returns:
        List of paths to cleaned PDB files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_paths = []
    for pdb_path in pdb_paths:
        try:
            output_path = output_dir / Path(pdb_path).name
            cleaned = remove_linker_from_output(
                pdb_path=pdb_path,
                target_length=target_length,
                linker_length=linker_length,
                keep_linker_end_n=keep_linker_end_n,
                output_path=str(output_path)
            )
            cleaned_paths.append(cleaned)
        except Exception as e:
            logger.error(f"Failed to remove linker from {pdb_path}: {e}")
    
    logger.info(f"Removed linker from {len(cleaned_paths)}/{len(pdb_paths)} files")
    return cleaned_paths


def create_pyrosetta_input(
    pdb_path: str,
    original_target_pdb: str,
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_path: str
) -> str:
    """
    Create PyRosetta input: original starting target + extracted binder.
    
    This combines the target core with chain B built from
    (kept linker tail + generated binder) extracted from LaProteina output.
    
    Args:
        pdb_path: Path to LaProteina output PDB (target + linker + binder)
        original_target_pdb: Path to the original starting target PDB
        target_length: Length of original target
        linker_length: Length of linker to skip
        keep_linker_end_n: Number of linker residues to keep from C-terminus
            as binder seed
        output_path: Path to save the PyRosetta input PDB
        
    Returns:
        Path to the saved PyRosetta input PDB
    """
    # Load LaProteina output
    laproteina_output = strucio.load_structure(pdb_path)

    if keep_linker_end_n < 0:
        raise ValueError(f"keep_linker_end_n must be >= 0, got {keep_linker_end_n}")
    if keep_linker_end_n > linker_length:
        raise ValueError(
            f"keep_linker_end_n ({keep_linker_end_n}) cannot exceed linker_length ({linker_length})"
        )
    
    # Load original target
    original_target = strucio.load_structure(original_target_pdb)
    original_target = original_target[original_target.chain_id == "A"]
    target_core_ids = [r for r in sorted(np.unique(original_target.res_id)) if r <= target_length]
    if not target_core_ids:
        logger.warning(
            f"No target residues <= {target_length} found in {original_target_pdb}; "
            "falling back to first target_length residues by order."
        )
        ordered_ids = sorted(np.unique(original_target.res_id))
        target_core_ids = ordered_ids[:target_length]
    target_core_mask = np.isin(original_target.res_id, target_core_ids)
    original_target = original_target[target_core_mask]
    
    # Get all residue IDs from LaProteina output
    all_res_ids = sorted(np.unique(laproteina_output.res_id))
    
    # Identify binder residues (after target + linker)
    linker_start = target_length + 1
    linker_end = target_length + linker_length
    linker_res_ids = [r for r in all_res_ids if linker_start <= r <= linker_end]
    linker_keep_ids = linker_res_ids[-keep_linker_end_n:] if keep_linker_end_n > 0 else []
    binder_start = target_length + linker_length + 1
    binder_res_ids = [r for r in all_res_ids if r >= binder_start]
    
    if not binder_res_ids:
        logger.warning(f"No binder residues found in {pdb_path}")
        return pdb_path
    
    # Extract kept linker tail + binder atoms
    chain_b_res_ids = sorted(linker_keep_ids + binder_res_ids)
    binder_mask = np.isin(laproteina_output.res_id, chain_b_res_ids)
    binder_atoms = laproteina_output[binder_mask].copy()
    
    # Prepare original target (chain A, renumber 1..N)
    target_atoms = original_target.copy()
    target_atoms.chain_id[:] = "A"
    orig_res_ids = sorted(np.unique(target_atoms.res_id))
    for new_id, old_id in enumerate(orig_res_ids, 1):
        target_atoms.res_id[target_atoms.res_id == old_id] = new_id + 100000
    target_atoms.res_id = target_atoms.res_id - 100000
    
    # Prepare binder (chain B, renumber 1..M)
    binder_atoms.chain_id[:] = "B"
    for new_id, old_id in enumerate(chain_b_res_ids, 1):
        binder_atoms.res_id[binder_atoms.res_id == old_id] = new_id + 100000
    binder_atoms.res_id = binder_atoms.res_id - 100000
    
    # Combine
    combined = target_atoms + binder_atoms
    
    # Save
    strucio.save_structure(output_path, combined)
    
    logger.debug(f"Created PyRosetta input: {output_path}")
    return output_path


def batch_create_pyrosetta_input(
    pdb_paths: List[str],
    original_target_pdb: str,
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_dir: str
) -> List[str]:
    """
    Create PyRosetta input files for multiple LaProteina outputs.
    
    Each output contains: original starting target (chain A) + extracted binder (chain B)
    
    Args:
        pdb_paths: List of LaProteina output PDB paths
        original_target_pdb: Path to the original starting target PDB
        target_length: Length of original target
        linker_length: Length of linker to skip
        output_dir: Directory to save PyRosetta input PDBs
        
    Returns:
        List of paths to PyRosetta input PDB files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pyrosetta_paths = []
    for pdb_path in pdb_paths:
        try:
            output_path = output_dir / Path(pdb_path).name
            result = create_pyrosetta_input(
                pdb_path=pdb_path,
                original_target_pdb=original_target_pdb,
                target_length=target_length,
                linker_length=linker_length,
                keep_linker_end_n=keep_linker_end_n,
                output_path=str(output_path)
            )
            pyrosetta_paths.append(result)
        except Exception as e:
            logger.error(f"Failed to create PyRosetta input for {pdb_path}: {e}")
    
    logger.info(f"Created {len(pyrosetta_paths)}/{len(pdb_paths)} PyRosetta input files")
    return pyrosetta_paths
