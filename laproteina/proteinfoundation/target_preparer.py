"""
Target Preparer for LaProteina V2 Pipeline

Prepares target proteins with linkers for binder generation.

Process:
1. Extract the hotspot region from the target (e.g., res 300-350)
2. Fold the linker onto the hotspot region using AF2
3. Extract the linker coordinates from the folded structure
4. Append linker to the FULL original target (numbered at end, positioned near hotspot)
5. LaProteina sees: full target + floating linker in correct 3D position
6. Post-processing: Remove linker from output for scoring
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Sequence
from loguru import logger

import biotite.structure as struc
import biotite.structure.io as strucio

from laproteina.proteinfoundation.linker_generator import LinkerConfig


@dataclass
class PreparedTarget:
    """Result of target preparation with linker."""
    
    pdb_path: str                    # Path to the prepared PDB file
    contig_string: str               # Contig string for LaProteina
    target_length: int               # Length of original target
    linker_length: int               # Length of linker added
    total_fixed_length: int          # target_length + linker_length
    linker_config: LinkerConfig      # Original linker config used
    metadata: dict                   # Additional metadata


class TargetPreparer:
    """
    Prepares target proteins with linkers for binder generation.
    
    The linker is attached at the C-terminus of the full protein,
    ensuring LaProteina sees the complete protein context to avoid
    generating binders that collide with other protein regions.
    
    Example:
        preparer = TargetPreparer(
            output_dir="./prepared_targets",
            linker_mode="polyglycine"
        )
        
        result = preparer.prepare_target_with_linker(
            target_pdb="my_target.pdb",
            linker_config=LinkerConfig(attachment_residue=74, linker_length=20, ...),
            binder_length_range=(40, 80)
        )
    """
    
    def __init__(
        self,
        output_dir: str = "./prepared_targets",
        linker_mode: str = "polyglycine",
        af_params_dir: Optional[str] = None
    ):
        """
        Initialize the target preparer.
        
        Args:
            output_dir: Directory to save prepared PDB files
            linker_mode: "polyglycine" for simple Gly linker, "af2" for AF2-folded linker
            af_params_dir: AlphaFold params directory (required if linker_mode="af2")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.linker_mode = linker_mode
        self.af_params_dir = af_params_dir
        
        if linker_mode == "af2" and not af_params_dir:
            # Try to find default AF2 params
            bindcraft_root = Path(__file__).parents[2]
            default_params = bindcraft_root / "params"
            if default_params.exists():
                self.af_params_dir = str(default_params)
            else:
                logger.warning("AF2 params not found. Set af_params_dir or use linker_mode='polyglycine'")
        
        logger.info(f"TargetPreparer initialized with mode={linker_mode}, output_dir={output_dir}")
    
    def prepare_target_with_linker(
        self,
        target_pdb: str,
        linker_config: LinkerConfig,
        binder_length_range: Tuple[int, int],
        chain_id: str = "A"
    ) -> PreparedTarget:
        """
        Prepare a target PDB with a linker for LaProteina generation.
        
        Process:
        1. Extract hotspot region from target (based on linker_config.hotspot_range)
        2. Fold linker onto hotspot region using AF2
        3. Extract linker coordinates from folded structure
        4. Append linker to FULL original target (numbered at end, positioned near hotspot)
        5. Result: full target + floating linker ready for LaProteina
        
        Args:
            target_pdb: Path to the original target PDB
            linker_config: Linker configuration specifying length, attachment, and hotspot info
            binder_length_range: (min, max) length for the binder to be generated
            chain_id: Chain ID of the target (default "A")
            
        Returns:
            PreparedTarget with paths and metadata
        """
        # Load the full target structure
        full_target = strucio.load_structure(target_pdb)
        
        # Filter for the target chain
        if chain_id:
            full_target = full_target[full_target.chain_id == chain_id]
        
        if len(full_target) == 0:
            raise ValueError(f"No atoms found for chain {chain_id} in {target_pdb}")
        
        # Get target length (number of residues)
        target_residue_ids = sorted(np.unique(full_target.res_id))
        target_length = len(target_residue_ids)
        
        # Get hotspot range from linker config
        hotspot_start, hotspot_end = linker_config.hotspot_range
        
        logger.info(f"Target {Path(target_pdb).name}: {target_length} residues, chain {chain_id}")
        logger.info(f"Hotspot region: {hotspot_start}-{hotspot_end}")
        logger.info(f"Linker config: attachment={linker_config.attachment_residue}, "
                   f"length={linker_config.linker_length}, edge={linker_config.edge_type}")
        
        # Step 1: Extract hotspot region
        hotspot_structure = self._extract_hotspot_region(
            full_target, hotspot_start, hotspot_end
        )
        
        # Step 2: Fold linker onto hotspot region using AF2
        folded_linker = self._fold_linker_on_hotspot(
            hotspot_structure,
            linker_config,
            target_pdb
        )
        
        # Step 3: Prepare the full target (renumber 1..N)
        prepared_target = full_target.copy()
        prepared_target.chain_id[:] = "A"
        
        # Renumber target residues 1..N
        res_map = {old: new for new, old in enumerate(target_residue_ids, 1)}
        for old, new in res_map.items():
            mask = prepared_target.res_id == old
            prepared_target.res_id[mask] = new + 100000  # Temp offset
        prepared_target.res_id = prepared_target.res_id - 100000
        
        # Step 4: Append linker to full target
        # Linker residues are numbered starting from target_length + 1
        linker_start_res = target_length + 1
        folded_linker.chain_id[:] = "A"
        
        # Renumber linker residues
        linker_res_ids = sorted(np.unique(folded_linker.res_id))
        for i, old_res in enumerate(linker_res_ids):
            mask = folded_linker.res_id == old_res
            folded_linker.res_id[mask] = linker_start_res + i
        
        # Combine: full target + floating linker
        combined_structure = prepared_target + folded_linker
        
        # Calculate total fixed length (target + linker)
        total_fixed_length = target_length + linker_config.linker_length
        
        # Build contig string
        binder_min, binder_max = binder_length_range
        contig_string = f"A1-{total_fixed_length}/{binder_min}-{binder_max}"
        
        # Generate output filename
        target_name = Path(target_pdb).stem
        output_filename = f"{target_name}_{linker_config.config_id}.pdb"
        output_path = self.output_dir / output_filename
        
        # Save prepared structure
        strucio.save_structure(str(output_path), combined_structure)
        
        logger.info(f"Prepared target saved: {output_path}")
        logger.info(f"  Full target: residues 1-{target_length}")
        logger.info(f"  Linker: residues {linker_start_res}-{total_fixed_length} (floating near hotspot)")
        logger.info(f"Contig string: {contig_string}")
        
        return PreparedTarget(
            pdb_path=str(output_path),
            contig_string=contig_string,
            target_length=target_length,
            linker_length=linker_config.linker_length,
            total_fixed_length=total_fixed_length,
            linker_config=linker_config,
            metadata={
                "original_target": target_pdb,
                "chain_id": chain_id,
                "binder_length_range": binder_length_range,
                "linker_mode": self.linker_mode,
                "hotspot_range": linker_config.hotspot_range,
                "attachment_residue": linker_config.attachment_residue
            }
        )
    
    def _extract_hotspot_region(
        self,
        structure: struc.AtomArray,
        hotspot_start: int,
        hotspot_end: int
    ) -> struc.AtomArray:
        """
        Extract the hotspot region from the structure.
        
        Returns a new structure containing only the hotspot residues,
        renumbered from 1.
        """
        # Get residues in the hotspot range
        res_ids = np.unique(structure.res_id)
        hotspot_res_ids = [r for r in res_ids if hotspot_start <= r <= hotspot_end]
        
        if not hotspot_res_ids:
            raise ValueError(f"No residues found in hotspot range {hotspot_start}-{hotspot_end}")
        
        # Extract hotspot atoms
        hotspot_mask = np.isin(structure.res_id, hotspot_res_ids)
        hotspot = structure[hotspot_mask].copy()
        
        # Renumber 1..N
        for new_id, old_id in enumerate(sorted(hotspot_res_ids), 1):
            hotspot.res_id[hotspot.res_id == old_id] = new_id + 100000
        hotspot.res_id = hotspot.res_id - 100000
        
        hotspot.chain_id[:] = "A"
        
        logger.info(f"Extracted hotspot region: {len(hotspot_res_ids)} residues")
        return hotspot
    
    def _fold_linker_on_hotspot(
        self,
        hotspot_structure: struc.AtomArray,
        linker_config: LinkerConfig,
        original_pdb: str
    ) -> struc.AtomArray:
        """
        Fold a linker onto the hotspot region using AF2.
        
        Returns the linker portion only (not the hotspot), positioned
        correctly in 3D space.
        """
        from laproteina.proteinfoundation.utils.linker_folder import fold_linked_pdb
        
        # Save hotspot to temp file
        temp_hotspot = self.output_dir / f"temp_hotspot_{linker_config.config_id}.pdb"
        strucio.save_structure(str(temp_hotspot), hotspot_structure)
        
        # Generate linker sequence (GSGSGS pattern for flexibility)
        linker_length = linker_config.linker_length
        linker_seq = "GS" * (linker_length // 2)
        if linker_length % 2 == 1:
            linker_seq += "G"
        
        # Fold linker with AF2
        temp_folded = self.output_dir / f"temp_folded_{linker_config.config_id}.pdb"
        
        try:
            logger.info(f"Folding {linker_length}-residue linker onto hotspot...")
            fold_linked_pdb(
                target_pdb=str(temp_hotspot),
                linker_sequence=linker_seq,
                output_pdb=str(temp_folded),
                af_params_dir=self.af_params_dir
            )
            
            # Load folded structure
            folded = strucio.load_structure(str(temp_folded))
            
            # Extract just the linker portion (last N residues)
            hotspot_len = len(np.unique(hotspot_structure.res_id))
            all_res_ids = sorted(np.unique(folded.res_id))
            linker_res_ids = all_res_ids[hotspot_len:]  # Linker is after hotspot
            
            if not linker_res_ids:
                raise ValueError("No linker residues found in folded structure")
            
            # Extract linker atoms
            linker_mask = np.isin(folded.res_id, linker_res_ids)
            linker = folded[linker_mask].copy()
            
            logger.info(f"Extracted {len(linker_res_ids)} linker residues from folded structure")
            return linker
            
        finally:
            # Cleanup temp files
            if temp_hotspot.exists():
                temp_hotspot.unlink()
            if temp_folded.exists():
                temp_folded.unlink()
    
    def prepare_batch(
        self,
        target_pdb: str,
        linker_configs: List[LinkerConfig],
        binder_length_range: Tuple[int, int],
        chain_id: str = "A"
    ) -> List[PreparedTarget]:
        """
        Prepare multiple target+linker combinations.
        
        Args:
            target_pdb: Path to the target PDB
            linker_configs: List of linker configurations
            binder_length_range: (min, max) binder length
            chain_id: Target chain ID
            
        Returns:
            List of PreparedTarget objects
        """
        results = []
        
        for i, config in enumerate(linker_configs):
            try:
                logger.info(f"Preparing config {i+1}/{len(linker_configs)}: {config}")
                result = self.prepare_target_with_linker(
                    target_pdb=target_pdb,
                    linker_config=config,
                    binder_length_range=binder_length_range,
                    chain_id=chain_id
                )
                results.append(result)
                
                # Clear GPU memory after each linker preparation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to prepare target with config {config}: {e}")
                continue
        
        logger.info(f"Prepared {len(results)}/{len(linker_configs)} target+linker combinations")
        return results


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
    output_path: str
) -> str:
    """
    Remove the linker from LaProteina output to get clean target + binder.
    
    LaProteina output has: target (1..target_len) + linker (target_len+1..target_len+linker_len) + binder
    We want: target (chain A) + binder (chain B)
    
    Args:
        pdb_path: Path to LaProteina output PDB
        target_length: Length of original target
        linker_length: Length of linker to remove
        output_path: Path to save cleaned PDB
        
    Returns:
        Path to cleaned PDB file
    """
    structure = strucio.load_structure(pdb_path)
    
    # Get all residue IDs
    all_res_ids = sorted(np.unique(structure.res_id))
    
    # Identify residue ranges
    linker_start = target_length + 1
    linker_end = target_length + linker_length
    
    # Target residues: 1 to target_length
    target_res_ids = [r for r in all_res_ids if r <= target_length]
    
    # Linker residues: target_length+1 to target_length+linker_length
    linker_res_ids = [r for r in all_res_ids if linker_start <= r <= linker_end]
    
    # Binder residues: everything after linker
    binder_res_ids = [r for r in all_res_ids if r > linker_end]
    
    if not binder_res_ids:
        logger.warning(f"No binder residues found in {pdb_path}")
        return pdb_path
    
    logger.info(f"Removing linker: target={len(target_res_ids)} res, "
               f"linker={len(linker_res_ids)} res (removed), "
               f"binder={len(binder_res_ids)} res")
    
    # Extract target and binder (skip linker)
    target_mask = np.isin(structure.res_id, target_res_ids)
    binder_mask = np.isin(structure.res_id, binder_res_ids)
    
    target_atoms = structure[target_mask].copy()
    binder_atoms = structure[binder_mask].copy()
    
    # Assign chains
    target_atoms.chain_id[:] = "A"
    binder_atoms.chain_id[:] = "B"
    
    # Renumber target 1..N
    for new_id, old_id in enumerate(sorted(target_res_ids), 1):
        target_atoms.res_id[target_atoms.res_id == old_id] = new_id + 100000
    target_atoms.res_id = target_atoms.res_id - 100000
    
    # Renumber binder 1..M
    for new_id, old_id in enumerate(sorted(binder_res_ids), 1):
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
    output_path: str
) -> str:
    """
    Create PyRosetta input: original starting target + extracted binder.
    
    This combines the ORIGINAL target PDB (not the one with linker) with
    the binder extracted from LaProteina output.
    
    Args:
        pdb_path: Path to LaProteina output PDB (target + linker + binder)
        original_target_pdb: Path to the original starting target PDB
        target_length: Length of original target
        linker_length: Length of linker to skip
        output_path: Path to save the PyRosetta input PDB
        
    Returns:
        Path to the saved PyRosetta input PDB
    """
    # Load LaProteina output
    laproteina_output = strucio.load_structure(pdb_path)
    
    # Load original target
    original_target = strucio.load_structure(original_target_pdb)
    original_target = original_target[original_target.chain_id == "A"]
    
    # Get all residue IDs from LaProteina output
    all_res_ids = sorted(np.unique(laproteina_output.res_id))
    
    # Identify binder residues (after target + linker)
    binder_start = target_length + linker_length + 1
    binder_res_ids = [r for r in all_res_ids if r >= binder_start]
    
    if not binder_res_ids:
        logger.warning(f"No binder residues found in {pdb_path}")
        return pdb_path
    
    # Extract binder atoms
    binder_mask = np.isin(laproteina_output.res_id, binder_res_ids)
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
    for new_id, old_id in enumerate(sorted(binder_res_ids), 1):
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
                output_path=str(output_path)
            )
            pyrosetta_paths.append(result)
        except Exception as e:
            logger.error(f"Failed to create PyRosetta input for {pdb_path}: {e}")
    
    logger.info(f"Created {len(pyrosetta_paths)}/{len(pdb_paths)} PyRosetta input files")
    return pyrosetta_paths

