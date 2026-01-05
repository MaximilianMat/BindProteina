"""
PDB preprocessing utilities for web visualization.
"""

import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
from pathlib import Path
from typing import Dict, List, Optional


def preprocess_pdb_for_web(pdb_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Preprocess a PDB file for web viewing.
    Removes unnecessary data and ensures proper formatting.
    
    Args:
        pdb_path: Input PDB file path
        output_path: Optional output path (if None, returns PDB string)
        
    Returns:
        PDB file content as string
    """
    structure = strucio.load_structure(str(pdb_path))
    
    # Write to string
    pdb_string = strucio.save_structure(structure, format="pdb")
    
    if output_path:
        output_path.write_text(pdb_string)
    
    return pdb_string


def extract_interface_residues(
    structure: struc.AtomArray,
    chain1: str = "A",
    chain2: str = "B",
    distance_cutoff: float = 8.0
) -> Dict[str, List[int]]:
    """
    Extract interface residues between two chains.
    
    Args:
        structure: Protein structure
        chain1: First chain ID
        chain2: Second chain ID
        distance_cutoff: Distance cutoff in Angstroms
        
    Returns:
        Dictionary with chain IDs as keys and lists of residue IDs as values
    """
    chain1_mask = structure.chain_id == chain1
    chain2_mask = structure.chain_id == chain2
    
    chain1_structure = structure[chain1_mask]
    chain2_structure = structure[chain2_mask]
    
    # Get CA atoms
    chain1_ca = chain1_structure[chain1_structure.atom_name == "CA"]
    chain2_ca = chain2_structure[chain2_structure.atom_name == "CA"]
    
    # Compute distances
    interface_res1 = set()
    interface_res2 = set()
    
    for ca1 in chain1_ca:
        for ca2 in chain2_ca:
            distance = np.linalg.norm(ca1.coord - ca2.coord)
            if distance < distance_cutoff:
                interface_res1.add(int(ca1.res_id))
                interface_res2.add(int(ca2.res_id))
    
    return {
        chain1: sorted(list(interface_res1)),
        chain2: sorted(list(interface_res2))
    }

