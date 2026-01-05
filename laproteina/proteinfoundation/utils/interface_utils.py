"""
Utility functions for analyzing protein interfaces.
"""

import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, Selection
from typing import Set
from loguru import logger


def get_target_interface_residues(
    pdb_path: str, 
    binder_chain: str = "B", 
    distance_cutoff: float = 4.0
) -> Set[int]:
    """
    Returns set of target chain (A) residue IDs that are within distance_cutoff of binder chain.
    
    Args:
        pdb_path: Path to PDB file containing target (chain A) and binder (binder_chain)
        binder_chain: Chain ID of the binder (default: "B")
        distance_cutoff: Distance cutoff in Angstroms (default: 4.0)
        
    Returns:
        Set of target chain residue IDs (integers) that are at the interface
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_path)
        
        # Get binder atoms
        binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
        if not binder_atoms:
            logger.warning(f"No atoms found for binder chain {binder_chain} in {pdb_path}")
            return set()
        
        binder_coords = np.array([atom.coord for atom in binder_atoms])
        
        # Get target atoms (chain A)
        target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
        if not target_atoms:
            logger.warning(f"No atoms found for target chain A in {pdb_path}")
            return set()
        
        target_coords = np.array([atom.coord for atom in target_atoms])
        
        # Build KD trees for both chains
        binder_tree = cKDTree(binder_coords)
        target_tree = cKDTree(target_coords)
        
        # Query the tree for pairs of atoms within the distance cutoff
        # This finds target atoms within cutoff of binder atoms
        pairs = binder_tree.query_ball_tree(target_tree, distance_cutoff)
        
        # Collect target residue IDs that are at the interface
        target_interface_residues = set()
        
        for binder_idx, close_indices in enumerate(pairs):
            if close_indices:  # If this binder atom has nearby target atoms
                for close_idx in close_indices:
                    target_residue = target_atoms[close_idx].get_parent()
                    target_residue_id = target_residue.id[1]  # Get residue number
                    target_interface_residues.add(target_residue_id)
        
        return target_interface_residues
        
    except Exception as e:
        logger.error(f"Error calculating target interface residues from {pdb_path}: {e}")
        return set()

