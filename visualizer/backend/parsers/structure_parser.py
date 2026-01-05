"""
Parser for PDB structure files.
Handles reading and preprocessing PDB files for visualization.
"""

import biotite.structure.io as strucio
import biotite.structure as struc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class StructureParser:
    """Parser for PDB structure files."""
    
    @staticmethod
    def load_structure(pdb_path: str) -> struc.AtomArray:
        """Load a PDB structure."""
        return strucio.load_structure(pdb_path)
    
    @staticmethod
    def get_chain_info(structure: struc.AtomArray) -> Dict[str, List[int]]:
        """Get residue ranges for each chain."""
        chains = {}
        unique_chains = np.unique(structure.chain_id)
        
        for chain_id in unique_chains:
            chain_mask = structure.chain_id == chain_id
            chain_structure = structure[chain_mask]
            if len(chain_structure) > 0:
                res_ids = np.unique(chain_structure.res_id)
                chains[chain_id] = {
                    "start": int(np.min(res_ids)),
                    "end": int(np.max(res_ids)),
                    "num_residues": len(res_ids)
                }
        
        return chains
    
    @staticmethod
    def extract_chain(structure: struc.AtomArray, chain_id: str) -> struc.AtomArray:
        """Extract a specific chain from structure."""
        chain_mask = structure.chain_id == chain_id
        return structure[chain_mask]
    
    @staticmethod
    def get_sequence(structure: struc.AtomArray, chain_id: Optional[str] = None) -> str:
        """Extract sequence from structure."""
        if chain_id:
            structure = StructureParser.extract_chain(structure, chain_id)
        
        # Get CA atoms for sequence
        ca_mask = structure.atom_name == "CA"
        ca_atoms = structure[ca_mask]
        
        if len(ca_atoms) == 0:
            return ""
        
        # Map residue names to single letter codes
        three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        sequence = []
        for res_name in ca_atoms.res_name:
            seq_char = three_to_one.get(res_name, 'X')
            sequence.append(seq_char)
        
        return ''.join(sequence)
    
    @staticmethod
    def get_plddt_values(structure: struc.AtomArray) -> Dict[int, float]:
        """Extract pLDDT values from B-factor column."""
        plddt_values = {}
        
        if "b_factor" in structure.get_annotation_categories():
            # Get CA atoms for per-residue pLDDT
            ca_mask = structure.atom_name == "CA"
            ca_atoms = structure[ca_mask]
            
            for res_id, b_factor in zip(ca_atoms.res_id, ca_atoms.b_factor):
                if res_id not in plddt_values:
                    plddt_values[int(res_id)] = float(b_factor)
        
        return plddt_values
    
    @staticmethod
    def get_structure_info(pdb_path: str) -> Dict:
        """Get comprehensive information about a structure."""
        structure = StructureParser.load_structure(pdb_path)
        
        chains = StructureParser.get_chain_info(structure)
        
        info = {
            "path": str(pdb_path),
            "num_atoms": len(structure),
            "chains": chains,
            "sequences": {}
        }
        
        # Get sequences for each chain
        for chain_id in chains.keys():
            info["sequences"][chain_id] = StructureParser.get_sequence(structure, chain_id)
        
        # Get pLDDT if available
        plddt = StructureParser.get_plddt_values(structure)
        if plddt:
            info["plddt"] = plddt
            info["avg_plddt"] = float(np.mean(list(plddt.values())))
        
        return info

