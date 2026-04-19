import os
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa

from functions.biopython_utils import three_to_one_map as three_to_one

def extract_sequence_from_pdb(pdb_path: str, chain_id: str) -> str:
    """
    Extracts the amino acid sequence from a specific chain in a PDB file.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    
    sequence = ""
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if is_aa(residue, standard=True):
                        resname = residue.get_resname()
                        if resname in three_to_one:
                            sequence += three_to_one[resname]
                return sequence
    return sequence

def check_chains_exist(pdb_path: str, required_chains: list) -> bool:
    """Checks if the required chains exist in the PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    existing_chains = set()
    for model in structure:
        for chain in model:
            existing_chains.add(chain.id)
    return all(c in existing_chains for c in required_chains)
