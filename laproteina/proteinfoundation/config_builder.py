"""
Configuration builder utilities for LaProteina.

Translates simple BindCraft-style parameters into LaProteina motif configs.
"""

from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


def get_pdb_chain_length(pdb_path: str, chain_id: str = "A") -> int:
    """
    Get the number of residues in a specific chain of a PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain identifier to measure
        
    Returns:
        Number of residues in the chain
    """
    residue_ids = set()
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # PDB format: chain ID is column 22 (0-indexed 21)
                # Residue number is columns 23-26 (0-indexed 22:26)
                current_chain = line[21]
                if current_chain == chain_id:
                    try:
                        res_num = int(line[22:26].strip())
                        residue_ids.add(res_num)
                    except ValueError:
                        continue
    
    if not residue_ids:
        raise ValueError(f"No residues found for chain {chain_id} in {pdb_path}")
    
    # Return the count of unique residue IDs
    return len(residue_ids)


def build_contig_string(
    target_chain: str,
    target_start: int,
    target_end: int,
    binder_min_length: int,
    binder_max_length: int,
    linker_length: int = 0,
) -> str:
    """
    Build a LaProteina contig string from explicit parameters.
    
    Args:
        target_chain: Chain ID of target (e.g., "A")
        target_start: First residue number of target
        target_end: Last residue number of target
        binder_min_length: Minimum length of generated binder
        binder_max_length: Maximum length of generated binder
        
    Returns:
        Contig string like "A1-705/20-80"
    """
    motif_end = target_end + linker_length
    return f"{target_chain}{target_start}-{motif_end}/{binder_min_length}-{binder_max_length}"


def auto_build_motif_config(
    target_pdb: str,
    target_chain: str = "A",
    binder_max_length: int = 80,
    atom_selection_mode: str = "tip_atoms",
    linker_length: int = 0,
) -> dict:
    """
    Automatically build a motif config by parsing the target PDB.
    
    This determines the target length from the PDB file and constructs
    the appropriate contig string and length bounds.
    
    Args:
        target_pdb: Path to target PDB file
        target_chain: Chain ID of target in PDB
        binder_min_length: Minimum binder length
        binder_max_length: Maximum binder length
        atom_selection_mode: How to select atoms for motif conditioning
        
    Returns:
        Dictionary containing motif configuration
    """
    pdb_path = Path(target_pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Target PDB not found: {target_pdb}")
    
    # Get target chain length
    target_length = get_pdb_chain_length(str(pdb_path), target_chain)
    
    # Assume target is residues 1 to target_length
    # (This is a simplification - real PDBs may have different numbering)
    target_start = 1
    target_end = target_length
    
    # Build contig string
    contig_string = build_contig_string(
        target_chain, target_start, target_end,
        binder_min_length, binder_max_length,
        linker_length=linker_length
    )
    
    # Calculate total length bounds
    total_min = target_length + linker_length + binder_min_length
    total_max = target_length + linker_length + binder_max_length
    
    return {
        "contig_string": contig_string,
        "motif_pdb_path": str(pdb_path.absolute()),
        "motif_only": True,
        "motif_min_length": total_min,
        "motif_max_length": total_max,
        "segment_order": target_chain,
        "atom_selection_mode": atom_selection_mode,
        # Metadata for downstream processing
        "_target_length": target_length,
        "_linker_length": linker_length,
        "_binder_min_length": binder_min_length,
        "_binder_max_length": binder_max_length,
    }


def validate_motif_config(config: dict) -> bool:
    """
    Validate a motif configuration dictionary.
    
    Args:
        config: Motif config dict to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = [
        "contig_string", "motif_pdb_path", "motif_only",
        "motif_min_length", "motif_max_length", "segment_order",
        "atom_selection_mode"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    valid_modes = ["tip_atoms", "all_atom", "ca", "bb3o"]
    if config["atom_selection_mode"] not in valid_modes:
        raise ValueError(
            f"Invalid atom_selection_mode: {config['atom_selection_mode']}. "
            f"Must be one of: {valid_modes}"
        )
    
    if config["motif_min_length"] > config["motif_max_length"]:
        raise ValueError(
            f"motif_min_length ({config['motif_min_length']}) > "
            f"motif_max_length ({config['motif_max_length']})"
        )
    
    return True


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_builder.py <target_pdb>")
        sys.exit(1)
    
    target_pdb = sys.argv[1]
    config = auto_build_motif_config(
        target_pdb=target_pdb,
        binder_min_length=20,
        binder_max_length=80,
    )
    
    print("Generated motif config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
