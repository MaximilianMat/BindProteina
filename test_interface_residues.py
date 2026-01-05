#!/usr/bin/env python3
"""
Quick test script for target interface residue extraction.
Tests the PyRosetta-based interface residue detection without full scoring pipeline.
"""

import sys
import os

# Add BindCraft root to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import pyrosetta as pr
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, NeighborhoodResidueSelector
from pyrosetta.rosetta.core.select import get_residues_from_subset

def get_target_interface_residues(pdb_path: str, binder_chain: str = "B", distance_cutoff: float = 8.0):
    """
    Extract target chain (A) interface residues using PyRosetta.
    
    Args:
        pdb_path: Path to PDB file with target (A) and binder (B) chains
        binder_chain: Chain ID of binder (default: "B")
        distance_cutoff: Distance cutoff in Angstroms (default: 8.0)
        
    Returns:
        List of target chain residue IDs (integers)
    """
    # Load pose
    pose = pr.pose_from_pdb(pdb_path)
    pdb_info = pose.pdb_info()
    target_chain_id = 'A'
    binder_chain_id = binder_chain
    
    # Get residue indices for binder chain
    binder_residues = []
    for res_num in range(1, pose.size() + 1):
        if pdb_info.chain(res_num) == binder_chain_id:
            binder_residues.append(res_num)
    
    if not binder_residues:
        print(f"Warning: No residues found for binder chain {binder_chain_id}")
        return []
    
    # Create selector for binder chain
    binder_selector = ChainSelector(binder_chain_id)
    
    # Use NeighborhoodResidueSelector to find target residues near binder
    neighbor_selector = NeighborhoodResidueSelector()
    neighbor_selector.set_distance(distance_cutoff)
    neighbor_selector.set_focus_selector(binder_selector)
    
    # Apply selector to get interface residues
    interface_subset = neighbor_selector.apply(pose)
    interface_residue_indices = get_residues_from_subset(interface_subset)
    
    # Filter for target chain (A) residues only
    target_interface_residues = []
    for res_num in interface_residue_indices:
        chain_id = pdb_info.chain(res_num)
        if chain_id == target_chain_id:
            pdb_res_num = pdb_info.number(res_num)
            target_interface_residues.append(pdb_res_num)
    
    return sorted(set(target_interface_residues))

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_interface_residues.py <pdb_file> [binder_chain] [distance_cutoff]")
        print("Example: python test_interface_residues.py complex.pdb B 8.0")
        sys.exit(1)
    
    pdb_path = sys.argv[1]
    binder_chain = sys.argv[2] if len(sys.argv) > 2 else "B"
    distance_cutoff = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file not found: {pdb_path}")
        sys.exit(1)
    
    print(f"Testing interface residue extraction:")
    print(f"  PDB: {pdb_path}")
    print(f"  Binder chain: {binder_chain}")
    print(f"  Distance cutoff: {distance_cutoff} Å")
    print()
    
    # Initialize PyRosetta (this is the slow part, but only happens once)
    print("Initializing PyRosetta...")
    pr.init()
    print("PyRosetta initialized.")
    print()
    
    # Extract interface residues
    print("Extracting target interface residues...")
    try:
        target_residues = get_target_interface_residues(pdb_path, binder_chain, distance_cutoff)
        
        print(f"Found {len(target_residues)} target interface residues:")
        if target_residues:
            # Print in groups of 10 for readability
            for i in range(0, len(target_residues), 10):
                chunk = target_residues[i:i+10]
                print(f"  {', '.join(map(str, chunk))}")
            
            # Also print as comma-separated string (CSV format)
            print()
            print("CSV format:")
            print(','.join(map(str, target_residues)))
        else:
            print("  (none found)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

