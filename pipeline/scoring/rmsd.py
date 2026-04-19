import numpy as np

def compute_binder_rmsd(ref_pdb: str, mobile_pdb: str, target_chain: str = "A", binder_chain: str = "B") -> float:
    """
    Computes RMSD of the binder (chain B) after superimposing the target (chain A) CA atoms.
    """
    try:
        from Bio import PDB
        parser = PDB.PDBParser(QUIET=True)
        ref_struct = parser.get_structure("ref", ref_pdb)
        mob_struct = parser.get_structure("mob", mobile_pdb)
        
        # Get target CA atoms
        ref_target_ca = [atom for atom in ref_struct.get_atoms() if atom.get_parent().get_parent().id == target_chain and atom.name == "CA"]
        mob_target_ca = [atom for atom in mob_struct.get_atoms() if atom.get_parent().get_parent().id == target_chain and atom.name == "CA"]
        
        # Superimpose Target
        min_len = min(len(ref_target_ca), len(mob_target_ca))
        if min_len == 0:
            return float('nan')
            
        sup = PDB.Superimposer()
        sup.set_atoms(ref_target_ca[:min_len], mob_target_ca[:min_len])
        sup.apply(mob_struct.get_atoms())
        
        # Get Binder CA atoms
        ref_binder_ca = [atom for atom in ref_struct.get_atoms() if atom.get_parent().get_parent().id == binder_chain and atom.name == "CA"]
        mob_binder_ca = [atom for atom in mob_struct.get_atoms() if atom.get_parent().get_parent().id == binder_chain and atom.name == "CA"]
        
        # Compute RMSD on Binder CA atoms
        min_binder_len = min(len(ref_binder_ca), len(mob_binder_ca))
        if min_binder_len == 0:
            return float('nan')
            
        ref_coords = np.array([atom.coord for atom in ref_binder_ca[:min_binder_len]])
        mob_coords = np.array([atom.coord for atom in mob_binder_ca[:min_binder_len]])
        
        diff = ref_coords - mob_coords
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        return float(rmsd)
    except Exception as e:
        print(f"RMSD calculation failed: {e}")
        return float('nan')
