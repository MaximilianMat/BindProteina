"""
Utilities for generating animation sequences from evolution data.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import biotite.structure.io as strucio
import biotite.structure as struc


def interpolate_structures(
    structure1: struc.AtomArray,
    structure2: struc.AtomArray,
    num_frames: int = 30
) -> List[struc.AtomArray]:
    """
    Interpolate between two structures.
    
    Args:
        structure1: First structure
        structure2: Second structure
        num_frames: Number of interpolation frames
        
    Returns:
        List of interpolated structures
    """
    # Ensure structures have same number of atoms
    if len(structure1) != len(structure2):
        raise ValueError("Structures must have the same number of atoms")
    
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)  # 0 to 1
        interpolated_coords = (
            structure1.coord * (1 - alpha) + structure2.coord * alpha
        )
        
        # Create new structure with interpolated coordinates
        interpolated_structure = structure1.copy()
        interpolated_structure.coord = interpolated_coords
        
        frames.append(interpolated_structure)
    
    return frames


def align_structures(
    structure1: struc.AtomArray,
    structure2: struc.AtomArray,
    selection: Optional[str] = None
) -> Tuple[struc.AtomArray, struc.AtomArray]:
    """
    Align two structures using Kabsch algorithm.
    
    Args:
        structure1: Reference structure
        structure2: Structure to align
        selection: Atom selection (e.g., "CA" for alpha carbons)
        
    Returns:
        Tuple of (aligned_structure1, aligned_structure2)
    """
    # Use CA atoms for alignment if selection not specified
    if selection is None:
        mask1 = structure1.atom_name == "CA"
        mask2 = structure2.atom_name == "CA"
    else:
        mask1 = structure1.atom_name == selection
        mask2 = structure2.atom_name == selection
    
    coords1 = structure1[mask1].coord
    coords2 = structure2[mask2].coord
    
    if len(coords1) != len(coords2):
        raise ValueError("Structures must have same number of selected atoms")
    
    # Center coordinates
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    
    centered1 = coords1 - center1
    centered2 = coords2 - center2
    
    # Compute covariance matrix
    covariance = np.dot(centered1.T, centered2)
    
    # SVD
    U, S, Vt = np.linalg.svd(covariance)
    
    # Rotation matrix
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    rotation = Vt.T @ np.diag([1, 1, d]) @ U.T
    
    # Translation vector
    translation = center1 - np.dot(rotation, center2)
    
    # Apply transformation to full structure2
    aligned_structure2 = structure2.copy()
    aligned_structure2.coord = (
        np.dot(structure2.coord, rotation.T) + translation
    )
    
    return structure1, aligned_structure2


def create_evolution_animation_frames(
    pdb_paths: List[Path],
    align: bool = True
) -> List[struc.AtomArray]:
    """
    Create animation frames from a sequence of PDB files.
    
    Args:
        pdb_paths: List of PDB file paths
        align: Whether to align structures before interpolation
        
    Returns:
        List of structures for animation
    """
    if len(pdb_paths) < 2:
        raise ValueError("Need at least 2 structures for animation")
    
    structures = []
    for pdb_path in pdb_paths:
        structure = strucio.load_structure(str(pdb_path))
        structures.append(structure)
    
    # Align structures if requested
    if align:
        reference = structures[0]
        aligned_structures = [reference]
        
        for structure in structures[1:]:
            try:
                _, aligned = align_structures(reference, structure)
                aligned_structures.append(aligned)
            except Exception as e:
                print(f"Warning: Could not align structure {structure}: {e}")
                aligned_structures.append(structure)
        
        structures = aligned_structures
    
    # Interpolate between consecutive structures
    all_frames = []
    for i in range(len(structures) - 1):
        frames = interpolate_structures(structures[i], structures[i + 1])
        all_frames.extend(frames)
    
    return all_frames

