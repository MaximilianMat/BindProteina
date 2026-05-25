"""
Seed placement engine.

Takes a target structure, an anchor position, a direction vector, and a seed
fragment, then returns one or more linked PDB strings ready for LaProteina.
"""

from __future__ import annotations

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

from .seeds import get_seed_structure, get_seed_pdb


# ---------------------------------------------------------------------------
# Core placement
# ---------------------------------------------------------------------------

def place_seed(
    target: struc.AtomArray,
    anchor_pos: np.ndarray,
    direction: np.ndarray,
    distance: float,
    seed_type: str,
    rotation_deg: float = 0.0,
    custom_pdb: Optional[str] = None,
) -> Tuple[struc.AtomArray, int]:
    """
    Place a seed fragment so its first CA ends up at anchor_pos + distance * direction.

    Args:
        target:       Target AtomArray (chain A, already filtered).
        anchor_pos:   3D position of the chosen anchor (CA of hotspot res or centroid).
        direction:    Unit vector indicating placement direction.
        distance:     How far from anchor_pos to place the seed's first CA (Å).
        seed_type:    Key into the seed library.
        rotation_deg: Rotation around the anchor→placement axis (only matters for
                      multi-residue seeds, changes orientation of the helix/strand).
        custom_pdb:   PDB string for 'custom' seed type.

    Returns:
        (linked_structure, seed_n_residues): linked AtomArray (target + seed, all chain A)
        and the number of seed residues.
    """
    direction = _normalize(direction)
    placement_point = anchor_pos + distance * direction

    seed_struct = get_seed_structure(seed_type, custom_pdb)

    # Translate: first CA → placement_point
    first_ca_mask = seed_struct.atom_name == "CA"
    if not first_ca_mask.any():
        raise ValueError("Seed structure has no CA atoms")
    first_ca = seed_struct[first_ca_mask].coord[0]
    seed_struct = seed_struct.copy()
    seed_struct.coord += (placement_point - first_ca)

    # Rotate around the anchor→placement axis if requested
    if abs(rotation_deg) > 1e-3:
        axis = direction  # the direction from anchor to placement IS the rotation axis
        seed_struct.coord = _rotate_points(
            seed_struct.coord, axis, np.radians(rotation_deg), pivot=placement_point
        )

    # Renumber seed: target occupies 1..target_length, seed follows
    target_res_ids = sorted(np.unique(target.res_id).tolist())
    target_length = len(target_res_ids)

    seed_res_ids = sorted(np.unique(seed_struct.res_id).tolist())
    n_seed_residues = len(seed_res_ids)

    # Renumber target 1..N (in case original numbering has gaps)
    target_out = target.copy()
    for new_id, old_id in enumerate(target_res_ids, 1):
        target_out.res_id[target_out.res_id == old_id] = new_id + 100_000
    target_out.res_id -= 100_000

    # Renumber seed target_length+1 .. target_length+n_seed
    seed_out = seed_struct.copy()
    for new_id, old_id in enumerate(seed_res_ids, target_length + 1):
        seed_out.res_id[seed_out.res_id == old_id] = new_id + 100_000
    seed_out.res_id -= 100_000

    # Force everything to the same chain
    chain_id = target.chain_id[0] if len(target.chain_id) > 0 else "A"
    target_out.chain_id[:] = chain_id
    seed_out.chain_id[:] = chain_id

    linked = target_out + seed_out
    return linked, n_seed_residues


def generate_candidates(
    target: struc.AtomArray,
    anchor_pos: np.ndarray,
    direction: np.ndarray,
    distance: float,
    seed_type: str,
    n_candidates: int,
    custom_pdb: Optional[str] = None,
) -> List[Tuple[struc.AtomArray, int, float]]:
    """
    Generate N linked structures, each rotated evenly around the direction axis.

    Returns list of (linked_structure, n_seed_residues, rotation_deg_used).
    """
    angles = np.linspace(0, 360, n_candidates, endpoint=False)
    results = []
    for angle in angles:
        linked, n_seed = place_seed(
            target, anchor_pos, direction, distance,
            seed_type, rotation_deg=angle, custom_pdb=custom_pdb,
        )
        results.append((linked, n_seed, float(angle)))
    return results


# ---------------------------------------------------------------------------
# PDB I/O helpers
# ---------------------------------------------------------------------------

def structure_to_pdb_string(structure: struc.AtomArray) -> str:
    """Serialise an AtomArray to a PDB-format string."""
    f = StringIO()
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, structure)
    pdb_file.write(f)
    return f.getvalue()


def save_linked_pdb(structure: struc.AtomArray, output_path: str) -> str:
    """Write linked structure to disk, creating parent dirs if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdb_file = pdb_io.PDBFile()
    pdb_io.set_structure(pdb_file, structure)
    pdb_file.write(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalise a zero vector")
    return v / n


def _rotate_points(
    points: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
    pivot: np.ndarray,
) -> np.ndarray:
    """Rodrigues rotation of (N,3) points around axis through pivot."""
    k = _normalize(axis)
    pts = points - pivot
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    k_dot_v = (pts * k).sum(axis=1, keepdims=True)
    rotated = pts * cos_a + np.cross(k, pts) * sin_a + k * k_dot_v * (1.0 - cos_a)
    return rotated + pivot
