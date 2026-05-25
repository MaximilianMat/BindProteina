"""
Hotspot residue parsing, anchor selection, and surface-normal computation.
"""

from __future__ import annotations

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io
from io import StringIO
from typing import List, Tuple


def parse_hotspot_string(hotspot_str: str) -> List[int]:
    """
    Parse a hotspot residue string like "50-107,150,200-205" into a sorted list of ints.
    """
    residues = []
    for part in hotspot_str.replace(" ", "").split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            residues.extend(range(int(start), int(end) + 1))
        elif part:
            residues.append(int(part))
    return sorted(set(residues))


def get_ca_coords(structure: struc.AtomArray, residue_ids: List[int]) -> np.ndarray:
    """
    Return (N, 3) array of CA coordinates for the given residue IDs.
    Raises if any residue is missing.
    """
    coords = []
    missing = []
    for res_id in residue_ids:
        mask = (structure.res_id == res_id) & (structure.atom_name == "CA")
        if not mask.any():
            missing.append(res_id)
            continue
        coords.append(structure[mask].coord[0])
    if missing:
        raise ValueError(f"Residues not found in structure: {missing}")
    return np.array(coords)


def get_anchor_ca(structure: struc.AtomArray, residue_id: int) -> np.ndarray:
    """Return CA coordinate of a single anchor residue."""
    mask = (structure.res_id == residue_id) & (structure.atom_name == "CA")
    if not mask.any():
        raise ValueError(f"Residue {residue_id} CA not found in structure")
    return structure[mask].coord[0].copy()


def hotspot_centroid(structure: struc.AtomArray, residue_ids: List[int]) -> np.ndarray:
    """Return the centroid (mean CA position) of hotspot residues."""
    coords = get_ca_coords(structure, residue_ids)
    return coords.mean(axis=0)


def compute_surface_normal(
    structure: struc.AtomArray,
    anchor_pos: np.ndarray,
    radius: float = 8.0,
    outward: bool = True,
) -> np.ndarray:
    """
    Estimate a surface normal at anchor_pos by fitting a plane to nearby CA atoms.

    The returned vector points away from the overall CA centroid (outward=True)
    or toward it (outward=False).
    """
    ca_mask = structure.atom_name == "CA"
    ca_coords = structure[ca_mask].coord

    # Select CAs within radius of anchor
    dists = np.linalg.norm(ca_coords - anchor_pos, axis=1)
    nearby = ca_coords[dists <= radius]

    if len(nearby) < 3:
        # Fallback: use global centroid direction
        global_centroid = ca_coords.mean(axis=0)
        v = anchor_pos - global_centroid
        return _normalize(v if outward else -v)

    # SVD to find plane normal (smallest singular value = normal direction)
    centered = nearby - nearby.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # last row = direction of least variance = normal

    # Flip so it points away from global centroid
    global_centroid = ca_coords.mean(axis=0)
    centroid_to_anchor = anchor_pos - global_centroid
    if np.dot(normal, centroid_to_anchor) < 0:
        normal = -normal

    return _normalize(normal if outward else -normal)


def direction_from_preset(
    preset: str,
    structure: struc.AtomArray,
    anchor_pos: np.ndarray,
) -> np.ndarray:
    """
    Resolve a direction preset string to a unit vector.

    Presets: "+x", "-x", "+y", "-y", "+z", "-z",
             "surface-normal-outward", "surface-normal-inward"
    """
    presets = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    if preset in presets:
        return presets[preset]
    if preset == "surface-normal-outward":
        return compute_surface_normal(structure, anchor_pos, outward=True)
    if preset == "surface-normal-inward":
        return compute_surface_normal(structure, anchor_pos, outward=False)
    raise ValueError(f"Unknown direction preset: {preset!r}")


def parse_custom_direction(xyz_str: str) -> np.ndarray:
    """Parse a user-entered 'x,y,z' string into a unit vector."""
    parts = [float(v.strip()) for v in xyz_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Direction must be three comma-separated numbers, e.g. '1,0,0'")
    v = np.array(parts, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        raise ValueError("Direction vector must not be zero")
    return v / norm


def residue_ids_in_chain(structure: struc.AtomArray, chain_id: str = "A") -> List[int]:
    """Return sorted unique residue IDs for a given chain."""
    chain_mask = structure.chain_id == chain_id
    return sorted(np.unique(structure[chain_mask].res_id).tolist())


def load_structure_from_path(pdb_path: str, chain_id: str = "A") -> struc.AtomArray:
    """Load a PDB file and return the first model of the given chain."""
    stack = pdb_io.PDBFile.read(pdb_path).get_structure(model=1)
    return stack[stack.chain_id == chain_id]


def load_structure_from_string(pdb_str: str, chain_id: str | None = "A") -> struc.AtomArray:
    """Load a PDB from a string, optionally filtering to one chain."""
    stack = pdb_io.PDBFile.read(StringIO(pdb_str)).get_structure(model=1)
    if chain_id is not None:
        stack = stack[stack.chain_id == chain_id]
    return stack


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a zero vector")
    return v / n
