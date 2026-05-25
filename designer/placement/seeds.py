"""
Canonical seed fragment library.

Fragments are generated programmatically using ideal backbone geometry
(Engh & Huber parameters + NeRF placement). Each fragment starts with
its first CA at the origin so the placer can freely translate/rotate it.
"""

from __future__ import annotations

import numpy as np
from io import StringIO
from typing import Dict, List, Optional, Tuple

import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io


# ---------------------------------------------------------------------------
# Ideal geometry constants (Engh & Huber 1991/2001)
# ---------------------------------------------------------------------------

_BL_N_CA = 1.459   # N-CA bond length (Å)
_BL_CA_C = 1.525   # CA-C
_BL_C_N  = 1.336   # C-N (peptide)
_BL_C_O  = 1.229   # C=O

_BA_N_CA_C = np.radians(111.2)   # bond angle at CA
_BA_CA_C_N = np.radians(116.2)   # bond angle at C (toward next N)
_BA_C_N_CA = np.radians(121.7)   # bond angle at N
_BA_CA_C_O = np.radians(120.8)   # carbonyl angle

_OMEGA = np.radians(180.0)       # trans peptide

# Phi / psi for canonical secondary structures
_HELIX_PHI = np.radians(-57.8)
_HELIX_PSI = np.radians(-47.0)
_STRAND_PHI = np.radians(-119.0)
_STRAND_PSI = np.radians(113.0)


SEED_TYPES = ["single_gly", "helix_3", "helix_5", "strand_3", "custom"]

SEED_LABELS = {
    "single_gly": "Single Gly (1 residue)",
    "helix_3":    "Alpha helix — 3 residues",
    "helix_5":    "Alpha helix — 5 residues",
    "strand_3":   "Beta strand — 3 residues",
    "custom":     "Custom (paste PDB)",
}


# ---------------------------------------------------------------------------
# NeRF backbone builder
# ---------------------------------------------------------------------------

def _nerf(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
    d_len: float, bond_angle: float, dihedral: float,
) -> np.ndarray:
    """
    Natural Extension Reference Frame.
    Given atoms A, B, C returns D such that:
      |CD| = d_len, angle(B,C,D) = bond_angle, dihedral(A,B,C,D) = dihedral.
    """
    bc = c - b   # forward direction (b → c); b - c gave π-angle instead of angle
    bc_norm = np.linalg.norm(bc)
    if bc_norm < 1e-8:
        raise ValueError("B and C are degenerate (distance < 1e-8)")
    bc_hat = bc / bc_norm

    n_vec = np.cross(a - b, bc)
    n_norm = np.linalg.norm(n_vec)
    if n_norm < 1e-8:
        # Collinear fallback: pick arbitrary perpendicular
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(bc_hat, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        n_vec = np.cross(bc_hat, tmp)
    n_hat = n_vec / np.linalg.norm(n_vec)

    m_hat = np.cross(n_hat, bc_hat)

    d_local = np.array([
        -d_len * np.cos(bond_angle),
        d_len * np.sin(bond_angle) * np.cos(dihedral),
        d_len * np.sin(bond_angle) * np.sin(dihedral),
    ])

    return c + bc_hat * d_local[0] + m_hat * d_local[1] + n_hat * d_local[2]


def _build_backbone(n_residues: int, phi: float, psi: float) -> List[Dict[str, np.ndarray]]:
    """
    Build ideal backbone for n_residues with given phi/psi (radians).

    Returns list of dicts: {"N": ..., "CA": ..., "C": ..., "O": ...}
    The first CA is placed at the origin after generation.
    """
    # Seed the first residue in a canonical orientation
    N0  = np.array([0.000,  0.000, 0.000])
    CA0 = np.array([_BL_N_CA, 0.000, 0.000])
    # C: angle N-CA-C = 111.2° in the XY plane
    ang = np.pi - _BA_N_CA_C          # angle from +x at CA
    C0 = CA0 + _BL_CA_C * np.array([np.cos(ang), np.sin(ang), 0.0])
    # O: rough carbonyl placement (in XY plane)
    O0 = C0 + _BL_C_O * np.array([np.cos(ang + np.pi * 0.6), np.sin(ang + np.pi * 0.6), 0.0])

    residues = [{"N": N0, "CA": CA0, "C": C0, "O": O0}]

    for i in range(1, n_residues):
        prev = residues[-1]
        prev2 = residues[-2] if i >= 2 else None

        # N_{i} placed by psi of residue i-1: dihedral N_{i-1}-CA_{i-1}-C_{i-1}-N_{i}
        if prev2 is not None:
            N_new = _nerf(prev2["N"], prev["CA"], prev["C"],
                          _BL_C_N, _BA_CA_C_N, psi)
        else:
            N_new = _nerf(prev["N"], prev["CA"], prev["C"],
                          _BL_C_N, _BA_CA_C_N, psi)

        # CA_{i}: dihedral CA_{i-1}-C_{i-1}-N_{i}-CA_{i} = omega
        CA_new = _nerf(prev["CA"], prev["C"], N_new,
                       _BL_N_CA, _BA_C_N_CA, _OMEGA)

        # C_{i}: dihedral C_{i-1}-N_{i}-CA_{i}-C_{i} = phi
        C_new = _nerf(prev["C"], N_new, CA_new,
                      _BL_CA_C, _BA_N_CA_C, phi)

        # O_{i}: approximate carbonyl placement
        O_new = _nerf(N_new, CA_new, C_new,
                      _BL_C_O, _BA_CA_C_O, psi + np.pi)

        residues.append({"N": N_new, "CA": CA_new, "C": C_new, "O": O_new})

    # Translate so first CA is at origin
    offset = residues[0]["CA"].copy()
    for res in residues:
        for atom in res:
            res[atom] = res[atom] - offset

    return residues


def _backbone_to_pdb_string(
    residues: List[Dict[str, np.ndarray]],
    chain: str = "A",
    res_name: str = "GLY",
    start_res: int = 1,
) -> str:
    lines = []
    atom_idx = 1
    atom_order = ["N", "CA", "C", "O"]
    for i, res in enumerate(residues):
        res_num = start_res + i
        for aname in atom_order:
            coord = res[aname]
            element = aname[0]
            lines.append(
                f"ATOM  {atom_idx:5d}  {aname:<3s} {res_name} {chain}{res_num:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00  0.00           {element:>2s}\n"
            )
            atom_idx += 1
    lines.append("END\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_seed_pdb(seed_type: str, custom_pdb: Optional[str] = None) -> Tuple[str, int]:
    """
    Return (pdb_string, n_residues) for the named seed type.
    The fragment is centered so its first CA is at the origin.
    """
    if seed_type == "single_gly":
        pdb_str = (
            "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  GLY A   1       1.459   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   GLY A   1       2.006   1.411   0.000  1.00  0.00           C\n"
            "ATOM      4  O   GLY A   1       3.228   1.589   0.000  1.00  0.00           O\n"
            "END\n"
        )
        return pdb_str, 1

    elif seed_type == "helix_3":
        residues = _build_backbone(3, _HELIX_PHI, _HELIX_PSI)
        return _backbone_to_pdb_string(residues), 3

    elif seed_type == "helix_5":
        residues = _build_backbone(5, _HELIX_PHI, _HELIX_PSI)
        return _backbone_to_pdb_string(residues), 5

    elif seed_type == "strand_3":
        residues = _build_backbone(3, _STRAND_PHI, _STRAND_PSI)
        return _backbone_to_pdb_string(residues), 3

    elif seed_type == "custom":
        if custom_pdb is None:
            raise ValueError("custom_pdb must be provided for 'custom' seed type")
        n_res = _count_residues_in_pdb(custom_pdb)
        return custom_pdb, n_res

    else:
        raise ValueError(f"Unknown seed type: {seed_type!r}. Must be one of {SEED_TYPES}")


def get_seed_structure(seed_type: str, custom_pdb: Optional[str] = None) -> struc.AtomArray:
    """Return the seed as a biotite AtomArray (first CA at origin)."""
    pdb_str, _ = get_seed_pdb(seed_type, custom_pdb)
    return pdb_io.PDBFile.read(StringIO(pdb_str)).get_structure(model=1)


def _count_residues_in_pdb(pdb_str: str) -> int:
    res_ids = set()
    for line in pdb_str.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                res_ids.add(int(line[22:26].strip()))
            except ValueError:
                pass
    return len(res_ids)
