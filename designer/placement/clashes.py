"""
Clash detection between placed seed and target.
"""

from __future__ import annotations

import numpy as np
import biotite.structure as struc
from typing import List, Tuple


CLASH_THRESHOLD_DEFAULT = 2.0  # Å — hard clash for heavy atoms


def find_clashes(
    target: struc.AtomArray,
    seed: struc.AtomArray,
    threshold: float = CLASH_THRESHOLD_DEFAULT,
) -> List[Tuple[int, str, int, str, float]]:
    """
    Return list of clashing atom pairs between target and seed.

    Each entry: (target_res_id, target_atom_name, seed_res_id, seed_atom_name, distance)
    """
    # Exclude H atoms
    t_mask = ~np.isin(target.element, ["H"])
    s_mask = ~np.isin(seed.element, ["H"])

    t_coords = target[t_mask].coord
    s_coords = seed[s_mask].coord

    clashes = []
    for si, sc in enumerate(s_coords):
        dists = np.linalg.norm(t_coords - sc, axis=1)
        hits = np.where(dists < threshold)[0]
        for ti in hits:
            clashes.append((
                target[t_mask].res_id[ti],
                target[t_mask].atom_name[ti],
                seed[s_mask].res_id[si],
                seed[s_mask].atom_name[si],
                float(dists[ti]),
            ))
    return clashes


def clash_summary(clashes: List[Tuple]) -> str:
    if not clashes:
        return "No clashes detected."
    lines = [f"{len(clashes)} clash(es) detected:"]
    for t_res, t_atom, s_res, s_atom, dist in clashes[:10]:
        lines.append(f"  target {t_res}/{t_atom} — seed {s_res}/{s_atom}  d={dist:.2f} Å")
    if len(clashes) > 10:
        lines.append(f"  ... and {len(clashes) - 10} more")
    return "\n".join(lines)
