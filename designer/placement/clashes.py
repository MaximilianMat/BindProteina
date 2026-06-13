"""
Clash / proximity detection between placed seed and target.
"""

from __future__ import annotations

import numpy as np
import biotite.structure as struc
from typing import List, Tuple

# 2.0 Å = hard van-der-Waals overlap.
# 4.0 Å = proximity contact — seed residue is very close to the target,
#          which would restrict the space available for the binder.
#          Used as the default to avoid seeds that are too tightly packed
#          against the target surface.
CLASH_THRESHOLD_DEFAULT = 4.0


def _heavy_coords(arr: struc.AtomArray) -> np.ndarray:
    """Return (N, 3) coordinates of all non-hydrogen atoms."""
    return arr[~np.isin(arr.element, ["H", "D"])].coord


def has_clash(
    target: struc.AtomArray,
    seed: struc.AtomArray,
    threshold: float = CLASH_THRESHOLD_DEFAULT,
) -> bool:
    """
    Fast boolean check: is any seed heavy atom within `threshold` Å of any target heavy atom?
    Uses broadcasting so it avoids a Python loop.
    """
    t_coords = _heavy_coords(target)
    s_coords = _heavy_coords(seed)
    if len(t_coords) == 0 or len(s_coords) == 0:
        return False
    # (n_seed, 1, 3) - (1, n_target, 3) → (n_seed, n_target)
    dists = np.linalg.norm(
        s_coords[:, np.newaxis, :] - t_coords[np.newaxis, :, :], axis=2
    )
    return bool(np.any(dists < threshold))


def find_clashes(
    target: struc.AtomArray,
    seed: struc.AtomArray,
    threshold: float = CLASH_THRESHOLD_DEFAULT,
) -> List[Tuple[int, str, int, str, float]]:
    """
    Return every (target_res_id, target_atom, seed_res_id, seed_atom, distance)
    pair where distance < threshold.
    """
    t_heavy = target[~np.isin(target.element, ["H", "D"])]
    s_heavy = seed[~np.isin(seed.element, ["H", "D"])]

    t_coords = t_heavy.coord
    s_coords = s_heavy.coord

    clashes = []
    for si in range(len(s_coords)):
        dists = np.linalg.norm(t_coords - s_coords[si], axis=1)
        for ti in np.where(dists < threshold)[0]:
            clashes.append((
                int(t_heavy.res_id[ti]),
                str(t_heavy.atom_name[ti]),
                int(s_heavy.res_id[si]),
                str(s_heavy.atom_name[si]),
                float(dists[ti]),
            ))
    return clashes


def clash_summary(clashes: List[Tuple], threshold: float = CLASH_THRESHOLD_DEFAULT) -> str:
    if not clashes:
        return "No contacts detected."
    hard = [c for c in clashes if c[4] < 2.0]
    label = "hard clash" if hard else "proximity contact"
    lines = [f"{len(clashes)} {label}(s) within {threshold:.1f} Å:"]
    for t_res, t_atom, s_res, s_atom, dist in clashes[:10]:
        tag = " ⚠️ hard" if dist < 2.0 else ""
        lines.append(f"  target {t_res}/{t_atom} — seed {s_res}/{s_atom}  {dist:.2f} Å{tag}")
    if len(clashes) > 10:
        lines.append(f"  … and {len(clashes) - 10} more")
    return "\n".join(lines)
