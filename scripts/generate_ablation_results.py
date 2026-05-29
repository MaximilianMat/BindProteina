"""
Generate Ablation Results
=========================

Walks an ablation study directory tree and extracts condition parameters from
folder names.  Parameters (dimensions) are discovered dynamically from the
naming convention rather than being hard-coded, so the same script works for
old-style (PDL1) and new-style (NPT2B_2, CLD18_exp, …) layouts.

Naming convention (underscore-delimited tokens in the *condition* folder name):
    TARGET_hsRESIDUE_FRAGMENT_ORIENTATION_DISTANCE[_rotANGLE[_tiltTILT]]

Old-style layouts have a *separate* ``_rotANGLE`` sub-folder beneath the
condition directory; new-style layouts embed ``rot`` and ``tilt`` directly
in the condition folder name.

The script discovers ``initial_*`` sub-trees as leaf experiments and counts
successes (relaxed PDBs) vs. total samples (protenix_batch PDBs or generated
linked_sample dirs).
"""

import json
import re
from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# Regex-based token extractors
# ---------------------------------------------------------------------------

# Fragment codes we recognise (order matters: longer codes first)
_FRAGMENT_CODES = {
    "3hxc": "3_res_helix_capped",
    "3hx":  "3_res_helix",
    "5hx":  "5_res_helix",
    "3gx":  "3_res_gly_helix",
    "3bs":  "3_res_beta",
    "3bt":  "3_res_beta",
    "1gly": "single_gly",
    "gly":  "single_gly",
}

# Orientation codes – longer codes first to avoid partial matches
_ORIENTATION_CODES = {
    "sno":       "outward",
    "sni":       "inward",
    "negy":      "negy",
    "xneg":      "xneg",
    "xz":        "xz",
    "x":         "x",
    "z":         "z",
    "reference": "reference",
}

# Distance pattern: handles integer (10A) and fractional (6_5A → 6.5)
_DIST_RE = re.compile(r"_(\d+(?:_\d+)?)A(?:_|$)")

# Rotation and tilt
_ROT_RE  = re.compile(r"_?rot(\d+)")
_TILT_RE = re.compile(r"_?tilt(\d+)")

# Hotspot residue
_HS_RE = re.compile(r"_hs(\d+)")


def _parse_distance(s: str) -> str | None:
    """Return distance as a string like '10' or '6.5', or None."""
    m = _DIST_RE.search(s)
    if m:
        raw = m.group(1)
        return raw.replace("_", ".")  # 6_5 → 6.5
    return None


def _parse_fragment(s: str) -> str | None:
    for code, name in _FRAGMENT_CODES.items():
        # Match as a whole token (delimited by underscores or start/end)
        if re.search(rf"(?:^|_){re.escape(code)}(?:_|$)", s):
            return name
    return None


def _parse_orientation(s: str) -> str | None:
    for code, name in _ORIENTATION_CODES.items():
        if re.search(rf"(?:^|_){re.escape(code)}(?:_|$)", s):
            return name
    return None


def _parse_rot(s: str) -> str | None:
    m = _ROT_RE.search(s)
    return m.group(1) if m else None


def _parse_tilt(s: str) -> str | None:
    m = _TILT_RE.search(s)
    return m.group(1) if m else None


def _parse_hotspot(s: str) -> str | None:
    m = _HS_RE.search(s)
    return m.group(1) if m else None


def _parse_target(s: str) -> str | None:
    """First token before the first underscore-separated dimension."""
    parts = s.split("_")
    if parts:
        return parts[0]
    return None


def parse_condition_from_path(path_str):
    """
    Tries to infer orientation, distance, fragment, angle, and tilt from the base_run name.
    Expected folder naming conventions:
      - sni -> inward, sno -> outward
      - 8A, 10A, 12A, 14A, 6_5A etc. -> distance (fractional allowed)
      - gly -> single_gly, 3hx -> 3_res_helix, 3bt -> 3_res_beta, 5hx -> 5_res_helix, 3hxc -> 3hxc, x -> x, xneg -> xneg, etc.
      - _rot72 -> angle 72
      - tilt140 -> tilt angle 140 (optional)
    """
    path = Path(path_str)
    
    # Orientation
    orientation = "unknown"
    if "sno" in path_str:
        orientation = "outward"
    elif "sni" in path_str:
        orientation = "inward"
    # Also check for explicit orientation codes like 'negy', 'xz', 'z', 'x', 'xneg', 'reference'
    for code in ["negy", "xz", "z", "x", "xneg", "reference"]:
        if code in path_str:
            orientation = code
            break
    
    # Distance – supports fractional like 6_5A (6.5 Å)
    distance = 0
    dist_match = re.search(r"(\d+_?\d*)A", path_str)
    if dist_match:
        d_str = dist_match.group(1)
        if "_" in d_str:
            distance = float(d_str.replace("_", "."))
        else:
            distance = int(d_str)
    
    # Fragment
    fragment = "unknown"
    fragment_map = {
        "gly": "single_gly",
        "3hx": "3_res_helix",
        "3hxc": "3hxc",
        "3bt": "3_res_beta",
        "3bs": "3_res_beta",
        "5hx": "5_res_helix",
        "x": "x",
        "xneg": "xneg"
    }
    for key, val in fragment_map.items():
        if key in path_str:
            fragment = val
            break
    
    # Angle from _rot*
    angle = 0
    for part in path.parts:
        if part.startswith("_rot"):
            try:
                angle = int(part.replace("_rot", ""))
            except Exception:
                pass
    
    # Tilt – optional
    tilt = None
    tilt_match = re.search(r"tilt(\d+)", path_str)
    if tilt_match:
        tilt = int(tilt_match.group(1))
    
    return orientation, angle, distance, fragment, tilt


def parse_condition(name: str) -> dict:
    """
    Parse a condition folder name into a dict of dimension → value.
    Only dimensions that are actually present in the name are included.
    """
    cond: dict = {}

    target = _parse_target(name)
    if target:
        cond["target"] = target

    hs = _parse_hotspot(name)
    if hs is not None:
        cond["hotspot"] = hs

    frag = _parse_fragment(name)
    if frag is not None:
        cond["fragment"] = frag

    orient = _parse_orientation(name)
    if orient is not None:
        cond["orientation"] = orient

    dist = _parse_distance(name)
    if dist is not None:
        cond["distance"] = dist

    rot = _parse_rot(name)
    if rot is not None:
        cond["rotation"] = rot

    tilt = _parse_tilt(name)
    if tilt is not None:
        cond["tilt"] = tilt

    return cond


# ---------------------------------------------------------------------------
# Leaf experiment discovery
# ---------------------------------------------------------------------------

def _find_initial_dirs(base: Path):
    """
    Yield every ``initial_*`` directory anywhere under *base*.
    Each represents one experimental leaf.
    """
    for p in sorted(base.rglob("initial_*")):
        if p.is_dir():
            yield p


def _count_success(initial_dir: Path) -> int:
    return len(list(initial_dir.rglob("relaxed/*.pdb")))


def _count_samples(initial_dir: Path) -> int:
    # Prefer protenix_batch count
    n = len(list(initial_dir.rglob("protenix_batch/*.pdb")))
    if n > 0:
        return n
    # Fallback: count generated linked_sample dirs
    n = len([d for d in (initial_dir / "generated").iterdir() if d.is_dir()]) if (initial_dir / "generated").is_dir() else 0
    if n > 0:
        return n
    # Last resort
    return 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Walk an ablation study directory tree and extract results."
    )
    parser.add_argument("--input", required=True, help="Base directory of ablation study")
    parser.add_argument("--output", default="ablation_results.json", help="Output JSON file")
    parser.add_argument(
        "--residue",
        help="Optional residue substring filter (e.g. 'hs60').  "
             "Matched via simple substring match against the condition path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    results = []
    seen = set()

    for initial_dir in _find_initial_dirs(input_dir):
        # Build the "condition path" from everything between input_dir and
        # the initial_* directory (may be one or two levels).
        rel = initial_dir.relative_to(input_dir)
        # Everything before the initial_* component is the condition hierarchy
        condition_parts = list(rel.parts[:-1])  # drop the initial_* part itself

        if not condition_parts:
            continue

        condition_key = "/".join(condition_parts)

        # Avoid counting the same condition twice (shouldn't happen, but guard)
        if condition_key in seen:
            continue
        seen.add(condition_key)

        # Build full condition string for parsing
        condition_str = "_".join(condition_parts)

        # Residue filter: simple substring match
        if args.residue and args.residue not in condition_str:
            continue

        cond = parse_condition(condition_str)

        # Must have at least fragment or distance to be useful
        if "fragment" not in cond and "distance" not in cond:
            continue

        n_success = _count_success(initial_dir)
        n_samples = _count_samples(initial_dir)
        success_rate = n_success / n_samples if n_samples > 0 else 0.0

        record = {
            **cond,
            "success_rate": round(success_rate, 4),
            "n_samples": n_samples,
            "n_success": n_success,
            "condition_path": condition_key,
        }
        results.append(record)

    # Sort for reproducibility
    results.sort(key=lambda r: r.get("condition_path", ""))

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    # Summary
    all_dims = set()
    for r in results:
        all_dims.update(k for k in r if k not in ("success_rate", "n_samples", "n_success", "condition_path"))
    print(f"Generated {len(results)} records → {args.output}")
    print(f"Discovered dimensions: {sorted(all_dims)}")


if __name__ == "__main__":
    main()
