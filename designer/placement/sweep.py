"""
Sweep generation for parameter space exploration.

Placement direction model
-------------------------
The placement direction for a seed is parameterised as a point on a cone:

    direction(θ, φ) = cos(θ) * normal + sin(θ) * tangential(φ)

where
  θ  (cone_angle)  is the elevation from the surface normal.
       0°  → pure outward (directly above hotspot)
      45°  → equal outward + lateral  (x=1,y=0,z=1 style)
      90°  → purely lateral along the surface
  φ  (azimuth)     sweeps 360° around the normal axis.

Sweeping multiple cone angles at multiple azimuths, for one or more anchor
residues, gives a systematic coverage of the binding face.  Distance from
the anchor is fixed (8–16 Å has little influence on binding mode) or swept
over a small set of values.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalise a zero vector")
    return v / n


def _perpendicular_to(v: np.ndarray) -> np.ndarray:
    """Consistent unit vector perpendicular to v (reference tangential direction)."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    return _normalize(np.cross(v, ref))


def _rotate_around_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of v around axis by angle_rad."""
    k = _normalize(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1.0 - c)


def cone_directions(
    normal: np.ndarray,
    cone_angle_deg: float,
    n_azimuths: int,
) -> List[Tuple[np.ndarray, str]]:
    """
    Return unit placement directions on a cone at cone_angle_deg from the surface normal.

    cone_angle_deg=0  → single direction (pure outward, azimuth irrelevant).
    cone_angle_deg=45 → 8 directions (with n_azimuths=8), each at 45° off the
                        normal, sweeping the azimuth in 45° steps.
                        The first direction is (normal + tangential)/√2, i.e.
                        the x=1,y=0,z=1 case when normal=(0,0,1).

    Returns list of (direction_unit_vector, label) tuples.
    """
    normal = _normalize(normal)

    if cone_angle_deg <= 1e-3:
        return [(normal, "c0az0")]

    theta = np.radians(cone_angle_deg)
    perp = _perpendicular_to(normal)   # fixed reference tangential

    results = []
    for i in range(n_azimuths):
        phi = 2.0 * np.pi * i / n_azimuths
        # Rotate the reference tangential around the normal by phi
        tangential = _rotate_around_axis(perp, normal, phi)
        direction = np.cos(theta) * normal + np.sin(theta) * tangential
        az_label = f"c{cone_angle_deg:.0f}az{round(np.degrees(phi))}"
        results.append((_normalize(direction), az_label))

    return results


def fibonacci_sphere(
    center: np.ndarray,
    radius: float,
    n_points: int,
) -> List[np.ndarray]:
    """
    Evenly distributed points on a sphere (Fibonacci spiral).
    Used as an unbiased anchor baseline.
    """
    if n_points < 2:
        return [center + np.array([0.0, 0.0, radius])]

    phi = np.pi * (3.0 - np.sqrt(5.0))
    points = []
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        points.append(center + radius * np.array([np.cos(theta) * r, y, np.sin(theta) * r]))
    return points


def generate_sweep_configs(
    centroid: np.ndarray,
    surface_normal: np.ndarray,
    hotspot_cas: List[np.ndarray],
    seed_type: str,
    custom_pdb: Optional[str] = None,
    # anchor strategy
    use_centroid: bool = True,
    use_individual: bool = True,
    use_sphere: bool = False,
    sphere_n: int = 12,
    sphere_r: float = 5.0,
    # direction strategy
    cone_angles: Optional[List[float]] = None,   # degrees, e.g. [0, 45]
    n_azimuths: int = 8,
    # distance — minimal variation since 8–16 Å has little influence
    distances: Optional[List[float]] = None,     # e.g. [12.0, 18.0]
    # seed orientation
    n_spins: int = 4,
    tilts: Optional[List[float]] = None,         # tilt angles in degrees, e.g. [0, 45, 90]
) -> List[Dict[str, Any]]:
    """
    Build the flat list of placement configs for a parameter sweep.

    For each (anchor, direction, distance, spin) combination, one config is
    produced.  The direction is taken from a set of cone directions:

      direction(θ, φ) = cos(θ)*normal + sin(θ)*tangential(φ)

    so a cone_angle of 45° places the seed at (1,0,1)-style positions
    (equal outward and lateral components).

    Config dict keys: anchor_pos, anchor_label, direction, distance,
                      rotation_deg, tilt_deg, seed_type, custom_pdb.
    """
    if cone_angles is None:
        cone_angles = [0.0, 45.0]
    if distances is None:
        distances = [12.0, 18.0]
    if tilts is None:
        tilts = [0.0]

    surface_normal = _normalize(surface_normal)

    # ── Anchors ────────────────────────────────────────────────────────────
    # Cone anchors (centroid + individual hotspot CAs) get the full direction sweep.
    # Sphere anchors use the outward vector from centroid → sphere point directly —
    # the sphere geometry already provides directional coverage.
    cone_anchors: List[Tuple[np.ndarray, str]] = []
    if use_centroid:
        cone_anchors.append((centroid, "cen"))
    if use_individual and hotspot_cas:
        for i, ca in enumerate(hotspot_cas):
            cone_anchors.append((ca, f"hs{i}"))

    sphere_anchors: List[Tuple[np.ndarray, np.ndarray, str]] = []  # (pos, direction, label)
    if use_sphere:
        for i, pt in enumerate(fibonacci_sphere(centroid, sphere_r, sphere_n)):
            sphere_anchors.append((pt, _normalize(pt - centroid), f"sp{i}"))

    if not cone_anchors and not sphere_anchors:
        raise ValueError("No anchors generated — enable at least one anchor strategy.")

    # ── Directions for cone anchors ────────────────────────────────────────
    cone_directions_list: List[Tuple[np.ndarray, str]] = []
    for cone_angle in sorted(set(cone_angles)):
        cone_directions_list.extend(cone_directions(surface_normal, cone_angle, n_azimuths))

    # ── Seed spins ─────────────────────────────────────────────────────────
    spins = np.linspace(0.0, 360.0, max(1, n_spins), endpoint=False).tolist()

    def _format_distance(d: float) -> str:
        if abs(d - round(d)) < 1e-3:
            return f"{round(d)}"
        return f"{d}".replace(".", "_")

    # ── Build configs ──────────────────────────────────────────────────────
    configs: List[Dict[str, Any]] = []

    for anchor_pos, anchor_label in cone_anchors:
        for direction, dir_label in cone_directions_list:
            for distance in distances:
                _dist_str = f"_{_format_distance(distance)}A" if len(distances) > 1 else ""
                for spin in spins:
                    for tilt in tilts:
                        configs.append({
                            "anchor_pos":   anchor_pos,
                            "anchor_label": f"{anchor_label}_{dir_label}{_dist_str}",
                            "direction":    direction,
                            "distance":     float(distance),
                            "rotation_deg": float(spin),
                            "tilt_deg":     float(tilt),
                            "seed_type":    seed_type,
                            "custom_pdb":   custom_pdb,
                        })

    for anchor_pos, direction, anchor_label in sphere_anchors:
        for distance in distances:
            _dist_str = f"_{_format_distance(distance)}A" if len(distances) > 1 else ""
            for spin in spins:
                for tilt in tilts:
                    configs.append({
                        "anchor_pos":   anchor_pos,
                        "anchor_label": f"{anchor_label}{_dist_str}",
                        "direction":    direction,
                        "distance":     float(distance),
                        "rotation_deg": float(spin),
                        "tilt_deg":     float(tilt),
                        "seed_type":    seed_type,
                        "custom_pdb":   custom_pdb,
                    })

    return configs


def count_sweep_configs(
    n_cone_anchors: int,
    n_sphere_anchors: int,
    cone_angles: List[float],
    n_azimuths: int,
    n_distances: int,
    n_spins: int,
    n_tilts: int = 1,
) -> int:
    """Return expected config count without generating them."""
    n_dirs = sum(1 if a <= 1e-3 else n_azimuths for a in cone_angles)
    return (n_cone_anchors * n_dirs + n_sphere_anchors) * n_distances * n_spins * n_tilts
