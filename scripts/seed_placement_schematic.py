"""
PyMOL seed-placement schematic — PDL1 target (interactive).

What it draws
─────────────
  • PDL1 ribbon + transparent molecular surface
  • Hotspot residues as orange spheres
  • Outward surface-normal arrow (dark blue)
  • Placement points at d ∈ {8, 10, 12, 14} Å (coloured dots)
  • 5 seed-fragment arrows at θ = 90° (tangent) + φ ∈ {0°,72°,144°,216°,288°}

Run (inside PyMOL)
──────────────────
  PyMOL> run /path/to/seed_placement_schematic.py
"""

import math
import numpy as np

# ── PyMOL imports (run this via: pymol -r script.py  OR  PyMOL> run script.py)
from pymol import cmd, stored
from pymol.cgo import CYLINDER, CONE, SPHERE, COLOR, BEGIN, END, LINES, VERTEX

# ── Parameters ─────────────────────────────────────────────────────────────────
PDB_PATH   = "/Users/maximilianmatscher/remote/nfs/homedirs/matscher/targets/pdl1/pdl1.pdb"
HOTSPOTS   = [37, 39, 49, 98]       # PDL1 hotspot residue IDs (chain A)
INPUT_SPEC = (1, 115)
DISTANCES  = [8.0, 10.0, 12.0, 14.0]
SPINS      = [0, 72, 144, 216, 288]   # degrees
TILT       = 90.0                      # degrees – tangent placement

SEED_LEN   = 5.0      # arrow length (Å)
NORM_EXT   = 3.0      # how far past the last distance to extend the normal arrow
SHAFT_R    = 0.25     # cylinder radius
HEAD_R     = 0.60     # cone base radius
HEAD_L     = 1.20     # cone length

# ── Colour palette ─────────────────────────────────────────────────────────────
HOTSPOT_RGB  = (0.96, 0.65, 0.14)    # orange
ANCHOR_RGB   = (0.82, 0.40, 0.08)    # darker orange (centroid)
NORMAL_RGB   = (0.10, 0.36, 0.68)    # dark blue
DIST_RGB     = [                      # one per distance
    (0.20, 0.53, 0.84),              # d=8  blue
    (0.13, 0.70, 0.55),              # d=10 teal
    (0.95, 0.72, 0.02),              # d=12 amber
    (0.84, 0.18, 0.15),              # d=14 red
]
SPIN_RGB     = [                      # one per spin angle (tab-style)
    (0.12, 0.47, 0.71),              # φ=0°   blue
    (0.20, 0.63, 0.17),              # φ=72°  green
    (0.89, 0.10, 0.11),              # φ=144° red
    (1.00, 0.50, 0.00),              # φ=216° orange
    (0.42, 0.24, 0.60),              # φ=288° purple
]


# ── Geometry helpers ───────────────────────────────────────────────────────────

def norm(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)

def perp(v):
    ref = np.array([0., 0., 1.])
    if abs(np.dot(norm(v), ref)) > 0.9:
        ref = np.array([1., 0., 0.])
    return norm(np.cross(v, ref))

def rodrigues(v, axis, angle_rad):
    k = norm(axis)
    v = np.asarray(v, float)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)

def seed_dir(base_dir, tilt_rad, spin_rad):
    d = rodrigues(base_dir.copy(), perp(base_dir), tilt_rad)
    return norm(rodrigues(d, base_dir, spin_rad))


# ── CGO helpers ────────────────────────────────────────────────────────────────

def cgo_arrow(p0, p1, rgb, shaft_r=SHAFT_R, head_r=HEAD_R, head_l=HEAD_L):
    """Cylinder shaft + cone head from p0 toward p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    shaft_len = np.linalg.norm(p1 - p0) - head_l
    if shaft_len < 0.1:
        shaft_len = np.linalg.norm(p1 - p0) * 0.7
    d = norm(p1 - p0)
    shaft_end = p0 + d * shaft_len
    r, g, b = rgb
    return [
        CYLINDER,
        *p0.tolist(), *shaft_end.tolist(), shaft_r,
        r, g, b, r, g, b,
        CONE,
        *shaft_end.tolist(), *p1.tolist(), head_r, 0.0,
        r, g, b, r, g, b,
        1.0, 1.0,
    ]

def cgo_sphere(pos, rgb, r=0.5):
    return [COLOR, *rgb, SPHERE, *np.asarray(pos, float).tolist(), r]


# ── Load structure ─────────────────────────────────────────────────────────────

cmd.load(PDB_PATH, "target")
cmd.remove("hetatm")

# Keep only input residues (A1–115)
cmd.select("keep_res",
           f"target and chain A and resi {INPUT_SPEC[0]}-{INPUT_SPEC[1]}")
cmd.remove("target and not keep_res")
cmd.delete("keep_res")

# Fetch CA coordinates using pymol.stored (the correct headless way)
stored.ca = []
cmd.iterate_state(1, "target and name CA",
                  "stored.ca.append((int(resi), x, y, z))")
ca_by_id = {r: np.array([x, y, z]) for r, x, y, z in stored.ca}
all_ca   = np.array(list(ca_by_id.values()))

# Verify hotspot residues exist
found = [r for r in HOTSPOTS if r in ca_by_id]
missing = [r for r in HOTSPOTS if r not in ca_by_id]
if missing:
    print(f"WARNING: hotspot residues not found: {missing}")
    print(f"Available range: {min(ca_by_id)} – {max(ca_by_id)}")
hotspot_ca = np.array([ca_by_id[r] for r in found])

anchor    = hotspot_ca.mean(axis=0)
# Surface normal: SVD of local CA neighbourhood
dists     = np.linalg.norm(all_ca - anchor, axis=1)
nearby    = all_ca[dists <= 12.0]
_, _, vh  = np.linalg.svd(nearby - nearby.mean(axis=0))
n_raw     = vh[-1]
if np.dot(n_raw, anchor - all_ca.mean(axis=0)) < 0:
    n_raw = -n_raw
direction = norm(n_raw)

print(f"Anchor: {anchor.round(2)}")
print(f"Normal: {direction.round(3)}")
print(f"Hotspot CAs found: {found}")


# ── Visual setup ───────────────────────────────────────────────────────────────

cmd.hide("everything", "target")
cmd.show("cartoon",  "target")
cmd.show("surface",  "target")

# Colour ribbon by secondary structure (built-in PyMOL colour scheme)
cmd.util.cbc("target")                         # colour by chain (clean blue)
cmd.set("cartoon_color",    "skyblue",  "target")
cmd.set("surface_color",    "lightblue","target")
cmd.set("transparency",      0.55,       "target")
cmd.set("cartoon_transparency", 0.05,   "target")

# Hotspot residues — orange spheres
hs_sel = "target and chain A and resi " + "+".join(str(r) for r in found)
cmd.show("spheres", hs_sel)
cmd.color("tv_orange", hs_sel)
cmd.set("sphere_scale", 0.55, hs_sel)


# ── CGO objects ────────────────────────────────────────────────────────────────

cgo_all = []

# 1. Hotspot centroid (slightly larger sphere)
cgo_all += cgo_sphere(anchor, ANCHOR_RGB, r=0.80)

# 2. Surface-normal arrow
tip_norm = anchor + direction * (max(DISTANCES) + NORM_EXT)
cgo_all += cgo_arrow(anchor, tip_norm, NORMAL_RGB,
                     shaft_r=0.22, head_r=0.55, head_l=1.4)

# 3. Placement dots at each distance + distance tick
for dist, d_rgb in zip(DISTANCES, DIST_RGB):
    pl = anchor + direction * dist
    cgo_all += cgo_sphere(pl, d_rgb, r=0.55)
    # small cross-tick on the normal shaft
    t  = perp(direction)
    for s in [-0.3, 0.3]:
        cgo_all += [
            CYLINDER,
            *(pl + t * s - perp(t) * 0.0).tolist(),
            *(pl + t * s + perp(t) * 0.0).tolist(),
        ]

# 4. Seed arrows at the furthest distance — θ=90°, 5 spin angles
pl_seeds = anchor + direction * max(DISTANCES)
for spin_deg, s_rgb in zip(SPINS, SPIN_RGB):
    d   = seed_dir(direction, math.radians(TILT), math.radians(spin_deg))
    end = pl_seeds + d * SEED_LEN
    cgo_all += cgo_arrow(pl_seeds, end, s_rgb,
                         shaft_r=0.20, head_r=0.50, head_l=1.0)

cmd.load_cgo(cgo_all, "placement")

# 5. Label the normal (PyMOL label object)
tip_lbl = anchor + direction * (max(DISTANCES) + NORM_EXT + 1.2)
cmd.pseudoatom("lbl_n", pos=tip_lbl.tolist(), label="n-hat")
cmd.hide("everything", "lbl_n")
cmd.label("lbl_n", '"n̂"')
cmd.set("label_size",  22)
cmd.set("label_color", "marine")


# ── Camera & render settings ───────────────────────────────────────────────────

cmd.orient("target")
cmd.zoom("all", buffer=5)
cmd.turn("x", -15)
cmd.turn("y",  25)

cmd.bg_color("white")
cmd.set("ray_shadows",           0)
cmd.set("ray_opaque_background", 1)
cmd.set("ambient",               0.40)
cmd.set("specular",              0.25)
cmd.set("ray_trace_mode",        1)   # photorealistic
cmd.set("antialias",             2)
cmd.set("ray_trace_gain",        0.0)
cmd.set("depth_cue",             0)


print("Scene ready. Use PyMOL GUI to view and create GIF manually.")
