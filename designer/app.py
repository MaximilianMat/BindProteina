"""
LaProteina Linked-PDB Designer
================================
Streamlit app for placing seed fragments on a target protein and
generating settings_target JSON files ready for SLURM submission.

Run with:
    streamlit run designer/app.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from designer.placement.anchors import (
    compute_surface_normal,
    direction_from_preset,
    get_anchor_ca,
    hotspot_centroid,
    load_structure_from_path,
    load_structure_from_string,
    parse_custom_direction,
    parse_hotspot_string,
    residue_ids_in_chain,
)
from designer.placement.clashes import CLASH_THRESHOLD_DEFAULT, clash_summary, find_clashes, has_clash
from designer.placement.placer import (
    generate_candidates,
    place_seed,
    save_linked_pdb,
    structure_to_pdb_string,
)
from designer.placement.seeds import SEED_LABELS, SEED_TYPES, get_seed_structure
from designer.placement.sweep import count_sweep_configs, generate_sweep_configs
from designer.settings.builder import (
    batch_slurm_command,
    build_settings,
    save_batch_manifest,
    save_settings,
    settings_to_json,
    slurm_command,
)
from designer.settings.templates import get_defaults, get_path_translator
from designer.viewer.render import make_linked_viewer, make_sweep_preview_viewer, make_viewer, view_to_html

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LaProteina Designer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 LaProteina Linked-PDB Designer")
st.caption(
    "Place a seed fragment on a target PDB and generate a `settings_target` JSON "
    "for LaProteina binder generation."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIRECTION_PRESETS = [
    "surface-normal-outward",
    "surface-normal-inward",
    "+x", "-x", "+y", "-y", "+z", "-z",
    "custom (enter below)",
]


@st.cache_data(show_spinner=False)
def _load_target(pdb_path: str, chain: str):
    return load_structure_from_path(pdb_path, chain_id=chain)


def _show_viewer(html: str, height: int = 500):
    components.html(html, height=height + 20, scrolling=False)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _pdb_sequence(pdb_path: str, chain_id: str) -> str:
    """Extract 1-letter sequence for one chain from a PDB file.

    PDB column layout (0-indexed): atom_name=12:16, res_name=17:20,
    chain_id=21, res_seq+icode=22:27.
    """
    seen: set = set()
    residues: list = []
    with open(pdb_path) as _f:
        for _line in _f:
            if len(_line) < 27:
                continue
            if not _line.startswith("ATOM"):
                continue
            if _line[12:16].strip() != "CA":
                continue
            if _line[21] != chain_id:
                continue
            _key = _line[22:27]  # residue seq number (4) + insertion code (1)
            if _key not in seen:
                seen.add(_key)
                residues.append(_AA3TO1.get(_line[17:20].strip(), "X"))
    return "".join(residues)


def _hotspot_positions_str(hotspot_ids: list, sorted_chain_ids: list) -> str:
    """
    Convert author-numbered hotspot residue IDs to 1-based sequential positions.

    The seed placer (placer.place_seed) renumbers the target chain to 1..N in
    ascending order of the original residue IDs, so the generated / refolded
    structures the scorer sees are 1-based-sequential — NOT the author numbering
    the user typed.  This maps each hotspot to its position (rank) in the sorted
    chain so the downstream hotspot-contact check compares like-for-like.

    Residues not present in the chain are skipped (they were already dropped by
    the Section-2 validation).
    """
    rank = {rid: i for i, rid in enumerate(sorted_chain_ids, 1)}
    positions = sorted(rank[h] for h in hotspot_ids if h in rank)
    return ",".join(str(p) for p in positions)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_d = get_defaults()

_STATE_DEFAULTS: dict = {
    # --- structural state ---
    "target_structure":   None,
    "target_pdb_str":     "",
    "target_length":      0,
    "hotspot_ids":        [],
    "linked_structures":  [],
    "linked_pdb_strings": [],
    # --- settings fields (prefixed s_) — persisted via key= on widgets ---
    "s_binder_name":       f"binder_{_ts()}",
    "s_n_samples":         _d["n_samples"],
    "s_batch_size":        _d["batch_size"],
    "s_binder_len_min":    _d["binder_length_min"],
    "s_binder_len_max":    _d["binder_length_max"],
    "s_keep_linker_end_n": 1,
    "s_laproteina_mode":   _d["laproteina_mode"],
    "s_rosetta_relax":     _d["rosetta_relax"],
    "s_checkpoint_path":   _d["checkpoint_path"],
    "s_autoencoder_path":  _d["autoencoder_path"],
    "s_folding_model":     _d.get("folding_model", "af2"),
    "s_protenix_model":         _d.get("protenix_model", "protenix-v2"),
    "s_target_msa":             "",
    "s_protenix_use_template":  False,
    "s_original_pdb_local":     "",
    "s_output_dir":           _d["default_output_dir"],
    "s_save_dir_local":       "",
    # batch queue
    "batch_queue":            [],
    "s_batch_save_dir":       "",
    "s_batch_max_concurrent": 6,
    # placement angles
    "s_rotation_deg":         0,
    "s_tilt_deg":             0,
    # sweep
    "s_sweep_use_centroid":       True,
    "s_sweep_use_individual":     True,
    "s_sweep_use_sphere":         False,
    "s_sweep_sphere_radius":      5.0,
    "s_sweep_sphere_npoints":     12,
    "s_sweep_cone_angles_str":    "0,45",
    "s_sweep_n_azimuths":         8,
    "s_sweep_distances_str":      "12,18",
    "s_sweep_n_spins":            4,
    "s_sweep_tilts_str":          "0",
    "s_sweep_samples_per_config": 100,
    "current_direction":          None,
    "sweep_preview_points":       None,
    # guard flags for preventing double-execution during Streamlit reruns
    "_generating_sweep":          False,
    "_adding_placements":         False,
}

for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# ── Section 1: Target PDB ──
# ---------------------------------------------------------------------------

st.header("1 · Target PDB")

col_path, col_chain = st.columns([4, 1])
with col_path:
    pdb_source = st.radio(
        "Source", ["File path (NFS mount)", "Upload"], horizontal=True, label_visibility="collapsed"
    )

with col_chain:
    chain_id = st.text_input("Chain", value="A", max_chars=1).strip().upper() or "A"

if pdb_source == "File path (NFS mount)":
    pdb_path_input = st.text_input(
        "Path to target PDB",
        placeholder="/Users/maximilianmatscher/remote/nfs/homedirs/matscher/targets/pdl1/pdl1.pdb",
    )
    if pdb_path_input and Path(pdb_path_input).exists():
        try:
            st.session_state.target_structure = _load_target(pdb_path_input, chain_id)
            with open(pdb_path_input) as f:
                st.session_state.target_pdb_str = f.read()
            ids = residue_ids_in_chain(st.session_state.target_structure, chain_id)
            st.session_state.target_length = len(ids)
            st.session_state.s_original_pdb_local = pdb_path_input
            st.success(f"Loaded  {st.session_state.target_length} residues (chain {chain_id})")
        except Exception as e:
            st.error(f"Failed to load PDB: {e}")
    elif pdb_path_input:
        st.warning("File not found at that path.")
else:
    uploaded = st.file_uploader("Upload PDB file", type=["pdb"])
    if uploaded:
        pdb_str = uploaded.read().decode()
        st.session_state.target_pdb_str = pdb_str
        try:
            st.session_state.target_structure = load_structure_from_string(pdb_str, chain_id)
            ids = residue_ids_in_chain(st.session_state.target_structure, chain_id)
            st.session_state.target_length = len(ids)
            st.success(f"Loaded  {st.session_state.target_length} residues (chain {chain_id})")
        except Exception as e:
            st.error(f"Failed to parse uploaded PDB: {e}")

target = st.session_state.target_structure

# ---------------------------------------------------------------------------
# ── Section 2: Hotspot residues ──
# ---------------------------------------------------------------------------

st.header("2 · Hotspot Residues")
st.caption(
    "These are used for scoring guidance (`target_hotspot_residues` in the JSON) "
    "and anchor selection."
)

hotspot_str = st.text_input(
    "Hotspot residues",
    value="",
    placeholder="50-107  or  50,52,87,90-95",
    help="Ranges and individual IDs, comma-separated.",
)

if hotspot_str.strip() and target is not None:
    try:
        hotspot_ids = parse_hotspot_string(hotspot_str)
        # Validate: check they exist in the structure
        all_ids = residue_ids_in_chain(target, chain_id)
        bad = [r for r in hotspot_ids if r not in all_ids]
        if bad:
            st.warning(f"Residues not found in chain {chain_id}: {bad[:10]}")
            hotspot_ids = [r for r in hotspot_ids if r not in bad]
        st.session_state.hotspot_ids = hotspot_ids
        st.info(f"{len(hotspot_ids)} hotspot residues")
    except Exception as e:
        st.error(f"Could not parse hotspot string: {e}")
        hotspot_ids = []
else:
    hotspot_ids = st.session_state.hotspot_ids

# 1-based sequential positions (matches the placer's target renumbering) — this
# is the numbering the generated/refolded structures use, so it is what the
# scorer's hotspot-contact check must compare against.  See _hotspot_positions_str.
hotspot_positions_str = (
    _hotspot_positions_str(hotspot_ids, residue_ids_in_chain(target, chain_id))
    if hotspot_ids and target is not None else ""
)

# Quick structure preview when target is loaded
if target is not None and st.session_state.target_pdb_str:
    with st.expander("Target preview", expanded=False):
        view = make_viewer(
            st.session_state.target_pdb_str,
            hotspot_res_ids=hotspot_ids or None,
            width=780, height=420,
        )
        _show_viewer(view_to_html(view), height=420)
        st.caption("🖱 Left-drag: rotate · Right-drag: pan/translate · Scroll: zoom")

# ---------------------------------------------------------------------------
# ── Section 3: Placement ──
# ---------------------------------------------------------------------------

st.header("3 · Seed Placement")

if target is None:
    st.info("Load a target PDB first.")
    st.stop()

col_anchor, col_seed = st.columns(2)

with col_anchor:
    st.subheader("Anchor")
    anchor_options = ["centroid of all hotspots"] + (
        [str(r) for r in hotspot_ids] if hotspot_ids else []
    )
    anchor_choice = st.selectbox("Anchor point", anchor_options)

    direction_choice = st.selectbox("Direction", DIRECTION_PRESETS)
    custom_dir_str = ""
    if direction_choice == "custom (enter below)":
        custom_dir_str = st.text_input("Direction vector (x,y,z)", value="0,0,1")

    distance = st.slider("Distance from anchor (Å)", min_value=3.0, max_value=20.0, value=7.0, step=0.5)

with col_seed:
    st.subheader("Seed fragment")
    seed_type = st.selectbox(
        "Seed type",
        SEED_TYPES,
        format_func=lambda k: SEED_LABELS[k],
    )
    custom_pdb_input = ""
    if seed_type == "custom":
        custom_pdb_input = st.text_area("Paste seed PDB", height=150)

    rotation_deg = 0.0
    tilt_deg = 0.0
    if seed_type != "single_gly":
        rotation_deg = st.slider(
            "Spin (°) — around placement axis",
            min_value=0, max_value=355, step=5,
            key="s_rotation_deg",
            help="Spins the seed around the anchor→placement direction vector.",
        )
        tilt_deg = st.slider(
            "Tilt (°) — away from placement direction",
            min_value=0, max_value=180, step=5,
            key="s_tilt_deg",
            help="0° = seed extends along the placement direction. "
                 "90° = seed lies tangent to the surface. "
                 "180° = seed points back toward the protein.",
        )

    n_candidates = st.number_input(
        "Generate N orientation candidates",
        min_value=1, max_value=36, value=1,
        help="Evenly spaces N rotations from 0° to 360°. Produces N linked PDBs.",
    )

# Placement button
if st.button("🔬 Preview placement", use_container_width=True):
    if not hotspot_ids and anchor_choice == "centroid of all hotspots":
        st.error("Enter hotspot residues first to use the centroid anchor.")
        st.stop()

    with st.spinner("Placing seed…"):
        try:
            # --- Anchor position ---
            if anchor_choice == "centroid of all hotspots":
                anchor_pos = hotspot_centroid(target, hotspot_ids)
                anchor_res_id = None
            else:
                anchor_res_id = int(anchor_choice)
                anchor_pos = get_anchor_ca(target, anchor_res_id)

            # --- Direction ---
            if direction_choice == "custom (enter below)":
                direction = parse_custom_direction(custom_dir_str)
            else:
                direction = direction_from_preset(direction_choice, target, anchor_pos)

            st.session_state.current_direction = direction

            # --- Generate ---
            if n_candidates == 1:
                linked, n_seed = place_seed(
                    target, anchor_pos, direction, distance,
                    seed_type, rotation_deg, tilt_deg,
                    custom_pdb=custom_pdb_input or None,
                )
                st.session_state.linked_structures = [(linked, n_seed, rotation_deg, tilt_deg)]
            else:
                results = generate_candidates(
                    target, anchor_pos, direction, distance,
                    seed_type, int(n_candidates),
                    tilt_deg=tilt_deg,
                    custom_pdb=custom_pdb_input or None,
                )
                st.session_state.linked_structures = results

            # Build PDB strings for display
            pdb_strings = [
                structure_to_pdb_string(s) for s, _, _, _ in st.session_state.linked_structures
            ]
            st.session_state.linked_pdb_strings = pdb_strings

        except Exception as e:
            st.error(f"Placement failed: {e}")
            st.stop()

# --- Visualise results ---
if st.session_state.linked_pdb_strings:
    n_res_seed = st.session_state.linked_structures[0][1]

    if len(st.session_state.linked_pdb_strings) == 1:
        pdb_str = st.session_state.linked_pdb_strings[0]
        linked, n_seed, rot, tilt = st.session_state.linked_structures[0]

        # Clash check — re-extract seed atoms from linked (residue IDs > target_length)
        tl = st.session_state.target_length
        seed_atoms = linked[linked.res_id > tl]
        clashes = find_clashes(target, seed_atoms)
        if clashes:
            st.warning(f"⚠️ {clash_summary(clashes)}")
        else:
            st.success(f"✅ No contacts within {CLASH_THRESHOLD_DEFAULT:.0f} Å.")

        view = make_linked_viewer(
            pdb_str, tl,
            hotspot_res_ids=hotspot_ids or None,
            anchor_res_id=int(anchor_choice) if anchor_choice != "centroid of all hotspots" else None,
            width=780, height=460,
        )
        _show_viewer(view_to_html(view), height=460)
        st.caption("🖱 Left-drag: rotate · Right-drag: pan/translate · Scroll: zoom")

    else:
        st.info(f"Generated {len(st.session_state.linked_pdb_strings)} orientation candidates.")
        tab_labels = [
            f"Spin {r:.0f}° / Tilt {t:.0f}°" for _, _, r, t in st.session_state.linked_structures
        ]
        tabs = st.tabs(tab_labels)
        for tab, (linked, n_seed, rot, tilt), pdb_str in zip(
            tabs, st.session_state.linked_structures, st.session_state.linked_pdb_strings
        ):
            with tab:
                tl = st.session_state.target_length
                view = make_linked_viewer(
                    pdb_str, tl,
                    hotspot_res_ids=hotspot_ids or None,
                    width=780, height=420,
                )
                _show_viewer(view_to_html(view), height=420)
                st.caption("🖱 Left-drag: rotate · Right-drag: pan/translate · Scroll: zoom")

# ---------------------------------------------------------------------------
# ── Section 3b: Parameter Sweep ──
# ---------------------------------------------------------------------------

st.header("3b · Parameter Sweep")
st.caption(
    "Systematically probe the space around hotspot residues. "
    "Run short scouting rounds (low samples/config) to learn where good seeds live, "
    "then iterate with higher budget on the winners."
)

if not hotspot_ids:
    st.info("Set hotspot residues in Section 2 to enable the parameter sweep.")
elif target is None:
    st.info("Load a target PDB in Section 1 first.")
else:
    # ── Anchor positions ──────────────────────────────────────────────────────
    st.subheader("Anchors")
    _sc1, _sc2 = st.columns(2)
    with _sc1:
        sw_centroid   = st.checkbox("Hotspot centroid", key="s_sweep_use_centroid")
        sw_individual = st.checkbox("Individual hotspot residues", key="s_sweep_use_individual")
    with _sc2:
        sw_sphere = st.checkbox(
            "Fibonacci sphere around centroid",
            key="s_sweep_use_sphere",
            help="Evenly-distributed anchor points on a sphere — unbiased baseline.",
        )
        if sw_sphere:
            _sp1, _sp2 = st.columns(2)
            with _sp1:
                st.number_input("Sphere points", min_value=4, max_value=50, step=1,
                                key="s_sweep_sphere_npoints")
            with _sp2:
                st.number_input("Sphere radius (Å)", min_value=2.0, max_value=20.0, step=0.5,
                                key="s_sweep_sphere_radius")

    # ── Placement directions (cone model) ─────────────────────────────────────
    st.subheader("Placement direction")
    st.caption(
        "Direction = cos(θ)·normal + sin(θ)·lateral(φ).  "
        "θ=0° → seed directly above hotspot.  "
        "θ=45° → seed at a 45° angle (equal outward + lateral — the (1,0,1) style).  "
        "φ sweeps 360° around the normal for each non-zero θ."
    )
    _dc1, _dc2, _dc3 = st.columns(3)
    with _dc1:
        sw_cone_angles_str = st.text_input(
            "Cone angles θ (°, comma-separated)",
            key="s_sweep_cone_angles_str",
            help="e.g. '0,45' or '0,30,45,60'",
        )
        try:
            sw_cone_angles = sorted({float(x.strip()) for x in sw_cone_angles_str.split(",") if x.strip()})
            if not sw_cone_angles:
                sw_cone_angles = [0.0]
        except ValueError:
            st.error("Invalid cone angles.")
            sw_cone_angles = [0.0]
    with _dc2:
        sw_n_azimuths = st.slider(
            "Azimuths φ per cone angle",
            min_value=4, max_value=24, step=2,
            key="s_sweep_n_azimuths",
            help="Number of equally spaced lateral directions per non-zero cone angle.",
        )
    with _dc3:
        sw_distances_str = st.text_input(
            "Distances (Å, comma-separated)",
            key="s_sweep_distances_str",
            help="e.g. '12,18' — 8–16 Å has little influence, keep to 1–2 values.",
        )
        try:
            sw_distances = sorted({float(x.strip()) for x in sw_distances_str.split(",") if x.strip()})
            if not sw_distances:
                sw_distances = [12.0]
        except ValueError:
            st.error("Invalid distances.")
            sw_distances = [12.0]

    # ── Seed orientation ──────────────────────────────────────────────────────
    _oc1, _oc2 = st.columns(2)
    with _oc1:
        sw_n_spins = st.slider(
            "Seed spin orientations per direction",
            min_value=1, max_value=16, step=1,
            key="s_sweep_n_spins",
            help="Evenly spaced spin rotations around the placement direction axis.",
        )
    with _oc2:
        sw_tilts_str = st.text_input(
            "Tilt angles (°, comma-separated)",
            key="s_sweep_tilts_str",
            help="e.g. '0,45,90' — tilt rotates the seed around a perpendicular to the placement direction.",
        )
        try:
            sw_tilts = sorted({float(x.strip()) for x in sw_tilts_str.split(",") if x.strip()})
            if not sw_tilts:
                sw_tilts = [0.0]
        except ValueError:
            st.error("Invalid tilt angles.")
            sw_tilts = [0.0]

    # ── Budget ────────────────────────────────────────────────────────────────
    sw_samples = st.number_input(
        "Samples per config",
        min_value=10, max_value=2000, step=10,
        key="s_sweep_samples_per_config",
        help="Keep low (50–200) for scouting. Raise once you've found promising regions.",
    )

    # ── Preview ───────────────────────────────────────────────────────────────
    _sw_n_cone_anchors = (
        (1 if sw_centroid else 0)
        + (len(hotspot_ids) if sw_individual else 0)
    )
    _sw_n_sphere_anchors = int(st.session_state.s_sweep_sphere_npoints) if sw_sphere else 0
    _sw_n_dirs = sum(1 if a <= 1e-3 else sw_n_azimuths for a in sw_cone_angles)
    _sw_n_configs = count_sweep_configs(
        n_cone_anchors=_sw_n_cone_anchors,
        n_sphere_anchors=_sw_n_sphere_anchors,
        cone_angles=sw_cone_angles,
        n_azimuths=sw_n_azimuths,
        n_distances=len(sw_distances),
        n_spins=sw_n_spins,
        n_tilts=len(sw_tilts),
    )
    _sw_info_parts = []
    if _sw_n_cone_anchors:
        _sw_info_parts.append(f"**{_sw_n_cone_anchors} cone anchors** × **{_sw_n_dirs} dirs**")
    if _sw_n_sphere_anchors:
        _sw_info_parts.append(f"**{_sw_n_sphere_anchors} sphere anchors** (own direction)")
    st.info(
        " + ".join(_sw_info_parts)
        + f" × **{len(sw_distances)} dist** × **{sw_n_spins} spins** × **{len(sw_tilts)} tilts**"
        + f" = **{_sw_n_configs} configs** × {sw_samples} = **{_sw_n_configs * sw_samples:,} total samples**"
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    if st.button("🎯 Generate sweep → add to batch queue", use_container_width=True):
        if st.session_state._generating_sweep:
            st.stop()
        st.session_state._generating_sweep = True

        if st.session_state.current_direction is None:
            st.error("Run a placement in Section 3 first to register the surface normal direction.")
            st.session_state._generating_sweep = False
            st.stop()

        _sw_centroid = hotspot_centroid(target, hotspot_ids)
        _sw_normal   = compute_surface_normal(target, _sw_centroid)
        _sw_hotspot_cas = [
            target[(target.res_id == _rid) & (target.atom_name == "CA")].coord[0]
            for _rid in hotspot_ids
            if len(target[(target.res_id == _rid) & (target.atom_name == "CA")]) > 0
        ]
        _sw_binder_name = st.session_state.s_binder_name
        _sw_output_dir  = st.session_state.s_output_dir

        _sw_configs = generate_sweep_configs(
            centroid=_sw_centroid,
            surface_normal=_sw_normal,
            hotspot_cas=_sw_hotspot_cas,
            seed_type=seed_type,
            custom_pdb=(custom_pdb_input or None) if seed_type == "custom" else None,
            use_centroid=bool(sw_centroid),
            use_individual=bool(sw_individual),
            use_sphere=bool(sw_sphere),
            sphere_n=int(st.session_state.s_sweep_sphere_npoints),
            sphere_r=float(st.session_state.s_sweep_sphere_radius),
            cone_angles=sw_cone_angles,
            n_azimuths=int(sw_n_azimuths),
            distances=sw_distances,
            n_spins=int(sw_n_spins),
            tilts=sw_tilts,
        )

        _sw_added, _sw_skipped = 0, []
        _existing_labels = {item["label"] for item in st.session_state.batch_queue}
        for _cfg in _sw_configs:
            try:
                _sw_linked, _sw_nseed = place_seed(
                    target,
                    _cfg["anchor_pos"], _cfg["direction"], _cfg["distance"],
                    _cfg["seed_type"], _cfg["rotation_deg"], _cfg["tilt_deg"],
                    _cfg["custom_pdb"],
                )
                _sw_seed_atoms = _sw_linked[_sw_linked.res_id > st.session_state.target_length]
                if has_clash(target, _sw_seed_atoms):
                    _sw_skipped.append(
                        f"{_cfg['anchor_label']}: seed within {CLASH_THRESHOLD_DEFAULT:.0f} Å of target"
                    )
                    continue

                _tilt_str = f"_tilt{_cfg['tilt_deg']:.0f}" if (len(sw_tilts) > 1 or abs(_cfg['tilt_deg']) > 0.5) else ""
                _sw_suffix = f"_sw_{_cfg['anchor_label']}_s{_cfg['rotation_deg']:.0f}{_tilt_str}"
                _sw_label = _sw_binder_name + _sw_suffix
                if _sw_label in _existing_labels:
                    _sw_skipped.append(
                        f"{_cfg['anchor_label']}: already in batch queue (duplicate)"
                    )
                    continue

                st.session_state.batch_queue.append({
                    "label":            _sw_label,
                    "suffix":           _sw_suffix,
                    "base_binder_name": _sw_binder_name,
                    "base_output_dir":  _sw_output_dir,
                    "n_seed":           _sw_nseed,
                    "rotation":         _cfg["rotation_deg"],
                    "tilt":             _cfg["tilt_deg"],
                    "pdb_string":       structure_to_pdb_string(_sw_linked),
                    "settings_snapshot": {
                        "n_samples":         int(sw_samples),
                        "batch_size":        int(st.session_state.s_batch_size),
                        "binder_length_min": int(st.session_state.s_binder_len_min),
                        "binder_length_max": int(st.session_state.s_binder_len_max),
                        "keep_linker_end_n": int(st.session_state.s_keep_linker_end_n),
                        "laproteina_mode":   st.session_state.s_laproteina_mode,
                        "rosetta_relax":     bool(st.session_state.s_rosetta_relax),
                        "checkpoint_path":   st.session_state.s_checkpoint_path,
                        "autoencoder_path":  st.session_state.s_autoencoder_path,
                        "folding_model":     st.session_state.s_folding_model,
                        "protenix_model":    st.session_state.s_protenix_model,
                        "target_msa":        st.session_state.s_target_msa,
                        "hotspot_str":       hotspot_str.strip(),
                        "hotspot_positions_str": hotspot_positions_str,
                        "target_length":     st.session_state.target_length,
                    },
                })
                _existing_labels.add(_sw_label)
                _sw_added += 1
            except Exception as _e:
                _sw_skipped.append(f"{_cfg['anchor_label']}: {_e}")

        if _sw_skipped:
            with st.expander(f"{len(_sw_skipped)} configs skipped"):
                st.text("\n".join(_sw_skipped))
        st.success(
            f"Added **{_sw_added}** sweep configs to batch queue "
            f"({len(_sw_configs) - _sw_added} skipped). "
            f"Total in queue: **{len(st.session_state.batch_queue)}**."
        )
        st.session_state._generating_sweep = False

        # Build placement point spheres (deduplicate by anchor+direction+distance, ignore spin)
        _anchor_color = {"cen": "#f4a261", "hs": "#ffd166", "sp": "#4ecdc4"}
        _seen_pts: set = set()
        _preview_pts = []
        for _cfg in _sw_configs:
            _pt_key = (_cfg["anchor_label"], _cfg["distance"])
            if _pt_key in _seen_pts:
                continue
            _seen_pts.add(_pt_key)
            _pt = _cfg["anchor_pos"] + _cfg["direction"] * _cfg["distance"]
            _lbl = _cfg["anchor_label"]
            _col = next(
                (v for k, v in _anchor_color.items() if _lbl.startswith(k)),
                "#aaaaaa",
            )
            _preview_pts.append({"x": float(_pt[0]), "y": float(_pt[1]), "z": float(_pt[2]),
                                  "color": _col, "radius": 1.2})
        st.session_state.sweep_preview_points = _preview_pts

    # ── Sweep placement preview viewer ────────────────────────────────────────
    if st.session_state.sweep_preview_points and st.session_state.target_pdb_str:
        st.caption(
            "🟠 centroid · 🟡 individual hotspot residue · 🩵 Fibonacci sphere  "
            "— each sphere is a seed placement point"
        )
        _pv = make_sweep_preview_viewer(
            st.session_state.target_pdb_str,
            hotspot_res_ids=hotspot_ids,
            placement_spheres=st.session_state.sweep_preview_points,
        )
        _show_viewer(view_to_html(_pv), height=460)

# ---------------------------------------------------------------------------
# ── Section 4: LaProteina Settings ──
# ---------------------------------------------------------------------------

st.header("4 · LaProteina Settings")

if not st.session_state.linked_pdb_strings:
    st.info("Run placement first to enable settings generation.")
    st.stop()

n_seed = st.session_state.linked_structures[0][1]
# Sync keep_linker_end_n default to current seed size (only when a new placement runs)
if st.session_state.s_keep_linker_end_n > n_seed:
    st.session_state.s_keep_linker_end_n = n_seed

col_a, col_b = st.columns(2)
with col_a:
    binder_name = st.text_input(
        "Binder name",
        key="s_binder_name",
        help="Used as output directory name.",
    )
    n_samples  = st.number_input("n_samples",  min_value=1,  step=100, key="s_n_samples")
    batch_size = st.number_input("batch_size", min_value=1,  max_value=16, key="s_batch_size")
    binder_len_min = st.number_input("Binder min length", min_value=1, key="s_binder_len_min")
    binder_len_max = st.number_input("Binder max length", min_value=1, key="s_binder_len_max")

with col_b:
    keep_linker_end_n = st.number_input(
        "keep_linker_end_n",
        min_value=0,
        max_value=n_seed,
        key="s_keep_linker_end_n",
        help="Residues from the seed C-terminus to keep as binder start. Default = full seed.",
    )
    laproteina_mode = st.selectbox(
        "laproteina_mode",
        ["all_atom", "tip_atoms", "ca", "bb3o"],
        index=["all_atom", "tip_atoms", "ca", "bb3o"].index(
            st.session_state.s_laproteina_mode
        ),
        key="s_laproteina_mode",
    )
    rosetta_relax = st.checkbox("rosetta_relax", key="s_rosetta_relax")

    st.markdown("**Cluster paths** (from config.yaml — edit to override)")
    checkpoint_path  = st.text_input("checkpoint_path",  key="s_checkpoint_path")
    autoencoder_path = st.text_input("autoencoder_path", key="s_autoencoder_path")

tr = get_path_translator()

st.subheader("Folding model")
_folding_col, _protenix_col = st.columns(2)

with _folding_col:
    folding_model = st.selectbox(
        "Folding model",
        ["af2", "protenix"],
        index=["af2", "protenix"].index(st.session_state.s_folding_model),
        key="s_folding_model",
        help="af2 = AlphaFold2-Multimer  ·  protenix = ByteDance Protenix",
    )

with _protenix_col:
    _protenix_models = _d["protenix_models"]
    if folding_model == "protenix":
        _cur = st.session_state.s_protenix_model
        _idx = _protenix_models.index(_cur) if _cur in _protenix_models else 0
        protenix_model = st.selectbox(
            "Protenix model version",
            _protenix_models,
            index=_idx,
            key="s_protenix_model",
        )
        target_msa = st.text_input(
            "Target MSA path (optional, cluster path to .a3m)",
            key="s_target_msa",
            help="Pre-computed MSA for the target. Leave blank to skip.",
        )
        use_template = st.checkbox(
            "Use target PDB as Protenix template",
            key="s_protenix_use_template",
            help="Passes the original target PDB as a structural template to Protenix "
                 "(templatesPath). Helps preserve the target conformation and typically "
                 "improves sc_RMSD.",
        )
        if use_template:
            _orig = st.session_state.s_original_pdb_local
            _tpl_default = tr.to_cluster(_orig) if _orig else ""
            target_template = st.text_input(
                "Template PDB path (cluster path)",
                value=_tpl_default,
                help="Auto-filled from the target PDB loaded in Section 1. Edit if needed.",
            )
        else:
            target_template = ""
    else:
        protenix_model = _d["protenix_model"]
        target_msa = ""
        target_template = ""
        st.caption("Protenix options only shown when protenix is selected.")

# Auto-suggest output dir when binder name changes; user can override freely
_output_dir_default = f"{_d['default_output_dir']}/{binder_name}"
if st.session_state.s_output_dir == _d["default_output_dir"] or not st.session_state.s_output_dir:
    st.session_state.s_output_dir = _output_dir_default
output_dir_cluster = st.text_input("Output dir (cluster path)", key="s_output_dir")

# ---------------------------------------------------------------------------
# ── Section 5: Batch Queue ──
# ---------------------------------------------------------------------------

st.header("5 · Batch Queue")
st.caption(
    "Accumulate multiple seed placements and submit them as a single SLURM array job. "
    "Place seeds in Section 3, adjust settings in Section 4, then add to the queue."
)

if st.button("➕ Add current placements to batch queue", use_container_width=True):
    if st.session_state._adding_placements:
        st.stop()
    st.session_state._adding_placements = True

    _multi = len(st.session_state.linked_structures) > 1
    _added = 0
    _skipped = 0
    _existing_labels = {item["label"] for item in st.session_state.batch_queue}
    for (_linked_s, _n_seed_i, _rot, _tilt), _pdb_str in zip(
        st.session_state.linked_structures, st.session_state.linked_pdb_strings
    ):
        _spin_str = f"_rot{_rot:.0f}" if _multi else ""
        _tilt_str = f"_tilt{_tilt:.0f}" if abs(_tilt) > 0.5 else ""
        _suffix = _spin_str + _tilt_str
        _label = f"{binder_name}{_suffix}"
        if _label in _existing_labels:
            _skipped += 1
            continue
        st.session_state.batch_queue.append({
            "label":            _label,
            "suffix":           _suffix,
            "base_binder_name": binder_name,
            "base_output_dir":  output_dir_cluster,
            "n_seed":           _n_seed_i,
            "rotation":         _rot,
            "tilt":             _tilt,
            "pdb_string":       _pdb_str,
            "settings_snapshot": {
                "n_samples":         int(st.session_state.s_n_samples),
                "batch_size":        int(st.session_state.s_batch_size),
                "binder_length_min": int(st.session_state.s_binder_len_min),
                "binder_length_max": int(st.session_state.s_binder_len_max),
                "keep_linker_end_n": int(st.session_state.s_keep_linker_end_n),
                "laproteina_mode":   st.session_state.s_laproteina_mode,
                "rosetta_relax":     bool(st.session_state.s_rosetta_relax),
                "checkpoint_path":   st.session_state.s_checkpoint_path,
                "autoencoder_path":  st.session_state.s_autoencoder_path,
                "folding_model":          st.session_state.s_folding_model,
                "protenix_model":         st.session_state.s_protenix_model,
                "target_msa":             st.session_state.s_target_msa,
                "protenix_use_template":  st.session_state.s_protenix_use_template,
                "original_pdb_local":     st.session_state.s_original_pdb_local,
                "hotspot_str":            hotspot_str.strip(),
                "hotspot_positions_str":  hotspot_positions_str,
                "target_length":     st.session_state.target_length,
            },
        })
        _existing_labels.add(_label)
        _added += 1
    _msg = f"Added {_added} placement(s). Queue total: {len(st.session_state.batch_queue)}"
    if _skipped > 0:
        _msg += f" ({_skipped} duplicate(s) skipped)"
    st.success(_msg)
    st.session_state._adding_placements = False

# Queue display
if st.session_state.batch_queue:
    st.markdown(f"**{len(st.session_state.batch_queue)} item(s) in queue:**")
    _to_remove = None
    for _qi, _qitem in enumerate(st.session_state.batch_queue):
        _c1, _c2 = st.columns([7, 1])
        with _c1:
            st.markdown(
                f"`{_qi + 1}.` **{_qitem['label']}** — "
                f"seed_len={_qitem['n_seed']}, "
                f"spin={_qitem['rotation']:.0f}°, tilt={_qitem.get('tilt', 0):.0f}°"
            )
        with _c2:
            if st.button("✕", key=f"rm_batch_{_qi}", help="Remove from queue"):
                _to_remove = _qi
    if _to_remove is not None:
        st.session_state.batch_queue.pop(_to_remove)
        st.rerun()

    col_clr, _ = st.columns([2, 5])
    with col_clr:
        if st.button("🗑 Clear queue"):
            st.session_state.batch_queue = []
            st.rerun()

    st.markdown("---")
    st.subheader("Generate batch array job")

    # Auto-populate save dir from first queue item the first time
    if not st.session_state.s_batch_save_dir and st.session_state.batch_queue:
        st.session_state.s_batch_save_dir = tr.to_local(
            st.session_state.batch_queue[0]["base_output_dir"]
        ).rstrip("/")

    _batch_save_dir = st.text_input(
        "Save directory (local NFS path — all batch PDBs, JSONs, and manifest go here)",
        key="s_batch_save_dir",
    )
    _max_concurrent = st.number_input(
        "Max concurrent SLURM array tasks",
        min_value=1, max_value=50, step=1,
        key="s_batch_max_concurrent",
    )

    if st.button("🚀 Save all + generate batch job", use_container_width=True):
        if not _batch_save_dir.strip():
            st.error("Enter a save directory.")
            st.stop()
        _bsave = Path(_batch_save_dir.strip())
        try:
            _bsave.mkdir(parents=True, exist_ok=True)
        except Exception as _e:
            st.error(f"Cannot create directory {_bsave}: {_e}")
            st.stop()

        _batch_tpl_cluster = None
        _batch_protenix_mt_cmd = None
        for _item in st.session_state.batch_queue:
            _ss0 = _item["settings_snapshot"]
            if _ss0.get("protenix_use_template") and _ss0.get("folding_model") == "protenix" and _ss0.get("original_pdb_local"):
                _orig = Path(_ss0["original_pdb_local"])
                if _orig.exists():
                    _tpl_copy = _bsave / _orig.name
                    if not _tpl_copy.exists():
                        shutil.copy2(_orig, _tpl_copy)
                    _target_seq = _pdb_sequence(str(_tpl_copy), chain_id)
                    _tpl_input_local = _bsave / "template_input.json"
                    with open(_tpl_input_local, "w") as _fj:
                        json.dump([{"name": "target", "sequences": [{"proteinChain": {"count": 1, "sequence": _target_seq, "modifications": []}}]}], _fj, indent=2)
                    _tpl_input_cluster = tr.to_cluster(str(_tpl_input_local))
                    _tpl_search_dir_cluster = tr.to_cluster(str(_bsave / "template_search"))
                    _batch_tpl_cluster = f"{_tpl_search_dir_cluster}/target/msa/0/hmmsearch.a3m"
                    _batch_protenix_mt_cmd = f"python scripts/prepare_template.py -i {_tpl_input_cluster} -o {_tpl_search_dir_cluster}"
                break

        _cluster_json_paths = []
        for _item in st.session_state.batch_queue:
            _bn = _item["label"]
            _pdb_file  = str(_bsave / f"{_bn}_linked.pdb")
            _json_file = str(_bsave / f"{_bn}_settings.json")

            with open(_pdb_file, "w") as _f:
                _f.write(_item["pdb_string"])

            _ss = _item["settings_snapshot"]
            _settings = build_settings(
                binder_name=_bn,
                starting_pdb_local=_pdb_file,
                output_dir_cluster=_item["base_output_dir"] + _item["suffix"],
                target_length=_ss["target_length"],
                hotspot_residues_str=_ss["hotspot_str"],
                hotspot_positions_str=_ss.get("hotspot_positions_str", ""),
                linker_length=_item["n_seed"],
                keep_linker_end_n=_ss["keep_linker_end_n"],
                n_samples=_ss["n_samples"],
                batch_size=_ss["batch_size"],
                binder_length_min=_ss["binder_length_min"],
                binder_length_max=_ss["binder_length_max"],
                laproteina_mode=_ss["laproteina_mode"],
                checkpoint_path=_ss["checkpoint_path"],
                autoencoder_path=_ss["autoencoder_path"],
                folding_model=_ss["folding_model"],
                protenix_model=_ss["protenix_model"] if _ss["folding_model"] == "protenix" else None,
                target_msa=_ss["target_msa"] if _ss["folding_model"] == "protenix" else None,
                target_template=(
                    _batch_tpl_cluster
                    if _ss["folding_model"] == "protenix" and _ss.get("protenix_use_template")
                    else None
                ),
                rosetta_relax=_ss["rosetta_relax"],
            )
            save_settings(_settings, _json_file)
            _cluster_json_paths.append(tr.to_cluster(_json_file))

        # Deduplicate paths while preserving order
        _unique_paths = list(dict.fromkeys(_cluster_json_paths))

        _manifest_local = str(_bsave / f"batch_manifest_{_ts()}.txt")
        save_batch_manifest(_unique_paths, _manifest_local)
        _manifest_cluster = tr.to_cluster(_manifest_local)

        st.success(
            f"Saved {len(_unique_paths)} configuration(s) + manifest to `{_bsave}`"
        )
        st.subheader("SLURM commands")
        if _batch_protenix_mt_cmd:
            st.markdown("**Step 1 — Template preparation** (run on a cluster login node with internet access):")
            st.code(_batch_protenix_mt_cmd, language="bash")
            st.markdown("**Step 2 — Binder generation** (run after Step 1 completes):")
        else:
            st.caption("Copy this to your cluster terminal:")
        st.code(
            batch_slurm_command(_manifest_cluster, len(_unique_paths), int(_max_concurrent)),
            language="bash",
        )
        with st.expander("Manifest contents"):
            st.code("\n".join(_unique_paths), language="text")

# ---------------------------------------------------------------------------
# ── Section 6: Output ──
# ---------------------------------------------------------------------------

st.header("6 · Output")

_local_default = tr.to_local(output_dir_cluster)
if not st.session_state.s_save_dir_local:
    st.session_state.s_save_dir_local = _local_default
save_dir_local = st.text_input(
    "Save directory (local path on NFS mount)",
    key="s_save_dir_local",
    help="Linked PDB and settings JSON are written here.",
)

if st.button("💾 Save files + show SLURM commands", use_container_width=True):
    if not binder_name.strip():
        st.error("Enter a binder name.")
        st.stop()

    save_path = Path(save_dir_local)
    try:
        save_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Cannot create directory {save_path}: {e}")
        st.stop()

    _single_tpl_cluster = None
    _single_protenix_mt_cmd = None
    if folding_model == "protenix" and use_template and st.session_state.s_original_pdb_local:
        _orig = Path(st.session_state.s_original_pdb_local)
        if _orig.exists():
            _tpl_copy = save_path / _orig.name
            if not _tpl_copy.exists():
                shutil.copy2(_orig, _tpl_copy)
            _target_seq = _pdb_sequence(str(_tpl_copy), chain_id)
            _tpl_input_local = save_path / f"{binder_name}_template_input.json"
            with open(_tpl_input_local, "w") as _fj:
                json.dump([{"name": "target", "sequences": [{"proteinChain": {"count": 1, "sequence": _target_seq, "modifications": []}}]}], _fj, indent=2)
            _tpl_input_cluster = tr.to_cluster(str(_tpl_input_local))
            _tpl_search_dir_cluster = tr.to_cluster(str(save_path / "template_search"))
            _single_tpl_cluster = f"{_tpl_search_dir_cluster}/target/msa/0/hmmsearch.a3m"
            _single_protenix_mt_cmd = f"python scripts/prepare_template.py -i {_tpl_input_cluster} -o {_tpl_search_dir_cluster}"

    saved_pairs = []
    for i, (linked, n_seed_i, rot, tilt_i) in enumerate(st.session_state.linked_structures):
        _spin_str = f"_rot{rot:.0f}" if len(st.session_state.linked_structures) > 1 else ""
        _tilt_str = f"_tilt{tilt_i:.0f}" if abs(tilt_i) > 0.5 else ""
        suffix = _spin_str + _tilt_str
        pdb_filename  = f"{binder_name}{suffix}_linked.pdb"
        json_filename = f"{binder_name}{suffix}_settings.json"

        pdb_local  = str(save_path / pdb_filename)
        json_local = str(save_path / json_filename)

        # Save PDB
        save_linked_pdb(linked, pdb_local)

        # Build and save settings JSON
        settings = build_settings(
            binder_name=f"{binder_name}{suffix}",
            starting_pdb_local=pdb_local,
            output_dir_cluster=f"{output_dir_cluster}{suffix}",
            target_length=st.session_state.target_length,
            hotspot_residues_str=hotspot_str.strip(),
            hotspot_positions_str=hotspot_positions_str,
            linker_length=n_seed_i,
            keep_linker_end_n=int(keep_linker_end_n),
            n_samples=int(n_samples),
            batch_size=int(batch_size),
            binder_length_min=int(binder_len_min),
            binder_length_max=int(binder_len_max),
            laproteina_mode=laproteina_mode,
            checkpoint_path=checkpoint_path,
            autoencoder_path=autoencoder_path,
            folding_model=folding_model,
            protenix_model=protenix_model if folding_model == "protenix" else None,
            target_msa=target_msa if folding_model == "protenix" else None,
            target_template=_single_tpl_cluster if folding_model == "protenix" else None,
            rosetta_relax=rosetta_relax,
        )
        save_settings(settings, json_local)
        saved_pairs.append((pdb_local, json_local, settings))

    st.success(f"Saved {len(saved_pairs)} set(s) to `{save_path}`")

    # Show SLURM commands
    st.subheader("SLURM commands")
    if _single_protenix_mt_cmd:
        st.markdown("**Step 1 — Template preparation** (run on a cluster login node with internet access):")
        st.code(_single_protenix_mt_cmd, language="bash")
        st.markdown("**Step 2 — Binder generation** (run after Step 1 completes):")
    else:
        st.caption("Copy these to your cluster terminal:")
    cmds = []
    for pdb_local, json_local, settings in saved_pairs:
        json_cluster = tr.to_cluster(json_local)
        cmd = slurm_command(json_cluster)
        cmds.append(cmd)
    st.code("\n".join(cmds), language="bash")

    # Expandable JSON previews
    for pdb_local, json_local, settings in saved_pairs:
        label = Path(json_local).name
        with st.expander(f"📄 {label}"):
            st.json(settings)
