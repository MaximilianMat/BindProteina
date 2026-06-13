"""
py3Dmol-based viewer for Streamlit.

Renders target structure with hotspot highlights and seed overlay.
"""

from __future__ import annotations

from typing import List, Optional

import py3Dmol


def make_viewer(
    target_pdb: str,
    hotspot_res_ids: Optional[List[int]] = None,
    anchor_res_id: Optional[int] = None,
    seed_pdb: Optional[str] = None,
    seed_res_ids: Optional[List[int]] = None,
    width: int = 780,
    height: int = 480,
) -> py3Dmol.view:
    """
    Build a py3Dmol view showing:
      - Target: grey cartoon
      - Hotspot residues: red transparent spheres
      - Anchor residue: orange sphere (slightly larger)
      - Seed atoms: cyan sticks

    For use in Streamlit via:
        from viewer.render import make_viewer, view_to_html
        import streamlit.components.v1 as components
        components.html(view_to_html(make_viewer(...)), height=500)
    """
    view = py3Dmol.view(width=width, height=height)

    # --- Target ---
    view.addModel(target_pdb, "pdb")
    view.setStyle({}, {"cartoon": {"color": "lightgray", "opacity": 0.85}})
    # Show backbone sticks as well for clarity near the hotspot
    view.addStyle({}, {"stick": {"color": "lightgray", "radius": 0.05, "opacity": 0.3}})

    # --- Hotspot residues ---
    if hotspot_res_ids:
        for res_id in hotspot_res_ids:
            view.addStyle(
                {"resi": str(res_id)},
                {"sphere": {"color": "#e63946", "opacity": 0.5, "radius": 1.4}},
            )
        # Stick representation on top so atoms are visible
        view.addStyle(
            {"resi": [str(r) for r in hotspot_res_ids]},
            {"stick": {"color": "#e63946", "radius": 0.15}},
        )

    # --- Anchor residue (highlighted stronger) ---
    if anchor_res_id is not None:
        view.addStyle(
            {"resi": str(anchor_res_id)},
            {"sphere": {"color": "#f4a261", "opacity": 0.85, "radius": 1.8}},
        )

    # --- Seed fragment (second model) ---
    if seed_pdb:
        view.addModel(seed_pdb, "pdb")
        seed_model_idx = 1  # zero-indexed; target is 0
        view.addStyle(
            {"model": seed_model_idx},
            {"stick": {"color": "#06d6a0", "radius": 0.25}},
        )
        view.addStyle(
            {"model": seed_model_idx, "atom": "CA"},
            {"sphere": {"color": "#06d6a0", "radius": 0.55}},
        )

    view.zoomTo()
    view.spin(False)
    return view


def make_sweep_preview_viewer(
    target_pdb: str,
    hotspot_res_ids: Optional[List[int]] = None,
    placement_spheres: Optional[List[dict]] = None,
    width: int = 780,
    height: int = 480,
) -> py3Dmol.view:
    """
    Show target with hotspot highlights and colored spheres at seed placement points.

    placement_spheres: list of dicts with keys x, y, z, color, radius.
      Color convention:
        orange (#f4a261) → centroid anchor
        yellow (#ffd166) → individual hotspot residue anchor
        teal   (#4ecdc4) → Fibonacci sphere anchor
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(target_pdb, "pdb")
    view.setStyle({}, {"cartoon": {"color": "lightgray", "opacity": 0.85}})
    view.addStyle({}, {"stick": {"color": "lightgray", "radius": 0.05, "opacity": 0.3}})

    if hotspot_res_ids:
        for res_id in hotspot_res_ids:
            view.addStyle(
                {"resi": str(res_id)},
                {"sphere": {"color": "#e63946", "opacity": 0.5, "radius": 1.4}},
            )
        view.addStyle(
            {"resi": [str(r) for r in hotspot_res_ids]},
            {"stick": {"color": "#e63946", "radius": 0.15}},
        )

    for sp in (placement_spheres or []):
        view.addSphere({
            "center": {"x": sp["x"], "y": sp["y"], "z": sp["z"]},
            "radius": sp.get("radius", 1.0),
            "color":  sp["color"],
            "opacity": 0.75,
        })

    view.zoomTo()
    view.spin(False)
    return view


def view_to_html(view: py3Dmol.view) -> str:
    """Extract embeddable HTML string from a py3Dmol view."""
    return view._make_html()


def make_linked_viewer(
    linked_pdb: str,
    target_length: int,
    hotspot_res_ids: Optional[List[int]] = None,
    anchor_res_id: Optional[int] = None,
    width: int = 780,
    height: int = 480,
) -> py3Dmol.view:
    """
    Render a linked PDB (target + seed in one chain) with color split at target_length.
    Residues 1..target_length = grey, target_length+1..end = cyan.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(linked_pdb, "pdb")

    # Target part
    view.addStyle(
        {"resi": f"1-{target_length}"},
        {"cartoon": {"color": "lightgray", "opacity": 0.85}},
    )
    view.addStyle(
        {"resi": f"1-{target_length}"},
        {"stick": {"color": "lightgray", "radius": 0.05, "opacity": 0.3}},
    )

    # Seed part
    view.addStyle(
        {"resi": f"{target_length + 1}-9999"},
        {"stick": {"color": "#06d6a0", "radius": 0.25}},
    )
    view.addStyle(
        {"resi": f"{target_length + 1}-9999", "atom": "CA"},
        {"sphere": {"color": "#06d6a0", "radius": 0.55}},
    )

    if hotspot_res_ids:
        for res_id in hotspot_res_ids:
            view.addStyle(
                {"resi": str(res_id)},
                {"sphere": {"color": "#e63946", "opacity": 0.5, "radius": 1.4}},
            )

    if anchor_res_id is not None:
        view.addStyle(
            {"resi": str(anchor_res_id)},
            {"sphere": {"color": "#f4a261", "opacity": 0.85, "radius": 1.8}},
        )

    view.zoomTo()
    view.spin(False)
    return view
