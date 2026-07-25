"""
Configuration for the re-seeding ablation.

A run is fully described by a flat JSON file in ``settings_reseed/<target>.json`` (see
``settings_reseed/EXAMPLE.json``). This module loads that JSON into a ``ReseedConfig``
dataclass, applying the spec defaults (§3, §11) for any absent key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List


# Canonical seed-cut fragmentation patterns (spec §5.2). Each pattern partitions the
# 5 seed residues into fragments swept from maximally dispersed (P1) to contiguous (P5).
CUT_PATTERNS = {
    "P1": [1, 1, 1, 1, 1],
    "P2": [2, 1, 1, 1],
    "P3": [2, 2, 1],
    "P4": [3, 2],
    "P5": [5],
}


@dataclass
class ReseedConfig:
    """Fully-resolved configuration for one target's re-seeding run."""

    target: str
    target_pdb: str
    baseline_run_dir: str          # prior blind run -> founders + RQ1 baseline
    checkpoint: str                # LaProteina main checkpoint (.ckpt)
    autoencoder_path: str          # LaProteina autoencoder checkpoint (.ckpt)
    output_base_dir: str           # runs/<target> is created under here

    # Generation / scoring
    laproteina_mode: str = "all_atom"
    inference_config: Optional[str] = None
    filter_config: str = "settings_filters/laProteina_filters.json"
    folding_model: str = "af2"     # "af2" or "protenix"
    protenix_model: str = "protenix-v2"
    device: str = "cuda"
    n_steps: int = 400             # LaProteina diffusion steps (match current runs)
    max_batch_size: int = 2

    # Founder source (prior blind run may only have the folding prefilter stage)
    #   "auto"      -> use PyRosetta successes (scoring_summary.csv) when present, and
    #                  supplement with folding-passing designs if fewer than n_strains.
    #   "pyrosetta" -> only PyRosetta-passing founders (scoring_summary.csv).
    #   "folding"   -> only folding-passing founders (protenix_prescoring_results.csv).
    founder_filter_stage: str = "auto"
    target_len: Optional[int] = None   # skip get_pdb_chain_length if provided

    # Budget (spec §3 canonical config)
    n_strains: int = 20            # S
    samples_per_cut: int = 20
    n_cuts: int = 5                # P1..P5
    seed_n_res: int = 5

    # Seed residue selection:
    #   "random"    -> up to seed_n_res residues drawn anywhere on the binder (default).
    #   "interface" -> up to seed_n_res residues drawn ONLY from binder residues that
    #                  contact the target hotspots (or the whole interface if no hotspots),
    #                  so the seed carries the actual binding geometry (helps low-yield
    #                  targets). Hotspot ids matched against the parent complex chain-A
    #                  numbering (1-based for refolded complexes). Falls back to random if
    #                  fewer than seed_n_res_min interface residues are found.
    seed_selection: str = "random"
    seed_interface_cutoff: float = 8.0        # Å, binder-residue → hotspot heavy-atom dist
    seed_n_res_min: int = 1                    # min interface residues to attempt a seed
    seed_interface_hotspots: Optional[str] = None  # e.g. "37,38,301-304"; else target_hotspot_residues
    rounds: int = 4                # R

    # Target context
    target_hotspot_residues: Optional[str] = None
    target_msa: Optional[str] = None
    target_template: Optional[str] = None

    # Champion selection strategy (which winner seeds the next round)
    #   "uniform"       -> uniform random among a strain's winners (spec §7, no pressure).
    #   "ss_stratified" -> bin the strain's winners into a champion_ss_grid x champion_ss_grid
    #                      helix x sheet grid over [0,1]^2, pick a non-empty cell uniformly,
    #                      then a winner uniformly within it -> uniform coverage of SS space
    #                      (pushes lineages into under-sampled beta regions; imposes an
    #                      SS-coverage pressure, so the uniform arm stays the no-pressure ref).
    champion_selection: str = "uniform"
    champion_ss_grid: int = 3          # bins per axis for the ss_stratified grid

    # Budget redistribution: when strains go extinct, spread their freed generation
    # budget EVENLY across the surviving strains that round (keeps ~n_strains×n_cuts×
    # samples_per_cut folds/round instead of decaying). Deepens real lineages — does not
    # respawn founders, so extinction is still tracked and generations stay interpretable.
    redistribute_budget: bool = False

    # Knobs (spec §11)
    seed_free_sequence: bool = False   # §11-c: free the seed sequence, fix only backbone
    ss_bias_target: bool = False       # §11-j: flag as a strong-SS-bias target for RQ2
    beta_heavy_cutoff: float = 0.30    # §11-j: binder-relative sheet fraction to call "beta-heavy"

    # Reproducibility
    master_seed: int = 0

    @property
    def run_dir(self) -> Path:
        """runs/<target> root for this run's outputs."""
        return Path(self.output_base_dir) / "runs" / self.target

    def to_json_dict(self) -> dict:
        return asdict(self)


def effective_samples_per_cut(base: int, n_strains: int, n_alive: int,
                              enabled: bool) -> int:
    """Per-cut sample count for a round after redistributing extinct strains' budget.

    When ``enabled`` and some strains have died (``n_alive < n_strains``), the full round
    budget ``n_strains*base`` per cut-slot is spread EVENLY over the ``n_alive`` survivors,
    so each survivor generates ``round(base*n_strains/n_alive)`` per cut. Otherwise ``base``.
    """
    if not enabled or n_alive <= 0 or n_alive >= n_strains:
        return int(base)
    return max(int(base), round(base * n_strains / n_alive))


# Keys that MUST be present in the settings JSON (no sensible default).
_REQUIRED = ("target", "target_pdb", "baseline_run_dir", "checkpoint",
             "autoencoder_path", "output_base_dir")


def load_config(settings_path: str, overrides: Optional[dict] = None) -> ReseedConfig:
    """Load a ``ReseedConfig`` from a settings JSON, applying CLI overrides.

    Relative ``settings_path`` is resolved against the repo root (two levels up).
    """
    path = Path(settings_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    missing = [k for k in _REQUIRED if not data.get(k)]
    if missing:
        raise ValueError(f"Settings {path} missing required keys: {missing}")

    # Keep only keys ReseedConfig knows about, so a settings file may carry extra notes.
    known = set(ReseedConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in known}
    cfg = ReseedConfig(**filtered)

    if cfg.champion_selection not in ("uniform", "ss_stratified"):
        raise ValueError(
            f"champion_selection must be 'uniform' or 'ss_stratified', "
            f"got {cfg.champion_selection!r}")
    if cfg.seed_selection not in ("random", "interface"):
        raise ValueError(
            f"seed_selection must be 'random' or 'interface', got {cfg.seed_selection!r}")
    if cfg.champion_ss_grid < 1:
        raise ValueError(f"champion_ss_grid must be >= 1, got {cfg.champion_ss_grid}")
    return cfg
