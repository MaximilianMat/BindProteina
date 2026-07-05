"""
Build a settings_target JSON that matches the PDL1_linked.json format exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .templates import get_defaults, get_path_translator


def build_settings(
    # --- identity ---
    binder_name: str,
    # --- paths ---
    starting_pdb_local: str,      # local path to the linked PDB we just wrote
    output_dir_cluster: Optional[str] = None,
    input_dir_cluster: Optional[str] = None,
    # --- target ---
    target_chain: str = "A",
    target_length: Optional[int] = None,    # auto-derived if not given
    hotspot_residues_str: str = "",         # e.g. "50-107" (author numbering, human-readable)
    hotspot_positions_str: str = "",        # e.g. "1-58" (1-based sequential; matches the
                                            # placer's target renumbering — used by the scorer)
    # --- linker / seed ---
    linker_length: int = 1,
    keep_linker_end_n: Optional[int] = None,  # defaults to linker_length
    # --- generation ---
    n_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    binder_length_min: Optional[int] = None,
    binder_length_max: Optional[int] = None,
    n_linker_starting_points: Optional[int] = None,
    # --- laproteina model ---
    laproteina_mode: Optional[str] = None,
    inference_config: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    autoencoder_path: Optional[str] = None,
    # --- folding model ---
    folding_model: Optional[str] = None,     # "af2" or "protenix"
    protenix_model: Optional[str] = None,    # only used when folding_model="protenix"
    protenix_n_diffusion_samples: Optional[int] = None,  # diffusion samples per design (default 5; 3 ≈ 40% faster)
    target_msa: Optional[str] = None,        # optional cluster path to .a3m MSA file
    target_template: Optional[str] = None,   # optional cluster path to target PDB for Protenix templating
    # --- misc ---
    rosetta_relax: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Return a settings dict matching the PDL1_linked.json schema.
    All None values fall back to config.yaml defaults.
    """
    d = get_defaults()
    tr = get_path_translator()

    # Resolve optional args
    n_samples         = n_samples         or d["n_samples"]
    batch_size        = batch_size        or d["batch_size"]
    binder_length_min = binder_length_min or d["binder_length_min"]
    binder_length_max = binder_length_max or d["binder_length_max"]
    n_lsp             = n_linker_starting_points or d["n_linker_starting_points"]
    laproteina_mode   = laproteina_mode   or d["laproteina_mode"]
    inference_config  = inference_config  or d["inference_config"]
    checkpoint_path   = checkpoint_path   or d["checkpoint_path"]
    autoencoder_path  = autoencoder_path  or d["autoencoder_path"]
    folding_model     = folding_model     or d["folding_model"]
    protenix_model    = protenix_model    or d["protenix_model"]
    rosetta_relax     = rosetta_relax if rosetta_relax is not None else d["rosetta_relax"]
    keep_linker_end_n = keep_linker_end_n if keep_linker_end_n is not None else linker_length

    output_dir_cluster = output_dir_cluster or f"{d['default_output_dir']}/{binder_name}"
    input_dir_cluster  = input_dir_cluster  or d["default_input_dir"]

    starting_pdb_cluster = tr.to_cluster(starting_pdb_local)

    settings: Dict[str, Any] = {
        "input_dir":          input_dir_cluster,
        "output_dir":         output_dir_cluster,
        "starting_pdb":       starting_pdb_cluster,
        "chains":             target_chain,
        "binder_name":        binder_name,
        "linked_input_mode":  True,
        "direct_binder_mode": False,
        "linker_length":      linker_length,
        "keep_linker_end_n":  keep_linker_end_n,
        "laproteina_settings": {
            "checkpoint_path":          checkpoint_path,
            "autoencoder_path":         autoencoder_path,
            "laproteina_mode":          laproteina_mode,
            "inference_config":         inference_config,
            "n_samples":                n_samples,
            "batch_size":               batch_size,
            "n_linker_starting_points": n_lsp,
            "binder_length_range":      [binder_length_min, binder_length_max],
        },
        "rosetta_relax":  rosetta_relax,
        "folding_model":  folding_model,
    }

    # Protenix-specific fields — only added when protenix is selected
    if folding_model == "protenix":
        settings["protenix_model"] = protenix_model
        if protenix_n_diffusion_samples is not None:
            settings["protenix_n_diffusion_samples"] = protenix_n_diffusion_samples
        if target_msa and target_msa.strip():
            settings["target_msa"] = target_msa.strip()
        if target_template and target_template.strip():
            settings["target_template"] = target_template.strip()

    # Optional fields
    if target_length is not None:
        settings["target_length"] = target_length
    if hotspot_residues_str.strip():
        settings["target_hotspot_residues"] = hotspot_residues_str.strip()
    # 1-based sequential hotspot positions (matches the placer's target
    # renumbering) — this is what the scorer's hotspot-contact check consumes,
    # since the generated/refolded structures are numbered 1..N, not by the
    # original (possibly non-1-based) author numbering.
    if hotspot_positions_str.strip():
        settings["target_hotspot_positions"] = hotspot_positions_str.strip()

    return settings


def settings_to_json(settings: Dict[str, Any], indent: int = 4) -> str:
    return json.dumps(settings, indent=indent)


def save_settings(settings: Dict[str, Any], output_path: str) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(settings, f, indent=4)
    return output_path


def slurm_command(settings_cluster_path: str, slurm_script: str = "laproteina_generate.slurm") -> str:
    return f"sbatch {slurm_script} --settings {settings_cluster_path}"


def save_batch_manifest(settings_cluster_paths: List[str], output_path: str) -> str:
    """Write a newline-delimited manifest file (one cluster JSON path per line)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(settings_cluster_paths) + "\n")
    return output_path


def batch_slurm_command(
    manifest_cluster_path: str, n_jobs: int, max_concurrent: int = 6
) -> str:
    """Return the sbatch command for a SLURM array job over the given manifest."""
    return (
        f"sbatch --array=0-{n_jobs - 1}%{max_concurrent} "
        f"laproteina_generate_batch.slurm --batch {manifest_cluster_path}"
    )
