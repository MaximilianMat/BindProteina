"""
Default values for settings fields, loaded from config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_defaults() -> Dict[str, Any]:
    cfg = load_config()
    d = cfg.get("defaults", {})
    ck = cfg.get("checkpoints", {})
    paths = cfg.get("paths", {})
    protenix_models = cfg.get("protenix", {}).get("models", ["protenix-v2"])
    return {
        "n_samples":               d.get("n_samples", 4000),
        "batch_size":              d.get("batch_size", 2),
        "binder_length_min":       d.get("binder_length_min", 30),
        "binder_length_max":       d.get("binder_length_max", 80),
        "n_linker_starting_points": d.get("n_linker_starting_points", 1),
        "laproteina_mode":         d.get("laproteina_mode", "all_atom"),
        "inference_config":        d.get("inference_config", "inference_motif_idx_aa"),
        "rosetta_relax":           d.get("rosetta_relax", True),
        "checkpoint_path":         ck.get("laproteina", ""),
        "autoencoder_path":        ck.get("autoencoder", ""),
        "default_output_dir":      paths.get("default_output_dir", "/nfs/students/matscher/samples"),
        "default_input_dir":       paths.get("default_input_dir", "/nfs/homedirs/matscher/samples"),
        "folding_model":           d.get("folding_model", "af2"),
        "protenix_model":          d.get("protenix_model", protenix_models[0]),
        "protenix_models":         protenix_models,
    }


def get_protenix_models() -> list:
    cfg = load_config()
    return cfg.get("protenix", {}).get("models", ["protenix-v2"])


def get_path_translator():
    from .paths import PathTranslator
    cfg = load_config()
    cl = cfg.get("cluster", {})
    return PathTranslator(
        local_prefix=cl.get("nfs_local_prefix", ""),
        cluster_prefix=cl.get("nfs_cluster_prefix", "/nfs"),
    )
