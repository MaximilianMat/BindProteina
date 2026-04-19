#!/usr/bin/env python
"""Run generate_initial.py while keeping autoencoder.encoder.

This script monkey-patches LaProteinaGenerator._load_model so the
autoencoder encoder is not deleted before inference, matching the
encoder-retention A/B test you requested.
"""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from pathlib import Path


def _import_generator_module():
    candidate_modules = (
        "laproteina.proteinfoundation.laproteina_generator",
        "proteinfoundation.laproteina_generator",
    )
    for module_name in candidate_modules:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "Could not import LaProteinaGenerator module. Tried: "
        + ", ".join(candidate_modules)
    )


def _import_proteina_module():
    candidate_modules = (
        "laproteina.proteinfoundation.proteina",
        "proteinfoundation.proteina",
    )
    for module_name in candidate_modules:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        "Could not import Proteina module. Tried: "
        + ", ".join(candidate_modules)
    )


def _patch_load_model_keep_encoder(generator_module, proteina_module):
    original_load_model = generator_module.LaProteinaGenerator._load_model

    def _load_model_keep_encoder(self):
        if self.model is not None:
            return

        self.model = proteina_module.Proteina.load_from_checkpoint(
            str(self.checkpoint_path),
            strict=False,
            autoencoder_ckpt_path=str(self.autoencoder_path) if self.autoencoder_path else None,
        )

        self.model.eval()
        print(
            f"[INFO] Loaded model without dropping autoencoder.encoder: {self.checkpoint_path}"
        )

    generator_module._original_load_model_for_encoder_debug = original_load_model
    generator_module.LaProteinaGenerator._load_model = _load_model_keep_encoder


def _find_generate_initial_script(repo_root: Path) -> Path:
    candidates = (
        repo_root / "laproteina" / "proteinfoundation" / "generate_initial.py",
        repo_root / "proteinfoundation" / "generate_initial.py",
        repo_root / "generate_initial.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find generate_initial.py in:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run generate_initial.py with autoencoder.encoder preserved in memory. "
            "Pass any generate_initial.py args after '--'."
        )
    )
    parser.add_argument(
        "--repo-root",
        default="/nfs/homedirs/matscher/repos/BindProteina",
        help="Absolute path to repository root.",
    )
    parser.add_argument(
        "--generate-script",
        default=None,
        help="Explicit path to generate_initial.py (optional).",
    )
    parser.add_argument(
        "generate_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to generate_initial.py.",
    )
    args = parser.parse_args()

    generate_args = args.generate_args
    if generate_args and generate_args[0] == "--":
        generate_args = generate_args[1:]
    if not generate_args:
        raise SystemExit(
            "No generate_initial.py arguments provided. "
            "Example:\n  python scripts/run_generate_initial_keep_encoder.py -- "
            "/path/to/settings.yaml"
        )

    repo_root = Path(args.repo_root).expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    generator_module = _import_generator_module()
    proteina_module = _import_proteina_module()
    _patch_load_model_keep_encoder(generator_module, proteina_module)

    generate_script = (
        Path(args.generate_script).expanduser().resolve()
        if args.generate_script
        else _find_generate_initial_script(repo_root)
    )

    sys.argv = [str(generate_script)] + generate_args
    runpy.run_path(str(generate_script), run_name="__main__")


if __name__ == "__main__":
    main()
