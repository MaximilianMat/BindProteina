"""
Initial Sample Generation for LaProteina V2 Pipeline

Generates a diverse pool of initial binder candidates with:
- Multiple linker configurations (varying lengths and attachment edges)
- Full protein context for collision-free generation
- Quick Rosetta pre-scoring (no relaxation) for baseline metrics
- AF2 re-prediction and full Rosetta scoring for all samples
- Binder position overlap metric (Jaccard) to detect position drift

Usage:
    python generate_initial.py --settings settings_generation/my_target.json
    
    # Can also use existing evolution settings (with sensible defaults for new params):
    python generate_initial.py --settings settings_evolution/CLD18.json
"""

import argparse
import os
import sys
import json
import subprocess
import shutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Set
from loguru import logger

# Adjust path to import BindProteina modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pipeline.target_preparer import (
    get_target_length,
    remove_linker_from_output,
    create_pyrosetta_input
)
from pipeline.laproteina_generator import LaProteinaGenerator


def load_settings(settings_path: str) -> Dict:
    """Load settings from a JSON file."""
    with open(settings_path, 'r') as f:
        return json.load(f)


def fix_path(p: str) -> str:
    """Fix paths that might be missing leading slash."""
    if p and (p.startswith("nfs/") or p.startswith("home/")):
        return "/" + p
    return p


    # legacy function removed


def collect_all_pdbs(output_dir: Path) -> List[str]:
    """Collect all generated PDB files from output directory."""
    pdbs = []
    for pdb_path in output_dir.rglob("*.pdb"):
        # Filter for generated samples (not prepared targets)
        if "_sample_" in pdb_path.name or "job_" in pdb_path.name:
            pdbs.append(str(pdb_path))
    return pdbs


def pdb_has_chain_b(pdb_path: str) -> bool:
    """Return True if a PDB contains at least one ATOM record on chain B."""
    try:
        with open(pdb_path, "r", errors="ignore") as fh:
            for line in fh:
                if line.startswith("ATOM") and len(line) > 21 and line[21] == "B":
                    return True
    except Exception:
        return False
    return False


def compute_interface_overlap(pre_residues_str: str, post_residues_str: str) -> float:
    """
    Compute Jaccard overlap of target interface residue sets.
    
    Compares the set of target (chain A) residues at the interface before AF2
    and after AF2. A low overlap indicates the binder moved to a different region
    of the target during AF2 re-prediction.
    
    Args:
        pre_residues_str: Comma-separated target interface residue numbers (pre-AF2).
        post_residues_str: Comma-separated target interface residue numbers (post-AF2).
        
    Returns:
        Jaccard similarity (0.0 - 1.0). 0.0 means no overlap, 1.0 means identical sets.
    """
    if not pre_residues_str or not post_residues_str:
        return 0.0
    
    try:
        pre_set = set(int(r.strip()) for r in str(pre_residues_str).split(",") if r.strip())
        post_set = set(int(r.strip()) for r in str(post_residues_str).split(",") if r.strip())
    except (ValueError, TypeError):
        return 0.0
    
    if not pre_set or not post_set:
        return 0.0
    
    intersection = pre_set & post_set
    union = pre_set | post_set
    return len(intersection) / len(union) if union else 0.0


def _resolve_filter_settings_path(settings: Dict[str, Any], repo_root: Path) -> Path:
    """Resolve filter JSON path from settings with sane defaults."""
    lp_settings = settings.get("laproteina_settings", {})
    filter_path = (
        settings.get("filters")
        or settings.get("filter_settings_path")
        or lp_settings.get("filters")
        or lp_settings.get("filter_settings_path")
        or "settings_filters/laProteina_filters.json"
    )
    filter_path = Path(fix_path(str(filter_path)))
    if not filter_path.is_absolute():
        filter_path = repo_root / filter_path
    return filter_path


def _pick_filter_spec(
    filters_cfg: Dict[str, Dict[str, Any]],
    metric_suffix: str,
    model_num: int
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Pick filter in priority order: model-specific -> model 1 -> average."""
    candidate_keys = [
        f"{model_num}_{metric_suffix}",
        f"1_{metric_suffix}",
        f"Average_{metric_suffix}",
    ]
    for key in candidate_keys:
        spec = filters_cfg.get(key)
        if isinstance(spec, dict) and spec.get("threshold") is not None:
            return key, spec
    return None, None


def main():
    import time
    pipeline_start_time = time.time()
    
    # Set matmul precision for performance on Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    parser = argparse.ArgumentParser(
        description="Initial Sample Generation for LaProteina V2 Pipeline"
    )
    parser.add_argument(
        "--settings", 
        type=str, 
        required=True,
        help="Path to JSON settings file (e.g., settings_generation/my_target.json)"
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip folding+scoring (generate only)"
    )
    parser.add_argument(
        "--folding_model",
        type=str,
        default=None,
        help="Folding model to use for refolding (af2 or protenix)"
    )
    parser.add_argument(
        "--protenix_model",
        type=str,
        default=None,
        help="Protenix model name"
    )
    parser.add_argument(
        "--target_msa",
        type=str,
        default=None,
        help="Optional path to pre-computed target MSA (.a3m file or directory) for Chain A"
    )
    parser.add_argument(
        "--msa_db",
        type=str,
        default=None,
        help="Path to MSA database directory for auto-generating target MSAs"
    )
    parser.add_argument(
        "--target_template",
        type=str,
        default=None,
        help="Optional path to structural template for target Chain A. Accepts a .pdb/.cif "
             "file (protenix mt is run online to find matching PDB templates) or a pre-computed "
             ".a3m/.hhr template file. Enables --use_template true in Protenix."
    )
    parser.add_argument(
        "--protenix_n_diffusion_samples",
        type=int,
        default=None,
        help="Number of diffusion samples per design for Protenix (default: 5; use 3 for ~40%% speedup)"
    )

    args = parser.parse_args()
    
    # Load settings
    settings_path = Path(args.settings)
    if not settings_path.is_absolute():
        repo_root = Path(__file__).parent.parent
        settings_path = repo_root / settings_path
    
    if not settings_path.exists():
        logger.error(f"Settings file not found: {args.settings}")
        sys.exit(1)
    
    settings = load_settings(settings_path)
    logger.info(f"Loaded settings from: {settings_path}")

    # Helper to merge CLI and JSON settings
    def get_setting(key, cli_val, default=None):
        if cli_val is not None:
            return cli_val
        return settings.get(key, default)

    folding_model = get_setting("folding_model", args.folding_model, "af2")
    protenix_model = get_setting("protenix_model", args.protenix_model, "protenix-v2")
    protenix_n_diffusion_samples = int(get_setting("protenix_n_diffusion_samples", args.protenix_n_diffusion_samples, 5))
    target_msa = get_setting("target_msa", args.target_msa)
    msa_db = get_setting("msa_db", args.msa_db)
    target_template = get_setting("target_template", args.target_template)
    
    if target_msa:
        target_msa = fix_path(target_msa)
    if msa_db:
        msa_db = fix_path(msa_db)
    if target_template:
        target_template = fix_path(target_template)
    
    # Extract settings (settings_target format)
    lp_settings = settings.get("laproteina_settings", {})
    repo_root = Path(__file__).parent.parent
    filter_settings_path = _resolve_filter_settings_path(settings, repo_root)
    if filter_settings_path.exists():
        with open(filter_settings_path, "r") as f:
            filter_settings = json.load(f)
        logger.info(f"Using filter settings: {filter_settings_path}")
    else:
        filter_settings = {}
        logger.warning(
            f"Filter settings not found at {filter_settings_path}. "
            "Proceeding without JSON-based filtering."
        )
    
    # Extract CATH fold conditioning parameters
    cath_codes = lp_settings.get("cath_codes") or settings.get("cath_codes", None)
    len_cath_code_path = lp_settings.get("len_cath_code_path") or settings.get("len_cath_code_path", None)
    cath_code_level = lp_settings.get("cath_code_level") or settings.get("cath_code_level", None)
    if len_cath_code_path:
        len_cath_code_path = fix_path(len_cath_code_path)
    
    # Required fields
    target_pdb = fix_path(settings["starting_pdb"])
    checkpoint = fix_path(lp_settings["checkpoint_path"])
    output_base_dir = settings.get("output_dir", "./initial_generation")
    hotspot_string = settings.get("target_hotspot_residues", "")
    
    # Linker length range: parse from string "5-30" format (like hotspot_residues)
    # Check both top-level and nested in laproteina_settings
    linker_range_str = settings.get("linker_length_range") or lp_settings.get("linker_length_range", "15-30")
    if isinstance(linker_range_str, str) and "-" in linker_range_str:
        parts = linker_range_str.split("-")
        linker_length_range = (int(parts[0]), int(parts[1]))
    elif isinstance(linker_range_str, (list, tuple)):
        linker_length_range = tuple(linker_range_str)
    else:
        linker_length_range = (15, 30)
    
    linker_mode = settings.get("linker_generation_mode", "round_robin")
    
    binder_length_range = tuple(lp_settings.get("binder_length_range", [40, 80]))
    # n_samples is the TOTAL number of samples (distributed evenly across linker starting points)
    n_total_samples = lp_settings.get("n_samples", 50)
    # batch_size for generation (affects GPU memory usage)
    batch_size = lp_settings.get("batch_size", 2)
    # n_linker_starting_points: number of different linker attachment positions.
    # Total samples are split evenly among starting points.
    # e.g., n_samples=24, n_linker_starting_points=4 → 6 samples per starting point
    # Check laproteina_settings first, then top-level n_linker_configs for backwards compat
    n_linker_starting_points = lp_settings.get(
        "n_linker_starting_points",
        settings.get("n_linker_configs", min(n_total_samples, 8))
    )
    # Ensure we don't have more starting points than samples
    n_linker_starting_points = min(n_linker_starting_points, n_total_samples)
    
    samples_per_starting_point = n_total_samples // n_linker_starting_points
    remainder_samples = n_total_samples % n_linker_starting_points
    
    device = settings.get("device", "cuda")
    direct_binder_mode = settings.get(
        "direct_binder_mode",
        lp_settings.get("direct_binder_mode", False),
    )
    if isinstance(direct_binder_mode, str):
        direct_binder_mode = direct_binder_mode.strip().lower() in {"1", "true", "yes", "on"}

    linked_input_mode = settings.get(
        "linked_input_mode",
        lp_settings.get("linked_input_mode", False),
    )
    if isinstance(linked_input_mode, str):
        linked_input_mode = linked_input_mode.strip().lower() in {"1", "true", "yes", "on"}

    def _parse_optional_int(value, field_name: str) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.error(f"{field_name} must be an integer, got: {value!r}")
            sys.exit(1)

    linked_target_length = _parse_optional_int(
        settings.get("target_length", lp_settings.get("target_length")),
        "target_length",
    )
    linked_linker_length = _parse_optional_int(
        settings.get("linker_length", lp_settings.get("linker_length")),
        "linker_length",
    )
    keep_linker_end_n = _parse_optional_int(
        settings.get("keep_linker_end_n", lp_settings.get("keep_linker_end_n", 0)),
        "keep_linker_end_n",
    )
    keep_linker_end_n = keep_linker_end_n if keep_linker_end_n is not None else 0
    
    # LaProteina mode: "all_atom" (LD4) or "tip_atom" (LD5)
    # Controls which model config, feature factory class, and atom selection are used
    laproteina_mode = lp_settings.get("laproteina_mode", "all_atom")
    if laproteina_mode not in ("all_atom", "tip_atom"):
        logger.error(f"Invalid laproteina_mode: {laproteina_mode!r}. Must be 'all_atom' or 'tip_atom'")
        sys.exit(1)

    inference_config = settings.get("inference_config", lp_settings.get("inference_config"))
    if linked_input_mode and not inference_config:
        inference_config = "inference_motif_idx_aa"
    
    logger.info(f"Settings:")
    logger.info(f"  target_pdb: {target_pdb}")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  laproteina_mode: {laproteina_mode}")
    logger.info(f"  inference_config: {inference_config or '(mode default)'}")
    logger.info(f"  linker_length_range: {linker_length_range}")
    logger.info(f"  n_total_samples: {n_total_samples}")
    logger.info(f"  n_linker_starting_points: {n_linker_starting_points}")
    logger.info(f"  samples_per_starting_point: {samples_per_starting_point}" +
               (f" (+1 for first {remainder_samples})" if remainder_samples else ""))
    logger.info(f"  binder_length_range: {binder_length_range}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  direct_binder_mode: {direct_binder_mode}")
    logger.info(f"  linked_input_mode: {linked_input_mode}")
    logger.info(f"  cath_codes: {cath_codes}")
    logger.info(f"  len_cath_code_path: {len_cath_code_path}")
    logger.info(f"  cath_code_level: {cath_code_level}")
    
    # Create output directory
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(output_base_dir) / f"initial_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger.add(output_dir / "generation.log")
    logger.info(f"Starting Initial Generation Pipeline")
    logger.info(f"Output directory: {output_dir}")
    
    # Save settings copy
    with open(output_dir / "settings_used.json", 'w') as f:
        json.dump(settings, f, indent=2)
    
    # Validate inputs
    if not os.path.exists(target_pdb):
        logger.error(f"Target PDB not found: {target_pdb}")
        sys.exit(1)

    if direct_binder_mode and linked_input_mode:
        logger.error("direct_binder_mode and linked_input_mode are mutually exclusive.")
        sys.exit(1)

    if linked_input_mode:
        if linked_target_length is None or linked_linker_length is None:
            logger.error(
                "linked_input_mode requires explicit target_length and linker_length in settings."
            )
            sys.exit(1)
        if linked_target_length <= 0:
            logger.error(f"target_length must be > 0, got {linked_target_length}")
            sys.exit(1)
        if linked_linker_length < 0:
            logger.error(f"linker_length must be >= 0, got {linked_linker_length}")
            sys.exit(1)
        if keep_linker_end_n < 0:
            logger.error(f"keep_linker_end_n must be >= 0, got {keep_linker_end_n}")
            sys.exit(1)
        if keep_linker_end_n > linked_linker_length:
            logger.error(
                f"keep_linker_end_n ({keep_linker_end_n}) cannot exceed linker_length ({linked_linker_length})"
            )
            sys.exit(1)
    elif not direct_binder_mode and not hotspot_string:
        logger.error("target_hotspot_residues is required for linker generation")
        sys.exit(1)

    # Get target chain-A length from input PDB.
    chain_a_length = get_target_length(target_pdb, chain_id="A")
    if linked_input_mode:
        if linked_target_length + linked_linker_length > chain_a_length:
            logger.error(
                "Invalid linked-input boundaries: target_length + linker_length exceeds "
                f"chain A length ({linked_target_length} + {linked_linker_length} > {chain_a_length})."
            )
            sys.exit(1)
        target_length = linked_target_length
        logger.info(
            f"Linked input boundaries: target_length={linked_target_length}, "
            f"linker_length={linked_linker_length}, keep_linker_end_n={keep_linker_end_n}"
        )
        logger.info(
            f"Input chain A length: {chain_a_length}, fixed generation context: "
            f"A1-{linked_target_length + linked_linker_length}"
        )
    else:
        target_length = chain_a_length

    logger.info(f"Target protein: {Path(target_pdb).name}, chain-A length: {chain_a_length}")
    
    if not direct_binder_mode:
        if linked_input_mode:
            logger.info("=" * 60)
            logger.info("Skipping synthetic linker generation. Using linked input PDB with explicit boundaries.")
            logger.info("=" * 60)
        else:
            logger.error("Only 'direct_binder_mode' and 'linked_input_mode' are supported. Legacy linker generation has been removed.")
            sys.exit(1)
    else:
        logger.info("=" * 60)
        logger.info("Phase 1/2: Direct Binder Mode (no linker generation)")
        logger.info("=" * 60)
        logger.info("Skipping linker configuration and target preparation.")
    
    # =========================================================================
    # Phase 3: Generate Binder Samples
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 3: Generating Binder Samples")
    logger.info("=" * 60)
    
    # Clear GPU memory before loading LaProteina
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    
    # Initialize LaProteina generator
    logger.info("Loading LaProteina model...")
    ckpt_path = Path(checkpoint)
    
    # Use explicit autoencoder_path from settings, or derive from checkpoint location
    autoencoder_path = lp_settings.get("autoencoder_path")
    if autoencoder_path:
        ae_path = Path(fix_path(autoencoder_path))
    else:
        ae_path = ckpt_path.parent / "AE3_motif.ckpt"
    
    generator = LaProteinaGenerator(
        checkpoint_path=str(ckpt_path),
        autoencoder_path=str(ae_path) if ae_path.exists() else None,
        device=device,
        laproteina_mode=laproteina_mode,
        inference_config=inference_config,
    )
    generator._load_model()  # Preload model
    
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    
    all_generated_pdbs = []
    if direct_binder_mode:
        logger.info(f"Generating {n_total_samples} direct binder samples (single pass)")
        direct_samples = generator.generate_binder(
            target_pdb=target_pdb,
            target_residue_range=(1, target_length),
            binder_length_range=binder_length_range,
            n_samples=n_total_samples,
            output_dir=str(generated_dir),
            design_id="direct",
            direct_binder_mode=True,
            binder_target_nres=target_length,
            max_batch_size=batch_size,
            cath_codes=cath_codes,
            len_cath_code_path=len_cath_code_path,
            cath_code_level=cath_code_level,
        )
        all_generated_pdbs.extend(direct_samples)
        logger.info(f"Direct mode: Generated {len(direct_samples)} samples")
    elif linked_input_mode:
        logger.info(f"Generating {n_total_samples} linked input samples")
        binder_min, binder_max = binder_length_range
        total_fixed_length = linked_target_length + linked_linker_length
        contig_string = f"A1-{total_fixed_length}/{binder_min}-{binder_max}"
        
        linked_samples = generator.generate_binder(
            target_pdb=target_pdb,
            target_residue_range=(1, total_fixed_length),
            binder_length_range=binder_length_range,
            n_samples=n_total_samples,
            output_dir=str(generated_dir),
            contig_string=contig_string,
            design_id="linked",
            direct_binder_mode=False,
            max_batch_size=batch_size,
            cath_codes=cath_codes,
            len_cath_code_path=len_cath_code_path,
            cath_code_level=cath_code_level,
        )
        all_generated_pdbs.extend(linked_samples)
        logger.info(f"Linked mode: Generated {len(linked_samples)} samples")
    else:
        logger.error("Neither mode is enabled. Stopping.")
        sys.exit(1)
    
    # Unload generator to free GPU memory
    generator.unload()
    
    logger.info(f"Total generated samples: {len(all_generated_pdbs)}")
    
    # Collect all PDBs (in case some were not tracked)
    all_generated_pdbs = collect_all_pdbs(generated_dir)
    logger.info(f"Collected {len(all_generated_pdbs)} PDB files from {generated_dir}")
    
    # =========================================================================
    # Phase 4: Remove Linker & Prepare for Scoring
    # =========================================================================
    if all_generated_pdbs:
        logger.info("=" * 60)
        logger.info("Phase 4: Remove Linker & Prepare for Scoring")
        logger.info("=" * 60)
        
        if direct_binder_mode:
            # Direct mode outputs are already target+binder complexes. Just copy into scoring_input.
            scoring_input_dir = output_dir / "scoring_input"
            scoring_input_dir.mkdir(parents=True, exist_ok=True)
            scoring_pdbs = []
            for pdb_path in all_generated_pdbs:
                out_path = scoring_input_dir / Path(pdb_path).name
                shutil.copy2(pdb_path, out_path)
                scoring_pdbs.append(str(out_path))
            logger.info(f"Prepared {len(scoring_pdbs)} direct-binder structures for scoring")
            # Guardrail: direct-binder mode must materialize an explicit chain B.
            missing_chain_b = [p for p in scoring_pdbs if not pdb_has_chain_b(p)]
            if missing_chain_b:
                preview = ", ".join(Path(p).name for p in missing_chain_b[:5])
                raise RuntimeError(
                    "Direct-binder generation produced complexes without chain B "
                    f"({len(missing_chain_b)}/{len(scoring_pdbs)} files; examples: {preview}). "
                    "This indicates chain separation metadata was not propagated."
                )
        else:
            # Remove uniform linker from ALL generated PDBs
            scoring_input_dir = output_dir / "scoring_input"
            scoring_input_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Removing uniform linkers (length: {linked_linker_length}) from {len(all_generated_pdbs)} structures...")
            scoring_pdbs = []
            for pdb_path in all_generated_pdbs:
                result = remove_linker_from_output(
                    pdb_path=pdb_path,
                    target_length=target_length,
                    linker_length=linked_linker_length,
                    keep_linker_end_n=keep_linker_end_n,
                    output_path=str(scoring_input_dir / Path(pdb_path).name)
                )
                if result:
                    scoring_pdbs.append(result)
            
            logger.info(f"Prepared {len(scoring_pdbs)} structures for scoring")
            
            # Create PyRosetta input files (original target + extracted binder)
            pyrosetta_input_dir = output_dir / "pyrosetta_input"
            pyrosetta_input_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating PyRosetta input files (original target + binder)...")
            pyrosetta_pdbs = []
            for pdb_path in all_generated_pdbs:
                output_path = pyrosetta_input_dir / Path(pdb_path).name
                result = create_pyrosetta_input(
                    pdb_path=pdb_path,
                    original_target_pdb=target_pdb,
                    target_length=target_length,
                    linker_length=linked_linker_length,
                    keep_linker_end_n=keep_linker_end_n,
                    output_path=str(output_path)
                )
                if result:
                    pyrosetta_pdbs.append(result)
            logger.info(f"Created {len(pyrosetta_pdbs)} PyRosetta input files in {pyrosetta_input_dir}")
    else:
        scoring_pdbs = []
        logger.warning("No generated PDBs found, skipping linker removal")
    
    # =========================================================================
    # Phase 5 & 6: Unified Modular Scoring (AF2 + PyRosetta)
    # =========================================================================
    if scoring_pdbs and not args.skip_scoring:
        logger.info("=" * 60)
        logger.info("Phase 5 & 6: Unified Modular Scoring Pipeline (AF2/protenix + PyRosetta)")
        logger.info("=" * 60)
        from pipeline.scoring.scorer import score_complexes
        
        try:
            score_complexes(
                input_dir=str(scoring_input_dir),
                output_dir=str(output_dir),
                filters_path=str(filter_settings_path) if filter_settings_path.exists() else None,
                af2_model_idx=None,
                folding_model=folding_model,
                protenix_model_name=protenix_model,
                target_msa=target_msa,
                msa_db_path=msa_db,
                target_template=target_template,
                n_diffusion_samples=protenix_n_diffusion_samples,
            )
        except Exception as e:
            logger.error(f"Modular scoring failed: {e}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Initial Generation Pipeline Complete")
    logger.info("=" * 60)
    
    pipeline_end_time = time.time()
    total_elapsed = pipeline_end_time - pipeline_start_time
    hours, rem = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"Total Pipeline Runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} "
                f"({total_elapsed:.2f} seconds)")

    summary = {
        "pipeline_runtime_seconds": total_elapsed,
        "target_pdb": target_pdb,
        "n_linker_starting_points": n_linker_starting_points,
        "direct_binder_mode": direct_binder_mode,
        "linked_input_mode": linked_input_mode,
        "keep_linker_end_n": keep_linker_end_n,
        "samples_per_starting_point": samples_per_starting_point,
        "n_total_samples_requested": n_total_samples,
        "total_generated": len(all_generated_pdbs),
        "total_scored": len(scoring_pdbs),
        "output_dir": str(output_dir)
    }
    
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")
    
    with open(output_dir / "generation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")
    
    # Print path to scored results for use with evolve_v2.py
    scoring_summary = output_dir / "scored" / "scoring_summary.csv"
    if scoring_summary.exists():
        logger.info(f"\nTo continue with evolution, use:")
        logger.info(f"  --input_csv {scoring_summary}")
        logger.info(f"  --seeds_dir {output_dir / 'scored'}")


if __name__ == "__main__":
    main()

