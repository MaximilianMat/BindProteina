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
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Set
from loguru import logger

# Adjust path to import BindProteina modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from laproteina.proteinfoundation.linker_generator import (
    LinkerConfigGenerator, 
    LinkerConfig, 
    generate_linker_configs
)
from laproteina.proteinfoundation.target_preparer import (
    TargetPreparer, 
    PreparedTarget,
    get_target_length,
    remove_linker_from_output,
    create_pyrosetta_input
)
from laproteina.proteinfoundation.laproteina_generator import LaProteinaGenerator


def load_settings(settings_path: str) -> Dict:
    """Load settings from a JSON file."""
    with open(settings_path, 'r') as f:
        return json.load(f)


def fix_path(p: str) -> str:
    """Fix paths that might be missing leading slash."""
    if p and (p.startswith("nfs/") or p.startswith("home/")):
        return "/" + p
    return p


def generate_samples_for_config(
    generator: LaProteinaGenerator,
    prepared_target: PreparedTarget,
    n_samples: int,
    output_dir: Path,
    config_idx: int,
    max_batch_size: int = 2
) -> List[str]:
    """
    Generate samples for a single linker configuration.
    
    Returns list of generated PDB paths.
    """
    config = prepared_target.linker_config
    config_output_dir = output_dir / f"config_{config_idx:03d}_{config.config_id}"
    
    logger.info(f"Generating {n_samples} samples for config {config_idx}: {config}")
    
    # Generate binders
    try:
        generated_pdbs = generator.generate_binder(
            target_pdb=prepared_target.pdb_path,
            target_residue_range=(1, prepared_target.total_fixed_length),
            binder_length_range=prepared_target.metadata["binder_length_range"],
            n_samples=n_samples,
            output_dir=str(config_output_dir),
            contig_string=prepared_target.contig_string,
            design_id=f"cfg{config_idx:03d}",
            max_batch_size=max_batch_size  # Reduce batch size for memory
        )
        return generated_pdbs
    except Exception as e:
        logger.error(f"Generation failed for config {config_idx}: {e}")
        return []


def collect_all_pdbs(output_dir: Path) -> List[str]:
    """Collect all generated PDB files from output directory."""
    pdbs = []
    for pdb_path in output_dir.rglob("*.pdb"):
        # Filter for generated samples (not prepared targets)
        if "_sample_" in pdb_path.name or "job_" in pdb_path.name:
            pdbs.append(str(pdb_path))
    return pdbs


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
        help="Skip AF2 scoring (generate only)"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings_path = Path(args.settings)
    if not settings_path.is_absolute():
        repo_root = Path(__file__).parent.parent.parent
        settings_path = repo_root / settings_path
    
    if not settings_path.exists():
        logger.error(f"Settings file not found: {args.settings}")
        sys.exit(1)
    
    settings = load_settings(settings_path)
    logger.info(f"Loaded settings from: {settings_path}")
    
    # Extract settings (settings_target format)
    lp_settings = settings.get("laproteina_settings", {})
    repo_root = Path(__file__).parent.parent.parent
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
    
    # LaProteina mode: "all_atom" (LD4) or "tip_atom" (LD5)
    # Controls which model config, feature factory class, and atom selection are used
    laproteina_mode = lp_settings.get("laproteina_mode", "all_atom")
    if laproteina_mode not in ("all_atom", "tip_atom"):
        logger.error(f"Invalid laproteina_mode: {laproteina_mode!r}. Must be 'all_atom' or 'tip_atom'")
        sys.exit(1)
    
    logger.info(f"Settings:")
    logger.info(f"  target_pdb: {target_pdb}")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  laproteina_mode: {laproteina_mode}")
    logger.info(f"  linker_length_range: {linker_length_range}")
    logger.info(f"  n_total_samples: {n_total_samples}")
    logger.info(f"  n_linker_starting_points: {n_linker_starting_points}")
    logger.info(f"  samples_per_starting_point: {samples_per_starting_point}" +
               (f" (+1 for first {remainder_samples})" if remainder_samples else ""))
    logger.info(f"  binder_length_range: {binder_length_range}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  direct_binder_mode: {direct_binder_mode}")
    
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
    
    if not direct_binder_mode and not hotspot_string:
        logger.error("target_hotspot_residues is required for linker generation")
        sys.exit(1)
    
    # Get target length
    target_length = get_target_length(target_pdb, chain_id="A")
    logger.info(f"Target protein: {Path(target_pdb).name}, length: {target_length}")
    
    prepared_targets = []
    if not direct_binder_mode:
        # =========================================================================
        # Phase 1: Generate Linker Configurations
        # =========================================================================
        logger.info("=" * 60)
        logger.info("Phase 1: Generating Linker Configurations")
        logger.info("=" * 60)
        
        linker_generator = LinkerConfigGenerator(
            hotspot_string=hotspot_string,
            linker_length_range=linker_length_range,
            mode=linker_mode,
            n_attachment_points=n_linker_starting_points  # Evenly space attachment points across hotspot
        )
        
        linker_configs = linker_generator.generate_configs(n_linker_starting_points)
        logger.info(f"Generated {len(linker_configs)} linker configurations")
        logger.info(linker_generator.get_edge_summary())
        
        # Save linker configs
        linker_configs_data = [
            {
                "config_id": cfg.config_id,
                "attachment_residue": cfg.attachment_residue,
                "linker_length": cfg.linker_length,
                "edge_type": cfg.edge_type,
                "hotspot_range": cfg.hotspot_range
            }
            for cfg in linker_configs
        ]
        with open(output_dir / "linker_configs.json", 'w') as f:
            json.dump(linker_configs_data, f, indent=2)
        
        # =========================================================================
        # Phase 2: Prepare Targets with Linkers
        # =========================================================================
        logger.info("=" * 60)
        logger.info("Phase 2: Preparing Targets with Linkers")
        logger.info("=" * 60)
        
        prepared_targets_dir = output_dir / "prepared_targets"
        target_preparer = TargetPreparer(
            output_dir=str(prepared_targets_dir),
            linker_mode="polyglycine"  # Fast mode for initial generation
        )
        
        prepared_targets = target_preparer.prepare_batch(
            target_pdb=target_pdb,
            linker_configs=linker_configs,
            binder_length_range=binder_length_range,
            chain_id="A"
        )
        
        logger.info(f"Prepared {len(prepared_targets)} target+linker combinations")
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
            max_batch_size=batch_size,
        )
        all_generated_pdbs.extend(direct_samples)
        logger.info(f"Direct mode: Generated {len(direct_samples)} samples")
    else:
        # Distribute total samples evenly across linker starting points
        n_configs = len(prepared_targets)
        samples_per_config = n_total_samples // n_configs
        remainder = n_total_samples % n_configs
        
        logger.info(f"Distributing {n_total_samples} samples across {n_configs} linker starting points")
        logger.info(f"  Samples per starting point: {samples_per_config}" +
                   (f" (+1 for first {remainder})" if remainder else ""))
        
        for idx, prepared_target in enumerate(prepared_targets):
            # First 'remainder' starting points get one extra sample
            n_samples_this_config = samples_per_config + (1 if idx < remainder else 0)
            
            if n_samples_this_config == 0:
                logger.info(f"Starting point {idx}: Skipping (no samples allocated)")
                continue
                
            pdbs = generate_samples_for_config(
                generator=generator,
                prepared_target=prepared_target,
                n_samples=n_samples_this_config,
                output_dir=generated_dir,
                config_idx=idx,
                max_batch_size=batch_size
            )
            all_generated_pdbs.extend(pdbs)
            logger.info(f"Starting point {idx}: Generated {len(pdbs)} samples (total: {len(all_generated_pdbs)})")
            
            # Clear GPU cache between configs to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
        else:
            # Build mapping from config_id to linker_length
            # Each sample may have a different linker length based on its config
            config_to_linker_len = {}
            for pt in prepared_targets:
                config_to_linker_len[pt.linker_config.config_id] = pt.linker_length
            
            def get_linker_length_for_pdb(pdb_path: str) -> int:
                """Extract linker length from PDB path based on config_id or L{num} pattern."""
                import re
                path_str = str(pdb_path)
                
                # Try to match config_id (e.g., cfg006)
                for config_id, linker_len in config_to_linker_len.items():
                    if config_id in path_str:
                        return linker_len
                
                # Fallback: try to parse L{num} from filename
                match = re.search(r'_L(\d+)_', path_str)
                if match:
                    return int(match.group(1))
                
                # Last resort: use first config's linker length
                if prepared_targets:
                    return prepared_targets[0].linker_length
                return linker_length_range[0]
            
            # Remove linker from ALL generated PDBs
            # This creates clean target (chain A) + binder (chain B) structures
            scoring_input_dir = output_dir / "scoring_input"
            scoring_input_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Removing linkers from {len(all_generated_pdbs)} structures (varying linker lengths)...")
            scoring_pdbs = []
            for pdb_path in all_generated_pdbs:
                linker_len = get_linker_length_for_pdb(pdb_path)
                result = remove_linker_from_output(
                    pdb_path=pdb_path,
                    target_length=target_length,
                    linker_length=linker_len,
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
                linker_len = get_linker_length_for_pdb(pdb_path)
                output_path = pyrosetta_input_dir / Path(pdb_path).name
                result = create_pyrosetta_input(
                    pdb_path=pdb_path,
                    original_target_pdb=target_pdb,
                    target_length=target_length,
                    linker_length=linker_len,
                    output_path=str(output_path)
                )
                if result:
                    pyrosetta_pdbs.append(result)
            logger.info(f"Created {len(pyrosetta_pdbs)} PyRosetta input files in {pyrosetta_input_dir}")
    else:
        scoring_pdbs = []
        logger.warning("No generated PDBs found, skipping linker removal")
    
    # =========================================================================
    # Phase 5: AF2 Re-prediction (no relaxation) — pre-scoring
    # =========================================================================
    af2_output_pdbs = []  # list of (design_name, af2_pdb_path)
    if scoring_pdbs and not args.skip_scoring:
        logger.info("=" * 60)
        logger.info("Phase 5: AF2 Re-prediction (no relaxation)")
        logger.info("=" * 60)
        
        import time as _time
        import re as _re
        import random as _random
        
        # Clear GPU memory before loading AF2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Initialize ColabDesign AF2 model
        from colabdesign import mk_afdesign_model, clear_mem
        from colabdesign.shared.utils import copy_dict
        from laproteina_scoring.utils import extract_sequence_from_pdb
        
        clear_mem()
        logger.info("Initializing AF2 model for re-prediction...")
        complex_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=1,
            use_multimer=True,
            use_initial_guess=False,
            use_initial_atom_pos=False
        )
        
        # Pick a random AF2 model (1 of 5)
        af2_model_num = _random.randint(0, 4)
        logger.info(f"AF2 model initialized. Using model {af2_model_num + 1}")
        
        af2_predictions_dir = output_dir / "af2_predictions"
        af2_predictions_dir.mkdir(parents=True, exist_ok=True)
        
        all_af2_prescores = []
        af2_total_time = 0.0
        
        logger.info(f"Running AF2 re-prediction on {len(scoring_pdbs)} structures...")
        for idx, pdb_path in enumerate(scoring_pdbs):
            pdb_name = Path(pdb_path).stem
            
            try:
                # Extract binder sequence from chain B
                binder_seq = extract_sequence_from_pdb(pdb_path, "B")
                if not binder_seq:
                    logger.warning(f"No binder sequence found in {pdb_name}, skipping")
                    all_af2_prescores.append({"design_name": pdb_name, "status": "failed", "reason": "no_sequence"})
                    continue
                
                binder_seq = _re.sub("[^A-Z]", "", binder_seq.upper())
                binder_len = len(binder_seq)
                
                # Prep AF2 model inputs (target = chain A from the scoring_input PDB)
                complex_model.prep_inputs(
                    pdb_filename=pdb_path,
                    chain="A",
                    binder_len=binder_len,
                    rm_target_seq=True,
                    rm_target_sc=True
                )
                
                # Run AF2 prediction (timed)
                t_start = _time.time()
                complex_model.predict(
                    seq=binder_seq,
                    models=[af2_model_num],
                    num_recycles=1,
                    verbose=False
                )
                t_predict = _time.time() - t_start
                af2_total_time += t_predict
                
                # Save predicted PDB
                af2_pdb_path = str(af2_predictions_dir / f"{pdb_name}_model{af2_model_num + 1}.pdb")
                complex_model.save_pdb(af2_pdb_path)
                
                # Extract AF2 metrics
                metrics = copy_dict(complex_model.aux["log"])
                binder_plddt = None
                residue_plddt = complex_model.aux.get("plddt")
                if residue_plddt is not None:
                    residue_plddt = np.asarray(residue_plddt).reshape(-1)
                    if residue_plddt.size >= binder_len:
                        binder_plddt = round(float(np.mean(residue_plddt[-binder_len:])), 3)
                af2_stats = {
                    "design_name": pdb_name,
                    "status": "predicted",
                    "sequence": binder_seq,
                    "binder_length": binder_len,
                    "af2_model": af2_model_num + 1,
                    "pLDDT": round(metrics["plddt"], 3),
                    "binder_pLDDT": binder_plddt,
                    "pTM": round(metrics["ptm"], 3),
                    "i_pTM": round(metrics["i_ptm"], 3),
                    "pAE": round(metrics["pae"], 3),
                    "i_pAE": round(metrics["i_pae"], 3),
                    "af2_predict_seconds": round(t_predict, 1),
                }
                all_af2_prescores.append(af2_stats)
                af2_output_pdbs.append((pdb_name, af2_pdb_path))
                
                logger.info(
                    f"[{idx+1}/{len(scoring_pdbs)}] AF2 predicted {pdb_name} in {t_predict:.1f}s — "
                    f"pLDDT={af2_stats['pLDDT']:.2f}, pTM={af2_stats['pTM']:.2f}, "
                    f"i_pTM={af2_stats['i_pTM']:.2f}, i_pAE={af2_stats['i_pAE']:.2f}"
                    + (
                        f", binder_pLDDT={af2_stats['binder_pLDDT']:.2f}"
                        if af2_stats["binder_pLDDT"] is not None else ""
                    )
                )
                
            except Exception as e:
                logger.error(f"AF2 prediction failed for {pdb_name}: {e}")
                all_af2_prescores.append({"design_name": pdb_name, "status": "failed", "reason": str(e)})
        
        # Save AF2 pre-scores
        af2_prescores_df = pd.DataFrame(all_af2_prescores)
        af2_prescoring_csv = output_dir / "af2_prescoring_results.csv"
        af2_prescores_df.to_csv(af2_prescoring_csv, index=False)
        
        af2_avg_time = af2_total_time / len(af2_output_pdbs) if af2_output_pdbs else 0
        logger.info(f"AF2 pre-scoring complete:")
        logger.info(f"  Total AF2 prediction time: {af2_total_time:.1f}s ({af2_total_time/60:.1f} min)")
        logger.info(f"  Average per sample: {af2_avg_time:.1f}s")
        logger.info(f"  Successfully predicted: {len(af2_output_pdbs)}/{len(scoring_pdbs)}")
        logger.info(f"  Saved AF2 metrics to: {af2_prescoring_csv}")
        
        if not af2_prescores_df.empty and 'pLDDT' in af2_prescores_df.columns:
            predicted = af2_prescores_df[af2_prescores_df['status'] == 'predicted']
            if not predicted.empty:
                logger.info(f"  AF2 metric summary:")
                logger.info(f"    pLDDT: mean={predicted['pLDDT'].mean():.3f}, min={predicted['pLDDT'].min():.3f}, max={predicted['pLDDT'].max():.3f}")
                logger.info(f"    pTM:   mean={predicted['pTM'].mean():.3f}, min={predicted['pTM'].min():.3f}, max={predicted['pTM'].max():.3f}")
                logger.info(f"    i_pTM: mean={predicted['i_pTM'].mean():.3f}, min={predicted['i_pTM'].min():.3f}, max={predicted['i_pTM'].max():.3f}")
        
        # Free GPU memory - AF2 model no longer needed
        del complex_model
        clear_mem()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # ----- AF2 Filtering: keep only samples passing JSON-driven AF2 filters -----
        af2_metric_to_suffix = {
            "pLDDT": "pLDDT",
            "binder_pLDDT": "Binder_pLDDT",
            "pTM": "pTM",
            "i_pTM": "i_pTM",
            "pAE": "pAE",
            "i_pAE": "i_pAE",
        }
        af2_filters = {}
        for metric_name, metric_suffix in af2_metric_to_suffix.items():
            filter_key, filter_spec = _pick_filter_spec(
                filter_settings,
                metric_suffix,
                af2_model_num + 1,
            )
            if filter_key and filter_spec:
                af2_filters[metric_name] = {
                    "threshold": filter_spec.get("threshold"),
                    "higher": filter_spec.get("higher", True),
                    "filter_key": filter_key,
                }
        
        logger.info("AF2 filter thresholds (loaded from filter JSON):")
        if not af2_filters:
            logger.info("  No AF2 filters with thresholds were found; keeping all AF2 predictions.")
        for metric, cfg in af2_filters.items():
            direction = ">=" if cfg["higher"] else "<="
            logger.info(f"  {cfg['filter_key']} ({metric}) {direction} {cfg['threshold']}")
        
        passing_af2 = []
        filter_failure_counts = {}
        
        for design_name, af2_pdb_path in af2_output_pdbs:
            row = next(
                (r for r in all_af2_prescores if r.get("design_name") == design_name and r.get("status") == "predicted"),
                None
            )
            if row is None:
                continue
            
            passed = True
            for metric, cfg in af2_filters.items():
                value = row.get(metric)
                if value is None:
                    continue
                if cfg["higher"] and value < cfg["threshold"]:
                    passed = False
                    key = cfg["filter_key"]
                    filter_failure_counts[key] = filter_failure_counts.get(key, 0) + 1
                elif not cfg["higher"] and value > cfg["threshold"]:
                    passed = False
                    key = cfg["filter_key"]
                    filter_failure_counts[key] = filter_failure_counts.get(key, 0) + 1
            
            if passed:
                passing_af2.append((design_name, af2_pdb_path))
        
        logger.info(f"AF2 filter results: {len(passing_af2)}/{len(af2_output_pdbs)} passed")
        if filter_failure_counts:
            for metric, count in sorted(filter_failure_counts.items()):
                logger.info(f"  {metric}: {count} failures")
        
        af2_output_pdbs = passing_af2
        
    elif args.skip_scoring:
        logger.info("Skipping AF2 scoring")
    else:
        logger.warning("No structures available for scoring after linker removal")
    
    # =========================================================================
    # Phase 6: PyRosetta Relax + Interface Scoring (only AF2-passing samples)
    # =========================================================================
    if af2_output_pdbs and not args.skip_scoring:
        logger.info("=" * 60)
        logger.info(f"Phase 6: PyRosetta Relax + Interface Scoring ({len(af2_output_pdbs)} passing samples)")
        logger.info("=" * 60)
        
        import time as _time
        
        # Initialize PyRosetta
        import pyrosetta as pr
        repo_root = Path(__file__).parent.parent.parent
        dalphaball_path = str(repo_root / 'functions' / 'DAlphaBall.gcc')
        try:
            from functions.pyrosetta_utils import get_pyrosetta_init_cmd
            init_cmd = get_pyrosetta_init_cmd(dalphaball_path)
            pr.init(init_cmd)
            logger.info(f"PyRosetta initialized with multithreading ({os.cpu_count() or 4} threads)")
        except Exception:
            pass  # Already initialized
        
        from functions.pyrosetta_utils import score_interface, pr_relax
        
        relaxed_dir = output_dir / "relaxed"
        relaxed_dir.mkdir(parents=True, exist_ok=True)
        scoring_output_dir = output_dir / "scored"
        scoring_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_final_scores = []
        relax_total_time = 0.0
        score_total_time = 0.0
        
        logger.info(f"Running PyRosetta relax + scoring on {len(af2_output_pdbs)} AF2-predicted structures...")
        for idx, (design_name, af2_pdb_path) in enumerate(af2_output_pdbs):
            try:
                # Step 1: Relax (timed)
                relaxed_pdb_path = str(relaxed_dir / f"{design_name}_relaxed.pdb")
                t_start = _time.time()
                pr_relax(af2_pdb_path, relaxed_pdb_path)
                t_relax = _time.time() - t_start
                relax_total_time += t_relax
                
                # Step 2: Score interface (timed)
                t_start = _time.time()
                interface_scores, interface_AA, interface_residues = score_interface(
                    relaxed_pdb_path, binder_chain="B"
                )
                t_score = _time.time() - t_start
                score_total_time += t_score
                
                # Combine with AF2 pre-scores
                # Find the AF2 pre-score row for this design
                af2_row = next(
                    (r for r in all_af2_prescores if r.get("design_name") == design_name),
                    {}
                )
                
                final_row = {
                    "design_name": design_name,
                    "status": "scored",
                    # AF2 metrics
                    "pLDDT": af2_row.get("pLDDT"),
                    "binder_pLDDT": af2_row.get("binder_pLDDT"),
                    "pTM": af2_row.get("pTM"),
                    "i_pTM": af2_row.get("i_pTM"),
                    "pAE": af2_row.get("pAE"),
                    "i_pAE": af2_row.get("i_pAE"),
                    "af2_model": af2_row.get("af2_model"),
                    "binder_length": af2_row.get("binder_length"),
                    # PyRosetta metrics
                    **interface_scores,
                    # Timing
                    "af2_predict_seconds": af2_row.get("af2_predict_seconds"),
                    "relax_seconds": round(t_relax, 1),
                    "rosetta_score_seconds": round(t_score, 1),
                    # Sequence
                    "sequence": af2_row.get("sequence"),
                }

                # Apply all available JSON-backed filters for metrics present in final_row.
                # Any filter key not mappable to computed metrics is skipped.
                final_metric_to_suffix = {
                    "pLDDT": "pLDDT",
                    "binder_pLDDT": "Binder_pLDDT",
                    "pTM": "pTM",
                    "i_pTM": "i_pTM",
                    "pAE": "pAE",
                    "i_pAE": "i_pAE",
                    "binder_score": "Binder_Energy_Score",
                    "surface_hydrophobicity": "Surface_Hydrophobicity",
                    "interface_sc": "ShapeComplementarity",
                    "interface_packstat": "PackStat",
                    "interface_dG": "dG",
                    "interface_dSASA": "dSASA",
                    "interface_nres": "n_InterfaceResidues",
                    "interface_interface_hbonds": "n_InterfaceHbonds",
                    "interface_delta_unsat_hbonds": "n_InterfaceUnsatHbonds",
                    "interface_hydrophobicity": "Interface_Hydrophobicity",
                }
                failed_filters = []
                model_num = int(final_row.get("af2_model") or (af2_model_num + 1))
                for metric_name, metric_suffix in final_metric_to_suffix.items():
                    metric_value = final_row.get(metric_name)
                    if metric_value is None:
                        continue
                    filter_key, filter_spec = _pick_filter_spec(
                        filter_settings,
                        metric_suffix,
                        model_num,
                    )
                    if not filter_key or not filter_spec:
                        continue
                    threshold = filter_spec.get("threshold")
                    if threshold is None:
                        continue
                    higher = filter_spec.get("higher", True)
                    passed = metric_value >= threshold if higher else metric_value <= threshold
                    if not passed:
                        failed_filters.append(filter_key)

                final_row["overall_passed_filters"] = len(failed_filters) == 0
                final_row["failed_filters"] = ";".join(failed_filters)
                if failed_filters:
                    final_row["status"] = "failed_filter"
                all_final_scores.append(final_row)
                
                logger.info(
                    f"[{idx+1}/{len(af2_output_pdbs)}] {design_name}: "
                    f"relax={t_relax:.1f}s, score={t_score:.1f}s — "
                    f"dG={interface_scores.get('interface_dG', 'N/A')}, "
                    f"sc={interface_scores.get('interface_sc', 'N/A')}"
                )
                
            except Exception as e:
                logger.error(f"PyRosetta scoring failed for {design_name}: {e}")
                all_final_scores.append({
                    "design_name": design_name,
                    "status": "rosetta_failed",
                    "reason": str(e)
                })
        
        # Save final scoring summary
        scoring_df = pd.DataFrame(all_final_scores)
        scoring_summary_path = scoring_output_dir / "scoring_summary.csv"
        
        # Reorder columns
        priority_cols = ["design_name", "status", "pLDDT", "pTM", "i_pTM", "pAE", "i_pAE"]
        all_cols = priority_cols + [c for c in scoring_df.columns if c not in priority_cols and c != "sequence"] + ["sequence"]
        all_cols = [c for c in all_cols if c in scoring_df.columns]
        scoring_df = scoring_df[all_cols]
        
        scoring_df.to_csv(scoring_summary_path, index=False)
        
        relax_avg = relax_total_time / len(af2_output_pdbs) if af2_output_pdbs else 0
        score_avg = score_total_time / len(af2_output_pdbs) if af2_output_pdbs else 0
        
        logger.info(f"PyRosetta scoring complete:")
        logger.info(f"  Total relax time:  {relax_total_time:.1f}s ({relax_total_time/60:.1f} min)")
        logger.info(f"  Total score time:  {score_total_time:.1f}s ({score_total_time/60:.1f} min)")
        logger.info(f"  Avg relax/sample:  {relax_avg:.1f}s")
        logger.info(f"  Avg score/sample:  {score_avg:.1f}s")
        logger.info(f"  Saved results to: {scoring_summary_path}")
        
        # Print timing comparison
        logger.info("=" * 60)
        logger.info("TIMING COMPARISON")
        logger.info("=" * 60)
        af2_avg = af2_total_time / len(af2_output_pdbs) if af2_output_pdbs else 0
        total_pipeline_time = af2_total_time + relax_total_time + score_total_time
        logger.info(f"  AF2 prediction:    {af2_total_time:.1f}s total, {af2_avg:.1f}s/sample")
        logger.info(f"  PyRosetta relax:   {relax_total_time:.1f}s total, {relax_avg:.1f}s/sample")
        logger.info(f"  PyRosetta scoring: {score_total_time:.1f}s total, {score_avg:.1f}s/sample")
        logger.info(f"  Total pipeline:    {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} min)")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Initial Generation Pipeline Complete")
    logger.info("=" * 60)
    
    summary = {
        "target_pdb": target_pdb,
        "n_linker_starting_points": n_linker_starting_points,
        "direct_binder_mode": direct_binder_mode,
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

