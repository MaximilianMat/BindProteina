"""
Initial Sample Generation for LaProteina V2 Pipeline

Generates a diverse pool of initial binder candidates with:
- Multiple linker configurations (varying lengths and attachment edges)
- Full protein context for collision-free generation
- Rosetta pre-filtering to eliminate poor designs
- AF2 scoring for final candidate selection

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
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
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
from laproteina_scoring.rosetta_prefilter import RosettaPreFilter, PrefilterResult


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
        "--skip_prefilter",
        action="store_true",
        help="Skip Rosetta pre-filtering (not recommended)"
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip AF2 scoring (generate and pre-filter only)"
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
    # n_samples is the TOTAL number of samples (distributed across configs)
    n_total_samples = lp_settings.get("n_samples", 50)
    # batch_size for generation (affects GPU memory usage)
    batch_size = lp_settings.get("batch_size", 2)
    
    prefilter_settings_path = settings.get(
        "rosetta_prefilter_settings", 
        "settings_filters/laProteina_filters.json"
    )
    # Enable PyRosetta relaxation before scoring (improves metrics but slower)
    do_rosetta_relax = settings.get("rosetta_relax", True)
    device = settings.get("device", "cuda")
    
    # Number of linker configs: we'll generate one config per sample batch
    # Each batch uses a different linker config (cycling through)
    n_linker_configs = settings.get("n_linker_configs", min(n_total_samples, 8))
    
    logger.info(f"Settings:")
    logger.info(f"  target_pdb: {target_pdb}")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  linker_length_range: {linker_length_range}")
    logger.info(f"  n_total_samples: {n_total_samples}")
    logger.info(f"  n_linker_configs: {n_linker_configs}")
    logger.info(f"  binder_length_range: {binder_length_range}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  rosetta_relax: {do_rosetta_relax}")
    
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
    
    if not hotspot_string:
        logger.error("target_hotspot_residues is required for linker generation")
        sys.exit(1)
    
    # Get target length
    target_length = get_target_length(target_pdb, chain_id="A")
    logger.info(f"Target protein: {Path(target_pdb).name}, length: {target_length}")
    
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
        n_attachment_points=n_linker_configs  # Distribute attachment points based on config count
    )
    
    linker_configs = linker_generator.generate_configs(n_linker_configs)
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
        device=device
    )
    generator._load_model()  # Preload model
    
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    
    # Distribute total samples across configs (one config per batch)
    n_configs = len(prepared_targets)
    samples_per_config = n_total_samples // n_configs
    remainder = n_total_samples % n_configs
    
    logger.info(f"Distributing {n_total_samples} samples across {n_configs} configs")
    logger.info(f"  Base samples per config: {samples_per_config}")
    logger.info(f"  Remainder (extra samples for first configs): {remainder}")
    
    all_generated_pdbs = []
    for idx, prepared_target in enumerate(prepared_targets):
        # First 'remainder' configs get one extra sample
        n_samples_this_config = samples_per_config + (1 if idx < remainder else 0)
        
        if n_samples_this_config == 0:
            logger.info(f"Config {idx}: Skipping (no samples allocated)")
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
        logger.info(f"Config {idx}: Generated {len(pdbs)} samples (total: {len(all_generated_pdbs)})")
        
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
    # Phase 4: Rosetta Pre-Filtering
    # =========================================================================
    if not args.skip_prefilter:
        logger.info("=" * 60)
        logger.info("Phase 4: Rosetta Pre-Filtering")
        logger.info("=" * 60)
        
        # Resolve prefilter settings path
        prefilter_settings_full = fix_path(prefilter_settings_path)
        if not os.path.isabs(prefilter_settings_full):
            repo_root = Path(__file__).parent.parent.parent
            prefilter_settings_full = str(repo_root / prefilter_settings_full)
        
        prefilter_output_dir = output_dir / "prefilter_results"
        
        # Determine target+linker length for chain splitting
        # Use the first prepared target's total_fixed_length as reference
        if prepared_targets:
            split_point = prepared_targets[0].total_fixed_length
        else:
            split_point = target_length  # Fallback
        
        prefilter = RosettaPreFilter(
            filter_settings_path=prefilter_settings_full,
            do_relax=do_rosetta_relax,
            hotspot_residues=hotspot_string  # Filter for binders interacting with hotspot region
        )
        
        prefilter_result = prefilter.prefilter_batch(
            pdb_paths=all_generated_pdbs,
            target_len=split_point,
            output_dir=str(prefilter_output_dir),
            save_split_pdbs=True  # Save split PDBs for inspection
        )
        
        passing_pdbs = prefilter_result.passing_pdbs
        logger.info(f"Pre-filter: {len(passing_pdbs)}/{len(all_generated_pdbs)} passed")
    else:
        logger.info("Skipping Rosetta pre-filtering")
        passing_pdbs = all_generated_pdbs
    
    # =========================================================================
    # Phase 5: Remove Linker & Prepare for Scoring
    # =========================================================================
    if not args.skip_scoring and passing_pdbs:
        logger.info("=" * 60)
        logger.info("Phase 5: Remove Linker & Prepare for Scoring")
        logger.info("=" * 60)
        
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
        
        # Remove linker from all passing PDbs
        # This creates clean target (chain A) + binder (chain B) structures
        scoring_input_dir = output_dir / "scoring_input"
        scoring_input_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Removing linkers from {len(passing_pdbs)} structures (varying linker lengths)...")
        scoring_pdbs = []
        for pdb_path in passing_pdbs:
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
        for pdb_path in passing_pdbs:
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
        
        # =========================================================================
        # Phase 6: AF2 Scoring
        # =========================================================================
        if scoring_pdbs:
            logger.info("=" * 60)
            logger.info("Phase 6: AF2 Scoring")
            logger.info("=" * 60)
            
            scoring_output_dir = output_dir / "scored"
            
            score_cmd = [
                sys.executable,
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                    "laproteina_scoring", "score.py"
                ),
                "--input_dir", str(scoring_input_dir),
                "--output_dir", str(scoring_output_dir),
            ]
            
            logger.info(f"Running scoring: {' '.join(score_cmd)}")
            
            try:
                subprocess.run(score_cmd, check=True)
                logger.info(f"Scoring complete. Results in: {scoring_output_dir}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Scoring failed: {e}")
        else:
            logger.warning("No structures available for scoring after linker removal")
    else:
        if args.skip_scoring:
            logger.info("Skipping AF2 scoring")
        else:
            logger.info("No samples passed pre-filter, skipping scoring")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Initial Generation Pipeline Complete")
    logger.info("=" * 60)
    
    summary = {
        "target_pdb": target_pdb,
        "n_linker_configs": len(linker_configs),
        "n_total_samples_requested": n_total_samples,
        "total_generated": len(all_generated_pdbs),
        "passed_prefilter": len(passing_pdbs) if not args.skip_prefilter else "skipped",
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

