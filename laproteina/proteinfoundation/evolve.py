
import argparse
import os
import sys
import subprocess
import json
import torch
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

# Adjust path to import BindCraft modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from laproteina.proteinfoundation.optimizer import IterativeOptimizer
from laproteina.proteinfoundation.laproteina_generator import LaProteinaGenerator, GenerationConfig
from laproteina.proteinfoundation.config_builder import get_pdb_chain_length

def load_settings(settings_path):
    """Load settings from a JSON file."""
    with open(settings_path, 'r') as f:
        return json.load(f)

def main():
    # Set matmul precision for performance on Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    parser = argparse.ArgumentParser(description="Iterative Optimization Workflow (Binder Evolution)")
    
    # Settings file (alternative to individual arguments)
    parser.add_argument("--settings", type=str, default=None, help="Path to JSON settings file (e.g., settings_evolution/CLD18.json). Command-line arguments override settings file values.")
    
    # Inputs
    parser.add_argument("--target_pdb", type=str, default=None, help="Original native target PDB")
    parser.add_argument("--input_csv", type=str, default=None, help="Scoring summary CSV from the previous run")
    parser.add_argument("--seeds_dir", type=str, default=None, help="Directory containing the PDBs listed in input_csv")
    parser.add_argument("--checkpoint", type=str, default=None, help="LaProteina checkpoint path")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base output directory")
    
    # Evolution Parameters
    parser.add_argument("--rounds", type=int, default=None, help="Number of evolution rounds")
    parser.add_argument("--n_samples_per_seed", type=int, default=None, help="Number of samples to generate per seed")
    parser.add_argument("--metric", type=str, default=None, help="Metric to rank seeds")
    parser.add_argument("--top_n", type=int, default=None, help="Number of seeds to pick per round")
    parser.add_argument("--plddt_threshold", type=float, default=None, help="pLDDT threshold for fixing binder residues")
    parser.add_argument("--linker_length", type=int, default=None, help="Length of linker (used for binder extraction)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--target_hotspot_residues", type=str, default=None, help="Target hotspot regions (e.g., '300-350,400-450'). Seeds without interface residues in these regions will be filtered out.")
    parser.add_argument("--target_blacklist_residues", type=str, default=None, help="Target blacklist regions (e.g., '1-100,200-300'). Seeds with interface residues in these regions will be filtered out.")

    args = parser.parse_args()
    
    # Load settings from file if provided
    settings = {}
    if args.settings:
        settings_path = Path(args.settings)
        if not settings_path.is_absolute():
            # Try relative to repo root
            repo_root = Path(__file__).parent.parent.parent
            settings_path = repo_root / settings_path
        if not settings_path.exists():
            logger.error(f"Settings file not found: {args.settings}")
            sys.exit(1)
        settings = load_settings(settings_path)
        logger.info(f"Loaded settings from: {settings_path}")
    
    # Merge settings with command-line arguments (CLI args take precedence)
    def get_arg(key, cli_value, default=None):
        """Get argument value: CLI > settings > default"""
        if cli_value is not None:
            return cli_value
        if key in settings:
            return settings[key]
        return default
    
    # Required arguments (must be in settings or CLI)
    target_pdb = get_arg("target_pdb", args.target_pdb)
    input_csv = get_arg("input_csv", args.input_csv)
    seeds_dir = get_arg("seeds_dir", args.seeds_dir)
    checkpoint = get_arg("checkpoint", args.checkpoint)
    
    if not all([target_pdb, input_csv, seeds_dir, checkpoint]):
        parser.error("Missing required arguments. Either provide --settings file or specify --target_pdb, --input_csv, --seeds_dir, and --checkpoint")
    
    # Optional arguments with defaults
    output_base_dir = get_arg("output_base_dir", args.output_base_dir, "./evolution_output")
    rounds = get_arg("rounds", args.rounds, 3)
    n_samples_per_seed = get_arg("n_samples_per_seed", args.n_samples_per_seed, 6)
    metric = get_arg("metric", args.metric, "i_pTM")
    top_n = get_arg("top_n", args.top_n, 3)
    plddt_threshold = get_arg("plddt_threshold", args.plddt_threshold, 55.0)
    linker_length = get_arg("linker_length", args.linker_length, 0)
    device = get_arg("device", args.device, "cuda")
    target_hotspot_residues = get_arg("target_hotspot_residues", args.target_hotspot_residues, None)
    target_blacklist_residues = get_arg("target_blacklist_residues", args.target_blacklist_residues, None)
    
    # Create a namespace-like object for compatibility with rest of code
    class Args:
        pass
    args_obj = Args()
    args_obj.target_pdb = target_pdb
    args_obj.input_csv = input_csv
    args_obj.seeds_dir = seeds_dir
    args_obj.checkpoint = checkpoint
    args_obj.output_base_dir = output_base_dir
    args_obj.rounds = rounds
    args_obj.n_samples_per_seed = n_samples_per_seed
    args_obj.metric = metric
    args_obj.top_n = top_n
    args_obj.plddt_threshold = plddt_threshold
    args_obj.linker_length = linker_length
    args_obj.device = device
    args_obj.target_hotspot_residues = target_hotspot_residues
    args_obj.target_blacklist_residues = target_blacklist_residues
    
    args = args_obj

    # Setup Output
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    main_output_dir = Path(args.output_base_dir) / f"evolution_{date_str}"
    main_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.add(main_output_dir / "evolution.log")
    logger.info(f"Starting Evolution Workflow: {args.rounds} rounds, top {args.top_n} seeds based on {args.metric}")

    # Ensure paths are absolute if they look like they might be missing a leading slash on this system
    def fix_path(p):
        if p and (p.startswith("nfs/") or p.startswith("home/")):
            return "/" + p
        return p

    # Initialize Components
    optimizer = IterativeOptimizer(str(main_output_dir))
    
    # Generator (Load Model Once)
    logger.info("Loading LaProteina Model...")
    ckpt_path = Path(fix_path(args.checkpoint))
    ae_path = ckpt_path.parent / "AE3_motif.ckpt" # Common default?
    
    generator = LaProteinaGenerator(
        checkpoint_path=str(ckpt_path),
        autoencoder_path=str(ae_path) if ae_path.exists() else None,
        device=args.device
    )
    generator._load_model() # Preload

    current_csv = fix_path(args.input_csv)
    current_pdb_dir = Path(fix_path(args.seeds_dir))

    target_length = get_pdb_chain_length(fix_path(args.target_pdb), "A")
    logger.info(f"Native Target Length: {target_length}")

    for round_idx in range(1, args.rounds + 1):
        logger.info(f"=== ROUND {round_idx} ===")
        
        # 1. Select Seeds
        try:
            seeds = optimizer.select_seeds(
                scoring_csv=current_csv,
                metric=args.metric,
                top_n=args.top_n,
                target_hotspot_residues=args.target_hotspot_residues,
                target_blacklist_residues=args.target_blacklist_residues
            )
        except Exception as e:
            logger.error(f"Seed selection failed: {e}")
            break
            
        if not seeds:
            logger.warning("No valid seeds found. Stopping evolution.")
            break
            
        # Prepare Round Directory
        round_dir = optimizer.prepare_round_directory(round_idx, date_str)
        generated_dir = round_dir / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Generate for each seed
        for seed in seeds:
            design_name = seed["design_name"]
            design_model = seed["model"]
            
            # Generate short ID for this design
            short_id = optimizer.get_or_create_id(design_name, round_idx)
            logger.info(f"Design {design_name} -> Short ID: {short_id}")
            
            # Try exact name first - check af2_predictions subdirectory
            seed_pdb_path = current_pdb_dir / "af2_predictions" / f"{design_name}_model{design_model}.pdb"
            # Fallback: check relaxed subdirectory or parent directory
            if not seed_pdb_path.exists():
                seed_pdb_path = current_pdb_dir / "relaxed" / f"{design_name}_model{design_model}.pdb"
            if not seed_pdb_path.exists():
                seed_pdb_path = current_pdb_dir / f"{design_name}_model{design_model}.pdb"
            if not seed_pdb_path.exists():
                logger.warning(f"Could not find PDB for seed {design_name} at any location. Skipping.")
                continue
                
            logger.info(f"Processing Seed: {short_id} (from {seed_pdb_path.name})")
            
            # 3. Optimize/Mask
            try:
                # Create masks directory for this round
                masks_dir = round_dir / "masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                
                contig, explanation, masqueraded_pdb = optimizer.generate_mask_and_contig(
                    pdb_path=str(seed_pdb_path),
                    target_len=target_length,
                    plddt_threshold=args.plddt_threshold,
                    save_dir=masks_dir
                )
                logger.info(f"  > Strategy: {explanation}")
                logger.info(f"  > Masqueraded PDB: {masqueraded_pdb}")
                
                # 4. Generate
                # Use the masqueraded PDB as source (contains Chain A + Stitching)
                # Note: masked PDB now only has fixed parts, contig handles the rest
                generator.generate_binder(
                    target_pdb=masqueraded_pdb,
                    target_residue_range=(1, target_length),
                    n_samples=args.n_samples_per_seed,
                    output_dir=str(generated_dir / f"{short_id}_evolved"),
                    contig_string=contig,
                    linker_length=args.linker_length,
                    design_id=short_id
                )
                
            except Exception as e:
                logger.error(f"  > Generation failed for {short_id}: {e}")
                
        # 5. Score this round
        logger.info(f"Scoring Round {round_idx}...")
        
        # We need to collect all generated files from subfolders to a common scoring dir or score in place
        # score.py takes input_dir and recurses? No, usually glob *.pdb.
        # Let's aggregate first.
        network_scoring_dir = round_dir / "scoring_input"
        network_scoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy and Extract all evolved PDBs (split Chain B)
        # We need to split the single chain into Target (A) and Binder (B) for score.py
        import biotite.structure.io as strucio
        import biotite.structure as struc
        
        target_len = get_pdb_chain_length(fix_path(args.target_pdb), 'A')
        split_point = target_len

        for pdb in generated_dir.rglob("*.pdb"):
             # We assume they are valid PDBs.
             # Generator creates files named like {design_id}_sample_{idx}.pdb
             if "final_complex" in pdb.name or "binder" in pdb.name or "_sample_" in pdb.name: 
                 try:
                     # Load structure
                     structure = strucio.load_structure(str(pdb))
                     
                     # Identify binder residues (residue ID usually resets or continues? LaProteina usually 1-indexed continuous or split?)
                     # If single chain, it's continuous.
                     # But we can just use residue index if we trust the count.
                     # CAUTION: Biotite res_id might be PDB numbering.
                     # Let's rely on atom count or residue index?
                     # Simplest: Split by residue index in the array.
                     
                     # Get all residues
                     residue_starts = struc.get_residue_starts(structure)
                     n_residues = len(residue_starts)
                     
                     # Check if we have enough residues
                     if n_residues <= split_point:
                         logger.warning(f"Skipping {pdb.name}: Too few residues ({n_residues}) for split point {split_point}")
                         continue

                     # Assign Chain IDs
                     # Target + Linker -> A
                     # Binder -> B
                     # Actually, Linker should be part of A? or skipped?
                     # Standard extraction skips linker. 
                     # But score.py might expect full complex? 
                     # If "No binder sequence found", it scans Chain B.
                     # So we MUST have Chain B.
                     # Let's assign Chain B to residues > split_point.
                     
                     # We need to iterate over atoms and set chain_id
                     # This is slow in Python loop, vectorization is better.
                     
                     # Identify residues that are binder
                     # residue_starts returns indices of first atom of each residue.
                     # The binder starts at residue index `split_point` (0-indexed logic: 0 to split_point-1 is Target+Linker?).
                     # Wait, split_point = target_len + linker_len.
                     # So regex 1..split_point is Target+Linker.
                     # residue index `split_point` is the first residue of Binder.
                     
                     binder_start_atom_idx = residue_starts[split_point]
                     
                     # Set Chain B for binder atoms
                     structure.chain_id[binder_start_atom_idx:] = "B"
                     
                     # Renumber Chain B residues starting from 1
                     # Get mask for Chain B
                     chain_b_mask = structure.chain_id == "B"
                     
                     # Get residue starts within Chain B
                     # We can just iterate over residues in Chain B and set their res_id
                     # struc.get_residue_starts returns indices into the atom array
                     # We need to map these to the actual atoms
                     
                     # Efficient way:
                     # Get all residues again (chain IDs changed)
                     # Iterate residues, if chain is B, set res_id
                     
                     # Helper to renumber
                     # A simpler approach using residue_starts for the whole structure:
                     # Any residue index >= split_point is Chain B.
                     # The new res_id should be (residue_index - split_point) + 1
                     
                     for i in range(split_point, n_residues):
                         start = residue_starts[i]
                         if i < n_residues - 1:
                             end = residue_starts[i+1]
                         else:
                             end = len(structure)
                         
                         structure.res_id[start:end] = (i - split_point) + 1
                     
                     # Save to scoring dir - use the PDB filename as-is (already contains short ID if from generator)
                     # The filename will be used as design_name by score.py
                     strucio.save_structure(str(network_scoring_dir / pdb.name), structure)
                     
                 except Exception as e:
                     logger.warning(f"Failed to process {pdb} for scoring: {e}")

        score_output_dir = round_dir / "scored"
        
        # Call score.py
        score_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "laproteina_scoring", "score.py"),
            "--input_dir", str(network_scoring_dir),
            "--output_dir", str(score_output_dir),
            # Do we need settings? Maybe for chains? 
            # We know it's Chain A/B or whatever generator output.
            # Generator outputs A/B usually? 
            # Wait, generator outputs standard LaProteina format.
            # score.py expects?
            # score.py handles it.
             "--filters", "" # Use defaults
        ]
        
        # We might need to pass target pdb? 
        # Actually score.py infers target from input PDB if not specified (self-contained).
        # This is safe.
        
        logger.info(f"Running scoring: {' '.join(score_cmd)}")
        subprocess.run(score_cmd, check=True)
        
        # Update current state for next round
        current_csv = str(score_output_dir / "scoring_summary.csv")
        current_pdb_dir = score_output_dir / "af2_predictions"  # PDBs are in af2_predictions subdirectory
        
        logger.info(f"Round {round_idx} complete. Summary at {current_csv}")

    logger.info("Evolution Workflow Completed.")

if __name__ == "__main__":
    main()
