#!/usr/bin/env python
"""
Test script for LaProteina Generator API.

Usage:
    python test_generator.py --target_pdb <path> --output_dir <dir> [--n_samples N]
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    # Set matmul precision for performance on Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description="Test LaProteina Generator")
    parser.add_argument(
        "--target_pdb",
        type=str,
        required=True,
        help="Path to target PDB file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Output directory for generated PDBs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints_laproteina/LD5_motif_idx_tip.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--autoencoder",
        type=str,
        default="./checkpoints_laproteina/AE3_motif.ckpt",
        help="Path to autoencoder checkpoint",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--binder_min",
        type=int,
        default=20,
        help="Minimum binder length",
    )
    parser.add_argument(
        "--binder_max",
        type=int,
        default=80,
        help="Maximum binder length",
    )
    parser.add_argument(
        "--atom_mode",
        type=str,
        default="tip_atoms",
        choices=["tip_atoms", "all_atom", "ca_only", "backbone"],
        help="Atom selection mode for motif conditioning",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.target_pdb):
        print(f"Error: Target PDB not found: {args.target_pdb}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Import after path setup
    from proteinfoundation.laproteina_generator import LaProteinaGenerator
    from proteinfoundation.config_builder import get_pdb_chain_length
    
    print(f"LaProteina Generator Test")
    print(f"=" * 50)
    print(f"Target PDB: {args.target_pdb}")
    print(f"Output dir: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"N samples:  {args.n_samples}")
    print(f"Binder length: {args.binder_min}-{args.binder_max}")
    print(f"Atom mode:  {args.atom_mode}")
    print(f"=" * 50)
    
    # Get target length
    target_length = get_pdb_chain_length(args.target_pdb, "A")
    print(f"\nDetected target length: {target_length} residues")
    
    # Initialize generator
    print(f"\nInitializing generator...")
    generator = LaProteinaGenerator(
        checkpoint_path=args.checkpoint,
        autoencoder_path=args.autoencoder if os.path.exists(args.autoencoder) else None,
    )
    
    # Generate binders
    print(f"\nGenerating {args.n_samples} binder candidates...")
    pdbs = generator.generate_binder(
        target_pdb=args.target_pdb,
        target_residue_range=(1, target_length),
        binder_length_range=(args.binder_min, args.binder_max),
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        atom_selection_mode=args.atom_mode,
    )
    
    print(f"\nGeneration complete!")
    print(f"Generated {len(pdbs)} structures:")
    for pdb in pdbs:
        print(f"  - {pdb}")


if __name__ == "__main__":
    main()
