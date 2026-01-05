#!/usr/bin/env python
"""
LaProteina Pipeline CLI

End-to-end binder generation and scoring.

Usage:
    python pipeline_cli.py --target_pdb <path> --output_dir <dir> [options]
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="LaProteina Binder Generation & Scoring Pipeline"
    )
    parser.add_argument(
        "--target_pdb",
        type=str,
        required=True,
        help="Path to target PDB file (with linker if applicable)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pipeline_output",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints_laproteina/LD5_motif_idx_tip.ckpt",
        help="Path to LaProteina model checkpoint",
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
        default=10,
        help="Number of binder candidates to generate",
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
        "--linker_length",
        type=int,
        default=0,
        help="Length of linker between target and binder to skip",
    )
    parser.add_argument(
        "--atom_mode",
        type=str,
        default="tip_atoms",
        choices=["tip_atoms", "all_atom", "ca", "bb3o"],
        help="Atom selection mode for conditioning",
    )
    parser.add_argument(
        "--skip_scoring",
        action="store_true",
        help="Skip scoring step (generate and extract only)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=None,
        help="Target length (auto-detected if not provided)",
    )
    parser.add_argument(
        "--reference_pdb",
        type=str,
        default=None,
        help="Path to reference full PDB for merging (if different from target_pdb)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.target_pdb):
        print(f"Error: Target PDB not found: {args.target_pdb}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Import modules
    from proteinfoundation.scoring_pipeline import LaProteinaScoringPipeline
    from proteinfoundation.config_builder import get_pdb_chain_length
    
    # Get target length
    if args.target_length is None:
        target_length = get_pdb_chain_length(args.target_pdb, "A")
        print(f"Auto-detected target length: {target_length}")
    else:
        target_length = args.target_length
    
    # Print configuration
    print("\n" + "=" * 60)
    print("LaProteina Pipeline Configuration")
    print("=" * 60)
    print(f"Target PDB:      {args.target_pdb}")
    print(f"Target length:   {target_length}")
    print(f"Binder length:   {args.binder_min}-{args.binder_max}")
    print(f"Linker length:   {args.linker_length}")
    print(f"N samples:       {args.n_samples}")
    print(f"Atom mode:       {args.atom_mode}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Score:           {not args.skip_scoring}")
    print("=" * 60 + "\n")
    
    # Initialize pipeline
    pipeline = LaProteinaScoringPipeline(
        checkpoint_path=args.checkpoint,
        autoencoder_path=args.autoencoder if os.path.exists(args.autoencoder) else None,
    )
    
    # Run pipeline
    candidates = pipeline.run(
        target_pdb=args.target_pdb,
        target_length=target_length,
        binder_length_range=(args.binder_min, args.binder_max),
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        linker_length=args.linker_length,
        batch_size=args.batch_size,
        score_candidates=not args.skip_scoring,
        atom_selection_mode=args.atom_mode,
        reference_pdb=args.reference_pdb,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Generated {len(candidates)} candidates\n")
    
    for i, candidate in enumerate(candidates):
        print(f"Candidate {i+1}:")
        print(f"  PDB:      {candidate.pdb_path}")
        print(f"  Length:   {len(candidate.binder_sequence)} aa")
        print(f"  Sequence: {candidate.binder_sequence[:30]}...")
        if candidate.scores:
            print(f"  Scores:   {candidate.scores}")
        print()
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
