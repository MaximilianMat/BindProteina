"""
LaProteina Scoring Integration

This module integrates LaProteina generation with the BindCraft scoring pipeline.
It provides an end-to-end workflow: generate binders → extract → score.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import shutil
import json
from datetime import datetime

from loguru import logger


@dataclass
class BinderCandidate:
    """Represents a generated binder candidate with metadata."""
    
    pdb_path: str
    target_pdb: str
    target_length: int
    binder_start: int
    binder_end: int
    binder_sequence: Optional[str] = None
    scores: Dict = field(default_factory=dict)
    generation_config: Dict = field(default_factory=dict)


class LaProteinaScoringPipeline:
    """
    End-to-end pipeline: LaProteina generation → Binder extraction → Scoring.
    
    Example:
        pipeline = LaProteinaScoringPipeline(
            checkpoint_path="checkpoints/LD5_motif_idx_tip.ckpt",
            autoencoder_path="checkpoints/AE3_motif.ckpt"
        )
        
        results = pipeline.run(
            target_pdb="target.pdb",
            target_length=705,
            binder_length_range=(20, 80),
            n_samples=10,
            output_dir="./results"
        )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        autoencoder_path: Optional[str] = None,
        scoring_module_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize the pipeline.
        
        Args:
            checkpoint_path: Path to LaProteina model checkpoint
            autoencoder_path: Path to autoencoder checkpoint
            scoring_module_path: Path to BindCraft functions/ directory for scoring
            device: Device to run on
        """
        self.checkpoint_path = checkpoint_path
        self.autoencoder_path = autoencoder_path
        self.device = device
        self._generator = None
        self._scoring_module_path = scoring_module_path
    @property
    def generator(self):
        """Lazy-load the generator."""
        if self._generator is None:
            from proteinfoundation.laproteina_generator import LaProteinaGenerator
            self._generator = LaProteinaGenerator(
                checkpoint_path=self.checkpoint_path,
                autoencoder_path=self.autoencoder_path,
                device=self.device,
            )
        return self._generator

    def cleanup(self):
        """Free up resources and GPU memory."""
        if self._generator is not None:
            self._generator.unload()
            self._generator = None
    
    def run(
        self,
        target_pdb: str,
        target_length: int,
        binder_length_range: Tuple[int, int] = (20, 80),
        n_samples: int = 10,
        output_dir: str = "./pipeline_output",
        linker_length: int = 0,
        batch_size: int = 4,
        score_candidates: bool = True,
        atom_selection_mode: str = "tip_atoms",
        reference_pdb: Optional[str] = None,
    ) -> List[BinderCandidate]:
        """
        Run the full pipeline.
        
        Args:
            target_pdb: Path to target PDB
            target_length: Number of residues in target (needed for binder extraction)
            binder_length_range: (min, max) length of generated binder
            n_samples: Number of candidates to generate
            output_dir: Output directory
            linker_length: Length of linker to skip during extraction
            score_candidates: Whether to run scoring after generation
            atom_selection_mode: Atom selection for conditioning
            
        Returns:
            List of BinderCandidate objects with results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate candidates
        logger.info(f"Step 1: Generating {n_samples} candidates...")
        generated_dir = output_path / "generated"
        
        generated_pdbs = self.generator.generate_binder(
            target_pdb=target_pdb,
            target_residue_range=(1, target_length),
            binder_length_range=binder_length_range,
            n_samples=n_samples,
            output_dir=str(generated_dir),
            atom_selection_mode=atom_selection_mode,
            linker_length=linker_length,
            max_batch_size=batch_size,
        )
        
        logger.info(f"Generated {len(generated_pdbs)} structures")
        
        # Step 2: Extract binders from generated monomers
        logger.info("Step 2: Extracting binders...")
        extracted_dir = output_path / "extracted"
        complexes_dir = extracted_dir / "complexes"
        binders_dir = extracted_dir / "binders"
        complexes_dir.mkdir(parents=True, exist_ok=True)
        binders_dir.mkdir(parents=True, exist_ok=True)
        
        candidates = self.extract_binders(
            generated_pdbs=generated_pdbs,
            target_pdb=target_pdb,
            target_length=target_length,
            linker_length=linker_length,
            output_dir=str(extracted_dir),
            reference_pdb=reference_pdb,
        )
        
        logger.info(f"Extracted {len(candidates)} binder candidates")
        
        # Step 3: Score candidates (optional)
        if score_candidates:
            logger.info("Step 3: Scoring candidates...")
            scored_dir = output_path / "scored"
            candidates = self.score_binders(
                candidates=candidates,
                output_dir=str(scored_dir),
            )
            logger.info("Scoring complete")
        
        # Save summary
        self._save_summary(candidates, output_path / "summary.json")
        
        return candidates
    
    def extract_binders(
        self,
        generated_pdbs: List[str],
        target_pdb: str,
        target_length: int,
        linker_length: int = 0,
        output_dir: str = "./extracted",
        reference_pdb: Optional[str] = None,
    ) -> List[BinderCandidate]:
        """
        Extract binder portions from generated monomers.
        """
        output_path = Path(output_dir)
        complexes_dir = output_path / "complexes"
        binders_dir = output_path / "binders"
        complexes_dir.mkdir(parents=True, exist_ok=True)
        binders_dir.mkdir(parents=True, exist_ok=True)
        
        candidates = []
        
        for i, gen_pdb in enumerate(generated_pdbs):
            try:
                candidate = self._extract_single_binder(
                    generated_pdb=gen_pdb,
                    target_pdb=target_pdb,
                    target_length=target_length,
                    linker_length=linker_length,
                    complexes_dir=complexes_dir,
                    binders_dir=binders_dir,
                    idx=i,
                    reference_pdb=reference_pdb,
                )
                candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Failed to extract binder from {gen_pdb}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        return candidates
    
    def _extract_single_binder(
        self,
        generated_pdb: str,
        target_pdb: str,
        target_length: int,
        linker_length: int,
        complexes_dir: Path,
        binders_dir: Path,
        idx: int,
        reference_pdb: Optional[str] = None,
    ) -> BinderCandidate:
        """Extract binder from a single generated structure."""
        
        # Ranges (using 1-indexed residue numbers from PDB)
        target_end = target_length
        binder_start = target_length + linker_length + 1
        
        # Read generated structure
        with open(generated_pdb, 'r') as f:
            gen_lines = f.readlines()
        
        # Split into target and binder atoms directly from the generated PDB
        # This ensures they are in the same coordinate space!
        target_atoms = []
        binder_atoms = []
        
        # If a reference PDB is provided, use it for target atoms (Chain A)
        if reference_pdb and os.path.exists(reference_pdb):
            with open(reference_pdb, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        # Force to Chain A
                        new_line = line[:21] + 'A' + line[22:]
                        target_atoms.append(new_line)
        
        current_resnum = -1
        remapped_resnum = 0
        
        for line in gen_lines:
            if not line.startswith('ATOM'):
                continue
                
            try:
                res_num = int(line[22:26].strip())
                
                if res_num <= target_end:
                    if not reference_pdb:
                        # Target part: Force to Chain A
                        new_line = line[:21] + 'A' + line[22:]
                        target_atoms.append(new_line)
                elif res_num >= binder_start:
                    # Binder part: Force to Chain B and renumber from 1
                    if res_num != current_resnum:
                        current_resnum = res_num
                        remapped_resnum += 1
                    
                    new_line = line[:21] + 'B' + f"{remapped_resnum:4d}" + line[26:]
                    binder_atoms.append(new_line)
            except ValueError:
                continue
        
        # Paths
        complex_path = complexes_dir / f"complex_{idx:04d}.pdb"
        binder_only_path = binders_dir / f"binder_{idx:04d}.pdb"
        
        # Write files with proper PDB formatting (MODEL/ENDMDL)
        def write_pdb(path, atoms):
            with open(path, 'w') as f:
                f.write("MODEL     1\n")
                last_chain = None
                for line in atoms:
                    # Check if line is long enough to contain a chain ID at index 21
                    if len(line) > 21 and line.startswith(('ATOM', 'HETATM')):
                        chain = line[21]
                        if last_chain is not None and chain != last_chain:
                            f.write("TER\n")
                        f.write(line)
                        last_chain = chain
                    else:
                        # For shorter lines or non-atom lines, just write them as is
                        f.write(line)
                        # If it's a TER line, we might want to reset last_chain or handle it
                
                # Ensure TER at end if not present
                if atoms and not atoms[-1].startswith(('TER', 'END')):
                    f.write("TER\n")
                f.write("ENDMDL\n")
                f.write("END\n")
        
        write_pdb(complex_path, target_atoms + binder_atoms)
        write_pdb(binder_only_path, binder_atoms)
        
        # Extract sequence
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }
        
        seen_res = set()
        sequence = []
        for line in binder_atoms:
            res_num = int(line[22:26].strip())
            if res_num not in seen_res:
                seen_res.add(res_num)
                resname = line[17:20].strip()
                sequence.append(three_to_one.get(resname, 'X'))
        
        return BinderCandidate(
            pdb_path=str(complex_path),
            target_pdb=target_pdb,
            target_length=target_length,
            binder_start=1,
            binder_end=len(sequence),
            binder_sequence=''.join(sequence),
            generation_config={
                "source_pdb": generated_pdb,
                "linker_length": linker_length,
            }
        )
    
    def score_binders(
        self,
        candidates: List[BinderCandidate],
        output_dir: str = "./scored",
        scoring_config: Optional[Dict] = None,
    ) -> List[BinderCandidate]:
        """
        Score binder candidates using BindCraft scoring functions.
        
        Integrates with the existing laproteina_scoring module.
        
        Args:
            candidates: List of BinderCandidate objects
            output_dir: Output directory for scored structures
            scoring_config: Optional config dict for scoring settings
            
        Returns:
            Updated candidates with scores
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Import scoring functions from BindCraft
        try:
            from laproteina_scoring.config import ScoringConfig
            from laproteina_scoring.pipeline import LaProteinaScorer
            
            # Create scoring config
            if scoring_config:
                config = ScoringConfig(**scoring_config)
            else:
                # Use default config
                config = ScoringConfig()
            
            # Initialize scorer
            scorer = LaProteinaScorer(config)
            
            for candidate in candidates:
                try:
                    # Get target PDB for template masking
                    target_pdb = candidate.target_pdb
                    
                    # Score the complex
                    scores = scorer.score_design(
                        pdb_path=candidate.pdb_path,
                        target_pdb=target_pdb,
                        output_dir=str(output_path),
                        design_name=Path(candidate.pdb_path).stem,
                    )
                    candidate.scores = scores
                    
                except Exception as e:
                    logger.warning(f"Scoring failed for {candidate.pdb_path}: {e}")
                    candidate.scores = {"error": str(e)}
                    
        except ImportError as e:
            logger.warning(
                f"BindCraft scoring module not available: {e}. "
                "Skipping scoring - candidates can be scored later."
            )
            for candidate in candidates:
                candidate.scores = {"status": "not_scored"}
        
        return candidates
    
    def _save_summary(self, candidates: List[BinderCandidate], path: Path):
        """Save pipeline results summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "n_candidates": len(candidates),
            "candidates": [
                {
                    "pdb_path": c.pdb_path,
                    "target_pdb": c.target_pdb,
                    "binder_length": c.binder_end - c.binder_start + 1,
                    "binder_sequence": c.binder_sequence,
                    "scores": c.scores,
                }
                for c in candidates
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {path}")


# Convenience function for standalone use
def run_generation_and_score(
    target_pdb: str,
    output_dir: str,
    checkpoint_path: str,
    autoencoder_path: Optional[str] = None,
    n_samples: int = 10,
    binder_min: int = 20,
    binder_max: int = 80,
) -> List[BinderCandidate]:
    """
    Convenience function to run the full pipeline.
    
    Automatically detects target length from PDB.
    """
    from proteinfoundation.config_builder import get_pdb_chain_length
    
    target_length = get_pdb_chain_length(target_pdb, "A")
    
    pipeline = LaProteinaScoringPipeline(
        checkpoint_path=checkpoint_path,
        autoencoder_path=autoencoder_path,
    )
    
    return pipeline.run(
        target_pdb=target_pdb,
        target_length=target_length,
        binder_length_range=(binder_min, binder_max),
        n_samples=n_samples,
        output_dir=output_dir,
    )
