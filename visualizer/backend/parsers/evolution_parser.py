"""
Parser for evolution output directory structure.
Reads evolution directories and extracts metadata, metrics, and file paths.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import numpy as np


@dataclass
class EvolutionRun:
    """Represents a single evolution run."""
    name: str  # e.g., "evolution_20251230_1813" or "evolution_v2_20260207_1500"
    date_str: str  # e.g., "20251230_1813"
    path: Path
    rounds: List['EvolutionRound'] = field(default_factory=list)
    is_lineage_based: bool = False  # True for evolve_v2 runs with lineage_* dirs
    lineage_ids: List[str] = field(default_factory=list)  # e.g., ["L0000", "L0001", ...]


@dataclass
class EvolutionRound:
    """Represents a single evolution round."""
    round_number: int
    path: Path
    designs: List['Design'] = field(default_factory=list)
    scoring_summary_path: Optional[Path] = None
    af2_predictions_dir: Optional[Path] = None
    relaxed_dir: Optional[Path] = None
    generated_dir: Optional[Path] = None


@dataclass
class Design:
    """Represents a single design with its metrics and file paths."""
    design_name: str
    model: int
    round_number: int
    metrics: Dict[str, any] = field(default_factory=dict)
    af2_pdb_path: Optional[Path] = None
    relaxed_pdb_path: Optional[Path] = None
    laproteina_pdb_path: Optional[Path] = None
    status: str = "unknown"


class EvolutionParser:
    """Parser for evolution output directories."""
    
    def __init__(self, evolution_output_dir: str = "evolution_output"):
        # Resolve to absolute path to ensure consistent path handling
        self.evolution_output_dir = Path(evolution_output_dir).resolve()
        if not self.evolution_output_dir.exists():
            raise ValueError(f"Evolution output directory not found: {evolution_output_dir}")
        self._run_cache = {}  # Cache for parsed evolution runs
    
    def list_evolution_runs(self) -> List[str]:
        """List all evolution run directories."""
        runs = []
        for item in self.evolution_output_dir.iterdir():
            if item.is_dir() and item.name.startswith("evolution_"):
                runs.append(item.name)
        return sorted(runs, reverse=True)  # Most recent first
    
    def parse_evolution_run(self, run_name: str) -> EvolutionRun:
        """Parse basic info about an evolution run directory with caching.
        
        Handles two directory layouts:
        - Classic: evolution_{date}/round_*/scored/scoring_summary.csv
        - Lineage (v2): evolution_v2_{date}/lineage_*/round_*/scored/scoring_summary.csv
        """
        if run_name in self._run_cache:
            return self._run_cache[run_name]
            
        run_path = self.evolution_output_dir / run_name
        
        if not run_path.exists():
            raise ValueError(f"Evolution run not found: {run_name}")
        
        # Extract date string
        date_str = run_name.replace("evolution_v2_", "").replace("evolution_", "")
        
        # Detect layout: check for lineage_* directories
        lineage_dirs = sorted(
            [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("lineage_")],
            key=lambda x: x.name
        )
        is_lineage_based = len(lineage_dirs) > 0
        
        # Also check for classic round_* directories at top level
        top_level_round_dirs = sorted(
            [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("round_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        
        run = EvolutionRun(
            name=run_name,
            date_str=date_str,
            path=run_path,
            is_lineage_based=is_lineage_based,
            lineage_ids=[d.name.replace("lineage_", "") for d in lineage_dirs]
        )
        
        if is_lineage_based:
            # Lineage layout: discover rounds across all lineages
            all_round_numbers = set()
            for lineage_dir in lineage_dirs:
                for d in lineage_dir.iterdir():
                    if d.is_dir() and d.name.startswith("round_"):
                        try:
                            round_num = int(d.name.split("_")[1])
                            all_round_numbers.add(round_num)
                        except (ValueError, IndexError):
                            pass
            
            for round_num in sorted(all_round_numbers):
                # For lineage runs, the "path" points to the run root (rounds are virtual)
                evolution_round = EvolutionRound(
                    round_number=round_num,
                    path=run_path  # Virtual: actual data is inside lineage dirs
                )
                run.rounds.append(evolution_round)
        else:
            # Classic layout: round_* directories at top level
            for round_dir in top_level_round_dirs:
                round_num = int(round_dir.name.split("_")[1])
                evolution_round = EvolutionRound(
                    round_number=round_num,
                    path=round_dir
                )
                run.rounds.append(evolution_round)
        
        self._run_cache[run_name] = run
        return run

    def parse_round(self, run_name: str, round_number: int) -> Optional[EvolutionRound]:
        """Parse a specific round of an evolution run on demand.
        
        For lineage-based runs, aggregates designs from all lineages for the given round.
        """
        run = self.parse_evolution_run(run_name)
        
        # Find the round object
        round_obj = None
        for r in run.rounds:
            if r.round_number == round_number:
                round_obj = r
                break
        
        if not round_obj:
            return None
            
        # If already parsed (has designs), return it
        if round_obj.designs:
            return round_obj
        
        if run.is_lineage_based:
            # Lineage layout: aggregate across all lineage directories
            round_obj.designs = self._parse_lineage_round(run, round_number)
            # Link PDB files for lineage designs
            self._link_pdb_files_lineage(run, round_obj, round_number)
        else:
            # Classic layout
            round_dir = round_obj.path
            scored_dir = round_dir / "scored"
            if scored_dir.exists():
                csv_path = scored_dir / "scoring_summary.csv"
                if csv_path.exists():
                    round_obj.scoring_summary_path = csv_path
                    round_obj.designs = self._parse_scoring_csv(csv_path, round_number)
                
                af2_dir = scored_dir / "af2_predictions"
                relaxed_dir = scored_dir / "relaxed"
                generated_dir = round_dir / "generated"
                
                if af2_dir.exists():
                    round_obj.af2_predictions_dir = af2_dir
                if relaxed_dir.exists():
                    round_obj.relaxed_dir = relaxed_dir
                if generated_dir.exists():
                    round_obj.generated_dir = generated_dir
                
                self._link_pdb_files(round_obj)
            
        return round_obj
    
    def _parse_lineage_round(self, run: EvolutionRun, round_number: int) -> List[Design]:
        """Parse designs from all lineages for a specific round number."""
        all_designs = []
        
        for lineage_id in run.lineage_ids:
            lineage_dir = run.path / f"lineage_{lineage_id}"
            round_dir = lineage_dir / f"round_{round_number}"
            
            if not round_dir.exists():
                continue
            
            scored_dir = round_dir / "scored"
            csv_path = scored_dir / "scoring_summary.csv" if scored_dir.exists() else None
            
            if csv_path and csv_path.exists():
                designs = self._parse_scoring_csv(csv_path, round_number)
                # Tag each design with its lineage ID
                for design in designs:
                    design.metrics["lineage_id"] = lineage_id
                all_designs.extend(designs)
        
        return all_designs
    
    def _link_pdb_files_lineage(self, run: EvolutionRun, round_obj: EvolutionRound, round_number: int):
        """Link PDB files for lineage-based runs.
        
        PDBs are inside: lineage_{id}/round_{N}/scored/af2_predictions/ etc.
        """
        if not round_obj.designs:
            return
        
        output_dir_name = self.evolution_output_dir.name
        
        # Build a lookup of design_name -> lineage_id from the designs (tagged in _parse_lineage_round)
        design_to_lineage = {}
        for design in round_obj.designs:
            lineage_id = design.metrics.get("lineage_id", "")
            if lineage_id:
                design_to_lineage[design.design_name] = lineage_id
        
        for design in round_obj.designs:
            lineage_id = design_to_lineage.get(design.design_name, "")
            if not lineage_id:
                continue
            
            lineage_dir = run.path / f"lineage_{lineage_id}"
            round_dir = lineage_dir / f"round_{round_number}"
            scored_dir = round_dir / "scored"
            
            # Build relative path prefix for this lineage's round
            rel_prefix = Path(output_dir_name) / run.name / f"lineage_{lineage_id}" / f"round_{round_number}"
            
            # Expected filename patterns
            expected_stem = f"{design.design_name}_model{design.model}"
            expected_pdb = f"{expected_stem}.pdb"
            
            # AF2
            af2_dir = scored_dir / "af2_predictions"
            if af2_dir.exists():
                af2_pdb = af2_dir / expected_pdb
                if af2_pdb.exists():
                    design.af2_pdb_path = rel_prefix / "scored" / "af2_predictions" / expected_pdb
                else:
                    # Fuzzy match
                    for f in af2_dir.glob("*.pdb"):
                        if design.design_name in f.name:
                            design.af2_pdb_path = rel_prefix / "scored" / "af2_predictions" / f.name
                            break
            
            # Relaxed
            relaxed_dir = scored_dir / "relaxed"
            if relaxed_dir.exists():
                relaxed_pdb = relaxed_dir / expected_pdb
                if relaxed_pdb.exists():
                    design.relaxed_pdb_path = rel_prefix / "scored" / "relaxed" / expected_pdb
                else:
                    for f in relaxed_dir.glob("*.pdb"):
                        if design.design_name in f.name:
                            design.relaxed_pdb_path = rel_prefix / "scored" / "relaxed" / f.name
                            break
            
            # LaProteina (generated variants)
            generated_dir = round_dir / "generated"
            if generated_dir.exists():
                for variants_dir in generated_dir.iterdir():
                    if variants_dir.is_dir():
                        for sample_dir in variants_dir.iterdir():
                            if sample_dir.is_dir():
                                for pdb_file in sample_dir.glob("*.pdb"):
                                    if design.design_name in pdb_file.stem:
                                        design.laproteina_pdb_path = (
                                            rel_prefix / "generated" / variants_dir.name / sample_dir.name / pdb_file.name
                                        )
                                        break
    
    def _parse_scoring_csv(self, csv_path: Path, round_number: int) -> List[Design]:
        """Parse scoring summary CSV and create Design objects."""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not parse CSV {csv_path}: {e}")
            return []
        
        designs = []
        for _, row in df.iterrows():
            design_name = row.get("design_name", "")
            model = row.get("model", 1)
            
            # Extract all metrics, converting NaN to None for JSON serialization
            metrics = {}
            for col in df.columns:
                if col not in ["design_name", "model", "sequence", "status"]:
                    value = row.get(col)
                    # Convert NaN/NaT to None for JSON compatibility
                    if pd.isna(value):
                        metrics[col] = None
                    else:
                        # Convert numpy types to Python native types
                        if pd.api.types.is_integer_dtype(type(value)):
                            metrics[col] = int(value)
                        elif pd.api.types.is_float_dtype(type(value)):
                            metrics[col] = float(value)
                        else:
                            metrics[col] = value
            
            design = Design(
                design_name=str(design_name),
                model=int(model) if pd.notna(model) else 1,
                round_number=round_number,
                metrics=metrics,
                status=row.get("status", "unknown")
            )
            
            designs.append(design)
        
        return designs
    
    def _link_pdb_files(self, round_obj: EvolutionRound):
        """Link PDB file paths to designs without expensive directory scanning."""
        if not round_obj.designs:
            return
        
        # Get base paths for relative path construction
        # We use strings for speed and avoid repeated resolve() calls
        output_dir_name = self.evolution_output_dir.name
        round_rel_path = Path(output_dir_name) / round_obj.path.relative_to(self.evolution_output_dir)
        
        af2_dir = round_obj.af2_predictions_dir
        relaxed_dir = round_obj.relaxed_dir
        generated_dir = round_obj.generated_dir
        
        # Use a simple set of filenames for fast existence checks
        af2_filenames = set()
        if af2_dir and af2_dir.exists():
            try:
                af2_filenames = {f.name for f in af2_dir.iterdir() if f.suffix == '.pdb'}
            except Exception: pass
            
        relaxed_filenames = set()
        if relaxed_dir and relaxed_dir.exists():
            try:
                relaxed_filenames = {f.name for f in relaxed_dir.iterdir() if f.suffix == '.pdb'}
            except Exception: pass

        # For LaProteina, we'll still need some directory traversal but we'll keep it light
        # Mapping: stem -> relative path string
        laproteina_lookup = {}
        if generated_dir and generated_dir.exists():
            try:
                # Optimized scan: only look at the expected structure
                for evolved_dir in generated_dir.iterdir():
                    if evolved_dir.is_dir() and evolved_dir.name.endswith("_evolved"):
                        for sample_dir in evolved_dir.iterdir():
                            if sample_dir.is_dir():
                                for pdb_file in sample_dir.glob("*.pdb"):
                                    # Store relative path string immediately
                                    rel_path = round_rel_path / "generated" / evolved_dir.name / sample_dir.name / pdb_file.name
                                    laproteina_lookup[pdb_file.stem] = rel_path
            except Exception: pass

        for design in round_obj.designs:
            # Expected filenames
            expected_stem = f"{design.design_name}_model{design.model}"
            expected_pdb = f"{expected_stem}.pdb"
            
            # AF2
            if expected_pdb in af2_filenames:
                design.af2_pdb_path = round_rel_path / "scored" / "af2_predictions" / expected_pdb
            else:
                # Fuzzy match if needed (less common)
                for fname in af2_filenames:
                    if design.design_name in fname and f"model{design.model}" in fname:
                        design.af2_pdb_path = round_rel_path / "scored" / "af2_predictions" / fname
                        break
            
            # Relaxed
            if expected_pdb in relaxed_filenames:
                design.relaxed_pdb_path = round_rel_path / "scored" / "relaxed" / expected_pdb
            else:
                for fname in relaxed_filenames:
                    if design.design_name in fname and f"model{design.model}" in fname:
                        design.relaxed_pdb_path = round_rel_path / "scored" / "relaxed" / fname
                        break
            
            # LaProteina
            if "_model" in design.design_name:
                design_base_with_sample = design.design_name.split("_model")[0]
            else:
                design_base_with_sample = design.design_name
            
            # Direct lookup in our optimized scan
            if design_base_with_sample in laproteina_lookup:
                design.laproteina_pdb_path = laproteina_lookup[design_base_with_sample]
            else:
                # Last resort pattern match
                design_base = design_base_with_sample.split('_')[0] if '_' in design_base_with_sample else design_base_with_sample
                for stem, rel_path in laproteina_lookup.items():
                    if design_base_with_sample in stem or (design_base in stem and "_sample_" in stem):
                        design.laproteina_pdb_path = rel_path
                        break
    
    def get_run_notes(self, run_name: str) -> str:
        """Get notes for an evolution run."""
        run_path = self.evolution_output_dir / run_name
        notes_path = run_path / "notes.txt"
        
        if notes_path.exists():
            try:
                return notes_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Error reading notes for {run_name}: {e}")
                return ""
        return ""

    def save_run_notes(self, run_name: str, notes: str) -> bool:
        """Save notes for an evolution run."""
        run_path = self.evolution_output_dir / run_name
        
        if not run_path.exists():
            return False
            
        notes_path = run_path / "notes.txt"
        try:
            notes_path.write_text(notes, encoding="utf-8")
            return True
        except Exception as e:
            print(f"Error saving notes for {run_name}: {e}")
            return False

    def get_design_summary(self, run_name: str) -> Dict:
        """Get a lightweight summary of an evolution run."""
        run = self.parse_evolution_run(run_name)
        
        summary = {
            "run_name": str(run.name),
            "date": str(run.date_str),
            "num_rounds": len(run.rounds),
            "is_lineage_based": run.is_lineage_based,
            "num_lineages": len(run.lineage_ids) if run.is_lineage_based else 0,
            "rounds": []
        }
        
        for round_obj in run.rounds:
            round_summary = {
                "round_number": int(round_obj.round_number),
                "path": str(round_obj.path)
            }
            summary["rounds"].append(round_summary)
        
        return summary

