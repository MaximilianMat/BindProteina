"""
Lineage Evolution Script for LaProteina V2 Pipeline (evolve_v2.py)

Evolves each seed independently with strict 1:1 best-variant selection per round.
This prevents one design from dominating the entire pool.

Key differences from evolve.py:
- Each initial seed is an independent lineage
- Lineages evolve separately (no mixing between lineages)
- Per round, each lineage keeps ONLY the single best variant
- Checkpointing per lineage for fault tolerance and resume capability

Supports two input modes:
1. CSV-based: Provide input_csv + seeds_dir (classic mode, requires scoring results)
2. PDB directory: Provide seeds_pdb_dir (no CSV needed, uses ALL PDBs as seeds)

Usage:
    # From scoring CSV (classic):
    python evolve_v2.py --settings settings_evolution/my_target.json

    # Directly from PDB directory (no scoring needed):
    python evolve_v2.py --settings settings_evolution/my_target.json
    # where settings has "seeds_pdb_dir" instead of "input_csv"
"""

import argparse
import os
import sys
import subprocess
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import shutil

# Adjust path to import BindProteina modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from laproteina.proteinfoundation.optimizer import IterativeOptimizer
from laproteina.proteinfoundation.laproteina_generator import LaProteinaGenerator
from laproteina.proteinfoundation.config_builder import get_pdb_chain_length


@dataclass
class Lineage:
    """Represents an independent evolution lineage."""
    
    lineage_id: str                  # Unique ID for this lineage
    original_seed_name: str          # Original design name from initial pool
    original_seed_pdb: str           # Path to original seed PDB
    original_score: float            # Initial score of the seed
    
    current_round: int = 0           # Current evolution round
    current_best_pdb: str = ""       # Path to current best variant
    current_best_score: float = 0.0  # Score of current best
    
    # Evolution history: list of {round, pdb, score, n_variants_generated}
    history: List[Dict] = field(default_factory=list)
    
    # Status: 'active', 'completed', 'failed'
    status: str = "active"
    
    def __post_init__(self):
        if not self.current_best_pdb:
            self.current_best_pdb = self.original_seed_pdb
        if self.current_best_score == 0.0:
            self.current_best_score = self.original_score


@dataclass
class LineageCheckpoint:
    """Checkpoint data for resume capability."""
    
    lineage_id: str
    current_round: int
    current_best_pdb: str
    current_best_score: float
    status: str


def load_settings(settings_path: str) -> Dict:
    """Load settings from a JSON file."""
    with open(settings_path, 'r') as f:
        return json.load(f)


def fix_path(p: str) -> str:
    """Fix paths that might be missing leading slash."""
    if p and (p.startswith("nfs/") or p.startswith("home/")):
        return "/" + p
    return p


def load_lineages_from_csv(
    scoring_csv: str,
    seeds_dir: str,
    metric: str = "i_pTM",
    higher_is_better: bool = True,
    target_hotspot_residues: Optional[str] = None,
    target_blacklist_residues: Optional[str] = None
) -> List[Lineage]:
    """
    Load initial seeds from scoring CSV and create lineages.
    
    Each seed becomes an independent lineage.
    """
    from laproteina.proteinfoundation.utils.pdb_slicer import parse_hotspot_string
    
    df = pd.read_csv(scoring_csv)
    
    # Filter for successful designs
    if "status" in df.columns:
        df = df[df["status"] != "failed"]
    
    # Ensure metric is numeric
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df.dropna(subset=[metric])
    
    # Get best model per design
    best_models = df.groupby("design_name").first().reset_index()
    
    # Apply hotspot/blacklist filtering if specified
    if target_hotspot_residues or target_blacklist_residues:
        if "target_interface_residues" not in best_models.columns:
            logger.warning("target_interface_residues column not found. Skipping interface filtering.")
        else:
            filtered_seeds = []
            hotspot_ranges = parse_hotspot_string(target_hotspot_residues or "")
            blacklist_ranges = parse_hotspot_string(target_blacklist_residues or "")
            
            for _, row in best_models.iterrows():
                interface_str = row.get("target_interface_residues", "")
                if pd.isna(interface_str) or not interface_str:
                    if hotspot_ranges:
                        continue  # Skip if hotspot filtering required but no interface data
                    filtered_seeds.append(row)
                    continue
                
                try:
                    interface_residues = [int(r.strip()) for r in str(interface_str).split(",") if r.strip()]
                except ValueError:
                    if hotspot_ranges:
                        continue
                    filtered_seeds.append(row)
                    continue
                
                # Check blacklist
                if blacklist_ranges:
                    in_blacklist = any(
                        any(start <= r <= end for start, end in blacklist_ranges)
                        for r in interface_residues
                    )
                    if in_blacklist:
                        continue
                
                # Check hotspot
                if hotspot_ranges:
                    in_hotspot = any(
                        any(start <= r <= end for start, end in hotspot_ranges)
                        for r in interface_residues
                    )
                    if not in_hotspot:
                        continue
                
                filtered_seeds.append(row)
            
            best_models = pd.DataFrame(filtered_seeds)
    
    # Sort by metric
    best_models = best_models.sort_values(by=metric, ascending=not higher_is_better)
    
    # Create lineages
    lineages = []
    seeds_path = Path(seeds_dir)
    
    for idx, row in best_models.iterrows():
        design_name = row["design_name"]
        model_num = row.get("model", 1)
        score = row[metric]
        
        # Find the seed PDB
        pdb_path = None
        for subdir in ["af2_predictions", "relaxed", ""]:
            candidate = seeds_path / subdir / f"{design_name}_model{model_num}.pdb"
            if candidate.exists():
                pdb_path = str(candidate)
                break
        
        if not pdb_path:
            # Try without model number
            for subdir in ["af2_predictions", "relaxed", ""]:
                candidate = seeds_path / subdir / f"{design_name}.pdb"
                if candidate.exists():
                    pdb_path = str(candidate)
                    break
        
        if not pdb_path:
            logger.warning(f"Could not find PDB for seed {design_name}. Skipping.")
            continue
        
        lineage = Lineage(
            lineage_id=f"L{len(lineages):04d}",
            original_seed_name=design_name,
            original_seed_pdb=pdb_path,
            original_score=float(score),
            current_best_pdb=pdb_path,
            current_best_score=float(score)
        )
        lineages.append(lineage)
    
    logger.info(f"Created {len(lineages)} lineages from {scoring_csv}")
    return lineages


def load_lineages_from_pdb_dir(
    pdb_dir: str,
    csv_path: Optional[str] = None,
) -> List[Lineage]:
    """
    Create lineages directly from a directory of PDB files.
    
    No scoring CSV required — every PDB in the directory becomes a seed.
    
    Metrics are loaded for sorting (best seeds first) from:
      1. An explicit csv_path if provided
      2. Auto-discovered: scored/scoring_summary.csv  (full PyRosetta scoring)
      3. Auto-discovered: af2_prescoring_results.csv  (AF2-only metrics)
    Lineages are sorted by i_pTM descending so the most promising seeds
    are evolved first (faster initial feedback, graceful early termination).
    
    Searches PDBs in: pdb_dir/af2_predictions/, pdb_dir/scoring_pdbs/, pdb_dir/
    """
    pdb_path = Path(pdb_dir)
    
    # Collect PDB files from the directory and common subdirectories
    # Prefer af2_predictions (has pLDDT in B-factors) > scoring_pdbs > root
    search_dirs = []
    for subdir in ["af2_predictions", "scoring_pdbs", ""]:
        candidate = pdb_path / subdir if subdir else pdb_path
        if candidate.is_dir():
            search_dirs.append(candidate)
    
    # Collect unique PDBs (prefer af2_predictions over others)
    seen_stems = set()
    pdb_files = []
    for search_dir in search_dirs:
        for pdb_file in sorted(search_dir.glob("*.pdb")):
            # Extract design name from filename
            # Handles: design_name_model1.pdb, design_name.pdb
            stem = pdb_file.stem
            # Strip _modelN suffix if present
            design_name = stem
            for suffix in ["_model1", "_model2", "_model3", "_model4", "_model5"]:
                if stem.endswith(suffix):
                    design_name = stem[:-len(suffix)]
                    break
            
            if design_name not in seen_stems:
                seen_stems.add(design_name)
                pdb_files.append((design_name, str(pdb_file)))
    
    if not pdb_files:
        logger.warning(f"No PDB files found in {pdb_dir}")
        return []
    
    # Load metrics from CSV for sorting.  Priority:
    #   1. Explicit csv_path argument
    #   2. scored/scoring_summary.csv  (full scoring, if generate_initial produced it)
    #   3. af2_prescoring_results.csv  (AF2-only, always produced by generate_initial)
    metrics_map = {}
    csv_source = None
    
    csv_candidates = []
    if csv_path:
        csv_candidates.append(Path(csv_path))
    csv_candidates.extend([
        pdb_path / "scored" / "scoring_summary.csv",
        pdb_path / "af2_prescoring_results.csv",
    ])
    
    for candidate_csv in csv_candidates:
        if candidate_csv.exists():
            try:
                df = pd.read_csv(candidate_csv)
                if "design_name" in df.columns and "i_pTM" in df.columns:
                    for _, row in df.iterrows():
                        name = row.get("design_name", "")
                        if name:
                            metrics_map[name] = row.to_dict()
                    csv_source = str(candidate_csv)
                    break
            except Exception as e:
                logger.warning(f"Could not read CSV {candidate_csv}: {e}")
    
    # Build (design_name, pdb_file, score) tuples
    scored_pdbs = []
    for design_name, pdb_file in pdb_files:
        score = 0.0
        csv_row = metrics_map.get(design_name, {})
        if "i_pTM" in csv_row:
            try:
                score = float(csv_row["i_pTM"])
            except (ValueError, TypeError):
                pass
        scored_pdbs.append((design_name, pdb_file, score))
    
    # Sort by i_pTM descending so best seeds get processed first
    scored_pdbs.sort(key=lambda x: x[2], reverse=True)
    
    # Create lineages (IDs reflect sorted order: L0000 = best seed)
    lineages = []
    for idx, (design_name, pdb_file, score) in enumerate(scored_pdbs):
        lineage = Lineage(
            lineage_id=f"L{idx:04d}",
            original_seed_name=design_name,
            original_seed_pdb=pdb_file,
            original_score=score,
            current_best_pdb=pdb_file,
            current_best_score=score,
        )
        lineages.append(lineage)
    
    logger.info(f"Created {len(lineages)} lineages from PDB directory: {pdb_dir}")
    if csv_source:
        logger.info(f"  Sorted by i_pTM from: {csv_source}")
        if lineages:
            logger.info(f"  Best seed:  {lineages[0].original_seed_name} (i_pTM={lineages[0].original_score:.4f})")
            logger.info(f"  Worst seed: {lineages[-1].original_seed_name} (i_pTM={lineages[-1].original_score:.4f})")
    else:
        logger.info(f"  No scoring CSV found — lineages are in filesystem order (unsorted)")
    
    return lineages


def save_lineage_checkpoint(lineage: Lineage, checkpoint_dir: Path):
    """Save checkpoint for a single lineage."""
    checkpoint_file = checkpoint_dir / f"{lineage.lineage_id}_checkpoint.json"
    checkpoint_data = {
        "lineage_id": lineage.lineage_id,
        "original_seed_name": lineage.original_seed_name,
        "original_seed_pdb": lineage.original_seed_pdb,
        "original_score": lineage.original_score,
        "current_round": lineage.current_round,
        "current_best_pdb": lineage.current_best_pdb,
        "current_best_score": lineage.current_best_score,
        "history": lineage.history,
        "status": lineage.status
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_lineage_checkpoint(checkpoint_file: Path) -> Optional[Lineage]:
    """Load a lineage from checkpoint file."""
    if not checkpoint_file.exists():
        return None
    
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    return Lineage(
        lineage_id=data["lineage_id"],
        original_seed_name=data["original_seed_name"],
        original_seed_pdb=data["original_seed_pdb"],
        original_score=data["original_score"],
        current_round=data["current_round"],
        current_best_pdb=data["current_best_pdb"],
        current_best_score=data["current_best_score"],
        history=data.get("history", []),
        status=data.get("status", "active")
    )


def evolve_lineage_single_round(
    lineage: Lineage,
    round_idx: int,
    generator: LaProteinaGenerator,
    optimizer: IterativeOptimizer,
    target_pdb: str,
    target_len: int,
    n_samples: int,
    plddt_threshold: float,
    metric: str,
    output_dir: Path,
    target_hotspot_residues: Optional[str] = None,
    target_blacklist_residues: Optional[str] = None
) -> Lineage:
    """
    Evolve a single lineage for one round.
    
    Generates variants, scores them, and keeps ONLY the single best.
    """
    lineage_dir = output_dir / f"lineage_{lineage.lineage_id}"
    round_dir = lineage_dir / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evolving lineage {lineage.lineage_id} (round {round_idx})")
    logger.info(f"  Current best: {Path(lineage.current_best_pdb).name} (score: {lineage.current_best_score:.4f})")
    
    # Phase 1: Generate mask and contig
    masks_dir = round_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    
    try:
        contig, explanation, masqueraded_pdb = optimizer.generate_mask_and_contig(
            pdb_path=lineage.current_best_pdb,
            target_len=target_len,
            plddt_threshold=plddt_threshold,
            save_dir=masks_dir
        )
        logger.info(f"  Mask strategy: {explanation}")
    except Exception as e:
        logger.error(f"  Mask generation failed: {e}")
        lineage.status = "failed"
        return lineage
    
    # Phase 2: Generate variants
    generated_dir = round_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    
    try:
        generator.generate_binder(
            target_pdb=masqueraded_pdb,
            target_residue_range=(1, target_len),
            n_samples=n_samples,
            output_dir=str(generated_dir / f"{lineage.lineage_id}_variants"),
            contig_string=contig,
            design_id=f"{lineage.lineage_id}_R{round_idx}"
        )
    except Exception as e:
        logger.error(f"  Generation failed: {e}")
        lineage.status = "failed"
        return lineage
    
    # Phase 3: Prepare for scoring (split chains)
    import biotite.structure.io as strucio
    import biotite.structure as struc
    
    scoring_input_dir = round_dir / "scoring_input"
    scoring_input_dir.mkdir(exist_ok=True)
    
    generated_pdbs = list(generated_dir.rglob("*.pdb"))
    n_prepared = 0
    
    for pdb in generated_pdbs:
        if "_sample_" not in pdb.name and "job_" not in pdb.name:
            continue
        
        try:
            structure = strucio.load_structure(str(pdb))
            residue_starts = struc.get_residue_starts(structure)
            n_residues = len(residue_starts)
            
            if n_residues <= target_len:
                continue
            
            # Assign chain IDs
            structure.chain_id[:] = "A"
            binder_start_atom_idx = residue_starts[target_len]
            structure.chain_id[binder_start_atom_idx:] = "B"
            
            # Renumber Chain B
            for i in range(target_len, n_residues):
                start = residue_starts[i]
                end = residue_starts[i + 1] if i < n_residues - 1 else len(structure)
                structure.res_id[start:end] = (i - target_len) + 1
            
            strucio.save_structure(str(scoring_input_dir / pdb.name), structure)
            n_prepared += 1
        except Exception as e:
            logger.warning(f"  Failed to prepare {pdb.name}: {e}")
    
    if n_prepared == 0:
        logger.warning(f"  No variants prepared for scoring")
        # Keep current best unchanged
        lineage.history.append({
            "round": round_idx,
            "pdb": lineage.current_best_pdb,
            "score": lineage.current_best_score,
            "n_variants": 0,
            "improved": False
        })
        lineage.current_round = round_idx
        return lineage
    
    logger.info(f"  Prepared {n_prepared} variants for scoring")
    
    # Phase 4: Score variants
    score_output_dir = round_dir / "scored"
    
    score_cmd = [
        sys.executable,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
            "laproteina_scoring", "score.py"
        ),
        "--input_dir", str(scoring_input_dir),
        "--output_dir", str(score_output_dir),
    ]
    
    try:
        subprocess.run(score_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"  Scoring failed: {e}")
        lineage.status = "failed"
        return lineage
    
    # Phase 5: Select single best variant
    scoring_csv = score_output_dir / "scoring_summary.csv"
    if not scoring_csv.exists():
        logger.warning(f"  No scoring results found")
        lineage.history.append({
            "round": round_idx,
            "pdb": lineage.current_best_pdb,
            "score": lineage.current_best_score,
            "n_variants": n_prepared,
            "improved": False
        })
        lineage.current_round = round_idx
        return lineage
    
    scores_df = pd.read_csv(scoring_csv)
    
    if len(scores_df) == 0:
        logger.warning(f"  Empty scoring results")
        lineage.history.append({
            "round": round_idx,
            "pdb": lineage.current_best_pdb,
            "score": lineage.current_best_score,
            "n_variants": n_prepared,
            "improved": False
        })
        lineage.current_round = round_idx
        return lineage
    
    # Filter by hotspot/blacklist if specified
    if target_hotspot_residues or target_blacklist_residues:
        from laproteina.proteinfoundation.utils.pdb_slicer import parse_hotspot_string
        hotspot_ranges = parse_hotspot_string(target_hotspot_residues or "")
        blacklist_ranges = parse_hotspot_string(target_blacklist_residues or "")
        
        if "target_interface_residues" in scores_df.columns:
            filtered_rows = []
            for _, row in scores_df.iterrows():
                interface_str = row.get("target_interface_residues", "")
                if pd.isna(interface_str) or not interface_str:
                    if not hotspot_ranges:
                        filtered_rows.append(row)
                    continue
                
                try:
                    interface_residues = [int(r.strip()) for r in str(interface_str).split(",") if r.strip()]
                except ValueError:
                    continue
                
                # Check blacklist
                if blacklist_ranges:
                    in_blacklist = any(
                        any(start <= r <= end for start, end in blacklist_ranges)
                        for r in interface_residues
                    )
                    if in_blacklist:
                        continue
                
                # Check hotspot
                if hotspot_ranges:
                    in_hotspot = any(
                        any(start <= r <= end for start, end in hotspot_ranges)
                        for r in interface_residues
                    )
                    if not in_hotspot:
                        continue
                
                filtered_rows.append(row)
            
            scores_df = pd.DataFrame(filtered_rows)
    
    if len(scores_df) == 0:
        logger.warning(f"  No variants passed interface filter")
        lineage.history.append({
            "round": round_idx,
            "pdb": lineage.current_best_pdb,
            "score": lineage.current_best_score,
            "n_variants": n_prepared,
            "improved": False
        })
        lineage.current_round = round_idx
        return lineage
    
    # Get best variant by metric
    scores_df[metric] = pd.to_numeric(scores_df[metric], errors='coerce')
    scores_df = scores_df.dropna(subset=[metric])
    
    if len(scores_df) == 0:
        logger.warning(f"  No valid scores for metric {metric}")
        lineage.history.append({
            "round": round_idx,
            "pdb": lineage.current_best_pdb,
            "score": lineage.current_best_score,
            "n_variants": n_prepared,
            "improved": False
        })
        lineage.current_round = round_idx
        return lineage
    
    best_row = scores_df.loc[scores_df[metric].idxmax()]
    best_score = float(best_row[metric])
    best_design_name = best_row["design_name"]
    best_model = best_row.get("model", 1)
    
    # Find the best PDB
    best_pdb = score_output_dir / "af2_predictions" / f"{best_design_name}_model{best_model}.pdb"
    if not best_pdb.exists():
        best_pdb = score_output_dir / "relaxed" / f"{best_design_name}_model{best_model}.pdb"
    
    if not best_pdb.exists():
        logger.warning(f"  Could not find best PDB: {best_design_name}")
        lineage.history.append({
            "round": round_idx,
            "pdb": lineage.current_best_pdb,
            "score": lineage.current_best_score,
            "n_variants": n_prepared,
            "improved": False
        })
        lineage.current_round = round_idx
        return lineage
    
    # Check if improved
    improved = best_score > lineage.current_best_score
    
    # Update lineage
    lineage.history.append({
        "round": round_idx,
        "pdb": str(best_pdb),
        "score": best_score,
        "n_variants": n_prepared,
        "improved": improved
    })
    
    if improved:
        lineage.current_best_pdb = str(best_pdb)
        lineage.current_best_score = best_score
        logger.info(f"  IMPROVED: {best_score:.4f} (was {lineage.history[-1]['score'] if len(lineage.history) > 1 else lineage.original_score:.4f})")
    else:
        # Keep the better of current best and new best
        if best_score > lineage.current_best_score:
            lineage.current_best_pdb = str(best_pdb)
            lineage.current_best_score = best_score
        logger.info(f"  Best this round: {best_score:.4f} (keeping: {lineage.current_best_score:.4f})")
    
    lineage.current_round = round_idx
    return lineage


def main():
    # Set matmul precision for performance on Tensor Cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    parser = argparse.ArgumentParser(
        description="Lineage Evolution for LaProteina V2 Pipeline"
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Path to JSON settings file"
    )
    
    # Individual arguments (can override settings file)
    parser.add_argument("--input_csv", type=str, default=None,
                        help="Path to scoring CSV (optional if --seeds_pdb_dir is provided)")
    parser.add_argument("--seeds_dir", type=str, default=None,
                        help="Directory containing seed PDBs (used with --input_csv)")
    parser.add_argument("--seeds_pdb_dir", type=str, default=None,
                        help="Directory of PDBs to use as seeds directly (no CSV needed)")
    parser.add_argument("--target_pdb", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--autoencoder_path", type=str, default=None)
    parser.add_argument("--laproteina_mode", type=str, default=None,
                        help="'all_atom' (LD4) or 'tip_atom' (LD5)")
    parser.add_argument("--output_base_dir", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--n_samples_per_seed", type=int, default=None)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--plddt_threshold", type=float, default=None)
    parser.add_argument("--target_hotspot_residues", type=str, default=None)
    parser.add_argument("--target_blacklist_residues", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    
    args = parser.parse_args()
    
    # Load settings from file if provided
    settings = {}
    if args.settings:
        settings_path = Path(args.settings)
        if not settings_path.is_absolute():
            repo_root = Path(__file__).parent.parent.parent
            settings_path = repo_root / settings_path
        if settings_path.exists():
            settings = load_settings(settings_path)
            logger.info(f"Loaded settings from: {settings_path}")
    
    # Merge settings with command-line arguments
    def get_arg(key, cli_value, default=None):
        if cli_value is not None:
            return cli_value
        if key in settings:
            return settings[key]
        return default
    
    # Input arguments — two modes:
    #   Mode A (CSV): input_csv + seeds_dir
    #   Mode B (PDB dir): seeds_pdb_dir  (no CSV needed)
    input_csv = get_arg("input_csv", args.input_csv)
    seeds_dir = get_arg("seeds_dir", args.seeds_dir)
    seeds_pdb_dir = get_arg("seeds_pdb_dir", args.seeds_pdb_dir)
    target_pdb = get_arg("target_pdb", args.target_pdb)
    checkpoint_path = get_arg("checkpoint", args.checkpoint)
    autoencoder_path = get_arg("autoencoder_path", args.autoencoder_path)
    laproteina_mode = get_arg("laproteina_mode", args.laproteina_mode, "all_atom")
    
    # Validate: need either (input_csv + seeds_dir) OR seeds_pdb_dir
    has_csv_mode = bool(input_csv and seeds_dir)
    has_pdb_dir_mode = bool(seeds_pdb_dir)
    
    if not has_csv_mode and not has_pdb_dir_mode:
        parser.error(
            "Provide either (input_csv + seeds_dir) for CSV mode, "
            "or seeds_pdb_dir for direct PDB directory mode."
        )
    
    if not target_pdb or not checkpoint_path:
        parser.error("target_pdb and checkpoint are required.")
    
    # Optional arguments with defaults
    output_base_dir = get_arg("output_base_dir", args.output_base_dir, "./evolution_v2_output")
    rounds = get_arg("rounds", args.rounds, 3)
    n_samples_per_seed = get_arg("n_samples_per_seed", args.n_samples_per_seed, 6)
    metric = get_arg("metric", args.metric, "i_pTM")
    plddt_threshold = get_arg("plddt_threshold", args.plddt_threshold, 55.0)
    target_hotspot_residues = get_arg("target_hotspot_residues", args.target_hotspot_residues)
    target_blacklist_residues = get_arg("target_blacklist_residues", args.target_blacklist_residues)
    device = get_arg("device", args.device, "cuda")
    
    # Fix paths
    if input_csv:
        input_csv = fix_path(input_csv)
    if seeds_dir:
        seeds_dir = fix_path(seeds_dir)
    if seeds_pdb_dir:
        seeds_pdb_dir = fix_path(seeds_pdb_dir)
    if autoencoder_path:
        autoencoder_path = fix_path(autoencoder_path)
    target_pdb = fix_path(target_pdb)
    checkpoint_path = fix_path(checkpoint_path)
    
    # Create output directory
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(output_base_dir) / f"evolution_v2_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger.add(output_dir / "evolution.log")
    logger.info("=" * 60)
    logger.info("Starting Lineage Evolution V2")
    logger.info("=" * 60)
    logger.info(f"Rounds: {rounds}, Samples per seed: {n_samples_per_seed}, Metric: {metric}")
    logger.info(f"Output directory: {output_dir}")
    
    # Save settings used
    settings_used = {
        "input_csv": input_csv,
        "seeds_dir": seeds_dir,
        "seeds_pdb_dir": seeds_pdb_dir,
        "target_pdb": target_pdb,
        "checkpoint": checkpoint_path,
        "autoencoder_path": autoencoder_path,
        "laproteina_mode": laproteina_mode,
        "output_base_dir": output_base_dir,
        "rounds": rounds,
        "n_samples_per_seed": n_samples_per_seed,
        "metric": metric,
        "plddt_threshold": plddt_threshold,
        "target_hotspot_residues": target_hotspot_residues,
        "target_blacklist_residues": target_blacklist_residues,
        "device": device
    }
    with open(output_dir / "settings_used.json", 'w') as f:
        json.dump(settings_used, f, indent=2)
    
    # Get target length
    target_len = get_pdb_chain_length(target_pdb, "A")
    logger.info(f"Target length: {target_len}")
    
    # Load or create lineages
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    if args.resume:
        # Try to load existing lineages from checkpoints
        lineages = []
        for checkpoint_file in checkpoints_dir.glob("*_checkpoint.json"):
            lineage = load_lineage_checkpoint(checkpoint_file)
            if lineage:
                lineages.append(lineage)
        
        if lineages:
            logger.info(f"Resumed {len(lineages)} lineages from checkpoints")
        else:
            logger.info("No checkpoints found. Creating new lineages.")
            lineages = None  # Fall through to creation below
    else:
        lineages = None
    
    if lineages is None:
        if has_pdb_dir_mode:
            # Mode B: Load directly from PDB directory (all PDBs become seeds)
            logger.info(f"Loading seeds from PDB directory: {seeds_pdb_dir}")
            lineages = load_lineages_from_pdb_dir(
                pdb_dir=seeds_pdb_dir,
                csv_path=input_csv,  # Optional: for informational metrics only
            )
        elif has_csv_mode:
            # Mode A: Load from scoring CSV (classic mode)
            logger.info(f"Loading seeds from CSV: {input_csv}")
            lineages = load_lineages_from_csv(
                scoring_csv=input_csv,
                seeds_dir=seeds_dir,
                metric=metric,
                target_hotspot_residues=target_hotspot_residues,
                target_blacklist_residues=target_blacklist_residues
            )
    
    logger.info(f"Total lineages: {len(lineages)}")
    
    # Initialize components
    optimizer = IterativeOptimizer(str(output_dir))
    
    logger.info("Loading LaProteina model...")
    ckpt_path = Path(checkpoint_path)
    
    # Resolve autoencoder path
    if autoencoder_path:
        ae_path = Path(autoencoder_path)
    else:
        ae_path = ckpt_path.parent / "AE3_motif.ckpt"
    
    generator = LaProteinaGenerator(
        checkpoint_path=str(ckpt_path),
        autoencoder_path=str(ae_path) if ae_path.exists() else None,
        device=device,
        laproteina_mode=laproteina_mode,
    )
    generator._load_model()
    
    # Evolution loop: lineages first, then rounds (all rounds of a lineage before moving to next)
    completed_count = 0
    for lin_idx, lineage in enumerate(lineages):
        if lineage.status != "active":
            continue
        logger.info("=" * 60)
        logger.info(f"LINEAGE {lineage.lineage_id} ({lin_idx + 1}/{len(lineages)})")
        logger.info("=" * 60)

        for round_idx in range(1, rounds + 1):
            if lineage.current_round >= round_idx:
                continue  # Already completed (resume support)
            try:
                lineage = evolve_lineage_single_round(
                    lineage=lineage,
                    round_idx=round_idx,
                    generator=generator,
                    optimizer=optimizer,
                    target_pdb=target_pdb,
                    target_len=target_len,
                    n_samples=n_samples_per_seed,
                    plddt_threshold=plddt_threshold,
                    metric=metric,
                    output_dir=output_dir,
                    target_hotspot_residues=target_hotspot_residues,
                    target_blacklist_residues=target_blacklist_residues
                )
            except Exception as e:
                logger.error(f"Lineage {lineage.lineage_id} failed: {e}")
                lineage.status = "failed"
                break  # Stop this lineage, move to next
            save_lineage_checkpoint(lineage, checkpoints_dir)

        # Per-lineage summary
        completed_count += 1
        improved = any(h.get("improved") for h in (lineage.history or [])) if lineage.history else False
        logger.info(
            f"Lineage {lineage.lineage_id} complete: {lineage.current_round}/{rounds} rounds, "
            f"score {lineage.current_best_score:.4f}"
            + (f" (improved)" if improved else "")
        )
        logger.info(f"Progress: {completed_count} lineages fully processed")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 60)
    
    # Create final summary
    final_results = []
    for lineage in lineages:
        final_results.append({
            "lineage_id": lineage.lineage_id,
            "original_seed": lineage.original_seed_name,
            "original_score": lineage.original_score,
            "final_score": lineage.current_best_score,
            "improvement": lineage.current_best_score - lineage.original_score,
            "final_pdb": lineage.current_best_pdb,
            "rounds_completed": lineage.current_round,
            "status": lineage.status
        })
    
    summary_df = pd.DataFrame(final_results)
    summary_df = summary_df.sort_values("final_score", ascending=False)
    summary_df.to_csv(output_dir / "final_summary.csv", index=False)
    
    # Log top results
    logger.info("\nTop 10 lineages by final score:")
    for _, row in summary_df.head(10).iterrows():
        logger.info(f"  {row['lineage_id']}: {row['final_score']:.4f} (improvement: {row['improvement']:+.4f})")
    
    # Copy top results to final directory
    final_dir = output_dir / "final_best"
    final_dir.mkdir(exist_ok=True)
    
    for _, row in summary_df.head(20).iterrows():
        if row["final_pdb"] and os.path.exists(row["final_pdb"]):
            shutil.copy(row["final_pdb"], final_dir / f"{row['lineage_id']}_final.pdb")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Final summary: {output_dir / 'final_summary.csv'}")
    logger.info(f"Top PDBs: {final_dir}")


if __name__ == "__main__":
    main()

