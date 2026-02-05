"""
Rosetta Pre-Filter for LaProteina V2 Pipeline

Fast filtering of raw LaProteina outputs using Rosetta scoring
before expensive AlphaFold2 reprediction.

Runs score_interface() on raw generated structures (without relaxation)
to quickly eliminate physically unrealistic designs.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

import biotite.structure as struc
import biotite.structure.io as strucio


@dataclass
class PrefilterResult:
    """Result of Rosetta pre-filtering."""
    
    passing_pdbs: List[str]           # PDBs that passed all filters
    failed_pdbs: List[str]            # PDBs that failed filters
    scores_df: pd.DataFrame           # All scores for all PDBs
    filter_summary: Dict[str, int]    # Count of failures per filter


class RosettaPreFilter:
    """
    Fast pre-filtering of LaProteina outputs using Rosetta scoring.
    
    Scores raw generated structures (without relaxation) to quickly
    eliminate designs with poor physical properties before running
    expensive AF2 reprediction.
    
    Example:
        prefilter = RosettaPreFilter(
            filter_settings_path="settings_filters/laProteina_filters.json"
        )
        
        result = prefilter.prefilter_batch(
            pdb_paths=[...],
            target_len=291,
            output_dir="./prefilter_results"
        )
    """
    
    # Mapping from filter settings keys to score_interface output keys
    FILTER_TO_SCORE_MAPPING = {
        "dG": "interface_dG",
        "ShapeComplementarity": "interface_sc",
        "dSASA": "interface_dSASA",
        "n_InterfaceResidues": "interface_nres",
        "Surface_Hydrophobicity": "surface_hydrophobicity",
        "PackStat": "interface_packstat",
        "Interface_Hydrophobicity": "interface_hydrophobicity",
        "n_InterfaceHbonds": "interface_interface_hbonds",
        "n_InterfaceUnsatHbonds": "interface_delta_unsat_hbonds",
        "Binder_Energy_Score": "binder_score",
    }
    
    def __init__(
        self,
        filter_settings_path: Optional[str] = None,
        filter_settings: Optional[Dict] = None,
        init_pyrosetta: bool = True,
        do_relax: bool = False,
        hotspot_residues: Optional[str] = None
    ):
        """
        Initialize the Rosetta pre-filter.
        
        Args:
            filter_settings_path: Path to filter settings JSON file
            filter_settings: Direct filter settings dict (alternative to path)
            init_pyrosetta: Whether to initialize PyRosetta on construction
            do_relax: Whether to run FastRelax before scoring (improves metrics but slower)
            hotspot_residues: Hotspot regions string (e.g., "300-350,400-450").
                             If provided, binders must have interface residues in these regions.
        """
        self.filters = {}
        self.do_relax = do_relax
        self.hotspot_residues = hotspot_residues
        self.hotspot_ranges = []
        
        # Parse hotspot ranges if provided
        if hotspot_residues:
            from laproteina.proteinfoundation.utils.pdb_slicer import parse_hotspot_string
            self.hotspot_ranges = parse_hotspot_string(hotspot_residues)
            if not self.hotspot_ranges:
                logger.warning(f"Could not parse hotspot_residues: {hotspot_residues}")
        
        if filter_settings_path:
            self.filters = self._load_filters(filter_settings_path)
        elif filter_settings:
            self.filters = self._parse_filters(filter_settings)
        
        if init_pyrosetta:
            self._init_pyrosetta()
        
        relax_status = "with relaxation" if do_relax else "without relaxation"
        hotspot_status = f", hotspot filter: {hotspot_residues}" if hotspot_residues else ""
        logger.info(f"RosettaPreFilter initialized with {len(self.filters)} active filters ({relax_status}{hotspot_status})")
    
    def _init_pyrosetta(self):
        """Initialize PyRosetta with appropriate settings."""
        import pyrosetta as pr
        
        # Find DAlphaBall path
        bindcraft_root = Path(__file__).parents[1]
        dalphaball_path = bindcraft_root / "functions" / "DAlphaBall.gcc"
        
        cmd = (
            f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
            f'-holes:dalphaball {dalphaball_path} '
            f'-corrections::beta_nov16 true -relax:default_repeats 1'
        )
        
        try:
            pr.init(cmd)
            logger.info("PyRosetta initialized for pre-filtering")
        except Exception:
            pass  # Already initialized
    
    def _load_filters(self, filter_path: str) -> Dict[str, Dict]:
        """
        Load and parse filter settings from JSON file.
        
        Extracts Rosetta-applicable filters (dG, SC, dSASA, etc.)
        that can be computed without AF2 reprediction.
        """
        with open(filter_path, 'r') as f:
            all_filters = json.load(f)
        
        return self._parse_filters(all_filters)
    
    def _parse_filters(self, all_filters: Dict) -> Dict[str, Dict]:
        """
        Extract Rosetta-applicable filters from settings.
        
        Only keeps filters that:
        1. Have non-null thresholds
        2. Can be computed by score_interface() (no AF2 metrics)
        """
        active_filters = {}
        
        # Look for single-model filters (1_*) or Average_* filters
        for filter_key, filter_config in all_filters.items():
            if not isinstance(filter_config, dict):
                continue
            if "threshold" not in filter_config:
                continue
            if filter_config.get("threshold") is None:
                continue
            
            # Extract the metric name (remove prefix like "1_", "Average_")
            metric_name = filter_key
            for prefix in ["1_", "2_", "Average_"]:
                if metric_name.startswith(prefix):
                    metric_name = metric_name[len(prefix):]
                    break
            
            # Check if this is a Rosetta-computable metric
            if metric_name in self.FILTER_TO_SCORE_MAPPING:
                # Only keep one filter per metric (prefer 1_ over Average_)
                if metric_name not in active_filters or filter_key.startswith("1_"):
                    active_filters[metric_name] = {
                        "threshold": filter_config["threshold"],
                        "higher": filter_config.get("higher", True),
                        "score_key": self.FILTER_TO_SCORE_MAPPING[metric_name],
                        "original_key": filter_key
                    }
        
        logger.info(f"Active Rosetta filters: {list(active_filters.keys())}")
        return active_filters
    
    def prefilter_batch(
        self,
        pdb_paths: List[str],
        target_len: int,
        output_dir: Optional[str] = None,
        save_split_pdbs: bool = False,
        do_relax: Optional[bool] = None
    ) -> PrefilterResult:
        """
        Pre-filter a batch of LaProteina-generated PDBs.
        
        Args:
            pdb_paths: List of paths to generated PDB files
            target_len: Length of the target protein (for chain splitting)
            output_dir: Optional directory to save results
            save_split_pdbs: Whether to save the split A/B chain PDBs
            do_relax: Override instance-level relax setting (None uses instance default)
            
        Returns:
            PrefilterResult with passing PDBs, scores, and statistics
        """
        from functions.pyrosetta_utils import score_interface, pr_relax
        
        # Use instance default if not specified
        relax_enabled = do_relax if do_relax is not None else self.do_relax
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Prefilter output directory: {output_path}")
            split_dir = output_path / "split_pdbs"
            if save_split_pdbs:
                split_dir.mkdir(exist_ok=True)
            if relax_enabled:
                relaxed_dir = output_path / "relaxed_pdbs"
                relaxed_dir.mkdir(exist_ok=True)
            else:
                relaxed_dir = None
        else:
            output_path = None
            split_dir = None
            relaxed_dir = None
        
        all_scores = []
        passing_pdbs = []
        failed_pdbs = []
        filter_failures = {metric: 0 for metric in self.filters}
        
        relax_status = "with relaxation" if relax_enabled else "without relaxation"
        logger.info(f"Pre-filtering {len(pdb_paths)} PDBs with {len(self.filters)} filters ({relax_status})")
        
        for pdb_path in pdb_paths:
            pdb_name = Path(pdb_path).stem
            
            try:
                # Split chains
                split_pdb_path = self._split_chains(
                    pdb_path, 
                    target_len, 
                    split_dir if save_split_pdbs else None
                )
                
                if split_pdb_path is None:
                    logger.warning(f"Skipping {pdb_name}: Chain splitting failed")
                    failed_pdbs.append(pdb_path)
                    continue
                
                # Optional relaxation
                pdb_to_score = split_pdb_path
                if relax_enabled:
                    try:
                        if relaxed_dir:
                            relaxed_pdb_path = str(relaxed_dir / f"{pdb_name}_relaxed.pdb")
                        else:
                            relaxed_pdb_path = split_pdb_path.replace(".pdb", "_relaxed.pdb")
                        
                        pr_relax(split_pdb_path, relaxed_pdb_path)
                        pdb_to_score = relaxed_pdb_path
                        logger.debug(f"Relaxed {pdb_name}")
                    except Exception as e:
                        logger.warning(f"Relaxation failed for {pdb_name}: {e}, using unrelaxed")
                        pdb_to_score = split_pdb_path
                
                # Score interface
                try:
                    interface_scores, interface_AA, interface_residues = score_interface(
                        pdb_to_score, 
                        binder_chain="B"
                    )
                except Exception as e:
                    logger.warning(f"Scoring failed for {pdb_name}: {e}")
                    failed_pdbs.append(pdb_path)
                    continue
                
                # Add metadata
                scores_row = {
                    "pdb_path": pdb_path,
                    "pdb_name": pdb_name,
                    **interface_scores
                }
                all_scores.append(scores_row)
                
                # Check filters
                passed_all = True
                for metric_name, filter_config in self.filters.items():
                    score_key = filter_config["score_key"]
                    threshold = filter_config["threshold"]
                    higher_is_better = filter_config["higher"]
                    
                    if score_key not in interface_scores:
                        continue
                    
                    value = interface_scores[score_key]
                    if value is None:
                        continue
                    
                    if higher_is_better:
                        passed = value >= threshold
                    else:
                        passed = value <= threshold
                    
                    if not passed:
                        passed_all = False
                        filter_failures[metric_name] += 1
                
                # Check hotspot interface filter
                if passed_all and self.hotspot_ranges:
                    target_interface_str = interface_scores.get("target_interface_residues", "")
                    if not target_interface_str:
                        # No interface residues found - fail hotspot filter
                        passed_all = False
                        if "hotspot_interface" not in filter_failures:
                            filter_failures["hotspot_interface"] = 0
                        filter_failures["hotspot_interface"] += 1
                    else:
                        # Parse target interface residues
                        try:
                            target_interface_residues = [
                                int(r.strip()) for r in str(target_interface_str).split(",") if r.strip()
                            ]
                        except ValueError:
                            target_interface_residues = []
                        
                        # Check if any interface residue is in the hotspot region
                        in_hotspot = any(
                            any(start <= r <= end for start, end in self.hotspot_ranges)
                            for r in target_interface_residues
                        )
                        
                        if not in_hotspot:
                            passed_all = False
                            if "hotspot_interface" not in filter_failures:
                                filter_failures["hotspot_interface"] = 0
                            filter_failures["hotspot_interface"] += 1
                
                if passed_all:
                    passing_pdbs.append(pdb_path)
                else:
                    failed_pdbs.append(pdb_path)
                
                # Cleanup temporary files if not saving
                if not save_split_pdbs and split_pdb_path != pdb_path:
                    if os.path.exists(split_pdb_path):
                        os.remove(split_pdb_path)
                
                # Cleanup temporary relaxed PDB if not saving in dedicated dir
                if relax_enabled and relaxed_dir is None and pdb_to_score != split_pdb_path:
                    if os.path.exists(pdb_to_score):
                        os.remove(pdb_to_score)
                        
            except Exception as e:
                logger.error(f"Error processing {pdb_name}: {e}")
                failed_pdbs.append(pdb_path)
                continue
        
        # Create DataFrame
        scores_df = pd.DataFrame(all_scores)
        
        # Save results if output_dir specified
        if output_path:
            scores_csv = output_path / "prefilter_scores.csv"
            scores_df.to_csv(scores_csv, index=False)
            logger.info(f"Saved prefilter scores to: {scores_csv}")
            
            # Save passing PDB list
            passing_txt = output_path / "passing_pdbs.txt"
            with open(passing_txt, 'w') as f:
                for pdb in passing_pdbs:
                    f.write(f"{pdb}\n")
            logger.info(f"Saved passing PDBs list to: {passing_txt}")
            
            # Save filter summary
            summary = {
                "total_pdbs": len(pdb_paths),
                "passing": len(passing_pdbs),
                "failed": len(failed_pdbs),
                "filter_failures": filter_failures
            }
            summary_json = output_path / "prefilter_summary.json"
            with open(summary_json, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved prefilter summary to: {summary_json}")
        
        logger.info(f"Pre-filter complete: {len(passing_pdbs)}/{len(pdb_paths)} passed")
        for metric, count in filter_failures.items():
            if count > 0:
                logger.info(f"  - {metric}: {count} failures")
        
        return PrefilterResult(
            passing_pdbs=passing_pdbs,
            failed_pdbs=failed_pdbs,
            scores_df=scores_df,
            filter_summary=filter_failures
        )
    
    def _split_chains(
        self,
        pdb_path: str,
        target_len: int,
        output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """
        Split a single-chain LaProteina PDB into chains A (target+linker) and B (binder).
        
        Args:
            pdb_path: Path to the generated PDB
            target_len: Number of residues in target+linker
            output_dir: Optional directory to save split PDB
            
        Returns:
            Path to the split PDB (or None if failed)
        """
        try:
            structure = strucio.load_structure(pdb_path)
            
            # Get residue information
            residue_starts = struc.get_residue_starts(structure)
            n_residues = len(residue_starts)
            
            # Check if we have enough residues
            if n_residues <= target_len:
                logger.warning(f"Too few residues ({n_residues}) for split at {target_len}")
                return None
            
            # Assign chain IDs
            # Target+Linker (residues 0 to target_len-1) -> Chain A
            # Binder (residues target_len onwards) -> Chain B
            structure.chain_id[:] = "A"  # Default all to A
            
            binder_start_atom_idx = residue_starts[target_len]
            structure.chain_id[binder_start_atom_idx:] = "B"
            
            # Renumber Chain B residues starting from 1
            for i in range(target_len, n_residues):
                start = residue_starts[i]
                if i < n_residues - 1:
                    end = residue_starts[i + 1]
                else:
                    end = len(structure)
                
                structure.res_id[start:end] = (i - target_len) + 1
            
            # Save split structure
            if output_dir:
                output_path = output_dir / f"{Path(pdb_path).stem}_split.pdb"
            else:
                import tempfile
                output_path = Path(tempfile.mktemp(suffix="_split.pdb"))
            
            strucio.save_structure(str(output_path), structure)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Chain splitting failed for {pdb_path}: {e}")
            return None
    
    def score_single(
        self,
        pdb_path: str,
        target_len: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Score a single PDB and check if it passes filters.
        
        Args:
            pdb_path: Path to the generated PDB
            target_len: Length of target+linker for chain splitting
            
        Returns:
            Tuple of (passed_filters, scores_dict)
        """
        from functions.pyrosetta_utils import score_interface
        
        split_pdb = self._split_chains(pdb_path, target_len)
        if split_pdb is None:
            return False, {}
        
        try:
            interface_scores, _, _ = score_interface(split_pdb, binder_chain="B")
            
            # Check filters
            passed_all = True
            for metric_name, filter_config in self.filters.items():
                score_key = filter_config["score_key"]
                threshold = filter_config["threshold"]
                higher_is_better = filter_config["higher"]
                
                if score_key not in interface_scores:
                    continue
                
                value = interface_scores[score_key]
                if value is None:
                    continue
                
                if higher_is_better:
                    passed = value >= threshold
                else:
                    passed = value <= threshold
                
                if not passed:
                    passed_all = False
                    break
            
            return passed_all, interface_scores
            
        finally:
            # Cleanup
            if os.path.exists(split_pdb):
                os.remove(split_pdb)


def create_default_prefilter() -> RosettaPreFilter:
    """
    Create a RosettaPreFilter with sensible default thresholds.
    
    Uses conservative defaults for quick filtering without a settings file.
    """
    default_filters = {
        "dG": {"threshold": 0, "higher": False},           # Binding energy < 0
        "ShapeComplementarity": {"threshold": 0.5, "higher": True},  # SC > 0.5
        "dSASA": {"threshold": 400, "higher": True},       # Interface area > 400
        "n_InterfaceResidues": {"threshold": 5, "higher": True},  # At least 5 interface residues
    }
    
    return RosettaPreFilter(filter_settings=default_filters)

