
import os
import pandas as pd
import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from loguru import logger
import hashlib

# Import utilities for interface filtering
from laproteina.proteinfoundation.utils.pdb_slicer import parse_hotspot_string

class IterativeOptimizer:
    """
    Handles the logic for the "Binder Evolution" workflow.
    Selects seeds from scoring results, generates masks based on pLDDT,
    and constructs contigs for the next round of generation.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Track ID mappings: {original_design_name: short_id}
        self.id_mapping: Dict[str, str] = {}
        # Track ID counter for sequential IDs
        self.id_counter = 0
        
    def select_seeds(
        self, 
        scoring_csv: str, 
        metric: str = "i_pTM", 
        top_n: int = 5, 
        higher_is_better: bool = True,
        filter_status: bool = True,
        target_hotspot_residues: Optional[str] = None,
        target_blacklist_residues: Optional[str] = None
    ) -> List[Dict]:
        """
        Select top N candidates from the scoring summary CSV.
        
        Args:
            scoring_csv: Path to CSV file with scoring results
            metric: Metric to rank seeds by
            top_n: Number of top seeds to select
            higher_is_better: Whether higher metric values are better
            filter_status: Whether to filter out failed designs
            target_hotspot_residues: Optional hotspot regions string (e.g., "300-350,400-450").
                                    Seeds without interface residues in these regions will be filtered out.
                                    Requires 'target_interface_residues' column in CSV (from PyRosetta scoring).
            target_blacklist_residues: Optional blacklist regions string (e.g., "1-100,200-300").
                                      Seeds with interface residues in these regions will be filtered out.
                                      Requires 'target_interface_residues' column in CSV (from PyRosetta scoring).
        """
        df = pd.read_csv(scoring_csv)
        
        # Filter for successful designs
        if filter_status and "status" in df.columns:
            # We accept 'scored' or 'failed_filter' (because maybe we want to fix failed ones!)
            # But usually we want to evolve the best ones. 
            # User said "take samples that have Hbonds... and try to regenerate". 
            # So we might take even those that didn't pass "overall_passed" if they are promising in specific metrics.
            # For now, let's filter out total technical failures.
            df = df[df["status"] != "failed"]
            
        # Ensure numeric
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
        df = df.dropna(subset=[metric])
        
        # Pick unique designs (best model for each design)
        # Assuming we want distinct designs to evolve
        best_models = df.groupby("design_name").first().reset_index()
        
        # Filter by interface residues in hotspot/blacklist regions if specified
        filtered_seeds = []
        filtered_out = []
        
        # Check if we need interface filtering (hotspot or blacklist)
        needs_interface_filtering = (
            (target_hotspot_residues and target_hotspot_residues.strip()) or
            (target_blacklist_residues and target_blacklist_residues.strip())
        )
        
        if needs_interface_filtering:
            # Parse hotspot and blacklist ranges
            hotspot_ranges = []
            blacklist_ranges = []
            
            if target_hotspot_residues and target_hotspot_residues.strip():
                hotspot_ranges = parse_hotspot_string(target_hotspot_residues)
                if not hotspot_ranges:
                    logger.warning(f"Could not parse hotspot regions from '{target_hotspot_residues}'. Skipping hotspot filter.")
            
            if target_blacklist_residues and target_blacklist_residues.strip():
                blacklist_ranges = parse_hotspot_string(target_blacklist_residues)
                if not blacklist_ranges:
                    logger.warning(f"Could not parse blacklist regions from '{target_blacklist_residues}'. Skipping blacklist filter.")
            
            if "target_interface_residues" not in best_models.columns:
                logger.warning("target_interface_residues column not found in CSV. Interface filtering requires re-scoring with updated code.")
                # Fall back to no filtering
                best_models = best_models.sort_values(by=metric, ascending=not higher_is_better)
                seeds = best_models.head(top_n).to_dict('records')
            else:
                # Iterate over ALL seeds and filter by interface residues
                for _, seed_row in best_models.iterrows():
                    seed = seed_row.to_dict()
                    design_name = seed["design_name"]
                    
                    # Get target interface residues from CSV (comma-separated string)
                    target_interface_residues_str = seed.get("target_interface_residues", "")
                    
                    if not target_interface_residues_str or pd.isna(target_interface_residues_str):
                        # If hotspot filtering is required, filter out seeds without interface data
                        if hotspot_ranges:
                            logger.info(f"Seed {design_name} has no target interface residues recorded. Filtering out.")
                            filtered_out.append(design_name)
                            continue
                        # If only blacklist filtering, we can't check, so skip blacklist check for this seed
                        target_interface_residues = []
                    else:
                        # Parse comma-separated residue IDs
                        try:
                            target_interface_residues = [int(r.strip()) for r in str(target_interface_residues_str).split(",") if r.strip()]
                        except ValueError:
                            logger.warning(f"Could not parse target_interface_residues for {design_name}: {target_interface_residues_str}")
                            if hotspot_ranges:
                                filtered_out.append(design_name)
                                continue
                            target_interface_residues = []
                    
                    # Check blacklist first (if any interface residue is in blacklist, exclude seed)
                    if blacklist_ranges and target_interface_residues:
                        has_interface_in_blacklist = False
                        for res_id in target_interface_residues:
                            for start, end in blacklist_ranges:
                                if start <= res_id <= end:
                                    has_interface_in_blacklist = True
                                    break
                            if has_interface_in_blacklist:
                                break
                        
                        if has_interface_in_blacklist:
                            logger.info(f"Seed {design_name} has interface residues in blacklisted regions {blacklist_ranges}. Filtering out.")
                            filtered_out.append(design_name)
                            continue
                    
                    # Check hotspot (if hotspot filtering is enabled, seed must have interface in hotspot)
                    if hotspot_ranges:
                        if not target_interface_residues:
                            logger.info(f"Seed {design_name} has no interface residues. Filtering out.")
                            filtered_out.append(design_name)
                            continue
                        
                        has_interface_in_hotspot = False
                        for res_id in target_interface_residues:
                            for start, end in hotspot_ranges:
                                if start <= res_id <= end:
                                    has_interface_in_hotspot = True
                                    break
                            if has_interface_in_hotspot:
                                break
                        
                        if has_interface_in_hotspot:
                            filtered_seeds.append(seed)
                        else:
                            logger.info(f"Seed {design_name} has interface residues {target_interface_residues[:5]}{'...' if len(target_interface_residues) > 5 else ''} but none in hotspot regions {hotspot_ranges}. Filtering out.")
                            filtered_out.append(design_name)
                    else:
                        # No hotspot filtering, but blacklist check passed (or no blacklist), so include seed
                        filtered_seeds.append(seed)
                
                if filtered_out:
                    filter_reasons = []
                    if hotspot_ranges:
                        filter_reasons.append("hotspot regions")
                    if blacklist_ranges:
                        filter_reasons.append("blacklist regions")
                    logger.info(f"Filtered out {len(filtered_out)} seeds based on {', '.join(filter_reasons)}: {filtered_out[:5]}{'...' if len(filtered_out) > 5 else ''}")
                
                # Convert filtered seeds back to DataFrame for sorting
                if filtered_seeds:
                    filtered_df = pd.DataFrame(filtered_seeds)
                    filtered_df[metric] = pd.to_numeric(filtered_df[metric], errors='coerce')
                    filtered_df = filtered_df.dropna(subset=[metric])
                    filtered_df = filtered_df.sort_values(by=metric, ascending=not higher_is_better)
                    seeds = filtered_df.head(top_n).to_dict('records')
                else:
                    logger.warning("No seeds passed interface filter. Returning empty list.")
                    seeds = []
        else:
            # No filtering, just sort and take top N
            best_models = best_models.sort_values(by=metric, ascending=not higher_is_better)
            seeds = best_models.head(top_n).to_dict('records')
        
        logger.info(f"Selected {len(seeds)} seeds based on {metric}")
        return seeds

    def generate_mask_and_contig(
        self, 
        pdb_path: str, 
        target_len: int,
        plddt_threshold: float = 55.0,
        gap_flexibility: int = 0,
        save_dir: Path = None
    ) -> Tuple[str, str, str]:
        """
        Analyzes a PDB (Target+Binder).
        If separate Chain B exists, stitches it to Chain A (renumbering) to create a Monomer Masquerade PDB.
        Constructs a contig string for partial redesign.
        
        Args:
            save_dir: Optional directory to save the masqueraded PDB. 
                      If None, saves alongside input PDB.
        
        Returns:
            Tuple[contig_string, explanation, masqueraded_pdb_path]
        """
        structure = strucio.load_structure(pdb_path, extra_fields=["b_factor"])
        
        # Check for Chain B
        chains = np.unique(structure.chain_id)
        has_chain_b = "B" in chains
        
        masqueraded_pdb_path = pdb_path
        
        if has_chain_b:
            # === Monomer Masquerade Logic ===
            logger.info(f"Detected Chain B in {Path(pdb_path).name}. Performing Monomer Masquerade stitching...")
            
            chain_a_mask = structure.chain_id == "A"
            chain_b_mask = structure.chain_id == "B"
            
            chain_a = structure[chain_a_mask]
            chain_b = structure[chain_b_mask]
            
            # Renumber Chain B to follow Chain A
            # We assume Chain A is 1..target_len (or at least ends at target_len)
            # Safe bet: start B at target_len + 1
            # Or max(chain_a) + 1
            last_a_res = np.max(chain_a.res_id) if len(chain_a) > 0 else target_len
            start_b_res = last_a_res + 1
            
            # Shift B residues
            # Check if B starts at 1
            min_b = np.min(chain_b.res_id)
            offset = start_b_res - min_b
            chain_b.res_id += offset
            
            # Rename B to A
            chain_b.chain_id[:] = "A"
            
            # Merge
            structure = chain_a + chain_b
            
            # NOTE: We save the *filtered* structure later, so we don't need to save here immediately
            # unless we want the intermediate full stitched one. 
            # But the logic below creates a 'filtered_structure' from 'structure'.
            # So 'structure' now holds the stitched version.

        # Now analyze the (single chain) structure
        # Filter for CA to get residues and B-factors (pLDDT)
        ca = structure[structure.atom_name == "CA"]
        
        # Identify Target vs Binder/Linker
        # Target: Residue IDs 1 to target_len
        # Note: If masqueraded, binder starts > target_len
        target_mask = (ca.res_id <= target_len)
        binder_mask = (ca.res_id > target_len)
        
        if not np.any(binder_mask):
             # Just in case, try searching for residues > target_len even if no Chain B was found originally
             # (e.g. if input was already stitched)
             if not has_chain_b:
                  raise ValueError(f"No binder residues found (residues > {target_len}) in {pdb_path}")
        
        binder_residues = ca[binder_mask]
        
        # Analyze pLDDT of binder residues
        # Safe access to B-factor
        if "b_factor" not in binder_residues.get_annotation_categories():
            logger.warning(f"No B-factor (pLDDT) found in {pdb_path}. Assuming pLDDT=0 (Redesign all).")
            # Create dummy low pLDDT
            plddts = np.zeros(len(binder_residues))
        else:
            plddts = binder_residues.b_factor
            
        res_ids = binder_residues.res_id
        
        # Dynamic pLDDT threshold: Ensure we never keep more than 80% of the binder
        # Start with the base threshold
        current_threshold = plddt_threshold
        max_binder_fraction = 0.80  # Never keep more than 80% of binder
        min_keep_residues = 5  # Always keep at least 5 residues
        
        # Calculate how many residues pass the threshold
        good_mask = (plddts >= current_threshold)
        n_good = np.sum(good_mask)
        total_binder = len(plddts)
        
        # If more than 80% pass, increase threshold until we're at or below 80%
        if n_good > max_binder_fraction * total_binder:
            # Sort pLDDT values in descending order
            sorted_indices = np.argsort(plddts)[::-1]
            max_keep = int(max_binder_fraction * total_binder)
            
            # Find the threshold that gives us exactly max_keep residues (or closest)
            # We want the pLDDT value at position max_keep-1 in sorted order
            if max_keep > 0:
                threshold_value = plddts[sorted_indices[max_keep - 1]]
                # Use a slightly higher threshold to ensure we're at or below 80%
                current_threshold = threshold_value + 0.1
                logger.info(f"Dynamic threshold adjustment: {n_good}/{total_binder} ({100*n_good/total_binder:.1f}%) residues passed base threshold {plddt_threshold:.1f}. "
                          f"Increasing threshold to {current_threshold:.1f} to keep ≤{max_keep} residues ({100*max_binder_fraction:.0f}%).")
            else:
                current_threshold = np.max(plddts) + 1.0  # Keep none if binder is too short
            
            # Recalculate mask with new threshold
            good_mask = (plddts >= current_threshold)
            n_good = np.sum(good_mask)
        
        # Ensure we keep at least min_keep_residues (but still respect 80% max)
        if n_good < min_keep_residues and total_binder >= min_keep_residues:
            # Keep the top min_keep_residues by pLDDT
            sorted_indices = np.argsort(plddts)[::-1]
            force_keep_indices = sorted_indices[:min_keep_residues]
            good_mask = np.zeros(len(plddts), dtype=bool)
            good_mask[force_keep_indices] = True
            n_good = min_keep_residues
            logger.info(f"Only {np.sum(plddts >= plddt_threshold)} residues > base threshold {plddt_threshold:.1f} pLDDT. "
                      f"Forcing keep of top {min_keep_residues} binder residues by pLDDT.")
        elif n_good < min_keep_residues:
            # Binder is shorter than min_keep_residues, keep all
            good_mask[:] = True
            n_good = len(plddts)
            logger.info(f"Binder has only {len(plddts)} residues. Keeping all residues.")
        
        # Start contig with Target (Fixed)
        contig_parts = [f"A1-{target_len}"] 
        
        current_segment_type = None # 'fixed' or 'gap'
        current_res_start = None
        current_res_end = None
        
        # Track which binder residues are kept for the PDB file
        binder_kept_indices = []
        
        # Iterate through binder residues to build segments
        for i, (rid, is_good) in enumerate(zip(res_ids, good_mask)):
            seg_type = 'fixed' if is_good else 'gap'
            
            if is_good:
                binder_kept_indices.append(i)
            
            if current_segment_type is None:
                current_segment_type = seg_type
                current_res_start = rid
                current_res_end = rid
            elif seg_type == current_segment_type:
                # Extend current segment
                current_res_end = rid
            else:
                # Switch segment -> Flush previous
                self._flush_segment(contig_parts, current_segment_type, current_res_start, current_res_end, gap_flexibility)
                
                # Start new
                current_segment_type = seg_type
                current_res_start = rid
                current_res_end = rid
                
        # Flush last segment
        if current_segment_type is not None:
             self._flush_segment(contig_parts, current_segment_type, current_res_start, current_res_end, gap_flexibility)
             
        contig_string = "/".join(contig_parts)
        
        # Save "Masqueraded" PDB containing ONLY the FIXED parts (Target + Fixed Binder)
        
        # Filter structure to keep Target + Kept Binder
        # Re-evaluate masks on the full 'structure' object
        final_keep_mask = np.zeros(len(structure), dtype=bool)
        
        res_ids_all = structure.res_id
        
        # Target
        final_keep_mask[res_ids_all <= target_len] = True
        
        # Binder: Kept binder residues
        kept_binder_res_ids = res_ids[binder_kept_indices] 
        final_keep_mask[np.isin(res_ids_all, kept_binder_res_ids)] = True
        
        filtered_structure = structure[final_keep_mask]
        
        p = Path(pdb_path)
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            masqueraded_pdb_path = str(save_dir / f"{p.stem}_masqueraded.pdb")
        else:
            masqueraded_pdb_path = str(p.parent / f"{p.stem}_masqueraded.pdb")
            
        strucio.save_structure(masqueraded_pdb_path, filtered_structure)
        logger.info(f"Saved masqueraded (stripped) PDB to {masqueraded_pdb_path}")
        
        # Construct summary
        fixed_count = np.sum(good_mask)
        total_binder = len(plddts)
        kept_fraction = fixed_count / total_binder if total_binder > 0 else 0
        explanation = f"Binder ({total_binder} res): Kept {fixed_count} res ({100*kept_fraction:.1f}%) at threshold {current_threshold:.1f}. Contig: {contig_string}"
        
        return contig_string, explanation, masqueraded_pdb_path

    def _flush_segment(self, contig_parts, seg_type, start, end, flex):
        if seg_type == 'fixed':
            # Add fixed segment: A{start}-{end}
            contig_parts.append(f"A{start}-{end}")
        else:
            # Add gap with VARIABLE length
            # Allow segment to shrink by 20% or grow by 40% from original length
            # This allows for longer binders while still allowing some flexibility
            length = end - start + 1
            
            # Calculate range: 80% to 140% of original length
            min_len = max(1, int(length * 0.85))  # At least 1 residue, round down
            max_len = int(length * 1.4)  # Allow up to 40% longer binders
            
            contig_parts.append(f"{min_len}-{max_len}")

    def generate_short_id(self, design_name: str, round_idx: int = 1) -> str:
        """
        Generate a short unique ID for a design.
        Uses hash-based approach for consistency across runs.
        
        Args:
            design_name: Original design name
            round_idx: Current round number (for prefix)
            
        Returns:
            Short ID like 'R1A3F2B1' (6-8 chars)
        """
        if design_name in self.id_mapping:
            return self.id_mapping[design_name]
        
        # Generate hash-based ID for consistency
        hash_obj = hashlib.md5(design_name.encode())
        hash_hex = hash_obj.hexdigest()[:6].upper()
        
        # Format: R{round}{hash} -> e.g., R1A3F2B1
        short_id = f"R{round_idx}{hash_hex}"
        
        self.id_mapping[design_name] = short_id
        return short_id
    
    def get_or_create_id(self, design_name: str, round_idx: int = 1) -> str:
        """Get existing ID or create new one."""
        return self.generate_short_id(design_name, round_idx)

    def prepare_round_directory(self, round_idx: int, date_str: str = None) -> Path:
        """Creates the directory for the current evolution round."""
        # self.output_dir is already set to the run specific folder (e.g. evolution_20251227_1500)
        # So we just append round_X
        round_dir = self.output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        return round_dir
