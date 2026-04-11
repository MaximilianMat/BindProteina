"""
Target Preparer for LaProteina V2 Pipeline

Prepares target proteins with linkers for binder generation.

Process:
1. Extract a LARGE focused region around the hotspot (max context for robust alignment)
2. Fold the linker onto the focused region using AF2
3. Align the AF2 result back to the full target coordinates
4. If alignment RMSD > threshold, fall back to direct coordinate positioning
5. Append positioned linker to the FULL original target
6. LaProteina sees: full target + floating linker in correct 3D position
7. Post-processing: Remove linker from output for scoring
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Sequence, Dict
from loguru import logger

import biotite.structure as struc
import biotite.structure.io as strucio

from laproteina.proteinfoundation.linker_generator import LinkerConfig


# RMSD threshold (Å). If alignment RMSD exceeds this, switch to direct
# coordinate positioning instead of alignment-based placement.
DEFAULT_RMSD_THRESHOLD = 3.0


@dataclass
class PreparedTarget:
    """Result of target preparation with linker."""
    
    pdb_path: str                    # Path to the prepared PDB file
    contig_string: str               # Contig string for LaProteina
    target_length: int               # Length of original target
    linker_length: int               # Length of linker added
    total_fixed_length: int          # target_length + linker_length
    linker_config: LinkerConfig      # Original linker config used
    metadata: dict                   # Additional metadata


class TargetPreparer:
    """
    Prepares target proteins with linkers for binder generation.
    
    Uses maximum context window for robust AF2 alignment, with automatic
    fallback to direct coordinate positioning when alignment RMSD is too high.
    
    The linker is attached at the C-terminus of the full protein,
    ensuring LaProteina sees the complete protein context to avoid
    generating binders that collide with other protein regions.
    
    Example:
        preparer = TargetPreparer(
            output_dir="./prepared_targets",
            linker_mode="polyglycine"
        )
        
        result = preparer.prepare_target_with_linker(
            target_pdb="my_target.pdb",
            linker_config=LinkerConfig(attachment_residue=74, linker_length=20, ...),
            binder_length_range=(40, 80)
        )
    """
    
    def __init__(
        self,
        output_dir: str = "./prepared_targets",
        linker_mode: str = "polyglycine",
        af_params_dir: Optional[str] = None,
        rmsd_threshold: float = DEFAULT_RMSD_THRESHOLD,
    ):
        """
        Initialize the target preparer.
        
        Args:
            output_dir: Directory to save prepared PDB files
            linker_mode: "polyglycine" for simple Gly linker, "af2" for AF2-folded linker
            af_params_dir: AlphaFold params directory (required if linker_mode="af2")
            rmsd_threshold: RMSD threshold (Å) above which direct coordinate
                            positioning is used instead of alignment-based placement
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.linker_mode = linker_mode
        self.af_params_dir = af_params_dir
        self.rmsd_threshold = rmsd_threshold
        
        if linker_mode == "af2" and not af_params_dir:
            # Try to find default AF2 params
            bindcraft_root = Path(__file__).parents[2]
            default_params = bindcraft_root / "params"
            if default_params.exists():
                self.af_params_dir = str(default_params)
            else:
                logger.warning("AF2 params not found. Set af_params_dir or use linker_mode='polyglycine'")
        
        logger.info(f"TargetPreparer initialized with mode={linker_mode}, "
                    f"rmsd_threshold={rmsd_threshold} Å, output_dir={output_dir}")
    
    def prepare_target_with_linker(
        self,
        target_pdb: str,
        linker_config: LinkerConfig,
        binder_length_range: Tuple[int, int],
        chain_id: str = "A"
    ) -> PreparedTarget:
        """
        Prepare a target PDB with a linker for LaProteina generation.
        
        Process:
        1. Extract focused region from target (hotspot + large buffer for robust alignment)
        2. Fold linker onto focused region using AF2
        3. Align AF2 result to full target; check RMSD
        4. If RMSD > threshold, fall back to direct coordinate positioning
        5. Append positioned linker to FULL original target
        6. Result: full target + floating linker ready for LaProteina
        
        Args:
            target_pdb: Path to the original target PDB
            linker_config: Linker configuration specifying length, attachment, and hotspot info
            binder_length_range: (min, max) length for the binder to be generated
            chain_id: Chain ID of the target (default "A")
            
        Returns:
            PreparedTarget with paths and metadata
        """
        # Load the full target structure
        full_target = strucio.load_structure(target_pdb)
        
        # Filter for the target chain
        if chain_id:
            full_target = full_target[full_target.chain_id == chain_id]
        
        if len(full_target) == 0:
            raise ValueError(f"No atoms found for chain {chain_id} in {target_pdb}")
        
        # Get target length (number of residues)
        target_residue_ids = sorted(np.unique(full_target.res_id))
        target_length = len(target_residue_ids)
        
        # Get hotspot range from linker config
        hotspot_start, hotspot_end = linker_config.hotspot_range
        hotspot_size = hotspot_end - hotspot_start + 1
        
        logger.info(f"Target {Path(target_pdb).name}: {target_length} residues, chain {chain_id}")
        logger.info(f"Hotspot region: {hotspot_start}-{hotspot_end} ({hotspot_size} residues)")
        logger.info(f"Linker config: attachment={linker_config.attachment_residue}, "
                    f"length={linker_config.linker_length}, position={linker_config.edge_type}")

        # Special case: zero-length linker means "use current chain end as-is".
        # This enables linked-input workflows without direct_binder_mode.
        if linker_config.linker_length <= 0:
            prepared_target = full_target.copy()
            prepared_target.chain_id[:] = "A"

            target_residue_ids = sorted(np.unique(prepared_target.res_id))
            res_map = {old: new for new, old in enumerate(target_residue_ids, 1)}
            for old, new in res_map.items():
                mask = prepared_target.res_id == old
                prepared_target.res_id[mask] = new + 100000
            prepared_target.res_id = prepared_target.res_id - 100000

            binder_min, binder_max = binder_length_range
            total_fixed_length = target_length
            contig_string = f"A1-{total_fixed_length}/{binder_min}-{binder_max}"

            target_name = Path(target_pdb).stem
            output_filename = f"{target_name}_{linker_config.config_id}.pdb"
            output_path = self.output_dir / output_filename
            strucio.save_structure(str(output_path), prepared_target)

            logger.info(
                "Zero-length linker config detected; using original target as fixed context "
                f"(contig: {contig_string})"
            )

            return PreparedTarget(
                pdb_path=str(output_path),
                contig_string=contig_string,
                target_length=target_length,
                linker_length=0,
                total_fixed_length=total_fixed_length,
                linker_config=linker_config,
                metadata={
                    "original_target": target_pdb,
                    "chain_id": chain_id,
                    "binder_length_range": binder_length_range,
                    "linker_mode": self.linker_mode,
                    "hotspot_range": linker_config.hotspot_range,
                    "attachment_residue": linker_config.attachment_residue,
                    "positioning_method": "none_zero_length_linker",
                    "context_length": target_length,
                },
            )
        
        # Step 1: Extract LARGE focused region around the attachment point
        # Use maximum context: entire hotspot + 100 residues buffer on each side
        max_context = min(hotspot_size + 200, target_length)
        
        focused_structure, res_id_mapping = self._extract_focused_region(
            full_target,
            linker_config.attachment_residue,
            context_length=max_context,
        )
        
        logger.info(f"Using maximum context window: {len(np.unique(focused_structure.res_id))} residues "
                    f"(hotspot={hotspot_size}, buffer=±100, target={target_length})")
        
        # Step 2: Fold linker onto focused region using AF2
        folded_linker, positioning_method = self._fold_linker_on_focused_region(
            focused_structure,
            full_target,
            linker_config,
            res_id_mapping,
            target_pdb,
        )
        
        # Step 3: Prepare the full target (renumber 1..N)
        prepared_target = full_target.copy()
        prepared_target.chain_id[:] = "A"
        
        # Renumber target residues 1..N
        res_map = {old: new for new, old in enumerate(target_residue_ids, 1)}
        for old, new in res_map.items():
            mask = prepared_target.res_id == old
            prepared_target.res_id[mask] = new + 100000  # Temp offset
        prepared_target.res_id = prepared_target.res_id - 100000
        
        # Step 4: Append linker to full target
        # Linker residues are numbered starting from target_length + 1
        linker_start_res = target_length + 1
        folded_linker.chain_id[:] = "A"
        
        # Renumber linker residues
        linker_res_ids = sorted(np.unique(folded_linker.res_id))
        for i, old_res in enumerate(linker_res_ids):
            mask = folded_linker.res_id == old_res
            folded_linker.res_id[mask] = linker_start_res + i
        
        # Combine: full target + floating linker
        combined_structure = prepared_target + folded_linker
        
        # Calculate total fixed length (target + linker)
        total_fixed_length = target_length + linker_config.linker_length
        
        # Build contig string
        binder_min, binder_max = binder_length_range
        contig_string = f"A1-{total_fixed_length}/{binder_min}-{binder_max}"
        
        # Generate output filename
        target_name = Path(target_pdb).stem
        output_filename = f"{target_name}_{linker_config.config_id}.pdb"
        output_path = self.output_dir / output_filename
        
        # Save prepared structure
        strucio.save_structure(str(output_path), combined_structure)
        
        logger.info(f"Prepared target saved: {output_path}")
        logger.info(f"  Full target: residues 1-{target_length}")
        logger.info(f"  Linker: residues {linker_start_res}-{total_fixed_length} "
                    f"(positioned via {positioning_method})")
        logger.info(f"Contig string: {contig_string}")
        
        return PreparedTarget(
            pdb_path=str(output_path),
            contig_string=contig_string,
            target_length=target_length,
            linker_length=linker_config.linker_length,
            total_fixed_length=total_fixed_length,
            linker_config=linker_config,
            metadata={
                "original_target": target_pdb,
                "chain_id": chain_id,
                "binder_length_range": binder_length_range,
                "linker_mode": self.linker_mode,
                "hotspot_range": linker_config.hotspot_range,
                "attachment_residue": linker_config.attachment_residue,
                "positioning_method": positioning_method,
                "context_length": len(np.unique(focused_structure.res_id)),
            }
        )
    
    def _extract_focused_region(
        self,
        structure: struc.AtomArray,
        attachment_residue: int,
        context_length: int = 250,
    ) -> Tuple[struc.AtomArray, Dict[int, int]]:
        """
        Extract a large focused region centered on the attachment point.
        
        Unlike _extract_hotspot_region which only takes the hotspot,
        this extracts the hotspot + large buffer for robust alignment.
        Keeps original 3D coordinates but renumbers residues 1..N for AF2.
        
        Args:
            structure: Full target structure.
            attachment_residue: Residue ID where the linker attaches.
            context_length: Total number of residues to include.
            
        Returns:
            Tuple of (focused_structure, res_id_mapping) where res_id_mapping
            maps new_res_id -> original_res_id for coordinate alignment.
        """
        all_res_ids = np.sort(np.unique(structure.res_id))
        n_residues = len(all_res_ids)
        
        # Find attachment residue index
        attachment_idx = np.searchsorted(all_res_ids, attachment_residue)
        if attachment_idx >= n_residues or all_res_ids[attachment_idx] != attachment_residue:
            # Find nearest residue
            nearest_idx = np.argmin(np.abs(all_res_ids - attachment_residue))
            logger.warning(
                f"Attachment residue {attachment_residue} not found. "
                f"Using nearest: {all_res_ids[nearest_idx]}"
            )
            attachment_idx = nearest_idx
        
        # Calculate extraction window centered on attachment point
        half_context = context_length // 2
        start_idx = max(0, attachment_idx - half_context)
        end_idx = min(n_residues, attachment_idx + half_context + 1)
        
        # Adjust to fill context_length if near edges
        actual_length = end_idx - start_idx
        if actual_length < context_length:
            if start_idx == 0:
                end_idx = min(n_residues, context_length)
            elif end_idx == n_residues:
                start_idx = max(0, n_residues - context_length)
        
        focused_res_ids = all_res_ids[start_idx:end_idx]
        
        # Extract atoms for focused region
        focused_mask = np.isin(structure.res_id, focused_res_ids)
        focused = structure[focused_mask].copy()
        
        # Renumber 1..N for AF2 compatibility, keeping coordinate mapping
        # res_id_mapping: {new_id: original_id}
        res_id_mapping = {}
        for new_id, old_id in enumerate(sorted(focused_res_ids), 1):
            res_id_mapping[new_id] = int(old_id)
            focused.res_id[focused.res_id == old_id] = new_id + 100000
        focused.res_id = focused.res_id - 100000
        
        focused.chain_id[:] = "A"
        
        logger.info(
            f"Extracted focused region: {len(focused_res_ids)} residues "
            f"(original {focused_res_ids[0]}-{focused_res_ids[-1]}), "
            f"centered on attachment residue {attachment_residue}"
        )
        
        return focused, res_id_mapping
    
    def _fold_linker_on_focused_region(
        self,
        focused_structure: struc.AtomArray,
        full_target: struc.AtomArray,
        linker_config: LinkerConfig,
        res_id_mapping: Dict[int, int],
        original_pdb: str,
    ) -> Tuple[struc.AtomArray, str]:
        """
        Fold a linker onto the focused region using AF2, then align
        back to the full target.
        
        If alignment RMSD exceeds the threshold, falls back to direct
        coordinate positioning using the attachment point.
        
        Args:
            focused_structure: Large focused region (renumbered 1..N)
            full_target: Full original target structure (original coordinates)
            linker_config: Linker configuration
            res_id_mapping: {new_res_id: original_res_id} from extraction
            original_pdb: Path to original full target PDB
            
        Returns:
            Tuple of (linker_atoms, positioning_method) where
            positioning_method is "alignment" or "direct_coordinates".
        """
        from laproteina.proteinfoundation.utils.linker_folder import fold_linked_pdb
        
        # Save focused region to temp file
        temp_focused = self.output_dir / f"temp_focused_{linker_config.config_id}.pdb"
        strucio.save_structure(str(temp_focused), focused_structure)
        
        # Generate linker sequence (GSGSGS pattern for flexibility)
        linker_length = linker_config.linker_length
        linker_seq = "GS" * (linker_length // 2)
        if linker_length % 2 == 1:
            linker_seq += "G"
        
        # Fold linker with AF2
        temp_folded = self.output_dir / f"temp_folded_{linker_config.config_id}.pdb"
        
        try:
            logger.info(f"Folding {linker_length}-residue linker onto focused region "
                       f"({len(np.unique(focused_structure.res_id))} residues)...")
            
            result = fold_linked_pdb(
                target_pdb=str(temp_focused),
                linker_sequence=linker_seq,
                output_pdb=str(temp_folded),
                af_params_dir=self.af_params_dir,
            )
            
            # fold_linked_pdb returns (output_path, alignment_rmsd)
            if isinstance(result, tuple):
                _, alignment_rmsd = result
            else:
                # Backwards compatibility: old version returns just path
                alignment_rmsd = float('inf')
            
            logger.info(f"AF2 folding alignment RMSD: {alignment_rmsd:.3f} Å")
            
            # Load the folded/stitched structure
            folded = strucio.load_structure(str(temp_folded))
            
            # Extract just the linker portion (last N residues)
            focused_len = len(np.unique(focused_structure.res_id))
            all_res_ids = sorted(np.unique(folded.res_id))
            linker_res_ids = all_res_ids[focused_len:]  # Linker is after focused region
            
            if not linker_res_ids:
                raise ValueError("No linker residues found in folded structure")
            
            linker_mask = np.isin(folded.res_id, linker_res_ids)
            linker_atoms = folded[linker_mask].copy()
            
            # Check alignment quality
            if alignment_rmsd > self.rmsd_threshold:
                logger.warning(
                    f"Alignment RMSD ({alignment_rmsd:.3f} Å) exceeds threshold "
                    f"({self.rmsd_threshold} Å). Using direct coordinate positioning."
                )
                linker_atoms = self._position_linker_by_coordinates(
                    linker_atoms,
                    full_target,
                    linker_config.attachment_residue,
                )
                positioning_method = "direct_coordinates"
            else:
                logger.info(
                    f"Alignment-based positioning (RMSD={alignment_rmsd:.3f} Å, "
                    f"threshold={self.rmsd_threshold} Å)"
                )
                positioning_method = "alignment"
            
            logger.info(f"Extracted {len(linker_res_ids)} linker residues "
                       f"(method: {positioning_method})")
            return linker_atoms, positioning_method
            
        finally:
            # Cleanup temp files
            if temp_focused.exists():
                temp_focused.unlink()
            if temp_folded.exists():
                temp_folded.unlink()
    
    def _position_linker_by_coordinates(
        self,
        linker: struc.AtomArray,
        full_target: struc.AtomArray,
        attachment_residue: int,
    ) -> struc.AtomArray:
        """
        Position linker directly using attachment point coordinates.
        
        This bypasses alignment and places the linker's first residue
        at the exact attachment point coordinate from the full target.
        Used as fallback when Kabsch alignment RMSD is too high.
        
        The method translates the entire linker so that its attachment
        atom (N of first residue or C of last residue, depending on 
        which is closest to the target attachment point) coincides with
        the CA atom of the attachment residue on the full target.
        
        Args:
            linker: Linker structure from AF2
            full_target: Full target structure (original coordinates)
            attachment_residue: Residue ID on the target where linker attaches
            
        Returns:
            Positioned linker structure
        """
        # Get attachment point CA coordinate from full target
        attachment_mask = full_target.res_id == attachment_residue
        target_ca = full_target[attachment_mask & (full_target.atom_name == "CA")]
        
        if len(target_ca) == 0:
            # Try nearby residues
            nearby_mask = np.abs(full_target.res_id - attachment_residue) <= 2
            target_ca = full_target[nearby_mask & (full_target.atom_name == "CA")]
            if len(target_ca) == 0:
                raise ValueError(f"Could not find CA atom near residue {attachment_residue}")
            # Use the closest one
            target_ca = target_ca[:1]
        
        target_attach_coord = target_ca.coord[0]  # (3,)
        
        # Get linker attachment atom coordinates
        linker_res_ids = sorted(np.unique(linker.res_id))
        
        # Try C-terminal of linker (last residue, C atom) — connects to target
        last_res = linker_res_ids[-1]
        last_c = linker[(linker.res_id == last_res) & (linker.atom_name == "C")]
        
        # Try N-terminal of linker (first residue, N atom) — connects to target  
        first_res = linker_res_ids[0]
        first_n = linker[(linker.res_id == first_res) & (linker.atom_name == "N")]
        
        # Pick whichever endpoint is closer to the target attachment point
        candidates = []
        if len(last_c) > 0:
            dist_c = np.linalg.norm(last_c.coord[0] - target_attach_coord)
            candidates.append((last_c.coord[0], dist_c, "C-terminal"))
        if len(first_n) > 0:
            dist_n = np.linalg.norm(first_n.coord[0] - target_attach_coord)
            candidates.append((first_n.coord[0], dist_n, "N-terminal"))
        
        if not candidates:
            # Fallback: use CA of first residue
            first_ca = linker[(linker.res_id == first_res) & (linker.atom_name == "CA")]
            if len(first_ca) == 0:
                raise ValueError("Could not find any attachment atoms on linker")
            candidates.append((first_ca.coord[0], 0.0, "CA-fallback"))
        
        # Sort by distance, pick closest
        candidates.sort(key=lambda x: x[1])
        linker_attach_coord, _, attach_type = candidates[0]
        
        # Calculate translation
        translation = target_attach_coord - linker_attach_coord
        
        # Apply translation to entire linker
        positioned_linker = linker.copy()
        positioned_linker.coord = linker.coord + translation
        
        logger.info(
            f"Direct coordinate positioning: translated {np.linalg.norm(translation):.3f} Å "
            f"(attachment={attach_type}, target residue {attachment_residue})"
        )
        return positioned_linker
    
    def _extract_hotspot_region(
        self,
        structure: struc.AtomArray,
        hotspot_start: int,
        hotspot_end: int
    ) -> struc.AtomArray:
        """
        Extract the hotspot region from the structure.
        
        Returns a new structure containing only the hotspot residues,
        renumbered from 1.
        
        Note: This is the original small-context extraction. For alignment,
        use _extract_focused_region() which provides much more context.
        """
        # Get residues in the hotspot range
        res_ids = np.unique(structure.res_id)
        hotspot_res_ids = [r for r in res_ids if hotspot_start <= r <= hotspot_end]
        
        if not hotspot_res_ids:
            raise ValueError(f"No residues found in hotspot range {hotspot_start}-{hotspot_end}")
        
        # Extract hotspot atoms
        hotspot_mask = np.isin(structure.res_id, hotspot_res_ids)
        hotspot = structure[hotspot_mask].copy()
        
        # Renumber 1..N
        for new_id, old_id in enumerate(sorted(hotspot_res_ids), 1):
            hotspot.res_id[hotspot.res_id == old_id] = new_id + 100000
        hotspot.res_id = hotspot.res_id - 100000
        
        hotspot.chain_id[:] = "A"
        
        logger.info(f"Extracted hotspot region: {len(hotspot_res_ids)} residues")
        return hotspot
    
    def prepare_batch(
        self,
        target_pdb: str,
        linker_configs: List[LinkerConfig],
        binder_length_range: Tuple[int, int],
        chain_id: str = "A"
    ) -> List[PreparedTarget]:
        """
        Prepare multiple target+linker combinations.
        
        Args:
            target_pdb: Path to the target PDB
            linker_configs: List of linker configurations
            binder_length_range: (min, max) binder length
            chain_id: Target chain ID
            
        Returns:
            List of PreparedTarget objects
        """
        results = []
        
        for i, config in enumerate(linker_configs):
            try:
                logger.info(f"Preparing config {i+1}/{len(linker_configs)}: {config}")
                result = self.prepare_target_with_linker(
                    target_pdb=target_pdb,
                    linker_config=config,
                    binder_length_range=binder_length_range,
                    chain_id=chain_id
                )
                results.append(result)
                
                # Clear GPU memory after each linker preparation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to prepare target with config {config}: {e}")
                continue
        
        logger.info(f"Prepared {len(results)}/{len(linker_configs)} target+linker combinations")
        return results


def get_target_length(pdb_path: str, chain_id: str = "A") -> int:
    """
    Utility function to get the number of residues in a PDB chain.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain ID to analyze
        
    Returns:
        Number of residues in the chain
    """
    structure = strucio.load_structure(pdb_path)
    if chain_id:
        structure = structure[structure.chain_id == chain_id]
    
    unique_res_ids = np.unique(structure.res_id)
    return len(unique_res_ids)


def remove_linker_from_output(
    pdb_path: str,
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_path: str
) -> str:
    """
    Remove the linker from LaProteina output to get clean target + binder.
    
    LaProteina output has: target (1..target_len) + linker (target_len+1..target_len+linker_len) + binder.
    We output: target (chain A) + (kept linker tail + binder) (chain B).
    
    Args:
        pdb_path: Path to LaProteina output PDB
        target_length: Length of original target
        linker_length: Length of linker to remove
        keep_linker_end_n: Number of residues to keep from linker C-terminus as
            binder seed (0 means remove full linker)
        output_path: Path to save cleaned PDB
        
    Returns:
        Path to cleaned PDB file
    """
    structure = strucio.load_structure(pdb_path)

    if keep_linker_end_n < 0:
        raise ValueError(f"keep_linker_end_n must be >= 0, got {keep_linker_end_n}")
    if keep_linker_end_n > linker_length:
        raise ValueError(
            f"keep_linker_end_n ({keep_linker_end_n}) cannot exceed linker_length ({linker_length})"
        )
    
    # Get all residue IDs
    all_res_ids = sorted(np.unique(structure.res_id))
    
    # Identify residue ranges
    linker_start = target_length + 1
    linker_end = target_length + linker_length
    
    # Target residues: 1 to target_length
    target_res_ids = [r for r in all_res_ids if r <= target_length]
    
    # Linker residues: target_length+1 to target_length+linker_length
    linker_res_ids = [r for r in all_res_ids if linker_start <= r <= linker_end]

    # Keep only the final N linker residues as binder seed (if requested)
    linker_keep_ids = linker_res_ids[-keep_linker_end_n:] if keep_linker_end_n > 0 else []
    linker_removed_ids = linker_res_ids[:-keep_linker_end_n] if keep_linker_end_n > 0 else linker_res_ids
    
    # Binder residues: everything after linker
    binder_res_ids = [r for r in all_res_ids if r > linker_end]
    
    if not binder_res_ids:
        logger.warning(f"No binder residues found in {pdb_path}")
        return pdb_path
    
    logger.info(
        f"Removing linker: target={len(target_res_ids)} res, "
        f"linker_removed={len(linker_removed_ids)} res, "
        f"linker_kept={len(linker_keep_ids)} res, "
        f"binder={len(binder_res_ids)} res"
    )
    
    # Extract target and chain-B residues (kept linker tail + binder)
    chain_b_res_ids = sorted(linker_keep_ids + binder_res_ids)
    target_mask = np.isin(structure.res_id, target_res_ids)
    binder_mask = np.isin(structure.res_id, chain_b_res_ids)
    
    target_atoms = structure[target_mask].copy()
    binder_atoms = structure[binder_mask].copy()
    
    # Assign chains
    target_atoms.chain_id[:] = "A"
    binder_atoms.chain_id[:] = "B"
    
    # Renumber target 1..N
    for new_id, old_id in enumerate(sorted(target_res_ids), 1):
        target_atoms.res_id[target_atoms.res_id == old_id] = new_id + 100000
    target_atoms.res_id = target_atoms.res_id - 100000
    
    # Renumber chain B 1..M (kept linker tail first, then binder)
    for new_id, old_id in enumerate(chain_b_res_ids, 1):
        binder_atoms.res_id[binder_atoms.res_id == old_id] = new_id + 100000
    binder_atoms.res_id = binder_atoms.res_id - 100000
    
    # Combine
    cleaned = target_atoms + binder_atoms
    
    # Save
    strucio.save_structure(output_path, cleaned)
    
    logger.info(f"Saved cleaned structure (no linker): {output_path}")
    return output_path


def batch_remove_linker(
    pdb_paths: List[str],
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_dir: str
) -> List[str]:
    """
    Remove linker from multiple LaProteina output files.
    
    Args:
        pdb_paths: List of LaProteina output PDB paths
        target_length: Length of original target
        linker_length: Length of linker to remove
        output_dir: Directory to save cleaned PDBs
        
    Returns:
        List of paths to cleaned PDB files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_paths = []
    for pdb_path in pdb_paths:
        try:
            output_path = output_dir / Path(pdb_path).name
            cleaned = remove_linker_from_output(
                pdb_path=pdb_path,
                target_length=target_length,
                linker_length=linker_length,
                keep_linker_end_n=keep_linker_end_n,
                output_path=str(output_path)
            )
            cleaned_paths.append(cleaned)
        except Exception as e:
            logger.error(f"Failed to remove linker from {pdb_path}: {e}")
    
    logger.info(f"Removed linker from {len(cleaned_paths)}/{len(pdb_paths)} files")
    return cleaned_paths


def create_pyrosetta_input(
    pdb_path: str,
    original_target_pdb: str,
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_path: str
) -> str:
    """
    Create PyRosetta input: original starting target + extracted binder.
    
    This combines the target core with chain B built from
    (kept linker tail + generated binder) extracted from LaProteina output.
    
    Args:
        pdb_path: Path to LaProteina output PDB (target + linker + binder)
        original_target_pdb: Path to the original starting target PDB
        target_length: Length of original target
        linker_length: Length of linker to skip
        keep_linker_end_n: Number of linker residues to keep from C-terminus
            as binder seed
        output_path: Path to save the PyRosetta input PDB
        
    Returns:
        Path to the saved PyRosetta input PDB
    """
    # Load LaProteina output
    laproteina_output = strucio.load_structure(pdb_path)

    if keep_linker_end_n < 0:
        raise ValueError(f"keep_linker_end_n must be >= 0, got {keep_linker_end_n}")
    if keep_linker_end_n > linker_length:
        raise ValueError(
            f"keep_linker_end_n ({keep_linker_end_n}) cannot exceed linker_length ({linker_length})"
        )
    
    # Load original target
    original_target = strucio.load_structure(original_target_pdb)
    original_target = original_target[original_target.chain_id == "A"]
    target_core_ids = [r for r in sorted(np.unique(original_target.res_id)) if r <= target_length]
    if not target_core_ids:
        logger.warning(
            f"No target residues <= {target_length} found in {original_target_pdb}; "
            "falling back to first target_length residues by order."
        )
        ordered_ids = sorted(np.unique(original_target.res_id))
        target_core_ids = ordered_ids[:target_length]
    target_core_mask = np.isin(original_target.res_id, target_core_ids)
    original_target = original_target[target_core_mask]
    
    # Get all residue IDs from LaProteina output
    all_res_ids = sorted(np.unique(laproteina_output.res_id))
    
    # Identify binder residues (after target + linker)
    linker_start = target_length + 1
    linker_end = target_length + linker_length
    linker_res_ids = [r for r in all_res_ids if linker_start <= r <= linker_end]
    linker_keep_ids = linker_res_ids[-keep_linker_end_n:] if keep_linker_end_n > 0 else []
    binder_start = target_length + linker_length + 1
    binder_res_ids = [r for r in all_res_ids if r >= binder_start]
    
    if not binder_res_ids:
        logger.warning(f"No binder residues found in {pdb_path}")
        return pdb_path
    
    # Extract kept linker tail + binder atoms
    chain_b_res_ids = sorted(linker_keep_ids + binder_res_ids)
    binder_mask = np.isin(laproteina_output.res_id, chain_b_res_ids)
    binder_atoms = laproteina_output[binder_mask].copy()
    
    # Prepare original target (chain A, renumber 1..N)
    target_atoms = original_target.copy()
    target_atoms.chain_id[:] = "A"
    orig_res_ids = sorted(np.unique(target_atoms.res_id))
    for new_id, old_id in enumerate(orig_res_ids, 1):
        target_atoms.res_id[target_atoms.res_id == old_id] = new_id + 100000
    target_atoms.res_id = target_atoms.res_id - 100000
    
    # Prepare binder (chain B, renumber 1..M)
    binder_atoms.chain_id[:] = "B"
    for new_id, old_id in enumerate(chain_b_res_ids, 1):
        binder_atoms.res_id[binder_atoms.res_id == old_id] = new_id + 100000
    binder_atoms.res_id = binder_atoms.res_id - 100000
    
    # Combine
    combined = target_atoms + binder_atoms
    
    # Save
    strucio.save_structure(output_path, combined)
    
    logger.debug(f"Created PyRosetta input: {output_path}")
    return output_path


def batch_create_pyrosetta_input(
    pdb_paths: List[str],
    original_target_pdb: str,
    target_length: int,
    linker_length: int,
    keep_linker_end_n: int,
    output_dir: str
) -> List[str]:
    """
    Create PyRosetta input files for multiple LaProteina outputs.
    
    Each output contains: original starting target (chain A) + extracted binder (chain B)
    
    Args:
        pdb_paths: List of LaProteina output PDB paths
        original_target_pdb: Path to the original starting target PDB
        target_length: Length of original target
        linker_length: Length of linker to skip
        output_dir: Directory to save PyRosetta input PDBs
        
    Returns:
        List of paths to PyRosetta input PDB files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pyrosetta_paths = []
    for pdb_path in pdb_paths:
        try:
            output_path = output_dir / Path(pdb_path).name
            result = create_pyrosetta_input(
                pdb_path=pdb_path,
                original_target_pdb=original_target_pdb,
                target_length=target_length,
                linker_length=linker_length,
                keep_linker_end_n=keep_linker_end_n,
                output_path=str(output_path)
            )
            pyrosetta_paths.append(result)
        except Exception as e:
            logger.error(f"Failed to create PyRosetta input for {pdb_path}: {e}")
    
    logger.info(f"Created {len(pyrosetta_paths)}/{len(pdb_paths)} PyRosetta input files")
    return pyrosetta_paths
