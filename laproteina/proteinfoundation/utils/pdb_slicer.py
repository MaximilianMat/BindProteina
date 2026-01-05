import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from loguru import logger

def slice_pdb_by_ranges(
    input_pdb: str,
    output_pdb: str,
    residue_ranges: List[Tuple[int, int]],
    chain_id: str = "A",
    renumber: bool = False
) -> str:
    """
    Slices a PDB file to include only residues within the specified ranges.
    
    Args:
        input_pdb: Path to the source PDB.
        output_pdb: Path to save the sliced PDB.
        residue_ranges: List of (start, end) residue numbers.
        chain_id: The chain to slice from.
        renumber: If True, renumbers residues starting from 1.
        
    Returns:
        Path to the output PDB.
    """
    selected_lines = []
    
    # For renumbering
    res_map = {}
    next_res_num = 1
    
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    res_num = int(line[22:26])
                    current_chain = line[21]
                    
                    if current_chain == chain_id:
                        in_range = any(start <= res_num <= end for start, end in residue_ranges)
                        if in_range:
                            if renumber:
                                if res_num not in res_map:
                                    res_map[res_num] = next_res_num
                                    next_res_num += 1
                                new_res_num = res_map[res_num]
                                line = line[:22] + f"{new_res_num:4d}" + line[26:]
                            selected_lines.append(line)
                except ValueError:
                    continue
            elif line.startswith("TER"):
                # Only keep TER if the preceding atom was selected
                if selected_lines and selected_lines[-1].startswith(("ATOM", "HETATM")):
                    selected_lines.append(line)
    
    if not selected_lines:
        raise ValueError(f"No residues found in specified ranges {residue_ranges} for chain {chain_id}")

    with open(output_pdb, 'w') as f:
        for line in selected_lines:
            f.write(line)
        f.write("END\n")
        
    logger.info(f"Sliced PDB saved to: {output_pdb}")
    return output_pdb

def parse_hotspot_string(hotspot_str: str) -> List[Tuple[int, int]]:
    """
    Parses a string like "122-135,234-362" into a list of tuples.
    """
    if not hotspot_str or not hotspot_str.strip():
        return []
    
    ranges = []
    parts = hotspot_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            ranges.append((start, end))
        else:
            res = int(part)
            ranges.append((res, res))
    return ranges
