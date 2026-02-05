"""
Linker Configuration Generator for LaProteina V2 Pipeline

Generates diverse linker configurations based on hotspot edges with variable lengths.
Used in the initial generation phase to create diverse starting candidates.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
from loguru import logger

from laproteina.proteinfoundation.utils.pdb_slicer import parse_hotspot_string


@dataclass
class LinkerConfig:
    """Configuration for a single linker attachment."""
    
    attachment_residue: int      # Which residue the linker attaches to (hotspot edge)
    linker_length: int           # Length of the linker in residues
    edge_type: str               # "start" or "end" of the hotspot range
    hotspot_range: Tuple[int, int]  # Original hotspot range (start, end)
    config_id: str = ""          # Unique identifier for this config
    
    def __post_init__(self):
        if not self.config_id:
            self.config_id = f"L{self.linker_length}_E{self.attachment_residue}_{self.edge_type[0]}"
    
    def __repr__(self):
        return (f"LinkerConfig(residue={self.attachment_residue}, "
                f"length={self.linker_length}, edge={self.edge_type})")


class LinkerConfigGenerator:
    """
    Generates linker configurations based on hotspot regions.
    
    Distributes attachment points evenly across the hotspot region(s),
    with the number of points determined by n_configs.
    
    Example:
        generator = LinkerConfigGenerator(
            hotspot_string="28-74,157-174",
            linker_length_range=(15, 30),
            mode="round_robin",
            n_attachment_points=8  # Will create 8 evenly spaced attachment points
        )
        configs = generator.generate_configs(n_configs=10)
    """
    
    def __init__(
        self,
        hotspot_string: str,
        linker_length_range: Tuple[int, int],
        mode: str = "round_robin",
        seed: Optional[int] = None,
        n_attachment_points: Optional[int] = None
    ):
        """
        Initialize the linker configuration generator.
        
        Args:
            hotspot_string: Hotspot regions string (e.g., "28-74,157-174")
            linker_length_range: (min_length, max_length) for linker
            mode: "round_robin" cycles through edges, "random" picks randomly
            seed: Random seed for reproducibility (used in random mode)
            n_attachment_points: Number of attachment points to create across the
                                 hotspot region(s). If None, defaults to evenly 
                                 distributing based on region size.
        """
        self.hotspot_string = hotspot_string
        self.linker_length_range = linker_length_range
        self.mode = mode
        self.n_attachment_points = n_attachment_points
        
        if seed is not None:
            random.seed(seed)
        
        # Parse hotspot ranges
        self.hotspot_ranges = parse_hotspot_string(hotspot_string)
        if not self.hotspot_ranges:
            raise ValueError(f"Could not parse hotspot string: {hotspot_string}")
        
        # Extract attachment points (evenly distributed across hotspot regions)
        self.edges = self._extract_attachment_points(n_attachment_points)
        
        logger.info(f"LinkerConfigGenerator initialized with {len(self.edges)} attachment points: {[e[0] for e in self.edges]}")
        logger.info(f"Linker length range: {linker_length_range}, mode: {mode}")
    
    def _extract_attachment_points(
        self, 
        n_points: Optional[int] = None
    ) -> List[Tuple[int, str, Tuple[int, int]]]:
        """
        Extract evenly distributed attachment points from hotspot ranges.
        
        Distributes attachment points evenly across all hotspot residues,
        ensuring we always include the start and end of each range.
        
        Args:
            n_points: Number of attachment points to create. If None, 
                      uses a reasonable default based on region size.
        
        Returns:
            List of (residue_number, position_type, original_range) tuples
        """
        # Collect all residues from all hotspot ranges
        all_residues = []
        residue_to_range = {}
        for start, end in self.hotspot_ranges:
            for res in range(start, end + 1):
                all_residues.append(res)
                residue_to_range[res] = (start, end)
        
        total_residues = len(all_residues)
        
        # Determine number of attachment points
        if n_points is None:
            # Default: one point every ~10 residues, minimum 3
            n_points = max(3, total_residues // 10)
        
        # Ensure we have at least 2 points and at most the total residues
        n_points = max(2, min(n_points, total_residues))
        
        # Select evenly spaced indices
        if n_points >= total_residues:
            # Use all residues
            selected_residues = all_residues
        else:
            # Evenly space the attachment points
            step = (total_residues - 1) / (n_points - 1) if n_points > 1 else 0
            indices = [int(round(i * step)) for i in range(n_points)]
            # Ensure uniqueness
            indices = sorted(set(indices))
            selected_residues = [all_residues[i] for i in indices]
        
        # Create edge tuples with position type
        edges = []
        for idx, res in enumerate(selected_residues):
            hotspot_range = residue_to_range[res]
            # Determine position type based on position in the range
            start, end = hotspot_range
            if res == start:
                pos_type = "start"
            elif res == end:
                pos_type = "end"
            else:
                # Calculate relative position (0-100%)
                rel_pos = (res - start) / (end - start) * 100 if end != start else 50
                pos_type = f"p{int(rel_pos)}"  # e.g., p25, p50, p75
            
            edges.append((res, pos_type, hotspot_range))
        
        return edges
    
    def _get_random_length(self) -> int:
        """Get a random linker length within the configured range."""
        min_len, max_len = self.linker_length_range
        return random.randint(min_len, max_len)
    
    def _create_config(
        self, 
        edge: Tuple[int, str, Tuple[int, int]], 
        length: int,
        index: int
    ) -> LinkerConfig:
        """Create a LinkerConfig from an edge tuple."""
        residue, pos_type, hotspot_range = edge
        # Use position type in config_id: s=start, e=end, p25/p50/etc for interior positions
        return LinkerConfig(
            attachment_residue=residue,
            linker_length=length,
            edge_type=pos_type,
            hotspot_range=hotspot_range,
            config_id=f"cfg{index:03d}_L{length}_R{residue}_{pos_type}"
        )
    
    def generate_configs(self, n_configs: int) -> List[LinkerConfig]:
        """
        Generate n linker configurations.
        
        Args:
            n_configs: Number of configurations to generate
            
        Returns:
            List of LinkerConfig objects
        """
        configs = []
        
        if self.mode == "round_robin":
            configs = self._generate_round_robin(n_configs)
        elif self.mode == "random":
            configs = self._generate_random(n_configs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'round_robin' or 'random'.")
        
        logger.info(f"Generated {len(configs)} linker configurations")
        return configs
    
    def _generate_round_robin(self, n_configs: int) -> List[LinkerConfig]:
        """
        Generate configs by cycling through edges in order.
        
        Distributes configs evenly across all edges, with varying lengths.
        """
        configs = []
        edge_cycle = self._edge_cycle_iterator()
        
        for i in range(n_configs):
            edge = next(edge_cycle)
            length = self._get_random_length()
            config = self._create_config(edge, length, i)
            configs.append(config)
        
        return configs
    
    def _generate_random(self, n_configs: int) -> List[LinkerConfig]:
        """Generate configs by randomly selecting edges and lengths."""
        configs = []
        
        for i in range(n_configs):
            edge = random.choice(self.edges)
            length = self._get_random_length()
            config = self._create_config(edge, length, i)
            configs.append(config)
        
        return configs
    
    def _edge_cycle_iterator(self) -> Iterator[Tuple[int, str, Tuple[int, int]]]:
        """Infinite iterator that cycles through edges."""
        while True:
            for edge in self.edges:
                yield edge
    
    def generate_all_combinations(
        self, 
        length_step: int = 5
    ) -> List[LinkerConfig]:
        """
        Generate all combinations of edges and lengths.
        
        Useful for systematic exploration.
        
        Args:
            length_step: Step size for linker lengths (e.g., 5 means 15, 20, 25, 30)
            
        Returns:
            List of all possible LinkerConfig combinations
        """
        min_len, max_len = self.linker_length_range
        lengths = list(range(min_len, max_len + 1, length_step))
        
        configs = []
        idx = 0
        for edge in self.edges:
            for length in lengths:
                config = self._create_config(edge, length, idx)
                configs.append(config)
                idx += 1
        
        logger.info(f"Generated {len(configs)} exhaustive linker configurations "
                   f"({len(self.edges)} edges x {len(lengths)} lengths)")
        return configs
    
    def get_edge_summary(self) -> str:
        """Get a human-readable summary of attachment points."""
        lines = ["Linker Configuration Generator Summary:"]
        lines.append(f"  Hotspot string: {self.hotspot_string}")
        lines.append(f"  Linker length range: {self.linker_length_range}")
        lines.append(f"  Mode: {self.mode}")
        lines.append(f"  Attachment points ({len(self.edges)} total):")
        for residue, pos_type, hotspot_range in self.edges:
            lines.append(f"    - Residue {residue} ({pos_type} in range {hotspot_range})")
        return "\n".join(lines)


def validate_linker_configs(configs: List[LinkerConfig]) -> bool:
    """
    Validate a list of linker configurations.
    
    Checks for:
    - Non-empty list
    - Valid linker lengths (positive)
    - Valid attachment residues (positive)
    
    Returns:
        True if all configs are valid
        
    Raises:
        ValueError if any config is invalid
    """
    if not configs:
        raise ValueError("Empty config list")
    
    for config in configs:
        if config.linker_length <= 0:
            raise ValueError(f"Invalid linker length {config.linker_length} in {config}")
        if config.attachment_residue <= 0:
            raise ValueError(f"Invalid attachment residue {config.attachment_residue} in {config}")
        if config.edge_type not in ("start", "end"):
            raise ValueError(f"Invalid edge type '{config.edge_type}' in {config}")
    
    return True


# Convenience function for quick config generation
def generate_linker_configs(
    hotspot_string: str,
    linker_length_range: Tuple[int, int],
    n_configs: int,
    mode: str = "round_robin",
    seed: Optional[int] = None
) -> List[LinkerConfig]:
    """
    Convenience function to generate linker configs in one call.
    
    Args:
        hotspot_string: Hotspot regions (e.g., "28-74,157-174")
        linker_length_range: (min, max) linker length
        n_configs: Number of configs to generate
        mode: "round_robin" or "random"
        seed: Random seed for reproducibility
        
    Returns:
        List of LinkerConfig objects
    """
    generator = LinkerConfigGenerator(
        hotspot_string=hotspot_string,
        linker_length_range=linker_length_range,
        mode=mode,
        seed=seed
    )
    return generator.generate_configs(n_configs)

