import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AF2Settings:
    num_recycles: int = 1  # Reduced from 3
    num_models: int = 1
    use_multimer: bool = True
    model_type: str = "multimer_v3"
    
    # Weights for selecting "best" model or just scoring
    # Reusing BindCraft default weights where applicable
    weights: Dict[str, float] = field(default_factory=lambda: {
        "plddt": 0.0, # We are not "designing" so we might not need optimization weights, but ColabDesign uses them.
        # However, for pure prediction/scoring, we just want the output metrics.
        # These weights are more relevant if we were doing hallucination/design steps.
    })

@dataclass
class RosettaSettings:
    relax_repeats: int = 3
    # BindCraft uses: -ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball ... -corrections::beta_nov16 true
    dalphaball_path: str = "/path/to/Dalphaball" # Needs to be configured

@dataclass
class ScoringConfig:
    target_chains: List[str] = field(default_factory=lambda: ["A"])
    binder_chain: str = "B"
    target_hotspot_residues: Optional[str] = None
    af2_settings: AF2Settings = field(default_factory=AF2Settings)
    rosetta_settings: RosettaSettings = field(default_factory=RosettaSettings)
    
    # Path settings
    af_params_dir: str = "./params"
    
def load_config(path: str) -> ScoringConfig:
    # TODO: Implement generic JSON loader to override defaults
    return ScoringConfig()
