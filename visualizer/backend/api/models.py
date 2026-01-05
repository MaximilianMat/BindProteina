"""
Data models for the API.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class DesignModel(BaseModel):
    """Design model for API responses."""
    design_name: str
    model: int
    round_number: int
    status: str
    metrics: Dict[str, Any]
    af2_pdb_path: Optional[str] = None
    relaxed_pdb_path: Optional[str] = None


class RoundModel(BaseModel):
    """Round model for API responses."""
    round_number: int
    num_designs: int
    designs: List[DesignModel]


class EvolutionRunModel(BaseModel):
    """Evolution run model for API responses."""
    run_name: str
    date: str
    num_rounds: int
    rounds: List[RoundModel]


class StructureInfoModel(BaseModel):
    """Structure information model."""
    path: str
    num_atoms: int
    chains: Dict[str, Dict[str, int]]
    sequences: Dict[str, str]
    plddt: Optional[Dict[int, float]] = None
    avg_plddt: Optional[float] = None

