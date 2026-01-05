"""
LaProteina Integration for BindCraft

This package provides LaProteina protein generation integrated with
BindCraft scoring capabilities.

Usage:
    from laproteina.proteinfoundation import LaProteinaGenerator, LaProteinaScoringPipeline
"""

import os
import sys

# Add this directory to sys.path so that absolute imports like 
# 'import proteinfoundation' work correctly as internal library calls
_lib_path = os.path.dirname(os.path.realpath(__file__))
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

# Re-export main components
from laproteina.proteinfoundation import (
    auto_build_motif_config,
    build_contig_string,
    get_pdb_chain_length,
)

# Heavy imports are lazy-loaded from proteinfoundation module
def __getattr__(name):
    if name in ("LaProteinaGenerator", "LaProteinaScoringPipeline", "BinderCandidate"):
        from laproteina import proteinfoundation
        return getattr(proteinfoundation, name)
    raise AttributeError(f"module 'laproteina' has no attribute '{name}'")

__all__ = [
    "LaProteinaGenerator",
    "LaProteinaScoringPipeline", 
    "BinderCandidate",
    "auto_build_motif_config",
    "build_contig_string",
    "get_pdb_chain_length",
]
