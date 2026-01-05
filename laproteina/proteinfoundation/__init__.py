"""
ProteInFoundation Module - LaProteina Library

This module provides a library-style API for LaProteina protein generation
with integration for BindCraft scoring.

Main components:
- LaProteinaGenerator: Core generation API
- LaProteinaScoringPipeline: End-to-end generation + scoring
- config_builder: Utilities for config translation

Note: Heavy imports (torch, lightning) are deferred until actually used.
"""

# Light imports (no heavy dependencies)
from proteinfoundation.config_builder import (
    auto_build_motif_config,
    build_contig_string,
    get_pdb_chain_length,
)


def __getattr__(name):
    """Lazy import for heavy dependencies."""
    if name in ("LaProteinaGenerator", "MotifConfig", "GenerationConfig", "extract_binder_sequence"):
        from proteinfoundation.laproteina_generator import (
            LaProteinaGenerator,
            MotifConfig,
            GenerationConfig,
            extract_binder_sequence,
        )
        return {
            "LaProteinaGenerator": LaProteinaGenerator,
            "MotifConfig": MotifConfig,
            "GenerationConfig": GenerationConfig,
            "extract_binder_sequence": extract_binder_sequence,
        }[name]
    
    if name in ("LaProteinaScoringPipeline", "BinderCandidate", "run_generation_and_score"):
        from proteinfoundation.scoring_pipeline import (
            LaProteinaScoringPipeline,
            BinderCandidate,
            run_generation_and_score,
        )
        return {
            "LaProteinaScoringPipeline": LaProteinaScoringPipeline,
            "BinderCandidate": BinderCandidate,
            "run_generation_and_score": run_generation_and_score,
        }[name]
    
    raise AttributeError(f"module 'proteinfoundation' has no attribute '{name}'")


__all__ = [
    # Generator
    "LaProteinaGenerator",
    "MotifConfig", 
    "GenerationConfig",
    "extract_binder_sequence",
    # Config
    "auto_build_motif_config",
    "build_contig_string",
    "get_pdb_chain_length",
    # Pipeline
    "LaProteinaScoringPipeline",
    "BinderCandidate",
    "run_generation_and_score",
]
