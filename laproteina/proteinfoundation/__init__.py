"""
ProteInFoundation Module - LaProteina Library

This module provides a library-style API for LaProteina protein generation
with integration for BindCraft scoring.
"""

# Light imports (no heavy dependencies)
# Note: config_builder and laproteina_generator have been migrated to the pipeline/ directory.

def __getattr__(name):
    """Lazy import for any remaining core components if needed."""
    # Add future lazy imports here if any core components are added back
    raise AttributeError(f"module 'proteinfoundation' has no attribute '{name}'")


__all__ = [
    # Core components only (generator and utils are now in pipeline/)
]
