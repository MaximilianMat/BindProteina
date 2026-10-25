"""
Vendored copy of NVIDIA's La-Proteina.

Upstream: https://github.com/NVIDIA-Digital-Bio/la-proteina, commit cde5de3.
Code is Apache-2.0, released weights are under the NVIDIA Open Model License;
both texts are in ``licenses/``. Of the files vendored here, 110 are
byte-identical to upstream, 22 are modified and 11 were added for this thesis.
``PROVENANCE.md`` lists every one of them and what each change does.

This module only makes the vendored tree importable: ``laproteina/`` is put on
``sys.path`` so that La-Proteina's own absolute imports (``import
proteinfoundation...``) resolve as they do upstream. Generation is driven from
``pipeline/laproteina_generator.py``, which imports ``proteinfoundation``
directly; nothing imports this package for its contents.
"""

import os
import sys

# La-Proteina's internal imports are absolute (``import proteinfoundation``),
# so the vendored directory itself has to be importable as a root.
_lib_path = os.path.dirname(os.path.realpath(__file__))
if _lib_path not in sys.path:
    sys.path.insert(0, _lib_path)

__all__ = []
