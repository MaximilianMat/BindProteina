# `functions/` — vendored BindCraft

Not my work. This directory is a partial copy of **BindCraft**
(<https://github.com/martinpacesa/BindCraft>, commit `0366085`, MIT,
© 2024 Martin Pacesa). The license text is in `../licenses/LICENSE-bindcraft.txt`.

The files are deliberately left as close to upstream as possible so the diff
stays easy to verify, which is also why unused BindCraft helpers were not
trimmed out of them.

| File | Status vs. upstream |
|---|---|
| `biopython_utils.py` | verbatim |
| `generic_utils.py` | 2 lines changed |
| `pyrosetta_utils.py` | 123 lines changed — interface-analysis entry points adapted to this pipeline's chain convention and threading model |
| `__init__.py` | one star-import line removed (see below) |
| `DAlphaBall.gcc` | upstream binary, unmodified |

Only these pieces are actually reached from this pipeline:

- `pyrosetta_utils.pr_relax`, `score_interface`, `get_pyrosetta_init_cmd`
  — used by `pipeline/scoring/pyrosetta_scorer.py`
- `biopython_utils.three_to_one_map` — used by `pipeline/scoring/utils.py`
- `generic_utils.clean_pdb` — used by `pyrosetta_utils.py`
- `DAlphaBall.gcc` — PyRosetta shape-complementarity calculation

Everything else here is upstream code that this method never calls. Two
leftovers follow from that, both intentional:

- `generic_utils.py` still contains BindCraft's CLI helpers
  (`perform_input_check`, `perform_advanced_settings_check`) whose default
  paths point at `settings_advanced/` and `functions/dssp`. Both were removed
  during cleanup for the thesis submission because this method never runs
  BindCraft's trajectory-design protocol. Those code paths are unreachable from
  here, so the dangling defaults are harmless.
- `__init__.py` no longer does `from .colabdesign_utils import *`, since
  `colabdesign_utils.py` was removed for the same reason. The AF2 oracle in
  `pipeline/scoring/folding.py` imports the `colabdesign` library directly and
  never needed BindCraft's wrapper.

See `../PROVENANCE.md` for the full attribution across the repository.
