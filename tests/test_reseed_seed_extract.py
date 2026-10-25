"""
Tests for the re-seeding ablation seed-extraction (spec §5).

Pure combinatorial tests run anywhere (numpy only). The biotite-dependent test that
builds a real motif PDB from a founder complex is skipped when biotite is unavailable
(i.e. runs on the cluster env, and against ``top_binders/*.pdb`` if present).

    python -m pytest tests/test_reseed_seed_extract.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.reseed.config import CUT_PATTERNS
from pipeline.reseed.seed_extract import (
    draw_fragments, assemble_contig, contig_total_length, make_seed_cuts,
)


def test_patterns_sum_to_five():
    for cut_id, lengths in CUT_PATTERNS.items():
        assert sum(lengths) == 5, cut_id


@pytest.mark.parametrize("parent_len", [12, 30, 60, 80])
def test_draw_fragments_valid(parent_len):
    rng = np.random.default_rng(0)
    for cut_id, lengths in CUT_PATTERNS.items():
        for _ in range(200):
            frags = draw_fragments(parent_len, lengths, rng)
            idx = [i for a, b in frags for i in range(a, b + 1)]
            assert len(idx) == 5 and len(set(idx)) == 5           # exactly 5, no overlap
            assert min(idx) >= 1 and max(idx) <= parent_len        # in bounds
            assert sorted(b - a + 1 for a, b in frags) == sorted(lengths)  # right lengths
            for (a1, b1), (a2, b2) in zip(sorted(frags), sorted(frags)[1:]):
                assert a2 - b1 >= 2                                # >= 1 residue apart


def test_contig_fixed_length():
    rng = np.random.default_rng(1)
    target_len, parent_len = 200, 60
    for cut_id, lengths in CUT_PATTERNS.items():
        frags = draw_fragments(parent_len, lengths, rng)
        contig = assemble_contig(target_len, parent_len, frags)
        assert contig_total_length(contig) == target_len + parent_len


def test_draw_fragments_too_short_raises():
    with pytest.raises(ValueError):
        draw_fragments(8, [1, 1, 1, 1, 1], np.random.default_rng(0))  # needs >= 9


def test_make_seed_cuts_labels_ss():
    ss = ["H"] * 20 + ["E"] * 20 + ["L"] * 20
    cuts = make_seed_cuts(60, np.random.default_rng(2), ss_per_res=ss)
    assert len(cuts) == 5
    for c in cuts:
        assert len(c.seed_res_indices) == 5
        assert len(c.seed_res_ss) == 5
        assert all(s in "HEL" for s in c.seed_res_ss)


def test_build_motif_pdb_on_founder():
    biotite = pytest.importorskip("biotite.structure.io")
    from pipeline.config_builder import get_pdb_chain_length
    from pipeline.reseed.seed_extract import build_contig_and_motif_pdb, make_seed_cuts

    founders = sorted((REPO / "top_binders").glob("*_relaxed.pdb"))
    if not founders:
        pytest.skip("no top_binders/*.pdb available")
    parent = str(founders[0])
    target_len = get_pdb_chain_length(parent, "A")

    import biotite.structure.io as strucio
    _, plen = None, None
    from pipeline.reseed.seed_extract import _load_and_stitch
    _, tlen, plen = _load_and_stitch(parent, target_len)

    cuts = make_seed_cuts(plen, np.random.default_rng(0))
    import tempfile
    out = tempfile.mkdtemp()
    for cut in cuts:
        contig, motif_pdb, parent_len = build_contig_and_motif_pdb(
            parent, target_len, cut, out)
        assert parent_len == plen
        assert contig_total_length(contig) == tlen + plen
        motif = strucio.load_structure(motif_pdb)
        n_res = len(np.unique(motif.res_id))
        assert n_res == tlen + 5, (cut.cut_id, n_res, tlen)
