"""
Tests for SS-stratified champion selection (plan: ADAPTATION — SS-stratified champion).

Exercises the picker logic directly (numpy only) via a lightweight stand-in that mirrors
``ReseedRun._pick_ss_stratified`` / ``_pick_champion``, plus the config parsing/validation.

    python -m pytest tests/test_reseed_selection.py -v
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# --- Picker stand-in: identical body to ReseedRun._pick_ss_stratified / _pick_champion,
#     but free of torch/biotite imports so it runs anywhere. -------------------------------
def _isna(v):
    return v is None or (isinstance(v, float) and v != v)


def pick_ss_stratified(winners, rng, g):
    cells = {}
    for w in winners:
        h, s = w.get("ss_helix_frac"), w.get("ss_sheet_frac")
        if _isna(h) or _isna(s):
            continue
        ch = min(int(min(max(float(h), 0.0), 1.0) * g), g - 1)
        cs = min(int(min(max(float(s), 0.0), 1.0) * g), g - 1)
        cells.setdefault((ch, cs), []).append(w)
    if not cells:
        return None
    keys = list(cells.keys())
    bucket = cells[keys[int(rng.integers(0, len(keys)))]]
    return bucket[int(rng.integers(0, len(bucket)))]


def pick_champion(winners, rng, g, mode):
    if mode == "ss_stratified":
        p = pick_ss_stratified(winners, rng, g)
        if p is not None:
            return p
    return winners[int(rng.integers(0, len(winners)))]


def _make(idx, h, s):
    return {"sample_id": f"w{idx}", "ss_helix_frac": h, "ss_sheet_frac": s}


def test_uniform_over_nonempty_cells_and_rare_beta_upweight():
    # 30 helix winners in one cell + 1 lone beta-heavy winner in its own cell (g=3).
    winners = [_make(i, 0.9, 0.05) for i in range(30)]          # cell (2,0)
    winners.append(_make(99, 0.05, 0.9))                        # cell (0,2)  (lone beta)
    g = 3
    rng = np.random.default_rng(0)
    picks = [pick_champion(winners, rng, g, "ss_stratified")["sample_id"]
             for _ in range(20000)]
    c = Counter(picks)
    beta_prob = c["w99"] / len(picks)
    # 2 non-empty cells -> lone beta picked ~1/2, hugely up from uniform 1/31.
    assert 0.45 < beta_prob < 0.55, beta_prob
    assert beta_prob > 5 * (1 / len(winners))


def test_within_cell_uniform():
    # Two winners share one cell; expect ~50/50 within it (single non-empty cell).
    winners = [_make(0, 0.9, 0.05), _make(1, 0.85, 0.08)]
    rng = np.random.default_rng(1)
    picks = Counter(pick_champion(winners, rng, 3, "ss_stratified")["sample_id"]
                    for _ in range(10000))
    assert 0.45 < picks["w0"] / 10000 < 0.55


def test_fallback_to_uniform_when_no_ss():
    winners = [_make(i, None, float("nan")) for i in range(5)]
    rng = np.random.default_rng(2)
    got = {pick_champion(winners, rng, 3, "ss_stratified")["sample_id"] for _ in range(500)}
    assert got == {f"w{i}" for i in range(5)}          # every winner reachable
    assert pick_ss_stratified(winners, rng, 3) is None


def test_config_parses_and_validates(tmp_path):
    import json
    from pipeline.reseed.config import load_config
    base = {"target": "T", "target_pdb": "/x/t.pdb", "baseline_run_dir": "/x/r",
            "checkpoint": "/x/a.ckpt", "autoencoder_path": "/x/b.ckpt",
            "output_base_dir": "/x/o"}

    p = tmp_path / "ok.json"
    p.write_text(json.dumps({**base, "champion_selection": "ss_stratified",
                             "champion_ss_grid": 4}))
    c = load_config(str(p))
    assert c.champion_selection == "ss_stratified" and c.champion_ss_grid == 4

    d = tmp_path / "default.json"
    d.write_text(json.dumps(base))
    assert load_config(str(d)).champion_selection == "uniform"   # backward-compatible

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**base, "champion_selection": "greedy"}))
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_effective_samples_per_cut():
    from pipeline.reseed.config import effective_samples_per_cut as f
    assert f(20, 20, 20, True) == 20            # all alive -> base
    assert f(20, 20, 10, True) == 40            # half dead -> 2x
    assert f(20, 20, 15, True) == round(20 * 20 / 15) == 27
    assert f(20, 20, 1, True) == 400            # lone survivor gets the full budget
    assert f(20, 20, 10, False) == 20           # disabled -> base
    assert f(20, 20, 0, True) == 20             # no survivors -> base (round ends anyway)
    assert f(20, 20, 25, True) == 20            # n_alive > n_strains -> base


def test_interface_seed_parsing_and_merge():
    from pipeline.reseed import seed_extract as SE
    assert SE.parse_residue_spec("37,38,301-304") == {37, 38, 301, 302, 303, 304}
    assert SE.parse_residue_spec(None) == set() and SE.parse_residue_spec("") == set()
    assert SE._merge_consecutive([5, 6, 7, 10, 12, 13]) == [(5, 7), (10, 10), (12, 13)]
    assert SE._merge_consecutive([]) == []


def test_interface_seed_cuts_only_interface(monkeypatch):
    from pipeline.reseed import seed_extract as SE
    monkeypatch.setattr(SE, "interface_binder_indices",
                        lambda p, h, c: [10, 11, 12, 30, 31, 55])
    cuts = SE.make_interface_seed_cuts(
        "x.pdb", 60, np.random.default_rng(0), hotspot_ids={1}, cutoff=8.0,
        seed_n_res=5, seed_n_res_min=1, n_cuts=5, ss_per_res=["E"] * 60)
    assert len(cuts) == 5
    for cut in cuts:
        idx = cut.seed_res_indices
        assert 1 <= len(idx) <= 5 and len(set(idx)) == len(idx)
        assert all(i in {10, 11, 12, 30, 31, 55} for i in idx)   # interface residues only
        frag_idx = [i for a, b in cut.fragments for i in range(a, b + 1)]
        assert sorted(frag_idx) == sorted(idx)                   # fragments cover the picks
        assert len(cut.seed_res_ss) == len(idx)
    # too few interface residues -> empty (caller skips)
    monkeypatch.setattr(SE, "interface_binder_indices", lambda p, h, c: [])
    assert SE.make_interface_seed_cuts("x", 60, np.random.default_rng(0),
                                       seed_n_res_min=1) == []


def test_seed_selection_config(tmp_path):
    import json
    from pipeline.reseed.config import load_config
    base = {"target": "T", "target_pdb": "/x/t.pdb", "baseline_run_dir": "/x/r",
            "checkpoint": "/x/a.ckpt", "autoencoder_path": "/x/b.ckpt",
            "output_base_dir": "/x/o"}
    p = tmp_path / "i.json"
    p.write_text(json.dumps({**base, "seed_selection": "interface",
                             "seed_interface_hotspots": "37,38,301-304"}))
    c = load_config(str(p))
    assert c.seed_selection == "interface" and c.seed_n_res_min == 1
    d = tmp_path / "d.json"
    d.write_text(json.dumps(base))
    assert load_config(str(d)).seed_selection == "random"        # default
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**base, "seed_selection": "hotspots"}))
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_redistribute_budget_config(tmp_path):
    import json
    from pipeline.reseed.config import load_config
    base = {"target": "T", "target_pdb": "/x/t.pdb", "baseline_run_dir": "/x/r",
            "checkpoint": "/x/a.ckpt", "autoencoder_path": "/x/b.ckpt",
            "output_base_dir": "/x/o"}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({**base, "redistribute_budget": True}))
    assert load_config(str(p)).redistribute_budget is True
    d = tmp_path / "d.json"
    d.write_text(json.dumps(base))
    assert load_config(str(d)).redistribute_budget is False   # default off
