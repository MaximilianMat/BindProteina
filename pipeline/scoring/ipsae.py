"""
ipSAE interface scoring, adapted for the BindProteina scoring pipeline.

The ipSAE score (Dunbrack lab, https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1)
is a PAE-derived measure of protein-protein interface confidence.  This module is a
self-contained port of the *core* algorithm from the reference implementation

    Roland Dunbrack, ipsae.py, version 3 (April 2025), MIT license
    https://github.com/DunbrackLab/IPSAE  (or the copy the user maintains locally)

The reference is a standalone CLI that writes .txt/.pml side-files and iterates over
every chain pair.  Here we expose a single pure function that computes the bundle for
the target<->binder interface (target = chain A, binder = the remaining chain(s)),
returns a flat dict of scalars, and never touches disk.

Definitions kept identical to the reference:
  * ptm_func(x, d0) = 1 / (1 + (x/d0)^2)
  * calc_d0(L)      = max(1.0, 1.24*(L-15)^(1/3) - 1.8)   (Yang & Skolnick 2004)
  * ipSAE (d0res)   = per aligned-residue mean of ptm over interchain partners with
                      PAE < pae_cutoff, using a per-residue d0; the chain-pair value is
                      the max over aligned residues (asym), then max/min over the two
                      directions.

For the usual two-chain (A target, B binder) case the numbers match the reference
`ipSAE_d0res` "max" (our ``ipsae``) and the min over the two "asym" rows
(our ``ipsae_min``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


# Columns this module always emits (so the CSV schema is stable even when PAE is
# unavailable or malformed — values are then None, except the cutoffs).
BUNDLE_KEYS = (
    "ipsae",            # ipSAE d0res, max over the two chain directions (primary)
    "ipsae_min",        # ipSAE d0res, min over the two directions (meta-analysis discriminator)
    "ipsae_d0chn",      # ipSAE with d0 from total chain-pair length (max)
    "ipsae_d0dom",      # ipSAE with d0 from interface-residue count (max)
    "iptm_pae",         # PAE-derived ipTM (no PAE cutoff), i.e. "pTM-from-PAE" (max)
    "pdockq",
    "pdockq2",
    "lis",
    "ipsae_pae_cutoff",
    "ipsae_dist_cutoff",
)


def _ptm_func(x: np.ndarray, d0: float) -> np.ndarray:
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def _calc_d0(L: float, pair_type: str = "protein") -> float:
    L = float(L)
    if L < 27:
        L = 27.0
    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
    return max(min_value, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def _calc_d0_array(L: np.ndarray, pair_type: str = "protein") -> np.ndarray:
    L = np.array(L, dtype=float)
    L = np.maximum(27.0, L)
    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
    return np.maximum(min_value, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def _null_bundle(pae_cutoff: float, dist_cutoff: float) -> Dict[str, Optional[float]]:
    bundle: Dict[str, Optional[float]] = {k: None for k in BUNDLE_KEYS}
    bundle["ipsae_pae_cutoff"] = float(pae_cutoff)
    bundle["ipsae_dist_cutoff"] = float(dist_cutoff)
    return bundle


def _parse_pdb_ca_cb(pdb_path: str):
    """
    Read one CA and one CB (CA for GLY) per residue from a PDB file, in file order.

    Returns (chains, resnums, ca_coords, cb_coords) as parallel arrays.  Only the
    first model is read; alt-loc is ignored (first occurrence wins per residue).
    """
    chains: List[str] = []
    resnums: List[int] = []
    ca_coords: List[np.ndarray] = []
    cb_coords: List[np.ndarray] = []
    seen_ca = set()
    seen_cb = set()

    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip() or " "
            try:
                res_seq = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            key = (chain_id, res_seq, line[26])  # include insertion code

            if atom_name == "CA" and key not in seen_ca:
                seen_ca.add(key)
                chains.append(chain_id)
                resnums.append(res_seq)
                ca_coords.append(np.array([x, y, z]))
            is_cb = atom_name == "CB" or (res_name == "GLY" and atom_name == "CA")
            if is_cb and key not in seen_cb:
                seen_cb.add(key)
                cb_coords.append(np.array([x, y, z]))

    return (
        np.array(chains),
        np.array(resnums, dtype=int),
        np.array(ca_coords) if ca_coords else np.zeros((0, 3)),
        np.array(cb_coords) if cb_coords else np.zeros((0, 3)),
    )


def _directed_ipsae(pae, align_mask, scored_mask, pae_cutoff):
    """
    Compute the asymmetric (align-chain -> scored-chain) ipSAE/ipTM values.

    Mirrors the reference by-residue loop: for each aligned residue, average the
    ptm-transformed PAE over interchain partners (with PAE < cutoff for ipSAE), then
    take the max over aligned residues.  Returns the four asym scalars.
    """
    align_idx = np.where(align_mask)[0]
    if align_idx.size == 0 or not scored_mask.any():
        return {"iptm_d0chn": 0.0, "ipsae_d0chn": 0.0, "ipsae_d0dom": 0.0, "ipsae_d0res": 0.0}

    n0chn = int(align_mask.sum() + scored_mask.sum())
    d0chn = _calc_d0(n0chn)
    ptm_d0chn = _ptm_func(pae, d0chn)

    valid_matrix = scored_mask[None, :] & (pae < pae_cutoff)  # (N, N)

    # First pass over d0chn + collect unique interface residues for n0dom
    iptm_byres = np.zeros(pae.shape[0])
    ipsae_d0chn_byres = np.zeros(pae.shape[0])
    uniq_align = set()
    uniq_scored = set()
    for i in align_idx:
        iptm_byres[i] = ptm_d0chn[i, scored_mask].mean()
        vp = valid_matrix[i]
        if vp.any():
            ipsae_d0chn_byres[i] = ptm_d0chn[i, vp].mean()
            uniq_align.add(int(i))
            uniq_scored.update(np.where(vp)[0].tolist())

    n0dom = len(uniq_align) + len(uniq_scored)
    d0dom = _calc_d0(n0dom)
    ptm_d0dom = _ptm_func(pae, d0dom)

    n0res_byres = valid_matrix.sum(axis=1)
    d0res_byres = _calc_d0_array(n0res_byres)

    ipsae_d0dom_byres = np.zeros(pae.shape[0])
    ipsae_d0res_byres = np.zeros(pae.shape[0])
    for i in align_idx:
        vp = valid_matrix[i]
        if vp.any():
            ipsae_d0dom_byres[i] = ptm_d0dom[i, vp].mean()
            ptm_row = _ptm_func(pae[i], d0res_byres[i])
            ipsae_d0res_byres[i] = ptm_row[vp].mean()

    return {
        "iptm_d0chn": float(iptm_byres[align_idx].max()),
        "ipsae_d0chn": float(ipsae_d0chn_byres[align_idx].max()),
        "ipsae_d0dom": float(ipsae_d0dom_byres[align_idx].max()),
        "ipsae_d0res": float(ipsae_d0res_byres[align_idx].max()),
    }


def _pdockq(distances, pae, plddt, align_mask, scored_mask, contact_cutoff=8.0):
    """pDockQ and pDockQ2 for the directed align->scored interface (reference formulas)."""
    align_idx = np.where(align_mask)[0]
    interface_residues = set()
    npairs = 0
    ptm_sum = 0.0
    for i in align_idx:
        contacts = scored_mask & (distances[i] <= contact_cutoff)
        n = int(contacts.sum())
        if n:
            npairs += n
            interface_residues.add(int(i))
            interface_residues.update(np.where(contacts)[0].tolist())
            ptm_sum += _ptm_func(pae[i][contacts], 10.0).sum()

    if npairs == 0 or plddt is None or len(interface_residues) == 0:
        return 0.0, 0.0

    idx = list(interface_residues)
    mean_plddt = float(np.asarray(plddt)[idx].mean())
    x = mean_plddt * math.log10(npairs)
    pdockq = 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018

    mean_ptm = ptm_sum / npairs
    x2 = mean_plddt * mean_ptm
    pdockq2 = 1.31 / (1.0 + math.exp(-0.075 * (x2 - 84.733))) + 0.005
    return float(pdockq), float(pdockq2)


def _lis(pae, align_mask, scored_mask, thresh=12.0):
    """Local Interaction Score for the directed align->scored block (reference formula)."""
    block = pae[np.ix_(np.where(align_mask)[0], np.where(scored_mask)[0])]
    if block.size == 0:
        return 0.0
    valid = block[block <= thresh]
    if valid.size == 0:
        return 0.0
    return float(((thresh - valid) / thresh).mean())


def compute_ipsae_bundle(
    pae_matrix: np.ndarray,
    pdb_path: str,
    plddt: Optional[np.ndarray] = None,
    target_chains: str = "A",
    binder_chains: str = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> Dict[str, Optional[float]]:
    """
    Compute the ipSAE bundle for the target<->binder interface.

    Args:
        pae_matrix:    (n_token, n_token) PAE array; token order must match the CA
                       order in ``pdb_path`` (true for protein-only Protenix output).
        pdb_path:      refolded complex PDB (chain ``target_chains`` = target,
                       ``binder_chains`` = binder; multi-char binders like "BC" ok).
        plddt:         optional per-residue pLDDT (any 0-1 or 0-100 scale) for
                       pDockQ/pDockQ2; if None those are returned as 0.
        pae_cutoff/dist_cutoff: reference defaults 10/10.  (dist_cutoff does not affect
                       ipSAE itself; kept for provenance.)

    Returns a dict with the keys in ``BUNDLE_KEYS``.  On any malformed input (missing
    chains, PAE shape mismatch) returns a null bundle rather than raising.
    """
    bundle = _null_bundle(pae_cutoff, dist_cutoff)
    try:
        pae = np.asarray(pae_matrix, dtype=float)
        chains, _resnums, _ca, cb = _parse_pdb_ca_cb(pdb_path)
        numres = len(chains)

        if numres == 0 or pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
            return bundle
        # Token/residue alignment guard: PAE must be residue-sized.
        if pae.shape[0] != numres:
            return bundle

        target_mask = np.isin(chains, list(target_chains))
        binder_mask = np.isin(chains, list(binder_chains))
        if not target_mask.any() or not binder_mask.any():
            return bundle

        # pLDDT normalised to 0-100 for pDockQ formulas.
        plddt_arr = None
        if plddt is not None:
            plddt_arr = np.asarray(plddt, dtype=float).ravel()
            if plddt_arr.shape[0] != numres:
                plddt_arr = None
            elif plddt_arr.size and plddt_arr.max() <= 1.0:
                plddt_arr = plddt_arr * 100.0

        # CB distance matrix (falls back to CA count mismatch -> skip pDockQ distances).
        if cb.shape[0] == numres:
            distances = np.sqrt(((cb[:, None, :] - cb[None, :, :]) ** 2).sum(axis=2))
        else:
            distances = None

        # Two directions: target->binder and binder->target.
        d_tb = _directed_ipsae(pae, target_mask, binder_mask, pae_cutoff)
        d_bt = _directed_ipsae(pae, binder_mask, target_mask, pae_cutoff)

        bundle["ipsae"] = round(max(d_tb["ipsae_d0res"], d_bt["ipsae_d0res"]), 6)
        bundle["ipsae_min"] = round(min(d_tb["ipsae_d0res"], d_bt["ipsae_d0res"]), 6)
        bundle["ipsae_d0chn"] = round(max(d_tb["ipsae_d0chn"], d_bt["ipsae_d0chn"]), 6)
        bundle["ipsae_d0dom"] = round(max(d_tb["ipsae_d0dom"], d_bt["ipsae_d0dom"]), 6)
        bundle["iptm_pae"] = round(max(d_tb["iptm_d0chn"], d_bt["iptm_d0chn"]), 6)

        if distances is not None:
            p_tb = _pdockq(distances, pae, plddt_arr, target_mask, binder_mask)
            p_bt = _pdockq(distances, pae, plddt_arr, binder_mask, target_mask)
            bundle["pdockq"] = round(max(p_tb[0], p_bt[0]), 4)
            bundle["pdockq2"] = round(max(p_tb[1], p_bt[1]), 4)
        lis_tb = _lis(pae, target_mask, binder_mask)
        lis_bt = _lis(pae, binder_mask, target_mask)
        bundle["lis"] = round((lis_tb + lis_bt) / 2.0, 4)

        return bundle
    except Exception:
        return _null_bundle(pae_cutoff, dist_cutoff)
