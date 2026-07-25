"""
Per-sample metrics that the existing scorer does not produce but the ablation needs:

  - binder-relative secondary-structure fractions (RQ2), and per-residue SS (seed_res_ss),
    via mdtraj DSSP (spec §9.4);
  - RMSD of a progeny binder to its parent (novelty guard, spec §9.3), reusing the
    scoring stack's ``compute_binder_rmsd``;
  - positional sequence identity to the parent (novelty guard).

mdtraj is imported lazily so this module can be imported (for its pure-python helpers)
in environments without the heavy cluster dependencies.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple


def _simplify(state: str) -> str:
    """Map an mdtraj simplified DSSP code to H (helix) / E (sheet) / L (loop)."""
    if state == "H":
        return "H"
    if state == "E":
        return "E"
    return "L"


def _chain_residue_indices(topology, chain_id: str) -> List[int]:
    """Global mdtraj residue indices for a chain, in sequence order."""
    idx = []
    for chain in topology.chains:
        if chain.chain_id == chain_id:
            idx.extend(r.index for r in chain.residues)
    return idx


def ss_per_residue(pdb_path: str, chain_id: str = "B") -> Optional[List[str]]:
    """Per-residue simplified SS (H/E/L) for a chain, in binder-local sequence order.

    Element ``i`` is the SS of the ``i``-th residue (0-based) of ``chain_id``. Returns
    ``None`` on failure or if the chain is absent.
    """
    try:
        import mdtraj as md
    except ImportError:
        return None
    try:
        traj = md.load(pdb_path)
        residue_indices = _chain_residue_indices(traj.topology, chain_id)
        if not residue_indices:
            return None
        ss = md.compute_dssp(traj, simplified=True)[0]
        return [_simplify(ss[i]) for i in residue_indices]
    except Exception:  # noqa: BLE001
        return None


def ss_fractions(pdb_path: str, chain_id: str = "B") -> Optional[Tuple[float, float, float]]:
    """(helix, sheet, loop) fractions for a chain, or ``None`` on failure."""
    per_res = ss_per_residue(pdb_path, chain_id=chain_id)
    if not per_res:
        return None
    counts = Counter(per_res)
    total = len(per_res)
    return (counts["H"] / total, counts["E"] / total, counts["L"] / total)


def rmsd_to_parent(sample_complex_pdb: str, parent_complex_pdb: str,
                   target_chain: str = "A", binder_chain: str = "B") -> float:
    """Binder-CA RMSD of a progeny to its parent after superposing the target chain.

    Thin wrapper over the scoring stack's ``compute_binder_rmsd`` so novelty uses the
    same RMSD definition as self-consistency scoring.
    """
    from pipeline.scoring.rmsd import compute_binder_rmsd
    return compute_binder_rmsd(parent_complex_pdb, sample_complex_pdb,
                               target_chain=target_chain, binder_chain=binder_chain)


def seqid_to_parent(sample_seq: str, parent_seq: str) -> float:
    """Positional sequence identity (fraction) between progeny and parent binder.

    Lengths are equal by construction (total length pinned to the parent), but we
    guard against off-by-one by comparing over the shorter length. Returns ``nan`` if
    either sequence is empty.
    """
    if not sample_seq or not parent_seq:
        return float("nan")
    n = min(len(sample_seq), len(parent_seq))
    if n == 0:
        return float("nan")
    matches = sum(1 for a, b in zip(sample_seq[:n], parent_seq[:n]) if a == b)
    return matches / n
