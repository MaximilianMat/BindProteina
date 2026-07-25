"""
Seed extraction — the core of the re-seeding ablation (spec §5).

Given a parent binder complex (chain A = target, chain B = binder), we:

  1. draw five residues from the binder in each of the five fragmentation patterns
     P1..P5 (dispersed -> contiguous), redrawn independently per pattern (§5.2);
  2. build a LaProteina motif: the target plus those five residues, with the parent's
     backbone coordinates and (by default) amino-acid identities inherited (§5.3-5.4);
  3. build a fixed-length contig so the scaffolded total length equals the parent
     length exactly (§5.5).

The transmission of interface geometry works because target and seed live in the *same*
motif chain — LaProteina centres the whole motif to its centroid, so only the seed's
geometry *relative to the target* matters, which is preserved (this is the same
"monomer masquerade" mechanism as ``IterativeOptimizer.generate_mask_and_contig``).

Pure combinatorial helpers (``draw_fragments``, ``assemble_contig``) have no heavy
dependencies and are unit-tested directly; PDB I/O uses biotite lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from pipeline.reseed.config import CUT_PATTERNS


@dataclass
class SeedCut:
    """One fragmentation configuration extracted from a parent (spec §5, §8 fields)."""

    cut_id: str                       # "P1".."P5"
    cut_pattern: str                  # e.g. "3-2"
    fragments: List[Tuple[int, int]]  # binder-local 1-based (start, end) inclusive
    seed_res_indices: List[int]       # binder-local 1-based indices of all 5 residues
    seed_res_ss: List[str] = field(default_factory=list)  # H/E/L per seed residue


# --------------------------------------------------------------------------------------
# Pure combinatorial logic (no biotite / mdtraj) — unit-testable anywhere.
# --------------------------------------------------------------------------------------

def draw_fragments(parent_len: int, lengths: Sequence[int], rng) -> List[Tuple[int, int]]:
    """Place fragments of the given lengths in the binder [1..parent_len].

    Fragments are non-overlapping and separated by >= 1 residue, so they read as distinct
    motif segments (spec §5.1). Their order along the chain is randomised. ``rng`` is a
    ``numpy.random.Generator``. Raises ``ValueError`` if the binder is too short.
    """
    lengths = list(lengths)
    k = len(lengths)
    total_frag = sum(lengths)
    min_between = k - 1  # >= 1 residue between consecutive fragments
    free = parent_len - total_frag - min_between
    if free < 0:
        raise ValueError(
            f"Binder length {parent_len} too short for pattern {lengths} "
            f"(needs >= {total_frag + min_between})"
        )

    # Randomise which length sits where along the chain.
    order = rng.permutation(k)
    ordered_lengths = [lengths[i] for i in order]

    # Distribute the `free` slack across k+1 gap slots (before, between*(k-1), after).
    # Between-slots additionally carry their mandatory +1 separation.
    slots = k + 1
    extra = _random_composition(free, slots, rng)
    gaps = list(extra)
    for i in range(1, k):  # add the mandatory separation to the between-slots
        gaps[i] += 1

    fragments: List[Tuple[int, int]] = []
    pos = 1 + gaps[0]
    for i, ln in enumerate(ordered_lengths):
        start = pos
        end = pos + ln - 1
        fragments.append((start, end))
        pos = end + 1 + (gaps[i + 1] if i + 1 < len(gaps) else 0)
    return sorted(fragments)


def _random_composition(total: int, bins: int, rng) -> List[int]:
    """Split ``total`` non-negative units into ``bins`` bins uniformly at random."""
    if bins <= 0:
        return []
    if total <= 0:
        return [0] * bins
    # Stars-and-bars: choose bins-1 cut points among total+bins-1 positions.
    counts = [0] * bins
    idx = rng.integers(0, bins, size=total)
    for i in idx:
        counts[int(i)] += 1
    return counts


def assemble_contig(target_len: int, parent_len: int,
                    fragments: Sequence[Tuple[int, int]]) -> str:
    """Build a fixed-length contig string for one seed-cut.

    Target is a fixed motif ``A1-{target_len}``. The binder region [1..parent_len] is
    emitted as interleaved fixed integer gaps (scaffolded) and fixed motif segments
    ``A{gid_start}-{gid_end}`` (seed residues, referenced by their global res-id
    ``target_len + local_index``). Total length is pinned to ``target_len + parent_len``.
    """
    parts = [f"A1-{target_len}"]
    pos = 1
    for (start, end) in sorted(fragments):
        gap = start - pos
        if gap > 0:
            parts.append(str(gap))
        parts.append(f"A{target_len + start}-{target_len + end}")
        pos = end + 1
    trailing = parent_len - pos + 1
    if trailing > 0:
        parts.append(str(trailing))
    return "/".join(parts)


def contig_total_length(contig: str) -> int:
    """Total residues encoded by a contig (motif residues + scaffold gaps)."""
    total = 0
    for part in contig.split("/"):
        part = part.strip()
        if not part:
            continue
        if part[0].isalpha():
            # Motif segment like "A1-58" or "A205-205".
            rng = part[1:]
            if "-" in rng:
                a, b = rng.split("-")
                total += int(b) - int(a) + 1
            else:
                total += 1
        elif "-" in part:
            # A sampled range "40-80" — not used here, count the max.
            total += int(part.split("-")[1])
        else:
            total += int(part)
    return total


# --------------------------------------------------------------------------------------
# Seed-cut construction (SS labelling needs mdtraj, passed in by the caller).
# --------------------------------------------------------------------------------------

def make_seed_cuts(parent_len: int, rng, seed_n_res: int = 5,
                   ss_per_res: Optional[Sequence[str]] = None,
                   patterns: Optional[dict] = None) -> List[SeedCut]:
    """Produce the five seed-cuts for a parent (residues redrawn independently per cut).

    ``ss_per_res`` (optional) is the parent binder's per-residue SS (H/E/L) in binder
    order, used to label each seed residue for RQ2b.
    """
    patterns = patterns or CUT_PATTERNS
    cuts: List[SeedCut] = []
    for cut_id, lengths in patterns.items():
        if sum(lengths) != seed_n_res:
            raise ValueError(f"Pattern {cut_id}={lengths} does not sum to {seed_n_res}")
        fragments = draw_fragments(parent_len, lengths, rng)
        indices = [i for (a, b) in fragments for i in range(a, b + 1)]
        seed_ss = []
        if ss_per_res is not None:
            seed_ss = [ss_per_res[i - 1] if 0 <= i - 1 < len(ss_per_res) else "?"
                       for i in indices]
        cuts.append(SeedCut(
            cut_id=cut_id,
            cut_pattern="-".join(str(x) for x in lengths),
            fragments=fragments,
            seed_res_indices=indices,
            seed_res_ss=seed_ss,
        ))
    return cuts


# --------------------------------------------------------------------------------------
# Interface-restricted seed selection (mode "interface").
# --------------------------------------------------------------------------------------

def parse_residue_spec(spec) -> set:
    """Parse '37,38,301-304' -> {37,38,301,302,303,304}. None/'' -> empty set."""
    out = set()
    if not spec:
        return out
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def interface_binder_indices(parent_pdb: str, hotspot_ids=None,
                             cutoff: float = 8.0) -> List[int]:
    """Binder (chain B) residues within ``cutoff`` Å of the target hotspots.

    ``hotspot_ids`` are target chain-A residue ids (matched against the parent complex's
    chain-A numbering; None/empty -> the whole target = general interface). Returns
    binder-local 1-based indices (chain-B order), matching the seed-cut convention.
    """
    import numpy as np
    import biotite.structure.io as strucio

    structure = strucio.load_structure(parent_pdb)
    a = structure[structure.chain_id == "A"]
    b = structure[structure.chain_id == "B"]
    if a.array_length() == 0 or b.array_length() == 0:
        return []
    if hotspot_ids:
        a = a[np.isin(a.res_id, list(hotspot_ids))]
        if a.array_length() == 0:      # supplied hotspots not present in this complex
            return []
    a_coord = a.coord                  # (Na, 3)
    b_res_ids = _ordered_unique(b.res_id)
    b_coord = b.coord
    b_resid_arr = b.res_id
    cand = []
    for local_idx, rid in enumerate(b_res_ids, start=1):
        res_coord = b_coord[b_resid_arr == rid]      # (Nr, 3)
        d2 = ((res_coord[:, None, :] - a_coord[None, :, :]) ** 2).sum(-1)
        if float(np.sqrt(d2.min())) <= cutoff:
            cand.append(local_idx)
    return cand


def _merge_consecutive(indices) -> List[Tuple[int, int]]:
    """Sorted unique residue indices -> (start,end) fragments, merging runs."""
    idx = sorted(set(int(i) for i in indices))
    if not idx:
        return []
    frags = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            frags.append((start, prev))
            start = prev = i
    frags.append((start, prev))
    return frags


def make_interface_seed_cuts(parent_pdb: str, parent_len: int, rng,
                             hotspot_ids=None, cutoff: float = 8.0,
                             seed_n_res: int = 5, seed_n_res_min: int = 1,
                             n_cuts: int = 5,
                             ss_per_res: Optional[Sequence[str]] = None) -> List[SeedCut]:
    """Seed-cuts drawn only from interface residues (mode 'interface').

    Computes the binder interface residues once, then for each of ``n_cuts`` draws up to
    ``seed_n_res`` of them at random and merges consecutive picks into fragments. Returns
    ``[]`` if fewer than ``seed_n_res_min`` interface residues exist (caller then skips /
    may fall back to random).
    """
    cand = list(interface_binder_indices(parent_pdb, hotspot_ids, cutoff))
    if len(cand) < max(1, seed_n_res_min):
        return []
    cuts: List[SeedCut] = []
    for c in range(n_cuts):
        k = min(seed_n_res, len(cand))
        pick = sorted(rng.choice(len(cand), size=k, replace=False))
        indices = sorted(cand[i] for i in pick)
        fragments = _merge_consecutive(indices)
        seed_ss = ([ss_per_res[i - 1] if 0 <= i - 1 < len(ss_per_res) else "?"
                    for i in indices] if ss_per_res is not None else [])
        cuts.append(SeedCut(
            cut_id=f"I{c + 1}",
            cut_pattern="-".join(str(b - a + 1) for a, b in fragments),
            fragments=fragments,
            seed_res_indices=indices,
            seed_res_ss=seed_ss,
        ))
    return cuts


# --------------------------------------------------------------------------------------
# biotite PDB I/O (lazy import).
# --------------------------------------------------------------------------------------

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def _load_and_stitch(parent_pdb: str, target_len_hint: int):
    """Load a complex PDB and stitch chain B onto chain A (single-chain motif frame).

    Returns ``(structure, target_len, parent_len)`` where ``structure`` is a biotite
    AtomArray in one chain "A": target residues renumbered 1..target_len, binder
    residues renumbered target_len+1..target_len+parent_len. Mirrors the "monomer
    masquerade" stitch of ``IterativeOptimizer.generate_mask_and_contig``.
    """
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io as strucio

    structure = strucio.load_structure(parent_pdb)

    chains = np.unique(structure.chain_id)
    a_mask = structure.chain_id == "A"
    b_mask = structure.chain_id == "B"
    if not np.any(a_mask):
        raise ValueError(f"{parent_pdb}: no chain A (target) found")

    chain_a = structure[a_mask]
    # Renumber target A residues to 1..nA in order.
    a_res_ids = _ordered_unique(chain_a.res_id)
    a_remap = {old: i + 1 for i, old in enumerate(a_res_ids)}
    chain_a = chain_a.copy()
    chain_a.res_id = np.array([a_remap[r] for r in chain_a.res_id])
    chain_a.chain_id[:] = "A"
    target_len = len(a_res_ids)

    if "B" in chains and np.any(b_mask):
        chain_b = structure[b_mask].copy()
        b_res_ids = _ordered_unique(chain_b.res_id)
        b_remap = {old: target_len + i + 1 for i, old in enumerate(b_res_ids)}
        chain_b.res_id = np.array([b_remap[r] for r in chain_b.res_id])
        chain_b.chain_id[:] = "A"
        parent_len = len(b_res_ids)
        stitched = chain_a + chain_b
    else:
        parent_len = 0
        stitched = chain_a

    return stitched, target_len, parent_len


def _ordered_unique(arr) -> list:
    seen = set()
    out = []
    for v in arr:
        v = int(v)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def chain_sequence(pdb_path: str, chain_id: str = "B") -> str:
    """One-letter sequence of a chain (used for seqid-to-parent)."""
    import numpy as np
    import biotite.structure as struc
    import biotite.structure.io as strucio

    structure = strucio.load_structure(pdb_path)
    mask = structure.chain_id == chain_id
    if not np.any(mask):
        return ""
    chain = structure[mask]
    _, res_names = struc.get_residues(chain)
    return "".join(_THREE_TO_ONE.get(str(n), "X") for n in res_names)


def build_contig_and_motif_pdb(parent_pdb: str, target_len: int, seed_cut: SeedCut,
                               out_dir: str, free_sequence: bool = False):
    """Build the contig string and write the motif PDB for one seed-cut.

    Returns ``(contig_string, motif_pdb_path, parent_len)``. The motif PDB holds the
    target (residues 1..target_len) plus the five seed residues (global res-ids
    target_len + local_index), carrying inherited backbone coords and — unless
    ``free_sequence`` — inherited amino-acid identities.
    """
    import numpy as np
    import biotite.structure.io as strucio

    stitched, target_len_eff, parent_len = _load_and_stitch(parent_pdb, target_len)
    if target_len_eff != target_len:
        from loguru import logger
        logger.warning(
            f"{parent_pdb}: chain-A length {target_len_eff} != configured target_len "
            f"{target_len}; using {target_len_eff}"
        )
        target_len = target_len_eff

    seed_global_ids = [target_len + i for i in seed_cut.seed_res_indices]
    keep_ids = list(range(1, target_len + 1)) + seed_global_ids
    keep_mask = np.isin(stitched.res_id, keep_ids)
    motif = stitched[keep_mask].copy()

    if free_sequence:
        # §11-c knob: keep backbone, drop identity by relabelling seed residues to GLY.
        seed_mask = np.isin(motif.res_id, seed_global_ids)
        motif.res_name[seed_mask] = "GLY"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motif_path = out_dir / f"{Path(parent_pdb).stem}_{seed_cut.cut_id}_motif.pdb"
    strucio.save_structure(str(motif_path), motif)

    contig = assemble_contig(target_len, parent_len, seed_cut.fragments)
    # Sanity: fixed length must equal target + parent.
    assert contig_total_length(contig) == target_len + parent_len, (
        f"contig length {contig_total_length(contig)} != {target_len + parent_len} "
        f"for {seed_cut.cut_id} ({contig})"
    )
    return contig, str(motif_path), parent_len
