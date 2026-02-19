#!/usr/bin/env python3
"""
Quick structural sanity checks for binder-generation PDB outputs.

Usage:
  python script_utils/analyze_binder_outputs.py \
    --label finetuned --root ./inference/inference_motif_idx_aa_binder_TARGET_BINDER_AA

  python script_utils/analyze_binder_outputs.py \
    --label baseline --root ./inference/benchmark_ld4_pretrained \
    --label finetuned --root ./inference/benchmark_ld4_finetuned
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ChainStats:
    n_res: int
    min_res: int
    max_res: int
    index_breaks: int
    ca_large_jumps: int
    ca_mean: float
    ca_max: float
    bad_intra_backbone_bonds: int
    bad_peptide_bonds: int
    max_intra_backbone_bond: float
    max_peptide_bond: float


@dataclass
class SampleStats:
    path: Path
    chains: Dict[str, ChainStats]
    has_chain_a: bool
    has_chain_b: bool
    has_residue_zero: bool


def parse_atom_line(line: str) -> Tuple[str, str, int, Tuple[float, float, float]]:
    atom_name = line[12:16].strip()
    chain_id = (line[21] or " ").strip() or "?"
    res_id = int(line[22:26])
    xyz = (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    )
    return atom_name, chain_id, res_id, xyz


def analyze_pdb(path: Path, ca_jump_threshold: float) -> SampleStats:
    residues_by_chain: Dict[str, set] = {}
    ca_by_res: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
    atoms_by_res: Dict[Tuple[str, int], Dict[str, Tuple[float, float, float]]] = {}

    # Loose thresholds that still catch obviously broken backbone geometry.
    # Distances are in angstrom.
    intra_backbone_cutoff = 2.2  # for N-CA, CA-C, C-O
    peptide_bond_cutoff = 2.1  # for C(i)-N(i+1)

    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name, chain_id, res_id, xyz = parse_atom_line(line)
        residues_by_chain.setdefault(chain_id, set()).add(res_id)
        atoms_by_res.setdefault((chain_id, res_id), {})[atom_name] = xyz
        if atom_name == "CA":
            ca_by_res[(chain_id, res_id)] = xyz

    chains: Dict[str, ChainStats] = {}
    has_residue_zero = False
    for chain_id, residues in residues_by_chain.items():
        ordered = sorted(residues)
        has_residue_zero = has_residue_zero or (ordered and ordered[0] == 0)
        index_breaks = sum(
            1 for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1] + 1
        )

        dists: List[float] = []
        bad_intra_backbone_bonds = 0
        bad_peptide_bonds = 0
        max_intra_backbone_bond = float("nan")
        max_peptide_bond = float("nan")

        for i in range(1, len(ordered)):
            prev_xyz = ca_by_res.get((chain_id, ordered[i - 1]))
            cur_xyz = ca_by_res.get((chain_id, ordered[i]))
            if prev_xyz is None or cur_xyz is None:
                continue
            dists.append(math.dist(prev_xyz, cur_xyz))
        ca_large_jumps = sum(1 for d in dists if d > ca_jump_threshold)
        ca_mean = float(sum(dists) / len(dists)) if dists else float("nan")
        ca_max = float(max(dists)) if dists else float("nan")

        for i, res_id in enumerate(ordered):
            cur_atoms = atoms_by_res.get((chain_id, res_id), {})
            # Intra-residue backbone bonds
            for a1, a2 in (("N", "CA"), ("CA", "C"), ("C", "O")):
                if a1 in cur_atoms and a2 in cur_atoms:
                    d = math.dist(cur_atoms[a1], cur_atoms[a2])
                    if math.isnan(max_intra_backbone_bond):
                        max_intra_backbone_bond = d
                    else:
                        max_intra_backbone_bond = max(max_intra_backbone_bond, d)
                    if d > intra_backbone_cutoff:
                        bad_intra_backbone_bonds += 1

            # Peptide C(i)-N(i+1)
            if i + 1 < len(ordered):
                next_atoms = atoms_by_res.get((chain_id, ordered[i + 1]), {})
                if "C" in cur_atoms and "N" in next_atoms:
                    d = math.dist(cur_atoms["C"], next_atoms["N"])
                    if math.isnan(max_peptide_bond):
                        max_peptide_bond = d
                    else:
                        max_peptide_bond = max(max_peptide_bond, d)
                    if d > peptide_bond_cutoff:
                        bad_peptide_bonds += 1

        chains[chain_id] = ChainStats(
            n_res=len(ordered),
            min_res=ordered[0],
            max_res=ordered[-1],
            index_breaks=index_breaks,
            ca_large_jumps=ca_large_jumps,
            ca_mean=ca_mean,
            ca_max=ca_max,
            bad_intra_backbone_bonds=bad_intra_backbone_bonds,
            bad_peptide_bonds=bad_peptide_bonds,
            max_intra_backbone_bond=max_intra_backbone_bond,
            max_peptide_bond=max_peptide_bond,
        )

    return SampleStats(
        path=path,
        chains=chains,
        has_chain_a="A" in chains,
        has_chain_b="B" in chains,
        has_residue_zero=has_residue_zero,
    )


def summarize(label: str, root: Path, ca_jump_threshold: float) -> None:
    pdb_paths = sorted(
        p for p in root.rglob("*.pdb") if not p.name.startswith("._")
    )
    print(f"\n=== {label} ===")
    print(f"root: {root}")
    print(f"samples: {len(pdb_paths)}")
    if not pdb_paths:
        return

    samples = [analyze_pdb(p, ca_jump_threshold) for p in pdb_paths]
    with_chain_b = sum(1 for s in samples if s.has_chain_b)
    residue_zero = sum(1 for s in samples if s.has_residue_zero)
    print(f"with_chain_B: {with_chain_b}/{len(samples)}")
    print(f"residue_index_zero_present: {residue_zero}/{len(samples)}")

    # Aggregate chain-A jump diagnostics (most relevant for target validity)
    a_jump_counts = [s.chains["A"].ca_large_jumps for s in samples if "A" in s.chains]
    a_jump_max = [s.chains["A"].ca_max for s in samples if "A" in s.chains]
    a_bad_intra = [s.chains["A"].bad_intra_backbone_bonds for s in samples if "A" in s.chains]
    a_bad_peptide = [s.chains["A"].bad_peptide_bonds for s in samples if "A" in s.chains]
    a_max_intra = [s.chains["A"].max_intra_backbone_bond for s in samples if "A" in s.chains and not math.isnan(s.chains["A"].max_intra_backbone_bond)]
    a_max_peptide = [s.chains["A"].max_peptide_bond for s in samples if "A" in s.chains and not math.isnan(s.chains["A"].max_peptide_bond)]
    if a_jump_counts:
        print(
            "chain_A_large_CA_jumps(>{:.1f}A): min={} median={} max={}".format(
                ca_jump_threshold,
                min(a_jump_counts),
                sorted(a_jump_counts)[len(a_jump_counts) // 2],
                max(a_jump_counts),
            )
        )
        print(
            "chain_A_max_CA_distance: min={:.2f} median={:.2f} max={:.2f}".format(
                min(a_jump_max),
                sorted(a_jump_max)[len(a_jump_max) // 2],
                max(a_jump_max),
            )
        )
    if a_bad_intra:
        print(
            "chain_A_bad_intra_backbone_bonds(>2.2A): min={} median={} max={}".format(
                min(a_bad_intra),
                sorted(a_bad_intra)[len(a_bad_intra) // 2],
                max(a_bad_intra),
            )
        )
    if a_bad_peptide:
        print(
            "chain_A_bad_peptide_bonds(>2.1A): min={} median={} max={}".format(
                min(a_bad_peptide),
                sorted(a_bad_peptide)[len(a_bad_peptide) // 2],
                max(a_bad_peptide),
            )
        )
    if a_max_intra:
        print(
            "chain_A_max_intra_backbone_bond: min={:.2f} median={:.2f} max={:.2f}".format(
                min(a_max_intra),
                sorted(a_max_intra)[len(a_max_intra) // 2],
                max(a_max_intra),
            )
        )
    if a_max_peptide:
        print(
            "chain_A_max_peptide_bond: min={:.2f} median={:.2f} max={:.2f}".format(
                min(a_max_peptide),
                sorted(a_max_peptide)[len(a_max_peptide) // 2],
                max(a_max_peptide),
            )
        )

    # Show the first sample as concrete reference.
    ref = samples[0]
    print(f"example: {ref.path}")
    for cid, st in sorted(ref.chains.items()):
        print(
            f"  chain {cid}: nres={st.n_res}, range={st.min_res}-{st.max_res}, "
            f"index_breaks={st.index_breaks}, large_CA_jumps={st.ca_large_jumps}, "
            f"mean_CA={st.ca_mean:.2f}, max_CA={st.ca_max:.2f}, "
            f"bad_intra_backbone={st.bad_intra_backbone_bonds}, "
            f"bad_peptide={st.bad_peptide_bonds}, "
            f"max_intra={st.max_intra_backbone_bond:.2f}, "
            f"max_peptide={st.max_peptide_bond:.2f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for each root (repeat once per --root).",
    )
    p.add_argument(
        "--root",
        action="append",
        required=True,
        help="Root directory containing generated PDBs (repeatable).",
    )
    p.add_argument(
        "--ca_jump_threshold",
        type=float,
        default=4.5,
        help="CA-CA distance threshold to flag local geometry jumps.",
    )
    args = p.parse_args()
    if len(args.label) != len(args.root):
        raise ValueError("Need equal number of --label and --root arguments.")
    return args


def main() -> None:
    args = parse_args()
    for label, root in zip(args.label, args.root):
        summarize(label=label, root=Path(root), ca_jump_threshold=args.ca_jump_threshold)


if __name__ == "__main__":
    main()
