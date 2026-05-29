#!/usr/bin/env python3

import os
import argparse
from collections import Counter

import mdtraj as md
from tqdm import tqdm


def find_pdb_files(root_dir):
    """Recursively collect all PDB files from subfolders."""
    pdb_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".pdb"):
                pdb_files.append(os.path.join(dirpath, f))
    return pdb_files


def analyze_secondary_structure(input_dir, chain_id="B", residue_filter=None):

    global_counts = Counter()
    protein_stats = []

    pdb_files = find_pdb_files(input_dir)
    if residue_filter:
        pdb_files = [p for p in pdb_files if residue_filter in p]

    if len(pdb_files) == 0:
        print(f"No PDB files found in: {input_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files (recursive search)")
    print(f"Analyzing only chain: {chain_id}\n")

    for pdb_path in tqdm(pdb_files, desc="Processing PDBs"):

        try:
            traj = md.load(pdb_path)
            topology = traj.topology

            selected_residues = []

            for chain in topology.chains:
                if chain.chain_id == chain_id:
                    for residue in chain.residues:
                        selected_residues.append(residue.index)

            if len(selected_residues) == 0:
                continue

            ss = md.compute_dssp(traj, simplified=True)[0]

            counts = Counter()

            for residue_index in selected_residues:

                state = ss[residue_index]

                if state == "H":
                    counts["helix"] += 1
                elif state == "E":
                    counts["sheet"] += 1
                else:
                    counts["loop"] += 1

            total = sum(counts.values())

            if total == 0:
                continue

            protein_stats.append((counts["helix"]/total,
                                  counts["sheet"]/total,
                                  counts["loop"]/total))

            global_counts.update(counts)

        except Exception:
            continue

    # =========================
    # GLOBAL SUMMARY
    # =========================

    total_all = sum(global_counts.values())

    if total_all == 0:
        print("No valid residues processed.")
        return

    print("\n===================================")
    print("GLOBAL DATASET STATISTICS")
    print("===================================")

    print(f"Total residues analyzed: {total_all}")
    print(f"Helix fraction: {global_counts['helix']/total_all:.3f}")
    print(f"Sheet fraction: {global_counts['sheet']/total_all:.3f}")
    print(f"Loop fraction : {global_counts['loop']/total_all:.3f}")

    # =========================
    # MEAN PER-PROTEIN
    # =========================

    mean_helix = sum(x[0] for x in protein_stats) / len(protein_stats)
    mean_sheet = sum(x[1] for x in protein_stats) / len(protein_stats)
    mean_loop = sum(x[2] for x in protein_stats) / len(protein_stats)

    print("\n===================================")
    print("MEAN PER-PROTEIN FRACTIONS")
    print("===================================")

    print(f"Mean helix fraction: {mean_helix:.3f}")
    print(f"Mean sheet fraction: {mean_sheet:.3f}")
    print(f"Mean loop fraction : {mean_loop:.3f}")


def main():

    parser = argparse.ArgumentParser(
        description="Secondary structure analysis (recursive PDB scan)"
    )

    parser.add_argument(
        "pdb_dir",
        help="Root directory containing PDB files in subfolders"
    )

    parser.add_argument(
        "--chain",
        default="B",
        help="Chain ID to analyze (default: B)"
    )
    parser.add_argument(
        "--residue",
        help="Optional hotspot residue substring filter (e.g., 'hs60')"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.pdb_dir):
        print(f"Directory not found: {args.pdb_dir}")
        return

    analyze_secondary_structure(args.pdb_dir, args.chain, args.residue)


if __name__ == "__main__":
    main()