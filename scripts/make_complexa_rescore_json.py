"""Build a Protenix-style scoring JSON for the top-N Complexa designs by ipSAE_min.

Complexa handoff sheets (e.g. PD-L1-Complexa.xlsx, CD19_Complexa.xlsx) contain only
binder sequences plus Complexa's own AF3 self-consistency metrics -- no PDB
structures. This selects the top N rows by a chosen metric (default:
ipSAE_min_mean, descending -- higher ipSAE means a more confident interface) and
pairs each binder sequence with the target's chain-A sequence (extracted from
target_pdb) to build the JSON input expected by pipeline/score_json.py.

Usage:
    python scripts/make_complexa_rescore_json.py \\
        --xlsx /path/to/PD-L1-Complexa.xlsx \\
        --target_pdb /nfs/homedirs/matscher/targets/pdl1/pdl1.pdb \\
        --out top300_pdl1.json \\
        --top_n 300
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from Bio import PDB
from Bio.PDB.Polypeptide import is_aa

# Local copy of pipeline.scoring.utils.extract_sequence_from_pdb's logic. Not
# imported directly: pipeline.scoring.utils pulls in functions/__init__.py,
# which unconditionally imports pyrosetta -- overkill (and often unavailable)
# for this Bio.PDB-only, no-GPU JSON-prep step.
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def extract_sequence_from_pdb(pdb_path: str, chain_id: str) -> str:
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return "".join(
                    THREE_TO_ONE[r.get_resname()]
                    for r in chain
                    if is_aa(r, standard=True) and r.get_resname() in THREE_TO_ONE
                )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Protenix scoring JSON for the top-N Complexa designs by ipSAE_min."
    )
    parser.add_argument("--xlsx", required=True, help="Complexa handoff .xlsx path")
    parser.add_argument("--target_pdb", required=True, help="Target PDB to pull the chain-A sequence from")
    parser.add_argument("--target_chain", default="A", help="Chain in --target_pdb to extract (default: A)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--top_n", type=int, default=300, help="Number of top designs to select (default: 300)")
    parser.add_argument("--sort_col", default="ipSAE_min_mean", help="Column to rank by (default: ipSAE_min_mean)")
    parser.add_argument("--name_col", default="af_name", help="Column to use as the design name (default: af_name)")
    parser.add_argument("--sequence_col", default="binder_sequence", help="Column with the binder sequence")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx)
    for col in (args.sort_col, args.name_col, args.sequence_col):
        if col not in df.columns:
            raise SystemExit(f"Column {col!r} not found in {args.xlsx}. Available: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[args.sort_col, args.sequence_col])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows missing {args.sort_col!r} or {args.sequence_col!r}")

    top = df.sort_values(args.sort_col, ascending=False).head(args.top_n)
    print(f"Selected top {len(top)} of {before} rows by {args.sort_col} (descending)")
    print(f"  {args.sort_col} range: {top[args.sort_col].min():.4f} - {top[args.sort_col].max():.4f}")

    target_seq = extract_sequence_from_pdb(args.target_pdb, args.target_chain)
    if not target_seq:
        raise SystemExit(f"Could not extract chain {args.target_chain} sequence from {args.target_pdb}")
    print(f"Target chain {args.target_chain} sequence: {len(target_seq)} residues")

    entries = []
    for _, row in top.iterrows():
        binder_seq = str(row[args.sequence_col]).strip().upper()
        entries.append({
            "name": str(row[args.name_col]),
            "covalent_bonds": [],
            "sequences": [
                {"proteinChain": {"count": 1, "sequence": target_seq, "modifications": []}},
                {"proteinChain": {"count": 1, "sequence": binder_seq, "modifications": []}},
            ],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(entries, fh, indent=2)
    print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
