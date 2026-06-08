#!/usr/bin/env python3
"""
Template preparation for Protenix inference on internet-restricted cluster nodes.

Workflow (run on a login node with internet access):
  1. Run `protenix mt` to generate MSA + hmmsearch.a3m template hits.
  2. Parse PDB IDs from hmmsearch.a3m.
  3. Download required mmCIF files from PDBe into $PROTENIX_ROOT_DIR/mmcif/.

Usage:
  python scripts/prepare_template.py \
      --input  /nfs/.../template_input.json \
      --output /nfs/.../template_search

After this script completes, the hmmsearch.a3m path to pass as target_template is:
  <output>/target/msa/0/hmmsearch.a3m
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import requests


PDBE_CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}.cif"


def parse_pdb_ids_from_a3m(a3m_path: str) -> list[str]:
    """Extract unique PDB IDs from hmmsearch.a3m header lines (e.g. '>7tdm_A/1-187 ...')."""
    pdb_ids = []
    seen: set[str] = set()
    with open(a3m_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            # Match 4-char PDB ID at start of header, e.g. ">7tdm_A/..." or ">7TDM_A ..."
            m = re.match(r">([0-9A-Za-z]{4})[_\s/]", line)
            if m:
                pdb_id = m.group(1).lower()
                if pdb_id not in seen:
                    seen.add(pdb_id)
                    pdb_ids.append(pdb_id)
    return pdb_ids


def download_cif_files(pdb_ids: list[str], mmcif_dir: str) -> None:
    """Download .cif files from PDBe for each PDB ID, skipping those already cached."""
    mmcif_path = Path(mmcif_dir)
    mmcif_path.mkdir(parents=True, exist_ok=True)

    for pdb_id in pdb_ids:
        cif_path = mmcif_path / f"{pdb_id}.cif"
        if cif_path.exists():
            print(f"  [cached] {pdb_id}.cif")
            continue
        url = PDBE_CIF_URL.format(pdb_id=pdb_id)
        try:
            print(f"  [fetch]  {pdb_id}.cif  ← {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            cif_path.write_text(r.text)
        except Exception as e:
            print(f"  [WARN]   Failed to download {pdb_id}.cif: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  "-i", required=True,  help="Path to template_input.json (protenix mt input)")
    parser.add_argument("--output", "-o", required=True,  help="Output directory for protenix mt results")
    parser.add_argument("--msa_server_mode", "-m", default="protenix", help="MSA server mode (default: protenix)")
    parser.add_argument("--skip_mt", action="store_true", help="Skip protenix mt (use existing hmmsearch.a3m)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run protenix mt
    if not args.skip_mt:
        print(f"\n=== Step 1: protenix mt ===")
        cmd = ["protenix", "mt", "-i", args.input, "-o", str(out_dir), "-m", args.msa_server_mode]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: protenix mt failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    else:
        print("Skipping protenix mt (--skip_mt)")

    # Step 2: Locate hmmsearch.a3m
    candidates = list(out_dir.rglob("hmmsearch.a3m"))
    if not candidates:
        print(f"ERROR: hmmsearch.a3m not found under {out_dir}")
        sys.exit(1)
    hmmsearch_a3m = candidates[0]
    print(f"\n=== Step 2: Parsing template hits from {hmmsearch_a3m} ===")

    pdb_ids = parse_pdb_ids_from_a3m(str(hmmsearch_a3m))
    print(f"Found {len(pdb_ids)} unique PDB IDs: {pdb_ids}")

    # Step 3: Download mmCIF files
    protenix_root = os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))
    mmcif_dir = os.path.join(protenix_root, "mmcif")
    print(f"\n=== Step 3: Pre-downloading mmCIF files to {mmcif_dir} ===")
    download_cif_files(pdb_ids, mmcif_dir)

    print(f"\nDone. Template path for settings JSON:")
    print(f"  {hmmsearch_a3m.absolute()}")


if __name__ == "__main__":
    main()
