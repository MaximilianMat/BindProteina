"""
Protenix folding adapter.

Runs Protenix structure prediction via its CLI (`protenix pred`),
then converts the CIF output to PDB and extracts confidence metrics
so that the downstream scoring pipeline sees the same interface as AF2.

MSA workflow
------------
Generate the target MSA once with `generate_target_msa_dir()`, which calls
`protenix msa --msa_server_mode protenix` and returns the directory that
contains `pairing.a3m` + `non_pairing.a3m`.  Pass that directory as
`target_msa_path` to the folding functions; it is wired into the
`pairedMsaPath` / `unpairedMsaPath` fields of the target chain entry.

`--use_msa false` is always passed to `protenix pred` so that no per-sample
MSA search is triggered (the binder is a de novo design with no MSA, and
the target MSA is already embedded in the JSON).
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_protenix_cmd() -> str:
    """Return the protenix binary path, preferring the protenix_env conda env."""
    try:
        conda_base = subprocess.check_output(["conda", "info", "--base"], text=True).strip()
        protenix_bin = os.path.join(conda_base, "envs", "protenix_env", "bin", "protenix")
        if os.path.exists(protenix_bin):
            return protenix_bin
    except Exception:
        pass
    return "protenix"


def _resolve_kalign_cmd() -> Optional[str]:
    """Return the kalign binary path from the same bin/ dir as protenix, or None if not found."""
    protenix = _resolve_protenix_cmd()
    candidate = os.path.join(os.path.dirname(protenix), "kalign")
    if os.path.exists(candidate):
        return candidate
    import shutil
    return shutil.which("kalign")


def _attach_msa_to_chain(chain: dict, target_msa_path: Optional[str]) -> None:
    """
    Attach precomputed MSA to a proteinChain dict in-place.

    Accepts either:
    - a directory containing pairing.a3m + non_pairing.a3m
      → sets pairedMsaPath / unpairedMsaPath directly (Protenix modern format)
    - a single .a3m file (backward-compat)
      → sets BOTH pairedMsaPath and unpairedMsaPath to the same file.
        This prevents Protenix from searching for a paired MSA when --use_msa true is set.
    """
    if not target_msa_path:
        return
    # Skip if paths already exist
    if "pairedMsaPath" in chain and "unpairedMsaPath" in chain:
        return
    p = Path(target_msa_path)
    if p.is_dir():
        paired = p / "pairing.a3m"
        unpaired = p / "non_pairing.a3m"
        if paired.exists():
            chain["pairedMsaPath"] = str(paired)
        if unpaired.exists():
            chain["unpairedMsaPath"] = str(unpaired)
        if not paired.exists() and not unpaired.exists():
            logger.warning(f"MSA directory found but contains no pairing.a3m or non_pairing.a3m: {p}")
    elif p.is_file():
        # Set both to avoid trigger search for paired
        chain["unpairedMsaPath"] = str(p)
        chain["pairedMsaPath"] = str(p)
    else:
        logger.warning(f"Target MSA path provided but NOT FOUND: {p}")


def _write_minimal_binder_a3m(binder_seq: str, out_dir: Path) -> str:
    """
    Write a minimal single-sequence a3m for a de novo binder and return the path.

    Format matches Protenix's own output for sequences with no homologs found:
    two lines — ">query" header, then the bare uppercase sequence.
    The same file is used for both pairedMsaPath and unpairedMsaPath.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    a3m_path = out_dir / "binder.a3m"
    if not a3m_path.exists():
        a3m_path.write_text(f">query\n{binder_seq}\n")
    return str(a3m_path)


def _build_protenix_input_json(
    target_seq: str,
    binder_seq: str,
    job_name: str,
    out_path: Path,
    target_msa_path: Optional[str] = None,
    target_template_path: Optional[str] = None,
    binder_msa_path: Optional[str] = None,
) -> str:
    """Build a Protenix input JSON with two protein chains."""
    target_chain: dict = {"count": 1, "sequence": target_seq, "modifications": []}
    if target_msa_path:
        from pipeline.generate import fix_path
        resolved = Path(fix_path(target_msa_path)).absolute()
        _attach_msa_to_chain(target_chain, str(resolved))
        if "pairedMsaPath" in target_chain or "unpairedMsaPath" in target_chain or "msa" in target_chain:
            logger.info(f"Using pre-computed target MSA: {resolved}")

    if target_template_path:
        tp = Path(target_template_path).absolute()
        if tp.exists():
            target_chain["templatesPath"] = str(tp)
            logger.info(f"Using template: {tp}")
        else:
            logger.warning(f"Template path provided but NOT FOUND: {tp}")

    binder_chain: dict = {"count": 1, "sequence": binder_seq, "modifications": []}
    if binder_msa_path:
        binder_chain["unpairedMsaPath"] = binder_msa_path
        binder_chain["pairedMsaPath"] = binder_msa_path

    payload = [
        {
            "name": job_name,
            "covalent_bonds": [],
            "sequences": [
                {"proteinChain": target_chain},
                {"proteinChain": binder_chain},
            ],
        }
    ]
    json_path = out_path / f"{job_name}_input.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(json_path)


def _cif_to_pdb(cif_path: str, pdb_path: str) -> str:
    """Convert a CIF file to PDB format using biotite."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc

    pdbx_file = pdbx.CIFFile.read(cif_path)
    block = list(pdbx_file.values())[0]
    atoms = pdbx.get_structure(block, model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]

    pdb_file = pdb_io.PDBFile()
    pdb_file.set_structure(atoms)
    pdb_file.write(pdb_path)
    return pdb_path


def _pick_best_sample(
    search_dir: Path,
    design_name: Optional[str] = None,
) -> Tuple[Optional[Path], dict]:
    """
    Scan *search_dir* for summary_confidence JSON files, pick the one with the
    highest ranking_score, and return (best_cif_path, confidence_data).

    If *design_name* is given, only files whose name starts with
    ``{design_name}_summary_confidence`` or whose parent dir matches the name
    are considered.
    """
    if design_name:
        conf_files = [
            cf for cf in search_dir.rglob("*_summary_confidence_sample_*.json")
            if cf.name.startswith(f"{design_name}_summary_confidence")
            or cf.parent.name == design_name
        ]
    else:
        conf_files = list(search_dir.rglob("*_summary_confidence_sample_*.json"))

    best_cif: Optional[Path] = None
    best_data: dict = {}

    if conf_files:
        ranked = []
        for cf in conf_files:
            try:
                with open(cf) as fh:
                    data = json.load(fh)
                score = data.get("ranking_score", data.get("iptm", 0.0))
                ranked.append((cf, score, data))
            except Exception:
                continue

        if ranked:
            ranked.sort(key=lambda x: x[1], reverse=True)
            best_conf_file, _, best_data = ranked[0]
            # Derive CIF path: {name}_summary_confidence_sample_N.json → {name}_sample_N.cif
            cif_name = best_conf_file.name.replace("_summary_confidence_", "_").replace(".json", ".cif")
            potential_cif = best_conf_file.parent / cif_name
            if potential_cif.exists():
                best_cif = potential_cif

    if best_cif is None:
        # Fallback to any matching CIF
        if design_name:
            cifs = [
                c for c in search_dir.rglob("*.cif")
                if c.name.startswith(f"{design_name}_sample_") or c.parent.name == design_name
            ]
        else:
            cifs = list(search_dir.rglob("*.cif"))
        if cifs:
            best_cif = cifs[0]

    return best_cif, best_data


def _build_metrics_dict(
    best_data: dict,
    binder_seq: str,
    binder_len: int,
    model_name: str,
) -> dict:
    """Build the standardised metrics dict from a Protenix confidence JSON."""
    plddt = best_data.get("plddt", 0.0)
    if plddt < 1.0:
        plddt *= 100.0

    chain_plddt = best_data.get("chain_plddt") or []
    binder_plddt = (
        sum(chain_plddt[1:]) / len(chain_plddt[1:]) * 100.0
        if len(chain_plddt) > 1 else plddt
    )

    chain_iptm = best_data.get("chain_iptm") or []
    binder_iptm = (
        sum(chain_iptm[1:]) / len(chain_iptm[1:])
        if len(chain_iptm) > 1 else None
    )

    chain_pair_iptm = best_data.get("chain_pair_iptm") or []
    binder_cols = chain_pair_iptm[0][1:] if (len(chain_pair_iptm) > 0 and len(chain_pair_iptm[0]) > 1) else []
    cross_iptm = sum(binder_cols) / len(binder_cols) if binder_cols else None

    return {
        "pLDDT":           plddt,
        "binder_pLDDT":    binder_plddt,
        "pTM":             best_data.get("ptm", 0.0),
        "i_pTM":           best_data.get("iptm", 0.0),
        "ranking_score":   best_data.get("ranking_score", 0.0),
        "gpde":            best_data.get("gpde"),
        "binder_iptm":     binder_iptm,
        "chain_pair_iptm": cross_iptm,
        "has_clash":       best_data.get("has_clash"),
        "disorder":        best_data.get("disorder"),
        "num_recycles":    best_data.get("num_recycles"),
        "pAE":             None,   # scalar PAE not reported by Protenix summary
        "i_pAE":           None,
        "sequence":        binder_seq,
        "binder_length":   binder_len,
        "folding_model":   model_name,
    }


# Module-level ipSAE cutoffs (reference defaults; see pipeline/scoring/ipsae.py).
IPSAE_PAE_CUTOFF = 10.0
IPSAE_DIST_CUTOFF = 10.0


def _load_sample_pae(best_cif: Path):
    """
    Load the full token-level confidence (PAE matrix + token pLDDT) that Protenix
    writes next to the chosen sample's CIF.

    Protenix's ``pred`` output only exposes scalar summaries in
    ``*_summary_confidence_sample_*.json``; the token PAE lives in a sibling
    confidence file whose exact name/format varies by version.  We therefore search
    the sample's directory for a non-``summary`` confidence/pae file matching the
    design prefix + sample index and try the common containers (JSON keys
    ``pae``/``predicted_aligned_error``; NPZ arrays of the same names).  Returns
    ``(pae_matrix, plddt)`` or ``(None, None)`` if nothing usable is found — the
    caller then leaves ipSAE blank rather than failing.
    """
    import re
    import numpy as np

    best_cif = Path(best_cif)
    stem = best_cif.stem                      # {design}_sample_{N}
    m = re.search(r"sample_(\d+)", stem)
    sample_tag = f"sample_{m.group(1)}" if m else ""
    prefix = stem.split("_sample_")[0] if "_sample_" in stem else stem

    def _load(path: Path):
        try:
            if path.suffix == ".npz":
                d = np.load(path)
                for k in ("pae", "predicted_aligned_error", "pae_matrix"):
                    if k in d:
                        plddt = None
                        for pk in ("plddt", "token_plddt", "atom_plddts"):
                            if pk in d:
                                plddt = np.asarray(d[pk]); break
                        return np.asarray(d[k]), plddt
            elif path.suffix == ".json":
                with open(path) as fh:
                    d = json.load(fh)
                for k in ("pae", "predicted_aligned_error"):
                    if k in d:
                        plddt = d.get("plddt") or d.get("token_plddt") or d.get("atom_plddts")
                        return np.asarray(d[k]), (np.asarray(plddt) if plddt is not None else None)
        except Exception:
            return None, None
        return None, None

    def _candidates(files):
        out = []
        for p in files:
            if not p.is_file() or p.suffix not in (".json", ".npz"):
                continue
            name = p.name
            if "summary_confidence" in name:
                continue
            if sample_tag and sample_tag not in name:
                continue
            if prefix and not name.startswith(prefix):
                continue
            if not any(t in name.lower() for t in ("confidence", "pae", "full_data")):
                continue
            out.append(p)
        out.sort(key=lambda p: (p.suffix != ".json", len(p.name)))
        return out

    for search in (best_cif.parent.glob("*"), best_cif.parent.rglob("*")):
        for c in _candidates(search):
            pae, plddt = _load(c)
            if pae is not None:
                logger.info(f"ipSAE: loaded PAE from {c.name} (shape {getattr(pae, 'shape', None)})")
                return pae, plddt
    return None, None


def _augment_metrics_with_ipsae(metrics: dict, best_cif, pdb_path: str) -> dict:
    """
    Compute the ipSAE bundle for one design and merge it into ``metrics`` (in place).

    Requires the design's PDB on disk (chain A = target, ``binder_chains`` = binder)
    and a loadable Protenix PAE matrix.  No-op if either is missing; never raises.
    """
    try:
        if not best_cif or not pdb_path or not os.path.exists(pdb_path):
            return metrics
        pae, plddt = _load_sample_pae(Path(best_cif))
        if pae is None:
            return metrics
        from pipeline.scoring.ipsae import compute_ipsae_bundle
        binder_chains = metrics.get("binder_chains", "B") or "B"
        bundle = compute_ipsae_bundle(
            pae, str(pdb_path), plddt=plddt,
            target_chains="A", binder_chains=binder_chains,
            pae_cutoff=IPSAE_PAE_CUTOFF, dist_cutoff=IPSAE_DIST_CUTOFF,
        )
        metrics.update(bundle)
    except Exception as e:
        logger.warning(f"ipSAE computation failed for a design: {e}")
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_protenix_refolding(
    pdb_path: str,
    design_name: str,
    output_dir: Path,
    model_name: str = "protenix-v2",
    seed: int = 42,
    target_msa_path: Optional[str] = None,
    target_template_path: Optional[str] = None,
    n_diffusion_samples: int = 5,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Run Protenix structure prediction on a single complex PDB.

    Returns: (path_to_protenix_pdb, metrics_dict)
    """
    from pipeline.scoring.utils import extract_sequence_from_pdb

    target_seq = extract_sequence_from_pdb(pdb_path, "A")
    binder_seq = extract_sequence_from_pdb(pdb_path, "B")

    if not binder_seq:
        return None, {"error": "No binder sequence found (Chain B missing?)"}
    if not target_seq:
        return None, {"error": "No target sequence found (Chain A missing?)"}

    binder_seq = "".join(c for c in binder_seq.upper() if "A" <= c <= "Z")
    target_seq = "".join(c for c in target_seq.upper() if "A" <= c <= "Z")
    binder_len = len(binder_seq)

    if binder_len == 0:
        return None, {"error": "Empty binder sequence"}

    work_dir = output_dir / f"{design_name}_protenix"
    work_dir.mkdir(parents=True, exist_ok=True)

    binder_msa = _write_minimal_binder_a3m(binder_seq, work_dir / "binder_msa")

    input_json = _build_protenix_input_json(
        target_seq=target_seq,
        binder_seq=binder_seq,
        job_name=design_name,
        out_path=work_dir,
        target_msa_path=target_msa_path,
        target_template_path=target_template_path,
        binder_msa_path=binder_msa,
    )

    protenix_cmd = _resolve_protenix_cmd()
    use_template_flag = "true" if target_template_path else "false"
    cmd = [
        protenix_cmd, "pred",
        "-i", input_json,
        "-o", str(work_dir),
        "-n", model_name,
        "--seeds", str(seed),
        "--sample", str(n_diffusion_samples),
        "--use_msa", "true",       # precomputed MSA in JSON; suppress per-sample MSA search
        "--use_template", use_template_flag,
        "--enable_cache", "true",
    ]
    if target_template_path:
        kalign = _resolve_kalign_cmd()
        if kalign:
            cmd += ["--kalign_binary_path", kalign]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return None, {"error": f"Protenix failed: {result.stderr[:500]}"}
    except subprocess.TimeoutExpired:
        return None, {"error": "Protenix prediction timed out (600s)"}
    except FileNotFoundError:
        return None, {"error": "protenix CLI not found. Is protenix installed?"}

    best_cif, best_data = _pick_best_sample(work_dir)
    if best_cif is None:
        return None, {"error": f"No Protenix output (.cif) found in {work_dir}"}

    protenix_pdb_path = str(work_dir / f"{design_name}_protenix.pdb")
    try:
        _cif_to_pdb(str(best_cif), protenix_pdb_path)
    except Exception:
        pass

    metrics = _build_metrics_dict(best_data, binder_seq, binder_len, model_name)
    _augment_metrics_with_ipsae(metrics, best_cif, protenix_pdb_path)
    return protenix_pdb_path, metrics


def _get_completed_designs(work_dir, payload: list, job_info: dict, model_name: str):
    import json
    
    completed_results = []
    remaining = []
    
    for cp in payload:
        dn = cp["name"]
        pdb_path = work_dir / f"{dn}_protenix.pdb"
        meta_path = work_dir / f"{dn}_protenix_meta.json"
        
        if pdb_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    metrics = json.load(f)
                completed_results.append((dn, str(pdb_path), metrics))
                continue
            except Exception:
                pass
        remaining.append(cp)
        
    if not remaining:
        return completed_results, remaining
        
    chunk_dirs = [d for d in work_dir.glob("chunk_*") if d.is_dir()]
    if chunk_dirs:
        design_to_chunk = {}
        for cdir in chunk_dirs:
            cjson = cdir / "chunk_input.json"
            if cjson.exists():
                try:
                    with open(cjson, "r") as f:
                        for item in json.load(f):
                            design_to_chunk[item["name"]] = cdir
                except Exception:
                    pass
                    
        still_remaining = []
        for cp in remaining:
            dn = cp["name"]
            cdir = design_to_chunk.get(dn)
            found = False
            if cdir:
                jinfo = job_info.get(dn)
                if jinfo and "error" not in jinfo:
                    best_cif, best_data = _pick_best_sample(cdir, dn)
                    if best_cif is not None:
                        pdb_path = work_dir / f"{dn}_protenix.pdb"
                        meta_path = work_dir / f"{dn}_protenix_meta.json"
                        try:
                            _cif_to_pdb(str(best_cif), str(pdb_path))
                        except Exception:
                            pass
                        metrics = _build_metrics_dict(best_data, jinfo["binder_seq"], jinfo["binder_len"], model_name)
                        if "binder_chains" in jinfo:
                            metrics["binder_chains"] = jinfo["binder_chains"]
                        _augment_metrics_with_ipsae(metrics, best_cif, str(pdb_path))
                        with open(meta_path, "w") as f:
                            json.dump(metrics, f)
                        completed_results.append((dn, str(pdb_path), metrics))
                        found = True
            if not found:
                still_remaining.append(cp)
        remaining = still_remaining
        
    return completed_results, remaining


def run_protenix_refolding_batch(
    pdb_paths: List[str],
    output_dir: Path,
    model_name: str = "protenix-v2",
    seed: int = 42,
    target_msa_path: Optional[str] = None,
    target_template_path: Optional[str] = None,
    chunk_size: int = 35,
    fetch_all_msas: bool = False,
    n_diffusion_samples: int = 5,
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """
    Run Protenix structure prediction on multiple complex PDBs using chunked batches.

    Protenix can struggle with very large input JSONs due to preprocessing overhead.
    This function splits work into chunks of *chunk_size* designs and runs a separate
    `protenix pred` subprocess per chunk while showing a live progress bar.

    Returns: List of (design_name, path_to_protenix_pdb, metrics_dict)
    """
    import time
    from tqdm import tqdm
    from pipeline.scoring.utils import extract_sequence_from_pdb

    work_dir = output_dir / "protenix_batch"
    work_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resolve precomputed MSA + template paths once for all designs
    # -----------------------------------------------------------------------
    import hashlib
    msa_cache_dir = work_dir / "msa_cache"
    msa_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_msa_for_seq(seq: str) -> str:
        s_hash = hashlib.md5(seq.encode()).hexdigest()
        s_dir = msa_cache_dir / s_hash
        if not (s_dir / "non_pairing.a3m").exists():
            return generate_target_msa_dir(seq, s_dir)
        cands = list(s_dir.rglob("non_pairing.a3m"))
        return str(cands[0].parent.absolute())
    from pipeline.generate import fix_path

    resolved_msa_path: Optional[str] = None
    if target_msa_path:
        msa_obj = Path(fix_path(target_msa_path)).absolute()
        if msa_obj.exists():
            resolved_msa_path = str(msa_obj)
            logger.info(f"Using pre-computed target MSA: {resolved_msa_path}")
        else:
            logger.warning(f"Target MSA path provided but NOT FOUND: {msa_obj}")

    resolved_template_path: Optional[str] = None
    if target_template_path:
        tpl_obj = Path(fix_path(target_template_path)).absolute()
        if tpl_obj.exists():
            resolved_template_path = str(tpl_obj)
            logger.info(f"Using template: {resolved_template_path}")
        else:
            logger.warning(f"Template path provided but NOT FOUND: {tpl_obj}")

    # -----------------------------------------------------------------------
    # Build full payload (sequence extraction + MSA embedding)
    # -----------------------------------------------------------------------
    binder_msa_dir = work_dir / "binder_msas"
    binder_msa_dir.mkdir(parents=True, exist_ok=True)

    payload: list = []
    job_info: dict = {}

    for pdb_path in pdb_paths:
        design_name = Path(pdb_path).stem
        target_seq = extract_sequence_from_pdb(pdb_path, "A")
        binder_seq = extract_sequence_from_pdb(pdb_path, "B")

        if not binder_seq:
            job_info[design_name] = {"error": "No binder sequence found (Chain B missing?)"}
            continue
        if not target_seq:
            job_info[design_name] = {"error": "No target sequence found (Chain A missing?)"}
            continue

        binder_seq = "".join(c for c in binder_seq.upper() if "A" <= c <= "Z")
        target_seq = "".join(c for c in target_seq.upper() if "A" <= c <= "Z")
        binder_len = len(binder_seq)

        if binder_len == 0:
            job_info[design_name] = {"error": "Empty binder sequence"}
            continue

        # Chain A (Target)
        target_chain: dict = {"count": 1, "sequence": target_seq, "modifications": []}
        # Check if already has MSAs (from input PDB or previous step)
        if "pairedMsaPath" not in target_chain:
            if resolved_msa_path:
                _attach_msa_to_chain(target_chain, resolved_msa_path)
            elif fetch_all_msas:
                _attach_msa_to_chain(target_chain, get_msa_for_seq(target_seq))
        
        if resolved_template_path:
            target_chain["templatesPath"] = resolved_template_path

        # Chain B (Binder)
        binder_chain: dict = {"count": 1, "sequence": binder_seq, "modifications": []}
        if "pairedMsaPath" not in binder_chain:
            if fetch_all_msas:
                _attach_msa_to_chain(binder_chain, get_msa_for_seq(binder_seq))
            else:
                # Default to dummy for de novo binders
                binder_a3m = binder_msa_dir / f"{design_name}_binder.a3m"
                if not binder_a3m.exists():
                    binder_a3m.write_text(f">query\n{binder_seq}\n")
                binder_chain["unpairedMsaPath"] = str(binder_a3m)
                binder_chain["pairedMsaPath"] = str(binder_a3m)

        payload.append({
            "name": design_name,
            "covalent_bonds": [],
            "sequences": [
                {"proteinChain": target_chain},
                {"proteinChain": binder_chain},
            ],
        })
        job_info[design_name] = {"binder_seq": binder_seq, "binder_len": binder_len}

    results: list = []
    if not payload:
        for p in pdb_paths:
            dnn = Path(p).stem
            err = job_info.get(dnn, {}).get("error", "Failed precondition")
            results.append((dnn, None, {"error": err}))
        return results

    with open(work_dir / "batch_input.json", "w") as f:
        json.dump(payload, f, indent=2)

    protenix_cmd = _resolve_protenix_cmd()

    completed_results, missing_payload = _get_completed_designs(work_dir, payload, job_info, model_name)
    results.extend(completed_results)

    if not missing_payload:
        logger.info(f"Protenix batch complete: All {len(payload)} designs already processed.")
        return results

    n_chunks = (len(missing_payload) + chunk_size - 1) // chunk_size
    logger.info(f"Protenix Batch: {len(missing_payload)} missing designs → {n_chunks} chunk(s) of ≤{chunk_size}")

    existing_chunks = [int(d.name.split('_')[1]) for d in work_dir.glob("chunk_*") if d.is_dir() and d.name.split('_')[1].isdigit()]
    next_chunk_idx = max(existing_chunks) + 1 if existing_chunks else 0

    pbar = tqdm(total=len(payload), desc="Protenix Batch Prediction", unit="design")
    pbar.update(len(completed_results))
    global_start = time.time()

    for i in range(n_chunks):
        chunk_idx = next_chunk_idx + i
        chunk_start = i * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(missing_payload))
        chunk_payload = missing_payload[chunk_start:chunk_end]

        chunk_dir = work_dir / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_json = chunk_dir / "chunk_input.json"
        with open(chunk_json, "w") as f:
            json.dump(chunk_payload, f, indent=2)

        cmd = [
            protenix_cmd, "pred",
            "-i", str(chunk_json),
            "-o", str(chunk_dir),
            "-n", model_name,
            "--seeds", str(seed),
            "--sample", str(n_diffusion_samples),
            "--use_msa", "true",
            "--use_template", "true" if resolved_template_path else "false",
            "--enable_cache", "true",
        ]
        if resolved_template_path:
            kalign = _resolve_kalign_cmd()
            if kalign:
                cmd += ["--kalign_binary_path", kalign]

        if i == 0:
            logger.info(f"Protenix command: {' '.join(cmd)}")

        chunk_start_time = time.time()
        
        try:
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                logger.error(f"Chunk {chunk_idx} exited {process.returncode}. Stderr: {process.stderr[-3000:]}")

            for cp in chunk_payload:
                design_name = cp["name"]
                jinfo = job_info.get(design_name)
                
                best_cif, best_data = _pick_best_sample(chunk_dir, design_name)
                if best_cif is None:
                    results.append((design_name, None, {"error": "No output produced or protenix failed"}))
                else:
                    protenix_pdb_path = str(work_dir / f"{design_name}_protenix.pdb")
                    meta_path = work_dir / f"{design_name}_protenix_meta.json"
                    try:
                        _cif_to_pdb(str(best_cif), protenix_pdb_path)
                    except Exception:
                        pass
                    
                    metrics = _build_metrics_dict(best_data, jinfo["binder_seq"], jinfo["binder_len"], model_name)
                    if "binder_chains" in jinfo:
                        metrics["binder_chains"] = jinfo["binder_chains"]
                    _augment_metrics_with_ipsae(metrics, best_cif, protenix_pdb_path)
                    with open(meta_path, "w") as f:
                        json.dump(metrics, f)
                        
                    results.append((design_name, protenix_pdb_path, metrics))
                    
                pbar.update(1)
                
            chunk_elapsed = time.time() - chunk_start_time
            logger.info(f"Chunk {i+1}/{n_chunks}: {len(chunk_payload)} designs in {chunk_elapsed:.1f}s")
            
            # Clean up the chunk directory to save disk space
            import shutil
            try:
                shutil.rmtree(chunk_dir)
            except Exception as e:
                logger.warning(f"Could not delete chunk dir {chunk_dir}: {e}")

        except Exception as e:
            err = f"Protenix chunk {chunk_idx} exception: {e}"
            logger.error(err)
            for cp in chunk_payload:
                results.append((cp["name"], None, {"error": err}))
                pbar.update(1)
                
    pbar.close()
    n_ok = sum(1 for r in results if r[1] is not None)
    total_elapsed = time.time() - global_start
    logger.info(f"Protenix batch complete: {n_ok}/{len(payload)} designs succeeded in {total_elapsed:.1f}s")

    return results


def run_protenix_from_json_payload(
    entries: List[dict],
    output_dir: Path,
    model_name: str = "protenix-v2",
    seed: int = 42,
    target_msa_path: Optional[str] = None,
    target_template_path: Optional[str] = None,
    chunk_size: int = 40,
    fetch_all_msas: bool = False,
    n_diffusion_samples: int = 5,
) -> List[Tuple[str, Optional[str], Dict[str, Any]]]:
    """
    Run Protenix on complexes defined by a Protenix-style input JSON list.

    Sequences are taken directly from the JSON entries (no PDB files needed).
    Convention: first chain in each entry = target; all remaining chains = binder.
    `id` fields are stripped so Protenix assigns chains A, B, C... in sequence order.

    The returned metrics_dict contains a ``"binder_chains"`` key (e.g. ``"BC"``) for
    use as the PyRosetta interface specification.

    Returns: List of (design_name, path_to_protenix_pdb, metrics_dict)
    """
    import copy
    import time
    import hashlib
    from tqdm import tqdm

    work_dir = output_dir / "protenix_json_batch"
    work_dir.mkdir(parents=True, exist_ok=True)
    binder_msa_dir = work_dir / "binder_msas"
    binder_msa_dir.mkdir(parents=True, exist_ok=True)
    
    msa_cache_dir = work_dir / "msa_cache"
    msa_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_msa_for_seq(seq: str) -> str:
        s_hash = hashlib.md5(seq.encode()).hexdigest()
        s_dir = msa_cache_dir / s_hash
        if not (s_dir / "non_pairing.a3m").exists():
            return generate_target_msa_dir(seq, s_dir)
        cands = list(s_dir.rglob("non_pairing.a3m"))
        return str(cands[0].parent.absolute())

    resolved_msa_path: Optional[str] = None
    if target_msa_path:
        msa_obj = Path(target_msa_path).absolute()
        if msa_obj.exists():
            resolved_msa_path = str(msa_obj)
            logger.info(f"Using pre-computed target MSA: {resolved_msa_path}")
        else:
            logger.warning(f"Target MSA path provided but NOT FOUND: {msa_obj}")

    resolved_template_path: Optional[str] = None
    if target_template_path:
        tpl_obj = Path(target_template_path).absolute()
        if tpl_obj.exists():
            resolved_template_path = str(tpl_obj)
        else:
            logger.warning(f"Template path provided but NOT FOUND: {tpl_obj}")

    payload: list = []
    job_info: dict = {}

    for entry in entries:
        design_name = entry.get("name", "unknown")
        raw_chains = entry.get("sequences", [])
        if not raw_chains:
            job_info[design_name] = {"error": "No sequences in entry"}
            continue

        chains = copy.deepcopy(raw_chains)

        chain_seqs: List[str] = []
        for chain_entry in chains:
            pc = chain_entry.get("proteinChain", {})
            seq = "".join(c for c in pc.get("sequence", "").upper() if "A" <= c <= "Z")
            pc.pop("id", None)
            chain_seqs.append(seq)

        if not chain_seqs[0]:
            job_info[design_name] = {"error": "Empty target sequence"}
            continue
        binder_seqs = chain_seqs[1:]
        if not binder_seqs:
            job_info[design_name] = {"error": "No binder chains in entry"}
            continue

        binder_chain_labels = "".join(chr(ord('B') + i) for i in range(len(binder_seqs)))
        binder_seq_combined = "/".join(binder_seqs)
        binder_len_total = sum(len(s) for s in binder_seqs)

        new_entry = copy.deepcopy(entry)
        new_entry["name"] = design_name
        new_entry["sequences"] = chains
        new_entry["covalent_bonds"] = entry.get("covalent_bonds", [])
        
        for i, chain_entry in enumerate(new_entry["sequences"]):
            pc = chain_entry["proteinChain"]
            seq = pc["sequence"]
            
            if "pairedMsaPath" in pc and "unpairedMsaPath" in pc:
                continue

            if i == 0:
                if resolved_msa_path:
                    _attach_msa_to_chain(pc, resolved_msa_path)
                elif fetch_all_msas:
                    _attach_msa_to_chain(pc, get_msa_for_seq(seq))
                if resolved_template_path:
                    pc["templatesPath"] = resolved_template_path
            else:
                if fetch_all_msas:
                    _attach_msa_to_chain(pc, get_msa_for_seq(seq))
                else:
                    a3m_path = binder_msa_dir / f"{design_name}_chain{i}.a3m"
                    if not a3m_path.exists():
                        a3m_path.write_text(f">query\n{seq}\n")
                    pc["unpairedMsaPath"] = str(a3m_path)
                    pc["pairedMsaPath"] = str(a3m_path)

        payload.append(new_entry)
        job_info[design_name] = {
            "binder_seq": binder_seq_combined,
            "binder_len": binder_len_total,
            "binder_chains": binder_chain_labels,
        }

    results: list = []
    if not payload:
        for entry in entries:
            dn = entry.get("name", "unknown")
            results.append((dn, None, {"error": job_info.get(dn, {}).get("error", "Failed precondition")}))
        return results

    with open(work_dir / "batch_input.json", "w") as f:
        json.dump(payload, f, indent=2)

    protenix_cmd = _resolve_protenix_cmd()

    completed_results, missing_payload = _get_completed_designs(work_dir, payload, job_info, model_name)
    results.extend(completed_results)

    if not missing_payload:
        logger.info(f"Protenix JSON batch complete: All {len(payload)} complexes already processed.")
        return results

    n_chunks = (len(missing_payload) + chunk_size - 1) // chunk_size
    logger.info(f"Protenix JSON Batch: {len(missing_payload)} missing complexes → {n_chunks} chunk(s) of ≤{chunk_size}")

    existing_chunks = [int(d.name.split('_')[1]) for d in work_dir.glob("chunk_*") if d.is_dir() and d.name.split('_')[1].isdigit()]
    next_chunk_idx = max(existing_chunks) + 1 if existing_chunks else 0

    pbar = tqdm(total=len(payload), desc="Protenix JSON Batch", unit="complex")
    pbar.update(len(completed_results))
    global_start = time.time()

    for i in range(n_chunks):
        chunk_idx = next_chunk_idx + i
        chunk_start_idx = i * chunk_size
        chunk_end_idx = min(chunk_start_idx + chunk_size, len(missing_payload))
        chunk_payload = missing_payload[chunk_start_idx:chunk_end_idx]

        chunk_dir = work_dir / f"chunk_{chunk_idx:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_json = chunk_dir / "chunk_input.json"

        with open(chunk_json, "w") as f:
            json.dump(chunk_payload, f, indent=2)

        cmd = [
            protenix_cmd, "pred",
            "-i", str(chunk_json),
            "-o", str(chunk_dir),
            "-n", model_name,
            "--seeds", str(seed),
            "--sample", str(n_diffusion_samples),
            "--use_msa", "true",
            "--use_template", "true" if resolved_template_path else "false",
            "--enable_cache", "true",
        ]
        if resolved_template_path:
            kalign = _resolve_kalign_cmd()
            if kalign:
                cmd += ["--kalign_binary_path", kalign]

        if i == 0:
            logger.info(f"Protenix command: {' '.join(cmd)}")

        chunk_start_time = time.time()
        
        try:
            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.returncode != 0:
                logger.error(f"Chunk {chunk_idx} exited {process.returncode}. Stderr: {process.stderr[-3000:]}")

            for cp in chunk_payload:
                design_name = cp["name"]
                jinfo = job_info.get(design_name)
                
                best_cif, best_data = _pick_best_sample(chunk_dir, design_name)
                if best_cif is None:
                    results.append((design_name, None, {"error": "No output cif found"}))
                else:
                    protenix_pdb_path = str(work_dir / f"{design_name}_protenix.pdb")
                    meta_path = work_dir / f"{design_name}_protenix_meta.json"
                    try:
                        _cif_to_pdb(str(best_cif), protenix_pdb_path)
                    except Exception:
                        pass
                    metrics = _build_metrics_dict(best_data, jinfo["binder_seq"], jinfo["binder_len"], model_name)
                    metrics["binder_chains"] = jinfo["binder_chains"]
                    _augment_metrics_with_ipsae(metrics, best_cif, protenix_pdb_path)
                    with open(meta_path, "w") as f:
                        json.dump(metrics, f)
                    results.append((design_name, protenix_pdb_path, metrics))
                    
                pbar.update(1)

            chunk_elapsed = time.time() - chunk_start_time
            logger.info(f"Chunk {i+1}/{n_chunks}: {len(chunk_payload)} complexes in {chunk_elapsed:.1f}s")
            
            # Clean up the chunk directory to save disk space
            import shutil
            try:
                shutil.rmtree(chunk_dir)
            except Exception as e:
                logger.warning(f"Could not delete chunk dir {chunk_dir}: {e}")

        except Exception as e:
            err = f"Protenix chunk {chunk_idx} exception: {e}"
            logger.error(err)
            for cp in chunk_payload:
                results.append((cp["name"], None, {"error": err}))
                pbar.update(1)

    pbar.close()
    n_ok = sum(1 for r in results if r[1] is not None)
    logger.info(f"Protenix JSON batch complete: {n_ok}/{len(payload)} complexes in {time.time()-global_start:.1f}s")
    return results


def prepare_msa_dir_from_a3m(
    non_pairing_a3m: str,
    output_dir: Path,
) -> str:
    """
    Build the Protenix MSA directory structure from an existing non-pairing .a3m file.

    Creates *output_dir* and symlinks (or copies) the supplied file as
    ``non_pairing.a3m`` inside it.  No ``pairing.a3m`` is created — Protenix
    treats the field as optional and will simply skip paired-MSA features.

    Returns the absolute path to *output_dir*.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(non_pairing_a3m).absolute()
    if not src.exists():
        raise FileNotFoundError(f"Supplied non-pairing MSA not found: {src}")

    dst = output_dir / "non_pairing.a3m"
    if not dst.exists():
        try:
            dst.symlink_to(src)
        except OSError:
            import shutil
            shutil.copy2(src, dst)

    logger.info(f"MSA directory prepared from {src} → {output_dir}")
    return str(output_dir.absolute())


def generate_target_msa_dir(
    target_seq: str,
    output_dir: Path,
    msa_server_mode: str = "protenix",
) -> str:
    """
    Generate a precomputed MSA directory for the target sequence using
    `protenix msa --msa_server_mode <mode>`.

    Returns the absolute path to the directory containing
    `pairing.a3m` and `non_pairing.a3m`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = output_dir / "target.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">target\n{target_seq}\n")

    protenix_cmd = _resolve_protenix_cmd()
    cmd = [
        protenix_cmd, "msa",
        "--input", str(fasta_path),
        "--out_dir", str(output_dir),
        "--msa_server_mode", msa_server_mode,
    ]
    logger.info(f"Generating target MSA: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"protenix msa failed: {result.stderr}")

    # Find the directory containing non_pairing.a3m — take the first match
    candidates = list(output_dir.rglob("non_pairing.a3m"))
    if not candidates:
        raise FileNotFoundError(f"protenix msa produced no non_pairing.a3m in {output_dir}")

    msa_dir = candidates[0].parent
    logger.info(f"Target MSA directory: {msa_dir}")
    return str(msa_dir.absolute())


def generate_target_msa_and_template_dir(
    target_seq: str,
    output_dir: Path,
    msa_server_mode: str = "protenix",
) -> Tuple[str, Optional[str]]:
    """
    Generate MSA + structural templates for the target sequence in one step using
    `protenix mt` (msatemplate — online search).

    `protenix mt` searches PDB for homologous structures, which Protenix then uses
    as structural templates to bias target folding.  For well-characterised targets
    (e.g. PDL1, EGFR) this typically finds the exact crystal structure.

    Returns: (msa_dir_path, template_a3m_path)
      - msa_dir_path     : directory containing pairing.a3m + non_pairing.a3m
      - template_a3m_path: path to hmmsearch.a3m (or .hhr) for use as templatesPath;
                           None if not found (template inference is then skipped)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a minimal Protenix JSON for the target sequence
    input_json = output_dir / "target_input.json"
    with open(input_json, "w") as f:
        json.dump([{
            "name": "target",
            "sequences": [{"proteinChain": {"count": 1, "sequence": target_seq, "modifications": []}}],
        }], f, indent=2)

    protenix_cmd = _resolve_protenix_cmd()
    cmd = [
        protenix_cmd, "mt",
        "-i", str(input_json),
        "-o", str(output_dir),
        "-m", msa_server_mode,
    ]
    logger.info(f"Generating target MSA + template: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"protenix mt failed: {result.stderr}")

    # Locate MSA directory
    candidates = list(output_dir.rglob("non_pairing.a3m"))
    if not candidates:
        raise FileNotFoundError(f"protenix mt produced no non_pairing.a3m in {output_dir}")
    msa_dir = candidates[0].parent

    # Locate template file (.a3m or .hhr from hmmsearch)
    template_path: Optional[str] = None
    for pattern in ("hmmsearch.a3m", "*.hhr", "*.a3m"):
        hits = [
            p for p in output_dir.rglob(pattern)
            if p.name != "non_pairing.a3m" and p.name != "pairing.a3m"
        ]
        if hits:
            template_path = str(hits[0].absolute())
            logger.info(f"Template file: {template_path}")
            break

    return str(msa_dir.absolute()), template_path


def prepare_template_from_pdb(
    pdb_or_cif_path: str,
    output_dir: Path,
    msa_server_mode: str = "protenix",
) -> Tuple[str, Optional[str]]:
    """
    Convenience wrapper: extract Chain A sequence from a PDB/CIF file, then call
    `generate_target_msa_and_template_dir` to search PDB for matching templates.

    Returns: (msa_dir_path, template_a3m_path)
    """
    from pipeline.scoring.utils import extract_sequence_from_pdb
    target_seq = extract_sequence_from_pdb(pdb_or_cif_path, "A")
    if not target_seq:
        raise ValueError(f"Could not extract Chain A sequence from {pdb_or_cif_path}")
    target_seq = "".join(c for c in target_seq.upper() if "A" <= c <= "Z")
    return generate_target_msa_and_template_dir(target_seq, output_dir, msa_server_mode)


def auto_generate_msa(
    pdb_path: str,
    output_dir: Path,
    db_path: str = "",
    msa_server_mode: str = "protenix",
) -> str:
    """
    Generate a target MSA from a PDB file (extracts Chain A sequence, then calls
    `generate_target_msa_dir`).

    Returns the absolute path to the MSA directory.
    The *db_path* parameter is kept for backward compatibility but is unused when
    *msa_server_mode* is "protenix" (online server is used instead).
    """
    from pipeline.scoring.utils import extract_sequence_from_pdb
    target_seq = extract_sequence_from_pdb(pdb_path, "A")
    if not target_seq:
        raise ValueError(f"Could not extract Chain A sequence from {pdb_path}")
    target_seq = "".join(c for c in target_seq.upper() if "A" <= c <= "Z")
    return generate_target_msa_dir(target_seq, output_dir, msa_server_mode=msa_server_mode)
