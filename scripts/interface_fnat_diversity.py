#!/usr/bin/env python3
"""
Interface *mechanism* diversity for a re-seeding run: pairwise fnat on top of
epitope overlap.

Motivation
----------
``target_interface_residues`` (PyRosetta, 8 A) answers "which target residues are
touched" -- the *epitope*. It cannot answer "is the same interaction being made",
because it is a target-side set: it records no pairing, no orientation and no
chemistry. Two designs can share an epitope and still bind through completely
different mechanisms (documented repeatedly for antibodies: same epitope,
divergent binding orientations).

The field's operational definition of "same binding mode" is CAPRI's, of which
the contact term is **fnat** -- the fraction of reference residue-residue
contacts recovered. This script computes fnat between designs and crosses it with
epitope overlap, which splits every pair into three states:

    different patch            epitope Jaccard <  --epitope-thresh
    same patch, same mech.     epitope Jaccard >= --epitope-thresh, fnat >= --fnat-thresh
    same patch, NEW mech.      epitope Jaccard >= --epitope-thresh, fnat <  --fnat-thresh

The last row is the interesting one: re-seeding that keeps finding the same patch
but engages it differently *is* producing mechanism diversity, and a target-side
set metric is blind to it.

Two fnat flavours
-----------------
fnat needs residue correspondence between the two binders.

``index`` -- true CAPRI fnat. A contact is the pair ``(target_resid,
    binder_resid)``; fnat is straight set recovery. Valid only where binder
    numbering corresponds, i.e. **within one lineage** -- re-seeding keeps the
    binder length fixed per strain, so all progeny of a founder are mutually
    indexable. Directly comparable to CAPRI thresholds (0.1 acceptable,
    0.3 medium, 0.5 high).

``spatial`` -- target-anchored fallback for cross-lineage pairs, where binder
    indices mean nothing. The target is the same molecule in every complex, so
    each design is superposed onto a common frame via its target CA atoms, and
    a shared epitope residue counts as recovered when the two designs engage it
    from within ``--pos-tol`` A of each other (contact position = centroid of the
    binder heavy atoms within the cutoff of that target residue).

``auto`` (default) picks ``index`` when the pair shares ``founder_id`` and binder
length, else ``spatial``. Every output row carries the ``fnat_mode`` used, and
the two flavours are never averaged together in a headline number.

Contacts use a --contact-cutoff of 4.5 A between heavy atoms (the iAlign/CAPRI
convention), NOT the 8 A boundary shell that ``target_interface_residues`` uses.
Epitope sets are therefore recomputed here from the PDBs and will be tighter --
and pairwise Jaccard correspondingly lower -- than the values in records.csv.

Outputs (under --out-dir, default <run-dir>/analysis/fnat/)
----------------------------------------------------------
per_generation.csv    within-generation pair stats: does generation g engage its
                      own shared patches consistently, or many ways?
vs_parent.csv         each design against its direct parent -> the cleanest
                      per-generation "did the mechanism change this round" signal
vs_founder.csv        each design against its lineage founder -> cumulative drift
epitope_clusters.csv  greedy epitope clusters, their generation spread and mean
                      intra-cluster fnat
fold_cluster_epitope.csv   with --cluster-csv: per FoldSeek cluster, the epitopes
                      it spans and whether a run-exclusive cluster is a new
                      epitope or merely a new fold on an old epitope
pairwise.csv.gz       every pair, for your own slicing
designs.csv           per-design epitope size, contact count, resolved PDB

Examples
--------
    # CD19_v4
    python scripts/interface_fnat_diversity.py \\
        --run-dir /nfs/students/matscher/reseed_output/cd19_v4/merged/runs/CD19

    # PDL1_v2, including the "are the new clusters a new epitope" question
    python scripts/interface_fnat_diversity.py \\
        --run-dir /nfs/students/matscher/reseed_output/pdl1_v2/merged/runs/PDL1 \\
        --cluster-csv /nfs/students/matscher/samples/foldseek_clusters_PDL1_complexa_vs_laproteina/cluster_assignments_with_metadata.csv \\
        --cluster-root pdl1_v2
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:                                                    # optional, --engine dockq
    from DockQ.DockQ import load_PDB as _dockq_load
    from DockQ.DockQ import run_on_all_native_interfaces as _dockq_run
    HAVE_DOCKQ = True
except ImportError:                                     # pragma: no cover
    HAVE_DOCKQ = False


# --------------------------------------------------------------------------------------
# records / PDB resolution
# --------------------------------------------------------------------------------------

def _truthy(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "pass", "passed", "yes")


def load_records(run_dir: Path) -> pd.DataFrame:
    """Prefer parquet, but fall back to the CSV twin when no parquet engine is
    installed -- the two are written together and carry the same columns."""
    for name in ("records.parquet", "records.csv"):
        p = run_dir / name
        if not p.exists():
            continue
        if p.suffix == ".csv":
            return pd.read_csv(p)
        try:
            return pd.read_parquet(p)
        except ImportError as e:
            print(f"  (no parquet engine: {e}); falling back to records.csv")
    raise FileNotFoundError(f"no readable records.parquet/csv in {run_dir}")


def search_roots(run_dir: Path, extra: Sequence[str] = ()) -> List[Path]:
    """
    Directories to look for archived PDBs in.

    A ``merged`` run directory carries the pooled records.csv but usually only the
    gen-0 founder PDBs; every progeny structure stays in the shard it was produced
    in. So for ``<base>/merged/runs/<TARGET>`` we also search
    ``<base>/shard*/runs/<TARGET>``.
    """
    roots = [run_dir]
    parts = run_dir.parts
    if len(parts) >= 3 and parts[-2] == "runs" and parts[-3] == "merged":
        base = run_dir.parent.parent.parent
        target = parts[-1]
        roots += sorted(p / "runs" / target for p in base.glob("shard*")
                        if (p / "runs" / target).is_dir())
    roots += [Path(e).expanduser() for e in extra]
    return [r for r in roots if r.is_dir()]


class PdbIndex:
    """Lazy recursive stem -> path index over every search root."""

    def __init__(self, roots: Sequence[Path]):
        self.roots = list(roots)
        self._idx: Optional[Dict[str, Path]] = None

    def build(self) -> None:
        if self._idx is not None:
            return
        print(f"  building PDB index over {len(self.roots)} root(s) ...", flush=True)
        self._idx = {}
        for root in self.roots:
            for p in root.rglob("*.pdb"):
                if p.name.startswith("._"):
                    continue
                key = p.stem[:-8] if p.stem.endswith("_relaxed") else p.stem
                self._idx.setdefault(key, p)
        print(f"  indexed {len(self._idx)} PDBs", flush=True)

    def get(self, stem: str) -> Optional[Path]:
        self.build()
        assert self._idx is not None
        return self._idx.get(stem)


def resolve_pdb(roots: Sequence[Path], row: pd.Series, index: PdbIndex) -> Optional[Path]:
    """Archived complex PDB for a records row, across the layouts in the wild."""
    sid = str(row["sample_id"])
    stem = sid.split("/")[-1]
    gen = int(row["generation"])
    strain = int(row["strain_id"]) if pd.notna(row.get("strain_id")) else 0

    cands: List[Path] = []
    for run_dir in roots:
        if gen == 0:
            g0 = run_dir / "gen0"
            cands.append(g0 / f"strain{strain:02d}_{stem}_relaxed.pdb")
            cands.append(g0 / f"strain{strain:02d}_{stem}.pdb")
            if g0.is_dir():
                cands += sorted(g0.glob(f"strain{strain:02d}_*{stem}*.pdb"))
        else:
            gd = run_dir / f"gen{gen}"
            cands.append(gd / "_scored" / "relaxed" / f"{stem}_relaxed.pdb")
            cands.append(gd / "_scored" / "relaxed" / f"{stem}.pdb")
            cands.append(gd / "_scored" / f"{stem}_relaxed.pdb")
            cut = row.get("cut_id")
            if isinstance(cut, str) and cut.strip():
                m = re.search(r"_sample_(\d+)", stem)
                if m:
                    cands.append(gd / f"strain{strain:02d}" / f"cut{cut}"
                                 / f"sample{m.group(1)}.pdb")
    for c in cands:
        if c.exists():
            return c
    return index.get(stem)


# --------------------------------------------------------------------------------------
# PDB parsing
# --------------------------------------------------------------------------------------

def _is_hydrogen(line: str) -> bool:
    el = line[76:78].strip()
    if el:
        return el == "H"
    return line[12:16].strip()[:1] == "H"


def parse_complex(pdb: Path, target_chain: str, binder_chain: str
                  ) -> Optional[Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray,
                                      Dict[int, np.ndarray], Dict[int, str]]]:
    """
    Returns
        tgt_ca      resid -> CA xyz (for superposition)
        b_xyz       (N,3) binder heavy-atom coords
        b_res       (N,)  binder resid per atom
        tgt_heavy   resid -> (K,3) target heavy atoms
        b_restype   binder resid -> 3-letter residue name
    """
    tgt_ca: Dict[int, np.ndarray] = {}
    tgt_heavy: Dict[int, List[Sequence[float]]] = {}
    b_xyz: List[Sequence[float]] = []
    b_res: List[int] = []
    b_restype: Dict[int, str] = {}
    try:
        with open(pdb) as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                    continue
                if _is_hydrogen(line):
                    continue
                ch = line[21]
                if ch not in (target_chain, binder_chain):
                    continue
                try:
                    resid = int(line[22:26])
                    xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError:
                    continue
                if ch == target_chain:
                    tgt_heavy.setdefault(resid, []).append(xyz)
                    if line[12:16].strip() == "CA":
                        tgt_ca[resid] = np.asarray(xyz, dtype=float)
                else:
                    b_xyz.append(xyz)
                    b_res.append(resid)
                    b_restype.setdefault(resid, line[17:20].strip())
    except OSError:
        return None
    if not tgt_heavy or not b_xyz:
        return None
    return (tgt_ca,
            np.asarray(b_xyz, dtype=float),
            np.asarray(b_res, dtype=int),
            {r: np.asarray(v, dtype=float) for r, v in tgt_heavy.items()},
            b_restype)


def kabsch(mobile: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rotation+translation taking `mobile` onto `ref`; also returns RMSD."""
    mc, rc = mobile.mean(axis=0), ref.mean(axis=0)
    P, Q = mobile - mc, ref - rc
    V, _, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    aligned = P @ R
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean()))
    return R, rc - mc @ R, rmsd


# --------------------------------------------------------------------------------------
# per-design interface extraction
# --------------------------------------------------------------------------------------

class Design:
    __slots__ = ("sid", "gen", "strain", "founder", "parent", "blen", "passed",
                 "epitope", "pos", "pairs", "restype_at", "n_contacts", "pdb")

    def __init__(self, sid, gen, strain, founder, parent, blen, passed, pdb):
        self.sid, self.gen, self.strain = sid, gen, strain
        self.founder, self.parent, self.blen, self.passed = founder, parent, blen, passed
        self.pdb = pdb
        self.epitope: frozenset = frozenset()
        self.pos: Dict[int, np.ndarray] = {}      # target resid -> binder contact centroid
        self.pairs: frozenset = frozenset()       # (target resid, binder resid)
        self.restype_at: Dict[int, str] = {}      # target resid -> dominant binder restype
        self.n_contacts = 0


def build_design(row, pdb: Path, target_chain: str, binder_chain: str,
                 cutoff: float, ref_ca: Optional[Dict[int, np.ndarray]]
                 ) -> Optional[Tuple[Design, Dict[int, np.ndarray], float]]:
    parsed = parse_complex(pdb, target_chain, binder_chain)
    if parsed is None:
        return None
    tgt_ca, b_xyz, b_res, tgt_heavy, b_restype = parsed

    sup_rmsd = 0.0
    if ref_ca:
        shared = sorted(set(tgt_ca) & set(ref_ca))
        if len(shared) >= 3:
            mob = np.asarray([tgt_ca[r] for r in shared])
            rf = np.asarray([ref_ca[r] for r in shared])
            R, t, sup_rmsd = kabsch(mob, rf)
            b_xyz = b_xyz @ R + t
            tgt_heavy = {r: v @ R + t for r, v in tgt_heavy.items()}

    d = Design(str(row["sample_id"]), int(row["generation"]),
               int(row["strain_id"]) if pd.notna(row.get("strain_id")) else -1,
               str(row.get("founder_id", "")), str(row.get("parent_id", "")),
               int(row["binder_length"]) if pd.notna(row.get("binder_length")) else -1,
               _truthy(row.get("passed_filter")), str(pdb))

    c2 = cutoff * cutoff
    epi, pos, pairs, rtype = [], {}, set(), {}
    for resid, tatoms in tgt_heavy.items():
        # squared distances target-residue atoms x binder atoms
        diff = b_xyz[None, :, :] - tatoms[:, None, :]
        d2 = (diff * diff).sum(axis=-1)
        hit = d2.min(axis=0) <= c2           # per binder atom
        if not hit.any():
            continue
        epi.append(resid)
        pos[resid] = b_xyz[hit].mean(axis=0)
        br = b_res[hit]
        for x in np.unique(br):
            pairs.add((resid, int(x)))
        vals, counts = np.unique(br, return_counts=True)
        rtype[resid] = b_restype.get(int(vals[counts.argmax()]), "UNK")
    if not epi:
        return None
    d.epitope = frozenset(epi)
    d.pos = pos
    d.pairs = frozenset(pairs)
    d.restype_at = rtype
    d.n_contacts = len(pairs)
    return d, tgt_ca, sup_rmsd


# --------------------------------------------------------------------------------------
# pairwise metrics
# --------------------------------------------------------------------------------------

def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def dockq_pair(i: Design, j: Design, structs: Dict[str, object],
               target_chain: str, binder_chain: str) -> Optional[dict]:
    """
    Reference-implementation fnat via DockQ, treating design *i* as the native.

    DockQ returns nat_correct / nat_total / model_total for the interface, so one
    call yields BOTH directions plus the symmetric form:

        fnat      = nat_correct / nat_total      (CAPRI, native = i)
        fnat_rev  = nat_correct / model_total    (native = j)
        fnat_sym  = 2*nat_correct / (nat_total + model_total)

    Also carries DockQ's own composite score, iRMSD and LRMSD, which the internal
    engine has no equivalent for.

    CAVEAT worth knowing: DockQ establishes residue correspondence by SEQUENCE
    ALIGNMENT of model to native. Our same-lineage binders share length but only
    ~40-45% identity, and the aligner drops residues it cannot match -- so DockQ's
    model_total is smaller than the true contact count and its fnat is a
    conservative (low) estimate relative to positional correspondence. Validated
    against the internal engine: nat_total agrees exactly, model_total does not.
    """
    si, sj = structs.get(i.sid), structs.get(j.sid)
    if si is None or sj is None:
        return None
    try:
        res, _ = _dockq_run(sj, si, chain_map={target_chain: target_chain,
                                               binder_chain: binder_chain})
    except Exception:                                   # noqa: BLE001
        return None
    if not res:
        return None
    v = res.get((target_chain, binder_chain)) or next(iter(res.values()))
    nc = float(v.get("nat_correct", 0))
    nt = float(v.get("nat_total", 0))
    mt = float(v.get("model_total", 0))
    return {
        "dockq": float(v.get("DockQ", float("nan"))),
        "irmsd": float(v.get("iRMSD", float("nan"))),
        "lrmsd": float(v.get("LRMSD", float("nan"))),
        "fnat": (nc / nt) if nt else float("nan"),
        "fnat_rev": (nc / mt) if mt else float("nan"),
        "fnat_sym": (2 * nc / (nt + mt)) if (nt + mt) else float("nan"),
        "nat_correct": nc, "nat_total": nt, "model_total": mt,
    }


def pair_metrics(i: Design, j: Design, mode: str, pos_tol: float) -> dict:
    epi_j = jaccard(i.epitope, j.epitope)
    shared = i.epitope & j.epitope

    use_index = mode == "index" or (
        mode == "auto" and i.founder == j.founder and i.blen == j.blen and i.blen > 0)

    if use_index:
        inter = len(i.pairs & j.pairs)
        fnat_ij = inter / len(i.pairs) if i.pairs else float("nan")
        fnat_ji = inter / len(j.pairs) if j.pairs else float("nan")
        denom = len(i.pairs) + len(j.pairs)
        fnat_sym = (2 * inter / denom) if denom else float("nan")
        fnat_shared = fnat_sym
        used = "index"
    else:
        matched = sum(
            1 for r in shared
            if float(np.linalg.norm(i.pos[r] - j.pos[r])) <= pos_tol)
        fnat_shared = matched / len(shared) if shared else float("nan")
        fnat_ij = matched / len(i.epitope) if i.epitope else float("nan")
        fnat_ji = matched / len(j.epitope) if j.epitope else float("nan")
        fnat_sym = (2 * matched / (len(i.epitope) + len(j.epitope))) \
            if (i.epitope or j.epitope) else float("nan")
        used = "spatial"

    chem = float(np.mean([i.restype_at[r] == j.restype_at[r] for r in shared])) \
        if shared else float("nan")

    return {
        "sample_i": i.sid, "sample_j": j.sid,
        "gen_i": i.gen, "gen_j": j.gen,
        "founder_i": i.founder, "founder_j": j.founder,
        "same_lineage": i.founder == j.founder,
        "epitope_jaccard": epi_j,
        "n_shared_epitope": len(shared),
        "fnat_mode": used,
        "fnat_shared": fnat_shared, "fnat_sym": fnat_sym,
        "fnat_ij": fnat_ij, "fnat_ji": fnat_ji,
        "restype_agreement": chem,
    }


def classify(row: dict, epi_thresh: float, fnat_thresh: float) -> str:
    if not np.isfinite(row["epitope_jaccard"]) or row["epitope_jaccard"] < epi_thresh:
        return "different_patch"
    f = row["fnat_shared"]
    if not np.isfinite(f):
        return "different_patch"
    return "same_patch_same_mech" if f >= fnat_thresh else "same_patch_new_mech"


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------

def _agg(rows: List[dict], epi_thresh: float, fnat_thresh: float) -> dict:
    """
    Aggregate a bag of pairs.

    NOTE the two fnat flavours are reported SEPARATELY and never averaged into one
    another. They are not on the same scale: ``index`` demands the identical
    (target_resid, binder_resid) pair, while ``spatial`` only demands the shared
    epitope residue be engaged from within --pos-tol, which is strictly looser and
    reads systematically higher. A mixed mean would drift purely with the
    lineage composition of the pair set.
    """
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    sp = df[df["state"] != "different_patch"]
    out = {
        "n_pairs": len(df),
        "mean_epitope_jaccard": df["epitope_jaccard"].mean(),
        "frac_same_patch": (df["state"] != "different_patch").mean(),
        "n_same_patch_pairs": len(sp),
        "frac_same_patch_same_mech": (df["state"] == "same_patch_same_mech").mean(),
        "frac_same_patch_new_mech": (df["state"] == "same_patch_new_mech").mean(),
        "mean_restype_agreement": df["restype_agreement"].mean(),
        "pct_index_mode": (df["fnat_mode"] == "index").mean(),
    }
    out["new_mech_share_of_same_patch"] = (
        (sp["state"] == "same_patch_new_mech").mean() if len(sp) else float("nan"))
    # per-mode, same-patch-only. "dockq" is mode-free (one scale); index/spatial
    # are NOT interchangeable and are kept apart.
    for mode in ("dockq", "index", "spatial"):
        m = sp[sp["fnat_mode"] == mode]
        if not len(m) and mode != "dockq":
            continue
        out[f"n_{mode}_pairs"] = len(m)
        out[f"mean_fnat_{mode}"] = m["fnat_shared"].mean() if len(m) else float("nan")
        out[f"median_fnat_{mode}"] = m["fnat_shared"].median() if len(m) else float("nan")
        out[f"new_mech_share_{mode}"] = (
            (m["fnat_shared"] < fnat_thresh).mean() if len(m) else float("nan"))
    for extra in ("dockq", "irmsd", "lrmsd"):
        if extra in df.columns:
            out[f"mean_{extra}"] = df[extra].mean()
    return out


def greedy_epitope_clusters(designs: List[Design], thresh: float) -> List[int]:
    reps: List[frozenset] = []
    assign: List[int] = []
    for d in designs:
        for ci, rep in enumerate(reps):
            if jaccard(d.epitope, rep) >= thresh:
                assign.append(ci)
                break
        else:
            reps.append(d.epitope)
            assign.append(len(reps) - 1)
    return assign


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pairwise fnat x epitope overlap for a re-seeding run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="runs/<TARGET> directory")
    ap.add_argument("--out-dir", default=None, help="default <run-dir>/analysis/fnat")
    ap.add_argument("--pdb-root", action="append", default=[],
                    help="extra directory to search for archived PDBs (repeatable). "
                         "A merged run dir already auto-searches its sibling shard*/ "
                         "dirs, where the progeny structures actually live.")
    ap.add_argument("--contact-cutoff", type=float, default=4.5,
                    help="heavy-atom contact cutoff in A (iAlign/CAPRI convention)")
    ap.add_argument("--epitope-thresh", type=float, default=0.5,
                    help="epitope Jaccard at/above which two designs share a patch")
    ap.add_argument("--fnat-thresh", type=float, default=0.3,
                    help="fnat at/above which a shared patch counts as the same "
                         "mechanism (CAPRI: 0.1 acceptable, 0.3 medium, 0.5 high)")
    ap.add_argument("--pos-tol", type=float, default=4.0,
                    help="spatial mode: A tolerance on the binder contact centroid")
    ap.add_argument("--engine", choices=("dockq", "internal"), default="dockq",
                    help="dockq = reference implementation (needs `pip install "
                         "DockQ`); internal = the built-in index/spatial fnat. "
                         "DockQ handles differing binder lengths via sequence "
                         "alignment, so it works cross-lineage where the internal "
                         "index mode cannot.")
    ap.add_argument("--fnat-mode", choices=("auto", "index", "spatial"), default="auto",
                    help="internal engine only; ignored when --engine dockq")
    ap.add_argument("--target-chain", default="A")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--passing-only", action="store_true",
                    help="restrict to passed_filter designs")
    ap.add_argument("--max-designs", type=int, default=0,
                    help="0 = no cap; else evenly subsample per generation")
    ap.add_argument("--cluster-csv", default=None,
                    help="FoldSeek cluster_assignments_with_metadata.csv")
    ap.add_argument("--cluster-root", default=None,
                    help="experiment prefix selecting this run in --cluster-csv "
                         "(e.g. pdl1_v2)")
    ap.add_argument("--no-pairwise-dump", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir).expanduser()
    if not run.is_dir():
        print(f"ERROR: no such run dir: {run}", file=sys.stderr)
        return 2
    out = Path(args.out_dir).expanduser() if args.out_dir else run / "analysis" / "fnat"
    out.mkdir(parents=True, exist_ok=True)

    df = load_records(run)
    df = df[df["status"].astype(str) == "scored"].copy()
    if args.passing_only:
        df = df[df["passed_filter"].apply(_truthy)]
    if args.max_designs and len(df) > args.max_designs:
        # even stride within each generation, so a cap stays representative of the
        # strains present rather than collapsing onto whichever one is listed first
        per = max(2, args.max_designs // max(1, df["generation"].nunique()))
        keep = []
        for _, g in df.groupby("generation", sort=True):
            if len(g) <= per:
                keep.append(g)
            else:
                idx = np.linspace(0, len(g) - 1, per).astype(int)
                keep.append(g.iloc[np.unique(idx)])
        df = pd.concat(keep, ignore_index=True)
    print(f"records selected: {len(df)}")

    # ---- build designs ----
    roots = search_roots(run, args.pdb_root)
    print(f"PDB search roots ({len(roots)}):")
    for r in roots:
        print(f"  {r}")
    index = PdbIndex(roots)
    designs: List[Design] = []
    ref_ca: Optional[Dict[int, np.ndarray]] = None
    n_missing = 0
    sup_rmsds: List[float] = []
    for _, row in df.iterrows():
        pdb = resolve_pdb(roots, row, index)
        if pdb is None:
            n_missing += 1
            continue
        built = build_design(row, pdb, args.target_chain, args.binder_chain,
                             args.contact_cutoff, ref_ca)
        if built is None:
            n_missing += 1
            continue
        d, tgt_ca, rms = built
        if ref_ca is None:
            ref_ca = tgt_ca
        else:
            sup_rmsds.append(rms)
        designs.append(d)
    print(f"designs with parsed interface: {len(designs)}  (unresolved/unparsed: {n_missing})")
    if len(designs) < 2:
        print("ERROR: need >=2 designs", file=sys.stderr)
        return 3
    if sup_rmsds:
        print(f"target superposition RMSD: mean={np.mean(sup_rmsds):.3f} "
              f"max={np.max(sup_rmsds):.3f} A "
              f"({'common frame' if np.max(sup_rmsds) < 0.1 else 'superposed'})")
    print(f"epitope size at {args.contact_cutoff} A: "
          f"mean={np.mean([len(d.epitope) for d in designs]):.1f} residues; "
          f"contacts mean={np.mean([d.n_contacts for d in designs]):.1f}")

    pd.DataFrame([{
        "sample_id": d.sid, "generation": d.gen, "strain_id": d.strain,
        "founder_id": d.founder, "parent_id": d.parent, "binder_length": d.blen,
        "passed_filter": d.passed, "epitope_size": len(d.epitope),
        "n_contacts": d.n_contacts,
        "epitope_residues": ",".join(map(str, sorted(d.epitope))), "pdb": d.pdb,
    } for d in designs]).to_csv(out / "designs.csv", index=False)

    # ---- all pairs ----
    # ---- optional DockQ structures (loaded once, reused across all pairs) ----
    structs: Dict[str, object] = {}
    if args.engine == "dockq":
        if not HAVE_DOCKQ:
            print("ERROR: --engine dockq but DockQ is not importable "
                  "(pip install DockQ)", file=sys.stderr)
            return 4
        print(f"loading {len(designs)} structures for DockQ ...", flush=True)
        for k, d in enumerate(designs):
            try:
                structs[d.sid] = _dockq_load(d.pdb)
            except Exception as e:                       # noqa: BLE001
                print(f"  load failed {d.sid}: {e}")
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(designs)}", flush=True)
        print(f"  loaded {len(structs)}/{len(designs)}")

    fail = {"n": 0}

    def compute_pair(i: Design, j: Design) -> dict:
        """Single entry point so every comparison (all-pairs, vs-parent,
        vs-founder) goes through the same engine. Previously only the all-pairs
        loop was DockQ-augmented, which left the lineage tables on the internal
        metric and their dockq columns NaN."""
        m = pair_metrics(i, j, args.fnat_mode, args.pos_tol)
        if args.engine == "dockq":
            dq = dockq_pair(i, j, structs, args.target_chain, args.binder_chain)
            if dq is None:
                fail["n"] += 1
            else:
                m.update(dq)
                # DockQ's directional fnat becomes the headline number; it is
                # mode-free, so index/spatial no longer applies.
                m["fnat_shared"] = dq["fnat"]
                m["fnat_mode"] = "dockq"
        m["state"] = classify(m, args.epitope_thresh, args.fnat_thresh)
        return m

    n_pairs = len(designs) * (len(designs) - 1) // 2
    print(f"computing {n_pairs} pairs (engine={args.engine}) ...", flush=True)
    rows: List[dict] = []
    for k, (a, b) in enumerate(itertools.combinations(range(len(designs)), 2)):
        rows.append(compute_pair(designs[a], designs[b]))
        if n_pairs > 20000 and (k + 1) % 20000 == 0:
            print(f"  {k+1}/{n_pairs}", flush=True)
    if fail["n"]:
        print(f"  DockQ returned no interface for {fail['n']}/{n_pairs} pairs "
              f"(internal fnat kept for those)")
    pw = pd.DataFrame(rows)
    if not args.no_pairwise_dump:
        with gzip.open(out / "pairwise.csv.gz", "wt") as fh:
            pw.to_csv(fh, index=False)

    # ---- per generation (within-generation pairs) ----
    per_gen = []
    for gen in sorted({d.gen for d in designs}):
        sub = pw[(pw["gen_i"] == gen) & (pw["gen_j"] == gen)]
        rec = {"generation": gen,
               "n_designs": sum(1 for d in designs if d.gen == gen)}
        rec.update(_agg(sub.to_dict("records"), args.epitope_thresh, args.fnat_thresh))
        per_gen.append(rec)
    per_gen_df = pd.DataFrame(per_gen)
    per_gen_df.to_csv(out / "per_generation.csv", index=False)

    # ---- fnat threshold sweep -------------------------------------------------
    # The "same mechanism" call is threshold-dependent; if every pair sits well
    # above --fnat-thresh the classification is saturated and says nothing. This
    # sweep shows where the signal actually lives before you quote a number.
    sp_all = pw[pw["epitope_jaccard"] >= args.epitope_thresh]
    sweep = []
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        rec = {"fnat_thresh": t}
        for mode in ("dockq", "index", "spatial"):
            m = sp_all[sp_all["fnat_mode"] == mode]
            if not len(m):
                continue
            rec[f"new_mech_share_{mode}"] = (m["fnat_shared"] < t).mean()
        for gen in sorted({d.gen for d in designs}):
            g = sp_all[(sp_all["gen_i"] == gen) & (sp_all["gen_j"] == gen)]
            rec[f"gen{gen}_new_mech_share"] = ((g["fnat_shared"] < t).mean()
                                               if len(g) else float("nan"))
        sweep.append(rec)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out / "fnat_threshold_sweep.csv", index=False)

    # ---- vs direct parent ----
    by_sid = {d.sid: d for d in designs}
    by_stem = {d.sid.split("/")[-1]: d for d in designs}
    vs_parent = []
    for d in designs:
        if d.gen == 0:
            continue
        p = by_sid.get(d.parent) or by_stem.get(str(d.parent).split("/")[-1])
        if p is None:
            continue
        m = compute_pair(p, d)
        m["generation"] = d.gen
        vs_parent.append(m)
    vp = pd.DataFrame(vs_parent) if vs_parent else pd.DataFrame()
    if len(vp):
        vp.to_csv(out / "vs_parent_pairs.csv", index=False)
        agg = []
        for gen, g in vp.groupby("generation"):
            rec = {"generation": int(gen), "n_child_parent_pairs": len(g)}
            rec.update(_agg(g.to_dict("records"), args.epitope_thresh, args.fnat_thresh))
            agg.append(rec)
        pd.DataFrame(agg).to_csv(out / "vs_parent.csv", index=False)
    else:
        print("NOTE: no parent_id links resolved -- vs_parent.csv skipped")

    # ---- vs lineage founder ----
    founders = {d.founder: d for d in designs if d.gen == 0}
    vs_founder = []
    for d in designs:
        if d.gen == 0:
            continue
        f = founders.get(d.founder)
        if f is None:
            continue
        m = compute_pair(f, d)
        m["generation"] = d.gen
        vs_founder.append(m)
    vf = pd.DataFrame(vs_founder) if vs_founder else pd.DataFrame()
    if len(vf):
        vf.to_csv(out / "vs_founder_pairs.csv", index=False)
        agg = []
        for gen, g in vf.groupby("generation"):
            rec = {"generation": int(gen), "n_vs_founder_pairs": len(g)}
            rec.update(_agg(g.to_dict("records"), args.epitope_thresh, args.fnat_thresh))
            agg.append(rec)
        pd.DataFrame(agg).to_csv(out / "vs_founder.csv", index=False)
    else:
        print("NOTE: no gen-0 founders in selection -- vs_founder.csv skipped")

    # ---- epitope clusters ----
    epi_assign = greedy_epitope_clusters(designs, args.epitope_thresh)
    sid_to_epi = {d.sid: c for d, c in zip(designs, epi_assign)}
    epi_rows = []
    for c in sorted(set(epi_assign)):
        mem = [d for d, a in zip(designs, epi_assign) if a == c]
        sids = {d.sid for d in mem}
        intra = pw[pw["sample_i"].isin(sids) & pw["sample_j"].isin(sids)]
        epi_rows.append({
            "epitope_cluster": c, "n": len(mem),
            "generations": ",".join(map(str, sorted({d.gen for d in mem}))),
            "first_generation": min(d.gen for d in mem),
            "founders": len({d.founder for d in mem}),
            "strains": ",".join(map(str, sorted({d.strain for d in mem}))),
            "mean_intra_fnat_shared": intra["fnat_shared"].mean() if len(intra) else float("nan"),
            "frac_intra_new_mech": (intra["state"] == "same_patch_new_mech").mean()
                                   if len(intra) else float("nan"),
            "consensus_epitope": ",".join(map(str, sorted(
                set.intersection(*[set(d.epitope) for d in mem])))) if mem else "",
        })
    pd.DataFrame(epi_rows).to_csv(out / "epitope_clusters.csv", index=False)

    # ---- fold cluster x epitope (the "are new clusters a new epitope" question) ----
    if args.cluster_csv:
        cpath = Path(args.cluster_csv).expanduser()
        if not cpath.exists():
            print(f"WARNING: --cluster-csv not found: {cpath}")
        else:
            cl = pd.read_csv(cpath)
            cl["root"] = cl["experiment"].astype(str).str.split("/").str[0]
            root = args.cluster_root
            if root is None:
                print("WARNING: --cluster-root not given; inferring from run path")
                root = run.parts[-4] if len(run.parts) >= 4 else ""
            mine = cl[cl["root"] == root].copy()
            if not len(mine):
                print(f"WARNING: no rows with root=={root!r} in cluster CSV")
            else:
                mine["stem"] = (mine["original_path"].astype(str).str.split("/").str[-1]
                                .str.replace("_relaxed.pdb", "", regex=False)
                                .str.replace(".pdb", "", regex=False))
                stem_to_epi = {d.sid.split("/")[-1]: sid_to_epi[d.sid] for d in designs}
                stem_to_gen = {d.sid.split("/")[-1]: d.gen for d in designs}
                mine["epitope_cluster"] = mine["stem"].map(stem_to_epi)
                mine["generation"] = mine["stem"].map(stem_to_gen)
                # exclusivity is judged on the FULL cluster CSV, not just our rows
                sizes = cl.groupby("cluster_id").size()
                mine_sizes = mine.groupby("cluster_id").size()
                frows = []
                for cid, g in mine.groupby("cluster_id"):
                    total = int(sizes.get(cid, 0))
                    n_mine = int(mine_sizes.get(cid, 0))
                    others = cl[(cl["cluster_id"] == cid) & (cl["root"] != root)]
                    epis = sorted(set(g["epitope_cluster"].dropna().astype(int)))
                    sids = {d.sid for d in designs
                            if d.sid.split("/")[-1] in set(g["stem"])}
                    intra = pw[pw["sample_i"].isin(sids) & pw["sample_j"].isin(sids)]
                    outer = pw[(pw["sample_i"].isin(sids) ^ pw["sample_j"].isin(sids))]
                    exclusive = len(others) == 0
                    # is a run-exclusive fold cluster ALSO a new epitope?
                    if exclusive:
                        if epis and all(
                            # epitope cluster seen only inside this fold cluster
                            set(mine[mine["epitope_cluster"] == e]["cluster_id"]) == {cid}
                            and (min(g["generation"].dropna(), default=1) > 0)
                            for e in epis
                        ):
                            verdict = "NEW_FOLD_AND_NEW_EPITOPE"
                        else:
                            verdict = "NEW_FOLD_ON_EXISTING_EPITOPE"
                    else:
                        verdict = "shared_with_other_sources"
                    frows.append({
                        "fold_cluster_id": cid,
                        "n_from_this_run": n_mine,
                        "n_total_members": total,
                        "n_other_source_members": len(others),
                        "exclusive_to_run": exclusive,
                        "other_sources": ",".join(sorted(set(others["root"]))) if len(others) else "",
                        "generations": ",".join(map(str, sorted(set(
                            g["generation"].dropna().astype(int))))),
                        "epitope_clusters_spanned": ",".join(map(str, epis)),
                        "n_epitope_clusters": len(epis),
                        "mean_intra_fnat_shared": intra["fnat_shared"].mean() if len(intra) else float("nan"),
                        "mean_epitope_jaccard_vs_other_folds": outer["epitope_jaccard"].mean() if len(outer) else float("nan"),
                        "mean_fnat_vs_other_folds": outer["fnat_shared"].mean() if len(outer) else float("nan"),
                        "verdict": verdict,
                    })
                fdf = pd.DataFrame(frows).sort_values("n_from_this_run", ascending=False)
                fdf.to_csv(out / "fold_cluster_epitope.csv", index=False)
                print("\n--- fold clusters exclusive to this run ---")
                ex = fdf[fdf["exclusive_to_run"]]
                print(ex.to_string(index=False) if len(ex)
                      else "  none (every fold cluster is shared with another source)")

    # ---- console summary ----
    print("\n" + "=" * 78)
    print(f"PER-GENERATION  (epitope>={args.epitope_thresh}, fnat>={args.fnat_thresh} "
          f"= same mechanism, contacts at {args.contact_cutoff} A)")
    print("=" * 78)
    cols = ["generation", "n_designs", "n_pairs", "mean_epitope_jaccard",
            "n_dockq_pairs", "mean_fnat_dockq", "new_mech_share_dockq",
            "mean_dockq", "mean_irmsd",
            "n_index_pairs", "mean_fnat_index", "new_mech_share_index",
            "n_spatial_pairs", "mean_fnat_spatial", "new_mech_share_spatial",
            "mean_restype_agreement"]
    print(per_gen_df[[c for c in cols if c in per_gen_df.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  (index = true CAPRI fnat, same lineage; spatial = target-anchored, "
          "cross-lineage. Different scales -- do not compare across columns.)")
    c2 = ["generation", "n_child_parent_pairs", "n_vs_founder_pairs",
          "mean_epitope_jaccard", "mean_fnat_dockq", "new_mech_share_dockq",
          "mean_dockq", "mean_irmsd", "mean_fnat_index", "new_mech_share_index",
          "mean_restype_agreement"]
    if vs_parent:
        print("\n--- child vs DIRECT PARENT (did this round change the mechanism?) ---")
        vpd = pd.read_csv(out / "vs_parent.csv")
        print(vpd[[c for c in c2 if c in vpd.columns]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if vs_founder:
        print("\n--- design vs LINEAGE FOUNDER (cumulative mechanism drift) ---")
        vfd = pd.read_csv(out / "vs_founder.csv")
        print(vfd[[c for c in c2 if c in vfd.columns]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n--- fnat threshold sweep: share of same-patch pairs called NEW MECHANISM ---")
    print(sweep_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  A column that is 0.000 everywhere means the threshold is too permissive")
    print("  to discriminate; pick a threshold where the generations separate.")
    print(f"\nwrote -> {out}")
    print("\nHow to read this:")
    print("  mean_fnat_index HIGH + new_mech_share_index LOW")
    print("      -> same patch engaged the same way: refinement only, no new mechanism.")
    print("  mean_epitope_jaccard HIGH + new_mech_share_index HIGH")
    print("      -> same patch engaged DIFFERENTLY: real mechanism diversity that a")
    print("         target-side residue-set metric cannot see.")
    print("  mean_restype_agreement LOW while fnat stays HIGH")
    print("      -> same residues engaged from the same geometry but by different")
    print("         amino acids: chemical, not geometric, mechanism diversity.")
    print("  Quote a threshold only after checking fnat_threshold_sweep.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
