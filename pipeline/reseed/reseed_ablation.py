"""
Re-seeding ablation driver (spec §4).

Runs the iterative, self-contained-strain re-seeding loop for one target:

    G0        = post-filter successes from the prior blind run (generation 0, free)
    Founders  = diversity-select up to S founders  (one strain each)
    for k in 1..R:
        for each alive strain:
            parent = strain champion
            cuts   = 5 seed-cuts (P1..P5) drawn from the parent
            for each cut: LaProteina scaffolds `samples_per_cut` progeny
        refold + score everything (existing pipeline, WITH the filter config)
        winners      = samples that passed the filter
        champion[s]  = uniform random winner   (strain dies if none)

Outputs (spec §8): runs/<target>/{gen0,gen<k>,records.parquet,lineage.json,checkpoints}.

Usage:
    python -m pipeline.reseed.reseed_ablation --settings settings_reseed/<target>.json [--resume]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pipeline.reseed.config import ReseedConfig, load_config, effective_samples_per_cut
from pipeline.reseed import founders as founders_mod
from pipeline.reseed import seed_extract
from pipeline.reseed import metrics_extra

# design_name -> (generation, strain, cut, sample) e.g. "NPT2B_g1_s03_P4_sample_007"
# cut id is P1..P5 (random mode) or I1..I5 (interface mode) -> match any letter+digits
_NAME_RE = re.compile(r"_g(\d+)_s(\d+)_([A-Za-z]+\d+)_sample_(\d+)")

# Identity/lineage columns the ablation adds on top of the scorer's row (spec §8).
_LINEAGE_COLS = [
    "target", "sample_id", "generation", "strain_id", "founder_id", "parent_id",
    "cut_id", "cut_pattern", "seed_res_indices", "seed_res_ss", "length",
    "passed_filter", "ss_helix_frac", "ss_sheet_frac", "ss_loop_frac",
    "rmsd_to_parent", "seqid_to_parent",
]


@dataclass
class Strain:
    strain_id: int
    founder_id: str
    champion_pdb: str           # complex PDB seeding the next round
    champion_id: str            # sample_id of the current champion
    alive: bool = True
    parent_len: int = -1


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def split_target_binder(pdb_in: str, pdb_out: str, target_len: int) -> bool:
    """Split a single-chain LaProteina output into chain A (target) + chain B (binder).

    Residues 1..target_len -> chain A; the rest -> chain B, renumbered from 1.
    Mirrors ``evolve_v2.py`` Phase 3. Returns False if the sample has no binder region.
    """
    import biotite.structure as struc
    import biotite.structure.io as strucio

    structure = strucio.load_structure(pdb_in)
    residue_starts = struc.get_residue_starts(structure)
    n_residues = len(residue_starts)
    if n_residues <= target_len:
        return False

    structure.chain_id[:] = "A"
    binder_start_atom_idx = residue_starts[target_len]
    structure.chain_id[binder_start_atom_idx:] = "B"
    for i in range(target_len, n_residues):
        start = residue_starts[i]
        end = residue_starts[i + 1] if i < n_residues - 1 else len(structure)
        structure.res_id[start:end] = (i - target_len) + 1

    Path(pdb_out).parent.mkdir(parents=True, exist_ok=True)
    strucio.save_structure(pdb_out, structure)
    return True


def _resolve_scored_complex(scored_dir: Path, design_name: str) -> Optional[Path]:
    """Find the design's complex PDB used for scoring (chain A + B)."""
    for sub in ("scoring_input", "af2_predictions", "relaxed", "protenix_predictions"):
        base = scored_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.glob(f"{design_name}*.pdb")):
            return p
    return None


# --------------------------------------------------------------------------------------
# Round execution
# --------------------------------------------------------------------------------------

class ReseedRun:
    def __init__(self, cfg: ReseedConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        # Persist the resolved config so downstream analysis/viewer can find the
        # baseline_run_dir + founder_filter_stage (e.g. for cluster-novelty).
        try:
            with open(self.run_dir / "config.json", "w") as _fh:
                json.dump(cfg.to_json_dict(), _fh, indent=2)
        except Exception:  # noqa: BLE001
            pass
        self.records_path = self.run_dir / "records.parquet"
        self.records_csv = self.run_dir / "records.csv"
        self.lineage_path = self.run_dir / "lineage.json"

        import numpy as np
        self.np = np
        self.master_rng = np.random.default_rng(cfg.master_seed)

        self.target_len = self._target_length()
        # Hotspots for interface-mode seed selection (falls back to target_hotspot_residues).
        self.seed_hotspot_ids = seed_extract.parse_residue_spec(
            cfg.seed_interface_hotspots or cfg.target_hotspot_residues)
        self.records: List[dict] = []
        self.strains: Dict[int, Strain] = {}
        self.lineage: Dict[str, dict] = {}   # strain_id -> {founder, rounds: {k: champion_id}}
        self.generator = None                # lazy: LaProteinaGenerator (loads the model)

    # ---- setup -------------------------------------------------------------------
    def _target_length(self) -> int:
        if self.cfg.target_len:
            return int(self.cfg.target_len)
        from pipeline.config_builder import get_pdb_chain_length
        return get_pdb_chain_length(self.cfg.target_pdb, "A")

    def _get_generator(self):
        if self.generator is None:
            from pipeline.laproteina_generator import LaProteinaGenerator
            logger.info("Loading LaProteina model (once) ...")
            self.generator = LaProteinaGenerator(
                checkpoint_path=self.cfg.checkpoint,
                autoencoder_path=self.cfg.autoencoder_path,
                device=self.cfg.device,
                laproteina_mode=self.cfg.laproteina_mode,
                inference_config=self.cfg.inference_config,
            )
        return self.generator

    # ---- generation 0 ------------------------------------------------------------
    def init_generation_zero(self):
        all_founders = founders_mod.load_success_set(
            self.cfg.baseline_run_dir,
            filter_stage=self.cfg.founder_filter_stage,
            min_needed=self.cfg.n_strains)
        if not all_founders:
            raise RuntimeError(f"No founders found in {self.cfg.baseline_run_dir}")
        chosen = founders_mod.select_strain_founders(all_founders, S=self.cfg.n_strains)

        # Strain sharding. Founder selection above is deterministic, so every shard
        # derives the identical global ordering and then keeps a disjoint slice of it.
        # Strain ids stay GLOBAL (not renumbered per shard), so output paths are unique
        # across shards and the trees merge by union.
        indexed = list(enumerate(chosen))
        if self.cfg.n_shards > 1:
            indexed = [(sid, f) for sid, f in indexed
                       if sid % self.cfg.n_shards == self.cfg.shard_index]
            logger.info(
                f"Shard {self.cfg.shard_index}/{self.cfg.n_shards}: "
                f"{len(indexed)} of {len(chosen)} strains "
                f"(ids {[sid for sid, _ in indexed]})")
            if not indexed:
                raise RuntimeError(
                    f"Shard {self.cfg.shard_index} got no strains: only {len(chosen)} "
                    f"founders were selected for n_shards={self.cfg.n_shards}.")

        gen0_dir = self.run_dir / "gen0"
        gen0_dir.mkdir(exist_ok=True)
        for sid, f in indexed:
            binder_len = f.binder_len
            if binder_len is None or binder_len < 0:
                binder_len = seed_extract._load_and_stitch(f.pdb_path, self.target_len)[2]
            # copy founder complex into gen0/
            dst = gen0_dir / f"strain{sid:02d}_{Path(f.pdb_path).stem}.pdb"
            try:
                shutil.copy(f.pdb_path, dst)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not copy founder {f.pdb_path}: {e}")
                dst = Path(f.pdb_path)

            ss = f.ss or metrics_extra.ss_fractions(f.pdb_path, "B") or (None, None, None)
            self.strains[sid] = Strain(
                strain_id=sid, founder_id=f.founder_id,
                champion_pdb=str(dst), champion_id=f.founder_id,
                alive=True, parent_len=int(binder_len),
            )
            self.lineage[str(sid)] = {"founder_id": f.founder_id, "rounds": {}}
            row = self._base_row(
                sample_id=f.founder_id, generation=0, strain_id=sid,
                founder_id=f.founder_id, parent_id=f.founder_id,
                cut_id=None, cut_pattern=None, seed_res_indices=None, seed_res_ss=None,
                length=int(binder_len), passed_filter=True,
                ss=ss, rmsd_to_parent=0.0, seqid_to_parent=1.0,
                scorer_metrics=f.metrics,
            )
            self.records.append(row)

        logger.info(f"Generation 0: {len(self.strains)} strains established")
        self._flush()

    # ---- one round ---------------------------------------------------------------
    def run_round(self, k: int):
        gen_dir = self.run_dir / f"gen{k}"

        # Checkpointing is per-round: a round only counts once it has been scored,
        # collected and flushed. So any gen<k> already on disk here is debris from an
        # attempt that was killed mid-round (time limit, preemption, node failure) and
        # was never recorded. Leaving it would feed stale, half-generated designs into
        # this round's scoring pass, where _collect_round would attribute them by name
        # with no parent context (founder_id/parent_id None, RMSD-to-parent NaN) and
        # let them compete for champion. Start every round from a clean directory.
        if gen_dir.exists():
            logger.warning(f"Round {k}: discarding leftover {gen_dir} from an "
                           f"interrupted earlier attempt at this round")
            shutil.rmtree(gen_dir, ignore_errors=True)

        scoring_input = gen_dir / "_scoring_input"
        scored_dir = gen_dir / "_scored"
        scoring_input.mkdir(parents=True, exist_ok=True)

        alive = [s for s in self.strains.values() if s.alive]
        spc = effective_samples_per_cut(
            self.cfg.samples_per_cut, self.cfg.n_strains, len(alive),
            self.cfg.redistribute_budget)
        msg = f"=== Round {k}: {len(alive)} alive strains ==="
        if spc != self.cfg.samples_per_cut:
            msg += (f" | redistributing freed budget: {spc} samples/cut "
                    f"(base {self.cfg.samples_per_cut})")
        logger.info(msg)

        # (design_name) -> lineage context, so we can attribute scored rows back.
        ctx: Dict[str, dict] = {}

        for strain in alive:
            parent_pdb = strain.champion_pdb
            parent_len = strain.parent_len
            parent_seq = seed_extract.chain_sequence(parent_pdb, "B")
            ss_per_res = metrics_extra.ss_per_residue(parent_pdb, "B")

            try:
                if self.cfg.seed_selection == "interface":
                    cuts = seed_extract.make_interface_seed_cuts(
                        parent_pdb, parent_len, self.master_rng,
                        hotspot_ids=self.seed_hotspot_ids,
                        cutoff=self.cfg.seed_interface_cutoff,
                        seed_n_res=self.cfg.seed_n_res,
                        seed_n_res_min=self.cfg.seed_n_res_min,
                        n_cuts=self.cfg.n_cuts, ss_per_res=ss_per_res)
                    if not cuts:
                        logger.warning(f"Strain {strain.strain_id}: <{self.cfg.seed_n_res_min} "
                                       f"interface residues near hotspots; skipping")
                        continue
                else:
                    cuts = seed_extract.make_seed_cuts(
                        parent_len, self.master_rng, seed_n_res=self.cfg.seed_n_res,
                        ss_per_res=ss_per_res)
            except ValueError as e:
                logger.warning(f"Strain {strain.strain_id}: cannot cut parent ({e}); skipping")
                continue

            for cut in cuts:
                motif_dir = gen_dir / "_motifs" / f"s{strain.strain_id:02d}"
                contig, motif_pdb, plen = seed_extract.build_contig_and_motif_pdb(
                    parent_pdb, self.target_len, cut, str(motif_dir),
                    free_sequence=self.cfg.seed_free_sequence)

                design_id = f"{self.cfg.target}_g{k}_s{strain.strain_id:02d}_{cut.cut_id}"
                gen_seed = int(self.master_rng.integers(0, 2**31 - 1))
                out_dir = gen_dir / "_generated" / design_id
                total_len = self.target_len + plen
                try:
                    gen_pdbs = self._generate(motif_pdb, contig, total_len,
                                              design_id, gen_seed, str(out_dir), spc)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Generation failed for {design_id}: {e}")
                    continue

                for pdb in gen_pdbs:
                    stem = Path(pdb).stem
                    split_out = scoring_input / f"{stem}.pdb"
                    try:
                        if split_target_binder(pdb, str(split_out), self.target_len):
                            ctx[stem] = {
                                "strain_id": strain.strain_id,
                                "founder_id": strain.founder_id,
                                "parent_id": strain.champion_id,
                                "parent_pdb": parent_pdb,
                                "parent_seq": parent_seq,
                                "parent_len": parent_len,
                                "cut_id": cut.cut_id,
                                "cut_pattern": cut.cut_pattern,
                                "seed_res_indices": cut.seed_res_indices,
                                "seed_res_ss": cut.seed_res_ss,
                            }
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Chain-split failed for {pdb}: {e}")

        if not any(scoring_input.glob("*.pdb")):
            logger.warning(f"Round {k}: no samples generated; all remaining strains die")
            for s in alive:
                s.alive = False
            self._flush()
            return

        # Score the whole round at once (one folding pass).
        self._score(scoring_input, scored_dir)

        # Attribute scored rows, compute extras, append records.
        round_rows = self._collect_round(k, scored_dir, ctx, gen_dir)

        # Champion selection per strain (uniform random among winners).
        self._select_champions(k, round_rows, gen_dir)
        self._flush()

    def _generate(self, motif_pdb: str, contig: str, total_len: int,
                  design_id: str, seed: int, out_dir: str, n_samples: int):
        """Scaffold ``n_samples`` progeny around one seed-cut.

        Uses an explicit MotifConfig so ``motif_min_length == motif_max_length ==
        total_len`` — this pins the (already deterministic) fixed-integer contig length
        and avoids generate_binder's dummy 50/2000 bounds rejecting large targets.
        """
        from pipeline.laproteina_generator import MotifConfig, GenerationConfig
        gen = self._get_generator()
        atom_mode = gen.MODE_ATOM_SELECTION_MAP[self.cfg.laproteina_mode]
        motif_cfg = MotifConfig(
            contig_string=contig,
            motif_pdb_path=str(Path(motif_pdb).absolute()),
            motif_only=True,
            motif_min_length=total_len,
            motif_max_length=total_len,
            segment_order="A",
            atom_selection_mode=atom_mode,
            design_id=design_id,
            direct_binder_mode=False,
            binder_target_nres=self.target_len,
        )
        gen_cfg = GenerationConfig(
            n_samples=int(n_samples),
            n_steps=self.cfg.n_steps,
            seed=seed,
            max_batch_size=self.cfg.max_batch_size,
        )
        return gen.generate_from_motif_config(motif_cfg, gen_cfg, output_dir=out_dir)

    def _score(self, scoring_input: Path, scored_dir: Path):
        from pipeline.scoring.scorer import score_complexes
        filters_path = self.cfg.filter_config
        if filters_path and not Path(filters_path).is_absolute():
            filters_path = str(Path(__file__).resolve().parents[2] / filters_path)
        score_complexes(
            input_dir=str(scoring_input),
            output_dir=str(scored_dir),
            filters_path=filters_path,
            folding_model=self.cfg.folding_model,
            protenix_model_name=self.cfg.protenix_model,
            target_msa=self.cfg.target_msa,
            target_template=self.cfg.target_template,
            target_hotspot_residues=self.cfg.target_hotspot_residues,
            target_reference_pdb=self.cfg.target_pdb,
        )

    def _collect_round(self, k: int, scored_dir: Path, ctx: Dict[str, dict],
                       gen_dir: Path) -> List[dict]:
        summary_csv = scored_dir / "scoring_summary.csv"
        if not summary_csv.exists():
            logger.warning(f"Round {k}: no scoring_summary.csv produced")
            return []
        df = pd.read_csv(summary_csv)
        rows: List[dict] = []
        for _, r in df.iterrows():
            design_name = str(r["design_name"])
            info = ctx.get(design_name)
            if info is None:
                # fall back to parsing the name
                m = _NAME_RE.search(design_name)
                if not m:
                    continue
                info = {"strain_id": int(m.group(2)), "founder_id": None,
                        "parent_id": None, "parent_pdb": None, "parent_seq": "",
                        "cut_id": m.group(3), "cut_pattern": None,
                        "seed_res_indices": None, "seed_res_ss": None}

            complex_pdb = _resolve_scored_complex(scored_dir, design_name)
            ss = (None, None, None)
            rmsd_p = float("nan")
            if complex_pdb is not None:
                ss = metrics_extra.ss_fractions(str(complex_pdb), "B") or (None, None, None)
                if info.get("parent_pdb"):
                    rmsd_p = metrics_extra.rmsd_to_parent(str(complex_pdb), info["parent_pdb"])
            seqid_p = metrics_extra.seqid_to_parent(str(r.get("sequence", "")),
                                                    info.get("parent_seq", ""))
            passed = founders_mod._truthy(r.get("pyrosetta_filter_pass"))

            # Persist the sample into the spec's gen<k>/strain<ss>/cut<Pn>/ layout.
            self._archive_sample(gen_dir, info, design_name, complex_pdb, r, ss,
                                 rmsd_p, seqid_p, passed)

            row = self._base_row(
                sample_id=design_name, generation=k, strain_id=info["strain_id"],
                founder_id=info.get("founder_id"), parent_id=info.get("parent_id"),
                cut_id=info.get("cut_id"), cut_pattern=info.get("cut_pattern"),
                seed_res_indices=info.get("seed_res_indices"),
                seed_res_ss=info.get("seed_res_ss"),
                length=_safe_int(r.get("binder_length"), info.get("parent_len", -1)),
                passed_filter=bool(passed), ss=ss,
                rmsd_to_parent=rmsd_p, seqid_to_parent=seqid_p,
                scorer_metrics=r.to_dict(),
            )
            rows.append(row)
            self.records.append(row)
        logger.info(f"Round {k}: scored {len(rows)} samples, "
                    f"{sum(1 for x in rows if x['passed_filter'])} passed")
        return rows

    def _archive_sample(self, gen_dir, info, design_name, complex_pdb, r, ss,
                        rmsd_p, seqid_p, passed):
        sid = info["strain_id"]
        cut = info.get("cut_id") or "NA"
        m = _NAME_RE.search(design_name)
        sample_idx = m.group(4) if m else "000"
        dst_dir = gen_dir / f"strain{sid:02d}" / f"cut{cut}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if complex_pdb is not None:
            try:
                shutil.copy(str(complex_pdb), dst_dir / f"sample{sample_idx}.pdb")
            except Exception:  # noqa: BLE001
                pass
        with open(dst_dir / f"sample{sample_idx}.json", "w") as fh:
            json.dump({"design_name": design_name, "passed_filter": bool(passed),
                       "ss": ss, "rmsd_to_parent": rmsd_p, "seqid_to_parent": seqid_p,
                       "metrics": {k2: _jsonable(v) for k2, v in r.to_dict().items()}},
                      fh, indent=2)

    def _pick_champion(self, winners: List[dict]) -> dict:
        """Choose the next parent among a strain's winners (spec §7 / SS-stratified)."""
        if self.cfg.champion_selection == "ss_stratified":
            pick = self._pick_ss_stratified(winners)
            if pick is not None:
                return pick
        return winners[int(self.master_rng.integers(0, len(winners)))]

    def _pick_ss_stratified(self, winners: List[dict]) -> Optional[dict]:
        """Uniform draw over a helix x sheet SS grid (uniform SS-space coverage).

        Bin winners into a champion_ss_grid x champion_ss_grid grid over [0,1]^2 by
        (helix_frac, sheet_frac), pick a NON-EMPTY cell uniformly, then a winner uniformly
        within it. Returns None if no winner has usable SS (caller falls back to uniform).
        """
        g = int(self.cfg.champion_ss_grid)
        cells: Dict[tuple, List[dict]] = {}
        for w in winners:
            h, s = w.get("ss_helix_frac"), w.get("ss_sheet_frac")
            if h is None or s is None or pd.isna(h) or pd.isna(s):
                continue
            ch = min(int(min(max(float(h), 0.0), 1.0) * g), g - 1)
            cs = min(int(min(max(float(s), 0.0), 1.0) * g), g - 1)
            cells.setdefault((ch, cs), []).append(w)
        if not cells:
            return None
        keys = list(cells.keys())
        bucket = cells[keys[int(self.master_rng.integers(0, len(keys)))]]
        return bucket[int(self.master_rng.integers(0, len(bucket)))]

    def _select_champions(self, k: int, round_rows: List[dict], gen_dir: Path):
        by_strain: Dict[int, List[dict]] = {}
        for row in round_rows:
            by_strain.setdefault(row["strain_id"], []).append(row)

        for strain in [s for s in self.strains.values() if s.alive]:
            winners = [row for row in by_strain.get(strain.strain_id, [])
                       if row["passed_filter"]]
            if not winners:
                strain.alive = False
                self.lineage[str(strain.strain_id)]["rounds"][str(k)] = None
                logger.info(f"Strain {strain.strain_id}: extinct at round {k}")
                continue
            pick = self._pick_champion(winners)
            # resolve the champion's archived complex PDB
            champ_pdb = self._archived_pdb_path(gen_dir, pick)
            strain.champion_pdb = champ_pdb
            strain.champion_id = pick["sample_id"]
            strain.parent_len = int(pick["length"]) if pick["length"] and pick["length"] > 0 else strain.parent_len
            self.lineage[str(strain.strain_id)]["rounds"][str(k)] = pick["sample_id"]

    def _archived_pdb_path(self, gen_dir: Path, row: dict) -> str:
        cut = row.get("cut_id") or "NA"
        m = _NAME_RE.search(row["sample_id"])
        sample_idx = m.group(4) if m else "000"
        return str(gen_dir / f"strain{row['strain_id']:02d}" / f"cut{cut}" / f"sample{sample_idx}.pdb")

    # ---- record assembly / persistence ------------------------------------------
    def _base_row(self, sample_id, generation, strain_id, founder_id, parent_id,
                  cut_id, cut_pattern, seed_res_indices, seed_res_ss, length,
                  passed_filter, ss, rmsd_to_parent, seqid_to_parent,
                  scorer_metrics) -> dict:
        row = {k: _jsonable(v) for k, v in (scorer_metrics or {}).items()}
        row.update({
            "target": self.cfg.target,
            "sample_id": sample_id,
            "generation": generation,
            "strain_id": strain_id,
            "founder_id": founder_id,
            "parent_id": parent_id,
            "cut_id": cut_id,
            "cut_pattern": cut_pattern,
            "seed_res_indices": json.dumps(seed_res_indices) if seed_res_indices is not None else None,
            "seed_res_ss": json.dumps(seed_res_ss) if seed_res_ss is not None else None,
            "length": length,
            "passed_filter": bool(passed_filter),
            "ss_helix_frac": ss[0], "ss_sheet_frac": ss[1], "ss_loop_frac": ss[2],
            "rmsd_to_parent": rmsd_to_parent,
            "seqid_to_parent": seqid_to_parent,
        })
        return row

    def _flush(self):
        df = pd.DataFrame(self.records)
        # order lineage cols first for readability
        cols = [c for c in _LINEAGE_COLS if c in df.columns] + \
               [c for c in df.columns if c not in _LINEAGE_COLS]
        df = df[cols]
        try:
            df.to_parquet(self.records_path, index=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Parquet write failed ({e}); CSV only")
        df.to_csv(self.records_csv, index=False)
        with open(self.lineage_path, "w") as fh:
            json.dump(self.lineage, fh, indent=2)

    def _checkpoint(self, last_round: int):
        ckpt = {
            "last_completed_round": last_round,
            "strains": {str(sid): asdict(s) for sid, s in self.strains.items()},
            "lineage": self.lineage,
            # Persist the master RNG so a resumed run continues the same random stream.
            # Without this the generator is rebuilt from master_seed on every restart,
            # so a resumed round k redraws seed-cuts and generation seeds from the
            # stream position round 1 saw - i.e. a run that was interrupted produces
            # different progeny than the same run left alone. With it, resume is
            # equivalent to an uninterrupted run.
            "rng_state": self.master_rng.bit_generator.state,
        }
        with open(self.run_dir / "checkpoints" / f"{self.cfg.target}_state.json", "w") as fh:
            json.dump(ckpt, fh, indent=2)

    def load_checkpoint(self) -> int:
        ckpt_path = self.run_dir / "checkpoints" / f"{self.cfg.target}_state.json"
        if not ckpt_path.exists():
            return 0
        with open(ckpt_path) as fh:
            ckpt = json.load(fh)
        self.strains = {int(sid): Strain(**s) for sid, s in ckpt["strains"].items()}
        self.lineage = ckpt["lineage"]
        rng_state = ckpt.get("rng_state")
        if rng_state is not None:
            try:
                self.master_rng.bit_generator.state = rng_state
            except (ValueError, KeyError, TypeError) as e:  # noqa: BLE001
                logger.warning(f"Could not restore RNG state ({e}); continuing from "
                               f"master_seed. Progeny will differ from an "
                               f"uninterrupted run, but the run stays valid.")
        else:
            logger.warning("Checkpoint predates RNG-state persistence; resuming from "
                           "master_seed. Progeny from here on will differ from what an "
                           "uninterrupted run would have produced.")
        if self.records_path.exists():
            self.records = pd.read_parquet(self.records_path).to_dict("records")
        elif self.records_csv.exists():
            self.records = pd.read_csv(self.records_csv).to_dict("records")
        logger.info(f"Resumed from round {ckpt['last_completed_round']}")
        return int(ckpt["last_completed_round"])


def _safe_int(v, default: int = -1) -> int:
    """int(v) with NaN/None/blank -> default."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return int(default)
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _jsonable(v):
    """Coerce numpy / NaN scalars to plain python for JSON + parquet safety."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
    except Exception:  # noqa: BLE001
        pass
    return v


def main():
    ap = argparse.ArgumentParser(description="Re-seeding ablation driver")
    ap.add_argument("--settings", required=True, help="settings_reseed/<target>.json")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument("--rounds", type=int, default=None, help="Override number of rounds")
    ap.add_argument("--n_strains", type=int, default=None)
    ap.add_argument("--samples_per_cut", type=int, default=None)
    ap.add_argument("--folding_model", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--output_base_dir", type=str, default=None,
                    help="Override output_base_dir (e.g. a throwaway dir for a smoke test)")
    ap.add_argument("--n_shards", type=int, default=None,
                    help="Split the n_strains strains across this many independent jobs. "
                         "Keep --n_strains at its GLOBAL value in every shard; each shard "
                         "needs its own --output_base_dir.")
    ap.add_argument("--shard_index", type=int, default=None,
                    help="Which shard this job runs, in [0, n_shards).")
    args = ap.parse_args()

    overrides = {k: getattr(args, k) for k in
                 ("rounds", "n_strains", "samples_per_cut", "folding_model", "device",
                  "output_base_dir", "n_shards", "shard_index")}
    cfg = load_config(args.settings, overrides)
    sel = cfg.champion_selection
    if sel == "ss_stratified":
        sel += f" ({cfg.champion_ss_grid}x{cfg.champion_ss_grid} helix×sheet grid)"
    seedsel = cfg.seed_selection
    if seedsel == "interface":
        seedsel += f" (cutoff {cfg.seed_interface_cutoff}Å, hotspots={cfg.seed_interface_hotspots or cfg.target_hotspot_residues})"
    shard = "" if cfg.n_shards == 1 else f", shard={cfg.shard_index}/{cfg.n_shards}"
    logger.info(f"Re-seeding ablation for {cfg.target}: S={cfg.n_strains}, "
                f"cuts={cfg.n_cuts}x{cfg.samples_per_cut}, R={cfg.rounds}, "
                f"folding={cfg.folding_model}, champion={sel}, "
                f"redistribute_budget={cfg.redistribute_budget}, seed={seedsel}{shard}")

    run = ReseedRun(cfg)
    start_round = run.load_checkpoint() if args.resume else 0
    if start_round == 0 and not run.strains:
        run.init_generation_zero()

    for k in range(start_round + 1, cfg.rounds + 1):
        run.run_round(k)
        run._checkpoint(k)

    logger.info(f"Done. Records: {run.records_path}")


if __name__ == "__main__":
    main()
