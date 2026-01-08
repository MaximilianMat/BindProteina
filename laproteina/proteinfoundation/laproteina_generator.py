"""
LaProteina Generator Library API

Provides a clean programmatic interface to LaProteina for binder design.
Can be called iteratively for multi-round design workflows.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import lightning as L
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from loguru import logger

# Optimize for A100 Tensor Cores
torch.set_float32_matmul_precision('medium')

@dataclass
class MotifConfig:
    """Configuration for motif-conditioned generation."""
    
    contig_string: str
    motif_pdb_path: str
    motif_only: bool = True
    motif_min_length: int = 50
    motif_max_length: int = 150
    segment_order: str = "A"
    atom_selection_mode: str = "tip_atoms"  # tip_atoms, all_atom, ca_only, backbone
    design_id: Optional[str] = None  # Short unique ID for naming (used in evolution workflow)


@dataclass
class GenerationConfig:
    """Configuration for generation parameters."""
    
    n_samples: int = 28
    max_batch_size: int = 2
    n_steps: int = 400
    seed: int = 5
    device: str = "cuda"
    linker_length: int = 0


class LaProteinaGenerator:
    """
    Library-style wrapper for LaProteina protein generation.
    
    Loads the model once and can be called multiple times for iterative design.
    
    Example:
        generator = LaProteinaGenerator(
            checkpoint_path="checkpoints_laproteina/LD5_motif_idx_tip.ckpt",
            autoencoder_path="checkpoints_laproteina/AE3_motif.ckpt"
        )
        
        pdbs = generator.generate_binder(
            target_pdb="my_target.pdb",
            target_residue_range=(1, 705),
            binder_length_range=(20, 80),
            n_samples=10,
            output_dir="./output"
        )
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        autoencoder_path: Optional[str] = None,
        base_config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize the generator by loading model weights.
        
        Args:
            checkpoint_path: Path to main model checkpoint (.ckpt)
            autoencoder_path: Path to autoencoder checkpoint (.ckpt), if using latent model
            base_config_path: Optional path to base config directory
            device: Device to run on ("cuda" or "cpu")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.autoencoder_path = Path(autoencoder_path) if autoencoder_path else None
        self.device = device
        self.model = None
        self._base_config = None
        
        # Resolve config path relative to this file or use provided
        if base_config_path:
            self._config_dir = Path(base_config_path)
        else:
            # Default: assume configs are in ../configs relative to proteinfoundation
            self._config_dir = Path(__file__).parent.parent / "configs"
        
        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        if self.autoencoder_path and not self.autoencoder_path.exists():
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {self.autoencoder_path}")
        
        logger.info(f"LaProteinaGenerator initialized with checkpoint: {self.checkpoint_path}")
    
    def _load_model(self):
        """Load the model if not already loaded."""
        if self.model is not None:
            return
        
        from proteinfoundation.proteina import Proteina
        
        logger.info(f"Loading model from {self.checkpoint_path}")
        
        self.model = Proteina.load_from_checkpoint(
            str(self.checkpoint_path),
            strict=False,
            autoencoder_ckpt_path=str(self.autoencoder_path) if self.autoencoder_path else None
        )
        
        # Optimization: Remove encoder from autoencoder during generation
        if self.model.autoencoder is not None and hasattr(self.model.autoencoder, 'encoder'):
            if self.model.autoencoder.encoder is not None:
                logger.info("Removing autoencoder encoder to save memory during generation")
                del self.model.autoencoder.encoder
                self.model.autoencoder.encoder = None
        
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def unload(self):
        """Unload the model and free GPU memory."""
        if self.model is not None:
            logger.info("Unloading LaProteina model and clearing GPU cache")
            self.model.cpu()
            del self.model
            self.model = None
            
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    def _build_generation_config(
        self,
        motif_config: MotifConfig,
        gen_config: GenerationConfig,
    ) -> DictConfig:
        """Build OmegaConf config for generation."""
        
        # Build the config structure that GenDataset and Proteina expect
        config = {
            "seed": gen_config.seed,
            "generation": {
                "n_recycle": 0,
                "args": {
                    "motif_cond": True,
                    "nsteps": gen_config.n_steps,
                    "self_cond": True,
                    "fold_cond": False,
                    "guidance_w": 1.0,
                    "ag_ratio": 0.0,
                    "save_trajectory_every": 0,
                },
                "model": {
                    "bb_ca": {
                        "schedule": {"mode": "log", "p": 2.0},
                        "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
                        "simulation_step_params": {
                            "sampling_mode": "sc",
                            "sc_scale_noise": 0.1,
                            "sc_scale_score": 1.0,
                            "t_lim_ode": 0.98,
                            "t_lim_ode_below": 0.02,
                            "center_every_step": False,  # Matched with inference_motif_idx_tip.yaml
                        }
                    },
                    "local_latents": {
                        "schedule": {"mode": "power", "p": 2.0},
                        "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
                        "simulation_step_params": {
                            "sampling_mode": "sc",
                            "sc_scale_noise": 0.1,
                            "sc_scale_score": 1.0,
                            "t_lim_ode": 0.98,
                            "t_lim_ode_below": 0.02,
                            "center_every_step": False,
                        }
                    }
                },
                "dataset": {
                    "motif_task_name": "TARGET_BINDER_TIP",
                    "nsamples": gen_config.n_samples,
                    "max_nsamples_per_batch": gen_config.max_batch_size,
                    "motif_dict_cfg": {
                        "TARGET_BINDER_TIP": {
                            "contig_string": motif_config.contig_string,
                            "motif_pdb_path": motif_config.motif_pdb_path,
                            "motif_only": motif_config.motif_only,
                            "motif_min_length": motif_config.motif_min_length,
                            "motif_max_length": motif_config.motif_max_length,
                            "segment_order": motif_config.segment_order,
                            "atom_selection_mode": motif_config.atom_selection_mode,
                        }
                    },
                    "nlens_cfg": {
                        "nres_lens": None,
                        "min_len": None,
                        "max_len": None,
                        "step_len": None,
                    },
                    "cath_codes": None,
                    "empirical_distribution_cfg": {
                        "len_cath_code_path": None,
                        "cath_code_level": None,
                        "bucket_min_len": 50,
                        "bucket_max_len": 274,
                        "bucket_step_size": 25,
                    },
                },
                "metric": {
                    "compute_designability": True,
                    "compute_codesignability": False,
                    "compute_motif_rmsd": True,
                    "compute_novelty_pdb": False,
                    "compute_novelty_afdb": False,
                    "compute_fid": False,
                    "keep_folding_outputs": True,
                    "designability_modes": ["ca", "bb3o"],
                },
            }
        }
        
        return OmegaConf.create(config)
    
    def generate_binder(
        self,
        target_pdb: str,
        target_residue_range: Tuple[int, int],
        binder_length_range: Tuple[int, int] = (20, 80),
        target_chain: str = "A",
        n_samples: int = 28,  # Match motif.yaml default
        output_dir: str = "./generated_binders",
        atom_selection_mode: str = "tip_atoms",
        n_steps: int = 400,  # Match inference_base.yaml default
        seed: int = 5,       # Match inference_base.yaml default
        linker_length: int = 0,
        max_batch_size: int = 2,
        contig_string: Optional[str] = None, # Allow override for complex contigs
        design_id: Optional[str] = None  # Short unique ID for naming (used in evolution workflow)
    ) -> List[str]:
        """
        Generate binder candidates for a target protein.
        
        Args:
            target_pdb: Path to target PDB file
            target_residue_range: (start, end) residue numbers of target in the PDB
            binder_length_range: (min, max) length of generated binder
            target_chain: Chain ID of target in PDB (default "A")
            n_samples: Number of binder candidates to generate
            output_dir: Directory to save generated PDBs
            atom_selection_mode: Atom selection for motif conditioning
            n_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            contig_string: Optional custom contig string (e.g. A1-100/10-20/A120-150)
                           If provided, this overrides standard contig construction.
            
        Returns:
            List of paths to generated PDB files
        """
        # Ensure model is loaded
        self._load_model()
        
        # Build contig string
        # Build contig string
        if contig_string:
            logger.info(f"Using custom contig: {contig_string}")
    
            total_min = 50 # minimal dummy
            total_max = 2000 # maximal dummy
        else:
            # Build standard contig string: target_chain + residue range / binder length range
            target_start, target_end = target_residue_range
            binder_min, binder_max = binder_length_range
            
            # Format: A1-705/20-80 (target chain + range / binder length range)
            # Use provided linker length if defined globally or default
            # Use provided linker length
            linker_len = linker_length 
            
            motif_end = target_end + linker_len
            contig_string = f"{target_chain}{target_start}-{motif_end}/{binder_min}-{binder_max}"
            
            # Calculate total length bounds
            target_len = target_end - target_start + 1
            total_min = target_len + linker_len + binder_min
            total_max = target_len + linker_len + binder_max
        
        logger.info(f"Generating binders with contig: {contig_string}")
        
        # Build configs
        motif_cfg = MotifConfig(
            contig_string=contig_string,
            motif_pdb_path=str(Path(target_pdb).absolute()),
            motif_only=True,
            motif_min_length=total_min,
            motif_max_length=total_max,
            segment_order=target_chain,
            atom_selection_mode=atom_selection_mode,
            design_id=design_id,
        )
        
        gen_cfg = GenerationConfig(
            n_samples=n_samples,
            n_steps=n_steps,
            seed=seed,
            max_batch_size=max_batch_size,
        )
        
        # Generate
        return self._run_generation(motif_cfg, gen_cfg, output_dir)
    
    def generate_from_motif_config(
        self,
        motif_config: MotifConfig,
        gen_config: Optional[GenerationConfig] = None,
        output_dir: str = "./generated",
    ) -> List[str]:
        """
        Generate using explicit motif configuration (for advanced use).
        
        Args:
            motif_config: Full motif configuration
            gen_config: Generation parameters (uses defaults if None)
            output_dir: Directory to save generated PDBs
            
        Returns:
            List of paths to generated PDB files
        """
        self._load_model()
        
        if gen_config is None:
            gen_config = GenerationConfig()
        
        return self._run_generation(motif_config, gen_config, output_dir)
    
    def _run_generation(
        self,
        motif_cfg: MotifConfig,
        gen_cfg: GenerationConfig,
        output_dir: str,
    ) -> List[str]:
        """Internal method to run the actual generation."""
        from proteinfoundation.datasets.gen_dataset import GenDataset
        from proteinfoundation.utils.pdb_utils import write_prot_to_pdb
        from proteinfoundation.utils.motif_utils import extract_motif_from_pdb
        from proteinfoundation.utils.align_utils import mean_w_mask
        
        # Calculate original center to un-center later and avoid jumps
        # We use the same extraction logic as the dataset to find the exact center used for normalization
        m_mask, x_m, _ = extract_motif_from_pdb(
            motif_cfg.contig_string, 
            motif_cfg.motif_pdb_path, 
            motif_only=motif_cfg.motif_only,
            atom_selection_mode=motif_cfg.atom_selection_mode,
            coors_to_nm=False # Get center in Angstroms
        )
        # mean_w_mask returns [3] tensor
        original_center = mean_w_mask(x_m.flatten(0, 1), m_mask.flatten(0, 1)).cpu().numpy()
        logger.info(f"Original motif center: {original_center}. Will apply this offset to generated coordinates.")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build config
        cfg = self._build_generation_config(motif_cfg, gen_cfg)
        
        # Set seed
        L.seed_everything(gen_cfg.seed)
        
        # Configure model for inference
        self.model.configure_inference(cfg.generation, nn_ag=None)
        
        # Create dataset
        motif_csv_path = str(output_path / "motif_info.csv")
        dataset = GenDataset(
            motif_csv_path=motif_csv_path,
            **cfg.generation.dataset
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Run prediction
        trainer = L.Trainer(
            accelerator=self.device if self.device != "cpu" else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        predictions = trainer.predict(self.model, dataloader)
        
        # Save generated structures
        generated_pdbs = []
        predictions_flat = [sample for sublist in predictions for sample in sublist]
        
        motif_pdb_name = Path(motif_cfg.motif_pdb_path).stem
        
        for idx, pred in enumerate(predictions_flat):
            coors_atom37, residue_type = pred
            
            # Un-center coordinates to match original PDB space
            # coors_atom37 is [N, 37, 3]
            coors_np = coors_atom37.float().detach().cpu().numpy()
            coors_np = coors_np + original_center
            
            # Use short ID if available (for evolution workflow), otherwise use original naming
            if motif_cfg.design_id:
                dir_name = f"{motif_cfg.design_id}_sample_{idx:03d}"
            else:
                dir_name = f"job_0_id_{idx}_motif_{motif_pdb_name}"
            
            sample_root_path = output_path / dir_name
            sample_root_path.mkdir(parents=True, exist_ok=True)
            
            pdb_name = f"{dir_name}.pdb"
            pdb_path = sample_root_path / pdb_name
            
            # Save PDB with original write_prot_to_pdb parameters
            write_prot_to_pdb(
                prot_pos=coors_np,
                aatype=residue_type.detach().cpu().numpy(),
                file_path=str(pdb_path),
                overwrite=True,
                no_indexing=True,
            )
            
            generated_pdbs.append(str(pdb_path))
            logger.info(f"Saved: {pdb_path}")
        
        logger.info(f"Generated {len(generated_pdbs)} structures in {output_dir}")
        return generated_pdbs


def extract_binder_sequence(
    generated_pdb: str,
    target_length: int,
    linker_length: int = 0,
) -> Tuple[str, int, int]:
    """
    Extract the binder portion from a generated monomer.
    
    LaProteina generates a single chain containing target + binder.
    This function identifies the binder residues.
    
    Args:
        generated_pdb: Path to generated PDB
        target_length: Length of the target protein (known from input)
        linker_length: Length of linker between target and binder (if any)
        
    Returns:
        Tuple of (binder_sequence, start_residue, end_residue)
    """
    from Bio import PDB
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("generated", generated_pdb)
    
    # Get first model, first chain
    model = structure[0]
    chain = list(model.get_chains())[0]
    residues = list(chain.get_residues())
    
    # Binder starts after target + linker
    binder_start = target_length + linker_length
    binder_residues = residues[binder_start:]
    
    # Extract sequence
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    sequence = ""
    for res in binder_residues:
        resname = res.get_resname()
        sequence += three_to_one.get(resname, 'X')
    
    return sequence, binder_start + 1, binder_start + len(binder_residues)
