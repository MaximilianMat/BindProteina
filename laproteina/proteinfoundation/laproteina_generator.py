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

# Set dummy DATA_PATH if not set - needed for OmegaConf interpolation in model checkpoint
if "DATA_PATH" not in os.environ:
    os.environ["DATA_PATH"] = "."

import torch
import lightning as L
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from loguru import logger
import hydra
from hydra import compose, initialize_config_dir
from contextlib import contextmanager

# Note: Removed torch.set_float32_matmul_precision('medium') - can cause numerical artifacts


@contextmanager
def hydra_config_context(config_dir: str):
    """Context manager for Hydra config initialization."""
    # Use GlobalHydra to check if already initialized
    from hydra.core.global_hydra import GlobalHydra
    
    was_initialized = GlobalHydra.instance().is_initialized()
    if was_initialized:
        GlobalHydra.instance().clear()
    
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            yield
    finally:
        if was_initialized:
            # Re-clear since we modified state
            GlobalHydra.instance().clear()

@dataclass
class MotifConfig:
    """Configuration for motif-conditioned generation."""
    
    contig_string: str
    motif_pdb_path: str
    motif_only: bool = True
    motif_min_length: int = 50
    motif_max_length: int = 150
    segment_order: str = "A"
    atom_selection_mode: str = "all_atom"  # all_atom (LD4), tip_atoms (LD5), ca_only, backbone
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
            checkpoint_path="checkpoints_laproteina/LD4_motif_idx_aa.ckpt",
            autoencoder_path="checkpoints_laproteina/AE3_motif.ckpt",
            laproteina_mode="all_atom",  # "all_atom" for LD4, "tip_atom" for LD5
        )
        
        pdbs = generator.generate_binder(
            target_pdb="my_target.pdb",
            target_residue_range=(1, 705),
            binder_length_range=(20, 80),
            n_samples=10,
            output_dir="./output"
        )
    """
    
    # Map from laproteina_mode to Hydra config name
    MODE_CONFIG_MAP = {
        "all_atom": "inference_motif_idx_aa",
        "tip_atom": "inference_motif_idx_tip",
    }
    
    # Map from laproteina_mode to atom_selection_mode used in motif data processing
    MODE_ATOM_SELECTION_MAP = {
        "all_atom": "all_atom",
        "tip_atom": "tip_atoms",
    }
    
    def __init__(
        self,
        checkpoint_path: str,
        autoencoder_path: Optional[str] = None,
        base_config_path: Optional[str] = None,
        device: str = "cuda",
        laproteina_mode: str = "all_atom",
    ):
        """
        Initialize the generator by loading model weights.
        
        Args:
            checkpoint_path: Path to main model checkpoint (.ckpt)
            autoencoder_path: Path to autoencoder checkpoint (.ckpt), if using latent model
            base_config_path: Optional path to base config directory
            device: Device to run on ("cuda" or "cpu")
            laproteina_mode: "all_atom" for LD4 model, "tip_atom" for LD5 model.
                Controls which Hydra config, feature factory class, and atom selection
                mode are used.
        """
        if laproteina_mode not in self.MODE_CONFIG_MAP:
            raise ValueError(
                f"Unknown laproteina_mode: {laproteina_mode!r}. "
                f"Supported: {list(self.MODE_CONFIG_MAP.keys())}"
            )
        
        self.laproteina_mode = laproteina_mode
        self.checkpoint_path = Path(checkpoint_path)
        self.autoencoder_path = Path(autoencoder_path) if autoencoder_path else None
        self.device = device
        self.model = None
        self._base_config = None
        
        # Set the feature factory module-level variable BEFORE model loading
        import proteinfoundation.nn.feature_factory as ff
        ff.MOTIF_ATOM_MODE = laproteina_mode
        logger.info(f"Set MOTIF_ATOM_MODE = {laproteina_mode!r}")
        
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
        
        logger.info(
            f"LaProteinaGenerator initialized: mode={laproteina_mode}, "
            f"config={self.MODE_CONFIG_MAP[laproteina_mode]}, "
            f"checkpoint={self.checkpoint_path}"
        )
    
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
        """
        Build OmegaConf config for generation using Hydra composition.
        
        This uses the official LaProteina config files via Hydra to ensure
        exact match with the expected config structure and defaults.
        """
        config_dir = str(self._config_dir.absolute())
        
        try:
            # Try to use Hydra to load the official config
            with hydra_config_context(config_dir):
                # Load the motif generation config based on laproteina_mode
                config_name = self.MODE_CONFIG_MAP[self.laproteina_mode]
                logger.info(f"Loading Hydra config: {config_name}")
                cfg = compose(config_name=config_name)
                
                # Override with our specific settings
                cfg.seed = gen_config.seed
                cfg.generation.args.nsteps = gen_config.n_steps
                cfg.generation.dataset.nsamples = gen_config.n_samples
                cfg.generation.dataset.max_nsamples_per_batch = gen_config.max_batch_size
                
                # Set up our custom motif task
                cfg.generation.dataset.motif_task_name = "BINDER_GENERATION"
                
                # Disable all metrics
                cfg.generation.metric.compute_designability = False
                cfg.generation.metric.compute_codesignability = False
                cfg.generation.metric.compute_motif_rmsd = False
                cfg.generation.metric.compute_novelty_pdb = False
                cfg.generation.metric.compute_novelty_afdb = False
                cfg.generation.metric.compute_fid = False
                cfg.generation.metric.keep_folding_outputs = False
                
                # Add our custom motif configuration
                OmegaConf.set_struct(cfg, False)
                cfg.generation.dataset.motif_dict_cfg = {
                    "BINDER_GENERATION": {
                        "contig_string": motif_config.contig_string,
                        "motif_pdb_path": motif_config.motif_pdb_path,
                        "motif_only": motif_config.motif_only,
                        "motif_min_length": motif_config.motif_min_length,
                        "motif_max_length": motif_config.motif_max_length,
                        "segment_order": motif_config.segment_order,
                        "atom_selection_mode": motif_config.atom_selection_mode,
                    }
                }
                OmegaConf.set_struct(cfg, True)
                
                logger.info("Config loaded via Hydra composition")
                return cfg
                
        except Exception as e:
            logger.warning(f"Failed to load config via Hydra: {e}")
            logger.warning("Falling back to manual config construction")
            return self._build_generation_config_manual(motif_config, gen_config)
    
    def _build_generation_config_manual(
        self,
        motif_config: MotifConfig,
        gen_config: GenerationConfig,
    ) -> DictConfig:
        """
        Fallback: Build config manually using OmegaConf.create().
        Used when Hydra config loading fails.
        """
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
                    "ag_ckpt_path": None,
                    "save_trajectory_every": 0,
                },
                "model": {
                    "bb_ca": {
                        "schedule": {
                            "mode": "log",
                            "p": 2.0,
                        },
                        "gt": {
                            "mode": "1/t",
                            "p": 1.0,
                            "clamp_val": None,
                        },
                        "simulation_step_params": {
                            "sampling_mode": "sc",
                            "sc_scale_noise": 0.1,
                            "sc_scale_score": 1.0,
                            "t_lim_ode": 0.98,
                            "t_lim_ode_below": 0.02,
                            "center_every_step": False,
                        }
                    },
                    "local_latents": {
                        "schedule": {
                            "mode": "power",
                            "p": 2.0,
                        },
                        "gt": {
                            "mode": "tan",
                            "p": 1.0,
                            "clamp_val": None,
                        },
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
                    "motif_task_name": "BINDER_GENERATION",
                    "nsamples": gen_config.n_samples,
                    "max_nsamples_per_batch": gen_config.max_batch_size,
                    "motif_dict_cfg": {
                        "BINDER_GENERATION": {
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
                    "compute_designability": False,
                    "compute_codesignability": False,
                    "compute_motif_rmsd": False,
                    "compute_novelty_pdb": False,
                    "compute_novelty_afdb": False,
                    "compute_fid": False,
                    "keep_folding_outputs": False,
                    "metric_factory": None,
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
        atom_selection_mode: Optional[str] = None,
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
            atom_selection_mode: Atom selection for motif conditioning.
                If None, automatically set based on laproteina_mode.
            n_steps: Number of diffusion steps
            seed: Random seed for reproducibility
            contig_string: Optional custom contig string (e.g. A1-100/10-20/A120-150)
                           If provided, this overrides standard contig construction.
            
        Returns:
            List of paths to generated PDB files
        """
        # Default atom_selection_mode based on laproteina_mode
        if atom_selection_mode is None:
            atom_selection_mode = self.MODE_ATOM_SELECTION_MAP[self.laproteina_mode]
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
        )
        predictions = trainer.predict(self.model, dataloader)
        
        # Save generated structures
        generated_pdbs = []
        predictions_flat = [sample for sublist in predictions for sample in sublist]
        
        motif_pdb_name = Path(motif_cfg.motif_pdb_path).stem
        
        for idx, pred in enumerate(predictions_flat):
            coors_atom37, residue_type = pred
            
            # Use short ID if available (for evolution workflow), otherwise use original naming
            if motif_cfg.design_id:
                dir_name = f"{motif_cfg.design_id}_sample_{idx:03d}"
            else:
                dir_name = f"job_0_id_{idx}_motif_{motif_pdb_name}"
            
            sample_root_path = output_path / dir_name
            sample_root_path.mkdir(parents=True, exist_ok=True)
            
            pdb_name = f"{dir_name}.pdb"
            pdb_path = sample_root_path / pdb_name
            
            # Save PDB
            write_prot_to_pdb(
                prot_pos=coors_atom37.float().detach().cpu().numpy(),
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
