import os
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

# BindCraft imports
from colabdesign import mk_afdesign_model, clear_mem
from functions.colabdesign_utils import predict_binder_complex
from functions.pyrosetta_utils import score_interface
from colabdesign.shared.utils import copy_dict

# Local imports
from .config import ScoringConfig
from .utils import extract_sequence_from_pdb

import pyrosetta as pr

class LaProteinaScorer:
    def __init__(self, config: ScoringConfig):
        self.config = config
        
        # Randomly select N models from the available 5
        import random
        all_models = list(range(5)) # Standard AF2 has 5 models
        num_to_select = min(self.config.af2_settings.num_models, 5)
        self.prediction_models = sorted(random.sample(all_models, num_to_select))
        
        print(f"LaProteinaScorer initialized to use random models: {[m+1 for m in self.prediction_models]}")
        
        self.complex_model = None
        
        # Initialize AF2 model (lazy load or immediate?)
        # Better to initialize once to save compilation time
        self._init_pyrosetta()
        self._init_af2_model()

    def _init_pyrosetta(self):
        """Initializes PyRosetta with BindCraft settings."""
        # Locate DAlphaBall relative to repo root
        bindcraft_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        dalphaball_path = os.path.join(bindcraft_root, 'functions', 'DAlphaBall.gcc')
        
        cmd = f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1'
        try:
            pr.init(cmd)
        except Exception:
            pass # Already initialized

    def _init_af2_model(self):
        """Initializes the AF2 model for complex prediction."""
        print("Initializing AF2 model...")
        clear_mem()
        self.complex_model = mk_afdesign_model(
            protocol="binder",
            num_recycles=self.config.af2_settings.num_recycles,
            data_dir=self.config.af_params_dir,
            use_multimer=self.config.af2_settings.use_multimer,
            use_initial_guess=False, # Standard prediction
            use_initial_atom_pos=False
        )
        print("AF2 model initialized.")

    def score_design(self, 
                     pdb_path: str, 
                     target_pdb: str, 
                     output_dir: str, 
                     design_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Scores a single design PDB.
        
        Args:
            pdb_path: Path to the input PDB (Target+Binder).
            target_pdb: Path to the original target PDB (for template masking if needed).
            output_dir: Directory to save results.
            design_name: Optional name, defaults to filename stem.
            
        Returns:
            Dictionary containing metrics.
        """
        if design_name is None:
            design_name = Path(pdb_path).stem
            
        # 1. Extract Sequence
        try:
            binder_seq = extract_sequence_from_pdb(pdb_path, self.config.binder_chain)
        except Exception as e:
            print(f"Error extracting sequence from {pdb_path}: {e}")
            return {"design_name": design_name, "error": f"Sequence extraction failed: {e}"}

        if not binder_seq:
            return {"design_name": design_name, "error": "No binder sequence found (Chain B missing?)"}
            
        length = len(binder_seq)
        
        # 2. Prepare Design Paths (Mocking BindCraft structure)
        # predict_binder_complex expects: design_paths["MPNN"], design_paths["MPNN/Relaxed"]
        design_paths = {
            "MPNN": os.path.join(output_dir, "af2_predictions"),
            "MPNN/Relaxed": os.path.join(output_dir, "relaxed"),
            # Needed for failure csv updates if filter fails
        }
        os.makedirs(design_paths["MPNN"], exist_ok=True)
        os.makedirs(design_paths["MPNN/Relaxed"], exist_ok=True)
        
        # failure_csv path (dummy for now, or real if we want to track)
        failure_csv = os.path.join(output_dir, "failures.csv") # predict_binder_complex writes to this
        if not os.path.exists(failure_csv):
             pd.DataFrame(columns=["design", "reason"]).to_csv(failure_csv, index=False) # Simplified
        
        # 3. Predict Complex (AF2)
        # We use strict BindCraft re-prediction
        # Note: predict_binder_complex uses 'trajectory_pdb' for alignment in some cases, 
        # but mostly for MPNN logic. Here we just pass target_pdb or input pdb as trajectory?
        # Examining predict_binder_complex: it doesn't use trajectory_pdb for complex prediction logic itself,
        # unless it's doing something specific with initial guess which we disabled.
        # However, it calls pr_relax on the output.
        
        # Prepare "advanced_settings" dict that BindCraft functions expect
        advanced_settings = {
            "num_recycles_validation": self.config.af2_settings.num_recycles,
            "rm_template_seq_predict": True, 
            "rm_template_sc_predict": True,
            # We don't have rejection checks here yet
        }
        
        # Filters: Empty for now to force "pass" and get metrics
        # If we pass filters, it returns False if they fail.
        # We can pass an empty dict to skip filtering inside the function, 
        # OR we can pass the real filters if we want to skip relaxing bad designs.
        filters = {} 
        
        # Pre-prep inputs for the model for this specific target/length
        self.complex_model.prep_inputs(
            pdb_filename=target_pdb,
            chain=",".join(self.config.target_chains),
            binder_len=length,
            rm_target_seq=advanced_settings["rm_template_seq_predict"],
            rm_target_sc=advanced_settings["rm_template_sc_predict"]
        )
        
        # Force cleanup of existing predictions to ensure we get metrics
        # BindCraft's predict_binder_complex skips prediction (and thus stats generation) if files exist.
        for model_num in self.prediction_models:
             complex_pdb = os.path.join(design_paths["MPNN"], f"{design_name}_model{model_num+1}.pdb")
             if os.path.exists(complex_pdb):
                 os.remove(complex_pdb)
        
        try:
            prediction_stats, pass_af2 = predict_binder_complex(
                self.complex_model,
                binder_seq,
                design_name,
                target_pdb,
                ",".join(self.config.target_chains),
                length,
                pdb_path, # trajectory_pdb - acting as reference??
                self.prediction_models,
                advanced_settings,
                filters,
                design_paths,
                failure_csv
            )
        except Exception as e:
             print(f"Error during AF2 prediction for {design_name}: {e}")
             return {"design_name": design_name, "error": f"AF2 prediction failed: {e}"}

        # 4. Interface Scoring (PyRosetta)
        # If pass_af2 is True (which it should be with empty filters), 
        # predict_binder_complex HAS ALREADY RUN RELAXATION!
        # It calls pr_relax() internally.
        
        # Now we just need to run score_interface on the *best* model or all models.
        # Let's iterate over models as BindCraft does.
        
        final_metrics = {}
        
        for model_num in self.prediction_models:
             model_id = model_num + 1
             relaxed_pdb = os.path.join(design_paths["MPNN/Relaxed"], f"{design_name}_model{model_id}.pdb")
             
             if not os.path.exists(relaxed_pdb):
                 continue
                 
             # Score Interface
             try:
                 interface_scores, interface_AA, interface_residues = score_interface(relaxed_pdb, self.config.binder_chain)
             except Exception as e:
                 print(f"Rosetta scoring failed for {design_name} model {model_id}: {e}")
                 continue
                 
             # Combine AF2 metrics and Rosetta metrics
             af2_stats = prediction_stats.get(model_id, {})
             
             model_data = {
                 "design_name": design_name,
                 "model": model_id,
                 "sequence": binder_seq,
                 "binder_length": len(binder_seq),
                 **af2_stats,
                 **interface_scores
             }
             
             final_metrics[model_id] = model_data
             
        # Clear memory after each design to prevent cuSolver handle pool exhaustion
        # Only do gc.collect() here - clear_mem() is too aggressive and causes "Array has been deleted" errors
        try:
            import gc
            gc.collect()
        except:
            pass
        
        # Return all model metrics
        return {"design_name": design_name, "models": final_metrics}
