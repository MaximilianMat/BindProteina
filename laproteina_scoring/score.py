import argparse
import os
import sys

# Set JAX/XLA environment variables BEFORE any JAX/XLA imports
# This prevents JAX from preallocating all VRAM and helps avoid cuSolver errors
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"  # Show full JAX tracebacks for debugging

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Optimize for A100 Tensor Cores
import torch
torch.set_float32_matmul_precision('medium')

# Ensure we can import from BindCraft root
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def main():
    # Lazy imports to ensure environment variables are set and cleanup has a chance to work
    from laproteina_scoring.config import ScoringConfig
    from laproteina_scoring.pipeline import LaProteinaScorer

    parser = argparse.ArgumentParser(description="LaProteina-AF2 Scoring Pipeline")
    parser.add_argument("--input_dir", "-i", type=str, help="Directory containing PDB files to score")
    parser.add_argument("--output_dir", "-o", type=str, help="Directory to save results")
    parser.add_argument("--settings", "-s", type=str, help="Path to target settings JSON file")
    parser.add_argument("--filters", "-f", type=str, help="Path to filters JSON file")
    
    args = parser.parse_args()
    
    # 1. Setup Configuration
    config = ScoringConfig()
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Load settings from JSON if provided
    if args.settings:
        with open(args.settings, 'r') as f:
            settings_data = json.load(f)
            
        # Parse input/output dirs from settings (CLI overrides)
        if "input_dir" in settings_data and not input_dir:
            input_dir = settings_data["input_dir"]
        if "output_dir" in settings_data and not output_dir:
            output_dir = settings_data["output_dir"]
            
        # Parse chains from settings (e.g. "A,B") or default to config
        if "chains" in settings_data:
            config.target_chains = settings_data["chains"].split(",")
            
        # Parse hotspot residues (will be used for metrics later)
        if "target_hotspot_residues" in settings_data:
            config.target_hotspot_residues = settings_data["target_hotspot_residues"]
    
    # Validate required args
    if not input_dir or not output_dir:
        parser.error("--input_dir and --output_dir are required (either via CLI or settings file)")
    
    # Load filters (default to BindCraft default_filters.json if not provided)
    filters = {}
    filter_path = args.filters
    
    scoring_dir = os.path.dirname(os.path.realpath(__file__))
    bindcraft_root = os.path.dirname(scoring_dir)
    
    if not filter_path:
        default_filter_path = os.path.join(bindcraft_root, "settings_filters", "laProteina_filters.json")
        if os.path.exists(default_filter_path):
            filter_path = default_filter_path
            print(f"No filters specified, using default: {filter_path}")
            
    if filter_path:
        with open(filter_path, 'r') as f:
            filters = json.load(f)
            
    # Initialize paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Check for LaProteina Generation
    laproteina_results = None
    if args.settings and "laproteina_settings" in settings_data:
        lp_settings = settings_data["laproteina_settings"]
        print("\n" + "="*60)
        print("LaProteina Generation Phase")
        print("="*60)
        
        # Determine target PDB
        target_pdb = settings_data.get("starting_pdb")
        if not target_pdb:
            # Fallback: try to find a PDB in input_dir with binder_name
            binder_name = settings_data.get("binder_name", "target")
            potential_pdb = Path(input_dir) / f"{binder_name}.pdb"
            if potential_pdb.exists():
                target_pdb = str(potential_pdb)
            else:
                # Last resort: use input_dir as the path if it ends in .pdb
                if input_dir.endswith(".pdb"):
                    target_pdb = input_dir
        
        if not target_pdb or not os.path.exists(target_pdb):
            print(f"Error: LaProteina requires a target PDB. Please specify 'starting_pdb' in {args.settings}")
            return

        import subprocess
        
        # Run generation and extraction
        lp_output_dir = output_path / "laproteina_gen"
        lp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for hotspots
        hotspot_str = settings_data.get("target_hotspot_residues", "")
        from laproteina.proteinfoundation.utils.pdb_slicer import parse_hotspot_string, slice_pdb_by_ranges
        hotspots = parse_hotspot_string(hotspot_str)
        
        # Handle multiple hotspots or full target
        if not hotspots:
            from laproteina.proteinfoundation.config_builder import get_pdb_chain_length
            target_len = get_pdb_chain_length(target_pdb, "A")
            hotspot_tasks = [(target_pdb, None, lp_settings.get("n_samples", 10), target_len)]
        else:
            n_total = lp_settings.get("n_samples", 10)
            batch_size = lp_settings.get("batch_size", lp_settings.get("batchSize", 4))
            
            # Distribute samples: at least one batch per hotspot
            samples_per_hotspot = max(batch_size, n_total // len(hotspots))
            hotspot_tasks = []
            
            for i, hs in enumerate(hotspots):
                hs_pdb = str(lp_output_dir / f"target_hs_{i}.pdb")
                # Slice with renumber=True to ensure motif starts at 1
                slice_pdb_by_ranges(target_pdb, hs_pdb, [hs], renumber=True)
                # Length of the slice
                hs_len = hs[1] - hs[0] + 1
                hotspot_tasks.append((hs_pdb, target_pdb, samples_per_hotspot, hs_len))
        
        print(f"Starting generation for {len(hotspot_tasks)} hotspot regions...")
        
        import subprocess
        
        # Determine paths
        scoring_dir = os.path.dirname(os.path.realpath(__file__))
        bindcraft_root = os.path.dirname(scoring_dir)
        cli_path = os.path.join(bindcraft_root, "laproteina", "proteinfoundation", "pipeline_cli.py")
        
        linker_length = lp_settings.get("linker_length", 0)
        linker_sequence = lp_settings.get("linker_sequence", "GGGGS")
        
        all_extracted_complexes = []
        
        for i, (current_target, reference_target, n_samples, hs_len) in enumerate(hotspot_tasks):
            print(f"\n--- Hotspot {i+1}/{len(hotspot_tasks)} ---")
            
            # 1. Build linker for this target/slice
            # 1. Fold linker for this target/slice using AF2
            if linker_length > 0:
                print(f"Folding automated AF2 linker for hotspot...")
                try:
                    from laproteina.proteinfoundation.utils.linker_folder import fold_linked_pdb
                    autoinfolded_pdb = str(lp_output_dir / f"target_autofolded_hs{i}.pdb")
                    
                    # Compute full linker sequence
                    full_linker_seq = (linker_sequence * (linker_length // len(linker_sequence) + 1))[:linker_length]
                    
                    # Attempt to find af_params_dir (default to root/params)
                    af_params_dir = os.path.join(bindcraft_root, "params")
                    if not os.path.exists(af_params_dir):
                        af_params_dir = "./params"

                    current_target = fold_linked_pdb(
                        target_pdb=current_target,
                        linker_sequence=full_linker_seq,
                        output_pdb=autoinfolded_pdb,
                        af_params_dir=af_params_dir
                    )
                except Exception as e:
                    print(f"Warning: Automated AF2 linker folding failed: {e}")
            
            # 2. Run generation subprocess
            hs_output_dir = lp_output_dir / f"hs_{i}"
            hs_output_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                sys.executable,
                cli_path,
                "--target_pdb", current_target,
                "--output_dir", str(hs_output_dir),
                "--checkpoint", lp_settings.get("checkpoint_path"),
                "--n_samples", str(n_samples),
                "--target_length", str(hs_len),
                "--binder_min", str(lp_settings.get("binder_length_range", [20, 80])[0]),
                "--binder_max", str(lp_settings.get("binder_length_range", [20, 80])[1]),
                "--linker_length", str(linker_length),
                "--batch_size", str(batch_size),
                "--atom_mode", lp_settings.get("atom_selection_mode", "tip_atoms"),
                "--skip_scoring"
            ]
            
            if reference_target:
                cmd.extend(["--reference_pdb", reference_target])
            
            if lp_settings.get("autoencoder_path"):
                cmd.extend(["--autoencoder", lp_settings.get("autoencoder_path")])
            
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Collect results
            hs_complexes = list((hs_output_dir / "extracted" / "complexes").glob("*.pdb"))
            all_extracted_complexes.extend(hs_complexes)

        # Update input_dir to the aggregated complexes
        # We move them to a main extracted folder for the rest of score.py to find
        final_complex_dir = lp_output_dir / "extracted" / "complexes"
        final_complex_dir.mkdir(parents=True, exist_ok=True)
        
        for i, source_pdb in enumerate(all_extracted_complexes):
            import shutil
            shutil.copy(source_pdb, final_complex_dir / f"final_complex_{i:04d}.pdb")
            
        input_dir = str(final_complex_dir)
        input_path = Path(input_dir)
        print(f"\nGeneration complete. Total aggregated candidates: {len(all_extracted_complexes)}")
        print(f"Aggregated complexes saved to: {input_dir}")
        print("="*60 + "\n")

    # 3. Initialize Scorer
    print("Initializing LaProteinaScorer...")
    scorer = LaProteinaScorer(config)
    
    # 4. Process Input Directory (re-initialize in case generation updated them)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    pdb_files = list(input_path.glob("*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {input_path}")
        return
        
    print(f"Found {len(pdb_files)} PDB files to score.")
    
    all_results = []
    
    # Define metric mappings (our column name -> filter key template)
    metric_filters = {
        "pLDDT": "{model}_pLDDT",
        "pTM": "{model}_pTM",
        "i_pTM": "{model}_i_pTM",
        "pAE": "{model}_pAE",
        "i_pAE": "{model}_i_pAE",
        "binder_score": "{model}_Binder_Energy_Score",
        "surface_hydrophobicity": "{model}_Surface_Hydrophobicity",
        "interface_sc": "{model}_ShapeComplementarity",
        "interface_packstat": "{model}_PackStat",
        "interface_dG": "{model}_dG",
        "interface_dSASA": "{model}_dSASA",
        "interface_nres": "{model}_n_InterfaceResidues",
        "interface_interface_hbonds": "{model}_n_InterfaceHbonds",
        "interface_delta_unsat_hbonds": "{model}_n_InterfaceUnsatHbonds",
        "interface_hydrophobicity": "{model}_Interface_Hydrophobicity",
    }
    
    def check_metric_filter(row, metric_name, filter_key_template, filters_dict):
        """Check if a metric passes its filter."""
        model_num = row.get("model", 1)
        filter_key = filter_key_template.format(model=model_num)
        
        # If no filter exists for this model, try falling back to model 1
        if filter_key not in filters_dict:
            fallback_key = filter_key_template.format(model=1)
            if fallback_key in filters_dict:
                filter_key = fallback_key
            else:
                return "SKIPPED"
            
        filter_spec = filters_dict[filter_key]
        threshold = filter_spec.get("threshold")
        
        if threshold is None and model_num != 1:
            fallback_key = filter_key_template.format(model=1)
            if fallback_key in filters_dict:
                threshold = filters_dict[fallback_key].get("threshold")
                filter_spec = filters_dict[fallback_key]
        
        if threshold is None:
            return "SKIPPED"
            
        value = row.get(metric_name)
        if value is None or value == "":
            return False
            
        higher = filter_spec.get("higher", True)
        if higher:
            return value >= threshold
        else:
            return value <= threshold
            
    # 4. Score Loop
    # Periodic memory clearing to prevent cuSolver handle exhaustion
    # Use gc.collect() only - clear_mem() is too aggressive and causes "Array has been deleted" errors
    clear_every_n = 5  # Clear memory every N designs (increased from 3)
    
    for idx, pdb_file in enumerate(tqdm(pdb_files, desc="Scoring Designs")):
        # Periodic memory cleanup - only gc.collect(), not clear_mem()
        # Removed artificial delays (0.2s, 0.5s) now that cuDNN is fixed
        if idx > 0 and idx % clear_every_n == 0:
            try:
                import gc
                gc.collect()
            except:
                pass
        
        design_name = pdb_file.stem
        
        # Use the input PDB itself as the target structure source (Chain A)
        target_pdb = str(pdb_file)
        
        try:
            results = scorer.score_design(
                pdb_path=str(pdb_file),
                target_pdb=target_pdb,
                output_dir=str(output_path),
                design_name=design_name
            )
        except Exception as e:
            # Handle cuSolver errors gracefully
            error_msg = str(e)
            if "cuSolver" in error_msg or "gpusolverDnCreate" in error_msg:
                print(f"cuSolver error for {design_name}: {error_msg}")
                # Try to clear memory and continue - only gc.collect(), not clear_mem()
                try:
                    import gc
                    gc.collect()
                except:
                    pass
                all_results.append({
                    "design_name": design_name,
                    "status": "failed",
                    "reason": f"cuSolver error: {error_msg[:100]}"
                })
                continue
            else:
                raise
        
        if "error" in results:
            print(f"Failed {design_name}: {results['error']}")
            all_results.append({
                "design": design_name,
                "status": "failed",
                "reason": results["error"]
            })
            continue
            
        # Flatten results for CSV
        if "models" in results:
            for model_id, data in results["models"].items():
                flat_data = data.copy()
                flat_data["status"] = "scored"
                
                # Check metrics against filters
                if filters:
                    passed_all = True
                    for metric_name, filter_template in metric_filters.items():
                        res = check_metric_filter(flat_data, metric_name, filter_template, filters)
                        flat_data[f"{metric_name}_passed"] = res
                        if res == False:
                            passed_all = False
                    
                    flat_data["overall_passed"] = passed_all
                    if not passed_all:
                        flat_data["status"] = "failed_filter"
                
                all_results.append(flat_data)
        else:
            # Should not happen if no error and no models?
            pass

    # 5. Save Summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Return all model metrics
        
        # Reorder columns to put design/model first
        cols = ["design_name", "model", "status"] + [c for c in df.columns if c not in ["design_name", "model", "status", "sequence"]] + ["sequence"]
        # Filter cols that actually exist
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        csv_path = output_path / "scoring_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"Scoring complete. Results saved to {csv_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
