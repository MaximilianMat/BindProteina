import os
import json
from pathlib import Path
import argparse

def parse_condition_from_path(path_str):
    """
    Tries to infer orientation, distance, and fragment from the base_run name,
    and angle from the rotation folder.
    Expected folder naming conventions:
      - sni -> inward, sno -> outward
      - 8A, 10A, 12A, 14A -> distance
      - gly -> single_gly, 3hx -> 3_res_helix, 3bt -> 3_res_beta, 5hx -> 5_res_helix
      - _rot72 -> angle 72
    """
    path = Path(path_str)
    
    orientation = "unknown"
    if "sno" in path_str: orientation = "outward"
    elif "sni" in path_str: orientation = "inward"
    
    distance = 0
    for d in [8, 10, 12, 14]:
        if f"{d}A" in path_str:
            distance = d
            break
            
    fragment = "unknown"
    if "gly" in path_str: fragment = "single_gly"
    elif "3hx" in path_str: fragment = "3_res_helix"
    elif "3bt" in path_str or "3bs" in path_str: fragment = "3_res_beta"
    elif "5hx" in path_str: fragment = "5_res_helix"
    
    angle = 0
    for part in path.parts:
        if part.startswith("_rot"):
            try:
                angle = int(part.replace("_rot", ""))
            except:
                pass
                
    return orientation, angle, distance, fragment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Base directory of ablation study")
    parser.add_argument("--output", default="ablation_results.json", help="Output JSON file")
    parser.add_argument("--residue", help="Optional hotspot residue number to filter by (e.g. '60' for hs60)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    results = []
    
    # Find all rotation folders
    rot_folders = set()
    for p in input_dir.rglob("_rot*"):
        if p.is_dir():
            rot_folders.add(p)
            
    for rot_dir in sorted(rot_folders):
        # Filter by hotspot residue if specified
        if args.residue and f"hs{args.residue}" not in str(rot_dir):
            continue
            
        orientation, angle, distance, fragment = parse_condition_from_path(str(rot_dir))
        
        # If it doesn't match our ablation patterns, skip
        if distance == 0 or fragment == "unknown":
            continue
            
        # Count successes (files in any relaxed/ folder inside this rotation)
        n_success = len(list(rot_dir.rglob("relaxed/*.pdb")))
        
        # Count total samples (files in protenix_batch, or just assume 100 for now, or count generated)
        # To be safe, we count the number of generated fastas or linked samples, 
        # or we can count how many were sent to protenix_batch
        n_samples = len(list(rot_dir.rglob("protenix_batch/*.pdb")))
        if n_samples == 0:
            # Maybe they haven't been scored yet, fallback to 100
            n_samples = 100
            
        success_rate = n_success / n_samples if n_samples > 0 else 0
        
        results.append({
            "orientation": orientation,
            "angle": angle,
            "distance": distance,
            "fragment": fragment,
            "success_rate": round(success_rate, 3),
            "n_samples": n_samples,
            "n_success": n_success
        })
        
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Generated {len(results)} records and saved to {args.output}")

if __name__ == "__main__":
    main()
