import json
from pathlib import Path

indices = [
    24, 27, 27, 40, 40, 40, 44, 47, 
    74, 75, 75, 76, 77, 77, 78, 78, 79, 83, 
    90, 90, 90, 90, 98, 99, 
    107, 108, 110, 111, 111, 113, 115, 116, 119, 
    120, 120, 121, 122, 125, 126, 128, 128, 132, 133, 134, 134
]

file_path = Path("/Users/maximilianmatscher/remote/nfs/homedirs/matscher/targets/ulocuplumab_variants.json")

with open(file_path, 'r') as f:
    data = json.load(f)

if len(data) != len(indices):
    print(f"Warning: JSON has {len(data)} entries but you provided {len(indices)} indices.")
    # We will process as many as possible
    
for i, (entry, idx) in enumerate(zip(data, indices)):
    entry["name"] = f"ulocuplumab_{idx}"

with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Successfully updated {min(len(data), len(indices))} names.")
