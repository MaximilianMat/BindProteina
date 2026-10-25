import os
import shutil
from pathlib import Path

def find_and_delete_old_chunks(base_dir: str):
    base = Path(base_dir).resolve()
    print(f"Scanning {base} for orphaned chunk_* directories from Protenix...")
    
    deleted_count = 0
    deleted_size = 0
    
    # We look for directories named 'chunk_*' that are inside 'protenix_batch' or 'protenix_json_batch'
    for batch_type in ["protenix_batch", "protenix_json_batch"]:
        for batch_dir in base.rglob(batch_type):
            if not batch_dir.is_dir():
                continue
            
            for chunk_dir in batch_dir.glob("chunk_*"):
                if not chunk_dir.is_dir():
                    continue
                
                # Check if it was processed. Usually, the final PDBs are extracted and saved
                # in the batch_dir itself. If chunk_dir exists, it's leftover.
                size = get_dir_size(chunk_dir)
                print(f"Found orphaned chunk: {chunk_dir} (Size: {size / (1024**3):.2f} GB)")
                
                try:
                    shutil.rmtree(chunk_dir)
                    deleted_count += 1
                    deleted_size += size
                    print(f"  -> Deleted successfully.")
                except Exception as e:
                    print(f"  -> Failed to delete {chunk_dir}: {e}")

    print(f"\nCleanup complete!")
    print(f"Deleted {deleted_count} chunk directories.")
    print(f"Freed {deleted_size / (1024**3):.2f} GB of disk space.")

def get_dir_size(path: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean up old Protenix chunk directories.")
    parser.add_argument("--dir", type=str, default=".", help="Base directory to scan (e.g. /nfs/students/matscher/samples)")
    args = parser.parse_args()
    
    find_and_delete_old_chunks(args.dir)
