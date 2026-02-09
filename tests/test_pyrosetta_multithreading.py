"""
Benchmark test for PyRosetta multithreading performance.

This test compares PyRosetta relaxation and scoring performance
with single-threaded vs multithreaded initialization.
"""

import os
import sys
import time
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyrosetta as pr
from functions.pyrosetta_utils import (
    get_pyrosetta_init_cmd,
    pr_relax,
    score_interface
)


class PyRosettaBenchmark:
    """Benchmark PyRosetta performance with different threading configurations."""
    
    def __init__(self, test_pdb_path: str, num_iterations: int = 3, binder_chain: str = None):
        """
        Initialize benchmark.
        
        Args:
            test_pdb_path: Path to a test PDB file (should have chain A and optionally chain B)
            num_iterations: Number of times to run each test for averaging
            binder_chain: Chain ID for binder (default: auto-detect, tries 'B' then other chains)
        """
        self.test_pdb_path = Path(test_pdb_path)
        if not self.test_pdb_path.exists():
            raise FileNotFoundError(f"Test PDB not found: {test_pdb_path}")
        
        self.num_iterations = num_iterations
        self.repo_root = Path(__file__).parent.parent
        self.dalphaball_path = str(self.repo_root / 'functions' / 'DAlphaBall.gcc')
        
        if not Path(self.dalphaball_path).exists():
            raise FileNotFoundError(f"DAlphaBall not found: {self.dalphaball_path}")
        
        # Use SLURM_CPUS_PER_TASK if available, otherwise auto-detect
        self.default_threads = None
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            try:
                self.default_threads = int(os.environ['SLURM_CPUS_PER_TASK'])
                print(f"Using SLURM_CPUS_PER_TASK: {self.default_threads} threads")
            except ValueError:
                pass
        
        # Detect binder chain if not provided
        self.binder_chain = binder_chain or self._detect_binder_chain()
    
    def _detect_binder_chain(self) -> str:
        """Detect binder chain from PDB file."""
        try:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('test', str(self.test_pdb_path))
            
            # Get all chain IDs
            chains = [chain.id for chain in structure[0]]
            print(f"Detected chains in PDB: {chains}")
            
            # Prefer chain B, otherwise use first non-A chain, or just use B as default
            if 'B' in chains:
                return 'B'
            elif len(chains) > 1:
                # Use first chain that's not A
                for chain_id in chains:
                    if chain_id != 'A':
                        print(f"Using chain {chain_id} as binder chain")
                        return chain_id
            # If only chain A exists, we'll use B anyway and let score_interface handle it
            return 'B'
        except Exception as e:
            print(f"Warning: Could not detect chains, using 'B' as default: {e}")
            return 'B'
    
    def _init_pyrosetta_single_threaded(self):
        """Initialize PyRosetta with single-threaded configuration."""
        cmd = (
            f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all '
            f'-holes:dalphaball {self.dalphaball_path} -corrections::beta_nov16 true '
            f'-relax:default_repeats 1'
        )
        try:
            pr.init(cmd)
        except Exception:
            pass  # Already initialized
    
    def _init_pyrosetta_multithreaded(self, num_threads: int = None):
        """Initialize PyRosetta with multithreaded configuration."""
        cmd = get_pyrosetta_init_cmd(self.dalphaball_path, num_threads)
        try:
            pr.init(cmd)
        except Exception:
            pass  # Already initialized
    
    def benchmark_relax(self, num_threads: int = None) -> Tuple[float, float]:
        """
        Benchmark relaxation performance.
        
        Args:
            num_threads: Number of threads (None = use default from SLURM or auto-detect)
        
        Returns:
            Tuple of (avg_time_single_threaded, avg_time_multithreaded)
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking PyRosetta Relaxation")
        print(f"{'='*60}")
        print(f"Test PDB: {self.test_pdb_path}")
        print(f"Iterations per test: {self.num_iterations}")
        
        if num_threads is None:
            num_threads = self.default_threads or os.cpu_count() or 4
        print(f"Multithreaded config: {num_threads} threads")
        
        # Single-threaded benchmark
        print(f"\n[1/2] Running single-threaded relaxation...")
        single_times = []
        for i in range(self.num_iterations):
            self._init_pyrosetta_single_threaded()
            # Use unique temp file for each iteration
            tmp_path = str(Path(tempfile.gettempdir()) / f"relax_bench_{uuid.uuid4()}.pdb")
            
            # Ensure file doesn't exist
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Verify file doesn't exist before relaxation
            if os.path.exists(tmp_path):
                raise RuntimeError(f"Temp file {tmp_path} already exists!")
            
            start = time.time()
            pr_relax(str(self.test_pdb_path), tmp_path)
            elapsed = time.time() - start
            
            # Verify relaxation actually created the file
            if not os.path.exists(tmp_path):
                raise RuntimeError(f"Relaxation did not create output file {tmp_path}")
            
            single_times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s (output file size: {os.path.getsize(tmp_path)} bytes)")
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        avg_single = sum(single_times) / len(single_times)
        print(f"  Average: {avg_single:.2f}s")
        
        # Multithreaded benchmark
        print(f"\n[2/2] Running multithreaded relaxation ({num_threads} threads)...")
        multi_times = []
        for i in range(self.num_iterations):
            self._init_pyrosetta_multithreaded(num_threads)
            # Use unique temp file for each iteration
            tmp_path = str(Path(tempfile.gettempdir()) / f"relax_bench_{uuid.uuid4()}.pdb")
            
            # Ensure file doesn't exist
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            # Verify file doesn't exist before relaxation
            if os.path.exists(tmp_path):
                raise RuntimeError(f"Temp file {tmp_path} already exists!")
            
            start = time.time()
            pr_relax(str(self.test_pdb_path), tmp_path)
            elapsed = time.time() - start
            
            # Verify relaxation actually created the file
            if not os.path.exists(tmp_path):
                raise RuntimeError(f"Relaxation did not create output file {tmp_path}")
            
            multi_times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.2f}s (output file size: {os.path.getsize(tmp_path)} bytes)")
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        avg_multi = sum(multi_times) / len(multi_times)
        print(f"  Average: {avg_multi:.2f}s")
        
        # Results
        speedup = avg_single / avg_multi if avg_multi > 0 else 0
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Single-threaded: {avg_single:.2f}s ± {self._std_dev(single_times):.2f}s")
        print(f"  Multithreaded:   {avg_multi:.2f}s ± {self._std_dev(multi_times):.2f}s")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        return avg_single, avg_multi
    
    def benchmark_scoring(self, num_threads: int = None) -> Tuple[float, float]:
        """
        Benchmark interface scoring performance.
        
        Args:
            num_threads: Number of threads (None = use default from SLURM or auto-detect)
        
        Returns:
            Tuple of (avg_time_single_threaded, avg_time_multithreaded)
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking PyRosetta Interface Scoring")
        print(f"{'='*60}")
        print(f"Test PDB: {self.test_pdb_path}")
        print(f"Iterations per test: {self.num_iterations}")
        
        if num_threads is None:
            num_threads = self.default_threads or os.cpu_count() or 4
        print(f"Multithreaded config: {num_threads} threads")
        
        # Single-threaded benchmark
        print(f"\n[1/2] Running single-threaded scoring...")
        print(f"Using binder chain: {self.binder_chain}")
        single_times = []
        for i in range(self.num_iterations):
            self._init_pyrosetta_single_threaded()
            
            try:
                start = time.time()
                score_interface(str(self.test_pdb_path), binder_chain=self.binder_chain)
                elapsed = time.time() - start
                single_times.append(elapsed)
                print(f"  Iteration {i+1}: {elapsed:.2f}s")
            except Exception as e:
                print(f"  Iteration {i+1}: ERROR - {e}")
                raise
        
        avg_single = sum(single_times) / len(single_times)
        print(f"  Average: {avg_single:.2f}s")
        
        # Multithreaded benchmark
        print(f"\n[2/2] Running multithreaded scoring ({num_threads} threads)...")
        print(f"Using binder chain: {self.binder_chain}")
        multi_times = []
        for i in range(self.num_iterations):
            self._init_pyrosetta_multithreaded(num_threads)
            
            try:
                start = time.time()
                score_interface(str(self.test_pdb_path), binder_chain=self.binder_chain)
                elapsed = time.time() - start
                multi_times.append(elapsed)
                print(f"  Iteration {i+1}: {elapsed:.2f}s")
            except Exception as e:
                print(f"  Iteration {i+1}: ERROR - {e}")
                raise
        
        avg_multi = sum(multi_times) / len(multi_times)
        print(f"  Average: {avg_multi:.2f}s")
        
        # Results
        speedup = avg_single / avg_multi if avg_multi > 0 else 0
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Single-threaded: {avg_single:.2f}s ± {self._std_dev(single_times):.2f}s")
        print(f"  Multithreaded:   {avg_multi:.2f}s ± {self._std_dev(multi_times):.2f}s")
        print(f"  Speedup:         {speedup:.2f}x")
        print(f"{'='*60}\n")
        
        return avg_single, avg_multi
    
    def run_full_benchmark(self, num_threads: int = None):
        """
        Run complete benchmark suite.
        
        Args:
            num_threads: Number of threads (None = use default from SLURM or auto-detect)
        """
        print(f"\n{'#'*60}")
        print(f"PyRosetta Multithreading Benchmark")
        print(f"{'#'*60}")
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            print(f"SLURM CPUs allocated: {os.environ['SLURM_CPUS_PER_TASK']}")
        print(f"CPU cores detected: {os.cpu_count()}")
        print(f"Test PDB: {self.test_pdb_path}")
        print(f"Iterations per test: {self.num_iterations}")
        
        if num_threads is None:
            num_threads = self.default_threads or os.cpu_count() or 4
        print(f"Multithreaded config: {num_threads} threads")
        print(f"{'#'*60}\n")
        
        # Benchmark relaxation
        relax_single, relax_multi = self.benchmark_relax(num_threads)
        
        # Benchmark scoring
        score_single, score_multi = self.benchmark_scoring(num_threads)
        
        # Summary
        relax_speedup = relax_single / relax_multi if relax_multi > 0 else 0
        score_speedup = score_single / score_multi if score_multi > 0 else 0
        
        print(f"\n{'#'*60}")
        print(f"SUMMARY")
        print(f"{'#'*60}")
        print(f"Relaxation speedup: {relax_speedup:.2f}x")
        print(f"Scoring speedup:    {score_speedup:.2f}x")
        print(f"Overall improvement: {((relax_speedup + score_speedup) / 2):.2f}x")
        print(f"{'#'*60}\n")
    
    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


def main():
    """Run benchmark if executed directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark PyRosetta multithreading")
    parser.add_argument(
        "--pdb",
        type=str,
        required=True,
        help="Path to test PDB file (should have chain A and chain B)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per test (default: 3)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads for multithreaded test (default: auto-detect)"
    )
    parser.add_argument(
        "--binder-chain",
        type=str,
        default=None,
        help="Chain ID for binder (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    benchmark = PyRosettaBenchmark(args.pdb, args.iterations, binder_chain=args.binder_chain)
    benchmark.run_full_benchmark(args.threads)


if __name__ == "__main__":
    main()
