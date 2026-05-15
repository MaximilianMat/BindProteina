import os
from pathlib import Path
from typing import Dict, Any

def run_pyrosetta_scoring(
    af2_pdb: str,
    output_dir: Path,
    design_name: str,
    dalphaball_path: str,
    target_chains: str = "A",
    binder_chains: str = "B",
) -> Dict[str, Any]:
    """
    Relaxes the Protenix/AF2 prediction and scores the interface.

    Args:
        target_chains: Chain ID(s) forming the target (default "A").
        binder_chains: Chain ID(s) forming the binder (default "B").
                       Multi-chain binders are supported, e.g. "BC" for heavy+light.
    """
    import pyrosetta as pr
    from functions.pyrosetta_utils import get_pyrosetta_init_cmd, pr_relax, score_interface

    try:
        pr.init(get_pyrosetta_init_cmd(dalphaball_path))
    except Exception:
        pass  # Already initialized

    relaxed_pdb = output_dir / "relaxed" / f"{design_name}_relaxed.pdb"
    relaxed_pdb.parent.mkdir(parents=True, exist_ok=True)

    pr_relax(af2_pdb, str(relaxed_pdb))
    interface_scores, _, _ = score_interface(
        str(relaxed_pdb),
        target_chains=target_chains,
        binder_chains=binder_chains,
    )

    return interface_scores
