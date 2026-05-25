"""
Local ↔ cluster NFS path translation.

The user mounts the cluster NFS locally under ~/remote/nfs/,
so /nfs/... on the cluster becomes ~/remote/nfs/... locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class PathTranslator:
    def __init__(self, local_prefix: str, cluster_prefix: str):
        self.local_prefix = local_prefix.rstrip("/")
        self.cluster_prefix = cluster_prefix.rstrip("/")

    def to_cluster(self, local_path: str) -> str:
        """Convert a local NFS path to the cluster-side path stored in settings JSON."""
        p = str(local_path)
        if p.startswith(self.local_prefix):
            return self.cluster_prefix + p[len(self.local_prefix):]
        return p  # already a cluster path or unrelated

    def to_local(self, cluster_path: str) -> str:
        """Convert a cluster path to the locally mounted path."""
        p = str(cluster_path)
        if p.startswith(self.cluster_prefix):
            return self.local_prefix + p[len(self.cluster_prefix):]
        return p

    def local_exists(self, cluster_path: str) -> bool:
        return Path(self.to_local(cluster_path)).exists()
