#===- act/back_end/validation/reproducibility.py - Reproducibility Tools -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025 ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
#===---------------------------------------------------------------------====#

"""
Reproducibility Tools for Experiment Validation

Ensures all experiments can be exactly reproduced with the same seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from act.back_end.core import Net
    from act.back_end.net_factory import NetFactory


# ============================================================================
# Seed Management
# ============================================================================

def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Set all random number generator seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Enable fully deterministic mode (may reduce performance)

    Note:
        Must be called BEFORE any random operations!
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def derive_seed(base_seed: int, *components) -> int:
    """
    Derive a new seed from a base seed and additional components.

    Args:
        base_seed: Base seed value
        *components: Additional components to differentiate seeds

    Returns:
        Derived 32-bit unsigned integer seed

    Examples:
        >>> derive_seed(42, "rq1", 0)  # RQ1 experiment, network 0
        >>> derive_seed(42, "rq1", 0, "weights")  # Weight initialization
        >>> derive_seed(42, "rq1", 0, "scc", 5)  # SCC sample 5
    """
    payload = "|".join(str(c) for c in [base_seed] + list(components))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def hash_file(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()[:16]}"


def hash_dict(d: Dict) -> str:
    """Compute hash of a dictionary."""
    serialized = json.dumps(d, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(serialized.encode()).hexdigest()[:16]}"


def get_git_commit() -> str:
    """Get current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def get_environment_info() -> Dict[str, str]:
    """Get environment information for reproducibility."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    if torch.cuda.is_available():
        info["cuda"] = str(torch.version.cuda)
        info["cudnn"] = str(torch.backends.cudnn.version())
        info["gpu"] = torch.cuda.get_device_name(0)

    return info


@dataclass
class ExperimentMetadata:
    """
    Experiment metadata for reproducibility.

    Records all information needed to reproduce an experiment exactly.
    """
    # Basic information
    experiment_id: str
    description: str = ""

    # Seed information
    master_seed: int = 42
    seed_offset: int = 0

    # Configuration
    config_path: Optional[str] = None
    config_hash: Optional[str] = None

    # Version control
    git_commit: str = field(default_factory=get_git_commit)

    # Environment
    environment: Dict[str, str] = field(default_factory=get_environment_info)

    # Timestamps
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None

    # Network records
    num_networks: int = 0
    networks: List[Dict[str, Any]] = field(default_factory=list)

    def get_experiment_seed(self) -> int:
        """Get the experiment-specific seed."""
        return self.master_seed + self.seed_offset

    def get_network_seed(self, network_idx: int, instance_id: str = "") -> int:
        """Get seed for a specific network."""
        return derive_seed(self.get_experiment_seed(), network_idx, instance_id)

    def record_network(self, name: str, seed: int, **kwargs) -> None:
        """Record a generated network."""
        record = {
            "name": name,
            "seed": seed,
            "index": len(self.networks),
            **kwargs
        }
        self.networks.append(record)
        self.num_networks = len(self.networks)

    def finalize(self) -> None:
        """Finalize the experiment, record end time."""
        self.end_time = datetime.now().isoformat()

    def save(self, path: str) -> None:
        """Save metadata to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentMetadata":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ReproducibleNetFactory:
    """
    Reproducible network factory wrapper.

    Ensures each network generation has a deterministic seed
    and records all information for reproducibility.
    """

    def __init__(
        self,
        base_factory: "NetFactory",
        metadata: ExperimentMetadata
    ):
        """
        Args:
            base_factory: NetFactory instance to wrap
            metadata: Experiment metadata for recording
        """
        self.factory = base_factory
        self.metadata = metadata
        self._network_idx = 0
        self._generated_nets: List["Net"] = []

    def generate_one(self, **kwargs) -> "Net":
        """
        Generate a single network with reproducible seed.

        Returns:
            Generated Net object
        """
        # Derive network seed
        temp_id = f"net_{self._network_idx}"
        network_seed = self.metadata.get_network_seed(self._network_idx, temp_id)

        # Set seeds before generation
        set_all_seeds(network_seed)

        # Sample and build network
        instance = self.factory._sample_instance(self._network_idx)
        name = instance["instance_id"]

        # Get dtype from config
        dtype = self.factory.config.get("common", {}).get("dtype", "torch.float32")
        spec = self.factory._build_network_spec(instance, dtype=dtype)
        net = self.factory.create_network(name, spec)

        # Record network info
        self.metadata.record_network(
            name=name,
            seed=network_seed,
            family=instance["family"],
            input_shape=list(instance["model_cfg"]["input_shape"]),
            num_classes=instance["model_cfg"].get("num_classes", 10),
            input_spec_kind=instance["input_spec"]["kind"],
            output_spec_kind=instance["output_spec"]["kind"],
        )

        self._generated_nets.append(net)
        self._network_idx += 1

        return net

    def generate_batch(self, n: int) -> List["Net"]:
        """
        Generate a batch of networks.

        Args:
            n: Number of networks to generate

        Returns:
            List of generated Net objects
        """
        return [self.generate_one() for _ in range(n)]

    def get_network_by_index(self, idx: int) -> "Net":
        """Get a generated network by index."""
        if idx >= len(self._generated_nets):
            raise IndexError(f"Network index {idx} not generated yet")
        return self._generated_nets[idx]

    def get_network_seed(self, idx: int) -> int:
        """Get the seed used for a specific network."""
        if idx >= len(self.metadata.networks):
            raise IndexError(f"Network index {idx} not recorded")
        return self.metadata.networks[idx]["seed"]


def verify_reproducibility(
    expected_path: str,
    actual_path: str,
    tolerance: float = 1e-10
) -> bool:
    """
    Verify that two experiment results match exactly.

    Args:
        expected_path: Path to expected results directory
        actual_path: Path to actual results directory
        tolerance: Tolerance for floating point comparisons

    Returns:
        True if results match, False otherwise
    """
    expected_dir = Path(expected_path)
    actual_dir = Path(actual_path)

    mismatches = []

    # Compare metadata
    expected_meta = json.load(open(expected_dir / "metadata.json"))
    actual_meta = json.load(open(actual_dir / "metadata.json"))

    if expected_meta["master_seed"] != actual_meta["master_seed"]:
        mismatches.append("master_seed mismatch")

    if expected_meta["num_networks"] != actual_meta["num_networks"]:
        mismatches.append("num_networks mismatch")

    # Compare network records
    for i, (exp_net, act_net) in enumerate(zip(
        expected_meta["networks"], actual_meta["networks"]
    )):
        if exp_net["seed"] != act_net["seed"]:
            mismatches.append(f"network {i} seed mismatch")
        if exp_net["name"] != act_net["name"]:
            mismatches.append(f"network {i} name mismatch")

    # Compare results if present
    results_file = "results.json"
    if (expected_dir / results_file).exists() and (actual_dir / results_file).exists():
        expected_results = json.load(open(expected_dir / results_file))
        actual_results = json.load(open(actual_dir / results_file))

        # Compare detection results
        if "detection_results" in expected_results:
            for key in expected_results["detection_results"]:
                if key not in actual_results["detection_results"]:
                    mismatches.append(f"missing detection key: {key}")
                    continue

                exp_list = expected_results["detection_results"][key]
                act_list = actual_results["detection_results"][key]

                if len(exp_list) != len(act_list):
                    mismatches.append(f"{key}: different number of results")
                    continue

                for j, (exp_r, act_r) in enumerate(zip(exp_list, act_list)):
                    for field in ["level1", "level2", "network_seed"]:
                        if field in exp_r and exp_r[field] != act_r.get(field):
                            mismatches.append(f"{key}[{j}].{field} mismatch")

    if mismatches:
        print("Reproducibility check FAILED:")
        for m in mismatches:
            print(f"  - {m}")
        return False

    print("All results match exactly!")
    return True
