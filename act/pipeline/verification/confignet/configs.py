from __future__ import annotations

import dataclasses
import json
import hashlib
from typing import Any, Dict, List, Optional


def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def canonical_hash(obj: Any) -> str:
    """Stable SHA256 over canonical JSON."""
    return hashlib.sha256(_canonical_dumps(obj).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    arch: str  # "mlp" or "cnn"
    template_name: str  # existing template in examples_config.yaml
    input_shape: List[int]
    num_classes: int
    activation: str = "relu"
    dataset: str = "runtime"
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(**d)

    def stable_hash(self) -> str:
        return canonical_hash(self.to_dict())


@dataclasses.dataclass(frozen=True)
class SpecConfig:
    eps: float
    norm: str = "linf"
    targeted: bool = False  # Week1: must remain False (untargeted)
    true_label: int = 0
    target_label: Optional[int] = None  # ignored when targeted=False
    assert_kind: str = "TOP1_ROBUST"  # Week1: untargeted

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpecConfig":
        return SpecConfig(**d)

    def stable_hash(self) -> str:
        return canonical_hash(self.to_dict())
