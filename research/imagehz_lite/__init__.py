"""ImageHZ-lite Phase 0 — VGG/Tiny conv-body representation prototype.

This package implements the representation-only experiment authorized
by the §6b VGG mini-atlas gate (PROCEED) and constrained by the
§9-resolved design lock in `research/imagehz_vgg_prototype_plan.md`.

Phase 0 has NO connection to the production verifier:
- no `verify_once_hz` call
- no witness sidecar
- no LP solver
- no FAL receipts
- no CIFAR path

The package's only consumers are:
- the unit tests under `research/imagehz_lite/tests/`
- the sentinel driver `research/imagehz_lite/run_vgg_phase0.py`
"""

from research.imagehz_lite.domain import (  # noqa: F401
    TileBlock,
    ImageHZLite,
    Phase0FlattenSnapshot,
)
from research.imagehz_lite.budget import (  # noqa: F401
    BudgetExceeded,
    Budget,
)
