"""ImageHZ — spatial-locality-preserving HZ for dense-conv bodies.

Phase 2.0 prototype (see audit_results/cifar_unknown_margin_atlas_20260603/PHASE2_DESIGN.md).

This package is research-only. The production verifier path is NOT
wired to it; the only consumer is the offline prototype driver at
`research/imagehz_cifar_prototype.py`.

Three hard guards per advisor 2026-06-03 rev-2:
1. MaxPool is FORBIDDEN — raises NotImplementedError.
2. NO Ac/Ab/b in the ImageHZ body; constraints come back at the
   Flatten → SparseGcZ bridge.
3. All tensors are float64; per-generator storage is
   (C_region, H_region, W_region).
"""
from act.back_end.imagehz.representation import (
    BoundingBox,
    ImageHZ,
    SpatialGenerator,
    merge_generators_by_xi_id,
)
from act.back_end.imagehz.operators import (
    apply_add,
    apply_avgpool2d,
    apply_conv2d,
    apply_maxpool2d,  # raises NotImplementedError on call
    apply_relu_triangle,
)
from act.back_end.imagehz.bridge import flatten_to_sparsegcz

__all__ = [
    "BoundingBox",
    "ImageHZ",
    "SpatialGenerator",
    "apply_add",
    "apply_avgpool2d",
    "apply_conv2d",
    "apply_maxpool2d",
    "apply_relu_triangle",
    "flatten_to_sparsegcz",
    "merge_generators_by_xi_id",
]
