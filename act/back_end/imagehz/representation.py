"""ImageHZ representation — spatial-locality-preserving HZ.

A minimal, NO-constraint variant of HZ for the Phase 2 prototype.
The forward HZ here keeps the center as a (C, H, W) tensor and
each generator as a localized (C, H, W) value block over a bounded
spatial region (per advisor 2026-06-03 rev-2: NO Ac/Ab/b in the
body; constraints re-enter only at the Flatten bridge to
SparseGcZ).

All tensors are float64. Soundness comes from the operator-level
semantics in `operators.py`, not from any constraint system here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class BoundingBox:
    """Closed-open bounding box in (C, H, W) coordinate space.

    Semantics: the generator's `values` tensor has shape
    (c_hi - c_lo, h_hi - h_lo, w_hi - w_lo). Contributions outside
    this region are implicitly zero — that IS the spatial-locality
    structure.
    """
    c_lo: int
    c_hi: int
    h_lo: int
    h_hi: int
    w_lo: int
    w_hi: int

    def __post_init__(self) -> None:
        if not (0 <= self.c_lo <= self.c_hi
                and 0 <= self.h_lo <= self.h_hi
                and 0 <= self.w_lo <= self.w_hi):
            raise ValueError(f"invalid box: {self}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.c_hi - self.c_lo,
                self.h_hi - self.h_lo,
                self.w_hi - self.w_lo)

    @property
    def numel(self) -> int:
        c, h, w = self.shape
        return c * h * w

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Smallest box containing both."""
        return BoundingBox(
            c_lo=min(self.c_lo, other.c_lo),
            c_hi=max(self.c_hi, other.c_hi),
            h_lo=min(self.h_lo, other.h_lo),
            h_hi=max(self.h_hi, other.h_hi),
            w_lo=min(self.w_lo, other.w_lo),
            w_hi=max(self.w_hi, other.w_hi),
        )

    def is_full(self, C: int, H: int, W: int) -> bool:
        return (self.c_lo == 0 and self.c_hi == C
                and self.h_lo == 0 and self.h_hi == H
                and self.w_lo == 0 and self.w_hi == W)


@dataclass
class SpatialGenerator:
    """One generator localized to a bounding box.

    `values` shape MUST equal `region.shape`. The full-image
    contribution at any (c, h, w) outside the region is 0.

    `xi_id` identifies which input symbol this generator scales
    with — either a root input pixel id (0 .. root_ng - 1) or an
    aux id (root_ng .. ng - 1) assigned by ReLU triangle.
    """
    region: BoundingBox
    values: torch.Tensor
    xi_id: int

    def __post_init__(self) -> None:
        if self.values.shape != self.region.shape:
            raise ValueError(
                f"generator values shape {tuple(self.values.shape)} != "
                f"region shape {self.region.shape}"
            )
        if self.values.dtype != torch.float64:
            raise ValueError(
                f"ImageHZ requires float64; got {self.values.dtype}"
            )


@dataclass
class ImageHZ:
    """Spatial-locality-preserving HZ.

    `c` is the center activation of shape (C, H, W).
    `generators` is a list of `SpatialGenerator` — each with its
    own region and value tensor.

    NO Ac/Ab/b in the body (advisor rev-2 guard 2). Constraints
    re-enter at the Flatten bridge to SparseGcZ in `bridge.py`.

    Semantics: the concrete set is

        { c + Σ_g G.embed_full() * xi[G.xi_id]
          | xi ∈ [-1, 1]^ng }

    where `embed_full` is the embedding of the local values tensor
    into the full (C, H, W) space (zero outside the region).
    """
    c: torch.Tensor                              # (C, H, W) float64
    generators: List[SpatialGenerator] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.c.dim() != 3:
            raise ValueError(
                f"ImageHZ.c must be (C, H, W); got shape {tuple(self.c.shape)}"
            )
        if self.c.dtype != torch.float64:
            raise ValueError(
                f"ImageHZ requires float64 center; got {self.c.dtype}"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.c.shape)  # type: ignore[return-value]

    @property
    def C(self) -> int:
        return int(self.c.shape[0])

    @property
    def H(self) -> int:
        return int(self.c.shape[1])

    @property
    def W(self) -> int:
        return int(self.c.shape[2])

    @property
    def ng(self) -> int:
        return len(self.generators)

    @property
    def device(self) -> torch.device:
        return self.c.device

    @property
    def dtype(self) -> torch.dtype:
        return self.c.dtype

    # ── Embedding helpers ─────────────────────────────────────────

    def embed_generator(self, g: SpatialGenerator) -> torch.Tensor:
        """Return the full (C, H, W) tensor with g's values placed
        inside its region and 0 elsewhere."""
        out = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        r = g.region
        out[r.c_lo:r.c_hi, r.h_lo:r.h_hi, r.w_lo:r.w_hi] = g.values
        return out

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-position box bounds (lb, ub) of shape (C, H, W).

        lb[i] = c[i] - Σ_g |G.embed_full()[i]|
        ub[i] = c[i] + Σ_g |G.embed_full()[i]|

        Sound (over-approximate) because each xi[G.xi_id] ∈ [-1, 1]
        independently; total radius is at most the sum of absolute
        contributions per position. Tight on each individual
        generator; loose only via the independence assumption when
        multiple generators share an xi_id (which is OK for an
        over-approx).
        """
        rad = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        for g in self.generators:
            r = g.region
            rad[r.c_lo:r.c_hi, r.h_lo:r.h_hi, r.w_lo:r.w_hi] += g.values.abs()
        return self.c - rad, self.c + rad

    def num_unique_xi_ids(self) -> int:
        """Count of distinct xi_ids used by the generators."""
        return len({g.xi_id for g in self.generators})

    def stats(self) -> dict:
        """Per-layer telemetry per advisor 2026-06-03 Step 2.0 guard 2.

        Returns:
            dict with keys:
              - num_generators: total generator count
              - num_unique_xi_id: count of distinct xi_ids
              - total_nonzeros: sum of generator value-tensor elements
              - max_region_numel: max generator region size (C*H*W)
              - avg_region_numel: mean region size
              - dim_C, dim_H, dim_W: spatial center dims
        """
        n_gen = len(self.generators)
        ids = {g.xi_id for g in self.generators}
        sizes = [g.region.numel for g in self.generators]
        return {
            "num_generators": int(n_gen),
            "num_unique_xi_id": int(len(ids)),
            "total_nonzeros": int(sum(int(g.values.numel()) for g in self.generators)),
            "max_region_numel": int(max(sizes)) if sizes else 0,
            "avg_region_numel": (float(sum(sizes)) / n_gen) if n_gen else 0.0,
            "dim_C": int(self.C),
            "dim_H": int(self.H),
            "dim_W": int(self.W),
        }


def merge_generators_by_xi_id(
    generators: List[SpatialGenerator],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> List[SpatialGenerator]:
    """Per advisor 2026-06-03 Step 2.0 guard 1.

    Group generators by `xi_id`. For each group, merge by:
    - region = union of input regions
    - values = sum over the union frame (zero outside each
      operand's original region)

    This preserves the *factor-aware* semantics — same xi_id is
    treated as ONE variable downstream, not multiple independent
    columns. Without this, the LP would treat shared-xi columns as
    independent and produce strictly LOOSER bounds (sound but
    spurious phantom margin), defeating the whole point of
    spatial-locality preservation.
    """
    by_id: dict[int, list[SpatialGenerator]] = {}
    for g in generators:
        by_id.setdefault(g.xi_id, []).append(g)
    out: List[SpatialGenerator] = []
    for xi_id, gs in by_id.items():
        if len(gs) == 1:
            out.append(gs[0])
            continue
        region = gs[0].region
        for g in gs[1:]:
            region = region.union(g.region)
        merged = torch.zeros(
            region.shape, dtype=dtype, device=device,
        )
        for g in gs:
            r = g.region
            merged[
                (r.c_lo - region.c_lo):(r.c_hi - region.c_lo),
                (r.h_lo - region.h_lo):(r.h_hi - region.h_lo),
                (r.w_lo - region.w_lo):(r.w_hi - region.w_lo),
            ] += g.values
        out.append(SpatialGenerator(
            region=region, values=merged, xi_id=xi_id,
        ))
    return out


def fresh_aux_xi_id(root_ng: int, used: Sequence[int]) -> int:
    """Return a fresh aux xi_id strictly greater than all in `used`
    and at least `root_ng`. Used by ReLU triangle to assign ids
    that don't collide with root input pixels.
    """
    max_used = max(list(used) + [root_ng - 1])
    return max_used + 1
