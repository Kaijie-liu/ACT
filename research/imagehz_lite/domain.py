"""ImageHZ-lite Phase 0 domain types.

Per §9R-1 of the prototype plan: a tile carries N independent generators
that share the same spatial footprint, NOT a single generator with an
opaque factor_ids list.

The concretization at (c, h, w) with (c0, h0, w0) the tile origin and
(tc, th, tw) the tile shape is:

    sum over k in [0, n_gen_tile):
        G_tile[k, c-c0, h-h0, w-w0] * xi[factor_ids[k]]

Multiple TileBlocks may overlap spatially provided their `factor_ids`
lists are disjoint. The container `ImageHZLite` carries the center +
the list of TileBlocks and enforces invariants on every mutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TileBlock:
    """A block of N independent generators sharing one spatial footprint.

    Fields:
        origin_chw  : (c0, h0, w0) — top-left in the feature map.
        shape       : (tc, th, tw) — spatial footprint.
        G_tile      : (n_gen_tile, tc, th, tw) float64 — generator
                      coefficient stack along the leading axis.
        factor_ids  : list[int] len == n_gen_tile — one root-factor id
                      per generator column.
        aux_meta    : dict with keys
                        kind         : 'root' | 'relu_aux'
                        spawn_layer  : int
                        spawn_op     : str
                        parent_block : int | None

    Invariants (enforced in `validate`):
        - G_tile.ndim == 4
        - G_tile.shape == (n_gen_tile,) + shape
        - len(factor_ids) == n_gen_tile
        - all factor_ids unique within this block
        - G_tile.dtype == float64
    """

    origin_chw: Tuple[int, int, int]
    shape: Tuple[int, int, int]
    G_tile: torch.Tensor
    factor_ids: Tuple[int, ...]
    aux_meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_gen_tile(self) -> int:
        return int(self.G_tile.shape[0])

    @property
    def numel_per_gen(self) -> int:
        tc, th, tw = self.shape
        return tc * th * tw

    def validate(self) -> None:
        if self.G_tile.dim() != 4:
            raise ValueError(
                f"TileBlock.G_tile must be 4D, got shape={tuple(self.G_tile.shape)}"
            )
        n_gen, tc, th, tw = self.G_tile.shape
        if (tc, th, tw) != tuple(self.shape):
            raise ValueError(
                f"TileBlock.G_tile spatial dims {(tc, th, tw)} != shape {self.shape}"
            )
        if len(self.factor_ids) != n_gen:
            raise ValueError(
                f"TileBlock: len(factor_ids)={len(self.factor_ids)} "
                f"!= n_gen_tile={n_gen}"
            )
        if len(set(self.factor_ids)) != len(self.factor_ids):
            raise ValueError(
                f"TileBlock: factor_ids contains duplicates: {self.factor_ids}"
            )
        if self.G_tile.dtype != torch.float64:
            raise ValueError(
                f"TileBlock.G_tile must be float64; got {self.G_tile.dtype}"
            )
        ak = self.aux_meta.get("kind")
        if ak not in (None, "root", "relu_aux"):
            raise ValueError(
                f"TileBlock.aux_meta['kind'] must be 'root' or 'relu_aux', got {ak}"
            )

    def bounds_radius(
        self, shape_chw: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Per-position radius contribution = sum of |G_tile| over the
        generator axis, embedded into the full (C, H, W) frame.

        Used to compute per-position bounds when combined with the
        center and the radius contributions from all other tiles.
        """
        C, H, W = shape_chw
        c0, h0, w0 = self.origin_chw
        tc, th, tw = self.shape
        rad = self.G_tile.abs().sum(dim=0)  # (tc, th, tw)
        out = torch.zeros((C, H, W), dtype=torch.float64, device=self.G_tile.device)
        out[c0:c0 + tc, h0:h0 + th, w0:w0 + tw] = rad
        return out


@dataclass
class ImageHZLite:
    """Center + list of TileBlocks. The concrete reachable set is

        { c + sum_block sum_k G_block[k] * xi[factor_id_block_k]
          | xi in [-1, 1]^N_total_factors }

    Container invariants:
        - c.shape == (C, H, W)
        - c.dtype == float64
        - tiles is a list of TileBlock (possibly empty)
        - factor_ids across tiles need NOT be disjoint globally; they
          are aliased — multiple blocks sharing factor i share that xi
          variable. (This is the factor-aware semantics.)
    """

    c: torch.Tensor                                  # (C, H, W) float64
    tiles: List[TileBlock] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.c.dim() != 3:
            raise ValueError(
                f"ImageHZLite.c must be (C, H, W); got {tuple(self.c.shape)}"
            )
        if self.c.dtype != torch.float64:
            raise ValueError(
                f"ImageHZLite.c must be float64; got {self.c.dtype}"
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
    def n_tile(self) -> int:
        return len(self.tiles)

    @property
    def total_generator_count(self) -> int:
        return sum(t.n_gen_tile for t in self.tiles)

    @property
    def unique_factor_ids(self) -> set:
        ids: set = set()
        for t in self.tiles:
            ids.update(t.factor_ids)
        return ids

    @property
    def n_root_factors(self) -> int:
        """Number of distinct factor ids whose kind is 'root'."""
        ids: set = set()
        for t in self.tiles:
            if t.aux_meta.get("kind") == "root":
                ids.update(t.factor_ids)
        return len(ids)

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-position (lb, ub) tensors of shape (C, H, W).

        Sound: the radius at each position is at most the sum of
        absolute generator contributions because each xi ∈ [-1, +1].
        Loose only via the independence assumption when multiple tiles
        share a factor id (which is OK for an over-approximation).
        """
        rad = torch.zeros(self.shape, dtype=torch.float64, device=self.c.device)
        for t in self.tiles:
            rad = rad + t.bounds_radius(self.shape)
        return self.c - rad, self.c + rad


@dataclass
class Phase0FlattenSnapshot:
    """Output of `apply_flatten` in Phase 0.

    Per §9R-4: representation metrics + deterministic column metadata.
    No SparseGcZ, no HZono, no verifier connection.
    """

    c_flat: torch.Tensor                                 # (n_flat,) float64
    blocks_meta: List[Dict[str, Any]]                    # one per TileBlock
    root_ng_at_flatten: int
    total_aux_count: int
    per_layer_girard_fires_observed: List[Dict[str, Any]]
    peak_memory_bytes: int
    wall_s: float
