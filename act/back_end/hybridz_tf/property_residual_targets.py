"""Property-driven schedules for ReLU residual normal form.

This module is deliberately outside the proof kernel.  It consumes final
ASSERT rival rows and DAG-aware DualSolver adjoints, then returns explicit
``(ReLU layer, row, guard)`` targets.  Scores, dual margins, propagated facts,
and selector statuses never alter a coefficient, bound, or verdict.

The multi-rival selector keeps a small top pool per rival and applies a
deterministic facility-location greedy rule.  This prevents a large budget
from repeatedly serving only one hard rival while avoiding an
``M * all_neurons`` score copy on top of DualSolver's adjoint tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from act.back_end.core import Bounds, Fact, Net


@dataclass(frozen=True)
class PropertyResidualTarget:
    layer_id: int
    row: int
    guard: str
    score: float
    facility_gain: float
    dominant_rival: int

    def builder_tuple(self) -> Tuple[int, int, str]:
        return int(self.layer_id), int(self.row), str(self.guard)


@dataclass(frozen=True)
class PropertyResidualPlan:
    targets: Tuple[PropertyResidualTarget, ...]
    property_sha256: str
    targets_sha256: str
    receipt: Mapping[str, Any]

    @property
    def builder_targets(self) -> Tuple[Tuple[int, int, str], ...]:
        return tuple(target.builder_tuple() for target in self.targets)


@dataclass(frozen=True)
class PropertySparseQueryTarget:
    """One non-authoritative property-selected query row."""

    layer_id: int
    row: int
    score: float
    facility_gain: float
    dominant_rival: int


@dataclass(frozen=True)
class PropertySparseQueryPlan:
    """Deterministic per-layer sparse query schedule.

    The order within each layer is the nested facility-location prefix.  It is
    intentionally kept separate from ``PropertyResidualPlan``: selecting a
    row grants no authority to either the candidate optimizer or its scores.
    """

    targets: Tuple[PropertySparseQueryTarget, ...]
    property_sha256: str
    selection_sha256: str
    receipt: Mapping[str, Any]

    @property
    def rows_by_layer(self) -> Dict[int, Tuple[int, ...]]:
        grouped: Dict[int, list[int]] = {}
        for target in self.targets:
            grouped.setdefault(int(target.layer_id), []).append(int(target.row))
        return {
            layer_id: tuple(rows)
            for layer_id, rows in sorted(grouped.items())
        }

    def rows_for_layer(self, layer_id: int) -> Tuple[int, ...]:
        return self.rows_by_layer.get(int(layer_id), ())


def property_correlation_layer_quotas(
    net: Net,
    *,
    budget: int,
    per_layer_cap: int = 64,
    before: Optional[Mapping[int, Any]] = None,
) -> Dict[int, int]:
    """Allocate a nested budget across targetable residual-block ReLUs.

    A targetable route is deliberately identical to the first operator
    candidate: ``ADD -> [FLATTEN] -> (DENSE|CONV2D) -> RELU`` or the direct
    ``ADD -> RELU`` case, with one successor at every step.  Allocation is
    deterministic round-robin in topological order, capped by layer width and
    ``per_layer_cap``.  Consequently every residual depth receives coverage
    before any layer receives a second row.
    """

    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (Integral, np.integer))
        or int(budget) < 0
    ):
        raise ValueError("correlation budget must be a nonnegative integer")
    if (
        isinstance(per_layer_cap, (bool, np.bool_))
        or not isinstance(per_layer_cap, (Integral, np.integer))
        or int(per_layer_cap) <= 0
    ):
        raise ValueError("per_layer_cap must be a positive integer")

    layers = list(net.layers)
    by_id = {int(layer.id): layer for layer in layers}
    position = {int(layer.id): index for index, layer in enumerate(layers)}
    successors: Dict[int, list[int]] = {layer_id: [] for layer_id in by_id}
    for layer in layers:
        for parent in net.preds.get(int(layer.id), ()):
            parent_id = int(parent)
            if parent_id not in successors:
                raise ValueError(
                    f"layer {layer.id} references missing predecessor "
                    f"{parent_id}"
                )
            successors[parent_id].append(int(layer.id))

    target_ids: set[int] = set()
    for layer in layers:
        if _kind_token(layer.kind) != "ADD":
            continue
        for raw_successor in successors[int(layer.id)]:
            current = int(raw_successor)
            current_layer = by_id[current]
            if _kind_token(current_layer.kind) == "RELU":
                target_ids.add(current)
                continue
            if _kind_token(current_layer.kind) == "FLATTEN":
                current_successors = successors[current]
                if len(current_successors) != 1:
                    continue
                current = int(current_successors[0])
                current_layer = by_id[current]
            if _kind_token(current_layer.kind) not in {"DENSE", "CONV2D"}:
                continue
            current_successors = successors[current]
            if len(current_successors) != 1:
                continue
            relu_id = int(current_successors[0])
            relu = by_id[relu_id]
            if _kind_token(relu.kind) == "RELU":
                target_ids.add(relu_id)

    ordered = sorted(target_ids, key=position.__getitem__)
    caps = {}
    for layer_id in ordered:
        cap = min(
            int(per_layer_cap), int(len(by_id[layer_id].out_vars))
        )
        if before is not None:
            if layer_id not in before:
                raise ValueError(
                    f"missing preactivation bounds for ReLU {layer_id}"
                )
            bounds = _as_bounds(before[layer_id], layer_id=layer_id)
            lower = bounds.lb.detach().reshape(-1)
            upper = bounds.ub.detach().reshape(-1)
            unstable = int(
                torch.count_nonzero((lower < 0.0) & (upper > 0.0)).item()
            )
            cap = min(cap, unstable)
        caps[layer_id] = cap
    quotas = {layer_id: 0 for layer_id in ordered}
    remaining = int(budget)
    while remaining > 0:
        progressed = False
        for layer_id in ordered:
            if quotas[layer_id] >= caps[layer_id]:
                continue
            quotas[layer_id] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quotas


def _binary64_sha256(
    C: np.ndarray,
    thresholds: np.ndarray,
    *,
    kind: str,
) -> str:
    digest = hashlib.sha256()
    for value in (C, thresholds):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(str(kind).encode("utf-8"))
    return digest.hexdigest()


def _targets_sha256(
    property_sha256: str,
    targets: Sequence[PropertyResidualTarget],
) -> str:
    payload = {
        "property_sha256": str(property_sha256),
        "targets": [
            {
                "layer_id": int(target.layer_id),
                "row": int(target.row),
                "guard": str(target.guard),
            }
            for target in targets
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sparse_selection_sha256(
    property_sha256: str,
    layer_quotas: Sequence[Tuple[int, int]],
    targets: Sequence[PropertySparseQueryTarget],
) -> str:
    return _canonical_sha256(
        {
            "property_sha256": str(property_sha256),
            "layer_quotas": [
                [int(layer_id), int(quota)]
                for layer_id, quota in layer_quotas
            ],
            "selected": [
                [int(target.layer_id), int(target.row)]
                for target in targets
            ],
        }
    )


def _kind_token(kind: Any) -> str:
    return str(getattr(kind, "value", kind)).upper()


def _as_bounds(value: Any, *, layer_id: int) -> Bounds:
    if isinstance(value, Bounds):
        return value
    if isinstance(value, Fact):
        return value.bounds
    bounds = getattr(value, "bounds", None)
    if isinstance(bounds, Bounds):
        return bounds
    raise ValueError(f"layer {layer_id} does not provide Bounds")


def _finite_property(
    C: Any,
    thresholds: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(C, torch.Tensor):
        C_np = C.detach().cpu().double().numpy()
    else:
        C_np = np.asarray(C, dtype=np.float64)
    if isinstance(thresholds, torch.Tensor):
        thresholds_np = thresholds.detach().cpu().double().numpy()
    else:
        thresholds_np = np.asarray(thresholds, dtype=np.float64)
    if C_np.ndim != 2:
        raise ValueError(f"property C must be 2-D, got {C_np.shape}")
    thresholds_np = thresholds_np.reshape(-1)
    if thresholds_np.size != C_np.shape[0]:
        raise ValueError(
            f"threshold count {thresholds_np.size} != rival rows {C_np.shape[0]}"
        )
    if (
        C_np.shape[0] == 0
        or C_np.shape[1] == 0
        or not np.all(np.isfinite(C_np))
        or not np.all(np.isfinite(thresholds_np))
    ):
        raise ValueError("property rows must be nonempty and finite")
    return (
        np.ascontiguousarray(C_np, dtype=np.float64),
        np.ascontiguousarray(thresholds_np, dtype=np.float64),
    )


def _relu_bounds(
    before: Mapping[int, Any],
    layer_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(layer_id) not in before:
        raise ValueError(f"missing preactivation bounds for ReLU {layer_id}")
    bounds = _as_bounds(before[int(layer_id)], layer_id=int(layer_id))
    lower = bounds.lb.detach().flatten()
    upper = bounds.ub.detach().flatten()
    if (
        lower.numel() != upper.numel()
        or not bool(torch.isfinite(lower).all())
        or not bool(torch.isfinite(upper).all())
        or bool((lower > upper).any())
    ):
        raise ValueError(f"invalid preactivation bounds for ReLU {layer_id}")
    return lower, upper


def _validated_layer_quotas(
    layer_quotas: Mapping[int, int],
) -> Tuple[Tuple[int, int], ...]:
    if not isinstance(layer_quotas, Mapping):
        raise ValueError("layer_quotas must be a mapping")
    normalized = []
    for raw_layer_id, raw_quota in layer_quotas.items():
        if (
            isinstance(raw_layer_id, (bool, np.bool_))
            or not isinstance(raw_layer_id, (Integral, np.integer))
            or int(raw_layer_id) < 0
        ):
            raise ValueError("layer quota ids must be nonnegative integers")
        if (
            isinstance(raw_quota, (bool, np.bool_))
            or not isinstance(raw_quota, (Integral, np.integer))
            or int(raw_quota) < 0
        ):
            raise ValueError("layer quotas must be nonnegative integers")
        normalized.append((int(raw_layer_id), int(raw_quota)))
    normalized.sort()
    if len({layer_id for layer_id, _quota in normalized}) != len(normalized):
        raise ValueError("layer quota ids must be unique")
    return tuple(normalized)


def _facility_weights(
    rival_ids: Tuple[int, ...],
    hardness: np.ndarray,
) -> np.ndarray:
    hard_order = sorted(
        range(len(rival_ids)),
        key=lambda index: (-float(hardness[index]), rival_ids[index]),
    )
    weights = np.ones(len(rival_ids), dtype=np.float64)
    if len(rival_ids) == 1:
        weights[0] = 2.0
    else:
        for rank, rival in enumerate(hard_order):
            weights[rival] = 2.0 - rank / (len(rival_ids) - 1)
    return weights


def _deterministic_top_score_rows(
    scores: torch.Tensor,
    take: int,
) -> torch.Tensor:
    """Return exact top-score local rows with ascending-row tie breaks."""

    width = int(scores.shape[1])
    if int(take) == width:
        return torch.arange(
            width, dtype=torch.long, device=scores.device
        ).expand(int(scores.shape[0]), width)
    values = torch.topk(
        scores,
        k=int(take),
        dim=1,
        largest=True,
        sorted=False,
    ).values
    cutoffs = values.min(dim=1).values
    chosen = []
    for rival in range(int(scores.shape[0])):
        row_scores = scores[rival]
        strict = torch.nonzero(
            row_scores > cutoffs[rival], as_tuple=False
        ).flatten()
        remaining = int(take) - int(strict.numel())
        tied = torch.nonzero(
            row_scores == cutoffs[rival], as_tuple=False
        ).flatten()[:remaining]
        rows = torch.cat([strict, tied])
        if int(rows.numel()) != int(take):
            raise RuntimeError("deterministic score pool is incomplete")
        chosen.append(rows)
    return torch.stack(chosen, dim=0)


def plan_sparse_query_rows_from_property_adjoints(
    property_adjoints: Mapping[int, torch.Tensor],
    before: Mapping[int, Any],
    *,
    layer_quotas: Mapping[int, int],
    rival_ids: Sequence[int],
    rival_hardness: Sequence[float],
    all_rivals_processed: bool,
    property_sha256: str,
    pool_per_rival: int = 64,
    facility_beta: float = 0.25,
) -> PropertySparseQueryPlan:
    """Select a fixed, deterministic query quota independently per ReLU.

    Only unstable rows are eligible.  Each rival contributes a fixed-size
    score pool, then facility-location greedy selection produces a nested
    prefix.  Requiring ``pool_per_rival >= quota`` makes the candidate universe
    independent of the requested prefix length and guarantees that every
    feasible quota is filled without materializing all rival-by-neuron scores
    on the CPU.
    """

    quotas = _validated_layer_quotas(layer_quotas)
    if int(pool_per_rival) <= 0:
        raise ValueError("pool_per_rival must be positive")
    if any(quota > int(pool_per_rival) for _layer_id, quota in quotas):
        raise ValueError("each layer quota must not exceed pool_per_rival")
    if not math.isfinite(float(facility_beta)) or float(facility_beta) < 0.0:
        raise ValueError("facility_beta must be finite and nonnegative")
    rival_ids_input = tuple(int(value) for value in rival_ids)
    hardness = np.asarray(rival_hardness, dtype=np.float64).reshape(-1)
    rival_count = len(rival_ids_input)
    if (
        rival_count == 0
        or hardness.size != rival_count
        or not np.all(np.isfinite(hardness))
    ):
        raise ValueError("processed rivals/hardness are malformed")
    if (
        len(set(rival_ids_input)) != rival_count
        or min(rival_ids_input) < 0
    ):
        raise ValueError("processed rival ids must be unique and nonnegative")
    # Canonicalizing the reduction axis prevents a caller-side rival
    # permutation from changing floating-point summation order.
    canonical_rivals = tuple(
        sorted(range(rival_count), key=lambda index: rival_ids_input[index])
    )
    rival_ids_tuple = tuple(
        rival_ids_input[index] for index in canonical_rivals
    )
    hardness = hardness[np.asarray(canonical_rivals, dtype=np.int64)]
    weights = _facility_weights(rival_ids_tuple, hardness)

    selected_all: list[PropertySparseQueryTarget] = []
    layer_receipts: list[Dict[str, Any]] = []
    adjoint_cells = 0
    for layer_id, quota in quotas:
        if layer_id not in property_adjoints:
            raise ValueError(f"missing property adjoint for ReLU {layer_id}")
        adjoint = property_adjoints[layer_id].detach()
        if adjoint.dim() < 2 or int(adjoint.shape[0]) != rival_count:
            raise ValueError(
                f"ReLU {layer_id} adjoint shape {tuple(adjoint.shape)} "
                f"does not have rival axis {rival_count}"
            )
        flat = adjoint.flatten(start_dim=1)
        flat = flat[
            torch.as_tensor(
                canonical_rivals,
                dtype=torch.long,
                device=flat.device,
            )
        ]
        lower, upper = _relu_bounds(before, layer_id)
        if int(flat.shape[1]) != int(lower.numel()):
            raise ValueError(
                f"ReLU {layer_id} adjoint width {flat.shape[1]} != "
                f"bounds width {lower.numel()}"
            )
        lower = lower.to(device=flat.device, dtype=flat.dtype)
        upper = upper.to(device=flat.device, dtype=flat.dtype)
        unstable_mask = (lower < 0.0) & (upper > 0.0)
        eligible_tensor = torch.nonzero(
            unstable_mask, as_tuple=False
        ).flatten()
        eligible = tuple(
            int(value)
            for value in eligible_tensor.detach().cpu().tolist()
        )
        adjoint_cells += int(flat.numel())

        selected_layer: list[PropertySparseQueryTarget] = []
        candidate_rows: Tuple[int, ...] = ()
        if quota > 0 and eligible:
            l = lower[eligible_tensor]
            u = upper[eligible_tensor]
            gap = -l * u / (u - l)
            scores = flat[:, eligible_tensor].abs() * gap.unsqueeze(0)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(
                    f"ReLU {layer_id} produced non-finite sparse scores"
                )
            take = min(int(pool_per_rival), len(eligible))
            # Exact boundary-tie handling avoids a full-width argsort while
            # retaining ascending row-id semantics.  Only the small union
            # below crosses to CPU.
            order = _deterministic_top_score_rows(scores, take)
            pooled_local = sorted(
                {
                    int(value)
                    for value in order.detach().cpu().reshape(-1).tolist()
                }
            )
            candidate_rows = tuple(eligible[position] for position in pooled_local)
            candidate_index = torch.as_tensor(
                pooled_local,
                dtype=torch.long,
                device=scores.device,
            )
            score_matrix = (
                scores[:, candidate_index].detach().cpu().double().numpy()
            )
            rival_max = (
                scores.max(dim=1).values.detach().cpu().double().numpy()
            )
            normalized = score_matrix / np.maximum(
                rival_max[:, None],
                np.finfo(np.float64).tiny,
            )
            coverage = np.zeros(rival_count, dtype=np.float64)
            available = np.ones(len(candidate_rows), dtype=bool)
            max_take = min(quota, len(eligible))
            if len(candidate_rows) < max_take:
                raise RuntimeError(
                    "fixed sparse quota exceeds deterministic candidate union"
                )
            for _step in range(max_take):
                incremental = np.maximum(
                    normalized - coverage[:, None], 0.0
                )
                gains = (weights[:, None] * incremental).sum(axis=0)
                gains += float(facility_beta) * np.max(
                    weights[:, None] * normalized,
                    axis=0,
                )
                gains[~available] = -np.inf
                best_gain = float(np.max(gains))
                if not math.isfinite(best_gain):
                    raise RuntimeError("sparse facility gain is non-finite")
                # candidate_rows is ascending: first maximum is the exact,
                # deterministic row-id tie break.
                position = int(np.flatnonzero(gains == best_gain)[0])
                row = int(candidate_rows[position])
                dominant_local = min(
                    range(rival_count),
                    key=lambda rival: (
                        -float(score_matrix[rival, position]),
                        int(rival_ids_tuple[rival]),
                    ),
                )
                selected_layer.append(
                    PropertySparseQueryTarget(
                        layer_id=int(layer_id),
                        row=row,
                        score=float(score_matrix[dominant_local, position]),
                        facility_gain=best_gain,
                        dominant_rival=int(
                            rival_ids_tuple[dominant_local]
                        ),
                    )
                )
                available[position] = False
                coverage = np.maximum(
                    coverage, normalized[:, position]
                )

        selected_rows = tuple(target.row for target in selected_layer)
        selected_set = set(selected_rows)
        omitted = tuple(row for row in eligible if row not in selected_set)
        selected_all.extend(selected_layer)
        layer_receipts.append(
            {
                "layer_id": int(layer_id),
                "quota": int(quota),
                "eligible_rows": list(eligible),
                "eligible_rows_sha256": _canonical_sha256(list(eligible)),
                "selected_rows": list(selected_rows),
                "selected_rows_sha256": _canonical_sha256(
                    list(selected_rows)
                ),
                "omitted_rows": list(omitted),
                "omitted_rows_sha256": _canonical_sha256(list(omitted)),
                "eligible_count": int(len(eligible)),
                "selected_count": int(len(selected_rows)),
                "omitted_count": int(len(omitted)),
                "candidate_union": int(len(candidate_rows)),
                "quota_filled": len(selected_rows) == min(quota, len(eligible)),
                "partition_complete": set(eligible)
                == selected_set.union(omitted),
                "partition_disjoint": selected_set.isdisjoint(omitted),
            }
        )

    targets = tuple(selected_all)
    selection_hash = _sparse_selection_sha256(
        property_sha256, quotas, targets
    )
    return PropertySparseQueryPlan(
        targets=targets,
        property_sha256=str(property_sha256),
        selection_sha256=selection_hash,
        receipt={
            "schema": "property_sparse_query_selector_v3",
            "status": "selected" if targets else "no_selected_rows",
            "candidate_only": True,
            "proof_authority": False,
            "property_sha256": str(property_sha256),
            "selection_sha256": selection_hash,
            "layer_quotas": [
                [int(layer_id), int(quota)]
                for layer_id, quota in quotas
            ],
            "pool_per_rival": int(pool_per_rival),
            "facility_beta": float(facility_beta),
            "rivals_processed": int(rival_count),
            "rival_ids": [int(value) for value in rival_ids_tuple],
            "all_rivals_processed": bool(all_rivals_processed),
            "adjoint_cells": int(adjoint_cells),
            "adjoint_solver_calls": 0,
            "targets_selected": int(len(targets)),
            "layers": layer_receipts,
            "schedule": [
                {
                    "layer_id": int(target.layer_id),
                    "row": int(target.row),
                    "score": float(target.score),
                    "facility_gain": float(target.facility_gain),
                    "dominant_rival": int(target.dominant_rival),
                }
                for target in targets
            ],
        },
    )


def plan_from_property_adjoints(
    property_adjoints: Mapping[int, torch.Tensor],
    before: Mapping[int, Any],
    *,
    budget: int,
    rival_ids: Sequence[int],
    rival_hardness: Sequence[float],
    all_rivals_processed: bool,
    property_sha256: str,
    pool_per_rival: int = 8,
    facility_beta: float = 0.25,
    phase_joint_focus_after_first: bool = False,
) -> PropertyResidualPlan:
    """Create a deterministic nested multi-rival target prefix.

    ``property_adjoints[layer]`` is the signed coefficient of the final
    violation form with respect to the *output* of that ReLU.  Its first axis
    is the processed rival.  The function is heuristic-only and does not
    consume any proof-producing object.
    """

    if int(budget) < 0:
        raise ValueError("property residual budget must be nonnegative")
    if int(pool_per_rival) <= 0:
        raise ValueError("pool_per_rival must be positive")
    if not math.isfinite(float(facility_beta)) or float(facility_beta) < 0.0:
        raise ValueError("facility_beta must be finite and nonnegative")
    rival_ids_tuple = tuple(int(value) for value in rival_ids)
    hardness = np.asarray(rival_hardness, dtype=np.float64).reshape(-1)
    M = len(rival_ids_tuple)
    if M == 0 or hardness.size != M or not np.all(np.isfinite(hardness)):
        raise ValueError("processed rivals/hardness are malformed")
    if len(set(rival_ids_tuple)) != M or min(rival_ids_tuple) < 0:
        raise ValueError("processed rival ids must be unique and nonnegative")

    per_rival_entries = [[] for _ in range(M)]
    rival_max = np.zeros(M, dtype=np.float64)
    relu_width_sum = 0
    unstable_total = 0
    adjoint_cells = 0

    for layer_id in sorted(int(value) for value in property_adjoints):
        adjoint = property_adjoints[layer_id].detach()
        if adjoint.dim() < 2 or int(adjoint.shape[0]) != M:
            raise ValueError(
                f"ReLU {layer_id} adjoint shape {tuple(adjoint.shape)} "
                f"does not have rival axis {M}"
            )
        flat = adjoint.flatten(start_dim=1)
        lower, upper = _relu_bounds(before, layer_id)
        if flat.shape[1] != lower.numel():
            raise ValueError(
                f"ReLU {layer_id} adjoint width {flat.shape[1]} != "
                f"bounds width {lower.numel()}"
            )
        lower = lower.to(device=flat.device, dtype=flat.dtype)
        upper = upper.to(device=flat.device, dtype=flat.dtype)
        unstable = (lower < 0.0) & (upper > 0.0)
        relu_width_sum += int(lower.numel())
        unstable_total += int(unstable.sum().item())
        adjoint_cells += int(flat.numel())
        if not bool(unstable.any()):
            continue
        gap = torch.zeros_like(lower)
        denominator = upper[unstable] - lower[unstable]
        gap[unstable] = (
            -lower[unstable] * upper[unstable] / denominator
        )
        scores = flat.abs() * gap.unsqueeze(0)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError(f"ReLU {layer_id} produced non-finite scores")
        layer_max = scores.max(dim=1).values.detach().cpu().double().numpy()
        rival_max = np.maximum(rival_max, layer_max)
        take = min(int(pool_per_rival), int(scores.shape[1]))
        # ``torch.topk`` does not promise which indices survive an exact
        # cutoff tie.  Reconstruct the pool from the kth score so membership
        # is score-first with an ascending local-row tie break on every
        # device/backend.  The global sort below then applies the stable
        # layer-id/row-id tie break across ReLUs.
        rows = _deterministic_top_score_rows(scores, take)
        values = torch.gather(scores, dim=1, index=rows)
        values_np = values.detach().cpu().double().numpy()
        rows_np = rows.detach().cpu().numpy()
        for rival in range(M):
            for position in range(take):
                score = float(values_np[rival, position])
                if score > 0.0:
                    per_rival_entries[rival].append(
                        (score, int(layer_id), int(rows_np[rival, position]))
                    )

    for rival in range(M):
        per_rival_entries[rival].sort(
            key=lambda item: (-item[0], item[1], item[2])
        )
        per_rival_entries[rival] = per_rival_entries[rival][
            : int(pool_per_rival)
        ]
    coordinates = sorted(
        {
            (int(layer_id), int(row))
            for entries in per_rival_entries
            for _score, layer_id, row in entries
        }
    )

    if int(budget) == 0 or not coordinates:
        targets: Tuple[PropertyResidualTarget, ...] = ()
        return PropertyResidualPlan(
            targets=targets,
            property_sha256=str(property_sha256),
            targets_sha256=_targets_sha256(property_sha256, targets),
            receipt={
                "schema": "property_residual_selector_v1",
                "status": "empty_budget" if int(budget) == 0 else "no_candidates",
                "proof_authority": False,
                "rivals_processed": M,
                "all_rivals_processed": bool(all_rivals_processed),
                "relu_width_sum": int(relu_width_sum),
                "unstable_total": int(unstable_total),
                "adjoint_cells": int(adjoint_cells),
                "candidate_union": int(len(coordinates)),
                "targets_selected": 0,
            },
        )

    score_matrix = np.zeros((M, len(coordinates)), dtype=np.float64)
    sign_matrix = np.zeros((M, len(coordinates)), dtype=np.float64)
    coordinate_position = {
        coordinate: position
        for position, coordinate in enumerate(coordinates)
    }
    grouped: Dict[int, list[Tuple[int, int]]] = {}
    for position, (layer_id, row) in enumerate(coordinates):
        grouped.setdefault(int(layer_id), []).append((position, int(row)))
    for layer_id, entries in grouped.items():
        adjoint = property_adjoints[layer_id].detach().flatten(start_dim=1)
        lower, upper = _relu_bounds(before, layer_id)
        lower = lower.to(device=adjoint.device, dtype=adjoint.dtype)
        upper = upper.to(device=adjoint.device, dtype=adjoint.dtype)
        rows = torch.as_tensor(
            [row for _position, row in entries],
            dtype=torch.long,
            device=adjoint.device,
        )
        l = lower[rows]
        u = upper[rows]
        gap = -l * u / (u - l)
        signed = adjoint[:, rows]
        scores = signed.abs() * gap.unsqueeze(0)
        scores_np = scores.detach().cpu().double().numpy()
        signed_np = signed.detach().cpu().double().numpy()
        for local, (position, _row) in enumerate(entries):
            score_matrix[:, position] = scores_np[:, local]
            sign_matrix[:, position] = signed_np[:, local]

    denominators = np.maximum(rival_max, np.finfo(np.float64).tiny)
    normalized = score_matrix / denominators[:, None]
    # Harder rivals receive weight two, easiest weight one.  Stable original
    # rival ids break hardness ties, making row permutations reproducible once
    # mapped back to those ids.
    hard_order = sorted(
        range(M),
        key=lambda index: (-float(hardness[index]), rival_ids_tuple[index]),
    )
    weights = np.ones(M, dtype=np.float64)
    if M == 1:
        weights[0] = 2.0
    else:
        for rank, rival in enumerate(hard_order):
            weights[rival] = 2.0 - rank / (M - 1)

    coverage = np.zeros(M, dtype=np.float64)
    available = np.ones(len(coordinates), dtype=bool)
    selected: list[PropertyResidualTarget] = []
    joint_focus_local: Optional[int] = None
    max_take = min(int(budget), len(coordinates))
    for _step in range(max_take):
        if (
            bool(phase_joint_focus_after_first)
            and selected
            and joint_focus_local is not None
        ):
            # Exact phase depth is exponential and must close the first
            # unresolved rival, not spend its second bit on facility
            # diversity.  Rank complementary coordinates by that rival's
            # own normalized relaxation contribution.
            gains = normalized[joint_focus_local, :].copy()
        else:
            incremental = np.maximum(
                normalized - coverage[:, None], 0.0
            )
            gains = (weights[:, None] * incremental).sum(axis=0)
            gains += float(facility_beta) * np.max(
                weights[:, None] * normalized,
                axis=0,
            )
        gains[~available] = -np.inf
        best_gain = float(np.max(gains))
        if not math.isfinite(best_gain) or best_gain <= 0.0:
            break
        # ``coordinates`` is lexicographically sorted, so the first exact
        # maximum is the deterministic layer/row tie break.
        position = int(np.flatnonzero(gains == best_gain)[0])
        layer_id, row = coordinates[position]
        signed = sign_matrix[:, position]
        magnitude = float(np.max(np.abs(signed))) if signed.size else 0.0
        sign_tol = 64.0 * np.finfo(np.float64).eps * max(1.0, magnitude)
        all_nonnegative = (
            bool(all_rivals_processed)
            and bool(np.all(signed >= -sign_tol))
            and bool(np.any(signed > sign_tol))
        )
        guard = "none" if all_nonnegative else "both"
        dominant_local = (
            int(joint_focus_local)
            if (
                bool(phase_joint_focus_after_first)
                and selected
                and joint_focus_local is not None
            )
            else int(np.argmax(score_matrix[:, position]))
        )
        selected.append(
            PropertyResidualTarget(
                layer_id=int(layer_id),
                row=int(row),
                guard=guard,
                score=float(score_matrix[dominant_local, position]),
                facility_gain=best_gain,
                dominant_rival=int(rival_ids_tuple[dominant_local]),
            )
        )
        available[position] = False
        coverage = np.maximum(coverage, normalized[:, position])
        if (
            bool(phase_joint_focus_after_first)
            and joint_focus_local is None
        ):
            joint_focus_local = int(dominant_local)

    targets_tuple = tuple(selected)
    return PropertyResidualPlan(
        targets=targets_tuple,
        property_sha256=str(property_sha256),
        targets_sha256=_targets_sha256(property_sha256, targets_tuple),
        receipt={
            "schema": "property_residual_selector_v1",
            "status": "selected" if targets_tuple else "no_positive_gain",
            "proof_authority": False,
            "rivals_processed": int(M),
            "rival_ids": [int(value) for value in rival_ids_tuple],
            "all_rivals_processed": bool(all_rivals_processed),
            "relu_width_sum": int(relu_width_sum),
            "unstable_total": int(unstable_total),
            "adjoint_cells": int(adjoint_cells),
            "pool_per_rival": int(pool_per_rival),
            "candidate_union": int(len(coordinates)),
            "targets_selected": int(len(targets_tuple)),
            "guard_none": int(
                sum(target.guard == "none" for target in targets_tuple)
            ),
            "guard_both": int(
                sum(target.guard == "both" for target in targets_tuple)
            ),
            "facility_beta": float(facility_beta),
            "selection_policy": (
                "facility_first_then_same_rival_joint"
                if phase_joint_focus_after_first
                else "multi_rival_facility"
            ),
            "joint_focus_rival_id": (
                None
                if joint_focus_local is None
                else int(rival_ids_tuple[joint_focus_local])
            ),
            "schedule": [
                {
                    "layer_id": int(target.layer_id),
                    "row": int(target.row),
                    "guard": str(target.guard),
                    "score": float(target.score),
                    "facility_gain": float(target.facility_gain),
                    "dominant_rival": int(target.dominant_rival),
                }
                for target in targets_tuple
            ],
        },
    )


def select_property_residual_targets(
    *,
    net: Net,
    before: Mapping[int, Any],
    after: Mapping[int, Any],
    C: Any,
    thresholds: Any,
    kind: Any,
    output_layer_id: int,
    budget: int,
    time_limit: float,
    deadline: Optional[float] = None,
    max_adjoint_cells: int = 30_000_000,
    pool_per_rival: int = 8,
    allowed_relu_layer_ids: Optional[Sequence[int]] = None,
    phase_joint_focus_after_first: bool = False,
) -> PropertyResidualPlan:
    """Run one bounded, DAG-aware DualSolver scheduling pass.

    If the all-rival adjoint tensor estimate exceeds ``max_adjoint_cells``,
    only the hardest interval-surviving rivals are processed and every chosen
    target retains both lower guards.  Thus a memory stop-loss can reduce
    scheduling quality but cannot alter graph containment.
    """

    started = time.monotonic()
    if int(budget) < 0:
        raise ValueError("property residual budget must be nonnegative")
    if not math.isfinite(float(time_limit)) or float(time_limit) < 0.0:
        raise ValueError("property residual time limit must be finite/nonnegative")
    if int(max_adjoint_cells) <= 0:
        raise ValueError("max_adjoint_cells must be positive")
    C_np, thresholds_np = _finite_property(C, thresholds)
    kind_name = _kind_token(kind)
    property_hash = _binary64_sha256(C_np, thresholds_np, kind=kind_name)

    def empty(status: str, **extra: Any) -> PropertyResidualPlan:
        targets: Tuple[PropertyResidualTarget, ...] = ()
        return PropertyResidualPlan(
            targets=targets,
            property_sha256=property_hash,
            targets_sha256=_targets_sha256(property_hash, targets),
            receipt={
                "schema": "property_residual_selector_v1",
                "status": str(status),
                "proof_authority": False,
                "elapsed_seconds": float(time.monotonic() - started),
                "targets_selected": 0,
                **extra,
            },
        )

    if int(budget) == 0 or float(time_limit) == 0.0:
        return empty("disabled")
    if kind_name == "UNSAFE_LINEAR":
        return empty("unsupported_joint_unsafe")
    local_deadline = started + float(time_limit)
    if deadline is not None:
        if not math.isfinite(float(deadline)):
            raise ValueError("selector deadline must be finite")
        local_deadline = min(local_deadline, float(deadline))
    if time.monotonic() >= local_deadline:
        return empty("deadline_before")

    if int(output_layer_id) not in after:
        raise ValueError(f"missing output bounds for layer {output_layer_id}")
    output_bounds = _as_bounds(
        after[int(output_layer_id)],
        layer_id=int(output_layer_id),
    )
    output_lower = output_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    output_upper = output_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    if (
        output_lower.size != C_np.shape[1]
        or output_upper.size != C_np.shape[1]
        or not np.all(np.isfinite(output_lower))
        or not np.all(np.isfinite(output_upper))
    ):
        raise ValueError("output bounds/property width mismatch")
    positive = np.maximum(C_np, 0.0)
    negative = np.minimum(C_np, 0.0)
    hardness = (
        positive @ output_upper
        + negative @ output_lower
        - thresholds_np
    )
    survivors = np.flatnonzero(hardness >= 0.0).astype(np.int64, copy=False)
    if not survivors.size:
        return empty("no_interval_survivors", rivals_total=int(C_np.shape[0]))

    all_relu_layers = [
        layer
        for layer in net.layers
        if _kind_token(layer.kind) == "RELU"
    ]
    all_relu_ids = {int(layer.id) for layer in all_relu_layers}
    if allowed_relu_layer_ids is None:
        allowed_relu_ids = set(all_relu_ids)
    else:
        allowed_relu_ids = set()
        for raw_layer_id in allowed_relu_layer_ids:
            if isinstance(raw_layer_id, (bool, np.bool_)) or not isinstance(
                raw_layer_id, (Integral, np.integer)
            ):
                raise ValueError("allowed ReLU layer ids must be integers")
            layer_id = int(raw_layer_id)
            if layer_id in allowed_relu_ids:
                raise ValueError("allowed ReLU layer ids must be unique")
            allowed_relu_ids.add(layer_id)
        unknown_allowed = allowed_relu_ids - all_relu_ids
        if unknown_allowed:
            raise ValueError(
                "allowed ReLU layer ids contain non-ReLU layers: "
                f"{sorted(unknown_allowed)}"
            )
    if not allowed_relu_ids:
        return empty("no_allowed_relu_layers")
    relu_width_sum = sum(len(layer.out_vars) for layer in all_relu_layers)
    if relu_width_sum <= 0:
        return empty("no_relu_layers")
    rival_capacity = max(1, int(max_adjoint_cells) // int(relu_width_sum))
    survivor_order = sorted(
        survivors.tolist(),
        key=lambda row: (-float(hardness[row]), int(row)),
    )
    selected_rivals = np.asarray(
        survivor_order[:rival_capacity],
        dtype=np.int64,
    )
    all_rivals_processed = selected_rivals.size == survivors.size

    scales = np.maximum(
        np.maximum(
            np.abs(C_np[selected_rivals]).sum(axis=1),
            np.abs(thresholds_np[selected_rivals]),
        ),
        1.0e-30,
    )
    normalized_C = C_np[selected_rivals] / scales[:, None]
    normalized_hardness = hardness[selected_rivals] / scales

    bounds_dict: Dict[int, Bounds] = {
        int(layer_id): _as_bounds(value, layer_id=int(layer_id))
        for layer_id, value in after.items()
    }
    for layer in all_relu_layers:
        layer_id = int(layer.id)
        if layer_id not in before:
            raise ValueError(f"missing before fact for ReLU {layer_id}")
        bounds_dict[layer_id] = _as_bounds(
            before[layer_id],
            layer_id=layer_id,
        )
    from act.util.device_manager import get_default_device, get_default_dtype

    target_device = get_default_device()
    target_dtype = get_default_dtype()
    bounds_dict = {
        layer_id: Bounds(
            bounds.lb.to(device=target_device, dtype=target_dtype),
            bounds.ub.to(device=target_device, dtype=target_dtype),
        )
        for layer_id, bounds in bounds_dict.items()
    }
    objective = torch.as_tensor(
        -normalized_C,
        dtype=target_dtype,
        device=target_device,
    )
    original_hash = _binary64_sha256(C_np, thresholds_np, kind=kind_name)

    from act.back_end.solver.solver_dual import DualSolver

    with torch.no_grad():
        result = DualSolver().compute_certified_bound(
            net,
            bounds_dict,
            objective,
            M=int(selected_rivals.size),
            optimize=False,
            return_nu_per_layer=True,
        )
    if time.monotonic() >= local_deadline:
        return empty(
            "deadline_after_adjoint",
            rivals_total=int(C_np.shape[0]),
            interval_survivors=int(survivors.size),
            rivals_processed=int(selected_rivals.size),
            adjoint_cells_estimate=int(
                selected_rivals.size * relu_width_sum
            ),
        )
    if result.nu_per_layer is None:
        return empty("dual_returned_no_adjoints")
    if _binary64_sha256(C_np, thresholds_np, kind=kind_name) != original_hash:
        raise RuntimeError("property arrays changed during residual selection")
    # DualSolver computed a lower-bound adjoint for -C.  Negating it recovers
    # the signed derivative convention of the final violation form C@y-t.
    property_adjoints = {
        int(layer_id): -value
        for layer_id, value in result.nu_per_layer.items()
        if int(layer_id) in allowed_relu_ids
    }
    if not property_adjoints:
        return empty(
            "dual_returned_no_allowed_adjoints",
            allowed_relu_layer_ids=sorted(allowed_relu_ids),
        )
    plan = plan_from_property_adjoints(
        property_adjoints,
        before,
        budget=int(budget),
        rival_ids=selected_rivals.tolist(),
        rival_hardness=normalized_hardness.tolist(),
        all_rivals_processed=bool(all_rivals_processed),
        property_sha256=property_hash,
        pool_per_rival=int(pool_per_rival),
        phase_joint_focus_after_first=bool(
            phase_joint_focus_after_first
        ),
    )
    receipt = dict(plan.receipt)
    receipt.update(
        {
            "elapsed_seconds": float(time.monotonic() - started),
            "time_limit": float(time_limit),
            "rivals_total": int(C_np.shape[0]),
            "interval_survivors": int(survivors.size),
            "rivals_processed": int(selected_rivals.size),
            "all_interval_survivors_processed": bool(all_rivals_processed),
            "adjoint_cells_estimate": int(
                selected_rivals.size * relu_width_sum
            ),
            "max_adjoint_cells": int(max_adjoint_cells),
            "adjoint_device": str(target_device),
            "adjoint_dtype": str(target_dtype),
            "property_sha256": property_hash,
            "allowed_relu_layer_ids": sorted(allowed_relu_ids),
            "allowed_relu_layer_count": int(len(allowed_relu_ids)),
            "candidate_only": True,
            "proof_authority": False,
        }
    )
    return PropertyResidualPlan(
        targets=plan.targets,
        property_sha256=plan.property_sha256,
        targets_sha256=plan.targets_sha256,
        receipt=receipt,
    )


def select_property_sparse_query_rows(
    *,
    net: Net,
    before: Mapping[int, Any],
    after: Mapping[int, Any],
    C: Any,
    thresholds: Any,
    kind: Any,
    output_layer_id: int,
    layer_quotas: Mapping[int, int],
    time_limit: float,
    deadline: Optional[float] = None,
    max_adjoint_cells: int = 30_000_000,
    pool_per_rival: int = 64,
) -> PropertySparseQueryPlan:
    """Run exactly one property-adjoint pass for all requested query stages."""

    started = time.monotonic()
    quotas = _validated_layer_quotas(layer_quotas)
    if int(pool_per_rival) <= 0:
        raise ValueError("pool_per_rival must be positive")
    if any(quota > int(pool_per_rival) for _layer_id, quota in quotas):
        raise ValueError("each layer quota must not exceed pool_per_rival")
    if not math.isfinite(float(time_limit)) or float(time_limit) < 0.0:
        raise ValueError("sparse selector time limit must be finite/nonnegative")
    if int(max_adjoint_cells) <= 0:
        raise ValueError("max_adjoint_cells must be positive")
    C_np, thresholds_np = _finite_property(C, thresholds)
    kind_name = _kind_token(kind)
    property_hash = _binary64_sha256(C_np, thresholds_np, kind=kind_name)

    def empty(status: str, **extra: Any) -> PropertySparseQueryPlan:
        targets: Tuple[PropertySparseQueryTarget, ...] = ()
        selection_hash = _sparse_selection_sha256(
            property_hash, quotas, targets
        )
        return PropertySparseQueryPlan(
            targets=targets,
            property_sha256=property_hash,
            selection_sha256=selection_hash,
            receipt={
                "schema": "property_sparse_query_selector_v3",
                "status": str(status),
                "candidate_only": True,
                "proof_authority": False,
                "property_sha256": property_hash,
                "selection_sha256": selection_hash,
                "layer_quotas": [
                    [int(layer_id), int(quota)]
                    for layer_id, quota in quotas
                ],
                "targets_selected": 0,
                "adjoint_solver_calls": 0,
                "elapsed_seconds": float(time.monotonic() - started),
                **extra,
            },
        )

    if not any(quota > 0 for _layer_id, quota in quotas):
        return empty("disabled_empty_quotas")
    if float(time_limit) == 0.0:
        return empty("disabled")
    if kind_name == "UNSAFE_LINEAR":
        return empty("unsupported_joint_unsafe")
    local_deadline = started + float(time_limit)
    if deadline is not None:
        if not math.isfinite(float(deadline)):
            raise ValueError("selector deadline must be finite")
        local_deadline = min(local_deadline, float(deadline))
    if time.monotonic() >= local_deadline:
        return empty("deadline_before")

    if int(output_layer_id) not in after:
        raise ValueError(f"missing output bounds for layer {output_layer_id}")
    output_bounds = _as_bounds(
        after[int(output_layer_id)],
        layer_id=int(output_layer_id),
    )
    output_lower = output_bounds.lb.detach().cpu().double().numpy().reshape(-1)
    output_upper = output_bounds.ub.detach().cpu().double().numpy().reshape(-1)
    if (
        output_lower.size != C_np.shape[1]
        or output_upper.size != C_np.shape[1]
        or not np.all(np.isfinite(output_lower))
        or not np.all(np.isfinite(output_upper))
    ):
        raise ValueError("output bounds/property width mismatch")
    positive = np.maximum(C_np, 0.0)
    negative = np.minimum(C_np, 0.0)
    hardness = (
        positive @ output_upper
        + negative @ output_lower
        - thresholds_np
    )
    survivors = np.flatnonzero(hardness >= 0.0).astype(
        np.int64, copy=False
    )
    if not survivors.size:
        return empty("no_interval_survivors", rivals_total=int(C_np.shape[0]))

    relu_layers = [
        layer
        for layer in net.layers
        if _kind_token(layer.kind) == "RELU"
    ]
    relu_ids = {int(layer.id) for layer in relu_layers}
    requested_ids = {layer_id for layer_id, quota in quotas if quota > 0}
    missing_requested = sorted(requested_ids.difference(relu_ids))
    if missing_requested:
        raise ValueError(
            f"layer quotas identify non-ReLU layers: {missing_requested}"
        )
    relu_width_sum = sum(len(layer.out_vars) for layer in relu_layers)
    if relu_width_sum <= 0:
        return empty("no_relu_layers")
    rival_capacity = max(
        1, int(max_adjoint_cells) // int(relu_width_sum)
    )
    survivor_order = sorted(
        survivors.tolist(),
        key=lambda row: (-float(hardness[row]), int(row)),
    )
    selected_rivals = np.asarray(
        survivor_order[:rival_capacity],
        dtype=np.int64,
    )
    all_rivals_processed = selected_rivals.size == survivors.size
    scales = np.maximum(
        np.maximum(
            np.abs(C_np[selected_rivals]).sum(axis=1),
            np.abs(thresholds_np[selected_rivals]),
        ),
        1.0e-30,
    )
    normalized_C = C_np[selected_rivals] / scales[:, None]
    normalized_hardness = hardness[selected_rivals] / scales

    bounds_dict: Dict[int, Bounds] = {
        int(layer_id): _as_bounds(value, layer_id=int(layer_id))
        for layer_id, value in after.items()
    }
    for layer in relu_layers:
        layer_id = int(layer.id)
        if layer_id not in before:
            raise ValueError(f"missing before fact for ReLU {layer_id}")
        bounds_dict[layer_id] = _as_bounds(
            before[layer_id], layer_id=layer_id
        )
    from act.util.device_manager import get_default_device, get_default_dtype

    target_device = get_default_device()
    target_dtype = get_default_dtype()
    bounds_dict = {
        layer_id: Bounds(
            bounds.lb.to(device=target_device, dtype=target_dtype),
            bounds.ub.to(device=target_device, dtype=target_dtype),
        )
        for layer_id, bounds in bounds_dict.items()
    }
    objective = torch.as_tensor(
        -normalized_C,
        dtype=target_dtype,
        device=target_device,
    )
    original_hash = _binary64_sha256(C_np, thresholds_np, kind=kind_name)

    from act.back_end.solver.solver_dual import DualSolver

    with torch.no_grad():
        result = DualSolver().compute_certified_bound(
            net,
            bounds_dict,
            objective,
            M=int(selected_rivals.size),
            optimize=False,
            return_nu_per_layer=True,
        )
    if time.monotonic() >= local_deadline:
        return empty(
            "deadline_after_adjoint",
            rivals_total=int(C_np.shape[0]),
            interval_survivors=int(survivors.size),
            rivals_processed=int(selected_rivals.size),
            adjoint_cells_estimate=int(
                selected_rivals.size * relu_width_sum
            ),
            adjoint_solver_calls=1,
        )
    if result.nu_per_layer is None:
        return empty("dual_returned_no_adjoints", adjoint_solver_calls=1)
    if _binary64_sha256(C_np, thresholds_np, kind=kind_name) != original_hash:
        raise RuntimeError("property arrays changed during sparse selection")
    property_adjoints = {
        int(layer_id): -value
        for layer_id, value in result.nu_per_layer.items()
    }
    plan = plan_sparse_query_rows_from_property_adjoints(
        property_adjoints,
        before,
        layer_quotas=dict(quotas),
        rival_ids=selected_rivals.tolist(),
        rival_hardness=normalized_hardness.tolist(),
        all_rivals_processed=bool(all_rivals_processed),
        property_sha256=property_hash,
        pool_per_rival=int(pool_per_rival),
    )
    receipt = dict(plan.receipt)
    receipt.update(
        {
            "elapsed_seconds": float(time.monotonic() - started),
            "time_limit": float(time_limit),
            "rivals_total": int(C_np.shape[0]),
            "interval_survivors": int(survivors.size),
            "rivals_processed": int(selected_rivals.size),
            "all_interval_survivors_processed": bool(
                all_rivals_processed
            ),
            "adjoint_cells_estimate": int(
                selected_rivals.size * relu_width_sum
            ),
            "max_adjoint_cells": int(max_adjoint_cells),
            "adjoint_device": str(target_device),
            "adjoint_dtype": str(target_dtype),
            "adjoint_solver_calls": 1,
            "candidate_only": True,
            "proof_authority": False,
        }
    )
    return PropertySparseQueryPlan(
        targets=plan.targets,
        property_sha256=plan.property_sha256,
        selection_sha256=plan.selection_sha256,
        receipt=receipt,
    )


__all__ = [
    "PropertyResidualPlan",
    "PropertyResidualTarget",
    "PropertySparseQueryPlan",
    "PropertySparseQueryTarget",
    "plan_from_property_adjoints",
    "plan_sparse_query_rows_from_property_adjoints",
    "property_correlation_layer_quotas",
    "select_property_residual_targets",
    "select_property_sparse_query_rows",
]
