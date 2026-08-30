#!/usr/bin/env python3
"""Disconnected F-prime one-owner phase-update ability probe.

This scratch program preserves the frozen 59-path cell construction and
terminal, but intercepts its one LP transaction with exactly one request-local
``highspy.Highs`` owner.  If the base model is optimal with a negative margin,
the unique update set is every tight upper row with a strictly negative row
dual.  If the base model is infeasible, the unique update set is every mapped
phase row with an exact-nonzero coefficient in a validated HiGHS dual ray.
The complete set is flipped once, the same owner is cleared and loaded once,
and at most one updated LP is solved.  There is no solver fallback or retry.

LP values, row duals, and dual rays are candidate-selection diagnostics only.
Only the unchanged raw BOX, verifier-owned zero-width interval forward, and
stored-binary64 Fraction margin terminal can authorize a falsification.

No input sampling, input-point ONNX execution, PGD, BaB/splitting/enumeration,
backward bounds, dual tightening, external-label-dependent rule, cross-request
cache, parameter scan, or runtime menu is used.  This file is disconnected
scratch and never modifies production.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import highspy
import numpy as np
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parent
ONE_MULTI_PATH = ROOT / "scratch_phase_projection_one_multi_flip_probe.py"
FIXED400 = (
    ROOT
    / "artifacts/hybridz_largecls_gates/phase_projection_gpu_csr_fixed400_20260814.jsonl"
)
OWNER_CONTRACT = ROOT / "scratch_phase_projection_highs_owner_contract.py"
OWNER_CONTRACT_SHA256 = (
    "fe9caab1d8452a9c3125696b067f2470050610e03a64fa8cd73e1ef855bcd9cf"
)
HIGHS_SMALL = 1.0e-12
HIGHS_LARGE = 1.0e15
HIGHS_INFINITE_BOUND = 1.0e20
LP_TIME_LIMIT = 30.0
PINNED_HIGHS_VERSION = "1.15.0"
PINNED_HIGHS_GITHASH = "8396001"
PINNED_SIMPLEX_SCALE_STRATEGY = 2
PINNED_SIMPLEX_STRATEGY = 1
PINNED_RANDOM_SEED = 0

# This is an offline control manifest, not an outcome-label table.  The runtime
# rule receives only the chosen ONNX/VNNLIB pair and never reads benchmark
# results, SAT labels, old margins, or expected outcomes.
CONTROLS: Dict[str, Tuple[str, str, str]] = {
    "cifar100_medium_iid2": (
        "cifar100_2024",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_medium.onnx",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib",
    ),
    "tinyimagenet_medium_iid143": (
        "tinyimagenet_2024",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/onnx/TinyImageNet_resnet_medium.onnx",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/vnnlib/TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib",
    ),
    "cifar100_large_iid153": (
        "cifar100_2024",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_large.onnx",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_large_prop_idx_4652_sidx_1371_eps_0.0039.vnnlib",
    ),
    "cifar100_large_iid166": (
        "cifar100_2024",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/onnx/CIFAR100_resnet_large.onnx",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/cifar100_2024/vnnlib/CIFAR100_resnet_large_prop_idx_2630_sidx_1753_eps_0.0039.vnnlib",
    ),
    "tinyimagenet_medium_iid153": (
        "tinyimagenet_2024",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/onnx/TinyImageNet_resnet_medium.onnx",
        "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/vnnlib/TinyImageNet_resnet_medium_prop_idx_2493_sidx_4209_eps_0.0039.vnnlib",
    ),
}


class ProbeStop(RuntimeError):
    """Fail-closed disconnected-probe stop."""

    def __init__(self, reason: str, **details: Any):
        super().__init__(reason)
        self.reason = reason
        self.details = details


class OwnerPoisoned(ProbeStop):
    """A warning/error or malformed solver result permanently poisons owner."""


def add_secondary_type_note(primary: BaseException, note: str) -> None:
    try:
        primary.add_note(note)
    except BaseException:
        pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def restrictions() -> Dict[str, Any]:
    return {
        "production_modified": False,
        "input_center_boundary_or_random_sampling_used": False,
        "onnx_input_point_execution_used": False,
        "pgd_used": False,
        "bab_split_or_enumeration_used": False,
        "backward_bounds_used": False,
        "dual_tightening_used": False,
        "candidate_lp_marginal_or_ray_has_authority": False,
        "external_labels_read_by_runtime_rule": False,
        "second_solver_used": False,
        "fallback_or_runtime_menu": False,
        "parameter_scan": False,
        "phase_updates_max": 1,
        "resolves_after_base_max": 1,
    }


def fraction_upper(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ProbeStop("exact loader RHS overflowed")
    if Fraction.from_float(rounded) < value:
        rounded = float(np.nextafter(rounded, np.inf))
    if not math.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise ProbeStop("exact loader RHS could not be rounded outward")
    return rounded


def filter_tiny_upper_rows(
    matrix: sp.csr_matrix,
    rhs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Project deleted tiny terms from ``A*x <= b`` with exact dyadics.

    For each removed term the safe projected row is
    ``A_keep*x <= b - min(a*l, a*u)``.  The exact stored-binary64 sum is
    rounded only once toward positive infinity.  Logical screening and row
    mapping always remain those of the unfiltered matrix.
    """

    if not sp.isspmatrix_csr(matrix) or matrix.dtype != np.dtype(np.float64):
        raise ProbeStop("loader requires an existing float64 CSR")
    logical = matrix.copy()
    rhs = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1)
    lower = np.ascontiguousarray(lower, dtype=np.float64).reshape(-1)
    upper = np.ascontiguousarray(upper, dtype=np.float64).reshape(-1)
    if (
        logical.shape != (rhs.size, lower.size)
        or lower.shape != upper.shape
        or not np.all(np.isfinite(logical.data))
        or not np.all(np.isfinite(rhs))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise ProbeStop("loader input shape/finiteness/bounds contract failed")
    int32_max = np.iinfo(np.int32).max
    if (
        logical.indptr.dtype != np.dtype(np.int32)
        or logical.indices.dtype != np.dtype(np.int32)
        or logical.data.dtype != np.dtype(np.float64)
        or not logical.indptr.flags.c_contiguous
        or not logical.indices.flags.c_contiguous
        or not logical.data.flags.c_contiguous
    ):
        raise ProbeStop("loader requires contiguous int32/int32/float64 CSR arrays")
    if (
        logical.shape[0] > int32_max
        or logical.shape[1] > int32_max
        or logical.nnz > int32_max
        or logical.indptr.shape != (logical.shape[0] + 1,)
        or int(logical.indptr[0]) != 0
        or int(logical.indptr[-1]) != logical.nnz
        or np.any(np.diff(logical.indptr) < 0)
        or (logical.indices.size and np.min(logical.indices) < 0)
        or (logical.indices.size and np.max(logical.indices) >= logical.shape[1])
        or (logical.indptr.size and np.max(logical.indptr) > int32_max)
        or (logical.indices.size and np.max(logical.indices) > int32_max)
    ):
        raise ProbeStop("loader CSR structure/int32 contract failed")
    if not logical.has_sorted_indices or not logical.has_canonical_format:
        raise ProbeStop("loader rejects unsorted or duplicate CSR input")
    if np.any(logical.data == 0.0):
        raise ProbeStop("loader rejects explicit-zero CSR input")
    if np.any(np.abs(logical.data) >= HIGHS_LARGE):
        raise ProbeStop("loader rejects matrix coefficient at HiGHS large threshold")

    tiny = (logical.data != 0.0) & (np.abs(logical.data) <= HIGHS_SMALL)
    deleted = int(np.count_nonzero(tiny))
    loaded_rhs = rhs.copy()
    maximum_relaxation = 0.0
    affected_rows = 0
    if deleted:
        row_for_entry = np.repeat(
            np.arange(logical.shape[0], dtype=np.int64), np.diff(logical.indptr)
        )
        affected = np.unique(row_for_entry[tiny])
        affected_rows = int(affected.size)
        for row in affected:
            start, stop = int(logical.indptr[row]), int(logical.indptr[row + 1])
            local_tiny = tiny[start:stop]
            exact_rhs = Fraction.from_float(float(rhs[row]))
            columns = logical.indices[start:stop][local_tiny]
            values = logical.data[start:stop][local_tiny]
            for column, value in zip(columns, values):
                coefficient = Fraction.from_float(float(value))
                lo = coefficient * Fraction.from_float(float(lower[column]))
                hi = coefficient * Fraction.from_float(float(upper[column]))
                exact_rhs -= min(lo, hi)
            loaded_rhs[row] = fraction_upper(exact_rhs)
            maximum_relaxation = max(
                maximum_relaxation, float(loaded_rhs[row] - rhs[row])
            )
        kept = ~tiny
        row_counts = np.bincount(
            row_for_entry[kept], minlength=logical.shape[0]
        ).astype(np.int64, copy=False)
        loaded_indptr64 = np.empty(logical.shape[0] + 1, dtype=np.int64)
        loaded_indptr64[0] = 0
        np.cumsum(row_counts, out=loaded_indptr64[1:])
        if int(loaded_indptr64[-1]) > int32_max:
            raise ProbeStop("tiny partition exceeds int32 CSR capacity")
        loaded_indptr = loaded_indptr64.astype(np.int32)
        logical = sp.csr_matrix(
            (
                logical.data[kept].copy(),
                logical.indices[kept].copy(),
                loaded_indptr,
            ),
            shape=logical.shape,
        )
        if (
            not logical.has_sorted_indices
            or not logical.has_canonical_format
            or np.any(logical.data == 0.0)
        ):
            raise ProbeStop("explicit tiny partition broke CSR canonical form")

    if not np.all(np.isfinite(loaded_rhs)):
        raise ProbeStop("loader outward RHS is nonfinite")
    return logical, loaded_rhs, {
        "deleted_tiny_nnz": deleted,
        "affected_rows": affected_rows,
        "maximum_rhs_relaxation": maximum_relaxation,
        "formula": "b - exact_sum(min(a*l,a*u)), rounded once toward +inf",
    }


@dataclass
class SolveRecord:
    tag: str
    model_status: Any
    logical_rows: int
    logical_nnz: int
    loaded_nnz: int
    loaded_rhs: np.ndarray
    loaded_matrix: sp.csr_matrix
    loader: Dict[str, Any]
    row_mapping: Optional[Tuple[Tuple[int, int, int], ...]] = None
    row_mapping_sha256: Optional[str] = None
    objective_nonzero_count: int = 0
    objective_tiny_nonzero_count: int = 0
    objective_min_nonzero_abs: Optional[float] = None
    factors: Optional[np.ndarray] = None
    row_value: Optional[np.ndarray] = None
    row_dual: Optional[np.ndarray] = None
    objective_value: Optional[float] = None
    dual_ray: Optional[np.ndarray] = None
    dual_ray_validation: Optional[Dict[str, Any]] = None
    simplex_iterations: int = 0

    @property
    def optimal(self) -> bool:
        return self.model_status == highspy.HighsModelStatus.kOptimal

    @property
    def infeasible(self) -> bool:
        return self.model_status == highspy.HighsModelStatus.kInfeasible


class SafeHighsOwner:
    """Exactly one fail-closed request-local HiGHS owner."""

    def __init__(self, *, deadline: Optional[float]):
        self.highs: Any = None
        self.deadline = deadline
        self.poisoned = False
        self.state = "NEW"
        self.clear_attempted = False
        self.model_loaded = False
        self.model_loads = 0
        self.solve_count = 0
        self.dual_ray_exist_calls = 0
        self.dual_ray_calls = 0
        self.records: list[SolveRecord] = []
        try:
            self.highs = highspy.Highs()
            self._configure()
            self.state = "READY_BASE"
        except BaseException as primary:
            self.poisoned = True
            self.state = "POISONED"
            if self.highs is not None and not self.clear_attempted:
                try:
                    self.clear_attempted = True
                    status = self.highs.clear()
                    if status != highspy.HighsStatus.kOk:
                        add_secondary_type_note(
                            primary,
                            "secondary constructor cleanup failure type=HighsStatus"
                        )
                except BaseException as cleanup:
                    add_secondary_type_note(
                        primary,
                        "secondary constructor cleanup failure type="
                        f"{type(cleanup).__name__}"
                    )
            raise

    def _poison(self, reason: str, **details: Any) -> None:
        self.poisoned = True
        self.state = "POISONED"
        primary = OwnerPoisoned(reason, **details)
        try:
            status = None
            if self.highs is not None and not self.clear_attempted:
                self.clear_attempted = True
                status = self.highs.clear()
            if (
                self.highs is not None
                and status is not None
                and status != highspy.HighsStatus.kOk
            ):
                add_secondary_type_note(
                    primary,
                    "secondary HiGHS poison cleanup failure type=HighsStatus",
                )
        except BaseException as cleanup:
            add_secondary_type_note(
                primary,
                f"secondary HiGHS poison cleanup failure type={type(cleanup).__name__}"
            )
        raise primary

    def _require_live(self, operation: str) -> None:
        if self.state == "CLOSED":
            raise OwnerPoisoned(
                "closed HiGHS owner rejects every entry", operation=operation
            )
        if self.state == "POISONED" or self.poisoned:
            raise OwnerPoisoned(
                "poisoned HiGHS owner rejects every entry", operation=operation
            )

    def _require_ok(self, status: Any, operation: str) -> None:
        if status != highspy.HighsStatus.kOk:
            self._poison(
                "HiGHS warning/error poisoned the only owner",
                operation=operation,
                highs_status_type=type(status).__name__,
            )

    def _configure(self) -> None:
        if (
            self.highs.version() != PINNED_HIGHS_VERSION
            or self.highs.githash() != PINNED_HIGHS_GITHASH
        ):
            self._poison(
                "unpinned highspy/HiGHS implementation",
                version=self.highs.version(),
                githash=self.highs.githash(),
            )
        for key, value in (
            ("output_flag", False),
            ("solver", "simplex"),
            ("simplex_strategy", PINNED_SIMPLEX_STRATEGY),
            ("simplex_scale_strategy", PINNED_SIMPLEX_SCALE_STRATEGY),
            ("presolve", "off"),
            ("threads", 1),
            ("parallel", "off"),
            ("random_seed", PINNED_RANDOM_SEED),
            ("allow_unbounded_or_infeasible", False),
            ("small_matrix_value", HIGHS_SMALL),
            ("large_matrix_value", HIGHS_LARGE),
            ("infinite_bound", HIGHS_INFINITE_BOUND),
            ("primal_feasibility_tolerance", 1.0e-9),
            ("dual_feasibility_tolerance", 1.0e-9),
        ):
            self._require_ok(self.highs.setOptionValue(key, value), f"set {key}")
        self._verify_pinned_options("initial")
        self._require_ok(
            self.highs.changeObjectiveSense(highspy.ObjSense.kMinimize),
            "set objective sense",
        )
        sense_status, sense = self.highs.getObjectiveSense()
        self._require_ok(sense_status, "get objective sense")
        if sense != highspy.ObjSense.kMinimize:
            self._poison("HiGHS objective sense is not minimize")

    def _verify_pinned_options(self, stage: str) -> None:
        for key, expected in (
            ("output_flag", False),
            ("solver", "simplex"),
            ("simplex_strategy", PINNED_SIMPLEX_STRATEGY),
            ("simplex_scale_strategy", PINNED_SIMPLEX_SCALE_STRATEGY),
            ("presolve", "off"),
            ("threads", 1),
            ("parallel", "off"),
            ("random_seed", PINNED_RANDOM_SEED),
            ("allow_unbounded_or_infeasible", False),
            ("small_matrix_value", HIGHS_SMALL),
            ("large_matrix_value", HIGHS_LARGE),
            ("infinite_bound", HIGHS_INFINITE_BOUND),
            ("primal_feasibility_tolerance", 1.0e-9),
            ("dual_feasibility_tolerance", 1.0e-9),
        ):
            status, observed = self.highs.getOptionValue(key)
            self._require_ok(status, f"{stage} get {key}")
            if observed != expected:
                self._poison(
                    "HiGHS option readback disagrees with pinned value",
                    stage=stage,
                    option=key,
                    expected=expected,
                    observed=observed,
                )

    def check_deadline(self, stage: str) -> None:
        self._require_live("check_deadline")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._poison("deadline expired", stage=stage)

    def _remaining(self, requested: float) -> float:
        self._require_live("remaining")
        remaining = float(requested)
        if self.deadline is not None:
            remaining = min(remaining, self.deadline - time.monotonic())
        if not math.isfinite(remaining) or remaining <= 0.0:
            self._poison("deadline expired before HiGHS transaction")
        return remaining

    def load_and_solve(
        self,
        *,
        tag: str,
        cost: np.ndarray,
        matrix: sp.csr_matrix,
        rhs: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        time_limit: float,
        row_mapping: Optional[Sequence[Tuple[int, int, int]]],
        selector_readback: bool,
    ) -> SolveRecord:
        self.check_deadline(f"{tag}:entry")
        if tag == "base":
            if not (
                self.state == "READY_BASE"
                and self.model_loads == 0
                and not self.model_loaded
                and selector_readback is True
                and row_mapping is not None
            ):
                self._poison("invalid base load state or selector contract")
            self.state = "BASE_LOADING"
        elif tag == "one_update":
            if not (
                self.state == "BASE_SOLVED"
                and self.model_loads == 1
                and self.model_loaded
                and selector_readback is False
                and row_mapping is None
            ):
                self._poison("invalid one-update load state or selector contract")
            self.state = "UPDATED_LOADING"
            self.check_deadline(f"{tag}:before_clear")
            self._require_ok(self.highs.clearModel(), "clearModel before one update")
            self.model_loaded = False
            if (
                self.highs.getNumCol() != 0
                or self.highs.getNumRow() != 0
                or self.highs.getNumNz() != 0
            ):
                self._poison("clearModel postcondition failed")
            self._verify_pinned_options("after_clearModel")
            sense_status, sense = self.highs.getObjectiveSense()
            self._require_ok(sense_status, "after_clearModel get objective sense")
            if sense != highspy.ObjSense.kMinimize:
                self._poison("objective sense changed after clearModel")
        else:
            self._poison("unknown model load tag")

        cost = np.ascontiguousarray(cost, dtype=np.float64).reshape(-1)
        if not sp.isspmatrix_csr(matrix) or matrix.dtype != np.dtype(np.float64):
            self._poison("model is not a pre-existing float64 CSR")
        logical = matrix.copy()
        rhs = np.ascontiguousarray(rhs, dtype=np.float64).reshape(-1)
        lower = np.ascontiguousarray(lower, dtype=np.float64).reshape(-1)
        upper = np.ascontiguousarray(upper, dtype=np.float64).reshape(-1)
        if (
            cost.size != lower.size
            or logical.shape != (rhs.size, cost.size)
            or not np.all(np.isfinite(cost))
        ):
            self._poison("objective/model dimension or finiteness contract failed")
        if (
            np.any(np.abs(cost) >= HIGHS_INFINITE_BOUND)
            or np.any(np.abs(rhs) >= HIGHS_INFINITE_BOUND)
            or np.any(np.abs(lower) >= HIGHS_INFINITE_BOUND)
            or np.any(np.abs(upper) >= HIGHS_INFINITE_BOUND)
        ):
            self._poison("objective/RHS/bound reaches HiGHS infinite-bound threshold")
        objective_nonzero = cost != 0.0
        objective_tiny = objective_nonzero & (np.abs(cost) <= HIGHS_SMALL)
        objective_nonzero_count = int(np.count_nonzero(objective_nonzero))
        objective_tiny_count = int(np.count_nonzero(objective_tiny))
        objective_min_abs = (
            float(np.min(np.abs(cost[objective_nonzero])))
            if objective_nonzero_count
            else None
        )
        if objective_tiny_count:
            self._poison(
                "objective contains an unproven tiny nonzero coefficient",
                tiny_nonzero_count=objective_tiny_count,
                minimum_nonzero_abs=objective_min_abs,
            )
        if tag == "base":
            if row_mapping is None or len(row_mapping) != logical.shape[0]:
                self._poison("base load lacks a synchronous sealed row mapping")
            sealed_mapping = tuple(
                (int(layer), int(position), int(row))
                for layer, position, row in row_mapping
            )
            mapping_sha256 = selected_digest(sealed_mapping)
        else:
            sealed_mapping = None
            mapping_sha256 = None

        self.check_deadline(f"{tag}:before_safe_filter")
        loaded, loaded_rhs, loader_receipt = filter_tiny_upper_rows(
            logical, rhs, lower, upper
        )
        self.check_deadline(f"{tag}:after_safe_filter")
        n = int(cost.size)
        self._require_ok(
            self.highs.setOptionValue("time_limit", self._remaining(time_limit)),
            "set time_limit",
        )
        sense_status, sense = self.highs.getObjectiveSense()
        self._require_ok(sense_status, "read objective sense before load")
        if sense != highspy.ObjSense.kMinimize:
            self._poison("objective sense changed before model load")
        self.check_deadline(f"{tag}:before_addCols")
        self._require_ok(
            self.highs.addCols(
                n,
                cost,
                lower,
                upper,
                0,
                np.zeros(n + 1, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
            ),
            "addCols",
        )
        lp = self.highs.getLp()
        if not (
            np.array_equal(np.asarray(lp.col_cost_, dtype=np.float64), cost)
            and np.array_equal(np.asarray(lp.col_lower_, dtype=np.float64), lower)
            and np.array_equal(np.asarray(lp.col_upper_, dtype=np.float64), upper)
        ):
            self._poison("HiGHS changed objective coefficients or column bounds")
        self.check_deadline(f"{tag}:before_addRows")
        self._require_ok(
            self.highs.addRows(
                int(loaded.shape[0]),
                np.full(loaded.shape[0], -np.inf, dtype=np.float64),
                loaded_rhs,
                int(loaded.nnz),
                loaded.indptr.astype(np.int32, copy=False),
                loaded.indices.astype(np.int32, copy=False),
                loaded.data,
            ),
            "addRows",
        )
        self.model_loaded = True
        self.model_loads += 1
        if (
            self.highs.getNumCol() != n
            or self.highs.getNumRow() != loaded.shape[0]
            or self.highs.getNumNz() != loaded.nnz
        ):
            self._poison("HiGHS post-load row/column/nnz contract failed")

        self.check_deadline(f"{tag}:before_run")
        latest_remaining = self._remaining(time_limit)
        self._require_ok(
            self.highs.setOptionValue("time_limit", latest_remaining),
            "latest set time_limit",
        )
        limit_status, observed_limit = self.highs.getOptionValue("time_limit")
        self._require_ok(limit_status, "latest get time_limit")
        if observed_limit != latest_remaining:
            self._poison("latest time_limit did not round-trip")
        self._require_ok(self.highs.run(), "run")
        self.check_deadline(f"{tag}:after_run")
        self.solve_count += 1
        status = self.highs.getModelStatus()
        record = SolveRecord(
            tag=tag,
            model_status=status,
            logical_rows=int(logical.shape[0]),
            logical_nnz=int(logical.nnz),
            loaded_nnz=int(loaded.nnz),
            loaded_rhs=loaded_rhs,
            loaded_matrix=loaded,
            loader=loader_receipt,
            row_mapping=sealed_mapping,
            row_mapping_sha256=mapping_sha256,
            objective_nonzero_count=objective_nonzero_count,
            objective_tiny_nonzero_count=objective_tiny_count,
            objective_min_nonzero_abs=objective_min_abs,
            simplex_iterations=int(self.highs.getInfo().simplex_iteration_count),
        )

        if status == highspy.HighsModelStatus.kOptimal:
            self.check_deadline(f"{tag}:before_primal_readback")
            solution = self.highs.getSolution()
            self.check_deadline(f"{tag}:after_primal_readback")
            factors = np.asarray(solution.col_value, dtype=np.float64)
            if not (
                solution.value_valid
                and factors.shape == (n,)
                and np.all(np.isfinite(factors))
            ):
                self._poison("optimal HiGHS primal readback is malformed")
            if np.any(factors < lower - 1.0e-9) or np.any(
                factors > upper + 1.0e-9
            ):
                self._poison("optimal HiGHS primal violates a factor bound")
            record.factors = factors
            if selector_readback:
                row_value = np.asarray(solution.row_value, dtype=np.float64)
                row_dual = np.asarray(solution.row_dual, dtype=np.float64)
                if not (
                    solution.dual_valid
                    and row_value.shape == (logical.shape[0],)
                    and row_dual.shape == (logical.shape[0],)
                    and np.all(np.isfinite(row_value))
                    and np.all(np.isfinite(row_dual))
                ):
                    self._poison("base HiGHS selector readback is malformed")
                if np.any(row_value > loaded_rhs + 1.0e-9):
                    self._poison("base HiGHS primal violates an upper row")
                if np.any(row_dual > 1.0e-9):
                    self._poison("MIN upper-row dual has the wrong sign")
                record.row_value = row_value
                record.row_dual = row_dual
            record.objective_value = float(self.highs.getObjectiveValue())
            self.check_deadline(f"{tag}:after_objective_readback")
            if not math.isfinite(record.objective_value):
                self._poison("optimal HiGHS objective is nonfinite")
        elif status == highspy.HighsModelStatus.kInfeasible and tag == "base":
            self.check_deadline("base:before_getDualRayExist")
            exist_status, ray_exists = self.highs.getDualRayExist()
            self.dual_ray_exist_calls += 1
            self.check_deadline("base:after_getDualRayExist")
            self._require_ok(exist_status, "getDualRayExist")
            if ray_exists is not True:
                self._poison("base infeasible model reports no dual ray")
            if self.dual_ray_exist_calls != 1 or self.dual_ray_calls != 0:
                self._poison("dual-ray API call count contract failed before getDualRay")
            self.check_deadline("base:before_getDualRay")
            ray_status, has_ray, ray_value = self.highs.getDualRay()
            self.dual_ray_calls += 1
            self.check_deadline("base:after_getDualRay")
            self._require_ok(ray_status, "getDualRay")
            if has_ray is not True:
                self._poison("getDualRay did not return a ray")
            if self.dual_ray_calls != 1:
                self._poison("getDualRay was invoked more than once")
            ray = np.array(ray_value, dtype=np.float64, order="C", copy=True)
            ray.setflags(write=False)
            try:
                validation = validate_dual_ray_shape_sign_support(
                    ray=ray,
                    has_ray=has_ray,
                    rows=int(logical.shape[0]),
                )
            except ProbeStop as exc:
                self._poison(exc.reason, **exc.details)
            record.dual_ray = ray
            record.dual_ray_validation = validation
        else:
            # Resource exhaustion and every other solver status fail closed.
            pass
        if tag == "base":
            self.state = (
                "BASE_SOLVED"
                if status
                in {
                    highspy.HighsModelStatus.kOptimal,
                    highspy.HighsModelStatus.kInfeasible,
                }
                else "BASE_FAILED"
            )
        else:
            self.state = "UPDATED_SOLVED"
        self.records.append(record)
        return record

    def scipy_compatible_linprog(
        self,
        cost: np.ndarray,
        *,
        A_ub: sp.csr_matrix,
        b_ub: np.ndarray,
        bounds: Sequence[Tuple[float, float]],
        method: str,
        options: Mapping[str, Any],
    ) -> Any:
        self._require_live("scipy_compatible_linprog")
        if method != "highs-ds" or bool(options.get("presolve", True)):
            self._poison("frozen builder requested a noncontract LP mode")
        requested = float(options.get("time_limit", LP_TIME_LIMIT))
        bound_array = np.asarray(bounds, dtype=np.float64)
        if bound_array.ndim != 2 or bound_array.shape[1] != 2:
            self._poison("frozen builder supplied malformed factor bounds")
        if not sp.isspmatrix_csr(A_ub) or A_ub.dtype != np.dtype(np.float64):
            self._poison("frozen builder did not supply a float64 CSR base model")
        row_mapping = seal_base_screened_row_mapping_from_caller(
            caller=sys._getframe(1), matrix=A_ub
        )
        record = self.load_and_solve(
            tag="base",
            cost=np.asarray(cost, dtype=np.float64),
            matrix=A_ub,
            rhs=np.asarray(b_ub, dtype=np.float64),
            lower=bound_array[:, 0],
            upper=bound_array[:, 1],
            time_limit=requested,
            row_mapping=row_mapping,
            selector_readback=True,
        )
        if record.optimal:
            return SimpleNamespace(
                success=True,
                x=record.factors,
                status=0,
                message="single-owner highspy optimal",
            )
        if record.infeasible:
            return SimpleNamespace(
                success=False,
                x=None,
                status=2,
                message="single-owner highspy infeasible with validated dual ray",
            )
        return SimpleNamespace(
            success=False,
            x=None,
            status=4,
            message=f"single-owner highspy fail-closed status {record.model_status}",
        )

    def close(self) -> Tuple[str, Optional[str]]:
        """Bounded cleanup reported only by a constant, never by exception."""
        if self.state == "CLOSED":
            raise OwnerPoisoned("closed HiGHS owner rejects repeated close")
        if self.clear_attempted:
            self.model_loaded = False
            self.state = "CLOSED"
            return "cleanup_ok", None
        try:
            self.clear_attempted = True
            status = self.highs.clear()
            if status != highspy.HighsStatus.kOk:
                return "cleanup_failed", "HighsStatus"
        except BaseException as cleanup:
            return "cleanup_failed", type(cleanup).__name__
        finally:
            self.model_loaded = False
            self.state = "CLOSED"
        return "cleanup_ok", None


def validate_dual_ray_shape_sign_support(
    *, ray: np.ndarray, has_ray: bool, rows: int
) -> Dict[str, Any]:
    """Validate the selector-facing part of a HiGHS upper-row dual ray.

    The selector-only contract requires the API success bit, exact row shape,
    finite coefficients, upper-row sign, and at least one stored-binary64
    coefficient that is exactly nonzero.  It deliberately does not consume
    implicit bound components or claim an algebraic Farkas proof: the ray never
    has proof, tightening, or verdict authority.
    """

    if (
        not has_ray
        or ray.dtype != np.dtype(np.float64)
        or not ray.flags.c_contiguous
        or ray.flags.writeable
        or not ray.flags.owndata
        or ray.shape != (rows,)
        or not np.all(np.isfinite(ray))
        or np.any(ray > 0.0)
    ):
        raise ProbeStop("HiGHS dual-ray API/shape/finiteness/sign contract failed")
    exact_nonzero = np.fromiter(
        (Fraction.from_float(float(value)) != 0 for value in ray),
        dtype=bool,
        count=rows,
    )
    if not np.any(exact_nonzero):
        raise ProbeStop("HiGHS dual ray has no exact-nonzero mapped support")
    return {
        "validation_scope": "api_state_shape_finite_upper_sign_exact_nonzero_only",
        "has_ray": True,
        "rows": rows,
        "finite": True,
        "upper_row_sign_nonpositive": True,
        "exact_nonzero_count": int(np.count_nonzero(exact_nonzero)),
        "owned_contiguous_float64_read_only": True,
        "algebraic_farkas_replay_used": False,
        "implicit_bound_components_consumed": False,
        "proof_or_tightening_authority": False,
    }


def seal_base_screened_row_mapping_from_caller(
    *, caller: Any, matrix: sp.csr_matrix
) -> Tuple[Tuple[int, int, int], ...]:
    """Seal the actual base screened-row map before the one owner loads it.

    The shim executes synchronously inside the frozen builder's LP call.  The
    mapping is therefore derived from the same live ``keep`` and phase frames
    that produced ``A_ub``, before solve/readback or any later reconstruction.
    """

    local = caller.f_locals
    required = (
        "order",
        "original_frames",
        "target_assign",
        "target_pre",
        "keep",
        "screened_A",
    )
    if any(name not in local for name in required):
        raise ProbeStop("base LP caller lacks synchronous phase-row mapping state")
    caller_matrix = local["screened_A"]
    if (
        not sp.isspmatrix_csr(caller_matrix)
        or caller_matrix.dtype != np.dtype(np.float64)
        or not sp.isspmatrix_csr(matrix)
        or matrix.dtype != np.dtype(np.float64)
    ):
        raise ProbeStop("caller screened matrix is not an existing float64 CSR")
    if not (
        caller_matrix.shape == matrix.shape
        and caller_matrix.nnz == matrix.nnz
        and np.array_equal(caller_matrix.indptr, matrix.indptr)
        and np.array_equal(caller_matrix.indices, matrix.indices)
        and np.array_equal(caller_matrix.data, matrix.data)
    ):
        raise ProbeStop("LP argument is not the caller's actual screened matrix")

    full: list[Tuple[int, int, int]] = []
    physical: set[Tuple[int, int]] = set()
    for layer in local["order"]:
        layer_id = int(layer.id)
        original = local["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        exact = np.asarray(original.exact, dtype=np.int64)
        stream_rows = np.asarray(original.stream_rows, dtype=np.int64)
        selected = np.asarray(local["target_assign"].get(layer_id))
        pre = np.asarray(local["target_pre"].get(layer_id))
        if not (
            exact.ndim == 1
            and stream_rows.ndim == 1
            and selected.dtype == np.dtype(bool)
            and selected.ndim == 1
            and pre.ndim == 2
            and exact.size == stream_rows.size == selected.size == pre.shape[0]
        ):
            raise ProbeStop(
                "base phase layer widths disagree before row-map seal",
                layer_id=layer_id,
            )
        for position, relu_row in enumerate(stream_rows):
            physical_key = (layer_id, int(relu_row))
            if physical_key in physical:
                raise ProbeStop(
                    "duplicate physical phase row before row-map seal",
                    layer_id=layer_id,
                    relu_row=int(relu_row),
                )
            physical.add(physical_key)
            full.append((layer_id, int(position), int(relu_row)))
    keep = np.asarray(local["keep"])
    if keep.dtype != np.dtype(bool) or keep.shape != (len(full),):
        raise ProbeStop("base LP caller keep mask is malformed")
    mapping = tuple(full[int(index)] for index in np.flatnonzero(keep))
    if (
        len(mapping) != matrix.shape[0]
        or len(set(mapping)) != len(mapping)
        or len({(layer, row) for layer, _position, row in mapping}) != len(mapping)
    ):
        raise ProbeStop("sealed screened-row mapping failed shape/uniqueness")
    return mapping


def import_frozen_helpers(case: str):
    category, onnx, vnnlib = CONTROLS[case]
    os.environ["ACT_PHASE_PROJECTION_CASE"] = case
    os.environ["ACT_PHASE_PROJECTION_CATEGORY"] = category
    os.environ["ACT_PHASE_PROJECTION_ONNX"] = onnx
    os.environ["ACT_PHASE_PROJECTION_VNNLIB"] = vnnlib
    spec = importlib.util.spec_from_file_location("fprime_frozen_helpers", ONE_MULTI_PATH)
    if spec is None or spec.loader is None:
        raise ProbeStop("could not import frozen scratch helpers")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    def local_fail(reason: str, **details: Any) -> None:
        raise ProbeStop(reason, **details)

    helper.fail = local_fail
    return helper


def capture_base_cell(helper: Any, owner: SafeHighsOwner, net: Any, entry: int, before: Any, after: Any):
    phase = helper.phase
    captured: Dict[str, Any] = {}
    target_code = phase.build_forward_exact_relu_phase_projection_candidate.__code__

    def tracer(frame: Any, event: str, arg: Any):
        if frame.f_code is target_code and event == "exception":
            captured.update(frame.f_locals)
        return tracer

    original_linprog = phase.linprog
    primary: Optional[BaseException] = None
    result = None
    error = None
    started = time.monotonic()
    sys.settrace(tracer)
    phase.linprog = owner.scipy_compatible_linprog
    try:
        result = phase.build_forward_exact_relu_phase_projection_candidate(
            net,
            int(entry),
            before,
            after,
            deadline=owner.deadline,
            lp_time_limit=LP_TIME_LIMIT,
        )
    except BaseException as exc:
        primary = exc
        error = exc
        if not isinstance(exc, Exception):
            # SystemExit, KeyboardInterrupt, and GeneratorExit retain identity
            # and traceback; the outer finally still performs bounded cleanup.
            raise
    finally:
        phase.linprog = original_linprog
        sys.settrace(None)
    return captured, result, error, time.monotonic() - started


def select_update_rows(c: Mapping[str, Any], base: SolveRecord) -> Tuple[str, list[Tuple[int, int, int]], Dict[str, Any]]:
    if base.row_mapping is None or len(base.row_mapping) != base.logical_rows:
        raise ProbeStop("base SolveRecord lacks its sealed screened-row mapping")

    if base.optimal:
        margin = float(c["candidate_margin"])
        if not math.isfinite(margin) or not margin < 0.0:
            raise ProbeStop("optimal selector invoked without a strictly negative margin")
        assert base.row_value is not None and base.row_dual is not None
        tolerance = float(c["_SOLVER_TOLERANCE"] if "_SOLVER_TOLERANCE" in c else 1.0e-9)
        slack = base.loaded_rhs - base.row_value
        tight = np.abs(slack) <= tolerance * (1.0 + np.abs(base.loaded_rhs))
        eligible = tight & (base.row_dual < 0.0)
        rule = "optimal_negative_all_tight_strict_negative_upper_row_dual"
        diagnostics = {
            "margin": margin,
            "tight_count": int(np.count_nonzero(tight)),
            "strict_negative_row_dual_count": int(np.count_nonzero(base.row_dual < 0.0)),
            "selected_count": int(np.count_nonzero(eligible)),
            "tightness_uses_loaded_safe_upper": True,
        }
    elif base.infeasible:
        if base.dual_ray is None or base.dual_ray_validation is None:
            raise ProbeStop("infeasible base model lacks a validated dual ray")
        eligible = np.fromiter(
            (
                Fraction.from_float(float(value)) != 0
                for value in base.dual_ray
            ),
            dtype=bool,
            count=base.logical_rows,
        )
        rule = "infeasible_all_exact_nonzero_validated_dual_ray_phase_rows"
        diagnostics = dict(base.dual_ray_validation)
        diagnostics["selected_count"] = int(np.count_nonzero(eligible))
    else:
        raise ProbeStop("base solver status is not eligible for the single update")

    selected = [base.row_mapping[int(index)] for index in np.flatnonzero(eligible)]
    if not selected:
        raise ProbeStop("the frozen F-prime selector produced an empty update set")
    return rule, selected, diagnostics


def assemble_updated_cell(helper: Any, c: Mapping[str, Any], rebuilt: Mapping[str, Any]):
    phase = helper.phase
    blocks = []
    rhs = []
    total_phases = 0
    for layer in c["order"]:
        layer_id = int(layer.id)
        original = c["original_frames"].get(layer_id)
        if original is None or not original.exact.size:
            continue
        matrix = sp.csr_matrix(rebuilt["pre"][layer_id])
        selected = np.asarray(rebuilt["assign"][layer_id], dtype=bool)
        blocks.append(matrix.multiply(np.where(selected, -1.0, 1.0)[:, None]).tocsr())
        center = np.asarray(rebuilt["pre_center"][layer_id], dtype=np.float64)
        rhs.append(np.where(selected, center, -center))
        total_phases += int(original.exact.size)
    matrix = sp.vstack(blocks, format="csr")
    full_rhs = np.ascontiguousarray(np.concatenate(rhs), dtype=np.float64)
    row_max = phase._csr_box_upper(matrix, c["factor_lower"], c["factor_upper"])
    keep = row_max > full_rhs
    screened = matrix[keep].tocsr()
    screened_rhs = full_rhs[keep]
    coeff = np.asarray(c["C"][[c["rival"]]] @ rebuilt["output"], dtype=np.float64).reshape(-1)
    center = float(
        c["C"][c["rival"]] @ rebuilt["output_center"]
        - c["thresholds"][c["rival"]]
    )
    if (
        matrix.shape != (total_phases, int(c["input_rows"].size))
        or not np.all(np.isfinite(matrix.data))
        or not np.all(np.isfinite(full_rhs))
        or not np.all(np.isfinite(coeff))
        or not math.isfinite(center)
    ):
        raise ProbeStop("updated cell assembly shape/finiteness contract failed")
    return screened, screened_rhs, coeff, center, keep


def unchanged_terminal_with_deadline(
    helper: Any,
    c: Mapping[str, Any],
    factors: np.ndarray,
    deadline: Optional[float],
) -> Dict[str, Any]:
    """The frozen terminal, with its existing optional deadline connected."""

    decoded = np.asarray(c["raw_lower"], dtype=np.float64).copy()
    for column, raw_row in enumerate(c["input_rows"]):
        row = int(raw_row)
        exact_value = Fraction.from_float(float(c["input_center"][row]))
        exact_value += Fraction.from_float(
            float(c["input_radius"][row])
        ) * Fraction.from_float(float(factors[column]))
        decoded[row] = float(exact_value)
    in_box = bool(
        np.all(np.isfinite(decoded))
        and np.all(decoded >= c["raw_lower"])
        and np.all(decoded <= c["raw_upper"])
    )
    if not in_box:
        return {"raw_box": False, "verified": False, "seconds": 0.0}
    started = time.monotonic()
    lower, upper = helper.phase._singleton_interval_forward(
        c["net"],
        c["order"],
        c["affines"],
        decoded.reshape(c["input_shape"]),
        c["output_layer_id"],
        pointwise=c["pointwise"],
        deadline=deadline,
    )
    exact = helper.phase._exact_singleton_margin_lower(
        c["C"][c["rival"]], c["thresholds"][c["rival"]], lower, upper
    )
    return {
        "raw_box": True,
        "zero_width_interval": True,
        "fraction_margin_lower": float(exact),
        "verified": bool(exact > 0),
        "seconds": time.monotonic() - started,
    }


def perform_single_update(
    helper: Any,
    c: Dict[str, Any],
    owner: SafeHighsOwner,
    net: Any,
    base: SolveRecord,
    base_payload: Dict[str, Any],
) -> Dict[str, Any]:
    owner.check_deadline("selection")
    rule, selected, selection_diagnostics = select_update_rows(c, base)
    assignments = {
        key: np.asarray(value, dtype=bool).copy()
        for key, value in c["target_assign"].items()
    }
    for layer_id, position, _row in selected:
        assignments[layer_id][position] = ~assignments[layer_id][position]
    owner.check_deadline("after_complete_set_flip")
    c["net"] = net
    rebuilt = helper.rebuild_one_cell(c, assignments)
    owner.check_deadline("after_one_cell_rebuild")
    matrix, rhs, coeff, center, _keep = assemble_updated_cell(helper, c, rebuilt)
    owner.check_deadline("after_one_update_assembly")
    updated = owner.load_and_solve(
        tag="one_update",
        cost=-coeff,
        matrix=matrix,
        rhs=rhs,
        lower=np.asarray(c["factor_lower"], dtype=np.float64),
        upper=np.asarray(c["factor_upper"], dtype=np.float64),
        time_limit=LP_TIME_LIMIT,
        row_mapping=None,
        selector_readback=False,
    )
    updated_payload = record_summary(updated)
    terminal_payload = None
    if updated.optimal and updated.factors is not None:
        updated_margin = float(center + coeff @ updated.factors)
        if not math.isfinite(updated_margin):
            raise ProbeStop("updated margin is nonfinite")
        updated_payload["margin"] = updated_margin
        if updated_margin > 0.0:
            owner.check_deadline("before_unchanged_terminal")
            terminal_payload = unchanged_terminal_with_deadline(
                helper, c, updated.factors, owner.deadline
            )
            owner.check_deadline("after_unchanged_terminal")
    return {
        "status": (
            "UPDATED_TERMINAL_VERIFIED"
            if terminal_payload and terminal_payload.get("verified")
            else "UPDATED_POSITIVE_TERMINAL_REJECTED"
            if terminal_payload
            else "UPDATED_NONPOSITIVE_OR_UNSOLVED"
        ),
        "base": base_payload,
        "selection_rule": rule,
        "selection_diagnostics": selection_diagnostics,
        "selected_flip_count": len(selected),
        "selected_rows_sha256": selected_digest(selected),
        "selected_rows_first_ten": [
            {"layer_id": a, "position": b, "relu_row": row}
            for a, b, row in selected[:10]
        ],
        "updated": updated_payload,
        "terminal": terminal_payload,
    }


def record_summary(record: SolveRecord) -> Dict[str, Any]:
    result = {
        "tag": record.tag,
        "model_status": str(record.model_status),
        "logical_rows": record.logical_rows,
        "logical_nnz": record.logical_nnz,
        "loaded_nnz": record.loaded_nnz,
        "simplex_iterations": record.simplex_iterations,
        "loader": record.loader,
        "objective": {
            "nonzero_count": record.objective_nonzero_count,
            "tiny_nonzero_count": record.objective_tiny_nonzero_count,
            "minimum_nonzero_abs": record.objective_min_nonzero_abs,
            "tiny_nonzero_fail_closed": True,
        },
    }
    if record.row_mapping_sha256 is not None:
        result["sealed_row_mapping_sha256"] = record.row_mapping_sha256
    if record.dual_ray_validation is not None:
        result["dual_ray_validation"] = record.dual_ray_validation
    return result


def selected_digest(selected: Sequence[Tuple[int, int, int]]) -> str:
    payload = [[int(a), int(b), int(c)] for a, b, c in selected]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def run_case(case: str) -> Dict[str, Any]:
    helper = import_frozen_helpers(case)
    phase = helper.phase
    helper.initialize_device(device="cuda", dtype="float64")
    helper.set_solver_mode("hybridz")
    helper.set_transfer_function_mode("interval")
    category, onnx, vnnlib = CONTROLS[case]

    total_started = time.monotonic()
    sr = helper.create_specs_from_paths(onnx, vnnlib, category=category)
    vm = next(iter(helper.synthesize_models_from_specs([sr]).values()))
    net = helper.TorchToACT(vm).run()
    entry = helper.find_entry_layer_id(net)
    specs = helper.gather_input_spec_layers(net)
    seed = helper.seed_from_input_specs(specs)
    fact = helper.Fact(bounds=seed, cons=helper.ConSet())
    helper.add_all_input_specs(fact.cons, helper.get_input_ids(net), specs)
    before, after, _ = helper.analyze(net, entry, fact)

    owner = SafeHighsOwner(deadline=time.monotonic() + LP_TIME_LIMIT)
    primary: Optional[BaseException] = None
    receipt: Dict[str, Any]
    try:
        captured, base_result, base_error, base_seconds = capture_base_cell(
            helper, owner, net, entry, before, after
        )
        if not owner.records:
            raise ProbeStop(
                "frozen cell construction stopped before the single-owner base LP",
                builder_error_type=(
                    None if base_error is None else type(base_error).__name__
                ),
            )
        base = owner.records[0]
        base_payload = record_summary(base)
        base_payload["seconds_instrumented"] = base_seconds

        if base_result is not None:
            receipt = {
                "status": "BASE_TERMINAL_VERIFIED",
                "base": base_payload,
                "selected_flip_count": 0,
                "updated": None,
                "terminal": {
                    "verified": True,
                    "fraction_margin_lower": float(
                        base_result.receipt.singleton_margin_lower
                    ),
                },
            }
        elif base.optimal and base.objective_value is not None:
            if "candidate_margin" not in captured:
                raise ProbeStop(
                    "optimal base did not expose a candidate margin",
                    builder_error_type=(
                        None if base_error is None else type(base_error).__name__
                    ),
                )
            base_margin = float(captured["candidate_margin"])
            base_payload["margin"] = base_margin
            if base_margin > 0.0:
                # A positive base candidate already reached the unchanged
                # terminal and failed there.  F-prime has no fallback/update.
                raise ProbeStop(
                    "positive base candidate was rejected by the unchanged terminal",
                    base_margin=base_margin,
                    builder_error_type=(
                        None if base_error is None else type(base_error).__name__
                    ),
                )
            if base_margin == 0.0:
                raise ProbeStop("zero base margin is not update-eligible")
            receipt = perform_single_update(
                helper, captured, owner, net, base, base_payload
            )
        elif base.infeasible:
            receipt = perform_single_update(
                helper, captured, owner, net, base, base_payload
            )
        else:
            raise ProbeStop(
                "base HiGHS status is fail-closed",
                model_status=str(base.model_status),
            )
    except BaseException as exc:
        primary = exc
        if not isinstance(exc, Exception):
            # Preserve SystemExit, KeyboardInterrupt, and GeneratorExit.  The
            # finally below still cleans the one owner.
            raise
        if isinstance(exc, ProbeStop):
            reason = exc.reason
            details = exc.details
        else:
            reason = (
                "unexpected disconnected-probe failure type="
                f"{type(exc).__name__}"
            )
            details = {}
        receipt = {
            "status": "STOP_LOSS_UNKNOWN",
            "reason": reason,
            "details": details,
        }
    finally:
        cleanup_note, cleanup_failure_type = owner.close()
        if cleanup_note != "cleanup_ok" and primary is not None:
            add_secondary_type_note(
                primary,
                "secondary single-owner final cleanup failure "
                f"type={cleanup_failure_type}"
            )
    if cleanup_note != "cleanup_ok" and primary is None:
        receipt = {
            "status": "STOP_LOSS_UNKNOWN",
            "reason": "single-owner cleanup failed",
        }

    receipt.update(
        {
            "schema": "act.scratch.phase_projection_fprime_single_owner.v1",
            "case": case,
            "external_labels_read_by_runtime_rule": False,
            "owner": {
                "highspy_version": PINNED_HIGHS_VERSION,
                "instances_constructed": 1,
                "model_loads": owner.model_loads,
                "solves": owner.solve_count,
                "dual_ray_exist_calls": owner.dual_ray_exist_calls,
                "dual_ray_calls": owner.dual_ray_calls,
                "poisoned": owner.poisoned,
                "state": owner.state,
                "cleanup": cleanup_note,
                "pinned_highs_githash": PINNED_HIGHS_GITHASH,
            },
            "timing": {
                "end_to_end_instrumented_seconds": time.monotonic() - total_started,
                "timing_authority": False,
            },
            "restrictions": restrictions(),
            "production_modified": False,
            "source": {
                "scratch_path": str(Path(__file__).resolve()),
                "frozen_phase_sha256": sha256(Path(phase.__file__).resolve()),
                "frozen_live_sha256": sha256(Path(helper.live.__file__).resolve()),
                "cpu_owner_contract_path": str(OWNER_CONTRACT),
                "cpu_owner_contract_sha256": sha256(OWNER_CONTRACT),
            },
        }
    )
    return receipt


def static_check() -> Dict[str, Any]:
    expected = {
        "cifar100_medium_iid2",
        "tinyimagenet_medium_iid143",
        "cifar100_large_iid153",
        "cifar100_large_iid166",
        "tinyimagenet_medium_iid153",
    }
    if set(CONTROLS) != expected:
        raise ProbeStop("control manifest is not the frozen five-case set")
    if sha256(OWNER_CONTRACT) != OWNER_CONTRACT_SHA256:
        raise ProbeStop("frozen CPU owner contract hash changed")
    for case, (_category, onnx, vnnlib) in CONTROLS.items():
        if not Path(onnx).is_file() or not Path(vnnlib).is_file():
            raise ProbeStop("control input is unavailable", case=case)
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    highs_constructors = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "highspy"
            and node.func.attr == "Highs"
        ):
            highs_constructors += 1
    forbidden = ("scipy.optimize", "onnxruntime")
    hits = [name for name in imported if name.startswith(forbidden)]
    if hits:
        raise ProbeStop("static forbidden dependency token present", hits=hits)
    if highs_constructors != 1:
        raise ProbeStop(
            "static owner-constructor count is not exactly one",
            observed=highs_constructors,
        )
    run_case_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_case"
    )
    closed_backend_refs = [
        node
        for node in ast.walk(run_case_node)
        if isinstance(node, ast.Attribute)
        and node.attr == "highs"
        and isinstance(node.value, ast.Name)
        and node.value.id == "owner"
    ]
    if closed_backend_refs:
        raise ProbeStop("run_case may access owner.highs after final close")
    return {
        "schema": "act.scratch.phase_projection_fprime_single_owner.static.v1",
        "status": "STATIC_READY_FOR_INDEPENDENT_PRE_GPU_AUDIT",
        "five_cases": list(CONTROLS),
        "one_highspy_constructor_site": True,
        "production_modified": False,
        "gpu_started": False,
        "cpu_owner_contract_sha256": OWNER_CONTRACT_SHA256,
        "restrictions": restrictions(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CONTROLS))
    parser.add_argument("--static-check", action="store_true")
    args = parser.parse_args()
    if args.static_check:
        print(json.dumps(static_check(), sort_keys=True, separators=(",", ":")))
        return
    if args.case is None:
        parser.error("--case is required unless --static-check is used")
    print(
        json.dumps(
            run_case(args.case),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
