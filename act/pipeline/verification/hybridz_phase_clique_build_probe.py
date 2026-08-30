"""Bounded, verdict-free Operator-HZ large-class build probe.

This command loads exactly one official-CSV iid (hard-wired to iid=2), performs
interval analysis, builds Operator-HZ, and runs the default raw-property K4
pipeline, the explicit ``localized_e2`` build-only diagnostic, or the
strictly pre-registered ``rbs_adaptive_k4`` experiment.  The default-disabled
``pcoh_k2_build_only`` route reuses the raw focused-rival and verified literal
front half, then invokes the receipt-only PCOH K2 transaction.  It deliberately
never imports or calls ``hz_objbound_decide``.  Historical and reference labels
are never consulted or included in a decision or receipt.

When a fresh K4 frame is materialized, one focused property objective is
solved in the source and private fresh LP relaxations.  HiGHS' multipliers are
not trusted: each reported upper bound is reconstructed by the native-block,
no-stack independent long-double Lagrangian checker.  These bounds remain
diagnostic and have no proof/verdict authority.  Localized E2 likewise uses
only a structurally revalidated private pair-cut copy and can pass only a
controlled build-only gate.  Adaptive K4 splits one selector prefix into four
primaries and a same-layer reserve, screens the resulting build before entering
K4, and requires an independently certified LP drop afterwards.  Neither experiment
gains production authorization.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from fractions import Fraction
import gc
import hashlib
import itertools
import json
import math
import multiprocessing
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import resource
import secrets
import signal
import stat
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence
import weakref

import numpy as np
import scipy.sparse as sp

_SCHEMA = "act.hybridz_phase_clique_build_probe.v1"
_MAX_WALL_SECONDS = 60.0
_ONLY_IID = 2
_CANDIDATE_MODES = (
    "k4",
    "localized_e2",
    "rbs_adaptive_k4",
    "pcoh_k2_build_only",
    "pcoh_k3_build_only",
)
_PCOH_K2_BUILD_ONLY_MODE = "pcoh_k2_build_only"
_PCOH_K3_BUILD_ONLY_MODE = "pcoh_k3_build_only"
_PCOH_K2_FAMILY = "cifar100_medium"
_PCOH_K2_MAX_PHASE_SECONDS = 25.0
_PCOH_K2_RESIDUAL_SECONDS = 4.0
_PCOH_K2_MAX_SOURCE_BUILD_SECONDS = 27.0
_PCOH_K2_FINALIZATION_RESERVE_SECONDS = 1.0
_PCOH_K2_SELECTION_SECONDS = 2.0
_PCOH_K2_SELECTION_WORK_ITEMS = 1_000_000
_PCOH_K3_FAMILY = "cifar100_medium"
_PCOH_K3_CLI_PHASE_SECONDS = 25.0
_PCOH_K3_INTERNAL_PHASE_SECONDS = 22.0
_PCOH_K3_FINALIZATION_RESERVE_SECONDS = 3.0
_PCOH_K3_RESIDUAL_SECONDS = 4.0
_PCOH_K3_SELECTION_SECONDS = 2.0
_PCOH_K3_SELECTION_WORK_ITEMS = 1_000_000
_PCOH_K3_TRANSACTION_SCHEMA = (
    "act.hybridz_pcoh_k3_build_only_probe_transaction.v3"
)
_PCOH_K3_TIGHTNESS_GATE_SCHEMA = (
    "act.hybridz_pcoh_k3_strong_tightness_gate.v1"
)
_PCOH_K3_INTEGRITY_SCHEMA = (
    "act.hybridz_pcoh_k3_build_only_probe_integrity.v2"
)
_PCOH_K3_SOURCE_PREFLIGHT_SCHEMA = (
    "act.hybridz_pcoh_k3_source_build_preflight.v2"
)
_PCOH_K3_BASELINE_ARTIFACT_SHA256 = (
    "01625add9f435eefef20e3eaa6dcaf72f2ce0f50137f19a611c576c1829846b0"
)
_PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH = (
    "artifacts/hybridz_largecls_gates/"
    "pcoh_k2_materialized_tightness_cifar100_medium_iid2_first.json"
)
_PCOH_K3_BASELINE_ANCHOR_SCHEMA = (
    "act.hybridz_pcoh_k3_fixed_k2_baseline_anchor.v2"
)
_PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SCHEMA = (
    "act.hybridz_pcoh_focused_front_half_semantic_anchor.v1"
)
_PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SHA256 = (
    "bb938b9f23f4e0909a77f8f547d30402121ca3592267e817e3ff6083fb62c862"
)
_PCOH_K3_BASELINE_SUMMARY_SHA256 = (
    "1ab9e71a1c7ed74ece34731e030bd0e1f8d27284d01d19aae851f5239e1471b2"
)
_PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST = (
    "288961a458b8628840bb43efc75d82502acfd03aef5b9d661d847c00a80a4dac"
)
_PCOH_K3_EXPECTED_SELECTION_DIGEST = (
    "ad3abbbe05b99b7c135e4faded67bb9256673e4ee2d33371d3fecff3f55563dd"
)
_PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST = (
    "f50d17c9660016f1404bfd2cc57271a93e0a014f84614c7181764d70b9c066c5"
)
_PCOH_K3_EXPECTED_FULL_BATCH_SHA256 = (
    "c898fb8e5d99b86dc9a05f98fecd24a3dbf64ead0fd440616df7ef4bbcb0dd50"
)
_PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST = (
    "0a20af0be91b8b9c03bf32dc8bcc00a91c9f949ce82be00a871fbadde3db2e9f"
)
_PCOH_K3_FOCUS_METHOD = (
    "caller_bound_residual_joint_focus_encoded_row_v1"
)
_PCOH_K3_FOCUS_COUNT = 1
_PCOH_K3_FOCUSED_ENCODED_ROW = 50
_PCOH_K3_FOCUSED_RIVAL_ID = 51
_PCOH_K3_RETAINED_K2_STABLE_BIT_IDS = (52557, 52558)
_PCOH_K3_GLOBAL_CUBE_UPPER_HEX = "0x1.bb75dc1b70296p+6"
_PCOH_K3_GLOBAL_CUBE_UPPER_EXACT = Fraction(
    17155542887308652655950798511,
    154742504910672534362390528,
)
_PCOH_K2_MAX_RSS_BYTES = 5 * 1024**3 // 2
_PCOH_K2_SOURCE_OUTPUTS = 100
_PCOH_K2_SOURCE_BINARIES = 4
_PCOH_K2_MAX_SOURCE_CONTINUOUS = 60_000
_PCOH_K2_MAX_SOURCE_ROWS_PLUS_OUTPUTS = 105_000
_PCOH_K2_MAX_SOURCE_CONSTRAINT_NONZEROS = 11_000_000
_PCOH_K2_MAX_SOURCE_GENERATOR_NONZEROS = 20_000
_PCOH_K2_TRANSACTION_SCHEMA = (
    "act.hybridz_pcoh_k2_build_only_probe_transaction.v2"
)
_PCOH_K2_TIGHTNESS_GATE_SCHEMA = (
    "act.hybridz_pcoh_k2_materialized_tightness_gate.v1"
)
_PCOH_K2_INTEGRITY_SCHEMA = (
    "act.hybridz_pcoh_k2_build_only_probe_integrity.v2"
)
_PCOH_K2_TRANSACTION_FIELDS = frozenset({
    "schema",
    "status",
    "reason",
    "failed_stage",
    "diagnostic_only",
    "candidate_only",
    "build_only",
    "instance_count",
    "proof_authority",
    "verdict_authority",
    "ground_truth_loaded",
    "reference_label_used",
    "build_only_transaction_called",
    "transaction_verified_before_serialization",
    "solver_handoff_called",
    "diagnostic_lp_called",
    "hz_base_feasibility_called",
    "hz_objbound_decide_called",
    "strict_replay_called",
    "fresh_build_returned",
    "full_parent_lp_called",
    "full_parent_lp_solver_called",
    "input_sha256",
    "implementation_sha256",
    "full_batch_sha256",
    "focused_subset_digest",
    "focused_encoded_row",
    "focused_rival_id",
    "successful_selection_binding_retained",
    "selection_digest",
    "selection_property_digest",
    "selection_parent_semantic_digest",
    "selection_operator_row_tag_digest",
    "stable_bit_selection_method",
    "stable_bit_ids",
    "diagnostic_schema",
    "diagnostic_sha256",
    "transaction_receipt_sha256",
    "source_semantic_digest",
    "fresh_semantic_digest",
    "source_dimensions",
    "fresh_dimensions",
    "conditional_certificate_sha256",
    "pair_bundle_sha256",
    "fresh_issuance_sha256",
    "materialized_tightness_summary_sha256",
    "materialized_tightness_summary",
    "tightness_gate",
    "resource_preflight",
    "resource_postflight",
    "stage_resources",
    "timings",
    "receipt_sha256",
})
_PCOH_K2_TIGHTNESS_GATE_FIELDS = frozenset({
    "schema",
    "status",
    "diagnostic_only",
    "candidate_only",
    "full_parent_lp_called",
    "full_parent_lp_solver_called",
    "proof_authority",
    "verdict_authority",
    "materialized_tightness_summary_sha256",
    "global_cube_upper_fraction",
    "final_structural_upper_fraction",
    "ideal_union_upper_fraction",
    "rounding_tax_fraction",
    "delta_fraction",
    "continuation_scale_threshold_fraction",
    "strong_scale_threshold_fraction",
    "rounding_tax_threshold_fraction",
    "delta_nonnegative",
    "global_positive",
    "continuation_scale_met",
    "strong_scale_met",
    "rounding_tax_dominance_met",
    "continuation_candidate",
    "strong_candidate",
    "cube_already_sufficient",
    "zero_crossing",
    "receipt_sha256",
})
_PCOH_K3_TRANSACTION_FIELDS = frozenset({
    "schema", "status", "reason", "failed_stage", "diagnostic_only",
    "candidate_only", "build_only", "instance_count", "proof_authority",
    "verdict_authority", "provenance_authority",
    "authenticity_authority", "ground_truth_loaded",
    "reference_label_used",
    "k3_transaction_called", "k3_same_process_verified",
    "trusted_outcome_digest_captured_before_detach",
    "k3_detached_verified", "k2_build_only_called",
    "phase_transaction_called", "solver_handoff_called",
    "hz_base_feasibility_called", "hz_objbound_decide_called",
    "full_parent_lp_called", "full_parent_lp_solver_called",
    "fresh_build_returned", "input_sha256", "implementation_sha256",
    "baseline_artifact_sha256", "baseline_summary_sha256",
    "baseline_anchor_receipt_sha256", "baseline_anchor_verified",
    "full_batch_sha256", "focused_subset_digest",
    "residual_selector_receipt_sha256", "focused_semantic_anchor",
    "focused_semantic_anchor_sha256", "focused_encoded_row",
    "focused_rival_id", "source_semantic_digest",
    "selection_digest", "selection_property_digest",
    "selection_parent_semantic_digest", "selection_operator_row_tag_digest",
    "stable_bit_selection_method", "retained_k2_stable_bit_ids",
    "stable_bit_ids", "third_stable_bit_id", "outcome_kind",
    "outcome_schema", "outcome_status", "trusted_outcome_sha256",
    "outcome_receipt_sha256", "detached_outcome",
    "resource_gate_rejection_sha256", "resource_gate_rejection",
    "pair_bundle_sha256", "active_pattern_mask", "evaluation_schedule",
    "threshold_pattern_indices", "source_dimensions", "fresh_dimensions",
    "fresh_semantic_digest", "materialized_tightness_summary_sha256",
    "materialized_tightness_summary", "strong_tightness_gate",
    "pair_local_lp_actual_calls", "conditional_local_lp_actual_calls",
    "total_local_lp_actual_calls", "conditional_checker_actual_calls",
    "local_lp_actual_call_cap", "conditional_checker_actual_call_cap",
    "stage_resources", "timings", "receipt_sha256",
})
_PCOH_K3_TIGHTNESS_GATE_FIELDS = frozenset({
    "schema", "status", "diagnostic_only", "candidate_only",
    "proof_authority", "verdict_authority", "full_parent_lp_called",
    "full_parent_lp_solver_called", "baseline_artifact_sha256",
    "baseline_summary_sha256", "source_semantic_digest",
    "selection_digest", "focused_encoded_row", "focused_rival_id",
    "retained_k2_stable_bit_ids", "stable_bit_ids",
    "materialized_tightness_summary_sha256",
    "global_cube_upper_fraction", "final_structural_upper_fraction",
    "ideal_union_upper_fraction", "rounding_tax_fraction",
    "delta_fraction", "strong_target_fraction",
    "strong_scale_threshold_fraction", "rounding_tax_threshold_fraction",
    "global_anchor_matches", "source_anchor_matches",
    "selection_anchor_matches", "focus_anchor_matches",
    "retained_ids_anchor_matches", "final_at_most_strong_target",
    "strong_scale_met", "rounding_tax_dominance_met",
    "strong_candidate", "receipt_sha256",
})


@dataclass(frozen=True)
class _PCOHK2TrustedTransactionAnchor:
    transaction: dict[str, Any]
    process_id: int
    transaction_receipt_sha256: str
    materialized_tightness_summary_sha256: str


_PCOH_K2_TRUSTED_TRANSACTION_LOCK = threading.Lock()
_PCOH_K2_TRUSTED_TRANSACTIONS: dict[
    int, _PCOHK2TrustedTransactionAnchor
] = {}


@dataclass(frozen=True)
class _PCOHK3TrustedTransactionAnchor:
    transaction: dict[str, Any]
    outcome: Any
    process_id: int
    transaction_receipt_sha256: str
    outcome_sha256: str
    outcome_kind: str


_PCOH_K3_TRUSTED_TRANSACTION_LOCK = threading.Lock()
_PCOH_K3_TRUSTED_TRANSACTIONS: dict[
    int, _PCOHK3TrustedTransactionAnchor
] = {}
_PCOH_K3_CONSUMED_OUTCOMES: dict[int, weakref.ReferenceType[Any]] = {}
_LOCALIZED_E2_MIN_RELATIVE_DROP = 0.05
_LOCALIZED_E2_MAX_RSS_BYTES = 5 * 1024**3 // 2
_LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES = 8 * 1024**3
_RBS_ADAPTIVE_K4_FAMILY = "cifar100_medium"
_RBS_ADAPTIVE_K4_PRIMARY_BUDGET = 4
_RBS_ADAPTIVE_K4_SELECTOR_BUDGET = 16
_RBS_ADAPTIVE_K4_RESIDUAL_SECONDS = 4.0
_RBS_ADAPTIVE_K4_PHASE_SECONDS = 30.0
_RBS_ADAPTIVE_K4_CPU_THREADS = 20
_RBS_ADAPTIVE_K4_MAX_BACKUPS_PER_PRIMARY = 3
_RBS_ADAPTIVE_K4_MAX_BUILD_SECONDS = 20.0
_RBS_ADAPTIVE_K4_MAX_CANDIDATE_SECONDS = 12.0
_RBS_ADAPTIVE_K4_FINALIZATION_RESERVE_SECONDS = 1.0
_RBS_ADAPTIVE_K4_EXPECTED_CONV_LAYERS = 19
_RBS_ADAPTIVE_K4_MIN_ABSOLUTE_DROP = 1.0
_RBS_ADAPTIVE_K4_MIN_RELATIVE_DROP = 0.10
_RBS_ADAPTIVE_K4_MAX_CUBE_UPPER = 75.0
_RBS_ADAPTIVE_K4_MAX_RSS_BYTES = 5 * 1024**3 // 2
_RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES = 64 * 1024**2
_RBS_ADAPTIVE_K4_MAX_CUDA_ALLOCATED_BYTES = 8 * 1024**3
_RBS_ADAPTIVE_K4_COMPACT_STATUS = (
    "focused_rival_clique_compact_candidate"
)
_RBS_ADAPTIVE_K4_COMPACT_TELEMETRY_SCHEMA = (
    "act.operator_exact_relu_phase_clique_compact_candidate.v1"
)
_RBS_ADAPTIVE_K4_PROGRESS_SCHEMA = (
    "act.operator_exact_relu_phase_clique_progress.v1"
)
_RBS_ADAPTIVE_K4_COMPACT_REPRESENTATION = (
    "exact_certificates_and_clique_descriptor_only"
)
_RBS_ADAPTIVE_K4_ORACLE_BACKEND = (
    "highspy_persistent_simplex_presolve_lazy_dual_ray_v2"
)
_RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE = (
    "split_continuous_rows_binary_change_coeff_v1"
)
_RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP = 65_536
_RBS_ADAPTIVE_K4_CANDIDATE_DUST_ABS = 1.0e-12
_RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_SCHEMA = (
    "hz_lp_lagrangian_split_blocks_longdouble_v1"
)
_RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_ROUTE = (
    "native_hz_split_csr_blocks_no_stack_v1"
)
_RBS_ADAPTIVE_K4_OBJECTIVE_DUAL_PROPOSAL_SCHEMA = (
    "act.hybridz.native_split_row_objective_dual_proposal.v1"
)
_RBS_ADAPTIVE_K4_OBJECTIVE_DUAL_BACKEND = (
    "highspy_one_shot_simplex_presolve_split_rows_v1"
)
_RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256 = (
    "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4"
)
_RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256 = (
    "33e795c8421b7b19125f32415adb9cee09b2f90cb83152c4cd3aa03810e91ec3"
)
_RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256 = (
    "aa656d7a73529ba7c41b5618440f543ba4677418bb44115d384b644cc034f9ee"
)
_RBS_ADAPTIVE_K4_HISTORICAL_SHAPE = {
    "output_dimension": 100,
    "continuous_columns": 52_359,
    "binary_columns": 4,
    "upper_rows": 98_378,
    "equality_rows": 0,
    "constraint_nonzeros": 9_267_556,
    "generator_nonzeros": 10_100,
}
_RBS_ADAPTIVE_K4_MAX_UPPER_ROWS = 98_500
_RBS_ADAPTIVE_K4_MAX_CONSTRAINT_NONZEROS = 9_448_000
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_ROOT = (_REPO_ROOT / "artifacts").resolve()
_DEFAULT_BENCHMARK_ROOT = Path(
    "/data1/Kane/data/vnncomp2025_benchmarks/benchmarks"
)
_FAMILIES = {
    "cifar100_medium": {
        "benchmark": "cifar100_2024",
        "model_basename": "CIFAR100_resnet_medium.onnx",
        "iid_min": 0,
        "iid_max": 99,
    },
    "cifar100_large": {
        "benchmark": "cifar100_2024",
        "model_basename": "CIFAR100_resnet_large.onnx",
        "iid_min": 100,
        "iid_max": 199,
    },
    "tinyimagenet_medium": {
        "benchmark": "tinyimagenet_2024",
        "model_basename": "TinyImageNet_resnet_medium.onnx",
        "iid_min": 0,
        "iid_max": 199,
    },
}


class PhaseCliqueBuildProbeError(RuntimeError):
    """A fail-closed probe invariant failed."""


class _RBSAdaptiveK4StopLoss(RuntimeError):
    """Internal control flow for an expected, receipt-backed early stop."""


class _PCOHK2BuildOnlyStopLoss(RuntimeError):
    """Internal control flow after a receipt-backed PCOH stop loss."""


class _PCOHK3BuildOnlyStopLoss(RuntimeError):
    """Internal control flow after a receipt-backed K3 terminal stop."""


@dataclass(frozen=True)
class _ProbeInstance:
    family: str
    iid: int
    onnx_path: Path
    vnnlib_path: Path
    csv_path: Path


@dataclass(frozen=True)
class _OutputSlot:
    display_path: Path
    basename: str
    parent_fd: int
    parent_identity: tuple[int, int]
    parent_relative: tuple[str, ...]
    artifact_root_fd: int
    artifact_root_identity: tuple[int, int]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IMPLEMENTATION_RELATIVE_PATHS = (
    "act/pipeline/verification/hybridz_phase_clique_build_probe.py",
    "act/back_end/hybridz_tf/operator_hz.py",
    "act/back_end/hybridz_tf/property_residual_targets.py",
    "act/back_end/hybridz_tf/operator_phase_clique_pipeline.py",
    "act/back_end/hybridz_tf/operator_exact_relu_phase_literals.py",
    "act/back_end/hybridz_tf/operator_exact_relu_phase_cliques.py",
    "act/back_end/hybridz_tf/operator_exact_relu_phase_clique_materializer.py",
    "act/back_end/hybridz_tf/adaptive_phase_forest.py",
    "act/back_end/hybridz_tf/property_phase_conflict_clique.py",
    "act/back_end/hybridz_tf/persistent_phase_conflict_oracle.py",
    "act/back_end/hybridz_tf/raw_vnnlib_focused_rival_bridge.py",
    "act/back_end/hybridz_tf/raw_vnnlib_rival_adapter.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_build_only.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_objective_bounds.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_pair_infeasibility.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_live_adapter.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_objective_hull.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_objective_hull_row_materializer.py",
    "act/back_end/hybridz_tf/operator_phase_conditioned_objective_hull_fresh_materializer.py",
    "act/back_end/solver/solver_hz.py",
)
_PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS = (
    *_IMPLEMENTATION_RELATIVE_PATHS,
    "act/back_end/hybridz_tf/operator_phase_conditioned_k3_build_only.py",
)


def _implementation_sha256() -> dict[str, str]:
    """Bind a one-shot artifact to the exact uncommitted implementation."""

    result: dict[str, str] = {}
    for relative in _IMPLEMENTATION_RELATIVE_PATHS:
        path = (_REPO_ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(_REPO_ROOT)
        except ValueError as exc:
            raise PhaseCliqueBuildProbeError(
                "implementation source escaped the repository"
            ) from exc
        if not path.is_file():
            raise PhaseCliqueBuildProbeError(
                f"implementation source is not a file: {relative}"
            )
        result[relative] = _sha256_file(path)
    return result


def _pcoh_k3_implementation_sha256() -> dict[str, str]:
    """Bind K3 without changing the frozen K2 implementation keyset."""

    result: dict[str, str] = {}
    for relative in _PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS:
        path = (_REPO_ROOT / relative).resolve(strict=True)
        try:
            path.relative_to(_REPO_ROOT)
        except ValueError as exc:
            raise PhaseCliqueBuildProbeError(
                "K3 implementation source escaped the repository"
            ) from exc
        if not path.is_file():
            raise PhaseCliqueBuildProbeError(
                f"K3 implementation source is not a file: {relative}"
            )
        result[relative] = _sha256_file(path)
    return result


def _checksummed(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["receipt_sha256"] = hashlib.sha256(
        _canonical_json(result)
    ).hexdigest()
    return result


def _strict_csv_artifact(base: Path, raw: str, suffix: str) -> Path:
    relative = Path(raw)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix.lower() != suffix
    ):
        raise PhaseCliqueBuildProbeError(
            f"invalid strict-relative CSV artifact: {raw!r}"
        )
    resolved_base = base.resolve(strict=True)
    resolved = (resolved_base / relative).resolve(strict=True)
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise PhaseCliqueBuildProbeError(
            f"CSV artifact escapes benchmark directory: {raw!r}"
        ) from exc
    if not resolved.is_file():
        raise PhaseCliqueBuildProbeError(f"CSV artifact is not a file: {resolved}")
    return resolved


def _select_instance(
    benchmark_root: Path, family: str, iid: int
) -> _ProbeInstance:
    """Resolve one official CSV row without opening a gate/GT manifest."""

    if type(iid) is not int or iid != _ONLY_IID:
        raise PhaseCliqueBuildProbeError("probe is hard-wired to iid=2")
    if family not in _FAMILIES:
        raise PhaseCliqueBuildProbeError(f"unknown family: {family!r}")
    family_spec = _FAMILIES[family]
    if not int(family_spec["iid_min"]) <= iid <= int(family_spec["iid_max"]):
        raise PhaseCliqueBuildProbeError(
            f"iid={iid} lies outside family {family}"
        )
    root = benchmark_root.expanduser().resolve(strict=True)
    category = (root / str(family_spec["benchmark"])).resolve(strict=True)
    try:
        category.relative_to(root)
    except ValueError as exc:
        raise PhaseCliqueBuildProbeError("benchmark category escapes root") from exc
    csv_path = category / "instances.csv"
    if not csv_path.is_file():
        raise PhaseCliqueBuildProbeError(f"official instances.csv missing: {csv_path}")
    selected: Optional[list[str]] = None
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row_iid, row in enumerate(csv.reader(stream, strict=True)):
            if row_iid != iid:
                continue
            selected = row
            break
    if selected is None or len(selected) != 3 or any(not cell.strip() for cell in selected):
        raise PhaseCliqueBuildProbeError(
            f"instances.csv iid={iid} is absent or malformed"
        )
    onnx_rel, vnnlib_rel, raw_timeout = (cell.strip() for cell in selected)
    try:
        csv_timeout = float(raw_timeout)
    except ValueError as exc:
        raise PhaseCliqueBuildProbeError("CSV timeout is malformed") from exc
    if not math.isfinite(csv_timeout) or csv_timeout <= 0.0:
        raise PhaseCliqueBuildProbeError("CSV timeout must be finite and positive")
    if Path(onnx_rel).name != family_spec["model_basename"]:
        raise PhaseCliqueBuildProbeError(
            f"iid={iid} model does not belong to family {family}"
        )
    expected_vnnlib_prefix = Path(onnx_rel).stem + "_prop_"
    if not Path(vnnlib_rel).name.startswith(expected_vnnlib_prefix):
        raise PhaseCliqueBuildProbeError("VNNLIB/model family prefix mismatch")
    return _ProbeInstance(
        family=family,
        iid=iid,
        onnx_path=_strict_csv_artifact(category, onnx_rel, ".onnx"),
        vnnlib_path=_strict_csv_artifact(category, vnnlib_rel, ".vnnlib"),
        csv_path=csv_path.resolve(strict=True),
    )


def _validate_new_output_path(
    raw_path: Path, *, protected_paths: Sequence[Path]
) -> _OutputSlot:
    """Resolve a new artifacts/*.json target without following a final link."""

    if raw_path.suffix != ".json":
        raise PhaseCliqueBuildProbeError("output must have a .json suffix")
    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    target = raw_path.expanduser()
    if not target.is_absolute():
        target = (_REPO_ROOT / target)
    if os.path.lexists(target):
        raise PhaseCliqueBuildProbeError("output already exists; overwrite forbidden")
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(_ARTIFACT_ROOT)
    except ValueError as exc:
        raise PhaseCliqueBuildProbeError(
            f"output must lie below {_ARTIFACT_ROOT}"
        ) from exc
    if os.path.lexists(resolved):
        raise PhaseCliqueBuildProbeError("output already exists; overwrite forbidden")
    parent = resolved.parent.resolve(strict=True)
    try:
        parent.relative_to(_ARTIFACT_ROOT)
    except ValueError as exc:
        raise PhaseCliqueBuildProbeError("output parent escapes artifacts") from exc
    protected = {path.expanduser().resolve(strict=True) for path in protected_paths}
    if resolved in protected:
        raise PhaseCliqueBuildProbeError("output aliases a protected input")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    artifact_root_fd = os.open(_ARTIFACT_ROOT, directory_flags)
    try:
        parent_fd = os.open(parent, directory_flags)
    except Exception:
        os.close(artifact_root_fd)
        raise
    root_info = os.fstat(artifact_root_fd)
    parent_info = os.fstat(parent_fd)
    return _OutputSlot(
        display_path=resolved,
        basename=resolved.name,
        parent_fd=parent_fd,
        parent_identity=(int(parent_info.st_dev), int(parent_info.st_ino)),
        parent_relative=tuple(parent.relative_to(_ARTIFACT_ROOT).parts),
        artifact_root_fd=artifact_root_fd,
        artifact_root_identity=(int(root_info.st_dev), int(root_info.st_ino)),
    )


def _close_output_slot(slot: _OutputSlot) -> None:
    for fd in (slot.parent_fd, slot.artifact_root_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def _validate_output_slot_live(
    slot: _OutputSlot,
    *,
    expected_published_identity: Optional[tuple[int, int]] = None,
) -> None:
    if (
        not slot.basename
        or slot.basename in {".", ".."}
        or Path(slot.basename).name != slot.basename
        or Path(slot.basename).suffix != ".json"
    ):
        raise PhaseCliqueBuildProbeError("output basename is not one JSON component")
    parent_info = os.fstat(slot.parent_fd)
    root_info = os.fstat(slot.artifact_root_fd)
    if (
        (int(parent_info.st_dev), int(parent_info.st_ino))
        != slot.parent_identity
        or (int(root_info.st_dev), int(root_info.st_ino))
        != slot.artifact_root_identity
    ):
        raise PhaseCliqueBuildProbeError("output directory FD identity changed")
    root_link = os.readlink(f"/proc/self/fd/{slot.artifact_root_fd}")
    parent_link = os.readlink(f"/proc/self/fd/{slot.parent_fd}")
    if root_link.endswith(" (deleted)") or parent_link.endswith(" (deleted)"):
        raise PhaseCliqueBuildProbeError("output directory was deleted or renamed")
    live_root = Path(root_link).resolve(strict=True)
    live_parent = Path(parent_link).resolve(strict=True)
    if live_root != _ARTIFACT_ROOT:
        raise PhaseCliqueBuildProbeError("artifact root FD moved")
    try:
        live_relative = tuple(live_parent.relative_to(live_root).parts)
    except ValueError as exc:
        raise PhaseCliqueBuildProbeError("output directory escaped artifacts") from exc
    if live_relative != slot.parent_relative:
        raise PhaseCliqueBuildProbeError("output directory canonical location changed")
    try:
        published = os.stat(
            slot.basename,
            dir_fd=slot.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if expected_published_identity is None:
            return
        raise PhaseCliqueBuildProbeError("published output basename is missing")
    observed_identity = (int(published.st_dev), int(published.st_ino))
    if expected_published_identity is None:
        raise PhaseCliqueBuildProbeError("output basename already exists")
    if (
        observed_identity != expected_published_identity
        or not stat.S_ISREG(published.st_mode)
    ):
        raise PhaseCliqueBuildProbeError("published output inode is not the worker inode")


def _write_private_worker_json_fd(
    fd: int,
    value: Mapping[str, Any],
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Write only the parent's still-open anonymous 0600 inode."""

    duplicate = os.dup(fd)
    try:
        info = os.fstat(duplicate)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 0
            or (int(info.st_dev), int(info.st_ino)) != expected_identity
        ):
            raise PhaseCliqueBuildProbeError(
                "worker output identity/mode is invalid"
            )
        os.ftruncate(duplicate, 0)
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(duplicate)
        except OSError:
            pass
        raise


def _all_finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _all_finite_json(item)
            for key, item in value.items()
        )
    return False


def _probe_worker_environment(
    ambient: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str], str]:
    """Build the exact Gate worker environment without loading its YAML."""

    from act.pipeline.verification.hybridz_largecls_gate import (
        _fixed_worker_environment,
    )

    fixed = _fixed_worker_environment({
        "row_workers": 4,
        "total_solver_threads": 20,
        "gpu_index": 0,
        "lp_prefilter_fraction": 0.9,
        "lp_prefilter_max_seconds": 60.0,
    })
    child = {
        str(key): str(value)
        for key, value in ambient.items()
        if not str(key).startswith("HZ_")
    }
    child.update(fixed)
    digest = hashlib.sha256(_canonical_json(fixed)).hexdigest()
    return child, fixed, digest


def _validate_worker_receipt_fd(
    fd: int,
    *,
    run_nonce: str,
    expected_identity: tuple[int, int],
) -> dict[str, Any]:
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 0
            or (int(info.st_dev), int(info.st_ino)) != expected_identity
        ):
            raise ValueError("receipt inode binding is invalid")
        raw = os.pread(fd, 16 * 1024 * 1024 + 1, 0)
        if not raw or len(raw) > 16 * 1024 * 1024:
            raise ValueError("receipt size is invalid")
        receipt = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseCliqueBuildProbeError("worker receipt is malformed") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != _SCHEMA
        or receipt.get("run_nonce") != run_nonce
        or not _all_finite_json(receipt)
    ):
        raise PhaseCliqueBuildProbeError("worker receipt binding is invalid")
    claimed = receipt.get("receipt_sha256")
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    expected = hashlib.sha256(_canonical_json(body)).hexdigest()
    if type(claimed) is not str or not secrets.compare_digest(claimed, expected):
        raise PhaseCliqueBuildProbeError("worker receipt checksum mismatch")
    return receipt


def _publish_new_json_fd(
    fd: int, slot: _OutputSlot, *, expected_identity: tuple[int, int]
) -> None:
    """Publish the continuously-held anonymous inode with no-clobber linkat."""

    _validate_output_slot_live(slot)
    info = os.fstat(fd)
    if (
        (int(info.st_dev), int(info.st_ino)) != expected_identity
        or info.st_nlink != 0
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise PhaseCliqueBuildProbeError("publication inode binding changed")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.linkat(
        ctypes.c_int(fd),
        ctypes.c_char_p(b""),
        ctypes.c_int(slot.parent_fd),
        ctypes.c_char_p(os.fsencode(slot.basename)),
        ctypes.c_int(0x1000),
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == 17:
            raise PhaseCliqueBuildProbeError("output publication collision")
        raise OSError(error, os.strerror(error), str(slot.display_path))
    try:
        _validate_output_slot_live(
            slot, expected_published_identity=expected_identity
        )
    except Exception:
        try:
            published = os.stat(
                slot.basename,
                dir_fd=slot.parent_fd,
                follow_symlinks=False,
            )
            if (int(published.st_dev), int(published.st_ino)) == expected_identity:
                os.unlink(slot.basename, dir_fd=slot.parent_fd)
                os.fsync(slot.parent_fd)
        except FileNotFoundError:
            pass
        raise
    os.fsync(slot.parent_fd)


def _hz_shape(hz: Any) -> dict[str, int]:
    return {
        "output_dimension": int(np.asarray(hz.c).size),
        "continuous_columns": int(hz.n_cont),
        "binary_columns": int(hz.n_bin),
        "upper_rows": int(hz.n_ub),
        "equality_rows": int(hz.n_eq),
        "constraint_nonzeros": int(
            hz.Auc.nnz + hz.Aub.nnz + hz.Ac.nnz + hz.Ab.nnz
        ),
        "generator_nonzeros": int(hz.Gc.nnz + hz.Gb.nnz),
    }


def _rbs_adaptive_k4_memory_forecast(hz: Any) -> dict[str, Any]:
    """Conservative payload lower bound for the native-objective K4 route.

    Relative to the live source HZ already represented in the entry RSS, the
    compact-candidate/native-LP stages require at least one additional HZ core
    plus one native candidate CSR payload (``C + S``).  Materialization may
    simultaneously retain two additional HZ cores (``2C``).  Those phases do
    not overlap after the one-use handoff, so the static lower bound is
    ``max(C + S, 2C)`` rather than their sum.  HiGHS structures, basis and
    allocator retention are deliberately excluded: this receipt may veto a
    run but can never authorize one without the independent RSS headroom gate.
    """

    full_hz_counts = {
        "compact_k4_native_model": 1,
        "verified_cut_reconstruction": 2,
        "materializer_private_handoff": 2,
        "native_objective_dual": 1,
    }
    candidate_csr_counts = {
        "compact_k4_native_model": 1,
        "verified_cut_reconstruction": 0,
        "materializer_private_handoff": 0,
        "native_objective_dual": 1,
    }
    try:
        dense_names = ("c", "b", "col_ids", "bcol_ids", "ub")
        csr_names = ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
        dense = [getattr(hz, name) for name in dense_names]
        csr = [getattr(hz, name) for name in csr_names]
        if (
            any(type(value) is not np.ndarray for value in dense)
            or any(type(value) is not sp.csr_matrix for value in csr)
        ):
            raise TypeError("noncanonical_hz_core")
        core_bytes = int(sum(value.nbytes for value in dense))
        core_bytes += int(
            sum(
                matrix.data.nbytes
                + matrix.indices.nbytes
                + matrix.indptr.nbytes
                for matrix in csr
            )
        )
        candidate_rows = int(hz.n_ub + hz.n_eq)
        candidate_nnz = int(
            hz.Auc.nnz + hz.Aub.nnz + hz.Ac.nnz + hz.Ab.nnz
        )
        candidate_csr_bytes = int(
            candidate_nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
            + (candidate_rows + 1) * np.dtype(np.int32).itemsize
        )
        candidate_phase_lower_bound = int(
            core_bytes + candidate_csr_bytes
        )
        materializer_phase_lower_bound = int(2 * core_bytes)
        native_objective_phase_lower_bound = int(
            core_bytes + candidate_csr_bytes
        )
        stage_increments = {
            "compact_k4_native_model": candidate_phase_lower_bound,
            "verified_cut_reconstruction": (
                materializer_phase_lower_bound
            ),
            "materializer_private_handoff": (
                materializer_phase_lower_bound
            ),
            "native_objective_dual": (
                native_objective_phase_lower_bound
            ),
        }
        static_lower_bound = int(
            max(stage_increments.values())
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return {
            "schema": "act.rbs_adaptive_k4_memory_forecast.v2",
            "status": "invalid",
            "candidate_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "entry_source_hz_core_already_resident": True,
            "hz_core_bytes": None,
            "candidate_csr_bytes": None,
            "peak_formula": (
                "max(2*hz_core_bytes,"
                "hz_core_bytes+candidate_csr_bytes)"
            ),
            "stage_additional_full_hz_core_counts": full_hz_counts,
            "stage_candidate_csr_payload_counts": candidate_csr_counts,
            "stage_increment_bytes": {
                name: None for name in full_hz_counts
            },
            "static_peak_increment_lower_bound_bytes": None,
            "native_model_release_between_stages_required": True,
            "highs_internal_overhead_included": False,
            "allocator_retention_included": False,
        }
    return {
        "schema": "act.rbs_adaptive_k4_memory_forecast.v2",
        "status": "computed",
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "entry_source_hz_core_already_resident": True,
        "hz_core_bytes": core_bytes,
        "candidate_csr_bytes": candidate_csr_bytes,
        "peak_formula": (
            "max(2*hz_core_bytes,"
            "hz_core_bytes+candidate_csr_bytes)"
        ),
        "stage_additional_full_hz_core_counts": full_hz_counts,
        "stage_candidate_csr_payload_counts": candidate_csr_counts,
        "stage_increment_bytes": stage_increments,
        "static_peak_increment_lower_bound_bytes": static_lower_bound,
        "native_model_release_between_stages_required": True,
        "highs_internal_overhead_included": False,
        "allocator_retention_included": False,
    }


def _rbs_adaptive_k4_memory_forecast_valid(
    receipt: Mapping[str, Any],
) -> bool:
    """Recompute the v2 lifecycle formula before using it as a veto."""

    keys = {
        "schema",
        "status",
        "candidate_only",
        "proof_authority",
        "verdict_authority",
        "entry_source_hz_core_already_resident",
        "hz_core_bytes",
        "candidate_csr_bytes",
        "peak_formula",
        "stage_additional_full_hz_core_counts",
        "stage_candidate_csr_payload_counts",
        "stage_increment_bytes",
        "static_peak_increment_lower_bound_bytes",
        "native_model_release_between_stages_required",
        "highs_internal_overhead_included",
        "allocator_retention_included",
    }
    stages = (
        "compact_k4_native_model",
        "verified_cut_reconstruction",
        "materializer_private_handoff",
        "native_objective_dual",
    )
    if (
        type(receipt) is not dict
        or set(receipt) != keys
        or receipt.get("schema")
        != "act.rbs_adaptive_k4_memory_forecast.v2"
        or receipt.get("status") != "computed"
        or receipt.get("candidate_only") is not True
        or receipt.get("proof_authority") is not False
        or receipt.get("verdict_authority") is not False
        or receipt.get("entry_source_hz_core_already_resident") is not True
        or receipt.get("peak_formula")
        != "max(2*hz_core_bytes,hz_core_bytes+candidate_csr_bytes)"
        or receipt.get("native_model_release_between_stages_required")
        is not True
        or receipt.get("highs_internal_overhead_included") is not False
        or receipt.get("allocator_retention_included") is not False
    ):
        return False
    core = receipt.get("hz_core_bytes")
    candidate = receipt.get("candidate_csr_bytes")
    full_counts = receipt.get("stage_additional_full_hz_core_counts")
    csr_counts = receipt.get("stage_candidate_csr_payload_counts")
    increments = receipt.get("stage_increment_bytes")
    if (
        type(core) is not int
        or core < 0
        or type(candidate) is not int
        or candidate < 0
        or type(full_counts) is not dict
        or set(full_counts) != set(stages)
        or type(csr_counts) is not dict
        or set(csr_counts) != set(stages)
        or type(increments) is not dict
        or set(increments) != set(stages)
    ):
        return False
    expected_full = dict(zip(stages, (1, 2, 2, 1)))
    expected_csr = dict(zip(stages, (1, 0, 0, 1)))
    expected_increments = {
        name: expected_full[name] * core + expected_csr[name] * candidate
        for name in stages
    }
    return bool(
        full_counts == expected_full
        and csr_counts == expected_csr
        and increments == expected_increments
        and receipt.get("static_peak_increment_lower_bound_bytes")
        == max(expected_increments.values())
    )


def _exact_candidate_kept_nonzeros(
    hz: Any,
    *,
    deadline: float,
) -> int:
    """Recompute the HiGHS dust-filtered parent nnz in bounded chunks."""

    total = 0
    for name in ("Auc", "Aub", "Ac", "Ab"):
        matrix = getattr(hz, name)
        if type(matrix) is not sp.csr_matrix:
            raise PhaseCliqueBuildProbeError(
                "candidate kept-nnz source is not canonical CSR"
            )
        data = matrix.data
        for start in range(0, int(data.size), 1 << 18):
            if time.monotonic() >= deadline:
                raise PhaseCliqueBuildProbeError(
                    "candidate kept-nnz replay exceeded deadline"
                )
            chunk = data[start : start + (1 << 18)]
            if not np.all(np.isfinite(chunk)):
                raise PhaseCliqueBuildProbeError(
                    "candidate kept-nnz replay saw nonfinite data"
                )
            total += int(
                np.count_nonzero(
                    np.abs(chunk)
                    > _RBS_ADAPTIVE_K4_CANDIDATE_DUST_ABS
                )
            )
    return total


def _certified_relaxed_upper(
    hz: Any,
    objective: np.ndarray,
    threshold: float,
    *,
    deadline: float,
) -> dict[str, Any]:
    """Propose one native-row dual and independently certify its upper."""

    started = time.monotonic()
    remaining = float(deadline - started)
    if remaining <= 0.0:
        return {"status": "skipped_deadline", "proof_authority": False}
    from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
        PersistentConflictOracleError,
        propose_native_split_row_objective_duals,
    )
    from act.back_end.solver.solver_hz import (
        _hz_independent_split_block_lp_lagrangian_upper,
        _hz_longdouble_to_outward_float64_upper,
    )

    objective = np.asarray(objective, dtype=np.float64).reshape(-1)
    if objective.size != int(np.asarray(hz.c).size):
        raise PhaseCliqueBuildProbeError("LP objective/output width mismatch")
    factor_objective = np.concatenate(
        [
            np.asarray(objective @ hz.Gc, dtype=np.float64).reshape(-1),
            np.asarray(objective @ hz.Gb, dtype=np.float64).reshape(-1),
        ]
    )
    remaining = float(deadline - time.monotonic())
    if remaining <= 0.0:
        return {
            "status": "lp_deadline_inconclusive",
            "proof_authority": False,
            "verdict_authority": False,
            "elapsed_seconds": float(time.monotonic() - started),
        }
    try:
        proposal = propose_native_split_row_objective_duals(
            hz,
            factor_objective,
            deadline=deadline,
        )
    except PersistentConflictOracleError as exc:
        status = (
            "lp_deadline_inconclusive"
            if time.monotonic() >= deadline
            or "deadline" in str(exc).lower()
            else "lp_inconclusive"
        )
        return {
            "status": status,
            "proof_authority": False,
            "verdict_authority": False,
            "candidate_error": f"{type(exc).__name__}:{str(exc)[:240]}",
            "elapsed_seconds": float(time.monotonic() - started),
        }
    if time.monotonic() >= deadline:
        return {
            "status": "lp_deadline_inconclusive",
            "proof_authority": False,
            "verdict_authority": False,
            "elapsed_seconds": float(time.monotonic() - started),
        }
    inequality_dual = proposal.upper_row_dual
    equality_dual = proposal.equality_row_dual
    certified, certificate = (
        _hz_independent_split_block_lp_lagrangian_upper(
            c=hz.c,
            Gc=hz.Gc,
            Gb=hz.Gb,
            C_row=objective,
            threshold=float(threshold),
            Auc=hz.Auc,
            Aub=hz.Aub,
            Ac=hz.Ac,
            Ab=hz.Ab,
            ub=hz.ub,
            b=hz.b,
            continuous_lb=np.full(
                hz.n_cont, -1.0, dtype=np.float64
            ),
            continuous_ub=np.full(
                hz.n_cont, 1.0, dtype=np.float64
            ),
            binary_lb=np.full(
                hz.n_bin, -1.0, dtype=np.float64
            ),
            binary_ub=np.full(
                hz.n_bin, 1.0, dtype=np.float64
            ),
            upper_row_dual=inequality_dual,
            equality_row_dual=equality_dual,
            deadline=deadline,
        )
    )
    if time.monotonic() >= deadline:
        return {
            "status": "lp_deadline_inconclusive",
            "proof_authority": False,
            "verdict_authority": False,
            "elapsed_seconds": float(time.monotonic() - started),
        }
    certified_float = None
    if certified is not None:
        try:
            certified_float = (
                _hz_longdouble_to_outward_float64_upper(certified)
            )
        except (TypeError, ValueError, OverflowError):
            certified = None
        if (
            certified is not None
            and (
                certificate.get("upper") != certified_float
                or certificate.get("upper_float64_rounding")
                != "toward_positive_infinity_from_longdouble_v1"
            )
        ):
            certified = None
            certified_float = None
    expected_input_sparse_nnz = int(
        hz.Gc.nnz
        + hz.Gb.nnz
        + hz.Auc.nnz
        + hz.Aub.nnz
        + hz.Ac.nnz
        + hz.Ab.nnz
    )
    certificate_route = {
        "schema": certificate.get("schema"),
        "route": certificate.get("route"),
        "uses_sparse_hstack": certificate.get("uses_sparse_hstack"),
        "uses_sparse_vstack": certificate.get("uses_sparse_vstack"),
        "assembled_sparse_nnz": certificate.get(
            "assembled_sparse_nnz"
        ),
        "input_sparse_nnz": certificate.get("input_sparse_nnz"),
        "recomputed_input_sparse_nnz": expected_input_sparse_nnz,
        "block_shapes": certificate.get("block_shapes"),
        "upper_float64_rounding": certificate.get(
            "upper_float64_rounding"
        ),
        "upper_outward_float64": certificate.get("upper"),
        "candidate_upper_row_dual_sha256": (
            _native_objective_f64_payload_sha256(
                inequality_dual
            )
        ),
        "candidate_equality_row_dual_sha256": (
            _native_objective_f64_payload_sha256(
                equality_dual
            )
        ),
    }
    proposal_receipt = dict(proposal.receipt)
    proposal_route = dict(proposal_receipt)
    status = (
        "certified_diagnostic_upper"
        if certified is not None
        else "certificate_rejected"
    )
    return {
        "status": status,
        "proof_authority": False,
        "verdict_authority": False,
        "solver_relaxation_value": float(
            np.dot(objective, np.asarray(hz.c).reshape(-1))
            - float(threshold)
            - float(proposal.solver_minimization_objective)
        ),
        "independently_certified_upper": (
            certified_float if certified is not None else None
        ),
        "certificate": certificate,
        "certificate_route": certificate_route,
        "objective_dual_proposal_receipt": proposal_receipt,
        "objective_dual_proposal_route": proposal_route,
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _current_rss_bytes() -> Optional[int]:
    """Read current Linux RSS for diagnostics; never grant gate authority."""

    try:
        fields = Path("/proc/self/statm").read_text(
            encoding="ascii"
        ).split()
        resident_pages = int(fields[1])
        page_bytes = int(os.sysconf("SC_PAGE_SIZE"))
        if resident_pages < 0 or page_bytes <= 0:
            return None
        return int(resident_pages * page_bytes)
    except (IndexError, OSError, TypeError, ValueError):
        return None


def _glibc_malloc_trim_diagnostic(
    *,
    rss_reader: Optional[Callable[[], Optional[int]]] = None,
    library_loader: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Best-effort Linux/glibc ``malloc_trim(0)`` with no gate authority.

    This helper is deliberately fail-closed as a diagnostic: an unsupported
    allocator, missing symbol, malformed return code, RSS read failure, or
    call exception is recorded and never converted into claimed headroom.
    Downstream gates continue to use a fresh, independently captured current
    RSS sample.
    """

    read_rss = _current_rss_bytes if rss_reader is None else rss_reader
    load_library = ctypes.CDLL if library_loader is None else library_loader

    def safe_rss() -> Optional[int]:
        try:
            value = read_rss()
            return value if type(value) is int and value >= 0 else None
        except Exception:
            return None

    before = safe_rss()
    status = "not_attempted"
    return_code: Optional[int] = None
    glibc_version: Optional[str] = None
    try:
        if sys.platform != "linux":
            status = "unsupported_non_linux"
        else:
            libc = load_library("libc.so.6", use_errno=True)
            version_function = getattr(libc, "gnu_get_libc_version")
            version_function.argtypes = []
            version_function.restype = ctypes.c_char_p
            raw_version = version_function()
            if type(raw_version) is not bytes or not raw_version:
                status = "unsupported_non_glibc"
            else:
                glibc_version = raw_version.decode("ascii", errors="strict")
                trim_function = getattr(libc, "malloc_trim")
                trim_function.argtypes = [ctypes.c_size_t]
                trim_function.restype = ctypes.c_int
                observed = trim_function(0)
                if type(observed) is not int:
                    status = "invalid_return_code"
                else:
                    return_code = observed
                    if observed == 1:
                        status = "called_memory_released"
                    elif observed == 0:
                        status = "called_no_memory_released"
                    else:
                        status = "invalid_return_code"
    except (AttributeError, OSError, UnicodeError):
        status = "unsupported_non_glibc_or_symbol_missing"
    except Exception:
        status = "call_error"
    after = safe_rss()
    released = (
        max(0, before - after)
        if type(before) is int and type(after) is int
        else None
    )
    return {
        "schema": "act.hybridz_glibc_malloc_trim_diagnostic.v1",
        "status": status,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "gate_authority": False,
        "platform": sys.platform,
        "allocator": "glibc" if glibc_version is not None else None,
        "glibc_version": glibc_version,
        "argument": 0,
        "return_code": return_code,
        "current_rss_before_bytes": before,
        "current_rss_after_bytes": after,
        "released_bytes": released,
    }


def _freeze_live_assert_value(value: Any) -> Any:
    """Detach a small assert payload without retaining the analyzed graph."""

    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).copy(order="C")
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if not callable(detach) or not callable(cpu):
        return value
    frozen = detach().cpu()
    contiguous = getattr(frozen, "contiguous", None)
    if callable(contiguous):
        frozen = contiguous()
    clone = getattr(frozen, "clone", None)
    if callable(clone):
        frozen = clone()
    return frozen


def _capture_resource_peaks(torch_module: Any = None) -> dict[str, Any]:
    """Capture monotone peaks plus non-authoritative current worker RSS."""

    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    allocated: Optional[int] = None
    reserved: Optional[int] = None
    cuda_initialized = False
    try:
        cuda = None if torch_module is None else torch_module.cuda
        cuda_initialized = bool(cuda is not None and cuda.is_initialized())
        if cuda_initialized:
            allocated = int(cuda.max_memory_allocated())
            reserved = int(cuda.max_memory_reserved())
    except Exception:
        allocated = None
        reserved = None
        cuda_initialized = False
    return {
        "peak_rss_bytes": rss_bytes,
        "current_rss_bytes": _current_rss_bytes(),
        "cuda_initialized": cuda_initialized,
        "cuda_peak_allocated_bytes": allocated,
        "cuda_peak_reserved_bytes": reserved,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _selector_target_record(target: Any) -> dict[str, Any]:
    """Return the exact public selector fields used by the split receipt."""

    layer_id = getattr(target, "layer_id", None)
    row = getattr(target, "row", None)
    guard = getattr(target, "guard", None)
    score = getattr(target, "score", None)
    facility_gain = getattr(target, "facility_gain", None)
    dominant_rival = getattr(target, "dominant_rival", None)
    if (
        type(layer_id) is not int
        or layer_id < 0
        or type(row) is not int
        or row < 0
        or guard not in {"none", "both"}
        or isinstance(score, (bool, np.bool_))
        or not isinstance(score, (int, float, np.integer, np.floating))
        or not math.isfinite(float(score))
        or float(score) < 0.0
        or isinstance(facility_gain, (bool, np.bool_))
        or not isinstance(
            facility_gain, (int, float, np.integer, np.floating)
        )
        or not math.isfinite(float(facility_gain))
        or float(facility_gain) < 0.0
        or type(dominant_rival) is not int
        or dominant_rival < 0
    ):
        raise PhaseCliqueBuildProbeError(
            "selector target contains malformed schedule fields"
        )
    return {
        "layer_id": int(layer_id),
        "row": int(row),
        "guard": str(guard),
        "score": float(score),
        "facility_gain": float(facility_gain),
        "dominant_rival": int(dominant_rival),
    }


def _split_rbs_adaptive_schedule(
    residual_plan: Any,
    *,
    primary_budget: int,
    expected_selector_budget: int,
    expected_property_sha256: Optional[str] = None,
    require_all_interval_survivors_processed: bool = False,
) -> tuple[
    tuple[tuple[int, int, str], ...],
    tuple[tuple[int, int], ...],
    dict[str, Any],
]:
    """Purely split one authenticated selector prefix into primary/reserve.

    The selector is never rerun here.  Every field in its stored schedule is
    checked against ``plan.targets`` before the first ``primary_budget`` rows
    are retained.  Only suffix rows from a primary layer can enter the
    reservoir, with the kernel's three-backups-per-primary cap applied per
    layer.  Nonmatching suffix rows remain visible in the receipt.
    """

    if (
        type(primary_budget) is not int
        or primary_budget <= 0
        or type(expected_selector_budget) is not int
        or expected_selector_budget <= primary_budget
        or type(require_all_interval_survivors_processed) is not bool
    ):
        raise PhaseCliqueBuildProbeError(
            "adaptive schedule budgets must be positive nested integers"
        )
    targets = getattr(residual_plan, "targets", None)
    plan_property_sha256 = getattr(
        residual_plan, "property_sha256", None
    )
    plan_targets_sha256 = getattr(residual_plan, "targets_sha256", None)
    selector_receipt = getattr(residual_plan, "receipt", None)
    if (
        not isinstance(targets, (tuple, list))
        or not _valid_sha256(plan_property_sha256)
        or not _valid_sha256(plan_targets_sha256)
        or not isinstance(selector_receipt, Mapping)
    ):
        raise PhaseCliqueBuildProbeError("selector plan is malformed")
    if (
        expected_property_sha256 is not None
        and (
            not _valid_sha256(expected_property_sha256)
            or plan_property_sha256 != expected_property_sha256
        )
    ):
        raise PhaseCliqueBuildProbeError(
            "selector property digest does not bind the live ASSERT"
        )

    full_schedule = [_selector_target_record(target) for target in targets]
    coordinates = [
        (item["layer_id"], item["row"]) for item in full_schedule
    ]
    if len(set(coordinates)) != len(coordinates):
        raise PhaseCliqueBuildProbeError(
            "selector schedule contains duplicate coordinates"
        )
    receipt_schedule = selector_receipt.get("schedule")
    rival_ids = selector_receipt.get("rival_ids")
    joint_focus_rival_id = selector_receipt.get("joint_focus_rival_id")
    selector_header_valid = bool(
        selector_receipt.get("schema")
        == "property_residual_selector_v1"
        and selector_receipt.get("status") == "selected"
        and selector_receipt.get("candidate_only") is True
        and selector_receipt.get("proof_authority") is False
        and selector_receipt.get("selection_policy")
        == "facility_first_then_same_rival_joint"
        and selector_receipt.get("property_sha256")
        == plan_property_sha256
        and isinstance(rival_ids, list)
        and bool(rival_ids)
        and all(type(value) is int and value >= 0 for value in rival_ids)
        and len(set(rival_ids)) == len(rival_ids)
        and type(joint_focus_rival_id) is int
        and joint_focus_rival_id in rival_ids
        and selector_receipt.get("rivals_processed") == len(rival_ids)
        and type(selector_receipt.get("targets_selected")) is int
        and selector_receipt.get("targets_selected") == len(full_schedule)
        and isinstance(receipt_schedule, list)
        and len(receipt_schedule) == len(full_schedule)
        and (
            not require_all_interval_survivors_processed
            or selector_receipt.get("all_interval_survivors_processed")
            is True
        )
    )
    if not selector_header_valid:
        raise PhaseCliqueBuildProbeError(
            "selector receipt header does not bind its schedule"
        )
    required_fields = {
        "layer_id",
        "row",
        "guard",
        "score",
        "facility_gain",
        "dominant_rival",
    }
    for receipt_item, target_item in zip(receipt_schedule, full_schedule):
        if (
            not isinstance(receipt_item, Mapping)
            or set(receipt_item) != required_fields
            or dict(receipt_item) != target_item
        ):
            raise PhaseCliqueBuildProbeError(
                "selector receipt schedule differs from plan.targets"
            )

    target_hash_payload = {
        "property_sha256": plan_property_sha256,
        "targets": [
            {
                "layer_id": item["layer_id"],
                "row": item["row"],
                "guard": item["guard"],
            }
            for item in full_schedule
        ],
    }
    recomputed_targets_sha256 = hashlib.sha256(
        _canonical_json(target_hash_payload)
    ).hexdigest()
    if recomputed_targets_sha256 != plan_targets_sha256:
        raise PhaseCliqueBuildProbeError(
            "selector targets digest does not match plan.targets"
        )
    builder_targets = tuple(
        (item["layer_id"], item["row"], item["guard"])
        for item in full_schedule
    )
    exposed_builder_targets = getattr(
        residual_plan, "builder_targets", builder_targets
    )
    if tuple(exposed_builder_targets) != builder_targets:
        raise PhaseCliqueBuildProbeError(
            "selector builder_targets differs from plan.targets"
        )

    primary_records = full_schedule[:primary_budget]
    primary_targets = builder_targets[:primary_budget]
    primary_counts: dict[int, int] = {}
    for item in primary_records:
        primary_counts[item["layer_id"]] = (
            primary_counts.get(item["layer_id"], 0) + 1
        )
    reserve_counts = {layer_id: 0 for layer_id in primary_counts}
    reserve_records: list[dict[str, Any]] = []
    dropped_cross_layer: list[dict[str, Any]] = []
    dropped_per_layer_cap: list[dict[str, Any]] = []
    for item in full_schedule[primary_budget:]:
        layer_id = item["layer_id"]
        if layer_id not in primary_counts:
            dropped_cross_layer.append(item)
            continue
        cap = (
            _RBS_ADAPTIVE_K4_MAX_BACKUPS_PER_PRIMARY
            * primary_counts[layer_id]
        )
        if reserve_counts[layer_id] >= cap:
            dropped_per_layer_cap.append(item)
            continue
        reserve_records.append(item)
        reserve_counts[layer_id] += 1
    reservoir = tuple(
        (item["layer_id"], item["row"]) for item in reserve_records
    )

    if len(full_schedule) != expected_selector_budget:
        status = "selector_prefix_incomplete"
    elif len(primary_targets) != primary_budget:
        status = "primary_prefix_incomplete"
    elif not reservoir:
        status = "no_same_layer_reserve"
    else:
        status = "ready"
    per_layer = [
        {
            "layer_id": int(layer_id),
            "primary_count": int(primary_counts[layer_id]),
            "reserve_count": int(reserve_counts[layer_id]),
            "reserve_cap": int(
                _RBS_ADAPTIVE_K4_MAX_BACKUPS_PER_PRIMARY
                * primary_counts[layer_id]
            ),
        }
        for layer_id in sorted(primary_counts)
    ]
    receipt = _checksummed({
        "schema": "act.rbs_adaptive_k4_schedule.v1",
        "status": status,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "selector_rerun": False,
        "all_interval_survivors_processed_required": bool(
            require_all_interval_survivors_processed
        ),
        "all_interval_survivors_processed": selector_receipt.get(
            "all_interval_survivors_processed"
        ),
        "selection_policy": selector_receipt.get("selection_policy"),
        "rival_ids": list(rival_ids),
        "joint_focus_rival_id": int(joint_focus_rival_id),
        "primary_budget": int(primary_budget),
        "expected_selector_budget": int(expected_selector_budget),
        "selector_target_count": int(len(full_schedule)),
        "property_sha256": plan_property_sha256,
        "selector_targets_sha256": plan_targets_sha256,
        "recomputed_targets_sha256": recomputed_targets_sha256,
        "selector_receipt_sha256": hashlib.sha256(
            _canonical_json(dict(selector_receipt))
        ).hexdigest(),
        "full_schedule_sha256": hashlib.sha256(
            _canonical_json(full_schedule)
        ).hexdigest(),
        "full_schedule": full_schedule,
        "primary_schedule": primary_records,
        "primary_builder_targets": [list(item) for item in primary_targets],
        "reserve_schedule": reserve_records,
        "exact_target_reservoir": [list(item) for item in reservoir],
        "dropped_cross_layer_schedule": dropped_cross_layer,
        "dropped_cross_layer_count": int(len(dropped_cross_layer)),
        "dropped_per_layer_cap_schedule": dropped_per_layer_cap,
        "dropped_per_layer_cap_count": int(len(dropped_per_layer_cap)),
        "per_layer": per_layer,
        "same_layer_only": all(
            layer_id in primary_counts for layer_id, _row in reservoir
        ),
        "per_layer_three_per_primary_cap_enforced": all(
            item["reserve_count"] <= item["reserve_cap"]
            for item in per_layer
        ),
        "selector_receipt_schedule_exactly_matched": True,
        "selector_targets_digest_recomputed": True,
    })
    return primary_targets, reservoir, receipt


def _binary_property_sha256(
    C: np.ndarray, thresholds: np.ndarray, *, kind: Any
) -> str:
    """Recompute the selector's binary64 ASSERT binding."""

    digest = hashlib.sha256()
    for value in (C, thresholds):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    kind_name = str(getattr(kind, "value", kind)).upper()
    digest.update(kind_name.encode("utf-8"))
    return digest.hexdigest()


def _f64_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _native_objective_f64_payload_sha256(value: Any) -> str:
    """Match the producer's raw contiguous binary64 payload digest."""

    array = np.ascontiguousarray(
        np.asarray(value, dtype=np.float64).reshape(-1)
    )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _rbs_adaptive_property_cube_receipt(
    hz: Any,
    C: np.ndarray,
    thresholds: np.ndarray,
    *,
    cube_upper: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Build the fixed iid2 constraint-free property-cube pre-gate."""

    if cube_upper is None:
        from act.back_end.solver.solver_hz import (
            _hz_cube_row_upper_bounds,
        )

        cube_upper = _hz_cube_row_upper_bounds
    C = np.ascontiguousarray(C, dtype=np.float64)
    thresholds = np.ascontiguousarray(
        thresholds, dtype=np.float64
    ).reshape(-1)
    if (
        C.ndim != 2
        or C.shape[1] != int(np.asarray(hz.c).size)
        or C.shape[0] != thresholds.size
        or not np.all(np.isfinite(C))
        or not np.all(np.isfinite(thresholds))
    ):
        raise PhaseCliqueBuildProbeError(
            "property cube inputs are malformed"
        )
    upper, guards = cube_upper(
        hz.c, hz.Gc, hz.Gb, C, thresholds
    )
    upper = np.ascontiguousarray(upper, dtype=np.float64).reshape(-1)
    guards = np.ascontiguousarray(guards, dtype=np.float64).reshape(-1)
    finite = bool(
        upper.size == C.shape[0]
        and guards.size == C.shape[0]
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(guards))
        and np.all(guards >= 0.0)
    )
    maximum = float(np.max(upper)) if upper.size and finite else None
    minimum = float(np.min(upper)) if upper.size and finite else None
    conditions = {
        "fixed_property_row_count_99": bool(C.shape[0] == 99),
        "fixed_output_dimension_100": bool(C.shape[1] == 100),
        "finite_outward_cube_bounds": finite,
        "worst_cube_upper_at_most_75": bool(
            maximum is not None
            and maximum <= _RBS_ADAPTIVE_K4_MAX_CUBE_UPPER
        ),
    }
    return _checksummed({
        "schema": "act.rbs_adaptive_k4_property_cube.v1",
        "status": (
            "passed"
            if all(value is True for value in conditions.values())
            else "rejected"
        ),
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "conditions": conditions,
        "row_count": int(C.shape[0]),
        "output_dimension": int(C.shape[1]),
        "minimum_upper": minimum,
        "maximum_upper": maximum,
        "nonpositive_rows": int(np.count_nonzero(upper <= 0.0))
        if finite
        else None,
        "maximum_roundoff_guard": float(np.max(guards))
        if guards.size and finite
        else None,
        "C_sha256": _f64_array_sha256(C),
        "thresholds_sha256": _f64_array_sha256(thresholds),
        "upper_sha256": _f64_array_sha256(upper),
        "guards_sha256": _f64_array_sha256(guards),
        "construction": "constraint_free_outward_guarded_hz_cube_support",
    })


def _local_receipt_checksum_valid(
    receipt: Any, *, schema: str
) -> bool:
    if not isinstance(receipt, Mapping) or receipt.get("schema") != schema:
        return False
    claimed = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    return bool(
        _valid_sha256(claimed)
        and secrets.compare_digest(
            claimed, hashlib.sha256(_canonical_json(payload)).hexdigest()
        )
    )


def _rbs_adaptive_reservoir_audit(
    metadata: Mapping[str, Any], schedule_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate the kernel's per-layer RBS replacement receipts."""

    primary_by_layer: dict[int, list[int]] = {}
    for raw in schedule_receipt.get("primary_builder_targets", ()):
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return {"passed": False, "reason": "primary_schedule_malformed"}
        layer_id, row, _guard = raw
        if type(layer_id) is not int or type(row) is not int:
            return {"passed": False, "reason": "primary_schedule_malformed"}
        primary_by_layer.setdefault(layer_id, []).append(row)
    reserve_by_layer: dict[int, list[int]] = {}
    for raw in schedule_receipt.get("exact_target_reservoir", ()):
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return {"passed": False, "reason": "reserve_schedule_malformed"}
        layer_id, row = raw
        if type(layer_id) is not int or type(row) is not int:
            return {"passed": False, "reason": "reserve_schedule_malformed"}
        reserve_by_layer.setdefault(layer_id, []).append(row)
    receipts = metadata.get("exact_target_reservoir_receipts")
    if not isinstance(receipts, list):
        return {"passed": False, "reason": "kernel_receipts_missing"}

    observed_layers: set[int] = set()
    replacement_slots: list[dict[str, int]] = []
    failures: list[str] = []
    selected_rows = 0

    def strict_int_set(value: Any) -> Optional[set[int]]:
        if not isinstance(value, list) or any(
            type(item) is not int or item < 0 for item in value
        ):
            return None
        if len(set(value)) != len(value):
            return None
        return set(value)

    for item in receipts:
        if not isinstance(item, Mapping):
            failures.append("kernel_receipt_malformed")
            continue
        layer_id = item.get("relu_layer_id")
        if type(layer_id) is not int or layer_id in observed_layers:
            failures.append("kernel_layer_identity_malformed")
            continue
        observed_layers.add(layer_id)
        expected_primary = primary_by_layer.get(layer_id)
        expected_reserve = reserve_by_layer.get(layer_id, [])
        if item.get("primary_rows") != expected_primary:
            failures.append(f"layer_{layer_id}_primary_mismatch")
        if item.get("reserve_rows") != expected_reserve:
            failures.append(f"layer_{layer_id}_reserve_mismatch")
        if (
            item.get("schema")
            != "operator_hz_exact_target_reservoir_v1"
            or item.get("enabled") is not True
            or item.get("candidate_only") is not True
            or item.get("proof_authority") is not False
            or item.get("same_layer_only") is not True
            or item.get("status") != "filled"
            or item.get("shortfall") != 0
            or item.get("all_primary_rows_rbs_tightened") is not True
            or item.get("all_selected_rows_rbs_tightened") is not True
            or item.get("unselected_reserves_use_ordinary_triangle")
            is not True
            or item.get("selected_rows_use_existing_exact_big_m")
            is not True
        ):
            failures.append(f"layer_{layer_id}_kernel_gate_rejected")
        raw_selected = item.get("selected_rows")
        if not isinstance(raw_selected, list) or any(
            type(row) is not int for row in raw_selected
        ):
            failures.append(f"layer_{layer_id}_selected_rows_malformed")
            raw_selected = []
        selected_rows += len(raw_selected)

        primary_set = set(expected_primary or ())
        reserve_set = set(expected_reserve)
        pre_unstable = strict_int_set(
            item.get("pre_screen_cube_unstable_primary")
        )
        tightened_primary = strict_int_set(
            item.get("primary_rows_rbs_tightened")
        )
        newly_stable = strict_int_set(
            item.get("rbs_newly_stabilized_primary")
        )
        stable_active = strict_int_set(
            item.get("post_screen_stabilized_active_primary")
        )
        stable_inactive = strict_int_set(
            item.get("post_screen_stabilized_inactive_primary")
        )
        selected_primary = strict_int_set(
            item.get("selected_primary_rows")
        )
        selected_reserve = strict_int_set(
            item.get("selected_reserve_rows")
        )
        selected_tightened = strict_int_set(
            item.get("selected_rows_rbs_tightened")
        )
        non_rbs_stable = strict_int_set(
            item.get("non_rbs_stable_primary_not_replaced")
        )
        set_fields = (
            pre_unstable,
            tightened_primary,
            newly_stable,
            stable_active,
            stable_inactive,
            selected_primary,
            selected_reserve,
            selected_tightened,
            non_rbs_stable,
        )
        if any(value is None for value in set_fields):
            failures.append(f"layer_{layer_id}_row_sets_malformed")
            pre_unstable = set()
            tightened_primary = set()
            newly_stable = set()
            stable_active = set()
            stable_inactive = set()
            selected_primary = set()
            selected_reserve = set()
            selected_tightened = set()
            non_rbs_stable = set()
        post_stable = stable_active | stable_inactive
        selected_set = set(raw_selected)
        if (
            pre_unstable != primary_set
            or tightened_primary != primary_set
            or not newly_stable.issubset(primary_set)
            or not newly_stable.issubset(post_stable)
            or non_rbs_stable
            or selected_primary != primary_set.difference(newly_stable)
            or not selected_reserve.issubset(reserve_set)
            or len(selected_reserve) != len(newly_stable)
            or selected_set != selected_primary.union(selected_reserve)
            or selected_tightened != selected_set
        ):
            failures.append(f"layer_{layer_id}_row_partition_mismatch")
        slots = item.get("replacement_slots")
        if not isinstance(slots, list):
            failures.append(f"layer_{layer_id}_replacement_slots_malformed")
            slots = []
        for slot in slots:
            if not isinstance(slot, Mapping):
                failures.append(f"layer_{layer_id}_replacement_slot_malformed")
                continue
            primary = slot.get("stabilized_primary_row")
            reserve = slot.get("selected_reserve_row")
            if (
                type(primary) is not int
                or type(reserve) is not int
                or primary not in pre_unstable
                or primary not in tightened_primary
                or primary not in newly_stable
                or primary not in post_stable
                or reserve not in selected_reserve
                or reserve not in reserve_set
            ):
                failures.append(
                    f"layer_{layer_id}_replacement_not_rbs_bound"
                )
                continue
            replacement_slots.append({
                "relu_layer_id": int(layer_id),
                "stabilized_primary_row": int(primary),
                "selected_reserve_row": int(reserve),
            })
        if item.get("replacement_count") != len(slots):
            failures.append(f"layer_{layer_id}_replacement_count_mismatch")
        if len(slots) != len(newly_stable):
            failures.append(f"layer_{layer_id}_replacement_bijection_mismatch")

    if observed_layers != set(primary_by_layer):
        failures.append("kernel_primary_layer_set_mismatch")
    if selected_rows != _RBS_ADAPTIVE_K4_PRIMARY_BUDGET:
        failures.append("kernel_selected_exact_count_mismatch")
    return {
        "schema": "act.rbs_adaptive_k4_reservoir_audit.v1",
        "passed": not failures,
        "candidate_only": True,
        "proof_authority": False,
        "expected_primary_by_layer": [
            {"layer_id": layer_id, "rows": rows}
            for layer_id, rows in sorted(primary_by_layer.items())
        ],
        "expected_reserve_by_layer": [
            {"layer_id": layer_id, "rows": rows}
            for layer_id, rows in sorted(reserve_by_layer.items())
        ],
        "observed_layer_count": int(len(observed_layers)),
        "selected_exact_rows": int(selected_rows),
        "replacement_slots": replacement_slots,
        "replacement_count": int(len(replacement_slots)),
        "replacement_binding_rule": (
            "primary_pre_cube_unstable+rbs_tightened+post_screen_stable+"
            "same_layer_selected_reserve"
        ),
        "failures": failures,
    }


def _rbs_adaptive_k4_pre_gate(
    source_build: Any,
    *,
    schedule_receipt: Mapping[str, Any],
    property_cube_receipt: Mapping[str, Any],
    build_seconds: float,
    input_sha256: Mapping[str, Any],
    resources: Mapping[str, Any],
    remaining_seconds: float,
) -> dict[str, Any]:
    """Apply every preregistered stop-loss before K4 is callable."""

    hz = source_build.hz
    shape = _hz_shape(hz)
    metadata = getattr(source_build, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise PhaseCliqueBuildProbeError(
            "Operator-HZ metadata is unavailable for adaptive pre-gate"
        )
    reservoir_audit = _rbs_adaptive_reservoir_audit(
        metadata, schedule_receipt
    )
    phase_receipts = metadata.get("residual_phase_screen_receipts")
    if not isinstance(phase_receipts, list):
        phase_receipts = []
    prepared = [
        item
        for item in phase_receipts
        if isinstance(item, Mapping) and item.get("status") == "prepared"
    ]
    strict_prepared = bool(
        len(prepared) == 4
        and all(
            item.get("schema")
            == "operator_hz_residual_phase_screen_v1"
            and item.get("mode") == "strict_bound_improvement"
            and item.get("proof_authority") is True
            and type(item.get("retained_count")) is int
            and item.get("retained_count") > 0
            for item in prepared
        )
    )
    prepared_retained_total = sum(
        int(item.get("retained_count", 0)) for item in prepared
    )
    layers = metadata.get("layers")
    if not isinstance(layers, list):
        layers = []
    conv_layers = [
        item
        for item in layers
        if isinstance(item, Mapping) and item.get("kind") == "CONV2D"
    ]
    traversal_release = metadata.get("traversal_cache_release")
    if not isinstance(traversal_release, Mapping):
        traversal_release = {}
    performance = getattr(source_build, "performance_diagnostic", None)
    if not isinstance(performance, Mapping):
        performance = {}
    performance_layers = performance.get("layers")
    performance_stages = performance.get("stages")
    applied_receipts = [
        item.get("residual_phase_screen")
        for item in layers
        if isinstance(item, Mapping)
        and item.get("kind") == "RELU"
        and isinstance(item.get("residual_phase_screen"), Mapping)
        and item["residual_phase_screen"].get("status") == "applied"
    ]
    applied_rows_total = sum(
        int(item.get("rows_applied", 0)) for item in applied_receipts
    )
    aggregate_tightened = metadata.get(
        "residual_bound_screen_rows_tightened"
    )
    rss = resources.get("peak_rss_bytes")
    current_rss = resources.get("current_rss_bytes")
    allocated = resources.get("cuda_peak_allocated_bytes")
    memory_forecast = _rbs_adaptive_k4_memory_forecast(hz)
    static_peak_increment = memory_forecast.get(
        "static_peak_increment_lower_bound_bytes"
    )
    conditions = {
        "fixed_onnx_sha256": bool(
            input_sha256.get("onnx")
            == _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256
        ),
        "fixed_vnnlib_sha256": bool(
            input_sha256.get("vnnlib")
            == _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256
        ),
        "fixed_instances_csv_sha256": bool(
            input_sha256.get("instances_csv")
            == _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256
        ),
        "schedule_receipt_valid": _local_receipt_checksum_valid(
            schedule_receipt,
            schema="act.rbs_adaptive_k4_schedule.v1",
        )
        and schedule_receipt.get("status") == "ready",
        "property_cube_gate_passed": _local_receipt_checksum_valid(
            property_cube_receipt,
            schema="act.rbs_adaptive_k4_property_cube.v1",
        )
        and property_cube_receipt.get("status") == "passed",
        "build_under_20_seconds": bool(
            type(build_seconds) is float
            and 0.0 <= build_seconds <= _RBS_ADAPTIVE_K4_MAX_BUILD_SECONDS
        ),
        "shape_output_100": bool(shape["output_dimension"] == 100),
        "shape_binary_4": bool(shape["binary_columns"] == 4),
        "shape_upper_rows_within_cap": bool(
            shape["upper_rows"] <= _RBS_ADAPTIVE_K4_MAX_UPPER_ROWS
        ),
        "shape_constraint_nonzeros_within_cap": bool(
            shape["constraint_nonzeros"]
            <= _RBS_ADAPTIVE_K4_MAX_CONSTRAINT_NONZEROS
        ),
        "exact_budget_requested_and_used_4": bool(
            metadata.get("exact_budget_requested") == 4
            and metadata.get("exact_budget_used") == 4
        ),
        "materialized_add_build": metadata.get("materialize_add") is True,
        "owned_canonical_sparse_hz_assembly": bool(
            metadata.get("sparse_hz_core_assembly")
            == "owned_canonical_no_recopy_v1"
        ),
        "all_19_convs_use_vectorized_exact_csr": bool(
            len(conv_layers) == _RBS_ADAPTIVE_K4_EXPECTED_CONV_LAYERS
            and all(
                item.get("operator_csr_builder")
                == "vectorized_exact_csr_v1"
                for item in conv_layers
            )
        ),
        "traversal_caches_released_before_final_assembly": bool(
            traversal_release.get("schema")
            == "operator_hz_traversal_cache_release_v1"
            and traversal_release.get("status")
            == "released_before_final_sparse_assembly"
            and traversal_release.get("numeric_semantics_changed") is False
            and traversal_release.get(
                "constraint_blocks_released_before_constructor"
            )
            is True
            and type(traversal_release.get("expr_count")) is int
            and traversal_release.get("expr_count") > 0
        ),
        "non_authoritative_build_telemetry_complete": bool(
            performance.get("schema")
            == "operator_hz_build_performance_diagnostic_v1"
            and performance.get("candidate_only") is True
            and performance.get("proof_authority") is False
            and performance.get("verdict_authority") is False
            and isinstance(performance_layers, list)
            and len(performance_layers) == metadata.get("n_layers")
            and isinstance(performance_stages, Mapping)
            and type(performance.get("total_wall_seconds")) is float
            and math.isfinite(performance.get("total_wall_seconds"))
            and performance.get("total_wall_seconds") >= 0.0
        ),
        "bound_screen_only": bool(
            metadata.get("residual_bound_screen_requested") is True
            and metadata.get("residual_phase_screen_requested") is False
        ),
        "rbs_scanned_1232": bool(
            metadata.get("residual_phase_screen_rows_scanned") == 1_232
        ),
        "rbs_tightened_1232": bool(
            metadata.get("residual_bound_screen_rows_tightened") == 1_232
        ),
        "rbs_four_strict_layers": strict_prepared,
        "rbs_prepare_apply_aggregate_match_1232": bool(
            prepared_retained_total
            == applied_rows_total
            == aggregate_tightened
            == 1_232
        ),
        "rbs_elapsed_at_most_1_second": bool(
            type(metadata.get("residual_phase_screen_elapsed_seconds"))
            is float
            and 0.0
            <= metadata.get("residual_phase_screen_elapsed_seconds")
            <= 1.0
        ),
        "rbs_active_at_least_26": bool(
            type(metadata.get("residual_phase_screen_stabilized_active"))
            is int
            and metadata.get("residual_phase_screen_stabilized_active")
            >= 26
        ),
        "rbs_inactive_at_least_296": bool(
            type(metadata.get("residual_phase_screen_stabilized_inactive"))
            is int
            and metadata.get("residual_phase_screen_stabilized_inactive")
            >= 296
        ),
        "reservoir_requested": bool(
            metadata.get("exact_target_reservoir_requested") is True
            and metadata.get("exact_target_reservoir_primary_count") == 4
            and metadata.get("exact_target_reservoir_backup_count")
            == len(schedule_receipt.get("exact_target_reservoir", ()))
        ),
        "reservoir_no_shortfall": bool(
            metadata.get("exact_target_reservoir_shortfall") == 0
        ),
        "reservoir_receipts_revalidated": bool(
            reservoir_audit.get("passed") is True
        ),
        "verified_frame_closed": bool(
            getattr(source_build, "verified_preactivation_frame", None)
            is None
            and metadata.get("verified_preactivation_frame_export_requested")
            is False
            and metadata.get("verified_preactivation_frame_exported") is False
        ),
        "peak_rss_within_2_5_gib": bool(
            type(rss) is int
            and 0 <= rss <= _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
        ),
        "current_rss_has_64_mib_phase_headroom": bool(
            type(current_rss) is int
            and 0
            <= current_rss
            <= (
                _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
                - _RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
            )
        ),
        "static_k4_memory_lower_bound_has_64_mib_headroom": bool(
            _rbs_adaptive_k4_memory_forecast_valid(memory_forecast)
            and type(static_peak_increment) is int
            and static_peak_increment >= 0
            and type(current_rss) is int
            and current_rss >= 0
            and current_rss + static_peak_increment
            <= (
                _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
                - _RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
            )
        ),
        "cuda_initialized": resources.get("cuda_initialized") is True,
        "cuda_allocated_within_8_gib": bool(
            type(allocated) is int
            and 0
            <= allocated
            <= _RBS_ADAPTIVE_K4_MAX_CUDA_ALLOCATED_BYTES
        ),
        "full_30_second_phase_window_remaining": bool(
            type(remaining_seconds) is float
            and remaining_seconds >= _RBS_ADAPTIVE_K4_PHASE_SECONDS
        ),
    }
    failures = [
        name for name, passed in conditions.items() if passed is not True
    ]
    return _checksummed({
        "schema": "act.rbs_adaptive_k4_pre_gate.v1",
        "status": "passed" if not failures else "rejected",
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "conditions": conditions,
        "failed_conditions": failures,
        "build_seconds": float(build_seconds),
        "shape": shape,
        "historical_shape_sanity": {
            "expected": dict(_RBS_ADAPTIVE_K4_HISTORICAL_SHAPE),
            "exact_match": shape == _RBS_ADAPTIVE_K4_HISTORICAL_SHAPE,
            "gate_authority": False,
        },
        "rbs": {
            "rows_scanned": metadata.get(
                "residual_phase_screen_rows_scanned"
            ),
            "rows_tightened": metadata.get(
                "residual_bound_screen_rows_tightened"
            ),
            "layers_prepared": metadata.get(
                "residual_phase_screen_layers_prepared"
            ),
            "stabilized_active": metadata.get(
                "residual_phase_screen_stabilized_active"
            ),
            "stabilized_inactive": metadata.get(
                "residual_phase_screen_stabilized_inactive"
            ),
            "elapsed_seconds": metadata.get(
                "residual_phase_screen_elapsed_seconds"
            ),
            "prepared_retained_rows": int(prepared_retained_total),
            "applied_rows": int(applied_rows_total),
            "aggregate_tightened_rows": aggregate_tightened,
        },
        "reservoir_audit": reservoir_audit,
        "conv_builder": {
            "expected_conv_layers": int(
                _RBS_ADAPTIVE_K4_EXPECTED_CONV_LAYERS
            ),
            "observed_conv_layers": int(len(conv_layers)),
            "layers": [
                {
                    "layer_id": item.get("layer_id"),
                    "operator_nnz": item.get("operator_nnz"),
                    "operator_csr_builder": item.get(
                        "operator_csr_builder"
                    ),
                }
                for item in conv_layers
            ],
        },
        "traversal_cache_release": dict(traversal_release),
        "performance_diagnostic": dict(performance),
        "property_cube_receipt_sha256": property_cube_receipt.get(
            "receipt_sha256"
        ),
        "schedule_receipt_sha256": schedule_receipt.get(
            "receipt_sha256"
        ),
        "resource_usage": dict(resources),
        "memory_forecast": memory_forecast,
        "remaining_seconds": float(max(0.0, remaining_seconds)),
    })


def _localized_e2_live_seals(hz: Any) -> tuple[str, str]:
    from act.back_end.hybridz_tf.adaptive_phase_forest import (
        sparse_hz_semantic_digest,
    )
    from act.back_end.hybridz_tf.operator_localized_phase_edge_candidate import (
        _row_tag_digest,
    )

    return sparse_hz_semantic_digest(hz), _row_tag_digest(hz)


def _localized_e2_adapter_checksum_valid(candidate: Any) -> bool:
    from act.back_end.hybridz_tf import (
        operator_localized_phase_edge_candidate as adapter_module,
    )
    from act.back_end.hybridz_tf.operator_localized_phase_edge_candidate import (
        OperatorLocalizedPhaseEdgeCandidateResult,
    )

    if type(candidate) is not OperatorLocalizedPhaseEdgeCandidateResult:
        return False
    try:
        observed = adapter_module._sha256(
            adapter_module._result_payload(candidate, include_digest=False)
        )
    except Exception:
        return False
    return _valid_sha256(candidate.result_sha256) and observed == candidate.result_sha256


def _verify_localized_e2_exact_candidate(
    source_build: Any,
    candidate: Any,
    selection: Any,
    *,
    deadline: float,
    candidate_kwargs: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Recheck the adapter receipt and exact certificate before any cut copy."""

    from act.back_end.hybridz_tf import (
        localized_phase_conflict_oracle as localized_module,
    )
    from act.back_end.hybridz_tf import (
        operator_exact_relu_phase_cliques as clique_module,
    )
    from act.back_end.hybridz_tf import (
        operator_localized_phase_edge_candidate as adapter_module,
    )
    from act.back_end.hybridz_tf.operator_localized_phase_edge_candidate import (
        OperatorLocalizedPhaseEdgeCaps,
    )
    from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
        verify_exact_dual_ray_conflict_certificate,
    )

    if not _localized_e2_adapter_checksum_valid(candidate):
        return False
    localized = candidate.localized_result
    certificate = candidate.certificate
    caps = candidate.caps
    try:
        import inspect

        signature = inspect.signature(
            adapter_module.run_operator_localized_phase_edge_candidate
        )
        option_names = tuple(
            name
            for name in signature.parameters
            if name
            not in {"build", "focused_rivals", "selection", "deadline", "enabled"}
        )
        supplied_options = dict(candidate_kwargs or {})
        if any(name not in option_names for name in supplied_options):
            return False
        expected_options = {
            name: supplied_options.get(name, signature.parameters[name].default)
            for name in option_names
        }
        expected_caps, clique_caps = adapter_module._normalize_caps(
            **expected_options
        )
        if (
            type(caps) is not OperatorLocalizedPhaseEdgeCaps
            or adapter_module._caps_payload(caps)
            != adapter_module._caps_payload(expected_caps)
        ):
            return False
        live_parent_digest, live_tag_digest = _localized_e2_live_seals(
            source_build.hz
        )
        expected_source_modes = adapter_module._EXPECTED_SOURCE_MODES
        expected_source_modes_sha256 = adapter_module._source_modes_digest(
            expected_source_modes
        )
        expected_build_binding = adapter_module._build_binding(
            source_build,
            parent_digest=live_parent_digest,
            row_tag_digest=live_tag_digest,
            producer_nonempty_seal_verified=True,
            source_modes=expected_source_modes,
            source_modes_sha256=expected_source_modes_sha256,
        )
        ranked, omitted, excluded = clique_module._ranked_subset(
            selection,
            caps=clique_caps,
            deadline=deadline,
        )
        expected_subset_digest = clique_module._subset_binding_digest(
            selection=selection,
            caps=clique_caps,
            ranked=ranked,
            omitted_zero_bcol_ids=omitted,
            excluded_selected_bcol_ids=excluded,
            deadline=deadline,
        )
        expected_literals = clique_module._make_bound_literals(
            parent_digest=live_parent_digest,
            subset_digest=expected_subset_digest,
            ranked=ranked,
        )
        localized_digest = localized_module._sha256(
            localized_module._result_payload(localized, include_digest=False)
        )
        selection_digest = selection.selection_digest
        property_digest = selection.property_digest
    except Exception:
        return False
    if (
        candidate.enabled is not True
        or candidate.status != "certified_localized_phase_edge"
        or candidate.edge_accepted is not True
        or candidate.parent_unchanged is not True
        or candidate.proof_authority is not False
        or candidate.producer_nonempty_seal_verified is not True
        or candidate.build_binding_sha256 != expected_build_binding
        or candidate.parent_semantic_digest != live_parent_digest
        or candidate.terminal_parent_semantic_digest != live_parent_digest
        or candidate.operator_row_tag_digest != live_tag_digest
        or candidate.terminal_operator_row_tag_digest != live_tag_digest
        or candidate.source_modes != expected_source_modes
        or candidate.source_modes_sha256 != expected_source_modes_sha256
        or type(candidate.literals) is not tuple
        or len(candidate.literals) != 2
        or adapter_module._ranked_payload(candidate.ranked_phases)
        != adapter_module._ranked_payload(ranked)
        or list(candidate.omitted_zero_bcol_ids) != list(omitted)
        or list(candidate.excluded_selected_bcol_ids) != list(excluded)
        or adapter_module._literal_payload(candidate.literals)
        != adapter_module._literal_payload(expected_literals)
        or candidate.subset_binding_digest != expected_subset_digest
        or certificate is None
        or localized is None
        or localized.edge_accepted is not True
        or localized.certificate != certificate
        or localized.proof_authority is not False
        or not _valid_sha256(localized.result_sha256)
        or localized_digest != localized.result_sha256
        or candidate.localized_result_sha256 != localized.result_sha256
        or candidate.selection_digest != selection_digest
        or candidate.focused_property_digest != property_digest
        or not _valid_sha256(candidate.subset_binding_digest)
        or not _valid_sha256(candidate.ordered_source_frame_sha256)
        or candidate.subset_binding_digest != certificate.property_digest
        or candidate.parent_semantic_digest != certificate.parent_semantic_digest
        or candidate.ordered_source_frame_sha256
        != certificate.ordered_source_frame_sha256
        or certificate.literals != candidate.literals
    ):
        return False
    try:
        return bool(
            verify_exact_dual_ray_conflict_certificate(
                source_build.hz,
                certificate,
                property_digest=candidate.subset_binding_digest,
                deadline=deadline,
                max_source_terms=caps.max_source_terms,
                max_multiplier_bits=caps.max_multiplier_bits,
                max_exact_bits=caps.max_exact_bits,
                max_exact_nonzeros=caps.max_exact_nonzeros,
            )
        )
    except Exception:
        return False


def _localized_e2_candidate_summary(candidate: Any) -> dict[str, Any]:
    certificate = getattr(candidate, "certificate", None)
    literals = getattr(candidate, "literals", ())
    return {
        "status": getattr(candidate, "status", None),
        "reason": getattr(candidate, "reason", None),
        "edge_accepted": getattr(candidate, "edge_accepted", False) is True,
        "literal_count": len(literals) if type(literals) is tuple else None,
        "result_sha256": getattr(candidate, "result_sha256", None),
        "certificate_sha256": getattr(certificate, "certificate_sha256", None),
        "localized_result_sha256": getattr(
            candidate, "localized_result_sha256", None
        ),
        "build_binding_sha256": getattr(candidate, "build_binding_sha256", None),
        "parent_semantic_digest": getattr(
            candidate, "parent_semantic_digest", None
        ),
        "terminal_parent_semantic_digest": getattr(
            candidate, "terminal_parent_semantic_digest", None
        ),
        "operator_row_tag_digest": getattr(
            candidate, "operator_row_tag_digest", None
        ),
        "terminal_operator_row_tag_digest": getattr(
            candidate, "terminal_operator_row_tag_digest", None
        ),
        "selection_digest": getattr(candidate, "selection_digest", None),
        "subset_binding_digest": getattr(
            candidate, "subset_binding_digest", None
        ),
        "focused_property_digest": getattr(
            candidate, "focused_property_digest", None
        ),
        "ordered_source_frame_sha256": getattr(
            candidate, "ordered_source_frame_sha256", None
        ),
        "source_modes_sha256": getattr(
            candidate, "source_modes_sha256", None
        ),
        "producer_nonempty_seal_verified": getattr(
            candidate, "producer_nonempty_seal_verified", False
        )
        is True,
        "proof_authority": False,
        "verdict_authority": False,
    }


def _private_localized_pair_cut_valid(
    live_hz: Any,
    cut_hz: Any,
    literals: Sequence[Any],
) -> bool:
    """Require an exact, no-alias one-row pair-cut copy of the live HZ."""

    from act.back_end.hybridz_tf.property_phase_conflict_clique import (
        PhaseLiteral,
    )
    from act.back_end.solver.solver_hz import SparseHZono

    dense_names = ("c", "b", "ub", "col_ids", "bcol_ids")
    sparse_names = ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")

    def array_exact(left: Any, right: Any) -> bool:
        return (
            type(left) is np.ndarray
            and type(right) is np.ndarray
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.flags.c_contiguous
            and right.flags.c_contiguous
            and np.array_equal(
                left.view(np.uint8).reshape(-1),
                right.view(np.uint8).reshape(-1),
            )
        )

    def csr_valid(value: Any) -> bool:
        return (
            type(value) is sp.csr_matrix
            and value.has_canonical_format
            and value.has_sorted_indices
            and value.data.flags.c_contiguous
            and value.indices.flags.c_contiguous
            and value.indptr.flags.c_contiguous
        )

    def csr_exact(left: Any, right: Any) -> bool:
        return (
            csr_valid(left)
            and csr_valid(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and array_exact(left.data, right.data)
            and array_exact(left.indices, right.indices)
            and array_exact(left.indptr, right.indptr)
        )

    def csr_prefix_exact(parent: Any, child: Any) -> bool:
        if (
            not csr_valid(parent)
            or not csr_valid(child)
            or child.dtype != parent.dtype
            or child.shape != (parent.shape[0] + 1, parent.shape[1])
        ):
            return False
        prefix_nnz = int(child.indptr[parent.shape[0]])
        return (
            prefix_nnz == parent.nnz
            and array_exact(parent.data, child.data[:prefix_nnz])
            and array_exact(parent.indices, child.indices[:prefix_nnz])
            and array_exact(
                parent.indptr, child.indptr[: parent.shape[0] + 1]
            )
        )

    if (
        type(live_hz) is not SparseHZono
        or type(cut_hz) is not SparseHZono
        or cut_hz is live_hz
        or type(literals) is not tuple
        or len(literals) != 2
        or any(type(literal) is not PhaseLiteral for literal in literals)
        or any(literal.phase not in {-1, 1} for literal in literals)
        or literals[0].stable_bcol_id == literals[1].stable_bcol_id
        or any(
            "conditional" in name.lower() or "active" in name.lower()
            for hz in (live_hz, cut_hz)
            for name in vars(hz)
        )
    ):
        return False
    if any(not csr_valid(getattr(hz, name, None)) for hz in (live_hz, cut_hz) for name in sparse_names):
        return False
    if (
        cut_hz.n_out != live_hz.n_out
        or cut_hz.n_cont != live_hz.n_cont
        or cut_hz.n_bin != live_hz.n_bin
        or cut_hz.n_eq != live_hz.n_eq
        or cut_hz.n_ub != live_hz.n_ub + 1
    ):
        return False
    for name in ("c", "b", "col_ids", "bcol_ids"):
        if not array_exact(getattr(live_hz, name), getattr(cut_hz, name)):
            return False
    for name in ("Gc", "Gb", "Ac", "Ab"):
        if not csr_exact(getattr(live_hz, name), getattr(cut_hz, name)):
            return False
    if (
        not array_exact(live_hz.ub, cut_hz.ub[:-1])
        or cut_hz.ub.dtype != np.dtype(np.float64)
        or not array_exact(
            cut_hz.ub[-1:], np.asarray([0.0], dtype=np.float64)
        )
        or not csr_prefix_exact(live_hz.Auc, cut_hz.Auc)
        or int(cut_hz.Auc.indptr[-1])
        != int(cut_hz.Auc.indptr[-2])
        or not csr_prefix_exact(live_hz.Aub, cut_hz.Aub)
    ):
        return False
    positions = {
        int(stable_id): position
        for position, stable_id in enumerate(live_hz.bcol_ids.tolist())
    }
    if len(positions) != live_hz.n_bin or any(
        literal.stable_bcol_id not in positions for literal in literals
    ):
        return False
    expected_items = sorted(
        (
            positions[literal.stable_bcol_id],
            float(literal.phase),
        )
        for literal in literals
    )
    last_start = int(cut_hz.Aub.indptr[-2])
    last_stop = int(cut_hz.Aub.indptr[-1])
    expected_indices = np.asarray(
        [position for position, _phase in expected_items],
        dtype=cut_hz.Aub.indices.dtype,
    )
    expected_data = np.asarray(
        [phase for _position, phase in expected_items],
        dtype=np.float64,
    )
    if (
        last_stop - last_start != 2
        or not array_exact(
            cut_hz.Aub.indices[last_start:last_stop], expected_indices
        )
        or not array_exact(
            cut_hz.Aub.data[last_start:last_stop], expected_data
        )
    ):
        return False

    live_buffers = [getattr(live_hz, name) for name in dense_names]
    cut_buffers = [getattr(cut_hz, name) for name in dense_names]
    for name in sparse_names:
        live_matrix = getattr(live_hz, name)
        cut_matrix = getattr(cut_hz, name)
        live_buffers.extend((live_matrix.data, live_matrix.indices, live_matrix.indptr))
        cut_buffers.extend((cut_matrix.data, cut_matrix.indices, cut_matrix.indptr))
    try:
        if any(
            np.shares_memory(live_buffer, cut_buffer)
            for live_buffer in live_buffers
            for cut_buffer in cut_buffers
        ):
            return False
    except Exception:
        return False
    return True


def _run_localized_e2_transaction(
    source_build: Any,
    focused_rivals: Sequence[Any],
    selection: Any,
    *,
    focused_encoded_row: int,
    objective_rows: np.ndarray,
    thresholds: np.ndarray,
    deadline: float,
    candidate_deadline: Optional[float] = None,
    overall_started: Optional[float] = None,
    run_candidate: Optional[Callable[..., Any]] = None,
    candidate_kwargs: Optional[Mapping[str, Any]] = None,
    copy_pair_cut: Optional[Callable[[Any, Sequence[Any]], Any]] = None,
    lp_upper: Callable[..., Mapping[str, Any]] = _certified_relaxed_upper,
    live_seals: Callable[[Any], tuple[str, str]] = _localized_e2_live_seals,
    validate_exact_candidate: Callable[..., bool] = (
        _verify_localized_e2_exact_candidate
    ),
    validate_adapter_checksum: Callable[[Any], bool] = (
        _localized_e2_adapter_checksum_valid
    ),
    validate_private_cut: Callable[[Any, Any, Sequence[Any]], bool] = (
        _private_localized_pair_cut_valid
    ),
    resource_peaks: Callable[[], Mapping[str, Any]] = _capture_resource_peaks,
) -> dict[str, Any]:
    """Run one E2 adapter and a private diagnostic pair-cut transaction."""

    transaction_started = time.monotonic()
    if candidate_deadline is None:
        candidate_deadline = deadline
    candidate_deadline = min(float(candidate_deadline), float(deadline))
    if overall_started is None:
        overall_started = transaction_started
    if run_candidate is None:
        from act.back_end.hybridz_tf.operator_localized_phase_edge_candidate import (
            run_operator_localized_phase_edge_candidate,
        )

        run_candidate = run_operator_localized_phase_edge_candidate
    candidate_options = dict(candidate_kwargs or {})
    if {"enabled", "deadline"}.intersection(candidate_options):
        raise PhaseCliqueBuildProbeError(
            "localized candidate options may not override authorization"
        )
    if copy_pair_cut is None:
        from act.back_end.hybridz_tf.property_phase_conflict_clique import (
            _copy_parent_with_clique_cut,
        )

        copy_pair_cut = _copy_parent_with_clique_cut

    initial_parent_digest, initial_tag_digest = live_seals(source_build.hz)
    candidate: Any = None
    adapter_call_count = 0
    cut_attempted = False
    cut_structurally_validated = False
    lp_attempted = False
    timings: dict[str, float] = {}

    def finish(
        status: str,
        reason: str,
        *,
        lp_tightness: Optional[Mapping[str, Any]] = None,
        gate_conditions: Optional[Mapping[str, bool]] = None,
        resource_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        terminal_parent_digest, terminal_tag_digest = live_seals(source_build.hz)
        resources = dict(
            resource_peaks() if resource_snapshot is None else resource_snapshot
        )
        total_seconds = float(max(0.0, time.monotonic() - overall_started))
        timings["localized_e2_transaction_seconds"] = float(
            max(0.0, time.monotonic() - transaction_started)
        )
        timings["total_seconds_at_gate"] = total_seconds
        unchanged = (
            terminal_parent_digest == initial_parent_digest
            and terminal_tag_digest == initial_tag_digest
        )
        conditions = dict(gate_conditions or {})
        conditions.setdefault("live_parent_unchanged", unchanged)
        conditions.setdefault("total_under_60_seconds", total_seconds < 60.0)
        rss = resources.get("peak_rss_bytes")
        allocated = resources.get("cuda_peak_allocated_bytes")
        conditions.setdefault(
            "peak_rss_within_cap",
            type(rss) is int and 0 <= rss <= _LOCALIZED_E2_MAX_RSS_BYTES,
        )
        conditions.setdefault(
            "cuda_allocated_within_cap",
            resources.get("cuda_initialized") is True
            and type(allocated) is int
            and 0 <= allocated <= _LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES,
        )
        promoted = (
            status == "localized_e2_promoted_diagnostic"
            and bool(conditions)
            and all(conditions.values())
        )
        reported_status = (
            "localized_e2_promotion_rejected"
            if status == "localized_e2_promoted_diagnostic" and not promoted
            else status
        )
        return {
            "candidate_mode": "localized_e2",
            "status": reported_status,
            "reason": reason,
            "diagnostic_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "adapter_called_once": adapter_call_count == 1,
            "adapter_call_count": adapter_call_count,
            "adapter": (
                None if candidate is None else _localized_e2_candidate_summary(candidate)
            ),
            "focused_encoded_row": focused_encoded_row,
            "initial_parent_semantic_digest": initial_parent_digest,
            "terminal_parent_semantic_digest": terminal_parent_digest,
            "initial_operator_row_tag_digest": initial_tag_digest,
            "terminal_operator_row_tag_digest": terminal_tag_digest,
            "live_parent_unchanged": unchanged,
            "diagnostic_cut": {
                "attempted": cut_attempted,
                "private_only": True,
                "structurally_validated": cut_structurally_validated,
                "live_parent_mutated": not unchanged,
            },
            "lp_attempted": lp_attempted,
            "lp_tightness": dict(
                lp_tightness
                or {"status": "not_run", "proof_authority": False, "verdict_authority": False}
            ),
            "promotion_gate": {
                "promoted": promoted,
                "minimum_relative_drop": _LOCALIZED_E2_MIN_RELATIVE_DROP,
                "max_peak_rss_bytes": _LOCALIZED_E2_MAX_RSS_BYTES,
                "max_cuda_allocated_bytes": _LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES,
                "conditions": conditions,
            },
            "controlled_build_only_gate": {"passed": promoted},
            "production_gate": {
                "authorized": False,
                "no_owner_capability": True,
            },
            "resource_usage": resources,
            "timings": dict(timings),
        }

    if time.monotonic() >= candidate_deadline:
        return finish("stop_loss_deadline", "deadline_expired_before_adapter")
    try:
        candidate_started = time.monotonic()
        adapter_call_count += 1
        candidate = run_candidate(
            source_build,
            focused_rivals,
            selection,
            deadline=candidate_deadline,
            enabled=True,
            **candidate_options,
        )
        timings["localized_adapter_seconds"] = float(
            time.monotonic() - candidate_started
        )
    except Exception as exc:
        text = f"{type(exc).__name__}:{exc}".lower()
        status = (
            "stop_loss_deadline"
            if "deadline" in text or "timeout" in text
            else "stop_loss_resource"
            if isinstance(exc, MemoryError)
            or any(word in text for word in ("resource", "cap", "nonzero", "buffer"))
            else "stop_loss_adapter_error"
        )
        return finish(status, f"adapter_rejected:{type(exc).__name__}")

    if not validate_adapter_checksum(candidate):
        return finish("stop_loss_receipt_rejected", "adapter_checksum_rejected")
    literals = getattr(candidate, "literals", ())
    if type(literals) is not tuple or len(literals) < 2:
        return finish(
            "stop_loss_insufficient_literals",
            "operator_selection_has_zero_or_one_literal",
        )
    if len(literals) != 2 or getattr(candidate, "edge_accepted", False) is not True:
        return finish("stop_loss_no_exact_edge", "no_exact_localized_pair_edge")
    if time.monotonic() >= candidate_deadline:
        return finish("stop_loss_deadline", "deadline_expired_after_adapter")

    resources_before_cut = dict(resource_peaks())
    rss_before = resources_before_cut.get("peak_rss_bytes")
    allocated_before = resources_before_cut.get("cuda_peak_allocated_bytes")
    if (
        type(rss_before) is not int
        or rss_before < 0
        or rss_before > _LOCALIZED_E2_MAX_RSS_BYTES
        or resources_before_cut.get("cuda_initialized") is not True
        or type(allocated_before) is not int
        or allocated_before < 0
        or allocated_before > _LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES
    ):
        return finish(
            "stop_loss_resource",
            "resource_peak_exceeded_before_cut",
            resource_snapshot=resources_before_cut,
        )

    if not validate_exact_candidate(
        source_build,
        candidate,
        selection,
        deadline=candidate_deadline,
        candidate_kwargs=candidate_options,
    ):
        return finish(
            "stop_loss_receipt_rejected",
            "independent_exact_candidate_replay_rejected",
        )
    if (
        getattr(candidate, "parent_semantic_digest", None)
        != initial_parent_digest
        or getattr(candidate, "terminal_parent_semantic_digest", None)
        != initial_parent_digest
        or getattr(candidate, "operator_row_tag_digest", None)
        != initial_tag_digest
        or getattr(candidate, "terminal_operator_row_tag_digest", None)
        != initial_tag_digest
    ):
        return finish(
            "stop_loss_receipt_rejected",
            "adapter_live_parent_or_tag_binding_rejected",
        )

    try:
        cut_started = time.monotonic()
        cut_attempted = True
        cut_hz = copy_pair_cut(source_build.hz, candidate.literals)
        timings["private_pair_cut_seconds"] = float(time.monotonic() - cut_started)
        if cut_hz is source_build.hz:
            raise PhaseCliqueBuildProbeError("pair cut did not create a private HZ")
        cut_parent_digest, cut_parent_tags = live_seals(source_build.hz)
        if (
            cut_parent_digest != initial_parent_digest
            or cut_parent_tags != initial_tag_digest
        ):
            raise PhaseCliqueBuildProbeError("pair cut mutated the live parent")
        if not validate_private_cut(
            source_build.hz, cut_hz, candidate.literals
        ):
            raise PhaseCliqueBuildProbeError(
                "private pair cut structural validation failed"
            )
        validated_parent_digest, validated_parent_tags = live_seals(
            source_build.hz
        )
        if (
            validated_parent_digest != initial_parent_digest
            or validated_parent_tags != initial_tag_digest
        ):
            raise PhaseCliqueBuildProbeError(
                "pair cut validator mutated the live parent"
            )
        cut_structurally_validated = True
    except Exception as exc:
        return finish(
            "stop_loss_private_cut_rejected",
            f"private_pair_cut_rejected:{type(exc).__name__}",
        )

    resources_after_cut = dict(resource_peaks())
    rss_after_cut = resources_after_cut.get("peak_rss_bytes")
    allocated_after_cut = resources_after_cut.get("cuda_peak_allocated_bytes")
    if (
        type(rss_after_cut) is not int
        or rss_after_cut < 0
        or rss_after_cut > _LOCALIZED_E2_MAX_RSS_BYTES
        or resources_after_cut.get("cuda_initialized") is not True
        or type(allocated_after_cut) is not int
        or allocated_after_cut < 0
        or allocated_after_cut > _LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES
    ):
        return finish(
            "stop_loss_resource",
            "resource_peak_exceeded_after_private_cut",
            resource_snapshot=resources_after_cut,
        )

    rows = np.asarray(objective_rows, dtype=np.float64)
    limits = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if (
        type(focused_encoded_row) is not int
        or rows.ndim != 2
        or not 0 <= focused_encoded_row < rows.shape[0]
        or limits.shape != (rows.shape[0],)
    ):
        return finish("stop_loss_property_frame", "focused_objective_row_invalid")

    lp_attempted = True
    try:
        before_deadline = min(deadline, time.monotonic() + 5.0)
        before = dict(
            lp_upper(
                source_build.hz,
                rows[focused_encoded_row],
                float(limits[focused_encoded_row]),
                deadline=before_deadline,
            )
        )
        before_completed = time.monotonic()
        before_completed_before_deadline = before_completed < before_deadline
        before_upper = before.get("independently_certified_upper")
        before_qualified = (
            before.get("status") == "certified_diagnostic_upper"
            and type(before_upper) is float
            and math.isfinite(before_upper)
            and before_upper > 0.0
            and before_completed_before_deadline
        )
        if not before_qualified:
            return finish(
                "localized_e2_promotion_rejected",
                "before_upper_not_certified_finite_positive_in_time",
                lp_tightness={
                    "status": "before_inconclusive",
                    "proof_authority": False,
                    "verdict_authority": False,
                    "focused_encoded_row": focused_encoded_row,
                    "before_deadline_monotonic": float(before_deadline),
                    "before_completed_before_deadline": (
                        before_completed_before_deadline
                    ),
                    "before": before,
                    "after": {"status": "not_run"},
                    "drop": None,
                    "relative_drop": None,
                },
                gate_conditions={
                    "edge_exact": True,
                    "before_upper_certified": False,
                    "after_upper_certified": False,
                    "before_positive": False,
                    "drop_positive": False,
                    "relative_drop_at_least_5pct": False,
                },
            )
        after_deadline = min(deadline, time.monotonic() + 5.0)
        after = dict(
            lp_upper(
                cut_hz,
                rows[focused_encoded_row],
                float(limits[focused_encoded_row]),
                deadline=after_deadline,
            )
        )
        after_completed = time.monotonic()
        after_completed_before_deadline = after_completed < after_deadline
    except Exception as exc:
        return finish(
            "stop_loss_lp_inconclusive",
            f"diagnostic_lp_error:{type(exc).__name__}",
        )
    before_upper = before.get("independently_certified_upper")
    after_upper = after.get("independently_certified_upper")
    both_certified = (
        before.get("status") == "certified_diagnostic_upper"
        and after.get("status") == "certified_diagnostic_upper"
        and type(before_upper) is float
        and type(after_upper) is float
        and math.isfinite(before_upper)
        and math.isfinite(after_upper)
        and before_completed_before_deadline
        and after_completed_before_deadline
    )
    drop = (
        float(before_upper - after_upper) if both_certified else None
    )
    relative_drop = (
        float(drop / max(abs(before_upper), 1.0e-12))
        if both_certified and drop is not None
        else None
    )
    tightness = {
        "status": "compared" if both_certified else "inconclusive",
        "proof_authority": False,
        "verdict_authority": False,
        "focused_encoded_row": focused_encoded_row,
        "before_deadline_monotonic": float(before_deadline),
        "after_deadline_monotonic": float(after_deadline),
        "before_completed_before_deadline": before_completed_before_deadline,
        "after_completed_before_deadline": after_completed_before_deadline,
        "before": before,
        "after": after,
        "drop": drop,
        "relative_drop": relative_drop,
    }
    conditions = {
        "edge_exact": True,
        "before_upper_certified": bool(both_certified),
        "after_upper_certified": bool(both_certified),
        "before_completed_before_deadline": before_completed_before_deadline,
        "after_completed_before_deadline": after_completed_before_deadline,
        "before_positive": bool(both_certified and before_upper > 0.0),
        "drop_positive": bool(both_certified and drop is not None and drop > 0.0),
        "relative_drop_at_least_5pct": bool(
            both_certified
            and relative_drop is not None
            and relative_drop >= _LOCALIZED_E2_MIN_RELATIVE_DROP
        ),
    }
    status = (
        "localized_e2_promoted_diagnostic"
        if all(conditions.values())
        else "localized_e2_promotion_rejected"
    )
    return finish(
        status,
        "fixed_promotion_gate_evaluated",
        lp_tightness=tightness,
        gate_conditions=conditions,
    )


def _localized_e2_preflight_stop(
    source_build: Any,
    *,
    status: str,
    reason: str,
    stage: str,
    started: float,
    timings: Mapping[str, float],
    resource_peaks: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    parent_digest, tag_digest = _localized_e2_live_seals(source_build.hz)
    resources = dict(resource_peaks())
    total_seconds = float(max(0.0, time.monotonic() - started))
    return {
        "candidate_mode": "localized_e2",
        "status": status,
        "reason": reason[:300],
        "failed_stage": stage,
        "diagnostic_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "adapter_called_once": False,
        "adapter_call_count": 0,
        "adapter": None,
        "initial_parent_semantic_digest": parent_digest,
        "terminal_parent_semantic_digest": parent_digest,
        "initial_operator_row_tag_digest": tag_digest,
        "terminal_operator_row_tag_digest": tag_digest,
        "live_parent_unchanged": True,
        "diagnostic_cut": {
            "attempted": False,
            "private_only": True,
            "structurally_validated": False,
            "live_parent_mutated": False,
        },
        "lp_attempted": False,
        "lp_tightness": {
            "status": "not_run",
            "proof_authority": False,
            "verdict_authority": False,
        },
        "promotion_gate": {
            "promoted": False,
            "minimum_relative_drop": _LOCALIZED_E2_MIN_RELATIVE_DROP,
            "max_peak_rss_bytes": _LOCALIZED_E2_MAX_RSS_BYTES,
            "max_cuda_allocated_bytes": _LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES,
            "conditions": {
                "preflight_complete": False,
                "total_under_60_seconds": total_seconds < 60.0,
            },
        },
        "controlled_build_only_gate": {"passed": False},
        "production_gate": {
            "authorized": False,
            "no_owner_capability": True,
        },
        "raw_audit": None,
        "resource_usage": resources,
        "timings": {
            **{str(key): float(value) for key, value in timings.items()},
            "total_seconds_at_gate": total_seconds,
        },
    }


def _run_localized_e2_pipeline(
    source_build: Any,
    *,
    vnnlib_path: Any,
    expected_vnnlib_sha256: str,
    live_assert_params: Any,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    residual_selector_receipt: Mapping[str, Any],
    residual_selector_property_sha256: str,
    objective_rows: np.ndarray,
    thresholds: np.ndarray,
    deadline: float,
    phase_time_limit: float,
    overall_started: float,
    torch_module: Any,
) -> dict[str, Any]:
    """Reuse the K4 raw/focus/literal audit, then run only localized E2."""

    from act.back_end.hybridz_tf.adaptive_phase_forest import (
        sparse_hz_semantic_digest,
    )
    from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
        derive_operator_exact_relu_property_phase_literals,
        verify_operator_exact_relu_property_phase_selection,
    )
    from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
        _check_deadline,
        _exact_interval_upper_violations,
        _interval_frame_digest,
        _normalize_caps,
        _snapshot_b1_bounds,
    )
    from act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge import (
        issue_raw_rival_exact_hardness_receipt,
        select_raw_focused_rivals,
        verify_raw_focused_rival_selection,
        verify_raw_rival_exact_hardness_receipt,
    )
    from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
        consume_raw_vnnlib_top1_candidate,
        issue_raw_vnnlib_top1_candidate,
        validate_consumed_raw_vnnlib_rival_batch,
    )

    started = time.monotonic()
    timings: dict[str, float] = {}
    stage = "localized_input_validation"
    resource_reader = lambda: _capture_resource_peaks(torch_module)
    try:
        caps = _normalize_caps(None)
        if (
            not _valid_sha256(expected_vnnlib_sha256)
            or type(residual_selector_receipt) is not dict
            or not _valid_sha256(residual_selector_property_sha256)
        ):
            raise PhaseCliqueBuildProbeError("raw_or_selector_binding_invalid")
        joint_focus = residual_selector_receipt.get("joint_focus_rival_id")
        if type(joint_focus) is not int:
            raise PhaseCliqueBuildProbeError("residual_joint_focus_row_invalid")
        now = time.monotonic()
        remaining = float(deadline - now)
        if remaining <= 10.0:
            raise TimeoutError("ten_second_lp_reserve_unavailable")
        candidate_deadline = min(
            now + float(phase_time_limit), deadline - 10.0
        )
        if candidate_deadline <= now:
            raise TimeoutError("localized_candidate_budget_unavailable")
        source_digest = sparse_hz_semantic_digest(source_build.hz)

        stage = "raw_vnnlib_batch_issue_consume"
        stage_started = time.monotonic()
        raw_candidate = issue_raw_vnnlib_top1_candidate(
            vnnlib_path,
            expected_vnnlib_sha256=expected_vnnlib_sha256,
            live_assert_params=live_assert_params,
            deadline=candidate_deadline,
        )
        batch = consume_raw_vnnlib_top1_candidate(
            raw_candidate,
            live_assert_params=live_assert_params,
            deadline=candidate_deadline,
        )
        if not validate_consumed_raw_vnnlib_rival_batch(batch):
            raise PhaseCliqueBuildProbeError("raw_consumed_batch_invalid")
        timings["raw_batch_seconds"] = float(time.monotonic() - stage_started)
        if not 1 <= len(batch.rivals) <= caps.max_full_rivals:
            raise PhaseCliqueBuildProbeError("raw_full_rival_count_out_of_cap")

        stage = "complete_b1_interval_hardness"
        stage_started = time.monotonic()
        lower, upper = _snapshot_b1_bounds(
            output_lower,
            output_upper,
            output_width=int(source_build.hz.n_out),
        )
        exact_hardness = _exact_interval_upper_violations(
            batch.rivals,
            lower.reshape(-1),
            upper.reshape(-1),
            deadline=candidate_deadline,
        )
        interval_digest = _interval_frame_digest(
            build_digest=source_digest,
            batch_sha256=batch.batch_sha256,
            live_assert_sha256=batch.live_assert_sha256,
            property_digest=residual_selector_property_sha256,
            lower=lower,
            upper=upper,
        )
        hardness = issue_raw_rival_exact_hardness_receipt(
            batch,
            exact_hardness,
            live_interval_bounds_sha256=interval_digest,
            deadline=candidate_deadline,
            max_rivals=caps.max_full_rivals,
            max_focus=1,
            max_exact_bits=caps.max_raw_exact_bits,
            max_work_items=caps.max_raw_work_items,
        )
        timings["complete_b1_hardness_seconds"] = float(
            time.monotonic() - stage_started
        )

        stage = "residual_joint_focus_and_double_verify"
        stage_started = time.monotonic()
        focused = select_raw_focused_rivals(
            batch,
            hardness,
            focus_count=1,
            explicit_encoded_focus_row=joint_focus,
            residual_selector_receipt=residual_selector_receipt,
            residual_selector_property_sha256=residual_selector_property_sha256,
            expected_exact_upper_violations=exact_hardness,
            expected_live_interval_bounds_sha256=interval_digest,
            deadline=candidate_deadline,
            max_rivals=caps.max_full_rivals,
            max_focus=1,
            max_exact_bits=caps.max_raw_exact_bits,
            max_work_items=caps.max_raw_work_items,
        )
        hardness_verified = verify_raw_rival_exact_hardness_receipt(
            batch,
            hardness,
            expected_exact_upper_violations=exact_hardness,
            expected_live_interval_bounds_sha256=interval_digest,
            deadline=candidate_deadline,
            max_rivals=caps.max_full_rivals,
            max_focus=1,
            max_exact_bits=caps.max_raw_exact_bits,
            max_work_items=caps.max_raw_work_items,
        )
        focus_verified = verify_raw_focused_rival_selection(
            batch,
            hardness,
            focused,
            expected_focus_count=1,
            expected_exact_upper_violations=exact_hardness,
            expected_live_interval_bounds_sha256=interval_digest,
            deadline=candidate_deadline,
            max_rivals=caps.max_full_rivals,
            max_focus=1,
            max_exact_bits=caps.max_raw_exact_bits,
            max_work_items=caps.max_raw_work_items,
        )
        if hardness_verified is not True or focus_verified is not True:
            raise PhaseCliqueBuildProbeError("focused_receipt_double_verify_rejected")
        timings["focus_and_double_verify_seconds"] = float(
            time.monotonic() - stage_started
        )

        stage = "operator_literal_derive_and_verify"
        stage_started = time.monotonic()
        remaining_selection = candidate_deadline - time.monotonic()
        if remaining_selection <= 0.006:
            raise TimeoutError("deadline_before_literal_audit")
        selection_timeout = min(
            caps.max_selection_seconds, remaining_selection / 6.0
        )
        selection = derive_operator_exact_relu_property_phase_literals(
            source_build,
            focused.rivals,
            max_rivals=1,
            max_binaries=caps.max_binaries,
            max_work_items=caps.max_selection_work_items,
            timeout_seconds=selection_timeout,
        )
        if not verify_operator_exact_relu_property_phase_selection(
            source_build,
            focused.rivals,
            selection,
            max_rivals=1,
            max_binaries=caps.max_binaries,
            max_work_items=caps.max_selection_work_items,
            timeout_seconds=selection_timeout,
        ):
            raise PhaseCliqueBuildProbeError("operator_literal_selection_rejected")
        _check_deadline(candidate_deadline, stage="localized_literal_selection")
        timings["literal_derive_verify_seconds"] = float(
            time.monotonic() - stage_started
        )

        stage = "localized_e2_transaction"
        transaction = _run_localized_e2_transaction(
            source_build,
            focused.rivals,
            selection,
            focused_encoded_row=joint_focus,
            objective_rows=objective_rows,
            thresholds=thresholds,
            deadline=deadline,
            candidate_deadline=candidate_deadline,
            overall_started=overall_started,
            candidate_kwargs={
                "selection_max_rivals": 1,
                "selection_max_binaries": caps.max_binaries,
                "selection_max_work_items": caps.max_selection_work_items,
                "selection_timeout_seconds": selection_timeout,
                "max_parent_variables": caps.max_parent_variables,
                "max_parent_rows": caps.max_parent_rows,
                "max_parent_nonzeros": caps.max_parent_nonzeros,
                "max_parent_buffer_items": caps.max_parent_buffer_items,
                "max_top_literals": 2,
                "max_total_pairs": 1,
                "max_source_terms": caps.max_source_terms,
                "max_multiplier_bits": caps.max_multiplier_bits,
                "max_exact_bits": caps.max_exact_bits,
                "max_exact_nonzeros": caps.max_exact_nonzeros,
                "localized_row_tiers": (64, 256, 1024, 4096),
                "localized_max_selected_nnz": 1_000_000,
                "localized_max_source_terms": caps.max_source_terms,
            },
            resource_peaks=resource_reader,
        )
        transaction["raw_audit"] = {
            "full_rival_count": len(batch.rivals),
            "full_batch_sha256": batch.batch_sha256,
            "full_live_assert_sha256": batch.live_assert_sha256,
            "full_property_digest": hardness.full_property_digest,
            "interval_frame_sha256": interval_digest,
            "hardness_vector_digest": hardness.vector_digest,
            "focused_subset_digest": focused.focused_subset_digest,
            "focused_encoded_row": joint_focus,
            "hardness_receipt_verified": True,
            "focused_receipt_verified": True,
            "selection_digest": selection.selection_digest,
            "focused_property_digest": selection.property_digest,
            "residual_selector_property_sha256": (
                residual_selector_property_sha256
            ),
        }
        transaction["budget"] = {
            "global_deadline_monotonic": float(deadline),
            "candidate_deadline_monotonic": float(candidate_deadline),
            "lp_reserve_seconds": 10.0,
            "per_lp_deadline_seconds": 5.0,
        }
        transaction["timings"] = {
            **timings,
            **dict(transaction.get("timings", {})),
        }
        return transaction
    except Exception as exc:
        text = f"{type(exc).__name__}:{exc}".lower()
        timeout = (
            isinstance(exc, TimeoutError)
            or bool(getattr(exc, "timeout", False))
            or "deadline" in text
            or "timeout" in text
        )
        resource_failure = isinstance(exc, MemoryError) or any(
            word in text for word in ("resource", "out_of_cap", "nonzero", "buffer")
        )
        status = (
            "stop_loss_deadline"
            if timeout
            else "stop_loss_resource"
            if resource_failure
            else "stop_loss_preflight_rejected"
        )
        return _localized_e2_preflight_stop(
            source_build,
            status=status,
            reason=f"{type(exc).__name__}:{str(exc)[:240]}",
            stage=stage,
            started=overall_started,
            timings=timings,
            resource_peaks=resource_reader,
        )


def _finalize_localized_e2_integrity(body: dict[str, Any]) -> None:
    """Bind final input/resource/time integrity into the promotion decision."""

    if body.get("candidate_mode") != "localized_e2":
        return
    transaction = body.get("localized_e2")
    resources = body.get("resource_usage")
    if transaction is None:
        return
    if not isinstance(transaction, dict) or not isinstance(resources, dict):
        body["phase_status"] = "error"
        body["failed_stage"] = "localized_terminal_integrity"
        body["error_type"] = "PhaseCliqueBuildProbeError"
        body["error"] = "localized terminal receipt is malformed"
        return
    gate = transaction.get("promotion_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("conditions"), dict):
        body["phase_status"] = "error"
        body["failed_stage"] = "localized_terminal_integrity"
        body["error_type"] = "PhaseCliqueBuildProbeError"
        body["error"] = "localized promotion gate is malformed"
        return
    conditions = gate["conditions"]
    rss = resources.get("peak_rss_bytes")
    allocated = resources.get("cuda_peak_allocated_bytes")
    conditions["inputs_unchanged"] = body.get("inputs_unchanged") is True
    conditions["final_peak_rss_within_cap"] = (
        type(rss) is int and 0 <= rss <= _LOCALIZED_E2_MAX_RSS_BYTES
    )
    conditions["final_cuda_allocated_within_cap"] = (
        resources.get("cuda_initialized") is True
        and type(allocated) is int
        and 0 <= allocated <= _LOCALIZED_E2_MAX_CUDA_ALLOCATED_BYTES
    )
    total = body.get("timings", {}).get("total_seconds")
    conditions["final_total_under_60_seconds"] = (
        type(total) is float and 0.0 <= total < 60.0
    )
    transaction["resource_usage"] = dict(resources)
    promoted = bool(gate.get("promoted")) and all(
        value is True for value in conditions.values()
    )
    gate["promoted"] = promoted
    if not promoted and transaction.get("status") == "localized_e2_promoted_diagnostic":
        transaction["status"] = "localized_e2_promotion_rejected"
        transaction["reason"] = "final_input_resource_or_time_integrity_rejected"
    controlled = transaction.get("controlled_build_only_gate")
    if isinstance(controlled, dict):
        controlled["passed"] = promoted
    body["phase_status"] = transaction.get("status")
    body["fallback_reason"] = transaction.get("reason")
    wall_timeout = body.get("wall_timeout_seconds")
    if type(wall_timeout) in {int, float} and type(total) is float:
        body["completed_before_deadline"] = bool(total < float(wall_timeout))


def _builtin_receipt_mapping(value: Any) -> Optional[dict[str, Any]]:
    """Detach a bounded JSON receipt mapping from a live result object."""

    if not isinstance(value, Mapping):
        return None

    def thaw(item: Any) -> Any:
        if item is None or type(item) in {bool, int, float, str}:
            return item
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise TypeError("receipt mapping key is not a string")
            return {key: thaw(nested) for key, nested in item.items()}
        if type(item) in {tuple, list}:
            return [thaw(nested) for nested in item]
        raise TypeError(
            f"receipt value is not JSON-safe: {type(item).__name__}"
        )

    try:
        copied = json.loads(_canonical_json(thaw(value)))
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return None
    return copied if type(copied) is dict else None


def _materializer_route_summary(
    pipeline_receipt: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    """Retain only the exact low-peak handoff contract needed by C88."""

    nested = _builtin_receipt_mapping(
        pipeline_receipt.get("materialization_receipt")
    )
    if nested is None:
        return None
    fields = (
        "schema",
        "receipt_sha256",
        "public_core_source",
        "parent_prefix_core",
        "parent_prefix_readonly",
        "parent_prefix_aliases_public_cut",
        "public_core_readonly",
        "materializer_full_core_copy_count",
        "private_solver_core",
        "public_private_core_no_alias",
        "producer_nonempty_seal_verified",
        "one_use_snapshot_consumed",
        "solver_handoff_one_use",
        "solver_handoff_owner_bound",
        "solver_handoff_pid_bound",
        "solver_handoff_private_core_readonly",
    )
    return {name: nested.get(name) for name in fields}


def _rss_sample() -> dict[str, Optional[int]]:
    """Capture current and monotone peak CPU RSS without verdict authority."""

    return {
        "current_rss_bytes": _current_rss_bytes(),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * 1024,
    }


def _pcoh_k2_source_build_preflight(
    shape: Mapping[str, Any],
    *,
    build_seconds: Any,
    input_sha256: Mapping[str, Any],
    implementation_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the fixed source-build stop-loss before entering PCOH."""

    expected_shape_fields = {
        "output_dimension",
        "continuous_columns",
        "binary_columns",
        "upper_rows",
        "equality_rows",
        "constraint_nonzeros",
        "generator_nonzeros",
    }
    shape_exact = type(shape) is dict and set(shape) == expected_shape_fields
    integers = bool(
        shape_exact
        and all(type(value) is int and value >= 0 for value in shape.values())
    )
    rows_plus_outputs = (
        shape["upper_rows"]
        + shape["equality_rows"]
        + shape["output_dimension"]
        if integers
        else None
    )
    conditions = {
        "build_seconds_at_most_27": bool(
            type(build_seconds) is float
            and math.isfinite(build_seconds)
            and 0.0
            <= build_seconds
            <= _PCOH_K2_MAX_SOURCE_BUILD_SECONDS
        ),
        "shape_fields_exact": shape_exact,
        "shape_values_nonnegative_builtin_int": integers,
        "output_dimension_exact_100": bool(
            integers
            and shape["output_dimension"] == _PCOH_K2_SOURCE_OUTPUTS
        ),
        "binary_columns_exact_4": bool(
            integers
            and shape["binary_columns"] == _PCOH_K2_SOURCE_BINARIES
        ),
        "continuous_columns_at_most_60000": bool(
            integers
            and shape["continuous_columns"]
            <= _PCOH_K2_MAX_SOURCE_CONTINUOUS
        ),
        "rows_plus_outputs_at_most_105000": bool(
            type(rows_plus_outputs) is int
            and rows_plus_outputs
            <= _PCOH_K2_MAX_SOURCE_ROWS_PLUS_OUTPUTS
        ),
        "constraint_nonzeros_at_most_11m": bool(
            integers
            and shape["constraint_nonzeros"]
            <= _PCOH_K2_MAX_SOURCE_CONSTRAINT_NONZEROS
        ),
        "generator_nonzeros_at_most_20k": bool(
            integers
            and shape["generator_nonzeros"]
            <= _PCOH_K2_MAX_SOURCE_GENERATOR_NONZEROS
        ),
    }
    failed = [name for name, passed in conditions.items() if passed is not True]
    return _checksummed({
        "schema": "act.hybridz_pcoh_k2_source_build_preflight.v1",
        "status": "passed" if not failed else "stop_loss",
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": True,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "input_sha256": dict(input_sha256),
        "implementation_sha256": dict(implementation_sha256),
        "source_shape": dict(shape) if type(shape) is dict else None,
        "source_build_seconds": build_seconds,
        "rows_plus_outputs": rows_plus_outputs,
        "conditions": conditions,
        "failed_conditions": failed,
    })


def _pcoh_k2_source_build_preflight_valid(value: Any) -> bool:
    if type(value) is not dict:
        return False
    try:
        checksum_valid = _local_receipt_checksum_valid(
            value, schema="act.hybridz_pcoh_k2_source_build_preflight.v1"
        )
    except (TypeError, ValueError, OverflowError):
        return False
    expected = {
        "schema",
        "status",
        "diagnostic_only",
        "candidate_only",
        "build_only",
        "instance_count",
        "proof_authority",
        "verdict_authority",
        "ground_truth_loaded",
        "reference_label_used",
        "input_sha256",
        "implementation_sha256",
        "source_shape",
        "source_build_seconds",
        "rows_plus_outputs",
        "conditions",
        "failed_conditions",
        "receipt_sha256",
    }
    structural = bool(
        checksum_valid
        and set(value) == expected
        and value.get("status") in {"passed", "stop_loss"}
        and value.get("diagnostic_only") is True
        and value.get("candidate_only") is True
        and value.get("build_only") is True
        and value.get("instance_count") == 1
        and value.get("proof_authority") is False
        and value.get("verdict_authority") is False
        and value.get("ground_truth_loaded") is False
        and value.get("reference_label_used") is False
        and type(value.get("conditions")) is dict
        and type(value.get("failed_conditions")) is list
        and value["failed_conditions"]
        == [
            name
            for name, passed in value["conditions"].items()
            if passed is not True
        ]
        and value["status"]
        == ("passed" if not value["failed_conditions"] else "stop_loss")
    )
    if not structural:
        return False
    input_sha256 = value.get("input_sha256")
    implementation_sha256 = value.get("implementation_sha256")
    if (
        type(input_sha256) is not dict
        or input_sha256
        != {
            "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
            "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
            "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
        }
        or type(implementation_sha256) is not dict
        or set(implementation_sha256) != set(_IMPLEMENTATION_RELATIVE_PATHS)
        or any(
            not _valid_sha256(digest)
            for digest in implementation_sha256.values()
        )
    ):
        return False
    try:
        recomputed = _pcoh_k2_source_build_preflight(
            value.get("source_shape"),
            build_seconds=value.get("source_build_seconds"),
            input_sha256=input_sha256,
            implementation_sha256=implementation_sha256,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return secrets.compare_digest(
        value["receipt_sha256"], recomputed["receipt_sha256"]
    ) and value == recomputed


def _pcoh_k2_fraction_pair(value: Fraction) -> list[int]:
    if type(value) is not Fraction:
        raise PhaseCliqueBuildProbeError("pcoh tightness value is not exact")
    return [int(value.numerator), int(value.denominator)]


def _pcoh_k2_strict_fraction_pair(value: Any, *, name: str) -> Fraction:
    if (
        type(value) not in {tuple, list}
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or value[1] <= 0
    ):
        raise PhaseCliqueBuildProbeError(
            f"pcoh tightness {name} fraction pair is malformed"
        )
    exact = Fraction(value[0], value[1])
    if [exact.numerator, exact.denominator] != list(value):
        raise PhaseCliqueBuildProbeError(
            f"pcoh tightness {name} fraction pair is noncanonical"
        )
    return exact


def _pcoh_k2_strict_hex_fraction(value: Any, *, name: str) -> Fraction:
    if type(value) is not str or len(value) > 32:
        raise PhaseCliqueBuildProbeError(
            f"pcoh tightness {name} hex is malformed"
        )
    try:
        stored = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise PhaseCliqueBuildProbeError(
            f"pcoh tightness {name} hex is malformed"
        ) from exc
    if not math.isfinite(stored) or stored.hex() != value:
        raise PhaseCliqueBuildProbeError(
            f"pcoh tightness {name} hex is noncanonical or nonfinite"
        )
    return Fraction.from_float(stored)


def _pcoh_k2_tightness_gate(
    summary: Mapping[str, Any],
    *,
    expected_summary_sha256: str,
) -> dict[str, Any]:
    """Compute the preregistered continuation gates over exact Fractions."""

    if (
        type(summary) is not dict
        or not _valid_sha256(expected_summary_sha256)
        or summary.get("summary_sha256") != expected_summary_sha256
    ):
        raise PhaseCliqueBuildProbeError(
            "pcoh tightness summary anchor is malformed or mismatched"
        )
    global_upper = _pcoh_k2_strict_hex_fraction(
        summary.get("global_cube_upper_hex"), name="global_cube_upper"
    )
    final_upper = _pcoh_k2_strict_hex_fraction(
        summary.get("final_structural_upper_hex"),
        name="final_structural_upper",
    )
    ideal_upper = _pcoh_k2_strict_hex_fraction(
        summary.get("ideal_union_upper_hex"), name="ideal_union_upper"
    )
    rounding_tax = _pcoh_k2_strict_fraction_pair(
        summary.get("rounding_tax_exact"), name="rounding_tax"
    )
    if rounding_tax < 0:
        raise PhaseCliqueBuildProbeError(
            "pcoh tightness rounding tax is negative"
        )
    delta = global_upper - final_upper
    if delta < 0:
        raise PhaseCliqueBuildProbeError(
            "pcoh final structural upper exceeds global cube upper"
        )
    unit_scale = max(Fraction(1), abs(global_upper))
    continuation_scale = max(
        Fraction(1, 4), Fraction(1, 200) * unit_scale
    )
    strong_scale = max(Fraction(1), Fraction(1, 50) * unit_scale)
    rounding_threshold = 8 * rounding_tax
    global_positive = global_upper > 0
    continuation_scale_met = delta >= continuation_scale
    strong_scale_met = delta >= strong_scale
    rounding_dominance_met = delta >= rounding_threshold
    return _checksummed({
        "schema": _PCOH_K2_TIGHTNESS_GATE_SCHEMA,
        "status": "evaluated",
        "diagnostic_only": True,
        "candidate_only": True,
        "full_parent_lp_called": False,
        "full_parent_lp_solver_called": False,
        "proof_authority": False,
        "verdict_authority": False,
        "materialized_tightness_summary_sha256": expected_summary_sha256,
        "global_cube_upper_fraction": _pcoh_k2_fraction_pair(global_upper),
        "final_structural_upper_fraction": _pcoh_k2_fraction_pair(final_upper),
        "ideal_union_upper_fraction": _pcoh_k2_fraction_pair(ideal_upper),
        "rounding_tax_fraction": _pcoh_k2_fraction_pair(rounding_tax),
        "delta_fraction": _pcoh_k2_fraction_pair(delta),
        "continuation_scale_threshold_fraction": _pcoh_k2_fraction_pair(
            continuation_scale
        ),
        "strong_scale_threshold_fraction": _pcoh_k2_fraction_pair(
            strong_scale
        ),
        "rounding_tax_threshold_fraction": _pcoh_k2_fraction_pair(
            rounding_threshold
        ),
        "delta_nonnegative": True,
        "global_positive": global_positive,
        "continuation_scale_met": continuation_scale_met,
        "strong_scale_met": strong_scale_met,
        "rounding_tax_dominance_met": rounding_dominance_met,
        "continuation_candidate": bool(
            global_positive
            and continuation_scale_met
            and rounding_dominance_met
        ),
        "strong_candidate": bool(
            global_positive and strong_scale_met and rounding_dominance_met
        ),
        "cube_already_sufficient": global_upper <= 0,
        "zero_crossing": final_upper < 0,
    })


def _pcoh_k2_tightness_gate_valid(
    value: Any,
    summary: Any,
    *,
    expected_summary_sha256: Any,
) -> bool:
    if type(value) is not dict or set(value) != _PCOH_K2_TIGHTNESS_GATE_FIELDS:
        return False
    try:
        if not _local_receipt_checksum_valid(
            value, schema=_PCOH_K2_TIGHTNESS_GATE_SCHEMA
        ):
            return False
        expected = _pcoh_k2_tightness_gate(
            summary,
            expected_summary_sha256=expected_summary_sha256,
        )
    except (KeyError, TypeError, ValueError, OverflowError, PhaseCliqueBuildProbeError):
        return False
    return bool(
        secrets.compare_digest(
            value["receipt_sha256"], expected["receipt_sha256"]
        )
        and value == expected
    )


def _register_pcoh_k2_trusted_transaction(
    transaction: dict[str, Any],
    *,
    trusted_summary_sha256: str,
) -> None:
    """Bind one live transaction identity to its separately captured anchor."""

    if type(transaction) is not dict:
        raise PhaseCliqueBuildProbeError(
            "pcoh trusted transaction registration rejected"
        )
    receipt_sha256 = transaction.get("receipt_sha256")
    if (
        not _valid_sha256(receipt_sha256)
        or not _valid_sha256(trusted_summary_sha256)
        or transaction.get("materialized_tightness_summary_sha256")
        != trusted_summary_sha256
    ):
        raise PhaseCliqueBuildProbeError(
            "pcoh trusted transaction registration rejected"
        )
    key = id(transaction)
    record = _PCOHK2TrustedTransactionAnchor(
        transaction=transaction,
        process_id=os.getpid(),
        transaction_receipt_sha256=receipt_sha256,
        materialized_tightness_summary_sha256=trusted_summary_sha256,
    )
    with _PCOH_K2_TRUSTED_TRANSACTION_LOCK:
        if key in _PCOH_K2_TRUSTED_TRANSACTIONS:
            raise PhaseCliqueBuildProbeError(
                "pcoh trusted transaction identity collision"
            )
        _PCOH_K2_TRUSTED_TRANSACTIONS[key] = record


def _pcoh_k2_trusted_transaction_anchor(
    transaction: Any,
) -> Optional[str]:
    if type(transaction) is not dict:
        return None
    with _PCOH_K2_TRUSTED_TRANSACTION_LOCK:
        record = _PCOH_K2_TRUSTED_TRANSACTIONS.get(id(transaction))
        if (
            type(record) is not _PCOHK2TrustedTransactionAnchor
            or record.transaction is not transaction
            or record.process_id != os.getpid()
            or transaction.get("receipt_sha256")
            != record.transaction_receipt_sha256
        ):
            return None
        return record.materialized_tightness_summary_sha256


def _release_pcoh_k2_trusted_transaction(transaction: Any) -> None:
    if type(transaction) is not dict:
        return
    with _PCOH_K2_TRUSTED_TRANSACTION_LOCK:
        record = _PCOH_K2_TRUSTED_TRANSACTIONS.get(id(transaction))
        if (
            type(record) is _PCOHK2TrustedTransactionAnchor
            and record.transaction is transaction
        ):
            _PCOH_K2_TRUSTED_TRANSACTIONS.pop(id(transaction), None)


def _adopt_pcoh_k2_trusted_transaction(
    body: dict[str, Any], transaction: Any
) -> None:
    """Transfer a registered transaction to the finalizer or release it."""

    try:
        if not _pcoh_k2_transaction_receipt_valid(transaction):
            raise PhaseCliqueBuildProbeError(
                "pcoh transaction returned malformed receipt"
            )
        body["pcoh_k2_build_only"] = transaction
    except BaseException:
        _release_pcoh_k2_trusted_transaction(transaction)
        transaction = None
        raise


def _pcoh_k2_materialized_tightness_payload_valid(
    payload: Any,
    *,
    source_semantic_digest: Any,
    stable_bit_ids: Any,
    conditional_certificate_sha256: Any,
    expected_summary_sha256: Any,
) -> bool:
    """Invoke the public detached JSON verifier with a caller-held anchor."""

    try:
        from act.back_end.hybridz_tf.operator_phase_conditioned_build_only import (
            verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload,
        )

        return bool(
            verify_phase_conditioned_objective_hull_build_only_materialized_tightness_payload(
                payload,
                expected_source_semantic_digest=source_semantic_digest,
                expected_stable_bit_ids=stable_bit_ids,
                expected_conditional_certificate_sha256=(
                    conditional_certificate_sha256
                ),
                expected_summary_sha256=expected_summary_sha256,
            )
        )
    except (ImportError, TypeError, ValueError, OverflowError, RuntimeError):
        return False


def _pcoh_k2_stop_loss_receipt(
    *,
    stage: str,
    reason: str,
    started: float,
    input_sha256: Optional[Mapping[str, Any]] = None,
    implementation_sha256: Optional[Mapping[str, Any]] = None,
    stage_resources: Optional[Mapping[str, Any]] = None,
    timings: Optional[Mapping[str, Any]] = None,
    build_only_transaction_called: bool = False,
) -> dict[str, Any]:
    elapsed = float(max(0.0, time.monotonic() - started))
    timing_receipt = dict(timings or {})
    timing_receipt["total_seconds"] = elapsed
    return _checksummed({
        "schema": _PCOH_K2_TRANSACTION_SCHEMA,
        "status": "stop_loss",
        "reason": reason[:300],
        "failed_stage": stage,
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": True,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "build_only_transaction_called": bool(
            build_only_transaction_called
        ),
        "transaction_verified_before_serialization": False,
        "solver_handoff_called": False,
        "diagnostic_lp_called": False,
        "hz_base_feasibility_called": False,
        "hz_objbound_decide_called": False,
        "strict_replay_called": False,
        "fresh_build_returned": False,
        "full_parent_lp_called": False,
        "full_parent_lp_solver_called": False,
        "input_sha256": dict(input_sha256 or {}),
        "implementation_sha256": dict(implementation_sha256 or {}),
        "full_batch_sha256": None,
        "focused_subset_digest": None,
        "focused_encoded_row": None,
        "focused_rival_id": None,
        "successful_selection_binding_retained": False,
        "selection_digest": None,
        "selection_property_digest": None,
        "selection_parent_semantic_digest": None,
        "selection_operator_row_tag_digest": None,
        "stable_bit_selection_method": None,
        "stable_bit_ids": [],
        "diagnostic_schema": None,
        "diagnostic_sha256": None,
        "transaction_receipt_sha256": None,
        "source_semantic_digest": None,
        "fresh_semantic_digest": None,
        "source_dimensions": None,
        "fresh_dimensions": None,
        "conditional_certificate_sha256": [],
        "pair_bundle_sha256": None,
        "fresh_issuance_sha256": None,
        "materialized_tightness_summary_sha256": None,
        "materialized_tightness_summary": None,
        "tightness_gate": None,
        "resource_preflight": None,
        "resource_postflight": None,
        "stage_resources": dict(stage_resources or {}),
        "timings": timing_receipt,
    })


def _pcoh_k2_transaction_receipt_valid(value: Any) -> bool:
    if type(value) is not dict:
        return False
    try:
        checksum_valid = _local_receipt_checksum_valid(
            value, schema=_PCOH_K2_TRANSACTION_SCHEMA
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if not checksum_valid:
        return False
    if set(value) != _PCOH_K2_TRANSACTION_FIELDS:
        return False
    input_sha256 = value.get("input_sha256")
    implementation_sha256 = value.get("implementation_sha256")
    stage_resources = value.get("stage_resources")
    timings = value.get("timings")
    common = bool(
        value.get("status") in {"built_and_released", "stop_loss"}
        and value.get("diagnostic_only") is True
        and value.get("candidate_only") is True
        and value.get("build_only") is True
        and value.get("instance_count") == 1
        and value.get("proof_authority") is False
        and value.get("verdict_authority") is False
        and value.get("ground_truth_loaded") is False
        and value.get("reference_label_used") is False
        and type(value.get("build_only_transaction_called")) is bool
        and type(value.get("transaction_verified_before_serialization"))
        is bool
        and value.get("solver_handoff_called") is False
        and value.get("diagnostic_lp_called") is False
        and value.get("hz_base_feasibility_called") is False
        and value.get("hz_objbound_decide_called") is False
        and value.get("strict_replay_called") is False
        and value.get("fresh_build_returned") is False
        and value.get("full_parent_lp_called") is False
        and value.get("full_parent_lp_solver_called") is False
        and type(input_sha256) is dict
        and set(input_sha256) == {"onnx", "vnnlib", "instances_csv"}
        and all(_valid_sha256(item) for item in input_sha256.values())
        and type(implementation_sha256) is dict
        and bool(implementation_sha256)
        and set(implementation_sha256) == set(_IMPLEMENTATION_RELATIVE_PATHS)
        and all(
            type(path) is str and _valid_sha256(digest)
            for path, digest in implementation_sha256.items()
        )
        and type(stage_resources) is dict
        and bool(stage_resources)
        and all(
            type(name) is str
            and type(snapshot) is dict
            and type(snapshot.get("peak_rss_bytes")) is int
            and snapshot["peak_rss_bytes"] >= 0
            and (
                snapshot.get("current_rss_bytes") is None
                or (
                    type(snapshot.get("current_rss_bytes")) is int
                    and snapshot["current_rss_bytes"] >= 0
                )
            )
            for name, snapshot in stage_resources.items()
        )
        and type(timings) is dict
        and type(timings.get("total_seconds")) is float
        and math.isfinite(timings["total_seconds"])
        and timings["total_seconds"] >= 0.0
    )
    if not common:
        return False
    if value["status"] == "stop_loss":
        return bool(
            type(value.get("failed_stage")) is str
            and bool(value["failed_stage"])
            and type(value.get("reason")) is str
            and bool(value["reason"])
            and value.get("transaction_verified_before_serialization")
            is False
            and value.get("successful_selection_binding_retained") is False
            and value.get("stable_bit_ids") == []
            and value.get("conditional_certificate_sha256") == []
            and all(
                value.get(name) is None
                for name in (
                    "full_batch_sha256",
                    "focused_subset_digest",
                    "focused_encoded_row",
                    "focused_rival_id",
                    "selection_digest",
                    "selection_property_digest",
                    "selection_parent_semantic_digest",
                    "selection_operator_row_tag_digest",
                    "stable_bit_selection_method",
                    "diagnostic_schema",
                    "diagnostic_sha256",
                    "transaction_receipt_sha256",
                    "source_semantic_digest",
                    "fresh_semantic_digest",
                    "source_dimensions",
                    "fresh_dimensions",
                    "pair_bundle_sha256",
                    "fresh_issuance_sha256",
                    "materialized_tightness_summary_sha256",
                    "materialized_tightness_summary",
                    "tightness_gate",
                    "resource_preflight",
                    "resource_postflight",
                )
            )
        )
    sha_fields = (
        "full_batch_sha256",
        "focused_subset_digest",
        "selection_digest",
        "selection_property_digest",
        "selection_parent_semantic_digest",
        "selection_operator_row_tag_digest",
        "diagnostic_sha256",
        "transaction_receipt_sha256",
        "source_semantic_digest",
        "fresh_semantic_digest",
        "pair_bundle_sha256",
        "fresh_issuance_sha256",
    )
    trusted_summary_sha256 = _pcoh_k2_trusted_transaction_anchor(value)
    return bool(
        value.get("reason") is None
        and value.get("failed_stage") is None
        and value.get("build_only_transaction_called") is True
        and value.get("transaction_verified_before_serialization") is True
        and value.get("successful_selection_binding_retained") is True
        and value.get("stable_bit_selection_method")
        == "lowest_two_canonical_ids_from_verified_selection"
        and type(value.get("focused_encoded_row")) is int
        and value["focused_encoded_row"] >= 0
        and type(value.get("focused_rival_id")) is int
        and value["focused_rival_id"] >= 0
        and type(value.get("stable_bit_ids")) is list
        and len(value["stable_bit_ids"]) == 2
        and all(
            type(stable_id) is int and stable_id >= 0
            for stable_id in value["stable_bit_ids"]
        )
        and sorted(value["stable_bit_ids"]) == value["stable_bit_ids"]
        and len(set(value["stable_bit_ids"])) == 2
        and type(value.get("conditional_certificate_sha256")) is list
        and len(value["conditional_certificate_sha256"]) == 4
        and len(set(value["conditional_certificate_sha256"])) == 4
        and all(_valid_sha256(value.get(name)) for name in sha_fields)
        and all(
            _valid_sha256(item)
            for item in value["conditional_certificate_sha256"]
        )
        and isinstance(value.get("resource_preflight"), dict)
        and value["resource_preflight"].get("passed") is True
        and value["resource_preflight"].get("caller_supplied") is False
        and isinstance(value.get("resource_postflight"), dict)
        and value["resource_postflight"].get("passed") is True
        and value["resource_postflight"].get("caller_supplied") is False
        and value.get("diagnostic_schema")
        == "act.hybridz_pcoh_build_only_diagnostic.v2"
        and value.get("selection_parent_semantic_digest")
        == value.get("source_semantic_digest")
        and _valid_sha256(
            value.get("materialized_tightness_summary_sha256")
        )
        and _valid_sha256(trusted_summary_sha256)
        and value.get("materialized_tightness_summary_sha256")
        == trusted_summary_sha256
        and type(value.get("materialized_tightness_summary")) is dict
        and value["materialized_tightness_summary"].get("summary_sha256")
        == value["materialized_tightness_summary_sha256"]
        and _pcoh_k2_materialized_tightness_payload_valid(
            value["materialized_tightness_summary"],
            source_semantic_digest=value["source_semantic_digest"],
            stable_bit_ids=value["stable_bit_ids"],
            conditional_certificate_sha256=value[
                "conditional_certificate_sha256"
            ],
            expected_summary_sha256=trusted_summary_sha256,
        )
        and _pcoh_k2_tightness_gate_valid(
            value.get("tightness_gate"),
            value["materialized_tightness_summary"],
            expected_summary_sha256=trusted_summary_sha256,
        )
        and set(stage_resources)
        == {
            "entry",
            "raw_batch",
            "focused_rival",
            "literal_selection",
            "build_only_transaction",
        }
        and type(value.get("source_dimensions")) is list
        and len(value["source_dimensions"]) == 5
        and type(value.get("fresh_dimensions")) is list
        and len(value["fresh_dimensions"]) == 5
        and all(
            type(dimension) is int and dimension >= 0
            for dimension in value["source_dimensions"]
        )
        and all(
            type(dimension) is int and dimension >= 0
            for dimension in value["fresh_dimensions"]
        )
        and value["fresh_dimensions"][0] == value["source_dimensions"][0]
        and value["fresh_dimensions"][1]
        == value["source_dimensions"][1] + 4
        and value["fresh_dimensions"][2] == value["source_dimensions"][2]
        and 3
        <= value["fresh_dimensions"][3] - value["source_dimensions"][3]
        <= 7
        and value["fresh_dimensions"][4]
        == value["source_dimensions"][4] + 1
    )


def _pcoh_k3_focused_semantic_anchor(
    *,
    source_semantic_digest: str,
    full_batch_sha256: str,
    focused_encoded_row: int,
    focused_rival_id: int,
    selection_digest: str,
    selection_property_digest: str,
    selection_parent_semantic_digest: str,
    selection_operator_row_tag_digest: str,
) -> dict[str, Any]:
    """Project run-local focus evidence onto a stable semantic identity."""

    sha_values = (
        source_semantic_digest,
        full_batch_sha256,
        selection_digest,
        selection_property_digest,
        selection_parent_semantic_digest,
        selection_operator_row_tag_digest,
    )
    if (
        any(not _valid_sha256(value) for value in sha_values)
        or type(focused_encoded_row) is not int
        or focused_encoded_row < 0
        or type(focused_rival_id) is not int
        or focused_rival_id < 0
        or selection_parent_semantic_digest != source_semantic_digest
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 focused semantic anchor inputs are invalid"
        )
    payload = {
        "schema": _PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SCHEMA,
        "candidate_only": True,
        "proof_authority": False,
        "focus_method": _PCOH_K3_FOCUS_METHOD,
        "focus_count": _PCOH_K3_FOCUS_COUNT,
        "source_semantic_digest": source_semantic_digest,
        "full_batch_sha256": full_batch_sha256,
        "focused_encoded_row": focused_encoded_row,
        "focused_rival_id": focused_rival_id,
        "selection_digest": selection_digest,
        "selection_property_digest": selection_property_digest,
        "selection_parent_semantic_digest": (
            selection_parent_semantic_digest
        ),
        "selection_operator_row_tag_digest": (
            selection_operator_row_tag_digest
        ),
    }
    return {
        **payload,
        "semantic_sha256": hashlib.sha256(
            _canonical_json(payload)
        ).hexdigest(),
    }


def _pcoh_k3_focused_semantic_anchor_valid(value: Any) -> bool:
    fields = {
        "schema", "candidate_only", "proof_authority", "focus_method",
        "focus_count", "source_semantic_digest", "full_batch_sha256",
        "focused_encoded_row", "focused_rival_id", "selection_digest",
        "selection_property_digest", "selection_parent_semantic_digest",
        "selection_operator_row_tag_digest", "semantic_sha256",
    }
    try:
        if (
            type(value) is not dict
            or set(value) != fields
            or type(value.get("schema")) is not str
            or value.get("schema")
            != _PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SCHEMA
            or value.get("candidate_only") is not True
            or value.get("proof_authority") is not False
            or type(value.get("focus_method")) is not str
            or value.get("focus_method") != _PCOH_K3_FOCUS_METHOD
            or type(value.get("focus_count")) is not int
            or value.get("focus_count") != _PCOH_K3_FOCUS_COUNT
            or not _valid_sha256(value.get("semantic_sha256"))
        ):
            return False
        expected = _pcoh_k3_focused_semantic_anchor(
            source_semantic_digest=value["source_semantic_digest"],
            full_batch_sha256=value["full_batch_sha256"],
            focused_encoded_row=value["focused_encoded_row"],
            focused_rival_id=value["focused_rival_id"],
            selection_digest=value["selection_digest"],
            selection_property_digest=value[
                "selection_property_digest"
            ],
            selection_parent_semantic_digest=value[
                "selection_parent_semantic_digest"
            ],
            selection_operator_row_tag_digest=value[
                "selection_operator_row_tag_digest"
            ],
        )
        return value == expected
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        PhaseCliqueBuildProbeError,
    ):
        return False


def _pcoh_k3_fixed_focused_semantic_anchor_valid(value: Any) -> bool:
    return bool(
        _pcoh_k3_focused_semantic_anchor_valid(value)
        and value.get("semantic_sha256")
        == _PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SHA256
        and value.get("source_semantic_digest")
        == _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
        and value.get("full_batch_sha256")
        == _PCOH_K3_EXPECTED_FULL_BATCH_SHA256
        and value.get("focused_encoded_row")
        == _PCOH_K3_FOCUSED_ENCODED_ROW
        and value.get("focused_rival_id") == _PCOH_K3_FOCUSED_RIVAL_ID
        and value.get("selection_digest")
        == _PCOH_K3_EXPECTED_SELECTION_DIGEST
        and value.get("selection_property_digest")
        == _PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
        and value.get("selection_parent_semantic_digest")
        == _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
        and value.get("selection_operator_row_tag_digest")
        == _PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
    )


def _pcoh_k3_baseline_anchor_receipt_valid(value: Any) -> bool:
    fields = {
        "schema", "status", "diagnostic_only", "candidate_only",
        "proof_authority", "verdict_authority", "ground_truth_loaded",
        "reference_label_used", "artifact_relative_path",
        "artifact_sha256", "artifact_bytes", "artifact_receipt_sha256",
        "baseline_summary_sha256", "source_semantic_digest",
        "selection_digest", "selection_property_digest",
        "selection_parent_semantic_digest",
        "selection_operator_row_tag_digest", "full_batch_sha256",
        "focused_subset_digest", "focused_semantic_anchor",
        "focused_semantic_anchor_sha256",
        "focused_encoded_row", "focused_rival_id",
        "retained_k2_stable_bit_ids", "global_cube_upper_hex",
        "materialized_payload_detached_verified",
        "tightness_gate_detached_verified", "receipt_sha256",
    }
    try:
        return bool(
            type(value) is dict
            and set(value) == fields
            and _local_receipt_checksum_valid(
                value, schema=_PCOH_K3_BASELINE_ANCHOR_SCHEMA
            )
            and value.get("status") == "fixed_baseline_verified"
            and value.get("diagnostic_only") is True
            and value.get("candidate_only") is True
            and value.get("proof_authority") is False
            and value.get("verdict_authority") is False
            and value.get("ground_truth_loaded") is False
            and value.get("reference_label_used") is False
            and value.get("artifact_relative_path")
            == _PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH
            and value.get("artifact_sha256")
            == _PCOH_K3_BASELINE_ARTIFACT_SHA256
            and type(value.get("artifact_bytes")) is int
            and 0 < value["artifact_bytes"] <= 16 * 1024 * 1024
            and _valid_sha256(value.get("artifact_receipt_sha256"))
            and value.get("baseline_summary_sha256")
            == _PCOH_K3_BASELINE_SUMMARY_SHA256
            and value.get("source_semantic_digest")
            == _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            and value.get("selection_digest")
            == _PCOH_K3_EXPECTED_SELECTION_DIGEST
            and value.get("selection_property_digest")
            == _PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
            and value.get("full_batch_sha256")
            == _PCOH_K3_EXPECTED_FULL_BATCH_SHA256
            and _valid_sha256(value.get("focused_subset_digest"))
            and _pcoh_k3_fixed_focused_semantic_anchor_valid(
                value.get("focused_semantic_anchor")
            )
            and value.get("focused_semantic_anchor_sha256")
            == _PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SHA256
            and value["focused_semantic_anchor"].get("semantic_sha256")
            == value.get("focused_semantic_anchor_sha256")
            and value.get("selection_parent_semantic_digest")
            == _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            and value.get("selection_operator_row_tag_digest")
            == _PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
            and value["focused_semantic_anchor"].get(
                "source_semantic_digest"
            ) == value.get("source_semantic_digest")
            and value["focused_semantic_anchor"].get("full_batch_sha256")
            == value.get("full_batch_sha256")
            and value["focused_semantic_anchor"].get("selection_digest")
            == value.get("selection_digest")
            and value["focused_semantic_anchor"].get(
                "selection_property_digest"
            ) == value.get("selection_property_digest")
            and value["focused_semantic_anchor"].get(
                "selection_parent_semantic_digest"
            ) == value.get("selection_parent_semantic_digest")
            and value["focused_semantic_anchor"].get(
                "selection_operator_row_tag_digest"
            ) == value.get("selection_operator_row_tag_digest")
            and value.get("focused_encoded_row")
            == _PCOH_K3_FOCUSED_ENCODED_ROW
            and value.get("focused_rival_id")
            == _PCOH_K3_FOCUSED_RIVAL_ID
            and value.get("retained_k2_stable_bit_ids")
            == list(_PCOH_K3_RETAINED_K2_STABLE_BIT_IDS)
            and value.get("global_cube_upper_hex")
            == _PCOH_K3_GLOBAL_CUBE_UPPER_HEX
            and value.get("materialized_payload_detached_verified") is True
            and value.get("tightness_gate_detached_verified") is True
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _pcoh_k3_fixed_baseline_artifact_anchor(
    *, deadline: float
) -> dict[str, Any]:
    """Read and verify the fixed K2 artifact through one no-follow FD."""

    if (
        type(deadline) is not float
        or not math.isfinite(deadline)
        or time.monotonic() >= deadline
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 baseline artifact deadline is invalid"
        )
    path = _REPO_ROOT / _PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH
    if path.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise PhaseCliqueBuildProbeError(
            "K3 baseline artifact path or parent is symlinked"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > 16 * 1024 * 1024
        ):
            raise PhaseCliqueBuildProbeError(
                "K3 baseline artifact file shape is invalid"
            )
        chunks = []
        total = 0
        digest = hashlib.sha256()
        while True:
            if time.monotonic() >= deadline:
                raise PhaseCliqueBuildProbeError(
                    "K3 baseline artifact read exceeded deadline"
                )
            chunk = os.read(fd, min(1024 * 1024, 16 * 1024 * 1024 + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise PhaseCliqueBuildProbeError(
                    "K3 baseline artifact exceeded byte cap"
                )
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(fd)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or total != before.st_size
        ):
            raise PhaseCliqueBuildProbeError(
                "K3 baseline artifact changed during read"
            )
    finally:
        os.close(fd)
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != _PCOH_K3_BASELINE_ARTIFACT_SHA256:
        raise PhaseCliqueBuildProbeError(
            "K3 baseline artifact SHA256 mismatch"
        )
    try:
        artifact = json.loads(
            b"".join(chunks).decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant:{value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseCliqueBuildProbeError(
            "K3 baseline artifact JSON is malformed"
        ) from exc
    if (
        type(artifact) is not dict
        or not _all_finite_json(artifact)
        or not _local_receipt_checksum_valid(artifact, schema=_SCHEMA)
        or artifact.get("candidate_mode") != _PCOH_K2_BUILD_ONLY_MODE
        or artifact.get("phase_status") != "built_and_released"
        or artifact.get("ground_truth_loaded") is not False
        or artifact.get("reference_label_used") is not False
        or artifact.get("input_sha256")
        != {
            "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
            "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
            "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
        }
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 baseline artifact top receipt failed verification"
        )
    transaction = artifact.get("pcoh_k2_build_only")
    if (
        type(transaction) is not dict
        or set(transaction) != _PCOH_K2_TRANSACTION_FIELDS
        or not _local_receipt_checksum_valid(
            transaction, schema=_PCOH_K2_TRANSACTION_SCHEMA
        )
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 baseline nested K2 transaction is malformed"
        )
    summary = transaction.get("materialized_tightness_summary")
    summary_sha256 = transaction.get(
        "materialized_tightness_summary_sha256"
    )
    materialized_valid = _pcoh_k2_materialized_tightness_payload_valid(
        summary,
        source_semantic_digest=transaction.get("source_semantic_digest"),
        stable_bit_ids=transaction.get("stable_bit_ids"),
        conditional_certificate_sha256=transaction.get(
            "conditional_certificate_sha256"
        ),
        expected_summary_sha256=summary_sha256,
    )
    tightness_valid = _pcoh_k2_tightness_gate_valid(
        transaction.get("tightness_gate"),
        summary,
        expected_summary_sha256=summary_sha256,
    )
    if (
        summary_sha256 != _PCOH_K3_BASELINE_SUMMARY_SHA256
        or type(summary) is not dict
        or summary.get("summary_sha256") != summary_sha256
        or summary.get("global_cube_upper_hex")
        != _PCOH_K3_GLOBAL_CUBE_UPPER_HEX
        or summary.get("global_cube_upper_exact")
        != [
            _PCOH_K3_GLOBAL_CUBE_UPPER_EXACT.numerator,
            _PCOH_K3_GLOBAL_CUBE_UPPER_EXACT.denominator,
        ]
        or Fraction.from_float(
            float.fromhex(_PCOH_K3_GLOBAL_CUBE_UPPER_HEX)
        )
        <= _PCOH_K3_GLOBAL_CUBE_UPPER_EXACT
        or transaction.get("source_semantic_digest")
        != _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
        or transaction.get("selection_parent_semantic_digest")
        != _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
        or transaction.get("selection_digest")
        != _PCOH_K3_EXPECTED_SELECTION_DIGEST
        or transaction.get("selection_property_digest")
        != _PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
        or transaction.get("selection_operator_row_tag_digest")
        != _PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
        or transaction.get("full_batch_sha256")
        != _PCOH_K3_EXPECTED_FULL_BATCH_SHA256
        or not _valid_sha256(transaction.get("focused_subset_digest"))
        or transaction.get("focused_encoded_row")
        != _PCOH_K3_FOCUSED_ENCODED_ROW
        or transaction.get("focused_rival_id") != _PCOH_K3_FOCUSED_RIVAL_ID
        or transaction.get("stable_bit_ids")
        != list(_PCOH_K3_RETAINED_K2_STABLE_BIT_IDS)
        or materialized_valid is not True
        or tightness_valid is not True
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 baseline nested fixed anchors failed verification"
        )
    focused_semantic_anchor = _pcoh_k3_focused_semantic_anchor(
        source_semantic_digest=transaction["source_semantic_digest"],
        full_batch_sha256=transaction["full_batch_sha256"],
        focused_encoded_row=transaction["focused_encoded_row"],
        focused_rival_id=transaction["focused_rival_id"],
        selection_digest=transaction["selection_digest"],
        selection_property_digest=transaction[
            "selection_property_digest"
        ],
        selection_parent_semantic_digest=transaction[
            "selection_parent_semantic_digest"
        ],
        selection_operator_row_tag_digest=transaction[
            "selection_operator_row_tag_digest"
        ],
    )
    if not _pcoh_k3_fixed_focused_semantic_anchor_valid(
        focused_semantic_anchor
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 baseline semantic anchor projection failed verification"
        )
    result = _checksummed({
        "schema": _PCOH_K3_BASELINE_ANCHOR_SCHEMA,
        "status": "fixed_baseline_verified",
        "diagnostic_only": True,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "artifact_relative_path": _PCOH_K3_BASELINE_ARTIFACT_RELATIVE_PATH,
        "artifact_sha256": observed_sha256,
        "artifact_bytes": total,
        "artifact_receipt_sha256": artifact["receipt_sha256"],
        "baseline_summary_sha256": summary_sha256,
        "source_semantic_digest": transaction["source_semantic_digest"],
        "selection_digest": transaction["selection_digest"],
        "selection_property_digest": transaction[
            "selection_property_digest"
        ],
        "selection_parent_semantic_digest": transaction[
            "selection_parent_semantic_digest"
        ],
        "selection_operator_row_tag_digest": transaction[
            "selection_operator_row_tag_digest"
        ],
        "full_batch_sha256": transaction["full_batch_sha256"],
        "focused_subset_digest": transaction["focused_subset_digest"],
        "focused_semantic_anchor": focused_semantic_anchor,
        "focused_semantic_anchor_sha256": focused_semantic_anchor[
            "semantic_sha256"
        ],
        "focused_encoded_row": transaction["focused_encoded_row"],
        "focused_rival_id": transaction["focused_rival_id"],
        "retained_k2_stable_bit_ids": transaction["stable_bit_ids"],
        "global_cube_upper_hex": summary["global_cube_upper_hex"],
        "materialized_payload_detached_verified": materialized_valid,
        "tightness_gate_detached_verified": tightness_valid,
    })
    if not _pcoh_k3_baseline_anchor_receipt_valid(result):
        raise PhaseCliqueBuildProbeError(
            "K3 baseline anchor receipt self-check failed"
        )
    return result


def _pcoh_k3_source_build_preflight(
    shape: Mapping[str, Any],
    *,
    build_seconds: Any,
    input_sha256: Mapping[str, Any],
    implementation_sha256: Mapping[str, Any],
    baseline_anchor_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the K2 size gates while keeping a K3-only receipt keyset."""

    if (
        type(implementation_sha256) is not dict
        or set(implementation_sha256)
        != set(_PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS)
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 source preflight implementation keyset mismatch"
        )
    if not _pcoh_k3_baseline_anchor_receipt_valid(
        baseline_anchor_receipt
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 source preflight baseline anchor mismatch"
        )
    k2_implementation = {
        path: implementation_sha256[path]
        for path in _IMPLEMENTATION_RELATIVE_PATHS
    }
    base = _pcoh_k2_source_build_preflight(
        shape,
        build_seconds=build_seconds,
        input_sha256=input_sha256,
        implementation_sha256=k2_implementation,
    )
    body = dict(base)
    body.pop("receipt_sha256", None)
    body["schema"] = _PCOH_K3_SOURCE_PREFLIGHT_SCHEMA
    body["implementation_sha256"] = dict(implementation_sha256)
    body["baseline_artifact_sha256"] = (
        _PCOH_K3_BASELINE_ARTIFACT_SHA256
    )
    body["baseline_summary_sha256"] = _PCOH_K3_BASELINE_SUMMARY_SHA256
    body["baseline_anchor_receipt_sha256"] = baseline_anchor_receipt[
        "receipt_sha256"
    ]
    body["baseline_anchor_receipt"] = dict(baseline_anchor_receipt)
    return _checksummed(body)


def _pcoh_k3_source_build_preflight_valid(value: Any) -> bool:
    if type(value) is not dict:
        return False
    try:
        if not _local_receipt_checksum_valid(
            value, schema=_PCOH_K3_SOURCE_PREFLIGHT_SCHEMA
        ):
            return False
        expected_fields = {
            "schema", "status", "diagnostic_only", "candidate_only",
            "build_only", "instance_count", "proof_authority",
            "verdict_authority", "ground_truth_loaded",
            "reference_label_used", "input_sha256",
            "implementation_sha256", "source_shape",
            "source_build_seconds", "rows_plus_outputs", "conditions",
            "failed_conditions", "baseline_artifact_sha256",
            "baseline_summary_sha256", "receipt_sha256",
            "baseline_anchor_receipt_sha256",
            "baseline_anchor_receipt",
        }
        implementation = value.get("implementation_sha256")
        if (
            set(value) != expected_fields
            or value.get("baseline_artifact_sha256")
            != _PCOH_K3_BASELINE_ARTIFACT_SHA256
            or value.get("baseline_summary_sha256")
            != _PCOH_K3_BASELINE_SUMMARY_SHA256
            or not _valid_sha256(
                value.get("baseline_anchor_receipt_sha256")
            )
            or not _pcoh_k3_baseline_anchor_receipt_valid(
                value.get("baseline_anchor_receipt")
            )
            or value["baseline_anchor_receipt"].get("receipt_sha256")
            != value.get("baseline_anchor_receipt_sha256")
            or type(implementation) is not dict
            or set(implementation)
            != set(_PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS)
            or any(not _valid_sha256(item) for item in implementation.values())
        ):
            return False
        expected = _pcoh_k3_source_build_preflight(
            value.get("source_shape"),
            build_seconds=value.get("source_build_seconds"),
            input_sha256=value.get("input_sha256"),
            implementation_sha256=implementation,
            baseline_anchor_receipt=value["baseline_anchor_receipt"],
        )
        return bool(
            value == expected
            and value.get("failed_conditions")
            == [
                name
                for name, passed in value.get("conditions", {}).items()
                if passed is not True
            ]
            and value.get("status")
            == (
                "passed"
                if not value.get("failed_conditions")
                else "stop_loss"
            )
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        PhaseCliqueBuildProbeError,
    ):
        return False


def _pcoh_k3_strong_tightness_gate(
    summary: Mapping[str, Any],
    *,
    source_semantic_digest: str,
    selection_digest: str,
    focused_encoded_row: int,
    focused_rival_id: int,
    retained_k2_stable_bit_ids: Sequence[int],
    stable_bit_ids: Sequence[int],
) -> dict[str, Any]:
    """Recompute the exact preregistered K3 promotion boundary."""

    from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
        _K3_STRONG_TARGET,
    )

    if type(summary) is not dict:
        raise PhaseCliqueBuildProbeError("K3 tightness summary is not JSON")
    retained = tuple(retained_k2_stable_bit_ids)
    stable = tuple(stable_bit_ids)
    global_upper = _pcoh_k2_strict_hex_fraction(
        summary.get("global_cube_upper_hex"), name="k3_global_cube_upper"
    )
    final_upper = _pcoh_k2_strict_hex_fraction(
        summary.get("final_structural_upper_hex"),
        name="k3_final_structural_upper",
    )
    ideal_upper = _pcoh_k2_strict_hex_fraction(
        summary.get("ideal_union_upper_hex"), name="k3_ideal_union_upper"
    )
    rounding_tax = _pcoh_k2_strict_fraction_pair(
        summary.get("rounding_tax_exact"), name="k3_rounding_tax"
    )
    expected_global_stored = Fraction.from_float(
        float.fromhex(_PCOH_K3_GLOBAL_CUBE_UPPER_HEX)
    )
    summary_global_exact = _pcoh_k2_strict_fraction_pair(
        summary.get("global_cube_upper_exact"),
        name="k3_global_cube_upper_exact",
    )
    if summary_global_exact != _PCOH_K3_GLOBAL_CUBE_UPPER_EXACT:
        raise PhaseCliqueBuildProbeError(
            "K3 global cube exact fixed-anchor mismatch"
        )
    if global_upper < summary_global_exact:
        raise PhaseCliqueBuildProbeError(
            "K3 global cube stored upper is not outward"
        )
    if global_upper == summary_global_exact:
        raise PhaseCliqueBuildProbeError(
            "K3 global cube exact/stored representations collapsed"
        )
    global_anchor = bool(
        global_upper == expected_global_stored
        and summary.get("global_cube_upper_hex")
        == _PCOH_K3_GLOBAL_CUBE_UPPER_HEX
    )
    source_anchor = bool(
        source_semantic_digest
        == _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
        and summary.get("parent_semantic_digest")
        == _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
    )
    selection_anchor = (
        selection_digest == _PCOH_K3_EXPECTED_SELECTION_DIGEST
    )
    focus_anchor = bool(
        focused_encoded_row == _PCOH_K3_FOCUSED_ENCODED_ROW
        and focused_rival_id == _PCOH_K3_FOCUSED_RIVAL_ID
    )
    retained_anchor = bool(
        retained == _PCOH_K3_RETAINED_K2_STABLE_BIT_IDS
        and len(stable) == 3
        and stable[:2] == retained
        and tuple(summary.get("stable_bit_ids", ())) == stable
    )
    if not all(
        (
            global_anchor,
            source_anchor,
            selection_anchor,
            focus_anchor,
            retained_anchor,
        )
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 live global/source/selection/focus anchor mismatch"
        )
    if rounding_tax < 0:
        raise PhaseCliqueBuildProbeError("K3 rounding tax is negative")
    delta = global_upper - final_upper
    if delta < 0:
        raise PhaseCliqueBuildProbeError(
            "K3 final structural upper exceeds global cube upper"
        )
    strong_scale = max(Fraction(1), abs(global_upper) / 50)
    rounding_threshold = 8 * rounding_tax
    if expected_global_stored - strong_scale != _K3_STRONG_TARGET:
        raise PhaseCliqueBuildProbeError(
            "K3 target no longer matches the fixed strong-scale boundary"
        )
    final_at_most_target = final_upper <= _K3_STRONG_TARGET
    strong_scale_met = delta >= strong_scale
    rounding_dominance = delta >= rounding_threshold
    strong = bool(
        final_at_most_target
        and strong_scale_met
        and rounding_dominance
    )
    return _checksummed({
        "schema": _PCOH_K3_TIGHTNESS_GATE_SCHEMA,
        "status": "strong_promotion" if strong else "built_but_not_strong",
        "diagnostic_only": True,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "full_parent_lp_called": False,
        "full_parent_lp_solver_called": False,
        "baseline_artifact_sha256": _PCOH_K3_BASELINE_ARTIFACT_SHA256,
        "baseline_summary_sha256": _PCOH_K3_BASELINE_SUMMARY_SHA256,
        "source_semantic_digest": source_semantic_digest,
        "selection_digest": selection_digest,
        "focused_encoded_row": focused_encoded_row,
        "focused_rival_id": focused_rival_id,
        "retained_k2_stable_bit_ids": list(retained),
        "stable_bit_ids": list(stable),
        "materialized_tightness_summary_sha256": summary.get(
            "summary_sha256"
        ),
        "global_cube_upper_fraction": _pcoh_k2_fraction_pair(global_upper),
        "final_structural_upper_fraction": _pcoh_k2_fraction_pair(final_upper),
        "ideal_union_upper_fraction": _pcoh_k2_fraction_pair(ideal_upper),
        "rounding_tax_fraction": _pcoh_k2_fraction_pair(rounding_tax),
        "delta_fraction": _pcoh_k2_fraction_pair(delta),
        "strong_target_fraction": _pcoh_k2_fraction_pair(_K3_STRONG_TARGET),
        "strong_scale_threshold_fraction": _pcoh_k2_fraction_pair(strong_scale),
        "rounding_tax_threshold_fraction": _pcoh_k2_fraction_pair(
            rounding_threshold
        ),
        "global_anchor_matches": global_anchor,
        "source_anchor_matches": source_anchor,
        "selection_anchor_matches": selection_anchor,
        "focus_anchor_matches": focus_anchor,
        "retained_ids_anchor_matches": retained_anchor,
        "final_at_most_strong_target": final_at_most_target,
        "strong_scale_met": strong_scale_met,
        "rounding_tax_dominance_met": rounding_dominance,
        "strong_candidate": strong,
    })


def _pcoh_k3_strong_tightness_gate_valid(
    value: Any,
    summary: Any,
    **anchors: Any,
) -> bool:
    if type(value) is not dict or set(value) != _PCOH_K3_TIGHTNESS_GATE_FIELDS:
        return False
    try:
        expected = _pcoh_k3_strong_tightness_gate(summary, **anchors)
        return bool(
            _local_receipt_checksum_valid(
                value, schema=_PCOH_K3_TIGHTNESS_GATE_SCHEMA
            )
            and value == expected
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        PhaseCliqueBuildProbeError,
    ):
        return False


def _register_pcoh_k3_trusted_transaction(
    transaction: dict[str, Any],
    *,
    outcome: Any,
    outcome_sha256: str,
    outcome_kind: str,
) -> None:
    from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
        PCOHK3BuildOnlyDiagnostic,
        PCOHK3BuildOnlyResourceStopDiagnostic,
        PCOHK3BuildOnlyStopDiagnostic,
        verify_phase_conditioned_objective_hull_k3_build_only_outcome,
    )

    outcome_contract = {
        "success": (PCOHK3BuildOnlyDiagnostic, "diagnostic_sha256"),
        "strong_target_stop": (
            PCOHK3BuildOnlyStopDiagnostic,
            "stop_sha256",
        ),
        "resource_stop": (
            PCOHK3BuildOnlyResourceStopDiagnostic,
            "resource_stop_sha256",
        ),
    }
    expected = outcome_contract.get(outcome_kind)
    if (
        type(transaction) is not dict
        or not _valid_sha256(transaction.get("receipt_sha256"))
        or not _valid_sha256(outcome_sha256)
        or type(outcome_kind) is not str
        or expected is None
        or type(outcome) is not expected[0]
        or getattr(outcome, expected[1], None) != outcome_sha256
        or transaction.get("trusted_outcome_sha256") != outcome_sha256
        or transaction.get("outcome_kind") != outcome_kind
        or verify_phase_conditioned_objective_hull_k3_build_only_outcome(
            outcome
        )
        is not True
        or not _pcoh_k3_transaction_structure_valid(
            transaction,
            expected_outcome_sha256=outcome_sha256,
            expected_outcome_kind=outcome_kind,
        )
    ):
        raise PhaseCliqueBuildProbeError(
            "K3 trusted transaction registration rejected"
        )
    record = _PCOHK3TrustedTransactionAnchor(
        transaction=transaction,
        outcome=outcome,
        process_id=os.getpid(),
        transaction_receipt_sha256=transaction["receipt_sha256"],
        outcome_sha256=outcome_sha256,
        outcome_kind=outcome_kind,
    )
    with _PCOH_K3_TRUSTED_TRANSACTION_LOCK:
        key = id(transaction)
        dead_outcomes = [
            item_key
            for item_key, item_ref in _PCOH_K3_CONSUMED_OUTCOMES.items()
            if item_ref() is None
        ]
        for item_key in dead_outcomes:
            _PCOH_K3_CONSUMED_OUTCOMES.pop(item_key, None)
        outcome_ref = _PCOH_K3_CONSUMED_OUTCOMES.get(id(outcome))
        if key in _PCOH_K3_TRUSTED_TRANSACTIONS:
            raise PhaseCliqueBuildProbeError(
                "K3 trusted transaction identity collision"
            )
        if outcome_ref is not None and outcome_ref() is outcome:
            raise PhaseCliqueBuildProbeError(
                "K3 trusted outcome identity already consumed"
            )
        try:
            _PCOH_K3_CONSUMED_OUTCOMES[id(outcome)] = weakref.ref(outcome)
        except TypeError as exc:
            raise PhaseCliqueBuildProbeError(
                "K3 trusted outcome is not weak-referenceable"
            ) from exc
        _PCOH_K3_TRUSTED_TRANSACTIONS[key] = record


def _pcoh_k3_trusted_transaction_anchor(
    transaction: Any,
) -> Optional[_PCOHK3TrustedTransactionAnchor]:
    if type(transaction) is not dict:
        return None
    with _PCOH_K3_TRUSTED_TRANSACTION_LOCK:
        record = _PCOH_K3_TRUSTED_TRANSACTIONS.get(id(transaction))
        if (
            type(record) is not _PCOHK3TrustedTransactionAnchor
            or record.transaction is not transaction
            or record.outcome is None
            or record.process_id != os.getpid()
            or transaction.get("receipt_sha256")
            != record.transaction_receipt_sha256
            or transaction.get("trusted_outcome_sha256")
            != record.outcome_sha256
            or transaction.get("outcome_kind") != record.outcome_kind
        ):
            return None
        return record


def _release_pcoh_k3_trusted_transaction(transaction: Any) -> None:
    if type(transaction) is not dict:
        return
    with _PCOH_K3_TRUSTED_TRANSACTION_LOCK:
        record = _PCOH_K3_TRUSTED_TRANSACTIONS.get(id(transaction))
        if (
            type(record) is _PCOHK3TrustedTransactionAnchor
            and record.transaction is transaction
        ):
            _PCOH_K3_TRUSTED_TRANSACTIONS.pop(id(transaction), None)


def _pcoh_k3_stop_loss_receipt(
    *,
    stage: str,
    reason: str,
    started: float,
    input_sha256: Optional[Mapping[str, Any]] = None,
    implementation_sha256: Optional[Mapping[str, Any]] = None,
    stage_resources: Optional[Mapping[str, Any]] = None,
    timings: Optional[Mapping[str, Any]] = None,
    k3_transaction_called: bool = False,
    baseline_anchor_receipt: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    timing_receipt = dict(timings or {})
    observed_stage_seconds = [
        item
        for item in timing_receipt.values()
        if type(item) is float and math.isfinite(item) and item >= 0.0
    ]
    timing_receipt["total_seconds"] = float(
        max(
            [0.0, time.monotonic() - started, *observed_stage_seconds]
        )
    )
    return _checksummed({
        "schema": _PCOH_K3_TRANSACTION_SCHEMA,
        "status": "stop_loss",
        "reason": str(reason)[:300],
        "failed_stage": str(stage),
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": True,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "provenance_authority": False,
        "authenticity_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "k3_transaction_called": bool(k3_transaction_called),
        "k3_same_process_verified": False,
        "trusted_outcome_digest_captured_before_detach": False,
        "k3_detached_verified": False,
        "k2_build_only_called": False,
        "phase_transaction_called": False,
        "solver_handoff_called": False,
        "hz_base_feasibility_called": False,
        "hz_objbound_decide_called": False,
        "full_parent_lp_called": False,
        "full_parent_lp_solver_called": False,
        "fresh_build_returned": False,
        "input_sha256": dict(input_sha256 or {}),
        "implementation_sha256": dict(implementation_sha256 or {}),
        "baseline_artifact_sha256": _PCOH_K3_BASELINE_ARTIFACT_SHA256,
        "baseline_summary_sha256": _PCOH_K3_BASELINE_SUMMARY_SHA256,
        "baseline_anchor_receipt_sha256": (
            baseline_anchor_receipt.get("receipt_sha256")
            if _pcoh_k3_baseline_anchor_receipt_valid(
                baseline_anchor_receipt
            )
            else None
        ),
        "baseline_anchor_verified": bool(
            _pcoh_k3_baseline_anchor_receipt_valid(
                baseline_anchor_receipt
            )
        ),
        "full_batch_sha256": None,
        "focused_subset_digest": None,
        "residual_selector_receipt_sha256": None,
        "focused_semantic_anchor": None,
        "focused_semantic_anchor_sha256": None,
        "focused_encoded_row": None,
        "focused_rival_id": None,
        "source_semantic_digest": None,
        "selection_digest": None,
        "selection_property_digest": None,
        "selection_parent_semantic_digest": None,
        "selection_operator_row_tag_digest": None,
        "stable_bit_selection_method": None,
        "retained_k2_stable_bit_ids": [],
        "stable_bit_ids": [],
        "third_stable_bit_id": None,
        "outcome_kind": None,
        "outcome_schema": None,
        "outcome_status": None,
        "trusted_outcome_sha256": None,
        "outcome_receipt_sha256": None,
        "detached_outcome": None,
        "resource_gate_rejection_sha256": None,
        "resource_gate_rejection": None,
        "pair_bundle_sha256": None,
        "active_pattern_mask": [],
        "evaluation_schedule": [],
        "threshold_pattern_indices": [],
        "source_dimensions": None,
        "fresh_dimensions": None,
        "fresh_semantic_digest": None,
        "materialized_tightness_summary_sha256": None,
        "materialized_tightness_summary": None,
        "strong_tightness_gate": None,
        "pair_local_lp_actual_calls": 0,
        "conditional_local_lp_actual_calls": 0,
        "total_local_lp_actual_calls": 0,
        "conditional_checker_actual_calls": 0,
        "local_lp_actual_call_cap": 20,
        "conditional_checker_actual_call_cap": 34,
        "stage_resources": dict(stage_resources or {}),
        "timings": timing_receipt,
    })


def _pcoh_k3_exact_builtin_json(value: Any) -> bool:
    """Accept only finite JSON trees made from exact builtin types."""

    if value is None or type(value) in {bool, int, str}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_pcoh_k3_exact_builtin_json(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _pcoh_k3_exact_builtin_json(item)
            for key, item in value.items()
        )
    return False


def _pcoh_k3_resource_snapshot_valid(value: Any) -> bool:
    fields = {
        "peak_rss_bytes",
        "current_rss_bytes",
        "cuda_initialized",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    }
    if type(value) is not dict or set(value) != fields:
        return False
    peak = value["peak_rss_bytes"]
    current = value["current_rss_bytes"]
    cuda = value["cuda_initialized"]
    allocated = value["cuda_peak_allocated_bytes"]
    reserved = value["cuda_peak_reserved_bytes"]
    if (
        type(peak) is not int
        or peak < 0
        or (
            current is not None
            and (
                type(current) is not int
                or current < 0
                or current > peak
            )
        )
        or type(cuda) is not bool
    ):
        return False
    if cuda:
        return bool(
            type(allocated) is int
            and allocated >= 0
            and type(reserved) is int
            and reserved >= allocated
        )
    return allocated is None and reserved is None


def _pcoh_k3_timings_valid(value: Any, *, stop_loss: bool) -> bool:
    fixed = {
        "raw_batch_seconds",
        "focused_rival_seconds",
        "literal_selection_seconds",
        "k3_transaction_seconds",
        "total_seconds",
    }
    if type(value) is not dict or not value or "total_seconds" not in value:
        return False
    if (not stop_loss and set(value) != fixed) or (
        stop_loss and not set(value).issubset(fixed)
    ):
        return False
    for item in value.values():
        if (
            type(item) is not float
            or not math.isfinite(item)
            or item < 0.0
            or (item == 0.0 and math.copysign(1.0, item) < 0.0)
        ):
            return False
    return value["total_seconds"] >= max(value.values())


def _pcoh_k3_int_list(value: Any, *, length: int) -> bool:
    return bool(
        type(value) is list
        and len(value) == length
        and all(type(item) is int for item in value)
    )


def _pcoh_k3_transaction_structure_valid(
    value: Any,
    *,
    expected_outcome_sha256: Optional[str] = None,
    expected_outcome_kind: Optional[str] = None,
) -> bool:
    if type(value) is not dict or set(value) != _PCOH_K3_TRANSACTION_FIELDS:
        return False
    try:
        if not _pcoh_k3_exact_builtin_json(value):
            return False
        if not _local_receipt_checksum_valid(
            value, schema=_PCOH_K3_TRANSACTION_SCHEMA
        ):
            return False
        implementation = value.get("implementation_sha256")
        inputs = value.get("input_sha256")
        resources = value.get("stage_resources")
        timings = value.get("timings")
        counts = tuple(
            value.get(name)
            for name in (
                "pair_local_lp_actual_calls",
                "conditional_local_lp_actual_calls",
                "total_local_lp_actual_calls",
                "conditional_checker_actual_calls",
            )
        )
        common = bool(
            value.get("status")
            in {
                "strong_promotion",
                "strong_target_stop",
                "built_but_not_strong",
                "resource_stop",
                "stop_loss",
            }
            and value.get("diagnostic_only") is True
            and value.get("candidate_only") is True
            and value.get("build_only") is True
            and type(value.get("instance_count")) is int
            and value.get("instance_count") == 1
            and value.get("proof_authority") is False
            and value.get("verdict_authority") is False
            and value.get("provenance_authority") is False
            and value.get("authenticity_authority") is False
            and value.get("ground_truth_loaded") is False
            and value.get("reference_label_used") is False
            and value.get("k2_build_only_called") is False
            and value.get("phase_transaction_called") is False
            and value.get("solver_handoff_called") is False
            and value.get("hz_base_feasibility_called") is False
            and value.get("hz_objbound_decide_called") is False
            and value.get("full_parent_lp_called") is False
            and value.get("full_parent_lp_solver_called") is False
            and value.get("fresh_build_returned") is False
            and value.get("baseline_artifact_sha256")
            == _PCOH_K3_BASELINE_ARTIFACT_SHA256
            and value.get("baseline_summary_sha256")
            == _PCOH_K3_BASELINE_SUMMARY_SHA256
            and type(value.get("baseline_anchor_verified")) is bool
            and (
                value.get("baseline_anchor_receipt_sha256") is None
                or _valid_sha256(
                    value.get("baseline_anchor_receipt_sha256")
                )
            )
            and type(inputs) is dict
            and set(inputs) == {"onnx", "vnnlib", "instances_csv"}
            and all(_valid_sha256(item) for item in inputs.values())
            and type(implementation) is dict
            and set(implementation)
            == set(_PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS)
            and all(_valid_sha256(item) for item in implementation.values())
            and type(resources) is dict
            and bool(resources)
            and all(_pcoh_k3_resource_snapshot_valid(item)
                    for item in resources.values())
            and _pcoh_k3_timings_valid(
                timings, stop_loss=value.get("status") == "stop_loss"
            )
            and all(type(item) is int and item >= 0 for item in counts)
            and type(value.get("local_lp_actual_call_cap")) is int
            and value.get("local_lp_actual_call_cap") == 20
            and type(value.get("conditional_checker_actual_call_cap")) is int
            and value.get("conditional_checker_actual_call_cap") == 34
            and value.get("total_local_lp_actual_calls")
            == value.get("pair_local_lp_actual_calls")
            + value.get("conditional_local_lp_actual_calls")
            and value.get("total_local_lp_actual_calls") <= 20
            and value.get("conditional_checker_actual_calls") <= 34
        )
        if not common:
            return False
        if value["status"] == "stop_loss":
            null_fields = (
                "full_batch_sha256",
                "focused_subset_digest",
                "residual_selector_receipt_sha256",
                "focused_semantic_anchor",
                "focused_semantic_anchor_sha256",
                "focused_encoded_row",
                "focused_rival_id",
                "source_semantic_digest",
                "selection_digest",
                "selection_property_digest",
                "selection_parent_semantic_digest",
                "selection_operator_row_tag_digest",
                "stable_bit_selection_method",
                "third_stable_bit_id",
                "outcome_kind",
                "outcome_schema",
                "outcome_status",
                "trusted_outcome_sha256",
                "outcome_receipt_sha256",
                "detached_outcome",
                "resource_gate_rejection_sha256",
                "resource_gate_rejection",
                "pair_bundle_sha256",
                "source_dimensions",
                "fresh_dimensions",
                "fresh_semantic_digest",
                "materialized_tightness_summary_sha256",
                "materialized_tightness_summary",
                "strong_tightness_gate",
            )
            empty_list_fields = (
                "retained_k2_stable_bit_ids",
                "stable_bit_ids",
                "active_pattern_mask",
                "evaluation_schedule",
                "threshold_pattern_indices",
            )
            return bool(
                type(value.get("reason")) is str
                and bool(value["reason"])
                and type(value.get("failed_stage")) is str
                and bool(value["failed_stage"])
                and type(value.get("k3_transaction_called")) is bool
                and value.get("k3_same_process_verified") is False
                and value.get(
                    "trusted_outcome_digest_captured_before_detach"
                )
                is False
                and value.get("k3_detached_verified") is False
                and all(value.get(name) is None for name in null_fields)
                and all(
                    type(value.get(name)) is list
                    and not value[name]
                    for name in empty_list_fields
                )
                and (
                    (
                        value.get("baseline_anchor_verified") is True
                        and _valid_sha256(
                            value.get("baseline_anchor_receipt_sha256")
                        )
                    )
                    or (
                        value.get("baseline_anchor_verified") is False
                        and value.get("baseline_anchor_receipt_sha256")
                        is None
                    )
                )
                and counts == (0, 0, 0, 0)
                and expected_outcome_sha256 is None
                and expected_outcome_kind is None
            )
        detached = value.get("detached_outcome")
        receipt = (
            detached.get("receipt") if type(detached) is dict else None
        )
        execution = (
            detached.get("execution_telemetry")
            if type(detached) is dict
            else None
        )
        retained_ids = value.get("retained_k2_stable_bit_ids")
        stable_ids = value.get("stable_bit_ids")
        active_mask = value.get("active_pattern_mask")
        schedule = value.get("evaluation_schedule")
        thresholds = value.get("threshold_pattern_indices")
        canonical_patterns = set(itertools.product((-1, 1), repeat=3))
        if (
            not _valid_sha256(expected_outcome_sha256)
            or expected_outcome_kind not in {
                "success", "strong_target_stop", "resource_stop"
            }
            or (
                value.get("status") != "resource_stop"
                and (
                    value.get("reason") is not None
                    or value.get("failed_stage") is not None
                )
            )
            or value.get("k3_transaction_called") is not True
            or value.get("k3_same_process_verified") is not True
            or value.get(
                "trusted_outcome_digest_captured_before_detach"
            )
            is not True
            or value.get("k3_detached_verified") is not True
            or type(detached) is not dict
            or type(receipt) is not dict
            or value.get("trusted_outcome_sha256")
            != expected_outcome_sha256
            or value.get("outcome_kind") != expected_outcome_kind
            or value.get("outcome_receipt_sha256")
            != receipt.get("receipt_sha256")
            or not _valid_sha256(value.get("outcome_receipt_sha256"))
            or value.get("input_sha256")
            != {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            or value.get("full_batch_sha256")
            != _PCOH_K3_EXPECTED_FULL_BATCH_SHA256
            or value.get("baseline_anchor_verified") is not True
            or not _valid_sha256(
                value.get("baseline_anchor_receipt_sha256")
            )
            or not _valid_sha256(value.get("focused_subset_digest"))
            or not _valid_sha256(
                value.get("residual_selector_receipt_sha256")
            )
            or not _pcoh_k3_fixed_focused_semantic_anchor_valid(
                value.get("focused_semantic_anchor")
            )
            or value.get("focused_semantic_anchor_sha256")
            != _PCOH_K3_FOCUSED_SEMANTIC_ANCHOR_SHA256
            or value["focused_semantic_anchor"].get("semantic_sha256")
            != value.get("focused_semantic_anchor_sha256")
            or value.get("focused_encoded_row")
            != _PCOH_K3_FOCUSED_ENCODED_ROW
            or value.get("focused_rival_id") != _PCOH_K3_FOCUSED_RIVAL_ID
            or value.get("source_semantic_digest")
            != _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            or value.get("selection_digest")
            != _PCOH_K3_EXPECTED_SELECTION_DIGEST
            or value.get("selection_property_digest")
            != _PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
            or value.get("selection_parent_semantic_digest")
            != _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            or value.get("selection_operator_row_tag_digest")
            != _PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
            or value["focused_semantic_anchor"].get(
                "source_semantic_digest"
            ) != value.get("source_semantic_digest")
            or value["focused_semantic_anchor"].get("full_batch_sha256")
            != value.get("full_batch_sha256")
            or value["focused_semantic_anchor"].get("focused_encoded_row")
            != value.get("focused_encoded_row")
            or value["focused_semantic_anchor"].get("focused_rival_id")
            != value.get("focused_rival_id")
            or value["focused_semantic_anchor"].get("selection_digest")
            != value.get("selection_digest")
            or value["focused_semantic_anchor"].get(
                "selection_property_digest"
            ) != value.get("selection_property_digest")
            or value["focused_semantic_anchor"].get(
                "selection_parent_semantic_digest"
            ) != value.get("selection_parent_semantic_digest")
            or value["focused_semantic_anchor"].get(
                "selection_operator_row_tag_digest"
            ) != value.get("selection_operator_row_tag_digest")
            or value.get("stable_bit_selection_method")
            != "retain_fixed_k2_pair_then_exact_focused_third"
            or not _pcoh_k3_int_list(retained_ids, length=2)
            or retained_ids != list(_PCOH_K3_RETAINED_K2_STABLE_BIT_IDS)
            or not _pcoh_k3_int_list(stable_ids, length=3)
            or stable_ids[:2]
            != list(_PCOH_K3_RETAINED_K2_STABLE_BIT_IDS)
            or len(set(stable_ids)) != 3
            or type(value.get("third_stable_bit_id")) is not int
            or value.get("third_stable_bit_id")
            != stable_ids[2]
            or type(active_mask) is not list
            or len(active_mask) != 8
            or any(type(item) is not bool for item in active_mask)
            or not any(active_mask)
            or type(schedule) is not list
            or len(schedule) != 8
            or any(
                type(pattern) is not list
                or len(pattern) != 3
                or any(type(phase) is not int or phase not in {-1, 1}
                       for phase in pattern)
                for pattern in schedule
            )
            or {tuple(pattern) for pattern in schedule} != canonical_patterns
            or type(thresholds) is not list
            or any(
                type(index) is not int or index < 0 or index >= 8
                for index in thresholds
            )
            or len(set(thresholds)) != len(thresholds)
            or detached.get("source_semantic_digest")
            != value.get("source_semantic_digest")
            or detached.get("focused_rival_id")
            != value.get("focused_rival_id")
            or detached.get("retained_k2_stable_bit_ids") != retained_ids
            or detached.get("stable_bit_ids") != stable_ids
            or detached.get("third_stable_bit_id")
            != value.get("third_stable_bit_id")
            or detached.get("pair_bundle_sha256")
            != value.get("pair_bundle_sha256")
            or detached.get("active_pattern_mask") != active_mask
            or detached.get("evaluation_schedule") != schedule
            or detached.get("threshold_pattern_indices") != thresholds
            or value.get("pair_local_lp_actual_calls") != 12
            or type(execution) is not dict
            or value.get("conditional_local_lp_actual_calls")
            != execution.get("scheduled_local_lp_actual_calls")
            or value.get("total_local_lp_actual_calls")
            != execution.get("local_lp_actual_calls")
            or value.get("conditional_checker_actual_calls")
            != execution.get("conditional_checker_actual_calls")
            or receipt.get("proof_authority") is not False
            or receipt.get("verdict_authority") is not False
            or receipt.get("full_parent_lp_called") is not False
        ):
            return False
        from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
            verify_detached_phase_conditioned_objective_hull_k3_build_only,
        )
        if not verify_detached_phase_conditioned_objective_hull_k3_build_only(
            detached, expected_sha256=expected_outcome_sha256
        ):
            return False
        if value["status"] == "resource_stop":
            resource_receipt = detached.get("receipt")
            rejection = (
                resource_receipt.get("resource_gate_rejection")
                if type(resource_receipt) is dict
                else None
            )
            stage = detached.get("stage")
            reason = detached.get("reason")
            scheduled_bundle = detached.get("scheduled_bundle_sha256")
            completed = detached.get(
                "completed_conditional_certificate_count"
            )
            return bool(
                expected_outcome_kind == "resource_stop"
                and type(value.get("status")) is str
                and value.get("outcome_kind") == "resource_stop"
                and type(value.get("outcome_kind")) is str
                and value.get("outcome_schema")
                == "act.hybridz_pcoh_k3_build_only_resource_stop.v1"
                and type(value.get("outcome_schema")) is str
                and value.get("outcome_status")
                == "stopped_by_resource_gate_no_partial_output"
                and type(value.get("outcome_status")) is str
                and stage in {
                    "pre_scheduled", "pre_fresh_materialization"
                }
                and type(stage) is str
                and type(reason) is str
                and bool(reason)
                and type(value.get("failed_stage")) is str
                and value.get("failed_stage") == stage
                and type(value.get("reason")) is str
                and value.get("reason") == reason
                and resource_receipt.get("stage") == stage
                and resource_receipt.get("reason") == reason
                and type(rejection) is dict
                and value.get("resource_gate_rejection") == rejection
                and _valid_sha256(rejection.get("rejection_sha256"))
                and value.get("resource_gate_rejection_sha256")
                == rejection.get("rejection_sha256")
                and detached.get("fresh_issue_called") is False
                and detached.get("fresh_build_returned") is False
                and detached.get("fresh_descriptor_returned") is False
                and detached.get("partial_certificates_returned") is False
                and detached.get(
                    "conditional_certificate_payload_returned"
                ) is False
                and detached.get("provenance_authority") is False
                and detached.get("authenticity_authority") is False
                and value.get("fresh_dimensions") is None
                and value.get("fresh_semantic_digest") is None
                and value.get(
                    "materialized_tightness_summary_sha256"
                ) is None
                and value.get("materialized_tightness_summary") is None
                and value.get("strong_tightness_gate") is None
                and type(value.get("source_dimensions")) is list
                and len(value["source_dimensions"]) == 5
                and all(
                    type(item) is int and item >= 0
                    for item in value["source_dimensions"]
                )
                and value["source_dimensions"]
                == resource_receipt.get("source_dimensions")
                and (
                    (
                        stage == "pre_scheduled"
                        and scheduled_bundle is None
                        and completed == 0
                        and value.get(
                            "conditional_local_lp_actual_calls"
                        ) == 0
                        and value.get("total_local_lp_actual_calls") == 12
                        and value.get(
                            "conditional_checker_actual_calls"
                        ) == 0
                    )
                    or (
                        stage == "pre_fresh_materialization"
                        and _valid_sha256(scheduled_bundle)
                        and completed == 8
                        and 12
                        <= value.get("total_local_lp_actual_calls")
                        <= 20
                    )
                )
                and set(resources)
                == {
                    "entry", "baseline_artifact", "raw_batch",
                    "focused_rival", "literal_selection",
                    "k3_transaction",
                }
            )
        if (
            value.get("resource_gate_rejection_sha256") is not None
            or value.get("resource_gate_rejection") is not None
        ):
            return False
        if value["status"] == "strong_target_stop":
            return bool(
                expected_outcome_kind == "strong_target_stop"
                and value.get("outcome_schema")
                == "act.hybridz_pcoh_k3_build_only_stop.v1"
                and value.get("outcome_status")
                == "stopped_by_strong_target_no_partial_output"
                and detached.get("fresh_issue_called") is False
                and detached.get("partial_certificates_returned") is False
                and value.get("fresh_dimensions") is None
                and value.get("fresh_semantic_digest") is None
                and value.get(
                    "materialized_tightness_summary_sha256"
                ) is None
                and value.get("materialized_tightness_summary") is None
                and value.get("strong_tightness_gate") is None
                and value.get("source_dimensions")
                == detached.get("source_dimensions")
                and value.get("source_dimensions")
                == receipt.get("source_dimensions")
                and set(resources)
                == {
                    "entry", "baseline_artifact", "raw_batch", "focused_rival",
                    "literal_selection", "k3_transaction",
                }
            )
        summary = value.get("materialized_tightness_summary")
        gate = value.get("strong_tightness_gate")
        source_dimensions = value.get("source_dimensions")
        fresh_dimensions = value.get("fresh_dimensions")
        expected_status = (
            "strong_promotion"
            if gate.get("strong_candidate") is True
            else "built_but_not_strong"
        ) if type(gate) is dict else None
        return bool(
            expected_outcome_kind == "success"
            and value["status"] == expected_status
            and value.get("outcome_schema")
            == "act.hybridz_pcoh_k3_build_only_diagnostic.v1"
            and value.get("outcome_status")
            == "k3_build_only_materialized_validated_consumed_and_released"
            and type(summary) is dict
            and summary.get("summary_sha256")
            == value.get("materialized_tightness_summary_sha256")
            and summary == detached.get("materialized_tightness_summary")
            and _pcoh_k3_strong_tightness_gate_valid(
                gate,
                summary,
                source_semantic_digest=value["source_semantic_digest"],
                selection_digest=value["selection_digest"],
                focused_encoded_row=value["focused_encoded_row"],
                focused_rival_id=value["focused_rival_id"],
                retained_k2_stable_bit_ids=value[
                    "retained_k2_stable_bit_ids"
                ],
                stable_bit_ids=value["stable_bit_ids"],
            )
            and type(source_dimensions) is list
            and len(source_dimensions) == 5
            and all(type(item) is int and item >= 0
                    for item in source_dimensions)
            and source_dimensions == detached.get("source_dimensions")
            and type(fresh_dimensions) is list
            and len(fresh_dimensions) == 5
            and all(type(item) is int and item >= 0
                    for item in fresh_dimensions)
            and fresh_dimensions == detached.get("fresh_dimensions")
            and fresh_dimensions[0] == source_dimensions[0]
            and fresh_dimensions[1] == source_dimensions[1] + 8
            and fresh_dimensions[2] == source_dimensions[2]
            and fresh_dimensions[3]
            == source_dimensions[3]
            + 4
            + value["active_pattern_mask"].count(False)
            and fresh_dimensions[4] == source_dimensions[4] + 1
            and _valid_sha256(value.get("fresh_semantic_digest"))
            and value.get("fresh_semantic_digest")
            == detached.get("fresh_semantic_digest")
            and set(resources)
            == {
                "entry", "baseline_artifact", "raw_batch", "focused_rival",
                "literal_selection", "k3_transaction",
            }
        )
    except (
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        RuntimeError,
        PhaseCliqueBuildProbeError,
    ):
        return False


def _pcoh_k3_transaction_receipt_valid(value: Any) -> bool:
    """Validate structure and, for trusted outcomes, the live local anchor."""

    if type(value) is not dict:
        return False
    if value.get("status") == "stop_loss":
        return bool(
            _pcoh_k3_trusted_transaction_anchor(value) is None
            and _pcoh_k3_transaction_structure_valid(value)
        )
    anchor = _pcoh_k3_trusted_transaction_anchor(value)
    return bool(
        type(anchor) is _PCOHK3TrustedTransactionAnchor
        and _pcoh_k3_transaction_structure_valid(
            value,
            expected_outcome_sha256=anchor.outcome_sha256,
            expected_outcome_kind=anchor.outcome_kind,
        )
    )


def _pcoh_k3_transaction_basic_receipt_valid(value: Any) -> bool:
    """Accept a safe self-signed envelope without granting live authority."""

    try:
        return bool(
            type(value) is dict
            and set(value) == _PCOH_K3_TRANSACTION_FIELDS
            and _pcoh_k3_exact_builtin_json(value)
            and _local_receipt_checksum_valid(
                value, schema=_PCOH_K3_TRANSACTION_SCHEMA
            )
        )
    except BaseException as exc:
        _clear_pcoh_k3_exception_traceback(exc)
        return False


def _adopt_pcoh_k3_trusted_transaction(
    body: dict[str, Any], transaction: Any
) -> None:
    diagnostically_safe = _pcoh_k3_transaction_basic_receipt_valid(
        transaction
    )
    if diagnostically_safe:
        body["pcoh_k3_build_only"] = transaction
    try:
        if not _pcoh_k3_transaction_receipt_valid(transaction):
            raise PhaseCliqueBuildProbeError(
                "K3 transaction returned malformed receipt"
            )
    except BaseException:
        _release_pcoh_k3_trusted_transaction(transaction)
        transaction = None
        raise


def _run_pcoh_k2_build_only_pipeline(
    source_build: Any,
    *,
    input_sha256: Mapping[str, Any],
    implementation_sha256: Mapping[str, Any],
    vnnlib_path: Any,
    expected_vnnlib_sha256: str,
    live_assert_params: Any,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    residual_selector_receipt: Mapping[str, Any],
    residual_selector_property_sha256: str,
    deadline: float,
    phase_time_limit: float,
    torch_module: Any,
) -> dict[str, Any]:
    """Run exactly one receipt-only PCOH K2 build transaction."""

    from act.back_end.hybridz_tf.adaptive_phase_forest import (
        sparse_hz_semantic_digest,
    )
    from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
        derive_operator_exact_relu_property_phase_literals,
        verify_operator_exact_relu_property_phase_selection,
    )
    from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
        _exact_interval_upper_violations,
        _interval_frame_digest,
        _snapshot_b1_bounds,
    )
    from act.back_end.hybridz_tf.operator_phase_conditioned_build_only import (
        run_phase_conditioned_objective_hull_build_only,
        verify_phase_conditioned_objective_hull_build_only_diagnostic,
    )
    from act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge import (
        issue_raw_rival_exact_hardness_receipt,
        select_raw_focused_rivals,
        verify_raw_focused_rival_selection,
        verify_raw_rival_exact_hardness_receipt,
    )
    from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
        consume_raw_vnnlib_top1_candidate,
        issue_raw_vnnlib_top1_candidate,
        validate_consumed_raw_vnnlib_rival_batch,
    )

    started = time.monotonic()
    stage_resources: dict[str, Any] = {
        "entry": _capture_resource_peaks(torch_module)
    }
    timings: dict[str, float] = {}
    build_only_transaction_called = False
    stage = "input_validation"
    try:
        if (
            not _valid_sha256(expected_vnnlib_sha256)
            or type(input_sha256) is not dict
            or set(input_sha256) != {"onnx", "vnnlib", "instances_csv"}
            or any(not _valid_sha256(value) for value in input_sha256.values())
            or input_sha256["vnnlib"] != expected_vnnlib_sha256
            or input_sha256
            != {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            or type(implementation_sha256) is not dict
            or not implementation_sha256
            or set(implementation_sha256)
            != set(_IMPLEMENTATION_RELATIVE_PATHS)
            or any(
                type(path) is not str or not _valid_sha256(value)
                for path, value in implementation_sha256.items()
            )
            or type(residual_selector_receipt) is not dict
            or not _valid_sha256(residual_selector_property_sha256)
            or type(phase_time_limit) is not float
            or not math.isfinite(phase_time_limit)
            or phase_time_limit != _PCOH_K2_MAX_PHASE_SECONDS
        ):
            raise PhaseCliqueBuildProbeError("pcoh_input_binding_invalid")
        focused_encoded_row = residual_selector_receipt.get(
            "joint_focus_rival_id"
        )
        if type(focused_encoded_row) is not int:
            raise PhaseCliqueBuildProbeError(
                "pcoh_residual_joint_focus_row_invalid"
            )
        now = time.monotonic()
        phase_deadline = min(deadline, now + float(phase_time_limit))
        if phase_deadline <= now:
            raise TimeoutError("pcoh_phase_budget_unavailable")
        source_digest = sparse_hz_semantic_digest(source_build.hz)

        stage = "raw_vnnlib_batch_issue_consume"
        stage_started = time.monotonic()
        raw_candidate = issue_raw_vnnlib_top1_candidate(
            vnnlib_path,
            expected_vnnlib_sha256=expected_vnnlib_sha256,
            live_assert_params=live_assert_params,
            deadline=phase_deadline,
        )
        batch = consume_raw_vnnlib_top1_candidate(
            raw_candidate,
            live_assert_params=live_assert_params,
            deadline=phase_deadline,
        )
        if not validate_consumed_raw_vnnlib_rival_batch(batch):
            raise PhaseCliqueBuildProbeError("pcoh_raw_batch_invalid")
        timings["raw_batch_seconds"] = float(
            time.monotonic() - stage_started
        )
        stage_resources["raw_batch"] = _capture_resource_peaks(torch_module)

        stage = "focused_rival_exact_interval_binding"
        stage_started = time.monotonic()
        lower, upper = _snapshot_b1_bounds(
            output_lower,
            output_upper,
            output_width=int(source_build.hz.n_out),
        )
        exact_hardness = _exact_interval_upper_violations(
            batch.rivals,
            lower.reshape(-1),
            upper.reshape(-1),
            deadline=phase_deadline,
        )
        interval_digest = _interval_frame_digest(
            build_digest=source_digest,
            batch_sha256=batch.batch_sha256,
            live_assert_sha256=batch.live_assert_sha256,
            property_digest=residual_selector_property_sha256,
            lower=lower,
            upper=upper,
        )
        hardness = issue_raw_rival_exact_hardness_receipt(
            batch,
            exact_hardness,
            live_interval_bounds_sha256=interval_digest,
            deadline=phase_deadline,
            max_rivals=256,
            max_focus=1,
            max_exact_bits=4096,
            max_work_items=5_000_000,
        )
        focused = select_raw_focused_rivals(
            batch,
            hardness,
            focus_count=1,
            explicit_encoded_focus_row=focused_encoded_row,
            residual_selector_receipt=residual_selector_receipt,
            residual_selector_property_sha256=(
                residual_selector_property_sha256
            ),
            expected_exact_upper_violations=exact_hardness,
            expected_live_interval_bounds_sha256=interval_digest,
            deadline=phase_deadline,
            max_rivals=256,
            max_focus=1,
            max_exact_bits=4096,
            max_work_items=5_000_000,
        )
        if (
            verify_raw_rival_exact_hardness_receipt(
                batch,
                hardness,
                expected_exact_upper_violations=exact_hardness,
                expected_live_interval_bounds_sha256=interval_digest,
                deadline=phase_deadline,
                max_rivals=256,
                max_focus=1,
                max_exact_bits=4096,
                max_work_items=5_000_000,
            )
            is not True
            or verify_raw_focused_rival_selection(
                batch,
                hardness,
                focused,
                expected_focus_count=1,
                expected_exact_upper_violations=exact_hardness,
                expected_live_interval_bounds_sha256=interval_digest,
                deadline=phase_deadline,
                max_rivals=256,
                max_focus=1,
                max_exact_bits=4096,
                max_work_items=5_000_000,
            )
            is not True
            or type(focused.rivals) is not tuple
            or len(focused.rivals) != 1
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_focused_rival_verification_failed"
            )
        focused_rival_id = focused.rivals[0].rival_id
        if type(focused_rival_id) is not int or focused_rival_id < 0:
            raise PhaseCliqueBuildProbeError(
                "pcoh_focused_rival_id_invalid"
            )
        timings["focused_rival_seconds"] = float(
            time.monotonic() - stage_started
        )
        stage_resources["focused_rival"] = _capture_resource_peaks(
            torch_module
        )

        stage = "verified_literal_selection"
        stage_started = time.monotonic()
        remaining = phase_deadline - time.monotonic()
        selection_seconds = min(
            _PCOH_K2_SELECTION_SECONDS, remaining / 3.0
        )
        if selection_seconds <= 0.0:
            raise TimeoutError("pcoh_selection_budget_unavailable")
        selection = derive_operator_exact_relu_property_phase_literals(
            source_build,
            focused.rivals,
            max_rivals=1,
            max_binaries=4,
            max_work_items=_PCOH_K2_SELECTION_WORK_ITEMS,
            timeout_seconds=selection_seconds,
        )
        if not verify_operator_exact_relu_property_phase_selection(
            source_build,
            focused.rivals,
            selection,
            max_rivals=1,
            max_binaries=4,
            max_work_items=_PCOH_K2_SELECTION_WORK_ITEMS,
            timeout_seconds=selection_seconds,
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_literal_selection_verification_failed"
            )
        canonical_ids = tuple(
            sorted(mapping.stable_bcol_id for mapping in selection.mappings)
        )
        if (
            len(canonical_ids) < 2
            or len(set(canonical_ids)) != len(canonical_ids)
            or any(type(value) is not int or value < 0 for value in canonical_ids)
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_verified_selection_has_fewer_than_two_ids"
            )
        stable_ids = canonical_ids[:2]
        timings["literal_selection_seconds"] = float(
            time.monotonic() - stage_started
        )
        stage_resources["literal_selection"] = _capture_resource_peaks(
            torch_module
        )

        stage = "no_verdict_build_only_transaction"
        stage_started = time.monotonic()
        build_only_transaction_called = True
        diagnostic = run_phase_conditioned_objective_hull_build_only(
            source_build,
            focused.rivals,
            selection,
            focused_rival_id=focused_rival_id,
            stable_bit_ids=stable_ids,
            deadline=phase_deadline,
        )
        if not verify_phase_conditioned_objective_hull_build_only_diagnostic(
            diagnostic
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_build_only_diagnostic_verification_failed"
            )
        live_summary = diagnostic.materialized_tightness_summary
        live_receipt = diagnostic.receipt
        summary_anchor = live_receipt.get(
            "materialized_tightness_summary_sha256"
        )
        if (
            diagnostic.schema
            != "act.hybridz_pcoh_build_only_diagnostic.v2"
            or diagnostic.full_parent_lp_called is not False
            or not isinstance(live_summary, Mapping)
            or not isinstance(live_receipt, Mapping)
            or live_receipt.get("materialized_tightness_summary")
            is not live_summary
            or live_summary.get("summary_sha256") != summary_anchor
            or not _valid_sha256(summary_anchor)
            or live_receipt.get("full_parent_lp_called") is not False
            or live_receipt.get("full_parent_lp_solver_called") is not False
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_build_only_tightness_anchor_or_lp_firewall_failed"
            )
        materialized_tightness_summary = _builtin_receipt_mapping(
            live_summary
        )
        transaction_receipt = _builtin_receipt_mapping(diagnostic.receipt)
        if (
            transaction_receipt is None
            or materialized_tightness_summary is None
            or transaction_receipt.get(
                "materialized_tightness_summary_sha256"
            )
            != summary_anchor
            or transaction_receipt.get("full_parent_lp_called") is not False
            or transaction_receipt.get("full_parent_lp_solver_called")
            is not False
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_transaction_receipt_not_json_safe"
            )
        tightness_gate = _pcoh_k2_tightness_gate(
            materialized_tightness_summary,
            expected_summary_sha256=summary_anchor,
        )
        timings["build_only_transaction_seconds"] = float(
            time.monotonic() - stage_started
        )
        timings["total_seconds"] = float(time.monotonic() - started)
        stage_resources["build_only_transaction"] = _capture_resource_peaks(
            torch_module
        )
        result = _checksummed({
            "schema": _PCOH_K2_TRANSACTION_SCHEMA,
            "status": "built_and_released",
            "reason": None,
            "failed_stage": None,
            "diagnostic_only": True,
            "candidate_only": True,
            "build_only": True,
            "instance_count": 1,
            "proof_authority": False,
            "verdict_authority": False,
            "ground_truth_loaded": False,
            "reference_label_used": False,
            "build_only_transaction_called": True,
            "solver_handoff_called": False,
            "diagnostic_lp_called": False,
            "hz_base_feasibility_called": False,
            "hz_objbound_decide_called": False,
            "strict_replay_called": False,
            "fresh_build_returned": False,
            "full_parent_lp_called": False,
            "full_parent_lp_solver_called": False,
            "transaction_verified_before_serialization": True,
            "input_sha256": dict(input_sha256),
            "implementation_sha256": dict(implementation_sha256),
            "full_batch_sha256": batch.batch_sha256,
            "focused_subset_digest": focused.focused_subset_digest,
            "focused_encoded_row": focused_encoded_row,
            "focused_rival_id": focused_rival_id,
            "successful_selection_binding_retained": True,
            "selection_digest": selection.selection_digest,
            "selection_property_digest": selection.property_digest,
            "selection_parent_semantic_digest": (
                selection.parent_semantic_digest
            ),
            "selection_operator_row_tag_digest": (
                selection.operator_row_tag_digest
            ),
            "stable_bit_selection_method": (
                "lowest_two_canonical_ids_from_verified_selection"
            ),
            "stable_bit_ids": list(stable_ids),
            "diagnostic_schema": diagnostic.schema,
            "diagnostic_sha256": diagnostic.diagnostic_sha256,
            "transaction_receipt_sha256": transaction_receipt[
                "receipt_sha256"
            ],
            "source_semantic_digest": diagnostic.source_semantic_digest,
            "fresh_semantic_digest": diagnostic.fresh_semantic_digest,
            "source_dimensions": list(diagnostic.source_dimensions),
            "fresh_dimensions": list(diagnostic.fresh_dimensions),
            "conditional_certificate_sha256": list(
                diagnostic.conditional_certificate_sha256
            ),
            "pair_bundle_sha256": diagnostic.pair_bundle_sha256,
            "fresh_issuance_sha256": diagnostic.fresh_issuance_sha256,
            "materialized_tightness_summary_sha256": summary_anchor,
            "materialized_tightness_summary": (
                materialized_tightness_summary
            ),
            "tightness_gate": tightness_gate,
            "resource_preflight": transaction_receipt[
                "resource_preflight"
            ],
            "resource_postflight": transaction_receipt[
                "resource_postflight"
            ],
            "stage_resources": dict(stage_resources),
            "timings": timings,
        })
        _register_pcoh_k2_trusted_transaction(
            result, trusted_summary_sha256=summary_anchor
        )
        try:
            if not _pcoh_k2_transaction_receipt_valid(result):
                raise PhaseCliqueBuildProbeError(
                    "pcoh_probe_transaction_receipt_self_check_failed"
                )
        except BaseException:
            _release_pcoh_k2_trusted_transaction(result)
            raise
        return result
    except Exception as exc:
        stage_resources["stop_loss"] = _capture_resource_peaks(torch_module)
        return _pcoh_k2_stop_loss_receipt(
            stage=stage,
            reason=f"{type(exc).__name__}:{str(exc)[:240]}",
            started=started,
            input_sha256=input_sha256,
            implementation_sha256=implementation_sha256,
            stage_resources=stage_resources,
            timings=timings,
            build_only_transaction_called=build_only_transaction_called,
        )


def _clear_pcoh_k3_exception_traceback(exc: BaseException) -> None:
    cursor = exc.__traceback__
    while cursor is not None:
        frame = cursor.tb_frame
        cursor = cursor.tb_next
        try:
            frame.clear()
        except RuntimeError:
            pass
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None


def _run_pcoh_k3_build_only_pipeline(
    source_build: Any,
    *,
    input_sha256: Mapping[str, Any],
    implementation_sha256: Mapping[str, Any],
    vnnlib_path: Any,
    expected_vnnlib_sha256: str,
    live_assert_params: Any,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    residual_selector_receipt: Mapping[str, Any],
    residual_selector_property_sha256: str,
    deadline: float,
    phase_time_limit: float,
    torch_module: Any,
    baseline_anchor_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the fixed-anchor pair-first K3 diagnostic transaction."""

    from act.back_end.hybridz_tf.adaptive_phase_forest import (
        sparse_hz_semantic_digest,
    )
    from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
        derive_operator_exact_relu_property_phase_literals,
        verify_operator_exact_relu_property_phase_selection,
    )
    from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
        _exact_interval_upper_violations,
        _interval_frame_digest,
        _snapshot_b1_bounds,
    )
    from act.back_end.hybridz_tf.operator_phase_conditioned_k3_build_only import (
        PCOHK3BuildOnlyDiagnostic,
        PCOHK3BuildOnlyResourceStopDiagnostic,
        PCOHK3BuildOnlyStopDiagnostic,
        export_phase_conditioned_objective_hull_k3_build_only_detached,
        run_phase_conditioned_objective_hull_k3_build_only,
        verify_detached_phase_conditioned_objective_hull_k3_build_only,
        verify_phase_conditioned_objective_hull_k3_build_only_outcome,
    )
    from act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge import (
        issue_raw_rival_exact_hardness_receipt,
        select_raw_focused_rivals,
        verify_raw_focused_rival_selection,
        verify_raw_rival_exact_hardness_receipt,
    )
    from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
        consume_raw_vnnlib_top1_candidate,
        issue_raw_vnnlib_top1_candidate,
        validate_consumed_raw_vnnlib_rival_batch,
    )

    started = time.monotonic()
    stage = "input_validation"
    stage_resources: dict[str, Any] = {
        "entry": _capture_resource_peaks(torch_module)
    }
    timings: dict[str, float] = {}
    k3_transaction_called = False
    registered_transaction = None
    handed_off = False
    outcome = None
    try:
        if (
            input_sha256
            != {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            or expected_vnnlib_sha256
            != _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256
            or type(implementation_sha256) is not dict
            or set(implementation_sha256)
            != set(_PCOH_K3_IMPLEMENTATION_RELATIVE_PATHS)
            or any(
                type(path) is not str or not _valid_sha256(digest)
                for path, digest in implementation_sha256.items()
            )
            or type(residual_selector_receipt) is not dict
            or not _pcoh_k3_baseline_anchor_receipt_valid(
                baseline_anchor_receipt
            )
            or not _valid_sha256(residual_selector_property_sha256)
            or type(phase_time_limit) is not float
            or phase_time_limit != _PCOH_K3_INTERNAL_PHASE_SECONDS
        ):
            raise PhaseCliqueBuildProbeError("pcoh_k3_input_binding_invalid")
        stage_resources["baseline_artifact"] = _capture_resource_peaks(
            torch_module
        )
        focused_encoded_row = residual_selector_receipt.get(
            "joint_focus_rival_id"
        )
        if focused_encoded_row != _PCOH_K3_FOCUSED_ENCODED_ROW:
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_fixed_focused_encoded_row_mismatch"
            )
        now = time.monotonic()
        phase_deadline = min(deadline, now + _PCOH_K3_INTERNAL_PHASE_SECONDS)
        if phase_deadline <= now:
            raise TimeoutError("pcoh_k3_phase_budget_unavailable")
        source_digest = sparse_hz_semantic_digest(source_build.hz)
        if source_digest != _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST:
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_fixed_source_semantic_digest_mismatch"
            )

        stage = "raw_vnnlib_batch_issue_consume"
        stage_started = time.monotonic()
        raw_candidate = issue_raw_vnnlib_top1_candidate(
            vnnlib_path,
            expected_vnnlib_sha256=expected_vnnlib_sha256,
            live_assert_params=live_assert_params,
            deadline=phase_deadline,
        )
        batch = consume_raw_vnnlib_top1_candidate(
            raw_candidate,
            live_assert_params=live_assert_params,
            deadline=phase_deadline,
        )
        raw_candidate = None
        if (
            not validate_consumed_raw_vnnlib_rival_batch(batch)
            or batch.batch_sha256 != _PCOH_K3_EXPECTED_FULL_BATCH_SHA256
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_fixed_raw_batch_anchor_mismatch"
            )
        timings["raw_batch_seconds"] = float(
            time.monotonic() - stage_started
        )
        stage_resources["raw_batch"] = _capture_resource_peaks(torch_module)

        stage = "focused_rival_exact_interval_binding"
        stage_started = time.monotonic()
        lower, upper = _snapshot_b1_bounds(
            output_lower,
            output_upper,
            output_width=int(source_build.hz.n_out),
        )
        exact_hardness = _exact_interval_upper_violations(
            batch.rivals,
            lower.reshape(-1),
            upper.reshape(-1),
            deadline=phase_deadline,
        )
        interval_digest = _interval_frame_digest(
            build_digest=source_digest,
            batch_sha256=batch.batch_sha256,
            live_assert_sha256=batch.live_assert_sha256,
            property_digest=residual_selector_property_sha256,
            lower=lower,
            upper=upper,
        )
        hardness = issue_raw_rival_exact_hardness_receipt(
            batch,
            exact_hardness,
            live_interval_bounds_sha256=interval_digest,
            deadline=phase_deadline,
            max_rivals=256,
            max_focus=1,
            max_exact_bits=4096,
            max_work_items=5_000_000,
        )
        focused = select_raw_focused_rivals(
            batch,
            hardness,
            focus_count=1,
            explicit_encoded_focus_row=focused_encoded_row,
            residual_selector_receipt=residual_selector_receipt,
            residual_selector_property_sha256=(
                residual_selector_property_sha256
            ),
            expected_exact_upper_violations=exact_hardness,
            expected_live_interval_bounds_sha256=interval_digest,
            deadline=phase_deadline,
            max_rivals=256,
            max_focus=1,
            max_exact_bits=4096,
            max_work_items=5_000_000,
        )
        if (
            verify_raw_rival_exact_hardness_receipt(
                batch,
                hardness,
                expected_exact_upper_violations=exact_hardness,
                expected_live_interval_bounds_sha256=interval_digest,
                deadline=phase_deadline,
                max_rivals=256,
                max_focus=1,
                max_exact_bits=4096,
                max_work_items=5_000_000,
            )
            is not True
            or verify_raw_focused_rival_selection(
                batch,
                hardness,
                focused,
                expected_focus_count=1,
                expected_exact_upper_violations=exact_hardness,
                expected_live_interval_bounds_sha256=interval_digest,
                deadline=phase_deadline,
                max_rivals=256,
                max_focus=1,
                max_exact_bits=4096,
                max_work_items=5_000_000,
            )
            is not True
            or type(focused.rivals) is not tuple
            or len(focused.rivals) != 1
            or focused.rivals[0].rival_id != _PCOH_K3_FOCUSED_RIVAL_ID
            or focused.method != _PCOH_K3_FOCUS_METHOD
            or focused.focus_count != _PCOH_K3_FOCUS_COUNT
            or not _valid_sha256(focused.focused_subset_digest)
            or not _valid_sha256(
                focused.residual_selector_receipt_sha256
            )
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_fixed_focused_rival_anchor_mismatch"
            )
        focused_rival_id = focused.rivals[0].rival_id
        timings["focused_rival_seconds"] = float(
            time.monotonic() - stage_started
        )
        stage_resources["focused_rival"] = _capture_resource_peaks(
            torch_module
        )

        stage = "verified_literal_selection_max4"
        stage_started = time.monotonic()
        remaining = phase_deadline - time.monotonic()
        selection_seconds = min(_PCOH_K3_SELECTION_SECONDS, remaining / 3.0)
        if selection_seconds <= 0.0:
            raise TimeoutError("pcoh_k3_selection_budget_unavailable")
        selection = derive_operator_exact_relu_property_phase_literals(
            source_build,
            focused.rivals,
            max_rivals=1,
            max_binaries=4,
            max_work_items=_PCOH_K3_SELECTION_WORK_ITEMS,
            timeout_seconds=selection_seconds,
        )
        if not verify_operator_exact_relu_property_phase_selection(
            source_build,
            focused.rivals,
            selection,
            max_rivals=1,
            max_binaries=4,
            max_work_items=_PCOH_K3_SELECTION_WORK_ITEMS,
            timeout_seconds=selection_seconds,
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_literal_selection_verification_failed"
            )
        canonical_ids = tuple(
            sorted(mapping.stable_bcol_id for mapping in selection.mappings)
        )
        if (
            selection.selection_digest != _PCOH_K3_EXPECTED_SELECTION_DIGEST
            or selection.property_digest
            != _PCOH_K3_EXPECTED_SELECTION_PROPERTY_DIGEST
            or selection.parent_semantic_digest
            != _PCOH_K3_EXPECTED_SOURCE_SEMANTIC_DIGEST
            or selection.operator_row_tag_digest
            != _PCOH_K3_EXPECTED_SELECTION_OPERATOR_ROW_TAG_DIGEST
            or len(canonical_ids) < 3
            or len(canonical_ids) > 4
            or len(set(canonical_ids)) != len(canonical_ids)
            or canonical_ids[:2] != _PCOH_K3_RETAINED_K2_STABLE_BIT_IDS
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_fixed_selection_or_retained_ids_mismatch"
            )
        focused_semantic_anchor = _pcoh_k3_focused_semantic_anchor(
            source_semantic_digest=source_digest,
            full_batch_sha256=batch.batch_sha256,
            focused_encoded_row=focused_encoded_row,
            focused_rival_id=focused_rival_id,
            selection_digest=selection.selection_digest,
            selection_property_digest=selection.property_digest,
            selection_parent_semantic_digest=(
                selection.parent_semantic_digest
            ),
            selection_operator_row_tag_digest=(
                selection.operator_row_tag_digest
            ),
        )
        if (
            not _pcoh_k3_fixed_focused_semantic_anchor_valid(
                focused_semantic_anchor
            )
            or focused_semantic_anchor["semantic_sha256"]
            != baseline_anchor_receipt[
                "focused_semantic_anchor_sha256"
            ]
            or focused_semantic_anchor
            != baseline_anchor_receipt["focused_semantic_anchor"]
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_focused_semantic_anchor_mismatch"
            )
        timings["literal_selection_seconds"] = float(
            time.monotonic() - stage_started
        )
        stage_resources["literal_selection"] = _capture_resource_peaks(
            torch_module
        )

        stage = "k3_build_only_transaction"
        stage_started = time.monotonic()
        k3_transaction_called = True
        outcome = run_phase_conditioned_objective_hull_k3_build_only(
            source_build,
            focused.rivals,
            selection,
            focused_rival_id=focused_rival_id,
            retained_k2_stable_bit_ids=(
                _PCOH_K3_RETAINED_K2_STABLE_BIT_IDS
            ),
            deadline=phase_deadline,
        )
        if not verify_phase_conditioned_objective_hull_k3_build_only_outcome(
            outcome
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_same_process_outcome_verification_failed"
            )
        if type(outcome) is PCOHK3BuildOnlyDiagnostic:
            outcome_kind = "success"
            outcome_sha256 = outcome.diagnostic_sha256
        elif type(outcome) is PCOHK3BuildOnlyStopDiagnostic:
            outcome_kind = "strong_target_stop"
            outcome_sha256 = outcome.stop_sha256
        elif type(outcome) is PCOHK3BuildOnlyResourceStopDiagnostic:
            outcome_kind = "resource_stop"
            outcome_sha256 = outcome.resource_stop_sha256
        else:
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_outcome_type_not_exact"
            )
        if not _valid_sha256(outcome_sha256):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_trusted_outcome_digest_invalid"
            )
        detached_mapping = (
            export_phase_conditioned_objective_hull_k3_build_only_detached(
                outcome
            )
        )
        detached = _builtin_receipt_mapping(detached_mapping)
        if (
            detached is None
            or not _all_finite_json(detached)
            or not verify_detached_phase_conditioned_objective_hull_k3_build_only(
                detached, expected_sha256=outcome_sha256
            )
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_detached_outcome_verification_failed"
            )
        execution = detached.get("execution_telemetry")
        detached_receipt = detached.get("receipt")
        stable_ids = list(detached.get("stable_bit_ids", ()))
        if (
            type(execution) is not dict
            or type(detached_receipt) is not dict
            or len(stable_ids) != 3
            or stable_ids[:2]
            != list(_PCOH_K3_RETAINED_K2_STABLE_BIT_IDS)
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_detached_execution_binding_failed"
            )
        pair_lp = execution.get("pair_local_lp_actual_calls")
        conditional_lp = execution.get("scheduled_local_lp_actual_calls")
        total_lp = execution.get("local_lp_actual_calls")
        checker_calls = execution.get("conditional_checker_actual_calls")
        if (
            pair_lp != 12
            or type(conditional_lp) is not int
            or type(total_lp) is not int
            or total_lp != pair_lp + conditional_lp
            or total_lp > 20
            or type(checker_calls) is not int
            or checker_calls > 34
            or execution.get("local_lp_actual_call_cap") != 20
            or execution.get("conditional_checker_actual_call_cap") != 34
        ):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_actual_lp_or_checker_counter_mismatch"
            )
        resource_gate_rejection = None
        resource_gate_rejection_sha256 = None
        if outcome_kind == "success":
            summary = detached.get("materialized_tightness_summary")
            if type(summary) is not dict:
                raise PhaseCliqueBuildProbeError(
                    "pcoh_k3_materialized_summary_missing"
                )
            gate = _pcoh_k3_strong_tightness_gate(
                summary,
                source_semantic_digest=detached[
                    "source_semantic_digest"
                ],
                selection_digest=selection.selection_digest,
                focused_encoded_row=focused_encoded_row,
                focused_rival_id=focused_rival_id,
                retained_k2_stable_bit_ids=(
                    _PCOH_K3_RETAINED_K2_STABLE_BIT_IDS
                ),
                stable_bit_ids=stable_ids,
            )
            status = gate["status"]
            summary_sha256 = summary.get("summary_sha256")
            fresh_dimensions = list(detached["fresh_dimensions"])
            fresh_semantic_digest = detached["fresh_semantic_digest"]
            source_dimensions = list(detached["source_dimensions"])
            transaction_reason = None
            transaction_failed_stage = None
        elif outcome_kind == "strong_target_stop":
            status = "strong_target_stop"
            summary = None
            summary_sha256 = None
            gate = None
            fresh_dimensions = None
            fresh_semantic_digest = None
            source_dimensions = list(detached_receipt["source_dimensions"])
            transaction_reason = None
            transaction_failed_stage = None
        else:
            status = "resource_stop"
            summary = None
            summary_sha256 = None
            gate = None
            fresh_dimensions = None
            fresh_semantic_digest = None
            source_dimensions = list(detached_receipt["source_dimensions"])
            transaction_reason = detached.get("reason")
            transaction_failed_stage = detached.get("stage")
            resource_gate_rejection = _builtin_receipt_mapping(
                detached_receipt.get("resource_gate_rejection")
            )
            resource_gate_rejection_sha256 = (
                resource_gate_rejection.get("rejection_sha256")
                if type(resource_gate_rejection) is dict
                else None
            )
            if (
                transaction_failed_stage not in {
                    "pre_scheduled", "pre_fresh_materialization"
                }
                or type(transaction_reason) is not str
                or not transaction_reason
                or type(resource_gate_rejection) is not dict
                or not _valid_sha256(resource_gate_rejection_sha256)
                or resource_gate_rejection.get("stage")
                != transaction_failed_stage
                or resource_gate_rejection.get("reason")
                != transaction_reason
                or any(
                    detached.get(name) is not False
                    for name in (
                        "partial_certificates_returned",
                        "conditional_certificate_payload_returned",
                        "fresh_issue_called",
                        "fresh_build_returned",
                        "fresh_descriptor_returned",
                        "provenance_authority",
                        "authenticity_authority",
                    )
                )
            ):
                raise PhaseCliqueBuildProbeError(
                    "pcoh_k3_resource_stop_binding_failed"
                )
        timings["k3_transaction_seconds"] = float(
            time.monotonic() - stage_started
        )
        timings["total_seconds"] = float(time.monotonic() - started)
        stage_resources["k3_transaction"] = _capture_resource_peaks(
            torch_module
        )
        transaction = _checksummed({
            "schema": _PCOH_K3_TRANSACTION_SCHEMA,
            "status": status,
            "reason": transaction_reason,
            "failed_stage": transaction_failed_stage,
            "diagnostic_only": True,
            "candidate_only": True,
            "build_only": True,
            "instance_count": 1,
            "proof_authority": False,
            "verdict_authority": False,
            "provenance_authority": False,
            "authenticity_authority": False,
            "ground_truth_loaded": False,
            "reference_label_used": False,
            "k3_transaction_called": True,
            "k3_same_process_verified": True,
            "trusted_outcome_digest_captured_before_detach": True,
            "k3_detached_verified": True,
            "k2_build_only_called": False,
            "phase_transaction_called": False,
            "solver_handoff_called": False,
            "hz_base_feasibility_called": False,
            "hz_objbound_decide_called": False,
            "full_parent_lp_called": False,
            "full_parent_lp_solver_called": False,
            "fresh_build_returned": False,
            "input_sha256": dict(input_sha256),
            "implementation_sha256": dict(implementation_sha256),
            "baseline_artifact_sha256": _PCOH_K3_BASELINE_ARTIFACT_SHA256,
            "baseline_summary_sha256": _PCOH_K3_BASELINE_SUMMARY_SHA256,
            "baseline_anchor_receipt_sha256": baseline_anchor_receipt[
                "receipt_sha256"
            ],
            "baseline_anchor_verified": True,
            "full_batch_sha256": batch.batch_sha256,
            "focused_subset_digest": focused.focused_subset_digest,
            "residual_selector_receipt_sha256": (
                focused.residual_selector_receipt_sha256
            ),
            "focused_semantic_anchor": focused_semantic_anchor,
            "focused_semantic_anchor_sha256": focused_semantic_anchor[
                "semantic_sha256"
            ],
            "focused_encoded_row": focused_encoded_row,
            "focused_rival_id": focused_rival_id,
            "source_semantic_digest": detached["source_semantic_digest"],
            "selection_digest": selection.selection_digest,
            "selection_property_digest": selection.property_digest,
            "selection_parent_semantic_digest": (
                selection.parent_semantic_digest
            ),
            "selection_operator_row_tag_digest": (
                selection.operator_row_tag_digest
            ),
            "stable_bit_selection_method": (
                "retain_fixed_k2_pair_then_exact_focused_third"
            ),
            "retained_k2_stable_bit_ids": list(
                _PCOH_K3_RETAINED_K2_STABLE_BIT_IDS
            ),
            "stable_bit_ids": stable_ids,
            "third_stable_bit_id": stable_ids[2],
            "outcome_kind": outcome_kind,
            "outcome_schema": detached["schema"],
            "outcome_status": detached["status"],
            "trusted_outcome_sha256": outcome_sha256,
            "outcome_receipt_sha256": detached_receipt["receipt_sha256"],
            "detached_outcome": detached,
            "resource_gate_rejection_sha256": (
                resource_gate_rejection_sha256
            ),
            "resource_gate_rejection": resource_gate_rejection,
            "pair_bundle_sha256": detached["pair_bundle_sha256"],
            "active_pattern_mask": list(detached["active_pattern_mask"]),
            "evaluation_schedule": list(detached["evaluation_schedule"]),
            "threshold_pattern_indices": list(
                detached["threshold_pattern_indices"]
            ),
            "source_dimensions": source_dimensions,
            "fresh_dimensions": fresh_dimensions,
            "fresh_semantic_digest": fresh_semantic_digest,
            "materialized_tightness_summary_sha256": summary_sha256,
            "materialized_tightness_summary": summary,
            "strong_tightness_gate": gate,
            "pair_local_lp_actual_calls": pair_lp,
            "conditional_local_lp_actual_calls": conditional_lp,
            "total_local_lp_actual_calls": total_lp,
            "conditional_checker_actual_calls": checker_calls,
            "local_lp_actual_call_cap": 20,
            "conditional_checker_actual_call_cap": 34,
            "stage_resources": dict(stage_resources),
            "timings": timings,
        })
        registered_transaction = transaction
        _register_pcoh_k3_trusted_transaction(
            transaction,
            outcome=outcome,
            outcome_sha256=outcome_sha256,
            outcome_kind=outcome_kind,
        )
        outcome = None
        if not _pcoh_k3_transaction_receipt_valid(transaction):
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_transaction_self_check_failed"
            )
        if time.monotonic() >= phase_deadline:
            raise TimeoutError("pcoh_k3_transaction_deadline_exhausted")
        handed_off = True
        return transaction
    except BaseException as exc:
        detail = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_pcoh_k3_exception_traceback(exc)
        return _pcoh_k3_stop_loss_receipt(
            stage=stage,
            reason=detail,
            started=started,
            input_sha256=input_sha256,
            implementation_sha256=implementation_sha256,
            stage_resources={
                **stage_resources,
                "stop_loss": _capture_resource_peaks(torch_module),
            },
            timings=timings,
            k3_transaction_called=k3_transaction_called,
            baseline_anchor_receipt=baseline_anchor_receipt,
        )
    finally:
        outcome = None
        if registered_transaction is not None and not handed_off:
            _release_pcoh_k3_trusted_transaction(registered_transaction)
        registered_transaction = None


def _run_phase_transaction(
    source_build: Any,
    *,
    pipeline_kwargs: Mapping[str, Any],
    objective_rows: np.ndarray,
    thresholds: np.ndarray,
    deadline: float,
    run_pipeline: Callable[..., Any],
    consume_handoff: Callable[..., Any],
    validate_consumed: Callable[[Any, Any], bool],
    lp_upper: Callable[..., Mapping[str, Any]] = _certified_relaxed_upper,
    lp_per_call_seconds: float = 5.0,
) -> tuple[dict[str, Any], Any]:
    """Run and consume the candidate transaction without a verdict engine."""

    if (
        isinstance(lp_per_call_seconds, bool)
        or not isinstance(lp_per_call_seconds, (int, float))
        or not math.isfinite(float(lp_per_call_seconds))
        or float(lp_per_call_seconds) <= 0.0
    ):
        raise PhaseCliqueBuildProbeError(
            "LP diagnostic budget must be finite and positive"
        )
    transaction_started = time.monotonic()
    rss_before_pipeline = _rss_sample()
    source_exact_kept_nonzeros = _exact_candidate_kept_nonzeros(
        source_build.hz,
        deadline=deadline,
    )
    pipeline_started = transaction_started
    result = run_pipeline(source_build, **dict(pipeline_kwargs))
    pipeline_seconds = float(time.monotonic() - pipeline_started)
    private_build = consume_handoff(
        source_build, result, deadline=deadline
    )
    if validate_consumed(result, private_build) is not True:
        raise PhaseCliqueBuildProbeError(
            "private solver build failed terminal handoff validation"
        )
    rss_after_handoff = _rss_sample()
    receipt = dict(result.receipt)
    candidate_route = _builtin_receipt_mapping(
        receipt.get("candidate_route_summary")
    )
    candidate_progress = _builtin_receipt_mapping(
        receipt.get("candidate_progress")
    )
    materializer_route = _materializer_route_summary(receipt)
    result_status = str(result.status)
    was_materialized = bool(result.materialized)
    identity_preserved = bool(result.identity_preserved)
    public_build_is_solver_build = bool(result.build is private_build)
    focused_encoded_row = receipt.get("focused_encoded_row")
    diagnostic: dict[str, Any] = {
        "status": result_status,
        "materialized": was_materialized,
        "identity_preserved": identity_preserved,
        "pipeline_seconds": pipeline_seconds,
        "source_upper_rows": int(source_build.hz.n_ub),
        "fresh_upper_rows": int(private_build.hz.n_ub),
        "clique_count": int(receipt.get("clique_count", 0)),
        "certified_edge_count": int(receipt.get("certified_edge_count", 0)),
        "cut_row_count": int(receipt.get("cut_row_count", 0)),
        "fallback_reason": receipt.get("fallback_reason"),
        "failed_stage": receipt.get("failed_stage"),
        "error_type": receipt.get("error_type"),
        "pipeline_receipt_sha256": receipt.get("receipt_sha256"),
        "candidate_result_status": receipt.get(
            "candidate_result_status"
        ),
        "candidate_telemetry_schema": receipt.get(
            "candidate_telemetry_schema"
        ),
        "candidate_representation": receipt.get(
            "candidate_representation"
        ),
        "candidate_cut_hz_emitted": receipt.get(
            "candidate_cut_hz_emitted"
        ),
        "candidate_descriptor_sha256": receipt.get(
            "candidate_descriptor_sha256"
        ),
        "candidate_route_summary": candidate_route,
        "candidate_progress_available": receipt.get(
            "candidate_progress_available"
        ),
        "candidate_progress": candidate_progress,
        "materialization_receipt_sha256": receipt.get(
            "materialization_receipt_sha256"
        ),
        "materializer_route_summary": materializer_route,
        "initial_budget_seconds": receipt.get("initial_budget_seconds"),
        "candidate_budget_seconds": receipt.get(
            "candidate_budget_seconds"
        ),
        "candidate_elapsed_seconds": receipt.get(
            "candidate_elapsed_seconds"
        ),
        "minimum_materializer_reserve_seconds": receipt.get(
            "minimum_materializer_reserve_seconds"
        ),
        "private_handoff_consumed": True,
        "terminal_handoff_validated": True,
        "public_build_is_solver_build": public_build_is_solver_build,
        "source_shape": _hz_shape(source_build.hz),
        "source_exact_kept_candidate_nonzeros": (
            source_exact_kept_nonzeros
        ),
        "fresh_private_shape": _hz_shape(private_build.hz),
        "pipeline_timings": dict(receipt.get("timings", {})),
        "phase_rss_before_pipeline": rss_before_pipeline,
        "phase_rss_after_handoff": rss_after_handoff,
        "lp_tightness": {"status": "not_materialized", "proof_authority": False},
    }
    # The one-use handoff has been consumed and independently validated.
    # Freeze every later scalar above, then release the public cut HZ before
    # either diagnostic LP allocates its candidate matrix.
    del result
    del receipt
    diagnostic["phase_rss_after_public_release"] = _rss_sample()
    if was_materialized:
        focused = focused_encoded_row
        if type(focused) is not int or not 0 <= focused < objective_rows.shape[0]:
            raise PhaseCliqueBuildProbeError("focused objective row is malformed")
        before_deadline = min(
            deadline, time.monotonic() + float(lp_per_call_seconds)
        )
        before = dict(
            lp_upper(
                source_build.hz,
                objective_rows[focused],
                float(thresholds[focused]),
                deadline=before_deadline,
            )
        )
        after_deadline = min(
            deadline, time.monotonic() + float(lp_per_call_seconds)
        )
        after = dict(
            lp_upper(
                private_build.hz,
                objective_rows[focused],
                float(thresholds[focused]),
                deadline=after_deadline,
            )
        )
        before_upper = before.get("independently_certified_upper")
        after_upper = after.get("independently_certified_upper")
        compared = bool(
            type(before_upper) is float
            and math.isfinite(before_upper)
            and type(after_upper) is float
            and math.isfinite(after_upper)
        )
        improvement = (
            float(before_upper - after_upper) if compared else None
        )
        relative_drop = (
            float(improvement / max(abs(before_upper), np.finfo(float).tiny))
            if compared
            else None
        )
        diagnostic["lp_tightness"] = {
            "status": "compared" if compared else "inconclusive",
            "proof_authority": False,
            "verdict_authority": False,
            "focused_encoded_row": focused,
            "before": before,
            "after": after,
            "independent_lp_call_count": 2,
            "per_call_budget_seconds": float(lp_per_call_seconds),
            "before_deadline_monotonic": float(before_deadline),
            "after_deadline_monotonic": float(after_deadline),
            "certified_upper_improvement": improvement,
            "relative_drop": relative_drop,
        }
    diagnostic["transaction_elapsed_seconds"] = float(
        time.monotonic() - transaction_started
    )
    diagnostic["phase_rss_after_transaction"] = _rss_sample()
    return diagnostic, private_build


def _rbs_adaptive_k4_split_lp_certificate_valid(
    bound: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> bool:
    """Bind one diagnostic upper to the native no-stack checker route."""

    if not isinstance(bound, Mapping) or not isinstance(shape, Mapping):
        return False
    route = bound.get("certificate_route")
    certificate = bound.get("certificate")
    route_keys = {
        "schema",
        "route",
        "uses_sparse_hstack",
        "uses_sparse_vstack",
        "assembled_sparse_nnz",
        "input_sparse_nnz",
        "recomputed_input_sparse_nnz",
        "block_shapes",
        "upper_float64_rounding",
        "upper_outward_float64",
        "candidate_upper_row_dual_sha256",
        "candidate_equality_row_dual_sha256",
    }
    block_keys = {"Gc", "Gb", "Auc", "Aub", "Ac", "Ab"}
    if (
        type(route) is not dict
        or set(route) != route_keys
        or not isinstance(certificate, Mapping)
        or certificate.get("status") != "verified_upper"
        or route.get("schema")
        != _RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_SCHEMA
        or route.get("route")
        != _RBS_ADAPTIVE_K4_SPLIT_LP_CERTIFICATE_ROUTE
        or route.get("uses_sparse_hstack") is not False
        or route.get("uses_sparse_vstack") is not False
        or route.get("assembled_sparse_nnz") != 0
        or type(route.get("input_sparse_nnz")) is not int
        or type(route.get("recomputed_input_sparse_nnz")) is not int
        or route.get("input_sparse_nnz")
        != route.get("recomputed_input_sparse_nnz")
        or route.get("upper_float64_rounding")
        != "toward_positive_infinity_from_longdouble_v1"
        or type(route.get("upper_outward_float64")) is not float
        or not math.isfinite(route.get("upper_outward_float64"))
        or bound.get("independently_certified_upper")
        != route.get("upper_outward_float64")
        or certificate.get("upper")
        != route.get("upper_outward_float64")
        or not _valid_sha256(
            route.get("candidate_upper_row_dual_sha256")
        )
        or not _valid_sha256(
            route.get("candidate_equality_row_dual_sha256")
        )
        or any(
            certificate.get(name) != route.get(name)
            for name in (
                "schema",
                "route",
                "uses_sparse_hstack",
                "uses_sparse_vstack",
                "assembled_sparse_nnz",
                "input_sparse_nnz",
                "block_shapes",
                "upper_float64_rounding",
            )
        )
    ):
        return False
    blocks = route.get("block_shapes")
    if type(blocks) is not dict or set(blocks) != block_keys:
        return False
    required_shape_keys = (
        "output_dimension",
        "continuous_columns",
        "binary_columns",
        "upper_rows",
        "equality_rows",
        "constraint_nonzeros",
        "generator_nonzeros",
    )
    if any(
        type(shape.get(name)) is not int or shape.get(name) < 0
        for name in required_shape_keys
    ):
        return False
    output = shape["output_dimension"]
    continuous = shape["continuous_columns"]
    binary = shape["binary_columns"]
    upper_rows = shape["upper_rows"]
    equality_rows = shape["equality_rows"]
    expected_blocks = {
        "Gc": [output, continuous],
        "Gb": [output, binary],
        "Auc": [upper_rows, continuous],
        "Aub": [upper_rows, binary],
        "Ac": [equality_rows, continuous],
        "Ab": [equality_rows, binary],
    }
    expected_nonzeros = (
        shape["constraint_nonzeros"] + shape["generator_nonzeros"]
    )
    return bool(
        blocks == expected_blocks
        and route.get("input_sparse_nnz") == expected_nonzeros
    )


def _rbs_adaptive_k4_objective_dual_proposal_valid(
    bound: Mapping[str, Any],
    shape: Mapping[str, Any],
    *,
    expected_kept_nonzeros: Optional[int] = None,
) -> bool:
    """Bind a bound to the closed, native split-row proposal route."""

    if not isinstance(bound, Mapping) or not isinstance(shape, Mapping):
        return False
    receipt = bound.get("objective_dual_proposal_receipt")
    route = bound.get("objective_dual_proposal_route")
    certificate_route = bound.get("certificate_route")
    receipt_keys = {
        "schema",
        "status",
        "candidate_only",
        "proof_authority",
        "verdict_authority",
        "backend",
        "highs_version",
        "presolve",
        "row_order",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "n_continuous",
        "n_binary",
        "n_upper",
        "n_equality",
        "objective_convention",
        "maximization_factor_objective_size",
        "maximization_factor_objective_sha256",
        "solver_cost_sha256",
        "upper_row_dual_size",
        "equality_row_dual_size",
        "upper_row_dual_sha256",
        "equality_row_dual_sha256",
        "solver_minimization_objective_hex",
        "pair_solve_calls",
        "objective_solve_calls",
        "native_model_closed_before_return",
        "uses_sparse_hstack",
        "uses_sparse_vstack",
        "used_merged_sparse_frame",
        "receipt_sha256",
    }
    shape_keys = (
        "continuous_columns",
        "binary_columns",
        "upper_rows",
        "equality_rows",
        "constraint_nonzeros",
    )
    if (
        type(receipt) is not dict
        or set(receipt) != receipt_keys
        or type(route) is not dict
        or route != receipt
        or type(certificate_route) is not dict
        or not _local_receipt_checksum_valid(
            receipt,
            schema=_RBS_ADAPTIVE_K4_OBJECTIVE_DUAL_PROPOSAL_SCHEMA,
        )
        or any(
            type(shape.get(name)) is not int or shape.get(name) < 0
            for name in shape_keys
        )
    ):
        return False
    n_continuous = shape["continuous_columns"]
    n_binary = shape["binary_columns"]
    n_upper = shape["upper_rows"]
    n_equality = shape["equality_rows"]
    candidate_nonzeros = receipt.get("candidate_nonzeros")
    solver_objective_hex = receipt.get(
        "solver_minimization_objective_hex"
    )
    try:
        parsed_solver_objective = float.fromhex(solver_objective_hex)
    except (TypeError, ValueError, OverflowError):
        return False
    hashes = (
        receipt.get("maximization_factor_objective_sha256"),
        receipt.get("solver_cost_sha256"),
        receipt.get("upper_row_dual_sha256"),
        receipt.get("equality_row_dual_sha256"),
    )
    kept_binding_valid = (
        expected_kept_nonzeros is None
        or (
            type(expected_kept_nonzeros) is int
            and expected_kept_nonzeros >= 0
            and candidate_nonzeros == expected_kept_nonzeros
        )
    )
    return bool(
        receipt.get("status") == "optimal_dual_candidate"
        and receipt.get("candidate_only") is True
        and receipt.get("proof_authority") is False
        and receipt.get("verdict_authority") is False
        and receipt.get("backend")
        == _RBS_ADAPTIVE_K4_OBJECTIVE_DUAL_BACKEND
        and type(receipt.get("highs_version")) is str
        and bool(receipt.get("highs_version"))
        and receipt.get("presolve") == "on"
        and receipt.get("row_order") == "upper_then_equality"
        and receipt.get("candidate_load_mode")
        == _RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE
        and receipt.get("binary_change_coefficient_cap")
        == _RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP
        and receipt.get("candidate_rows") == n_upper + n_equality
        and receipt.get("candidate_columns")
        == n_continuous + n_binary
        and type(candidate_nonzeros) is int
        and 0 <= candidate_nonzeros <= shape["constraint_nonzeros"]
        and kept_binding_valid
        and receipt.get("n_continuous") == n_continuous
        and receipt.get("n_binary") == n_binary
        and receipt.get("n_upper") == n_upper
        and receipt.get("n_equality") == n_equality
        and receipt.get("objective_convention")
        == "highs_minimize_cost_equals_negative_max_factor_objective"
        and receipt.get("maximization_factor_objective_size")
        == n_continuous + n_binary
        and receipt.get("upper_row_dual_size") == n_upper
        and receipt.get("equality_row_dual_size") == n_equality
        and all(_valid_sha256(value) for value in hashes)
        and certificate_route.get(
            "candidate_upper_row_dual_sha256"
        )
        == receipt.get("upper_row_dual_sha256")
        and certificate_route.get(
            "candidate_equality_row_dual_sha256"
        )
        == receipt.get("equality_row_dual_sha256")
        and type(parsed_solver_objective) is float
        and math.isfinite(parsed_solver_objective)
        and receipt.get("pair_solve_calls") == 0
        and receipt.get("objective_solve_calls") == 1
        and receipt.get("native_model_closed_before_return") is True
        and receipt.get("uses_sparse_hstack") is False
        and receipt.get("uses_sparse_vstack") is False
        and receipt.get("used_merged_sparse_frame") is False
    )


def _rbs_adaptive_k4_terminal_progress_valid(
    transaction: Mapping[str, Any],
) -> bool:
    """Require an exact terminal six-pair diagnostic progress frame."""

    progress = transaction.get("candidate_progress")
    source = transaction.get("source_shape")
    keys = {
        "schema",
        "status",
        "candidate_only",
        "proof_authority",
        "verdict_authority",
        "model_load_started",
        "model_loaded",
        "oracle_backend",
        "oracle_presolve",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "pair_target_count",
        "pair_attempted_count",
        "pair_completed_count",
        "certified_conflict_count",
        "last_pair_index",
        "terminal_complete",
        "candidate_cut_hz_emitted",
        "partial_never_authorizes_edge",
        "materializer_reached",
    }
    if (
        transaction.get("candidate_progress_available") is not True
        or type(progress) is not dict
        or set(progress) != keys
        or not isinstance(source, Mapping)
    ):
        return False
    required_source = (
        "upper_rows",
        "equality_rows",
        "continuous_columns",
        "binary_columns",
    )
    if any(
        type(source.get(name)) is not int or source.get(name) < 0
        for name in required_source
    ):
        return False
    return bool(
        progress.get("schema") == _RBS_ADAPTIVE_K4_PROGRESS_SCHEMA
        and progress.get("status") == "complete"
        and progress.get("candidate_only") is True
        and progress.get("proof_authority") is False
        and progress.get("verdict_authority") is False
        and progress.get("model_load_started") is True
        and progress.get("model_loaded") is True
        and progress.get("oracle_backend")
        == _RBS_ADAPTIVE_K4_ORACLE_BACKEND
        and progress.get("oracle_presolve") == "on"
        and progress.get("candidate_load_mode")
        == _RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE
        and progress.get("binary_change_coefficient_cap")
        == _RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP
        and progress.get("candidate_rows")
        == source.get("upper_rows") + source.get("equality_rows")
        and progress.get("candidate_columns")
        == source.get("continuous_columns")
        + source.get("binary_columns")
        and progress.get("candidate_nonzeros")
        == transaction.get("source_exact_kept_candidate_nonzeros")
        and progress.get("pair_target_count") == 6
        and progress.get("pair_attempted_count") == 6
        and progress.get("pair_completed_count") == 6
        and progress.get("certified_conflict_count") == 6
        and progress.get("last_pair_index") == 5
        and progress.get("terminal_complete") is True
        and progress.get("candidate_cut_hz_emitted") is False
        and progress.get("partial_never_authorizes_edge") is True
        and progress.get("materializer_reached") is False
    )


def _rbs_adaptive_k4_route_checks(
    transaction: Mapping[str, Any],
) -> dict[str, bool]:
    """Rebind the exact C88 low-peak K4 implementation route."""

    route = transaction.get("candidate_route_summary")
    source_shape = transaction.get("source_shape")
    materializer = transaction.get("materializer_route_summary")
    route_keys = {
        "schema",
        "result_mode",
        "result_status",
        "telemetry_schema",
        "hz_absent",
        "oracle_backend",
        "oracle_presolve",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "model_builds",
        "solve_calls",
        "base_solve_calls",
        "pair_count",
        "pair_status_counts",
        "completed_pair_count",
        "proof_authority",
    }
    materializer_keys = {
        "schema",
        "receipt_sha256",
        "public_core_source",
        "parent_prefix_core",
        "parent_prefix_readonly",
        "parent_prefix_aliases_public_cut",
        "public_core_readonly",
        "materializer_full_core_copy_count",
        "private_solver_core",
        "public_private_core_no_alias",
        "producer_nonempty_seal_verified",
        "one_use_snapshot_consumed",
        "solver_handoff_one_use",
        "solver_handoff_owner_bound",
        "solver_handoff_pid_bound",
        "solver_handoff_private_core_readonly",
    }
    route_exact = type(route) is dict and set(route) == route_keys
    source_exact = isinstance(source_shape, Mapping)
    materializer_exact = (
        type(materializer) is dict
        and set(materializer) == materializer_keys
    )
    descriptor_bound = bool(
        transaction.get("candidate_result_status")
        == _RBS_ADAPTIVE_K4_COMPACT_STATUS
        and transaction.get("candidate_telemetry_schema")
        == _RBS_ADAPTIVE_K4_COMPACT_TELEMETRY_SCHEMA
        and transaction.get("candidate_representation")
        == _RBS_ADAPTIVE_K4_COMPACT_REPRESENTATION
        and transaction.get("candidate_cut_hz_emitted") is False
        and _valid_sha256(
            transaction.get("candidate_descriptor_sha256")
        )
        and _valid_sha256(
            transaction.get("pipeline_receipt_sha256")
        )
        and route_exact
        and route.get("schema")
        == "act.operator_phase_clique_compact_route.v1"
        and route.get("result_mode") == "compact_exact_descriptor_v1"
        and route.get("result_status")
        == _RBS_ADAPTIVE_K4_COMPACT_STATUS
        and route.get("telemetry_schema")
        == _RBS_ADAPTIVE_K4_COMPACT_TELEMETRY_SCHEMA
        and route.get("hz_absent") is True
        and route.get("proof_authority") is False
    )
    presolve_route = bool(
        route_exact
        and route.get("oracle_backend")
        == _RBS_ADAPTIVE_K4_ORACLE_BACKEND
        and route.get("oracle_presolve") == "on"
        and route.get("model_builds") == 1
        and route.get("base_solve_calls") == 0
    )
    split_shape_route = bool(
        route_exact
        and source_exact
        and all(
            type(source_shape.get(name)) is int
            and source_shape.get(name) >= 0
            for name in (
                "upper_rows",
                "equality_rows",
                "continuous_columns",
                "binary_columns",
                "constraint_nonzeros",
            )
        )
        and route.get("candidate_load_mode")
        == _RBS_ADAPTIVE_K4_SPLIT_LOAD_MODE
        and route.get("binary_change_coefficient_cap")
        == _RBS_ADAPTIVE_K4_BINARY_CHANGE_COEFFICIENT_CAP
        and type(route.get("candidate_rows")) is int
        and route.get("candidate_rows")
        == source_shape.get("upper_rows", -1)
        + source_shape.get("equality_rows", -1)
        and type(route.get("candidate_columns")) is int
        and route.get("candidate_columns")
        == source_shape.get("continuous_columns", -1)
        + source_shape.get("binary_columns", -1)
        and type(route.get("candidate_nonzeros")) is int
        and route.get("candidate_nonzeros")
        == transaction.get("source_exact_kept_candidate_nonzeros")
        and type(
            transaction.get("source_exact_kept_candidate_nonzeros")
        )
        is int
        and 0
        <= transaction.get("source_exact_kept_candidate_nonzeros")
        <= source_shape.get("constraint_nonzeros")
    )
    pair_route = bool(
        route_exact
        and route.get("solve_calls") == 6
        and route.get("pair_count") == 6
        and route.get("completed_pair_count") == 6
        and route.get("pair_status_counts")
        == {
            "certified_conflict": 6,
            "feasible_or_unknown": 0,
            "infeasible_without_ray": 0,
            "exact_replay_rejected": 0,
        }
    )
    materializer_route = bool(
        materializer_exact
        and materializer.get("schema")
        == "act.operator_exact_relu_phase_clique_materialization.v2"
        and _valid_sha256(materializer.get("receipt_sha256"))
        and materializer.get("receipt_sha256")
        == transaction.get("materialization_receipt_sha256")
        and materializer.get("public_core_source")
        == "consumed_verified_cut_zero_copy"
        and materializer.get("parent_prefix_core")
        == "strict_readonly_zero_copy_view"
        and materializer.get("parent_prefix_readonly") is True
        and materializer.get("parent_prefix_aliases_public_cut") is True
        and materializer.get("public_core_readonly") is True
        and materializer.get("materializer_full_core_copy_count") == 1
        and materializer.get("private_solver_core")
        == "single_independent_snapshot"
        and materializer.get("public_private_core_no_alias") is True
        and materializer.get("producer_nonempty_seal_verified") is True
        and materializer.get("one_use_snapshot_consumed") is True
        and materializer.get("solver_handoff_one_use") is True
        and materializer.get("solver_handoff_owner_bound") is True
        and materializer.get("solver_handoff_pid_bound") is True
        and materializer.get("solver_handoff_private_core_readonly") is True
    )

    before = transaction.get("phase_rss_before_pipeline")
    after_handoff = transaction.get("phase_rss_after_handoff")
    after_public_release = transaction.get(
        "phase_rss_after_public_release"
    )
    after_transaction = transaction.get("phase_rss_after_transaction")

    def valid_sample(value: Any) -> bool:
        return bool(
            type(value) is dict
            and set(value) == {"current_rss_bytes", "peak_rss_bytes"}
            and type(value.get("current_rss_bytes")) is int
            and 0
            <= value.get("current_rss_bytes")
            <= _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
            and type(value.get("peak_rss_bytes")) is int
            and 0
            <= value.get("peak_rss_bytes")
            <= _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
        )

    samples_valid = bool(
        valid_sample(before)
        and valid_sample(after_handoff)
        and valid_sample(after_public_release)
        and valid_sample(after_transaction)
        and before["peak_rss_bytes"] <= after_handoff["peak_rss_bytes"]
        <= after_public_release["peak_rss_bytes"]
        <= after_transaction["peak_rss_bytes"]
    )
    phase_entry_headroom = bool(
        valid_sample(before)
        and before["current_rss_bytes"]
        <= (
            _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
            - _RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
        )
        and before["peak_rss_bytes"]
        <= (
            _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
            - _RBS_ADAPTIVE_K4_PHASE_ENTRY_HEADROOM_BYTES
        )
    )
    return {
        "compact_candidate_descriptor_bound": descriptor_bound,
        "terminal_six_pair_progress_receipt": (
            _rbs_adaptive_k4_terminal_progress_valid(transaction)
        ),
        "presolve_v2_candidate_route": presolve_route,
        "split_loader_exact_shape_route": split_shape_route,
        "six_pair_route_complete": pair_route,
        "unique_copy_materializer_route": materializer_route,
        "phase_entry_has_64_mib_rss_headroom": phase_entry_headroom,
        "phase_rss_samples_within_2_5_gib": samples_valid,
    }


def _rbs_adaptive_k4_post_gate(
    transaction: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a fresh K4 cut and a material independent-LP tightening."""

    lp = transaction.get("lp_tightness")
    if not isinstance(lp, Mapping):
        lp = {}
    before = lp.get("before")
    after = lp.get("after")
    if not isinstance(before, Mapping):
        before = {}
    if not isinstance(after, Mapping):
        after = {}
    before_upper = before.get("independently_certified_upper")
    after_upper = after.get("independently_certified_upper")
    improvement = lp.get("certified_upper_improvement")
    relative_drop = lp.get("relative_drop")
    recomputed_improvement = (
        float(before_upper - after_upper)
        if (
            type(before_upper) is float
            and math.isfinite(before_upper)
            and type(after_upper) is float
            and math.isfinite(after_upper)
        )
        else None
    )
    recomputed_relative_drop = (
        float(recomputed_improvement / before_upper)
        if (
            recomputed_improvement is not None
            and type(before_upper) is float
            and before_upper > 0.0
        )
        else None
    )
    initial_budget = transaction.get("initial_budget_seconds")
    candidate_budget = transaction.get("candidate_budget_seconds")
    candidate_elapsed = transaction.get("candidate_elapsed_seconds")
    materializer_reserve = transaction.get(
        "minimum_materializer_reserve_seconds"
    )
    pipeline_seconds = transaction.get("pipeline_seconds")
    transaction_seconds = transaction.get("transaction_elapsed_seconds")
    source_rows = transaction.get("source_upper_rows")
    fresh_rows = transaction.get("fresh_upper_rows")
    before_proposal = before.get("objective_dual_proposal_receipt")
    after_proposal = after.get("objective_dual_proposal_receipt")
    conditions = {
        "fresh_k4_materialized": bool(
            transaction.get("status")
            == "fresh_verified_k4_clique_materialized"
            and transaction.get("materialized") is True
            and transaction.get("identity_preserved") is False
        ),
        "six_certified_edges": bool(
            transaction.get("certified_edge_count") == 6
        ),
        "one_k4_clique": bool(transaction.get("clique_count") == 1),
        "one_fresh_cut": bool(
            transaction.get("cut_row_count") == 1
            and type(source_rows) is int
            and type(fresh_rows) is int
            and fresh_rows == source_rows + 1
        ),
        "private_handoff_valid": bool(
            transaction.get("private_handoff_consumed") is True
            and transaction.get("terminal_handoff_validated") is True
            and transaction.get("public_build_is_solver_build") is False
        ),
        "phase_window_at_most_30_seconds": bool(
            type(initial_budget) is float
            and 0.0 < initial_budget <= _RBS_ADAPTIVE_K4_PHASE_SECONDS
            and type(pipeline_seconds) is float
            and 0.0 <= pipeline_seconds <= _RBS_ADAPTIVE_K4_PHASE_SECONDS
        ),
        "whole_transaction_at_most_30_seconds": bool(
            type(transaction_seconds) is float
            and 0.0
            <= transaction_seconds
            <= _RBS_ADAPTIVE_K4_PHASE_SECONDS
        ),
        "candidate_budget_at_most_12_seconds": bool(
            type(candidate_budget) is float
            and 0.0
            < candidate_budget
            <= _RBS_ADAPTIVE_K4_MAX_CANDIDATE_SECONDS
        ),
        "candidate_elapsed_at_most_12_seconds": bool(
            type(candidate_elapsed) is float
            and 0.0
            <= candidate_elapsed
            <= _RBS_ADAPTIVE_K4_MAX_CANDIDATE_SECONDS
            and type(candidate_budget) is float
            and candidate_elapsed <= candidate_budget + 1e-6
        ),
        "candidate_and_reserve_match_40_60_split": bool(
            type(initial_budget) is float
            and type(candidate_budget) is float
            and type(materializer_reserve) is float
            and math.isclose(
                candidate_budget,
                0.4 * initial_budget,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            and math.isclose(
                materializer_reserve,
                0.6 * initial_budget,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ),
        "positive_baseline_upper": bool(
            type(before_upper) is float
            and math.isfinite(before_upper)
            and before_upper > 0.0
        ),
        "two_independently_certified_lp_bounds": bool(
            lp.get("status") == "compared"
            and lp.get("independent_lp_call_count") == 2
            and before.get("status") == "certified_diagnostic_upper"
            and after.get("status") == "certified_diagnostic_upper"
            and before.get("proof_authority") is False
            and after.get("proof_authority") is False
            and type(before_upper) is float
            and math.isfinite(before_upper)
            and type(after_upper) is float
            and math.isfinite(after_upper)
        ),
        "split_block_lp_certificate_no_stack_route": bool(
            _rbs_adaptive_k4_split_lp_certificate_valid(
                before,
                transaction.get("source_shape", {}),
            )
            and _rbs_adaptive_k4_split_lp_certificate_valid(
                after,
                transaction.get("fresh_private_shape", {}),
            )
        ),
        "native_split_objective_dual_candidate_route": bool(
            _rbs_adaptive_k4_objective_dual_proposal_valid(
                before,
                transaction.get("source_shape", {}),
                expected_kept_nonzeros=transaction.get(
                    "source_exact_kept_candidate_nonzeros"
                ),
            )
            and _rbs_adaptive_k4_objective_dual_proposal_valid(
                after,
                transaction.get("fresh_private_shape", {}),
            )
            and before_proposal.get(
                "maximization_factor_objective_sha256"
            )
            == after_proposal.get(
                "maximization_factor_objective_sha256"
            )
            and before_proposal.get("solver_cost_sha256")
            == after_proposal.get("solver_cost_sha256")
        ),
        "lp_drop_fields_recomputed": bool(
            type(improvement) is float
            and type(relative_drop) is float
            and recomputed_improvement is not None
            and recomputed_relative_drop is not None
            and math.isclose(
                improvement,
                recomputed_improvement,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                relative_drop,
                recomputed_relative_drop,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "absolute_drop_at_least_1": bool(
            recomputed_improvement is not None
            and recomputed_improvement
            >= _RBS_ADAPTIVE_K4_MIN_ABSOLUTE_DROP
        ),
        "relative_drop_at_least_10_percent": bool(
            recomputed_relative_drop is not None
            and recomputed_relative_drop
            >= _RBS_ADAPTIVE_K4_MIN_RELATIVE_DROP
        ),
    }
    conditions.update(_rbs_adaptive_k4_route_checks(transaction))
    failures = [
        name for name, passed in conditions.items() if passed is not True
    ]
    return _checksummed({
        "schema": "act.rbs_adaptive_k4_post_gate.v1",
        "status": "passed" if not failures else "rejected",
        "promoted": not failures,
        "candidate_only": True,
        "proof_authority": False,
        "verdict_authority": False,
        "conditions": conditions,
        "failed_conditions": failures,
        "minimum_absolute_drop": _RBS_ADAPTIVE_K4_MIN_ABSOLUTE_DROP,
        "minimum_relative_drop": _RBS_ADAPTIVE_K4_MIN_RELATIVE_DROP,
        "before_upper": before_upper,
        "after_upper": after_upper,
        "absolute_drop": recomputed_improvement,
        "relative_drop": recomputed_relative_drop,
        "pipeline_receipt_sha256": transaction.get(
            "pipeline_receipt_sha256"
        ),
    })


def _finalize_rbs_adaptive_k4_integrity(body: dict[str, Any]) -> None:
    """Veto adaptive promotion if final input/resource/time integrity moved."""

    if body.get("candidate_mode") != "rbs_adaptive_k4":
        return
    experiment = body.get("rbs_adaptive_k4")
    resources = body.get("resource_usage")
    if experiment is None:
        return
    if not isinstance(experiment, dict) or not isinstance(resources, dict):
        body["phase_status"] = "error"
        body["failed_stage"] = "rbs_adaptive_terminal_integrity"
        body["error_type"] = "PhaseCliqueBuildProbeError"
        body["error"] = "adaptive terminal receipt is malformed"
        return
    post_gate = experiment.get("post_gate")
    if not isinstance(post_gate, dict):
        return
    conditions = post_gate.get("conditions")
    if not isinstance(conditions, dict):
        body["phase_status"] = "error"
        body["failed_stage"] = "rbs_adaptive_terminal_integrity"
        body["error_type"] = "PhaseCliqueBuildProbeError"
        body["error"] = "adaptive post-gate conditions are malformed"
        return
    rss = resources.get("peak_rss_bytes")
    current_rss = resources.get("current_rss_bytes")
    allocated = resources.get("cuda_peak_allocated_bytes")
    total = body.get("timings", {}).get("total_seconds")
    conditions["final_inputs_unchanged"] = body.get("inputs_unchanged") is True
    conditions["final_implementation_unchanged"] = bool(
        body.get("implementation_sha256_after")
        == body.get("implementation_sha256")
        and body.get("implementation_integrity_error_type") is None
    )
    conditions["final_peak_rss_within_2_5_gib"] = bool(
        type(rss) is int and 0 <= rss <= _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
    )
    conditions["final_current_rss_within_2_5_gib"] = bool(
        type(current_rss) is int
        and 0 <= current_rss <= _RBS_ADAPTIVE_K4_MAX_RSS_BYTES
    )
    conditions["final_cuda_allocated_within_8_gib"] = bool(
        resources.get("cuda_initialized") is True
        and type(allocated) is int
        and 0
        <= allocated
        <= _RBS_ADAPTIVE_K4_MAX_CUDA_ALLOCATED_BYTES
    )
    conditions["final_total_under_60_seconds"] = bool(
        type(total) is float and 0.0 <= total < _MAX_WALL_SECONDS
    )
    conditions["final_shared_worker_deadline_met"] = bool(
        body.get("shared_worker_deadline_met") is True
    )
    promoted = bool(post_gate.get("promoted")) and all(
        value is True for value in conditions.values()
    )
    post_gate["promoted"] = promoted
    post_gate["status"] = "passed" if promoted else "rejected"
    post_gate["failed_conditions"] = [
        name for name, passed in conditions.items() if passed is not True
    ]
    post_gate.pop("receipt_sha256", None)
    post_gate["receipt_sha256"] = hashlib.sha256(
        _canonical_json(post_gate)
    ).hexdigest()
    experiment["resource_usage_final"] = dict(resources)
    if promoted:
        experiment["status"] = "rbs_adaptive_k4_promoted_diagnostic"
        experiment["reason"] = None
    elif experiment.get("phase_clique_attempted") is True:
        experiment["status"] = "rbs_adaptive_k4_post_gate_rejected"
        experiment["reason"] = "post_gate_or_terminal_integrity_rejected"
    body["phase_status"] = experiment.get("status")
    body["fallback_reason"] = experiment.get("reason")
    if type(total) is float:
        body["completed_before_deadline"] = bool(
            total < float(body.get("wall_timeout_seconds", 0.0))
        )


def _finalize_pcoh_k2_integrity(body: dict[str, Any]) -> None:
    """Release the process-local anchor even if terminal replay is interrupted."""

    registered_transaction = body.get("pcoh_k2_build_only")
    try:
        if body.get("candidate_mode") != _PCOH_K2_BUILD_ONLY_MODE:
            return
        _finalize_pcoh_k2_integrity_registered(body)
    finally:
        _release_pcoh_k2_trusted_transaction(registered_transaction)


def _finalize_pcoh_k2_integrity_registered(body: dict[str, Any]) -> None:
    """Fail closed while preserving the two-status build-only contract."""

    transaction = body.get("pcoh_k2_build_only")
    registered_transaction = transaction
    trusted_summary_sha256 = _pcoh_k2_trusted_transaction_anchor(
        registered_transaction
    )
    resources = body.get("resource_usage")
    original_transaction_valid = _pcoh_k2_transaction_receipt_valid(
        transaction
    )
    original_transaction_called = bool(
        type(transaction) is dict
        and transaction.get("build_only_transaction_called") is True
    )
    transaction_valid = original_transaction_valid
    source_preflight = body.get("pcoh_source_build_preflight")
    source_preflight_valid = _pcoh_k2_source_build_preflight_valid(
        source_preflight
    )
    if not original_transaction_valid:
        fallback_resources = (
            dict(resources)
            if type(resources) is dict
            else _capture_resource_peaks()
        )
        transaction = _pcoh_k2_stop_loss_receipt(
            stage="terminal_receipt_integrity",
            reason="pcoh_transaction_receipt_malformed",
            started=time.monotonic(),
            input_sha256=(
                body.get("input_sha256")
                if type(body.get("input_sha256")) is dict
                else {}
            ),
            implementation_sha256=(
                body.get("implementation_sha256")
                if type(body.get("implementation_sha256")) is dict
                else {}
            ),
            stage_resources={"terminal_integrity": fallback_resources},
            build_only_transaction_called=original_transaction_called,
        )
        body["pcoh_k2_build_only"] = transaction
        transaction_valid = _pcoh_k2_transaction_receipt_valid(transaction)

    total = body.get("timings", {}).get("total_seconds")
    rss = resources.get("peak_rss_bytes") if type(resources) is dict else None
    current_rss = (
        resources.get("current_rss_bytes")
        if type(resources) is dict
        else None
    )
    transaction_status = (
        transaction.get("status") if type(transaction) is dict else None
    )
    materialized_tightness_valid = bool(
        transaction_valid
        and transaction_status == "built_and_released"
        and _valid_sha256(trusted_summary_sha256)
        and transaction.get("materialized_tightness_summary_sha256")
        == trusted_summary_sha256
        and _pcoh_k2_materialized_tightness_payload_valid(
            transaction.get("materialized_tightness_summary"),
            source_semantic_digest=transaction.get(
                "source_semantic_digest"
            ),
            stable_bit_ids=transaction.get("stable_bit_ids"),
            conditional_certificate_sha256=transaction.get(
                "conditional_certificate_sha256"
            ),
            expected_summary_sha256=trusted_summary_sha256,
        )
    )
    tightness_gate_valid = bool(
        materialized_tightness_valid
        and _pcoh_k2_tightness_gate_valid(
            transaction.get("tightness_gate"),
            transaction.get("materialized_tightness_summary"),
            expected_summary_sha256=trusted_summary_sha256,
        )
    )
    conditions = {
        "original_transaction_receipt_valid": original_transaction_valid,
        "replacement_stop_receipt_valid": transaction_valid,
        "transaction_status_allowed": transaction_status
        in {"built_and_released", "stop_loss"},
        "transaction_built_and_released": (
            transaction_status == "built_and_released"
        ),
        "source_build_preflight_valid_and_passed": bool(
            source_preflight_valid
            and source_preflight.get("status") == "passed"
            and source_preflight.get("input_sha256")
            == body.get("input_sha256")
            and source_preflight.get("implementation_sha256")
            == body.get("implementation_sha256")
        ),
        "fixed_family": body.get("family") == _PCOH_K2_FAMILY,
        "fixed_iid": body.get("iid") == _ONLY_IID,
        "fixed_wall_timeout": (
            body.get("wall_timeout_seconds") == _MAX_WALL_SECONDS
        ),
        "fixed_phase_time_limit": bool(
            type(body.get("phase_time_limit_seconds")) is float
            and body["phase_time_limit_seconds"]
            == _PCOH_K2_MAX_PHASE_SECONDS
        ),
        "fixed_operator_exact_budget": body.get("operator_exact_budget") == 4,
        "fixed_residual_budget": body.get("residual_budget") == 4,
        "fixed_residual_time_limit": (
            body.get("residual_time_limit_seconds")
            == _PCOH_K2_RESIDUAL_SECONDS
        ),
        "fixed_cpu_threads": body.get("cpu_threads") == 20,
        "top_diagnostic_only": body.get("diagnostic_only") is True,
        "top_candidate_only": body.get("candidate_only") is True,
        "top_build_only": body.get("build_only") is True,
        "top_instance_count_one": body.get("instance_count") == 1,
        "top_proof_authority_false": body.get("proof_authority") is False,
        "top_verdict_authority_false": body.get("verdict_authority") is False,
        "top_ground_truth_false": body.get("ground_truth_loaded") is False,
        "top_reference_false": body.get("reference_label_used") is False,
        "solver_handoff_not_called": body.get("solver_handoff_called") is False,
        "diagnostic_lp_not_called": body.get("diagnostic_lp_called") is False,
        "hz_base_feasibility_not_called": (
            body.get("hz_base_feasibility_called") is False
        ),
        "hz_objbound_decide_not_called": (
            body.get("hz_objbound_decide_called") is False
        ),
        "strict_replay_not_called": body.get("strict_replay_called") is False,
        "full_parent_lp_not_called": bool(
            body.get("full_parent_lp_called") is False
            and transaction_valid
            and transaction.get("full_parent_lp_called") is False
        ),
        "full_parent_lp_solver_not_called": bool(
            body.get("full_parent_lp_solver_called") is False
            and transaction_valid
            and transaction.get("full_parent_lp_solver_called") is False
        ),
        "materialized_tightness_strictly_verified": (
            materialized_tightness_valid
        ),
        "tightness_gate_strictly_replayed": tightness_gate_valid,
        "certified_edge_count_zero": body.get("certified_edge_count") == 0,
        "fixed_input_sha256": body.get("input_sha256")
        == {
            "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
            "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
            "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
        },
        "input_sha_bound": bool(
            transaction_valid
            and transaction.get("input_sha256") == body.get("input_sha256")
        ),
        "inputs_unchanged": bool(
            body.get("inputs_unchanged") is True
            and body.get("input_sha256_after") == body.get("input_sha256")
        ),
        "implementation_sha_bound": bool(
            transaction_valid
            and transaction.get("implementation_sha256")
            == body.get("implementation_sha256")
        ),
        "implementation_unchanged": bool(
            body.get("implementation_unchanged") is True
            and body.get("implementation_sha256_after")
            == body.get("implementation_sha256")
        ),
        "terminal_peak_rss_recorded": bool(
            type(rss) is int and 0 <= rss <= _PCOH_K2_MAX_RSS_BYTES
        ),
        "terminal_current_rss_recorded": bool(
            current_rss is None
            or (
                type(current_rss) is int
                and 0 <= current_rss <= _PCOH_K2_MAX_RSS_BYTES
            )
        ),
        "stage_rss_recorded": bool(
            transaction_valid
            and type(transaction.get("stage_resources")) is dict
            and bool(transaction["stage_resources"])
        ),
        "nested_authority_firewall_matches_top": bool(
            transaction_valid
            and transaction.get("diagnostic_only")
            is body.get("diagnostic_only")
            and transaction.get("candidate_only")
            is body.get("candidate_only")
            and transaction.get("build_only") is body.get("build_only")
            and transaction.get("instance_count") == body.get("instance_count")
            and transaction.get("proof_authority")
            is body.get("proof_authority")
            and transaction.get("verdict_authority")
            is body.get("verdict_authority")
            and transaction.get("ground_truth_loaded")
            is body.get("ground_truth_loaded")
            and transaction.get("reference_label_used")
            is body.get("reference_label_used")
            and transaction.get("full_parent_lp_called")
            is body.get("full_parent_lp_called")
            and transaction.get("full_parent_lp_solver_called")
            is body.get("full_parent_lp_solver_called")
        ),
        "total_under_60_seconds": bool(
            type(total) is float
            and math.isfinite(total)
            and 0.0 <= total < _MAX_WALL_SECONDS
        ),
        "shared_worker_deadline_met": (
            body.get("shared_worker_deadline_met") is True
        ),
    }
    failures = [
        name for name, passed in conditions.items() if passed is not True
    ]
    terminal_success = bool(
        not failures and transaction_status == "built_and_released"
    )
    body["pcoh_transaction_sha256"] = (
        transaction.get("receipt_sha256")
        if transaction_valid
        else None
    )
    body["pcoh_diagnostic_sha256"] = (
        transaction.get("diagnostic_sha256")
        if transaction_valid
        else None
    )
    body["pcoh_source_build_preflight_sha256"] = (
        source_preflight.get("receipt_sha256")
        if source_preflight_valid
        else None
    )
    body["pcoh_stage_resources"] = (
        dict(transaction.get("stage_resources", {}))
        if transaction_valid
        else {}
    )
    body["pcoh_materialized_tightness_summary_sha256"] = (
        transaction.get("materialized_tightness_summary_sha256")
        if materialized_tightness_valid
        else None
    )
    body["pcoh_tightness_gate_sha256"] = (
        transaction.get("tightness_gate", {}).get("receipt_sha256")
        if tightness_gate_valid
        else None
    )
    body["pcoh_terminal_integrity"] = _checksummed({
        "schema": _PCOH_K2_INTEGRITY_SCHEMA,
        "status": "built_and_released" if terminal_success else "stop_loss",
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": True,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "conditions": conditions,
        "failed_conditions": failures,
        "transaction_sha256": body["pcoh_transaction_sha256"],
        "diagnostic_sha256": body["pcoh_diagnostic_sha256"],
        "source_build_preflight_sha256": body[
            "pcoh_source_build_preflight_sha256"
        ],
        "materialized_tightness_summary_sha256": body[
            "pcoh_materialized_tightness_summary_sha256"
        ],
        "tightness_gate_sha256": body["pcoh_tightness_gate_sha256"],
        "terminal_resource_usage": (
            dict(resources) if type(resources) is dict else None
        ),
    })
    body["phase_status"] = (
        "built_and_released" if terminal_success else "stop_loss"
    )
    if terminal_success:
        body["failed_stage"] = None
        body["fallback_reason"] = None
        body["error_type"] = None
        body["error"] = None
    else:
        if body.get("failed_stage") is None:
            body["failed_stage"] = "pcoh_terminal_integrity"
        if body.get("fallback_reason") is None:
            body["fallback_reason"] = (
                "pcoh_stop_loss:"
                + ",".join(failures or [str(transaction.get("reason"))])
            )[:300]
        body["error_type"] = None
        body["error"] = None
    if type(total) is float:
        body["completed_before_deadline"] = bool(
            total < float(body.get("wall_timeout_seconds", 0.0))
        )


def _pcoh_k3_emergency_checksummed(
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a K3 failure receipt without depending on the normal finalizer."""

    result = dict(body)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = hashlib.sha256(
        _canonical_json(result)
    ).hexdigest()
    return result


def _pcoh_k3_install_terminal_failure(
    body: dict[str, Any],
    *,
    transaction: Any,
    detail: str,
    failed_stage: str,
) -> None:
    """Install a separately sealed fail-closed K3 terminal diagnosis."""

    anchor = _pcoh_k3_trusted_transaction_anchor(transaction)
    basic_valid = _pcoh_k3_transaction_basic_receipt_valid(transaction)
    try:
        strict_valid = _pcoh_k3_transaction_receipt_valid(transaction)
    except BaseException as exc:
        _clear_pcoh_k3_exception_traceback(exc)
        strict_valid = False
    transaction_status = (
        transaction.get("status") if basic_valid else None
    )
    transaction_sha256 = (
        transaction.get("receipt_sha256") if basic_valid else None
    )
    trusted_resource_stop = bool(
        strict_valid
        and transaction_status == "resource_stop"
        and type(anchor) is _PCOHK3TrustedTransactionAnchor
        and anchor.outcome_kind == "resource_stop"
    )
    counter_names = (
        "pair_local_lp_actual_calls",
        "conditional_local_lp_actual_calls",
        "total_local_lp_actual_calls",
        "conditional_checker_actual_calls",
    )
    prior_integrity = body.get("pcoh_k3_terminal_integrity")
    try:
        prior_integrity_valid = bool(
            type(prior_integrity) is dict
            and _local_receipt_checksum_valid(
                prior_integrity, schema=_PCOH_K3_INTEGRITY_SCHEMA
            )
        )
    except BaseException as exc:
        _clear_pcoh_k3_exception_traceback(exc)
        prior_integrity_valid = False
    prior_trusted_resource_stop = bool(
        basic_valid
        and transaction_status == "resource_stop"
        and prior_integrity_valid
        and prior_integrity.get("conditions", {}).get(
            "original_transaction_receipt_valid"
        )
        is True
        and prior_integrity.get("original_transaction_sha256")
        == transaction_sha256
        and prior_integrity.get("trusted_outcome_sha256")
        == transaction.get("trusted_outcome_sha256")
        and all(
            prior_integrity.get(name) == transaction.get(name)
            for name in counter_names
        )
    )
    trusted_resource_stop = bool(
        trusted_resource_stop or prior_trusted_resource_stop
    )
    counters = tuple(
        transaction.get(name) if trusted_resource_stop else 0
        for name in counter_names
    )
    source_preflight = body.get("pcoh_k3_source_build_preflight")
    try:
        source_preflight_valid = (
            _pcoh_k3_source_build_preflight_valid(source_preflight)
        )
    except BaseException as exc:
        _clear_pcoh_k3_exception_traceback(exc)
        source_preflight_valid = False
    resources = body.get("resource_usage")
    trusted_outcome_sha256 = (
        anchor.outcome_sha256
        if trusted_resource_stop
        and type(anchor) is _PCOHK3TrustedTransactionAnchor
        else transaction.get("trusted_outcome_sha256")
        if prior_trusted_resource_stop
        else None
    )
    body["pcoh_k3_transaction_sha256"] = transaction_sha256
    body["pcoh_k3_trusted_outcome_sha256"] = trusted_outcome_sha256
    body["pcoh_k3_source_build_preflight_sha256"] = (
        source_preflight.get("receipt_sha256")
        if source_preflight_valid
        else None
    )
    body["pcoh_k3_baseline_anchor_receipt_sha256"] = None
    body["pcoh_k3_strong_tightness_gate_sha256"] = None
    body["pcoh_k3_terminal_integrity"] = _pcoh_k3_emergency_checksummed({
        "schema": _PCOH_K3_INTEGRITY_SCHEMA,
        "status": "stop_loss",
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": True,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "conditions": {"finalizer_completed": False},
        "failed_conditions": ["finalizer_completed"],
        "terminal_integrity_passed": False,
        "transaction_status": transaction_status,
        "transaction_terminal_candidate": transaction_status in {
            "strong_promotion",
            "strong_target_stop",
            "built_but_not_strong",
            "resource_stop",
        },
        "transaction_stop_loss": transaction_status == "stop_loss",
        "original_transaction_preserved": bool(
            basic_valid and body.get("pcoh_k3_build_only") is transaction
        ),
        "original_transaction_sha256": transaction_sha256,
        "transaction_sha256": transaction_sha256,
        "trusted_outcome_sha256": trusted_outcome_sha256,
        "source_build_preflight_sha256": body[
            "pcoh_k3_source_build_preflight_sha256"
        ],
        "baseline_anchor_receipt_sha256": None,
        "strong_tightness_gate_sha256": None,
        "pair_local_lp_actual_calls": counters[0],
        "conditional_local_lp_actual_calls": counters[1],
        "total_local_lp_actual_calls": counters[2],
        "conditional_checker_actual_calls": counters[3],
        "terminal_resource_usage": (
            dict(resources) if type(resources) is dict else None
        ),
        "terminal_failed_stage": failed_stage,
        "reason": detail,
    })
    body["phase_status"] = "stop_loss"
    body["failed_stage"] = failed_stage
    body["fallback_reason"] = f"pcoh_k3_stop_loss:{detail}"[:300]
    body["error_type"] = None
    body["error"] = None


def _finalize_pcoh_k3_integrity(body: dict[str, Any]) -> None:
    """Replay K3 without replacing its receipt; release local authority."""

    registered_transaction = body.get("pcoh_k3_build_only")
    try:
        if (
            body.get("candidate_mode") != _PCOH_K3_BUILD_ONLY_MODE
            and registered_transaction is None
        ):
            return
        _finalize_pcoh_k3_integrity_registered(body)
    except BaseException as exc:
        detail = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_pcoh_k3_exception_traceback(exc)
        try:
            _pcoh_k3_install_terminal_failure(
                body,
                transaction=registered_transaction,
                detail=detail,
                failed_stage="pcoh_k3_terminal_finalizer",
            )
        except BaseException as nested:
            _clear_pcoh_k3_exception_traceback(nested)
            body["pcoh_k3_terminal_integrity"] = None
    finally:
        _release_pcoh_k3_trusted_transaction(registered_transaction)
        current = body.get("pcoh_k3_build_only")
        if current is not registered_transaction:
            _release_pcoh_k3_trusted_transaction(current)
        registered_transaction = None


def _finalize_pcoh_k3_integrity_registered(body: dict[str, Any]) -> None:
    """Bind K3 and record terminal integrity separately from its receipt."""

    transaction = body.get("pcoh_k3_build_only")
    registered_transaction = transaction
    anchor = _pcoh_k3_trusted_transaction_anchor(registered_transaction)
    original_transaction_valid = _pcoh_k3_transaction_receipt_valid(
        transaction
    )
    resources = body.get("resource_usage")
    source_preflight = body.get("pcoh_k3_source_build_preflight")
    source_preflight_valid = _pcoh_k3_source_build_preflight_valid(
        source_preflight
    )
    terminal_baseline = None
    terminal_baseline_error = None
    try:
        parent_hard = body.get("parent_hard_deadline_monotonic")
        shared = body.get("shared_worker_deadline_monotonic")
        terminal_deadline = min(
            float(parent_hard)
            - _parent_term_reserve_seconds(
                float(body.get("wall_timeout_seconds"))
            ),
            float(shared) + _PCOH_K3_FINALIZATION_RESERVE_SECONDS,
        )
        terminal_baseline = _pcoh_k3_fixed_baseline_artifact_anchor(
            deadline=float(terminal_deadline)
        )
    except BaseException as exc:
        terminal_baseline_error = f"{type(exc).__name__}:{str(exc)[:200]}"
        _clear_pcoh_k3_exception_traceback(exc)
    terminal_baseline_valid = _pcoh_k3_baseline_anchor_receipt_valid(
        terminal_baseline
    )

    transaction_status = (
        transaction.get("status") if type(transaction) is dict else None
    )
    transaction_sha256 = (
        transaction.get("receipt_sha256")
        if type(transaction) is dict
        and _valid_sha256(transaction.get("receipt_sha256"))
        else None
    )
    transaction_terminal_candidate = transaction_status in {
        "strong_promotion",
        "strong_target_stop",
        "built_but_not_strong",
        "resource_stop",
    }
    transaction_stop_loss = transaction_status == "stop_loss"
    total = body.get("timings", {}).get("total_seconds")
    rss = resources.get("peak_rss_bytes") if type(resources) is dict else None
    current_rss = (
        resources.get("current_rss_bytes")
        if type(resources) is dict
        else None
    )
    top_counts = tuple(
        body.get(name)
        for name in (
            "pair_local_lp_actual_calls",
            "conditional_local_lp_actual_calls",
            "total_local_lp_actual_calls",
            "conditional_checker_actual_calls",
        )
    )
    nested_counts = tuple(
        transaction.get(name) if type(transaction) is dict else None
        for name in (
            "pair_local_lp_actual_calls",
            "conditional_local_lp_actual_calls",
            "total_local_lp_actual_calls",
            "conditional_checker_actual_calls",
        )
    )
    source_baseline = (
        source_preflight.get("baseline_anchor_receipt")
        if type(source_preflight) is dict
        else None
    )
    terminal_baseline_sha = (
        terminal_baseline.get("receipt_sha256")
        if terminal_baseline_valid
        else None
    )
    conditions = {
        "original_transaction_receipt_valid": original_transaction_valid,
        "transaction_status_allowed": transaction_status
        in {
            "strong_promotion",
            "strong_target_stop",
            "built_but_not_strong",
            "resource_stop",
            "stop_loss",
        },
        "source_build_preflight_valid_and_passed": bool(
            source_preflight_valid
            and source_preflight.get("status") == "passed"
            and source_preflight.get("input_sha256")
            == body.get("input_sha256")
            and source_preflight.get("implementation_sha256")
            == body.get("implementation_sha256")
        ),
        "terminal_fixed_baseline_reverified": terminal_baseline_valid,
        "baseline_anchor_bound_end_to_end": bool(
            terminal_baseline_valid
            and type(source_baseline) is dict
            and source_baseline == terminal_baseline
            and type(transaction) is dict
            and transaction.get("baseline_anchor_verified") is True
            and transaction.get("baseline_anchor_receipt_sha256")
            == terminal_baseline_sha
        ),
        "fixed_candidate_mode": (
            type(body.get("candidate_mode")) is str
            and body.get("candidate_mode") == _PCOH_K3_BUILD_ONLY_MODE
        ),
        "fixed_family": bool(
            type(body.get("family")) is str
            and body.get("family") == _PCOH_K3_FAMILY
        ),
        "fixed_iid": bool(
            type(body.get("iid")) is int
            and body.get("iid") == _ONLY_IID
        ),
        "fixed_wall_timeout": bool(
            type(body.get("wall_timeout_seconds")) is float
            and body.get("wall_timeout_seconds") == _MAX_WALL_SECONDS
        ),
        "fixed_phase_cli_and_internal_contract": bool(
            type(body.get("phase_time_limit_seconds")) is float
            and body.get("phase_time_limit_seconds")
            == _PCOH_K3_CLI_PHASE_SECONDS
            and type(transaction) is dict
            and transaction.get("timings", {}).get("total_seconds")
            is not None
        ),
        "fixed_operator_and_residual_budgets": bool(
            type(body.get("operator_exact_budget")) is int
            and body.get("operator_exact_budget") == 4
            and type(body.get("residual_budget")) is int
            and body.get("residual_budget") == 4
            and type(body.get("residual_time_limit_seconds")) is float
            and body.get("residual_time_limit_seconds")
            == _PCOH_K3_RESIDUAL_SECONDS
            and type(body.get("cpu_threads")) is int
            and body.get("cpu_threads") == 20
        ),
        "top_authority_firewall": bool(
            body.get("diagnostic_only") is True
            and body.get("candidate_only") is True
            and body.get("build_only") is True
            and type(body.get("instance_count")) is int
            and body.get("instance_count") == 1
            and body.get("proof_authority") is False
            and body.get("verdict_authority") is False
            and body.get("ground_truth_loaded") is False
            and body.get("reference_label_used") is False
        ),
        "forbidden_routes_not_called": bool(
            body.get("solver_handoff_called") is False
            and body.get("hz_base_feasibility_called") is False
            and body.get("hz_objbound_decide_called") is False
            and body.get("full_parent_lp_called") is False
            and body.get("full_parent_lp_solver_called") is False
            and body.get("k2_build_only_called") is False
            and body.get("phase_transaction_called") is False
            and type(transaction) is dict
            and transaction.get("k2_build_only_called") is False
            and transaction.get("phase_transaction_called") is False
            and transaction.get("solver_handoff_called") is False
            and transaction.get("full_parent_lp_called") is False
            and transaction.get("full_parent_lp_solver_called") is False
        ),
        "transaction_call_flags_bound": bool(
            type(transaction) is dict
            and type(body.get("k3_transaction_called")) is bool
            and body.get("k3_transaction_called")
            is transaction.get("k3_transaction_called")
            and type(body.get("k2_build_only_called")) is bool
            and body.get("k2_build_only_called")
            is transaction.get("k2_build_only_called")
            and type(body.get("phase_transaction_called")) is bool
            and body.get("phase_transaction_called")
            is transaction.get("phase_transaction_called")
        ),
        "explicit_actual_counts_bound": bool(
            top_counts == nested_counts
            and all(type(item) is int and item >= 0 for item in top_counts)
            and top_counts[2] == top_counts[0] + top_counts[1]
            and top_counts[2] <= 20
            and top_counts[3] <= 34
        ),
        "fixed_call_caps_and_zero_edges": bool(
            type(body.get("local_lp_actual_call_cap")) is int
            and body.get("local_lp_actual_call_cap") == 20
            and type(body.get("conditional_checker_actual_call_cap")) is int
            and body.get("conditional_checker_actual_call_cap") == 34
            and type(transaction) is dict
            and type(transaction.get("local_lp_actual_call_cap")) is int
            and transaction.get("local_lp_actual_call_cap") == 20
            and type(transaction.get("conditional_checker_actual_call_cap"))
            is int
            and transaction.get("conditional_checker_actual_call_cap") == 34
            and type(body.get("certified_edge_count")) is int
            and body.get("certified_edge_count") == 0
        ),
        "fixed_input_sha256": bool(
            body.get("input_sha256")
            == {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            and type(transaction) is dict
            and transaction.get("input_sha256")
            == body.get("input_sha256")
        ),
        "inputs_unchanged": bool(
            body.get("inputs_unchanged") is True
            and body.get("input_sha256_after") == body.get("input_sha256")
        ),
        "implementation_bound_and_unchanged": bool(
            type(transaction) is dict
            and transaction.get("implementation_sha256")
            == body.get("implementation_sha256")
            and body.get("implementation_unchanged") is True
            and body.get("implementation_sha256_after")
            == body.get("implementation_sha256")
        ),
        "terminal_resources_recorded": bool(
            type(rss) is int
            and 0 <= rss <= _PCOH_K2_MAX_RSS_BYTES
            and (
                current_rss is None
                or type(current_rss) is int
                and 0 <= current_rss <= _PCOH_K2_MAX_RSS_BYTES
            )
        ),
        "total_under_60_seconds": bool(
            type(total) is float
            and math.isfinite(total)
            and 0.0 <= total < _MAX_WALL_SECONDS
        ),
        "shared_worker_deadline_met": (
            body.get("shared_worker_deadline_met") is True
        ),
    }
    failures = [
        name for name, passed in conditions.items() if passed is not True
    ]
    terminal_integrity_passed = bool(not failures)
    trusted_original_resource_stop = bool(
        original_transaction_valid
        and transaction_status == "resource_stop"
        and type(anchor) is _PCOHK3TrustedTransactionAnchor
        and anchor.outcome_kind == "resource_stop"
    )
    evidence_counts = (
        nested_counts
        if trusted_original_resource_stop
        else top_counts
        if terminal_integrity_passed
        else (0, 0, 0, 0)
    )

    gate = (
        registered_transaction.get("strong_tightness_gate")
        if terminal_integrity_passed
        and type(registered_transaction) is dict
        and transaction_status == "strong_promotion"
        else None
    )
    body["pcoh_k3_transaction_sha256"] = transaction_sha256
    body["pcoh_k3_trusted_outcome_sha256"] = (
        anchor.outcome_sha256
        if (terminal_integrity_passed or trusted_original_resource_stop)
        and type(anchor) is _PCOHK3TrustedTransactionAnchor
        else None
    )
    body["pcoh_k3_source_build_preflight_sha256"] = (
        source_preflight.get("receipt_sha256")
        if source_preflight_valid
        else None
    )
    body["pcoh_k3_baseline_anchor_receipt_sha256"] = terminal_baseline_sha
    body["pcoh_k3_strong_tightness_gate_sha256"] = (
        gate.get("receipt_sha256") if type(gate) is dict else None
    )
    terminal_status = (
        transaction_status if terminal_integrity_passed else "stop_loss"
    )
    terminal_failed_stage = (
        None
        if terminal_integrity_passed
        else "pcoh_k3_terminal_integrity"
    )
    terminal_reason = (
        None
        if terminal_integrity_passed
        else (
            "terminal_integrity_rejected:"
            + ",".join(failures)
            + (
                f":{terminal_baseline_error}"
                if terminal_baseline_error
                else ""
            )
        )[:300]
    )
    body["pcoh_k3_terminal_integrity"] = _checksummed({
        "schema": _PCOH_K3_INTEGRITY_SCHEMA,
        "status": terminal_status,
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": True,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "conditions": conditions,
        "failed_conditions": failures,
        "terminal_integrity_passed": terminal_integrity_passed,
        "transaction_status": transaction_status,
        "transaction_terminal_candidate": transaction_terminal_candidate,
        "transaction_stop_loss": transaction_stop_loss,
        "original_transaction_preserved": (
            type(registered_transaction) is dict
            and body.get("pcoh_k3_build_only") is registered_transaction
        ),
        "original_transaction_sha256": transaction_sha256,
        "transaction_sha256": body["pcoh_k3_transaction_sha256"],
        "trusted_outcome_sha256": body[
            "pcoh_k3_trusted_outcome_sha256"
        ],
        "source_build_preflight_sha256": body[
            "pcoh_k3_source_build_preflight_sha256"
        ],
        "baseline_anchor_receipt_sha256": terminal_baseline_sha,
        "strong_tightness_gate_sha256": body[
            "pcoh_k3_strong_tightness_gate_sha256"
        ],
        "pair_local_lp_actual_calls": evidence_counts[0],
        "conditional_local_lp_actual_calls": evidence_counts[1],
        "total_local_lp_actual_calls": evidence_counts[2],
        "conditional_checker_actual_calls": evidence_counts[3],
        "terminal_resource_usage": (
            dict(resources) if type(resources) is dict else None
        ),
        "terminal_failed_stage": terminal_failed_stage,
        "reason": (
            None
            if terminal_integrity_passed
            and transaction_status == "strong_promotion"
            else "strong_target_stop"
            if terminal_integrity_passed
            and transaction_status == "strong_target_stop"
            else "built_but_not_strong"
            if terminal_integrity_passed
            and transaction_status == "built_but_not_strong"
            else transaction.get("reason")
            if terminal_integrity_passed
            and transaction_status == "resource_stop"
            and type(transaction) is dict
            else "upstream_stop_loss_preserved"
            if terminal_integrity_passed and transaction_stop_loss
            else terminal_reason
        ),
    })
    body["phase_status"] = terminal_status
    body["failed_stage"] = (
        transaction.get("failed_stage")
        if terminal_integrity_passed
        and (transaction_stop_loss or transaction_status == "resource_stop")
        and type(transaction) is dict
        else None
        if terminal_integrity_passed
        else "pcoh_k3_terminal_integrity"
    )
    body["fallback_reason"] = (
        None
        if terminal_integrity_passed
        and transaction_status == "strong_promotion"
        else "strong_target_stop"
        if terminal_integrity_passed
        and transaction_status == "strong_target_stop"
        else "built_but_not_strong"
        if terminal_integrity_passed
        and transaction_status == "built_but_not_strong"
        else transaction.get("reason")
        if terminal_integrity_passed
        and transaction_status == "resource_stop"
        and type(transaction) is dict
        else transaction.get("reason")
        if terminal_integrity_passed
        and transaction_stop_loss
        and type(transaction) is dict
        else ("pcoh_k3_stop_loss:" + ",".join(failures))[:300]
    )
    body["error_type"] = None
    body["error"] = None
    if type(total) is float:
        body["completed_before_deadline"] = bool(
            total < float(body.get("wall_timeout_seconds", 0.0))
        )


def _parent_term_reserve_seconds(wall_timeout: float) -> float:
    """Return the exact grace interval used before the parent hard stop."""

    return min(1.0, float(wall_timeout) / 2.0)


def _shared_worker_deadline(
    args: argparse.Namespace,
    *,
    now: Optional[float] = None,
) -> float:
    """Derive the child's work deadline from the parent's absolute deadline."""

    current = time.monotonic() if now is None else now
    parent_hard = getattr(args, "parent_hard_deadline_monotonic", None)
    if (
        type(current) is not float
        or not math.isfinite(current)
        or type(parent_hard) is not float
        or not math.isfinite(parent_hard)
    ):
        raise PhaseCliqueBuildProbeError(
            "parent/worker monotonic deadline is malformed"
        )
    wall_timeout = float(args.wall_timeout)
    remaining_to_parent_hard = parent_hard - current
    if not 0.0 < remaining_to_parent_hard <= wall_timeout + 1e-6:
        raise PhaseCliqueBuildProbeError(
            "parent hard deadline is expired or exceeds the wall contract"
        )
    finalization_reserve = (
        _RBS_ADAPTIVE_K4_FINALIZATION_RESERVE_SECONDS
        if args.candidate_mode == "rbs_adaptive_k4"
        else _PCOH_K3_FINALIZATION_RESERVE_SECONDS
        if args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE
        else _PCOH_K2_FINALIZATION_RESERVE_SECONDS
        if args.candidate_mode == _PCOH_K2_BUILD_ONLY_MODE
        else 0.0
    )
    deadline = (
        parent_hard
        - _parent_term_reserve_seconds(wall_timeout)
        - finalization_reserve
    )
    if deadline <= current:
        raise PhaseCliqueBuildProbeError(
            "shared worker deadline is already exhausted"
        )
    return float(deadline)


def _execute_probe_inner(
    args: argparse.Namespace,
    *,
    k3_cleanup_ref: list[Any],
) -> dict[str, Any]:
    """Execute inside the bounded child and always return a receipt body."""

    started = time.monotonic()
    deadline = _shared_worker_deadline(args, now=started)
    timings: dict[str, float] = {}
    failed_stage: Optional[str] = None
    torch_module: Any = None
    baseline_anchor_receipt: Optional[dict[str, Any]] = None
    instance = _select_instance(args.benchmark_root, args.family, args.iid)
    inputs = {
        "onnx": _sha256_file(instance.onnx_path),
        "vnnlib": _sha256_file(instance.vnnlib_path),
        "instances_csv": _sha256_file(instance.csv_path),
    }
    implementation = (
        _pcoh_k3_implementation_sha256()
        if args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE
        else _implementation_sha256()
    )
    body: dict[str, Any] = {
        "schema": _SCHEMA,
        "candidate_mode": args.candidate_mode,
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": bool(
            args.candidate_mode
            in {_PCOH_K2_BUILD_ONLY_MODE, _PCOH_K3_BUILD_ONLY_MODE}
        ),
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "hz_objbound_decide_called": False,
        "hz_base_feasibility_called": False,
        "solver_handoff_called": False,
        "diagnostic_lp_called": False,
        "strict_replay_called": False,
        "full_parent_lp_called": False,
        "full_parent_lp_solver_called": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "family": args.family,
        "iid": args.iid,
        "run_nonce": args.run_nonce,
        "fixed_environment_sha256": args.fixed_environment_sha256,
        "fixed_numerical_environment": {
            "HZ_QUERY_WORKERS": args.fixed_environment["HZ_QUERY_WORKERS"],
            "HZ_MILP_THREADS": args.fixed_environment["HZ_MILP_THREADS"],
            "HZ_LP_PREFILTER_THREADS": args.fixed_environment[
                "HZ_LP_PREFILTER_THREADS"
            ],
            "HZ_LP_PREFILTER_FRACTION": args.fixed_environment[
                "HZ_LP_PREFILTER_FRACTION"
            ],
            "HZ_LP_PREFILTER_MAX_SECONDS": args.fixed_environment[
                "HZ_LP_PREFILTER_MAX_SECONDS"
            ],
        },
        "wall_timeout_seconds": float(args.wall_timeout),
        "phase_time_limit_seconds": float(args.phase_time_limit),
        "operator_exact_budget": int(args.operator_exact_budget),
        "residual_budget": int(args.residual_budget),
        "residual_time_limit_seconds": float(args.residual_time_limit),
        "cpu_threads": int(args.cpu_threads),
        "parent_hard_deadline_monotonic": float(
            args.parent_hard_deadline_monotonic
        ),
        "shared_worker_deadline_monotonic": float(deadline),
        "shared_worker_deadline_met": False,
        "input_sha256": inputs,
        "implementation_sha256": implementation,
        "phase_status": "error",
        "failed_stage": None,
        "error_type": None,
        "error": None,
        "fallback_reason": None,
        "certified_edge_count": 0,
        "timings": timings,
    }
    if args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE:
        body.pop("diagnostic_lp_called", None)
        body.pop("strict_replay_called", None)
        body.update({
            "k3_transaction_called": False,
            "k2_build_only_called": False,
            "phase_transaction_called": False,
            "pair_local_lp_actual_calls": 0,
            "conditional_local_lp_actual_calls": 0,
            "total_local_lp_actual_calls": 0,
            "conditional_checker_actual_calls": 0,
            "local_lp_actual_call_cap": 20,
            "conditional_checker_actual_call_cap": 34,
        })
    try:
        if args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE:
            fixed_inputs = {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            if inputs != fixed_inputs:
                stop = _pcoh_k3_stop_loss_receipt(
                    stage="fixed_input_binding",
                    reason="fixed_iid2_input_sha256_mismatch",
                    started=started,
                    input_sha256=inputs,
                    implementation_sha256=implementation,
                    stage_resources={
                        "fixed_input_binding": _capture_resource_peaks()
                    },
                )
                body["pcoh_k3_build_only"] = stop
                body["phase_status"] = "stop_loss"
                body["failed_stage"] = "fixed_input_binding"
                body["fallback_reason"] = (
                    "fixed_iid2_input_sha256_mismatch"
                )
                body["error_type"] = None
                body["error"] = None
                raise _PCOHK3BuildOnlyStopLoss(
                    "fixed iid2 input binding rejected"
                )
            failed_stage = "pcoh_k3_fixed_baseline_artifact"
            baseline_anchor_receipt = (
                _pcoh_k3_fixed_baseline_artifact_anchor(
                    deadline=deadline
                )
            )
        if args.candidate_mode == _PCOH_K2_BUILD_ONLY_MODE:
            fixed_inputs = {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            if inputs != fixed_inputs:
                stop = _pcoh_k2_stop_loss_receipt(
                    stage="fixed_input_binding",
                    reason="fixed_iid2_input_sha256_mismatch",
                    started=started,
                    input_sha256=inputs,
                    implementation_sha256=implementation,
                    stage_resources={
                        "fixed_input_binding": _capture_resource_peaks()
                    },
                )
                body["pcoh_k2_build_only"] = stop
                body["phase_status"] = "stop_loss"
                body["failed_stage"] = "fixed_input_binding"
                body["fallback_reason"] = (
                    "fixed_iid2_input_sha256_mismatch"
                )
                body["error_type"] = None
                body["error"] = None
                raise _PCOHK2BuildOnlyStopLoss(
                    "fixed iid2 input binding rejected"
                )
        if args.candidate_mode == "rbs_adaptive_k4":
            fixed_inputs = {
                "onnx": _RBS_ADAPTIVE_K4_EXPECTED_ONNX_SHA256,
                "vnnlib": _RBS_ADAPTIVE_K4_EXPECTED_VNNLIB_SHA256,
                "instances_csv": _RBS_ADAPTIVE_K4_EXPECTED_CSV_SHA256,
            }
            if inputs != fixed_inputs:
                body["rbs_adaptive_k4"] = {
                    "status": "rbs_adaptive_k4_input_binding_stop_loss",
                    "reason": "fixed_iid2_input_sha256_mismatch",
                    "candidate_only": True,
                    "proof_authority": False,
                    "verdict_authority": False,
                    "phase_clique_attempted": False,
                    "expected_input_sha256": fixed_inputs,
                    "observed_input_sha256": dict(inputs),
                }
                body["phase_status"] = (
                    "rbs_adaptive_k4_input_binding_stop_loss"
                )
                body["failed_stage"] = "fixed_input_binding"
                body["fallback_reason"] = (
                    "fixed_iid2_input_sha256_mismatch"
                )
                body["error_type"] = None
                body["error"] = None
                raise _RBSAdaptiveK4StopLoss(
                    "fixed iid2 input binding rejected"
                )
        import torch
        torch_module = torch
        from act.back_end.analyze import analyze
        from act.back_end.core import ConSet, Fact
        from act.back_end.hybridz_tf.operator_hz import build_operator_hz
        from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
            consume_operator_phase_clique_pipeline_solver_handoff,
            maybe_run_operator_phase_clique_pipeline,
            validate_consumed_operator_phase_clique_solver_build,
        )
        from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
        from act.back_end.verifier import (
            _ensure_assert_linear_encoding,
            _get_output_layer_bounds,
            _get_output_layer_id,
            add_all_input_specs,
            find_entry_layer_id,
            gather_input_spec_layers,
            get_assert_layer,
            get_input_ids,
            seed_from_input_specs,
        )
        from act.front_end.model_synthesis import synthesize_models_from_specs
        from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
        from act.pipeline.verification.torch2act import TorchToACT
        from act.util.device_manager import initialize_device

        initialize_device("cuda", "float64")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
        torch.set_num_threads(int(args.cpu_threads))
        torch.set_num_interop_threads(1)
        set_solver_mode("hybridz")
        set_transfer_function_mode("interval")

        failed_stage = "parse_and_convert"
        stage = time.monotonic()
        spec = create_specs_from_paths(
            str(instance.onnx_path),
            str(instance.vnnlib_path),
            category=str(args.family),
        )
        wrapped = synthesize_models_from_specs([spec])
        if len(wrapped) != 1:
            raise PhaseCliqueBuildProbeError(
                f"expected one synthesized disjunct, observed {len(wrapped)}"
            )
        model = next(iter(wrapped.values())).to(device="cuda", dtype=torch.float64)
        net = TorchToACT(model).run()
        timings["parse_convert_synthesize_seconds"] = time.monotonic() - stage
        if time.monotonic() >= deadline:
            raise TimeoutError("deadline exhausted during conversion")

        failed_stage = "interval_analysis"
        stage = time.monotonic()
        entry_id = find_entry_layer_id(net)
        input_ids = get_input_ids(net)
        spec_layers = gather_input_spec_layers(net)
        seed = seed_from_input_specs(spec_layers)
        entry_fact = Fact(bounds=seed, cons=ConSet())
        add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
        before, after, _global = analyze(net, entry_id, entry_fact)
        timings["interval_analysis_seconds"] = time.monotonic() - stage

        failed_stage = "property_frame"
        output_bounds = _get_output_layer_bounds(net, after)
        assert_layer = get_assert_layer(net)
        B = int(seed.lb.shape[0])
        if B != 1:
            raise PhaseCliqueBuildProbeError(f"expected B=1, observed {B}")
        n_out = int(output_bounds.lb.shape[1])
        _ensure_assert_linear_encoding(
            assert_layer,
            B=1,
            n_out=n_out,
            device=output_bounds.lb.device,
            dtype=output_bounds.lb.dtype,
        )
        M = int(assert_layer.params["M"])
        C = np.ascontiguousarray(
            assert_layer.params["C"].detach().cpu().double().numpy().reshape(M, n_out)
        )
        thresholds = np.ascontiguousarray(
            assert_layer.params["thresholds"].detach().cpu().double().numpy().reshape(M)
        )
        output_lower = np.ascontiguousarray(output_bounds.lb.detach().cpu().double().numpy())
        output_upper = np.ascontiguousarray(output_bounds.ub.detach().cpu().double().numpy())

        failed_stage = "residual_selection"
        stage = time.monotonic()
        from act.back_end.hybridz_tf.property_residual_targets import (
            select_property_residual_targets,
        )
        residual_plan = select_property_residual_targets(
            net=net,
            before=before,
            after=after,
            C=C,
            thresholds=thresholds,
            kind=assert_layer.params.get("kind"),
            output_layer_id=_get_output_layer_id(net),
            budget=int(args.residual_budget),
            time_limit=float(args.residual_time_limit),
            deadline=deadline,
            max_adjoint_cells=30_000_000,
            pool_per_rival=8,
            allowed_relu_layer_ids=None,
            phase_joint_focus_after_first=True,
        )
        timings["residual_selection_seconds"] = time.monotonic() - stage

        schedule_receipt: Optional[dict[str, Any]] = None
        exact_target_reservoir: Optional[
            tuple[tuple[int, int], ...]
        ] = None
        builder_targets = residual_plan.builder_targets
        if args.candidate_mode == "rbs_adaptive_k4":
            failed_stage = "rbs_adaptive_schedule_split"
            live_property_sha256 = _binary_property_sha256(
                C,
                thresholds,
                kind=assert_layer.params.get("kind"),
            )
            (
                builder_targets,
                exact_target_reservoir,
                schedule_receipt,
            ) = _split_rbs_adaptive_schedule(
                residual_plan,
                primary_budget=_RBS_ADAPTIVE_K4_PRIMARY_BUDGET,
                expected_selector_budget=(
                    _RBS_ADAPTIVE_K4_SELECTOR_BUDGET
                ),
                expected_property_sha256=live_property_sha256,
                require_all_interval_survivors_processed=True,
            )
            body["rbs_adaptive_k4"] = {
                "status": "rbs_adaptive_k4_schedule_ready",
                "reason": None,
                "candidate_only": True,
                "proof_authority": False,
                "verdict_authority": False,
                "phase_clique_attempted": False,
                "schedule": schedule_receipt,
            }
            if schedule_receipt.get("status") != "ready":
                body["rbs_adaptive_k4"]["status"] = (
                    "rbs_adaptive_k4_selector_stop_loss"
                )
                body["rbs_adaptive_k4"]["reason"] = str(
                    schedule_receipt.get("status")
                )
                body["phase_status"] = (
                    "rbs_adaptive_k4_selector_stop_loss"
                )
                body["failed_stage"] = failed_stage
                body["fallback_reason"] = str(
                    schedule_receipt.get("status")
                )
                body["error_type"] = None
                body["error"] = None
                raise _RBSAdaptiveK4StopLoss(
                    "adaptive selector prefix did not fill its fixed schedule"
                )

        failed_stage = "operator_hz_build"
        stage = time.monotonic()
        build_kwargs: dict[str, Any] = {
            "exact_budget": int(args.operator_exact_budget),
            "materialize_add": True,
            "residual_targets": builder_targets,
            "issue_constructive_nonempty_seal": True,
            "deadline": deadline,
        }
        if args.candidate_mode == "rbs_adaptive_k4":
            build_kwargs.update({
                "residual_bound_screen": True,
                "residual_phase_screen": False,
                "exact_target_reservoir": exact_target_reservoir,
                "export_verified_preactivation_frame": False,
            })
        source_build = build_operator_hz(
            net, before, after, **build_kwargs
        )
        timings["operator_hz_build_seconds"] = time.monotonic() - stage
        body["operator_hz_build"] = {
            "elapsed_seconds": timings["operator_hz_build_seconds"],
            "shape": _hz_shape(source_build.hz),
            "input_columns": int(np.asarray(source_build.input_col_ids).size),
            "adaptive_schedule_receipt_sha256": (
                None
                if schedule_receipt is None
                else schedule_receipt.get("receipt_sha256")
            ),
        }

        # Freeze the small downstream bindings, then release conversion,
        # interval, and selector graphs before the large K4 LP is allowed to
        # allocate.  The resulting current RSS, not an asserted drop, governs
        # the adaptive pre-gate below.
        live_assert = {
            "kind": str(assert_layer.params.get("kind")),
            "C": _freeze_live_assert_value(
                assert_layer.params["C"]
            ),
            "thresholds": _freeze_live_assert_value(
                assert_layer.params["thresholds"]
            ),
            "M": M,
            "y_true": _freeze_live_assert_value(
                assert_layer.params["y_true"]
            ),
        }
        residual_selector_receipt = dict(residual_plan.receipt)
        residual_selector_property_sha256 = str(
            residual_plan.property_sha256
        )
        release_before = _capture_resource_peaks(torch)
        del build_kwargs
        del builder_targets
        del exact_target_reservoir
        del residual_plan
        del output_bounds
        del assert_layer
        del entry_fact
        del seed
        del spec_layers
        del input_ids
        del entry_id
        del _global
        del before
        del after
        del net
        del model
        del wrapped
        del spec
        collected_objects = int(gc.collect())
        empty_cache = getattr(torch.cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        allocator_trim = _glibc_malloc_trim_diagnostic()
        release_after = _capture_resource_peaks(torch)
        body["phase_input_release"] = {
            "schema": "act.hybridz_phase_clique_input_release.v1",
            "status": "released_before_phase_pipeline",
            "candidate_only": True,
            "proof_authority": False,
            "verdict_authority": False,
            "released_objects": (
                "torch_model",
                "act_graph",
                "interval_frames",
                "selector_plan",
                "synthesized_spec",
            ),
            "gc_collected_objects": collected_objects,
            "allocator_trim": allocator_trim,
            "resource_usage_before": dict(release_before),
            "resource_usage_after": dict(release_after),
        }

        adaptive_pre_gate_passed = True
        if args.candidate_mode == "rbs_adaptive_k4":
            failed_stage = "rbs_adaptive_k4_pre_gate"
            if schedule_receipt is None:
                raise PhaseCliqueBuildProbeError(
                    "adaptive schedule receipt disappeared before pre-gate"
                )
            property_cube = _rbs_adaptive_property_cube_receipt(
                source_build.hz, C, thresholds
            )
            pre_gate_resources = _capture_resource_peaks(torch)
            pre_gate = _rbs_adaptive_k4_pre_gate(
                source_build,
                schedule_receipt=schedule_receipt,
                property_cube_receipt=property_cube,
                build_seconds=float(
                    timings["operator_hz_build_seconds"]
                ),
                input_sha256=inputs,
                resources=pre_gate_resources,
                remaining_seconds=float(deadline - time.monotonic()),
            )
            experiment = body["rbs_adaptive_k4"]
            experiment.update({
                "property_cube": property_cube,
                "pre_gate": pre_gate,
                "resource_usage_pre_gate": pre_gate_resources,
            })
            adaptive_pre_gate_passed = pre_gate.get("status") == "passed"
            if not adaptive_pre_gate_passed:
                experiment["status"] = (
                    "rbs_adaptive_k4_build_stop_loss"
                )
                experiment["reason"] = "pre_gate_rejected:"
                experiment["reason"] += ",".join(
                    str(value)
                    for value in pre_gate.get("failed_conditions", ())
                )
                experiment["phase_clique_attempted"] = False
                body["phase_status"] = experiment["status"]
                body["failed_stage"] = failed_stage
                body["error_type"] = None
                body["fallback_reason"] = experiment["reason"]
                body["certified_edge_count"] = 0

        if (
            args.candidate_mode == "rbs_adaptive_k4"
            and not adaptive_pre_gate_passed
        ):
            pass
        elif args.candidate_mode == "k4":
            failed_stage = "phase_clique_pipeline_private_handoff"
            phase_deadline = min(
                deadline, time.monotonic() + float(args.phase_time_limit)
            )
            transaction, _private_build = _run_phase_transaction(
                source_build,
                pipeline_kwargs={
                    "enabled": True,
                    "vnnlib_path": str(instance.vnnlib_path),
                    "expected_vnnlib_sha256": inputs["vnnlib"],
                    "live_assert_params": live_assert,
                    "output_lower": output_lower,
                    "output_upper": output_upper,
                    "residual_selector_receipt": residual_selector_receipt,
                    "residual_selector_property_sha256": (
                        residual_selector_property_sha256
                    ),
                    "deadline": phase_deadline,
                    "caps": None,
                },
                objective_rows=C,
                thresholds=thresholds,
                deadline=phase_deadline,
                run_pipeline=maybe_run_operator_phase_clique_pipeline,
                consume_handoff=consume_operator_phase_clique_pipeline_solver_handoff,
                validate_consumed=validate_consumed_operator_phase_clique_solver_build,
            )
            body["phase_clique"] = transaction
            body["phase_status"] = transaction["status"]
            body["failed_stage"] = transaction["failed_stage"]
            body["error_type"] = transaction["error_type"]
            body["fallback_reason"] = transaction["fallback_reason"]
            body["certified_edge_count"] = transaction["certified_edge_count"]
        elif args.candidate_mode == "rbs_adaptive_k4":
            failed_stage = "rbs_adaptive_k4_phase_transaction"
            phase_deadline = min(
                deadline,
                time.monotonic() + _RBS_ADAPTIVE_K4_PHASE_SECONDS,
            )
            transaction, _private_build = _run_phase_transaction(
                source_build,
                pipeline_kwargs={
                    "enabled": True,
                    "vnnlib_path": str(instance.vnnlib_path),
                    "expected_vnnlib_sha256": inputs["vnnlib"],
                    "live_assert_params": live_assert,
                    "output_lower": output_lower,
                    "output_upper": output_upper,
                    "residual_selector_receipt": (
                        residual_selector_receipt
                    ),
                    "residual_selector_property_sha256": (
                        residual_selector_property_sha256
                    ),
                    "deadline": phase_deadline,
                    "caps": None,
                },
                objective_rows=C,
                thresholds=thresholds,
                deadline=phase_deadline,
                run_pipeline=maybe_run_operator_phase_clique_pipeline,
                consume_handoff=(
                    consume_operator_phase_clique_pipeline_solver_handoff
                ),
                validate_consumed=(
                    validate_consumed_operator_phase_clique_solver_build
                ),
                lp_per_call_seconds=5.0,
            )
            post_gate = _rbs_adaptive_k4_post_gate(transaction)
            experiment = body["rbs_adaptive_k4"]
            experiment.update({
                "phase_clique_attempted": True,
                "phase_clique": transaction,
                "post_gate": post_gate,
                "status": (
                    "rbs_adaptive_k4_post_gate_pending_terminal"
                    if post_gate.get("promoted") is True
                    else "rbs_adaptive_k4_post_gate_rejected"
                ),
                "reason": (
                    None
                    if post_gate.get("promoted") is True
                    else "post_gate_rejected:"
                    + ",".join(
                        str(value)
                        for value in post_gate.get(
                            "failed_conditions", ()
                        )
                    )
                ),
            })
            body["phase_status"] = experiment["status"]
            body["failed_stage"] = transaction.get("failed_stage")
            body["error_type"] = transaction.get("error_type")
            body["fallback_reason"] = experiment["reason"]
            body["certified_edge_count"] = transaction[
                "certified_edge_count"
            ]
        elif args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE:
            failed_stage = "pcoh_k3_source_build_preflight"
            if baseline_anchor_receipt is None:
                raise PhaseCliqueBuildProbeError(
                    "pcoh_k3_baseline_anchor_disappeared"
                )
            source_preflight = _pcoh_k3_source_build_preflight(
                body["operator_hz_build"]["shape"],
                build_seconds=float(
                    timings["operator_hz_build_seconds"]
                ),
                input_sha256=inputs,
                implementation_sha256=implementation,
                baseline_anchor_receipt=baseline_anchor_receipt,
            )
            body["pcoh_k3_source_build_preflight"] = source_preflight
            if (
                not _pcoh_k3_source_build_preflight_valid(
                    source_preflight
                )
                or source_preflight.get("status") != "passed"
            ):
                reason = "pcoh_k3_source_build_preflight_stop_loss:"
                reason += ",".join(
                    str(item)
                    for item in source_preflight.get(
                        "failed_conditions", ()
                    )
                )
                body["pcoh_k3_build_only"] = (
                    _pcoh_k3_stop_loss_receipt(
                        stage=failed_stage,
                        reason=reason,
                        started=started,
                        input_sha256=inputs,
                        implementation_sha256=implementation,
                        stage_resources={
                            "source_build_preflight": (
                                _capture_resource_peaks(torch)
                            )
                        },
                        baseline_anchor_receipt=(
                            baseline_anchor_receipt
                        ),
                    )
                )
                body["phase_status"] = "stop_loss"
                body["fallback_reason"] = reason[:300]
                body["error_type"] = None
                body["error"] = None
                raise _PCOHK3BuildOnlyStopLoss(reason)
            failed_stage = "pcoh_k3_build_only"
            transaction = _run_pcoh_k3_build_only_pipeline(
                source_build,
                input_sha256=inputs,
                implementation_sha256=implementation,
                vnnlib_path=str(instance.vnnlib_path),
                expected_vnnlib_sha256=inputs["vnnlib"],
                live_assert_params=live_assert,
                output_lower=output_lower,
                output_upper=output_upper,
                residual_selector_receipt=residual_selector_receipt,
                residual_selector_property_sha256=(
                    residual_selector_property_sha256
                ),
                deadline=deadline,
                phase_time_limit=_PCOH_K3_INTERNAL_PHASE_SECONDS,
                torch_module=torch,
                baseline_anchor_receipt=baseline_anchor_receipt,
            )
            k3_cleanup_ref[0] = transaction
            _adopt_pcoh_k3_trusted_transaction(body, transaction)
            body["phase_status"] = transaction["status"]
            body["failed_stage"] = transaction.get("failed_stage")
            body["error_type"] = None
            body["fallback_reason"] = transaction.get("reason")
            body["certified_edge_count"] = 0
            body["k3_transaction_called"] = transaction[
                "k3_transaction_called"
            ]
            body["k2_build_only_called"] = transaction[
                "k2_build_only_called"
            ]
            body["phase_transaction_called"] = transaction[
                "phase_transaction_called"
            ]
            for name in (
                "pair_local_lp_actual_calls",
                "conditional_local_lp_actual_calls",
                "total_local_lp_actual_calls",
                "conditional_checker_actual_calls",
            ):
                body[name] = transaction[name]
        elif args.candidate_mode == _PCOH_K2_BUILD_ONLY_MODE:
            failed_stage = "pcoh_k2_source_build_preflight"
            source_preflight = _pcoh_k2_source_build_preflight(
                body["operator_hz_build"]["shape"],
                build_seconds=float(
                    timings["operator_hz_build_seconds"]
                ),
                input_sha256=inputs,
                implementation_sha256=implementation,
            )
            body["pcoh_source_build_preflight"] = source_preflight
            if (
                not _pcoh_k2_source_build_preflight_valid(
                    source_preflight
                )
                or source_preflight.get("status") != "passed"
            ):
                reason = "pcoh_source_build_preflight_stop_loss:"
                reason += ",".join(
                    str(item)
                    for item in source_preflight.get(
                        "failed_conditions", ()
                    )
                )
                body["pcoh_k2_build_only"] = (
                    _pcoh_k2_stop_loss_receipt(
                        stage=failed_stage,
                        reason=reason,
                        started=started,
                        input_sha256=inputs,
                        implementation_sha256=implementation,
                        stage_resources={
                            "source_build_preflight": (
                                _capture_resource_peaks(torch)
                            )
                        },
                    )
                )
                body["phase_status"] = "stop_loss"
                body["fallback_reason"] = reason[:300]
                body["error_type"] = None
                body["error"] = None
                raise _PCOHK2BuildOnlyStopLoss(reason)
            failed_stage = "pcoh_k2_build_only"
            transaction = _run_pcoh_k2_build_only_pipeline(
                source_build,
                input_sha256=inputs,
                implementation_sha256=implementation,
                vnnlib_path=str(instance.vnnlib_path),
                expected_vnnlib_sha256=inputs["vnnlib"],
                live_assert_params=live_assert,
                output_lower=output_lower,
                output_upper=output_upper,
                residual_selector_receipt=residual_selector_receipt,
                residual_selector_property_sha256=(
                    residual_selector_property_sha256
                ),
                deadline=deadline,
                phase_time_limit=float(args.phase_time_limit),
                torch_module=torch,
            )
            _adopt_pcoh_k2_trusted_transaction(body, transaction)
            body["phase_status"] = transaction["status"]
            body["failed_stage"] = transaction.get("failed_stage")
            body["error_type"] = None
            body["fallback_reason"] = transaction.get("reason")
            body["certified_edge_count"] = 0
        else:
            failed_stage = "localized_e2_build_only"
            transaction = _run_localized_e2_pipeline(
                source_build,
                vnnlib_path=str(instance.vnnlib_path),
                expected_vnnlib_sha256=inputs["vnnlib"],
                live_assert_params=live_assert,
                output_lower=output_lower,
                output_upper=output_upper,
                residual_selector_receipt=residual_selector_receipt,
                residual_selector_property_sha256=(
                    residual_selector_property_sha256
                ),
                objective_rows=C,
                thresholds=thresholds,
                deadline=deadline,
                phase_time_limit=float(args.phase_time_limit),
                overall_started=started,
                torch_module=torch,
            )
            body["localized_e2"] = transaction
            body["phase_status"] = transaction["status"]
            body["failed_stage"] = transaction.get("failed_stage")
            body["error_type"] = None
            body["fallback_reason"] = transaction.get("reason")
            adapter_receipt = transaction.get("adapter")
            body["certified_edge_count"] = int(
                isinstance(adapter_receipt, dict)
                and adapter_receipt.get("edge_accepted") is True
            )
        body["error"] = None
        body["completed_before_deadline"] = bool(time.monotonic() < deadline)
    except (
        _RBSAdaptiveK4StopLoss,
        _PCOHK2BuildOnlyStopLoss,
        _PCOHK3BuildOnlyStopLoss,
    ):
        body["completed_before_deadline"] = bool(
            time.monotonic() < deadline
        )
    except Exception as exc:
        body["phase_status"] = (
            "stop_loss"
            if args.candidate_mode
            in {_PCOH_K2_BUILD_ONLY_MODE, _PCOH_K3_BUILD_ONLY_MODE}
            else "timeout"
            if isinstance(exc, TimeoutError)
            else "error"
        )
        body["failed_stage"] = failed_stage
        body["error_type"] = type(exc).__name__
        body["error"] = str(exc)[:1000]
        body["fallback_reason"] = (
            f"{type(exc).__name__}:{str(exc)[:240]}"
            if args.candidate_mode
            in {_PCOH_K2_BUILD_ONLY_MODE, _PCOH_K3_BUILD_ONLY_MODE}
            else None
        )
        if (
            args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE
            and "pcoh_k3_build_only" not in body
        ):
            body["pcoh_k3_build_only"] = _pcoh_k3_stop_loss_receipt(
                stage=str(failed_stage or "pcoh_k3_unhandled_exception"),
                reason=body["fallback_reason"],
                started=started,
                input_sha256=inputs,
                implementation_sha256=implementation,
                stage_resources={
                    "unhandled_exception": _capture_resource_peaks(
                        torch_module
                    )
                },
                baseline_anchor_receipt=baseline_anchor_receipt,
            )
        if (
            args.candidate_mode == _PCOH_K2_BUILD_ONLY_MODE
            and "pcoh_k2_build_only" not in body
        ):
            body["pcoh_k2_build_only"] = _pcoh_k2_stop_loss_receipt(
                stage=str(failed_stage or "pcoh_unhandled_exception"),
                reason=body["fallback_reason"],
                started=started,
                input_sha256=inputs,
                implementation_sha256=implementation,
                stage_resources={
                    "unhandled_exception": _capture_resource_peaks(
                        torch_module
                    )
                },
            )
        body["completed_before_deadline"] = bool(time.monotonic() < deadline)
    except BaseException as exc:
        if args.candidate_mode != _PCOH_K3_BUILD_ONLY_MODE:
            raise
        detail = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_pcoh_k3_exception_traceback(exc)
        body["phase_status"] = "stop_loss"
        body["failed_stage"] = failed_stage
        body["error_type"] = None
        body["error"] = None
        body["fallback_reason"] = detail
        if "pcoh_k3_build_only" not in body:
            body["pcoh_k3_build_only"] = _pcoh_k3_stop_loss_receipt(
                stage=str(failed_stage or "pcoh_k3_baseexception"),
                reason=detail,
                started=started,
                input_sha256=inputs,
                implementation_sha256=implementation,
                stage_resources={
                    "baseexception": _capture_resource_peaks(
                        torch_module
                    )
                },
                baseline_anchor_receipt=baseline_anchor_receipt,
            )
        body["completed_before_deadline"] = bool(time.monotonic() < deadline)
    k3_mode = args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE
    try:
        try:
            body["input_sha256_after"] = {
                "onnx": _sha256_file(instance.onnx_path),
                "vnnlib": _sha256_file(instance.vnnlib_path),
                "instances_csv": _sha256_file(instance.csv_path),
            }
            body["inputs_unchanged"] = body["input_sha256_after"] == inputs
            body["input_integrity_error_type"] = None
            body["input_integrity_error"] = None
        except Exception as exc:
            body["input_sha256_after"] = None
            body["inputs_unchanged"] = False
            body["input_integrity_error_type"] = type(exc).__name__
            body["input_integrity_error"] = str(exc)[:300]
        try:
            body["implementation_sha256_after"] = (
                _pcoh_k3_implementation_sha256()
                if k3_mode
                else _implementation_sha256()
            )
            body["implementation_unchanged"] = (
                body["implementation_sha256_after"] == implementation
            )
            body["implementation_integrity_error_type"] = None
            body["implementation_integrity_error"] = None
        except Exception as exc:
            body["implementation_sha256_after"] = None
            body["implementation_unchanged"] = False
            body["implementation_integrity_error_type"] = type(exc).__name__
            body["implementation_integrity_error"] = str(exc)[:300]
        body["resource_usage"] = _capture_resource_peaks(torch_module)
        finished = time.monotonic()
        timings["total_seconds"] = float(finished - started)
        body["shared_worker_deadline_met"] = bool(finished < deadline)
        _finalize_localized_e2_integrity(body)
        _finalize_rbs_adaptive_k4_integrity(body)
        _finalize_pcoh_k2_integrity(body)
    except BaseException as exc:
        if not k3_mode:
            raise
        detail = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_pcoh_k3_exception_traceback(exc)
        body["phase_status"] = "stop_loss"
        body["failed_stage"] = "pcoh_k3_terminal_processing"
        body["fallback_reason"] = detail
        body["error_type"] = None
        body["error"] = None
        body["shared_worker_deadline_met"] = False

    if k3_mode:
        try:
            _finalize_pcoh_k3_integrity(body)
        except BaseException as exc:
            detail = f"{type(exc).__name__}:{str(exc)[:240]}"
            _clear_pcoh_k3_exception_traceback(exc)
            _pcoh_k3_install_terminal_failure(
                body,
                transaction=body.get("pcoh_k3_build_only"),
                detail=detail,
                failed_stage="pcoh_k3_terminal_finalizer",
            )
            _release_pcoh_k3_trusted_transaction(
                body.get("pcoh_k3_build_only")
            )
    try:
        return _checksummed(body)
    except BaseException as exc:
        if not k3_mode:
            raise
        detail = f"{type(exc).__name__}:{str(exc)[:240]}"
        _clear_pcoh_k3_exception_traceback(exc)
        transaction = body.get("pcoh_k3_build_only")
        _pcoh_k3_install_terminal_failure(
            body,
            transaction=transaction,
            detail=detail,
            failed_stage="pcoh_k3_receipt_seal",
        )
        _release_pcoh_k3_trusted_transaction(transaction)
        return _pcoh_k3_emergency_checksummed(body)


def _execute_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run the probe under an outermost K3 transaction cleanup guard."""

    k3_mode = args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE
    k3_cleanup_ref: list[Any] = [None]
    try:
        return _execute_probe_inner(
            args, k3_cleanup_ref=k3_cleanup_ref
        )
    finally:
        if k3_mode:
            _release_pcoh_k3_trusted_transaction(k3_cleanup_ref[0])
        k3_cleanup_ref[0] = None


def _validate_args(args: argparse.Namespace) -> None:
    if args.iid != _ONLY_IID:
        raise PhaseCliqueBuildProbeError("only --iid 2 is permitted")
    if type(args.candidate_mode) is not str or args.candidate_mode not in _CANDIDATE_MODES:
        raise PhaseCliqueBuildProbeError("candidate mode is not supported")
    for name in ("wall_timeout", "phase_time_limit", "residual_time_limit"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise PhaseCliqueBuildProbeError(f"{name} must be finite numeric")
    if not 1.0 <= float(args.wall_timeout) <= _MAX_WALL_SECONDS:
        raise PhaseCliqueBuildProbeError("wall timeout must lie in [1, 60]")
    if not 0.1 <= float(args.phase_time_limit) <= 40.0:
        raise PhaseCliqueBuildProbeError("phase time limit must lie in [0.1, 40]")
    if not 0.1 <= float(args.residual_time_limit) <= 10.0:
        raise PhaseCliqueBuildProbeError("residual time limit must lie in [0.1, 10]")
    if type(args.operator_exact_budget) is not int or not 1 <= args.operator_exact_budget <= 16:
        raise PhaseCliqueBuildProbeError("operator exact budget must lie in [1, 16]")
    if type(args.residual_budget) is not int or not 1 <= args.residual_budget <= 16:
        raise PhaseCliqueBuildProbeError("residual budget must lie in [1, 16]")
    if type(args.cpu_threads) is not int or not 1 <= args.cpu_threads <= 20:
        raise PhaseCliqueBuildProbeError("cpu threads must lie in [1, 20]")
    if args.family not in _FAMILIES:
        raise PhaseCliqueBuildProbeError("family is not supported")
    if not args.benchmark_root.expanduser().is_dir():
        raise PhaseCliqueBuildProbeError("benchmark root does not exist")
    if args.candidate_mode == "rbs_adaptive_k4":
        fixed_contract = {
            "family": args.family == _RBS_ADAPTIVE_K4_FAMILY,
            "wall_timeout": (
                float(args.wall_timeout) == _MAX_WALL_SECONDS
            ),
            "phase_time_limit": (
                float(args.phase_time_limit)
                == _RBS_ADAPTIVE_K4_PHASE_SECONDS
            ),
            "operator_exact_budget": (
                args.operator_exact_budget
                == _RBS_ADAPTIVE_K4_PRIMARY_BUDGET
            ),
            "residual_budget": (
                args.residual_budget
                == _RBS_ADAPTIVE_K4_SELECTOR_BUDGET
            ),
            "residual_time_limit": (
                float(args.residual_time_limit)
                == _RBS_ADAPTIVE_K4_RESIDUAL_SECONDS
            ),
            "cpu_threads": (
                args.cpu_threads == _RBS_ADAPTIVE_K4_CPU_THREADS
            ),
        }
        rejected = [
            name for name, passed in fixed_contract.items() if not passed
        ]
        if rejected:
            raise PhaseCliqueBuildProbeError(
                "rbs_adaptive_k4 fixed contract mismatch: "
                + ",".join(rejected)
            )
    if args.candidate_mode == _PCOH_K2_BUILD_ONLY_MODE:
        fixed_contract = {
            "family": args.family == _PCOH_K2_FAMILY,
            "wall_timeout": float(args.wall_timeout) == _MAX_WALL_SECONDS,
            "phase_time_limit": (
                float(args.phase_time_limit)
                == _PCOH_K2_MAX_PHASE_SECONDS
            ),
            "operator_exact_budget": args.operator_exact_budget == 4,
            "residual_budget": args.residual_budget == 4,
            "residual_time_limit": (
                float(args.residual_time_limit)
                == _PCOH_K2_RESIDUAL_SECONDS
            ),
            "cpu_threads": args.cpu_threads == 20,
        }
        rejected = [
            name for name, passed in fixed_contract.items() if not passed
        ]
        if rejected:
            raise PhaseCliqueBuildProbeError(
                "pcoh_k2_build_only fixed contract mismatch: "
                + ",".join(rejected)
            )
    if args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE:
        fixed_contract = {
            "family": args.family == _PCOH_K3_FAMILY,
            "wall_timeout": float(args.wall_timeout) == _MAX_WALL_SECONDS,
            "phase_time_limit": (
                float(args.phase_time_limit)
                == _PCOH_K3_CLI_PHASE_SECONDS
            ),
            "operator_exact_budget": args.operator_exact_budget == 4,
            "residual_budget": args.residual_budget == 4,
            "residual_time_limit": (
                float(args.residual_time_limit)
                == _PCOH_K3_RESIDUAL_SECONDS
            ),
            "cpu_threads": args.cpu_threads == 20,
        }
        rejected = [
            name for name, passed in fixed_contract.items() if not passed
        ]
        if rejected:
            raise PhaseCliqueBuildProbeError(
                "pcoh_k3_build_only fixed contract mismatch: "
                + ",".join(rejected)
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root", type=Path, default=_DEFAULT_BENCHMARK_ROOT
    )
    parser.add_argument("--family", choices=tuple(_FAMILIES), default="cifar100_medium")
    parser.add_argument("--iid", type=int, default=_ONLY_IID)
    parser.add_argument(
        "--candidate-mode", choices=_CANDIDATE_MODES, default="k4"
    )
    parser.add_argument("--wall-timeout", type=float, default=60.0)
    parser.add_argument("--phase-time-limit", type=float, default=20.0)
    parser.add_argument("--operator-exact-budget", type=int, default=4)
    parser.add_argument("--residual-budget", type=int, default=4)
    parser.add_argument("--residual-time-limit", type=float, default=4.0)
    parser.add_argument("--cpu-threads", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _worker_payload(
    args: argparse.Namespace,
    *,
    run_nonce: str,
    parent_hard_deadline_monotonic: float,
    fixed_environment: Mapping[str, str],
    fixed_environment_sha256: str,
) -> dict[str, Any]:
    payload = {
        "benchmark_root": str(args.benchmark_root.expanduser().resolve(strict=True)),
        "family": args.family,
        "iid": args.iid,
        "candidate_mode": args.candidate_mode,
        "wall_timeout": float(args.wall_timeout),
        "phase_time_limit": float(args.phase_time_limit),
        "operator_exact_budget": args.operator_exact_budget,
        "residual_budget": args.residual_budget,
        "residual_time_limit": float(args.residual_time_limit),
        "cpu_threads": args.cpu_threads,
        "run_nonce": run_nonce,
        "parent_hard_deadline_monotonic": float(
            parent_hard_deadline_monotonic
        ),
        "fixed_environment": dict(fixed_environment),
        "fixed_environment_sha256": fixed_environment_sha256,
    }
    payload["worker_args_sha256"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    return payload


def _namespace_from_worker_payload(payload: Mapping[str, Any]) -> argparse.Namespace:
    expected = {
        "benchmark_root",
        "family",
        "iid",
        "candidate_mode",
        "wall_timeout",
        "phase_time_limit",
        "operator_exact_budget",
        "residual_budget",
        "residual_time_limit",
        "cpu_threads",
        "run_nonce",
        "parent_hard_deadline_monotonic",
        "fixed_environment",
        "fixed_environment_sha256",
        "worker_args_sha256",
    }
    if set(payload) != expected:
        raise PhaseCliqueBuildProbeError("worker payload fields mismatch")
    if (
        type(payload["run_nonce"]) is not str
        or len(payload["run_nonce"]) != 64
        or type(payload["parent_hard_deadline_monotonic"]) is not float
        or not math.isfinite(payload["parent_hard_deadline_monotonic"])
        or not isinstance(payload["fixed_environment"], dict)
        or type(payload["fixed_environment_sha256"]) is not str
        or not _valid_sha256(payload["worker_args_sha256"])
    ):
        raise PhaseCliqueBuildProbeError("worker private fields are malformed")
    checksum_body = dict(payload)
    observed_worker_checksum = checksum_body.pop("worker_args_sha256")
    if hashlib.sha256(_canonical_json(checksum_body)).hexdigest() != observed_worker_checksum:
        raise PhaseCliqueBuildProbeError("worker args checksum mismatch")
    namespace = argparse.Namespace(
        benchmark_root=Path(payload["benchmark_root"]),
        family=payload["family"],
        iid=payload["iid"],
        candidate_mode=payload["candidate_mode"],
        wall_timeout=payload["wall_timeout"],
        phase_time_limit=payload["phase_time_limit"],
        operator_exact_budget=payload["operator_exact_budget"],
        residual_budget=payload["residual_budget"],
        residual_time_limit=payload["residual_time_limit"],
        cpu_threads=payload["cpu_threads"],
        run_nonce=payload["run_nonce"],
        parent_hard_deadline_monotonic=payload[
            "parent_hard_deadline_monotonic"
        ],
        fixed_environment=dict(payload["fixed_environment"]),
        fixed_environment_sha256=payload["fixed_environment_sha256"],
    )
    _validate_args(namespace)
    _unused_child, canonical_fixed, canonical_digest = (
        _probe_worker_environment({})
    )
    if (
        any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in namespace.fixed_environment.items()
        )
        or namespace.fixed_environment != canonical_fixed
        or namespace.fixed_environment_sha256 != canonical_digest
        or any(
            os.environ.get(key) != value
            for key, value in namespace.fixed_environment.items()
        )
        or any(
            key.startswith("HZ_") and key not in namespace.fixed_environment
            for key in os.environ
        )
    ):
        raise PhaseCliqueBuildProbeError("worker numerical environment mismatch")
    _shared_worker_deadline(namespace)
    return namespace


def _multiprocessing_worker(
    payload: Mapping[str, Any],
    ready_connection: Any,
    output_handle: Any,
    output_identity: tuple[int, int],
) -> None:
    """Direct ``multiprocessing.Process`` target; never CLI-dispatched."""

    exit_code = 97
    try:
        parent = multiprocessing.parent_process()
        if parent is None or multiprocessing.current_process().name == "MainProcess":
            raise PhaseCliqueBuildProbeError("worker lacks multiprocessing parent")
        if payload.get("parent_pid") != os.getppid():
            raise PhaseCliqueBuildProbeError("worker parent PID mismatch")
        os.setsid()
        ready_connection.send({"session_ready": True, "pid": os.getpid()})
        ready_connection.close()
        args_payload = payload.get("args")
        if not isinstance(args_payload, dict):
            raise PhaseCliqueBuildProbeError("worker payload is malformed")
        args = _namespace_from_worker_payload(args_payload)
        receipt = _execute_probe(args)
        output_fd = output_handle.detach()
        _write_private_worker_json_fd(
            output_fd,
            receipt,
            expected_identity=output_identity,
        )
        os.close(output_fd)
        exit_code = 0 if receipt["phase_status"] not in {"error", "timeout"} else 2
    except Exception as exc:
        print(f"private worker authorization failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        try:
            ready_connection.close()
        except Exception:
            pass
    raise SystemExit(exit_code)


def _bounded_wait(
    child: Any,
    *,
    started: float,
    wall_timeout: float,
    clock: Callable[[], float] = time.monotonic,
    signal_group: Callable[[int, int], None] = os.killpg,
    session_ready: bool = True,
) -> dict[str, Any]:
    """Issue TERM before, and SIGKILL no later than, the hard deadline."""

    hard_deadline = started + float(wall_timeout)
    kill_deadline = hard_deadline - min(0.05, float(wall_timeout) / 20.0)
    term_reserve = _parent_term_reserve_seconds(float(wall_timeout))
    term_deadline = hard_deadline - term_reserve
    child.join(timeout=max(0.0, term_deadline - clock()))
    if child.exitcode is not None:
        return {
            "returncode": int(child.exitcode),
            "timed_out": False,
            "term_issued_seconds": None,
            "kill_issued_seconds": None,
            "reaped_before_deadline": True,
            "detached_unreaped_child": False,
            "detach_reason": None,
        }
    term_elapsed = float(max(0.0, clock() - started))
    try:
        if session_ready:
            signal_group(child.pid, signal.SIGTERM)
        else:
            child.terminate()
    except ProcessLookupError:
        pass
    child.join(timeout=max(0.0, kill_deadline - clock()))
    if child.exitcode is not None:
        return {
            "returncode": int(child.exitcode),
            "timed_out": True,
            "term_issued_seconds": term_elapsed,
            "kill_issued_seconds": None,
            "reaped_before_deadline": True,
            "detached_unreaped_child": False,
            "detach_reason": None,
        }
    kill_elapsed = float(max(0.0, clock() - started))
    try:
        if session_ready:
            signal_group(child.pid, signal.SIGKILL)
        else:
            child.kill()
    except ProcessLookupError:
        pass
    child.join(timeout=max(0.0, hard_deadline - clock()))
    detached = False
    detach_reason = None
    if child.exitcode is None:
        detached, detach_reason = _detach_unreaped_child(child)
    return {
        "returncode": (
            int(child.exitcode) if child.exitcode is not None else None
        ),
        "timed_out": True,
        "term_issued_seconds": term_elapsed,
        "kill_issued_seconds": kill_elapsed,
        "reaped_before_deadline": bool(child.exitcode is not None),
        "detached_unreaped_child": detached,
        "detach_reason": detach_reason,
    }


def _detach_unreaped_child(child: Any) -> tuple[bool, str]:
    """Prevent CPython's exit handler from unboundedly joining a killed child."""

    if sys.implementation.name != "cpython":
        return False, "unsupported_python_implementation"
    if not (3, 9) <= sys.version_info[:2] <= (3, 14):
        return False, "unsupported_cpython_version"
    try:
        import multiprocessing.process as process_module

        children = getattr(process_module, "_children")
        if not isinstance(children, set):
            return False, "active_child_registry_shape_mismatch"
        if getattr(child, "_parent_pid", None) != os.getpid():
            return False, "child_parent_identity_mismatch"
        if child not in children:
            return False, "child_not_in_active_registry"
        children.discard(child)
        if child in children:
            return False, "active_child_registry_discard_failed"
        return True, "cpython_active_child_registry_discarded_after_sigkill"
    except Exception as exc:
        return False, f"detach_error:{type(exc).__name__}"


def _stop_process_by_deadline(
    child: Any,
    *,
    started: float,
    wall_timeout: float,
    session_ready: bool,
) -> dict[str, Any]:
    """Immediately stop an exceptional child, with no post-deadline join."""

    deadline = started + float(wall_timeout)
    kill_deadline = deadline - min(0.05, float(wall_timeout) / 20.0)
    term_elapsed = float(max(0.0, time.monotonic() - started))
    try:
        if child.exitcode is None:
            if session_ready:
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
    except (ProcessLookupError, ValueError, AssertionError):
        pass
    child.join(timeout=max(0.0, kill_deadline - time.monotonic()))
    kill_elapsed = None
    if child.exitcode is None:
        kill_elapsed = float(max(0.0, time.monotonic() - started))
        try:
            if session_ready:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except (ProcessLookupError, ValueError, AssertionError):
            pass
        child.join(timeout=max(0.0, deadline - time.monotonic()))
    detached = False
    detach_reason = None
    if child.exitcode is None:
        detached, detach_reason = _detach_unreaped_child(child)
    return {
        "returncode": (
            int(child.exitcode) if child.exitcode is not None else None
        ),
        "timed_out": True,
        "term_issued_seconds": term_elapsed,
        "kill_issued_seconds": kill_elapsed,
        "reaped_before_deadline": bool(child.exitcode is not None),
        "detached_unreaped_child": detached,
        "detach_reason": detach_reason,
    }


def _parent_error_receipt(
    args: argparse.Namespace,
    *,
    run_nonce: str,
    failed_stage: str,
    error_type: str,
    error: str,
    elapsed_seconds: float,
    process: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    pcoh_k3_build_only = args.candidate_mode == _PCOH_K3_BUILD_ONLY_MODE
    pcoh_build_only = bool(
        args.candidate_mode == _PCOH_K2_BUILD_ONLY_MODE
        or pcoh_k3_build_only
    )
    body = {
        "schema": _SCHEMA,
        "candidate_mode": args.candidate_mode,
        "diagnostic_only": True,
        "candidate_only": True,
        "build_only": pcoh_build_only,
        "instance_count": 1,
        "proof_authority": False,
        "verdict_authority": False,
        "hz_objbound_decide_called": False,
        "hz_base_feasibility_called": False,
        "solver_handoff_called": False,
        "diagnostic_lp_called": False,
        "strict_replay_called": False,
        "ground_truth_loaded": False,
        "reference_label_used": False,
        "family": args.family,
        "iid": args.iid,
        "run_nonce": run_nonce,
        "wall_timeout_seconds": float(args.wall_timeout),
        "phase_time_limit_seconds": float(args.phase_time_limit),
        "operator_exact_budget": int(args.operator_exact_budget),
        "residual_budget": int(args.residual_budget),
        "residual_time_limit_seconds": float(args.residual_time_limit),
        "cpu_threads": int(args.cpu_threads),
        "phase_status": (
            "stop_loss"
            if pcoh_build_only
            else "timeout"
            if failed_stage == "outer_hard_stop"
            else "error"
        ),
        "fallback_reason": error[:300] if pcoh_build_only else None,
        "failed_stage": failed_stage,
        "error_type": error_type,
        "error": error[:1000],
        "certified_edge_count": 0,
        "timings": {"total_seconds": float(max(0.0, elapsed_seconds))},
        "process_control": dict(process or {}),
    }
    if pcoh_k3_build_only:
        body.pop("diagnostic_lp_called", None)
        body.pop("strict_replay_called", None)
        body.update({
            "k3_transaction_called": False,
            "k2_build_only_called": False,
            "phase_transaction_called": False,
            "full_parent_lp_called": False,
            "full_parent_lp_solver_called": False,
            "pair_local_lp_actual_calls": 0,
            "conditional_local_lp_actual_calls": 0,
            "total_local_lp_actual_calls": 0,
            "conditional_checker_actual_calls": 0,
            "local_lp_actual_call_cap": 20,
            "conditional_checker_actual_call_cap": 34,
        })
    return _checksummed(body)


def _child_receipt_exit_consistent(
    receipt: Mapping[str, Any], process: Mapping[str, Any]
) -> bool:
    status = receipt.get("phase_status")
    returncode = process.get("returncode")
    if process.get("timed_out") is not False or type(status) is not str:
        return False
    if returncode == 0:
        return status not in {"error", "timeout"}
    if returncode == 2:
        return status in {"error", "timeout"}
    return False


def _new_worker_inode(parent_fd: int) -> tuple[int, tuple[int, int]]:
    if not hasattr(os, "O_TMPFILE"):
        raise PhaseCliqueBuildProbeError("anonymous O_TMPFILE is unavailable")
    fd = os.open(
        ".",
        os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC,
        0o600,
        dir_fd=parent_fd,
    )
    info = os.fstat(fd)
    identity = (int(info.st_dev), int(info.st_ino))
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 0
    ):
        os.close(fd)
        raise PhaseCliqueBuildProbeError("anonymous worker inode is malformed")
    return fd, identity


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _build_parser().parse_args(raw_argv)
        _validate_args(args)
        instance = _select_instance(args.benchmark_root, args.family, args.iid)
        output_slot = _validate_new_output_path(
            args.output,
            protected_paths=(
                instance.onnx_path,
                instance.vnnlib_path,
                instance.csv_path,
                Path(__file__),
                _REPO_ROOT / "configs" / "hybridz_largecls_gates.yaml",
            ),
        )
        run_nonce = secrets.token_hex(32)
        _validate_output_slot_live(output_slot)
        worker_fd, worker_identity = _new_worker_inode(output_slot.parent_fd)
        child_environment, fixed_environment, fixed_environment_sha256 = (
            _probe_worker_environment(os.environ)
        )
        launched = time.monotonic()
        parent_hard_deadline_monotonic = (
            launched + float(args.wall_timeout)
        )
        payload = {
            "parent_pid": os.getpid(),
            "args": _worker_payload(
                args,
                run_nonce=run_nonce,
                parent_hard_deadline_monotonic=(
                    parent_hard_deadline_monotonic
                ),
                fixed_environment=fixed_environment,
                fixed_environment_sha256=fixed_environment_sha256,
            ),
        }
        context = multiprocessing.get_context("spawn")
        ready_parent, ready_child = context.Pipe(duplex=False)
        child = context.Process(
            target=_multiprocessing_worker,
            args=(
                payload,
                ready_child,
                DupFd(worker_fd),
                worker_identity,
            ),
            name="hybridz-phase-clique-build-worker",
            daemon=False,
        )
        previous_environment = dict(os.environ)
        child_started = False
        session_ready = False
        try:
            os.environ.clear()
            os.environ.update(child_environment)
            child.start()
            child_started = True
        except BaseException:
            try:
                ready_parent.close()
                ready_child.close()
                os.close(worker_fd)
                _close_output_slot(output_slot)
            except Exception:
                pass
            if child.pid is not None:
                _stop_process_by_deadline(
                    child,
                    started=launched,
                    wall_timeout=float(args.wall_timeout),
                    session_ready=False,
                )
            raise
        finally:
            os.environ.clear()
            os.environ.update(previous_environment)
        try:
            ready_child.close()
            ready_reserve = min(
                1.0, float(args.wall_timeout) / 2.0
            )
            ready_budget = max(
                0.0,
                min(
                    5.0,
                    launched
                    + float(args.wall_timeout)
                    - ready_reserve
                    - time.monotonic(),
                ),
            )
            if ready_parent.poll(ready_budget):
                ready = ready_parent.recv()
                session_ready = bool(
                    isinstance(ready, dict)
                    and ready.get("session_ready") is True
                    and ready.get("pid") == child.pid
                )
            ready_parent.close()
            process = _bounded_wait(
                child,
                started=launched,
                wall_timeout=float(args.wall_timeout),
                session_ready=session_ready,
            )
        except BaseException:
            try:
                ready_parent.close()
                ready_child.close()
            except Exception:
                pass
            if child_started:
                _stop_process_by_deadline(
                    child,
                    started=launched,
                    wall_timeout=float(args.wall_timeout),
                    session_ready=session_ready,
                )
            try:
                os.close(worker_fd)
            except OSError:
                pass
            _close_output_slot(output_slot)
            raise
        process["fixed_environment_sha256"] = fixed_environment_sha256
        process["HZ_QUERY_WORKERS"] = fixed_environment["HZ_QUERY_WORKERS"]
        process["HZ_MILP_THREADS"] = fixed_environment["HZ_MILP_THREADS"]
        process["HZ_LP_PREFILTER_THREADS"] = fixed_environment[
            "HZ_LP_PREFILTER_THREADS"
        ]
        try:
            if process["timed_out"]:
                raise TimeoutError("child exceeded the parent stop-loss deadline")
            child_receipt = _validate_worker_receipt_fd(
                worker_fd,
                run_nonce=run_nonce,
                expected_identity=worker_identity,
            )
            if not _child_receipt_exit_consistent(child_receipt, process):
                raise ChildProcessError(
                    "private worker receipt status conflicts with exit code "
                    f"{process['returncode']}"
                )
            _publish_new_json_fd(
                worker_fd,
                output_slot,
                expected_identity=worker_identity,
            )
            os.close(worker_fd)
            _close_output_slot(output_slot)
            return int(process["returncode"])
        except Exception as child_exc:
            try:
                os.close(worker_fd)
                worker_fd, worker_identity = _new_worker_inode(
                    output_slot.parent_fd
                )
                failed_stage = (
                    "outer_hard_stop"
                    if process["timed_out"]
                    else "parent_worker_validation"
                )
                parent_receipt = _parent_error_receipt(
                    args,
                    run_nonce=run_nonce,
                    failed_stage=failed_stage,
                    error_type=type(child_exc).__name__,
                    error=str(child_exc),
                    elapsed_seconds=time.monotonic() - launched,
                    process=process,
                )
                _write_private_worker_json_fd(
                    worker_fd,
                    parent_receipt,
                    expected_identity=worker_identity,
                )
                _validate_worker_receipt_fd(
                    worker_fd,
                    run_nonce=run_nonce,
                    expected_identity=worker_identity,
                )
                _publish_new_json_fd(
                    worker_fd,
                    output_slot,
                    expected_identity=worker_identity,
                )
            finally:
                try:
                    os.close(worker_fd)
                except OSError:
                    pass
                _close_output_slot(output_slot)
            return 2
    except (PhaseCliqueBuildProbeError, ValueError, OSError) as exc:
        print(f"probe configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
