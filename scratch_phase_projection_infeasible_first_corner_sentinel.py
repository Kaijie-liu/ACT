#!/usr/bin/env python3
"""Disconnected, one-shot status-corner ability sentinel.

The canonical production/configuration files are read-only inputs.  A private
copy of the HiGHS owner changes only its base-INFEASIBLE return to a typed,
status-only value and never asks HiGHS for a dual ray.  A private copy of the
current-v3 phase candidate keeps the complete OPTIMAL dispatch and uses the
already-computed first analytic inward BOX corner only for that typed
INFEASIBLE state.  Other owner states fail closed.

Candidate arithmetic has no authority.  FALSIFIED is accepted only from the
unchanged raw-BOX check, the same StoredBinary64Input passed to the
candidate-blind zero-width outward terminal, and a positive exact Fraction
property lower bound.  Static checks are CPU-only.  CUDA needs both an
explicit command flag and the separately frozen root authorization value.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import types
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
ARTIFACT_ROOT = ROOT / "artifacts/hybridz_largecls_gates"
PREREG_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_infeasible_first_corner_preregistration_20260820.json"
)
EVENTS_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_infeasible_first_corner_sentinel_20260820.events.jsonl"
)
RAW_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_infeasible_first_corner_sentinel_20260820.raw.jsonl"
)
RECEIPTS_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_infeasible_first_corner_sentinel_20260820.receipts.jsonl"
)
SUMMARY_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_infeasible_first_corner_sentinel_20260820.json"
)

RESULT_PREFIX = "@@ACT_INFEASIBLE_FIRST_CORNER_RESULT@@"
WORKER_TOKEN_ENV = "ACT_INFEASIBLE_FIRST_CORNER_ATTEMPT_TOKEN"
GPU_AUTH_ENV = "ACT_INFEASIBLE_FIRST_CORNER_GPU_AUTHORIZATION"
GPU_AUTH_VALUE = "ROOT_AUTHORIZED_INFEASIBLE_FIRST_CORNER_20260820"
REQUEST_SECONDS = 10.0
WORKER_TIMEOUT_SECONDS = 60.0
WORKER_STAGES = (
    "worker_entry",
    "authorized",
    "imports",
    "inputs_locked",
    "private_modules_loaded",
    "device_initialized",
    "model_synthesized",
    "network_converted",
    "verify_and_owner_cleanup",
    "cuda_synchronized",
    "postvalidate",
    "complete",
)

PREREG_SHA256 = "7ebb03b32b0eb2f5123f293a24c55887430b39a80317ff965f25b07ced31c682"
CANONICAL_PHASE_NAME = (
    "act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate"
)
PRIVATE_PHASE_NAME = (
    "act.back_end.hybridz_tf._scratch_infeasible_first_corner_candidate"
)
PRIVATE_OWNER_NAME = (
    "act.back_end.hybridz_tf._scratch_infeasible_status_only_owner"
)

MAX_PHASE_ROWS = 200_000
MAX_INPUT_COLUMNS = 200_000
MAX_SELECTED = 200_000
MAX_DENSE_ELEMENTS = 200_000_000
MAX_LOGICAL_NNZ = 200_000_000
MAX_TRANSACTION_BYTES = 2_000_000_000
INT32_MAX = int(np.iinfo(np.int32).max)


SOURCE_LOCKS: Mapping[str, str] = {
    "act/back_end/config.py": (
        "8d94960cf4686f8d9c894cc18c1483b57092faa52e794905dce9f622c8fddf21"
    ),
    "act/back_end/verifier.py": (
        "eb3dfc8611ee97262bf71d66b8deee58a6a2544d0934d94ca7b075424dbc3afd"
    ),
    "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py": (
        "13625f452c36a1b7844e4385b884471c8a0c82abf015bf2af417257e2c96c23a"
    ),
    "act/back_end/hybridz_tf/forward_exact_relu_live_row_stream_candidate.py": (
        "d53c2335c43905097e78bef8311175d7151d7e98293a6152fce62dba00d37511"
    ),
    "act/back_end/hybridz_tf/operator_hz.py": (
        "2502c009d6b5d37983e3f6d072e802c7734ac9bd106bb2a252e0a562da390fd8"
    ),
    "act/back_end/hybridz_tf/phase_projection_device_program.py": (
        "7f0cce0e461f63ff6599ddd82ad5e61ef7c921eb489ef7bbbf4d60cda9048962"
    ),
    "act/back_end/hybridz_tf/phase_projection_highs_owner.py": (
        "2f5678a5b3d2b098637b27558a8bdbffcc5160ca89cb8b4947f557320d03f5b7"
    ),
    "act/back_end/hybridz_tf/phase_projection_incremental_repair.py": (
        "acc1a98fa47d36c3b0bea7d10bf93af33d41ee9de108b0151ca01cd4822f997e"
    ),
}

CSV_LOCKS: Mapping[str, str] = {
    "cifar100_2024": (
        "aa656d7a73529ba7c41b5618440f543ba4677418bb44115d384b644cc034f9ee"
    ),
    "tinyimagenet_2024": (
        "188058624df1122f32295f99d83380485a7d736212555a5e8214204459c22b7e"
    ),
}


@dataclass(frozen=True)
class Case:
    name: str
    benchmark: str
    iid: int
    model_name: str
    spec_name: str
    onnx_sha256: str
    vnnlib_sha256: str


CASES: tuple[Case, ...] = (
    Case("cifar100_medium_iid2", "cifar100_2024", 2, "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_6232_sidx_3020_eps_0.0039.vnnlib", "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4", "33e795c8421b7b19125f32415adb9cee09b2f90cb83152c4cd3aa03810e91ec3"),
    Case("tinyimagenet_medium_iid143", "tinyimagenet_2024", 143, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "812bec2c0362d92d123380df161e1da6d5addbc84a27304d0a079090e814f5c7"),
    Case("cifar100_large_iid166", "cifar100_2024", 166, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_2630_sidx_1753_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "bdbd4493fbcc15ee7518afc86491eb183da4725a0525ad4af82cebc51b121b8c"),
    Case("tinyimagenet_medium_iid153", "tinyimagenet_2024", 153, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_2493_sidx_4209_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "f9c50760f284590d36366ffff8c9d4f628e554c11bc78d7e5d443b473dda8a17"),
    Case("cifar100_large_iid153", "cifar100_2024", 153, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_4652_sidx_1371_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "5b425e64c837085d070e219e8ac0e29012b30f2ef9f8e3af1ec7f5e00bc8e507"),
    Case("cifar100_medium_iid50", "cifar100_2024", 50, "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_913_sidx_2404_eps_0.0039.vnnlib", "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4", "295075c963461299d128f9514cefbc8b99082d11e42ee5403c57d39ec689addc"),
    Case("cifar100_large_iid101", "cifar100_2024", 101, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_4385_sidx_9116_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "c021ac7c83ad2405fbb2c32bf687413a9796ffaf90c98f8926073ca388d9bc6e"),
    Case("cifar100_large_iid120", "cifar100_2024", 120, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_2993_sidx_2485_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "2b5f71b56583f18c42a521b71dc637e6bcac3a970903c9e90479ed03a0add864"),
    Case("tinyimagenet_medium_iid9", "tinyimagenet_2024", 9, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_538_sidx_2467_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "17105dda046b27e5cd9eff31e4e97a32eed0e5b3656b81cb790ff5d0b5a41238"),
    Case("tinyimagenet_medium_iid173", "tinyimagenet_2024", 173, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_8444_sidx_2478_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "825a80c6622e501b61fe36eee0cf9c5a459c852e49b2e40ad7cb815119b344f0"),
    Case("tinyimagenet_medium_iid176", "tinyimagenet_2024", 176, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_3139_sidx_2973_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "50645c50e0a7dd906feb411451e71a734b4bd2913837f24b006f47ee768afea7"),
)

STAGE_A = tuple(case.name for case in CASES)
REQUIRED_FALSIFIED = (
    "cifar100_medium_iid2",
    "tinyimagenet_medium_iid143",
    "cifar100_large_iid166",
)
NEGATIVE_CONTROL = "cifar100_large_iid153"
OPTIONAL_TINY = "tinyimagenet_medium_iid153"
HISTORICAL_NO_CANDIDATE = tuple(case.name for case in CASES[5:])


class SentinelError(RuntimeError):
    """The disconnected experiment must stop without interpretation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_error_message(error: BaseException) -> str:
    if type(error).__module__ not in {"builtins", __name__} or len(error.args) != 1:
        return ""
    value = error.args[0]
    if type(value) is not str:
        return ""
    return "".join(c if c.isprintable() else "?" for c in value)[:240]


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _csr_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("ascii"))
    for array in (value.indptr, value.indices, value.data):
        digest.update(_array_sha256(array).encode("ascii"))
    return digest.hexdigest()


def _tuple_mapping_sha256(values: Sequence[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for value in values:
        normalized = tuple(
            bool(item) if type(item) is bool else int(item) for item in value
        )
        digest.update(_canonical_json(normalized).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_bundle() -> dict[str, str]:
    observed = {relative: _sha256(ROOT / relative) for relative in SOURCE_LOCKS}
    if observed != dict(SOURCE_LOCKS):
        changed = sorted(
            key for key, expected in SOURCE_LOCKS.items()
            if observed.get(key) != expected
        )
        raise SentinelError("frozen production source drifted: " + ",".join(changed))
    if _sha256(PREREG_PATH) != PREREG_SHA256:
        raise SentinelError("machine-readable preregistration drifted")
    return observed


def _bundle_sha256(bundle: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical_json(dict(bundle)).encode("utf-8")).hexdigest()


def _resolve(case: Case) -> tuple[Path, Path]:
    family = BENCHMARK_ROOT / case.benchmark
    csv_path = family / "instances.csv"
    if _sha256(csv_path) != CSV_LOCKS[case.benchmark]:
        raise SentinelError(f"{case.benchmark} instances.csv drifted")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if case.iid >= len(rows) or len(rows[case.iid]) != 3:
        raise SentinelError(f"{case.name} manifest row is malformed")
    model_rel, spec_rel, _timeout = rows[case.iid]
    model = (family / model_rel).resolve()
    spec = (family / spec_rel).resolve()
    if model.name != case.model_name or spec.name != case.spec_name:
        raise SentinelError(f"{case.name} manifest identity drifted")
    if not model.is_file() or not spec.is_file():
        raise SentinelError(f"{case.name} input is unavailable")
    if _sha256(model) != case.onnx_sha256 or _sha256(spec) != case.vnnlib_sha256:
        raise SentinelError(f"{case.name} input content drifted")
    return model, spec


_OWNER_CLASS_ANCHOR = """@dataclass(frozen=True)
class InfeasibleRaySelector:
    row_ray: np.ndarray
    row_ids: np.ndarray
    support_row_ids: tuple[int, ...]


@dataclass(frozen=True)
class OptimalCandidate:
"""
_OWNER_CLASS_REPLACEMENT = """@dataclass(frozen=True)
class InfeasibleRaySelector:
    row_ray: np.ndarray
    row_ids: np.ndarray
    support_row_ids: tuple[int, ...]


@dataclass(frozen=True)
class InfeasibleStatus:
    model_status: Any


@dataclass(frozen=True)
class OptimalCandidate:
"""
_OWNER_ALIAS_ANCHOR = (
    "BaseResult: TypeAlias = OptimalSelector | InfeasibleRaySelector | Unresolved\n"
)
_OWNER_ALIAS_REPLACEMENT = (
    "BaseResult: TypeAlias = "
    "OptimalSelector | InfeasibleRaySelector | InfeasibleStatus | Unresolved\n"
)
_OWNER_INFEASIBLE_ANCHOR = """            if status == highspy.HighsModelStatus.kInfeasible:
                result = self._read_base_ray(rows)
                self._base_columns = columns
                self._base_rows = rows
                self._state = "BASE_SOLVED"
                return result
"""
_OWNER_INFEASIBLE_REPLACEMENT = """            if status == highspy.HighsModelStatus.kInfeasible:
                result = InfeasibleStatus(model_status=status)
                self._base_columns = None
                self._base_rows = None
                self._state = "BASE_INFEASIBLE_STATUS"
                return result
"""

_PHASE_OWNER_IMPORT_ANCHOR = """        from act.back_end.hybridz_tf import (
            phase_projection_highs_owner as owner_module,
        )
"""
_PHASE_OWNER_IMPORT_REPLACEMENT = (
    "        owner_module = _scratch_infeasible_status_only_owner_module\n"
)
_PHASE_INFEASIBLE_ANCHOR = """        elif isinstance(base_result, owner_module.InfeasibleRaySelector):
            base_model_status = "INFEASIBLE"
            dual_ray_requests = 1
            if not np.array_equal(base_result.row_ids, base_rows.row_ids):
                raise ExactReLUPhaseProjectionUnknown(
                    "base infeasible row-id mapping drifted"
                )
            repair_selector_rule = (
                "infeasible_all_exact_nonzero_validated_dual_ray_phase_rows"
            )
            selected_ordinals = _select_infeasible_ray_rows(
                row_ray=base_result.row_ray,
                row_ids=base_result.row_ids,
                support_row_ids=base_result.support_row_ids,
            )
"""
_PHASE_INFEASIBLE_REPLACEMENT = """        elif isinstance(base_result, owner_module.InfeasibleStatus):
            base_model_status = "INFEASIBLE"
            repair_selector_rule = (
                "scratch_infeasible_first_analytic_inward_box_corner"
            )
            first_corner_margin = float(
                first_objective_center[int(rival)] + first_coeff @ first_factors
            )
            if not math.isfinite(first_corner_margin):
                raise ExactReLUPhaseProjectionUnknown(
                    "first analytic inward BOX corner margin is nonfinite"
                )
            final_factors = np.array(
                first_factors[: input_rows.size],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            candidate_margin = first_corner_margin
"""
_PHASE_SEAL_ANCHOR = """        if final_factors is None:
            if (
"""
_PHASE_SEAL_REPLACEMENT = """        if final_factors is None:
            selected_ordinals = _scratch_status_corner_seal_selected(
                selected_ordinals
            )
            if (
"""
_PHASE_PREFLIGHT_ANCHOR = """            repair_assign = {
                layer_id: np.asarray(value, dtype=np.bool_).copy()
                for layer_id, value in target_assign.items()
            }
"""
_PHASE_PREFLIGHT_REPLACEMENT = """            _scratch_status_corner_preflight(
                total_phases=total_phases,
                selected_ordinals=selected_ordinals,
                input_rows=input_rows,
                assert_width=int(candidate_program.assert_width),
                base_rows=base_rows,
                full_rows=A,
                keep=keep,
                physical_rows=physical_rows,
                deadline_monotonic=float(owner_deadline),
            )

            repair_assign = {
                layer_id: np.asarray(value, dtype=np.bool_).copy()
                for layer_id, value in target_assign.items()
            }
"""


def _named_ast(tree: ast.AST, class_name: str | None, function_name: str) -> ast.AST:
    nodes: Sequence[ast.stmt]
    if class_name is None:
        nodes = tree.body  # type: ignore[attr-defined]
    else:
        classes = [
            node for node in tree.body  # type: ignore[attr-defined]
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise SentinelError(f"AST class {class_name} is not unique")
        nodes = classes[0].body
    matches = [
        node for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise SentinelError(f"AST function {function_name} is not unique")
    return matches[0]


def _ast_sha(node: ast.AST) -> str:
    payload = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transformed_owner_source() -> tuple[str, str, str]:
    path = ROOT / "act/back_end/hybridz_tf/phase_projection_highs_owner.py"
    source = path.read_text(encoding="utf-8")
    replacements = (
        (_OWNER_CLASS_ANCHOR, _OWNER_CLASS_REPLACEMENT),
        (_OWNER_ALIAS_ANCHOR, _OWNER_ALIAS_REPLACEMENT),
        (_OWNER_INFEASIBLE_ANCHOR, _OWNER_INFEASIBLE_REPLACEMENT),
    )
    transformed = source
    for anchor, replacement in replacements:
        if transformed.count(anchor) != 1:
            raise SentinelError("private owner transform anchor is not unique")
        transformed = transformed.replace(anchor, replacement)
    original_tree = ast.parse(source)
    tree = ast.parse(transformed)
    optimal_original = _named_ast(original_tree, "SafeHighsOwner", "_read_base_optimal")
    optimal_private = _named_ast(tree, "SafeHighsOwner", "_read_base_optimal")
    if _ast_sha(optimal_original) != _ast_sha(optimal_private):
        raise SentinelError("private owner changed optimal readback semantics")
    solve = _named_ast(tree, "SafeHighsOwner", "solve_base")
    calls = [
        node for node in ast.walk(solve)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    if any(node.func.attr in {"_read_base_ray", "getDualRay", "getDualRayExist"} for node in calls):
        raise SentinelError("private owner INFEASIBLE dispatch still reads a ray")
    status_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InfeasibleStatus"
    ]
    if len(status_classes) != 1:
        raise SentinelError("private owner lacks one typed status-only result")
    digest = hashlib.sha256(transformed.encode("utf-8")).hexdigest()
    return transformed, digest, _ast_sha(optimal_private)


def _transformed_phase_source() -> tuple[str, str, str]:
    path = ROOT / "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py"
    source = path.read_text(encoding="utf-8")
    replacements = (
        (_PHASE_OWNER_IMPORT_ANCHOR, _PHASE_OWNER_IMPORT_REPLACEMENT),
        (_PHASE_INFEASIBLE_ANCHOR, _PHASE_INFEASIBLE_REPLACEMENT),
        (_PHASE_SEAL_ANCHOR, _PHASE_SEAL_REPLACEMENT),
        (_PHASE_PREFLIGHT_ANCHOR, _PHASE_PREFLIGHT_REPLACEMENT),
    )
    transformed = source
    for anchor, replacement in replacements:
        if transformed.count(anchor) != 1:
            raise SentinelError("private phase transform anchor is not unique")
        transformed = transformed.replace(anchor, replacement)
    original_tree = ast.parse(source)
    tree = ast.parse(transformed)
    selector_original = _named_ast(original_tree, None, "_select_optimal_negative_rows")
    selector_private = _named_ast(tree, None, "_select_optimal_negative_rows")
    if _ast_sha(selector_original) != _ast_sha(selector_private):
        raise SentinelError("private phase changed current-v3 optimal selector")
    legacy_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_select_infeasible_ray_rows"
    ]
    if legacy_calls:
        raise SentinelError("private phase still calls the infeasible-ray selector")
    preflight_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_scratch_status_corner_preflight"
    ]
    seal_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_scratch_status_corner_seal_selected"
    ]
    if len(preflight_calls) != 1 or len(seal_calls) != 1:
        raise SentinelError("private phase resource guard is not unique")
    digest = hashlib.sha256(transformed.encode("utf-8")).hexdigest()
    return transformed, digest, _ast_sha(selector_private)


def _load_private_modules() -> tuple[Any, Any, dict[str, str]]:
    owner_source, owner_sha, owner_optimal_sha = _transformed_owner_source()
    phase_source, phase_sha, phase_optimal_sha = _transformed_phase_source()
    if PRIVATE_OWNER_NAME in sys.modules or PRIVATE_PHASE_NAME in sys.modules:
        raise SentinelError("private module name is already occupied")
    owner = types.ModuleType(PRIVATE_OWNER_NAME)
    owner.__file__ = f"<{PRIVATE_OWNER_NAME}>"
    owner.__package__ = "act.back_end.hybridz_tf"
    sys.modules[PRIVATE_OWNER_NAME] = owner
    try:
        exec(compile(owner_source, owner.__file__, "exec"), owner.__dict__)
        phase = types.ModuleType(PRIVATE_PHASE_NAME)
        phase.__file__ = f"<{PRIVATE_PHASE_NAME}>"
        phase.__package__ = "act.back_end.hybridz_tf"
        phase.__dict__["_scratch_infeasible_status_only_owner_module"] = owner
        sys.modules[PRIVATE_PHASE_NAME] = phase
        exec(compile(phase_source, phase.__file__, "exec"), phase.__dict__)
    except BaseException:
        sys.modules.pop(PRIVATE_PHASE_NAME, None)
        sys.modules.pop(PRIVATE_OWNER_NAME, None)
        raise
    return phase, owner, {
        "private_owner_sha256": owner_sha,
        "private_phase_sha256": phase_sha,
        "owner_optimal_readback_ast_sha256": owner_optimal_sha,
        "phase_optimal_selector_ast_sha256": phase_optimal_sha,
    }


def _resource_bounds(
    *, P: int, K: int, X: int, O: int, B: int, L: int, base_loaded_nnz: int
) -> dict[str, int | str]:
    """Exact-integer fixed-cap bound, evaluated before repair allocation."""

    if any(type(value) is not int for value in (P, K, X, O, B, L, base_loaded_nnz)):
        raise SentinelError("resource preflight scalars must be exact integers")
    if (
        not (0 < K <= P <= MAX_PHASE_ROWS)
        or not (0 < X <= MAX_INPUT_COLUMNS)
        or not (0 < O <= INT32_MAX)
        or not (0 < B <= P)
        or not (0 <= L <= MAX_LOGICAL_NNZ)
        or not (0 <= base_loaded_nnz <= MAX_LOGICAL_NNZ)
        or K > MAX_SELECTED
        or X > INT32_MAX - K
        or P > INT32_MAX - K
        or K > MAX_DENSE_ELEMENTS // P
    ):
        raise SentinelError("resource preflight scalar cap failed")
    device_host_elements = (P + O) * K
    dense_elements = (3 * P + O) * K
    if device_host_elements > MAX_TRANSACTION_BYTES // 8:
        raise SentinelError("resource device host outputs exceed 2GB")
    if dense_elements > MAX_TRANSACTION_BYTES // 8:
        raise SentinelError("resource concurrent dense repair exceeds 2GB")
    if K > INT32_MAX // B:
        raise SentinelError("resource new-column multiplication exceeds int32")
    C = B * K
    R = L + (P - B) * K + K * (K + 1) // 2
    U = base_loaded_nnz + C + R
    if any(value > MAX_LOGICAL_NNZ or value > INT32_MAX for value in (C, R, U)):
        raise SentinelError("resource incremental nnz cap failed")
    D = 8 * dense_elements
    T = D + 64 * (L + C + R + P + B + K + X + O + 1)
    if T > MAX_TRANSACTION_BYTES:
        raise SentinelError("resource conservative transaction exceeds 2GB")
    return {
        "phase_rows_times_k": P * K,
        "k_squared": K * K,
        "device_host_output_bytes": 8 * device_host_elements,
        "concurrent_dense_bytes": D,
        "dense_peak_formula": "D=8*K*(3*P+O)",
        "C_new_csc_nnz_upper": C,
        "R_appended_nnz_upper": R,
        "R_formula": "R=L+(P-B)*K+K*(K+1)//2",
        "U_updated_nnz_upper": U,
        "U_formula": "U=base_loaded_nnz+C+R",
        "conservative_transaction_bytes": T,
        "transaction_formula": "T=D+64*(L+C+R+P+B+K+X+O+1)",
        "int32_max": INT32_MAX,
    }


def _owned_readonly_i64(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.int64) or raw.ndim != 1:
        raise SentinelError("selected row ids must be an int64 vector")
    owner = np.ascontiguousarray(raw, dtype=np.int64).tobytes(order="C")
    result = np.frombuffer(owner, dtype=np.int64)
    result.setflags(write=False)
    return result


def _exact_preflight(
    *,
    total_phases: Any,
    selected_ordinals: Any,
    input_rows: Any,
    assert_width: Any,
    base_rows: Any,
    full_rows: Any,
    keep: Any,
    physical_rows: Any,
    deadline_monotonic: Any,
) -> dict[str, Any]:
    import scipy.sparse as sp

    if type(total_phases) is not int or type(assert_width) is not int:
        raise SentinelError("preflight scalar ABI is malformed")
    P = total_phases
    O = assert_width
    selected = selected_ordinals
    input_vector = np.asarray(input_rows)
    keep_vector = np.asarray(keep)
    if (
        type(selected) is not np.ndarray
        or selected.dtype != np.dtype(np.int64)
        or selected.ndim != 1
        or not selected.flags.c_contiguous
        or selected.flags.writeable
        or type(selected.base) is not bytes
    ):
        raise SentinelError("selected rows are not an owned readonly int64 seal")
    K = int(selected.size)
    X = int(input_vector.size)
    if (
        type(input_rows) is not np.ndarray
        or input_vector.dtype != np.dtype(np.int64)
        or input_vector.ndim != 1
        or not input_vector.flags.c_contiguous
        or (X > 1 and np.any(input_vector[1:] <= input_vector[:-1]))
        or type(keep) is not np.ndarray
        or keep_vector.dtype != np.dtype(np.bool_)
        or keep_vector.shape != (P,)
        or not keep_vector.flags.c_contiguous
        or type(physical_rows) is not list
        or len(physical_rows) != P
        or not sp.isspmatrix_csr(full_rows)
        or full_rows.dtype != np.dtype(np.float64)
        or full_rows.shape != (P, X)
        or full_rows.indptr.dtype != np.dtype(np.int32)
        or full_rows.indices.dtype != np.dtype(np.int32)
        or full_rows.data.dtype != np.dtype(np.float64)
        or not full_rows.indptr.flags.c_contiguous
        or not full_rows.indices.flags.c_contiguous
        or not full_rows.data.flags.c_contiguous
    ):
        raise SentinelError("preflight phase/input frame is malformed")
    base_rows.assert_intact()
    A_nnz = int(full_rows.data.size)
    B = int(base_rows.rows)
    base_loaded_nnz = int(base_rows.data.size)
    if (
        type(deadline_monotonic) is not float
        or not math.isfinite(deadline_monotonic)
        or time.monotonic() >= deadline_monotonic
        or full_rows.indptr.shape != (P + 1,)
        or int(full_rows.indptr[0]) != 0
        or int(full_rows.indptr[-1]) != A_nnz
        or full_rows.indices.shape != full_rows.data.shape
        or (A_nnz and np.any(full_rows.indices < 0))
        or (A_nnz and np.any(full_rows.indices >= X))
        or (A_nnz and not np.all(np.isfinite(full_rows.data)))
        or (A_nnz and np.any(full_rows.data == 0.0))
        or np.any(selected < 0)
        or np.any(selected >= P)
        or (K > 1 and np.any(selected[1:] <= selected[:-1]))
        or not np.all(keep_vector[selected])
    ):
        raise SentinelError("preflight fixed integer/resource frame failed")
    resources = _resource_bounds(
        P=P,
        K=K,
        X=X,
        O=O,
        B=B,
        L=A_nnz,
        base_loaded_nnz=base_loaded_nnz,
    )
    kept = np.flatnonzero(keep_vector).astype(np.int64, copy=False)
    physical_valid = all(
        type(item) is tuple
        and len(item) == 3
        and all(type(component) is int for component in item)
        for item in physical_rows
    )
    if (
        kept.size != B
        or not np.array_equal(np.asarray(base_rows.row_ids), kept)
        or not physical_valid
        or len(set(tuple(item) for item in physical_rows)) != P
        or len({(int(item[0]), int(item[2])) for item in physical_rows}) != P
    ):
        raise SentinelError("preflight keep/physical mapping is not sealed")
    selected_physical = [
        (int(physical_rows[int(ordinal)][0]), int(physical_rows[int(ordinal)][2]))
        for ordinal in selected
    ]
    receipt = {
        "schema": "act.hybridz.status_corner_preflight.v1",
        "P_total_phase_rows": P,
        "K_selected_rows": K,
        "X_input_columns": X,
        "O_assert_width": O,
        "B_screened_base_rows": B,
        "A_logical_nnz": A_nnz,
        **resources,
        "selected_owned_immutable_bytes_readonly_int64": True,
        "selected_row_ids_sha256": _array_sha256(selected),
        "input_rows_sha256": _array_sha256(input_vector),
        "keep_sha256": _array_sha256(keep_vector),
        "base_row_ids_sha256": _array_sha256(np.asarray(base_rows.row_ids)),
        "base_rows_content_sha256": str(base_rows.content_sha256),
        "full_logical_csr_sha256": _csr_sha256(full_rows),
        "physical_mapping_sha256": _tuple_mapping_sha256(physical_rows),
        "selected_physical_mapping_sha256": _tuple_mapping_sha256(
            selected_physical
        ),
        "passed_before_repair_delta": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json(receipt).encode("utf-8")
    ).hexdigest()
    if time.monotonic() >= deadline_monotonic:
        raise SentinelError("preflight deadline expired after receipt sealing")
    return receipt


class _StatusCornerPatch:
    """Private status dispatch plus audited allocation/terminal boundaries."""

    def __init__(
        self,
        phase_module: Any,
        owner_module: Any,
        device_module: Any,
        repair_module: Any,
    ) -> None:
        self.phase = phase_module
        self.owner = owner_module
        self.device = device_module
        self.repair = repair_module
        self.real_optimal_selector = phase_module._select_optimal_negative_rows
        self.real_infeasible_selector = phase_module._select_infeasible_ray_rows
        self.real_build = phase_module.build_forward_exact_relu_phase_projection_candidate
        self.real_seal_delta = device_module.seal_delta_schedule
        self.real_build_repair = repair_module.build_incremental_repair
        self.real_seal_terminal = device_module.seal_terminal_input
        self.real_terminal_forward = device_module.terminal_interval_forward
        self._prior_canonical = sys.modules.get(CANONICAL_PHASE_NAME)
        self._preflight_receipt: dict[str, Any] | None = None
        self._preflight_bindings: dict[str, Any] | None = None
        self._terminal_object: Any | None = None
        self.audit: dict[str, Any] = {
            "schema": "act.hybridz.infeasible_first_corner_rule_audit.v1",
            "rule": "current_v3_OPTIMAL_else_typed_INFEASIBLE_first_corner",
            "optimal_selector_delegate_calls": 0,
            "legacy_infeasible_selector_calls": 0,
            "selected_seal_calls": 0,
            "delta_schedule_calls": 0,
            "incremental_repair_helper_calls": 0,
            "terminal_seal_calls": 0,
            "terminal_forward_calls": 0,
            "same_StoredBinary64Input_identity": False,
            "candidate_blind_terminal_signature": True,
            "preflight": None,
            "preflight_revalidated_at_repair_schedule": False,
            "preflight_revalidated_at_helper_entry": False,
            "preflight_bindings_released_at_helper_exit": False,
            "preflight_bindings_released_at_patch_exit": False,
            "returned_candidate_receipt": None,
        }

    def _unknown(self, message: str) -> BaseException:
        return self.phase.ExactReLUPhaseProjectionUnknown(message)

    def _optimal_delegate(self, **kwargs: Any) -> Any:
        self.audit["optimal_selector_delegate_calls"] += 1
        if self.audit["optimal_selector_delegate_calls"] != 1:
            raise self._unknown("current-v3 optimal selector was called twice")
        result = self.real_optimal_selector(**kwargs)
        selected, tight, negative = result
        if (
            type(selected) is not np.ndarray
            or selected.dtype != np.dtype(np.int64)
            or selected.ndim != 1
            or not selected.flags.c_contiguous
            or type(tight) is not int
            or type(negative) is not int
        ):
            raise self._unknown("current-v3 optimal selector ABI drifted")
        self.audit["optimal_v3_selection"] = {
            "rule": "all_primal_tight_and_strict_negative_upper_row_dual",
            "selected_count": int(selected.size),
            "selected_row_ids_sha256": _array_sha256(selected),
            "tight_count": tight,
            "strict_negative_upper_row_dual_count": negative,
            "delegated_without_candidate_or_row_space_change": True,
        }
        return result

    def _forbid_legacy_infeasible_selector(self, **_kwargs: Any) -> Any:
        self.audit["legacy_infeasible_selector_calls"] += 1
        raise self._unknown("typed INFEASIBLE must not invoke a ray selector")

    def _seal_selected(self, value: Any) -> np.ndarray:
        self.audit["selected_seal_calls"] += 1
        if self.audit["selected_seal_calls"] != 1:
            raise self._unknown("repair selection was sealed twice")
        try:
            sealed = _owned_readonly_i64(value)
        except Exception as exc:
            raise self._unknown("repair selection could not be sealed") from exc
        selection = self.audit.get("optimal_v3_selection")
        if (
            type(selection) is not dict
            or int(selection.get("selected_count", -1)) != int(sealed.size)
            or selection.get("selected_row_ids_sha256") != _array_sha256(sealed)
        ):
            raise self._unknown("sealed rows differ from current-v3 selection")
        return sealed

    def _preflight(self, **kwargs: Any) -> None:
        if self._preflight_receipt is not None:
            raise self._unknown("resource preflight was requested twice")
        try:
            receipt = _exact_preflight(**kwargs)
        except Exception as exc:
            raise self._unknown("resource preflight failed") from exc
        selection = self.audit.get("optimal_v3_selection")
        if (
            type(selection) is not dict
            or int(selection.get("selected_count", -1))
            != int(receipt["K_selected_rows"])
        ):
            raise self._unknown("preflight differs from current-v3 selection")
        self._preflight_bindings = dict(kwargs)
        self._preflight_receipt = receipt
        self.audit["preflight"] = dict(receipt)

    def _revalidate_preflight(self, stage: str) -> dict[str, Any]:
        receipt = self._preflight_receipt
        bindings = self._preflight_bindings
        if receipt is None or bindings is None:
            raise self._unknown(f"{stage} reached without preflight bindings")
        try:
            observed = _exact_preflight(**bindings)
        except Exception as exc:
            raise self._unknown(f"resource binding changed before {stage}") from exc
        if observed != receipt:
            raise self._unknown(f"preflight receipt changed before {stage}")
        return observed

    def _seal_delta_checked(
        self,
        program: Any,
        original_frames: Mapping[int, Any],
        target_frames: Mapping[int, Any],
        changes: Sequence[tuple[int, int, bool, bool]],
    ) -> Any:
        self.audit["delta_schedule_calls"] += 1
        if self._preflight_receipt is not None:
            receipt = self._revalidate_preflight("repair delta schedule")
            if self.audit["delta_schedule_calls"] != 2:
                raise self._unknown("repair delta is not the sole second schedule")
            if len(changes) != int(receipt["K_selected_rows"]):
                raise self._unknown("repair changes differ from preflight count")
            change_mapping = [(int(item[0]), int(item[1])) for item in changes]
            if (
                _tuple_mapping_sha256(change_mapping)
                != receipt["selected_physical_mapping_sha256"]
            ):
                raise self._unknown("repair physical mapping changed")
            self.audit["preflight_revalidated_at_repair_schedule"] = True
        return self.real_seal_delta(
            program, original_frames, target_frames, changes
        )

    def _build_repair_checked(self, *args: Any, **kwargs: Any) -> Any:
        self.audit["incremental_repair_helper_calls"] += 1
        receipt = self._revalidate_preflight("incremental repair helper entry")
        bindings = self._preflight_bindings
        if bindings is None:
            raise self._unknown("repair helper lacks preflight bindings")
        if (
            args
            or kwargs.get("full_oriented_rows") is not bindings["full_rows"]
            or kwargs.get("keep") is not bindings["keep"]
            or kwargs.get("base_rows") is not bindings["base_rows"]
            or _array_sha256(np.asarray(kwargs.get("selected_ordinals")))
            != receipt["selected_row_ids_sha256"]
        ):
            raise self._unknown("repair helper arguments differ from preflight")
        self.audit["preflight_revalidated_at_helper_entry"] = True
        try:
            return self.real_build_repair(**kwargs)
        finally:
            self._preflight_bindings = None
            self.audit["preflight_bindings_released_at_helper_exit"] = True

    def _seal_terminal_checked(self, decoded: Any) -> Any:
        self.audit["terminal_seal_calls"] += 1
        if self.audit["terminal_seal_calls"] != 1 or self._terminal_object is not None:
            raise self._unknown("terminal input was sealed more than once")
        sealed = self.real_seal_terminal(decoded)
        self._terminal_object = sealed
        self.audit["stored_binary64_input_sha256"] = _array_sha256(sealed.values)
        return sealed

    def _terminal_forward_checked(self, sealed: Any, program: Any) -> Any:
        self.audit["terminal_forward_calls"] += 1
        if (
            self.audit["terminal_forward_calls"] != 1
            or sealed is not self._terminal_object
            or type(sealed).__name__ != "StoredBinary64Input"
            or type(program).__name__ != "TerminalProgram"
        ):
            raise self._unknown("terminal did not receive the same sealed input")
        self.audit["same_StoredBinary64Input_identity"] = True
        return self.real_terminal_forward(sealed, program)

    def _build_wrapped(self, *args: Any, **kwargs: Any) -> Any:
        result = self.real_build(*args, **kwargs)
        receipt = result.receipt
        self.audit["returned_candidate_receipt"] = {
            "base_model_status": receipt.base_model_status,
            "repair_selector_rule": receipt.repair_selector_rule,
            "repair_updates": receipt.repair_updates,
            "owner_solves": receipt.owner_solves,
            "dual_ray_requests": receipt.dual_ray_requests,
            "singleton_margin_lower": receipt.singleton_margin_lower,
        }
        return result

    def __enter__(self) -> "_StatusCornerPatch":
        self.phase._select_optimal_negative_rows = self._optimal_delegate
        self.phase._select_infeasible_ray_rows = self._forbid_legacy_infeasible_selector
        self.phase._scratch_status_corner_seal_selected = self._seal_selected
        self.phase._scratch_status_corner_preflight = self._preflight
        self.phase.build_forward_exact_relu_phase_projection_candidate = self._build_wrapped
        self.device.seal_delta_schedule = self._seal_delta_checked
        self.repair.build_incremental_repair = self._build_repair_checked
        self.device.seal_terminal_input = self._seal_terminal_checked
        self.device.terminal_interval_forward = self._terminal_forward_checked
        sys.modules[CANONICAL_PHASE_NAME] = self.phase
        return self

    def finalize_audit(self) -> dict[str, Any]:
        return json.loads(_canonical_json(self.audit))

    def __exit__(self, _kind: Any, _value: Any, _tb: Any) -> bool:
        self.phase._select_optimal_negative_rows = self.real_optimal_selector
        self.phase._select_infeasible_ray_rows = self.real_infeasible_selector
        self.phase.__dict__.pop("_scratch_status_corner_seal_selected", None)
        self.phase.__dict__.pop("_scratch_status_corner_preflight", None)
        self.phase.build_forward_exact_relu_phase_projection_candidate = self.real_build
        self.device.seal_delta_schedule = self.real_seal_delta
        self.repair.build_incremental_repair = self.real_build_repair
        self.device.seal_terminal_input = self.real_seal_terminal
        self.device.terminal_interval_forward = self.real_terminal_forward
        self._preflight_bindings = None
        self.audit["preflight_bindings_released_at_patch_exit"] = True
        if self._prior_canonical is None:
            sys.modules.pop(CANONICAL_PHASE_NAME, None)
        else:
            sys.modules[CANONICAL_PHASE_NAME] = self._prior_canonical
        sys.modules.pop(PRIVATE_PHASE_NAME, None)
        sys.modules.pop(PRIVATE_OWNER_NAME, None)
        return False


class _OwnerInstrumentation:
    def __init__(self, owner_module: Any) -> None:
        self.owner_module = owner_module
        self.real_owner = owner_module.SafeHighsOwner
        self.real_highs = owner_module.highspy.Highs
        self.audit: dict[str, Any] = {
            "logical_owner_instances": 0,
            "logical_owner_close_calls": 0,
            "logical_owner_final_states": [],
            "base_solve_calls": 0,
            "base_result_types": [],
            "incremental_update_calls": 0,
            "native_owner_instances": 0,
            "native_run_calls": 0,
            "native_clear_calls": 0,
            "native_clear_model_calls": 0,
            "dual_ray_exist_calls": 0,
            "dual_ray_calls": 0,
        }

    def __enter__(self) -> "_OwnerInstrumentation":
        audit = self.audit
        real_owner = self.real_owner
        real_highs = self.real_highs

        class TrackingOwner(real_owner):
            def __init__(tracked_self: Any, *args: Any, **kwargs: Any) -> None:
                audit["logical_owner_instances"] += 1
                super().__init__(*args, **kwargs)

            def solve_base(tracked_self: Any, **kwargs: Any) -> Any:
                audit["base_solve_calls"] += 1
                result = super().solve_base(**kwargs)
                audit["base_result_types"].append(type(result).__name__)
                return result

            def apply_incremental_update(
                tracked_self: Any, **kwargs: Any
            ) -> Any:
                audit["incremental_update_calls"] += 1
                return super().apply_incremental_update(**kwargs)

            def close(tracked_self: Any) -> None:
                audit["logical_owner_close_calls"] += 1
                try:
                    super().close()
                finally:
                    audit["logical_owner_final_states"].append(
                        tracked_self.state
                    )

        class TrackingNative:
            def __init__(tracked_self: Any) -> None:
                audit["native_owner_instances"] += 1
                tracked_self._backend = real_highs()

            def __getattr__(tracked_self: Any, name: str) -> Any:
                return getattr(tracked_self._backend, name)

            def run(tracked_self: Any) -> Any:
                audit["native_run_calls"] += 1
                return tracked_self._backend.run()

            def clear(tracked_self: Any) -> Any:
                audit["native_clear_calls"] += 1
                return tracked_self._backend.clear()

            def clearModel(tracked_self: Any) -> Any:
                audit["native_clear_model_calls"] += 1
                return tracked_self._backend.clearModel()

            def getDualRayExist(tracked_self: Any) -> Any:
                audit["dual_ray_exist_calls"] += 1
                return tracked_self._backend.getDualRayExist()

            def getDualRay(tracked_self: Any) -> Any:
                audit["dual_ray_calls"] += 1
                return tracked_self._backend.getDualRay()

        self.owner_module.SafeHighsOwner = TrackingOwner
        self.owner_module.highspy.Highs = TrackingNative
        return self

    def __exit__(self, _kind: Any, _value: Any, _tb: Any) -> bool:
        self.owner_module.SafeHighsOwner = self.real_owner
        self.owner_module.highspy.Highs = self.real_highs
        return False


def _validate_owner_audit(audit: Mapping[str, Any]) -> None:
    integers = (
        "logical_owner_instances",
        "logical_owner_close_calls",
        "base_solve_calls",
        "incremental_update_calls",
        "native_owner_instances",
        "native_run_calls",
        "native_clear_calls",
        "native_clear_model_calls",
        "dual_ray_exist_calls",
        "dual_ray_calls",
    )
    if any(type(audit.get(name)) is not int for name in integers):
        raise SentinelError("owner audit counter is malformed")
    kinds = audit.get("base_result_types")
    states = audit.get("logical_owner_final_states")
    if (
        type(kinds) is not list
        or any(
            kind not in {"OptimalSelector", "InfeasibleStatus", "Unresolved"}
            for kind in kinds
        )
        or type(states) is not list
        or any(state != "CLOSED" for state in states)
    ):
        raise SentinelError("owner audit result/state receipt is malformed")
    logical = int(audit["logical_owner_instances"])
    native = int(audit["native_owner_instances"])
    if logical not in {0, 1} or native not in {0, 1} or native > logical:
        raise SentinelError("more than one private owner was observed")
    if int(audit["base_solve_calls"]) not in {0, 1}:
        raise SentinelError("base owner solve was called more than once")
    if len(kinds) > int(audit["base_solve_calls"]):
        raise SentinelError("owner result exists without a base solve")
    if int(audit["incremental_update_calls"]) not in {0, 1}:
        raise SentinelError("incremental update was called more than once")
    if int(audit["native_run_calls"]) not in {0, 1, 2}:
        raise SentinelError("native solve count exceeded current-v3")
    if int(audit["native_clear_model_calls"]) != 0:
        raise SentinelError("clearModel/reload was observed")
    if int(audit["dual_ray_exist_calls"]) != 0 or int(audit["dual_ray_calls"]) != 0:
        raise SentinelError("status-only owner touched a dual-ray API")
    if logical:
        if int(audit["logical_owner_close_calls"]) != 1 or states != ["CLOSED"]:
            raise SentinelError("logical owner cleanup is not exactly once")
        if native == 1 and int(audit["native_clear_calls"]) != 1:
            raise SentinelError("native owner cleanup is not exactly once")
        if native == 0 and any(
            int(audit[name]) != 0
            for name in (
                "native_run_calls",
                "native_clear_calls",
                "native_clear_model_calls",
                "dual_ray_exist_calls",
                "dual_ray_calls",
            )
        ):
            raise SentinelError("native activity exists without a native owner")
    elif states or any(int(audit[name]) for name in integers[1:]):
        raise SentinelError("owner activity exists without a logical owner")
    if kinds == ["InfeasibleStatus"] and (
        int(audit["native_run_calls"]) != 1
        or int(audit["incremental_update_calls"]) != 0
    ):
        raise SentinelError("typed INFEASIBLE performed a warm solve/update")
    if kinds == ["Unresolved"] and int(audit["incremental_update_calls"]) != 0:
        raise SentinelError("unresolved owner state performed an update")
    if kinds == ["OptimalSelector"] and not (
        1
        <= int(audit["native_run_calls"])
        <= 1 + int(audit["incremental_update_calls"])
    ):
        raise SentinelError("OPTIMAL native solve/update prefix is malformed")


def _validate_rule_audit(
    rule: Mapping[str, Any], owner: Mapping[str, Any]
) -> None:
    integers = (
        "optimal_selector_delegate_calls",
        "legacy_infeasible_selector_calls",
        "selected_seal_calls",
        "delta_schedule_calls",
        "incremental_repair_helper_calls",
        "terminal_seal_calls",
        "terminal_forward_calls",
    )
    if any(type(rule.get(name)) is not int for name in integers):
        raise SentinelError("rule audit counter is malformed")
    if (
        rule.get("schema")
        != "act.hybridz.infeasible_first_corner_rule_audit.v1"
        or rule.get("legacy_infeasible_selector_calls") != 0
        or rule.get("optimal_selector_delegate_calls") not in {0, 1}
        or rule.get("selected_seal_calls") not in {0, 1}
        or rule.get("delta_schedule_calls") not in {0, 1, 2}
        or rule.get("incremental_repair_helper_calls") not in {0, 1}
        or rule.get("terminal_seal_calls") not in {0, 1}
        or rule.get("terminal_forward_calls") not in {0, 1}
        or rule.get("terminal_forward_calls") > rule.get("terminal_seal_calls")
        or rule.get("candidate_blind_terminal_signature") is not True
        or rule.get("preflight_bindings_released_at_patch_exit") is not True
    ):
        raise SentinelError("rule audit violates the single-state shape")
    if rule.get("terminal_forward_calls") == 1 and (
        rule.get("same_StoredBinary64Input_identity") is not True
        or type(rule.get("stored_binary64_input_sha256")) is not str
        or len(rule["stored_binary64_input_sha256"]) != 64
    ):
        raise SentinelError("terminal did not consume the same stored object")
    preflight = rule.get("preflight")
    if type(preflight) is dict:
        if rule.get("preflight_bindings_released_at_patch_exit") is not True:
            raise SentinelError("preflight bindings survived patch exit")
        # A pre-native deadline can fire after either wrapper increments its
        # entry counter but before revalidation succeeds.  Only a path that
        # reached the owner update may claim completed schedule/helper checks;
        # every earlier UNKNOWN prefix is instead constrained explicitly below
        # and must release its bindings at patch exit.
        if owner.get("incremental_update_calls") == 1:
            if rule.get("delta_schedule_calls") == 2 and rule.get(
                "preflight_revalidated_at_repair_schedule"
            ) is not True:
                raise SentinelError("repair schedule lacked preflight revalidation")
            if rule.get("incremental_repair_helper_calls") == 1 and not (
                rule.get("preflight_revalidated_at_helper_entry") is True
                and rule.get("preflight_bindings_released_at_helper_exit") is True
            ):
                raise SentinelError("repair helper retained or skipped its binding")
    kinds = owner["base_result_types"]
    if kinds == ["InfeasibleStatus"]:
        if not (
            rule.get("optimal_selector_delegate_calls") == 0
            and rule.get("selected_seal_calls") == 0
            and rule.get("preflight") is None
            and rule.get("delta_schedule_calls") == 1
            and rule.get("incremental_repair_helper_calls") == 0
            and owner.get("incremental_update_calls") == 0
            and owner.get("native_run_calls") == 1
            and owner.get("dual_ray_exist_calls") == 0
            and owner.get("dual_ray_calls") == 0
        ):
            raise SentinelError("typed INFEASIBLE escaped status-corner dispatch")
    elif kinds == ["Unresolved"]:
        if not (
            rule.get("optimal_selector_delegate_calls") == 0
            and rule.get("selected_seal_calls") == 0
            and rule.get("preflight") is None
            and rule.get("incremental_repair_helper_calls") == 0
            and rule.get("terminal_seal_calls") == 0
            and rule.get("terminal_forward_calls") == 0
            and owner.get("incremental_update_calls") == 0
        ):
            raise SentinelError("unresolved owner state created a candidate")
    elif kinds == ["OptimalSelector"]:
        if owner.get("incremental_update_calls") == 1:
            if not (
                rule.get("optimal_selector_delegate_calls") == 1
                and rule.get("selected_seal_calls") == 1
                and type(preflight) is dict
                and preflight.get("schema") == "act.hybridz.status_corner_preflight.v1"
                and rule.get("delta_schedule_calls") == 2
                and rule.get("incremental_repair_helper_calls") == 1
                and rule.get("preflight_revalidated_at_repair_schedule") is True
                and rule.get("preflight_revalidated_at_helper_entry") is True
                and rule.get("preflight_bindings_released_at_helper_exit") is True
            ):
                raise SentinelError("OPTIMAL repair lacks the fixed preflight")
        elif rule.get("optimal_selector_delegate_calls") == 0:
            if preflight is not None or rule.get(
                "incremental_repair_helper_calls"
            ) != 0:
                raise SentinelError("base-positive OPTIMAL retained repair state")
        else:
            # A current-v3 negative selector may fail closed at any prefix
            # before native mutation: selection/seal, exact preflight,
            # repair schedule, or helper.  Such a prefix remains UNKNOWN;
            # every live binding must still be released by helper/patch exit.
            prefix = (
                rule.get("selected_seal_calls"),
                type(preflight) is dict,
                rule.get("delta_schedule_calls"),
                rule.get("preflight_revalidated_at_repair_schedule"),
                rule.get("incremental_repair_helper_calls"),
                rule.get("preflight_revalidated_at_helper_entry"),
                rule.get("preflight_bindings_released_at_helper_exit"),
            )
            reachable_prefixes = {
                (0, False, 1, False, 0, False, False),
                (1, False, 1, False, 0, False, False),
                (1, True, 1, False, 0, False, False),
                (1, True, 2, False, 0, False, False),
                (1, True, 2, True, 0, False, False),
                (1, True, 2, True, 1, False, False),
                (1, True, 2, True, 1, True, True),
            }
            if (
                prefix not in reachable_prefixes
                or owner.get("native_run_calls") != 1
                or rule.get("terminal_seal_calls") != 0
                or rule.get("terminal_forward_calls") != 0
            ):
                raise SentinelError("OPTIMAL fail-closed prefix is malformed")


def _validate_positive_receipt(
    receipt: Mapping[str, Any],
    rule: Mapping[str, Any],
    owner: Mapping[str, Any],
) -> None:
    margin = receipt.get("singleton_margin_lower")
    false_fields = (
        "candidate_authority",
        "proof_authority",
        "verdict_authority",
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
        "dual_ray_authority",
        "dual_selector_authority",
        "second_solver_used",
        "runtime_menu_used",
        "activation_split_used",
        "input_split_used",
        "enumeration_used",
        "cross_request_cache_used",
    )
    if (
        receipt.get("schema")
        != "act.hybridz.forward_exact_relu_phase_projection_candidate.v3"
        or receipt.get("status") != "singleton_verified"
        or receipt.get("singleton_interval_verified") is not True
        or type(margin) is not float
        or not math.isfinite(margin)
        or margin <= 0.0
        or any(receipt.get(field) is not False for field in false_fields)
        or receipt.get("fallbacks") != 0
        or receipt.get("retries") != 0
        or receipt.get("phase_retries") != 0
        or receipt.get("property_rows_selected") != 1
        or receipt.get("property_row_retries") != 0
        or receipt.get("owner_instances") != 1
        or receipt.get("repair_updates") not in {0, 1}
        or receipt.get("owner_solves") != 1 + receipt.get("repair_updates")
        or receipt.get("resolves_after_base") != receipt.get("repair_updates")
        or owner.get("native_run_calls") != receipt.get("owner_solves")
        or owner.get("incremental_update_calls") != receipt.get("repair_updates")
        or owner.get("dual_ray_exist_calls") != receipt.get("dual_ray_requests")
        or owner.get("dual_ray_calls") != receipt.get("dual_ray_requests")
        or receipt.get("same_owner_warm_update_used")
        is not bool(receipt.get("repair_updates"))
        or receipt.get("phase_delta_streams")
        != 1 + receipt.get("repair_updates")
        or receipt.get("same_stored_binary64_input_for_box_and_terminal") is not True
        or rule.get("terminal_seal_calls") != 1
        or rule.get("terminal_forward_calls") != 1
        or rule.get("same_StoredBinary64Input_identity") is not True
    ):
        raise SentinelError("formal positive receipt violates authority/scope")
    base_status = receipt.get("base_model_status")
    repair = int(receipt["repair_updates"])
    if base_status == "INFEASIBLE":
        if not (
            owner.get("base_result_types") == ["InfeasibleStatus"]
            and repair == 0
            and receipt.get("repair_selector_rule")
            == "scratch_infeasible_first_analytic_inward_box_corner"
            and receipt.get("repair_selected_rows") == 0
            and receipt.get("dual_ray_requests") == 0
            and receipt.get("dual_selector_used") is False
            and owner.get("native_run_calls") == 1
            and owner.get("incremental_update_calls") == 0
            and owner.get("dual_ray_exist_calls") == 0
            and owner.get("dual_ray_calls") == 0
        ):
            raise SentinelError("formal INFEASIBLE corner receipt drifted")
    elif base_status == "OPTIMAL":
        if owner.get("base_result_types") != ["OptimalSelector"]:
            raise SentinelError("formal OPTIMAL receipt lacks typed owner result")
        if repair == 0:
            if not (
                receipt.get("repair_selector_rule") == "base_positive_none"
                and receipt.get("repair_selected_rows") == 0
                and rule.get("optimal_selector_delegate_calls") == 0
            ):
                raise SentinelError("base-positive current-v3 receipt drifted")
        else:
            preflight = rule.get("preflight")
            if not (
                receipt.get("repair_selector_rule")
                == "optimal_negative_all_tight_strict_negative_upper_row_dual"
                and receipt.get("repair_selected_rows") > 0
                and receipt.get("dual_ray_requests") == 0
                and type(preflight) is dict
                and receipt.get("repair_selected_rows")
                == preflight.get("K_selected_rows")
                and rule.get("optimal_selector_delegate_calls") == 1
            ):
                raise SentinelError("optimal-negative current-v3 receipt drifted")
    else:
        raise SentinelError("formal positive has an unsupported owner status")


def _validate_record(record: Mapping[str, Any]) -> str:
    if record.get("schema") != "act.hybridz.infeasible_first_corner.worker.v1":
        raise SentinelError("worker schema drifted")
    if record.get("last_completed_stage") != "complete" or record.get(
        "error_stage"
    ) is not None:
        raise SentinelError("successful worker lacks a complete stage receipt")
    owner = record.get("owner_audit")
    rule = record.get("rule_audit")
    projection = record.get("phase_projection")
    if type(owner) is not dict or type(rule) is not dict or type(projection) is not dict:
        raise SentinelError("worker lacks owner/rule/projection receipts")
    _validate_owner_audit(owner)
    _validate_rule_audit(rule, owner)
    if projection.get("enabled") is not True or projection.get(
        "configured_seconds"
    ) != REQUEST_SECONDS:
        raise SentinelError("phase projection was not enabled for ten seconds")
    for field in (
        "input_sampling_used",
        "pgd_used",
        "concrete_onnx_execution_used",
        "bab_used",
        "backward_used",
        "dual_tightening_used",
    ):
        if projection.get(field) is not False:
            raise SentinelError("verifier reports a prohibited method")
    status = record.get("status")
    if status == "VerifyStatus.FALSIFIED":
        receipt = projection.get("candidate_receipt")
        if type(receipt) is not dict:
            raise SentinelError("formal positive lacks candidate receipt")
        _validate_positive_receipt(receipt, rule, owner)
        if not (
            record.get("has_counterexample") is True
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and projection.get("proof_rule")
            == "decoded_input_in_raw_BOX;verifier_owned_zero_width_forward_interval;"
            "exact_Fraction_property_lower_bound_positive"
        ):
            raise SentinelError("formal positive lacks verifier terminal authority")
        return "FALSIFIED"
    if status == "VerifyStatus.UNKNOWN":
        if record.get("has_counterexample") is not False or projection.get(
            "status"
        ) not in {"UNKNOWN", "not_run"}:
            raise SentinelError("UNKNOWN carries a counterexample or malformed phase status")
        return "UNKNOWN"
    raise SentinelError("worker returned a non fail-closed status")


def _identity(bundle: Mapping[str, str], private: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": "act.hybridz.infeasible_first_corner.identity.v1",
        "preregistration_sha256": PREREG_SHA256,
        "harness_sha256": _sha256(Path(__file__)),
        "canonical_source_bundle": dict(bundle),
        "canonical_source_bundle_sha256": _bundle_sha256(bundle),
        **dict(private),
        "csv_sha256": dict(CSV_LOCKS),
        "stage_a_order": list(STAGE_A),
        "input_sha256": {
            case.name: {"onnx": case.onnx_sha256, "vnnlib": case.vnnlib_sha256}
            for case in CASES
        },
        "request_seconds": REQUEST_SECONDS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "runtime_expected_or_public_labels": False,
        "production_mutations": 0,
        "gpu_requires_new_explicit_root_authorization": True,
    }


def _run_case(case: Case, stage_state: dict[str, str]) -> dict[str, Any]:
    stage_state["active"] = "imports"
    import torch

    from act.back_end.config import BackendConfig, HybridZConfig
    from act.back_end.hybridz_tf import phase_projection_device_program
    from act.back_end.hybridz_tf import phase_projection_incremental_repair
    from act.back_end.transfer_functions import (
        set_solver_mode,
        set_transfer_function_mode,
    )
    from act.back_end.verifier import verify_once
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.vnnlib_loader.create_specs import create_specs_from_paths
    from act.pipeline.verification.torch2act import TorchToACT
    from act.util.device_manager import initialize_device

    stage_state["last_completed"] = "imports"
    stage_state["active"] = "inputs_locked"
    bundle_before = _source_bundle()
    onnx, vnnlib = _resolve(case)
    input_before = {"onnx": _sha256(onnx), "vnnlib": _sha256(vnnlib)}
    stage_state["last_completed"] = "inputs_locked"
    stage_state["active"] = "private_modules_loaded"
    private_phase, private_owner, private_identity = _load_private_modules()
    stage_state["last_completed"] = "private_modules_loaded"
    started = time.monotonic()
    stage_state["active"] = "device_initialized"
    initialize_device(device="cuda", dtype="float64")
    set_solver_mode("hybridz")
    set_transfer_function_mode("interval")
    stage_state["last_completed"] = "device_initialized"
    stage_state["active"] = "model_synthesized"
    specs = create_specs_from_paths(str(onnx), str(vnnlib), category=case.benchmark)
    wrapped = synthesize_models_from_specs([specs])
    if len(wrapped) != 1:
        raise SentinelError("worker requires exactly one wrapped model")
    model = next(iter(wrapped.values())).to(
        device=torch.device("cuda"), dtype=torch.float64
    )
    stage_state["last_completed"] = "model_synthesized"
    stage_state["active"] = "network_converted"
    net = TorchToACT(model).run()
    stage_state["last_completed"] = "network_converted"
    rule = _StatusCornerPatch(
        private_phase,
        private_owner,
        phase_projection_device_program,
        phase_projection_incremental_repair,
    )
    owners = _OwnerInstrumentation(private_owner)
    stage_state["active"] = "verify_and_owner_cleanup"
    with rule:
        with owners:
            results = verify_once(
                net,
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cuda",
                    dtype="float64",
                    timeout=30.0,
                    bab_enabled=False,
                    lp_enabled=False,
                    hybridz=HybridZConfig(
                        timeout=20.0,
                        engine="operator_hz_objbound",
                        operator_exact_budget=-1,
                        operator_phase_projection_time_limit=REQUEST_SECONDS,
                        operator_materialize_add=True,
                    ),
                ),
            )
    stage_state["last_completed"] = "verify_and_owner_cleanup"
    stage_state["active"] = "cuda_synchronized"
    torch.cuda.synchronize()
    if len(results) != 1:
        raise SentinelError("worker returned more than one lane")
    stage_state["last_completed"] = "cuda_synchronized"
    stage_state["active"] = "postvalidate"
    result = results[0]
    bundle_after = _source_bundle()
    input_after = {"onnx": _sha256(onnx), "vnnlib": _sha256(vnnlib)}
    if bundle_after != bundle_before or input_after != input_before:
        raise SentinelError("canonical source or input changed during worker")
    observed_private = {
        "private_owner_sha256": _transformed_owner_source()[1],
        "private_phase_sha256": _transformed_phase_source()[1],
        "owner_optimal_readback_ast_sha256": _transformed_owner_source()[2],
        "phase_optimal_selector_ast_sha256": _transformed_phase_source()[2],
    }
    if observed_private != private_identity:
        raise SentinelError("private source identity changed during worker")
    record = {
        "schema": "act.hybridz.infeasible_first_corner.worker.v1",
        "case": case.name,
        "benchmark": case.benchmark,
        "iid": case.iid,
        "onnx": str(onnx),
        "vnnlib": str(vnnlib),
        "input_sha256": input_before,
        "canonical_source_bundle_sha256": _bundle_sha256(bundle_before),
        **private_identity,
        "status": str(result.status),
        "has_counterexample": result.counterexample is not None,
        "phase_projection": result.metadata.get("operator_phase_projection", {}),
        "owner_audit": owners.audit,
        "rule_audit": rule.finalize_audit(),
        "elapsed_seconds": time.monotonic() - started,
        "last_completed_stage": "complete",
        "error_stage": None,
        "scope": {
            "disconnected_scratch_only": True,
            "production_or_config_mutated": False,
            "runtime_expected_or_public_labels_read": False,
            "input_sampling_used": False,
            "onnx_point_execution_used": False,
            "pgd_used": False,
            "bab_split_or_enumeration_used": False,
            "backward_bounds_used": False,
            "dual_tightening_used": False,
            "fallback_menu_scan_used": False,
            "attempts_for_case": 1,
        },
    }
    record["validated_status"] = _validate_record(record)
    stage_state["last_completed"] = "complete"
    stage_state["active"] = "complete"
    return record


def _exclusive_jsonl(path: Path, first: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(first)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _worker_authorized(case_name: str, token: str) -> str:
    if type(token) is not str or len(token) < 32:
        raise SentinelError("worker lacks a parent attempt capability")
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with EVENTS_PATH.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    if not events or events[-1] != {
        "event": "case_attempt_started",
        "case": case_name,
        "attempt_token_sha256": digest,
    }:
        raise SentinelError("worker capability is not the last persisted attempt")
    with RAW_PATH.open("r", encoding="utf-8") as handle:
        if any(
            json.loads(line).get("case") == case_name
            for line in handle
            if line.strip()
        ):
            raise SentinelError("worker case already has a raw result")
    return digest


def _worker_entry(case_name: str) -> int:
    cases = {case.name: case for case in CASES}
    if case_name not in cases:
        raise SentinelError("worker case is outside the frozen manifest")
    token = os.environ.pop(WORKER_TOKEN_ENV, "")
    token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
    stage_state = {"active": "worker_entry", "last_completed": "worker_entry"}
    try:
        token_sha = _worker_authorized(case_name, token)
        stage_state["last_completed"] = "authorized"
        stage_state["active"] = "imports"
        with contextlib.redirect_stdout(sys.stderr):
            record = _run_case(cases[case_name], stage_state)
    except BaseException as exc:
        record = {
            "schema": "act.hybridz.infeasible_first_corner.worker.v1",
            "case": case_name,
            "status": "worker_error",
            "has_counterexample": False,
            "validated_status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message_safe": _safe_error_message(exc),
            "error_stage": stage_state["active"],
            "last_completed_stage": stage_state["last_completed"],
        }
    record["attempt_token_sha256"] = token_sha
    try:
        payload = _canonical_json(record)
    except BaseException as exc:
        record = {
            "schema": "act.hybridz.infeasible_first_corner.worker.v1",
            "case": case_name,
            "status": "worker_serialization_error",
            "has_counterexample": False,
            "validated_status": "ERROR",
            "error_type": type(exc).__name__,
            "error_message_safe": _safe_error_message(exc),
            "error_stage": "postvalidate",
            "last_completed_stage": stage_state["last_completed"],
            "attempt_token_sha256": token_sha,
        }
        payload = _canonical_json(record)
    print(RESULT_PREFIX + payload, flush=True)
    return 0 if record.get("validated_status") in {"FALSIFIED", "UNKNOWN"} else 2


def _validate_worker_error(
    record: Mapping[str, Any], case: Case, token_sha: str
) -> None:
    if not (
        record.get("schema") == "act.hybridz.infeasible_first_corner.worker.v1"
        and record.get("case") == case.name
        and record.get("validated_status") == "ERROR"
        and record.get("has_counterexample") is False
        and record.get("attempt_token_sha256") == token_sha
        and type(record.get("error_type")) is str
        and 0 < len(record["error_type"]) <= 128
        and record.get("error_stage") in WORKER_STAGES
        and record.get("last_completed_stage") in WORKER_STAGES
        and type(record.get("error_message_safe", "")) is str
        and len(record.get("error_message_safe", "")) <= 240
    ):
        raise SentinelError("worker ERROR receipt is malformed")


def _run_worker(case: Case, identity: Mapping[str, Any]) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
    _append_jsonl(
        EVENTS_PATH,
        {
            "event": "case_attempt_started",
            "case": case.name,
            "attempt_token_sha256": token_sha,
        },
    )
    env = dict(os.environ)
    env.pop(GPU_AUTH_ENV, None)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            WORKER_TOKEN_ENV: token,
        }
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        case.name,
    ]
    started = time.monotonic()
    stdout = ""
    stderr = ""
    timed_out = False
    transport_exception_type: str | None = None
    transport_exception_message = ""
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=WORKER_TIMEOUT_SECONDS,
        )
        stdout, stderr = completed.stdout, completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    except BaseException as exc:
        returncode = None
        transport_exception_type = type(exc).__name__
        transport_exception_message = _safe_error_message(exc)
    marked = [
        line[len(RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]

    def reject_nonfinite_json(_value: str) -> Any:
        raise ValueError("nonfinite JSON constant")

    try:
        record = (
            json.loads(marked[0], parse_constant=reject_nonfinite_json)
            if len(marked) == 1
            else {}
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        record = {}
    try:
        expected_stdout = (
            RESULT_PREFIX + _canonical_json(record) + "\n"
            if type(record) is dict and record
            else None
        )
    except (TypeError, ValueError, OverflowError):
        expected_stdout = None
    status_class = record.get("validated_status") if type(record) is dict else None
    expected_returncode = 0 if status_class in {"FALSIFIED", "UNKNOWN"} else 2
    if (
        timed_out
        or type(record) is not dict
        or record.get("case") != case.name
        or status_class not in {"FALSIFIED", "UNKNOWN", "ERROR"}
        or returncode != expected_returncode
        or expected_stdout is None
        or stdout != expected_stdout
    ):
        record = {
            "schema": "act.hybridz.infeasible_first_corner.worker.v1",
            "case": case.name,
            "status": "worker_transport_failure",
            "has_counterexample": False,
            "validated_status": "ERROR",
            "error_type": (
                "WorkerTimeout"
                if timed_out
                else transport_exception_type or "MalformedIsolatedResult"
            ),
            "error_message_safe": transport_exception_message,
            "attempt_token_sha256": token_sha,
            "error_stage": (
                record.get("error_stage", "transport")
                if type(record) is dict
                else "transport"
            ),
            "last_completed_stage": (
                record.get("last_completed_stage", "worker_entry")
                if type(record) is dict
                else "worker_entry"
            ),
        }
    elif record.get("validated_status") in {"FALSIFIED", "UNKNOWN"}:
        try:
            if record.get("attempt_token_sha256") != token_sha:
                raise SentinelError("worker attempt token digest drifted")
            if (
                record.get("canonical_source_bundle_sha256")
                != identity["canonical_source_bundle_sha256"]
            ):
                raise SentinelError("worker production source identity drifted")
            for field in (
                "private_owner_sha256",
                "private_phase_sha256",
                "owner_optimal_readback_ast_sha256",
                "phase_optimal_selector_ast_sha256",
            ):
                if record.get(field) != identity[field]:
                    raise SentinelError("worker private transform identity drifted")
            if record.get("input_sha256") != identity["input_sha256"][case.name]:
                raise SentinelError("worker input identity drifted")
            if _validate_record(record) != record["validated_status"]:
                raise SentinelError("worker/parent validation disagrees")
        except Exception as exc:
            record = {
                "schema": "act.hybridz.infeasible_first_corner.worker.v1",
                "case": case.name,
                "status": "parent_validation_failure",
                "has_counterexample": False,
                "validated_status": "ERROR",
                "error_type": type(exc).__name__,
                "error_message_safe": _safe_error_message(exc),
                "attempt_token_sha256": token_sha,
                "error_stage": "postvalidate",
                "last_completed_stage": "worker_entry",
            }
    else:
        try:
            _validate_worker_error(record, case, token_sha)
        except Exception as exc:
            record = {
                "schema": "act.hybridz.infeasible_first_corner.worker.v1",
                "case": case.name,
                "status": "parent_error_receipt_validation_failure",
                "has_counterexample": False,
                "validated_status": "ERROR",
                "error_type": type(exc).__name__,
                "error_message_safe": _safe_error_message(exc),
                "attempt_token_sha256": token_sha,
                "error_stage": "postvalidate",
                "last_completed_stage": "worker_entry",
            }
    stderr_tail = stderr.encode("utf-8")[-4096:]
    record["transport"] = {
        "child_wall_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stderr_tail_base64": base64.b64encode(stderr_tail).decode("ascii"),
        "stderr_tail_bytes": len(stderr_tail),
        "stderr_tail_truncated": len(stderr.encode("utf-8")) > len(stderr_tail),
        "isolated_result_records": len(marked),
        "stdout_exactly_one_canonical_marker_line": stdout == expected_stdout,
        "returncode_status_binding": "rc0=FALSIFIED_or_UNKNOWN;rc2=ERROR",
    }
    _append_jsonl(RAW_PATH, {"case": case.name, "record": record})
    receipt = {
        "case": case.name,
        "validated_status": record.get("validated_status"),
        "phase_projection": record.get("phase_projection"),
        "owner_audit": record.get("owner_audit"),
        "rule_audit": record.get("rule_audit"),
        "raw_record_sha256": hashlib.sha256(
            _canonical_json(record).encode("utf-8")
        ).hexdigest(),
    }
    _append_jsonl(RECEIPTS_PATH, receipt)
    _append_jsonl(
        EVENTS_PATH,
        {
            "event": "case_result_persisted",
            "case": case.name,
            "validated_status": record.get("validated_status"),
            "raw_record_sha256": receipt["raw_record_sha256"],
        },
    )
    return record


def _case_hard_gate(case_name: str, status: Any) -> tuple[bool, str]:
    if status == "ERROR":
        return False, "ERROR_first_hard_failure"
    if status not in {"FALSIFIED", "UNKNOWN"}:
        return False, "status_outside_FALSIFIED_UNKNOWN"
    if case_name in REQUIRED_FALSIFIED and status != "FALSIFIED":
        return False, "required_FALSIFIED_missing"
    if case_name == NEGATIVE_CONTROL and status != "UNKNOWN":
        return False, "negative_control_not_UNKNOWN"
    return True, "passed"


def _stage_a_success(
    results: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    statuses = {
        name: results.get(name, {}).get("validated_status") for name in STAGE_A
    }
    complete = len(results) == len(STAGE_A) and all(name in results for name in STAGE_A)
    required = [name for name in REQUIRED_FALSIFIED if statuses.get(name) == "FALSIFIED"]
    errors = [name for name, value in statuses.items() if value == "ERROR"]
    allowed = all(value in {"FALSIFIED", "UNKNOWN"} for value in statuses.values())
    passed = bool(
        complete
        and len(required) == len(REQUIRED_FALSIFIED)
        and statuses.get(NEGATIVE_CONTROL) == "UNKNOWN"
        and statuses.get(OPTIONAL_TINY) in {"FALSIFIED", "UNKNOWN"}
        and allowed
        and not errors
    )
    return passed, {
        "complete_all_11": complete,
        "required_falsified": required,
        "large153_status": statuses.get(NEGATIVE_CONTROL),
        "tiny153_status_ability_only": statuses.get(OPTIONAL_TINY),
        "historical_no_candidate_statuses": {
            name: statuses.get(name) for name in HISTORICAL_NO_CANDIDATE
        },
        "errors": errors,
    }


def _parent() -> int:
    if os.environ.pop(GPU_AUTH_ENV, "") != GPU_AUTH_VALUE:
        raise SentinelError("new explicit root GPU authorization value is absent")
    bundle = _source_bundle()
    for case in CASES:
        _resolve(case)
    owner_source, owner_sha, owner_optimal_sha = _transformed_owner_source()
    phase_source, phase_sha, phase_optimal_sha = _transformed_phase_source()
    del owner_source, phase_source
    private = {
        "private_owner_sha256": owner_sha,
        "private_phase_sha256": phase_sha,
        "owner_optimal_readback_ast_sha256": owner_optimal_sha,
        "phase_optimal_selector_ast_sha256": phase_optimal_sha,
    }
    identity = _identity(bundle, private)
    outputs = (EVENTS_PATH, RAW_PATH, RECEIPTS_PATH, SUMMARY_PATH)
    if any(path.exists() for path in outputs):
        raise SentinelError("exclusive status-corner output already exists; no retry")
    with Path(__file__).open("rb") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SentinelError("another status-corner parent owns the harness") from exc
        if any(path.exists() for path in outputs):
            raise SentinelError("exclusive output appeared while acquiring lock")
        identity_sha = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()
        _exclusive_jsonl(EVENTS_PATH, {"event": "run_created", "identity": identity})
        _exclusive_jsonl(
            RAW_PATH,
            {"event": "raw_ledger_created", "identity_sha256": identity_sha},
        )
        _exclusive_jsonl(
            RECEIPTS_PATH,
            {"event": "receipt_ledger_created", "identity_sha256": identity_sha},
        )
        _append_jsonl(EVENTS_PATH, {"event": "stage_started", "stage": "A"})
        results: dict[str, dict[str, Any]] = {}
        stop_reason: str | None = None
        for case in CASES:
            result = _run_worker(case, identity)
            results[case.name] = result
            gate, reason = _case_hard_gate(
                case.name, result.get("validated_status")
            )
            if not gate:
                stop_reason = f"{case.name}:{reason}"
                _append_jsonl(
                    EVENTS_PATH,
                    {
                        "event": "first_hard_failure_stop",
                        "case": case.name,
                        "reason": reason,
                    },
                )
                break
        stage_a_passed, stage_a_detail = _stage_a_success(results)
        _append_jsonl(
            EVENTS_PATH,
            {
                "event": "stage_completed",
                "stage": "A",
                "passed": stage_a_passed,
                "detail": stage_a_detail,
                "stop_reason": stop_reason,
            },
        )
        errors = [
            name
            for name, record in results.items()
            if record.get("validated_status") == "ERROR"
        ]
        if errors:
            status = "FAILED_CLOSED_ERROR"
        elif stage_a_passed:
            status = "COMPLETE_STAGE_A"
        else:
            status = "STAGE_A_FIRST_HARD_FAILURE_STOP"
        summary = {
            "schema": "act.hybridz.infeasible_first_corner.sentinel.v1",
            "status": status,
            "identity": identity,
            "stage_a_passed": stage_a_passed,
            "stage_a_detail": stage_a_detail,
            "first_hard_failure_stop_reason": stop_reason,
            "attempted": len(results),
            "falsified": [
                name
                for name, record in results.items()
                if record.get("validated_status") == "FALSIFIED"
            ],
            "unknown": [
                name
                for name, record in results.items()
                if record.get("validated_status") == "UNKNOWN"
            ],
            "errors": errors,
            "events_path": str(EVENTS_PATH.relative_to(ROOT)),
            "raw_path": str(RAW_PATH.relative_to(ROOT)),
            "receipts_path": str(RECEIPTS_PATH.relative_to(ROOT)),
            "formal_fixed400_changed": False,
            "production_or_config_changed": False,
            "tiny153_result_is_disconnected_ability_only": True,
        }
        _exclusive_json(SUMMARY_PATH, summary)
        _append_jsonl(
            EVENTS_PATH,
            {
                "event": "summary_persisted",
                "status": status,
                "summary_sha256": _sha256(SUMMARY_PATH),
            },
        )
        print(_canonical_json(summary), flush=True)
        return 0 if status == "COMPLETE_STAGE_A" else 2


def _cpu_hostile() -> dict[str, Any]:
    import scipy.sparse as sp

    bundle_before = _source_bundle()
    phase, owner, private_identity = _load_private_modules()
    try:
        safe_bounds = _resource_bounds(
            P=5_038,
            K=58,
            X=3_072,
            O=100,
            B=1_720,
            L=189_696,
            base_loaded_nnz=187_597,
        )
        overflow_rejected = False
        try:
            _resource_bounds(
                P=200_000,
                K=200_000,
                X=200_000,
                O=200,
                B=200_000,
                L=200_000_000,
                base_loaded_nnz=200_000_000,
            )
        except SentinelError:
            overflow_rejected = True
        if not overflow_rejected:
            raise SentinelError("resource hostile accepted an oversized repair")

        def f64(values: Sequence[float]) -> np.ndarray:
            return np.ascontiguousarray(values, dtype=np.float64)

        def i64(values: Sequence[int]) -> np.ndarray:
            return np.ascontiguousarray(values, dtype=np.int64)

        def csr(values: Sequence[Sequence[float]]) -> Any:
            matrix = sp.csr_matrix(np.asarray(values, dtype=np.float64))
            matrix.sort_indices()
            return matrix

        infeasible_rows = owner.FrozenRows.from_csr(
            csr([[1.0], [1.0]]),
            row_lower=f64([-np.inf, -np.inf]),
            row_upper=f64([-1.0, -2.0]),
            row_ids=i64([101, 202]),
            column_lower=f64([0.0]),
            column_upper=f64([1.0]),
        )
        infeasible_audit = _OwnerInstrumentation(owner)
        with infeasible_audit:
            with owner.SafeHighsOwner(
                deadline_monotonic=float(time.monotonic() + 10.0)
            ) as private_highs:
                infeasible_result = private_highs.solve_base(
                    cost=f64([0.0]),
                    column_lower=f64([0.0]),
                    column_upper=f64([1.0]),
                    rows=infeasible_rows,
                )
        _validate_owner_audit(infeasible_audit.audit)
        if (
            type(infeasible_result).__name__ != "InfeasibleStatus"
            or set(vars(infeasible_result)) != {"model_status"}
            or hasattr(infeasible_result, "row_ray")
            or hasattr(infeasible_result, "row_ids")
            or hasattr(infeasible_result, "support_row_ids")
            or infeasible_audit.audit["dual_ray_exist_calls"] != 0
            or infeasible_audit.audit["dual_ray_calls"] != 0
        ):
            raise SentinelError("private INFEASIBLE result is not status-only")

        with owner.SafeHighsOwner(
            deadline_monotonic=float(time.monotonic() + 10.0)
        ) as guarded_owner:
            guarded_result = guarded_owner.solve_base(
                cost=f64([0.0]),
                column_lower=f64([0.0]),
                column_upper=f64([1.0]),
                rows=infeasible_rows,
            )
            if (
                type(guarded_result).__name__ != "InfeasibleStatus"
                or guarded_owner.state != "BASE_INFEASIBLE_STATUS"
            ):
                raise SentinelError("typed INFEASIBLE owner state is not sealed")
            try:
                guarded_owner.apply_incremental_update(
                    new_columns=None,
                    existing_row_lower=None,
                    existing_row_upper=None,
                    appended_rows=None,
                )
            except owner.HighsOwnerUnknown:
                infeasible_update_rejected_before_arguments = bool(
                    guarded_owner.state == "POISONED"
                )
            else:
                infeasible_update_rejected_before_arguments = False
        if not infeasible_update_rejected_before_arguments:
            raise SentinelError("typed INFEASIBLE owner accepted an update")

        optimal_rows = owner.FrozenRows.from_csr(
            csr([[1.0]]),
            row_lower=f64([-np.inf]),
            row_upper=f64([0.75]),
            row_ids=i64([10]),
            column_lower=f64([-1.0]),
            column_upper=f64([1.0]),
        )
        optimal_audit = _OwnerInstrumentation(owner)
        with optimal_audit:
            with owner.SafeHighsOwner(
                deadline_monotonic=float(time.monotonic() + 10.0)
            ) as private_highs:
                optimal_result = private_highs.solve_base(
                    cost=f64([-2.0]),
                    column_lower=f64([-1.0]),
                    column_upper=f64([1.0]),
                    rows=optimal_rows,
                )
        _validate_owner_audit(optimal_audit.audit)
        if (
            type(optimal_result).__name__ != "OptimalSelector"
            or not np.array_equal(optimal_result.factors, f64([0.75]))
            or not np.array_equal(optimal_result.row_value, f64([0.75]))
            or not np.array_equal(optimal_result.row_dual, f64([-2.0]))
            or not np.array_equal(optimal_result.row_ids, i64([10]))
        ):
            raise SentinelError("private owner changed OPTIMAL readback")

        full = csr([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        keep = np.ascontiguousarray([True, True, False], dtype=np.bool_)
        base = owner.FrozenRows.from_csr(
            full[keep].tocsr(),
            row_lower=f64([-np.inf, -np.inf]),
            row_upper=f64([10.0, 10.0]),
            row_ids=i64([0, 1]),
            column_lower=f64([-1.0, -1.0]),
            column_upper=f64([1.0, 1.0]),
        )
        selected = _owned_readonly_i64(i64([1]))
        input_rows = i64([3, 7])
        physical = [(5, 0, 11), (5, 1, 12), (9, 0, 21)]

        class StoredBinary64Input:
            def __init__(self, decoded: Any) -> None:
                values = np.asarray(decoded, dtype=np.float64).reshape(-1)
                self.values = np.frombuffer(values.tobytes(), dtype=np.float64)

        class TerminalProgram:
            pass

        fake_device = types.SimpleNamespace(
            seal_delta_schedule=lambda *_args: "sealed_delta",
            seal_terminal_input=lambda decoded: StoredBinary64Input(decoded),
            terminal_interval_forward=lambda _sealed, _program: (
                f64([0.0]),
                f64([0.0]),
            ),
        )
        fake_repair = types.SimpleNamespace(
            build_incremental_repair=lambda **_kwargs: "repair"
        )
        patch = _StatusCornerPatch(phase, owner, fake_device, fake_repair)
        patch.audit["optimal_v3_selection"] = {
            "selected_count": 1,
            "selected_row_ids_sha256": _array_sha256(selected),
        }
        bindings = {
            "total_phases": 3,
            "selected_ordinals": selected,
            "input_rows": input_rows,
            "assert_width": 2,
            "base_rows": base,
            "full_rows": full,
            "keep": keep,
            "physical_rows": physical,
            "deadline_monotonic": float(time.monotonic() + 10.0),
        }
        patch._preflight(**bindings)
        keep[0] = False
        try:
            patch._revalidate_preflight("hostile keep mutation")
        except BaseException:
            keep_mutation_rejected = True
        else:
            keep_mutation_rejected = False
        finally:
            keep[0] = True
        prior_data = float(full.data[0])
        full.data[0] = prior_data + 1.0
        try:
            patch._revalidate_preflight("hostile CSR mutation")
        except BaseException:
            csr_mutation_rejected = True
        else:
            csr_mutation_rejected = False
        finally:
            full.data[0] = prior_data
        physical[1] = (5, 1, 99)
        try:
            patch._revalidate_preflight("hostile physical mutation")
        except BaseException:
            physical_mutation_rejected = True
        else:
            physical_mutation_rejected = False
        finally:
            physical[1] = (5, 1, 12)
        helper_result = patch._build_repair_checked(
            full_oriented_rows=full,
            keep=keep,
            base_rows=base,
            selected_ordinals=selected,
        )
        if (
            helper_result != "repair"
            or patch._preflight_bindings is not None
            or patch.audit["preflight_bindings_released_at_helper_exit"] is not True
            or not keep_mutation_rejected
            or not csr_mutation_rejected
            or not physical_mutation_rejected
        ):
            raise SentinelError("resource binding hostile check failed")

        failing_repair = types.SimpleNamespace(
            build_incremental_repair=lambda **_kwargs: (_ for _ in ()).throw(
                MemoryError("hostile helper")
            )
        )
        failure_patch = _StatusCornerPatch(
            phase, owner, fake_device, failing_repair
        )
        failure_patch.audit["optimal_v3_selection"] = {
            "selected_count": 1,
            "selected_row_ids_sha256": _array_sha256(selected),
        }
        failure_patch._preflight(**bindings)
        try:
            failure_patch._build_repair_checked(
                full_oriented_rows=full,
                keep=keep,
                base_rows=base,
                selected_ordinals=selected,
            )
        except MemoryError:
            helper_memoryerror_released = bool(
                failure_patch._preflight_bindings is None
                and failure_patch.audit[
                    "preflight_bindings_released_at_helper_exit"
                ]
                is True
            )
        else:
            helper_memoryerror_released = False
        if not helper_memoryerror_released:
            raise SentinelError("helper MemoryError retained preflight bindings")

        def prepared_deadline_patch() -> tuple[_StatusCornerPatch, np.ndarray]:
            deadline_patch = _StatusCornerPatch(
                phase, owner, fake_device, fake_repair
            )
            deadline_patch.audit["optimal_selector_delegate_calls"] = 1
            deadline_patch.audit["optimal_v3_selection"] = {
                "rule": "all_primal_tight_and_strict_negative_upper_row_dual",
                "selected_count": 1,
                "selected_row_ids_sha256": _array_sha256(selected),
                "tight_count": 1,
                "strict_negative_upper_row_dual_count": 1,
                "delegated_without_candidate_or_row_space_change": True,
            }
            sealed_selected = deadline_patch._seal_selected(selected)
            deadline_bindings = dict(bindings)
            deadline_bindings["selected_ordinals"] = sealed_selected
            deadline_bindings["deadline_monotonic"] = float(
                time.monotonic() + 10.0
            )
            # The first schedule is the immutable base schedule that precedes
            # the base solve in the real path.
            deadline_patch.audit["delta_schedule_calls"] = 1
            deadline_patch._preflight(**deadline_bindings)
            return deadline_patch, sealed_selected

        schedule_deadline_patch, _schedule_selected = prepared_deadline_patch()
        assert schedule_deadline_patch._preflight_bindings is not None
        schedule_deadline_patch._preflight_bindings["deadline_monotonic"] = float(
            time.monotonic() - 1.0
        )
        try:
            schedule_deadline_patch._seal_delta_checked(
                None, {}, {}, [(5, 12, False, True)]
            )
        except BaseException:
            schedule_deadline_prefix_raised = True
        else:
            schedule_deadline_prefix_raised = False
        finally:
            schedule_deadline_patch.__exit__(None, None, None)
        schedule_deadline_audit = schedule_deadline_patch.finalize_audit()
        _validate_rule_audit(schedule_deadline_audit, optimal_audit.audit)
        if not (
            schedule_deadline_prefix_raised
            and schedule_deadline_audit["delta_schedule_calls"] == 2
            and schedule_deadline_audit[
                "preflight_revalidated_at_repair_schedule"
            ]
            is False
            and schedule_deadline_audit["incremental_repair_helper_calls"] == 0
            and schedule_deadline_audit[
                "preflight_bindings_released_at_patch_exit"
            ]
            is True
        ):
            raise SentinelError("schedule-entry deadline prefix was rejected")

        helper_deadline_patch, helper_selected = prepared_deadline_patch()
        helper_deadline_patch._seal_delta_checked(
            None, {}, {}, [(5, 12, False, True)]
        )
        assert helper_deadline_patch._preflight_bindings is not None
        helper_deadline_patch._preflight_bindings["deadline_monotonic"] = float(
            time.monotonic() - 1.0
        )
        try:
            helper_deadline_patch._build_repair_checked(
                full_oriented_rows=full,
                keep=keep,
                base_rows=base,
                selected_ordinals=helper_selected,
            )
        except BaseException:
            helper_deadline_prefix_raised = True
        else:
            helper_deadline_prefix_raised = False
        finally:
            helper_deadline_patch.__exit__(None, None, None)
        helper_deadline_audit = helper_deadline_patch.finalize_audit()
        _validate_rule_audit(helper_deadline_audit, optimal_audit.audit)
        if not (
            helper_deadline_prefix_raised
            and helper_deadline_audit["delta_schedule_calls"] == 2
            and helper_deadline_audit[
                "preflight_revalidated_at_repair_schedule"
            ]
            is True
            and helper_deadline_audit["incremental_repair_helper_calls"] == 1
            and helper_deadline_audit[
                "preflight_revalidated_at_helper_entry"
            ]
            is False
            and helper_deadline_audit[
                "preflight_bindings_released_at_helper_exit"
            ]
            is False
            and helper_deadline_audit[
                "preflight_bindings_released_at_patch_exit"
            ]
            is True
        ):
            raise SentinelError("helper-entry deadline prefix was rejected")

        terminal_patch = _StatusCornerPatch(
            phase, owner, fake_device, fake_repair
        )
        sealed = terminal_patch._seal_terminal_checked(f64([0.25]))
        terminal_patch._terminal_forward_checked(sealed, TerminalProgram())
        if terminal_patch.audit["same_StoredBinary64Input_identity"] is not True:
            raise SentinelError("terminal identity hostile did not bind")
        wrong_terminal_patch = _StatusCornerPatch(
            phase, owner, fake_device, fake_repair
        )
        right = wrong_terminal_patch._seal_terminal_checked(f64([0.25]))
        wrong = StoredBinary64Input(f64([0.25]))
        try:
            wrong_terminal_patch._terminal_forward_checked(
                wrong, TerminalProgram()
            )
        except BaseException:
            wrong_terminal_identity_rejected = True
        else:
            wrong_terminal_identity_rejected = False
        if not wrong_terminal_identity_rejected or right is wrong:
            raise SentinelError("terminal accepted a different stored object")

        try:
            patch._forbid_legacy_infeasible_selector()
        except BaseException:
            legacy_ray_selector_rejected = True
        else:
            legacy_ray_selector_rejected = False
        if not legacy_ray_selector_rejected:
            raise SentinelError("legacy infeasible ray selector remained callable")

        forged_receipt: dict[str, Any] = {
            "schema": "act.hybridz.forward_exact_relu_phase_projection_candidate.v3",
            "status": "singleton_verified",
            "singleton_interval_verified": True,
            "singleton_margin_lower": 0.25,
            "fallbacks": 0,
            "retries": 0,
            "phase_retries": 0,
            "property_rows_selected": 1,
            "property_row_retries": 0,
            "owner_instances": 1,
            "repair_updates": 1,
            "owner_solves": 2,
            "resolves_after_base": 1,
            "same_owner_warm_update_used": True,
            "phase_delta_streams": 2,
            "same_stored_binary64_input_for_box_and_terminal": True,
            "base_model_status": "OPTIMAL",
            "repair_selector_rule": (
                "optimal_negative_all_tight_strict_negative_upper_row_dual"
            ),
            "repair_selected_rows": 1,
            "dual_ray_requests": 0,
        }
        for false_field in (
            "candidate_authority",
            "proof_authority",
            "verdict_authority",
            "input_sampling_used",
            "pgd_used",
            "concrete_onnx_execution_used",
            "bab_used",
            "backward_used",
            "dual_tightening_used",
            "dual_ray_authority",
            "dual_selector_authority",
            "second_solver_used",
            "runtime_menu_used",
            "activation_split_used",
            "input_split_used",
            "enumeration_used",
            "cross_request_cache_used",
        ):
            forged_receipt[false_field] = False
        forged_rule = {
            "terminal_seal_calls": 1,
            "terminal_forward_calls": 1,
            "same_StoredBinary64Input_identity": True,
            "optimal_selector_delegate_calls": 1,
            "preflight": {"K_selected_rows": 1},
        }
        forged_owner = {
            "base_result_types": ["OptimalSelector"],
            "native_run_calls": 1,
            "incremental_update_calls": 1,
            "dual_ray_exist_calls": 0,
            "dual_ray_calls": 0,
        }
        try:
            _validate_positive_receipt(
                forged_receipt, forged_rule, forged_owner
            )
        except SentinelError:
            forged_positive_counter_binding_rejected = True
        else:
            forged_positive_counter_binding_rejected = False
        if not forged_positive_counter_binding_rejected:
            raise SentinelError("formal positive accepted a missing native solve")
    finally:
        sys.modules.pop(PRIVATE_PHASE_NAME, None)
        sys.modules.pop(PRIVATE_OWNER_NAME, None)
    if _source_bundle() != bundle_before:
        raise SentinelError("CPU hostile changed canonical production")
    return {
        "private_transform_identity": private_identity,
        "typed_infeasible_status_only": True,
        "dualRayExist_calls": infeasible_audit.audit["dual_ray_exist_calls"],
        "dualRay_calls": infeasible_audit.audit["dual_ray_calls"],
        "typed_infeasible_one_base_solve_no_update": True,
        "typed_infeasible_update_rejected_before_arguments": True,
        "optimal_owner_readback_preserved": True,
        "resource_safe_profile_transaction_bytes": safe_bounds[
            "conservative_transaction_bytes"
        ],
        "oversized_resource_rejected": overflow_rejected,
        "keep_mutation_rejected_before_repair": keep_mutation_rejected,
        "csr_mutation_rejected_before_repair": csr_mutation_rejected,
        "physical_mutation_rejected_before_repair": physical_mutation_rejected,
        "helper_bindings_released_on_success_and_MemoryError": True,
        "schedule_entry_deadline_prefix_accepted_as_UNKNOWN": True,
        "helper_entry_deadline_prefix_accepted_as_UNKNOWN": True,
        "same_StoredBinary64Input_identity_enforced": True,
        "legacy_infeasible_ray_selector_rejected": True,
        "forged_positive_missing_native_solve_rejected": True,
        "gpu_called": False,
    }


def _static_check() -> dict[str, Any]:
    bundle = _source_bundle()
    for case in CASES:
        _resolve(case)
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if tuple(prereg["stage_a"]["case_order"]) != STAGE_A:
        raise SentinelError("preregistered Stage A order differs from harness")
    if tuple(prereg["stage_a"]["historical_no_candidate_status_cases"]) != (
        HISTORICAL_NO_CANDIDATE
    ):
        raise SentinelError("preregistered historical controls drifted")
    if len(CASES) != 11 or len({case.name for case in CASES}) != 11:
        raise SentinelError("case manifest is not eleven unique cases")
    evidence = prereg.get("frozen_optimal_retention_evidence", {})
    for path_field, hash_field in (
        ("summary_path", "summary_sha256"),
        ("events_path", "events_sha256"),
    ):
        evidence_path = ROOT / str(evidence.get(path_field, ""))
        if not evidence_path.is_file() or _sha256(evidence_path) != evidence.get(
            hash_field
        ):
            raise SentinelError("frozen OPTIMAL retention evidence drifted")
    owner_source, owner_sha, owner_optimal_sha = _transformed_owner_source()
    phase_source, phase_sha, phase_optimal_sha = _transformed_phase_source()
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_imports = {"onnxruntime", "scipy.optimize", "random"}
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    if forbidden_imports & imports:
        raise SentinelError("harness imports a forbidden search/second-solver module")
    verify_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "verify_once"
    ]
    if len(verify_calls) != 1:
        raise SentinelError("harness must have exactly one verify_once callsite")
    if any(path.exists() for path in (EVENTS_PATH, RAW_PATH, RECEIPTS_PATH, SUMMARY_PATH)):
        raise SentinelError("pre-GPU runtime output path is not absent")
    hostile = _cpu_hostile()
    if _source_bundle() != bundle:
        raise SentinelError("static/hostile check changed canonical production")
    return {
        "schema": "act.hybridz.infeasible_first_corner.static_check.v1",
        "status": "PASS_PRE_GPU",
        "harness_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": PREREG_SHA256,
        "canonical_source_bundle_sha256": _bundle_sha256(bundle),
        "private_owner_sha256": owner_sha,
        "private_phase_sha256": phase_sha,
        "owner_optimal_readback_ast_sha256": owner_optimal_sha,
        "phase_optimal_selector_ast_sha256": phase_optimal_sha,
        "private_owner_source_bytes": len(owner_source.encode("utf-8")),
        "private_phase_source_bytes": len(phase_source.encode("utf-8")),
        "runtime_output_paths_absent": True,
        "cuda_execution_started": False,
        "cpu_hostile": hostile,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--static-check", action="store_true")
    modes.add_argument("--run-authorized", action="store_true")
    modes.add_argument("--worker-case")
    args = parser.parse_args()
    if args.static_check:
        print(_canonical_json(_static_check()), flush=True)
        return 0
    if args.worker_case is not None:
        return _worker_entry(args.worker_case)
    if args.run_authorized:
        return _parent()
    raise SentinelError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
