#!/usr/bin/env python3
"""Disconnected, one-shot structural-support ability sentinel.

This file never changes ACT production, configuration, or tests.  At runtime
it privately loads the frozen production phase candidate with one call-site
change: ``row_dual`` is physically absent from the optimal-negative selector
signature.  The only experimental rule is the preregistered all-support rule.

``--static-check`` is CPU-only and creates no run artifact.  CUDA execution
requires both ``--run-authorized`` and the explicit root authorization
environment value frozen below.  Each case then runs in exactly one fresh
worker.  There is no retry, fallback, scan, menu, second update, or second
solver/owner.  Candidate/owner data have no authority; only the unchanged
raw-BOX, zero-width outward-forward, stored-binary64 Fraction terminal may
produce FALSIFIED.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import csv
from dataclasses import dataclass, replace
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path("/data1/Kane/data/vnncomp2025_benchmarks/benchmarks")
ARTIFACT_ROOT = ROOT / "artifacts/hybridz_largecls_gates"
PREREG_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_structural_support_preregistration_20260820.json"
)
EVENTS_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_structural_support_sentinel_20260820.events.jsonl"
)
RAW_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_structural_support_sentinel_20260820.raw.jsonl"
)
RECEIPTS_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_structural_support_sentinel_20260820.receipts.jsonl"
)
SUMMARY_PATH = (
    ARTIFACT_ROOT
    / "phase_projection_structural_support_sentinel_20260820.json"
)

RESULT_PREFIX = "@@ACT_STRUCTURAL_SUPPORT_RESULT@@"
WORKER_TOKEN_ENV = "ACT_STRUCTURAL_SUPPORT_ATTEMPT_TOKEN"
GPU_AUTH_ENV = "ACT_STRUCTURAL_SUPPORT_GPU_AUTHORIZATION"
GPU_AUTH_VALUE = "ROOT_AUTHORIZED_STRUCTURAL_SUPPORT_20260820"
REQUEST_SECONDS = 10.0
WORKER_TIMEOUT_SECONDS = 60.0
WORKER_STAGES = (
    "worker_entry",
    "authorized",
    "imports",
    "inputs_locked",
    "private_candidate_loaded",
    "device_initialized",
    "model_synthesized",
    "network_converted",
    "verify_and_owner_cleanup",
    "cuda_synchronized",
    "postvalidate",
    "complete",
)

PREREG_SHA256 = "a7619e1688a828b7455d50c86507e2d483570a45ec77a993da9a7e7eed2a7a1d"
CANONICAL_PHASE_NAME = (
    "act.back_end.hybridz_tf.forward_exact_relu_phase_projection_candidate"
)
PRIVATE_PHASE_NAME = "act.back_end.hybridz_tf._scratch_structural_support_candidate"

MAX_PHASE_ROWS = 200_000
MAX_INPUT_COLUMNS = 200_000
MAX_SELECTED = 200_000
MAX_DENSE_ELEMENTS = 200_000_000
MAX_LOGICAL_NNZ = 200_000_000
MAX_TRANSACTION_BYTES = 2_000_000_000
INT32_MAX = int(np.iinfo(np.int32).max)
SOLVER_TOLERANCE = 1.0e-9


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
    Case("tinyimagenet_medium_iid153", "tinyimagenet_2024", 153, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_2493_sidx_4209_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "f9c50760f284590d36366ffff8c9d4f628e554c11bc78d7e5d443b473dda8a17"),
    Case("cifar100_large_iid166", "cifar100_2024", 166, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_2630_sidx_1753_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "bdbd4493fbcc15ee7518afc86491eb183da4725a0525ad4af82cebc51b121b8c"),
    Case("cifar100_large_iid153", "cifar100_2024", 153, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_4652_sidx_1371_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "5b425e64c837085d070e219e8ac0e29012b30f2ef9f8e3af1ec7f5e00bc8e507"),
    Case("cifar100_large_iid110", "cifar100_2024", 110, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_1063_sidx_7948_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "f9b77b99fe82813df69b11e9ed2c378798c6d252277ab107ce3dd68429285b76"),
    Case("cifar100_large_iid160", "cifar100_2024", 160, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_5162_sidx_8126_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "c67e88f7a0a36f6dec78aa2d3d751c500e815078673662a1633b26b5e294e7be"),
    Case("cifar100_large_iid114", "cifar100_2024", 114, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_3585_sidx_3469_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "d19b43b56dcddb4af81cf81a44ef3f0a0f2b861c5da2a0729fb888dbe3759bfa"),
    Case("cifar100_large_iid161", "cifar100_2024", 161, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_8502_sidx_2893_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "34c4533bd3eeef8be75012a87cd6e1bed2999f10fd19c721d48509fda523d876"),
    Case("tinyimagenet_medium_iid93", "tinyimagenet_2024", 93, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_3437_sidx_2708_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "17428c1298247c3d94956164ca8784dec126b8a59bdfacd1f16556c080752651"),
    Case("cifar100_medium_iid50", "cifar100_2024", 50, "CIFAR100_resnet_medium.onnx", "CIFAR100_resnet_medium_prop_idx_913_sidx_2404_eps_0.0039.vnnlib", "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4", "295075c963461299d128f9514cefbc8b99082d11e42ee5403c57d39ec689addc"),
    Case("cifar100_large_iid101", "cifar100_2024", 101, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_4385_sidx_9116_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "c021ac7c83ad2405fbb2c32bf687413a9796ffaf90c98f8926073ca388d9bc6e"),
    Case("cifar100_large_iid120", "cifar100_2024", 120, "CIFAR100_resnet_large.onnx", "CIFAR100_resnet_large_prop_idx_2993_sidx_2485_eps_0.0039.vnnlib", "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16", "2b5f71b56583f18c42a521b71dc637e6bcac3a970903c9e90479ed03a0add864"),
    Case("tinyimagenet_medium_iid9", "tinyimagenet_2024", 9, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_538_sidx_2467_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "17105dda046b27e5cd9eff31e4e97a32eed0e5b3656b81cb790ff5d0b5a41238"),
    Case("tinyimagenet_medium_iid173", "tinyimagenet_2024", 173, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_8444_sidx_2478_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "825a80c6622e501b61fe36eee0cf9c5a459c852e49b2e40ad7cb815119b344f0"),
    Case("tinyimagenet_medium_iid176", "tinyimagenet_2024", 176, "TinyImageNet_resnet_medium.onnx", "TinyImageNet_resnet_medium_prop_idx_3139_sidx_2973_eps_0.0039.vnnlib", "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776", "50645c50e0a7dd906feb411451e71a734b4bd2913837f24b006f47ee768afea7"),
)

STAGE_A = tuple(case.name for case in CASES[:10])
STAGE_B = tuple(case.name for case in CASES[10:])
REQUIRED_RETAINED = (
    "cifar100_medium_iid2",
    "tinyimagenet_medium_iid143",
    "tinyimagenet_medium_iid153",
)
TARGET_CIFAR100 = (
    "cifar100_large_iid166",
    "cifar100_large_iid110",
    "cifar100_large_iid160",
    "cifar100_large_iid114",
    "cifar100_large_iid161",
)


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
    """Bound a primitive built-in message without invoking hostile __str__."""

    if type(error).__module__ not in {"builtins", __name__} or len(error.args) != 1:
        return ""
    value = error.args[0]
    if type(value) is not str:
        return ""
    printable = "".join(character if character.isprintable() else "?" for character in value)
    return printable[:240]


def _array_sha256(value: np.ndarray) -> str:
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
        normalized = tuple(int(item) if type(item) is not bool else bool(item) for item in value)
        digest.update(_canonical_json(normalized).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _owned_readonly_i64(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.int64) or raw.ndim != 1:
        raise SentinelError("row ids must enter as an int64 vector")
    owner = np.ascontiguousarray(raw, dtype=np.int64).tobytes(order="C")
    result = np.frombuffer(owner, dtype=np.int64)
    result.setflags(write=False)
    return result


def _owned_readonly_f64(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.float64) or raw.ndim != 1:
        raise SentinelError("opaque signal must enter as a float64 vector")
    owner = np.ascontiguousarray(raw, dtype=np.float64).tobytes(order="C")
    result = np.frombuffer(owner, dtype=np.float64)
    result.setflags(write=False)
    return result


def _source_bundle() -> dict[str, str]:
    observed = {relative: _sha256(ROOT / relative) for relative in SOURCE_LOCKS}
    if observed != dict(SOURCE_LOCKS):
        changed = sorted(key for key, expected in SOURCE_LOCKS.items() if observed.get(key) != expected)
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


_OPTIMAL_DUAL_ARGUMENT = "                    row_dual=base_result.row_dual,\n"
_OPAQUE_OLD_ARGUMENT = "                row_ray=base_result.row_ray,\n"
_OPAQUE_NEW_ARGUMENT = "                opaque_row_signal=base_result.row_ray,\n"
_PREFLIGHT_ANCHOR = """            repair_assign = {
                layer_id: np.asarray(value, dtype=np.bool_).copy()
                for layer_id, value in target_assign.items()
            }
"""
_PREFLIGHT_REPLACEMENT = """            _scratch_structural_support_preflight(
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


def _transformed_phase_source() -> tuple[str, str]:
    path = ROOT / "act/back_end/hybridz_tf/forward_exact_relu_phase_projection_candidate.py"
    source = path.read_text(encoding="utf-8")
    if source.count(_OPTIMAL_DUAL_ARGUMENT) != 1:
        raise SentinelError("optimal selector call-site transform is not unique")
    if source.count(_OPAQUE_OLD_ARGUMENT) != 1:
        raise SentinelError("opaque selector call-site transform is not unique")
    if source.count(_PREFLIGHT_ANCHOR) != 1:
        raise SentinelError("preflight insertion point is not unique")
    transformed = source.replace(_OPTIMAL_DUAL_ARGUMENT, "")
    transformed = transformed.replace(_OPAQUE_OLD_ARGUMENT, _OPAQUE_NEW_ARGUMENT)
    transformed = transformed.replace(_PREFLIGHT_ANCHOR, _PREFLIGHT_REPLACEMENT)
    tree = ast.parse(transformed)
    optimal_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_select_optimal_negative_rows"
    ]
    opaque_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_select_infeasible_ray_rows"
    ]
    preflight_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_scratch_structural_support_preflight"
    ]
    if len(optimal_calls) != 1 or {kw.arg for kw in optimal_calls[0].keywords} != {
        "row_value", "row_ids", "loaded_upper", "candidate_margin"
    }:
        raise SentinelError("private optimal selector still receives row_dual")
    if len(opaque_calls) != 1 or {kw.arg for kw in opaque_calls[0].keywords} != {
        "opaque_row_signal", "row_ids", "support_row_ids"
    }:
        raise SentinelError("private opaque selector ABI drifted")
    if len(preflight_calls) != 1:
        raise SentinelError("private candidate lacks exactly one early preflight")
    digest = hashlib.sha256(transformed.encode("utf-8")).hexdigest()
    return transformed, digest


def _load_private_candidate() -> tuple[Any, str]:
    transformed, digest = _transformed_phase_source()
    if PRIVATE_PHASE_NAME in sys.modules:
        raise SentinelError("private candidate module name is already occupied")
    module = types.ModuleType(PRIVATE_PHASE_NAME)
    module.__file__ = f"<{PRIVATE_PHASE_NAME}>"
    module.__package__ = "act.back_end.hybridz_tf"
    sys.modules[PRIVATE_PHASE_NAME] = module
    try:
        exec(compile(transformed, module.__file__, "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(PRIVATE_PHASE_NAME, None)
        raise
    return module, digest


def _resource_bounds(
    *, P: int, K: int, X: int, O: int, B: int, L: int, base_loaded_nnz: int
) -> dict[str, int | str]:
    """Exact-integer, fixed-cap repair bound; it performs no allocation."""

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
    if device_host_elements > MAX_TRANSACTION_BYTES // 8:
        raise SentinelError("resource device host outputs exceed 2GB")
    dense_elements = (3 * P + O) * K
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
        "base_loaded_nnz": base_loaded_nnz,
        "conservative_transaction_bytes": T,
        "transaction_formula": "T=D+64*(L+C+R+P+B+K+X+O+1)",
        "int32_max": INT32_MAX,
    }


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
        raise SentinelError("selected support is not owned readonly int64")
    K = int(selected.size)
    X = int(input_vector.size)
    if (
        type(input_rows) is not np.ndarray
        or input_vector.dtype != np.dtype(np.int64)
        or input_vector.ndim != 1
        or not input_vector.flags.c_contiguous
        or (input_vector.size > 1 and np.any(input_vector[1:] <= input_vector[:-1]))
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
        not math.isfinite(float(deadline_monotonic))
        or time.monotonic() >= float(deadline_monotonic)
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
        P=P, K=K, X=X, O=O, B=B, L=A_nnz,
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
        "schema": "act.hybridz.structural_support_preflight.v1",
        "P_total_phase_rows": P,
        "K_selected_support": K,
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
        "full_logical_csr_sha256": _csr_sha256(full_rows),
        "physical_mapping_sha256": _tuple_mapping_sha256(physical_rows),
        "selected_physical_mapping_sha256": _tuple_mapping_sha256(selected_physical),
        "passed_before_delta": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json(receipt).encode("utf-8")
    ).hexdigest()
    if time.monotonic() >= float(deadline_monotonic):
        raise SentinelError("preflight deadline expired after receipt sealing")
    return receipt


class _StructuralPatch:
    """Request-local private rule and immutable diagnostic instrumentation."""

    def __init__(
        self, phase_module: Any, owner_module: Any,
        device_module: Any, repair_module: Any,
    ) -> None:
        self.phase = phase_module
        self.owner = owner_module
        self.device = device_module
        self.repair = repair_module
        self.real_optimal_selector = phase_module._select_optimal_negative_rows
        self.real_opaque_selector = phase_module._select_infeasible_ray_rows
        self.real_build = phase_module.build_forward_exact_relu_phase_projection_candidate
        self.real_read_optimal = owner_module.SafeHighsOwner._read_base_optimal
        self.real_read_ray = owner_module.SafeHighsOwner._read_base_ray
        self.real_seal_delta = device_module.seal_delta_schedule
        self.real_build_repair = repair_module.build_incremental_repair
        self._prior_canonical = sys.modules.get(CANONICAL_PHASE_NAME)
        self._strict_negative_ids = np.empty(0, dtype=np.int64)
        self._tight_ids = np.empty(0, dtype=np.int64)
        self._preflight_receipt: dict[str, Any] | None = None
        self._preflight_bindings: dict[str, Any] | None = None
        self.audit: dict[str, Any] = {
            "schema": "act.hybridz.structural_support_rule_audit.v1",
            "rule": "all_primal_tight_or_all_opaque_exact_nonzero_once",
            "row_dual_consumed_by_selector": False,
            "opaque_signal_sign_or_magnitude_consumed_by_selector": False,
            "optimal_selector_calls": 0,
            "opaque_selector_calls": 0,
            "optimal_owner_abi_reads": 0,
            "opaque_owner_api_getDualRayExist_calls": 0,
            "opaque_owner_api_getDualRay_calls": 0,
            "delta_schedule_calls": 0,
            "selected_kind": "none",
            "preflight": None,
            "preflight_revalidated_at_repair_schedule": False,
            "preflight_revalidated_at_helper_entry": False,
            "preflight_bindings_released_at_helper_exit": False,
        }

    def _unknown(self, text: str) -> BaseException:
        return self.phase.ExactReLUPhaseProjectionUnknown(text)

    def _record_optimal_abi(self, result: Any) -> None:
        raw_dual = result.row_dual
        raw_ids = result.row_ids
        if (
            type(raw_dual) is not np.ndarray
            or raw_dual.dtype != np.dtype(np.float64)
            or raw_dual.ndim != 1
            or not raw_dual.flags.c_contiguous
            or type(raw_ids) is not np.ndarray
            or raw_ids.dtype != np.dtype(np.int64)
            or raw_ids.ndim != 1
            or not raw_ids.flags.c_contiguous
            or raw_ids.shape != raw_dual.shape
            or not np.all(np.isfinite(raw_dual))
            or np.any(raw_ids[1:] <= raw_ids[:-1])
        ):
            raise self._unknown("owner-sealed optimal dual ABI diagnostic is malformed")
        dual = _owned_readonly_f64(raw_dual)
        ids = _owned_readonly_i64(raw_ids)
        negative = dual < 0.0
        zero = dual == 0.0
        positive = dual > 0.0
        self._strict_negative_ids = np.ascontiguousarray(ids[negative], dtype=np.int64)
        self.audit["optimal_owner_abi_reads"] += 1
        self.audit["row_dual_diagnostic"] = {
            "role": "post_owner_validation_ABI_diagnostic_not_selector_input",
            "dtype": dual.dtype.str,
            "shape": list(dual.shape),
            "owned_immutable_bytes_snapshot": type(dual.base) is bytes,
            "readonly": not bool(dual.flags.writeable),
            "finite": True,
            "negative_count": int(np.count_nonzero(negative)),
            "zero_count": int(np.count_nonzero(zero)),
            "positive_count": int(np.count_nonzero(positive)),
            "negative_row_ids_sha256": _array_sha256(self._strict_negative_ids),
            "raw_binary64_sha256": _array_sha256(dual),
        }

    def _select_optimal(
        self,
        *,
        row_value: np.ndarray,
        row_ids: np.ndarray,
        loaded_upper: np.ndarray,
        candidate_margin: float,
    ) -> tuple[np.ndarray, int, int]:
        self.audit["optimal_selector_calls"] += 1
        raw_values = row_value
        raw_ids = row_ids
        raw_upper = loaded_upper
        if (
            type(raw_values) is not np.ndarray
            or raw_values.dtype != np.dtype(np.float64)
            or raw_values.ndim != 1
            or not raw_values.flags.c_contiguous
            or type(raw_ids) is not np.ndarray
            or raw_ids.dtype != np.dtype(np.int64)
            or raw_ids.ndim != 1
            or not raw_ids.flags.c_contiguous
            or type(raw_upper) is not np.ndarray
            or raw_upper.dtype != np.dtype(np.float64)
            or raw_upper.ndim != 1
            or not raw_upper.flags.c_contiguous
            or raw_values.shape != raw_ids.shape
            or raw_values.shape != raw_upper.shape
            or not raw_values.size
            or not np.all(np.isfinite(raw_values))
            or not np.all(np.isfinite(raw_upper))
            or np.any(raw_ids[1:] <= raw_ids[:-1])
            or not math.isfinite(float(candidate_margin))
            or not float(candidate_margin) < 0.0
        ):
            raise self._unknown("structural optimal-negative selector frame is malformed")
        values = _owned_readonly_f64(raw_values)
        ids = _owned_readonly_i64(raw_ids)
        upper = _owned_readonly_f64(raw_upper)
        residual = upper - values
        tight = residual <= SOLVER_TOLERANCE * (1.0 + np.abs(upper))
        selected = _owned_readonly_i64(ids[tight])
        if not selected.size:
            raise self._unknown("structural optimal-negative selector found no tight row")
        self._tight_ids = np.ascontiguousarray(selected, dtype=np.int64)
        self.audit["selected_kind"] = "optimal_all_primal_tight"
        self.audit["optimal_tight"] = {
            "formula": "residual=loaded_upper-row_value;tight=residual<=1e-9*(1+abs(loaded_upper))",
            "residual_absolute_value_used": False,
            "residual_nonnegative_gate_used": False,
            "screened_rows": int(ids.size),
            "tight_count": int(selected.size),
            "tight_row_ids_sha256": _array_sha256(selected),
            "row_value_sha256": _array_sha256(values),
            "loaded_upper_sha256": _array_sha256(upper),
            "selector_signature": ["row_value", "row_ids", "loaded_upper", "candidate_margin"],
        }
        # The third legacy return field is receipt-only ABI.  It is populated
        # from the independently sealed owner diagnostic after selection and
        # has no effect on ``selected``.
        strict_count = int(self.audit.get("row_dual_diagnostic", {}).get("negative_count", 0))
        return selected, int(selected.size), strict_count

    def _read_opaque(self, owner_self: Any, rows: Any) -> Any:
        if not rows.upper_only or owner_self._base_ray_requested:
            raise self.owner.HighsOwnerUnknown("opaque row signal requires one upper-only request")
        backend = owner_self._highs
        if backend is None:
            raise self.owner.HighsOwnerUnknown("single HiGHS backend is unavailable")
        rows.assert_intact()
        owner_self._remaining()
        self.audit["opaque_owner_api_getDualRayExist_calls"] += 1
        exist_status, exists = backend.getDualRayExist()
        owner_self._base_ray_requested = True
        owner_self._remaining()
        owner_self._require_ok(exist_status, "opaque getDualRayExist")
        if exists is not True:
            raise self.owner.HighsOwnerUnknown("infeasible owner reports no opaque row signal")
        owner_self._remaining()
        self.audit["opaque_owner_api_getDualRay_calls"] += 1
        signal_status, has_signal, raw_signal = backend.getDualRay()
        owner_self._remaining()
        owner_self._require_ok(signal_status, "opaque getDualRay")
        if has_signal is not True:
            raise self.owner.HighsOwnerUnknown("opaque row signal API returned no signal")
        if (
            type(raw_signal) is not np.ndarray
            or raw_signal.dtype != np.dtype(np.float64)
            or raw_signal.ndim != 1
            or not raw_signal.flags.c_contiguous
            or raw_signal.shape != (rows.rows,)
            or not np.all(np.isfinite(raw_signal))
            or not np.any(raw_signal != 0.0)
        ):
            raise self.owner.HighsOwnerUnknown("opaque row signal ABI is malformed")
        signal_owner = np.ascontiguousarray(raw_signal, dtype=np.float64).tobytes(order="C")
        signal = np.frombuffer(signal_owner, dtype=np.float64)
        signal.setflags(write=False)
        raw_row_ids = self.owner._readonly_i64(rows.row_ids)
        if (
            type(raw_row_ids) is not np.ndarray
            or raw_row_ids.dtype != np.dtype(np.int64)
            or raw_row_ids.ndim != 1
            or not raw_row_ids.flags.c_contiguous
            or raw_row_ids.shape != signal.shape
            or np.any(raw_row_ids[1:] <= raw_row_ids[:-1])
        ):
            raise self.owner.HighsOwnerUnknown("opaque row-id mapping ABI is malformed")
        row_ids = _owned_readonly_i64(raw_row_ids)
        if (
            not signal.flags.c_contiguous
            or signal.flags.writeable
            or signal.shape != row_ids.shape
            or np.any(row_ids[1:] <= row_ids[:-1])
        ):
            raise self.owner.HighsOwnerUnknown("opaque row signal ownership/mapping drifted")
        support = tuple(int(row_ids[index]) for index in np.flatnonzero(signal != 0.0))
        return self.owner.InfeasibleRaySelector(
            row_ray=signal,
            row_ids=row_ids,
            support_row_ids=support,
        )

    def _select_opaque(
        self,
        *,
        opaque_row_signal: np.ndarray,
        row_ids: np.ndarray,
        support_row_ids: tuple[int, ...],
    ) -> np.ndarray:
        self.audit["opaque_selector_calls"] += 1
        signal = opaque_row_signal
        raw_ids = row_ids
        if (
            type(signal) is not np.ndarray
            or signal.dtype != np.dtype(np.float64)
            or signal.ndim != 1
            or not signal.flags.c_contiguous
            or signal.flags.writeable
            or type(signal.base) is not bytes
            or type(raw_ids) is not np.ndarray
            or raw_ids.dtype != np.dtype(np.int64)
            or raw_ids.ndim != 1
            or not raw_ids.flags.c_contiguous
            or raw_ids.shape != signal.shape
            or not np.all(np.isfinite(signal))
            or np.any(raw_ids[1:] <= raw_ids[:-1])
        ):
            raise self._unknown("opaque support selector ABI is malformed")
        ids = _owned_readonly_i64(raw_ids)
        support_mask = signal != 0.0
        selected = _owned_readonly_i64(ids[support_mask])
        if not selected.size or tuple(int(value) for value in selected) != tuple(support_row_ids):
            raise self._unknown("opaque exact-nonzero support mapping drifted")
        negative = signal < 0.0
        positive = signal > 0.0
        positive_zero = (signal == 0.0) & ~np.signbit(signal)
        negative_zero = (signal == 0.0) & np.signbit(signal)
        self.audit["selected_kind"] = "infeasible_all_opaque_exact_nonzero"
        self.audit["opaque_signal"] = {
            "role": "opaque_exact_nonzero_support_only",
            "dtype": signal.dtype.str,
            "shape": list(signal.shape),
            "owned_immutable_bytes_snapshot": type(signal.base) is bytes,
            "readonly": not bool(signal.flags.writeable),
            "finite": True,
            "negative_count_diagnostic_only": int(np.count_nonzero(negative)),
            "positive_count_diagnostic_only": int(np.count_nonzero(positive)),
            "positive_zero_count_diagnostic_only": int(np.count_nonzero(positive_zero)),
            "negative_zero_count_diagnostic_only": int(np.count_nonzero(negative_zero)),
            "exact_nonzero_support_count": int(selected.size),
            "support_row_ids_sha256": _array_sha256(selected),
            "raw_binary64_sha256_diagnostic_only": _array_sha256(signal),
            "selector_predicate": "opaque_row_signal != 0.0",
        }
        return selected

    def _preflight(self, **kwargs: Any) -> None:
        if self._preflight_receipt is not None:
            raise self._unknown("structural support preflight was requested twice")
        try:
            receipt = _exact_preflight(**kwargs)
        except Exception as exc:
            raise self._unknown("structural support preflight failed") from exc
        selected_count = int(receipt["K_selected_support"])
        expected = 0
        if self.audit["selected_kind"] == "optimal_all_primal_tight":
            expected = int(self.audit["optimal_tight"]["tight_count"])
        elif self.audit["selected_kind"] == "infeasible_all_opaque_exact_nonzero":
            expected = int(self.audit["opaque_signal"]["exact_nonzero_support_count"])
        if selected_count != expected:
            raise self._unknown("preflight selection count differs from frozen support")
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
            raise self._unknown(f"structural binding changed before {stage}") from exc
        if observed != receipt:
            raise self._unknown(f"structural preflight receipt changed before {stage}")
        return observed

    def _seal_delta_checked(
        self,
        program: Any,
        original_frames: Mapping[int, Any],
        target_frames: Mapping[int, Any],
        changes: Sequence[tuple[int, int, bool, bool]],
    ) -> Any:
        self.audit["delta_schedule_calls"] += 1
        if self.audit["selected_kind"] != "none":
            receipt = self._revalidate_preflight("repair delta schedule")
            sealed = dict(receipt)
            claimed = sealed.pop("receipt_sha256", None)
            observed = hashlib.sha256(_canonical_json(sealed).encode("utf-8")).hexdigest()
            if claimed != observed or len(changes) != int(receipt["K_selected_support"]):
                raise self._unknown("repair preflight receipt changed before delta schedule")
            change_mapping = [(int(item[0]), int(item[1])) for item in changes]
            if _tuple_mapping_sha256(change_mapping) != receipt["selected_physical_mapping_sha256"]:
                raise self._unknown("repair physical support changed before delta schedule")
            if self.audit["delta_schedule_calls"] != 2:
                raise self._unknown("repair delta is not the sole second schedule")
            self.audit["preflight_revalidated_at_repair_schedule"] = True
        return self.real_seal_delta(program, original_frames, target_frames, changes)

    def _build_repair_checked(self, *args: Any, **kwargs: Any) -> Any:
        receipt = self._revalidate_preflight("incremental repair helper entry")
        bindings = self._preflight_bindings
        if bindings is None:
            raise self._unknown("repair helper lacks frozen bindings")
        if (
            args
            or kwargs.get("full_oriented_rows") is not bindings["full_rows"]
            or kwargs.get("keep") is not bindings["keep"]
            or kwargs.get("base_rows") is not bindings["base_rows"]
            or _array_sha256(np.asarray(kwargs.get("selected_ordinals")))
            != receipt["selected_row_ids_sha256"]
        ):
            raise self._unknown("repair helper arguments differ from frozen preflight")
        self.audit["preflight_revalidated_at_helper_entry"] = True
        try:
            return self.real_build_repair(**kwargs)
        finally:
            self._preflight_bindings = None
            self.audit["preflight_bindings_released_at_helper_exit"] = True

    def _build_wrapped(self, *args: Any, **kwargs: Any) -> Any:
        result = self.real_build(*args, **kwargs)
        receipt = result.receipt
        if receipt.repair_updates == 1:
            if self.audit["selected_kind"] == "optimal_all_primal_tight":
                rule = "scratch_optimal_negative_all_primal_tight_no_row_dual_input"
                receipt = replace(receipt, dual_selector_used=False)
            elif self.audit["selected_kind"] == "infeasible_all_opaque_exact_nonzero":
                rule = "scratch_infeasible_all_opaque_exact_nonzero_support"
            else:
                raise self._unknown("repair completed without the frozen structural rule")
            receipt = replace(receipt, repair_selector_rule=rule)
            result = replace(result, receipt=receipt)
        return result

    def __enter__(self) -> "_StructuralPatch":
        patch = self

        def read_optimal(owner_self: Any, columns: Any, rows: Any) -> Any:
            result = patch.real_read_optimal(owner_self, columns, rows)
            patch._record_optimal_abi(result)
            return result

        def read_opaque(owner_self: Any, rows: Any) -> Any:
            return patch._read_opaque(owner_self, rows)

        self._read_optimal_wrapper = read_optimal
        self._read_opaque_wrapper = read_opaque
        self.phase._select_optimal_negative_rows = self._select_optimal
        self.phase._select_infeasible_ray_rows = self._select_opaque
        self.phase._scratch_structural_support_preflight = self._preflight
        self.phase.build_forward_exact_relu_phase_projection_candidate = self._build_wrapped
        self.owner.SafeHighsOwner._read_base_optimal = read_optimal
        self.owner.SafeHighsOwner._read_base_ray = read_opaque
        self.device.seal_delta_schedule = self._seal_delta_checked
        self.repair.build_incremental_repair = self._build_repair_checked
        sys.modules[CANONICAL_PHASE_NAME] = self.phase
        return self

    def finalize_audit(self) -> dict[str, Any]:
        if self.audit["optimal_selector_calls"]:
            overlap = np.intersect1d(
                self._tight_ids, self._strict_negative_ids, assume_unique=True
            ).astype(np.int64, copy=False)
            self.audit["tight_vs_strict_negative_diagnostic"] = {
                "tight_count": int(self._tight_ids.size),
                "strict_negative_count": int(self._strict_negative_ids.size),
                "overlap_count": int(overlap.size),
                "tight_row_ids_sha256": _array_sha256(self._tight_ids),
                "strict_negative_row_ids_sha256": _array_sha256(self._strict_negative_ids),
                "overlap_row_ids_sha256": _array_sha256(overlap),
                "selection_depended_on_overlap": False,
            }
        self.audit["owner_api_per_call_caps"] = {
            "getDualRayExist_max": 1,
            "getDualRay_max": 1,
        }
        return json.loads(_canonical_json(self.audit))

    def __exit__(self, _kind: Any, _value: Any, _tb: Any) -> bool:
        self.phase._select_optimal_negative_rows = self.real_optimal_selector
        self.phase._select_infeasible_ray_rows = self.real_opaque_selector
        self.phase.__dict__.pop("_scratch_structural_support_preflight", None)
        self.phase.build_forward_exact_relu_phase_projection_candidate = self.real_build
        self.owner.SafeHighsOwner._read_base_optimal = self.real_read_optimal
        self.owner.SafeHighsOwner._read_base_ray = self.real_read_ray
        self.device.seal_delta_schedule = self.real_seal_delta
        self.repair.build_incremental_repair = self.real_build_repair
        self._preflight_bindings = None
        if self._prior_canonical is None:
            sys.modules.pop(CANONICAL_PHASE_NAME, None)
        else:
            sys.modules[CANONICAL_PHASE_NAME] = self._prior_canonical
        sys.modules.pop(PRIVATE_PHASE_NAME, None)
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

            def close(tracked_self: Any) -> None:
                audit["logical_owner_close_calls"] += 1
                try:
                    super().close()
                finally:
                    audit["logical_owner_final_states"].append(tracked_self.state)

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
        "logical_owner_instances", "logical_owner_close_calls",
        "native_owner_instances", "native_run_calls", "native_clear_calls",
        "native_clear_model_calls", "dual_ray_exist_calls", "dual_ray_calls",
    )
    if any(type(audit.get(name)) is not int for name in integers):
        raise SentinelError("owner audit counter is malformed")
    logical = int(audit["logical_owner_instances"])
    native = int(audit["native_owner_instances"])
    if logical not in {0, 1} or native not in {0, 1} or native > logical:
        raise SentinelError("more than one owner was observed")
    if int(audit["native_run_calls"]) not in {0, 1, 2}:
        raise SentinelError("owner solve count exceeded the frozen rule")
    if int(audit["dual_ray_exist_calls"]) not in {0, 1}:
        raise SentinelError("getDualRayExist exceeded one call")
    if int(audit["dual_ray_calls"]) not in {0, 1}:
        raise SentinelError("getDualRay exceeded one call")
    if int(audit["dual_ray_calls"]) > int(audit["dual_ray_exist_calls"]):
        raise SentinelError("getDualRay occurred without getDualRayExist")
    if int(audit["native_clear_model_calls"]) != 0:
        raise SentinelError("clearModel/reload was observed")
    states = audit.get("logical_owner_final_states")
    if type(states) is not list or any(state != "CLOSED" for state in states):
        raise SentinelError("owner did not finish CLOSED")
    if logical:
        if audit["logical_owner_close_calls"] != 1 or states != ["CLOSED"]:
            raise SentinelError("logical owner cleanup is not exactly once")
        if native == 1 and audit["native_clear_calls"] != 1:
            raise SentinelError("native owner cleanup is not exactly once")
        if native == 0 and any(
            int(audit[name]) != 0
            for name in (
                "native_run_calls", "native_clear_calls", "native_clear_model_calls",
                "dual_ray_exist_calls", "dual_ray_calls",
            )
        ):
            raise SentinelError("native activity exists without a native owner")
    elif states or any(int(audit[name]) for name in integers[1:]):
        raise SentinelError("owner activity exists without a logical owner")


def _validate_record(record: Mapping[str, Any]) -> str:
    if record.get("schema") != "act.hybridz.structural_support.worker.v1":
        raise SentinelError("worker schema drifted")
    if (
        record.get("last_completed_stage") != "complete"
        or record.get("error_stage") is not None
    ):
        raise SentinelError("successful worker lacks a complete stage receipt")
    owner = record.get("owner_audit")
    rule = record.get("rule_audit")
    projection = record.get("phase_projection")
    if type(owner) is not dict or type(rule) is not dict or type(projection) is not dict:
        raise SentinelError("worker lacks owner/rule/projection receipts")
    _validate_owner_audit(owner)
    if (
        rule.get("row_dual_consumed_by_selector") is not False
        or rule.get("opaque_signal_sign_or_magnitude_consumed_by_selector") is not False
        or rule.get("optimal_selector_calls") not in {0, 1}
        or rule.get("opaque_selector_calls") not in {0, 1}
        or rule.get("optimal_selector_calls") + rule.get("opaque_selector_calls") > 1
        or rule.get("opaque_owner_api_getDualRayExist_calls") != owner["dual_ray_exist_calls"]
        or rule.get("opaque_owner_api_getDualRay_calls") != owner["dual_ray_calls"]
        or rule.get("delta_schedule_calls") not in {0, 1, 2}
    ):
        raise SentinelError("structural rule audit violates the frozen shape")
    ray_exist_calls = int(owner["dual_ray_exist_calls"])
    ray_calls = int(owner["dual_ray_calls"])
    if int(rule.get("opaque_selector_calls", -1)) > ray_calls:
        raise SentinelError("opaque selector ran without a returned row signal")
    if (ray_exist_calls, ray_calls) == (1, 0) and not (
        rule.get("opaque_selector_calls") == 0
        and rule.get("optimal_selector_calls") == 0
        and rule.get("selected_kind") == "none"
        and rule.get("preflight") is None
        and rule.get("delta_schedule_calls") == 1
        and rule.get("preflight_revalidated_at_repair_schedule") is False
        and rule.get("preflight_revalidated_at_helper_entry") is False
        and rule.get("preflight_bindings_released_at_helper_exit") is False
    ):
        raise SentinelError("missing opaque signal performed downstream selection/update")
    if projection.get("enabled") is not True or projection.get("configured_seconds") != REQUEST_SECONDS:
        raise SentinelError("phase projection was not enabled for ten seconds")
    for field in (
        "input_sampling_used", "pgd_used", "concrete_onnx_execution_used",
        "bab_used", "backward_used", "dual_tightening_used",
    ):
        if projection.get(field) is not False:
            raise SentinelError("verifier reports a prohibited method")
    status = record.get("status")
    if status == "VerifyStatus.FALSIFIED":
        receipt = projection.get("candidate_receipt")
        if type(receipt) is not dict:
            raise SentinelError("formal positive lacks candidate receipt")
        margin = receipt.get("singleton_margin_lower")
        false_fields = (
            "candidate_authority", "proof_authority", "verdict_authority",
            "input_sampling_used", "pgd_used", "concrete_onnx_execution_used",
            "bab_used", "backward_used", "dual_tightening_used",
            "dual_ray_authority", "dual_selector_authority", "second_solver_used",
            "runtime_menu_used", "activation_split_used", "input_split_used",
            "enumeration_used", "cross_request_cache_used",
        )
        if (
            receipt.get("status") != "singleton_verified"
            or receipt.get("singleton_interval_verified") is not True
            or type(margin) is not float
            or not math.isfinite(margin)
            or margin <= 0.0
            or any(receipt.get(field) is not False for field in false_fields)
            or receipt.get("fallbacks") != 0
            or receipt.get("retries") != 0
            or receipt.get("phase_retries") != 0
            or receipt.get("property_row_retries") != 0
            or receipt.get("owner_instances") != 1
            or receipt.get("repair_updates") not in {0, 1}
            or receipt.get("owner_solves") != 1 + receipt.get("repair_updates")
            or receipt.get("resolves_after_base") != receipt.get("repair_updates")
            or receipt.get("same_stored_binary64_input_for_box_and_terminal") is not True
            or owner.get("logical_owner_instances") != 1
            or owner.get("native_owner_instances") != 1
        ):
            raise SentinelError("formal positive receipt violates authority/scope")
        if receipt.get("repair_updates") == 1:
            if (
                rule.get("preflight_revalidated_at_repair_schedule") is not True
                or rule.get("preflight_revalidated_at_helper_entry") is not True
                or rule.get("preflight_bindings_released_at_helper_exit") is not True
                or receipt.get("repair_selector_rule") not in {
                    "scratch_optimal_negative_all_primal_tight_no_row_dual_input",
                    "scratch_infeasible_all_opaque_exact_nonzero_support",
                }
                or receipt.get("repair_selected_rows") != rule["preflight"]["K_selected_support"]
            ):
                raise SentinelError("formal repair positive lacks structural preflight")
            if receipt.get("base_model_status") == "OPTIMAL":
                if not (
                    receipt.get("repair_selector_rule")
                    == "scratch_optimal_negative_all_primal_tight_no_row_dual_input"
                    and receipt.get("dual_selector_used") is False
                    and receipt.get("dual_ray_requests") == 0
                    and rule.get("selected_kind") == "optimal_all_primal_tight"
                    and owner.get("dual_ray_exist_calls") == 0
                    and owner.get("dual_ray_calls") == 0
                ):
                    raise SentinelError("optimal structural receipt falsely reports dual selection")
            elif receipt.get("base_model_status") == "INFEASIBLE":
                if not (
                    receipt.get("repair_selector_rule")
                    == "scratch_infeasible_all_opaque_exact_nonzero_support"
                    and receipt.get("dual_selector_used") is True
                    and receipt.get("dual_ray_requests") == 1
                    and rule.get("selected_kind")
                    == "infeasible_all_opaque_exact_nonzero"
                    and owner.get("dual_ray_exist_calls") == 1
                    and owner.get("dual_ray_calls") == 1
                ):
                    raise SentinelError("opaque structural receipt/status mapping drifted")
            else:
                raise SentinelError("repair positive has an unsupported base status")
        if not (
            record.get("has_counterexample") is True
            and projection.get("status") == "FALSIFIED"
            and projection.get("verifier_owned_proof_authority") is True
            and projection.get("proof_rule") == (
                "decoded_input_in_raw_BOX;verifier_owned_zero_width_forward_interval;"
                "exact_Fraction_property_lower_bound_positive"
            )
        ):
            raise SentinelError("formal positive lacks verifier-owned terminal authority")
        return "FALSIFIED"
    if status == "VerifyStatus.UNKNOWN":
        if record.get("has_counterexample") is not False or projection.get("status") not in {"UNKNOWN", "not_run"}:
            raise SentinelError("UNKNOWN carries a counterexample or malformed phase status")
        return "UNKNOWN"
    raise SentinelError("worker returned a non fail-closed status")


def _identity(bundle: Mapping[str, str], transformed_sha256: str) -> dict[str, Any]:
    return {
        "schema": "act.hybridz.structural_support.identity.v1",
        "preregistration_sha256": PREREG_SHA256,
        "harness_sha256": _sha256(Path(__file__)),
        "canonical_source_bundle": dict(bundle),
        "canonical_source_bundle_sha256": _bundle_sha256(bundle),
        "private_transformed_phase_sha256": transformed_sha256,
        "csv_sha256": dict(CSV_LOCKS),
        "stage_a_order": list(STAGE_A),
        "stage_b_order": list(STAGE_B),
        "input_sha256": {
            case.name: {"onnx": case.onnx_sha256, "vnnlib": case.vnnlib_sha256}
            for case in CASES
        },
        "request_seconds": REQUEST_SECONDS,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "runtime_expected_or_public_labels": False,
        "production_mutations": 0,
        "gpu_requires_explicit_root_authorization": True,
    }


def _run_case(case: Case, stage_state: dict[str, str]) -> dict[str, Any]:
    stage_state["active"] = "imports"
    import torch

    from act.back_end.config import BackendConfig, HybridZConfig
    from act.back_end.hybridz_tf import phase_projection_device_program
    from act.back_end.hybridz_tf import phase_projection_highs_owner
    from act.back_end.hybridz_tf import phase_projection_incremental_repair
    from act.back_end.transfer_functions import set_solver_mode, set_transfer_function_mode
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
    stage_state["active"] = "private_candidate_loaded"
    private_phase, private_sha = _load_private_candidate()
    stage_state["last_completed"] = "private_candidate_loaded"
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
    model = next(iter(wrapped.values())).to(device=torch.device("cuda"), dtype=torch.float64)
    stage_state["last_completed"] = "model_synthesized"
    stage_state["active"] = "network_converted"
    net = TorchToACT(model).run()
    stage_state["last_completed"] = "network_converted"
    structural = _StructuralPatch(
        private_phase, phase_projection_highs_owner,
        phase_projection_device_program, phase_projection_incremental_repair,
    )
    owners = _OwnerInstrumentation(phase_projection_highs_owner)
    stage_state["active"] = "verify_and_owner_cleanup"
    with structural:
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
    record = {
        "schema": "act.hybridz.structural_support.worker.v1",
        "case": case.name,
        "benchmark": case.benchmark,
        "iid": case.iid,
        "onnx": str(onnx),
        "vnnlib": str(vnnlib),
        "input_sha256": input_before,
        "canonical_source_bundle_sha256": _bundle_sha256(bundle_before),
        "private_transformed_phase_sha256": private_sha,
        "status": str(result.status),
        "has_counterexample": result.counterexample is not None,
        "phase_projection": result.metadata.get("operator_phase_projection", {}),
        "owner_audit": owners.audit,
        "rule_audit": structural.finalize_audit(),
        "elapsed_seconds": time.monotonic() - started,
        "last_completed_stage": "complete",
        "error_stage": None,
        "scope": {
            "disconnected_scratch_only": True,
            "production_or_config_mutated": False,
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
        if any(json.loads(line).get("case") == case_name for line in handle if line.strip()):
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
            "schema": "act.hybridz.structural_support.worker.v1",
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
            "schema": "act.hybridz.structural_support.worker.v1",
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


def _validate_worker_error(record: Mapping[str, Any], case: Case, token_sha: str) -> None:
    if not (
        record.get("schema") == "act.hybridz.structural_support.worker.v1"
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


def _stage_a_success(results: Mapping[str, Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    statuses = {name: results.get(name, {}).get("validated_status") for name in STAGE_A}
    retained = [name for name in REQUIRED_RETAINED if statuses.get(name) == "FALSIFIED"]
    new_cifar = [name for name in TARGET_CIFAR100 if statuses.get(name) == "FALSIFIED"]
    errors = [name for name in STAGE_A if statuses.get(name) == "ERROR"]
    passed = (
        len(results) >= len(STAGE_A)
        and len(retained) == len(REQUIRED_RETAINED)
        and bool(new_cifar)
        and statuses.get("cifar100_large_iid153") == "UNKNOWN"
        and not errors
    )
    return passed, {
        "required_retained": retained,
        "new_cifar100_falsified": new_cifar,
        "large153_status": statuses.get("cifar100_large_iid153"),
        "errors": errors,
    }


def _run_worker(case: Case, identity: Mapping[str, Any]) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
    _append_jsonl(EVENTS_PATH, {
        "event": "case_attempt_started", "case": case.name,
        "attempt_token_sha256": token_sha,
    })
    env = dict(os.environ)
    env.pop(GPU_AUTH_ENV, None)
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        WORKER_TOKEN_ENV: token,
    })
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-case", case.name]
    started = time.monotonic()
    stdout = ""
    stderr = ""
    timed_out = False
    transport_exception_type: str | None = None
    transport_exception_message = ""
    try:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
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
    marked = [line[len(RESULT_PREFIX):] for line in stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    def reject_nonfinite_json(_value: str) -> Any:
        raise ValueError("nonfinite JSON constant")

    try:
        record = (
            json.loads(marked[0], parse_constant=reject_nonfinite_json)
            if len(marked) == 1 else {}
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        record = {}
    try:
        expected_stdout = (
            RESULT_PREFIX + _canonical_json(record) + "\n"
            if type(record) is dict and record else None
        )
    except (TypeError, ValueError, OverflowError):
        expected_stdout = None
    status_class = record.get("validated_status") if type(record) is dict else None
    expected_returncode = 0 if status_class in {"FALSIFIED", "UNKNOWN"} else 2
    if (
        timed_out or type(record) is not dict or record.get("case") != case.name
        or status_class not in {"FALSIFIED", "UNKNOWN", "ERROR"}
        or returncode != expected_returncode
        or expected_stdout is None
        or stdout != expected_stdout
    ):
        record = {
            "schema": "act.hybridz.structural_support.worker.v1",
            "case": case.name,
            "status": "worker_transport_failure",
            "has_counterexample": False,
            "validated_status": "ERROR",
            "error_type": (
                "WorkerTimeout" if timed_out
                else transport_exception_type or "MalformedIsolatedResult"
            ),
            "error_message_safe": transport_exception_message,
            "attempt_token_sha256": token_sha,
            "error_stage": (
                record.get("error_stage", "transport")
                if type(record) is dict else "transport"
            ),
            "last_completed_stage": (
                record.get("last_completed_stage", "unknown")
                if type(record) is dict else "unknown"
            ),
        }
    elif record.get("validated_status") in {"FALSIFIED", "UNKNOWN"}:
        try:
            if record.get("attempt_token_sha256") != token_sha:
                raise SentinelError("worker attempt token digest drifted")
            if record.get("canonical_source_bundle_sha256") != identity["canonical_source_bundle_sha256"]:
                raise SentinelError("worker production source identity drifted")
            if record.get("private_transformed_phase_sha256") != identity["private_transformed_phase_sha256"]:
                raise SentinelError("worker private phase identity drifted")
            if record.get("input_sha256") != identity["input_sha256"][case.name]:
                raise SentinelError("worker input identity drifted")
            if _validate_record(record) != record["validated_status"]:
                raise SentinelError("worker/parent validation disagrees")
        except Exception as exc:
            record = {
                "schema": "act.hybridz.structural_support.worker.v1",
                "case": case.name,
                "status": "parent_validation_failure",
                "has_counterexample": False,
                "validated_status": "ERROR",
                "error_type": type(exc).__name__,
                "attempt_token_sha256": token_sha,
                "error_stage": "parent_postvalidate",
                "last_completed_stage": record.get("last_completed_stage", "unknown"),
            }
    elif record.get("validated_status") == "ERROR":
        try:
            _validate_worker_error(record, case, token_sha)
        except Exception as exc:
            record = {
                "schema": "act.hybridz.structural_support.worker.v1",
                "case": case.name,
                "status": "parent_error_receipt_validation_failure",
                "has_counterexample": False,
                "validated_status": "ERROR",
                "error_type": type(exc).__name__,
                "error_message_safe": _safe_error_message(exc),
                "attempt_token_sha256": token_sha,
                "error_stage": "parent_postvalidate",
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
        "raw_record_sha256": hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest(),
    }
    _append_jsonl(RECEIPTS_PATH, receipt)
    _append_jsonl(EVENTS_PATH, {
        "event": "case_result_persisted", "case": case.name,
        "validated_status": record.get("validated_status"),
        "raw_record_sha256": receipt["raw_record_sha256"],
    })
    return record


def _parent() -> int:
    if os.environ.pop(GPU_AUTH_ENV, "") != GPU_AUTH_VALUE:
        raise SentinelError("explicit root GPU authorization value is absent")
    bundle = _source_bundle()
    for case in CASES:
        _resolve(case)
    _transformed, private_sha = _transformed_phase_source()
    identity = _identity(bundle, private_sha)
    outputs = (EVENTS_PATH, RAW_PATH, RECEIPTS_PATH, SUMMARY_PATH)
    if any(path.exists() for path in outputs):
        raise SentinelError("exclusive 20260820 run output already exists; no resume or retry")
    with Path(__file__).open("rb") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SentinelError("another structural-support parent owns the harness") from exc
        if any(path.exists() for path in outputs):
            raise SentinelError("exclusive run output appeared while acquiring lock")
        header = {"event": "run_created", "identity": identity}
        _exclusive_jsonl(EVENTS_PATH, header)
        _exclusive_jsonl(RAW_PATH, {"event": "raw_ledger_created", "identity_sha256": hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()})
        _exclusive_jsonl(RECEIPTS_PATH, {"event": "receipt_ledger_created", "identity_sha256": hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()})
        results: dict[str, dict[str, Any]] = {}
        stage = "A"
        _append_jsonl(EVENTS_PATH, {"event": "stage_started", "stage": stage})
        for case in CASES[:10]:
            result = _run_worker(case, identity)
            results[case.name] = result
            if result.get("validated_status") == "ERROR":
                break
        stage_a_passed, stage_a_detail = _stage_a_success(results)
        _append_jsonl(EVENTS_PATH, {
            "event": "stage_completed", "stage": "A",
            "passed": stage_a_passed, "detail": stage_a_detail,
        })
        if stage_a_passed:
            stage = "B"
            _append_jsonl(EVENTS_PATH, {"event": "stage_started", "stage": stage})
            for case in CASES[10:]:
                result = _run_worker(case, identity)
                results[case.name] = result
                if result.get("validated_status") == "ERROR":
                    break
            stage_b_complete = all(name in results for name in STAGE_B)
            _append_jsonl(EVENTS_PATH, {
                "event": "stage_completed", "stage": "B",
                "complete": stage_b_complete,
            })
        else:
            stage_b_complete = False
        errors = [name for name, record in results.items() if record.get("validated_status") == "ERROR"]
        if errors:
            status = "FAILED_CLOSED_ERROR"
        elif not stage_a_passed:
            status = "STAGE_A_STOP_LOSS"
        elif stage_b_complete:
            status = "COMPLETE_STAGE_B"
        else:
            status = "STAGE_B_INCOMPLETE_NO_RETRY"
        summary = {
            "schema": "act.hybridz.structural_support.sentinel.v1",
            "status": status,
            "identity": identity,
            "stage_a_passed": stage_a_passed,
            "stage_a_detail": stage_a_detail,
            "stage_b_complete": stage_b_complete,
            "attempted": len(results),
            "falsified": [name for name, record in results.items() if record.get("validated_status") == "FALSIFIED"],
            "unknown": [name for name, record in results.items() if record.get("validated_status") == "UNKNOWN"],
            "errors": errors,
            "events_path": str(EVENTS_PATH.relative_to(ROOT)),
            "raw_path": str(RAW_PATH.relative_to(ROOT)),
            "receipts_path": str(RECEIPTS_PATH.relative_to(ROOT)),
            "formal_fixed400_changed": False,
            "production_or_config_changed": False,
            "historical_iid166_poison_evidence_role": "diagnostic_only_not_signal_sign_root_cause_assumption",
        }
        _exclusive_json(SUMMARY_PATH, summary)
        _append_jsonl(EVENTS_PATH, {
            "event": "summary_persisted", "status": status,
            "summary_sha256": _sha256(SUMMARY_PATH),
        })
        print(_canonical_json(summary), flush=True)
        return 0 if status == "COMPLETE_STAGE_B" else 2


def _cpu_hostile() -> dict[str, Any]:
    from act.back_end.hybridz_tf import phase_projection_device_program
    from act.back_end.hybridz_tf import phase_projection_highs_owner
    from act.back_end.hybridz_tf import phase_projection_incremental_repair

    private, private_sha = _load_private_candidate()
    patch = _StructuralPatch(
        private, phase_projection_highs_owner,
        phase_projection_device_program, phase_projection_incremental_repair,
    )
    canonical_before = sys.modules.get(CANONICAL_PHASE_NAME)
    bundle_before = _source_bundle()
    try:
        with patch:
            signature = inspect.signature(patch._select_optimal)
            if "row_dual" in signature.parameters:
                raise SentinelError("hostile: optimal selector signature contains row_dual")
            tiny_pos = np.nextafter(np.float64(0.0), np.float64(1.0))
            upper = np.asarray([5e-10, -5e-10, 0.0, 0.5], dtype=np.float64)
            just_over = np.nextafter(np.float64(1e-9), np.float64(np.inf))
            values = np.asarray([upper[0], upper[1], -just_over, upper[3] + 1.0], dtype=np.float64)
            ids = np.arange(4, dtype=np.int64)
            selected, _tight, _legacy = patch._select_optimal(
                row_value=values, row_ids=ids, loaded_upper=upper,
                candidate_margin=-1.0,
            )
            residual = upper - values
            expected = ids[residual <= SOLVER_TOLERANCE * (1.0 + np.abs(upper))]
            if not np.array_equal(selected, expected) or 2 in selected or 3 not in selected:
                raise SentinelError("hostile: exact tight formula/boundaries drifted")
            if not ({float(upper[0]), float(upper[1])} == {5e-10, -5e-10}):
                raise SentinelError("hostile: signed tiny upper coverage drifted")
            signal_owner = _owned_readonly_f64(
                np.asarray([0.0, -0.0, tiny_pos, -tiny_pos], dtype=np.float64)
            )
            opaque_ids = np.arange(10, 14, dtype=np.int64)
            opaque_ids.setflags(write=False)
            support = (12, 13)
            first = patch._select_opaque(
                opaque_row_signal=signal_owner, row_ids=opaque_ids,
                support_row_ids=support,
            )
            alternate = _owned_readonly_f64(
                np.asarray([-0.0, 0.0, -17.0, 31.0], dtype=np.float64)
            )
            second = patch._select_opaque(
                opaque_row_signal=alternate, row_ids=opaque_ids,
                support_row_ids=support,
            )
            if not np.array_equal(first, second) or tuple(first) != support:
                raise SentinelError("hostile: opaque selection consumed sign or magnitude")
            try:
                patch._select_opaque(
                    opaque_row_signal=signal_owner, row_ids=opaque_ids,
                    support_row_ids=(11, 12),
                )
            except private.ExactReLUPhaseProjectionUnknown:
                pass
            else:
                raise SentinelError("hostile: opaque support mapping drift was accepted")

            class OpaqueRows:
                upper_only = True
                rows = 4
                row_ids = np.arange(4, dtype=np.int64)
                def assert_intact(self) -> None:
                    return None

            class OpaqueBackend:
                def __init__(self, raw: Any) -> None:
                    self.raw = raw
                    self.exist_calls = 0
                    self.signal_calls = 0
                def getDualRayExist(self) -> tuple[object, bool]:
                    self.exist_calls += 1
                    return object(), True
                def getDualRay(self) -> tuple[object, bool, Any]:
                    self.signal_calls += 1
                    return object(), True, self.raw

            class OpaqueOwner:
                def __init__(self, raw: Any) -> None:
                    self._base_ray_requested = False
                    self._highs = OpaqueBackend(raw)
                def _remaining(self) -> float:
                    return 1.0
                def _require_ok(self, _status: Any, _stage: str) -> None:
                    return None

            valid_raw = np.array([0.0, -0.0, tiny_pos, -tiny_pos], dtype=np.float64)
            fake_owner = OpaqueOwner(valid_raw)
            opaque_result = patch._read_opaque(fake_owner, OpaqueRows())
            if (
                opaque_result.support_row_ids != (2, 3)
                or type(opaque_result.row_ray.base) is not bytes
                or opaque_result.row_ray.flags.writeable
                or fake_owner._highs.exist_calls != 1
                or fake_owner._highs.signal_calls != 1
            ):
                raise SentinelError("hostile: valid raw opaque owner ABI failed")
            try:
                patch._read_opaque(fake_owner, OpaqueRows())
            except phase_projection_highs_owner.HighsOwnerUnknown:
                pass
            else:
                raise SentinelError("hostile: opaque owner allowed a second API request")

            no_signal_owner = OpaqueOwner(valid_raw)
            def no_signal_exist() -> tuple[object, bool]:
                no_signal_owner._highs.exist_calls += 1
                return object(), False
            no_signal_owner._highs.getDualRayExist = no_signal_exist
            try:
                patch._read_opaque(no_signal_owner, OpaqueRows())
            except phase_projection_highs_owner.HighsOwnerUnknown:
                pass
            else:
                raise SentinelError("hostile: missing opaque signal did not fail closed")
            if (
                no_signal_owner._highs.exist_calls != 1
                or no_signal_owner._highs.signal_calls != 0
            ):
                raise SentinelError("hostile: missing signal still called getDualRay")

            noncontiguous_base = np.arange(8, dtype=np.float64)
            malformed_raw = (
                [0.0, 1.0],
                np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
                noncontiguous_base[::2],
                np.asarray([0.0, np.nan, 0.0, 1.0], dtype=np.float64),
                np.asarray([0.0, np.inf, 0.0, 1.0], dtype=np.float64),
                np.zeros(4, dtype=np.float64),
                np.asarray([], dtype=np.float64),
            )
            for malformed in malformed_raw:
                hostile_owner = OpaqueOwner(malformed)
                try:
                    patch._read_opaque(hostile_owner, OpaqueRows())
                except phase_projection_highs_owner.HighsOwnerUnknown:
                    pass
                else:
                    raise SentinelError("hostile: malformed raw opaque ABI was coerced")

            baseline_resources = {
                "P": 3, "K": 1, "X": 2, "O": 2,
                "B": 2, "L": 2, "base_loaded_nnz": 2,
            }
            resource_hostiles = (
                {"K": 0},
                {"P": 1, "K": 2, "B": 1},
                {"P": MAX_PHASE_ROWS + 1, "B": 1},
                {"X": MAX_INPUT_COLUMNS + 1},
                {"P": 200_000, "K": 1_001, "B": 200_000},
                {"P": 1, "K": 1, "B": 1, "O": 250_000_000},
                {"P": 200_000, "K": 500, "B": 200_000},
                {"P": 1, "K": 1, "B": 1, "L": 20_000_000},
                {"X": INT32_MAX, "K": 1},
                {"P": INT32_MAX, "K": 1, "B": 1},
                {"L": MAX_LOGICAL_NNZ + 1},
            )
            for overrides in resource_hostiles:
                arguments = dict(baseline_resources)
                arguments.update(overrides)
                try:
                    _resource_bounds(**arguments)
                except SentinelError:
                    pass
                else:
                    raise SentinelError("hostile: resource cap boundary+1 was accepted")
            selected_owned = _owned_readonly_i64(np.asarray([0, 2], dtype=np.int64))
            class Rows:
                rows = 2
                row_ids = _owned_readonly_i64(np.asarray([0, 2], dtype=np.int64))
                logical_nnz = 2
                data = np.asarray([1.0, 1.0], dtype=np.float64)
                def assert_intact(self) -> None:
                    return None
            import scipy.sparse as sp
            full = sp.csr_matrix(
                (np.asarray([1.0, 1.0]), np.asarray([0, 1], dtype=np.int32), np.asarray([0, 1, 2, 2], dtype=np.int32)),
                shape=(3, 2), dtype=np.float64,
            )
            full.sort_indices()
            input_good = np.asarray([0, 1], dtype=np.int64)
            keep_good = np.asarray([True, False, True], dtype=np.bool_)
            physical_good = [(1, 0, 4), (1, 1, 5), (2, 0, 6)]
            receipt = _exact_preflight(
                total_phases=3,
                selected_ordinals=selected_owned,
                input_rows=input_good,
                assert_width=2,
                base_rows=Rows(),
                full_rows=full,
                keep=keep_good,
                physical_rows=physical_good,
                deadline_monotonic=time.monotonic() + 5.0,
            )
            if receipt["passed_before_delta"] is not True:
                raise SentinelError("hostile: valid exact preflight failed")
            hostile_selected = np.asarray([0, 2], dtype=np.int64)
            try:
                _exact_preflight(
                    total_phases=3, selected_ordinals=hostile_selected,
                    input_rows=np.asarray([0, 1], dtype=np.int64), assert_width=2,
                    base_rows=Rows(), full_rows=full,
                    keep=np.asarray([True, False, True], dtype=np.bool_),
                    physical_rows=[(1, 0, 4), (1, 1, 5), (2, 0, 6)],
                    deadline_monotonic=time.monotonic() + 5.0,
                )
            except SentinelError:
                pass
            else:
                raise SentinelError("hostile: writable selected support passed preflight")
            structural_hostiles = (
                (_owned_readonly_i64(np.asarray([], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 1, 5), (2, 0, 6)], time.monotonic() + 5.0),
                (_owned_readonly_i64(np.asarray([2, 0], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 1, 5), (2, 0, 6)], time.monotonic() + 5.0),
                (_owned_readonly_i64(np.asarray([0, 0], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 1, 5), (2, 0, 6)], time.monotonic() + 5.0),
                (_owned_readonly_i64(np.asarray([3], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 1, 5), (2, 0, 6)], time.monotonic() + 5.0),
                (_owned_readonly_i64(np.asarray([1], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 1, 5), (2, 0, 6)], time.monotonic() + 5.0),
                (_owned_readonly_i64(np.asarray([0], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 0, 4), (2, 0, 6)], time.monotonic() + 5.0),
                (_owned_readonly_i64(np.asarray([0], dtype=np.int64)), np.asarray([True, False, True], dtype=np.bool_), [(1, 0, 4), (1, 1, 5), (2, 0, 6)], time.monotonic() - 1.0),
            )
            for bad_selected, bad_keep, bad_physical, bad_deadline in structural_hostiles:
                try:
                    _exact_preflight(
                        total_phases=3, selected_ordinals=bad_selected,
                        input_rows=np.asarray([0, 1], dtype=np.int64), assert_width=2,
                        base_rows=Rows(), full_rows=full, keep=bad_keep,
                        physical_rows=bad_physical,
                        deadline_monotonic=bad_deadline,
                    )
                except SentinelError:
                    pass
                else:
                    raise SentinelError("hostile: structural preflight corruption passed")
            try:
                selected_owned.setflags(write=True)
            except ValueError:
                pass
            else:
                raise SentinelError("hostile: immutable selected support allowed ABA mutation")

            bindings = {
                "total_phases": 3,
                "selected_ordinals": selected_owned,
                "input_rows": input_good,
                "assert_width": 2,
                "base_rows": Rows(),
                "full_rows": full,
                "keep": keep_good,
                "physical_rows": physical_good,
                "deadline_monotonic": time.monotonic() + 30.0,
            }
            bound_receipt = _exact_preflight(**bindings)
            repair_changes = (
                (1, 4, False, True),
                (2, 6, False, True),
            )
            real_schedule = patch.real_seal_delta
            real_helper = patch.real_build_repair

            def assert_mutation_stops_before_schedule(
                mutate: Any, restore: Any,
            ) -> None:
                calls = {"schedule": 0}
                def forbidden_schedule(*_args: Any, **_kwargs: Any) -> Any:
                    calls["schedule"] += 1
                    return None
                patch.real_seal_delta = forbidden_schedule
                patch._preflight_bindings = bindings
                patch._preflight_receipt = bound_receipt
                patch.audit["delta_schedule_calls"] = 1
                mutate()
                try:
                    patch._seal_delta_checked(None, {}, {}, repair_changes)
                except private.ExactReLUPhaseProjectionUnknown:
                    pass
                else:
                    raise SentinelError("hostile: mutated binding reached real schedule")
                finally:
                    restore()
                if calls["schedule"] != 0:
                    raise SentinelError("hostile: real schedule ran after binding mutation")

            assert_mutation_stops_before_schedule(
                lambda: keep_good.__setitem__(2, False),
                lambda: keep_good.__setitem__(2, True),
            )
            original_data = float(full.data[0])
            assert_mutation_stops_before_schedule(
                lambda: full.data.__setitem__(0, original_data + 1.0),
                lambda: full.data.__setitem__(0, original_data),
            )
            original_index = int(full.indices[0])
            assert_mutation_stops_before_schedule(
                lambda: full.indices.__setitem__(0, 1),
                lambda: full.indices.__setitem__(0, original_index),
            )
            assert_mutation_stops_before_schedule(
                lambda: physical_good.__setitem__(1, (9, 1, 5)),
                lambda: physical_good.__setitem__(1, (1, 1, 5)),
            )
            assert_mutation_stops_before_schedule(
                lambda: physical_good.__setitem__(0, (9, 0, 4)),
                lambda: physical_good.__setitem__(0, (1, 0, 4)),
            )
            assert_mutation_stops_before_schedule(
                lambda: input_good.__setitem__(1, 2),
                lambda: input_good.__setitem__(1, 1),
            )
            original_row_ids = bindings["base_rows"].row_ids
            assert_mutation_stops_before_schedule(
                lambda: setattr(bindings["base_rows"], "row_ids", _owned_readonly_i64(np.asarray([0, 1], dtype=np.int64))),
                lambda: setattr(bindings["base_rows"], "row_ids", original_row_ids),
            )

            helper_calls = {"count": 0}
            def forbidden_helper(**_kwargs: Any) -> Any:
                helper_calls["count"] += 1
                return None
            patch.real_build_repair = forbidden_helper
            keep_good[2] = False
            try:
                patch._build_repair_checked()
            except private.ExactReLUPhaseProjectionUnknown:
                pass
            else:
                raise SentinelError("hostile: mutated binding reached repair helper")
            finally:
                keep_good[2] = True
                patch.real_seal_delta = real_schedule
                patch.real_build_repair = real_helper
            if helper_calls["count"] != 0:
                raise SentinelError("hostile: real repair helper ran after binding mutation")

            helper_kwargs = {
                "full_oriented_rows": full,
                "keep": keep_good,
                "base_rows": bindings["base_rows"],
                "selected_ordinals": selected_owned,
            }
            patch._preflight_bindings = bindings
            patch._preflight_receipt = bound_receipt
            patch.real_build_repair = lambda **_kwargs: "ok"
            if patch._build_repair_checked(**helper_kwargs) != "ok":
                raise SentinelError("hostile: valid helper wrapper did not return result")
            if patch._preflight_bindings is not None:
                raise SentinelError("hostile: successful helper retained large bindings")
            def raising_helper(**_kwargs: Any) -> Any:
                raise MemoryError()
            patch._preflight_bindings = bindings
            patch._preflight_receipt = bound_receipt
            patch.real_build_repair = raising_helper
            try:
                patch._build_repair_checked(**helper_kwargs)
            except MemoryError:
                pass
            else:
                raise SentinelError("hostile: raising helper did not propagate")
            finally:
                patch.real_build_repair = real_helper
            if patch._preflight_bindings is not None:
                raise SentinelError("hostile: raising helper retained large bindings")
            owner_exist_false = {
                "logical_owner_instances": 1,
                "logical_owner_close_calls": 1,
                "logical_owner_final_states": ["CLOSED"],
                "native_owner_instances": 1,
                "native_run_calls": 1,
                "native_clear_calls": 1,
                "native_clear_model_calls": 0,
                "dual_ray_exist_calls": 1,
                "dual_ray_calls": 0,
            }
            _validate_owner_audit(owner_exist_false)
            missing_signal_rule = {
                "schema": "act.hybridz.structural_support_rule_audit.v1",
                "row_dual_consumed_by_selector": False,
                "opaque_signal_sign_or_magnitude_consumed_by_selector": False,
                "optimal_selector_calls": 0,
                "opaque_selector_calls": 0,
                "optimal_owner_abi_reads": 0,
                "opaque_owner_api_getDualRayExist_calls": 1,
                "opaque_owner_api_getDualRay_calls": 0,
                "delta_schedule_calls": 1,
                "selected_kind": "none",
                "preflight": None,
                "preflight_revalidated_at_repair_schedule": False,
                "preflight_revalidated_at_helper_entry": False,
                "preflight_bindings_released_at_helper_exit": False,
            }
            missing_signal_record = {
                "schema": "act.hybridz.structural_support.worker.v1",
                "status": "VerifyStatus.UNKNOWN",
                "has_counterexample": False,
                "last_completed_stage": "complete",
                "error_stage": None,
                "owner_audit": owner_exist_false,
                "rule_audit": missing_signal_rule,
                "phase_projection": {
                    "enabled": True,
                    "configured_seconds": REQUEST_SECONDS,
                    "status": "UNKNOWN",
                    "input_sampling_used": False,
                    "pgd_used": False,
                    "concrete_onnx_execution_used": False,
                    "bab_used": False,
                    "backward_used": False,
                    "dual_tightening_used": False,
                },
            }
            if _validate_record(missing_signal_record) != "UNKNOWN":
                raise SentinelError("hostile: legal missing signal did not remain UNKNOWN")
            forged_downstream = json.loads(_canonical_json(missing_signal_record))
            forged_downstream["rule_audit"]["opaque_selector_calls"] = 1
            forged_downstream["rule_audit"]["selected_kind"] = (
                "infeasible_all_opaque_exact_nonzero"
            )
            try:
                _validate_record(forged_downstream)
            except SentinelError:
                pass
            else:
                raise SentinelError("hostile: (1,0) missing signal forged downstream work")
            ray_without_exist = dict(owner_exist_false)
            ray_without_exist.update({"dual_ray_exist_calls": 0, "dual_ray_calls": 1})
            try:
                _validate_owner_audit(ray_without_exist)
            except SentinelError:
                pass
            else:
                raise SentinelError("hostile: getDualRay without existence call passed")
    finally:
        if sys.modules.get(CANONICAL_PHASE_NAME) is not canonical_before:
            raise SentinelError("hostile: canonical module was not restored")
        sys.modules.pop(PRIVATE_PHASE_NAME, None)
    if _source_bundle() != bundle_before:
        raise SentinelError("hostile: production source bundle changed")
    return {
        "private_transformed_phase_sha256": private_sha,
        "optimal_selector_signature_has_row_dual": False,
        "tight_upper_positive_5e10_covered": True,
        "tight_upper_negative_5e10_covered": True,
        "tight_just_over_threshold_rejected": True,
        "negative_residual_selected_without_extra_gate": True,
        "opaque_positive_negative_subnormal_selected": True,
        "opaque_positive_negative_zero_rejected": True,
        "opaque_sign_magnitude_invariance": True,
        "opaque_raw_abi_hostiles_rejected_without_coercion": True,
        "opaque_owner_each_api_at_most_once": True,
        "opaque_exist_false_is_legal_fail_closed_1_0": True,
        "opaque_exist_false_forbids_selector_preflight_repair_delta": True,
        "opaque_support_mapping_drift_rejected": True,
        "resource_cap_boundary_plus_one_rejected": True,
        "selected_unsorted_duplicate_range_keep_rejected": True,
        "expired_deadline_rejected": True,
        "physical_mapping_drift_rejected": True,
        "selected_immutable_ABA_rejected": True,
        "keep_csr_physical_input_base_ABA_rejected_before_schedule": True,
        "binding_mutation_rejected_before_helper_allocation": True,
        "preflight_bindings_released_on_helper_success_and_failure": True,
        "writable_selected_support_rejected": True,
        "production_bundle_unchanged": True,
        "gpu_called": False,
    }


def _static_check() -> dict[str, Any]:
    bundle = _source_bundle()
    for case in CASES:
        _resolve(case)
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if tuple(prereg["stage_a"]["case_order"]) != STAGE_A or tuple(prereg["stage_b"]["case_order"]) != STAGE_B:
        raise SentinelError("preregistered case order differs from harness")
    if len(CASES) != 16 or len({case.name for case in CASES}) != 16:
        raise SentinelError("case manifest is not sixteen unique cases")
    transformed, private_sha = _transformed_phase_source()
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden_imports = {"onnxruntime", "scipy.optimize", "random"}
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    if forbidden_imports & imports:
        raise SentinelError("harness imports a forbidden search/second-solver module")
    verify_calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "verify_once"
    ]
    if len(verify_calls) != 1:
        raise SentinelError("harness must have exactly one verify_once callsite")
    if any(path.exists() for path in (EVENTS_PATH, RAW_PATH, RECEIPTS_PATH, SUMMARY_PATH)):
        raise SentinelError("pre-GPU runtime output path is not absent")
    hostile = _cpu_hostile()
    if _source_bundle() != bundle:
        raise SentinelError("static/hostile check changed canonical production")
    return {
        "schema": "act.hybridz.structural_support.static_check.v1",
        "status": "PASS_PRE_GPU",
        "harness_sha256": _sha256(Path(__file__)),
        "preregistration_sha256": PREREG_SHA256,
        "canonical_source_bundle_sha256": _bundle_sha256(bundle),
        "private_transformed_phase_sha256": private_sha,
        "transformed_source_bytes": len(transformed.encode("utf-8")),
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
    if args.worker_case:
        return _worker_entry(args.worker_case)
    return _parent()


if __name__ == "__main__":
    raise SystemExit(main())
