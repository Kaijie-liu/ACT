#!/usr/bin/env python3
"""Fail-closed controlled promotion audit for query-dual V5.1.

The command constructs one deterministic synthetic residual network, obtains
one real :class:`QueryDualBoxCertificate`, and gives the exact same certified
bounds and query material to the frozen V3 transaction and the root-owned
V5.1 transaction.  It never accepts an ONNX/VNNLIB path and cannot produce a
solver verdict.

The public command has no performance knobs.  Its five-stage, four-thread,
five-pair configuration is sealed below.  Small profiles and parameter
injection are private test facilities and can only produce a non-official
receipt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from act.back_end.hybridz_tf import query_dual_replay as v3
from act.back_end.hybridz_tf import query_dual_replay_v51_session as v51_session
from act.back_end.hybridz_tf import query_dual_v51_authority as v51_authority
from act.back_end.hybridz_tf.query_dual_blas_contract import (
    QueryDualBlasContract,
    probe_query_dual_blas_contract,
    validate_query_dual_blas_contract,
)
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
    verify_query_dual_box_certificate,
)


SCHEMA = "act.query_dual_v51_controlled_audit.v1"
SEED = 20260728
EXPECTED_V3_SHA256 = (
    "6e291bdd4526518496e664c14e15664bf"
    "554c1e9f089d92f65f8097081db5d7e"
)
MIN_RELATED_TESTS = 104
_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024
_FALSE_ENV = frozenset({"0", "FALSE", "NO", "OFF"})
_HOST_CPU_SAMPLE_SECONDS = 0.25
_HOST_CPU_COMPETITOR_CORES = 0.50
_HOST_EXTERNAL_CPU_LIMIT_CORES = 0.50
_OFFICIAL_CPU_AFFINITY = (4, 5, 6, 7)


@dataclass(frozen=True)
class StageSpec:
    """One target/property use and its actual replay cone."""

    use_index: int
    target_relu_lid: Optional[int]
    cone_start_lid: Optional[int]
    objective_count: int


@dataclass(frozen=True)
class TopologyProfile:
    channels: int
    high_hw: int
    low_hw: int
    classes: int
    weight_scale: float


@dataclass(frozen=True)
class AuditParameters:
    """Private execution parameters; only one value is official."""

    stages: Tuple[StageSpec, ...]
    pairs: int
    blas_threads: int
    chunk_size: int
    workspace_bytes: int
    rss_limit_bytes: int
    bootstrap_samples: int
    session_timeout_s: float
    root_timeout_s: float


OFFICIAL_TOPOLOGY = TopologyProfile(
    channels=128,
    high_hw=8,
    low_hw=4,
    classes=100,
    weight_scale=0.025,
)
OFFICIAL_PARAMETERS = AuditParameters(
    stages=(
        StageSpec(0, 3, 2, 32),
        StageSpec(1, 6, 5, 16),
        StageSpec(2, 10, 9, 48),
        StageSpec(3, 15, 14, 32),
        StageSpec(4, None, None, 99),
    ),
    pairs=5,
    blas_threads=4,
    chunk_size=64,
    workspace_bytes=512 * _MIB,
    rss_limit_bytes=2 * _GIB,
    bootstrap_samples=20_000,
    session_timeout_s=300.0,
    root_timeout_s=300.0,
)

# This literal is filled from the canonical records, rather than accepted from
# a command-line value.  A source change changes the source manifest as well.
OFFICIAL_CONFIGURATION_SHA256 = (
    "1e9505d76376bad2cf5c4ba5e9d1972"
    "b9313ff738ba6a8a26f961b84c233ab82"
)
_REQUIRED_PROMOTION_GATE_KEYS = (
    "environment_passed",
    "host_initial_passed",
    "host_pre_timing_passed",
    "host_post_timing_passed",
    "warmup_and_timing_external_cpu_passed",
    "per_timed_implementation_external_cpu_passed",
    "unittest_fraction_passed",
    "pair_count_passed",
    "median_speedup_passed",
    "bootstrap_95_lower_passed",
    "tightness_passed",
    "rss_passed",
    "workspace_passed",
    "device_ast_passed",
    "live_blas_probe_and_commit_rechecks_passed",
    "same_root_certified_bounds_passed",
    "query_alpha_material_unchanged",
    "source_hashes_stable",
)

TEST_MODULES = (
    "act.back_end.hybridz_tf.test_query_dual_blas_contract",
    "act.back_end.hybridz_tf.test_query_dual_scalar_guard_v51",
    "act.back_end.hybridz_tf.test_query_dual_replay_v51_conv",
    "act.back_end.hybridz_tf.test_query_dual_replay_v51",
    "act.back_end.hybridz_tf.test_query_dual_v51_authority",
    "act.back_end.hybridz_tf.test_query_dual_replay_v51_session",
    "act.pipeline.verification.test_query_dual_v51_controlled_audit",
)
TEST_MODULE_MINIMUMS = MappingProxyType(
    {
        TEST_MODULES[0]: 3,
        TEST_MODULES[1]: 11,
        TEST_MODULES[2]: 10,
        TEST_MODULES[3]: 13,
        TEST_MODULES[4]: 28,
        TEST_MODULES[5]: 23,
        TEST_MODULES[6]: 16,
    }
)
FRACTION_TESTS = MappingProxyType(
    {
        "dense": {
            "module": TEST_MODULES[1],
            "class": "DenseScalarGuardV51Tests",
            "method": "test_fixed_5000_fraction_rows_and_v3_tightness",
            "minimum_rows": 5_000,
        },
        "conv": {
            "module": TEST_MODULES[2],
            "class": "QueryDualReplayV51ConvTests",
            "method": "test_5000_fraction_query_rows_and_v3_tightness_gate",
            "minimum_rows": 5_000,
        },
    }
)

NUMERIC_SOURCE_PATHS = (
    "act/back_end/hybridz_tf/query_dual_box_certifier.py",
    "act/back_end/hybridz_tf/query_dual_replay.py",
    "act/back_end/hybridz_tf/query_dual_scalar_guard.py",
    "act/back_end/hybridz_tf/query_dual_scalar_guard_v51.py",
    "act/back_end/hybridz_tf/query_dual_replay_v51_conv.py",
    "act/back_end/hybridz_tf/query_dual_replay_v51.py",
    "act/back_end/hybridz_tf/query_dual_v51_authority.py",
    "act/back_end/hybridz_tf/query_dual_blas_contract.py",
    "act/back_end/hybridz_tf/query_dual_replay_v51_session.py",
)
EXPECTED_NUMERIC_SOURCE_SHA256 = MappingProxyType(
    {
        "act/back_end/hybridz_tf/query_dual_box_certifier.py": (
            "c282f22e3510bd8427daa914cd85b8a89"
            "a36974f85e283e6b04048fda2ac0708"
        ),
        "act/back_end/hybridz_tf/query_dual_replay.py": (
            "6e291bdd4526518496e664c14e15664bf"
            "554c1e9f089d92f65f8097081db5d7e"
        ),
        "act/back_end/hybridz_tf/query_dual_scalar_guard.py": (
            "1c466f511f52f8f93e79bc6707d33495"
            "bf186f7958ea3fafbe5edfb8b50ed4f9"
        ),
        "act/back_end/hybridz_tf/query_dual_scalar_guard_v51.py": (
            "dad748771e9bf8ea7c4db8fb0a163a97"
            "5cfe3b3e5e4f6242b117a77d940b8e44"
        ),
        "act/back_end/hybridz_tf/query_dual_replay_v51_conv.py": (
            "cc58681fafd2a2a4827711164b894962f"
            "96b7931f639ffebeb03c074f0d97b56"
        ),
        "act/back_end/hybridz_tf/query_dual_replay_v51.py": (
            "bbf2f3ebcedcfe1c9e4d4bf8d5f2930"
            "4f06243b3c9738b1c26093c59916538ac"
        ),
        "act/back_end/hybridz_tf/query_dual_v51_authority.py": (
            "99b42d5275046e5f1195fe799409de9a"
            "69df21c416b4d15fe110e6a87f94f4a8"
        ),
        "act/back_end/hybridz_tf/query_dual_blas_contract.py": (
            "d2bc8974ebda1f8788dc3940e16aaa39"
            "5b8fc6e4beb4cb270d4ca0493d061d90"
        ),
        "act/back_end/hybridz_tf/query_dual_replay_v51_session.py": (
            "5b81f096c1d8279bae88e62f91bcdcd7"
            "8d8efa640227361ef0e7ed676ec6c172"
        ),
    }
)
SOURCE_PATHS = NUMERIC_SOURCE_PATHS + tuple(
    f"{module.replace('.', '/')}.py" for module in TEST_MODULES
) + (
    "act/pipeline/verification/query_dual_v51_controlled_audit.py",
)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0])
            )
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "<f8", "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_bits(value: np.ndarray) -> Tuple[str, ...]:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return tuple(f"{int(item):016x}" for item in array.view(np.uint64))


def _configuration_record(
    parameters: AuditParameters,
    topology: TopologyProfile,
) -> Mapping[str, Any]:
    return {
        "parameters": asdict(parameters),
        "topology": asdict(topology),
        "stage_semantics": (
            "target ReLU preactivation: both V3 and V5.1 replay from the "
            "target ReLU predecessor cone"
        ),
        "host_cpu_accounting": {
            "short_sample_seconds": _HOST_CPU_SAMPLE_SECONDS,
            "individual_process_limit_core_equivalents": (
                _HOST_CPU_COMPETITOR_CORES
            ),
            "individual_process_scope": (
                "all readable non-self PIDs; invocation ancestors excluded "
                "only from short host samples"
            ),
            "aggregate_external_limit_core_equivalents": (
                _HOST_EXTERNAL_CPU_LIMIT_CORES
            ),
            "aggregate_comparison": "strictly_less_than",
            "busy_fields": [
                "user",
                "nice",
                "system",
                "irq",
                "softirq",
                "steal",
            ],
            "excluded_fields": ["idle", "iowait"],
            "guest_policy": "already_in_user_nice_do_not_add",
            "cpu_scope": "current_affinity_cpuN_rows",
            "required_cpu_affinity": list(_OFFICIAL_CPU_AFFINITY),
            "timing_scope": (
                "every warmup, every timed implementation, and the "
                "complete warmup-plus-timing window"
            ),
        },
    }


def _assert_official_configuration(
    parameters: AuditParameters,
    topology: TopologyProfile,
) -> None:
    if parameters != OFFICIAL_PARAMETERS or topology != OFFICIAL_TOPOLOGY:
        raise RuntimeError("official V5.1 configuration was substituted")
    actual = _json_sha256(_configuration_record(parameters, topology))
    if not hmac.compare_digest(actual, OFFICIAL_CONFIGURATION_SHA256):
        raise RuntimeError("official V5.1 configuration seal changed")
    if (
        parameters.blas_threads != 4
        or parameters.pairs < 5
        or parameters.chunk_size != 64
        or parameters.workspace_bytes != 512 * _MIB
        or sum(stage.objective_count for stage in parameters.stages) != 227
        or tuple(stage.objective_count for stage in parameters.stages)
        != (32, 16, 48, 32, 99)
        or tuple(stage.cone_start_lid for stage in parameters.stages)
        != (2, 5, 9, 14, None)
        or tuple(stage.target_relu_lid for stage in parameters.stages)
        != (3, 6, 10, 15, None)
        or _HOST_CPU_SAMPLE_SECONDS != 0.25
        or _HOST_CPU_COMPETITOR_CORES != 0.50
        or _HOST_EXTERNAL_CPU_LIMIT_CORES != 0.50
        or _OFFICIAL_CPU_AFFINITY != (4, 5, 6, 7)
    ):
        raise RuntimeError("official V5.1 invariants changed")


def _layer(layer_id: int, kind: str, width: int, params=None) -> Any:
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[
            int(layer_id) * 1_000_000 + index
            for index in range(int(width))
        ],
        cache={},
    )


def _network(layers: Sequence[Any], predecessors: Mapping[int, Sequence[int]]):
    pred_map = {
        int(layer.id): tuple(
            int(parent) for parent in predecessors[int(layer.id)]
        )
        for layer in layers
    }
    successors = {int(layer.id): [] for layer in layers}
    for child, parents in pred_map.items():
        for parent in parents:
            successors[parent].append(child)
    return SimpleNamespace(
        layers=list(layers),
        preds=pred_map,
        succs=successors,
        by_id={int(layer.id): layer for layer in layers},
    )


def _conv(
    rng: np.random.Generator,
    layer_id: int,
    *,
    channels: int,
    input_hw: int,
    output_hw: int,
    kernel: int,
    stride: int,
    padding: int,
    weight_scale: float,
) -> Any:
    weight = np.ascontiguousarray(
        rng.normal(
            0.0,
            weight_scale,
            size=(channels, channels, kernel, kernel),
        ),
        dtype=np.float64,
    )
    bias = np.ascontiguousarray(
        rng.normal(0.0, weight_scale / 5.0, size=channels),
        dtype=np.float64,
    )
    return _layer(
        layer_id,
        "CONV2D",
        channels * output_hw * output_hw,
        {
            "weight": weight,
            "bias": bias,
            "in_channels": channels,
            "out_channels": channels,
            "kernel_size": (kernel, kernel),
            "stride": (stride, stride),
            "padding": (padding, padding),
            "dilation": (1, 1),
            "groups": 1,
            "padding_mode": "zeros",
            "input_shape": (1, channels, input_hw, input_hw),
            "output_shape": (1, channels, output_hw, output_hw),
        },
    )


def _build_topology(
    profile: TopologyProfile = OFFICIAL_TOPOLOGY,
    *,
    seed: int = SEED,
) -> Any:
    """Build the unchanged residual topology with a real BOX input."""

    if (
        profile.channels <= 0
        or profile.high_hw <= 1
        or profile.low_hw <= 0
        or profile.classes <= 0
        or profile.low_hw * 2 != profile.high_hw
        or not math.isfinite(profile.weight_scale)
        or profile.weight_scale <= 0.0
    ):
        raise ValueError("invalid controlled topology profile")
    rng = np.random.default_rng(seed)
    channels = profile.channels
    high = profile.high_hw
    low = profile.low_hw
    high_width = channels * high * high
    low_width = channels * low * low
    input_lower = np.full((1, high_width), -1.0, dtype=np.float64)
    input_upper = np.full((1, high_width), 1.0, dtype=np.float64)
    input_layer = _layer(
        0,
        "INPUT",
        high_width,
        {
            "shape": (1, channels, high, high),
            "dtype": "torch.float64",
        },
    )
    input_spec = _layer(
        1,
        "INPUT_SPEC",
        high_width,
        {
            "kind": "BOX",
            "lb": input_lower,
            "ub": input_upper,
        },
    )
    conv2 = _conv(
        rng,
        2,
        channels=channels,
        input_hw=high,
        output_hw=high,
        kernel=3,
        stride=1,
        padding=1,
        weight_scale=profile.weight_scale,
    )
    relu3 = _layer(3, "RELU", high_width)
    conv4 = _conv(
        rng,
        4,
        channels=channels,
        input_hw=high,
        output_hw=high,
        kernel=3,
        stride=1,
        padding=1,
        weight_scale=profile.weight_scale,
    )
    add5 = _layer(5, "ADD", high_width)
    relu6 = _layer(6, "RELU", high_width)
    main7 = _conv(
        rng,
        7,
        channels=channels,
        input_hw=high,
        output_hw=low,
        kernel=3,
        stride=2,
        padding=1,
        weight_scale=profile.weight_scale,
    )
    skip8 = _conv(
        rng,
        8,
        channels=channels,
        input_hw=high,
        output_hw=low,
        kernel=1,
        stride=2,
        padding=0,
        weight_scale=profile.weight_scale,
    )
    add9 = _layer(9, "ADD", low_width)
    relu10 = _layer(10, "RELU", low_width)
    conv11 = _conv(
        rng,
        11,
        channels=channels,
        input_hw=low,
        output_hw=low,
        kernel=3,
        stride=1,
        padding=1,
        weight_scale=profile.weight_scale,
    )
    relu12 = _layer(12, "RELU", low_width)
    conv13 = _conv(
        rng,
        13,
        channels=channels,
        input_hw=low,
        output_hw=low,
        kernel=3,
        stride=1,
        padding=1,
        weight_scale=profile.weight_scale,
    )
    add14 = _layer(14, "ADD", low_width)
    relu15 = _layer(15, "RELU", low_width)
    flatten16 = _layer(
        16,
        "FLATTEN",
        low_width,
        {"start_dim": 1, "end_dim": -1},
    )
    dense17 = _layer(
        17,
        "DENSE",
        profile.classes,
        {
            "weight": np.ascontiguousarray(
                rng.normal(
                    0.0,
                    profile.weight_scale,
                    size=(profile.classes, low_width),
                ),
                dtype=np.float64,
            ),
            "bias": np.ascontiguousarray(
                rng.normal(
                    0.0,
                    profile.weight_scale / 5.0,
                    size=profile.classes,
                ),
                dtype=np.float64,
            ),
            "in_features": low_width,
            "out_features": profile.classes,
        },
    )
    assertion18 = _layer(18, "ASSERT", profile.classes, {"kind": "AUDIT"})
    layers = (
        input_layer,
        input_spec,
        conv2,
        relu3,
        conv4,
        add5,
        relu6,
        main7,
        skip8,
        add9,
        relu10,
        conv11,
        relu12,
        conv13,
        add14,
        relu15,
        flatten16,
        dense17,
        assertion18,
    )
    predecessors = {
        0: (),
        1: (0,),
        2: (1,),
        3: (2,),
        4: (3,),
        5: (1, 4),
        6: (5,),
        7: (6,),
        8: (6,),
        9: (7, 8),
        10: (9,),
        11: (10,),
        12: (11,),
        13: (12,),
        14: (10, 13),
        15: (14,),
        16: (15,),
        17: (16,),
        18: (17,),
    }
    return _network(layers, predecessors)


def _ancestor_relus(net: Any, start_lid: Optional[int]) -> Tuple[int, ...]:
    if start_lid is None:
        assertion = next(
            layer for layer in net.layers if layer.kind == "ASSERT"
        )
        root = int(net.preds[int(assertion.id)][0])
    else:
        root = int(start_lid)
    seen: set[int] = set()
    relus: list[int] = []

    def visit(layer_id: int) -> None:
        if layer_id in seen:
            return
        seen.add(layer_id)
        layer = net.by_id[layer_id]
        if layer.kind == "RELU":
            relus.append(layer_id)
        for parent in net.preds[layer_id]:
            visit(parent)

    visit(root)
    return tuple(sorted(relus))


def _query_schedule(
    net: Any,
    parameters: AuditParameters,
    *,
    seed: int = SEED + 1,
) -> Tuple[Mapping[str, Any], ...]:
    rng = np.random.default_rng(seed)
    stages = []
    for spec in parameters.stages:
        if spec.cone_start_lid is None:
            assertion = next(
                layer for layer in net.layers if layer.kind == "ASSERT"
            )
            output_id = int(net.preds[int(assertion.id)][0])
            width = len(net.by_id[output_id].out_vars)
        else:
            width = len(net.by_id[spec.cone_start_lid].out_vars)
        mutable_rows = np.ascontiguousarray(
            rng.normal(
                0.0, 0.125, size=(spec.objective_count, width)
            ),
            dtype=np.float64,
        )
        # Keep every production row dense without changing the fixed RNG draw.
        mutable_rows[mutable_rows == 0.0] = np.nextafter(
            np.float64(0.0), np.float64(1.0)
        )
        rows = np.frombuffer(
            mutable_rows.tobytes(order="C"), dtype=np.float64
        ).reshape(mutable_rows.shape)
        alpha = MappingProxyType(
            {
                lid: np.frombuffer(
                    np.float64(0.5).tobytes(), dtype=np.float64
                ).reshape(())
                for lid in _ancestor_relus(net, spec.cone_start_lid)
            }
        )
        stages.append(
            MappingProxyType(
                {
                    "use_index": spec.use_index,
                    "target_relu_lid": spec.target_relu_lid,
                    "cone_start_lid": spec.cone_start_lid,
                    "objective_count": spec.objective_count,
                    "query_rows": rows,
                    "alpha": alpha,
                    "query_rows_sha256": _array_sha256(rows),
                    "alpha_sha256": _alpha_sha256(alpha),
                }
            )
        )
    return tuple(stages)


def _alpha_sha256(alpha: Mapping[int, np.ndarray]) -> str:
    manifest = [
        {
            "relu_lid": int(lid),
            "value_sha256": _array_sha256(np.asarray(value)),
        }
        for lid, value in sorted(alpha.items())
    ]
    return _json_sha256(manifest)


def _stage_material_seals(
    stages: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    seals = []
    for expected_index, stage in enumerate(stages):
        rows = np.asarray(stage["query_rows"])
        alpha = stage["alpha"]
        if (
            int(stage["use_index"]) != expected_index
            or type(rows) is not np.ndarray
            or rows.dtype != np.float64
            or rows.ndim != 2
            or rows.flags.writeable
            or not np.all(np.isfinite(rows))
            or not isinstance(alpha, Mapping)
            or any(
                type(np.asarray(value)) is not np.ndarray
                or np.asarray(value).dtype != np.float64
                or np.asarray(value).shape != ()
                or np.asarray(value).flags.writeable
                or not np.isfinite(np.asarray(value).item())
                for value in alpha.values()
            )
        ):
            raise RuntimeError("query/alpha material is not sealed CPU f64")
        query_sha = _array_sha256(rows)
        alpha_sha = _alpha_sha256(alpha)
        if (
            query_sha != stage["query_rows_sha256"]
            or alpha_sha != stage["alpha_sha256"]
        ):
            raise RuntimeError("query/alpha material binding changed")
        seals.append(
            {
                "use_index": expected_index,
                "query_rows_sha256": query_sha,
                "alpha_sha256": alpha_sha,
            }
        )
    return tuple(seals)


def _stage_uses(
    parameters: AuditParameters,
) -> Tuple[v51_authority.StageUse, ...]:
    uses = []
    for spec in parameters.stages:
        if spec.target_relu_lid is None:
            uses.append(
                v51_authority.StageUse(
                    use_index=spec.use_index,
                    stage_kind=v51_authority.STAGE_PROPERTY,
                    stage_index=None,
                    target_relu_lid=None,
                    cone_start_lid=None,
                )
            )
        else:
            uses.append(
                v51_authority.StageUse(
                    use_index=spec.use_index,
                    stage_kind=v51_authority.STAGE_TARGET,
                    stage_index=spec.use_index,
                    target_relu_lid=spec.target_relu_lid,
                    cone_start_lid=spec.cone_start_lid,
                )
            )
    return tuple(uses)


def _rss_bytes(field: str = "VmRSS") -> int:
    status = Path("/proc/self/status").read_text(encoding="ascii")
    match = re.search(
        rf"^{re.escape(field)}:\s+(\d+)\s+kB$", status, re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"/proc/self/status lacks {field}")
    return int(match.group(1)) * 1024


def _maximum_layer_width(net: Any) -> int:
    return max(len(layer.out_vars) for layer in net.layers)


def _workspace_record(
    net: Any,
    stage: Mapping[str, Any],
    parameters: AuditParameters,
) -> Mapping[str, int]:
    bytes_per_query = max(1, _maximum_layer_width(net) * 8 * 12)
    memory_limited = max(1, parameters.workspace_bytes // bytes_per_query)
    effective = min(
        parameters.chunk_size,
        memory_limited,
        int(stage["objective_count"]),
    )
    return {
        "requested_limit_bytes": parameters.workspace_bytes,
        "model_bytes_per_query": bytes_per_query,
        "effective_chunk_size": effective,
        "modeled_live_chunk_upper_bytes": effective * bytes_per_query,
    }


def _stage_record(
    stage: Mapping[str, Any],
    values: np.ndarray,
    seconds: float,
    workspace: Mapping[str, int],
    *,
    receipt_sha256: str,
) -> Mapping[str, Any]:
    array = np.asarray(values)
    if (
        type(array) is not np.ndarray
        or array.dtype != np.float64
        or array.ndim != 1
        or array.size != int(stage["objective_count"])
        or not np.all(np.isfinite(array))
    ):
        raise RuntimeError("controlled replay returned a non-CPU-f64 result")
    return {
        "use_index": int(stage["use_index"]),
        "target_relu_lid": stage["target_relu_lid"],
        "cone_start_lid": stage["cone_start_lid"],
        "objective_count": int(array.size),
        "seconds": float(seconds),
        "query_rows_sha256": str(stage["query_rows_sha256"]),
        "alpha_sha256": str(stage["alpha_sha256"]),
        "lower_bounds_sha256": _array_sha256(array),
        "lower_bounds_hex": [float(item).hex() for item in array],
        "lower_bounds_u64_hex": list(_array_bits(array)),
        "result_receipt_sha256": str(receipt_sha256),
        "workspace": dict(workspace),
    }


def _run_v3_schedule(
    net: Any,
    root_certificate: Any,
    stages: Sequence[Mapping[str, Any]],
    parameters: AuditParameters,
) -> Mapping[str, Any]:
    """Run all five V3 uses in one root-owned transaction and one frame."""

    total_start = time.perf_counter()
    deadline = time.monotonic() + parameters.session_timeout_s
    session = v3.create_query_dual_replay_session(
        net,
        root_certificate,
        tuple(stage["cone_start_lid"] for stage in stages),
        deadline=deadline,
    )
    try:
        setup_done = time.perf_counter()
        frame_start = setup_done
        frame = session.seal_bounds(
            root_certificate.bounds,
            start_lids=tuple(
                stage["cone_start_lid"] for stage in stages
            ),
        )
        frame_done = time.perf_counter()
        pending = []
        stage_seconds = []
        maximum_rss = _rss_bytes()
        maximum_hwm = _rss_bytes("VmHWM")
        for stage in stages:
            stage_start = time.perf_counter()
            pending.append(
                session.replay(
                    frame,
                    start_lid=stage["cone_start_lid"],
                    query_rows=stage["query_rows"],
                    alpha_by_relu=stage["alpha"],
                    chunk_size=parameters.chunk_size,
                    max_workspace_bytes=parameters.workspace_bytes,
                )
            )
            stage_seconds.append(time.perf_counter() - stage_start)
            maximum_rss = max(maximum_rss, _rss_bytes())
            maximum_hwm = max(maximum_hwm, _rss_bytes("VmHWM"))
        commit_start = time.perf_counter()
        committed = session.commit()
        commit_seconds = time.perf_counter() - commit_start
        total_seconds = time.perf_counter() - total_start
        maximum_rss = max(maximum_rss, _rss_bytes())
        maximum_hwm = max(maximum_hwm, _rss_bytes("VmHWM"))
    except Exception:
        session.abort()
        raise
    if len(committed) != len(stages):
        raise RuntimeError("V3 full-session result coverage changed")
    records = []
    arrays = []
    for stage, elapsed, result in zip(stages, stage_seconds, committed):
        if not v3.validate_query_dual_replay_result(result):
            raise RuntimeError("V3 committed result failed validation")
        values = np.asarray(result.lower_bounds)
        arrays.append(values.copy())
        records.append(
            _stage_record(
                stage,
                values,
                elapsed,
                _workspace_record(net, stage, parameters),
                receipt_sha256=str(
                    result.receipt["receipt_sha256"]
                ),
            )
        )
    return {
        "implementation": "v3_root_owned_session",
        "total_seconds": total_seconds,
        "session_setup_seconds": setup_done - total_start,
        "frame_seal_seconds": frame_done - frame_start,
        "commit_seconds": commit_seconds,
        "single_root_certificate": True,
        "single_bounds_frame": True,
        "same_certified_bounds_sha256": str(
            root_certificate.receipt["hashes"]["bounds_sha256"]
        ),
        "stages": records,
        "arrays": tuple(arrays),
        "maximum_rss_bytes": maximum_rss,
        "maximum_hwm_bytes": maximum_hwm,
    }


def _run_v51_schedule(
    net: Any,
    root_certificate: Any,
    stages: Sequence[Mapping[str, Any]],
    parameters: AuditParameters,
    blas_contract: QueryDualBlasContract,
) -> Mapping[str, Any]:
    """Run the required V5.1 root/session/frame/ledger/commit path."""

    total_start = time.perf_counter()
    deadline = time.monotonic() + parameters.session_timeout_s
    session = v51_session.create_query_dual_replay_v51_session(
        net,
        root_certificate,
        _stage_uses(parameters),
        deadline=deadline,
        blas_contract=blas_contract,
    )
    try:
        setup_done = time.perf_counter()
        frame_start = setup_done
        frame = session.seal_bounds(root_certificate.bounds)
        frame_sha256 = frame.frame_content_sha256
        frame_done = time.perf_counter()
        pending = []
        stage_seconds = []
        maximum_rss = _rss_bytes()
        maximum_hwm = _rss_bytes("VmHWM")
        for expected_index, stage in enumerate(stages):
            if int(stage["use_index"]) != expected_index:
                raise RuntimeError("V5.1 stage order changed")
            stage_start = time.perf_counter()
            pending.append(
                session.replay(
                    frame,
                    stage_use_index=expected_index,
                    query_rows=stage["query_rows"],
                    alpha_by_relu=stage["alpha"],
                    chunk_size=parameters.chunk_size,
                    max_workspace_bytes=parameters.workspace_bytes,
                )
            )
            stage_seconds.append(time.perf_counter() - stage_start)
            maximum_rss = max(maximum_rss, _rss_bytes())
            maximum_hwm = max(maximum_hwm, _rss_bytes("VmHWM"))
        catalog_entries_before_commit = frame.catalog_entry_count
        commit_start = time.perf_counter()
        committed = session.commit()
        commit_seconds = time.perf_counter() - commit_start
        total_seconds = time.perf_counter() - total_start
        maximum_rss = max(maximum_rss, _rss_bytes())
        maximum_hwm = max(maximum_hwm, _rss_bytes("VmHWM"))
    except Exception:
        session.abort()
        raise
    if len(committed) != 5 or len(committed) != len(stages):
        raise RuntimeError(
            "V5.1 full-session commit did not publish stages 0..4"
        )
    records = []
    arrays = []
    for stage, elapsed, result in zip(stages, stage_seconds, committed):
        if not v51_session.validate_query_dual_replay_v51_session_candidate(
            result
        ):
            raise RuntimeError("V5.1 committed candidate failed validation")
        if (
            result.proof_authority is not False
            or result.receipt["blas_contract_sha256"]
            != blas_contract.content_sha256
            or result.receipt["frame_content_sha256"] != frame_sha256
        ):
            raise RuntimeError("V5.1 commit/platform/frame binding changed")
        values = np.asarray(result.lower_bounds)
        arrays.append(values.copy())
        records.append(
            _stage_record(
                stage,
                values,
                elapsed,
                _workspace_record(net, stage, parameters),
                receipt_sha256=str(
                    result.receipt["receipt_sha256"]
                ),
            )
        )
    return {
        "implementation": "v51_root_owned_full_session_candidate",
        "proof_authority": False,
        "total_seconds": total_seconds,
        "session_setup_seconds": setup_done - total_start,
        "frame_seal_seconds": frame_done - frame_start,
        "commit_seconds": commit_seconds,
        "commit_live_blas_recheck_bound": True,
        "stage_uses_committed_once_in_order": True,
        "single_root_certificate": True,
        "single_bounds_frame": True,
        "frame_content_sha256": frame_sha256,
        "catalog_entries_before_commit": catalog_entries_before_commit,
        "same_certified_bounds_sha256": str(
            root_certificate.receipt["hashes"]["bounds_sha256"]
        ),
        "stages": records,
        "arrays": tuple(arrays),
        "maximum_rss_bytes": maximum_rss,
        "maximum_hwm_bytes": maximum_hwm,
    }


def _public_run_record(run: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in run.items() if key != "arrays"}


def _compare_runs(
    v3_run: Mapping[str, Any],
    v51_run: Mapping[str, Any],
) -> Mapping[str, Any]:
    old_arrays = tuple(v3_run["arrays"])
    new_arrays = tuple(v51_run["arrays"])
    if len(old_arrays) != 5 or len(new_arrays) != 5:
        raise RuntimeError("controlled comparison lacks five stages")
    regressions = []
    objective_count = 0
    equal_count = 0
    improved_count = 0
    for stage_index, (old_raw, new_raw) in enumerate(
        zip(old_arrays, new_arrays)
    ):
        old = np.asarray(old_raw)
        new = np.asarray(new_raw)
        if (
            old.dtype != np.float64
            or new.dtype != np.float64
            or old.shape != new.shape
            or not np.all(np.isfinite(old))
            or not np.all(np.isfinite(new))
        ):
            raise RuntimeError("controlled lower-bound arrays are incompatible")
        objective_count += int(old.size)
        mask = new < old
        for query_index in np.flatnonzero(mask):
            regressions.append(
                {
                    "stage_index": stage_index,
                    "query_index": int(query_index),
                    "v3_hex": float(old[query_index]).hex(),
                    "v51_hex": float(new[query_index]).hex(),
                    "v3_u64_hex": _array_bits(old[query_index : query_index + 1])[0],
                    "v51_u64_hex": _array_bits(
                        new[query_index : query_index + 1]
                    )[0],
                }
            )
        equal_count += int(np.count_nonzero(new == old))
        improved_count += int(np.count_nonzero(new > old))
    return {
        "objective_count": objective_count,
        "tightness_regression_count": len(regressions),
        "equal_count": equal_count,
        "improved_count": improved_count,
        "regression_preview": regressions[:16],
        "all_lower_bound_bits_recorded": True,
    }


def _bootstrap_lower(
    v3_seconds: Sequence[float],
    v51_seconds: Sequence[float],
    *,
    samples: int,
) -> float:
    old = np.asarray(v3_seconds, dtype=np.float64)
    new = np.asarray(v51_seconds, dtype=np.float64)
    if (
        old.shape != new.shape
        or old.ndim != 1
        or old.size < 5
        or isinstance(samples, bool)
        or samples <= 0
        or not np.all(np.isfinite(old))
        or not np.all(np.isfinite(new))
        or np.any(old <= 0.0)
        or np.any(new <= 0.0)
    ):
        raise RuntimeError("invalid five-pair bootstrap input")
    rng = np.random.default_rng(SEED + 2)
    indices = rng.integers(0, old.size, size=(samples, old.size))
    ratios = np.median(old[indices], axis=1) / np.median(
        new[indices], axis=1
    )
    return float(np.quantile(ratios, 0.025, method="lower"))


def _source_hashes(
    project_root: Path,
    relative_paths: Iterable[str] = SOURCE_PATHS,
) -> Mapping[str, str]:
    result = {}
    for relative in relative_paths:
        path = project_root / relative
        if not path.is_file():
            raise RuntimeError(f"required source is missing: {relative}")
        result[relative] = _file_sha256(path)
    return result


def _validate_expected_numeric_sources(
    observed: Mapping[str, str],
) -> Mapping[str, Any]:
    mismatches = {
        relative: {
            "expected": expected,
            "observed": observed.get(relative),
        }
        for relative, expected in EXPECTED_NUMERIC_SOURCE_SHA256.items()
        if observed.get(relative) != expected
    }
    if mismatches:
        raise RuntimeError(
            "frozen V5.1 numeric source closure changed: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "expected_numeric_source_sha256": dict(
            EXPECTED_NUMERIC_SOURCE_SHA256
        ),
        "observed_numeric_source_sha256": {
            relative: observed[relative]
            for relative in EXPECTED_NUMERIC_SOURCE_SHA256
        },
        "passed": True,
    }


def _dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return None if parent is None else f"{parent}.{node.attr}"
    return None


def _literal_contains_cuda(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "cuda" in node.value.lower()
    )


def _numeric_device_audit(
    project_root: Path,
    *,
    relative_paths: Sequence[str] = NUMERIC_SOURCE_PATHS,
) -> Mapping[str, Any]:
    """AST-audit the complete local numeric closure for GPU compute calls."""

    forbidden_roots = {"cupy", "jax", "tensorflow"}
    findings = []
    sources = []
    for relative in relative_paths:
        path = project_root / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        sources.append(
            {"path": relative, "sha256": hashlib.sha256(source.encode()).hexdigest()}
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in forbidden_roots:
                        findings.append(
                            {
                                "path": relative,
                                "line": node.lineno,
                                "kind": "forbidden_import",
                                "value": alias.name,
                            }
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".", 1)[0] in forbidden_roots:
                    findings.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "kind": "forbidden_import",
                            "value": node.module,
                        }
                    )
            if isinstance(node, ast.Call):
                name = _dotted_name(node.func) or ""
                if (
                    name == "torch.cuda"
                    or name.startswith("torch.cuda.")
                    or name.endswith(".cuda")
                ):
                    findings.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "kind": "cuda_call",
                            "value": name,
                        }
                    )
                if name.endswith(".to") and any(
                    _literal_contains_cuda(argument)
                    for argument in node.args
                ):
                    findings.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "kind": "cuda_device_transfer",
                            "value": name,
                        }
                    )
                if any(
                    keyword.arg == "device"
                    and _literal_contains_cuda(keyword.value)
                    for keyword in node.keywords
                ):
                    findings.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "kind": "cuda_device_keyword",
                            "value": name,
                        }
                    )
    return {
        "method": "AST audit of complete V3/V5.1 numeric source closure",
        "numeric_sources": sources,
        "forbidden_backend_roots": sorted(forbidden_roots),
        "forbidden_gpu_compute_findings": findings,
        "runtime_array_backend_required": "exact numpy.ndarray binary64",
        "ambient_torch_cuda_is_not_execution_evidence": True,
        "passed": not findings,
    }


def _operator_solver_usage_audit(project_root: Path) -> Mapping[str, Any]:
    """Distinguish package-bootstrap imports from audit-path solver use."""

    relative_paths = NUMERIC_SOURCE_PATHS + (
        "act/pipeline/verification/query_dual_v51_controlled_audit.py",
    )
    direct_imports = []
    direct_calls = []

    def is_solver_name(name: str) -> bool:
        lowered = name.lower()
        return bool(
            lowered == "gurobipy"
            or lowered.startswith("gurobipy.")
            or ".solver." in lowered
            or lowered.endswith(".solver")
            or "solver_gurobi" in lowered
            or "solver_hz" in lowered
            or "operator_hz" in lowered
        )

    for relative in relative_paths:
        source = (project_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if is_solver_name(alias.name):
                        direct_imports.append(
                            {
                                "path": relative,
                                "line": node.lineno,
                                "name": alias.name,
                            }
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if is_solver_name(node.module):
                    direct_imports.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "name": node.module,
                        }
                    )
            elif isinstance(node, ast.Call):
                name = _dotted_name(node.func) or ""
                if is_solver_name(name):
                    direct_calls.append(
                        {
                            "path": relative,
                            "line": node.lineno,
                            "name": name,
                        }
                    )
    ambient = sorted(
        name
        for name in sys.modules
        if is_solver_name(name)
        or name.startswith("act.back_end.solver")
    )
    return {
        "method": (
            "AST audit of the controlled numeric/harness path plus "
            "transparent sys.modules inventory"
        ),
        "direct_operator_or_solver_imports": direct_imports,
        "direct_operator_or_solver_imports_absent": not direct_imports,
        "direct_operator_or_solver_calls": direct_calls,
        "operator_or_solver_called": bool(direct_calls),
        "solver_verdict_created": False,
        "ambient_solver_modules_loaded": ambient,
        "ambient_import_explanation": (
            "Importing act.back_end.hybridz_tf executes act.back_end package "
            "bootstrap, which imports solver interfaces (and may import "
            "gurobipy). Ambient presence is recorded and is not represented "
            "as a direct audit-path call."
        ),
        "passed": not direct_imports and not direct_calls,
    }


def _find_method(
    tree: ast.Module, class_name: str, method_name: str
) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, ast.FunctionDef)
                    and child.name == method_name
                ):
                    return child
    raise RuntimeError(f"missing Fraction gate {class_name}.{method_name}")


def _fraction_gate_manifest(project_root: Path) -> Mapping[str, Any]:
    """Bind the two executed >=5,000-row Fraction tests to source."""

    records = {}
    for kind, spec in FRACTION_TESTS.items():
        relative = f"{str(spec['module']).replace('.', '/')}.py"
        source = (project_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        method = _find_method(
            tree, str(spec["class"]), str(spec["method"])
        )
        segment = ast.get_source_segment(source, method)
        if segment is None:
            raise RuntimeError(f"cannot bind Fraction method source: {kind}")
        integer_constants = {
            int(node.value)
            for node in ast.walk(method)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        }
        calls = {
            _dotted_name(node.func) or ""
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
        }
        minimum = int(spec["minimum_rows"])
        if kind == "dense":
            if minimum not in integer_constants or "_audit_fraction" not in calls:
                raise RuntimeError("Dense 5,000-row Fraction gate changed")
        else:
            assignments = {
                target.id: int(node.value.value)
                for node in tree.body
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, int)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if (
                assignments.get("FRACTION_QUERY_ROWS", -1) < minimum
                or "range" not in calls
                or "self._assert_sound" not in calls
            ):
                raise RuntimeError("Conv 5,000-row Fraction gate changed")
        full_id = (
            f"{spec['module']}.{spec['class']}.{spec['method']}"
        )
        records[kind] = {
            **dict(spec),
            "test_id": full_id,
            "source_path": relative,
            "method_source_sha256": hashlib.sha256(
                segment.encode("utf-8")
            ).hexdigest(),
            "static_minimum_rows_verified": True,
        }
    return records


def _fixed_environment(
    environment: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    values = dict(os.environ if environment is None else environment)
    required_threads = {
        "OPENBLAS_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    required_dynamic = ("MKL_DYNAMIC", "OMP_DYNAMIC")
    mismatches = {
        name: values.get(name, "")
        for name, expected in required_threads.items()
        if values.get(name, "") != expected
    }
    dynamic = {
        name: values.get(name, "") for name in required_dynamic
    }
    dynamic_bad = {
        name: value
        for name, value in dynamic.items()
        if value.strip().upper() not in _FALSE_ENV
    }
    return {
        "required_thread_environment": required_threads,
        "observed_thread_environment": {
            name: values.get(name, "") for name in required_threads
        },
        "required_dynamic_environment": {
            name: "explicit false" for name in required_dynamic
        },
        "observed_dynamic_environment": dynamic,
        "thread_mismatches": mismatches,
        "dynamic_mismatches": dynamic_bad,
        "passed": not mismatches and not dynamic_bad,
    }


def _parse_process_cpu_stat(
    raw_stat: str, *, expected_pid: int
) -> Mapping[str, int]:
    """Parse the PID, CPU and start-time fields from proc_pid_stat(5)."""

    prefix, separator, suffix = raw_stat.rpartition(")")
    fields = suffix.split()
    marker = f"{expected_pid} ("
    if (
        not separator
        or not prefix.startswith(marker)
        or len(fields) <= 19
    ):
        raise ValueError("malformed /proc PID stat record")
    return {
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "starttime_ticks": int(fields[19]),
        "parent_pid": int(fields[1]),
        "comm": prefix[len(marker) :],
    }


def _read_process_cpu_snapshot() -> Mapping[int, Mapping[str, Any]]:
    """Read PID-sealed process CPU counters without invoking ``ps``."""

    snapshot: Dict[int, Mapping[str, Any]] = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        try:
            parsed = _parse_process_cpu_stat(
                (directory / "stat").read_text(encoding="ascii"),
                expected_pid=pid,
            )
            uid = int(directory.stat().st_uid)
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            ValueError,
        ):
            continue
        # Cross-UID cmdline access can be denied even when the accounting
        # fields are readable.  Retain the process with its kernel comm.
        try:
            payload = (directory / "cmdline").read_bytes()
            command = payload.replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
        ):
            command = ""
        if not command:
            command = f"[{parsed['comm']}]"
        snapshot[pid] = {
            "pid": pid,
            "uid": uid,
            "cpu_ticks": int(parsed["cpu_ticks"]),
            "starttime_ticks": int(parsed["starttime_ticks"]),
            "command": command[:512],
        }
    return snapshot


def _read_affinity_cpu_counters(
    affinity: Sequence[int],
    *,
    proc_stat_text: Optional[str] = None,
) -> Mapping[str, Any]:
    """Sum busy counters only for CPUs on which this audit may execute."""

    cpu_ids = tuple(sorted(int(cpu) for cpu in affinity))
    if (
        not cpu_ids
        or len(set(cpu_ids)) != len(cpu_ids)
        or any(cpu < 0 for cpu in cpu_ids)
    ):
        raise RuntimeError("invalid CPU affinity for accounting")
    records = {}
    text = (
        Path("/proc/stat").read_text(encoding="ascii")
        if proc_stat_text is None
        else str(proc_stat_text)
    )
    for line in text.splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu\d+", fields[0]):
            continue
        records[int(fields[0][3:])] = fields[1:]
    busy = 0
    iowait = 0
    component_totals = {
        name: 0
        for name in ("user", "nice", "system", "irq", "softirq", "steal")
    }
    for cpu in cpu_ids:
        fields = records.get(cpu)
        if fields is None or len(fields) < 8:
            raise RuntimeError("affinity CPU is absent from /proc/stat")
        try:
            values = [int(value) for value in fields[:8]]
        except ValueError as exc:
            raise RuntimeError("non-integral per-CPU ticks") from exc
        user, nice, system, _idle, wait, irq, softirq, steal = values
        # user/nice already include guest/guest_nice.  Deliberately exclude
        # both idle and Linux's unstable iowait field from busy accounting.
        components = (user, nice, system, irq, softirq, steal)
        busy += sum(components)
        iowait += wait
        for name, value in zip(component_totals, components):
            component_totals[name] += value
    return {
        "cpu_ids": cpu_ids,
        "cpu_ids_sha256": _json_sha256(cpu_ids),
        "busy_ticks": busy,
        "iowait_ticks": iowait,
        "busy_components": component_totals,
        "field_policy": (
            "user+nice+system+irq+softirq+steal; "
            "idle/iowait excluded; guest already included in user/nice"
        ),
    }


def _read_self_cpu_identity() -> Mapping[str, int]:
    parsed = _parse_process_cpu_stat(
        Path("/proc/self/stat").read_text(encoding="ascii"),
        expected_pid=os.getpid(),
    )
    return {
        "pid": os.getpid(),
        "cpu_ticks": int(parsed["cpu_ticks"]),
        "starttime_ticks": int(parsed["starttime_ticks"]),
    }


def _external_cpu_window_from_counters(
    *,
    global_busy_ticks_start: int,
    global_busy_ticks_end: int,
    self_cpu_ticks_start: int,
    self_cpu_ticks_end: int,
    elapsed_ns: int,
    ticks_per_second: int,
    limit_cores: float,
) -> Mapping[str, Any]:
    """Account for every non-self CPU user over an observation window."""

    if (
        isinstance(elapsed_ns, bool)
        or not isinstance(elapsed_ns, int)
        or elapsed_ns <= 0
        or ticks_per_second <= 0
        or not math.isfinite(limit_cores)
        or limit_cores < 0.0
    ):
        raise ValueError("invalid external CPU window parameters")
    global_delta = int(global_busy_ticks_end) - int(
        global_busy_ticks_start
    )
    self_delta = int(self_cpu_ticks_end) - int(self_cpu_ticks_start)
    consistent = global_delta >= 0 and self_delta >= 0
    external_delta = global_delta - self_delta
    elapsed_s = elapsed_ns / 1_000_000_000
    if not consistent or external_delta < 0:
        return {
            "elapsed_nanoseconds": elapsed_ns,
            "elapsed_seconds": elapsed_s,
            "clock_ticks_per_second": ticks_per_second,
            "global_busy_ticks_delta": global_delta,
            "self_cpu_ticks_delta": self_delta,
            "external_cpu_ticks_delta": external_delta,
            "external_cpu_core_equivalents": math.inf,
            "external_cpu_limit_core_equivalents": limit_cores,
            "counter_consistent": False,
            "passed": False,
        }
    external_cores = external_delta / (ticks_per_second * elapsed_s)
    limit_numerator, limit_denominator = float(limit_cores).as_integer_ratio()
    measured_scaled = (
        external_delta * 1_000_000_000 * limit_denominator
    )
    limit_scaled = ticks_per_second * elapsed_ns * limit_numerator
    return {
        "elapsed_nanoseconds": elapsed_ns,
        "elapsed_seconds": elapsed_s,
        "clock_ticks_per_second": ticks_per_second,
        "global_busy_ticks_delta": global_delta,
        "self_cpu_ticks_delta": self_delta,
        "external_cpu_ticks_delta": external_delta,
        "external_cpu_core_equivalents": external_cores,
        "external_cpu_limit_core_equivalents": limit_cores,
        "strict_integer_comparison": {
            "measured_scaled": measured_scaled,
            "limit_scaled": limit_scaled,
            "operator": "<",
        },
        "counter_consistent": True,
        "passed": measured_scaled < limit_scaled,
    }


def _begin_external_cpu_window() -> Mapping[str, Any]:
    """Capture a private baseline for whole-window CPU accounting."""

    process_snapshot = _read_process_cpu_snapshot()
    ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    affinity = tuple(sorted(os.sched_getaffinity(0)))
    counters = _read_affinity_cpu_counters(affinity)
    started_ns = time.monotonic_ns()
    identity = _read_self_cpu_identity()
    return {
        "monotonic_started_ns": started_ns,
        "clock_ticks_per_second": ticks_per_second,
        "affinity": affinity,
        "affinity_sha256": counters["cpu_ids_sha256"],
        "global_busy_ticks": counters["busy_ticks"],
        "global_iowait_ticks": counters["iowait_ticks"],
        "global_busy_components": counters["busy_components"],
        "global_field_policy": counters["field_policy"],
        "self_cpu_ticks": identity["cpu_ticks"],
        "self_pid": identity["pid"],
        "self_starttime_ticks": identity["starttime_ticks"],
        "_process_snapshot": process_snapshot,
    }


def _finish_external_cpu_window(
    started: Mapping[str, Any],
    *,
    limit_cores: float = _HOST_EXTERNAL_CPU_LIMIT_CORES,
    individual_threshold_cores: float = _HOST_CPU_COMPETITOR_CORES,
    excluded_pids: Iterable[int] = (),
) -> Mapping[str, Any]:
    """Close a CPU window and retain both aggregate and PID evidence."""

    identity = _read_self_cpu_identity()
    finished_ns = time.monotonic_ns()
    affinity = tuple(sorted(os.sched_getaffinity(0)))
    counters = _read_affinity_cpu_counters(affinity)
    if (
        int(started["self_pid"]) != identity["pid"]
        or int(started["self_starttime_ticks"])
        != identity["starttime_ticks"]
        or tuple(started["affinity"]) != affinity
        or str(started["affinity_sha256"])
        != str(counters["cpu_ids_sha256"])
        or str(started["global_field_policy"])
        != str(counters["field_policy"])
    ):
        raise RuntimeError("CPU accounting identity or affinity changed")
    ticks_per_second = int(started["clock_ticks_per_second"])
    if ticks_per_second != int(os.sysconf("SC_CLK_TCK")):
        raise RuntimeError("CPU accounting clock tick rate changed")
    elapsed_ns = finished_ns - int(started["monotonic_started_ns"])
    aggregate = dict(_external_cpu_window_from_counters(
        global_busy_ticks_start=int(started["global_busy_ticks"]),
        global_busy_ticks_end=int(counters["busy_ticks"]),
        self_cpu_ticks_start=int(started["self_cpu_ticks"]),
        self_cpu_ticks_end=identity["cpu_ticks"],
        elapsed_ns=elapsed_ns,
        ticks_per_second=ticks_per_second,
        limit_cores=limit_cores,
    ))
    component_deltas = {
        name: int(counters["busy_components"][name])
        - int(started["global_busy_components"][name])
        for name in counters["busy_components"]
    }
    components_consistent = all(
        value >= 0 for value in component_deltas.values()
    )
    if not components_consistent:
        aggregate["counter_consistent"] = False
        aggregate["passed"] = False
    first = started["_process_snapshot"]
    second = _read_process_cpu_snapshot()
    excluded = {identity["pid"], *(int(pid) for pid in excluded_pids)}
    competitors = _cpu_competitors_from_snapshots(
        first,
        second,
        elapsed_s=elapsed_ns / 1_000_000_000,
        ticks_per_second=ticks_per_second,
        excluded_pids=excluded,
        threshold_cores=individual_threshold_cores,
    )
    first_pids = set(first)
    second_pids = set(second)
    passed = bool(aggregate["passed"] and not competitors)
    return {
        **aggregate,
        "affinity": affinity,
        "affinity_sha256": counters["cpu_ids_sha256"],
        "busy_field_policy": counters["field_policy"],
        "iowait_ticks_delta_recorded_not_gated": (
            int(counters["iowait_ticks"])
            - int(started["global_iowait_ticks"])
        ),
        "busy_component_tick_deltas": component_deltas,
        "busy_components_monotonic": components_consistent,
        "individual_threshold_cpu_core_equivalents": (
            individual_threshold_cores
        ),
        "high_cpu_competitors": competitors,
        "persistent_process_count": len(first_pids.intersection(second_pids)),
        "processes_started_during_window": len(second_pids - first_pids),
        "processes_exited_during_window": len(first_pids - second_pids),
        "aggregate_external_cpu_passed": bool(aggregate["passed"]),
        "passed": passed,
    }


def _cpu_competitors_from_snapshots(
    first: Mapping[int, Mapping[str, Any]],
    second: Mapping[int, Mapping[str, Any]],
    *,
    elapsed_s: float,
    ticks_per_second: int,
    excluded_pids: Iterable[int],
    threshold_cores: float,
) -> Sequence[Mapping[str, Any]]:
    """Return persistent non-authority processes consuming a CPU core."""

    if (
        not math.isfinite(elapsed_s)
        or elapsed_s <= 0.0
        or ticks_per_second <= 0
        or not math.isfinite(threshold_cores)
        or threshold_cores <= 0.0
    ):
        raise ValueError("invalid process CPU sample parameters")
    excluded = frozenset(int(pid) for pid in excluded_pids)
    competitors = []
    for pid in sorted(set(first).intersection(second)):
        if pid in excluded:
            continue
        before = first[pid]
        after = second[pid]
        if int(before["starttime_ticks"]) != int(after["starttime_ticks"]):
            continue
        delta_ticks = int(after["cpu_ticks"]) - int(before["cpu_ticks"])
        if delta_ticks < 0:
            continue
        cpu_cores = delta_ticks / (ticks_per_second * elapsed_s)
        if cpu_cores + 1e-12 < threshold_cores:
            continue
        competitors.append(
            {
                "pid": pid,
                "uid": int(after["uid"]),
                "cpu_core_equivalents": cpu_cores,
                "cpu_ticks_delta": delta_ticks,
                "starttime_ticks": int(after["starttime_ticks"]),
                "command": str(after["command"])[:512],
            }
        )
    return tuple(
        sorted(
            competitors,
            key=lambda item: (-item["cpu_core_equivalents"], item["pid"]),
        )
    )


def _sample_cpu_competitors(
    *,
    excluded_pids: Iterable[int],
    sample_seconds: float = _HOST_CPU_SAMPLE_SECONDS,
    threshold_cores: float = _HOST_CPU_COMPETITOR_CORES,
) -> Mapping[str, Any]:
    """Measure competing CPU use over a short interval outside timing."""

    if not math.isfinite(sample_seconds) or sample_seconds <= 0.0:
        raise ValueError("CPU sample duration must be positive and finite")
    started = _begin_external_cpu_window()
    time.sleep(sample_seconds)
    finished = _finish_external_cpu_window(
        started,
        limit_cores=_HOST_EXTERNAL_CPU_LIMIT_CORES,
        individual_threshold_cores=threshold_cores,
        excluded_pids=excluded_pids,
    )
    first = started["_process_snapshot"]
    second_count = int(finished["persistent_process_count"]) + int(
        finished["processes_started_during_window"]
    )
    competitors = tuple(finished["high_cpu_competitors"])
    return {
        "requested_sample_seconds": sample_seconds,
        "observed_sample_seconds": finished["elapsed_seconds"],
        "clock_ticks_per_second": finished["clock_ticks_per_second"],
        "threshold_cpu_core_equivalents": threshold_cores,
        "readable_process_count_first": len(first),
        "readable_process_count_second": second_count,
        "high_cpu_competitors": competitors,
        "aggregate_external_cpu": {
            key: finished[key]
            for key in (
                "elapsed_nanoseconds",
                "elapsed_seconds",
                "clock_ticks_per_second",
                "global_busy_ticks_delta",
                "self_cpu_ticks_delta",
                "external_cpu_ticks_delta",
                "external_cpu_core_equivalents",
                "external_cpu_limit_core_equivalents",
                "strict_integer_comparison",
                "counter_consistent",
                "processes_started_during_window",
                "processes_exited_during_window",
            )
        }
        | {"passed": finished["aggregate_external_cpu_passed"]},
        "passed": bool(
            finished["passed"] and not competitors
        ),
    }


def _host_preflight(
    *, load_limit: Optional[float] = 4.0
) -> Mapping[str, Any]:
    if not Path("/proc/loadavg").is_file():
        raise RuntimeError("official audit requires Linux /proc")
    load_fields = Path("/proc/loadavg").read_text(encoding="ascii").split()
    load = [float(value) for value in load_fields[:3]]
    affinity = sorted(os.sched_getaffinity(0))
    blockers = []
    own_pid = os.getpid()
    own_uid = os.getuid()
    ancestors = set()
    current = own_pid
    while current > 1:
        try:
            stat = (Path("/proc") / str(current) / "stat").read_text(
                encoding="ascii"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            break
        # Field two (comm) is parenthesized and may contain spaces or ')'.
        _, separator, suffix = stat.rpartition(")")
        fields = suffix.split()
        if not separator or len(fields) < 2:
            break
        current = int(fields[1])
        if current <= 0 or current in ancestors:
            break
        ancestors.add(current)
    excluded_pids = frozenset({own_pid, *ancestors})
    for directory in Path("/proc").iterdir():
        if (
            not directory.name.isdigit()
            or int(directory.name) in excluded_pids
        ):
            continue
        try:
            if directory.stat().st_uid != own_uid:
                continue
            payload = (directory / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = payload.replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        lowered = command.lower()
        if "query_dual" in lowered and (
            "audit" in lowered or "unittest" in lowered
        ):
            blockers.append(
                {"pid": int(directory.name), "command": command[:512]}
            )
    cpu_sample = _sample_cpu_competitors(excluded_pids=excluded_pids)
    memory_text = Path("/proc/meminfo").read_text(encoding="ascii")
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", memory_text, re.M)
    available = 0 if match is None else int(match.group(1)) * 1024
    passed = bool(
        tuple(affinity) == _OFFICIAL_CPU_AFFINITY
        and (load_limit is None or load[0] <= load_limit)
        and available >= 4 * _GIB
        and not blockers
        and cpu_sample["passed"]
    )
    return {
        "load_1_5_15": load,
        "load_1_limit": load_limit,
        "load_gate_enforced": load_limit is not None,
        "required_cpu_affinity": list(_OFFICIAL_CPU_AFFINITY),
        "cpu_affinity_count": len(affinity),
        "cpu_affinity": affinity,
        "cpu_affinity_sha256": _json_sha256(affinity),
        "memory_available_bytes": available,
        "excluded_process_ancestry": sorted(ancestors),
        "other_query_dual_audit_workers": blockers,
        "competing_cpu_process_gate_enforced": True,
        "competing_cpu_sample": cpu_sample,
        "high_cpu_competitors": cpu_sample["high_cpu_competitors"],
        "niceness": os.nice(0),
        "passed": passed,
    }


def _run_unittests(
    project_root: Path,
    fraction_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        *TEST_MODULES,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "MKL_DYNAMIC": "FALSE",
            "OMP_DYNAMIC": "FALSE",
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600.0,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout
    match = re.search(r"Ran (\d+) tests?", output)
    count = int(match.group(1)) if match else None
    module_counts = {
        module: len(
            re.findall(rf"\({re.escape(module)}\.[^)]+\)", output)
        )
        for module in TEST_MODULES
    }
    executed_fraction = {}
    for kind, record in fraction_manifest.items():
        test_id = str(record["test_id"])
        observed = (
            f"({test_id})" in output
            or test_id in output
        )
        executed_fraction[kind] = {
            "test_id": test_id,
            "minimum_rows": int(record["minimum_rows"]),
            "observed_in_verbose_output": observed,
        }
    passed = bool(
        completed.returncode == 0
        and count is not None
        and count >= MIN_RELATED_TESTS
        and all(
            module_counts[module] >= minimum
            for module, minimum in TEST_MODULE_MINIMUMS.items()
        )
        and all(
            item["observed_in_verbose_output"]
            for item in executed_fraction.values()
        )
    )
    return {
        "command": command,
        "module_count": len(TEST_MODULES),
        "module_minimum_test_counts": dict(TEST_MODULE_MINIMUMS),
        "observed_module_test_counts": module_counts,
        "minimum_total_test_count": MIN_RELATED_TESTS,
        "test_count": count,
        "returncode": completed.returncode,
        "seconds": elapsed,
        "fraction_gates_executed": executed_fraction,
        "output_sha256": hashlib.sha256(
            output.encode("utf-8")
        ).hexdigest(),
        "output_tail": output[-8000:],
        "passed": passed,
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish once by hard link; never replace an existing receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_query_dual_v51_controlled_audit_receipt(
    receipt: Mapping[str, Any],
) -> bool:
    """Check canonical integrity; the receipt never grants proof authority."""

    try:
        body = dict(receipt)
        claimed = str(body.pop("receipt_sha256"))
        solver_audit = body.get("operator_solver_usage_audit", {})
        gates = body.get("gates", {})
        gate_values_are_boolean = bool(
            isinstance(gates, Mapping)
            and all(
                isinstance(gates.get(name), bool)
                for name in _REQUIRED_PROMOTION_GATE_KEYS
            )
        )
        expected_status = (
            "passed"
            if gate_values_are_boolean
            and all(gates[name] for name in _REQUIRED_PROMOTION_GATE_KEYS)
            else "rejected"
        )
        return bool(
            body.get("schema") == SCHEMA
            and body.get("proof_authority") is False
            and body.get("controlled_synthetic_only") is True
            and body.get("real_onnx_or_vnnlib_accessed") is False
            and body.get("direct_operator_or_solver_imports") is False
            and body.get("operator_or_solver_called") is False
            and body.get("solver_verdict_created") is False
            and solver_audit.get(
                "direct_operator_or_solver_imports_absent"
            )
            is True
            and solver_audit.get("operator_or_solver_called") is False
            and solver_audit.get("solver_verdict_created") is False
            and isinstance(
                solver_audit.get("ambient_solver_modules_loaded"), list
            )
            and body.get("ambient_solver_modules_loaded")
            == solver_audit.get("ambient_solver_modules_loaded")
            and body.get("official_configuration_sha256")
            == OFFICIAL_CONFIGURATION_SHA256
            and _json_sha256(body.get("configuration"))
            == OFFICIAL_CONFIGURATION_SHA256
            and gate_values_are_boolean
            and body.get("status") == expected_status
            and hmac.compare_digest(_json_sha256(body), claimed)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def run_audit(
    *,
    project_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    """Run the immutable official audit; no benchmark knob is accepted.

    Publication occurs only after the entire audit and receipt self-check.
    Any fatal precondition, replay exception, warmup regression, or integrity
    failure leaves the requested output path absent; no partial receipt is
    promoted as an atomic controlled result.
    """

    parameters = OFFICIAL_PARAMETERS
    topology_profile = OFFICIAL_TOPOLOGY
    _assert_official_configuration(parameters, topology_profile)
    project_root = project_root.resolve()
    output = Path(os.path.abspath(output.expanduser()))
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite controlled receipt: {output}"
        )
    environment_gate = _fixed_environment()
    if not environment_gate["passed"]:
        raise RuntimeError(
            "official audit requires fixed four-thread, dynamic-off environment"
        )
    host_initial = _host_preflight()
    if not host_initial["passed"]:
        raise RuntimeError("official audit host is not sufficiently quiet")

    source_before = _source_hashes(project_root)
    numeric_source_gate = _validate_expected_numeric_sources(source_before)
    v3_relative = "act/back_end/hybridz_tf/query_dual_replay.py"
    if source_before[v3_relative] != EXPECTED_V3_SHA256:
        raise RuntimeError("frozen V3 replay source hash changed")
    device_audit = _numeric_device_audit(project_root)
    if not device_audit["passed"]:
        raise RuntimeError("numeric dependency closure contains GPU compute")
    operator_solver_audit = _operator_solver_usage_audit(project_root)
    if not operator_solver_audit["passed"]:
        raise RuntimeError("controlled path directly imports/calls a solver")
    fraction_manifest = _fraction_gate_manifest(project_root)
    unittest_gate = _run_unittests(project_root, fraction_manifest)
    if not unittest_gate["passed"]:
        raise RuntimeError("V5.1 related unittest/Fraction gate failed")

    with threadpool_limits(limits=parameters.blas_threads):
        initial_contract = probe_query_dual_blas_contract(
            required_threads=parameters.blas_threads,
            deadline=time.monotonic() + 60.0,
        )
        if not validate_query_dual_blas_contract(initial_contract):
            raise RuntimeError("initial live BLAS contract is invalid")

        net = _build_topology(topology_profile)
        root_start = time.perf_counter()
        root_certificate = certify_query_dual_boxes(
            net,
            deadline=time.monotonic() + parameters.root_timeout_s,
            conv_channel_chunk=32,
        )
        root_seconds = time.perf_counter() - root_start
        if not verify_query_dual_box_certificate(
            root_certificate, net=net
        ):
            raise RuntimeError("production-synthetic root certificate failed")
        stages = _query_schedule(net, parameters)
        if (
            len(stages) != 5
            or sum(int(stage["objective_count"]) for stage in stages)
            != 227
        ):
            raise RuntimeError("official five-stage schedule changed")
        stage_material_before = _stage_material_seals(stages)
        host_pre_timing = _host_preflight()
        if not host_pre_timing["passed"]:
            raise RuntimeError(
                "host changed before controlled warmup/timing"
            )

        timing_cpu_window_start = _begin_external_cpu_window()
        baseline_rss = _rss_bytes()
        baseline_hwm = _rss_bytes("VmHWM")
        warm_v3_cpu_start = _begin_external_cpu_window()
        warm_v3 = dict(
            _run_v3_schedule(
                net, root_certificate, stages, parameters
            )
        )
        warm_v3["external_cpu_window"] = _finish_external_cpu_window(
            warm_v3_cpu_start
        )
        if not warm_v3["external_cpu_window"]["passed"]:
            raise RuntimeError("external CPU interfered with V3 warmup")
        if _stage_material_seals(stages) != stage_material_before:
            raise RuntimeError("V3 warmup changed query/alpha material")
        warm_v51_cpu_start = _begin_external_cpu_window()
        warm_v51 = dict(
            _run_v51_schedule(
                net,
                root_certificate,
                stages,
                parameters,
                initial_contract,
            )
        )
        warm_v51["external_cpu_window"] = _finish_external_cpu_window(
            warm_v51_cpu_start
        )
        if not warm_v51["external_cpu_window"]["passed"]:
            raise RuntimeError("external CPU interfered with V5.1 warmup")
        if _stage_material_seals(stages) != stage_material_before:
            raise RuntimeError("V5.1 warmup changed query/alpha material")
        warm_comparison = _compare_runs(warm_v3, warm_v51)
        if warm_comparison["tightness_regression_count"]:
            raise RuntimeError("V5.1 warmup has a tightness regression")

        maximum_rss = max(
            baseline_rss,
            int(warm_v3["maximum_rss_bytes"]),
            int(warm_v51["maximum_rss_bytes"]),
        )
        maximum_hwm = max(
            baseline_hwm,
            int(warm_v3["maximum_hwm_bytes"]),
            int(warm_v51["maximum_hwm_bytes"]),
        )
        v3_seconds = []
        v51_seconds = []
        pair_records = []
        total_regressions = 0
        for pair_index in range(parameters.pairs):
            order = (
                ("v3", "v51")
                if pair_index % 2 == 0
                else ("v51", "v3")
            )
            observed: Dict[str, Mapping[str, Any]] = {}
            for implementation in order:
                implementation_cpu_start = _begin_external_cpu_window()
                if implementation == "v3":
                    run = _run_v3_schedule(
                        net,
                        root_certificate,
                        stages,
                        parameters,
                    )
                else:
                    run = _run_v51_schedule(
                        net,
                        root_certificate,
                        stages,
                        parameters,
                        initial_contract,
                    )
                run = dict(run)
                run["external_cpu_window"] = (
                    _finish_external_cpu_window(
                        implementation_cpu_start
                    )
                )
                observed[implementation] = run
                if not run["external_cpu_window"]["passed"]:
                    raise RuntimeError(
                        "external CPU interfered with timed "
                        f"{implementation} implementation"
                    )
                if _stage_material_seals(stages) != stage_material_before:
                    raise RuntimeError(
                        f"{implementation} pair changed query/alpha material"
                    )
                maximum_rss = max(
                    maximum_rss,
                    int(observed[implementation]["maximum_rss_bytes"]),
                )
                maximum_hwm = max(
                    maximum_hwm,
                    int(observed[implementation]["maximum_hwm_bytes"]),
                )
            comparison = _compare_runs(
                observed["v3"], observed["v51"]
            )
            total_regressions += int(
                comparison["tightness_regression_count"]
            )
            v3_seconds.append(float(observed["v3"]["total_seconds"]))
            v51_seconds.append(float(observed["v51"]["total_seconds"]))
            pair_records.append(
                {
                    "pair_index": pair_index,
                    "order": list(order),
                    "v3": _public_run_record(observed["v3"]),
                    "v51": _public_run_record(observed["v51"]),
                    "comparison": comparison,
                }
            )

        timing_cpu_window = _finish_external_cpu_window(
            timing_cpu_window_start
        )
        final_live_blas_recheck = validate_query_dual_blas_contract(
            initial_contract,
            recheck_current_platform=True,
            deadline=time.monotonic() + 60.0,
        )
        if not final_live_blas_recheck:
            raise RuntimeError("final live BLAS contract recheck failed")
        stage_material_after = _stage_material_seals(stages)
        if stage_material_after != stage_material_before:
            raise RuntimeError("query/alpha material changed after timing")
        # Our own four-thread work legitimately raises the one-minute load
        # average.  The post check retains its short aggregate/PID sample,
        # and the separate counter window covers all warmup and timed work,
        # including competing processes that exited or became idle.
        host_post_timing = dict(_host_preflight(load_limit=None))
        host_post_timing["warmup_and_timing_external_cpu_window"] = (
            timing_cpu_window
        )
        host_post_timing["passed"] = bool(
            host_post_timing["passed"] and timing_cpu_window["passed"]
        )

    median_v3 = float(np.median(v3_seconds))
    median_v51 = float(np.median(v51_seconds))
    speedup = median_v3 / median_v51
    bootstrap_lower = _bootstrap_lower(
        v3_seconds,
        v51_seconds,
        samples=parameters.bootstrap_samples,
    )
    source_after = _source_hashes(project_root)
    source_stable = source_before == source_after
    incremental_rss = max(0, maximum_rss - baseline_rss)
    incremental_hwm = max(0, maximum_hwm - baseline_hwm)
    maximum_workspace = max(
        int(stage["workspace"]["modeled_live_chunk_upper_bytes"])
        for pair in pair_records
        for implementation in ("v3", "v51")
        for stage in pair[implementation]["stages"]
    )
    bounds_sha = str(
        root_certificate.receipt["hashes"]["bounds_sha256"]
    )
    same_root_bounds = all(
        pair[implementation]["same_certified_bounds_sha256"]
        == bounds_sha
        for pair in pair_records
        for implementation in ("v3", "v51")
    )
    commit_recheck_count = int(
        bool(warm_v51["commit_live_blas_recheck_bound"])
    ) + sum(
        int(bool(pair["v51"]["commit_live_blas_recheck_bound"]))
        for pair in pair_records
    )
    expected_commit_rechecks = parameters.pairs + 1
    timed_cpu_windows = [
        pair[implementation]["external_cpu_window"]
        for pair in pair_records
        for implementation in ("v3", "v51")
    ]
    gates = {
        "environment_passed": bool(environment_gate["passed"]),
        "host_initial_passed": bool(host_initial["passed"]),
        "host_pre_timing_passed": bool(host_pre_timing["passed"]),
        "host_post_timing_passed": bool(host_post_timing["passed"]),
        "warmup_and_timing_external_cpu_passed": bool(
            timing_cpu_window["passed"]
        ),
        "per_timed_implementation_cpu_window_count_required": (
            parameters.pairs * 2
        ),
        "per_timed_implementation_cpu_window_count_measured": len(
            timed_cpu_windows
        ),
        "per_timed_implementation_external_cpu_passed": bool(
            len(timed_cpu_windows) == parameters.pairs * 2
            and all(window["passed"] for window in timed_cpu_windows)
        ),
        "unittest_fraction_passed": bool(unittest_gate["passed"]),
        "dense_fraction_rows_minimum": 5_000,
        "conv_fraction_rows_minimum": 5_000,
        "production_objective_count": 227,
        "pair_count_required": 5,
        "pair_count_measured": len(pair_records),
        "pair_count_passed": len(pair_records) >= 5,
        "median_speedup_required": 2.0,
        "median_speedup_measured": speedup,
        "median_speedup_passed": speedup >= 2.0,
        "bootstrap_95_lower_required": 1.8,
        "bootstrap_95_lower_measured": bootstrap_lower,
        "bootstrap_95_lower_passed": bootstrap_lower >= 1.8,
        "tightness_regression_count": total_regressions,
        "tightness_passed": total_regressions == 0,
        "rss_limit_bytes": parameters.rss_limit_bytes,
        "incremental_rss_bytes": incremental_rss,
        "incremental_hwm_bytes": incremental_hwm,
        "rss_passed": max(incremental_rss, incremental_hwm)
        <= parameters.rss_limit_bytes,
        "workspace_limit_bytes": parameters.workspace_bytes,
        "maximum_modeled_live_chunk_bytes": maximum_workspace,
        "workspace_passed": maximum_workspace
        <= parameters.workspace_bytes,
        "device_ast_passed": bool(device_audit["passed"]),
        "live_blas_probe_and_commit_rechecks_passed": bool(
            final_live_blas_recheck
            and commit_recheck_count == expected_commit_rechecks
        ),
        "v51_commit_live_recheck_count": commit_recheck_count,
        "v51_commit_live_recheck_count_required": expected_commit_rechecks,
        "same_root_certified_bounds_passed": same_root_bounds,
        "query_alpha_material_unchanged": (
            stage_material_before == stage_material_after
        ),
        "source_hashes_stable": source_stable,
    }
    passed = all(
        isinstance(gates[name], bool) and gates[name]
        for name in _REQUIRED_PROMOTION_GATE_KEYS
    )
    contract_receipt = _canonical_value(initial_contract.receipt)
    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if passed else "rejected",
        "proof_authority": False,
        "authority_scope": "controlled_candidate_only_no_solver_verdict",
        "controlled_synthetic_only": True,
        "real_onnx_or_vnnlib_accessed": False,
        "direct_operator_or_solver_imports": False,
        "operator_or_solver_called": False,
        "solver_verdict_created": False,
        "ambient_solver_modules_loaded": list(
            operator_solver_audit["ambient_solver_modules_loaded"]
        ),
        "official_configuration_sha256": (
            OFFICIAL_CONFIGURATION_SHA256
        ),
        "configuration": _configuration_record(
            parameters, topology_profile
        ),
        "seed": SEED,
        "stage_semantics": {
            "legacy_v5_start_ids_not_used": [3, 6, 10, 15],
            "target_relu_ids": [3, 6, 10, 15],
            "shared_v3_v51_predecessor_cone_start_ids": [2, 5, 9, 14],
            "reason": (
                "root-owned target stages bound the target ReLU "
                "preactivation and therefore replay its predecessor cone"
            ),
            "v3_and_v51_rows_alpha_identical": True,
        },
        "query_schedule": [
            {
                key: value
                for key, value in stage.items()
                if key not in {"query_rows", "alpha"}
            }
            for stage in stages
        ],
        "root_certificate": {
            "constructed_by_certify_query_dual_boxes": True,
            "shared_by_both_paths": True,
            "construction_excluded_from_replay_timing": True,
            "seconds": root_seconds,
            "receipt_sha256": str(
                root_certificate.receipt["receipt_sha256"]
            ),
            "net_sha256": str(
                root_certificate.receipt["hashes"]["net_sha256"]
            ),
            "bounds_sha256": bounds_sha,
        },
        "environment_gate": environment_gate,
        "host_preflight": {
            "initial": host_initial,
            "before_timing": host_pre_timing,
            "after_timing": host_post_timing,
        },
        "numeric_device_audit": device_audit,
        "operator_solver_usage_audit": operator_solver_audit,
        "numeric_source_gate": numeric_source_gate,
        "stage_material_seals_before": stage_material_before,
        "stage_material_seals_after": stage_material_after,
        "fraction_gate_manifest": fraction_manifest,
        "unittest_gate": unittest_gate,
        "blas_contract": contract_receipt,
        "warmup": {
            "v3": _public_run_record(warm_v3),
            "v51": _public_run_record(warm_v51),
            "comparison": warm_comparison,
            "excluded_from_timing_samples": True,
        },
        "alternating_pairs": pair_records,
        "performance": {
            "blas_threads": parameters.blas_threads,
            "dynamic_threads_disabled": True,
            "threadpools_after_final_recheck": threadpool_info(),
            "v3_total_seconds": v3_seconds,
            "v51_total_seconds": v51_seconds,
            "v3_median_seconds": median_v3,
            "v51_median_seconds": median_v51,
            "median_speedup": speedup,
            "paired_bootstrap_samples": parameters.bootstrap_samples,
            "paired_bootstrap_95_lower": bootstrap_lower,
            "timing_scope": (
                "new root-owned session + one frame + five replays + commit; "
                "shared root certificate excluded"
            ),
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "maximum_rss_bytes": maximum_rss,
            "baseline_hwm_bytes": baseline_hwm,
            "maximum_hwm_bytes": maximum_hwm,
            "incremental_rss_bytes": incremental_rss,
            "incremental_hwm_bytes": incremental_hwm,
            "maximum_modeled_live_chunk_bytes": maximum_workspace,
        },
        "gates": gates,
        "source_sha256": source_before,
        "source_sha256_after": source_after,
        "decision": (
            "eligible-only-for-separate-same-iid2-query-preregistration"
            if passed
            else "close-v5.1a-without-real-probe"
        ),
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    if not verify_query_dual_v51_controlled_audit_receipt(receipt):
        raise RuntimeError("controlled audit receipt self-check failed")
    _write_atomic(output, receipt)
    return receipt


def _parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the immutable synthetic V5.1 promotion audit. "
            "There are intentionally no timing/configuration overrides."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parse_arguments(argv)
    receipt = run_audit(
        project_root=arguments.project_root,
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "median_speedup": receipt["performance"][
                    "median_speedup"
                ],
                "bootstrap_95_lower": receipt["performance"][
                    "paired_bootstrap_95_lower"
                ],
                "tightness_regressions": receipt["gates"][
                    "tightness_regression_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
