#!/usr/bin/env python3
"""Fail-fast controlled audit for the isolated query-dual V5 candidate.

This command constructs only deterministic synthetic networks.  It does not
load an ONNX model, VNNLIB property, Operator-HZ object, or solver, and it
cannot produce a verification verdict.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from act.back_end.hybridz_tf import query_dual_replay as v3
from act.back_end.hybridz_tf import query_dual_replay_v5 as v5


SCHEMA = "act.query_dual_v5_controlled_audit.v1"
SEED = 20260728
EXPECTED_V3_SHA256 = (
    "6e291bdd4526518496e664c14e15664bf"
    "554c1e9f089d92f65f8097081db5d7e"
)
SCHEDULE = ((3, 32), (6, 16), (10, 48), (15, 32), (None, 99))
TEST_MODULES = (
    "act.back_end.hybridz_tf.test_query_dual_box_certifier",
    "act.back_end.hybridz_tf.test_query_dual_replay",
    "act.back_end.hybridz_tf.test_query_dual_replay_v3",
    "act.back_end.hybridz_tf.test_query_dual_replay_v4",
    "act.back_end.hybridz_tf.test_query_dual_scalar_guard",
    "act.back_end.hybridz_tf.test_query_dual_replay_v5_candidate",
    "act.back_end.hybridz_tf.test_query_dual_replay_v5",
    "act.back_end.hybridz_tf.test_query_dual_v5_authority",
    "act.back_end.hybridz_tf.test_query_dual_pipeline",
    "act.back_end.hybridz_tf.test_query_dual_pipeline_v3",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
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


def _layer(layer_id: int, kind: str, width: int, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[
            (int(layer_id), index) for index in range(int(width))
        ],
        cache={},
    )


def _network(layers, predecessors):
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


def _box(width: int) -> Mapping[str, np.ndarray]:
    lower = np.full(int(width), -1.0, dtype=np.float64)
    upper = np.full(int(width), 1.0, dtype=np.float64)
    return {"lb": lower, "ub": upper}


def _conv(
    rng: np.random.Generator,
    layer_id: int,
    *,
    input_hw: int,
    output_hw: int,
    kernel: int,
    stride: int,
    padding: int,
) -> Any:
    channels = 128
    weight = np.ascontiguousarray(
        rng.normal(
            0.0,
            0.025,
            size=(channels, channels, kernel, kernel),
        ),
        dtype=np.float64,
    )
    bias = np.ascontiguousarray(
        rng.normal(0.0, 0.005, size=channels), dtype=np.float64
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
            "input_shape": (1, channels, input_hw, input_hw),
            "output_shape": (
                1,
                channels,
                output_hw,
                output_hw,
            ),
        },
    )


def _build_topology():
    rng = np.random.default_rng(SEED)
    width8 = 128 * 8 * 8
    width4 = 128 * 4 * 4
    input_layer = _layer(
        0,
        "INPUT",
        width8,
        {"shape": (1, 128, 8, 8), "dtype": "torch.float64"},
    )
    input_spec = _layer(1, "INPUT_SPEC", width8, {"kind": "BOX"})
    conv2 = _conv(
        rng,
        2,
        input_hw=8,
        output_hw=8,
        kernel=3,
        stride=1,
        padding=1,
    )
    relu3 = _layer(3, "RELU", width8)
    conv4 = _conv(
        rng,
        4,
        input_hw=8,
        output_hw=8,
        kernel=3,
        stride=1,
        padding=1,
    )
    add5 = _layer(5, "ADD", width8)
    relu6 = _layer(6, "RELU", width8)
    main7 = _conv(
        rng,
        7,
        input_hw=8,
        output_hw=4,
        kernel=3,
        stride=2,
        padding=1,
    )
    skip8 = _conv(
        rng,
        8,
        input_hw=8,
        output_hw=4,
        kernel=1,
        stride=2,
        padding=0,
    )
    add9 = _layer(9, "ADD", width4)
    relu10 = _layer(10, "RELU", width4)
    conv11 = _conv(
        rng,
        11,
        input_hw=4,
        output_hw=4,
        kernel=3,
        stride=1,
        padding=1,
    )
    relu12 = _layer(12, "RELU", width4)
    conv13 = _conv(
        rng,
        13,
        input_hw=4,
        output_hw=4,
        kernel=3,
        stride=1,
        padding=1,
    )
    add14 = _layer(14, "ADD", width4)
    relu15 = _layer(15, "RELU", width4)
    flatten16 = _layer(16, "FLATTEN", width4, {"start_dim": 1})
    dense_weight = np.ascontiguousarray(
        rng.normal(0.0, 0.025, size=(100, width4)),
        dtype=np.float64,
    )
    dense_bias = np.ascontiguousarray(
        rng.normal(0.0, 0.005, size=100), dtype=np.float64
    )
    dense17 = _layer(
        17,
        "DENSE",
        100,
        {
            "weight": dense_weight,
            "bias": dense_bias,
            "in_features": width4,
            "out_features": 100,
        },
    )
    assertion18 = _layer(18, "ASSERT", 100, {"kind": "AUDIT"})
    layers = [
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
    ]
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
    bounds = {
        int(layer.id): _box(int(layer.out_vars.__len__()))
        for layer in layers
        if layer.kind not in {"INPUT", "ASSERT"}
    }
    topology = _network(layers, predecessors)
    return topology, bounds


def _ancestor_relus(net, start_lid: Optional[int]) -> Tuple[int, ...]:
    if start_lid is None:
        assert_layer = next(
            layer for layer in net.layers if layer.kind == "ASSERT"
        )
        root = int(net.preds[int(assert_layer.id)][0])
    else:
        root = int(start_lid)
    seen = set()
    relus = []

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


def _query_schedule(net) -> Tuple[Mapping[str, Any], ...]:
    rng = np.random.default_rng(SEED + 1)
    stages = []
    for stage_index, (start_lid, count) in enumerate(SCHEDULE):
        if start_lid is None:
            width = 100
        else:
            width = len(net.by_id[start_lid].out_vars)
        rows = np.ascontiguousarray(
            rng.normal(0.0, 0.125, size=(count, width)),
            dtype=np.float64,
        )
        # Deterministically exclude structural zeros so the production-like
        # schedule exercises the dense Conv branch.
        rows[rows == 0.0] = np.nextafter(
            np.float64(0.0), np.float64(1.0)
        )
        alpha = {
            lid: np.asarray(0.5, dtype=np.float64)
            for lid in _ancestor_relus(net, start_lid)
        }
        stages.append(
            {
                "stage_index": stage_index,
                "start_lid": start_lid,
                "count": count,
                "width": width,
                "query_rows": rows,
                "alpha": alpha,
                "query_sha256": _array_sha256(rows),
            }
        )
    return tuple(stages)


def _rss_bytes(field: str = "VmRSS") -> int:
    status = Path("/proc/self/status").read_text(encoding="ascii")
    match = re.search(rf"^{re.escape(field)}:\s+(\d+)\s+kB$", status, re.M)
    if match is None:
        raise RuntimeError(f"/proc/self/status lacks {field}")
    return int(match.group(1)) * 1024


def _run_schedule(
    implementation: str,
    net,
    bounds,
    stages,
) -> Mapping[str, Any]:
    runner = (
        v3.replay_query_lower_bounds
        if implementation == "v3"
        else v5.replay_query_lower_bounds_v5_candidate
    )
    stage_records = []
    arrays = []
    start = time.perf_counter()
    for stage in stages:
        stage_start = time.perf_counter()
        result = runner(
            net,
            bounds,
            start_lid=stage["start_lid"],
            query_rows=stage["query_rows"],
            alpha_by_relu=stage["alpha"],
            chunk_size=64,
            max_workspace_bytes=512 * 1024 * 1024,
            timeout_s=120.0,
        )
        if implementation == "v3":
            if (
                result.proof_authority is not True
                or not v3.verify_query_dual_replay_receipt(result.receipt)
            ):
                raise RuntimeError("V3 controlled receipt failed validation")
        elif (
            result.proof_authority is not False
            or not v5.verify_query_dual_replay_v5_candidate(result)
        ):
            raise RuntimeError("V5 controlled receipt failed validation")
        elapsed = time.perf_counter() - stage_start
        values = np.asarray(result.lower_bounds)
        arrays.append(values.copy())
        stage_records.append(
            {
                "stage_index": stage["stage_index"],
                "start_lid": stage["start_lid"],
                "objective_count": stage["count"],
                "seconds": elapsed,
                "lower_bounds_sha256": _array_sha256(values),
                "guard_total_hex": result.receipt["stats"][
                    "guard_total_hex"
                ],
                "guard_max_hex": result.receipt["stats"]["guard_max_hex"],
                "conv_sparse_blocks": result.receipt["stats"][
                    "conv_sparse_blocks"
                ],
                "conv_dense_blocks": result.receipt["stats"][
                    "conv_dense_blocks"
                ],
            }
        )
    return {
        "implementation": implementation,
        "seconds": time.perf_counter() - start,
        "stages": stage_records,
        "arrays": tuple(arrays),
        "rss_bytes": _rss_bytes(),
        "hwm_bytes": _rss_bytes("VmHWM"),
    }


def _strip_arrays(run: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value for key, value in run.items() if key != "arrays"
    }


def _compare_pair(v3_run, v5_run) -> Mapping[str, Any]:
    regressions = []
    total = 0
    equal = 0
    improved = 0
    maximum_regression = 0.0
    for stage_index, (old, new) in enumerate(
        zip(v3_run["arrays"], v5_run["arrays"])
    ):
        if old.shape != new.shape:
            raise RuntimeError("controlled V3/V5 shape mismatch")
        total += int(old.size)
        mask = new < old
        for query_index in np.flatnonzero(mask):
            gap = float(old[query_index] - new[query_index])
            maximum_regression = max(maximum_regression, gap)
            regressions.append(
                {
                    "stage_index": stage_index,
                    "query_index": int(query_index),
                    "v3_hex": float(old[query_index]).hex(),
                    "v5_hex": float(new[query_index]).hex(),
                    "gap_hex": gap.hex(),
                }
            )
        equal += int(np.count_nonzero(new == old))
        improved += int(np.count_nonzero(new > old))
    return {
        "objective_count": total,
        "tightness_regression_count": len(regressions),
        "equal_count": equal,
        "improved_count": improved,
        "maximum_regression_hex": maximum_regression.hex(),
        "regression_preview": regressions[:16],
    }


def _bootstrap_lower(
    v3_seconds: Sequence[float],
    v5_seconds: Sequence[float],
) -> float:
    old = np.asarray(v3_seconds, dtype=np.float64)
    new = np.asarray(v5_seconds, dtype=np.float64)
    if old.shape != new.shape or old.size < 3:
        raise RuntimeError("paired bootstrap requires at least three pairs")
    rng = np.random.default_rng(SEED + 2)
    indices = rng.integers(0, old.size, size=(20_000, old.size))
    ratios = np.median(old[indices], axis=1) / np.median(
        new[indices], axis=1
    )
    return float(np.quantile(ratios, 0.025, method="lower"))


def _run_unittests(project_root: Path) -> Mapping[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-q",
        *TEST_MODULES,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
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
        timeout=180.0,
        check=False,
    )
    output = completed.stdout
    match = re.search(r"Ran (\d+) tests?", output)
    record = {
        "command": command,
        "module_count": len(TEST_MODULES),
        "test_count": int(match.group(1)) if match else None,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "output_tail": output[-4000:],
        "passed": completed.returncode == 0,
    }
    if not record["passed"] or record["test_count"] is None:
        raise RuntimeError("controlled unittest gate failed")
    return record


def _source_paths(project_root: Path) -> Tuple[Path, ...]:
    relative = (
        "act/back_end/hybridz_tf/query_dual_replay.py",
        "act/back_end/hybridz_tf/query_dual_scalar_guard.py",
        "act/back_end/hybridz_tf/query_dual_replay_v5_candidate.py",
        "act/back_end/hybridz_tf/query_dual_replay_v5.py",
        "act/back_end/hybridz_tf/query_dual_v5_authority.py",
        "act/back_end/hybridz_tf/test_query_dual_replay_v5.py",
        "act/pipeline/verification/query_dual_v5_controlled_audit.py",
    )
    return tuple(project_root / item for item in relative)


def _source_hashes(paths: Iterable[Path]) -> Mapping[str, str]:
    return {str(path): _file_sha256(path) for path in paths}


def _numeric_device_audit(project_root: Path) -> Mapping[str, Any]:
    """Prove the replay dependency surface contains no GPU backend import.

    ACT's package initializer may eagerly import ``torch.cuda`` for unrelated
    device management.  Presence in ``sys.modules`` is therefore ambient
    process state, not evidence that this NumPy replay used CUDA.  The
    controlled gate instead parses the complete numeric source set consumed
    by V5 and rejects any import of a GPU-capable tensor backend.
    """

    relative = (
        "act/back_end/hybridz_tf/query_dual_replay.py",
        "act/back_end/hybridz_tf/query_dual_scalar_guard.py",
        "act/back_end/hybridz_tf/query_dual_replay_v5_candidate.py",
        "act/back_end/hybridz_tf/query_dual_replay_v5.py",
    )
    forbidden_roots = {"cupy", "jax", "tensorflow"}
    allowed_torch_attributes = {
        "torch.Tensor",
        "torch.float64",
        "torch.__version__",
    }
    imports = set()
    torch_attributes = set()
    for item in relative:
        source = (project_root / item).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=item)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            ):
                torch_attributes.add(f"torch.{node.attr}")
    forbidden = sorted(
        name
        for name in imports
        if name.split(".", 1)[0] in forbidden_roots
    )
    forbidden_torch = sorted(
        value
        for value in torch_attributes
        if value not in allowed_torch_attributes
    )
    return {
        "method": "AST import closure of all V3/V5 numeric replay sources",
        "numeric_sources": list(relative),
        "forbidden_backend_roots": sorted(forbidden_roots),
        "forbidden_imports_found": forbidden,
        "torch_attributes_found": sorted(torch_attributes),
        "allowed_torch_attributes": sorted(allowed_torch_attributes),
        "forbidden_torch_attributes_found": forbidden_torch,
        "passed": not forbidden and not forbidden_torch,
        "runtime_array_backend": "numpy.ndarray",
        "ambient_torch_cuda_preloaded": any(
            name == "torch.cuda" or name.startswith("torch.cuda.")
            for name in sys.modules
        ),
    }


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_audit(
    *,
    project_root: Path,
    output: Path,
    pairs: int,
    blas_threads: int,
) -> Mapping[str, Any]:
    if pairs < 3:
        raise ValueError("V5 controlled gate requires at least three pairs")
    if blas_threads <= 0:
        raise ValueError("BLAS thread count must be positive")
    sources = _source_paths(project_root)
    before_hashes = _source_hashes(sources)
    v3_path = project_root / (
        "act/back_end/hybridz_tf/query_dual_replay.py"
    )
    if before_hashes[str(v3_path)] != EXPECTED_V3_SHA256:
        raise RuntimeError("frozen V3 replay source hash changed")
    unittest_record = _run_unittests(project_root)
    net, bounds = _build_topology()
    stages = _query_schedule(net)
    baseline_rss = _rss_bytes()
    baseline_hwm = _rss_bytes("VmHWM")
    pair_records = []
    old_seconds = []
    new_seconds = []
    total_regressions = 0
    maximum_rss = baseline_rss
    maximum_hwm = baseline_hwm

    with threadpool_limits(limits=blas_threads):
        # One warm execution per implementation is excluded from timing.
        warm_v3 = _run_schedule("v3", net, bounds, stages)
        warm_v5 = _run_schedule("v5", net, bounds, stages)
        warm_comparison = _compare_pair(warm_v3, warm_v5)
        if warm_comparison["tightness_regression_count"]:
            raise RuntimeError("V5 warmup has a tightness regression")
        for pair_index in range(pairs):
            order = (
                ("v3", "v5")
                if pair_index % 2 == 0
                else ("v5", "v3")
            )
            observed = {}
            for implementation in order:
                observed[implementation] = _run_schedule(
                    implementation, net, bounds, stages
                )
                maximum_rss = max(
                    maximum_rss, observed[implementation]["rss_bytes"]
                )
                maximum_hwm = max(
                    maximum_hwm, observed[implementation]["hwm_bytes"]
                )
            comparison = _compare_pair(
                observed["v3"], observed["v5"]
            )
            total_regressions += comparison[
                "tightness_regression_count"
            ]
            old_seconds.append(observed["v3"]["seconds"])
            new_seconds.append(observed["v5"]["seconds"])
            pair_records.append(
                {
                    "pair_index": pair_index,
                    "order": list(order),
                    "v3": _strip_arrays(observed["v3"]),
                    "v5": _strip_arrays(observed["v5"]),
                    "comparison": comparison,
                }
            )

    median_old = float(np.median(old_seconds))
    median_new = float(np.median(new_seconds))
    speedup = median_old / median_new
    bootstrap_lower = _bootstrap_lower(old_seconds, new_seconds)
    source_hashes_after = _source_hashes(sources)
    source_stable = before_hashes == source_hashes_after
    incremental_rss = max(0, maximum_rss - baseline_rss)
    incremental_hwm = max(0, maximum_hwm - baseline_hwm)
    device_audit = _numeric_device_audit(project_root)
    gates = {
        "unittests_passed": bool(unittest_record["passed"]),
        "fraction_objective_minimum": 5000,
        "production_objective_count": sum(
            stage["count"] for stage in stages
        ),
        "pair_count_at_least_3": pairs >= 3,
        "median_speedup_required": 2.0,
        "median_speedup_measured": speedup,
        "median_speedup_passed": speedup >= 2.0,
        "bootstrap_95_lower_required": 1.8,
        "bootstrap_95_lower_measured": bootstrap_lower,
        "bootstrap_95_lower_passed": bootstrap_lower >= 1.8,
        "tightness_regression_count": total_regressions,
        "tightness_passed": total_regressions == 0,
        "incremental_rss_limit_bytes": 2 * 1024**3,
        "incremental_rss_bytes": incremental_rss,
        "incremental_hwm_bytes": incremental_hwm,
        "memory_passed": max(incremental_rss, incremental_hwm)
        <= 2 * 1024**3,
        "workspace_limit_bytes": 512 * 1024 * 1024,
        "hidden_cuda_absent": bool(device_audit["passed"]),
        "source_hashes_stable": source_stable,
    }
    passed = bool(
        gates["unittests_passed"]
        and gates["pair_count_at_least_3"]
        and gates["median_speedup_passed"]
        and gates["bootstrap_95_lower_passed"]
        and gates["tightness_passed"]
        and gates["memory_passed"]
        and gates["hidden_cuda_absent"]
        and gates["source_hashes_stable"]
    )
    receipt: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if passed else "rejected",
        "proof_authority": False,
        "controlled_toy_only": True,
        "real_onnx_or_vnnlib_accessed": False,
        "operator_or_solver_imported": False,
        "seed": SEED,
        "numeric_protocol": v5.NUMERIC_PROTOCOL,
        "topology": {
            "input_shape": [128, 8, 8],
            "block_8x8": (
                "two 128x128 3x3 convolutions plus identity residual"
            ),
            "downsample": (
                "128x128 3x3 stride-2 main plus 128x128 1x1 "
                "stride-2 skip"
            ),
            "block_4x4": (
                "two 128x128 3x3 convolutions plus identity residual"
            ),
            "head": "flatten 128x4x4 then dense 100",
            "weights": "deterministic synthetic Gaussian",
            "bounds": "synthetic supplied-certified [-1,1] boxes",
            "alpha": 0.5,
        },
        "query_schedule": [
            {
                key: value
                for key, value in stage.items()
                if key not in {"query_rows", "alpha"}
            }
            for stage in stages
        ],
        "unittest_gate": unittest_record,
        "warmup_comparison": warm_comparison,
        "alternating_pairs": pair_records,
        "performance": {
            "blas_threads": blas_threads,
            "threadpools": threadpool_info(),
            "v3_seconds": old_seconds,
            "v5_seconds": new_seconds,
            "v3_median_seconds": median_old,
            "v5_median_seconds": median_new,
            "median_speedup": speedup,
            "paired_bootstrap_samples": 20_000,
            "paired_bootstrap_95_lower": bootstrap_lower,
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "maximum_rss_bytes": maximum_rss,
            "baseline_hwm_bytes": baseline_hwm,
            "maximum_hwm_bytes": maximum_hwm,
            "incremental_rss_bytes": incremental_rss,
            "incremental_hwm_bytes": incremental_hwm,
        },
        "device_audit": device_audit,
        "gates": gates,
        "source_sha256": before_hashes,
        "source_sha256_after": source_hashes_after,
        "decision": (
            "eligible_for_same-iid2-query-only-preregistration"
            if passed
            else "close-v5-without-real-probe"
        ),
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    _write_atomic(output, receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--blas-threads", type=int, default=8)
    arguments = parser.parse_args(argv)
    receipt = run_audit(
        project_root=arguments.project_root.resolve(),
        output=arguments.output.resolve(),
        pairs=arguments.pairs,
        blas_threads=arguments.blas_threads,
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
