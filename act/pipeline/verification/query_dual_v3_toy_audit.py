#!/usr/bin/env python3
"""Reproducible controlled gates for sealed sparse query-dual V3.

This command runs no ONNX, VNNLIB, Operator-HZ, solver verdict, or benchmark
instance.  It combines the deterministic soundness/tightness unittest suite
with two warmed CPU stop-loss measurements:

* five overlapping static replay cones over at least 32 MiB of Dense weights;
* full V2 versus K64 V3 on one fixed wide residual Conv toy.

Run directly (never through pytest):

    python -m act.pipeline.verification.query_dual_v3_toy_audit \
      --output artifacts/hybridz_largecls_gates/query_dual_v3_controlled_audit_20260728.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import tempfile
import time
from typing import Any, Dict, Iterable, Mapping, Sequence
import unittest

import numpy as np
import torch

from act.back_end.hybridz_tf import query_dual_replay as replay_module
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
)
from act.back_end.hybridz_tf.query_dual_pipeline import (
    build_verified_query_dual_feedback,
)
from act.back_end.hybridz_tf.query_dual_pipeline_v3 import (
    build_verified_query_dual_feedback_v3,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    create_query_dual_replay_session,
)
from act.back_end.hybridz_tf.test_query_dual_box_certifier import (
    _input_pair,
    _layer,
    _net,
)
from act.back_end.hybridz_tf.test_query_dual_pipeline import (
    _IntervalCandidateSolver,
)
from act.back_end.hybridz_tf.test_query_dual_pipeline_v3 import (
    _toy_selector,
)
from act.util.device_manager import initialize_device


_SCHEMA = "act.query_dual_v3_controlled_audit.v2"
_STATIC_SAVING_GATE = 0.60
_TRANSACTION_RATIO_GATE = 0.45
_RSS_GATE_BYTES = 2 * 1024 * 1024 * 1024
_STATIC_WIDTH = 916
_STATIC_DEPTH = 5
_STATIC_MINIMUM_WEIGHT_BYTES = 32 * 1024 * 1024
_SOURCE_FILES = (
    "act/pipeline/verification/query_dual_v3_toy_audit.py",
    "act/back_end/hybridz_tf/query_dual_box_certifier.py",
    "act/back_end/hybridz_tf/query_dual_replay.py",
    "act/back_end/hybridz_tf/property_residual_targets.py",
    "act/back_end/hybridz_tf/query_dual_candidates.py",
    "act/back_end/hybridz_tf/query_dual_pipeline.py",
    "act/back_end/hybridz_tf/query_dual_pipeline_v3.py",
    "act/back_end/hybridz_tf/test_query_dual_box_certifier.py",
    "act/back_end/hybridz_tf/test_query_dual_replay.py",
    "act/back_end/hybridz_tf/test_query_dual_replay_v3.py",
    "act/back_end/hybridz_tf/test_property_residual_targets.py",
    "act/back_end/hybridz_tf/test_query_dual_candidates.py",
    "act/back_end/hybridz_tf/test_query_dual_pipeline.py",
    "act/back_end/hybridz_tf/test_query_dual_pipeline_v3.py",
    "act/pipeline/verification/query_dual_v3_cuda_toy_audit.py",
    "act/pipeline/verification/test_query_dual_v3_cuda_toy_audit.py",
    "act/pipeline/verification/query_dual_probe.py",
    "act/pipeline/verification/test_query_dual_probe.py",
    "act/pipeline/verification/test_query_dual_v3_toy_audit.py",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> Dict[str, str]:
    return {
        path: _file_sha256(Path(path).resolve()) for path in _SOURCE_FILES
    }


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"refusing to overwrite existing audit {path}"
                ) from exc
            temporary.unlink()
            temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _deep_dense_net(
    width: int = _STATIC_WIDTH, depth: int = _STATIC_DEPTH
):
    width = int(width)
    depth = int(depth)
    inp, spec = _input_pair(
        width,
        np.full(width, -1.0, dtype=np.float64),
        np.full(width, 1.0, dtype=np.float64),
    )
    layers = [inp, spec]
    preds = {0: [], 1: [0]}
    previous = 1
    starts = []
    rng = np.random.default_rng(2026072801)
    next_id = 2
    for _ in range(depth):
        weight = rng.normal(
            0.0, 0.01, size=(width, width)
        ).astype(np.float64)
        dense = _layer(
            next_id,
            "DENSE",
            width,
            {
                "weight": weight,
                "bias": np.zeros(width, dtype=np.float64),
            },
        )
        layers.append(dense)
        preds[next_id] = [previous]
        starts.append(next_id)
        previous = next_id
        next_id += 1
        relu = _layer(next_id, "RELU", width)
        layers.append(relu)
        preds[next_id] = [previous]
        previous = next_id
        next_id += 1
    assertion = _layer(next_id, "ASSERT", width, {"kind": "AUDIT"})
    layers.append(assertion)
    preds[next_id] = [previous]
    return _net(layers, preds), tuple(starts)


def _wide_conv_resnet(
    *,
    channels: int = 4,
    height: int = 8,
    width: int = 8,
    output_width: int = 16,
    seed: int = 2026072802,
):
    rng = np.random.default_rng(int(seed))
    channels, height, width = int(channels), int(height), int(width)
    output_width = int(output_width)
    flat_width = channels * height * width
    inp, spec = _input_pair(
        flat_width,
        np.full(flat_width, -1.0, dtype=np.float64),
        np.full(flat_width, 1.0, dtype=np.float64),
    )

    def conv(layer_id: int, weight: np.ndarray):
        return _layer(
            layer_id,
            "CONV2D",
            flat_width,
            {
                "weight": weight,
                "bias": np.zeros(channels, dtype=np.float64),
                "input_shape": (channels, height, width),
                "output_shape": (channels, height, width),
                "stride": (1, 1),
                "padding": (1, 1),
                "dilation": (1, 1),
                "groups": 1,
                "padding_mode": "zeros",
            },
        )

    conv1 = conv(
        2,
        rng.normal(
            0.0, 0.04, size=(channels, channels, 3, 3)
        ).astype(np.float64),
    )
    relu1 = _layer(3, "RELU", flat_width)
    conv2 = conv(
        4,
        rng.normal(
            0.0, 0.04, size=(channels, channels, 3, 3)
        ).astype(np.float64),
    )
    add = _layer(5, "ADD", flat_width)
    relu2 = _layer(6, "RELU", flat_width)
    flatten = _layer(
        7,
        "FLATTEN",
        flat_width,
        {
            "start_dim": 1,
            "end_dim": -1,
            "input_shape": (channels, height, width),
            "output_shape": (flat_width,),
        },
    )
    dense = _layer(
        8,
        "DENSE",
        output_width,
        {
            "weight": rng.normal(
                0.0, 0.03, size=(output_width, flat_width)
            ).astype(np.float64),
            "bias": np.zeros(output_width, dtype=np.float64),
        },
    )
    assertion = _layer(9, "ASSERT", output_width, {"kind": "AUDIT"})
    net = _net(
        [
            inp,
            spec,
            conv1,
            relu1,
            conv2,
            add,
            relu2,
            flatten,
            dense,
            assertion,
        ],
        {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [4, 1],
            6: [5],
            7: [6],
            8: [7],
            9: [8],
        },
    )
    return (
        net,
        np.eye(output_width, dtype=np.float64),
        np.zeros(output_width, dtype=np.float64),
    )


def _median(values: Iterable[float]) -> float:
    result = float(statistics.median(tuple(float(value) for value in values)))
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError("non-finite/non-positive benchmark median")
    return result


def _static_prepare_benchmark(repeats: int = 7) -> Dict[str, Any]:
    net, starts = _deep_dense_net()
    certificate = certify_query_dual_boxes(net)
    deadline = time.monotonic() + 120.0

    def legacy_once() -> float:
        started = time.perf_counter()
        for start_lid in starts:
            timer = replay_module._Deadline(end=deadline)
            replay_module._prepare(
                net,
                certificate.bounds,
                start_lid=start_lid,
                query_rows=np.zeros(
                    (1, _STATIC_WIDTH), dtype=np.float64
                ),
                one_hot=None,
                query_bias=None,
                alpha_by_relu=None,
                deadline=timer,
                expected_net_sha256=None,
                expected_bounds_sha256=None,
                expected_query_sha256=None,
                expected_alpha_sha256=None,
            )
        return time.perf_counter() - started

    def sealed_once() -> float:
        started = time.perf_counter()
        session = create_query_dual_replay_session(
            net,
            certificate,
            starts,
            deadline=deadline,
        )
        elapsed = time.perf_counter() - started
        session.abort()
        return elapsed

    legacy_once()
    sealed_once()
    legacy = []
    sealed = []
    for index in range(int(repeats)):
        # Alternate which implementation runs first so clock drift, cache
        # warmth, and thermal effects cannot systematically favor V3.
        if index % 2 == 0:
            legacy.append(legacy_once())
            sealed.append(sealed_once())
        else:
            sealed.append(sealed_once())
            legacy.append(legacy_once())
    legacy_median = _median(legacy)
    sealed_median = _median(sealed)
    ratio = sealed_median / legacy_median
    weight_bytes = _STATIC_DEPTH * _STATIC_WIDTH**2 * 8
    return {
        "weight_bytes": weight_bytes,
        "minimum_weight_bytes": _STATIC_MINIMUM_WEIGHT_BYTES,
        "overlapping_cones": len(starts),
        "repeats": int(repeats),
        "legacy_seconds": legacy,
        "sealed_seconds": sealed,
        "legacy_median_seconds": legacy_median,
        "sealed_median_seconds": sealed_median,
        "ratio": ratio,
        "saving_fraction": 1.0 - ratio,
        "required_saving_fraction": _STATIC_SAVING_GATE,
        "pass": (
            weight_bytes >= _STATIC_MINIMUM_WEIGHT_BYTES
            and (1.0 - ratio) >= _STATIC_SAVING_GATE
        ),
    }


def _transaction_benchmark(repeats: int = 3) -> Dict[str, Any]:
    net, rows, thresholds = _wide_conv_resnet()

    def v2_once() -> float:
        started = time.perf_counter()
        bundle = build_verified_query_dual_feedback(
            net,
            rows,
            thresholds,
            target_relu_ids=(6,),
            steps=2,
            block_size=1024,
            replay_chunk_size=1024,
            conv_channel_chunk=2,
            candidate_device="cpu",
            timeout_s=30.0,
            solver_factory=_IntervalCandidateSolver,
        )
        elapsed = time.perf_counter() - started
        if bundle.stages[0].strict_improvements != 256:
            raise RuntimeError("wide V2 toy did not cover all 256 target rows")
        return elapsed

    def v3_once() -> float:
        started = time.perf_counter()
        bundle = build_verified_query_dual_feedback_v3(
            net,
            rows,
            thresholds,
            target_relu_ids=(6,),
            stage_quotas=(64,),
            steps=2,
            block_size=1024,
            replay_chunk_size=1024,
            conv_channel_chunk=2,
            candidate_device="cpu",
            timeout_s=30.0,
            selector=_toy_selector,
            solver_factory=_IntervalCandidateSolver,
        )
        elapsed = time.perf_counter() - started
        if (
            bundle.stages[0].receipt["selected_row_ids"]
            != bundle.stages[0].candidate_receipt[
                "selected_target_row_ids"
            ]
            or bundle.stages[0].strict_improvements != 64
        ):
            raise RuntimeError("wide V3 toy did not preserve its K64 schedule")
        return elapsed

    v2_once()
    v3_once()
    v2 = []
    v3 = []
    for index in range(int(repeats)):
        # As above, alternate order instead of always making V3 the second,
        # warmer measurement.
        if index % 2 == 0:
            v2.append(v2_once())
            v3.append(v3_once())
        else:
            v3.append(v3_once())
            v2.append(v2_once())
    v2_median = _median(v2)
    v3_median = _median(v3)
    ratio = v3_median / v2_median
    return {
        "toy": "fixed_4x8x8_two_conv_residual_add",
        "eligible_target_rows": 256,
        "v2_target_objectives": 512,
        "v3_selected_target_rows": 64,
        "v3_target_objectives": 128,
        "property_objectives": 16,
        "repeats": int(repeats),
        "v2_seconds": v2,
        "v3_seconds": v3,
        "v2_median_seconds": v2_median,
        "v3_median_seconds": v3_median,
        "ratio": ratio,
        "required_max_ratio": _TRANSACTION_RATIO_GATE,
        "pass": ratio <= _TRANSACTION_RATIO_GATE,
    }


def _run_soundness_suite() -> Dict[str, Any]:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in (
        "act.back_end.hybridz_tf.test_query_dual_box_certifier",
        "act.back_end.hybridz_tf.test_query_dual_replay",
        "act.back_end.hybridz_tf.test_query_dual_replay_v3",
        "act.back_end.hybridz_tf.test_property_residual_targets",
        "act.back_end.hybridz_tf.test_query_dual_candidates",
        "act.back_end.hybridz_tf.test_query_dual_pipeline",
        "act.back_end.hybridz_tf.test_query_dual_pipeline_v3",
        "act.pipeline.verification.test_query_dual_v3_cuda_toy_audit",
        "act.pipeline.verification.test_query_dual_probe",
    ):
        suite.addTests(loader.loadTestsFromName(module))
    result = unittest.TestResult()
    suite.run(result)
    return {
        "tests_run": int(result.testsRun),
        "failures": [
            {"test": str(test), "traceback": traceback[-4000:]}
            for test, traceback in result.failures
        ],
        "errors": [
            {"test": str(test), "traceback": traceback[-4000:]}
            for test, traceback in result.errors
        ],
        "skipped": [
            {"test": str(test), "reason": str(reason)}
            for test, reason in result.skipped
        ],
        "fraction_dag_conv_objectives": 1000,
        "fraction_oracle_overestimate_violations": (
            0 if result.wasSuccessful() else None
        ),
        "residual_quarter_gate": (
            0.2500001 if result.wasSuccessful() else None
        ),
        "k64_minimum_full_gain_recovery": (
            0.80 if result.wasSuccessful() else None
        ),
        "pass": bool(result.wasSuccessful()),
    }


def run_audit() -> Dict[str, Any]:
    started = time.monotonic()
    source_before: Dict[str, str] = {}
    source_after: Dict[str, str] = {}
    soundness: Dict[str, Any] = {}
    static_prepare: Dict[str, Any] = {}
    transaction: Dict[str, Any] = {}
    rss_before = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    )
    try:
        initialize_device(device="cpu", dtype="float64")
        source_before = _source_hashes()
        soundness = _run_soundness_suite()
        if not soundness.get("pass", False):
            raise RuntimeError(
                "SOUNDNESS_TIGHTNESS_GATE_FAILED: performance benchmarks "
                "were skipped"
            )
        static_prepare = _static_prepare_benchmark()
        if not static_prepare.get("pass", False):
            raise RuntimeError(
                "STATIC_PREPARE_GATE_FAILED: transaction benchmark was "
                "skipped"
            )
        transaction = _transaction_benchmark()
        source_after = _source_hashes()
        source_stable = source_after == source_before
        rss_after = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        )
        rss_increment = max(0, rss_after - rss_before)
        passed = bool(
            soundness["pass"]
            and static_prepare["pass"]
            and transaction["pass"]
            and source_stable
            and rss_increment <= _RSS_GATE_BYTES
        )
        body: Dict[str, Any] = {
            "schema": _SCHEMA,
            "status": "pass" if passed else "fail",
            "proof_authority": False,
            "controlled_toy_only": True,
            "real_model_run": False,
            "onnx_loaded": False,
            "vnnlib_loaded": False,
            "operator_hz_called": False,
            "solver_verdict_called": False,
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "source_sha256": source_after,
            "source_integrity_stable": source_stable,
            "soundness_tightness": soundness,
            "static_prepare": static_prepare,
            "wide_conv_transaction": transaction,
            "timing_policy": (
                "warmed_interleaved_alternating_order_medians"
            ),
            "cpu_parallelism": {
                "transaction_workers": 1,
                "worker_policy": (
                    "single_sequential_proof_worker_with_internal_"
                    "tensor_parallelism"
                ),
                "torch_intraop_threads": int(torch.get_num_threads()),
                "torch_interop_threads": int(
                    torch.get_num_interop_threads()
                ),
                "logical_cpu_count": os.cpu_count(),
            },
            "cpu_rss": {
                "before_bytes": rss_before,
                "after_peak_bytes": rss_after,
                "increment_bytes": rss_increment,
                "maximum_increment_bytes": _RSS_GATE_BYTES,
                "pass": rss_increment <= _RSS_GATE_BYTES,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        try:
            source_after = _source_hashes()
        except Exception:
            source_after = {}
        rss_after = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        )
        rss_increment = max(0, rss_after - rss_before)
        source_stable = bool(
            source_before
            and source_after
            and source_before == source_after
        )
        body = {
            "schema": _SCHEMA,
            "status": "fail",
            "proof_authority": False,
            "controlled_toy_only": True,
            "real_model_run": False,
            "onnx_loaded": False,
            "vnnlib_loaded": False,
            "operator_hz_called": False,
            "solver_verdict_called": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            },
            "source_sha256_before": source_before,
            "source_sha256_after": source_after,
            "source_integrity_stable": source_stable,
            "cpu_rss": {
                "before_bytes": rss_before,
                "after_peak_bytes": rss_after,
                "increment_bytes": rss_increment,
                "maximum_increment_bytes": _RSS_GATE_BYTES,
                "pass": rss_increment <= _RSS_GATE_BYTES,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        if soundness:
            body["soundness_tightness"] = soundness
        if static_prepare:
            body["static_prepare"] = static_prepare
        if transaction:
            body["wide_conv_transaction"] = transaction
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run_audit()
    _atomic_json(args.output, result, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_sha256": result["receipt_sha256"],
                "tests_run": result.get(
                    "soundness_tightness", {}
                ).get("tests_run"),
                "static_ratio": result.get("static_prepare", {}).get(
                    "ratio"
                ),
                "transaction_ratio": result.get(
                    "wide_conv_transaction", {}
                ).get("ratio"),
                "error": result.get("error"),
                "elapsed_seconds": result["elapsed_seconds"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
