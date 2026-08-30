#!/usr/bin/env python3
"""Stop-loss audit for C10 disjoint property-row CPU replay.

This command uses only a deterministic residual Conv toy.  It never loads an
ONNX model or VNNLIB and never emits a solver verdict.  One warmed serial/four
worker pair is the 1.50x admission gate.  Only an admitted candidate receives
four more paired trials; its final gate is median speedup >=2.00x, paired
bootstrap 95% lower bound >=1.80x, bit-identical lower bounds, and zero
receipt-validation regressions.
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
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    create_query_dual_replay_session,
    validate_query_dual_replay_result,
)
from act.pipeline.verification.query_dual_v3_toy_audit import (
    _wide_conv_resnet,
)
from act.util.device_manager import initialize_device


_SCHEMA = "act.query_dual_c10_cpu_parallel_audit.v1"
_FIRST_PAIR_GATE = 1.50
_MEDIAN_GATE = 2.00
_BOOTSTRAP_LOWER_GATE = 1.80
_RSS_GATE_BYTES = 2 * 1024 * 1024 * 1024
_SOURCE_FILES = (
    "act/back_end/hybridz_tf/query_dual_replay.py",
    "act/back_end/hybridz_tf/test_query_dual_replay_v3.py",
    "act/pipeline/verification/query_dual_c10_cpu_parallel_audit.py",
    "act/pipeline/verification/query_dual_v3_toy_audit.py",
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
        path: _file_sha256(Path(path).resolve())
        for path in _SOURCE_FILES
    }


def _atomic_json(
    path: Path, value: Mapping[str, Any], *, overwrite: bool
) -> None:
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
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _property_rows(output_width: int = 100) -> np.ndarray:
    rows = np.zeros((output_width - 1, output_width), dtype=np.float64)
    rows[:, 0] = 1.0
    rows[np.arange(output_width - 1), np.arange(1, output_width)] = -1.0
    rows.setflags(write=False)
    return rows


def _run_once(
    net: Any,
    certificate: Any,
    rows: np.ndarray,
    *,
    workers: int,
) -> Tuple[float, Any]:
    deadline = time.monotonic() + 120.0
    started = time.perf_counter()
    session = create_query_dual_replay_session(
        net, certificate, [None], deadline=deadline
    )
    frame = session.seal_bounds(
        certificate.bounds, start_lids=(None,)
    )
    session.replay(
        frame,
        query_rows=rows,
        chunk_size=1024,
        max_workspace_bytes=512 * 1024 * 1024,
        proof_workers=int(workers),
    )
    result = session.commit()[0]
    elapsed = time.perf_counter() - started
    if (
        not validate_query_dual_replay_result(result)
        or result.receipt["proof_row_parallelism"]["requested_workers"]
        != int(workers)
    ):
        raise RuntimeError("replay receipt validation regression")
    return elapsed, result


def _paired_bootstrap_lower(
    speedups: Sequence[float],
    *,
    samples: int = 20_000,
) -> float:
    values = np.asarray(speedups, dtype=np.float64)
    if values.size != 5 or not np.all(np.isfinite(values)):
        raise RuntimeError("paired bootstrap requires five finite speedups")
    rng = np.random.default_rng(2026072901)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    medians = np.median(values[indices], axis=1)
    return float(np.quantile(medians, 0.025, method="lower"))


def run_audit() -> Dict[str, Any]:
    started = time.monotonic()
    initialize_device(device="cpu", dtype="float64")
    source_before = _source_hashes()
    rss_before = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    )
    net, _, _ = _wide_conv_resnet(
        channels=32,
        height=16,
        width=16,
        output_width=100,
        seed=2026072901,
    )
    rows = _property_rows()
    certificate = certify_query_dual_boxes(
        net, conv_channel_chunk=4
    )

    # Warm both paths before the pre-registered first pair.
    warm_serial, warm_serial_result = _run_once(
        net, certificate, rows, workers=1
    )
    warm_parallel, warm_parallel_result = _run_once(
        net, certificate, rows, workers=4
    )
    if not np.array_equal(
        warm_serial_result.lower_bounds,
        warm_parallel_result.lower_bounds,
    ):
        raise RuntimeError("warm serial/parallel lower bounds differ")

    serial_seconds = []
    parallel_seconds = []
    bit_identical = []
    # Alternate order across pairs to avoid a systematic warm-cache bias.
    for pair_index in range(5):
        if pair_index % 2 == 0:
            serial, serial_result = _run_once(
                net, certificate, rows, workers=1
            )
            parallel, parallel_result = _run_once(
                net, certificate, rows, workers=4
            )
        else:
            parallel, parallel_result = _run_once(
                net, certificate, rows, workers=4
            )
            serial, serial_result = _run_once(
                net, certificate, rows, workers=1
            )
        serial_seconds.append(serial)
        parallel_seconds.append(parallel)
        identical = bool(
            np.array_equal(
                serial_result.lower_bounds,
                parallel_result.lower_bounds,
            )
            and serial_result.receipt["lower_bounds_sha256"]
            == parallel_result.receipt["lower_bounds_sha256"]
        )
        bit_identical.append(identical)
        first_speedup = serial_seconds[0] / parallel_seconds[0]
        if pair_index == 0 and (
            not identical or first_speedup < _FIRST_PAIR_GATE
        ):
            break

    speedups = [
        serial / parallel
        for serial, parallel in zip(serial_seconds, parallel_seconds)
    ]
    admitted = bool(
        speedups
        and bit_identical[0]
        and speedups[0] >= _FIRST_PAIR_GATE
    )
    median_speedup = (
        float(statistics.median(speedups)) if speedups else 0.0
    )
    bootstrap_lower = (
        _paired_bootstrap_lower(speedups)
        if len(speedups) == 5
        else None
    )
    source_after = _source_hashes()
    rss_after = (
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    )
    result: Dict[str, Any] = {
        "schema": _SCHEMA,
        "status": (
            "pass"
            if (
                admitted
                and len(speedups) == 5
                and all(bit_identical)
                and median_speedup >= _MEDIAN_GATE
                and bootstrap_lower is not None
                and bootstrap_lower >= _BOOTSTRAP_LOWER_GATE
                and source_before == source_after
                and rss_after - rss_before <= _RSS_GATE_BYTES
            )
            else "closed_by_stop_loss"
        ),
        "controlled_toy_only": True,
        "onnx_loaded": False,
        "vnnlib_loaded": False,
        "solver_verdict_called": False,
        "topology": {
            "kind": "two_conv_residual_add",
            "channels": 32,
            "height": 16,
            "width": 16,
            "classes": 100,
            "property_objectives": 99,
        },
        "thread_contract": {
            "taskset_cpu_count_required": 4,
            "serial_workers": 1,
            "parallel_workers": 4,
            "blas_threads_required": 1,
        },
        "gates": {
            "first_pair_minimum_speedup": _FIRST_PAIR_GATE,
            "five_pair_median_minimum_speedup": _MEDIAN_GATE,
            "paired_bootstrap_95_lower_minimum": _BOOTSTRAP_LOWER_GATE,
            "zero_result_regressions": True,
        },
        "warmup_seconds": {
            "serial": warm_serial,
            "parallel": warm_parallel,
        },
        "first_pair_admitted": admitted,
        "pairs_run": len(speedups),
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "paired_speedups": speedups,
        "median_speedup": median_speedup,
        "paired_bootstrap_95_lower": bootstrap_lower,
        "bit_identical_by_pair": bit_identical,
        "result_regressions": int(
            sum(not value for value in bit_identical)
        ),
        "source_integrity_stable": source_before == source_after,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "cpu_rss": {
            "before_bytes": rss_before,
            "after_peak_bytes": rss_after,
            "increment_bytes": rss_after - rss_before,
            "maximum_increment_bytes": _RSS_GATE_BYTES,
            "pass": rss_after - rss_before <= _RSS_GATE_BYTES,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    body = dict(result)
    body["receipt_sha256"] = _canonical_sha256(body)
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run_audit()
    _atomic_json(args.output, result, overwrite=args.overwrite)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
