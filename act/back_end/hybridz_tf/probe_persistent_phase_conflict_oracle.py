#!/usr/bin/env python3
"""Reproduce and validate the fixed K7 persistent PC-PCC speed gate."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import highspy
import numpy as np
import scipy
import scipy.sparse as sp


_SCHEMA = "act.pc_pcc_persistent_pair_gate.v2"
_MEDIAN_SPEEDUP_MIN = 2.0
_BOOTSTRAP_LOWER_MIN = 1.8
_EXPECTED_PAIRS = 5
_EXPECTED_WARMUPS = 1
_BOOTSTRAP_SAMPLES = 200_000
_BOOTSTRAP_SEED = 20_260_730
_SOURCE_NAMES = (
    "adaptive_phase_forest.py",
    "persistent_phase_conflict_oracle.py",
    "property_phase_conflict_clique.py",
    "test_persistent_phase_conflict_oracle.py",
    "test_property_phase_conflict_clique.py",
    "probe_persistent_phase_conflict_oracle.py",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _format_affinity(cpus: list[int]) -> str:
    if not cpus:
        return ""
    ranges = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(
            str(start)
            if start == previous
            else f"{start}-{previous}"
        )
        start = previous = cpu
    ranges.append(
        str(start) if start == previous else f"{start}-{previous}"
    )
    return ",".join(ranges)


def _bootstrap_lower(
    speedups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        int(speedups.size),
        size=(samples, int(speedups.size)),
    )
    medians = np.median(speedups[indices], axis=1)
    return float(np.quantile(medians, 0.025, method="lower"))


def _load_act_symbols(root: Path) -> Mapping[str, Any]:
    # Several legacy ACT imports print environment diagnostics.  Preserve
    # those on stderr so stdout remains one machine-readable JSON document.
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    with redirect_stdout(sys.stderr):
        from act.back_end.hybridz_tf.adaptive_phase_forest import (
            ordered_property_digest,
            sparse_hz_semantic_digest,
        )
        from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
            make_persistent_pc_pcc_invocation_spec,
            run_persistent_pc_pcc_candidate,
            verify_persistent_pc_pcc_result,
        )
        from act.back_end.hybridz_tf.property_phase_conflict_clique import (
            run_pc_pcc_candidate,
            verify_pc_pcc_result,
        )
        from act.back_end.hybridz_tf import (
            test_property_phase_conflict_clique as pc_pcc_toys,
        )
    return {
        "ordered_property_digest": ordered_property_digest,
        "sparse_hz_semantic_digest": sparse_hz_semantic_digest,
        "make_invocation": (
            make_persistent_pc_pcc_invocation_spec
        ),
        "run_persistent": run_persistent_pc_pcc_candidate,
        "verify_persistent": verify_persistent_pc_pcc_result,
        "run_legacy": run_pc_pcc_candidate,
        "verify_legacy": verify_pc_pcc_result,
        "complete_c49": pc_pcc_toys._complete_c49,
        "lp_upper": pc_pcc_toys._lp_upper,
        "rivals": pc_pcc_toys._rivals,
    }


def _legacy_once(
    parent,
    rivals,
    symbols: Mapping[str, Any],
    *,
    gate_id: str,
):
    del gate_id
    deadline = time.monotonic() + 30.0
    started = time.perf_counter()
    result = symbols["run_legacy"](
        parent,
        rivals,
        deadline=deadline,
    )
    elapsed = time.perf_counter() - started
    verified = symbols["verify_legacy"](
        parent, rivals, result
    )
    upper = (
        symbols["lp_upper"](result.hz)
        if verified and result.hz is not None
        else None
    )
    valid = bool(
        verified
        and result.status == "unknown_to_safe_candidate"
        and upper is not None
        and abs(upper - 1.0) <= 1.0e-9
    )
    return elapsed, valid


def _persistent_once(
    parent,
    rivals,
    symbols: Mapping[str, Any],
    *,
    gate_id: str,
):
    invocation = symbols["make_invocation"](
        parent,
        rivals,
        deadline=time.monotonic() + 30.0,
        gate_id=gate_id,
    )
    started = time.perf_counter()
    result = symbols["run_persistent"](
        parent,
        rivals,
        invocation=invocation,
    )
    elapsed = time.perf_counter() - started
    verified = symbols["verify_persistent"](
        parent,
        rivals,
        result,
        invocation=invocation,
    )
    upper = (
        symbols["lp_upper"](result.hz)
        if verified and result.hz is not None
        else None
    )
    valid = bool(
        verified
        and result.status == "unknown_to_safe_candidate"
        and upper is not None
        and abs(upper - 1.0) <= 1.0e-9
        and result.oracle_result.telemetry.get("threads") == 1
    )
    return elapsed, valid


def _run_gate(root: Path) -> Mapping[str, Any]:
    symbols = _load_act_symbols(root)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 4:
        raise RuntimeError(
            "official gate requires exactly four affinity-bound CPUs"
        )
    if os.environ.get("HZ_MILP_THREADS") != "1":
        raise RuntimeError("official gate requires HZ_MILP_THREADS=1")

    parent = symbols["complete_c49"](7)
    rivals = symbols["rivals"]()
    base_upper = symbols["lp_upper"](parent)
    if abs(base_upper - 7.0 / 3.0) > 1.0e-9:
        raise RuntimeError("K7 baseline tightness changed")

    # One discarded alternating warm-up pair.
    warmup_results = (
        _legacy_once(
            parent,
            rivals,
            symbols,
            gate_id="warmup_legacy",
        ),
        _persistent_once(
            parent,
            rivals,
            symbols,
            gate_id="warmup_persistent",
        ),
    )
    if not all(valid for _elapsed, valid in warmup_results):
        raise RuntimeError("warm-up result validation failed")

    legacy_seconds = []
    persistent_seconds = []
    timing_order = []
    fallbacks = 0
    for pair_index in range(_EXPECTED_PAIRS):
        if pair_index % 2 == 0:
            order = ("legacy", "persistent")
        else:
            order = ("persistent", "legacy")
        timing_order.append(list(order))
        observed = {}
        for implementation in order:
            gate_id = f"pair_{pair_index}_{implementation}"
            if implementation == "legacy":
                elapsed, valid = _legacy_once(
                    parent,
                    rivals,
                    symbols,
                    gate_id=gate_id,
                )
            else:
                elapsed, valid = _persistent_once(
                    parent,
                    rivals,
                    symbols,
                    gate_id=gate_id,
                )
            observed[implementation] = float(elapsed)
            if not valid:
                fallbacks += 1
        legacy_seconds.append(observed["legacy"])
        persistent_seconds.append(observed["persistent"])

    legacy = np.asarray(legacy_seconds, dtype=np.float64)
    persistent = np.asarray(
        persistent_seconds, dtype=np.float64
    )
    speedups = legacy / persistent
    median_speedup = float(np.median(speedups))
    bootstrap_lower = _bootstrap_lower(
        speedups,
        samples=_BOOTSTRAP_SAMPLES,
        seed=_BOOTSTRAP_SEED,
    )
    median_legacy = float(np.median(legacy))
    median_persistent = float(np.median(persistent))

    module_dir = Path(__file__).resolve().parent
    source_hashes = {
        name: _sha256_file(module_dir / name)
        for name in _SOURCE_NAMES
    }
    upper = sp.hstack(
        [parent.Auc, parent.Aub], format="csr"
    )
    equality = sp.hstack(
        [parent.Ac, parent.Ab], format="csr"
    )
    dimensions = {
        "n_out": int(parent.n_out),
        "n_continuous": int(parent.n_cont),
        "n_binary": int(parent.n_bin),
        "n_upper": int(parent.n_ub),
        "n_equality": int(parent.n_eq),
        "constraint_nonzeros": int(upper.nnz + equality.nnz),
    }
    passed = bool(
        fallbacks == 0
        and median_speedup >= _MEDIAN_SPEEDUP_MIN
        and bootstrap_lower >= _BOOTSTRAP_LOWER_MIN
    )
    return {
        "schema": _SCHEMA,
        "date": date.today().isoformat(),
        "git_head": _git_head(root),
        "source_sha256": source_hashes,
        "generator": {
            "module": (
                "act.back_end.hybridz_tf."
                "probe_persistent_phase_conflict_oracle"
            ),
            "command": (
                "env CUDA_VISIBLE_DEVICES='' HZ_MILP_THREADS=1 "
                "taskset -c 8-11 python "
                "act/back_end/hybridz_tf/"
                "probe_persistent_phase_conflict_oracle.py"
            ),
            "validator_argument": "--validate-artifact",
        },
        "parent_semantic_digest": symbols[
            "sparse_hz_semantic_digest"
        ](parent),
        "property_digest": symbols[
            "ordered_property_digest"
        ](rivals),
        "dimensions": dimensions,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "highs": "{}.{}.{}".format(
                highspy.HIGHS_VERSION_MAJOR,
                highspy.HIGHS_VERSION_MINOR,
                highspy.HIGHS_VERSION_PATCH,
            ),
        },
        "cpu_affinity": _format_affinity(affinity),
        "affinity_core_count": len(affinity),
        "highs_threads": 1,
        "invocation_spec_prebuilt": True,
        "pairs": _EXPECTED_PAIRS,
        "warmup_pairs": _EXPECTED_WARMUPS,
        "timing_order": timing_order,
        "bootstrap_samples": _BOOTSTRAP_SAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "legacy_seconds": [
            round(float(value), 9) for value in legacy
        ],
        "persistent_seconds": [
            round(float(value), 9) for value in persistent
        ],
        "legacy_median_seconds": round(median_legacy, 9),
        "persistent_median_seconds": round(
            median_persistent, 9
        ),
        "paired_speedups": [
            round(float(value), 6) for value in speedups
        ],
        "median_paired_speedup": round(
            median_speedup, 6
        ),
        "paired_bootstrap_95_lower": round(
            bootstrap_lower, 6
        ),
        "result_fallbacks": int(fallbacks),
        "thresholds": {
            "median_paired_speedup_min": (
                _MEDIAN_SPEEDUP_MIN
            ),
            "paired_bootstrap_95_lower_min": (
                _BOOTSTRAP_LOWER_MIN
            ),
            "result_fallbacks_max": 0,
        },
        "tightness": {
            "toy": "K7 C49",
            "before": round(base_upper, 12),
            "after": 1.0,
        },
        "passed": passed,
    }


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    root: Path,
    check_live_sources: bool,
) -> None:
    if payload.get("schema") != _SCHEMA:
        raise ValueError("artifact schema mismatch")
    if payload.get("pairs") != _EXPECTED_PAIRS:
        raise ValueError("artifact pair count mismatch")
    legacy = np.asarray(
        payload.get("legacy_seconds"), dtype=np.float64
    )
    persistent = np.asarray(
        payload.get("persistent_seconds"), dtype=np.float64
    )
    speedups = np.asarray(
        payload.get("paired_speedups"), dtype=np.float64
    )
    if (
        legacy.shape != (_EXPECTED_PAIRS,)
        or persistent.shape != (_EXPECTED_PAIRS,)
        or speedups.shape != (_EXPECTED_PAIRS,)
        or np.any(legacy <= 0.0)
        or np.any(persistent <= 0.0)
        or not np.all(np.isfinite(legacy))
        or not np.all(np.isfinite(persistent))
        or not np.all(np.isfinite(speedups))
    ):
        raise ValueError("artifact timing arrays malformed")
    if not np.allclose(
        legacy / persistent,
        speedups,
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise ValueError("artifact paired speedups inconsistent")
    median_speedup = float(np.median(speedups))
    lower = _bootstrap_lower(
        speedups,
        samples=int(payload.get("bootstrap_samples")),
        seed=int(payload.get("bootstrap_seed")),
    )
    if abs(
        median_speedup
        - float(payload.get("median_paired_speedup"))
    ) > 2.0e-6:
        raise ValueError("artifact median speedup inconsistent")
    if abs(
        lower
        - float(payload.get("paired_bootstrap_95_lower"))
    ) > 2.0e-6:
        raise ValueError("artifact bootstrap lower inconsistent")
    expected_pass = bool(
        int(payload.get("result_fallbacks")) == 0
        and median_speedup >= _MEDIAN_SPEEDUP_MIN
        and lower >= _BOOTSTRAP_LOWER_MIN
    )
    if payload.get("passed") is not expected_pass:
        raise ValueError("artifact passed flag inconsistent")
    if check_live_sources:
        module_dir = Path(__file__).resolve().parent
        expected_hashes = payload.get("source_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError("artifact source hashes missing")
        live_hashes = {
            name: _sha256_file(module_dir / name)
            for name in _SOURCE_NAMES
        }
        if expected_hashes != live_hashes:
            raise ValueError("artifact source hashes are stale")
        if payload.get("git_head") != _git_head(root):
            raise ValueError("artifact git head is stale")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-artifact",
        type=Path,
        help="validate an existing JSON artifact without rerunning",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.validate_artifact is not None:
        payload = json.loads(
            args.validate_artifact.read_text(encoding="utf-8")
        )
        _validate_payload(
            payload, root=root, check_live_sources=True
        )
        print(
            json.dumps(
                {
                    "artifact": str(
                        args.validate_artifact.resolve()
                    ),
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    payload = _run_gate(root)
    _validate_payload(
        payload, root=root, check_live_sources=True
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
