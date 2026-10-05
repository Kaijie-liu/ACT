"""Process-local numerical MILP trace; no HZ changes or verdict authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import sparse

from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256


def array_digest(value):
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def problem_fingerprint(kwargs):
    c = np.asarray(kwargs["c"])
    bounds = kwargs["bounds"]
    constraint = kwargs.get("constraints")
    fields = {"objective": array_digest(c),
              "integrality": array_digest(np.broadcast_to(kwargs["integrality"], c.shape)),
              "var_lb": array_digest(np.broadcast_to(bounds.lb, c.shape)),
              "var_ub": array_digest(np.broadcast_to(bounds.ub, c.shape))}
    if constraint is None:
        shape, nnz = (0, len(c)), 0
    else:
        matrix = sparse.csr_matrix(constraint.A, copy=True)
        fields.update(matrix_data=array_digest(matrix.data),
                      matrix_indices=array_digest(matrix.indices),
                      matrix_indptr=array_digest(matrix.indptr),
                      row_lb=array_digest(np.broadcast_to(constraint.lb, (matrix.shape[0],))),
                      row_ub=array_digest(np.broadcast_to(constraint.ub, (matrix.shape[0],))))
        shape, nnz = matrix.shape, matrix.nnz
    payload = {"fields": fields, "shape": list(shape), "nnz": int(nnz)}
    return {**payload, "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def thread_count():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Threads:"):
            return int(line.split()[1])
    raise ValueError("no process thread count")


class MilpTrace:
    def __init__(self, solve, policy):
        if policy not in {"auto", "one"}:
            raise ValueError("unknown thread policy")
        self.solve, self.policy, self.records = solve, policy, []

    def __call__(self, **kwargs):
        observed_at = time.monotonic()
        before = problem_fingerprint(kwargs)
        call = dict(kwargs)
        call["options"] = dict(kwargs.get("options") or {})
        if self.policy == "one":
            call["options"]["threads"] = 1
        record = {"index": len(self.records), "problem": before,
                  "options": dict(call["options"]), "policy": self.policy,
                  "threads_before_monitor": thread_count(), "monitor_threads": 1}
        done = threading.Event()
        observations = []
        def monitor():
            while not done.is_set():
                observations.append(thread_count())
                done.wait(0.05)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        cpu_started, wall_started = time.process_time(), time.monotonic()
        record["pre_solve_instrumentation_s"] = wall_started - observed_at
        result = None
        try:
            result = self.solve(**call)
            return result
        except BaseException as exc:
            record["error"] = type(exc).__name__
            raise
        finally:
            solve_finished = time.monotonic()
            record["solve_wall_s"] = solve_finished - wall_started
            record["solve_cpu_s"] = time.process_time() - cpu_started
            done.set()
            watcher.join()
            record["threads_after_monitor"] = thread_count()
            record["observed_peak_threads_including_monitor"] = max(observations, default=0)
            record["problem_unchanged"] = before == problem_fingerprint(kwargs)
            if result is not None:
                record["scipy_status"] = int(result.status)
                record["message"] = str(result.message)
                record["mip_node_count"] = int(getattr(result, "mip_node_count", 0) or 0)
                x = getattr(result, "x", None)
                if x is not None:
                    values = np.asarray(x)
                    record["solution_sha256"] = array_digest(values)
                    record["max_variable_bound_violation"] = float(max(
                        0.0, np.max(np.asarray(kwargs["bounds"].lb) - values),
                        np.max(values - np.asarray(kwargs["bounds"].ub)),
                    ))
            record["post_solve_instrumentation_s"] = time.monotonic() - solve_finished
            record["total_wrapper_wall_s"] = time.monotonic() - observed_at
            self.records.append(record)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--highs-thread-policy", choices=("auto", "one"), required=True)
    args, remaining = parser.parse_known_args()
    output = args.trace_output.absolute()
    if not output.is_relative_to(ROOT / "experiments/neural_hz_20260831/results"):
        raise ValueError("trace output outside isolated results")
    if os.path.lexists(output):
        raise FileExistsError(output)
    from act.back_end.solver import solver_hz
    from experiments.neural_hz_20260831 import shadow_worker_dtype_v2 as worker
    from scipy.optimize._highspy import _core
    original = solver_hz.milp
    trace = MilpTrace(original, args.highs_thread_policy)
    solver_hz.milp = trace
    sys.argv = [worker.__file__, *remaining]
    try:
        worker.main()
    finally:
        solver_hz.milp = original
        _atomic_exclusive_json(output, {
            "schema": "milp_trace_v1", "policy": args.highs_thread_policy,
            "formal_gain": 0, "diagnostic_only": True, "worker_provenance": worker._provenance(ROOT),
            "embedded_highs_version": _core._Highs().version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "trace_source_sha256": _sha256(Path(__file__)), "calls": trace.records,
        })


if __name__ == "__main__":
    main()
