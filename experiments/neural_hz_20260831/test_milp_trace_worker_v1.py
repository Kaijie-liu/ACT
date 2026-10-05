from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint

from experiments.neural_hz_20260831.milp_trace_worker_v1 import MilpTrace, problem_fingerprint


def problem():
    return {"c": np.zeros(2), "integrality": np.array([0, 1]),
            "bounds": Bounds([-1, 0], [1, 1]),
            "constraints": LinearConstraint([[1., 2.]], [-np.inf], [1.]),
            "options": {"time_limit": 45., "presolve": True, "mip_rel_gap": 0.}}


def test_numeric_digest_binds_values_not_just_shape_and_nnz():
    p = problem()
    before = problem_fingerprint(p)
    p["constraints"].A[0, 1] = 3.
    assert problem_fingerprint(p)["sha256"] != before["sha256"]
    assert problem_fingerprint(p)["shape"] == before["shape"]


@pytest.mark.parametrize("policy", ["auto", "one"])
def test_trace_returns_original_result_and_preserves_call_except_thread_option(policy):
    p = problem()
    original_options = dict(p["options"])
    result = SimpleNamespace(status=0, message="test", x=np.array([0., 0.]), mip_node_count=0)
    def solve(**kw):
        assert kw["c"] is p["c"] and kw["constraints"] is p["constraints"]
        assert kw["options"].get("threads") == (1 if policy == "one" else None)
        return result
    trace = MilpTrace(solve, policy)
    assert trace(**p) is result
    assert p["options"] == original_options
    assert trace.records[0]["problem_unchanged"]
    assert trace.records[0]["max_variable_bound_violation"] == 0.


def test_rejected_bound_point_is_only_observed_never_repaired():
    result = SimpleNamespace(status=0, message="test", x=np.array([-1.0000000000000002, 0.]), mip_node_count=0)
    trace = MilpTrace(lambda **kw: result, "auto")
    assert trace(**problem()).x[0] < -1.
    assert trace.records[0]["max_variable_bound_violation"] > 0.


def test_fatal_exception_propagates_with_a_trace():
    def stop(**kwargs):
        raise KeyboardInterrupt()
    trace = MilpTrace(stop, "auto")
    with pytest.raises(KeyboardInterrupt):
        trace(**problem())
    assert trace.records[0]["error"] == "KeyboardInterrupt"
