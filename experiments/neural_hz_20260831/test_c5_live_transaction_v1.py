from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch

from act.back_end.core import Bounds, Fact, ConSet, Con
from experiments.neural_hz_20260831.c5_live_roots_v1 import collect
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import transaction, measured_build
from experiments.neural_hz_20260831.c5_native_budgeted_materialization_v1 import BudgetedMaterializer
from experiments.neural_hz_20260831.test_c5_native_budgeted_materialization_v1 import fixture


def context():
    expr, rows = fixture()
    bounds = Bounds(torch.zeros(expr.n_out), torch.ones(expr.n_out))
    cons = ConSet()
    cons.replace(Con("INEQ", (0,), {"tag": "test", "bound": torch.ones(1)}))
    fact = Fact(bounds, cons)
    tf = SimpleNamespace(cache={0: expr.terms[0].source}, expressions={1: expr}, before={1: fact}, after={0: fact},
                         frames={7: (2, 2)}, slots={(7, 1, 0): (0, 0, 0)}, arena=weakref.WeakValueDictionary())
    return tf, expr, rows


def test_all_fact_constraints_slots_and_payloads_are_in_fingerprint():
    tf, expr, rows = context()
    first = collect(tf)
    assert first.schema_counts["Fact"] == 1 and first.schema_counts["Con"] == 1
    assert len(first.numeric) >= 4 and first.measure().resident_bytes > 0
    tf.before[1].cons.S[("INEQ", (0,), "test")].meta["bound"][0] = 2.
    assert collect(tf).fingerprint != first.fingerprint


@pytest.mark.parametrize("mode", ["extra_field", "unknown", "cycle"])
def test_unknown_live_schema_and_cycles_reject(mode):
    tf, expr, rows = context()
    if mode == "extra_field":
        tf.before[1].hidden = np.ones(3)
    elif mode == "unknown":
        tf.unknown = SimpleNamespace(value=np.ones(3))
    else:
        tf.cycle = []
        tf.cycle.append(tf.cycle)
    with pytest.raises(ValueError):
        collect(tf)


@pytest.mark.parametrize("fault", [0, 1, "after_build"])
def test_failures_after_every_branch_and_before_return_do_not_publish(fault):
    tf, expr, rows = context()
    before = collect(tf).fingerprint
    budget = BudgetedMaterializer(2)
    published = []

    def observe(index, bound):
        if fault == index:
            raise MemoryError("injected branch failure")

    def after_build(result):
        if fault == "after_build":
            raise MemoryError("injected pre-return failure")

    with pytest.raises(MemoryError):
        result, stats = transaction(tf, {"expr": expr}, lambda: budget.run(expr, rows, observe=observe), after_build=after_build)
        published.append(result)
    assert not published and collect(tf).fingerprint == before


def test_successful_functional_return_leaves_all_incoming_roots_unchanged():
    tf, expr, rows = context()
    before = collect(tf).fingerprint
    result, stats = transaction(tf, {"expr": expr}, lambda: BudgetedMaterializer(2).run(expr, rows))
    assert result[0].exact and stats["input_roots_unchanged"]
    assert collect(tf).fingerprint == before


def test_changed_state_is_rejected_not_rolled_back_over_external_change():
    tf, expr, rows = context()

    def mutate(result):
        tf.frames[7] = (5, 2)

    with pytest.raises(ValueError, match="live roots changed"):
        transaction(tf, {}, lambda: object(), after_build=mutate)
    assert tf.frames[7] == (5, 2)


def test_operator_payload_drift_not_hidden_by_cached_content_key():
    tf, expr, rows = context()
    before = collect(tf).fingerprint
    expr.terms[0].operators[0]._kernel[0, 0, 0, 0] += .125
    assert collect(tf).fingerprint != before


def test_numpy_construction_is_visible_to_traced_allocation_peak():
    result, stats = measured_build(lambda: np.ones(1_000_000, dtype=np.float64))
    assert stats["traced_peak_bytes"] >= result.nbytes
    assert stats["measured_transient_gate"]
    assert stats["resident_growth_upper_bound_bytes"] >= 0
