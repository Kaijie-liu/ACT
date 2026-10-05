from types import SimpleNamespace

import numpy as np
import pytest

from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831 import c5_runtime_materializer_v2 as runtime
from experiments.neural_hz_20260831.test_c5_native_budgeted_materialization_v1 import fixture
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import compare_hz, reference_materialize


def test_disabled_context_does_not_patch_any_function():
    original = cnn._lazy_materialize, cnn._try_deferred_expr_conv_relu
    with runtime.installed():
        assert (cnn._lazy_materialize, cnn._try_deferred_expr_conv_relu) == original
    assert (cnn._lazy_materialize, cnn._try_deferred_expr_conv_relu) == original


def test_matching_two_stage_island_uses_one_budget_and_restores(monkeypatch):
    expr, rows = fixture()
    mask = np.ones(expr.n_out, dtype=bool)
    events = []

    def native(layer, incoming, result, tf):
        first = cnn._lazy_materialize(expr, mask, 64_000_000, allow_transient_sum=True)
        second = cnn._lazy_materialize(expr, mask, 64_000_000, allow_transient_sum=True)
        return first, second

    monkeypatch.setattr(cnn, "_try_deferred_expr_conv_relu", native)
    original = cnn._lazy_materialize
    with runtime.installed(enabled=True, emit=events.append):
        result = cnn._try_deferred_expr_conv_relu(SimpleNamespace(id=123456), None, None, None)
    assert cnn._lazy_materialize is original and cnn._try_deferred_expr_conv_relu is native
    assert all(compare_hz(result[0], reference_materialize(expr, rows)).values())
    assert all(compare_hz(result[0], result[1]).values())
    assert events[-1]["sequence_products"] == 32 and events[-1]["stages"] == 2


def test_outside_island_and_unmatched_profiles_use_original_before_selection(monkeypatch):
    expr, rows = fixture()
    calls = []

    def original(*args, **kwargs):
        calls.append(args[2])
        return "baseline"

    monkeypatch.setattr(cnn, "_lazy_materialize", original)
    monkeypatch.setattr(cnn, "_try_deferred_expr_conv_relu", lambda *args: cnn._lazy_materialize(expr, rows, 100))
    with runtime.installed(enabled=True):
        assert cnn._lazy_materialize(expr, rows, 64_000_000) == "baseline"
        assert cnn._try_deferred_expr_conv_relu(SimpleNamespace(id=1), None, None, None) == "baseline"
    assert calls == [64_000_000, 100]


def test_selected_failure_never_rescues_through_original(monkeypatch):
    expr, rows = fixture()
    original_calls = []
    monkeypatch.setattr(cnn, "_lazy_materialize", lambda *args, **kwargs: original_calls.append(True))

    class Reject:
        def __init__(self, branches):
            self.remaining_whole = 256_000_000
        def run(self, *args, **kwargs):
            raise MemoryError("injected")

    monkeypatch.setattr(runtime, "BudgetedMaterializer", Reject)

    def native(*args):
        for attempt in range(2):
            with pytest.raises(MemoryError):
                cnn._lazy_materialize(expr, np.ones(expr.n_out, dtype=bool), 64_000_000)

    monkeypatch.setattr(cnn, "_try_deferred_expr_conv_relu", native)
    with runtime.installed(enabled=True):
        cnn._try_deferred_expr_conv_relu(SimpleNamespace(id=2), None, None, None)
    assert not original_calls


def test_empty_frontier_preserves_full_hz_without_composition_or_candidate(monkeypatch):
    from dataclasses import replace
    from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
    from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
    expr, rows = fixture()
    expr = replace(expr, bias=np.arange(expr.n_out, dtype=np.float64) / 3.7)
    mask = np.zeros(expr.n_out, dtype=bool)
    expected = cnn._lazy_materialize(expr, mask, 64_000_000)
    before = source_digest(expr.terms[0].source)

    def forbidden(*args, **kwargs):
        raise AssertionError("empty frontier must not request any composed coefficient")

    monkeypatch.setattr(runtime, "BudgetedMaterializer", forbidden)
    monkeypatch.setattr(ImplicitConv2DOp, "_row", forbidden)
    monkeypatch.setattr(cnn, "_try_deferred_expr_conv_relu", lambda *args: cnn._lazy_materialize(expr, mask, 64_000_000))
    events = []
    with runtime.installed(enabled=True, emit=events.append):
        actual = cnn._try_deferred_expr_conv_relu(SimpleNamespace(id=9), None, None, None)
    assert all(compare_hz(actual, expected).values())
    assert actual.n_cont == actual.n_bin == 2 and actual.n_eq == actual.n_ineq == 1
    assert np.array_equal(actual.c, expr.bias) and actual.Gc.nnz == actual.Gb.nnz == 0
    assert source_digest(expr.terms[0].source) == before
    assert [event["event"] for event in events] == ["c5_empty_frontier_native"]
