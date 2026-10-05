import gc
import importlib.util
from pathlib import Path
import weakref
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.tf_cnn import (
    SparseHZAffineExpr,
    SparseHZAffineTerm,
)
from act.back_end.solver.solver_hz import SparseHZono
from act.config.config import HybridZConfig


def _load_shadow_worker():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments/neural_hz_20260831/shadow_worker.py"
    spec = importlib.util.spec_from_file_location("neural_hz_shadow_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def _source_hz():
    return SparseHZono(
        c=np.array([0.25, -0.5, 0.75, 0.0]),
        Gc=sp.eye(4, format="csr"),
        Gb=sp.csr_matrix([[0.0], [0.5], [0.0], [-0.25]]),
        Ac=sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 4)),
        Ab=sp.csr_matrix([[1.0]]),
        b=np.array([0.0]),
        Auc=sp.csr_matrix(([1.0], ([0], [1])), shape=(1, 4)),
        Aub=sp.csr_matrix([[-1.0]]),
        ub=np.array([1.0]),
        frame_id=91,
        exact=True,
    )


def test_implicit_conv_config_is_default_off_and_arena_is_weak():
    baseline = HybridzTF()
    candidate = HybridzTF(HybridZConfig(sparse_implicit_conv_dag=True))
    assert not baseline._neural_hz_sparse_implicit_conv_dag
    assert candidate._neural_hz_sparse_implicit_conv_dag
    assert isinstance(candidate._neural_hz_linear_op_arena, weakref.WeakValueDictionary)

    operator = ImplicitConv2DOp(
        np.ones((1, 1, 1, 1), dtype=np.float64),
        (1, 1, 2, 2),
    )
    candidate._neural_hz_linear_op_arena["one"] = operator
    reference = weakref.ref(operator)
    del operator
    gc.collect()
    assert reference() is None
    assert list(candidate._neural_hz_linear_op_arena) == []


def test_worker_operator_and_live_cache_ledgers_charge_resident_not_logical():
    worker, _ = _load_shadow_worker()
    source = _source_hz()
    operator = ImplicitConv2DOp(
        np.ones((2, 1, 3, 3), dtype=np.float64),
        (1, 1, 2, 2),
        padding=1,
    )
    logical, resident_entries, resident_bytes, implicit = (
        worker._operator_storage(operator)
    )
    assert implicit
    assert logical > resident_entries
    assert resident_entries == 18
    assert resident_bytes == 18 * 8

    expression = SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, (operator,)),),
        bias=np.zeros(operator.shape[0], dtype=np.float64),
        n_out=operator.shape[0],
        frame_id=91,
    )
    tf = HybridzTF(HybridZConfig(sparse_implicit_conv_dag=True))
    tf._sparse_affine_expr_cache[7] = expression
    tf._sparse_hz_cache[1] = source
    ledger = worker._live_cache_upper_bound_ledger(tf)

    assert ledger["scope"] == "unique_live_cache_upper_bound"
    assert not ledger["allocator_overhead_included"]
    assert ledger["operator_objects"] == 1
    assert ledger["implicit_conv_operator_objects"] == 1
    assert ledger["hz_objects"] == 1
    assert ledger["operator_logical_expanded_nnz"] == logical
    assert ledger["operator_resident_entries"] == resident_entries
    assert ledger["operator_resident_bytes"] == resident_bytes
    assert ledger["hz_value_entries"] > 0
    assert ledger["hz_predicate_entries"] > 0
    assert ledger["resident_bytes"] > resident_bytes


def test_worker_arm_and_provenance_cover_implicit_operator_source():
    worker, path = _load_shadow_worker()
    source = path.read_text()
    assert '"sparse_phase_implicit_relu_census"' in source
    assert '"act/back_end/hybridz_tf/exact_linear_op.py"' in source
    assert "_live_cache_upper_bound_ledger(tf)" in source
    assert callable(worker._provenance)


def test_consumer_gc_waits_for_all_residual_users_and_pins_endpoints():
    source = _source_hz()
    expression = SparseHZAffineExpr(
        terms=(SparseHZAffineTerm(source, ()),),
        bias=np.zeros(source.n_out, dtype=np.float64),
        n_out=source.n_out,
        frame_id=source.frame_id,
    )
    net = SimpleNamespace(
        preds={0: [], 1: [0], 2: [0], 3: [1, 2], 4: [3]},
    )
    tf = HybridzTF(HybridZConfig(sparse_implicit_conv_dag=True))
    tf._net = net
    tf._sparse_remaining_consumers = {0: 2, 1: 1, 2: 1, 3: 1, 4: 0}
    tf._sparse_pinned_layers = {0, 3}
    tf._sparse_hz_cache.update({0: source, 3: source})
    tf._sparse_affine_expr_cache.update({1: expression, 2: expression})
    tf._sparse_phase_output_bounds[1] = object()
    tf._sparse_phase_output_bounds[2] = object()

    tf._release_consumed_sparse_predecessors(SimpleNamespace(id=1))
    assert tf._sparse_remaining_consumers[0] == 1
    assert 0 in tf._sparse_hz_cache
    tf._release_consumed_sparse_predecessors(SimpleNamespace(id=2))
    assert tf._sparse_remaining_consumers[0] == 0
    assert 0 in tf._sparse_hz_cache

    tf._release_consumed_sparse_predecessors(SimpleNamespace(id=3))
    assert 1 not in tf._sparse_affine_expr_cache
    assert 2 not in tf._sparse_affine_expr_cache
    assert 1 not in tf._sparse_phase_output_bounds
    assert 2 not in tf._sparse_phase_output_bounds
    assert tf._neural_hz_released_sparse_states == 2

    tf._release_consumed_sparse_predecessors(SimpleNamespace(id=4))
    assert 3 in tf._sparse_hz_cache

    baseline = HybridzTF()
    baseline._net = net
    baseline._sparse_remaining_consumers = {0: 2}
    baseline._sparse_hz_cache[0] = source
    baseline._release_consumed_sparse_predecessors(SimpleNamespace(id=1))
    assert baseline._sparse_remaining_consumers[0] == 2
    assert 0 in baseline._sparse_hz_cache
