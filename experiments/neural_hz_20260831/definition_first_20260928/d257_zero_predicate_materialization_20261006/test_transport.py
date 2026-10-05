"""Native lazy transport tests; only test 07 makes two fixed ordinary LP calls.

The frozen D255 construction helpers are reused with this file's own Budget.
No old test, cache, evidence recorder or result writer is invoked.
"""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.solver.solver_hz import _lower_hz_milp, sparse_hz_pad_frame
from experiments.neural_hz_20260831.definition_first_20260928.d255_rebase_relation_transport_20261006 import test_rebased as old
from experiments.neural_hz_20260831.definition_first_20260928.d257_zero_predicate_materialization_20261006 import predicate_transport as pt


RUN = Path(__file__).resolve().parents[2] / "results/d257_zero_predicate_materialization_20261006_v1"
_NAMES = (
    "default_off_and_private_namespace", "reachable_source_and_fresh_widths",
    "joint_sources_skip_and_bias", "zero_chain_normalization_and_checkpoint",
    "masked_rows_retain_predicates", "implicit_operator_and_invalid_inputs",
    "full_materialized_terminal_relation", "integer_extensions_and_decoder",
    "internal_selective_materialization", "source_drift_and_atomic_rejection",
    "shared_budget_and_no_production_mutation", "summary_and_qualification_boundary",
)
_EVIDENCE, _CACHE = {}, {}
_BUDGET = None
_LIMIT = 1_000_000
_FUNCTIONS = ("_lazy_materialize", "_lazy_checkpoint", "_try_phase_selective_exact_relu",
              "_try_deferred_expr_conv_relu", "sparse_hz_apply_affine_expr_layer")
_ORIGINAL_FUNCTIONS = {name: getattr(cnn, name) for name in _FUNCTIONS}


def _record(number, **values):
    name = _NAMES[number - 1]
    assert name not in _EVIDENCE
    _EVIDENCE[name] = values


def _record_file(name, payload):
    assert name == "summary.json"
    assert Path(os.environ["NEURAL_HZ_ACTIVE_COMPONENT_RUN"]) == RUN
    assert RUN.is_dir() and not RUN.is_symlink()
    with (RUN / name).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _budget():
    global _BUDGET
    if _BUDGET is None:
        _BUDGET = pt.Budget()
    return _BUDGET


def _case():
    if not _CACHE:
        case = old._fixture(budget=_budget())
        case["snapshot"], case["result"] = old._attach(case)
        source = case["hz"]
        # A genuinely additional predicate, absent from the local Result S.
        # The earlier input source is genuinely narrower than the latest core.
        other_outputs = case["outputs"][:5] + (old._nf(),) * (source.n_out - 5)
        other = old._state(5, 0, other_outputs,
            (old._nf(0, ((0, F(1)), (1, F(1)))),), (F(0),), frame=source.frame_id)
        # Nonzero stored skip, plus bias, only at the final nonobjective output.
        skip = sp.csr_matrix(([0.5], ([source.n_out - 1], [0])),
                             shape=(source.n_out, source.n_out))
        bias = np.zeros(source.n_out)
        bias[-1] = 0.125
        views = (cnn._lazy_identity(source),
                 cnn._lazy_from_hz_linear(other, skip, bias, _LIMIT))
        batch = pt.transport(case["result"], source, views,
                             global_widths=(source.n_cont, source.n_bin), enabled=True)
        adapter = pt.make_adapter(budget=_budget(), enabled=True)
        _CACHE.update(case=case, source=source, other=other, views=views,
                      batch=batch, adapter=adapter)
    return _CACHE


def _joint(views):
    return cnn._lazy_add(views[0], views[1], _LIMIT)


def _materialize(expr, keep=None):
    if keep is None:
        keep = np.ones(expr.n_out, dtype=bool)
    return _case()["adapter"].call("_lazy_materialize", expr, keep, _LIMIT)


def _native_extended(case, extended):
    nc, nb = case["hz"].n_cont, case["hz"].n_bin
    # D255 certificate layout is oldcont, oldbin, aux; native is cont, bin.
    return tuple(extended[:nc]) + tuple(extended[nc + nb:]) + tuple(extended[nc:nc + nb])


def _values(hz, assignment):
    return tuple(old._nv(value, assignment[:hz.n_cont], assignment[hz.n_cont:])
                 for value in old._outputs(hz))


def _padded_rows(hz, width, equality):
    return old._native_rows(sparse_hz_pad_frame(hz, width, hz.n_bin), equality)


def _contains_predicates(output, source):
    for equality in (True, False):
        actual = old._native_rows(output, equality)
        assert all(row in actual for row in _padded_rows(source, output.n_cont, equality))


def _zero_expr(anchor, n_out, operators=None, bias=None):
    ops = (sp.csr_matrix((n_out, anchor.n_out), dtype=np.float64),) if operators is None else operators
    return cnn.SparseHZAffineExpr((cnn.SparseHZAffineTerm(anchor, ops),),
        np.zeros(n_out) if bias is None else bias, n_out, anchor.frame_id)


def _lower(hz):
    meter = _budget()._branch()
    pt.nm._charge_hz(hz, meter, 8)
    meter.charge(work=10 * (hz.n_cont + hz.n_bin), entries=10 * (hz.n_cont + hz.n_bin))
    model = _lower_hz_milp(hz, prune_unused=False, coalesce_rows=False,
                           project_inactive_cont=False, fix_implied_binary=False)
    pt.nm._audit_lowering(hz, model, meter)
    return model


def test_01_default_off_and_private_namespace():
    sentinel = object()
    assert pt.normalize(sentinel, budget=sentinel) is sentinel
    assert pt.make_adapter(budget=sentinel) is None
    assert pt.transport(sentinel, sentinel, sentinel, global_widths=sentinel) is None
    with pytest.raises(pt.Rejected):
        pt.make_adapter(budget=_budget(), enabled=1)
    adapter = pt.make_adapter(budget=_budget(), enabled=True)
    assert adapter.namespace["SparseHZAffineExpr"] is cnn.SparseHZAffineExpr
    for name in _FUNCTIONS:
        assert getattr(cnn, name) is _ORIGINAL_FUNCTIONS[name]
        assert adapter.namespace[name] is not getattr(cnn, name)
    cloned = adapter.namespace["_lazy_checkpoint"]
    assert cloned.__globals__ is not vars(cnn)
    assert cloned.__globals__["_lazy_materialize"] is adapter.namespace["_lazy_materialize"]
    with pytest.raises(TypeError):
        adapter.namespace["_lazy_materialize"] = None
    _record(1, default_off_without_data_inspection=True, original_classes=True,
            private_function_globals=True, production_monkeypatch=False)


def test_02_reachable_source_and_fresh_widths():
    c = _case()
    source, batch = c["source"], c["batch"]
    assert (source.n_cont, source.n_bin) == (15, 4)
    assert batch.new_widths == (21, 4)
    assert batch.source is source and batch.result is c["case"]["result"]
    assert batch.budget is _budget()
    assert batch.anchor.n_out == 1 and np.array_equal(batch.anchor.c, np.zeros(1))
    assert batch.anchor.Gc.nnz == batch.anchor.Gb.nnz == 0
    assert not batch.anchor.c.flags.writeable and not batch.anchor.Ac.data.flags.writeable
    _contains_predicates(batch.anchor, c["case"]["result"].hz)
    for before, after in zip(c["views"], batch.views):
        assert after.terms[:-1] == before.terms
        assert np.array_equal(after.bias, before.bias)
        assert after.terms[-1].source is batch.anchor
    original = old._bytes(source)
    with pytest.raises(pt.Rejected):
        pt.transport(c["case"]["result"], source, c["views"],
                     global_widths=(16, 4), enabled=True)
    assert old._bytes(source) == original and _budget()._failure is None
    _, wide_result = old._attach(c["case"], widths=(17, 5))
    wide_batch = pt.transport(wide_result, source, c["views"],
                              global_widths=(17, 5), enabled=True)
    assert wide_batch.new_widths == (23, 5)
    assert wide_batch.views[0].terms[0].source is source
    assert wide_batch.views[1].terms[0].source is c["other"]
    _record(2, original_columns=[15, 4], new_columns=[21, 4], shared_auxiliary_columns=6,
            source_reachable=True, narrower_source_under_global_highwater=[17, 5],
            online_allocator_reserved=False)


def test_03_joint_sources_skip_and_bias():
    c = _case()
    source, other = c["source"], c["other"]
    assert (other.n_cont, other.n_bin, other.n_eq) == (5, 0, 1)
    assert old._native_rows(other, True)[-1] not in old._native_rows(source, True)
    assert c["views"][1].terms[0].source is other and other.Gc.nnz > 0
    output = _materialize(_joint(c["batch"].views))
    _contains_predicates(output, source)
    _contains_predicates(output, other)
    _contains_predicates(output, c["batch"].anchor)
    inputs = old._transport(F(1, 2), F(-1, 2), F(1, 2))
    original = old._assignment(c["case"], inputs)
    extended = c["case"]["result"].canonical_extension(original)
    native = _native_extended(c["case"], extended)
    actual, expected = _values(output, native), list(_values(source, original))
    expected[-1] += inputs[0] / 2 + F(1, 8)
    assert actual == tuple(expected)
    assert old._old_holds(output, native, integral=True)
    assert c["case"]["result"].decode(extended) == inputs
    _record(3, local_source_does_not_cover_extra_suffix=True,
            narrower_skip_source_columns=[5, 0],
            all_14_original_outputs_retained_in_registered_view=True,
            nonzero_skip_and_bias=True, complete_decoder_coordinates=5)


def test_04_zero_chain_normalization_and_checkpoint():
    c = _case()
    anchor, adapter = c["batch"].anchor, c["adapter"]
    n = 8
    first = sp.csr_matrix((n, 1), dtype=np.float64)
    dense = sp.csr_matrix(np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n) / 64)
    diagonal = sp.diags(np.array([1, -1] * (n // 2), dtype=np.float64), format="csr")
    expr = _zero_expr(anchor, n, (first, dense, diagonal), np.arange(n, dtype=np.float64) / 8)
    normal = pt.normalize(expr, budget=_budget(), enabled=True)
    assert normal.terms[0].source is anchor
    assert len(normal.terms[0].operators) == 1
    assert normal.terms[0].operators[0].shape == (n, 1)
    assert normal.terms[0].operators[0].nnz == 0
    assert np.array_equal(normal.bias, expr.bias)
    checkpoint = adapter.call("_lazy_checkpoint", expr, _LIMIT)
    assert checkpoint.n_out == n and len(checkpoint.terms) == 1
    state = checkpoint.terms[0].source
    assert np.array_equal(state.c, expr.bias)
    assert state.Gc.nnz == state.Gb.nnz == 0
    _contains_predicates(state, anchor)
    assert adapter.normalized_terms > 0
    _record(4, zero_chain_length=3, canonical_zero_csr=True,
            checkpoint_preserves_all_predicates=True, nonzero_bias_preserved=True,
            runtime_speedup_claimed=False)


def test_05_masked_rows_retain_predicates():
    c = _case()
    expr = _joint(c["batch"].views)
    none = _materialize(expr, np.zeros(expr.n_out, dtype=bool))
    assert none.Gc.nnz == none.Gb.nnz == 0
    assert np.array_equal(none.c, expr.bias)
    _contains_predicates(none, c["other"])
    _contains_predicates(none, c["batch"].anchor)
    keep = np.arange(expr.n_out) % 2 == 0
    some = _materialize(expr, keep)
    full = _materialize(expr)
    assert (some.Gc[keep] != full.Gc[keep]).nnz == 0
    assert some.Gc[~keep].nnz == some.Gb[~keep].nnz == 0
    assert np.array_equal(some.c[~keep], expr.bias[~keep])
    _contains_predicates(some, c["batch"].anchor)
    _record(5, all_false_mask_retains_predicates=True,
            partial_mask_retains_predicates=True, original_unmasked_bias_semantics=True)


def test_06_implicit_operator_and_invalid_inputs():
    c = _case()
    anchor = c["batch"].anchor
    operator = ImplicitConv2DOp(np.array([[[[2.0]]]]), (1, 1, 1, 4))
    expr = _zero_expr(anchor, 4, (sp.csr_matrix((4, 1), dtype=np.float64), operator))
    normal = pt.normalize(expr, budget=_budget(), enabled=True)
    assert len(normal.terms[0].operators) == 1 and normal.terms[0].operators[0].nnz == 0
    _contains_predicates(_materialize(expr), anchor)
    unknown = SimpleNamespace(shape=(4, 1))
    invalid = _zero_expr(anchor, 4, (unknown,))
    with pytest.raises(pt.Rejected):
        pt.normalize(invalid, budget=_budget(), enabled=True)
    bad = sp.csr_matrix(([np.nan], ([0], [0])), shape=(4, 1))
    invalid = _zero_expr(anchor, 4, (bad,))
    with pytest.raises(pt.Rejected):
        pt.normalize(invalid, budget=_budget(), enabled=True)
    bad_shape = sp.csr_matrix((4, 1), dtype=np.float64)
    invalid = _zero_expr(anchor, 4, (bad_shape,))
    bad_shape.resize((5, 1))
    with pytest.raises(pt.Rejected):
        pt.normalize(invalid, budget=_budget(), enabled=True)
    operator._kernel.flat[0] = np.nan
    with pytest.raises(pt.Rejected):
        pt.normalize(expr, budget=_budget(), enabled=True)
    assert _budget()._failure is None
    _record(6, actual_implicit_conv_type=True, no_explicit_kernel_matrix_requested=True,
            unknown_nonfinite_shape_and_kernel_drift_rejected=True)


def test_07_full_materialized_terminal_relation():
    c = _case()
    case, result = c["case"], c["case"]["result"]
    certificate = old._certificate(case)
    plain = _materialize(_joint(c["views"]))
    enhanced = _materialize(_joint(c["batch"].views))
    _contains_predicates(enhanced, result.hz)
    _contains_predicates(enhanced, c["other"])
    objective = old._outputs(enhanced)[case["F_index"]]
    remapped = pt.nm.base.Row(tuple((i if i < case["hz"].n_cont else i + 6, a)
        for i, a in certificate.terms), certificate.rhs)
    assert remapped == old._row(old._flat(objective, enhanced.n_cont), F(265, 128))
    # These models are made from the COMPLETE actual lazy materializations.
    old_model, new_model = _lower(plain), _lower(enhanced)
    assert tuple(new_model.cont_source) == tuple(range(enhanced.n_cont))
    assert tuple(new_model.bin_source) == tuple(range(enhanced.n_bin))
    assert not new_model.bin_fixes and not new_model.cont_eliminations
    previous = old._normal_lp(plain, old_model, case["F_index"])
    current = old._normal_lp(enhanced, new_model, case["F_index"])
    assert previous >= float(F(8509, 4096)) - 1e-8
    assert current <= float(F(265, 128)) + 1e-8
    assert current >= float(F(527, 256)) - 1e-8
    assert current < previous and current < float(F(1061, 512))
    _record(7, actual_materialized_output_lowered=True, stored_exact_certificate=True,
            fixed_normal_LPs=2, old_LP=previous, new_LP=current, upper="265/128",
            same_D255_mathematical_control=True, bound_claimed_tight=False,
            original_bits=enhanced.n_bin, new_relation_bits=0, gains=0)


def test_08_integer_extensions_and_decoder():
    c = _case()
    case = c["case"]
    output = _materialize(_joint(c["batch"].views))
    witnesses = [(old._transport(1, -1, 1), None),
                 (old._transport(F(1, 2), F(-1, 2), F(-1, 2)), None)]
    witnesses.extend((old._transport(0, 0), tuple(map(F, bits)))
                     for bits in product((0, 1), repeat=4))
    for inputs, active in witnesses:
        original = old._assignment(case, inputs, active=active)
        extension = case["result"].canonical_extension(original)
        native = _native_extended(case, extension)
        assert old._old_holds(output, native, integral=True)
        assert case["result"].decode(extension) == inputs
        expected = list(_values(case["hz"], original))
        expected[-1] += inputs[0] / 2 + F(1, 8)
        assert _values(output, native) == tuple(expected)
    _record(8, integer_witnesses=len(witnesses), old_zero_label_combinations=16,
            complete_original_decoder=True, common_aux_assignment=True,
            certificate_to_native_column_reordering_explicit=True)


def test_09_internal_selective_materialization():
    c = _case()
    # Three real rows: stable-positive dense readout, crossing f1, negative constant.
    # The positive probe has 9 exact continuous supports, exceeding 3+1 mask
    # metadata. Its center is 8+10/32 and cube radius 201/512, so [1,100]
    # is conservative; the crossing row is exactly f1 in [-1,1].
    expr = _joint(c["batch"].views)
    matrix = np.zeros((3, expr.n_out), dtype=np.float64)
    matrix[0] = 1 / 32
    matrix[1, 5] = 1
    reduced = cnn._lazy_append_linear(expr, sp.csr_matrix(matrix), np.array([8.0, 0.0, -1.0]), _LIMIT)
    tf = HybridzTF()
    tf._SPARSE_MAX_AFFINE_CELLS = _LIMIT
    tf._neural_hz_sparse_phase_selective_materialization = True
    tf._neural_hz_compact_relu = False
    tf._sparse_frame_widths[expr.frame_id] = c["batch"].new_widths
    bounds = Bounds(torch.tensor([[1.0, -1.0, -1.0]], dtype=torch.float64),
                    torch.tensor([[100.0, 1.0, -1.0]], dtype=torch.float64))
    adapter = c["adapter"]
    before = adapter.materializations
    selected = adapter.call("_try_phase_selective_exact_relu", reduced, bounds, tf,
                            SimpleNamespace(id=25701, kind="RELU"))
    assert adapter.materializations > before
    assert selected is not None
    assert tf._neural_hz_phase_selective_profile[-1]["probe_generator_nnz"] == 9
    assert selected.expression.n_out == selected.core.n_out == 3
    assert selected.core.n_bin == c["batch"].new_widths[1] + 1
    _contains_predicates(selected.core, c["batch"].anchor)
    complete = _materialize(selected.expression)
    _contains_predicates(complete, c["other"])
    _contains_predicates(complete, c["batch"].anchor)
    assert complete.Gc.getrow(0).nnz > 0 and selected.core.Gc.getrow(0).nnz == 0
    assert complete.Gc.getrow(2).nnz == 0 and complete.c[2] == 0
    # Full witness of the genuinely appended original production ReLU graph.
    for x1, x2, t in ((1, -1, 1), (0, 0, 0)):
        original = old._assignment(c["case"], old._transport(x1, x2, t))
        extension = c["case"]["result"].canonical_extension(original)
        native = _native_extended(c["case"], extension)
        old_nc = c["batch"].new_widths[0]
        phases = (-1, 1) if x1 == 0 else (-1,)
        for bit in phases:
            q = max(F(0), F(x1))
            sval = (F(x1) - q) / F(-1, 2) - bit
            values = native[:old_nc] + (sval, 1 - 2 * q) + native[old_nc:] + (F(bit),)
            assert old._old_holds(complete, values, integral=True)
            expected = _values(_materialize(reduced), native)
            assert _values(complete, values) == tuple(max(F(0), v) for v in expected)
    _record(9, internal_wrapper_reached=True, selective_succeeded=True,
            stable_positive_and_core_joint_output=True, new_original_relu_bits=1,
            deferred_path_executed=False, installed_HybridzTF=False)


def test_10_source_drift_and_atomic_rejection():
    c = _case()
    source, views, result = c["source"], c["views"], c["case"]["result"]
    before = old._bytes(source)
    orphan = replace(source, c=source.c.copy())
    with pytest.raises(pt.Rejected):
        pt.transport(result, orphan, views, global_widths=(15, 4), enabled=True)
    wrong = replace(c["other"], frame_id=source.frame_id + 1)
    wrong_view = cnn._lazy_identity(wrong)
    with pytest.raises(pt.Rejected):
        pt.transport(result, source, (views[0], wrong_view), global_widths=(15, 4), enabled=True)
    wider = sparse_hz_pad_frame(c["other"], 16, 4)
    with pytest.raises(pt.Rejected):
        pt.transport(result, source, (views[0], cnn._lazy_identity(wider)),
                     global_widths=(15, 4), enabled=True)
    case = old._fixture(budget=_budget())
    _, isolated = old._attach(case)
    case["hz"].c[0] += 1 / 64
    changed = cnn._lazy_identity(case["hz"])
    with pytest.raises(pt.Rejected):
        pt.transport(isolated, case["hz"], (changed,), global_widths=(15, 4), enabled=True)
    assert old._bytes(source) == before and _budget()._failure is None
    assert _materialize(_joint(c["batch"].views)).n_bin == 4
    _record(10, reachability_frame_content_and_width_checked=True,
            invalid_later_view_rejects_whole_return=True, original_objects_unchanged=True,
            online_publication_claimed=False, adversarial_deserialization_claimed=False)


def test_11_shared_budget_and_no_production_mutation():
    c = _case()
    assert c["case"]["budget"] is c["batch"].budget is _budget()
    work, entries = _budget().work, _budget().entries
    pt.normalize(_joint(c["batch"].views), budget=_budget(), enabled=True)
    assert _budget().work > work and _budget().entries > entries
    for kwargs in ({"max_work": 0}, {"max_entries": 0}, {"max_branch": 0}):
        budget = pt.Budget(**kwargs)
        with pytest.raises(pt.Rejected):
            pt.normalize(c["batch"].views[0], budget=budget, enabled=True)
        assert budget._failure is not None
        with pytest.raises(pt.Rejected):
            pt.normalize(c["batch"].views[0], budget=budget, enabled=True)
    # The actual production apply catches ValueError, including Rejected.
    # A 2048-row bias alone costs 24576 work in the normalizer, exceeding
    # this declared small budget after the successful private adapter build.
    swallowed_budget = pt.Budget(max_work=20_000)
    adapter = pt.make_adapter(budget=swallowed_budget, enabled=True)
    wide = _zero_expr(c["batch"].anchor, 2048)
    lower = torch.zeros((1, 2048), dtype=torch.float64)
    lower[0, 0] = -1
    bounds = Bounds(lower, torch.ones((1, 2048), dtype=torch.float64))
    tf = HybridzTF()
    tf._SPARSE_MAX_AFFINE_CELLS = _LIMIT
    tf._neural_hz_sparse_phase_selective_materialization = True
    with pytest.raises(pt.Rejected):
        adapter.call("sparse_hz_apply_affine_expr_layer",
            SimpleNamespace(id=25702, kind="RELU"), wide, bounds,
            SimpleNamespace(bounds=bounds), tf)
    assert swallowed_budget._failure is not None and adapter._active_meter is None
    assert c["adapter"]._active_meter is None
    for name, original in _ORIGINAL_FUNCTIONS.items():
        assert getattr(cnn, name) is original
    assert c["batch"].views[0].__class__ is cnn.SparseHZAffineExpr
    assert _budget()._failure is None
    _record(11, ordinary_positive_shared_budget=True, failures_not_refunded=True,
            resource_failure_sticky=True, production_function_identity_unchanged=True,
            caught_resource_rejection_not_returned_as_success=True,
            active_meter_restored=True,
            work=_budget().work, entries=_budget().entries,
            entire_production_internal_work_metered=False)


def test_12_summary_and_qualification_boundary():
    assert tuple(_EVIDENCE) == _NAMES[:-1]
    _record(12, all_declared_cases_completed=True, gains=0,
            zero_predicate_materialization_passed=False, actual_model=False,
            online=False, GPU=False, complete_physical=False,
            new_domain=False, new_capability=False)
    _record_file("summary.json", {
        "schema": "d257_zero_predicate_materialization_v1",
        "completed": True, "tests": _EVIDENCE,
        "zero_predicate_materialization_passed": False,
        "actual_model": False, "online": False, "GPU": False,
        "complete_physical": False, "new_domain": False, "new_capability": False,
        "formal_gain": 0, "formal_baseline": "1870/2413", "independent": "61/400",
        "shared_budget": {"work": _budget().work, "entries": _budget().entries},
        "qualification_requires_full_runner_postchecks": True,
    })
