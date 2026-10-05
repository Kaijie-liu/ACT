"""Independent complete incidence checks for the isolated C128 support rule.

The scalar oracle below does not invoke Conv rows, floating convolution,
factorized support helpers, or the old engine.  Every result component and
packed owner word is checked; these small fixtures are unit tests, not target
substitutes or evidence of whole-source/native/LIVE admission.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import c24_dense_graph_v1 as old_graph
from experiments.neural_hz_20260831 import c128_birth_emission_v1 as birth
from experiments.neural_hz_20260831 import c128_channel_graph_v1 as new_graph
from experiments.neural_hz_20260831 import c128_channel_support_v1 as new
from experiments.neural_hz_20260831.c24_dense_ownership_v1 import DenseOwnerEngine, RADIX
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.test_c6_support_affine_plan_v1 import fixture as shared_fixture


def _geometry(case="dense"):
    batch, ci, co, height, width = 1, 4, 4, 4, 5
    groups, stride, padding, dilation, kh, kw = 1, 1, 1, 1, 3, 3
    if case == "groups":
        groups = 2
    elif case == "batch":
        batch, groups = 2, 2
    elif case == "stride":
        stride = 2
    elif case == "dilation":
        dilation, padding = 2, 2
    elif case == "asymmetric":
        stride, padding, dilation = (2, 1), (1, 2), (1, 2)
    elif case == "point":
        kh, kw, padding = 1, 1, 0
    elif case == "rectangle":
        kh, kw, padding = 2, 3, (0, 1)
    elif case == "row_mask":
        batch, groups = 2, 2
    raw = np.arange(co * (ci // groups) * kh * kw).reshape(co, ci // groups, kh, kw)
    kernel = (1 + raw % 3).astype(np.float64) / 8
    if case != "dense":
        kernel *= np.where(raw % 2, -1., 1.)
    if case == "sparse":
        kernel.reshape(-1)[::3] = 0.
    elif case == "zero":
        kernel[:] = 0.
    op = ImplicitConv2DOp(kernel, (batch, ci, height, width), groups=groups,
                          stride=stride, padding=padding, dilation=dilation)
    if case in ("row_mask", "asymmetric", "sparse"):
        # Distinct batch/channel masks: no spatial-only mask may substitute.
        row_mask = (np.arange(op.shape[0]).reshape(op.output_shape)
                    + np.arange(co)[None, :, None, None]) % 4 != 1
        op = ImplicitConv2DOp(kernel, op.input_shape, groups=groups,
                              stride=stride, padding=padding, dilation=dilation,
                              row_mask=row_mask.reshape(-1))
    return op


def _incidences(op):
    """Original structural incidences, each (output,input) exactly once."""
    if type(op) is sp.csr_matrix:
        return [(row, int(op.indices[pos]))
                for row in range(op.shape[0])
                for pos in range(int(op.indptr[row]), int(op.indptr[row + 1]))
                if op.data[pos] != 0.]
    batch, ci, hi, wi = op.input_shape
    _, co, ho, wo = op.output_shape
    _, cig, khn, kwn = op._kernel.shape
    cog = co // op._groups
    result = []
    for b in range(batch):
        for oc in range(co):
            group = oc // cog
            for oh in range(ho):
                for ow in range(wo):
                    row = ((b * co + oc) * ho + oh) * wo + ow
                    if op._row_mask is not None and not op._row_mask[row]:
                        continue
                    for local in range(cig):
                        channel = group * cig + local
                        for kh in range(khn):
                            ih = oh * op._stride[0] - op._padding[0] + kh * op._dilation[0]
                            if not 0 <= ih < hi:
                                continue
                            for kw in range(kwn):
                                iw = ow * op._stride[1] - op._padding[1] + kw * op._dilation[1]
                                if 0 <= iw < wi and op._kernel[oc, local, kh, kw] != 0.:
                                    col = ((b * ci + channel) * hi + ih) * wi + iw
                                    result.append((row, col))
    assert len(result) == len(set(result))
    return result


def _scalar(op, mask, *, transpose=False, owners=False):
    values = [int(value) for value in mask]
    if owners:
        # Ranks precede the OPERATOR row mask; excluded rows still reserve
        # their caller-mask rank.  Re-ranking after intersection is wrong.
        rank = 0
        for row, present in enumerate(values):
            if present:
                values[row] = RADIX + rank
                rank += 1
        transpose = True
    result = [0] * op.shape[1 if transpose else 0]
    for row, col in _incidences(op):
        if transpose:
            result[col] += values[row]
        else:
            result[row] += values[col]
    return np.asarray(result, dtype=np.int64)


def _check_vector(actual, old, scalar):
    assert actual.dtype == np.dtype(np.int64) and actual.ndim == 1
    assert not actual.flags.writeable
    assert np.array_equal(actual, old)
    assert np.array_equal(actual, scalar)


@pytest.mark.parametrize("case", [
    "dense", "signed", "groups", "batch", "stride", "dilation",
    "asymmetric", "point", "rectangle", "sparse", "zero", "row_mask",
])
def test_complete_forward_reverse_and_owner_words_match_two_independent_engines(case):
    op = _geometry(case)
    kernel_before = op._kernel.copy()
    rows_before = None if op._row_mask is None else op._row_mask.copy()
    forward = np.arange(op.shape[1], dtype=np.int64) % 5
    reverse = np.arange(op.shape[0], dtype=np.int64) % 7
    forward[1] = reverse[1] = 256_000_000  # Full old integer-mask domain.
    owners = np.arange(op.shape[0]) % 3 != 1
    masks_before = tuple(mask.copy() for mask in (forward, reverse, owners))
    engine, control = new.FactorizedOwnerEngine(enabled=True), DenseOwnerEngine()
    for mask, transpose in ((forward, False), (reverse, True)):
        actual = engine.compute(op, mask, transpose=transpose)
        _check_vector(actual, control.compute(op, mask, transpose=transpose),
                      _scalar(op, mask, transpose=transpose))
        assert not np.shares_memory(actual, mask)
    actual = engine.owners(op, owners)
    _check_vector(actual, control.owners(op, owners), _scalar(op, owners, owners=True))
    assert np.all(actual >= 0) and np.all(actual < (1 << 61))
    if case not in ("sparse", "zero"):
        assert actual.flags.owndata
    assert engine.channel_stats["new_persistent_workspaces"] == 0
    assert engine.channel_stats["source_or_LIVE_admitted"] is False
    assert engine.channel_stats["formal_gain"] == 0
    assert np.array_equal(op._kernel, kernel_before)
    if rows_before is not None:
        assert np.array_equal(op._row_mask, rows_before)
    for mask, before in zip((forward, reverse, owners), masks_before, strict=True):
        assert np.array_equal(mask, before)


def graph_fixture(mode="dense", keep="all"):
    """Reusable genuine binary/EQ/INEQ, shared-frame and shared-suffix DAG."""
    if mode not in ("dense", "sparse", "masked") or keep not in ("all", "partial"):
        raise ValueError("unknown deterministic C128 graph fixture")
    expr, op = shared_fixture()
    if mode != "sparse":
        op._kernel[op._kernel == 0.] = .125
    if mode == "masked":
        op._row_mask = np.arange(op.shape[0]) % 4 != 1
    wanted = np.ones(expr.n_out, dtype=bool)
    if keep == "partial":
        wanted[::2] = False
    return expr, wanted


def _source_support(source):
    result = []
    for row in range(source.n_out):
        present = bool(source.c[row] != 0.)
        for matrix in (source.Gc, source.Gb):
            present |= any(matrix.data[pos] != 0.
                           for pos in range(int(matrix.indptr[row]), int(matrix.indptr[row + 1])))
        result.append(present)
    return np.asarray(result, dtype=bool)


def _assert_scalar_graph(result, keep, uid_start):
    nodes, root, report, ownership, uid_bases = result
    supports, needed, owned, expected_bases = [], [], [], []
    cursor = uid_start
    for node in nodes:
        expected_bases.append(cursor)
        cursor += node["width"]
        if node["kind"] == "source":
            support = _source_support(node["source"])
        elif node["kind"] == "op":
            support = _scalar(node["op"], supports[node["parents"][0]]) != 0
        else:
            support = np.asarray([any(supports[p][row] for p in node["parents"])
                                  for row in range(node["width"])], dtype=bool)
        supports.append(support)
        needed.append(np.zeros(node["width"], dtype=bool))
        owned.append(np.zeros(node["width"], dtype=np.int64))
        assert np.array_equal(support, node["support"])
    needed[root] = keep & supports[root]
    for index in reversed(range(len(nodes))):
        node = nodes[index]
        if node["kind"] == "op":
            parent = node["parents"][0]
            words = _scalar(node["op"], needed[index], owners=True)
            for col, packed in enumerate(words):
                if packed and supports[parent][col]:
                    needed[parent][col] = True
                    count = int(packed) // RADIX
                    owned[parent][col] += int(packed) + count * expected_bases[index]
        elif node["kind"] == "sum":
            rank = 0
            for row, present in enumerate(needed[index]):
                if not present:
                    continue
                for parent in set(node["parents"]):
                    if supports[parent][row]:
                        needed[parent][row] = True
                        owned[parent][row] += RADIX + expected_bases[index] + rank
                rank += 1
    paths = []
    for index, (node, count) in enumerate(zip(nodes, report["node_counts"], strict=True)):
        assert np.array_equal(node["needed"], needed[index])
        assert np.array_equal(ownership[index], owned[index])
        nc = nb = centers = 0
        for row, present in enumerate(needed[index]):
            if not present:
                continue
            if node["kind"] == "source":
                source = node["source"]
                centers += int(source.c[row] != 0.)
                for matrix, is_binary in ((source.Gc, False), (source.Gb, True)):
                    edges = sum(matrix.data[pos] != 0. for pos in
                                range(int(matrix.indptr[row]), int(matrix.indptr[row + 1])))
                    if is_binary:
                        nb += edges
                    else:
                        nc += edges
            elif node["kind"] == "sum":
                nc += sum(bool(supports[parent][row]) for parent in set(node["parents"]))
        if node["kind"] == "op":
            forward = _scalar(node["op"], supports[node["parents"][0]])
            nc = sum(int(forward[row]) for row, present in enumerate(needed[index]) if present)
        assert count["auxiliaries"] == int(np.count_nonzero(needed[index]))
        assert (count["continuous_edges"], count["binary_edges"], count["center_edges"]) == (nc, nb, centers)
        assert count["encoding_work_upper"] == 16 * (nc + nb + centers + count["auxiliaries"])
        assert count["support_work"] == node["support_work"]
        local = count["support_work"] + count["encoding_work_upper"]
        paths.append(local + max((paths[parent] for parent in node["parents"]), default=0))
    assert uid_bases == expected_bases
    assert report["radix_uid_base"] == cursor
    assert report["largest_branch_work_upper"] == max(paths)
    assert report["support_work"] == sum(node["support_work"] for node in nodes)
    assert report["total_work_upper"] == report["support_work"] + sum(
        item["encoding_work_upper"] for item in report["node_counts"])


def _assert_graph_agreement(actual, control):
    nodes, root, report, words, bases = actual
    old_nodes, old_root, old_report, old_words, old_bases = control
    assert root == old_root and bases == old_bases and len(nodes) == len(old_nodes)
    for node, old_node, word, old_word in zip(nodes, old_nodes, words, old_words, strict=True):
        for field in ("kind", "width", "parents"):
            assert node[field] == old_node[field]
        for field in ("op", "source"):
            if field in node:
                assert node[field] is old_node[field]
        assert np.array_equal(node["support"], old_node["support"])
        assert np.array_equal(node["needed"], old_node["needed"])
        assert np.array_equal(word, old_word)
    for count, old_count in zip(report["node_counts"], old_report["node_counts"], strict=True):
        assert {k: v for k, v in count.items() if k != "support_work"} == {
            k: v for k, v in old_count.items() if k != "support_work"}
    for field in ("auxiliaries", "continuous_edges", "binary_edges", "temporary_support_cache_bytes",
                  "stable_uid_start", "radix_uid_base", "ownership_reverse_replaces_boolean_count",
                  "dense_node_local_UIDs", "sum_dense_rank_extra_work"):
        assert report[field] == old_report[field]


@pytest.mark.parametrize("mode", ["dense", "sparse", "masked"])
@pytest.mark.parametrize("keep", ["all", "partial"])
def test_complete_nonconvex_shared_frame_graph_matches_old_and_scalar_incidence(mode, keep):
    expr, wanted = graph_fixture(mode, keep)
    sources = {id(term.source): term.source for term in expr.terms}
    before = {key: source_digest(value) for key, value in sources.items()}
    assert len(sources) == 3 and all(source.n_bin == 2 for source in sources.values())
    assert all(source.n_eq > 0 and source.n_ineq > 0 for source in sources.values())
    control = old_graph.graph(expr, wanted, 256_000_000, uid_start=37)
    actual = new_graph.graph(expr, wanted, 256_000_000, uid_start=37, enabled=True)
    _assert_graph_agreement(actual, control)
    _assert_scalar_graph(actual, wanted, 37)
    assert before == {key: source_digest(value) for key, value in sources.items()}


def test_signed_dense_support_does_not_perform_floating_cancellation(monkeypatch):
    op = ImplicitConv2DOp(np.array([[[[1.]], [[-1.]]]]), (1, 2, 2, 2))
    def forbidden(*args, **kwargs):
        raise AssertionError("structural support requested floating Conv or rows")
    for name in ("_row", "to_csr_reference", "matvec"):
        monkeypatch.setattr(ImplicitConv2DOp, name, forbidden)
    actual = new.FactorizedOwnerEngine(enabled=True).compute(op, np.ones(8, dtype=bool))
    assert np.array_equal(actual, np.full(4, 2, dtype=np.int64))


def test_default_off_retains_old_result_and_tariff():
    op = _geometry("signed")
    values = np.arange(op.shape[1], dtype=np.int64) % 4
    owners = np.arange(op.shape[0]) % 2 == 0
    actual, control = new.FactorizedOwnerEngine(), DenseOwnerEngine()
    assert np.array_equal(actual.compute(op, values), control.compute(op, values))
    assert np.array_equal(actual.owners(op, owners), control.owners(op, owners))
    assert actual.visits == control.visits and actual.cache_bytes == control.cache_bytes


def test_zero_masks_skip_factorized_arithmetic_even_with_zero_budget(monkeypatch):
    op = _geometry()
    def forbidden(*args, **kwargs):
        raise AssertionError("zero support dispatched convolution arithmetic")
    monkeypatch.setattr(new.FactorizedOwnerEngine, "_conv_counts", forbidden)
    engine = new.FactorizedOwnerEngine(0, enabled=True)
    assert not engine.compute(op, np.zeros(op.shape[1], dtype=bool)).any()
    assert not engine.compute(op, np.zeros(op.shape[0], dtype=np.int64), transpose=True).any()
    assert not engine.owners(op, np.zeros(op.shape[0], dtype=bool)).any()
    assert engine.visits == 0


def test_complete_cache_hit_reuses_readonly_result_without_work():
    op = _geometry()
    engine = new.FactorizedOwnerEngine(enabled=True)
    mask = np.ones(op.shape[0], dtype=bool)
    a = engine.compute(op, mask, transpose=True)
    b = engine.owners(op, mask)
    visits, byte_count = engine.visits, engine.cache_bytes
    assert engine.compute(op, mask.copy(), transpose=True) is a
    assert engine.owners(op, mask.copy()) is b
    assert a is not b and engine.hits == 2
    assert engine.visits == visits and engine.cache_bytes == byte_count
    mask[0] = False
    fresh = engine.owners(op, mask)
    assert fresh is not b and np.array_equal(fresh, _scalar(op, mask, owners=True))
    assert np.array_equal(b, _scalar(op, np.ones(op.shape[0], dtype=bool), owners=True))


@pytest.mark.parametrize("mutation", ["kernel_zero", "kernel_sign", "row_mask"])
def test_source_operator_mutation_never_reuses_stale_support_certificate(mutation):
    op = _geometry()
    engine = new.FactorizedOwnerEngine(enabled=True)
    mask = np.ones(op.shape[0], dtype=bool)
    first = engine.owners(op, mask)
    saved, work = first.copy(), engine.visits
    if mutation == "kernel_zero":
        op._kernel.flat[0] = 0.
    elif mutation == "kernel_sign":
        op._kernel *= -1.
    else:
        op._row_mask = np.arange(op.shape[0]) % 2 == 0
    second = engine.owners(op, mask)
    _check_vector(second, DenseOwnerEngine().owners(op, mask), _scalar(op, mask, owners=True))
    assert second is not first and engine.visits > work
    assert np.array_equal(first, saved)


def test_nonfinite_source_is_rejected_even_when_prior_cache_key_exists():
    op = _geometry()
    engine = new.FactorizedOwnerEngine(enabled=True)
    mask = np.ones(op.shape[1], dtype=bool)
    engine.compute(op, mask)
    op._kernel.flat[-1] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        engine.compute(op, mask)


@pytest.mark.parametrize("bad", ["negative", "above_cap", "float", "shape", "owners_integer"])
def test_inherited_invalid_mask_domains_remain_rejected(bad):
    op = _geometry()
    engine = new.FactorizedOwnerEngine(enabled=True)
    mask = np.ones(op.shape[1], dtype=np.int64)
    if bad == "negative":
        mask[0] = -1
    elif bad == "above_cap":
        mask[0] = 256_000_001
    elif bad == "float":
        mask = mask.astype(np.float64)
    elif bad == "shape":
        mask = mask[:-1]
    else:
        with pytest.raises(ValueError):
            engine.owners(op, np.ones(op.shape[0], dtype=np.int64))
        return
    with pytest.raises(ValueError):
        engine.compute(op, mask)


def test_csr_fallback_preserves_explicit_zero_incidence_and_full_old_tariff():
    op = sp.csr_matrix((np.array([1., 0., -2., .5]), np.array([0, 2, 1, 3]),
                       np.array([0, 2, 3, 4])), shape=(3, 4))
    actual, control = new.FactorizedOwnerEngine(enabled=True), DenseOwnerEngine()
    for mask, transpose in ((np.array([2, 3, 5, 7]), False), (np.array([3, 4, 5]), True)):
        _check_vector(actual.compute(op, mask, transpose=transpose),
                      control.compute(op, mask, transpose=transpose),
                      _scalar(op, mask, transpose=transpose))
    wanted = np.array([True, False, True])
    _check_vector(actual.owners(op, wanted), control.owners(op, wanted), _scalar(op, wanted, owners=True))
    assert actual.visits == control.visits


def test_dense_fee_and_empty_output_rows_cover_complete_unfactored_geometry():
    op = _geometry("row_mask")
    plan = new.fee_plan(op)
    batch, _, hi, wi = op.input_shape
    _, _, ho, wo = op.output_shape
    khn, kwn = op._kernel.shape[-2:]
    spatial_visits = 0
    for _batch in range(batch):
        for _group in range(op._groups):
            for kh in range(khn):
                for kw in range(kwn):
                    for oh in range(ho):
                        for ow in range(wo):
                            ih = oh * op._stride[0] - op._padding[0] + kh * op._dilation[0]
                            iw = ow * op._stride[1] - op._padding[1] + kw * op._dilation[1]
                            spatial_visits += int(0 <= ih < hi and 0 <= iw < wi)
    inspection = 1024 + 4 * op._kernel.size + 16 * (khn + kwn)
    dense_work = 8 * (op.shape[1] + op.shape[0]) + 16 * spatial_visits
    assert plan["inspection_work"] == inspection
    assert plan["geometric_incidences"] == spatial_visits
    assert plan["dense_work"] == dense_work
    assert plan["total_work"] == inspection + dense_work
    assert plan["complete_kernel_density_not_observed"] is True
    assert plan["fallback_old_traversal_work_not_included"] is True
    engine = new.FactorizedOwnerEngine(enabled=True)
    mask = np.ones(op.shape[1], dtype=bool)
    wanted = np.ones(op.shape[0], dtype=bool)
    engine.compute(op, mask)
    engine.owners(op, wanted)
    assert engine.visits == 2 * plan["total_work"] + op.shape[0]
    op._row_mask[:] = False
    assert not engine.compute(op, mask).any()
    assert not engine.owners(op, wanted).any()
    assert engine.visits == 4 * plan["total_work"] + 2 * op.shape[0]


def test_both_inspection_and_complete_dense_work_are_prepaid(monkeypatch):
    op = _geometry()
    plan = new.fee_plan(op)
    mask = np.ones(op.shape[1], dtype=bool)
    def forbidden(*args, **kwargs):
        raise AssertionError("unpaid geometry scan or dense result allocation")
    with monkeypatch.context() as patch:
        patch.setattr(new, "_axis_ranges", forbidden)
        engine = new.FactorizedOwnerEngine(plan["inspection_work"] - 1, enabled=True)
        with pytest.raises(MemoryError):
            engine.compute(op, mask)
        assert engine.visits == 0 and not engine.cache
    with monkeypatch.context() as patch:
        patch.setattr(new.np, "zeros", forbidden)
        engine = new.FactorizedOwnerEngine(plan["total_work"] - 1, enabled=True)
        with pytest.raises(MemoryError):
            engine.compute(op, mask)
        assert engine.visits == plan["inspection_work"] and not engine.cache


def test_sparse_conv_keeps_every_old_visit_plus_fresh_full_kernel_inspection():
    op = _geometry("sparse")
    actual, control = new.FactorizedOwnerEngine(enabled=True), DenseOwnerEngine()
    for mask, transpose in ((np.ones(op.shape[1], dtype=bool), False),
                             (np.arange(op.shape[0], dtype=np.int64) % 3, True)):
        _check_vector(actual.compute(op, mask, transpose=transpose),
                      control.compute(op, mask, transpose=transpose),
                      _scalar(op, mask, transpose=transpose))
    mask = np.arange(op.shape[0]) % 3 != 0
    _check_vector(actual.owners(op, mask), control.owners(op, mask), _scalar(op, mask, owners=True))
    assert actual.visits == control.visits + 3 * new.fee_plan(op)["inspection_work"]
    assert actual.channel_stats["fallback_calls"] == 3
    assert actual.channel_stats["dense_calls"] == 0


def test_default_off_switch_requires_explicit_boolean():
    with pytest.raises(ValueError):
        new.FactorizedOwnerEngine(enabled=1)
    expr, keep = graph_fixture()
    with pytest.raises(ValueError):
        new_graph.graph(expr, keep, 256_000_000, uid_start=37, enabled=1)


def test_cache_bound_remains_one_gib_before_owner_result_allocation():
    op = _geometry()
    engine = new.FactorizedOwnerEngine(enabled=True)
    engine.cache_bytes = 1024 ** 3
    with pytest.raises(MemoryError):
        engine.owners(op, np.ones(op.shape[0], dtype=bool))
    assert not engine.cache and engine.visits == 0


@pytest.mark.parametrize("cap", [-1, 256_000_001, 1.5, True])
def test_unchanged_bounded_work_cap_rejects_invalid_limits(cap):
    with pytest.raises(ValueError):
        new.FactorizedOwnerEngine(cap, enabled=True)


def test_graph_default_off_retains_complete_old_graph_and_costs():
    expr, keep = graph_fixture()
    actual = new_graph.graph(expr, keep, 256_000_000, uid_start=37)
    control = old_graph.graph(expr, keep, 256_000_000, uid_start=37)
    _assert_graph_agreement(actual, control)
    assert actual[2] == control[2]


def test_graph_binding_restores_old_engine_after_budget_failure():
    expr, keep = graph_fixture()
    engine_before = old_graph.DenseOwnerEngine
    with pytest.raises(MemoryError):
        new_graph.graph(expr, keep, 0, uid_start=37, enabled=True)
    assert old_graph.DenseOwnerEngine is engine_before


def test_fresh_graph_source_support_includes_changed_original_source_rows():
    expr, keep = graph_fixture()
    source = expr.terms[0].source
    source.c[2] = .75
    actual = new_graph.graph(expr, keep, 256_000_000, uid_start=37, enabled=True)
    control = old_graph.graph(expr, keep, 256_000_000, uid_start=37)
    _assert_graph_agreement(actual, control)
    _assert_scalar_graph(actual, keep, 37)
    assert all(node["support"][2] for node in actual[0]
               if node["kind"] == "source" and node["source"] is source)


def test_birth_wrapper_default_off_does_not_read_arguments_and_rejects_integer_switch(monkeypatch):
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError("disabled source wrapper inspected an argument")
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled source wrapper invoked original construction")
    monkeypatch.setattr(birth.original, "lift", forbidden)
    value = Unreadable()
    assert birth.lift(value, value, max_work=value) is None
    with pytest.raises(ValueError, match="Boolean"):
        birth.lift(value, value, enabled=1, max_work=value)


@pytest.mark.parametrize("fails", [False, True])
def test_birth_wrapper_binds_enabled_graph_and_restores_after_success_or_failure(monkeypatch, fails):
    expression, keep, answer, graph_answer = object(), object(), object(), object()
    original_graph = birth.original.graph
    calls = []
    def fake_graph(*args, **kwargs):
        assert args == (expression, keep, 123)
        assert kwargs == {"uid_start": 9, "enabled": True}
        calls.append("graph")
        return graph_answer
    def fake_lift(expr, rows, *, enabled, **kwargs):
        assert expr is expression and rows is keep and enabled is True
        assert kwargs == {"max_work": 123, "max_branch_work": 99}
        assert birth.original.graph is not original_graph
        assert birth.original.graph(expression, keep, 123, uid_start=9) is graph_answer
        calls.append("lift")
        if fails:
            raise RuntimeError("intentional source stub failure")
        return answer
    monkeypatch.setattr(birth, "channel_graph", fake_graph)
    monkeypatch.setattr(birth.original, "lift", fake_lift)
    if fails:
        with pytest.raises(RuntimeError, match="intentional source stub failure"):
            birth.lift(expression, keep, enabled=True, max_work=123, max_branch_work=99)
    else:
        assert birth.lift(expression, keep, enabled=True, max_work=123, max_branch_work=99) is answer
    assert calls == ["graph", "lift"]
    assert birth.original.graph is original_graph


def test_birth_wrapper_rejects_conflicting_binding_without_replacing_it(monkeypatch):
    def conflicting(*args, **kwargs):
        raise AssertionError("conflicting graph must not run")
    def forbidden(*args, **kwargs):
        raise AssertionError("conflict must reject before original source construction")
    monkeypatch.setattr(birth.original, "graph", conflicting)
    monkeypatch.setattr(birth.original, "lift", forbidden)
    with pytest.raises(ValueError, match="conflicting"):
        birth.lift(object(), object(), enabled=True)
    assert birth.original.graph is conflicting
