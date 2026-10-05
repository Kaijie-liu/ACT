"""Seven frozen-population controls; no solver, model execution, or search."""

from fractions import Fraction as F

import pytest

from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import interval_capacity as cap


def _iv(lower, upper=None):
    return F(lower), F(lower if upper is None else upper)


def _compile(weights, bias, bounds, pairs, **kwargs):
    return cap.compile_capacities(weights, bias, bounds, pairs, enabled=True, **kwargs)


def _relu(value):
    return max(F(0), value)


def test_point_crossing_agrees_with_d024_closed_form():
    # The archived D024 point rule has K=(1/2,0), H=(0,5/8).
    result = _compile((_iv(1), _iv(F(-11, 10))), _iv(F(1, 50)),
                      (_iv(F(-5, 4), F(5, 4)),) * 2, (_iv(F(-1, 2), F(1, 2)),))
    assert type(result) is dict
    assert result["positive"] == (F(1, 2), F(0))
    assert result["negative"] == (F(0), F(5, 8))
    assert result["unpaired_positive"] == (F(5, 4), F(0))
    assert result["unpaired_negative"] == (F(0), F(11, 8))
    assert result["bias_positive"] == F(1, 50) and result["bias_negative"] == 0
    assert result["matches"] == ((0, 1, F(1)),) and result["nnz"] == 5
    odd = _compile((_iv(1), _iv(2), _iv(-3)), _iv(0),
                   (_iv(F(-5, 4), F(5, 4)),) * 3, (_iv(F(-1, 2), F(1, 2)),))
    assert not odd["matches"]
    assert odd["positive"] == (F(5, 4), F(5, 2), F(0))
    assert odd["negative"] == (F(0), F(0), F(15, 4))


def test_stable_zero_and_one_sided_bounds_keep_zero_choices():
    result = _compile((_iv(1), _iv(-1), _iv(1), _iv(-1)), _iv(0),
                      (_iv(1, 2), _iv(-3, -1), _iv(0), _iv(-2, 1)),
                      (_iv(2, 5), _iv(-1, 2)))
    assert result["positive"] == (F(2), F(0), F(0), F(0))
    assert result["negative"] == (F(0), F(0), F(0), F(1))
    reversed_order = _compile((_iv(1), _iv(-1)), _iv(0),
                              (_iv(-3, -1), _iv(1, 2)), (_iv(-5, -2),))
    assert reversed_order["positive"] == (F(0), F(0))
    assert reversed_order["negative"] == (F(0), F(2))
    zero = _compile((_iv(1), _iv(-1)), _iv(F(1, 50)), (_iv(0), _iv(0)), (_iv(0),))
    assert zero["positive"] == zero["negative"] == (F(0), F(0))
    # Four named unit fixtures, not runtime/network phase enumeration.
    for beta, eta in ((F(0), F(0)), (F(0), F(1)), (F(1), F(0)), (F(1), F(1))):
        g = F(1, 50)
        assert _relu(g) <= zero["bias_positive"] + zero["positive"][0] * beta
        assert _relu(-g) <= zero["bias_negative"] + zero["negative"][1] * eta


def test_interval_coefficients_and_ambiguous_signs():
    result = _compile((_iv(1, 2), _iv(-3, -2), _iv(-1, 2), _iv(1, 3)),
                      _iv(F(-1, 2), F(1, 4)), (_iv(-1, 1),) * 4,
                      (_iv(F(-1, 4), F(1, 2)), _iv(-1, 1)))
    assert result["positive"] == (F(3, 2), F(0), F(2), F(3))
    assert result["negative"] == (F(0), F(9, 4), F(1), F(0))
    assert result["matches"] == ((0, 1, F(1)),)
    assert (result["bias_positive"], result["bias_negative"]) == (F(1, 4), F(1, 2))
    assert all(a <= b for a, b in zip(result["positive"], result["unpaired_positive"]))
    assert all(a <= b for a, b in zip(result["negative"], result["unpaired_negative"]))
    f = (F(1, 4), F(1, 8), F(-1, 4), F(1, 2))
    weights = (F(3, 2), F(-5, 2), F(-1, 2), F(2))
    phases = (F(1), F(1), F(0), F(1))
    g = F(-1, 8) + sum(w * _relu(value) for w, value in zip(weights, f))
    assert _relu(g) <= result["bias_positive"] + sum(k * b for k, b in zip(result["positive"], phases))
    assert _relu(-g) <= result["bias_negative"] + sum(k * b for k, b in zip(result["negative"], phases))
    # The worker's two-product rule equals the inherited four-product bound,
    # including strictly positive activation lower bounds and all weight signs.
    from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import census
    kernel = cap.kernel
    budget = kernel.WorkBudget(enabled=True)
    coefficients = (_iv(1, 2), _iv(-3, -2), _iv(-1, 2), _iv(1, 3))
    bias = _iv(F(-1, 2), F(1, 4))
    activations = (_iv(F(1, 4), F(3, 4)), _iv(F(1, 8), F(5, 8)),
                   _iv(0, F(1, 2)), _iv(0, 1))
    inherited = bias
    for coefficient, activation in zip(coefficients, activations):
        inherited = kernel.add(inherited, kernel.mul(coefficient, activation, budget), budget)
    cached = census.receiver_interval(coefficients, bias, activations, kernel, budget)
    assert cached == inherited == _iv(F(-21, 8), F(11, 2))


def test_canonical_padding_slots_are_not_compacted():
    # f0=x, slot1=padding zero, f2=x, f3=x+y/4, x,y in [-1,1].
    # Removing slot1 would instead pair equal f0/f2 and change the rule.
    result = _compile((_iv(1), _iv(-1), _iv(-1), _iv(1)), _iv(0),
                      (_iv(-1, 1), _iv(0), _iv(-1, 1), _iv(F(-5, 4), F(5, 4))),
                      (_iv(-1, 1), _iv(F(-1, 4), F(1, 4))))
    assert result["positive"] == (F(1), F(0), F(0), F(1, 4))
    assert result["negative"] == (F(0), F(0), F(1, 4), F(0))
    assert result["matches"] == ((0, 1, F(1)), (3, 2, F(1)))
    assert result["positive"][1] == result["negative"][1] == 0
    assert result["nnz"] == 6
    # Only the center pixel is valid. Padding is zero AFTER preprocessing,
    # so its eight slots cannot each contribute the nonzero pre-affine bias.
    from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import census_worker_v2 as helper
    from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import census
    packet = {
        "input_shape": (1, 1, 1, 1),
        "pre_affine": {"scale": (F(2),), "bias": (F(3),)},
        "first_conv": {
            "weight_shape": (1, 1, 3, 3),
            "weights": tuple(F(value) for value in range(1, 10)),
            "bias": (F(7),), "strides": (1, 1), "dilations": (1, 1),
            "pads": (1, 1, 1, 1),
        },
        "first_post_ops": ({"kind": "affine", "scale": (F(2),), "bias": (F(-1),)},),
    }
    kernel = cap.kernel
    budget = kernel.WorkBudget(enabled=True)
    assert helper.receptive(packet["first_conv"], packet["input_shape"], 0, 0) == [(0, 0, 0, 4)]
    post = (helper.post_affine(packet["first_post_ops"], 0, kernel, budget),)
    inherited = helper.first_form(packet, 0, 0, 0, kernel, budget)
    cached = census.first_form(packet, (0, 0, 0), post, helper, kernel, budget)
    assert cached == inherited == (_iv(43), {0: _iv(20)})


def test_shared_original_input_cancellation_uses_old_kernel():
    kernel = cap.kernel
    budget = kernel.WorkBudget(enabled=True)
    box = {0: _iv(-1, 1), 1: _iv(-1, 1)}
    left = (_iv(0), {0: _iv(1), 1: _iv(F(1, 4))})
    right = (_iv(0), {0: _iv(1), 1: _iv(F(-1, 4))})
    difference = kernel.affine_add(left, kernel.affine_scale(right, _iv(-1), budget), budget)
    assert difference == (_iv(0), {1: _iv(F(1, 2))})
    bounds = (kernel.source_box_bounds(left, box, budget), kernel.source_box_bounds(right, box, budget))
    paired = kernel.source_box_bounds(difference, box, budget)
    assert paired == _iv(F(-1, 2), F(1, 2))
    result = _compile((_iv(1), _iv(-1)), _iv(0), bounds, (paired,), budget=budget)
    loose = kernel.add(bounds[0], kernel.neg(bounds[1], budget), budget)
    comparison = _compile((_iv(1), _iv(-1)), _iv(0), bounds, (loose,), budget=budget)
    assert result["positive"][0] == result["negative"][1] == F(1, 2)
    assert comparison["positive"][0] == comparison["negative"][1] == F(5, 4)


def test_general_bias_and_reversed_pair_orientation():
    result = _compile((_iv(F(-11, 10)), _iv(1)), _iv(F(-3, 100), F(-1, 50)),
                      (_iv(F(-5, 4), F(5, 4)),) * 2, (_iv(F(-1, 2), F(1, 2)),))
    assert (result["bias_positive"], result["bias_negative"]) == (F(0), F(3, 100))
    assert result["positive"] == (F(0), F(1, 2))
    assert result["negative"] == (F(5, 8), F(0))
    assert result["matches"] == ((1, 0, F(1)),)
    for f, h in ((F(5, 8), F(3, 8)), (F(-5, 8), F(-3, 8)), (F(1, 8), F(-1, 8))):
        beta, eta = F(int(f > 0)), F(int(h > 0))
        g = F(-1, 40) - F(11, 10) * _relu(f) + _relu(h)
        assert _relu(g) <= result["positive"][1] * eta
        assert _relu(-g) <= result["bias_negative"] + result["negative"][0] * beta


def test_default_types_shapes_work_and_bit_fail_closed():
    disabled = cap.WorkBudget()
    assert cap.compile_capacities(None, None, None, None, budget=disabled) is None
    assert disabled.used == 0
    weights, bias, bounds, pairs = (_iv(1),), _iv(0), (_iv(-1, 1),), ()
    with pytest.raises(cap.KernelError, match="enabled"):
        cap.compile_capacities(weights, bias, bounds, pairs, enabled=1)
    with pytest.raises(cap.kernel.KernelDisabled):
        _compile(weights, bias, bounds, pairs, budget=disabled)
    with pytest.raises(cap.KernelError, match="WorkBudget"):
        _compile(weights, bias, bounds, pairs, budget=object())
    with pytest.raises(cap.BudgetExceeded):
        _compile(weights, bias, bounds, pairs, budget=cap.WorkBudget(enabled=True, limit=0))
    invalid = (
        ([weights[0]], bias, bounds, pairs),
        (weights, bias, (), pairs),
        (weights, bias, bounds, (_iv(0),)),
        (((F(1), 1.0),), bias, bounds, pairs),
        ((_iv(2, 1),), bias, bounds, pairs),
        (weights, bias, (_iv(1, -1),), pairs),
        ((_iv(F(1 << 512)),), bias, bounds, pairs),
        (weights, _iv(F(1, 1 << 512)), bounds, pairs),
        (weights * (cap.MAX_SLOTS + 1), bias, bounds, pairs),
    )
    for args in invalid:
        with pytest.raises(cap.KernelError):
            _compile(*args)
    # Legal input endpoints, but a newly formed product exceeds 512 bits.
    with pytest.raises(cap.KernelError, match="bit bound"):
        _compile((_iv(F(1 << 511)),), bias, (_iv(-1, 2),), ())
    with pytest.raises(cap.KernelError, match="bit bound"):
        _compile(weights, _iv(F(1 << 16)), bounds, pairs,
                 budget=cap.WorkBudget(enabled=True, max_bits=16))
    budget = cap.WorkBudget(enabled=True)
    budget.charge(11)
    _compile(weights, bias, bounds, pairs, budget=budget)
    first_used = budget.used
    _compile(weights, bias, bounds, pairs, budget=budget)
    assert 11 < first_used < budget.used <= budget.limit
    # The real worker's bounded ledger and streaming encoder share one meter.
    import hashlib
    import json
    from pathlib import Path
    import tempfile
    from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence
    shared = (F(2, 3), F(3, 4))
    raw = b"abc"
    raw_digest = hashlib.sha256(raw).hexdigest()
    roots = {"first": shared, "second": shared, "raw": raw}
    meter = evidence.Meter(limit=40_000)
    ledger = evidence.bounded_ledger(roots, meter)
    assert ledger["retained_entries"] > 0 and ledger["held_instance_bytes"] > 0
    before_encode = meter.used
    with tempfile.TemporaryDirectory(prefix="d025-evidence-") as temporary:
        directory = Path(temporary)
        path = directory / "tiny-evidence.partial"
        receipt = evidence.write_evidence(path, roots, meter, {id(raw): raw_digest})
        encoded = path.read_bytes()
        assert json.loads(encoded) == {
            "first": [[2, 3], [3, 4]], "second": [[2, 3], [3, 4]],
            "raw": {"byte_count": 3, "sha256": raw_digest},
        }
        assert receipt == {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}
        assert before_encode < meter.used <= meter.limit
        before_second_ledger = meter.used
        evidence.bounded_ledger(roots, meter)
        assert meter.used > before_second_ledger
        with pytest.raises(ValueError, match="budget exhausted"):
            evidence.bounded_ledger(roots, evidence.Meter(limit=0))
        failed_path = directory / "unpaid-evidence.partial"
        with pytest.raises(ValueError, match="budget exhausted"):
            evidence.write_evidence(failed_path, roots, evidence.Meter(limit=0), {id(raw): raw_digest})
        assert failed_path.read_bytes() == b""
