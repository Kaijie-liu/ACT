from __future__ import annotations

import numpy as np
import pytest

from independent_vnnlib_replay import evaluate_vnnlib, parse_sexpressions


SPEC = """
; concrete unsafe-set property
(set-logic QF_LRA)
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (>= X_0 0))
(assert (<= X_0 1))
(assert (= X_1 (/ 1 2)))
(assert (or (and (>= Y_1 Y_0)) (and (> (- Y_0 Y_1) 2))))
"""


def test_zero_tolerance_input_and_unsafe_formula_pass():
    result = evaluate_vnnlib(SPEC, [0.0, 0.5], [1.0, 1.25])
    assert result.input_holds
    assert result.unsafe_holds
    assert result.all_assertions_hold
    assert result.input_min_slack == 0.0
    assert result.unsafe_min_slack == 0.25
    assert result.input_assertion_count == 3
    assert result.unsafe_assertion_count == 1


def test_nonstrict_boundary_passes_but_strict_boundary_does_not():
    nonstrict = evaluate_vnnlib(SPEC, [1.0, 0.5], [1.0, 1.0])
    assert nonstrict.all_assertions_hold
    strict_only = SPEC.replace(
        "(or (and (>= Y_1 Y_0)) (and (> (- Y_0 Y_1) 2)))",
        "(> Y_1 Y_0)",
    )
    strict = evaluate_vnnlib(strict_only, [1.0, 0.5], [1.0, 1.0])
    assert strict.input_holds
    assert not strict.unsafe_holds
    assert strict.unsafe_min_slack == 0.0


def test_input_violation_cannot_be_sat_witness():
    result = evaluate_vnnlib(SPEC, [1.0001, 0.5], [0.0, 2.0])
    assert not result.input_holds
    assert result.unsafe_holds
    assert not result.all_assertions_hold
    assert result.input_min_slack < 0.0


def test_arithmetic_and_boolean_operators():
    spec = SPEC.replace(
        "(or (and (>= Y_1 Y_0)) (and (> (- Y_0 Y_1) 2)))",
        "(and (not (< Y_1 Y_0)) (=> (> Y_1 0) (>= (* 2 Y_1) (+ Y_0 1))))",
    )
    assert evaluate_vnnlib(spec, [0.5, 0.5], [1.0, 1.0]).all_assertions_hold


def test_nonfinite_and_mixed_assertions_fail_closed():
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_vnnlib(SPEC, [np.nan, 0.5], [1.0, 2.0])
    mixed = SPEC.replace("(assert (>= X_0 0))", "(assert (>= X_0 Y_0))")
    with pytest.raises(ValueError, match="purely X or purely Y"):
        evaluate_vnnlib(mixed, [0.5, 0.5], [1.0, 2.0])


@pytest.mark.parametrize(
    "text,error",
    [
        ("(assert (>= X_0 0)", "unclosed"),
        (")",
         "unexpected closing"),
        ("(define-fun f () Real 0)", "unsupported top-level"),
    ],
)
def test_malformed_or_unsupported_syntax_fails_closed(text, error):
    if "define-fun" in text:
        with pytest.raises(ValueError, match=error):
            evaluate_vnnlib(text, [], [])
    else:
        with pytest.raises(ValueError, match=error):
            parse_sexpressions(text)
