#!/usr/bin/env python3
"""Fraction and scaling audits for final-ReLU property tail folding."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import random
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Layer, Net
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuildError,
    _property_relu_upper_planes,
    build_operator_hz,
)
from act.back_end.hybridz_tf.test_operator_residual_normal_form import (
    _scalar_relu_toy,
)
from act.back_end.hybridz_tf.test_operator_add_fusion import (
    _assemble_width_toy,
    _dense_matrix,
    _input_layers,
    _layer,
    _wide_layer,
)
from act.back_end.solver.solver_hz import hz_objbound_decide
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


def _f(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_property(
    C: np.ndarray,
    threshold: float,
    weight: np.ndarray,
    bias: np.ndarray,
    point: tuple[float, ...],
) -> Fraction:
    relu = [max(Fraction(0), _f(value)) for value in point]
    outputs = [
        sum((_f(weight[row, col]) * relu[col] for col in range(len(point))), Fraction(0))
        + _f(bias[row])
        for row in range(weight.shape[0])
    ]
    return (
        sum((_f(C[row]) * outputs[row] for row in range(C.size)), Fraction(0))
        - _f(threshold)
    )


def _stored_plane(
    coefficients: np.ndarray,
    intercept: float,
    point: tuple[float, ...],
) -> Fraction:
    return (
        _f(intercept)
        + sum(
            (_f(coefficients[col]) * _f(point[col]) for col in range(len(point))),
            Fraction(0),
        )
    )


def _verified_relu_net(threshold: float) -> Net:
    dtype = torch.float64
    input_vars = [0]
    pre_vars = [1]
    relu_vars = [2]
    output_vars = [3]
    identity = torch.ones((1, 1), dtype=dtype)
    zero = torch.zeros(1, dtype=dtype)
    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0], dtype=dtype),
        d=torch.tensor([float(threshold)], dtype=dtype),
    ).encode_linear(
        B=1,
        n_out=1,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=dtype),
                "ub": torch.tensor([[1.0]], dtype=dtype),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": identity,
                "bias": zero,
                "in_features": 1,
                "out_features": 1,
            },
            in_vars=input_vars,
            out_vars=pre_vars,
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=pre_vars,
            out_vars=relu_vars,
        ),
        Layer(
            id=4,
            kind="DENSE",
            params={
                "weight": identity,
                "bias": zero,
                "in_features": 1,
                "out_features": 1,
            },
            in_vars=relu_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=5,
            kind="ASSERT",
            params=assertion,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _verified_cancellation_net(threshold: float) -> Net:
    dtype = torch.float64
    input_vars = [0]
    pre_vars = [1, 2]
    relu_vars = [3, 4]
    output_vars = [5]
    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0], dtype=dtype),
        d=torch.tensor([float(threshold)], dtype=dtype),
    ).encode_linear(
        B=1,
        n_out=1,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=dtype),
                "ub": torch.tensor([[1.0]], dtype=dtype),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor([[1.0], [1.0]], dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
            in_vars=input_vars,
            out_vars=pre_vars,
        ),
        Layer(
            id=3,
            kind="RELU",
            params={},
            in_vars=pre_vars,
            out_vars=relu_vars,
        ),
        Layer(
            id=4,
            kind="DENSE",
            params={
                "weight": torch.tensor([[1.0, -1.0]], dtype=dtype),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 2,
                "out_features": 1,
            },
            in_vars=relu_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=5,
            kind="ASSERT",
            params=assertion,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        succs={0: [1], 1: [2], 2: [3], 3: [4], 4: [5], 5: []},
    )


def _verified_add_correlation_net(threshold: float) -> Net:
    """Two equal ADD coordinates whose final difference is exactly zero."""

    dtype = torch.float64
    input_vars = [0]
    left_vars = [1, 2]
    right_vars = [3, 4]
    add_vars = [5, 6]
    relu_vars = [7, 8]
    output_vars = [9, 10]
    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0, -1.0], dtype=dtype),
        d=torch.tensor([float(threshold)], dtype=dtype),
    ).encode_linear(
        B=1,
        n_out=2,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[1.0]], dtype=dtype),
                "ub": torch.tensor([[2.0]], dtype=dtype),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor([[1.0], [1.0]], dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
            in_vars=input_vars,
            out_vars=left_vars,
        ),
        Layer(
            id=3,
            kind="DENSE",
            params={
                "weight": torch.zeros((2, 1), dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
            in_vars=input_vars,
            out_vars=right_vars,
        ),
        Layer(
            id=4,
            kind="ADD",
            params={"x_vars": left_vars, "y_vars": right_vars},
            in_vars=left_vars + right_vars,
            out_vars=add_vars,
        ),
        Layer(
            id=5,
            kind="RELU",
            params={},
            in_vars=add_vars,
            out_vars=relu_vars,
        ),
        Layer(
            id=6,
            kind="DENSE",
            params={
                "weight": torch.eye(2, dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 2,
                "out_features": 2,
            },
            in_vars=relu_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=7,
            kind="ASSERT",
            params=assertion,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={
            0: [],
            1: [0],
            2: [1],
            3: [1],
            4: [2, 3],
            5: [4],
            6: [5],
            7: [6],
        },
        succs={
            0: [1],
            1: [2, 3],
            2: [4],
            3: [4],
            4: [5],
            5: [6],
            6: [7],
            7: [],
        },
    )


def _verified_live_tail_combo_net(threshold: float) -> Net:
    """ADD-fusion/property-tail composition with an exact zero output."""

    dtype = torch.float64
    input_vars = [0]
    left_vars = [1]
    right_vars = [2]
    add_vars = [3]
    flatten_vars = [4]
    pre_vars = [5, 6, 7]
    relu_vars = [8, 9, 10]
    output_vars = [11]
    assertion = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0], dtype=dtype),
        d=torch.tensor([float(threshold)], dtype=dtype),
    ).encode_linear(
        B=1,
        n_out=1,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=dtype),
                "ub": torch.tensor([[1.0]], dtype=dtype),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind="DENSE",
            params={
                "weight": torch.tensor([[0.5]], dtype=dtype),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 1,
                "out_features": 1,
            },
            in_vars=input_vars,
            out_vars=left_vars,
        ),
        Layer(
            id=3,
            kind="DENSE",
            params={
                "weight": torch.tensor([[0.5]], dtype=dtype),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 1,
                "out_features": 1,
            },
            in_vars=input_vars,
            out_vars=right_vars,
        ),
        Layer(
            id=4,
            kind="ADD",
            params={"x_vars": left_vars, "y_vars": right_vars},
            in_vars=left_vars + right_vars,
            out_vars=add_vars,
        ),
        Layer(
            id=5,
            kind="FLATTEN",
            params={},
            in_vars=add_vars,
            out_vars=flatten_vars,
        ),
        Layer(
            id=6,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [[1.0], [1.0], [0.0]], dtype=dtype
                ),
                "bias": torch.tensor([0.0, 0.0, -1.0], dtype=dtype),
                "in_features": 1,
                "out_features": 3,
            },
            in_vars=flatten_vars,
            out_vars=pre_vars,
        ),
        Layer(
            id=7,
            kind="RELU",
            params={},
            in_vars=pre_vars,
            out_vars=relu_vars,
        ),
        Layer(
            id=8,
            kind="DENSE",
            params={
                "weight": torch.tensor(
                    [[1.0, -1.0, 0.0]], dtype=dtype
                ),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 3,
                "out_features": 1,
            },
            in_vars=relu_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=9,
            kind="ASSERT",
            params=assertion,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={
            0: [],
            1: [0],
            2: [1],
            3: [1],
            4: [2, 3],
            5: [4],
            6: [5],
            7: [6],
            8: [7],
            9: [8],
        },
        succs={
            0: [1],
            1: [2, 3],
            2: [4],
            3: [4],
            4: [5],
            5: [6],
            6: [7],
            7: [8],
            8: [9],
            9: [],
        },
    )


class PropertyTailUpperTests(unittest.TestCase):
    def test_64_random_dyadic_networks_all_phase_endpoints(self) -> None:
        rng = random.Random(492017)
        for case in range(64):
            output_dim = 3
            width = 3
            C = np.asarray(
                [
                    [
                        float(
                            Fraction(
                                rng.choice((-3, -2, -1, 1, 2, 3)),
                                rng.choice((1, 2, 4)),
                            )
                        )
                        for _ in range(output_dim)
                    ],
                    [
                        float(
                            Fraction(
                                rng.choice((-2, -1, 0, 1, 2)),
                                rng.choice((1, 2, 4)),
                            )
                        )
                        for _ in range(output_dim)
                    ],
                ],
                dtype=np.float64,
            )
            weight = np.asarray(
                [
                    [
                        float(
                            Fraction(
                                rng.choice((-3, -2, -1, 0, 1, 2, 3)),
                                rng.choice((1, 2, 4)),
                            )
                        )
                        for _ in range(width)
                    ]
                    for _ in range(output_dim)
                ],
                dtype=np.float64,
            )
            bias = np.asarray(
                [
                    float(Fraction(rng.randint(-3, 3), 4))
                    for _ in range(output_dim)
                ],
                dtype=np.float64,
            )
            thresholds = np.asarray(
                [
                    float(Fraction(rng.randint(-3, 3), 4)),
                    float(Fraction(rng.randint(-3, 3), 4)),
                ],
                dtype=np.float64,
            )
            phase_kinds = [
                rng.choice(("unstable", "active", "inactive"))
                for _ in range(width)
            ]
            lower = []
            upper = []
            for phase in phase_kinds:
                if phase == "unstable":
                    lower.append(float(Fraction(-rng.randint(1, 7), 4)))
                    upper.append(float(Fraction(rng.randint(1, 7), 4)))
                elif phase == "active":
                    lo = Fraction(rng.randint(0, 3), 4)
                    lower.append(float(lo))
                    upper.append(float(lo + Fraction(rng.randint(1, 5), 4)))
                else:
                    hi = Fraction(-rng.randint(0, 3), 4)
                    upper.append(float(hi))
                    lower.append(float(hi - Fraction(rng.randint(1, 5), 4)))
            lower_np = np.asarray(lower, dtype=np.float64)
            upper_np = np.asarray(upper, dtype=np.float64)
            planes, intercepts, receipt = _property_relu_upper_planes(
                C,
                thresholds,
                sp.csr_matrix(weight),
                bias,
                lower_np,
                upper_np,
            )
            self.assertTrue(receipt["proof_authority"])
            choices = []
            for lo, hi in zip(lower_np, upper_np):
                row = [float(lo), float(hi)]
                if lo < 0.0 < hi:
                    row.append(0.0)
                choices.append(tuple(dict.fromkeys(row)))
            for point in itertools.product(*choices):
                for rival in range(C.shape[0]):
                    true_value = _exact_property(
                        C[rival],
                        thresholds[rival],
                        weight,
                        bias,
                        point,
                    )
                    plane_value = _stored_plane(
                        planes[rival],
                        intercepts[rival],
                        point,
                    )
                    self.assertGreaterEqual(
                        plane_value,
                        true_value,
                        (case, rival, point),
                    )

    def test_asymmetric_decimal_and_subnormal_endpoints(self) -> None:
        tiny = float(np.nextafter(0.0, np.inf))
        cases = (
            (-10.0, 0.01, 1.0),
            (-10.0, 0.2, 1.0),
            (-tiny, tiny, 1.0),
            (-1.0, 1.0, -1.0),
        )
        for lower, upper, coefficient in cases:
            planes, intercepts, _receipt = _property_relu_upper_planes(
                np.asarray([[coefficient]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
                sp.csr_matrix([[1.0]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
                np.asarray([lower], dtype=np.float64),
                np.asarray([upper], dtype=np.float64),
            )
            for point in (lower, 0.0, upper):
                true_value = _f(coefficient) * max(Fraction(0), _f(point))
                plane_value = _f(planes[0, 0]) * _f(point) + _f(intercepts[0])
                self.assertGreaterEqual(
                    plane_value,
                    true_value,
                    (lower, upper, coefficient, point),
                )

    def test_negative_unstable_coefficient_uses_zero_lower_facet(self) -> None:
        planes, intercepts, receipt = _property_relu_upper_planes(
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            sp.csr_matrix([[-3.0]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([-2.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        )
        self.assertEqual(planes[0, 0], 0.0)
        self.assertGreaterEqual(intercepts[0], 0.0)
        self.assertEqual(receipt["nonpositive_unstable_zero_facets"], 1)

    def test_arbitrary_negative_alpha_is_fraction_endpoint_sound(self) -> None:
        rng = np.random.default_rng(20260727)
        for _case in range(48):
            lower = -rng.uniform(1.0e-8, 20.0, size=4)
            upper = rng.uniform(1.0e-8, 20.0, size=4)
            weight = -rng.uniform(1.0e-8, 20.0, size=(1, 4))
            alpha = rng.uniform(0.0, 1.0, size=(1, 4))
            planes, intercepts, receipt = _property_relu_upper_planes(
                np.asarray([[1.0]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
                sp.csr_matrix(weight),
                np.asarray([0.0], dtype=np.float64),
                lower,
                upper,
                negative_alpha=alpha,
            )
            self.assertEqual(receipt["nonzero_negative_alpha"], 4)
            self.assertEqual(
                receipt["nonpositive_unstable_zero_facets"], 0
            )
            for point in itertools.product(
                *[(float(lo), 0.0, float(hi)) for lo, hi in zip(lower, upper)]
            ):
                true_value = _exact_property(
                    np.asarray([1.0]),
                    0.0,
                    weight,
                    np.asarray([0.0]),
                    point,
                )
                plane_value = _stored_plane(
                    planes[0], intercepts[0], point
                )
                self.assertGreaterEqual(
                    plane_value, true_value, (_case, point)
                )

    def test_negative_alpha_validation_and_subnormal_remainder(self) -> None:
        tiny = float(np.nextafter(0.0, np.inf))
        planes, intercepts, receipt = _property_relu_upper_planes(
            np.asarray([[-tiny]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            sp.csr_matrix([[0.75]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([-1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            negative_alpha=np.asarray([[1.0]], dtype=np.float64),
        )
        exact_q = -_f(tiny) * Fraction(3, 4)
        for point in (-1.0, 0.0, 1.0):
            true_value = exact_q * max(Fraction(0), _f(point))
            plane_value = (
                _f(planes[0, 0]) * _f(point) + _f(intercepts[0])
            )
            self.assertGreaterEqual(plane_value, true_value)
        self.assertEqual(receipt["negative_d_below_exact_q"], 1)
        self.assertGreater(
            receipt["max_negative_exact_endpoint_requirement"], 0.0
        )

        for invalid in (
            np.asarray([[-0.01]], dtype=np.float64),
            np.asarray([[1.01]], dtype=np.float64),
            np.asarray([[np.nan]], dtype=np.float64),
            np.zeros((2, 1), dtype=np.float64),
        ):
            with self.assertRaisesRegex(
                ValueError, "negative alpha"
            ):
                _property_relu_upper_planes(
                    np.asarray([[1.0]], dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                    sp.csr_matrix([[-1.0]], dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray([-1.0], dtype=np.float64),
                    np.asarray([1.0], dtype=np.float64),
                    negative_alpha=invalid,
                )

    def test_large_class_sparse_property_shape_is_bounded(self) -> None:
        rng = np.random.default_rng(77)
        output_dim = width = 100
        C = np.zeros((99, output_dim), dtype=np.float64)
        for rival in range(99):
            C[rival, rival] = 1.0
            C[rival, 99] = -1.0
        weight = rng.normal(size=(output_dim, width)).astype(np.float64)
        bias = rng.normal(size=output_dim).astype(np.float64)
        lower = -rng.uniform(0.1, 2.0, size=width)
        upper = rng.uniform(0.1, 2.0, size=width)
        started = time.monotonic()
        planes, intercepts, receipt = _property_relu_upper_planes(
            C,
            np.zeros(99, dtype=np.float64),
            sp.csr_matrix(weight),
            bias,
            lower,
            upper,
        )
        elapsed = time.monotonic() - started
        self.assertEqual(planes.shape, (99, 100))
        self.assertEqual(intercepts.shape, (99,))
        self.assertTrue(np.all(np.isfinite(planes)))
        self.assertTrue(np.all(np.isfinite(intercepts)))
        self.assertLess(elapsed, 5.0)
        self.assertEqual(receipt["rivals"], 99)

    def test_operator_tail_prunes_final_relu_and_certifies_safe_only(self) -> None:
        toy = _scalar_relu_toy()
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=np.asarray([[1.0]], dtype=np.float64),
            property_upper_thresholds=np.asarray([2.0], dtype=np.float64),
        )
        self.assertTrue(build.property_upper_output)
        tail = build.metadata["property_tail_upper"]
        self.assertTrue(tail["proof_authority"])
        self.assertTrue(tail["safe_only"])
        self.assertEqual(tail["relu_layer_id"], 3)
        self.assertEqual(tail["output_layer_id"], 4)
        self.assertEqual(tail["pruned_n_cont"], 1)
        self.assertEqual(tail["pruned_ub_rows"], 2)
        self.assertEqual(
            build.metadata["property_upper_semantics"],
            "safe_only_affine_dominating_rows",
        )

        verdict, witness = hz_objbound_decide(
            build.hz,
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)

    def test_operator_tail_handles_two_opposite_property_rows(self) -> None:
        toy = _scalar_relu_toy(
            pre_weight=Fraction(7, 5),
            pre_bias=Fraction(-1, 9),
            out_weight=Fraction(5, 7),
        )
        C = np.asarray([[1.0], [-1.0]], dtype=np.float64)
        thresholds = np.asarray([3.0, 0.25], dtype=np.float64)
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=C,
            property_upper_thresholds=thresholds,
        )
        self.assertEqual(build.hz.n_out, 2)
        self.assertEqual(
            build.metadata["property_tail_upper"]["rivals"],
            2,
        )
        verdict, witness = hz_objbound_decide(
            build.hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=1.0,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)

    def test_operator_selects_cancellation_alpha_per_rival_without_regression(
        self,
    ) -> None:
        input_layer, spec = _input_layers(-1, 1)
        pre = _dense_matrix(2, [[1], [1]], [0, 0])
        relu = _wide_layer(3, "RELU", 2)
        out = _dense_matrix(4, [[1, -1]], [0])
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, out, _layer(5, "ASSERT")],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            input_lb=-1,
            input_ub=1,
        )
        kwargs = {
            "exact_budget": 0,
            "materialize_add": True,
            "property_upper_C": np.asarray([[1.0]], dtype=np.float64),
            "property_upper_thresholds": np.asarray(
                [0.0], dtype=np.float64
            ),
        }
        baseline = build_operator_hz(
            toy.net, toy.facts, toy.facts, **kwargs
        )
        candidate = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            **kwargs,
            property_tail_alpha_steps=40,
            property_tail_alpha_time_limit=2.0,
            property_tail_alpha_max_cells=100,
            property_tail_alpha_device="cpu",
        )
        def cube_upper(build, row: int) -> float:
            return float(build.hz.c[row]) + sum(
                float(np.abs(matrix.getrow(row).data).sum())
                for matrix in (build.hz.Gc, build.hz.Gb)
            )

        self.assertEqual(baseline.hz.n_out, 1)
        self.assertEqual(candidate.hz.n_out, 2)
        self.assertEqual(candidate.property_upper_row_groups, ((0, 1),))
        self.assertLess(
            cube_upper(candidate, 1),
            cube_upper(candidate, 0),
        )
        alpha_receipt = candidate.metadata["property_tail_upper"][
            "negative_alpha_candidates"
        ]
        self.assertEqual(alpha_receipt["selected_rivals"], 1)
        self.assertGreater(
            alpha_receipt["cube_upper_improvement_max"], 0.4
        )
        self.assertTrue(
            alpha_receipt["exact_candidate_audit"]["proof_authority"]
        )
        verdict, witness = hz_objbound_decide(
            candidate.hz,
            np.eye(2, dtype=np.float64),
            np.full(2, 0.75, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            safe_row_groups=candidate.property_upper_row_groups,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)

    def test_operator_pairhull_joint_plane_closes_cancellation_gap(
        self,
    ) -> None:
        """PairHull must improve alpha without replacing either fallback."""

        from act.back_end.hybridz_tf.property_pairhull_candidates import (
            verify_property_pairhull_candidates_receipt,
        )

        input_layer, spec = _input_layers(-1, 1)
        pre = _dense_matrix(2, [[1], [1]], [0, 0])
        relu = _wide_layer(3, "RELU", 2)
        out = _dense_matrix(4, [[1, -1]], [0])
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, out, _layer(5, "ASSERT")],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            input_lb=-1,
            input_ub=1,
        )
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=np.asarray([[1.0]], dtype=np.float64),
            property_upper_thresholds=np.asarray(
                [0.25], dtype=np.float64
            ),
            property_tail_alpha_steps=40,
            property_tail_alpha_time_limit=2.0,
            property_tail_alpha_max_cells=100,
            property_tail_alpha_device="cpu",
            property_tail_pairhull_budget=1,
            property_tail_pairhull_time_limit=1.5,
        )

        def cube_upper(row: int) -> float:
            return float(build.hz.c[row]) + sum(
                float(np.abs(matrix.getrow(row).data).sum())
                for matrix in (build.hz.Gc, build.hz.Gb)
            )

        self.assertEqual(build.hz.n_out, 3)
        self.assertEqual(build.property_upper_row_groups, ((0, 1, 2),))
        tail = build.metadata["property_tail_upper"]
        self.assertEqual(
            tail["alternative_plane_kinds"],
            [
                "negative_alpha_materialized",
                "pairhull_joint_materialized",
            ],
        )
        pairhull = tail["pairhull_candidates"]
        self.assertEqual(pairhull["status"], "applied")
        self.assertTrue(pairhull["proof_authority"])
        self.assertTrue(pairhull["exact_search_complete"])
        self.assertEqual(pairhull["global_pair_count"], 1)
        self.assertEqual(pairhull["selected_rivals"], 1)
        self.assertEqual(pairhull["selected_rival_ids"], [0])
        self.assertEqual(pairhull["selected_pair_indices"], [[0, 1]])
        self.assertTrue(
            verify_property_pairhull_candidates_receipt(pairhull)
        )
        self.assertTrue(
            verify_property_pairhull_candidates_receipt(
                pairhull["candidate_receipt"]
            )
        )
        self.assertLess(cube_upper(2), cube_upper(1))
        self.assertLess(cube_upper(2), 0.0)

        verdict, witness = hz_objbound_decide(
            build.hz,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            safe_row_groups=build.property_upper_row_groups,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)

    def test_operator_pairhull_post_retention_timeout_cleans_fallback(
        self,
    ) -> None:
        """A late timeout must restore both the HZ and its receipt."""

        from act.back_end.hybridz_tf import operator_hz as operator_module
        from act.back_end.hybridz_tf import (
            property_pairhull_candidates as candidate_module,
        )
        from act.back_end.verifier import (
            _validate_property_tail_pairhull_receipt,
        )

        input_layer, spec = _input_layers(-1, 1)
        pre = _dense_matrix(2, [[1], [1]], [0, 0])
        relu = _wide_layer(3, "RELU", 2)
        out = _dense_matrix(4, [[1, -1]], [0])
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, out, _layer(5, "ASSERT")],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            input_lb=-1,
            input_ub=1,
        )

        original_candidates = (
            candidate_module.build_property_pairhull_candidates
        )
        candidate_returned = False

        def wrapped_candidates(*args, **kwargs):
            nonlocal candidate_returned
            result = original_candidates(*args, **kwargs)
            self.assertEqual(result.status, "generated")
            self.assertEqual(result.rival_ids.tolist(), [0])
            candidate_returned = True
            return result

        def controlled_monotonic() -> float:
            # Candidate construction and its final exact audit finish at t=0.
            # Immediately afterwards the operator observes that materializing
            # the retained row crossed its local 1.5-second deadline.
            return 2.0 if candidate_returned else 0.0

        with mock.patch.object(
            candidate_module,
            "build_property_pairhull_candidates",
            side_effect=wrapped_candidates,
        ), mock.patch.object(
            operator_module.time,
            "monotonic",
            side_effect=controlled_monotonic,
        ):
            build = build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=0,
                materialize_add=True,
                property_upper_C=np.asarray([[1.0]], dtype=np.float64),
                property_upper_thresholds=np.asarray(
                    [0.25], dtype=np.float64
                ),
                property_tail_alpha_steps=40,
                property_tail_alpha_time_limit=2.0,
                property_tail_alpha_max_cells=100,
                property_tail_alpha_device="cpu",
                property_tail_pairhull_budget=1,
                property_tail_pairhull_time_limit=1.5,
            )

        tail = build.metadata["property_tail_upper"]
        pairhull = tail["pairhull_candidates"]
        self.assertEqual(pairhull["status"], "error_fallback_foundations")
        self.assertFalse(pairhull["proof_authority"])
        self.assertFalse(pairhull["exact_search_complete"])
        self.assertFalse(pairhull["full_row_outward_affine"])
        self.assertEqual(pairhull["selected_rivals"], 0)
        self.assertEqual(pairhull["selected_rival_ids"], [])
        self.assertEqual(pairhull["selected_pair_indices"], [])
        self.assertEqual(pairhull["selected_foundation_indices"], [])
        self.assertIn(
            "total time limit expired", pairhull.get("error", "")
        )
        for applied_only_field in (
            "guarded_cube_improved_rivals",
            "guarded_cube_improvement_sum",
            "guarded_cube_improvement_max",
            "operator_discarded_nonimproving_rows",
        ):
            self.assertNotIn(applied_only_field, pairhull)

        empty_rows = np.empty((0, 2), dtype=np.float64)
        empty_intercepts = np.empty(0, dtype=np.float64)

        self.assertEqual(
            pairhull["candidate_rows_sha256"],
            hashlib.sha256(
                np.ascontiguousarray(empty_rows).tobytes()
            ).hexdigest(),
        )
        self.assertEqual(
            pairhull["candidate_intercepts_sha256"],
            hashlib.sha256(
                np.ascontiguousarray(empty_intercepts).tobytes()
            ).hexdigest(),
        )

        # Baseline and negative-alpha foundations survive; the appended
        # PairHull row and its group membership are both rolled back.
        self.assertEqual(build.hz.n_out, 2)
        self.assertEqual(build.property_upper_row_groups, ((0, 1),))
        self.assertEqual(
            tail["alternative_plane_kinds"],
            ["negative_alpha_materialized"],
        )
        self.assertEqual(tail["alternative_plane_rival_ids"], [0])

        self.assertTrue(
            candidate_module.verify_pairhull_candidate_receipt(pairhull)
        )
        self.assertTrue(
            candidate_module.verify_pairhull_candidate_receipt(
                pairhull["candidate_receipt"]
            )
        )
        self.assertTrue(
            _validate_property_tail_pairhull_receipt(
                pairhull,
                requested_budget=1,
                requested_time_limit=1.5,
                alternative_rivals=tail[
                    "alternative_plane_rival_ids"
                ],
                alternative_kinds=tail["alternative_plane_kinds"],
                rival_count=tail["baseline_plane_count"],
            )
        )

    def test_final_add_source_plane_recovers_cross_row_correlation(
        self,
    ) -> None:
        """The source row proves a cancellation hidden by ADD boxes.

        Both coordinates of the materialized ADD equal the same scalar input.
        Its independent local box loses that cross-row equality for a cube
        objective, while the pre-materialization source expression preserves
        the shared generator.  The original materialized row and both ADD
        equality bands must remain in the exported HZ.
        """

        input_layer, spec = _input_layers(1, 2)
        left = _dense_matrix(2, [[1], [2]], [0, 0])
        right = _dense_matrix(3, [[2], [1]], [0, 0])
        add = _wide_layer(4, "ADD", 2)
        flatten = _wide_layer(5, "FLATTEN", 2)
        bridge_dense = _dense_matrix(
            6, [[2, -1], [-1, 2]], [10, 10]
        )
        relu = _wide_layer(7, "RELU", 2)
        out = _dense_matrix(8, [[1, 0], [0, 1]], [0, 0])
        toy = _assemble_width_toy(
            [
                input_layer,
                spec,
                left,
                right,
                add,
                flatten,
                bridge_dense,
                relu,
                out,
                _wide_layer(9, "ASSERT", 2),
            ],
            {
                0: [],
                1: [0],
                2: [1],
                3: [1],
                4: [2, 3],
                5: [4],
                6: [5],
                7: [6],
                8: [7],
                9: [8],
            },
            input_lb=1,
            input_ub=2,
        )
        kwargs = {
            "exact_budget": 0,
            "materialize_add": True,
            "property_upper_C": np.asarray(
                [[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64
            ),
            "property_upper_thresholds": np.asarray(
                [0.1, 0.1], dtype=np.float64
            ),
        }
        baseline = build_operator_hz(
            toy.net, toy.facts, toy.facts, **kwargs
        )
        source = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            **kwargs,
            property_tail_add_source_planes=True,
        )

        def cube_upper(build, row: int) -> float:
            return float(build.hz.c[row]) + sum(
                float(np.abs(matrix.getrow(row).data).sum())
                for matrix in (build.hz.Gc, build.hz.Gb)
            )

        # Independent exact graph oracle exercises both ADD operands and the
        # nontrivial bridge matrix/bias at both interval endpoints.
        for exact_x in (Fraction(1), Fraction(2)):
            add_left = (exact_x, 2 * exact_x)
            add_right = (2 * exact_x, exact_x)
            z0 = add_left[0] + add_right[0]
            z1 = add_left[1] + add_right[1]
            bridge0 = 2 * z0 - z1 + 10
            bridge1 = -z0 + 2 * z1 + 10
            self.assertGreater(bridge0, 0)
            self.assertGreater(bridge1, 0)
            self.assertEqual(
                bridge0 - bridge1 - Fraction(1, 10),
                Fraction(-1, 10),
            )
        self.assertGreater(cube_upper(baseline, 0), 0.5)
        self.assertGreater(cube_upper(baseline, 1), 0.5)
        self.assertEqual(source.hz.n_out, 4)
        self.assertEqual(
            source.property_upper_row_groups, ((0, 2), (1, 3))
        )
        for baseline_row, source_row in ((0, 2), (1, 3)):
            self.assertLess(cube_upper(source, source_row), -0.09)
            self.assertGreater(
                cube_upper(source, baseline_row)
                - cube_upper(source, source_row),
                0.5,
            )

        tail = source.metadata["property_tail_upper"]
        add_receipt = tail["add_source_planes"]
        self.assertEqual(tail["alternative_plane_rival_ids"], [0, 1])
        self.assertEqual(
            tail["alternative_plane_kinds"],
            ["add_source_alpha0", "add_source_alpha0"],
        )
        self.assertEqual(add_receipt["status"], "applied")
        self.assertEqual(
            add_receipt["bridge_layer_ids"], [5, 6]
        )
        self.assertEqual(
            add_receipt["bridge_layer_kinds"], ["FLATTEN", "DENSE"]
        )
        self.assertEqual(
            add_receipt["bridge_topology"],
            "ADD->FLATTEN->DENSE->final_RELU",
        )
        self.assertTrue(add_receipt["proof_authority"])
        self.assertTrue(add_receipt["materialized_relation_retained"])
        self.assertFalse(add_receipt["prunes_materialized_frame"])
        self.assertGreaterEqual(add_receipt["materialized_new_ub"], 4)
        self.assertTrue(
            any(
                item["tag"].startswith("add_materialize:4:")
                for item in source.metadata["constraint_tags_ub"]
            )
        )

        verdict, witness = hz_objbound_decide(
            source.hz,
            np.eye(source.hz.n_out, dtype=np.float64),
            np.zeros(source.hz.n_out, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            safe_row_groups=source.property_upper_row_groups,
            expected_safe_group_count=2,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        winners = source.hz._solver_objbound_stats[
            "safe_row_group_winners"
        ]
        self.assertEqual(
            [(item["row"], item["stage"]) for item in winners],
            [(2, "cube"), (3, "cube")],
        )

    def test_final_add_source_requires_explicit_final_add_topology(
        self,
    ) -> None:
        toy = _scalar_relu_toy(
            pre_weight=Fraction(1),
            pre_bias=Fraction(0),
            out_weight=Fraction(1),
        )
        with self.assertRaisesRegex(
            OperatorHZBuildError, "require ADD"
        ):
            build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=0,
                materialize_add=True,
                property_upper_C=np.asarray(
                    [[1.0]], dtype=np.float64
                ),
                property_upper_thresholds=np.asarray(
                    [0.0], dtype=np.float64
                ),
                property_tail_add_source_planes=True,
            )

    def test_add_source_relation_receipt_handles_constant_rows(self) -> None:
        input_layer, spec = _input_layers(1, 2)
        left = _dense_matrix(2, [[1], [0]], [0, 0])
        right = _dense_matrix(3, [[0], [0]], [0, 0])
        toy = _assemble_width_toy(
            [
                input_layer,
                spec,
                left,
                right,
                _wide_layer(4, "ADD", 2),
                _wide_layer(5, "RELU", 2),
                _dense_matrix(6, [[1, 0], [0, 1]], [0, 0]),
                _wide_layer(7, "ASSERT", 2),
            ],
            {
                0: [],
                1: [0],
                2: [1],
                3: [1],
                4: [2, 3],
                5: [4],
                6: [5],
                7: [6],
            },
            input_lb=1,
            input_ub=2,
        )
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=np.asarray(
                [[0.0, 1.0]], dtype=np.float64
            ),
            property_upper_thresholds=np.asarray(
                [0.1], dtype=np.float64
            ),
            property_tail_add_source_planes=True,
        )
        receipt = build.metadata["property_tail_upper"][
            "add_source_planes"
        ]
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["materialized_new_ub"], 2)
        self.assertEqual(
            receipt["materialized_relation_block_rows"], [1, 1]
        )
        self.assertEqual(
            len(receipt["materialized_relation_blocks_sha256"]), 64
        )
        self.assertEqual(build.property_upper_row_groups, ((0, 1),))

    def test_verifier_tail_certifies_safe_and_demotes_unsafe_plane(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            config = BackendConfig(
                solver="hybridz",
                device="cpu",
                dtype="float64",
                hybridz=HybridZConfig(
                    timeout=2.0,
                    engine="operator_hz_objbound",
                    property_tail_upper=True,
                    lp_prefilter_fraction=1.0,
                    lp_prefilter_max_seconds=1.0,
                ),
            )
            safe = verify_once(
                _verified_relu_net(2.0),
                backend_cfg=config,
            )[0]
            self.assertEqual(safe.status, VerifyStatus.CERTIFIED)
            self.assertTrue(
                safe.metadata["operator_hz"]["property_upper_output"]
            )
            self.assertEqual(safe.metadata["hz_verdict"], "SAFE")

            unsafe_plane = verify_once(
                _verified_relu_net(0.0),
                backend_cfg=config,
            )[0]
            self.assertEqual(unsafe_plane.status, VerifyStatus.UNKNOWN)
            self.assertEqual(unsafe_plane.metadata["hz_verdict"], "UNKNOWN")
            self.assertEqual(
                unsafe_plane.metadata["reason"],
                "hybridz_verdict_unknown",
            )
            self.assertFalse(unsafe_plane.metadata["hz_has_witness"])
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verifier_plumbs_alpha_and_grouped_baseline_fallback(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                _verified_cancellation_net(0.75),
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=3.0,
                        engine="operator_hz_objbound",
                        property_tail_upper=True,
                        property_tail_alpha_steps=16,
                        property_tail_alpha_time_limit=1.0,
                        property_tail_alpha_device="cpu",
                        property_tail_mixture_grid_bits=4,
                    ),
                ),
            )[0]
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            tail = result.metadata["operator_hz"][
                "property_tail_upper"
            ]
            self.assertEqual(tail["baseline_plane_count"], 1)
            self.assertEqual(tail["alternative_plane_count"], 1)
            self.assertEqual(
                result.metadata["property_upper_row_groups"],
                [[0, 1]],
            )
            self.assertEqual(
                result.metadata["safe_row_group_count"], 1
            )
            self.assertEqual(
                result.metadata["cfg_property_tail_mixture_grid_bits"], 4
            )
            mixture = result.metadata["safe_row_dyadic_mixture"]
            self.assertTrue(mixture["enabled"])
            self.assertEqual(mixture["grid_bits"], 4)
            self.assertEqual(result.metadata["hz_verdict"], "SAFE")
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verifier_plumbs_pairhull_and_preserves_group_fallbacks(
        self,
    ) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                _verified_cancellation_net(0.25),
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=3.0,
                        engine="operator_hz_objbound",
                        property_tail_upper=True,
                        property_tail_alpha_steps=16,
                        property_tail_alpha_time_limit=1.0,
                        property_tail_alpha_device="cpu",
                        property_tail_pairhull_budget=1,
                        property_tail_pairhull_time_limit=1.5,
                        lp_prefilter_fraction=1.0,
                        lp_prefilter_max_seconds=1.0,
                    ),
                ),
            )[0]
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            self.assertEqual(result.metadata["hz_verdict"], "SAFE")
            self.assertEqual(
                result.metadata["cfg_property_tail_pairhull_budget"], 1
            )
            self.assertEqual(
                result.metadata[
                    "cfg_property_tail_pairhull_time_limit"
                ],
                1.5,
            )
            self.assertEqual(
                result.metadata["property_upper_row_groups"],
                [[0, 1, 2]],
            )
            tail = result.metadata["operator_hz"][
                "property_tail_upper"
            ]
            self.assertEqual(
                tail["alternative_plane_kinds"],
                [
                    "negative_alpha_materialized",
                    "pairhull_joint_materialized",
                ],
            )
            pairhull = tail["pairhull_candidates"]
            self.assertEqual(pairhull["status"], "applied")
            self.assertTrue(pairhull["proof_authority"])
            self.assertEqual(pairhull["selected_rival_ids"], [0])
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_live_add_fusion_composes_with_tail_alpha_and_mixture(self) -> None:
        """Exercise the real ADD->FLATTEN->DENSE->RELU tail as one proof."""

        threshold = Fraction(3, 4)
        for exact_x in (
            Fraction(-1),
            Fraction(-3, 4),
            Fraction(0),
            Fraction(1, 4),
            Fraction(1),
        ):
            left = exact_x / 2
            right = exact_x / 2
            added = left + right
            preactivation = (added, added, Fraction(-1))
            activated = tuple(
                max(Fraction(0), value) for value in preactivation
            )
            output = activated[0] - activated[1]
            self.assertEqual(added, exact_x)
            self.assertEqual(output, Fraction(0))
            self.assertEqual(output - threshold, -threshold)

        results = {}
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            for materialize_add in (True, False):
                result = verify_once(
                    _verified_live_tail_combo_net(float(threshold)),
                    backend_cfg=BackendConfig(
                        solver="hybridz",
                        device="cpu",
                        dtype="float64",
                        hybridz=HybridZConfig(
                            timeout=3.0,
                            engine="operator_hz_objbound",
                            operator_exact_budget=0,
                            operator_materialize_add=materialize_add,
                            property_tail_upper=True,
                            property_tail_alpha_steps=16,
                            property_tail_alpha_time_limit=1.0,
                            property_tail_alpha_device="cpu",
                            property_tail_mixture_grid_bits=4,
                            lp_prefilter_fraction=0.0,
                            lp_prefilter_max_seconds=0.0,
                        ),
                    ),
                )[0]
                results[materialize_add] = result

                self.assertEqual(result.status, VerifyStatus.CERTIFIED)
                self.assertEqual(result.metadata["hz_verdict"], "SAFE")
                self.assertEqual(
                    result.metadata["cfg_operator_materialize_add"],
                    materialize_add,
                )
                self.assertEqual(
                    result.metadata["cfg_property_tail_mixture_grid_bits"],
                    4,
                )
                self.assertEqual(
                    result.metadata["property_upper_row_groups"],
                    [[0, 1]],
                )
                self.assertEqual(result.metadata["safe_row_group_count"], 1)
                winners = result.metadata["safe_row_group_winners"]
                self.assertEqual(len(winners), 1)
                self.assertEqual(
                    (
                        winners[0]["group"],
                        winners[0]["row"],
                        winners[0]["stage"],
                    ),
                    (0, 1, "cube"),
                )
                self.assertLess(winners[0]["upper"], -0.24)
                self.assertGreater(result.metadata["cube_max_upper"], 0.24)
                self.assertLess(result.metadata["cube_min_upper"], -0.24)

                operator = result.metadata["operator_hz"]
                tail = operator["property_tail_upper"]
                self.assertEqual(
                    (
                        tail["phase_active"],
                        tail["phase_unstable"],
                        tail["phase_inactive"],
                    ),
                    (0, 2, 1),
                )
                self.assertEqual(tail["baseline_plane_count"], 1)
                self.assertEqual(tail["alternative_plane_count"], 1)
                self.assertEqual(
                    tail["alternative_plane_kinds"],
                    ["negative_alpha_materialized"],
                )
                self.assertEqual(tail["property_row_groups"], [[0, 1]])
                alpha = tail["negative_alpha_candidates"]
                self.assertTrue(
                    alpha["baseline_fallback_retained_per_rival"]
                )
                self.assertEqual(alpha["selected_rivals"], 1)
                self.assertEqual(alpha["alpha_max"], 0.5)
                self.assertGreater(
                    alpha["cube_upper_improvement_max"], 0.49
                )
                self.assertLess(
                    alpha["candidate_minus_baseline_cube_max"], -0.49
                )
                self.assertTrue(
                    alpha["exact_candidate_audit"]["proof_authority"]
                )
                self.assertEqual(
                    alpha["exact_candidate_audit"]["nonzero_negative_alpha"],
                    1,
                )

                mixture = result.metadata["safe_row_dyadic_mixture"]
                self.assertTrue(mixture["enabled"])
                self.assertEqual(mixture["grid_bits"], 4)
                self.assertEqual(
                    mixture["status"], "no_strict_proxy_improvement"
                )
                self.assertTrue(mixture["exact_search_complete"])
                self.assertTrue(mixture["dyadic_convexity_validated"])
                self.assertEqual(
                    mixture["guarded_cube_authority"],
                    "outward_hz_cube_checker",
                )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

        materialized = results[True]
        fused = results[False]
        materialized_operator = materialized.metadata["operator_hz"]
        fused_operator = fused.metadata["operator_hz"]
        self.assertFalse(materialized_operator["live_affine_relu_enabled"])
        self.assertEqual(
            materialized_operator["live_affine_relu_attempts"], []
        )
        self.assertEqual(
            len(materialized_operator["materialization_events"]), 1
        )
        self.assertTrue(fused_operator["live_affine_relu_enabled"])
        self.assertEqual(fused_operator["live_affine_relu_applied"], 1)
        self.assertEqual(
            fused_operator["live_affine_relu_box_inactive_rows"], 1
        )
        self.assertEqual(fused_operator["materialization_events"], [])
        attempt = fused_operator["live_affine_relu_attempts"][0]
        self.assertEqual(attempt["status"], "applied")
        self.assertEqual(attempt["add_origin_layer_id"], 4)
        self.assertEqual(attempt["layer_id"], 6)
        self.assertEqual(attempt["exact_inactive_rows"], 1)
        self.assertEqual(attempt["box_inactive_rows"], 1)
        self.assertTrue(attempt["proof_authority"])
        self.assertEqual(
            attempt["source_generator_sha256_before"],
            attempt["source_generator_sha256_after"],
        )
        self.assertEqual(
            len(attempt["source_generator_sha256_before"]), 64
        )

        self.assertLess(
            fused.metadata["operator_n_cont"],
            materialized.metadata["operator_n_cont"],
        )
        self.assertLess(
            fused.metadata["operator_n_ub"],
            materialized.metadata["operator_n_ub"],
        )
        self.assertLess(
            fused.metadata["operator_constraint_nnz"],
            materialized.metadata["operator_constraint_nnz"],
        )
        self.assertLess(
            fused_operator["property_tail_upper"]["prefix_n_cont"],
            materialized_operator["property_tail_upper"]["prefix_n_cont"],
        )

    def test_verifier_plumbs_final_add_source_groups(self) -> None:
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                _verified_add_correlation_net(0.1),
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=3.0,
                        engine="operator_hz_objbound",
                        property_tail_upper=True,
                        property_tail_add_source_planes=True,
                        lp_prefilter_fraction=0.0,
                        lp_prefilter_max_seconds=0.0,
                    ),
                ),
            )[0]
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            self.assertEqual(result.metadata["hz_verdict"], "SAFE")
            self.assertEqual(
                result.metadata["property_upper_row_groups"],
                [[0, 1]],
            )
            tail = result.metadata["operator_hz"][
                "property_tail_upper"
            ]
            self.assertEqual(
                tail["alternative_plane_kinds"],
                ["add_source_alpha0"],
            )
            self.assertEqual(
                tail["add_source_planes"]["status"], "applied"
            )
            self.assertTrue(
                tail["add_source_planes"]["materialized_relation_retained"]
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verifier_rejects_tampered_property_proof_object_receipt(
        self,
    ) -> None:
        from act.back_end.hybridz_tf import operator_hz as operator_module

        original_build = operator_module.build_operator_hz

        def tampered_build(*args, **kwargs):
            result = original_build(*args, **kwargs)
            result.metadata["property_tail_upper"][
                "upper_expression_center_sha256"
            ] = "0" * 64
            return result

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with mock.patch(
                "act.back_end.hybridz_tf.operator_hz.build_operator_hz",
                side_effect=tampered_build,
            ):
                result = verify_once(
                    _verified_relu_net(2.0),
                    backend_cfg=BackendConfig(
                        solver="hybridz",
                        device="cpu",
                        dtype="float64",
                        hybridz=HybridZConfig(
                            timeout=2.0,
                            engine="operator_hz_objbound",
                            property_tail_upper=True,
                        ),
                    ),
                )[0]
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            self.assertEqual(
                result.metadata["reason"],
                "hybridz_operator_build_failed",
            )
            self.assertIn(
                "grouped upper-plane receipt",
                result.metadata["operator_error"],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")


if __name__ == "__main__":
    unittest.main(verbosity=2)
