#!/usr/bin/env python3
"""Strict toy audits for ``operator_hz(materialize_add=False)``.

The tests in this module deliberately do not use interval-analysis facts as an
oracle.  Every scalar toy is evaluated independently by enumerating every ReLU
phase.  Within one phase every node is affine in the scalar input, so phase
feasibility and the output extrema are computed exactly with
:class:`fractions.Fraction`.

For ordinary-scale toys, a separate SciPy/HiGHS LP measures the projection of
the produced continuous HZ.  Soundness is checked without trusting that LP:
the exact graph range must be enclosed by the exact dyadic output cube of both
the materialized and fused builds.  The LP is used only to compare tightness.
The cancellation case around ``1e16`` therefore remains meaningful even on a
solver which scales or drops tiny coefficients.

Run from the repository root with::

    python -m act.back_end.hybridz_tf.test_operator_add_fusion
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import itertools
import random
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import operator_hz as operator_hz_module
from act.back_end.hybridz_tf.operator_hz import (
    _AffineExpr,
    _OperatorHZBuilder,
    _nonnegative_sum_upper,
    _row_l1_upper,
    OperatorHZBuild,
    OperatorHZBuildTimeout,
    build_operator_hz,
)


_DTYPE = torch.float64


@dataclass(frozen=True)
class _ScalarToy:
    net: Any
    facts: Mapping[int, Fact]
    input_lb: Fraction
    input_ub: Fraction


@dataclass(frozen=True)
class _ExactRange:
    lower: Fraction
    upper: Fraction
    phase_assignments: int
    feasible_phases: int


def _fraction(value: Any) -> Fraction:
    """Return the exact rational value of a stored binary64 scalar."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise AssertionError("the scalar oracle received a non-scalar tensor")
        value = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise AssertionError("the scalar oracle received a non-scalar array")
        value = value.reshape(-1)[0]
    return Fraction.from_float(float(value))


def _layer(
    layer_id: int,
    kind: str,
    params: Mapping[str, Any] | None = None,
) -> Any:
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[int(layer_id)],
    )


def _dense(
    layer_id: int,
    weight: Fraction | int | float,
    bias: Fraction | int | float,
) -> Any:
    return _layer(
        layer_id,
        "DENSE",
        {
            "weight": torch.tensor([[float(weight)]], dtype=_DTYPE),
            "bias": torch.tensor([float(bias)], dtype=_DTYPE),
            "in_features": 1,
            "out_features": 1,
        },
    )


def _conv1x1(
    layer_id: int,
    weight: Fraction | int | float,
    bias: Fraction | int | float,
    *,
    stride: int = 1,
) -> Any:
    return _layer(
        layer_id,
        "CONV2D",
        {
            "weight": torch.tensor(
                [[[[float(weight)]]]], dtype=_DTYPE
            ),
            "bias": torch.tensor([float(bias)], dtype=_DTYPE),
            "stride": (int(stride), int(stride)),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "input_shape": (1, 1, 1, 1),
            "output_shape": (1, 1, 1, 1),
            "data_format": "NCHW",
            "padding_mode": "zeros",
            "auto_pad": "NOTSET",
            "in_channels": 1,
            "out_channels": 1,
        },
    )


def _dense_matrix(
    layer_id: int,
    weight: Sequence[Sequence[Fraction | int | float]],
    bias: Sequence[Fraction | int | float],
) -> Any:
    weight_array = np.asarray(
        [[float(value) for value in row] for row in weight],
        dtype=np.float64,
    )
    bias_array = np.asarray([float(value) for value in bias], dtype=np.float64)
    if weight_array.ndim != 2 or weight_array.shape[0] != bias_array.size:
        raise AssertionError("invalid dense-matrix toy parameters")
    layer = _layer(
        layer_id,
        "DENSE",
        {
            "weight": torch.tensor(weight_array, dtype=_DTYPE),
            "bias": torch.tensor(bias_array, dtype=_DTYPE),
            "in_features": int(weight_array.shape[1]),
            "out_features": int(weight_array.shape[0]),
        },
    )
    layer.out_vars = [
        (int(layer_id), int(row)) for row in range(weight_array.shape[0])
    ]
    return layer


def _wide_layer(layer_id: int, kind: str, width: int) -> Any:
    layer = _layer(layer_id, kind)
    layer.out_vars = [
        (int(layer_id), int(row)) for row in range(int(width))
    ]
    return layer


def _assemble_scalar_toy(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    *,
    input_lb: Fraction | int | float,
    input_ub: Fraction | int | float,
) -> _ScalarToy:
    """Assemble a one-input/one-output ACT DAG with audit-only wide facts."""

    lower = _fraction(input_lb)
    upper = _fraction(input_ub)
    if lower > upper:
        raise AssertionError("invalid toy input interval")
    by_id = {int(layer.id): layer for layer in layers}
    if len(by_id) != len(layers):
        raise AssertionError("toy layer ids are not unique")
    pred_map = {
        int(layer.id): [int(parent) for parent in preds[int(layer.id)]]
        for layer in layers
    }
    succs: Dict[int, List[int]] = {int(layer.id): [] for layer in layers}
    for child, parents in pred_map.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=list(layers),
        preds=pred_map,
        succs=succs,
        by_id=by_id,
    )

    input_tensor_lb = torch.tensor([[float(lower)]], dtype=_DTYPE)
    input_tensor_ub = torch.tensor([[float(upper)]], dtype=_DTYPE)
    facts: Dict[int, Fact] = {}
    for layer in layers:
        if str(layer.kind).upper() in {"INPUT", "INPUT_SPEC"}:
            lb = input_tensor_lb.clone()
            ub = input_tensor_ub.clone()
        else:
            # Internal facts are intentionally non-informative.  The strict
            # builder must derive every local ReLU bound from its own frame.
            lb = torch.tensor([[-1.0e30]], dtype=_DTYPE)
            ub = torch.tensor([[1.0e30]], dtype=_DTYPE)
        facts[int(layer.id)] = Fact(Bounds(lb, ub), ConSet())
    return _ScalarToy(net, facts, lower, upper)


def _assemble_width_toy(
    layers: Sequence[Any],
    preds: Mapping[int, Sequence[int]],
    *,
    input_lb: Fraction | int | float,
    input_ub: Fraction | int | float,
) -> _ScalarToy:
    """Assemble a scalar-input DAG whose internal layers may be vectors."""

    lower = _fraction(input_lb)
    upper = _fraction(input_ub)
    by_id = {int(layer.id): layer for layer in layers}
    pred_map = {
        int(layer.id): [int(parent) for parent in preds[int(layer.id)]]
        for layer in layers
    }
    succs: Dict[int, List[int]] = {int(layer.id): [] for layer in layers}
    for child, parents in pred_map.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=list(layers),
        preds=pred_map,
        succs=succs,
        by_id=by_id,
    )
    facts: Dict[int, Fact] = {}
    for layer in layers:
        width = len(layer.out_vars)
        if str(layer.kind).upper() in {"INPUT", "INPUT_SPEC"}:
            lb = torch.full((1, width), float(lower), dtype=_DTYPE)
            ub = torch.full((1, width), float(upper), dtype=_DTYPE)
        else:
            lb = torch.full((1, width), -1.0e30, dtype=_DTYPE)
            ub = torch.full((1, width), 1.0e30, dtype=_DTYPE)
        facts[int(layer.id)] = Fact(Bounds(lb, ub), ConSet())
    return _ScalarToy(net, facts, lower, upper)


def _input_layers(
    lower: Fraction | int | float,
    upper: Fraction | int | float,
) -> Tuple[Any, Any]:
    lb = torch.tensor([[float(lower)]], dtype=_DTYPE)
    ub = torch.tensor([[float(upper)]], dtype=_DTYPE)
    return (
        _layer(0, "INPUT", {"shape": (1, 1)}),
        _layer(1, "INPUT_SPEC", {"kind": "BOX", "lb": lb, "ub": ub}),
    )


def _intersect_nonnegative(
    coefficient: Fraction,
    constant: Fraction,
    lower: Fraction,
    upper: Fraction,
) -> Tuple[Fraction, Fraction] | None:
    """Intersect ``[lower, upper]`` with ``coefficient*x+constant >= 0``."""

    if coefficient == 0:
        return (lower, upper) if constant >= 0 else None
    root = -constant / coefficient
    if coefficient > 0:
        lower = max(lower, root)
    else:
        upper = min(upper, root)
    if lower > upper:
        return None
    return lower, upper


def _exact_graph_range(toy: _ScalarToy) -> _ExactRange:
    """Enumerate all ReLU phases and optimize each affine piece exactly."""

    relu_ids = [
        int(layer.id)
        for layer in toy.net.layers
        if str(layer.kind).upper() == "RELU"
    ]
    lowers: List[Fraction] = []
    uppers: List[Fraction] = []
    feasible_phases = 0

    for assignment in itertools.product((False, True), repeat=len(relu_ids)):
        active = dict(zip(relu_ids, assignment))
        interval: Tuple[Fraction, Fraction] | None = (
            toy.input_lb,
            toy.input_ub,
        )
        # Each value is represented as coefficient*x + constant.
        exprs: Dict[int, Tuple[Fraction, Fraction]] = {}
        for layer in toy.net.layers:
            if interval is None:
                break
            lid = int(layer.id)
            kind = str(layer.kind).upper()
            parents = [int(parent) for parent in toy.net.preds[lid]]
            if kind == "INPUT":
                exprs[lid] = (Fraction(1), Fraction(0))
            elif kind in {"INPUT_SPEC", "FLATTEN", "ASSERT"}:
                if len(parents) != 1:
                    raise AssertionError(f"{kind} must have one parent")
                exprs[lid] = exprs[parents[0]]
            elif kind in {"DENSE", "CONV2D"}:
                if len(parents) != 1:
                    raise AssertionError(f"{kind} must have one parent")
                source_a, source_b = exprs[parents[0]]
                weight = _fraction(layer.params["weight"])
                bias = _fraction(layer.params["bias"])
                exprs[lid] = (
                    weight * source_a,
                    weight * source_b + bias,
                )
            elif kind == "ADD":
                if len(parents) != 2:
                    raise AssertionError("ADD must have two parents")
                left_a, left_b = exprs[parents[0]]
                right_a, right_b = exprs[parents[1]]
                exprs[lid] = (left_a + right_a, left_b + right_b)
            elif kind == "RELU":
                if len(parents) != 1:
                    raise AssertionError("RELU must have one parent")
                source_a, source_b = exprs[parents[0]]
                if active[lid]:
                    interval = _intersect_nonnegative(
                        source_a, source_b, *interval
                    )
                    exprs[lid] = (source_a, source_b)
                else:
                    interval = _intersect_nonnegative(
                        -source_a, -source_b, *interval
                    )
                    exprs[lid] = (Fraction(0), Fraction(0))
            else:
                raise AssertionError(f"unsupported exact-oracle layer {kind}")

        if interval is None:
            continue
        feasible_phases += 1
        assert_l, assert_u = interval
        assert_layers = [
            layer
            for layer in toy.net.layers
            if str(layer.kind).upper() == "ASSERT"
        ]
        if len(assert_layers) != 1:
            raise AssertionError("the scalar oracle requires one ASSERT")
        output_a, output_b = exprs[int(assert_layers[0].id)]
        endpoint_values = (
            output_a * assert_l + output_b,
            output_a * assert_u + output_b,
        )
        lowers.append(min(endpoint_values))
        uppers.append(max(endpoint_values))

    if not lowers:
        raise AssertionError("all exact ReLU phase assignments were infeasible")
    return _ExactRange(
        min(lowers),
        max(uppers),
        1 << len(relu_ids),
        feasible_phases,
    )


def _output_cube_fraction(build: OperatorHZBuild) -> Tuple[Fraction, Fraction]:
    """Return the exact dyadic cube projection of the scalar HZ output."""

    hz = build.hz
    if hz.n_out != 1:
        raise AssertionError("the scalar audit expected exactly one output")
    center = Fraction.from_float(float(hz.c[0]))
    radius = Fraction(0)
    for matrix in (hz.Gc, hz.Gb):
        for value in matrix.getrow(0).data:
            radius += abs(Fraction.from_float(float(value)))
    return center - radius, center + radius


def _lp_output_range(build: OperatorHZBuild) -> Tuple[float, float]:
    """Measure the continuous HZ projection; never use its status as proof."""

    hz = build.hz
    if hz.n_bin:
        raise AssertionError("the tightness LP requires exact_budget=0")
    objective = hz.Gc.getrow(0).toarray().reshape(-1)
    if hz.n_cont == 0:
        return float(hz.c[0]), float(hz.c[0])
    bounds = [(-1.0, 1.0)] * hz.n_cont
    kwargs = {
        "A_ub": hz.Auc if hz.n_ub else None,
        "b_ub": hz.ub if hz.n_ub else None,
        "A_eq": hz.Ac if hz.n_eq else None,
        "b_eq": hz.b if hz.n_eq else None,
        "bounds": bounds,
        "method": "highs",
    }
    minimum = linprog(objective, **kwargs)
    maximum = linprog(-objective, **kwargs)
    if not minimum.success or not maximum.success:
        raise AssertionError(
            "independent tightness LP failed: "
            f"min={minimum.status}/{minimum.message}; "
            f"max={maximum.status}/{maximum.message}"
        )
    return (
        float(hz.c[0] + minimum.fun),
        float(hz.c[0] - maximum.fun),
    )


def _size_receipt(build: OperatorHZBuild) -> Tuple[int, int]:
    hz = build.hz
    return int(hz.n_eq + hz.n_ub), int(hz.constraint_nnz)


def _build_pair(toy: _ScalarToy) -> Tuple[OperatorHZBuild, OperatorHZBuild]:
    materialized = build_operator_hz(
        toy.net,
        toy.facts,
        toy.facts,
        exact_budget=0,
        materialize_add=True,
    )
    fused = build_operator_hz(
        toy.net,
        toy.facts,
        toy.facts,
        exact_budget=0,
        materialize_add=False,
    )
    return materialized, fused


class OperatorAddFusionAuditTests(unittest.TestCase):
    """Soundness, identity, size and tightness gates for fused ADD frames."""

    def assertExactCubeEncloses(
        self,
        build: OperatorHZBuild,
        exact: _ExactRange,
        *,
        label: str,
    ) -> None:
        cube_lower, cube_upper = _output_cube_fraction(build)
        self.assertLessEqual(
            cube_lower,
            exact.lower,
            msg=f"{label}: exact lower endpoint escaped the HZ output cube",
        )
        self.assertGreaterEqual(
            cube_upper,
            exact.upper,
            msg=f"{label}: exact upper endpoint escaped the HZ output cube",
        )

    def assertLPEncloses(
        self,
        build: OperatorHZBuild,
        exact: _ExactRange,
        *,
        label: str,
        tolerance: float = 2.0e-8,
    ) -> Tuple[float, float]:
        lower, upper = _lp_output_range(build)
        scale = max(
            1.0,
            abs(float(exact.lower)),
            abs(float(exact.upper)),
        )
        slack = tolerance * scale
        self.assertLessEqual(
            lower,
            float(exact.lower) + slack,
            msg=f"{label}: LP projection lost the exact lower endpoint",
        )
        self.assertGreaterEqual(
            upper,
            float(exact.upper) - slack,
            msg=f"{label}: LP projection lost the exact upper endpoint",
        )
        return lower, upper

    def test_nonsemantic_fact_width_scan_is_absent_from_hot_path(self) -> None:
        toy = self._residual_add_relu_toy()
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=-1,
            materialize_add=False,
            export_verified_preactivation_frame=False,
        )
        for layer in build.metadata["layers"]:
            self.assertEqual(
                layer["fact_audit"],
                "omitted_nonsemantic_hot_path_v1",
            )
            self.assertNotIn("fact_width_mean", layer)
            self.assertNotIn("local_cube_width_mean", layer)

    def test_common_exact_path_counts_without_property_record_payloads(self) -> None:
        """Disabled property consumers must not allocate per-ReLU dicts."""

        toy = self._residual_add_relu_toy()
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=-1,
            materialize_add=False,
        )
        receipt = build.metadata["property_micro_rlt"]
        self.assertEqual(receipt["exact_record_count"], 1)
        self.assertEqual(receipt["exact_relu_records"], [])
        self.assertEqual(receipt["status"], "no_op_disabled")

    def test_direct_csr_row_l1_matches_scipy_reference_bitwise(self) -> None:
        rng = np.random.default_rng(0xA11CE20260811)
        for case in range(256):
            rows = int(rng.integers(1, 48))
            cols = int(rng.integers(1, 96))
            matrix = sp.random(
                rows,
                cols,
                density=float(rng.uniform(0.01, 0.65)),
                format="csr",
                random_state=rng,
                data_rvs=lambda count: (
                    rng.choice(np.asarray([-1.0, 1.0]), size=count)
                    * np.exp2(
                        rng.integers(-1000, 100, size=count)
                    ).astype(np.float64)
                ),
            )
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            reference_raw = np.asarray(
                abs(matrix).sum(axis=1), dtype=np.float64
            ).reshape(-1)
            nnz = np.diff(matrix.indptr).astype(np.float64)
            reference = operator_hz_module._inflate_nonnegative(
                reference_raw,
                2.0 * nnz + 2.0,
                active=nnz > 0.0,
                name=f"row_l1_reference[{case}]",
            )
            actual = _row_l1_upper(
                matrix, name=f"row_l1_candidate[{case}]"
            )
            np.testing.assert_array_equal(
                actual.view(np.uint64), reference.view(np.uint64)
            )

    def test_scalar_roundoff_count_matches_broadcast_reference_bitwise(
        self,
    ) -> None:
        rng = np.random.default_rng(0x6A66A20260811)
        for case in range(128):
            width = int(rng.integers(0, 257))
            rounded = np.abs(rng.standard_normal(width)).astype(np.float64)
            if width:
                rounded[rng.random(width) < 0.2] = 0.0
                rounded[rng.random(width) < 0.03] = np.nextafter(0.0, 1.0)
            active = (
                None
                if case % 3 == 0
                else rng.random(width) > 0.35
            )
            count = int((case % 17) + 1)
            scalar = operator_hz_module._inflate_nonnegative(
                rounded,
                count,
                active=active,
                name=f"scalar_count[{case}]",
            )
            broadcast = operator_hz_module._inflate_nonnegative(
                rounded,
                np.full(width, float(count), dtype=np.float64),
                active=active,
                name=f"broadcast_count[{case}]",
            )
            np.testing.assert_array_equal(
                scalar.view(np.uint64), broadcast.view(np.uint64)
            )

    def test_absolute_csr_view_reuses_topology_and_matches_scipy_bits(
        self,
    ) -> None:
        source = sp.csr_matrix(
            (
                np.asarray(
                    [-0.0, -4.0, 0.5, -0.25, 8.0], dtype=np.float64
                ),
                np.asarray([0, 3, 1, 2, 4], dtype=np.int32),
                np.asarray([0, 2, 3, 5], dtype=np.int32),
            ),
            shape=(3, 5),
        )
        reference = abs(source).tocsr()
        actual = operator_hz_module._absolute_csr_topology_view(
            source, name="absolute_topology_audit"
        )
        np.testing.assert_array_equal(actual.indptr, reference.indptr)
        np.testing.assert_array_equal(actual.indices, reference.indices)
        np.testing.assert_array_equal(
            actual.data.view(np.uint64), reference.data.view(np.uint64)
        )
        self.assertTrue(np.shares_memory(actual.indptr, source.indptr))
        self.assertTrue(np.shares_memory(actual.indices, source.indices))
        vector = np.asarray([1.0, 0.5, 2.0, 4.0, 0.25])
        expected_product = np.asarray(reference @ vector).reshape(-1)
        actual_product = np.asarray(actual @ vector).reshape(-1)
        np.testing.assert_array_equal(
            actual_product.view(np.uint64), expected_product.view(np.uint64)
        )

        nonnegative = sp.csr_matrix(
            np.asarray([[0.25, 0.0], [1.0, 2.0]], dtype=np.float64)
        )
        self.assertIs(
            operator_hz_module._absolute_csr_topology_view(
                nonnegative, name="nonnegative_topology_audit"
            ),
            nonnegative,
        )

    @staticmethod
    def _residual_add_relu_toy() -> _ScalarToy:
        """``ReLU(x + (x/2 - 1/4))`` over ``[-1,1]``."""

        lower, upper = Fraction(-1), Fraction(1)
        input_layer, spec_layer = _input_layers(lower, upper)
        layers = [
            input_layer,
            spec_layer,
            _dense(2, Fraction(1, 2), Fraction(-1, 4)),
            _layer(3, "ADD"),
            _layer(4, "RELU"),
            _layer(5, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {0: [], 1: [0], 2: [1], 3: [1, 2], 4: [3], 5: [4]},
            input_lb=lower,
            input_ub=upper,
        )

    @staticmethod
    def _shared_fanout_cancellation_toy() -> _ScalarToy:
        """A shared ReLU factor is consumed as ``r + (-r)``."""

        lower, upper = Fraction(-3, 4), Fraction(3, 4)
        input_layer, spec_layer = _input_layers(lower, upper)
        layers = [
            input_layer,
            spec_layer,
            _layer(2, "RELU"),
            _dense(3, Fraction(1), Fraction(0)),
            _dense(4, Fraction(-1), Fraction(0)),
            _layer(5, "ADD"),
            _layer(6, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {0: [], 1: [0], 2: [1], 3: [2], 4: [2], 5: [3, 4], 6: [5]},
            input_lb=lower,
            input_ub=upper,
        )

    @staticmethod
    def _catastrophic_add_toy() -> _ScalarToy:
        """Stored-float evaluation rounds ``1e16 + 1 - 1e16`` to zero."""

        input_layer, spec_layer = _input_layers(Fraction(0), Fraction(0))
        layers = [
            input_layer,
            spec_layer,
            _dense(2, 0, 1.0e16),
            _dense(3, 0, 1),
            _layer(4, "ADD"),
            _dense(5, 0, -1.0e16),
            _layer(6, "ADD"),
            _layer(7, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [1],
                4: [2, 3],
                5: [1],
                6: [4, 5],
                7: [6],
            },
            input_lb=Fraction(0),
            input_ub=Fraction(0),
        )

    @staticmethod
    def _add_affine_relu_toy(
        *,
        second_weight: Fraction | int | float = 1,
        second_bias: Fraction | int | float = 0,
    ) -> _ScalarToy:
        """``ReLU(w*(x + (x/2 - 1/4)) + b)`` over ``[-1,1]``."""

        lower, upper = Fraction(-1), Fraction(1)
        input_layer, spec_layer = _input_layers(lower, upper)
        layers = [
            input_layer,
            spec_layer,
            _dense(2, Fraction(1, 2), Fraction(-1, 4)),
            _layer(3, "ADD"),
            _dense(4, second_weight, second_bias),
            _layer(5, "RELU"),
            _layer(6, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [1, 2],
                4: [3],
                5: [4],
                6: [5],
            },
            input_lb=lower,
            input_ub=upper,
        )

    @staticmethod
    def _projection_skip_chain_toy() -> _ScalarToy:
        """Two residual joins with one 1x1 stride-2 projection skip."""

        lower, upper = Fraction(-1), Fraction(1)
        input_layer, spec_layer = _input_layers(lower, upper)
        input_layer.params["shape"] = (1, 1, 1, 1)
        layers = [
            input_layer,
            spec_layer,
            _conv1x1(2, 1, 0),
            _layer(3, "RELU"),
            _conv1x1(4, 1, Fraction(1, 8)),
            _layer(5, "ADD"),
            _conv1x1(6, 1, Fraction(-1, 8), stride=2),
            _layer(7, "RELU"),
            _conv1x1(8, Fraction(3, 4), 0),
            _conv1x1(9, Fraction(1, 2), 0, stride=2),
            _layer(10, "ADD"),
            _conv1x1(11, 1, Fraction(-1, 4)),
            _layer(12, "RELU"),
            _layer(13, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [2],
                4: [3],
                5: [4, 1],
                6: [5],
                7: [6],
                8: [7],
                9: [5],
                10: [8, 9],
                11: [10],
                12: [11],
                13: [12],
            },
            input_lb=lower,
            input_ub=upper,
        )

    @staticmethod
    def _catastrophic_affine_relu_toy() -> _ScalarToy:
        """Exact-real ``ReLU((1e16 + 1) - 1e16) == 1``."""

        input_layer, spec_layer = _input_layers(Fraction(0), Fraction(0))
        layers = [
            input_layer,
            spec_layer,
            _dense(2, 0, 1.0e16),
            _dense(3, 0, 1),
            _layer(4, "ADD"),
            _dense(5, 1, -1.0e16),
            _layer(6, "RELU"),
            _layer(7, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
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
            input_lb=Fraction(0),
            input_ub=Fraction(0),
        )

    @staticmethod
    def _mixed_phase_affine_toy() -> _ScalarToy:
        """One affine chunk containing active, unstable and inactive rows."""

        input_layer, spec_layer = _input_layers(Fraction(-1), Fraction(1))
        layers = [
            input_layer,
            spec_layer,
            _dense(2, 1, 0),
            _layer(3, "ADD"),
            _dense_matrix(
                4,
                [[1], [1], [1]],
                [Fraction(5, 2), 0, -3],
            ),
            _wide_layer(5, "RELU", 3),
            _dense_matrix(6, [[1, 1, 1]], [0]),
            _layer(7, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_width_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [1, 2],
                4: [3],
                5: [4],
                6: [5],
                7: [6],
            },
            input_lb=Fraction(-1),
            input_ub=Fraction(1),
        )

    @staticmethod
    def _affine_fanout_toy() -> _ScalarToy:
        """The fused affine has two consumers, forcing a pristine fallback."""

        input_layer, spec_layer = _input_layers(Fraction(-1), Fraction(1))
        layers = [
            input_layer,
            spec_layer,
            _dense(2, Fraction(1, 2), Fraction(1, 4)),
            _layer(3, "ADD"),
            _dense(4, 1, 0),
            _layer(5, "RELU"),
            _dense(6, -1, Fraction(1, 3)),
            _layer(7, "ADD"),
            _layer(8, "RELU"),
            _layer(9, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [1, 2],
                4: [3],
                5: [4],
                6: [4],
                7: [5, 6],
                8: [7],
                9: [8],
            },
            input_lb=Fraction(-1),
            input_ub=Fraction(1),
        )

    def test_residual_add_relu_is_sound_smaller_and_no_looser(self) -> None:
        toy = self._residual_add_relu_toy()
        exact = _exact_graph_range(toy)
        self.assertEqual(exact.phase_assignments, 2)
        self.assertEqual(exact.lower, Fraction(0))
        self.assertEqual(exact.upper, Fraction(5, 4))

        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        materialized_range = self.assertLPEncloses(
            materialized, exact, label="materialized"
        )
        fused_range = self.assertLPEncloses(fused, exact, label="fused")

        materialized_rows, materialized_nnz = _size_receipt(materialized)
        fused_rows, fused_nnz = _size_receipt(fused)
        self.assertLess(fused_rows, materialized_rows)
        self.assertLess(fused_nnz, materialized_nnz)
        materialized_width = materialized_range[1] - materialized_range[0]
        fused_width = fused_range[1] - fused_range[0]
        self.assertLessEqual(
            fused_width,
            materialized_width + 2.0e-8,
            msg="ADD fusion unexpectedly weakened the residual relaxation",
        )

        fused_add = next(
            item
            for item in fused.metadata["layers"]
            if item["layer_id"] == 3
        )
        self.assertFalse(fused_add["materialized"])
        self.assertEqual(fused_add["new_cont"], 0)
        self.assertEqual(fused_add["new_ub"], 0)

    def test_shared_fanout_cancels_without_losing_factor_identity(self) -> None:
        toy = self._shared_fanout_cancellation_toy()
        exact = _exact_graph_range(toy)
        self.assertEqual((exact.lower, exact.upper), (Fraction(0), Fraction(0)))
        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        self.assertLPEncloses(materialized, exact, label="materialized")
        self.assertLPEncloses(fused, exact, label="fused")

        # Stable input provenance must remain a unique global coordinate even
        # when its contribution (and the shared ReLU coordinate) cancel from
        # the final value expression.
        root_id = int(fused.input_col_ids[0])
        root_positions = np.flatnonzero(fused.hz.col_ids == root_id)
        self.assertEqual(root_positions.size, 1)
        root_col = int(root_positions[0])
        self.assertEqual(float(fused.hz.Gc[0, root_col]), 0.0)
        constraint_column_nnz = (
            fused.hz.Ac.getcol(root_col).nnz
            + fused.hz.Auc.getcol(root_col).nnz
        )
        self.assertGreater(
            constraint_column_nnz,
            0,
            msg="the stable input factor vanished instead of merely cancelling",
        )
        self.assertEqual(
            len(set(np.asarray(fused.hz.col_ids).reshape(-1).tolist())),
            fused.hz.n_cont,
            msg="ADD fusion aliased two stable continuous factor identities",
        )

        materialized_rows, materialized_nnz = _size_receipt(materialized)
        fused_rows, fused_nnz = _size_receipt(fused)
        self.assertLess(fused_rows, materialized_rows)
        self.assertLess(fused_nnz, materialized_nnz)

    def test_1e16_plus_one_minus_1e16_keeps_exact_real_one(self) -> None:
        toy = self._catastrophic_add_toy()
        exact = _exact_graph_range(toy)
        self.assertEqual((exact.lower, exact.upper), (Fraction(1), Fraction(1)))
        materialized, fused = _build_pair(toy)

        # Confirm that this really exercises catastrophic cancellation rather
        # than accidentally evaluating in extended precision.
        self.assertEqual(float(materialized.hz.c[0]), 0.0)
        self.assertEqual(float(fused.hz.c[0]), 0.0)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")

        materialized_rows, materialized_nnz = _size_receipt(materialized)
        fused_rows, fused_nnz = _size_receipt(fused)
        self.assertLess(fused_rows, materialized_rows)
        self.assertLess(fused_nnz, materialized_nnz)
        fused_cube = _output_cube_fraction(fused)
        self.assertGreater(
            fused_cube[1] - fused_cube[0],
            0,
            msg="the numerical allowance was dropped after fused ADD",
        )

    def test_add_affine_relu_fuses_without_chain_cut(self) -> None:
        toy = self._add_affine_relu_toy()
        exact = _exact_graph_range(toy)
        self.assertEqual(
            (exact.lower, exact.upper),
            (Fraction(0), Fraction(5, 4)),
        )
        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        materialized_range = self.assertLPEncloses(
            materialized, exact, label="materialized"
        )
        fused_range = self.assertLPEncloses(fused, exact, label="fused")
        self.assertLessEqual(
            fused_range[1] - fused_range[0],
            materialized_range[1] - materialized_range[0] + 2.0e-8,
        )
        self.assertLess(_size_receipt(fused)[0], _size_receipt(materialized)[0])
        self.assertLess(_size_receipt(fused)[1], _size_receipt(materialized)[1])

        attempts = fused.metadata["live_affine_relu_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "applied")
        self.assertEqual(attempts[0]["layer_id"], 4)
        self.assertEqual(attempts[0]["box_inactive_rows"], 0)
        self.assertEqual(
            attempts[0]["source_generator_sha256_before"],
            attempts[0]["source_generator_sha256_after"],
        )
        self.assertEqual(
            fused.metadata["materialization_events"],
            [],
            msg="the fused affine silently reintroduced a full chain cut",
        )

    def test_projection_skip_chain_preserves_one_affine_source(self) -> None:
        toy = self._projection_skip_chain_toy()
        exact = _exact_graph_range(toy)
        with mock.patch.object(
            _OperatorHZBuilder,
            "_projection_skip_chain",
            return_value=None,
        ):
            cut = build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=-1,
                materialize_add=False,
                export_verified_preactivation_frame=False,
            )
        preserved = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=-1,
            materialize_add=False,
            export_verified_preactivation_frame=False,
        )

        self.assertExactCubeEncloses(cut, exact, label="projection cut")
        self.assertExactCubeEncloses(
            preserved, exact, label="projection preserved"
        )
        self.assertEqual(
            [(item["layer_id"], item["reason"])
             for item in cut.metadata["materialization_events"]],
            [(5, "affine_chain_cut")],
        )
        self.assertEqual(preserved.metadata["materialization_events"], [])
        receipt = preserved.metadata[
            "projection_skip_chain_preservations"
        ]
        self.assertEqual(len(receipt), 1)
        self.assertEqual(receipt[0]["projection_layer_id"], 9)
        self.assertEqual(receipt[0]["target_add_layer_id"], 10)
        self.assertFalse(receipt[0]["runtime_fallback"])
        self.assertFalse(receipt[0]["second_representation_built"])
        self.assertLess(preserved.hz.n_cont, cut.hz.n_cont)
        self.assertLess(preserved.hz.n_ub, cut.hz.n_ub)
        self.assertLess(preserved.hz.constraint_nnz, cut.hz.constraint_nnz)
        self.assertEqual(
            sum(
                int(layer.get("relu_relaxed", 0))
                for layer in preserved.metadata["layers"]
            ),
            0,
        )

    def test_projection_skip_chain_never_falls_back_after_admission(self) -> None:
        toy = self._projection_skip_chain_toy()
        established = _OperatorHZBuilder._try_fuse_affine_into_relu

        def fail_only_required_downstream(builder, **kwargs):
            layer = kwargs["layer"]
            if int(layer.id) == 11:
                return None, {
                    "schema": "operator_hz_live_affine_relu_v1",
                    "layer_id": 11,
                    "status": "fallback:forced_test_stoploss",
                }
            return established(builder, **kwargs)

        with mock.patch.object(
            _OperatorHZBuilder,
            "_try_fuse_affine_into_relu",
            autospec=True,
            side_effect=fail_only_required_downstream,
        ):
            with self.assertRaisesRegex(
                operator_hz_module.OperatorHZBuildError,
                "projection skip chain downstream affine",
            ):
                build_operator_hz(
                    toy.net,
                    toy.facts,
                    toy.facts,
                    exact_budget=-1,
                    materialize_add=False,
                    export_verified_preactivation_frame=False,
                )

    def test_projection_skip_chain_is_not_used_by_relaxed_relu_mode(self) -> None:
        toy = self._projection_skip_chain_toy()
        relaxed = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=False,
            export_verified_preactivation_frame=False,
        )
        self.assertEqual(
            relaxed.metadata["projection_skip_chain_preservations"], []
        )
        self.assertTrue(
            any(
                item["layer_id"] == 5
                and item["reason"] == "affine_chain_cut"
                for item in relaxed.metadata["materialization_events"]
            )
        )

    def test_add_affine_relu_box_collapses_only_proven_inactive_row(self) -> None:
        toy = self._add_affine_relu_toy(second_bias=-4)
        exact = _exact_graph_range(toy)
        self.assertEqual(
            (exact.lower, exact.upper),
            (Fraction(0), Fraction(0)),
        )
        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        self.assertLPEncloses(materialized, exact, label="materialized")
        self.assertLPEncloses(fused, exact, label="fused")

        attempt = fused.metadata["live_affine_relu_attempts"][0]
        self.assertEqual(attempt["status"], "applied")
        self.assertEqual(attempt["exact_inactive_rows"], 1)
        self.assertEqual(attempt["box_inactive_rows"], 1)
        self.assertEqual(attempt["prescreen_inactive_rows"], 1)
        self.assertEqual(attempt["composed_rows_evaluated"], 0)
        self.assertEqual(
            attempt["full_composed_nnz_scope"],
            "post_prescreen_rows_only",
        )
        self.assertEqual(attempt["stored_nnz"], 0)
        self.assertLessEqual(attempt["closest_box_inactive_upper"], 0.0)
        self.assertEqual(fused.hz.Gc.nnz, 0)

    def test_one_chunk_mixes_active_unstable_and_inactive_rows(self) -> None:
        toy = self._mixed_phase_affine_toy()
        exact = _ExactRange(
            Fraction(1, 2),
            Fraction(13, 2),
            phase_assignments=2,
            feasible_phases=2,
        )
        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        materialized_range = self.assertLPEncloses(
            materialized, exact, label="materialized"
        )
        fused_range = self.assertLPEncloses(fused, exact, label="fused")
        self.assertLessEqual(
            fused_range[1] - fused_range[0],
            materialized_range[1] - materialized_range[0] + 2.0e-8,
        )
        attempt = fused.metadata["live_affine_relu_attempts"][0]
        self.assertEqual(attempt["status"], "applied")
        self.assertEqual(attempt["output_rows"], 3)
        self.assertEqual(attempt["box_inactive_rows"], 1)
        self.assertEqual(attempt["prescreen_inactive_rows"], 1)
        self.assertEqual(attempt["composed_rows_evaluated"], 2)
        relu = next(
            item
            for item in fused.metadata["layers"]
            if item["layer_id"] == 5
        )
        self.assertEqual(
            (
                relu["relu_active"],
                relu["relu_unstable"],
                relu["relu_inactive"],
            ),
            (1, 1, 1),
        )

    def test_zero_ulp_boundaries_and_subnormal_weight_fail_closed(self) -> None:
        positive_subnormal = float(np.nextafter(0.0, np.inf))
        negative_subnormal = float(np.nextafter(0.0, -np.inf))
        cases = (
            ("zero", self._add_affine_relu_toy(second_weight=0, second_bias=0)),
            (
                "positive_subnormal_bias",
                self._add_affine_relu_toy(
                    second_weight=0,
                    second_bias=positive_subnormal,
                ),
            ),
            (
                "negative_subnormal_bias",
                self._add_affine_relu_toy(
                    second_weight=0,
                    second_bias=negative_subnormal,
                ),
            ),
            (
                "positive_subnormal_weight",
                self._add_affine_relu_toy(
                    second_weight=positive_subnormal,
                    second_bias=0,
                ),
            ),
        )
        for label, toy in cases:
            with self.subTest(label=label):
                exact = _exact_graph_range(toy)
                materialized, fused = _build_pair(toy)
                self.assertExactCubeEncloses(
                    materialized, exact, label=f"{label} materialized"
                )
                self.assertExactCubeEncloses(
                    fused, exact, label=f"{label} fused"
                )
                attempt = fused.metadata["live_affine_relu_attempts"][0]
                self.assertEqual(attempt["status"], "applied")
                if exact.upper > 0:
                    self.assertEqual(
                        attempt["box_inactive_rows"],
                        0,
                        msg=f"{label}: a positive exact value was discarded",
                    )

    def test_collapsed_row_requires_its_own_outward_recheck(self) -> None:
        # This stored-float row is inactive under its original sparse cube,
        # while the extra outward reductions needed to collapse |G|_1 into
        # err cross zero.  A one-stage "inactive then erase G" rule would be
        # unsound; the production path must retain this row.
        center = np.asarray(
            [float.fromhex("-0x1.0000000000011p+0")],
            dtype=np.float64,
        )
        generators = sp.csr_matrix(
            np.asarray([[1.0]], dtype=np.float64)
        )
        original = _AffineExpr(
            center,
            generators,
            np.zeros(1, dtype=np.float64),
        )
        _, original_upper = _OperatorHZBuilder._cube_bounds(original)
        self.assertLessEqual(float(original_upper[0]), 0.0)

        collapsed_radius = _nonnegative_sum_upper(
            _row_l1_upper(generators, name="audit.row_l1"),
            original.err,
            name="audit.collapsed_radius",
        )
        collapsed = _AffineExpr(
            center,
            sp.csr_matrix((1, 1), dtype=np.float64),
            collapsed_radius,
        )
        _, collapsed_upper = _OperatorHZBuilder._cube_bounds(collapsed)
        self.assertGreater(float(collapsed_upper[0]), 0.0)

    def test_affine_prescreen_is_outward_of_full_composition(self) -> None:
        rng = np.random.default_rng(0xA771CE)
        builder = object.__new__(_OperatorHZBuilder)
        for case in range(64):
            center = rng.normal(0.0, 0.3, size=5).astype(np.float64)
            dense_G = rng.normal(0.0, 0.08, size=(5, 4)).astype(
                np.float64
            )
            dense_G[rng.random(dense_G.shape) < 0.45] = 0.0
            generators = sp.csr_matrix(dense_G)
            generators.eliminate_zeros()
            generators.sort_indices()
            source = _AffineExpr(
                c=center,
                G=generators,
                err=np.abs(
                    rng.normal(0.0, 0.01, size=5).astype(np.float64)
                ),
                affine_depth=1,
            )
            dense_W = rng.normal(0.0, 0.4, size=(7, 5)).astype(
                np.float64
            )
            dense_W[rng.random(dense_W.shape) < 0.35] = 0.0
            matrix = sp.csr_matrix(dense_W)
            matrix.eliminate_zeros()
            matrix.sort_indices()
            bias = rng.normal(-0.25, 0.35, size=7).astype(np.float64)

            source_lower, source_upper = builder._cube_bounds(source)
            lower_distance = source.c - source_lower
            upper_distance = source_upper - source.c
            variable = (lower_distance > 0.0) | (upper_distance > 0.0)
            lower_distance[variable] = np.nextafter(
                lower_distance[variable], np.inf
            )
            upper_distance[variable] = np.nextafter(
                upper_distance[variable], np.inf
            )
            source_radius = np.maximum(lower_distance, upper_distance)
            source_mass = _nonnegative_sum_upper(
                np.abs(source.c),
                _row_l1_upper(source.G, name="prescreen_test.source_l1"),
                source.err,
                name="prescreen_test.source_mass",
            )
            screen = builder._affine_collapse_prescreen(
                source,
                matrix,
                bias,
                source_radius=source_radius,
                source_mass_upper=source_mass,
                layer_id=case,
            )
            _, screen_upper = builder._cube_bounds(screen)

            exact = builder._affine(
                source,
                matrix,
                bias,
                layer_id=case,
                _source_mass_upper=source_mass,
            )
            exact_radius = _nonnegative_sum_upper(
                _row_l1_upper(
                    exact.G, name="prescreen_test.exact_l1"
                ),
                exact.err,
                name="prescreen_test.exact_radius",
            )
            collapsed = _AffineExpr(
                c=exact.c,
                G=sp.csr_matrix(
                    (exact.size, exact.G.shape[1]), dtype=np.float64
                ),
                err=exact_radius,
                affine_depth=exact.affine_depth,
            )
            _, exact_upper = builder._cube_bounds(collapsed)
            self.assertTrue(
                np.all(screen_upper >= exact_upper),
                msg=f"case {case}: prescreen rounded inward",
            )

            for row in range(matrix.shape[0]):
                exact_real_upper = Fraction.from_float(float(bias[row]))
                for source_row in range(matrix.shape[1]):
                    weight = Fraction.from_float(
                        float(dense_W[row, source_row])
                    )
                    exact_real_upper += weight * Fraction.from_float(
                        float(source.c[source_row])
                    )
                    exact_real_upper += abs(weight) * Fraction.from_float(
                        float(source.err[source_row])
                    )
                for factor in range(source.G.shape[1]):
                    coefficient = Fraction(0)
                    for source_row in range(matrix.shape[1]):
                        coefficient += Fraction.from_float(
                            float(dense_W[row, source_row])
                        ) * Fraction.from_float(
                            float(dense_G[source_row, factor])
                        )
                    exact_real_upper += abs(coefficient)
                self.assertGreaterEqual(
                    Fraction.from_float(float(screen_upper[row])),
                    exact_real_upper,
                    msg=f"case {case}, row {row}: exact-real escape",
                )

    def test_affine_fusion_keeps_catastrophic_positive_value_live(self) -> None:
        toy = self._catastrophic_affine_relu_toy()
        exact = _exact_graph_range(toy)
        self.assertEqual(
            (exact.lower, exact.upper),
            (Fraction(1), Fraction(1)),
        )
        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        self.assertLPEncloses(materialized, exact, label="materialized")
        self.assertLPEncloses(fused, exact, label="fused")
        attempt = fused.metadata["live_affine_relu_attempts"][0]
        self.assertEqual(attempt["status"], "applied")
        self.assertEqual(
            attempt["box_inactive_rows"],
            0,
            msg="catastrophic cancellation was falsely classified inactive",
        )

    @staticmethod
    def assertSameHZStructure(
        left: OperatorHZBuild,
        right: OperatorHZBuild,
        *,
        compare_ids: bool = True,
    ) -> None:
        vector_names = ["c", "b", "ub"]
        if compare_ids:
            vector_names.extend(("col_ids", "bcol_ids"))
        for name in vector_names:
            np.testing.assert_array_equal(
                np.asarray(getattr(left.hz, name)),
                np.asarray(getattr(right.hz, name)),
                err_msg=f"HZ vector {name} changed",
            )
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            lhs = getattr(left.hz, name).tocsr()
            rhs = getattr(right.hz, name).tocsr()
            if lhs.shape != rhs.shape:
                raise AssertionError(
                    f"HZ matrix {name} shape changed: {lhs.shape}/{rhs.shape}"
                )
            np.testing.assert_array_equal(
                lhs.indptr, rhs.indptr, err_msg=f"{name}.indptr changed"
            )
            np.testing.assert_array_equal(
                lhs.indices, rhs.indices, err_msg=f"{name}.indices changed"
            )
            np.testing.assert_array_equal(
                lhs.data, rhs.data, err_msg=f"{name}.data changed"
            )

    def test_chunk_size_is_semantically_and_structurally_invariant(self) -> None:
        toy = self._add_affine_relu_toy()
        with mock.patch.object(
            operator_hz_module, "_LIVE_AFFINE_CHUNK_ROWS", 1
        ):
            chunk_one = build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=0,
                materialize_add=False,
            )
        with mock.patch.object(
            operator_hz_module, "_LIVE_AFFINE_CHUNK_ROWS", 7
        ):
            chunk_seven = build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=0,
                materialize_add=False,
            )
        self.assertSameHZStructure(
            chunk_one,
            chunk_seven,
            compare_ids=False,
        )

    def test_precomputed_source_mass_is_bitwise_identical(self) -> None:
        builder = _OperatorHZBuilder(
            SimpleNamespace(layers=[], preds={}),
            {},
            {},
            exact_budget=0,
            materialize_add=False,
            preactivation_lp_budget=0,
            preactivation_lp_time_limit=0.0,
            deadline=None,
        )
        source = _AffineExpr(
            c=np.asarray([0.25, -0.5], dtype=np.float64),
            G=sp.csr_matrix(
                np.asarray(
                    [[1.0, -0.25], [0.5, 0.125]],
                    dtype=np.float64,
                )
            ),
            err=np.asarray([1.0e-12, 2.0e-12], dtype=np.float64),
            affine_depth=1,
        )
        matrix = sp.csr_matrix(
            np.asarray(
                [[1.5, -0.75], [-0.5, 2.0]],
                dtype=np.float64,
            )
        )
        bias = np.asarray([0.125, -0.25], dtype=np.float64)
        source_mass = _nonnegative_sum_upper(
            np.abs(source.c),
            _row_l1_upper(source.G, name="cached.source_G_l1"),
            source.err,
            name="cached.source_mass",
        )
        ordinary = builder._affine(
            source, matrix, bias, layer_id=7
        )
        cached = builder._affine(
            source,
            matrix,
            bias,
            layer_id=7,
            _source_mass_upper=source_mass,
        )
        np.testing.assert_array_equal(ordinary.c, cached.c)
        np.testing.assert_array_equal(ordinary.err, cached.err)
        np.testing.assert_array_equal(ordinary.G.indptr, cached.G.indptr)
        np.testing.assert_array_equal(ordinary.G.indices, cached.G.indices)
        np.testing.assert_array_equal(ordinary.G.data, cached.G.data)

    def test_opposite_difference_matches_independent_reverse_bitwise(self) -> None:
        builder = _OperatorHZBuilder(
            SimpleNamespace(layers=[], preds={}),
            {},
            {},
            exact_budget=-1,
            materialize_add=False,
            preactivation_lp_budget=0,
            preactivation_lp_time_limit=0.0,
            deadline=None,
        )
        rng = np.random.default_rng(0xADD20260811)
        for _case in range(64):
            left_dense = rng.integers(-8, 9, size=(5, 7)).astype(
                np.float64
            ) / 8.0
            right_dense = rng.integers(-8, 9, size=(5, 7)).astype(
                np.float64
            ) / 8.0
            left_dense[np.abs(left_dense) < 0.375] = 0.0
            right_dense[np.abs(right_dense) < 0.375] = 0.0
            left = _AffineExpr(
                c=rng.integers(-8, 9, size=5).astype(np.float64) / 8.0,
                G=sp.csr_matrix(left_dense),
                err=np.ldexp(
                    rng.integers(0, 5, size=5).astype(np.float64), -48
                ),
            )
            right = _AffineExpr(
                c=rng.integers(-8, 9, size=5).astype(np.float64) / 8.0,
                G=sp.csr_matrix(right_dense),
                err=np.ldexp(
                    rng.integers(0, 5, size=5).astype(np.float64), -49
                ),
            )
            builder.n_cont = 7
            forward, reused_reverse = builder._opposite_differences(
                left, right
            )
            independent_forward = builder._difference(left, right)
            independent_reverse = builder._difference(right, left)
            for reused, independent in (
                (forward, independent_forward),
                (reused_reverse, independent_reverse),
            ):
                np.testing.assert_array_equal(reused.c, independent.c)
                np.testing.assert_array_equal(reused.err, independent.err)
                np.testing.assert_array_equal(
                    reused.G.indptr, independent.G.indptr
                )
                np.testing.assert_array_equal(
                    reused.G.indices, independent.G.indices
                )
                np.testing.assert_array_equal(reused.G.data, independent.G.data)

    def test_candidate_nnz_stoploss_falls_back_without_frame_mutation(self) -> None:
        toy = self._add_affine_relu_toy()
        materialized = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
        )
        with mock.patch.object(
            operator_hz_module, "_LIVE_AFFINE_MAX_STORED_NNZ", 0
        ):
            fallback = build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=0,
                materialize_add=False,
            )
        attempt = fallback.metadata["live_affine_relu_attempts"][0]
        self.assertEqual(attempt["status"], "fallback:stored_nnz_stoploss")
        self.assertEqual(
            fallback.metadata["materialization_events"][0]["reason"],
            "affine_chain_cut",
        )
        # Stable ids are process-global and therefore intentionally differ
        # between independent builds.  Coefficient frames must still match.
        self.assertSameHZStructure(
            materialized,
            fallback,
            compare_ids=False,
        )

    def test_shared_affine_fanout_forces_sound_fallback(self) -> None:
        toy = self._affine_fanout_toy()
        exact = _exact_graph_range(toy)
        materialized, fused = _build_pair(toy)
        self.assertExactCubeEncloses(materialized, exact, label="materialized")
        self.assertExactCubeEncloses(fused, exact, label="fused")
        self.assertLPEncloses(materialized, exact, label="materialized")
        self.assertLPEncloses(fused, exact, label="fused")
        first_attempt = fused.metadata["live_affine_relu_attempts"][0]
        self.assertEqual(first_attempt["layer_id"], 4)
        self.assertEqual(first_attempt["status"], "not_eligible")
        first_cut = fused.metadata["materialization_events"][0]
        self.assertEqual(first_cut["layer_id"], 3)
        self.assertEqual(first_cut["reason"], "affine_chain_cut")

    def test_deterministic_rational_dag_fuzz(self) -> None:
        rng = random.Random(0xADD20260728)
        nonzero_weights = (
            Fraction(-3, 2),
            Fraction(-1),
            Fraction(-1, 2),
            Fraction(1, 2),
            Fraction(1),
            Fraction(3, 2),
        )
        biases = (
            Fraction(-1, 2),
            Fraction(-1, 4),
            Fraction(0),
            Fraction(1, 4),
            Fraction(1, 2),
        )
        for case in range(16):
            first_weight = rng.choice(nonzero_weights)
            first_bias = rng.choice(biases)
            skip_weight = rng.choice(nonzero_weights)
            skip_bias = rng.choice(biases)
            input_layer, spec_layer = _input_layers(Fraction(-1), Fraction(1))
            layers = [
                input_layer,
                spec_layer,
                _dense(2, first_weight, first_bias),
                _layer(3, "RELU"),
                _dense(4, skip_weight, skip_bias),
                _layer(5, "ADD"),
                _layer(6, "RELU"),
                _layer(7, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
            ]
            toy = _assemble_scalar_toy(
                layers,
                {
                    0: [],
                    1: [0],
                    2: [1],
                    3: [2],
                    4: [1],
                    5: [3, 4],
                    6: [5],
                    7: [6],
                },
                input_lb=Fraction(-1),
                input_ub=Fraction(1),
            )
            exact = _exact_graph_range(toy)
            self.assertEqual(exact.phase_assignments, 4)
            materialized, fused = _build_pair(toy)
            self.assertExactCubeEncloses(
                materialized, exact, label=f"case {case} materialized"
            )
            self.assertExactCubeEncloses(
                fused, exact, label=f"case {case} fused"
            )
            materialized_range = self.assertLPEncloses(
                materialized,
                exact,
                label=f"case {case} materialized",
            )
            fused_range = self.assertLPEncloses(
                fused,
                exact,
                label=f"case {case} fused",
            )
            materialized_rows, materialized_nnz = _size_receipt(materialized)
            fused_rows, fused_nnz = _size_receipt(fused)
            self.assertLessEqual(fused_rows, materialized_rows)
            self.assertLessEqual(fused_nnz, materialized_nnz)
            scale = max(
                1.0,
                abs(materialized_range[0]),
                abs(materialized_range[1]),
            )
            self.assertLessEqual(
                fused_range[1] - fused_range[0],
                materialized_range[1] - materialized_range[0] + 5.0e-8 * scale,
                msg=f"case {case}: fusion produced a looser LP projection",
            )

    def test_64_rational_add_affine_relu_dags_against_phase_oracle(self) -> None:
        rng = random.Random(0xAFF1AE20260728)
        weights = (
            Fraction(-3, 2),
            Fraction(-1),
            Fraction(-1, 2),
            Fraction(1, 2),
            Fraction(1),
            Fraction(3, 2),
        )
        biases = (
            Fraction(-3, 4),
            Fraction(-1, 2),
            Fraction(-1, 4),
            Fraction(0),
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(3, 4),
        )
        for case in range(64):
            input_layer, spec_layer = _input_layers(Fraction(-1), Fraction(1))
            layers = [
                input_layer,
                spec_layer,
                _dense(2, rng.choice(weights), rng.choice(biases)),
                _layer(3, "RELU"),
                _dense(4, rng.choice(weights), rng.choice(biases)),
                _layer(5, "ADD"),
                _dense(6, rng.choice(weights), rng.choice(biases)),
                _layer(7, "RELU"),
                _layer(8, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
            ]
            toy = _assemble_scalar_toy(
                layers,
                {
                    0: [],
                    1: [0],
                    2: [1],
                    3: [2],
                    4: [1],
                    5: [3, 4],
                    6: [5],
                    7: [6],
                    8: [7],
                },
                input_lb=Fraction(-1),
                input_ub=Fraction(1),
            )
            exact = _exact_graph_range(toy)
            materialized, fused = _build_pair(toy)
            self.assertExactCubeEncloses(
                materialized, exact, label=f"case {case} materialized"
            )
            self.assertExactCubeEncloses(
                fused, exact, label=f"case {case} fused"
            )
            materialized_range = self.assertLPEncloses(
                materialized,
                exact,
                label=f"case {case} materialized",
            )
            fused_range = self.assertLPEncloses(
                fused,
                exact,
                label=f"case {case} fused",
            )
            scale = max(
                1.0,
                abs(materialized_range[0]),
                abs(materialized_range[1]),
            )
            self.assertLessEqual(
                fused_range[1] - fused_range[0],
                materialized_range[1] - materialized_range[0]
                + 8.0e-8 * scale,
                msg=f"case {case}: direct fusion weakened the LP projection",
            )
            attempt = fused.metadata["live_affine_relu_attempts"][0]
            self.assertIn(
                attempt["status"],
                {"applied", "fallback:estimated_nnz_regression"},
            )

    def test_fused_add_honours_expired_shared_deadline(self) -> None:
        toy = self._residual_add_relu_toy()
        with self.assertRaises(OperatorHZBuildTimeout):
            build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                exact_budget=0,
                materialize_add=False,
                deadline=time.monotonic() - 1.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
