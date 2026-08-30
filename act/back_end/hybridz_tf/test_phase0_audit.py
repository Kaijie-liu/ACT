#!/usr/bin/env python3
# ===- test_phase0_audit.py - HybridZ Phase-0 soundness gates ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Small, deterministic HybridZ gates runnable without pytest.

Run from the repository root:

    python -m act.back_end.hybridz_tf.test_phase0_audit

The independent dense oracle enumerates ReLU phases and solves the resulting
continuous LPs with SciPy/HiGHS.  The sparse diamond test deliberately uses
two sibling nonlinear branches; local positional auxiliary-factor allocation
would incorrectly couple their phases and shrink ``ReLU(x) + ReLU(-x)`` from
``[0, 1]`` to ``{0}``.
"""

from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import importlib
import itertools
import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import (
    Bounds as SciPyBounds,
    LinearConstraint,
    linprog,
    milp,
)
import torch

import act.back_end.solver.solver_hz as solver_hz_module
from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Bounds, ConSet, Fact, Layer, Net
from act.back_end.hybridz_tf.tf_mlp import (
    hz_apply_relu,
    sparse_hz_add_same_frame,
    sparse_hz_apply_relu_exact,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)
from act.back_end.hybridz_tf.operator_hz import (
    _AffineExpr,
    _PreactivationLPBase,
    _independent_preactivation_lagrangian_upper,
    _relu_triangle_parameters,
    build_operator_hz,
    operator_hz_self_test,
)
from act.back_end.layer_schema import LayerKind
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _base_milp_matrices_from_blocks,
    _hz_cube_row_upper_bounds,
    _hz_gpu_dual_candidate_filter,
    _hz_independent_lp_lagrangian_upper,
    _hz_shared_deadline_self_test,
    hz_add_const,
    hz_base_feasibility,
    hz_from_bounds,
    hz_mark_constructively_nonempty,
    hz_mark_known_nonempty,
    hz_multiply,
    hz_objbound_decide,
    hz_objbound_safe_capability_receipt,
    hz_row_max,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


DTYPE = torch.float64
DEVICE = torch.device("cpu")

W1 = torch.tensor(
    [[1.2, -0.7], [-0.6, 1.1], [0.9, 0.4]],
    dtype=DTYPE,
)
B1 = torch.tensor([-0.15, 0.05, -0.35], dtype=DTYPE)
W2 = torch.tensor([[0.7, -1.1, 0.5]], dtype=DTYPE)
B2 = torch.tensor([0.03], dtype=DTYPE)


def _dense_toy_hz(lb: np.ndarray, ub: np.ndarray):
    """Propagate the fixed 2-3-1 toy through exact dense HybridZ."""

    lower = torch.as_tensor(np.asarray(lb), dtype=DTYPE).reshape(1, -1)
    upper = torch.as_tensor(np.asarray(ub), dtype=DTYPE).reshape(1, -1)
    hz = hz_from_bounds(
        Bounds(lb=lower, ub=upper),
        DTYPE,
        DEVICE,
        track_ids=True,
    )
    hz = hz_add_const(hz_multiply(hz, W1), B1.reshape(-1, 1))
    hz = hz_apply_relu(hz)
    return hz_add_const(hz_multiply(hz, W2), B2.reshape(-1, 1))


def _dense_scalar_range(hz) -> tuple[float, float]:
    """Exact integer support range of a scalar dense HZ."""

    upper = hz_row_max(
        hz,
        np.array([1.0], dtype=np.float64),
        integer=True,
        time_limit=5.0,
    )
    neg_lower = hz_row_max(
        hz,
        np.array([-1.0], dtype=np.float64),
        integer=True,
        time_limit=5.0,
    )
    if upper is None or neg_lower is None:
        raise AssertionError("dense HybridZ support solve returned UNKNOWN")
    return -float(neg_lower), float(upper)


def _phase_enumeration_oracle(
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float]:
    """Independent exact oracle for the fixed ReLU network.

    Every one of the three ReLU phase patterns is enumerated.  Within a fixed
    pattern the network and phase conditions are affine, so two HiGHS LPs give
    the exact scalar minimum and maximum.
    """

    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    w1 = W1.detach().cpu().numpy()
    b1 = B1.detach().cpu().numpy()
    w2 = W2.detach().cpu().numpy().reshape(-1)
    b2 = float(B2.item())
    minima: list[float] = []
    maxima: list[float] = []

    for active_bits in itertools.product((False, True), repeat=w1.shape[0]):
        active = np.asarray(active_bits, dtype=bool)
        rows = []
        rhs = []
        for neuron, is_active in enumerate(active):
            if is_active:
                # W x + b >= 0  <=>  -W x <= b.
                rows.append(-w1[neuron])
                rhs.append(b1[neuron])
            else:
                # W x + b <= 0  <=>   W x <= -b.
                rows.append(w1[neuron])
                rhs.append(-b1[neuron])

        objective = w2[active] @ w1[active] if active.any() else np.zeros(lb.size)
        constant = b2 + float(w2[active] @ b1[active]) if active.any() else b2
        common = {
            "A_ub": np.asarray(rows, dtype=np.float64),
            "b_ub": np.asarray(rhs, dtype=np.float64),
            "bounds": list(zip(lb, ub)),
            "method": "highs",
        }
        min_result = linprog(objective, **common)
        if not min_result.success:
            continue
        max_result = linprog(-objective, **common)
        if not max_result.success:
            raise AssertionError(
                f"phase LP max failed after min succeeded: {max_result.message}"
            )
        minima.append(constant + float(min_result.fun))
        maxima.append(constant - float(max_result.fun))

    if not minima:
        raise AssertionError("all independently enumerated ReLU phases infeasible")
    return min(minima), max(maxima)


def _identity_net(
    *,
    out_kind: str,
    c: torch.Tensor,
    d: torch.Tensor,
) -> Net:
    """Construct a one-dimensional B=1 identity network and assertion."""

    input_vars = [0]
    output_vars = [1]
    weight = torch.ones((1, 1), dtype=DTYPE)
    bias = torch.zeros(1, dtype=DTYPE)
    assert_params = OutputSpec(kind=out_kind, c=c, d=d).encode_linear(
        B=1,
        n_out=1,
        device=DEVICE,
        dtype=DTYPE,
    )
    layers = [
        Layer(
            id=0,
            kind=LayerKind.INPUT.value,
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=input_vars,
        ),
        Layer(
            id=1,
            kind=LayerKind.INPUT_SPEC.value,
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=DTYPE),
                "ub": torch.tensor([[1.0]], dtype=DTYPE),
            },
            in_vars=input_vars,
            out_vars=input_vars,
        ),
        Layer(
            id=2,
            kind=LayerKind.DENSE.value,
            params={
                "weight": weight,
                "weight_pos": weight,
                "weight_neg": torch.zeros_like(weight),
                "bias": bias,
                "in_features": 1,
                "out_features": 1,
                "input_shape": (1,),
            },
            in_vars=input_vars,
            out_vars=output_vars,
        ),
        Layer(
            id=3,
            kind=LayerKind.ASSERT.value,
            params=assert_params,
            in_vars=output_vars,
            out_vars=output_vars,
        ),
    ]
    return Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2]},
        succs={0: [1], 1: [2], 2: [3], 3: []},
    )


def _sparse_scalar_range(hz: SparseHZono) -> tuple[float, float]:
    """Exact scalar range of a sparse HZ via an independent SciPy MILP."""

    n_cont = hz.n_cont
    n_bin = hz.n_bin
    n_vars = n_cont + n_bin
    row_gc = np.asarray(hz.Gc.getrow(0).toarray()).reshape(-1)
    row_gb = np.asarray(hz.Gb.getrow(0).toarray()).reshape(-1)
    objective = np.concatenate([row_gc, 2.0 * row_gb])
    constant = float(hz.c[0] - row_gb.sum())
    constraints = []

    if hz.n_eq:
        matrix = sp.hstack([hz.Ac, 2.0 * hz.Ab], format="csr")
        rhs = hz.b + np.asarray(hz.Ab.sum(axis=1)).reshape(-1)
        constraints.append(LinearConstraint(matrix, lb=rhs, ub=rhs))
    if hz.n_ub:
        matrix = sp.hstack([hz.Auc, 2.0 * hz.Aub], format="csr")
        rhs = hz.ub + np.asarray(hz.Aub.sum(axis=1)).reshape(-1)
        constraints.append(
            LinearConstraint(
                matrix,
                lb=np.full(rhs.shape, -np.inf, dtype=np.float64),
                ub=rhs,
            )
        )

    variable_bounds = SciPyBounds(
        lb=np.concatenate([-np.ones(n_cont), np.zeros(n_bin)]),
        ub=np.ones(n_vars),
    )
    integrality = np.concatenate(
        [np.zeros(n_cont, dtype=np.int32), np.ones(n_bin, dtype=np.int32)]
    )
    options = {"time_limit": 5.0, "mip_rel_gap": 1e-9}
    minimum = milp(
        c=objective,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options,
    )
    maximum = milp(
        c=-objective,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=constraints,
        options=options,
    )
    if not minimum.success or not maximum.success:
        raise AssertionError(
            "sparse HZ oracle failed: "
            f"min={minimum.message!r}, max={maximum.message!r}"
        )
    return constant + float(minimum.fun), constant - float(maximum.fun)


_SPARSE_FIELDS = {item.name for item in fields(SparseHZono)}
_SPARSE_HAS_FACTOR_IDS = {"col_ids", "bcol_ids"} <= _SPARSE_FIELDS


def _unchanged(test):
    return test


_sparse_provenance_required = (
    _unchanged if _SPARSE_HAS_FACTOR_IDS else unittest.expectedFailure
)


class DenseExactOracleTests(unittest.TestCase):
    """Point, exact-box, and binary-count gates for dense HybridZ."""

    def test_degenerate_point_matches_concrete_network(self):
        point = np.array([0.2, -0.4], dtype=np.float64)
        hz = _dense_toy_hz(point, point)
        lower, upper = _dense_scalar_range(hz)
        concrete = float(
            (
                W2
                @ torch.relu(W1 @ torch.as_tensor(point, dtype=DTYPE) + B1)
                + B2
            ).item()
        )
        self.assertAlmostEqual(lower, concrete, places=10)
        self.assertAlmostEqual(upper, concrete, places=10)

    def test_box_matches_independent_phase_enumeration_oracle(self):
        lower_input = np.array([-0.9, -0.4], dtype=np.float64)
        upper_input = np.array([0.8, 1.1], dtype=np.float64)
        hz = _dense_toy_hz(lower_input, upper_input)
        hz_lower, hz_upper = _dense_scalar_range(hz)
        oracle_lower, oracle_upper = _phase_enumeration_oracle(
            lower_input,
            upper_input,
        )
        self.assertAlmostEqual(hz_lower, oracle_lower, places=8)
        self.assertAlmostEqual(hz_upper, oracle_upper, places=8)
        self.assertEqual(
            int(hz.Gb.shape[1]),
            3,
            "the three analytically unstable ReLUs must create three binaries",
        )
        self.assertEqual(int(hz.Ac.shape[0]), 9)

    def test_rival_queries_share_one_wall_clock_deadline(self):
        self.assertTrue(_hz_shared_deadline_self_test())


class DagSchedulingTests(unittest.TestCase):
    """A residual-style join must not be rebuilt once per arriving sibling."""

    @staticmethod
    def _diamond_net() -> Net:
        one = [0]
        layers = [
            Layer(
                id=0,
                kind=LayerKind.INPUT.value,
                params={"shape": (1, 1), "dtype": "torch.float64"},
                in_vars=[],
                out_vars=one,
            ),
            Layer(
                id=1,
                kind=LayerKind.INPUT_SPEC.value,
                params={
                    "kind": "BOX",
                    "lb": torch.tensor([[-1.0]], dtype=DTYPE),
                    "ub": torch.tensor([[1.0]], dtype=DTYPE),
                },
                in_vars=one,
                out_vars=one,
            ),
            Layer(
                id=2,
                kind=LayerKind.RELU.value,
                params={},
                in_vars=one,
                out_vars=[1],
            ),
            Layer(
                id=3,
                kind=LayerKind.RELU.value,
                params={},
                in_vars=one,
                out_vars=[2],
            ),
            Layer(
                id=4,
                kind=LayerKind.ADD.value,
                params={},
                in_vars=[1, 2],
                out_vars=[3],
            ),
            Layer(
                id=5,
                kind=LayerKind.ASSERT.value,
                params=OutputSpec(
                    kind=OutKind.LINEAR_LE,
                    c=torch.tensor([1.0], dtype=DTYPE),
                    d=torch.tensor([10.0], dtype=DTYPE),
                ).encode_linear(
                    B=1,
                    n_out=1,
                    device=DEVICE,
                    dtype=DTYPE,
                ),
                in_vars=[3],
                out_vars=[3],
            ),
        ]
        return Net(
            layers=layers,
            preds={0: [], 1: [0], 2: [1], 3: [1], 4: [2, 3], 5: [4]},
            succs={0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: []},
        )

    def test_diamond_join_dispatches_once_per_analysis(self):
        analyze_module = importlib.import_module("act.back_end.analyze")
        net = self._diamond_net()
        calls = {layer.id: 0 for layer in net.layers}

        class _NoSideState:
            @staticmethod
            def side_state_signature(_layer_id):
                return None

        def _dispatch(layer, before, _after, _net):
            calls[layer.id] += 1
            bounds = before[layer.id].bounds
            # Make every scheduled transfer finite and input-dependent while
            # keeping the stub independent of HybridZ implementation details.
            shift = torch.as_tensor(
                float(layer.id) / 100.0,
                dtype=bounds.lb.dtype,
                device=bounds.lb.device,
            )
            return Fact(
                bounds=Bounds(bounds.lb + shift, bounds.ub + shift),
                cons=before[layer.id].cons,
            )

        entry = Fact(
            bounds=Bounds(
                torch.tensor([[-1.0]], dtype=DTYPE),
                torch.tensor([[1.0]], dtype=DTYPE),
            ),
            cons=ConSet(),
        )
        with (
            patch.object(analyze_module, "dispatch_tf", side_effect=_dispatch),
            patch.object(
                analyze_module,
                "get_transfer_function",
                return_value=_NoSideState(),
            ),
        ):
            before, after, constraints = analyze_module.analyze(net, 1, entry)

        self.assertEqual(calls[4], 1)
        self.assertEqual(calls[5], 1)
        self.assertTrue(all(count == 1 for count in calls.values()), calls)

        calls = {layer.id: 0 for layer in net.layers}
        refined = Fact(
            bounds=Bounds(
                torch.tensor([[-0.5]], dtype=DTYPE),
                torch.tensor([[0.5]], dtype=DTYPE),
            ),
            cons=ConSet(),
        )
        cache = analyze_module.AnalyzeCache(before, after, constraints)
        with (
            patch.object(analyze_module, "dispatch_tf", side_effect=_dispatch),
            patch.object(
                analyze_module,
                "get_transfer_function",
                return_value=_NoSideState(),
            ),
        ):
            analyze_module.analyze(net, 1, refined, cache=cache)

        self.assertEqual(calls[4], 1)
        self.assertEqual(calls[5], 1)


class StrictHybridZEntryTests(unittest.TestCase):
    """End-to-end strict-backend result-policy gates."""

    def setUp(self):
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        self.config = BackendConfig(
            solver="hybridz",
            timeout=9.0,
            hybridz=HybridZConfig(
                timeout=2.0,
                engine="dense_hz_objbound",
            ),
        )

    def tearDown(self):
        set_solver_mode(None)
        set_transfer_function_mode("interval")

    def test_backend_config_reaches_strict_safe_path(self):
        net = _identity_net(
            out_kind=OutKind.LINEAR_LE,
            c=torch.tensor([1.0], dtype=DTYPE),
            d=torch.tensor([2.0], dtype=DTYPE),
        )
        result = verify_once(net, backend_cfg=self.config)[0]
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(result.metadata.get("solver"), "hybridz")
        self.assertEqual(result.metadata.get("engine"), "dense_hz_objbound")
        self.assertEqual(result.metadata.get("hz_verdict"), "SAFE")
        self.assertEqual(result.metadata.get("hz_timeout_s"), 2.0)

    def test_operator_engine_builds_from_interval_facts_end_to_end(self):
        net = _identity_net(
            out_kind=OutKind.LINEAR_LE,
            c=torch.tensor([1.0], dtype=DTYPE),
            d=torch.tensor([2.0], dtype=DTYPE),
        )
        config = BackendConfig(
            solver="hybridz",
            timeout=9.0,
            hybridz=HybridZConfig(
                timeout=2.0,
                engine="operator_hz_objbound",
                operator_exact_budget=0,
            ),
        )
        result = verify_once(net, backend_cfg=config)[0]
        self.assertEqual(result.status, VerifyStatus.CERTIFIED)
        self.assertEqual(result.metadata.get("engine"), "operator_hz_objbound")
        self.assertEqual(result.metadata.get("hz_verdict"), "SAFE")
        # One input factor plus one final-logit factor carrying the certified
        # affine roundoff envelope.  Treating W@x as exact binary64 arithmetic
        # was the source of the dense-cancellation false SAFE regression.
        self.assertEqual(result.metadata.get("operator_n_cont"), 2)
        operator_meta = result.metadata.get("operator_hz", {})
        self.assertEqual(
            operator_meta.get("output_roundoff_generator_count"),
            1,
        )
        self.assertGreater(
            float(operator_meta.get("output_roundoff_error_max", 0.0)),
            0.0,
        )
        self.assertEqual(
            operator_meta.get("schema"),
            "operator_hz_local_graph_v1",
        )

    def test_unsafe_without_raw_replay_must_remain_unknown(self):
        """An HZ factor witness alone is not a replayed concrete input."""

        net = _identity_net(
            out_kind=OutKind.UNSAFE_LINEAR,
            c=torch.tensor([[1.0]], dtype=DTYPE),
            d=torch.tensor([0.0], dtype=DTYPE),
        )
        result = verify_once(net, backend_cfg=self.config)[0]
        self.assertEqual(result.metadata.get("hz_verdict"), "UNSAFE")
        self.assertIsNone(result.counterexample)
        self.assertEqual(result.status, VerifyStatus.UNKNOWN)

    def test_unsafe_with_independent_replay_returns_counterexample(self):
        net = _identity_net(
            out_kind=OutKind.UNSAFE_LINEAR,
            c=torch.tensor([[1.0]], dtype=DTYPE),
            d=torch.tensor([0.0], dtype=DTYPE),
        )
        seen = {}

        def replay(x_batch):
            seen["x"] = x_batch.clone()
            return {
                "valid_counterexample": bool((x_batch <= 0.0).all().item()),
                "model_sha256": "toy-model",
                "spec_sha256": "toy-spec",
            }

        result = verify_once(
            net,
            backend_cfg=self.config,
            counterexample_replay_fn=replay,
        )[0]
        self.assertEqual(result.status, VerifyStatus.FALSIFIED)
        self.assertIsNotNone(result.counterexample)
        self.assertIn("x", seen)
        self.assertLessEqual(float(result.counterexample.item()), 0.0)
        self.assertEqual(
            result.metadata.get("hz_independent_replay"),
            "independent_replay_accepted",
        )

    def test_replay_rejection_or_model_conflict_fails_closed(self):
        net = _identity_net(
            out_kind=OutKind.UNSAFE_LINEAR,
            c=torch.tensor([[1.0]], dtype=DTYPE),
            d=torch.tensor([0.0], dtype=DTYPE),
        )
        rejected = verify_once(
            net,
            backend_cfg=self.config,
            counterexample_replay_fn=lambda _x: False,
        )[0]
        self.assertEqual(rejected.status, VerifyStatus.UNKNOWN)
        self.assertIsNone(rejected.counterexample)

        conflict = verify_once(
            net,
            backend_cfg=self.config,
            model_fn=lambda x: torch.ones_like(x),
            counterexample_replay_fn=lambda _x: True,
        )[0]
        self.assertEqual(conflict.status, VerifyStatus.UNKNOWN)
        self.assertEqual(conflict.metadata.get("reason"), "hybridz_replay_conflict")


class IndependentLPCertificateTests(unittest.TestCase):
    """The LP solver supplies candidates; stored-float arithmetic proves them."""

    @staticmethod
    def _upper(
        *,
        q,
        kappa,
        A,
        rl,
        ru,
        lb,
        ub,
        d,
    ):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=np.zeros(q.size, dtype=np.float64),
            Gc=sp.eye(q.size, format="csr", dtype=np.float64),
            C_row=q,
            threshold=-float(kappa),
            A=sp.csr_matrix(A, shape=(len(rl), q.size), dtype=np.float64),
            rl=np.asarray(rl, dtype=np.float64),
            ru=np.asarray(ru, dtype=np.float64),
            lb=np.asarray(lb, dtype=np.float64),
            ub=np.asarray(ub, dtype=np.float64),
            # HiGHS minimizes -q, hence its row dual is -d.
            row_dual=-np.asarray(d, dtype=np.float64),
        )
        if upper is None:
            raise AssertionError(f"LP certificate failed: {receipt}")
        return np.longdouble(upper), receipt

    def test_upper_lower_and_equality_dual_signs(self):
        upper, _ = self._upper(
            q=[1.0],
            kappa=0.0,
            A=[[1.0]],
            rl=[-np.inf],
            ru=[3.0],
            lb=[0.0],
            ub=[10.0],
            d=[1.0],
        )
        self.assertGreaterEqual(upper, np.longdouble(3.0))
        self.assertLess(upper, np.longdouble(3.0 + 1e-12))

        lower, _ = self._upper(
            q=[-1.0],
            kappa=0.0,
            A=[[1.0]],
            rl=[2.0],
            ru=[np.inf],
            lb=[0.0],
            ub=[10.0],
            d=[-1.0],
        )
        self.assertGreaterEqual(lower, np.longdouble(-2.0))
        self.assertLess(lower, np.longdouble(-2.0 + 1e-12))

        equality, _ = self._upper(
            q=[1.0],
            kappa=0.0,
            A=[[1.0]],
            rl=[2.0],
            ru=[2.0],
            lb=[0.0],
            ub=[10.0],
            d=[1.0],
        )
        self.assertGreaterEqual(equality, np.longdouble(2.0))
        self.assertLess(equality, np.longdouble(2.0 + 1e-12))

    def test_constraint_dual_is_tighter_than_the_cube(self):
        upper, receipt = self._upper(
            q=[1.0, 1.0],
            kappa=-0.2,
            A=[[1.0, 1.0]],
            rl=[-np.inf],
            ru=[0.0],
            lb=[-1.0, -1.0],
            ub=[1.0, 1.0],
            d=[1.0],
        )
        self.assertLess(upper, np.longdouble(-0.19))
        self.assertGreater(-0.2 + 2.0, 0.0)  # constraint-free cube is unsafe
        self.assertEqual(receipt["dual_nnz"], 1)

    def test_residual_cancellation_is_outward_not_naive_float(self):
        A = np.array([[1e16], [1.0], [-1e16]], dtype=np.float64)
        d = np.ones(3, dtype=np.float64)
        naive_residual = float((np.array([0.0]) - A.T @ d).item())
        self.assertEqual(naive_residual, 0.0)
        upper, receipt = self._upper(
            q=[0.0],
            kappa=0.0,
            A=A,
            rl=[0.0, 0.0, 0.0],
            ru=[0.0, 0.0, 0.0],
            lb=[-1.0],
            ub=[1.0],
            d=d,
        )
        # Exact stored-float A.T@d is one, so the residual box support is one.
        self.assertGreaterEqual(upper, np.longdouble(1.0))
        self.assertGreater(receipt["residual_guard"], 0.0)

    def test_objective_map_cancellation_cannot_false_safe(self):
        upper, receipt = _hz_independent_lp_lagrangian_upper(
            c=np.ones(3, dtype=np.float64),
            Gc=sp.csr_matrix((3, 0), dtype=np.float64),
            C_row=np.array([1e16, 1.0, -1e16], dtype=np.float64),
            threshold=0.5,
            A=sp.csr_matrix((0, 0), dtype=np.float64),
            rl=np.zeros(0, dtype=np.float64),
            ru=np.zeros(0, dtype=np.float64),
            lb=np.zeros(0, dtype=np.float64),
            ub=np.zeros(0, dtype=np.float64),
            row_dual=np.zeros(0, dtype=np.float64),
        )
        self.assertIsNotNone(upper, receipt)
        self.assertLess(
            float(
                np.array([1e16, 1.0, -1e16], dtype=np.float64)
                @ np.ones(3, dtype=np.float64)
                - 0.5
            ),
            0.0,
        )
        self.assertGreaterEqual(np.longdouble(upper), np.longdouble(0.5))

    def test_cutoff_tie_has_no_safe_authority(self):
        tol = 1e-9
        upper, _ = self._upper(
            q=[0.0],
            kappa=-tol,
            A=np.zeros((0, 1), dtype=np.float64),
            rl=[],
            ru=[],
            lb=[-1.0],
            ub=[1.0],
            d=[],
        )
        self.assertFalse(upper < -np.longdouble(tol))


class RivalCubePrefilterTests(unittest.TestCase):
    """Every skipped rival must be covered by an outward-guarded cube proof."""

    def _state(self) -> SparseHZono:
        hz = SparseHZono(
            c=np.array([0.0, 0.0], dtype=np.float64),
            Gc=sp.csr_matrix(
                np.array([[1.0], [0.5]], dtype=np.float64),
            ),
            Gb=sp.csr_matrix((2, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, 1), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            col_ids=np.array([901], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        return hz_mark_known_nonempty(hz, "cube_prefilter_test")

    def test_cube_bounds_cover_samples_and_only_prune_strict_safe_rows(self):
        hz = self._state()
        C = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=np.float64,
        )
        thresholds = np.array([2.0, 0.4, 1.1], dtype=np.float64)
        ub, guards = _hz_cube_row_upper_bounds(
            hz.c,
            hz.Gc,
            hz.Gb,
            C,
            thresholds,
        )
        for xi in np.linspace(-1.0, 1.0, 101):
            y = hz.c + np.asarray(hz.Gc @ np.array([xi])).reshape(-1)
            self.assertTrue(np.all(C @ y - thresholds <= ub))
        self.assertTrue(np.all(guards > 0.0))
        self.assertLess(ub[0], 0.0)
        self.assertGreater(ub[1], 0.0)
        self.assertLess(ub[2], 0.0)

    def test_all_pruned_rows_certify_without_losing_coverage(self):
        hz = self._state()
        verdict, witness = hz_objbound_decide(
            hz,
            np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float64),
            np.array([2.0, 1.1], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["cube_total_rows"], 2)
        self.assertEqual(stats["cube_pruned_rows"], 2)
        self.assertEqual(stats["cube_survivor_rows"], 0)
        self.assertTrue(stats["all_rivals_covered"])

    def test_persistent_lp_prunes_only_with_independent_certificate(self):
        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            # The output cube is [-1, 1], but the local graph constraint fixes
            # its sole factor to zero.  Cube cannot close y >= 0.1; LP can.
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([0.0], dtype=np.float64),
            col_ids=np.array([902], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_constructively_nonempty(hz, "persistent_lp_test")
        verdict, witness = hz_objbound_decide(
            hz,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=0.9,
            lp_prefilter_max_seconds=1.5,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["cube_pruned_rows"], 0)
        self.assertTrue(stats["lp_proof_authority"])
        self.assertEqual(stats["lp_candidate_empty_rows"], 1)
        self.assertEqual(stats["lp_pruned_rows"], 1)
        self.assertEqual(stats["lp_certified_rows"], 1)
        self.assertEqual(stats["lp_survivor_rows"], 0)
        self.assertTrue(stats["lp_coverage_ok"])
        self.assertTrue(stats["all_rivals_covered"])
        self.assertTrue(stats["lp_model_reused"])
        self.assertEqual(stats["base_feasibility_status"], "FEASIBLE")
        self.assertEqual(
            stats["base_feasibility_reason"],
            "constructive:persistent_lp_test",
        )

    @staticmethod
    def _tiny_coefficient_warning_state() -> SparseHZono:
        """A sound LP whose second row coefficient HiGHS must ignore."""

        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, 2), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix([[1.0, 1e-13]], dtype=np.float64),
            Aub=sp.csr_matrix((1, 0), dtype=np.float64),
            ub=np.array([0.0], dtype=np.float64),
            col_ids=np.array([920, 921], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        return hz_mark_constructively_nonempty(
            hz,
            "tiny_coefficient_warning_test",
        )

    def test_tiny_lp_coefficients_are_audited_and_certificate_uses_original(self):
        hz = self._tiny_coefficient_warning_state()
        verdict, witness = hz_objbound_decide(
            hz,
            np.array([[1.0]], dtype=np.float64),
            np.array([0.1], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            lp_prefilter_fraction=0.9,
            lp_prefilter_max_seconds=1.5,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertFalse(stats["lp_matrix_load_warning"])
        self.assertEqual(stats["lp_matrix_dropped_nnz"], 1)
        self.assertEqual(stats["lp_matrix_loaded_nnz"], 1)
        self.assertEqual(stats["lp_certified_rows"], 1)
        self.assertTrue(stats["lp_proof_authority"])

    def test_multi_rival_continuous_path_builds_one_persistent_highs_model(self):
        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([0.0], dtype=np.float64),
            col_ids=np.array([922], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_constructively_nonempty(hz, "persistent_multi_rival_test")
        real_highs = solver_hz_module._highspy.Highs
        constructions = []

        def counted_highs():
            constructions.append(1)
            return real_highs()

        with patch.object(
            solver_hz_module._highspy,
            "Highs",
            side_effect=counted_highs,
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.ones((9, 1), dtype=np.float64),
                np.linspace(0.1, 0.9, 9, dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        self.assertEqual(len(constructions), 1)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["lp_persistent_model_builds"], 1)
        self.assertEqual(stats["lp_completed_rows"], 9)
        self.assertEqual(stats["lp_certified_rows"], 9)
        self.assertFalse(
            stats.get("continuous_cutoff_fallback_skipped", False)
        )

    def test_time_limited_dual_has_no_status_authority_but_can_be_certified(self):
        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([0.0], dtype=np.float64),
            col_ids=np.array([923], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_constructively_nonempty(hz, "time_limited_dual_test")
        HS = solver_hz_module._highspy.HighsStatus
        MS = solver_hz_module._highspy.HighsModelStatus

        class TimeLimitedHighs:
            def __init__(self):
                self.ncol = 0
                self.nrow = 0
                self.nnz = 0

            def setOptionValue(self, *_args):
                return HS.kOk

            def addCols(self, ncol, *_args):
                self.ncol = int(ncol)
                return HS.kOk

            def addRows(self, nrow, _rl, _ru, nnz, *_args):
                self.nrow = int(nrow)
                self.nnz = int(nnz)
                return HS.kOk

            def getNumCol(self):
                return self.ncol

            def getNumRow(self):
                return self.nrow

            def getNumNz(self):
                return self.nnz

            def changeColsCost(self, *_args):
                return HS.kOk

            def run(self):
                return HS.kWarning

            def getModelStatus(self):
                return MS.kTimeLimit

            def getSolution(self):
                # For min -x with x==0, row_dual=-1 gives the exact
                # maximization multiplier d=1.  The kTimeLimit status itself
                # remains untrusted.
                return SimpleNamespace(
                    col_value=[0.0],
                    row_dual=[-1.0],
                )

        with patch.object(
            solver_hz_module._highspy,
            "Highs",
            TimeLimitedHighs,
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([0.1], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["lp_certificate_attempted_rows"], 1)
        self.assertEqual(stats["lp_certified_rows"], 1)
        self.assertEqual(
            stats["lp_solver_run_status_histogram"],
            {"HighsStatus.kWarning": 1},
        )
        self.assertEqual(
            stats["lp_model_status_histogram"],
            {"HighsModelStatus.kTimeLimit": 1},
        )

    def test_persistent_tiny_matrix_can_only_return_validated_witness(self):
        hz = self._tiny_coefficient_warning_state()
        with patch.dict(
            os.environ,
            {
                "HZ_MILP_BACKEND": "highs",
                "HZ_MILP_THREADS": "1",
                "HZ_LP_PREFILTER_THREADS": "1",
                "HZ_MILP_EQ_SUBST": "0",
                "HZ_MILP_ELIM_SINGLETONS": "0",
                "HZ_MILP_SCALE": "0",
            },
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([-0.1], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
            )
        self.assertEqual(verdict, "UNSAFE")
        self.assertIsNotNone(witness)
        # Recheck the returned factor point against the original, unmodified
        # tiny-coefficient row rather than HiGHS' dropped-coefficient model.
        self.assertLessEqual(
            float(witness[0] + 1e-13 * witness[1]),
            0.0,
        )


class GpuDualProductionIntegrationTests(unittest.TestCase):
    """CUDA duals are candidates; only original-frame checks may delete rows."""

    @staticmethod
    def _fixed_zero_state(n_factors: int) -> SparseHZono:
        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix(
                np.ones((1, n_factors), dtype=np.float64),
            ),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.eye(n_factors, format="csr", dtype=np.float64),
            Ab=sp.csr_matrix((n_factors, 0), dtype=np.float64),
            b=np.zeros(n_factors, dtype=np.float64),
            col_ids=np.arange(930, 930 + n_factors, dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        return hz_mark_constructively_nonempty(hz, "gpu_dual_fixed_zero")

    @staticmethod
    def _fixed_negative_binary_state() -> SparseHZono:
        """Return ``y=s`` with the exact binary constraint ``s=-1``."""

        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ac=sp.csr_matrix((1, 0), dtype=np.float64),
            Ab=sp.csr_matrix([[1.0]], dtype=np.float64),
            b=np.array([-1.0], dtype=np.float64),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.array([1842], dtype=np.int64),
        )
        return hz_mark_constructively_nonempty(
            hz, "gpu_dual_fixed_negative_binary"
        )

    @staticmethod
    def _candidate(row_dual, *, deadline_reached=False):
        dual = np.asarray(row_dual, dtype=np.float64)
        n_rivals = int(dual.shape[0])
        return SimpleNamespace(
            row_dual=dual,
            initial_support=np.full(n_rivals, 1.0, dtype=np.float64),
            candidate_support=np.zeros(n_rivals, dtype=np.float64),
            device="cuda:0",
            steps_completed=3,
            deadline_reached=bool(deadline_reached),
        )

    @staticmethod
    def _retain_all_lp(**kwargs):
        rows = np.asarray(kwargs["candidate_rows"], dtype=np.int64)
        return (
            rows,
            {
                "lp_coverage_ok": True,
                "lp_base_feasibility_conflict": False,
                "lp_certified_rows": 0,
                "lp_survivor_rows": int(rows.size),
                "lp_uncertified_rows": int(rows.size),
                "lp_status": "test_retain_all",
            },
            None,
        )

    def test_gpu_path_is_strictly_default_off(self):
        hz = self._fixed_zero_state(1)
        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "batched_original_frame_row_duals",
            side_effect=AssertionError(
                "default GPU settings must not launch the candidate helper"
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([0.1], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertFalse(stats["gpu_dual_enabled"])
        self.assertEqual(stats["gpu_dual_status"], "disabled")
        self.assertEqual(stats["gpu_dual_steps_requested"], 0)

    def test_verified_gpu_candidate_can_end_to_end_certify_without_highs(self):
        hz = self._fixed_zero_state(1)
        candidate = self._candidate([[-1.0]])
        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "batched_original_frame_row_duals",
                return_value=candidate,
            ),
            patch.object(
                solver_hz_module._highspy,
                "Highs",
                side_effect=AssertionError(
                    "persistent HiGHS must be skipped after complete GPU proof"
                ),
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([0.1], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["gpu_dual_status"], "completed")
        self.assertEqual(stats["gpu_dual_certificate_attempted_rows"], 1)
        self.assertEqual(stats["gpu_dual_certified_rows"], 1)
        self.assertEqual(stats["gpu_dual_checked_dual_nnz_total"], 1)
        self.assertTrue(stats["gpu_dual_proof_authority"])
        self.assertTrue(stats["gpu_dual_coverage_ok"])
        self.assertTrue(stats["all_rivals_covered"])
        self.assertEqual(stats["lp_status"], "skipped_all_gpu_certified")
        self.assertEqual(stats["lp_persistent_model_builds"], 0)

    def test_gpu_budget_is_independent_of_zero_lp_slice(self):
        hz = self._fixed_zero_state(1)
        candidate = self._candidate([[-1.0]])
        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "batched_original_frame_row_duals",
                return_value=candidate,
            ),
            patch.object(
                solver_hz_module._highspy,
                "Highs",
                side_effect=AssertionError(
                    "LP=0 must not suppress GPU or construct persistent HiGHS"
                ),
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([0.1], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertTrue(stats["gpu_dual_enabled"])
        self.assertEqual(stats["gpu_dual_status"], "completed")
        self.assertEqual(stats["gpu_dual_steps_completed"], 3)
        self.assertEqual(stats["gpu_dual_certified_rows"], 1)
        self.assertEqual(stats["lp_status"], "skipped_all_gpu_certified")

    def test_signed_binary_relaxation_candidate_is_rechecked_end_to_end(
        self,
    ):
        hz = self._fixed_negative_binary_state()
        captured = {}

        def candidate(frame, q, **_kwargs):
            captured["A"] = frame.A.copy()
            captured["rl"] = np.asarray(frame.rl).copy()
            captured["ru"] = np.asarray(frame.ru).copy()
            captured["lb"] = np.asarray(frame.lb).copy()
            captured["ub"] = np.asarray(frame.ub).copy()
            captured["q"] = np.asarray(q).copy()
            return self._candidate([[-1.0]])

        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "batched_original_frame_row_duals",
                side_effect=candidate,
            ),
            patch.object(
                solver_hz_module._highspy,
                "Highs",
                side_effect=AssertionError(
                    "independently certified binary GPU row must skip HiGHS"
                ),
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([-0.5], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
                safe_row_groups=((0,),),
                expected_safe_group_count=1,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        np.testing.assert_array_equal(
            captured["A"].toarray(), np.array([[2.0]])
        )
        np.testing.assert_array_equal(captured["rl"], np.array([0.0]))
        np.testing.assert_array_equal(captured["ru"], np.array([0.0]))
        np.testing.assert_array_equal(captured["lb"], np.array([0.0]))
        np.testing.assert_array_equal(captured["ub"], np.array([1.0]))
        np.testing.assert_array_equal(captured["q"], np.array([[2.0]]))
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertTrue(stats["gpu_dual_binary_relaxation_enabled"])
        self.assertEqual(stats["gpu_dual_binary_factor_count"], 1)
        self.assertFalse(stats["gpu_dual_candidate_witness_eligible"])
        self.assertEqual(stats["gpu_dual_certified_rows"], 1)
        self.assertTrue(stats["gpu_dual_proof_authority"])
        self.assertGreaterEqual(
            stats["gpu_dual_cert_center_transform_guard_max"], 0.0
        )
        self.assertEqual(
            stats["lp_status"],
            "skipped_all_property_groups_certified",
        )
        capability = hz_objbound_safe_capability_receipt(
            hz,
            np.array([[1.0]], dtype=np.float64),
            np.array([-0.5], dtype=np.float64),
            is_unsafe_linear=False,
            tol=1e-9,
            require_base_feasible=True,
            base_witness_precheck=False,
            safe_row_groups=((0,),),
            expected_safe_group_count=1,
            require_binary_relaxation_lp=True,
        )
        self.assertIsNotNone(capability)
        self.assertEqual(
            capability["proof_stage"], "gpu_dual_lagrangian"
        )

    def test_signed_binary_zero_candidate_cannot_false_prove(self):
        hz = self._fixed_negative_binary_state()
        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "batched_original_frame_row_duals",
                return_value=self._candidate([[0.0]]),
            ),
            patch.object(
                solver_hz_module,
                "_hz_persistent_lp_filter",
                side_effect=self._retain_all_lp,
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([-0.5], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
            )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(stats["gpu_dual_certified_rows"], 0)
        self.assertEqual(stats["gpu_dual_uncertified_rows"], 1)
        self.assertTrue(stats["gpu_dual_coverage_ok"])
        self.assertFalse(stats["gpu_dual_candidate_witness_eligible"])

    def test_binary_micro_rlt_gpu_schedules_one_objective_and_generated_rows(
        self,
    ):
        hz = SparseHZono(
            c=np.zeros(2, dtype=np.float64),
            Gc=sp.csr_matrix((2, 0), dtype=np.float64),
            Gb=sp.csr_matrix(np.ones((2, 1), dtype=np.float64)),
            Ac=sp.csr_matrix((1, 0), dtype=np.float64),
            Ab=sp.csr_matrix([[1.0]], dtype=np.float64),
            b=np.array([-1.0], dtype=np.float64),
            Auc=sp.csr_matrix((1, 0), dtype=np.float64),
            Aub=sp.csr_matrix((1, 1), dtype=np.float64),
            ub=np.array([1.0], dtype=np.float64),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.array([2842], dtype=np.int64),
        )
        hz_mark_constructively_nonempty(
            hz, "gpu_dual_binary_micro_rlt_schedule"
        )
        hz._solver_constraint_row_tags = (
            "base_binary_fix",
            "property_micro_rlt:generated:toy",
        )
        hz.operator_hz_metadata = {
            "property_micro_rlt": {
                "status": "applied",
                "common_focused_rival_id": 1,
            },
            "property_tail_upper": {
                "baseline_plane_count": 2,
                "property_row_groups": [[0], [1]],
            },
        }
        captured = {"postprocess_order": []}
        independent_checker = (
            solver_hz_module._hz_independent_lp_lagrangian_upper
        )
        support_attribution = (
            solver_hz_module._hz_candidate_support_attribution
        )

        def wavefront(frame, q, **_kwargs):
            captured["q_shape"] = np.asarray(q).shape
            captured["candidate_rows"] = np.asarray(
                frame.row_tags, dtype=np.int64
            )
            return SimpleNamespace(
                d=np.zeros((1, 1), dtype=np.float64),
                initial_support=np.ones(1, dtype=np.float64),
                candidate_support=np.ones(1, dtype=np.float64),
                updates=0,
                selected_constraint_count=0,
                elapsed_seconds=0.0,
                deadline_reached=False,
                method="test_packet_core_wavefront",
            )

        def checked_upper(*args, **kwargs):
            captured["postprocess_order"].append("independent_checker")
            return independent_checker(*args, **kwargs)

        def attribution(*args, **kwargs):
            captured["postprocess_order"].append("attribution")
            return support_attribution(*args, **kwargs)

        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "batched_original_frame_row_duals",
                side_effect=AssertionError(
                    "restricted packet core must not launch CUDA Adam"
                ),
            ),
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "property_conditioned_coordinate_wavefront_duals",
                side_effect=wavefront,
            ),
            patch.object(
                solver_hz_module,
                "_hz_persistent_lp_filter",
                side_effect=self._retain_all_lp,
            ),
            patch.object(
                solver_hz_module,
                "_hz_constraint_generation_dual_candidate",
                side_effect=AssertionError(
                    "restricted packet-core scope must skip full-frame "
                    "constraint generation"
                ),
            ),
            patch.object(
                solver_hz_module,
                "_hz_independent_lp_lagrangian_upper",
                side_effect=checked_upper,
            ),
            patch.object(
                solver_hz_module,
                "_hz_candidate_support_attribution",
                side_effect=attribution,
            ),
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.eye(2, dtype=np.float64),
                np.array([-0.5, -0.5], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
                safe_row_groups=((0,), (1,)),
                expected_safe_group_count=2,
            )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        self.assertEqual(captured["q_shape"], (1, 1))
        np.testing.assert_array_equal(
            captured["candidate_rows"], np.array([1], dtype=np.int64)
        )
        self.assertEqual(
            captured["postprocess_order"],
            ["independent_checker", "attribution"],
        )
        stats = getattr(hz, "_solver_objbound_stats")
        self.assertEqual(
            stats["gpu_dual_objective_scope"],
            "property_micro_rlt_focused_rival_baseline_plane",
        )
        self.assertEqual(stats["gpu_dual_objective_focus_rival_id"], 1)
        self.assertEqual(stats["gpu_dual_first_scheduled_objective_row"], 1)
        self.assertEqual(
            stats["gpu_dual_objective_focus_plane_kind"],
            "baseline_property_plane",
        )
        self.assertTrue(
            stats["gpu_dual_objective_focus_mapping_valid"]
        )
        self.assertEqual(stats["gpu_dual_total_input_rows"], 2)
        self.assertEqual(stats["gpu_dual_objective_rows_scheduled"], 1)
        self.assertEqual(stats["gpu_dual_objective_rows_deferred"], 1)
        self.assertEqual(
            stats["gpu_dual_candidate_constraint_scope"],
            "property_micro_rlt_generated_rows",
        )
        self.assertEqual(stats["gpu_dual_device"], "cpu_packet_core")
        self.assertTrue(stats["gpu_dual_packet_core_cpu_fallback"])
        self.assertEqual(
            stats["gpu_dual_candidate_constraint_rows_selected"], 1
        )
        self.assertEqual(
            stats["gpu_dual_candidate_constraint_rows_deferred"], 1
        )
        self.assertEqual(
            stats["gpu_dual_constraint_generation_status"],
            "skipped_restricted_constraint_scope",
        )
        self.assertEqual(stats["gpu_dual_uncertified_rows"], 2)
        self.assertTrue(stats["gpu_dual_coverage_ok"])

    def test_bad_and_zero_candidates_retain_every_rival(self):
        for label, candidate_or_error in (
            (
                "bad_shape",
                self._candidate(
                    [[-1.0, -1.0], [-1.0, -1.0]],
                ),
            ),
            (
                "zero",
                self._candidate([[0.0], [0.0]]),
            ),
            (
                "cuda_unavailable",
                RuntimeError(
                    "CUDA dual candidates requested but CUDA is unavailable"
                ),
            ),
        ):
            with self.subTest(label=label):
                hz = self._fixed_zero_state(1)
                patch_kwargs = (
                    {"side_effect": candidate_or_error}
                    if isinstance(candidate_or_error, BaseException)
                    else {"return_value": candidate_or_error}
                )
                with (
                    patch(
                        "act.back_end.hybridz_tf.gpu_dual_candidates."
                        "batched_original_frame_row_duals",
                        **patch_kwargs,
                    ),
                    patch.object(
                        solver_hz_module,
                        "_hz_persistent_lp_filter",
                        side_effect=self._retain_all_lp,
                    ),
                ):
                    verdict, witness = hz_objbound_decide(
                        hz,
                        np.ones((2, 1), dtype=np.float64),
                        np.array([0.1, 0.2], dtype=np.float64),
                        is_unsafe_linear=False,
                        time_limit=2.0,
                        base_witness_precheck=False,
                        lp_prefilter_fraction=0.9,
                        lp_prefilter_max_seconds=1.5,
                        gpu_dual_steps=3,
                        gpu_dual_time_limit=1.0,
                    )
                self.assertEqual(verdict, "UNKNOWN")
                self.assertIsNone(witness)
                stats = getattr(hz, "_solver_objbound_stats")
                self.assertEqual(stats["gpu_dual_certified_rows"], 0)
                self.assertEqual(stats["gpu_dual_uncertified_rows"], 2)
                self.assertTrue(stats["gpu_dual_coverage_ok"])
                self.assertTrue(stats["all_rivals_covered"])
                if label == "zero":
                    self.assertEqual(
                        stats["gpu_dual_certificate_attempted_rows"],
                        2,
                    )
                    self.assertEqual(stats["gpu_dual_errors"], 0)
                else:
                    self.assertEqual(stats["gpu_dual_errors"], 1)
                if label == "cuda_unavailable":
                    self.assertEqual(
                        stats["gpu_dual_status"],
                        "cuda_unavailable",
                    )
                    self.assertEqual(
                        stats["gpu_dual_error_stage"],
                        "candidate_generation",
                    )
                    self.assertIn(
                        "CUDA dual candidates requested",
                        stats["gpu_dual_error_message"],
                    )

    def test_expired_absolute_deadline_does_not_launch_candidate(self):
        hz = self._fixed_zero_state(1)
        c, Gc, Gb, Ace, Abe, be, Acl, Abl, bl = (
            solver_hz_module.hz_np_sparse(hz)
        )
        A, rl, ru, lb, ub, _ = _base_milp_matrices_from_blocks(
            Gc, Gb, Ace, Abe, be, Acl, Abl, bl
        )
        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "batched_original_frame_row_duals",
            side_effect=AssertionError("expired deadline must not launch CUDA"),
        ):
            rows, stats = _hz_gpu_dual_candidate_filter(
                c=c,
                Gc=Gc,
                Gb=Gb,
                C=np.array([[1.0]], dtype=np.float64),
                t=np.array([0.1], dtype=np.float64),
                candidate_rows=np.array([0], dtype=np.int64),
                A=A,
                rl=rl,
                ru=ru,
                lb=lb,
                ub=ub,
                deadline=time.monotonic() - 1.0,
                time_budget=1.0,
                steps=3,
                row_topk=0,
                learning_rate=0.08,
                tol=1e-9,
            )
        np.testing.assert_array_equal(rows, np.array([0], dtype=np.int64))
        self.assertEqual(stats["gpu_dual_status"], "no_budget")
        self.assertTrue(stats["gpu_dual_deadline_reached"])
        self.assertEqual(stats["gpu_dual_certified_rows"], 0)

    def test_topk_is_an_explicit_weakening_and_cannot_false_prove(self):
        candidate = self._candidate(
            [
                [-1.0, -1.0],
                [-1.0, -1.0],
            ]
        )
        full = self._fixed_zero_state(2)
        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "batched_original_frame_row_duals",
            return_value=candidate,
        ):
            full_verdict, _ = hz_objbound_decide(
                full,
                np.ones((2, 1), dtype=np.float64),
                np.array([0.1, 0.2], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
                gpu_dual_row_topk=0,
            )
        self.assertEqual(full_verdict, "SAFE")
        self.assertEqual(
            getattr(full, "_solver_objbound_stats")[
                "gpu_dual_certified_rows"
            ],
            2,
        )

        sparse = self._fixed_zero_state(2)
        with (
            patch(
                "act.back_end.hybridz_tf.gpu_dual_candidates."
                "batched_original_frame_row_duals",
                return_value=candidate,
            ),
            patch.object(
                solver_hz_module,
                "_hz_persistent_lp_filter",
                side_effect=self._retain_all_lp,
            ),
        ):
            sparse_verdict, _ = hz_objbound_decide(
                sparse,
                np.ones((2, 1), dtype=np.float64),
                np.array([0.1, 0.2], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.9,
                lp_prefilter_max_seconds=1.5,
                gpu_dual_steps=3,
                gpu_dual_time_limit=1.0,
                gpu_dual_row_topk=1,
            )
        self.assertEqual(sparse_verdict, "UNKNOWN")
        stats = getattr(sparse, "_solver_objbound_stats")
        self.assertEqual(stats["gpu_dual_candidate_dual_nnz_total"], 4)
        self.assertEqual(stats["gpu_dual_checked_dual_nnz_total"], 2)
        self.assertEqual(stats["gpu_dual_certified_rows"], 0)
        self.assertEqual(stats["gpu_dual_uncertified_rows"], 2)
        self.assertTrue(stats["all_rivals_covered"])


class CertifiedPreactivationTighteningTests(unittest.TestCase):
    """Controlled residual gates for proof-carrying ReLU pre-bounds."""

    @staticmethod
    def _residual_copy_toy():
        """Return ``p=x-(x+0)+1/4`` followed by ReLU.

        The first ADD is deliberately materialized in a fresh local frame.
        Ignoring its equality band makes the second residual expression range
        over ``[-3/4, 5/4]``.  The original constraints fix it to ``1/4``.
        Consequently the loose triangle admits ``5/8`` whereas the real graph
        and the certified active phase have maximum ``1/4``.
        """

        def layer(lid, kind, params):
            return SimpleNamespace(
                id=lid,
                kind=kind,
                params=params,
                out_vars=[lid],
                in_vars=[],
            )

        dtype = torch.float64
        zero = torch.zeros(1, dtype=dtype)
        half = torch.tensor([0.5], dtype=dtype)
        lower = -half
        layers = [
            layer(0, "INPUT", {"shape": (1, 1)}),
            layer(
                1,
                "INPUT_SPEC",
                {
                    "kind": "BOX",
                    "lb": lower.reshape(1, 1),
                    "ub": half.reshape(1, 1),
                },
            ),
            layer(
                2,
                "DENSE",
                {
                    "weight": torch.zeros((1, 1), dtype=dtype),
                    "bias": zero,
                    "in_features": 1,
                    "out_features": 1,
                },
            ),
            layer(3, "ADD", {}),
            layer(
                4,
                "DENSE",
                {
                    "weight": -torch.ones((1, 1), dtype=dtype),
                    "bias": zero,
                    "in_features": 1,
                    "out_features": 1,
                },
            ),
            layer(5, "ADD", {}),
            layer(
                6,
                "DENSE",
                {
                    "weight": torch.ones((1, 1), dtype=dtype),
                    "bias": torch.tensor([0.25], dtype=dtype),
                    "in_features": 1,
                    "out_features": 1,
                },
            ),
            layer(7, "RELU", {}),
            layer(8, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        preds = {
            0: [],
            1: [0],
            2: [1],
            3: [1, 2],
            4: [3],
            5: [1, 4],
            6: [5],
            7: [6],
            8: [7],
        }
        succs = {lid: [] for lid in preds}
        for child, parents in preds.items():
            for parent in parents:
                succs[parent].append(child)
        net = SimpleNamespace(
            layers=layers,
            preds=preds,
            succs=succs,
            by_id={item.id: item for item in layers},
        )
        facts = {}
        for item in layers:
            if item.id in {0, 1}:
                lb = lower.reshape(1, 1)
                ub = half.reshape(1, 1)
            else:
                # These facts are deliberately uninformative and audit-only.
                lb = torch.tensor([[-10.0]], dtype=dtype)
                ub = torch.tensor([[10.0]], dtype=dtype)
            facts[item.id] = Fact(
                Bounds(lb.clone(), ub.clone()),
                ConSet(),
            )
        return net, facts

    @staticmethod
    def _continuous_support_upper(hz: SparseHZono) -> float:
        objective = np.concatenate(
            (
                hz.Gc.getrow(0).toarray().reshape(-1),
                hz.Gb.getrow(0).toarray().reshape(-1),
            )
        )
        A_ub = sp.hstack((hz.Auc, hz.Aub), format="csr")
        A_eq = sp.hstack((hz.Ac, hz.Ab), format="csr")
        result = linprog(
            -objective,
            A_ub=A_ub if A_ub.shape[0] else None,
            b_ub=hz.ub if A_ub.shape[0] else None,
            A_eq=A_eq if A_eq.shape[0] else None,
            b_eq=hz.b if A_eq.shape[0] else None,
            bounds=(-1.0, 1.0),
            method="highs",
        )
        if not result.success:
            raise AssertionError(f"toy support LP failed: {result.message}")
        return float(hz.c[0] - result.fun)

    def test_residual_cube_125_triangle_0625_tightens_to_true_025(self):
        net, facts = self._residual_copy_toy()
        loose = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=0,
            materialize_add=True,
        )
        certified = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=0,
            materialize_add=True,
            preactivation_lp_budget=1,
            preactivation_lp_time_limit=2.0,
        )
        loose_relu = next(
            item for item in loose.metadata["layers"]
            if item["layer_id"] == 7
        )
        certified_relu = next(
            item for item in certified.metadata["layers"]
            if item["layer_id"] == 7
        )

        self.assertAlmostEqual(
            loose_relu["preactivation_cube_lb_min"], -0.75, places=10
        )
        self.assertAlmostEqual(
            loose_relu["preactivation_cube_ub_max"], 1.25, places=10
        )
        self.assertEqual(loose_relu["relu_relaxed"], 1)
        self.assertAlmostEqual(
            self._continuous_support_upper(loose.hz), 0.625, places=10
        )

        audit = certified_relu["preactivation_constrained_lp"]
        self.assertTrue(audit["proof_authority"])
        self.assertFalse(audit["candidate_solver_authority"])
        self.assertEqual(audit["directions_certified"], 2)
        self.assertEqual(audit["rows_tightened"], 1)
        self.assertEqual(audit["stabilized_active"], 1)
        self.assertEqual(certified_relu["relu_active"], 1)
        self.assertEqual(certified_relu["relu_relaxed"], 0)
        self.assertAlmostEqual(
            certified_relu["preactivation_certified_lb_min"],
            0.25,
            places=10,
        )
        self.assertAlmostEqual(
            certified_relu["preactivation_certified_ub_max"],
            0.25,
            places=10,
        )
        self.assertAlmostEqual(
            self._continuous_support_upper(certified.hz), 0.25, places=10
        )

        # Independent exact-dyadic explanation of the 5/8 loose value.
        slope, intercept, _ = _relu_triangle_parameters(
            np.asarray([-0.75]), np.asarray([1.25])
        )
        exact_line = (
            Fraction.from_float(float(slope[0])) * Fraction(1, 4)
            + Fraction.from_float(float(intercept[0]))
        )
        self.assertEqual(exact_line, Fraction(5, 8))

    @staticmethod
    def _equality_base(A, rhs) -> _PreactivationLPBase:
        matrix = sp.csr_matrix(np.asarray(A, dtype=np.float64))
        vector = np.asarray(rhs, dtype=np.float64).reshape(-1)
        n_var = int(matrix.shape[1])
        return _PreactivationLPBase(
            A=matrix,
            rl=vector.copy(),
            ru=vector.copy(),
            lb=np.full(n_var, -1.0, dtype=np.float64),
            ub=np.full(n_var, 1.0, dtype=np.float64),
            n_eq=int(vector.size),
            n_ub=0,
        )

    def test_fraction_phase_tie_and_cancellation_certificates(self):
        # Fraction oracle for the normalized duplicate-frame toy:
        # p = 1/4 + (xi_0-xi_1)/2 and xi_0 == xi_1.
        duplicate = _AffineExpr(
            c=np.asarray([0.25], dtype=np.float64),
            G=sp.csr_matrix([[0.5, -0.5]], dtype=np.float64),
            err=np.zeros(1, dtype=np.float64),
        )
        duplicate_base = self._equality_base([[1.0, -1.0]], [0.0])
        upper, upper_receipt = (
            _independent_preactivation_lagrangian_upper(
                duplicate,
                0,
                sign=1.0,
                base=duplicate_base,
                row_dual=np.asarray([-0.5]),
            )
        )
        neg_lower, lower_receipt = (
            _independent_preactivation_lagrangian_upper(
                duplicate,
                0,
                sign=-1.0,
                base=duplicate_base,
                row_dual=np.asarray([0.5]),
            )
        )
        self.assertTrue(upper_receipt["proof_authority"])
        self.assertTrue(lower_receipt["proof_authority"])
        exact_quarter = Fraction(1, 4)
        self.assertLessEqual(
            Fraction.from_float(float(-neg_lower)), exact_quarter
        )
        self.assertGreaterEqual(
            Fraction.from_float(float(upper)), exact_quarter
        )

        # At the phase tie x=0, both certified directions must continue to
        # enclose zero.  Positive/negative offsets must select only the phase
        # admitted by the independent concrete oracle.
        tie_base = self._equality_base([[1.0]], [0.0])
        phases = []
        for offset in (-0.125, 0.0, 0.125):
            expr = _AffineExpr(
                c=np.asarray([offset], dtype=np.float64),
                G=sp.csr_matrix([[1.0]], dtype=np.float64),
                err=np.zeros(1, dtype=np.float64),
            )
            hi, _ = _independent_preactivation_lagrangian_upper(
                expr,
                0,
                sign=1.0,
                base=tie_base,
                row_dual=np.asarray([-1.0]),
            )
            neg_lo, _ = _independent_preactivation_lagrangian_upper(
                expr,
                0,
                sign=-1.0,
                base=tie_base,
                row_dual=np.asarray([1.0]),
            )
            lo = -float(neg_lo)
            self.assertLessEqual(lo, offset)
            self.assertGreaterEqual(float(hi), offset)
            inactive = float(hi) <= 0.0
            active = lo >= 0.0 and float(hi) > 0.0
            unstable = lo < 0.0 and float(hi) > 0.0
            self.assertEqual(int(inactive) + int(active) + int(unstable), 1)
            phases.append(
                "inactive" if inactive else "active" if active else "unstable"
            )
        self.assertEqual(phases[0], "inactive")
        self.assertEqual(phases[1], "unstable")
        self.assertEqual(phases[2], "active")

        # A cancellation-heavy exact-real row is fixed by xi=(1,1,1).
        # Long-double guards may weaken the answer, but can never cut out the
        # Fraction value 10^12 + 1 - 10^12 = 1.
        scale = 1.0e12
        cancellation = _AffineExpr(
            c=np.zeros(1, dtype=np.float64),
            G=sp.csr_matrix([[scale, 1.0, -scale]], dtype=np.float64),
            err=np.zeros(1, dtype=np.float64),
        )
        cancellation_base = self._equality_base(np.eye(3), np.ones(3))
        q = np.asarray([scale, 1.0, -scale], dtype=np.float64)
        cancel_hi, _ = _independent_preactivation_lagrangian_upper(
            cancellation,
            0,
            sign=1.0,
            base=cancellation_base,
            row_dual=-q,
        )
        cancel_neg_lo, _ = _independent_preactivation_lagrangian_upper(
            cancellation,
            0,
            sign=-1.0,
            base=cancellation_base,
            row_dual=q,
        )
        exact_cancel = (
            Fraction.from_float(scale)
            + Fraction.from_float(1.0)
            - Fraction.from_float(scale)
        )
        self.assertLessEqual(
            Fraction.from_float(float(-cancel_neg_lo)), exact_cancel
        )
        self.assertGreaterEqual(
            Fraction.from_float(float(cancel_hi)), exact_cancel
        )

        # A wrong-sign multiplier for x<=0 is projected to zero.  The result
        # may lose the useful x<=0 tightening, but it must fall back to the
        # enclosing cube instead of manufacturing a false upper bound.
        one_sided_base = _PreactivationLPBase(
            A=sp.csr_matrix([[1.0]], dtype=np.float64),
            rl=np.asarray([-np.inf]),
            ru=np.asarray([0.0]),
            lb=np.asarray([-1.0]),
            ub=np.asarray([1.0]),
            n_eq=0,
            n_ub=1,
        )
        identity = _AffineExpr(
            c=np.zeros(1, dtype=np.float64),
            G=sp.csr_matrix([[1.0]], dtype=np.float64),
            err=np.zeros(1, dtype=np.float64),
        )
        wrong_sign, wrong_receipt = (
            _independent_preactivation_lagrangian_upper(
                identity,
                0,
                sign=1.0,
                base=one_sided_base,
                # raw=+1 becomes d=-1, illegal for an upper-only row.
                row_dual=np.asarray([1.0]),
            )
        )
        self.assertEqual(
            wrong_receipt["certificate"]["illegal_sign_projected"], 1
        )
        self.assertGreaterEqual(float(wrong_sign), 1.0)


class NumericalSoundnessRegressionTests(unittest.TestCase):
    """Cancellation, invalid tolerance, and vacuous-base proof blockers."""

    def test_operator_roundoff_envelopes_cover_fraction_oracles(self):
        receipt = operator_hz_self_test()
        self.assertTrue(receipt["ok"])
        self.assertTrue(
            receipt["audits"]["dense_cancellation_fraction_enclosed"]
        )
        self.assertTrue(
            receipt["audits"]["add_cancellation_fraction_enclosed"]
        )

    @staticmethod
    def _point_state() -> SparseHZono:
        return SparseHZono(
            c=np.ones(3, dtype=np.float64),
            Gc=sp.csr_matrix((3, 0), dtype=np.float64),
            Gb=sp.csr_matrix((3, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, 0), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )

    @staticmethod
    def _dummy_generator_state() -> SparseHZono:
        return SparseHZono(
            c=np.ones(3, dtype=np.float64),
            Gc=sp.csr_matrix((3, 1), dtype=np.float64),
            Gb=sp.csr_matrix((3, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, 1), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            col_ids=np.array([903], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )

    def test_point_cancellation_cannot_false_safe_or_or_and(self):
        # Binary64 BLAS commonly rounds both dot products below to zero, but
        # the exact-real sums of the stored floats are +1 and -1.
        or_c = np.array([[1e16, 1.0, -1e16]], dtype=np.float64)
        and_c = np.array([[1e16, -1.0, -1e16]], dtype=np.float64)
        self.assertEqual(
            hz_objbound_decide(
                self._point_state(),
                or_c,
                np.array([0.5], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
            )[0],
            "UNSAFE",
        )
        self.assertEqual(
            hz_objbound_decide(
                self._point_state(),
                and_c,
                np.array([-0.5], dtype=np.float64),
                is_unsafe_linear=True,
                time_limit=2.0,
            )[0],
            "UNSAFE",
        )

    def test_cutoff_cancellation_is_outward_widened_for_or_and(self):
        or_c = np.array([[1e16, 1.0, -1e16]], dtype=np.float64)
        and_c = np.array([[1e16, -1.0, -1e16]], dtype=np.float64)
        for C, threshold, unsafe_linear in (
            (or_c, 0.5, False),
            (and_c, -0.5, True),
        ):
            verdict, witness = hz_objbound_decide(
                self._dummy_generator_state(),
                C,
                np.array([threshold], dtype=np.float64),
                is_unsafe_linear=unsafe_linear,
                time_limit=2.0,
            )
            self.assertEqual(verdict, "UNSAFE")
            self.assertIsNotNone(witness)

    def test_negative_tolerance_and_contradictory_base_fail_closed(self):
        verdict, _ = hz_objbound_decide(
            self._dummy_generator_state(),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
            tol=-1.0,
        )
        self.assertEqual(verdict, "UNKNOWN")

        contradictory = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix((1, 0), dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([1.0], dtype=np.float64),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_known_nonempty(contradictory, "intentionally_false_marker")
        verdict, _ = hz_objbound_decide(
            contradictory,
            np.array([[1.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=2.0,
        )
        self.assertEqual(verdict, "UNKNOWN")

    def test_tiny_constant_row_contradiction_cannot_authorize_safe(self):
        # Exact stored-float semantics are 0 == 1e-12, hence the base HZ is
        # empty.  The historical +/-1e-9 feasibility tolerance incorrectly
        # authorized this state and the point fast path returned SAFE.
        contradictory = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix((1, 0), dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([1e-12], dtype=np.float64),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_known_nonempty(
            contradictory,
            "intentionally_false_tiny_constant_marker",
        )
        base_status, _ = hz_base_feasibility(
            contradictory,
            time_limit=1.0,
        )
        self.assertEqual(base_status, "INFEASIBLE")
        verdict, _ = hz_objbound_decide(
            contradictory,
            np.array([[1.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
        )
        self.assertEqual(verdict, "UNKNOWN")

    def test_near_bound_equality_cannot_authorize_safe(self):
        # x is constrained to [-1,1] by the HZ factor box but the stored
        # equality requires x=1+1e-8.  A tolerance-validated HiGHS incumbent
        # used to turn this mathematically empty state into FEASIBLE.
        contradictory = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[0.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([1.0 + 1e-8], dtype=np.float64),
            col_ids=np.array([904], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_known_nonempty(
            contradictory,
            "intentionally_false_near_bound_marker",
        )
        base_status, _ = hz_base_feasibility(
            contradictory,
            time_limit=1.0,
        )
        self.assertNotEqual(base_status, "FEASIBLE")
        verdict, _ = hz_objbound_decide(
            contradictory,
            np.array([[1.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
        )
        self.assertEqual(verdict, "UNKNOWN")

    def test_scipy_success_uses_same_exact_base_witness_validator(self):
        state = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[0.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix((1, 0), dtype=np.float64),
            b=np.array([0.0], dtype=np.float64),
            col_ids=np.array([909], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )

        class FakeMilpResult:
            success = True
            x = np.array([1e-12], dtype=np.float64)
            message = "fake tolerance-success"

        with (
            patch(
                "act.back_end.solver.solver_hz._HAS_HIGHSPY",
                False,
            ),
            patch(
                "act.back_end.solver.solver_hz._milp",
                return_value=FakeMilpResult(),
            ),
        ):
            status, message = hz_base_feasibility(
                state,
                time_limit=1.0,
            )
        self.assertEqual(status, "UNKNOWN")
        self.assertIn("not_exact", message)

    def test_binary_rhs_map_is_exact_for_equalities_and_outward_for_upper(self):
        empty_c = sp.csr_matrix((0, 0), dtype=np.float64)
        empty_b = sp.csr_matrix((0, 3), dtype=np.float64)
        upper_b = sp.csr_matrix(
            [[1e16, 1.0, -1e16]],
            dtype=np.float64,
        )
        _, _, ru, *_ = _base_milp_matrices_from_blocks(
            sp.csr_matrix((1, 0), dtype=np.float64),
            sp.csr_matrix((1, 3), dtype=np.float64),
            empty_c,
            empty_b,
            np.zeros(0, dtype=np.float64),
            sp.csr_matrix((1, 0), dtype=np.float64),
            upper_b,
            np.array([0.0], dtype=np.float64),
        )
        exact_upper = (
            Fraction.from_float(1e16)
            + Fraction.from_float(1.0)
            + Fraction.from_float(-1e16)
        )
        self.assertGreaterEqual(
            Fraction.from_float(float(ru[0])),
            exact_upper,
        )

        # 1e16 + 1 is not representable as binary64.  Rounding an equality
        # RHS would silently alter its feasible set, so the strict transform
        # must reject it.
        with self.assertRaises(ValueError):
            _base_milp_matrices_from_blocks(
                sp.csr_matrix((1, 0), dtype=np.float64),
                sp.csr_matrix((1, 1), dtype=np.float64),
                sp.csr_matrix((1, 0), dtype=np.float64),
                sp.csr_matrix([[1.0]], dtype=np.float64),
                np.array([1e16], dtype=np.float64),
                sp.csr_matrix((0, 0), dtype=np.float64),
                sp.csr_matrix((0, 1), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )

    def test_ill_conditioned_cutoff_infeasible_status_cannot_authorize_safe(self):
        # This exact candidate is inside every stored-float base inequality
        # and violates the property by ~1e-8.  HiGHS nevertheless reports the
        # cutoff MILP infeasible after numerical scaling/presolve.  SCIP and
        # unscaled HiGHS both find a witness.  A bare solver status must
        # therefore never authorize SAFE.
        Ac = np.array(
            [
                [-3.5702996391716793e-05, -4.8186540754284989e-04],
                [6.1351309191600138e-02, -8.4935335360188900e-03],
                [-2.5146514269878619e05, -1.9612896023818040e06],
                [1.0865942485058843e08, -8.2853638675315246e07],
            ],
            dtype=np.float64,
        )
        Ab = np.array(
            [
                [-1.2716528340094865e03, -6.9725573850599517e02],
                [8.6407162594525775e-07, -3.4709032262402831e-07],
                [-1.9719096875255330e-03, 1.8054824034457434e-02],
                [-3.7508279382209592e06, -6.0633582551243445e06],
            ],
            dtype=np.float64,
        )
        bl = np.array(
            [
                5.7439696878102927e02,
                -2.6009419021027907e-02,
                -4.8102211609657673e05,
                -6.8338224796549872e07,
            ],
            dtype=np.float64,
        )
        gc = np.array(
            [-6.589758673329295e-08, -2.720641322598071e-01],
            dtype=np.float64,
        )
        gb = np.array(
            [-2.5631753394172393e-07, -1.8013666159621009e-04],
            dtype=np.float64,
        )
        xc = np.array(
            [-0.38316764022084665, 0.2943856134052343],
            dtype=np.float64,
        )
        xb = np.array([-1.0, 1.0], dtype=np.float64)
        threshold = -0.08027163155510557

        def exact_dot(row, value):
            return sum(
                (
                    Fraction.from_float(float(a))
                    * Fraction.from_float(float(v))
                    for a, v in zip(row, value)
                ),
                Fraction(0),
            )

        for row_c, row_b, upper in zip(Ac, Ab, bl):
            lhs = exact_dot(row_c, xc) + exact_dot(row_b, xb)
            self.assertLessEqual(
                lhs,
                Fraction.from_float(float(upper)),
            )
        exact_output = exact_dot(gc, xc) + exact_dot(gb, xb)
        self.assertGreaterEqual(
            exact_output,
            Fraction.from_float(threshold),
        )

        hz = SparseHZono(
            c=np.array([0.0], dtype=np.float64),
            Gc=sp.csr_matrix(gc.reshape(1, -1)),
            Gb=sp.csr_matrix(gb.reshape(1, -1)),
            Ac=sp.csr_matrix((0, 2), dtype=np.float64),
            Ab=sp.csr_matrix((0, 2), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix(Ac),
            Aub=sp.csr_matrix(Ab),
            ub=bl,
            col_ids=np.array([905, 906], dtype=np.int64),
            bcol_ids=np.array([907, 908], dtype=np.int64),
        )
        hz_mark_known_nonempty(hz, "ill_conditioned_exact_candidate")
        self.assertEqual(
            hz_base_feasibility(hz, time_limit=2.0)[0],
            "FEASIBLE",
        )
        with patch.dict(
            os.environ,
            {
                "HZ_MILP_BACKEND": "highs",
                "HZ_MILP_EQ_SUBST": "0",
                "HZ_MILP_ELIM_SINGLETONS": "0",
                "HZ_HIGHS_OPTIONS": "",
                "HZ_LP_PREFILTER_FRACTION": "0",
                "HZ_LP_PREFILTER_MAX_SECONDS": "0",
            },
        ):
            verdict, _ = hz_objbound_decide(
                hz,
                np.array([[1.0]], dtype=np.float64),
                np.array([threshold], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
            )
        self.assertNotEqual(verdict, "SAFE")


class SparseSiblingJoinTests(unittest.TestCase):
    """Regression gate for sibling-local auxiliary/binary factor collisions."""

    @_sparse_provenance_required
    def test_relu_x_plus_relu_neg_x_keeps_independent_sibling_factors(self):
        bounds = Bounds(
            lb=torch.tensor([[-1.0]], dtype=DTYPE),
            ub=torch.tensor([[1.0]], dtype=DTYPE),
        )
        root = sparse_hz_from_bounds(bounds)
        if _SPARSE_HAS_FACTOR_IDS:
            root = SparseHZono(
                c=root.c,
                Gc=root.Gc,
                Gb=root.Gb,
                Ac=root.Ac,
                Ab=root.Ab,
                b=root.b,
                Auc=root.Auc,
                Aub=root.Aub,
                ub=root.ub,
                col_ids=np.array([100], dtype=np.int64),
                bcol_ids=np.empty(0, dtype=np.int64),
            )

        positive = sparse_hz_apply_relu_exact(
            sparse_hz_linear(root, np.array([[1.0]], dtype=np.float64)),
            bounds,
        )
        negative = sparse_hz_apply_relu_exact(
            sparse_hz_linear(root, np.array([[-1.0]], dtype=np.float64)),
            bounds,
        )

        if _SPARSE_HAS_FACTOR_IDS:
            root_ids = set(np.asarray(root.col_ids).reshape(-1).tolist())
            positive_aux = (
                set(np.asarray(positive.col_ids).reshape(-1).tolist()) - root_ids
            )
            negative_aux = (
                set(np.asarray(negative.col_ids).reshape(-1).tolist()) - root_ids
            )
            self.assertTrue(positive_aux.isdisjoint(negative_aux))
            self.assertTrue(
                set(np.asarray(positive.bcol_ids).reshape(-1).tolist()).isdisjoint(
                    set(np.asarray(negative.bcol_ids).reshape(-1).tolist())
                )
            )

        merged = sparse_hz_add_same_frame(positive, negative)
        lower, upper = _sparse_scalar_range(merged)
        self.assertAlmostEqual(lower, 0.0, places=8)
        self.assertAlmostEqual(
            upper,
            1.0,
            places=8,
            msg=(
                "sibling-local factor collision under-approximates "
                "ReLU(x)+ReLU(-x)=|x|"
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
