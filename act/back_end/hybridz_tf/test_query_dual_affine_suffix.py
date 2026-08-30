"""Soundness and integrity tests for truncated affine suffix replay."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Layer, Net
from act.back_end.hybridz_tf.query_dual_box_certifier import (
    certify_query_dual_boxes,
)
from act.back_end.hybridz_tf.gpu_dual_candidates import (
    BatchedDualCandidates,
)
from act.back_end.hybridz_tf.query_dual_replay import (
    QueryDualReplayError,
    replay_query_affine_lower_to_layer,
    validate_query_dual_affine_lower_plane,
)
from act.back_end.hybridz_tf.operator_hz import build_operator_hz
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _ordered_row_constraint_prefix_models,
    _solver_csr_sha256,
    hz_mark_constructively_nonempty,
    hz_objbound_decide,
)
from act.back_end.hybridz_tf.test_operator_add_fusion import (
    _assemble_width_toy,
    _dense_matrix,
    _input_layers,
    _wide_layer,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus
from act.back_end.hybridz_tf.test_query_dual_box_certifier import (
    _input_pair,
    _layer,
    _net,
)


def _two_block_scalar_residual_net():
    """x -> y=x+relu(x) -> q=y+relu(y-1/2)."""

    inp, spec = _input_pair(1, [-1.0], [1.0])
    stem = _layer(
        2,
        "DENSE",
        1,
        {
            "weight": np.asarray([[1.0]], dtype=np.float64),
            "bias": np.asarray([0.0], dtype=np.float64),
        },
    )
    first_relu = _layer(3, "RELU", 1)
    first_branch = _layer(
        4,
        "DENSE",
        1,
        {
            "weight": np.asarray([[1.0]], dtype=np.float64),
            "bias": np.asarray([0.0], dtype=np.float64),
        },
    )
    first_add = _layer(5, "ADD", 1)
    suffix_pre = _layer(
        6,
        "DENSE",
        1,
        {
            "weight": np.asarray([[1.0]], dtype=np.float64),
            "bias": np.asarray([-0.5], dtype=np.float64),
        },
    )
    suffix_relu = _layer(7, "RELU", 1)
    second_add = _layer(8, "ADD", 1)
    assertion = _layer(9, "ASSERT", 1, {"kind": "AUDIT"})
    return _net(
        [
            inp,
            spec,
            stem,
            first_relu,
            first_branch,
            first_add,
            suffix_pre,
            suffix_relu,
            second_add,
            assertion,
        ],
        {
            0: [],
            1: [0],
            2: [1],
            3: [2],
            4: [3],
            5: [2, 4],
            6: [5],
            7: [6],
            8: [5, 7],
            9: [8],
        },
    )


def _bypass_net():
    """Like the toy above, but layer 5 does not dominate the query."""

    net = _two_block_scalar_residual_net()
    net.preds[8] = [7, 2]
    net.succs[5].remove(8)
    net.succs[2].append(8)
    return net


def _exact_toy_values(x: Fraction) -> tuple[Fraction, Fraction]:
    first_relu = max(Fraction(0), x)
    stop = x + first_relu
    suffix_relu = max(Fraction(0), stop - Fraction(1, 2))
    return stop, stop + suffix_relu


def _operator_projection_toy():
    """Two ADDs with y-relu(y) cancellation hidden by materialization."""

    inp, spec = _input_layers(-1, 1)
    return _assemble_width_toy(
        [
            inp,
            spec,
            _dense_matrix(2, [[1]], [0]),
            _dense_matrix(3, [[0]], [0]),
            _wide_layer(4, "ADD", 1),
            _dense_matrix(5, [[1]], [0]),
            _wide_layer(6, "RELU", 1),
            _dense_matrix(7, [[-1]], [0]),
            _wide_layer(8, "ADD", 1),
            _dense_matrix(9, [[1]], [10]),
            _wide_layer(10, "RELU", 1),
            _dense_matrix(11, [[1]], [0]),
            _wide_layer(12, "ASSERT", 1),
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
            8: [4, 7],
            9: [8],
            10: [9],
            11: [10],
            12: [11],
        },
        input_lb=-1,
        input_ub=1,
    )


def _operator_mixed_alpha_projection_toy():
    """q=y-relu(y)-relu(-y); only mixed alpha=(1,0) proves q<=0."""

    inp, spec = _input_layers(-1, 1)
    return _assemble_width_toy(
        [
            inp,
            spec,
            _dense_matrix(2, [[1]], [0]),
            _dense_matrix(3, [[0]], [0]),
            _wide_layer(4, "ADD", 1),
            _dense_matrix(5, [[1]], [0]),
            _wide_layer(6, "RELU", 1),
            _dense_matrix(7, [[-1]], [0]),
            _dense_matrix(8, [[-1]], [0]),
            _wide_layer(9, "RELU", 1),
            _dense_matrix(10, [[-1]], [0]),
            _wide_layer(11, "ADD", 1),
            _wide_layer(12, "ADD", 1),
            _dense_matrix(13, [[1]], [10]),
            _wide_layer(14, "RELU", 1),
            _dense_matrix(15, [[1]], [0]),
            _wide_layer(16, "ASSERT", 1),
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
            8: [4],
            9: [8],
            10: [9],
            11: [4, 7],
            12: [11, 10],
            13: [12],
            14: [13],
            15: [14],
            16: [15],
        },
        input_lb=-1,
        input_ub=1,
    )


def _cube_upper(build, row: int) -> float:
    return float(build.hz.c[row]) + sum(
        float(np.abs(matrix.getrow(row).data).sum())
        for matrix in (build.hz.Gc, build.hz.Gb)
    )


def _verified_operator_projection_net(threshold: float) -> Net:
    dtype = torch.float64
    layers = []
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [2, 3],
        5: [4],
        6: [5],
        7: [6],
        8: [4, 7],
        9: [8],
        10: [9],
        11: [10],
        12: [11],
    }
    kinds = (
        "INPUT",
        "INPUT_SPEC",
        "DENSE",
        "DENSE",
        "ADD",
        "DENSE",
        "RELU",
        "DENSE",
        "ADD",
        "DENSE",
        "RELU",
        "DENSE",
        "ASSERT",
    )
    dense_params = {
        2: (1.0, 0.0),
        3: (0.0, 0.0),
        5: (1.0, 0.0),
        7: (-1.0, 0.0),
        9: (1.0, 10.0),
        11: (1.0, 0.0),
    }
    variables = {lid: [100 + lid] for lid in range(len(kinds))}
    variables[1] = variables[0]
    for lid, kind in enumerate(kinds):
        params = {}
        if kind == "INPUT":
            params = {"shape": (1, 1), "dtype": "torch.float64"}
        elif kind == "INPUT_SPEC":
            params = {
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=dtype),
                "ub": torch.tensor([[1.0]], dtype=dtype),
            }
        elif kind == "DENSE":
            weight, bias = dense_params[lid]
            params = {
                "weight": torch.tensor([[weight]], dtype=dtype),
                "bias": torch.tensor([bias], dtype=dtype),
                "in_features": 1,
                "out_features": 1,
            }
        elif kind == "ADD":
            params = {
                "x_vars": list(variables[preds[lid][0]]),
                "y_vars": list(variables[preds[lid][1]]),
            }
        elif kind == "ASSERT":
            params = OutputSpec(
                kind=OutKind.LINEAR_LE,
                c=torch.tensor([1.0], dtype=dtype),
                d=torch.tensor([float(threshold)], dtype=dtype),
            ).encode_linear(
                B=1,
                n_out=1,
                device=torch.device("cpu"),
                dtype=dtype,
            )
        in_vars = [
            value
            for parent in preds[lid]
            for value in variables[parent]
        ]
        layers.append(
            Layer(
                id=lid,
                kind=kind,
                params=params,
                in_vars=in_vars,
                out_vars=variables[lid],
            )
        )
    succs = {lid: [] for lid in range(len(kinds))}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    return Net(layers=layers, preds=preds, succs=succs)


class QueryDualAffineSuffixTests(unittest.TestCase):
    def test_prefix_models_preserve_hardest_first_outer_schedule(self):
        models = {
            (0, 3, "c", "d"): np.asarray([1, 5], dtype=np.int64),
            (0, 1, "a", "b"): np.asarray([2, 8], dtype=np.int64),
        }
        scheduled = _ordered_row_constraint_prefix_models(
            models,
            np.asarray([8, 5, 2, 1], dtype=np.int64),
        )
        self.assertEqual(
            [(key[:2], rows.tolist()) for key, rows in scheduled],
            [
                ((0, 1), [8, 2]),
                ((0, 3), [5, 1]),
            ],
        )

    @staticmethod
    def _prefix_lp_toy() -> SparseHZono:
        """One useful prefix row followed by deliberately irrelevant rows."""

        n_cont = 8
        # Output 0 is the ordinary fallback y; output 1 is the suffix plane x.
        Gc = sp.csr_matrix(
            (
                np.asarray([1.0, 1.0]),
                (
                    np.asarray([0, 1]),
                    np.asarray([1, 0]),
                ),
            ),
            shape=(2, n_cont),
            dtype=np.float64,
        )
        # The first stored row proves x <= -1/2.  Later rows touch only
        # unrelated variables and are intentionally absent from the prefix
        # relaxation.
        prefix = sp.csr_matrix(
            (
                np.asarray([1.0]),
                (
                    np.asarray([0]),
                    np.asarray([0]),
                ),
            ),
            shape=(1, n_cont),
            dtype=np.float64,
        )
        later = sp.csr_matrix(
            (
                np.ones(7, dtype=np.float64),
                (
                    np.arange(7, dtype=np.int64),
                    np.arange(1, 8, dtype=np.int64),
                ),
            ),
            shape=(7, n_cont),
            dtype=np.float64,
        )
        Auc = sp.vstack([prefix, later], format="csr")
        hz = SparseHZono(
            c=np.zeros(2, dtype=np.float64),
            Gc=Gc,
            Gb=sp.csr_matrix((2, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, n_cont), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=Auc,
            Aub=sp.csr_matrix((8, 0), dtype=np.float64),
            ub=np.asarray([-0.5, *([1.0] * 7)], dtype=np.float64),
        )
        hz_mark_constructively_nonempty(
            hz, "controlled_row_constraint_prefix_toy"
        )
        setattr(
            hz,
            "_solver_row_constraint_prefix_frames",
            {
                1: {
                    "schema": "operator_hz_row_constraint_prefix_v1",
                    "spec_row": 1,
                    "output_row": 1,
                    "stop_layer_id": 4,
                    "n_cont": 1,
                    "n_bin": 0,
                    "eq_rows": 0,
                    "ub_rows": 1,
                    "eq_csr_sha256": _solver_csr_sha256(hz.Ac[:0, :]),
                    "ub_csr_sha256": _solver_csr_sha256(hz.Auc[:1, :]),
                }
            },
        )
        return hz

    def test_row_constraint_prefix_lp_certifies_over_outer_relaxation(self):
        hz = self._prefix_lp_toy()
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=3.0,
            base_feas_time_limit=0.5,
            base_witness_precheck=False,
            lp_prefilter_fraction=1.0,
            lp_prefilter_max_seconds=2.0,
            safe_row_groups=((0, 1),),
            expected_safe_group_count=1,
        )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = hz._solver_objbound_stats
        self.assertEqual(stats["row_prefix_lp_valid_entries"], 1)
        self.assertEqual(stats["row_prefix_lp_certified_rows"], 1)
        self.assertEqual(stats["row_prefix_lp_certified_row_ids"], [1])
        self.assertEqual(stats["row_prefix_lp_full_constraint_rows"], 8)
        self.assertEqual(
            stats["row_prefix_lp_selected_constraint_rows_max"], 1
        )
        self.assertEqual(stats["row_prefix_lp_constraint_rows_dropped"], 7)
        self.assertEqual(
            stats["safe_row_group_winners"][0]["stage"],
            "row_constraint_prefix_lp_lagrangian",
        )
        self.assertEqual(
            stats["lp_status"],
            "skipped_all_property_groups_certified_"
            "by_row_constraint_prefix_lp",
        )

    def test_tampered_prefix_hash_has_no_proof_authority(self):
        hz = self._prefix_lp_toy()
        hz._solver_row_constraint_prefix_frames[1][
            "ub_csr_sha256"
        ] = "0" * 64
        verdict, witness = hz_objbound_decide(
            hz,
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            base_feas_time_limit=0.25,
            base_witness_precheck=False,
            lp_prefilter_fraction=0.0,
            lp_prefilter_max_seconds=0.0,
            safe_row_groups=((0, 1),),
            expected_safe_group_count=1,
        )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        stats = hz._solver_objbound_stats
        self.assertEqual(stats["row_prefix_lp_valid_entries"], 0)
        self.assertEqual(stats["row_prefix_lp_rejected_entries"], 1)
        self.assertEqual(stats["row_prefix_lp_certified_rows"], 0)

    def test_prefix_gpu_candidate_is_independently_reproved(self):
        hz = self._prefix_lp_toy()

        def candidate(_frame, q, **_kwargs):
            self.assertEqual(q.shape, (1, 8))
            return BatchedDualCandidates(
                # Prefix row is x <= -1/2; d=1 proves max x <= -1/2.
                # The candidate API returns the checker's row_dual=-d.
                row_dual=np.asarray([[-1.0]], dtype=np.float64),
                initial_support=np.asarray([1.0], dtype=np.float64),
                candidate_support=np.asarray([-0.5], dtype=np.float64),
                selected_rows=np.asarray([0], dtype=np.int64),
                device="cuda:0",
                dtype="torch.float64",
                steps_requested=1,
                steps_completed=1,
                elapsed_seconds=0.001,
                deadline_reached=False,
            )

        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "batched_original_frame_row_duals",
            side_effect=candidate,
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.eye(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=3.0,
                base_feas_time_limit=0.5,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
                gpu_dual_steps=1,
                gpu_dual_time_limit=1.0,
                safe_row_groups=((0, 1),),
                expected_safe_group_count=1,
            )
        self.assertEqual(verdict, "SAFE")
        self.assertIsNone(witness)
        stats = hz._solver_objbound_stats
        self.assertEqual(
            stats["gpu_dual_status"],
            "redirected_to_row_constraint_prefix",
        )
        self.assertEqual(stats["row_prefix_gpu_dual_certified_rows"], 1)
        self.assertEqual(
            stats["safe_row_group_winners"][0]["stage"],
            "row_constraint_prefix_gpu_lagrangian",
        )
        self.assertEqual(
            stats["lp_status"],
            "skipped_all_property_groups_certified_"
            "by_row_constraint_prefix_gpu",
        )

    def test_fake_negative_gpu_score_cannot_bypass_checker(self):
        hz = self._prefix_lp_toy()

        def fake_candidate(_frame, q, **_kwargs):
            return BatchedDualCandidates(
                # Zero multiplier cannot prove the prefix objective, even
                # though the untrusted diagnostic score lies about negativity.
                row_dual=np.zeros((q.shape[0], 1), dtype=np.float64),
                initial_support=np.ones(q.shape[0], dtype=np.float64),
                candidate_support=-np.ones(q.shape[0], dtype=np.float64),
                selected_rows=np.asarray([0], dtype=np.int64),
                device="cuda:0",
                dtype="torch.float64",
                steps_requested=1,
                steps_completed=1,
                elapsed_seconds=0.001,
                deadline_reached=False,
            )

        with patch(
            "act.back_end.hybridz_tf.gpu_dual_candidates."
            "batched_original_frame_row_duals",
            side_effect=fake_candidate,
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.eye(2, dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=2.0,
                base_feas_time_limit=0.5,
                base_witness_precheck=False,
                lp_prefilter_fraction=0.0,
                lp_prefilter_max_seconds=0.0,
                gpu_dual_steps=1,
                gpu_dual_time_limit=0.5,
                safe_row_groups=((0, 1),),
                expected_safe_group_count=1,
            )
        self.assertEqual(verdict, "UNKNOWN")
        self.assertIsNone(witness)
        stats = hz._solver_objbound_stats
        self.assertEqual(stats["row_prefix_gpu_dual_certified_rows"], 0)
        self.assertEqual(stats["safe_row_groups_resolved"], 0)

    def test_full_input_replay_exports_a_verified_constant_plane(self):
        toy = _operator_projection_toy()
        candidate = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=np.asarray([[1.0]], dtype=np.float64),
            property_upper_thresholds=np.asarray(
                [10.25], dtype=np.float64
            ),
            property_tail_suffix_blocks=8,
            property_tail_suffix_alpha_steps=8,
            property_tail_suffix_alpha_time_limit=2.0,
            property_tail_suffix_alpha_device="cpu",
        )
        receipt = candidate.metadata["property_tail_upper"][
            "shared_suffix_replay"
        ]
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(
            receipt["replay_strategy"], "optimized_only_full_input"
        )
        self.assertEqual(
            receipt["output_form"], "full_input_property_constant"
        )
        self.assertEqual(receipt["full_input_negative_rows"], 1)
        self.assertEqual(candidate.property_upper_row_groups, ((0, 1),))
        self.assertEqual(candidate.hz.Gc.getrow(1).nnz, 0)
        self.assertEqual(candidate.hz.Gb.getrow(1).nnz, 0)
        self.assertLess(float(candidate.hz.c[1]), -0.249)
        self.assertIsNotNone(
            candidate.hz._property_full_input_replay_result
        )
        self.assertEqual(
            candidate.hz._solver_row_constraint_prefix_frames, {}
        )

    def test_verifier_accepts_live_full_input_replay_object(self):
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                _verified_operator_projection_net(10.25),
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=4.0,
                        engine="operator_hz_objbound",
                        property_tail_upper=True,
                        property_tail_suffix_blocks=8,
                        property_tail_suffix_alpha_steps=8,
                        property_tail_suffix_alpha_time_limit=2.0,
                        property_tail_suffix_alpha_device="cpu",
                        lp_prefilter_fraction=0.0,
                        lp_prefilter_max_seconds=0.0,
                    ),
                ),
            )[0]
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            self.assertEqual(result.metadata["hz_verdict"], "SAFE")
            receipt = result.metadata["operator_hz"][
                "property_tail_upper"
            ]["shared_suffix_replay"]
            self.assertEqual(
                receipt["output_form"],
                "full_input_property_constant",
            )
            self.assertEqual(receipt["full_input_negative_rows"], 1)
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_adaptive_suffix_alpha_beats_both_uniform_extremes(self):
        toy = _operator_mixed_alpha_projection_toy()
        candidate = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            property_upper_C=np.asarray([[1.0]], dtype=np.float64),
            property_upper_thresholds=np.asarray(
                [10.25], dtype=np.float64
            ),
            property_tail_suffix_blocks=1,
            property_tail_suffix_alpha_steps=16,
            property_tail_suffix_alpha_time_limit=3.0,
            property_tail_suffix_alpha_device="cpu",
        )

        for x in (Fraction(-1), Fraction(0), Fraction(1)):
            q = x - max(Fraction(0), x) - max(Fraction(0), -x)
            self.assertLessEqual(q, Fraction(0))
        suffix = candidate.metadata["property_tail_upper"][
            "shared_suffix_replay"
        ]
        self.assertEqual(suffix["status"], "applied")
        self.assertEqual(suffix["stop_layer_id"], 4)
        self.assertEqual(suffix["optimized_alpha"]["status"], "replayed")
        self.assertEqual(suffix["optimized_alpha_selected_rows"], 1)
        self.assertEqual(suffix["alpha_one_selected_rows"], 0)
        self.assertLess(_cube_upper(candidate, 1), -0.249)
        self.assertGreater(suffix["free_cube_improvement_max"], 0.9)

    def test_verifier_accepts_only_the_grouped_sound_suffix_object(self):
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                _verified_operator_projection_net(10.25),
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=3.0,
                        engine="operator_hz_objbound",
                        property_tail_upper=True,
                        property_tail_suffix_blocks=1,
                        lp_prefilter_fraction=0.0,
                        lp_prefilter_max_seconds=0.0,
                    ),
                ),
            )[0]
            self.assertEqual(result.status, VerifyStatus.CERTIFIED)
            self.assertEqual(result.metadata["hz_verdict"], "SAFE")
            self.assertEqual(
                result.metadata["property_upper_row_groups"], [[0, 1]]
            )
            self.assertEqual(
                result.metadata["cfg_property_tail_suffix_blocks"], 1
            )
            suffix = result.metadata["operator_hz"][
                "property_tail_upper"
            ]["shared_suffix_replay"]
            self.assertEqual(suffix["status"], "applied")
            self.assertTrue(suffix["proof_authority"])
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_operator_projection_crosses_one_block_and_closes_cancellation(self):
        toy = _operator_projection_toy()
        kwargs = {
            "exact_budget": 0,
            "materialize_add": True,
            "property_upper_C": np.asarray([[1.0]], dtype=np.float64),
            "property_upper_thresholds": np.asarray(
                [10.25], dtype=np.float64
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
            property_tail_suffix_blocks=1,
        )

        # Exact graph: y=x, y-relu(y)+10 <= 10, hence the property upper is
        # at most -1/4 throughout both exact affine regions.
        for x in (Fraction(-1), Fraction(0), Fraction(1)):
            exact = x - max(Fraction(0), x) + Fraction(10)
            self.assertLessEqual(exact - Fraction(41, 4), Fraction(-1, 4))
        self.assertEqual(candidate.property_upper_row_groups, ((0, 1),))
        self.assertGreater(_cube_upper(baseline, 0), 0.7)
        self.assertLess(_cube_upper(candidate, 1), -0.249)
        self.assertGreater(
            _cube_upper(candidate, 0) - _cube_upper(candidate, 1),
            0.9,
        )
        receipt = candidate.metadata["property_tail_upper"][
            "shared_suffix_replay"
        ]
        self.assertEqual(receipt["status"], "applied")
        self.assertTrue(receipt["proof_authority"])
        self.assertEqual(receipt["stop_layer_id"], 4)
        self.assertEqual(receipt["alpha_one_selected_rows"], 1)
        self.assertGreater(receipt["free_cube_improvement_max"], 0.9)
        self.assertEqual(
            receipt["row_local_prefix_lp_schema"],
            "operator_hz_row_constraint_prefix_v1",
        )
        frames = candidate.hz._solver_row_constraint_prefix_frames
        self.assertEqual(sorted(frames), [1])
        self.assertEqual(frames[1]["stop_layer_id"], 4)
        self.assertEqual(
            frames[1]["ub_csr_sha256"],
            _solver_csr_sha256(
                candidate.hz.Auc[
                    : frames[1]["ub_rows"], :
                ]
            ),
        )

    def test_fraction_piecewise_oracle_proves_each_returned_plane(self):
        net = _two_block_scalar_residual_net()
        certificate = certify_query_dual_boxes(net)
        rows = np.asarray([[1.0], [-1.0], [2.0]], dtype=np.float64)
        bias = np.asarray([0.25, -0.5, 0.75], dtype=np.float64)
        plane = replay_query_affine_lower_to_layer(
            net,
            certificate.bounds,
            stop_lid=5,
            query_rows=rows,
            query_bias=bias,
            chunk_size=1,
        )

        # These are every endpoint of every affine region of the exact toy,
        # so checking them is an exact continuous-domain oracle, not sampling.
        breakpoints = (
            Fraction(-1),
            Fraction(0),
            Fraction(1, 4),
            Fraction(1),
        )
        for x in breakpoints:
            stop, output = _exact_toy_values(x)
            for row in range(rows.shape[0]):
                returned = (
                    Fraction.from_float(float(plane.scalar[row]))
                    + Fraction.from_float(float(plane.coefficients[row, 0]))
                    * stop
                )
                exact_query = (
                    Fraction.from_float(float(rows[row, 0])) * output
                    + Fraction.from_float(float(bias[row]))
                )
                self.assertLessEqual(returned, exact_query)

        self.assertTrue(validate_query_dual_affine_lower_plane(plane))
        self.assertFalse(plane.coefficients.flags.writeable)
        self.assertFalse(plane.scalar.flags.writeable)
        self.assertNotIn("coefficients_hex", plane.receipt)

    def test_non_dominating_stop_fails_closed(self):
        net = _bypass_net()
        certificate = certify_query_dual_boxes(net)
        with self.assertRaises(QueryDualReplayError) as caught:
            replay_query_affine_lower_to_layer(
                net,
                certificate.bounds,
                stop_lid=5,
                query_rows=np.asarray([[1.0]], dtype=np.float64),
            )
        self.assertEqual(caught.exception.code, "INVALID_GRAPH")

    def test_array_and_receipt_tampering_are_rejected(self):
        net = _two_block_scalar_residual_net()
        certificate = certify_query_dual_boxes(net)
        plane = replay_query_affine_lower_to_layer(
            net,
            certificate.bounds,
            stop_lid=5,
            query_rows=np.asarray([[1.0]], dtype=np.float64),
        )
        changed = plane.coefficients.copy()
        changed[0, 0] += 1.0
        changed.setflags(write=False)
        self.assertFalse(
            validate_query_dual_affine_lower_plane(
                replace(plane, coefficients=changed)
            )
        )
        receipt = dict(plane.receipt)
        receipt["stop_layer_id"] = 2
        self.assertFalse(
            validate_query_dual_affine_lower_plane(
                replace(plane, receipt=receipt)
            )
        )

    def test_cross_bound_hash_pin_is_rejected(self):
        net = _two_block_scalar_residual_net()
        certificate = certify_query_dual_boxes(net)
        first = replay_query_affine_lower_to_layer(
            net,
            certificate.bounds,
            stop_lid=5,
            query_rows=np.asarray([[1.0]], dtype=np.float64),
        )
        changed_bounds = dict(certificate.bounds)
        changed_bounds[6] = {
            "lb": np.asarray([-2.0], dtype=np.float64),
            "ub": np.asarray([2.0], dtype=np.float64),
        }
        with self.assertRaises(QueryDualReplayError) as caught:
            replay_query_affine_lower_to_layer(
                net,
                changed_bounds,
                stop_lid=5,
                query_rows=np.asarray([[1.0]], dtype=np.float64),
                expected_bounds_sha256=first.receipt["hashes"][
                    "bounds_sha256"
                ],
            )
        self.assertEqual(caught.exception.code, "HASH_MISMATCH")


if __name__ == "__main__":
    unittest.main()
