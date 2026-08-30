#!/usr/bin/env python3
"""Toy-first audits for exact HybridZ binary-phase covers.

The first test checks the substitution algebra directly.  The second uses two
duplicate ReLUs: the exact network computes

``ReLU(x) - ReLU(x) - 0.1 == -0.1``

but independent triangle relaxations permit a positive spurious margin.
Making only the first ReLU exact and enumerating its two binary phases restores
the correlation.  Both continuous children must independently certify the
property before the cover is considered SAFE.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Layer, Net
from act.back_end.hybridz_tf.operator_hz import (
    _OperatorHZBuilder,
    build_operator_hz,
)
from act.back_end.hybridz_tf.test_operator_add_fusion import (
    _assemble_scalar_toy,
    _dense,
    _exact_graph_range,
    _input_layers,
    _layer,
    _lp_output_range,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _hz_attach_exact_phase_conditional_property_rows_from_operator,
    _hz_conditional_applied_content_sha256,
    _hz_conditional_parent_content_sha256,
    hz_attach_exact_phase_conditional_property_rows,
    hz_base_feasibility,
    hz_enumerate_sparse_binary_phase_cover,
    hz_fix_sparse_binary_assignment,
    hz_mark_constructively_nonempty,
    hz_objbound_decide,
    hz_verify_sparse_binary_phase_child,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


def _verified_duplicate_relu_residual_net(
    *,
    interior_suffix_relu: bool = False,
    joint_suffix_relus: bool = False,
) -> Net:
    """Two duplicate ReLUs, two dominating ADDs, and a strict property tail."""

    interior_suffix_relu = bool(
        interior_suffix_relu or joint_suffix_relus
    )
    dtype = torch.float64
    kinds = list((
        "INPUT",
        "INPUT_SPEC",
        "RELU",
        "RELU",
        "DENSE",
        "ADD",
        "DENSE",
        "DENSE",
        "ADD",
        "DENSE",
        "RELU",
        "DENSE",
        "ASSERT",
    ))
    if interior_suffix_relu:
        kinds[7] = "RELU"
    kinds = tuple(kinds)
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [3],
        5: [2, 4],
        6: [5],
        7: [5],
        8: [6, 7],
        9: [8],
        10: [9],
        11: [10],
        12: [11],
    }
    dense_params = {
        4: (-1.0, 0.0),
        6: (1.0, 0.0),
        7: (0.0, 0.0),
        9: (1.0, 9.9),
        11: (1.0, 0.0),
    }
    if interior_suffix_relu:
        dense_params.pop(7)
    if joint_suffix_relus:
        dense_params[9] = (1.0, -0.5)
    variables = {layer_id: [200 + layer_id] for layer_id in range(len(kinds))}
    variables[1] = variables[0]
    layers = []
    for layer_id, kind in enumerate(kinds):
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
            weight, bias = dense_params[layer_id]
            params = {
                "weight": torch.tensor([[weight]], dtype=dtype),
                "bias": torch.tensor([bias], dtype=dtype),
                "in_features": 1,
                "out_features": 1,
            }
        elif kind == "ADD":
            params = {
                "x_vars": list(variables[preds[layer_id][0]]),
                "y_vars": list(variables[preds[layer_id][1]]),
            }
        elif kind == "ASSERT":
            params = OutputSpec(
                kind=OutKind.LINEAR_LE,
                c=torch.tensor([1.0], dtype=dtype),
                d=torch.tensor(
                    [0.01 if joint_suffix_relus else 9.95],
                    dtype=dtype,
                ),
            ).encode_linear(
                B=1,
                n_out=1,
                device=torch.device("cpu"),
                dtype=dtype,
            )
        layers.append(
            Layer(
                id=layer_id,
                kind=kind,
                params=params,
                in_vars=[
                    value
                    for parent in preds[layer_id]
                    for value in variables[parent]
                ],
                out_vars=variables[layer_id],
            )
        )
    succs = {layer_id: [] for layer_id in range(len(kinds))}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    return Net(layers=layers, preds=preds, succs=succs)


class SparseBinarySubstitutionTests(unittest.TestCase):
    def test_focused_conditional_alpha_preserves_rival_binding(self) -> None:
        full = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
        native = full.reshape(1, 4, 3)
        selected = _OperatorHZBuilder._slice_property_query_alpha(
            {
                3: full,
                5: native,
                7: np.asarray([0.1, 0.2, 0.3]),
                9: np.asarray(0.25),
            },
            rival_ids=(2,),
            full_query_count=4,
        )
        np.testing.assert_array_equal(selected[3], full[[2], :])
        np.testing.assert_array_equal(selected[5], native[:, [2], :])
        np.testing.assert_array_equal(
            selected[7], np.asarray([0.1, 0.2, 0.3])
        )
        self.assertEqual(float(selected[9].item()), 0.25)

    def test_conditional_property_rows_exist_only_on_matching_exact_child(
        self,
    ) -> None:
        parent = SparseHZono(
            c=np.asarray([1.0], dtype=np.float64),
            Gc=sp.csr_matrix([[0.25]], dtype=np.float64),
            Gb=sp.csr_matrix([[0.0]], dtype=np.float64),
            Ac=sp.csr_matrix((0, 1), dtype=np.float64),
            Ab=sp.csr_matrix((0, 1), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix((0, 1), dtype=np.float64),
            Aub=sp.csr_matrix((0, 1), dtype=np.float64),
            ub=np.zeros(0, dtype=np.float64),
            col_ids=np.asarray([10], dtype=np.int64),
            bcol_ids=np.asarray([20], dtype=np.int64),
        )
        hz_mark_constructively_nonempty(parent, "conditional_toy")
        parent._solver_continuous_column_layer_ids = np.asarray(
            [3], dtype=np.int64
        )
        parent._solver_constraint_row_tags = ()
        _hz_attach_exact_phase_conditional_property_rows_from_operator(
            parent,
            [
                {
                    "binary_col_id": 20,
                    "phase": -1,
                    "layer_id": 7,
                    "row": 2,
                    "center": np.asarray([-0.3]),
                    "generator": sp.csr_matrix([[0.1]]),
                    "error": np.asarray([0.01]),
                    "rival_ids": (7,),
                },
                {
                    "binary_col_id": 20,
                    "phase": 1,
                    "layer_id": 7,
                    "row": 2,
                    "center": np.asarray([-0.2]),
                    "generator": sp.csr_matrix([[0.05]]),
                    "error": np.asarray([0.02]),
                    "rival_ids": (7,),
                },
            ],
        )
        children = hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=2
        )
        self.assertEqual(parent.n_out, 1)
        for assignment, child in children:
            phase = assignment[0][1]
            self.assertEqual(child.n_out, 2)
            self.assertEqual(child.n_cont, 2)
            np.testing.assert_array_equal(
                child._solver_continuous_column_layer_ids,
                np.asarray([3, -2], dtype=np.int64),
            )
            self.assertEqual(child._solver_constraint_row_tags, ())
            applied = child._solver_conditional_property_rows_applied
            self.assertTrue(applied["proof_authority"])
            self.assertEqual(applied["rival_to_output_rows"], {7: (1,)})
            expected_center = -0.3 if phase == -1 else -0.2
            expected_error = 0.01 if phase == -1 else 0.02
            self.assertAlmostEqual(child.c[1], expected_center)
            self.assertAlmostEqual(child.Gc[1, 1], expected_error)
            self.assertEqual(child.Gc[1, 0], 0.1 if phase == -1 else 0.05)

    def test_joint_conditional_rows_cover_exactly_four_phase_children(
        self,
    ) -> None:
        def parent() -> SparseHZono:
            hz = SparseHZono(
                c=np.asarray([0.0], dtype=np.float64),
                Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
                Gb=sp.csr_matrix((1, 2), dtype=np.float64),
                Ac=sp.csr_matrix((0, 1), dtype=np.float64),
                Ab=sp.csr_matrix((0, 2), dtype=np.float64),
                b=np.zeros(0, dtype=np.float64),
                Auc=sp.csr_matrix((0, 1), dtype=np.float64),
                Aub=sp.csr_matrix((0, 2), dtype=np.float64),
                ub=np.zeros(0, dtype=np.float64),
                col_ids=np.asarray([10], dtype=np.int64),
                bcol_ids=np.asarray([20, 21], dtype=np.int64),
            )
            hz_mark_constructively_nonempty(hz, "joint_conditional_toy")
            return hz

        rows = []
        for left in (-1, 1):
            for right in (-1, 1):
                rows.append(
                    {
                        "binary_guards": (
                            {
                                "binary_col_id": 20,
                                "phase": left,
                                "layer_id": 7,
                                "row": 2,
                            },
                            {
                                "binary_col_id": 21,
                                "phase": right,
                                "layer_id": 7,
                                "row": 5,
                            },
                        ),
                        "center": np.asarray(
                            [10.0 * left + right], dtype=np.float64
                        ),
                        "generator": sp.csr_matrix(
                            [[0.25]], dtype=np.float64
                        ),
                        "error": np.asarray([0.0], dtype=np.float64),
                        "rival_ids": (3,),
                    }
                )
        hz = parent()
        _hz_attach_exact_phase_conditional_property_rows_from_operator(
            hz, rows
        )
        cover = hz_enumerate_sparse_binary_phase_cover(
            hz, max_children=4
        )
        self.assertEqual(len(cover), 4)
        for assignment, child in cover:
            phases = tuple(value for _position, value in assignment)
            self.assertEqual(child.n_out, 2)
            self.assertAlmostEqual(
                child.c[1], 10.0 * phases[0] + phases[1]
            )
            applied = child._solver_conditional_property_rows_applied
            self.assertEqual(applied["rival_to_output_rows"], {3: (1,)})
            self.assertEqual(len(applied["applied_guard_sets"]), 1)
            self.assertEqual(
                tuple(
                    guard["phase"]
                    for guard in applied["applied_guard_sets"][0]
                ),
                phases,
            )

        with self.assertRaisesRegex(ValueError, "complete exact phase cover"):
            _hz_attach_exact_phase_conditional_property_rows_from_operator(
                parent(), rows[:-1]
            )

    def _parent(self) -> SparseHZono:
        hz = SparseHZono(
            c=np.asarray([0.25, -0.5], dtype=np.float64),
            Gc=sp.csr_matrix(
                np.asarray([[1.0], [2.0]], dtype=np.float64)
            ),
            Gb=sp.csr_matrix(
                np.asarray([[3.0, -4.0], [5.0, 6.0]], dtype=np.float64)
            ),
            Ac=sp.csr_matrix(
                np.asarray([[2.0], [-3.0]], dtype=np.float64)
            ),
            Ab=sp.csr_matrix(
                np.asarray([[7.0, -8.0], [9.0, 10.0]], dtype=np.float64)
            ),
            b=np.asarray([11.0, -12.0], dtype=np.float64),
            Auc=sp.csr_matrix(
                np.asarray([[4.0], [-5.0]], dtype=np.float64)
            ),
            Aub=sp.csr_matrix(
                np.asarray([[13.0, 14.0], [-15.0, 16.0]], dtype=np.float64)
            ),
            ub=np.asarray([17.0, -18.0], dtype=np.float64),
            col_ids=np.asarray([101], dtype=np.int64),
            bcol_ids=np.asarray([201, 202], dtype=np.int64),
        )
        setattr(
            hz,
            "_solver_row_constraint_prefix_frames",
            {
                0: {
                    "schema": "operator_hz_row_constraint_prefix_v1",
                    "spec_row": 0,
                    "output_row": 0,
                    "stop_layer_id": 3,
                    "n_cont": 1,
                    "n_bin": 2,
                    "eq_rows": 2,
                    "ub_rows": 2,
                    "eq_csr_sha256": "0" * 64,
                    "ub_csr_sha256": "1" * 64,
                }
            },
        )
        return hz

    def _conditional_hash_fixture(
        self,
    ) -> tuple[SparseHZono, dict[int, int], SparseHZono]:
        parent = SparseHZono(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Gb=sp.csr_matrix([[0.0]], dtype=np.float64),
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix([[0.0]], dtype=np.float64),
            b=np.asarray([0.0], dtype=np.float64),
            Auc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Aub=sp.csr_matrix([[0.0]], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
            col_ids=np.asarray([10], dtype=np.int64),
            bcol_ids=np.asarray([20], dtype=np.int64),
        )
        hz_mark_constructively_nonempty(
            parent, "conditional_hash_fixture"
        )
        rows = [
            {
                "binary_col_id": 20,
                "phase": phase,
                "layer_id": 7,
                "row": 2,
                "center": np.asarray([0.5 * phase]),
                "generator": sp.csr_matrix([[0.25]]),
                "error": np.asarray([0.125]),
                "rival_ids": (3,),
                "receipt": {
                    "schema": "controlled_conditional_toy_source_v1",
                    "source": "duplicate_relu_exact_suffix_replay",
                    "source_receipt_sha256": "1" * 64,
                },
            }
            for phase in (-1, 1)
        ]
        _hz_attach_exact_phase_conditional_property_rows_from_operator(
            parent, rows
        )
        assignment = {0: 1}
        child = hz_fix_sparse_binary_assignment(parent, assignment)
        self.assertTrue(
            hz_verify_sparse_binary_phase_child(
                parent, assignment, child
            )
        )
        return parent, assignment, child

    def test_public_conditional_attach_has_no_proof_authority(self):
        parent = self._parent()
        rows = [
            {
                "binary_col_id": int(parent.bcol_ids[0]),
                "phase": phase,
                "layer_id": 7,
                "row": 2,
                "center": np.asarray([0.0]),
                "generator": sp.csr_matrix(
                    (1, parent.n_cont), dtype=np.float64
                ),
                "error": np.asarray([0.0]),
                "rival_ids": (0,),
            }
            for phase in (-1, 1)
        ]
        with self.assertRaisesRegex(
            PermissionError, "private proof producer"
        ):
            hz_attach_exact_phase_conditional_property_rows(parent, rows)
        with self.assertRaisesRegex(
            PermissionError, "private proof producer"
        ):
            hz_attach_exact_phase_conditional_property_rows(
                parent,
                rows,
                _producer_capability=object(),
            )
        self.assertFalse(
            hasattr(parent, "_solver_conditional_property_rows_token")
        )

    def test_conditional_attach_detaches_and_freezes_source_values(self):
        parent = SparseHZono(
            c=np.asarray([0.0]),
            Gc=sp.csr_matrix([[1.0]]),
            Gb=sp.csr_matrix([[0.0]]),
            Ac=sp.csr_matrix((0, 1)),
            Ab=sp.csr_matrix((0, 1)),
            b=np.zeros(0),
            Auc=sp.csr_matrix((0, 1)),
            Aub=sp.csr_matrix((0, 1)),
            ub=np.zeros(0),
            col_ids=np.asarray([10]),
            bcol_ids=np.asarray([20]),
        )
        hz_mark_constructively_nonempty(parent, "detached_source_toy")
        centers = {
            phase: np.asarray([0.25 * phase], dtype=np.float64)
            for phase in (-1, 1)
        }
        generators = {
            phase: sp.csr_matrix([[0.125]], dtype=np.float64)
            for phase in (-1, 1)
        }
        errors = {
            phase: np.asarray([0.0625], dtype=np.float64)
            for phase in (-1, 1)
        }
        receipts = {
            phase: {"source": ["toy", phase]}
            for phase in (-1, 1)
        }
        rows = [
            {
                "binary_col_id": 20,
                "phase": phase,
                "layer_id": 7,
                "row": 2,
                "center": centers[phase],
                "generator": generators[phase],
                "error": errors[phase],
                "rival_ids": (3,),
                "receipt": receipts[phase],
            }
            for phase in (-1, 1)
        ]
        _hz_attach_exact_phase_conditional_property_rows_from_operator(
            parent, rows
        )
        centers[1][0] = 99.0
        generators[1].data[0] = 99.0
        errors[1][0] = 99.0
        receipts[1]["source"].append("mutated")
        child = hz_fix_sparse_binary_assignment(parent, {0: 1})
        self.assertTrue(
            hz_verify_sparse_binary_phase_child(
                parent, {0: 1}, child
            )
        )
        self.assertEqual(float(child.c[1]), 0.25)
        self.assertEqual(float(child.Gc[1, 0]), 0.125)
        self.assertEqual(float(child.Gc[1, 1]), 0.0625)
        stored = parent._solver_conditional_property_rows[1]
        self.assertFalse(stored["center"].flags.writeable)
        self.assertFalse(stored["error"].flags.writeable)
        self.assertFalse(stored["generator"].data.flags.writeable)

    def test_conditional_parent_and_child_hashes_reject_mutations(self):
        def replace_record(parent, field, value):
            records = list(parent._solver_conditional_property_rows)
            replacement = dict(records[1])
            replacement[field] = value
            records[1] = replacement
            parent._solver_conditional_property_rows = tuple(records)

        parent, assignment, child = self._conditional_hash_fixture()
        mutations = {}

        center_parent = deepcopy(parent)
        replace_record(
            center_parent,
            "center",
            np.asarray([0.75], dtype=np.float64),
        )
        mutations["parent_center"] = center_parent

        rival_parent = deepcopy(parent)
        replace_record(rival_parent, "rival_ids", (4,))
        mutations["parent_rival_ids"] = rival_parent

        generator_parent = deepcopy(parent)
        replace_record(
            generator_parent,
            "generator",
            sp.csr_matrix([[0.5]], dtype=np.float64),
        )
        mutations["parent_generator_csr"] = generator_parent

        error_parent = deepcopy(parent)
        replace_record(
            error_parent,
            "error",
            np.asarray([0.25], dtype=np.float64),
        )
        mutations["parent_error"] = error_parent

        source_receipt_parent = deepcopy(parent)
        replace_record(
            source_receipt_parent,
            "receipt",
            {
                "schema": "wrong_source_v1",
                "source": "unbound_external_caller",
                "source_receipt_sha256": "2" * 64,
            },
        )
        mutations["parent_source_receipt"] = source_receipt_parent

        parent_receipt = deepcopy(parent)
        parent_receipt._solver_conditional_property_rows_receipt = dict(
            parent_receipt._solver_conditional_property_rows_receipt
        )
        parent_receipt._solver_conditional_property_rows_receipt[
            "live_content_sha256"
        ] = "0" * 64
        mutations["parent_live_receipt"] = parent_receipt

        token_parent = deepcopy(parent)
        token_parent._solver_conditional_property_rows_token = object()
        mutations["parent_token"] = token_parent

        for name, tampered_parent in mutations.items():
            with self.subTest(tamper=name):
                self.assertFalse(
                    hz_verify_sparse_binary_phase_child(
                        tampered_parent,
                        assignment,
                        child,
                    )
                )

        child_map = deepcopy(child)
        applied = dict(
            child_map._solver_conditional_property_rows_applied
        )
        applied["rival_to_output_rows"] = {3: (999,)}
        child_map._solver_conditional_property_rows_applied = applied
        self.assertFalse(
            hz_verify_sparse_binary_phase_child(
                parent,
                assignment,
                child_map,
            )
        )

        joint_parent = deepcopy(parent)
        replace_record(
            joint_parent,
            "center",
            np.asarray([0.75], dtype=np.float64),
        )
        joint_hash = _hz_conditional_parent_content_sha256(
            joint_parent,
            joint_parent._solver_conditional_property_rows,
        )
        joint_parent._solver_conditional_property_rows_receipt = dict(
            joint_parent._solver_conditional_property_rows_receipt
        )
        joint_parent._solver_conditional_property_rows_receipt[
            "live_content_sha256"
        ] = joint_hash
        joint_child = deepcopy(child)
        joint_child.c[1] = 0.75
        joint_applied = dict(
            joint_child._solver_conditional_property_rows_applied
        )
        joint_applied["parent_live_content_sha256"] = joint_hash
        joint_applied.pop("live_content_sha256")
        joint_applied["live_content_sha256"] = (
            _hz_conditional_applied_content_sha256(joint_applied)
        )
        joint_child._solver_conditional_property_rows_applied = (
            joint_applied
        )
        self.assertFalse(
            hz_verify_sparse_binary_phase_child(
                joint_parent,
                assignment,
                joint_child,
            )
        )

    def test_exact_substitution_preserves_values_and_row_residuals(self):
        parent = self._parent()
        for assignment, child in hz_enumerate_sparse_binary_phase_cover(
            parent, max_children=4
        ):
            fixed = np.asarray(
                [dict(assignment)[0], dict(assignment)[1]],
                dtype=np.float64,
            )
            self.assertEqual(child.n_bin, 0)
            self.assertEqual(
                child._solver_row_constraint_prefix_frames[0]["n_bin"], 0
            )
            self.assertEqual(
                child._solver_binary_phase_fix["fixed_bcol_ids"],
                [201, 202],
            )
            for continuous in (-1.0, -0.25, 0.0, 0.75, 1.0):
                xc = np.asarray([continuous], dtype=np.float64)
                parent_value = (
                    parent.c
                    + np.asarray(parent.Gc @ xc).reshape(-1)
                    + np.asarray(parent.Gb @ fixed).reshape(-1)
                )
                child_value = (
                    child.c + np.asarray(child.Gc @ xc).reshape(-1)
                )
                np.testing.assert_array_equal(child_value, parent_value)

                parent_eq_residual = (
                    np.asarray(parent.Ac @ xc).reshape(-1)
                    + np.asarray(parent.Ab @ fixed).reshape(-1)
                    - parent.b
                )
                child_eq_residual = (
                    np.asarray(child.Ac @ xc).reshape(-1) - child.b
                )
                np.testing.assert_array_equal(
                    child_eq_residual, parent_eq_residual
                )

                parent_ub_residual = (
                    np.asarray(parent.Auc @ xc).reshape(-1)
                    + np.asarray(parent.Aub @ fixed).reshape(-1)
                    - parent.ub
                )
                child_ub_residual = (
                    np.asarray(child.Auc @ xc).reshape(-1) - child.ub
                )
                np.testing.assert_array_equal(
                    child_ub_residual, parent_ub_residual
                )

    def test_phase_fix_hashes_each_distinct_constraint_prefix_once(self):
        from act.back_end.solver import solver_hz

        parent = self._parent()
        template = next(
            iter(parent._solver_row_constraint_prefix_frames.values())
        )
        frames = {}
        for row in range(198):
            entry = dict(template)
            entry["spec_row"] = row
            entry["output_row"] = row
            if row % 2:
                entry["eq_rows"] = 1
                entry["ub_rows"] = 1
            frames[row] = entry
        parent._solver_row_constraint_prefix_frames = frames

        with patch(
            "act.back_end.solver.solver_hz._solver_csr_sha256",
            wraps=solver_hz._solver_csr_sha256,
        ) as digest:
            child = hz_fix_sparse_binary_assignment(
                parent, {0: -1, 1: 1}
            )

        # Two distinct (eq_rows, ub_rows) pairs, with one equality and one
        # upper-matrix digest per pair.  This stays constant as the number of
        # property rows grows.
        self.assertEqual(digest.call_count, 4)
        self.assertEqual(
            len(child._solver_row_constraint_prefix_frames), 198
        )
        self.assertTrue(
            hz_verify_sparse_binary_phase_child(
                parent,
                {0: -1, 1: 1},
                child,
            )
        )

    def test_fraction_outward_substitution_contains_exact_projection(self):
        unit = 2.0**-54
        parent = SparseHZono(
            c=np.asarray([1.0], dtype=np.float64),
            Gc=sp.csr_matrix([[0.0]], dtype=np.float64),
            Gb=sp.csr_matrix(
                [[unit, unit, unit]],
                dtype=np.float64,
            ),
            Ac=sp.csr_matrix([[2.0]], dtype=np.float64),
            Ab=sp.csr_matrix(
                [[-unit, -unit, -unit]],
                dtype=np.float64,
            ),
            b=np.asarray([1.0], dtype=np.float64),
            Auc=sp.csr_matrix([[0.0]], dtype=np.float64),
            # Stored-order float CSR reduction loses ``-unit``:
            # 1 - unit - 1 evaluates to zero rather than -unit.
            Aub=sp.csr_matrix(
                [[1.0, -unit, -1.0]],
                dtype=np.float64,
            ),
            ub=np.asarray([1.0], dtype=np.float64),
            col_ids=np.asarray([10], dtype=np.int64),
            bcol_ids=np.asarray([20, 21, 22], dtype=np.int64),
        )
        fixed = np.ones(3, dtype=np.float64)
        old_float_upper = float(
            parent.ub[0]
            - np.asarray(parent.Aub @ fixed).reshape(-1)[0]
        )
        exact_unit = Fraction(1, 2**54)
        exact_center_and_rhs = Fraction(1) + 3 * exact_unit
        exact_upper_rhs = Fraction(1) + exact_unit
        self.assertEqual(old_float_upper, 1.0)
        self.assertLess(
            Fraction.from_float(old_float_upper),
            exact_upper_rhs,
        )

        child = hz_fix_sparse_binary_assignment(
            parent,
            {0: 1, 1: 1, 2: 1},
        )
        self.assertTrue(
            hz_verify_sparse_binary_phase_child(
                parent,
                {0: 1, 1: 1, 2: 1},
                child,
            )
        )
        receipt = child._solver_binary_phase_fix
        self.assertEqual(
            receipt["schema"],
            "sparse_hz_binary_phase_fix_v2",
        )
        self.assertEqual(
            receipt["projection_relation"],
            "exact_fixed_phase_projection_subset_of_child",
        )
        self.assertEqual(receipt["center_roundoff_generator_rows"], [0])
        self.assertEqual(
            receipt["equality_rhs_roundoff_generator_rows"],
            [0],
        )
        self.assertEqual(receipt["upper_rhs_outward_rounded_rows"], [0])
        self.assertEqual(receipt["roundoff_generator_count"], 2)
        self.assertEqual(child.n_cont, 3)
        self.assertEqual(
            len(set(child.col_ids.tolist())),
            child.n_cont,
        )

        center_radius = Fraction.from_float(float(child.Gc[0, 1]))
        equality_radius = Fraction.from_float(float(child.Ac[0, 2]))
        center_eta = (
            exact_center_and_rhs
            - Fraction.from_float(float(child.c[0]))
        ) / center_radius
        equality_eta = (
            Fraction.from_float(float(child.b[0]))
            - exact_center_and_rhs
        ) / equality_radius
        exact_xc = exact_center_and_rhs / 2
        child_xi = (exact_xc, center_eta, equality_eta)
        self.assertTrue(
            all(Fraction(-1) <= value <= Fraction(1) for value in child_xi)
        )

        def exact_sparse_row(matrix, row, vector):
            matrix = matrix.tocsr()
            total = Fraction(0)
            for offset in range(
                int(matrix.indptr[row]),
                int(matrix.indptr[row + 1]),
            ):
                total += (
                    Fraction.from_float(float(matrix.data[offset]))
                    * vector[int(matrix.indices[offset])]
                )
            return total

        child_output = (
            Fraction.from_float(float(child.c[0]))
            + exact_sparse_row(child.Gc, 0, child_xi)
        )
        self.assertEqual(child_output, exact_center_and_rhs)
        self.assertEqual(
            exact_sparse_row(child.Ac, 0, child_xi),
            Fraction.from_float(float(child.b[0])),
        )
        self.assertGreaterEqual(
            Fraction.from_float(float(child.ub[0])),
            exact_upper_rhs,
        )

        parent_output = (
            Fraction.from_float(float(parent.c[0]))
            + sum(
                Fraction.from_float(float(value))
                for value in parent.Gb.data
            )
        )
        self.assertEqual(parent_output, exact_center_and_rhs)
        self.assertEqual(
            2 * exact_xc - 3 * exact_unit,
            Fraction.from_float(float(parent.b[0])),
        )

    def test_live_phase_child_validator_rejects_all_tamper_surfaces(self):
        unit = 2.0**-54
        parent = SparseHZono(
            c=np.asarray([1.0], dtype=np.float64),
            Gc=sp.csr_matrix([[0.0]], dtype=np.float64),
            Gb=sp.csr_matrix(
                [[unit, unit, unit]],
                dtype=np.float64,
            ),
            Ac=sp.csr_matrix([[2.0]], dtype=np.float64),
            Ab=sp.csr_matrix(
                [[-unit, -unit, -unit]],
                dtype=np.float64,
            ),
            b=np.asarray([1.0], dtype=np.float64),
            Auc=sp.csr_matrix([[0.0]], dtype=np.float64),
            Aub=sp.csr_matrix(
                [[1.0, -unit, -1.0]],
                dtype=np.float64,
            ),
            ub=np.asarray([1.0], dtype=np.float64),
            col_ids=np.asarray([10], dtype=np.int64),
            bcol_ids=np.asarray([20, 21, 22], dtype=np.int64),
        )
        hz_mark_constructively_nonempty(parent, "live_validator_toy")
        assignment = {0: 1, 1: 1, 2: 1}
        child = hz_fix_sparse_binary_assignment(parent, assignment)
        self.assertTrue(
            hz_verify_sparse_binary_phase_child(
                parent,
                assignment,
                child,
            )
        )

        matrix_tamper = deepcopy(child)
        matrix_tamper.Gc = matrix_tamper.Gc.tolil()
        matrix_tamper.Gc[0, 1] = np.nextafter(
            float(matrix_tamper.Gc[0, 1]),
            0.0,
        )
        matrix_tamper.Gc = matrix_tamper.Gc.tocsr()

        rhs_tamper = deepcopy(child)
        rhs_tamper.ub[0] = 1.0

        receipt_tamper = deepcopy(child)
        receipt_tamper._solver_binary_phase_fix[
            "center_roundoff_radii_hex"
        ][0] = 0.0.hex()

        id_tamper = deepcopy(child)
        id_tamper.col_ids[1] = id_tamper.col_ids[0]

        token_tamper = deepcopy(child)
        token_tamper._solver_exact_phase_cover_member_token = (
            token_tamper._solver_binary_phase_fix
        )

        binary_tamper = deepcopy(child)
        binary_tamper.bcol_ids = np.asarray([999], dtype=np.int64)

        for name, tampered in (
            ("matrix", matrix_tamper),
            ("upper_rhs", rhs_tamper),
            ("receipt", receipt_tamper),
            ("ids", id_tamper),
            ("private_token", token_tamper),
            ("binary_columns", binary_tamper),
        ):
            with self.subTest(tamper=name):
                self.assertFalse(
                    hz_verify_sparse_binary_phase_child(
                        parent,
                        assignment,
                        tampered,
                    )
                )

    def test_live_validator_checks_conditional_trailing_error_columns(self):
        parent = SparseHZono(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Gb=sp.csr_matrix([[0.0]], dtype=np.float64),
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix([[0.0]], dtype=np.float64),
            b=np.asarray([0.0], dtype=np.float64),
            Auc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Aub=sp.csr_matrix([[0.0]], dtype=np.float64),
            ub=np.asarray([1.0], dtype=np.float64),
            col_ids=np.asarray([10], dtype=np.int64),
            bcol_ids=np.asarray([20], dtype=np.int64),
        )
        hz_mark_constructively_nonempty(parent, "conditional_validator_toy")
        _hz_attach_exact_phase_conditional_property_rows_from_operator(
            parent,
            [
                {
                    "binary_col_id": 20,
                    "phase": -1,
                    "layer_id": 7,
                    "row": 2,
                    "center": np.asarray([-0.5]),
                    "generator": sp.csr_matrix([[0.25]]),
                    "error": np.asarray([0.125]),
                    "rival_ids": (3,),
                },
                {
                    "binary_col_id": 20,
                    "phase": 1,
                    "layer_id": 7,
                    "row": 2,
                    "center": np.asarray([0.5]),
                    "generator": sp.csr_matrix([[0.25]]),
                    "error": np.asarray([0.125]),
                    "rival_ids": (3,),
                },
            ],
        )
        assignment = {0: 1}
        child = hz_fix_sparse_binary_assignment(parent, assignment)
        self.assertTrue(
            hz_verify_sparse_binary_phase_child(
                parent,
                assignment,
                child,
            )
        )
        self.assertEqual(child.n_out, 2)
        self.assertEqual(child.n_cont, 2)
        self.assertEqual(float(child.Ac[0, 1]), 0.0)
        self.assertEqual(float(child.Auc[0, 1]), 0.0)

        constraint_tamper = deepcopy(child)
        constraint_tamper.Ac = constraint_tamper.Ac.tolil()
        constraint_tamper.Ac[0, 1] = 0.125
        constraint_tamper.Ac = constraint_tamper.Ac.tocsr()
        self.assertFalse(
            hz_verify_sparse_binary_phase_child(
                parent,
                assignment,
                constraint_tamper,
            )
        )

        trailing_output_tamper = deepcopy(child)
        trailing_output_tamper.Gc = trailing_output_tamper.Gc.tolil()
        trailing_output_tamper.Gc[1, 0] = 0.5
        trailing_output_tamper.Gc = trailing_output_tamper.Gc.tocsr()
        self.assertFalse(
            hz_verify_sparse_binary_phase_child(
                parent,
                assignment,
                trailing_output_tamper,
            )
        )

    def test_fixed_phase_rlt_rows_remain_redundant_on_projection(self):
        # v = s*q signed-product hull, followed by the two RLT rows
        # (1+s)*(q-1/2)<=0 and (1-s)*(q-1/2)<=0.
        parent = SparseHZono(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0, 0.0]], dtype=np.float64),
            Gb=sp.csr_matrix([[0.0]], dtype=np.float64),
            Ac=sp.csr_matrix((0, 2), dtype=np.float64),
            Ab=sp.csr_matrix((0, 1), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix(
                [
                    [1.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ],
                dtype=np.float64,
            ),
            Aub=sp.csr_matrix(
                [[-1.0], [-1.0], [1.0], [1.0], [-0.5], [0.5]],
                dtype=np.float64,
            ),
            ub=np.asarray(
                [1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
                dtype=np.float64,
            ),
            col_ids=np.asarray([10, 11], dtype=np.int64),
            bcol_ids=np.asarray([20], dtype=np.int64),
        )

        for phase in (-1, 1):
            with self.subTest(phase=phase):
                child = hz_fix_sparse_binary_assignment(
                    parent,
                    {0: phase},
                )
                self.assertEqual(
                    child._solver_binary_phase_fix[
                        "roundoff_generator_count"
                    ],
                    0,
                )
                point = (Fraction(1, 2), Fraction(phase, 2))
                residuals = []
                for row in range(child.n_ub):
                    start = int(child.Auc.indptr[row])
                    end = int(child.Auc.indptr[row + 1])
                    lhs = sum(
                        (
                            Fraction.from_float(
                                float(child.Auc.data[offset])
                            )
                            * point[
                                int(child.Auc.indices[offset])
                            ]
                        )
                        for offset in range(start, end)
                    )
                    rhs = Fraction.from_float(float(child.ub[row]))
                    residuals.append(lhs - rhs)
                    self.assertLessEqual(lhs, rhs)
                redundant_rlt_row = 5 if phase == 1 else 4
                self.assertEqual(
                    residuals[redundant_rlt_row],
                    Fraction(0),
                )

    def test_nonfinite_duplicate_and_overflow_shifts_fail_closed(self):
        nonfinite = self._parent()
        nonfinite.c[0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            hz_fix_sparse_binary_assignment(nonfinite, {0: 1})

        duplicate_gb = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2], dtype=np.int32),
            ),
            shape=(1, 1),
        )
        duplicate_parent = SparseHZono(
            c=np.asarray([0.0]),
            Gc=sp.csr_matrix((1, 0)),
            Gb=duplicate_gb,
            Ac=sp.csr_matrix((0, 0)),
            Ab=sp.csr_matrix((0, 1)),
            b=np.zeros(0),
        )
        with self.assertRaisesRegex(ValueError, "canonical CSR"):
            hz_fix_sparse_binary_assignment(duplicate_parent, {0: 1})

        maximum = np.finfo(np.float64).max
        overflow_parent = SparseHZono(
            c=np.asarray([maximum]),
            Gc=sp.csr_matrix((1, 0)),
            Gb=sp.csr_matrix([[maximum]]),
            Ac=sp.csr_matrix((0, 0)),
            Ab=sp.csr_matrix((0, 1)),
            b=np.zeros(0),
        )
        with self.assertRaisesRegex(ValueError, "finite binary64"):
            hz_fix_sparse_binary_assignment(overflow_parent, {0: 1})

    def test_invalid_assignment_and_exponential_guard_fail_closed(self):
        parent = self._parent()
        with self.assertRaises(ValueError):
            hz_fix_sparse_binary_assignment(parent, {0: 0})
        with self.assertRaises(ValueError):
            hz_fix_sparse_binary_assignment(parent, {2: 1})
        with self.assertRaises(ValueError):
            hz_enumerate_sparse_binary_phase_cover(
                parent, max_children=2
            )

    def test_vacuous_grouped_safe_requires_private_exact_cover_capability(self):
        parent = SparseHZono(
            c=np.asarray([-1.0], dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix(np.asarray([[0.0]], dtype=np.float64)),
            Ac=sp.csr_matrix((0, 0), dtype=np.float64),
            Ab=sp.csr_matrix((0, 1), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix((0, 0), dtype=np.float64),
            Aub=sp.csr_matrix((0, 1), dtype=np.float64),
            ub=np.zeros(0, dtype=np.float64),
        )
        ordinary_child = hz_fix_sparse_binary_assignment(parent, {0: -1})
        # A copied public receipt cannot forge the module-private cover token.
        ordinary_child._solver_exact_phase_cover_member_token = (
            ordinary_child._solver_binary_phase_fix
        )
        verdict, _ = hz_objbound_decide(
            ordinary_child,
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            require_base_feasible=False,
            safe_row_groups=((0,),),
            expected_safe_group_count=1,
        )
        self.assertEqual(verdict, "UNKNOWN")

        hz_mark_constructively_nonempty(parent, "controlled_nonempty_parent")
        cover_child = hz_fix_sparse_binary_assignment(parent, {0: -1})
        verdict, _ = hz_objbound_decide(
            cover_child,
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            is_unsafe_linear=False,
            time_limit=1.0,
            require_base_feasible=False,
            safe_row_groups=((0,),),
            expected_safe_group_count=1,
        )
        self.assertEqual(verdict, "SAFE")
        stats = cover_child._solver_objbound_stats
        self.assertEqual(
            stats["base_feasibility_status"],
            "EXACT_COVER_MEMBER_NOT_REQUIRED",
        )
        self.assertTrue(stats["exact_phase_cover_member"])


class DuplicateReluPhaseCoverTests(unittest.TestCase):
    def test_two_suffix_relus_export_joint_four_quadrant_planes(self):
        net = _verified_duplicate_relu_residual_net(
            joint_suffix_relus=True
        )
        exact = _exact_graph_range(
            SimpleNamespace(
                net=net,
                input_lb=Fraction(-1),
                input_ub=Fraction(1),
            )
        )
        self.assertEqual(exact.lower, Fraction(0))
        self.assertEqual(exact.upper, Fraction(0))

        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                net,
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=10.0,
                        engine="operator_hz_objbound",
                        operator_exact_budget=2,
                        property_residual_budget=2,
                        property_residual_time_limit=1.0,
                        property_residual_max_adjoint_cells=128,
                        property_residual_pool_per_rival=4,
                        property_tail_upper=True,
                        property_tail_suffix_blocks=1,
                        property_tail_suffix_alpha_steps=1,
                        property_tail_suffix_alpha_time_limit=1.0,
                        property_tail_suffix_alpha_device="cpu",
                        lp_prefilter_fraction=0.0,
                        lp_prefilter_max_seconds=0.0,
                    ),
                ),
            )[0]
            selector = result.metadata["property_residual_selector"]
            self.assertEqual(selector["targets_selected"], 2)
            self.assertEqual(
                {item["layer_id"] for item in selector["schedule"]},
                {7, 10},
            )
            conditional = result.metadata["operator_hz"][
                "property_tail_upper"
            ]["shared_suffix_replay"][
                "exact_phase_conditional_suffix"
            ]
            self.assertEqual(conditional["status"], "applied")
            self.assertTrue(conditional["proof_authority"])
            self.assertEqual(conditional["joint_depth"], 2)
            self.assertEqual(len(conditional["assignments"]), 4)
            self.assertEqual(
                {
                    tuple(
                        guard["phase"]
                        for guard in assignment["binary_guards"]
                    )
                    for assignment in conditional["assignments"]
                },
                {(-1, -1), (-1, 1), (1, -1), (1, 1)},
            )
            phase = result.metadata["property_phase_split"]
            self.assertEqual(phase["actual_child_count"], 4)
            self.assertEqual(phase["expected_child_count"], 4)
            self.assertTrue(phase["all_assignments_enumerated"])
            self.assertTrue(
                all(
                    child["error"] is None
                    for child in phase["focused_rival_preflight"][
                        "children"
                    ]
                )
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_interior_suffix_exact_phase_exports_guarded_replay_rows(self):
        net = _verified_duplicate_relu_residual_net(
            interior_suffix_relu=True
        )
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            result = verify_once(
                net,
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=8.0,
                        engine="operator_hz_objbound",
                        operator_exact_budget=1,
                        property_residual_budget=1,
                        property_residual_time_limit=1.0,
                        property_residual_max_adjoint_cells=64,
                        property_residual_pool_per_rival=2,
                        property_tail_upper=True,
                        property_tail_suffix_blocks=1,
                        property_tail_suffix_alpha_steps=1,
                        property_tail_suffix_alpha_time_limit=1.0,
                        property_tail_suffix_alpha_device="cpu",
                        lp_prefilter_fraction=0.0,
                        lp_prefilter_max_seconds=0.0,
                    ),
                ),
            )[0]
            selector = result.metadata["property_residual_selector"]
            self.assertTrue(
                selector["conditional_suffix_replay_requested"]
            )
            self.assertTrue(
                selector["selected_layers_strictly_after_stop"]
            )
            self.assertEqual(selector["schedule"][0]["layer_id"], 7)
            conditional = result.metadata["operator_hz"][
                "property_tail_upper"
            ]["shared_suffix_replay"][
                "exact_phase_conditional_suffix"
            ]
            self.assertEqual(conditional["status"], "applied")
            self.assertTrue(conditional["proof_authority"])
            phase = result.metadata["property_phase_split"]
            focus = phase["focused_rival_preflight"]
            # Baseline + ordinary suffix + one matching conditional plane.
            self.assertEqual(len(focus["output_rows"]), 3)
            self.assertTrue(
                all(child["error"] is None for child in focus["children"])
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    @staticmethod
    def _toy():
        input_layer, input_spec = _input_layers(-1, 1)
        layers = [
            input_layer,
            input_spec,
            _layer(2, "RELU"),
            _layer(3, "RELU"),
            _dense(4, -1, 0),
            _layer(5, "ADD"),
            _dense(6, 1, Fraction(-1, 10)),
            _layer(7, "ASSERT", {"kind": "UNSAFE_LINEAR"}),
        ]
        return _assemble_scalar_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [1],
                4: [3],
                5: [2, 4],
                6: [5],
                7: [6],
            },
            input_lb=-1,
            input_ub=1,
        )

    def test_one_exact_relu_two_continuous_children_close_relaxation_gap(self):
        toy = self._toy()
        exact_range = _exact_graph_range(toy)
        stored_bias = Fraction.from_float(-0.1)
        self.assertEqual(exact_range.lower, stored_bias)
        self.assertEqual(exact_range.upper, stored_bias)

        relaxed = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
        )
        relaxed_lower, relaxed_upper = _lp_output_range(relaxed)
        self.assertLess(relaxed_lower, -0.09)
        self.assertGreater(relaxed_upper, 0.25)

        exact = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=1,
            materialize_add=True,
        )
        self.assertEqual(exact.hz.n_bin, 1)
        children = hz_enumerate_sparse_binary_phase_cover(
            exact.hz, max_children=2
        )
        self.assertEqual(len(children), 2)

        child_uppers = []
        for assignment, child in children:
            self.assertIn(assignment[0][1], {-1, 1})
            self.assertEqual(child.n_bin, 0)
            status, reason = hz_base_feasibility(
                child, time_limit=2.0
            )
            self.assertEqual(status, "FEASIBLE", reason)
            lower, upper = _lp_output_range(
                type("_Build", (), {"hz": child})()
            )
            child_uppers.append(upper)
            self.assertLessEqual(upper, -0.099999999)
            verdict, witness = hz_objbound_decide(
                child,
                np.asarray([[1.0]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
                is_unsafe_linear=False,
                time_limit=5.0,
                require_base_feasible=True,
            )
            self.assertEqual(verdict, "SAFE")
            self.assertIsNone(witness)
        self.assertLessEqual(max(child_uppers), -0.099999999)

    def test_verifier_parallel_phase_cover_certifies_when_baseline_is_unknown(self):
        net = _verified_duplicate_relu_residual_net()
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            baseline = verify_once(
                net,
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=5.0,
                        engine="operator_hz_objbound",
                        property_tail_upper=True,
                        property_tail_suffix_blocks=1,
                        lp_prefilter_fraction=1.0,
                        lp_prefilter_max_seconds=2.0,
                    ),
                ),
            )[0]
            self.assertEqual(baseline.status, VerifyStatus.UNKNOWN)

            split = verify_once(
                net,
                backend_cfg=BackendConfig(
                    solver="hybridz",
                    device="cpu",
                    dtype="float64",
                    hybridz=HybridZConfig(
                        timeout=5.0,
                        engine="operator_hz_objbound",
                        operator_exact_budget=1,
                        property_residual_budget=1,
                        property_residual_time_limit=1.0,
                        property_residual_max_adjoint_cells=32,
                        property_residual_pool_per_rival=2,
                        property_tail_upper=True,
                        property_tail_suffix_blocks=1,
                        lp_prefilter_fraction=1.0,
                        lp_prefilter_max_seconds=2.0,
                    ),
                ),
            )[0]
            self.assertEqual(split.status, VerifyStatus.CERTIFIED)
            self.assertEqual(split.metadata["hz_verdict"], "SAFE")
            selector = split.metadata["property_residual_selector"]
            self.assertEqual(
                selector["schema"], "property_phase_split_selector_v1"
            )
            self.assertEqual(selector["suffix_stop_layer_id"], 5)
            self.assertTrue(
                selector["selected_layers_strictly_before_stop"]
            )
            receipt = split.metadata["property_phase_split"]
            self.assertEqual(receipt["status"], "all_children_safe")
            self.assertTrue(receipt["proof_authority"])
            self.assertEqual(receipt["binary_depth"], 1)
            self.assertEqual(receipt["actual_child_count"], 2)
            self.assertTrue(receipt["all_assignments_enumerated"])
            self.assertTrue(receipt["children_run_in_parallel"])
            self.assertEqual(
                [child["verdict"] for child in receipt["children"]],
                ["SAFE", "SAFE"],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_one_unknown_child_blocks_cover_safe(self):
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            with patch(
                "act.back_end.solver.solver_hz.hz_objbound_decide",
                side_effect=(("SAFE", None), ("UNKNOWN", None)),
            ):
                result = verify_once(
                    _verified_duplicate_relu_residual_net(),
                    backend_cfg=BackendConfig(
                        solver="hybridz",
                        device="cpu",
                        dtype="float64",
                        hybridz=HybridZConfig(
                            timeout=5.0,
                            engine="operator_hz_objbound",
                            operator_exact_budget=1,
                            property_residual_budget=1,
                            property_residual_time_limit=1.0,
                            property_residual_max_adjoint_cells=32,
                            property_residual_pool_per_rival=2,
                            property_tail_upper=True,
                            property_tail_suffix_blocks=1,
                            lp_prefilter_fraction=1.0,
                            lp_prefilter_max_seconds=2.0,
                        ),
                    ),
                )[0]
            self.assertEqual(result.status, VerifyStatus.UNKNOWN)
            self.assertEqual(result.metadata["hz_verdict"], "UNKNOWN")
            receipt = result.metadata["property_phase_split"]
            self.assertEqual(
                receipt["status"], "focused_rival_unresolved"
            )
            self.assertFalse(receipt["proof_authority"])
            self.assertCountEqual(
                [child["verdict"] for child in receipt["children"]],
                ["SAFE", "UNKNOWN"],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_phase_cover_configuration_contract_is_explicit(self):
        valid = HybridZConfig(
            engine="operator_hz_objbound",
            operator_exact_budget=1,
            property_residual_budget=1,
            property_residual_time_limit=1.0,
            property_tail_upper=True,
            property_tail_suffix_blocks=1,
        )
        self.assertEqual(valid.operator_exact_budget, 1)
        for field in (
            "operator_exact_budget",
            "property_residual_budget",
        ):
            for value in (True, 1.5, "1"):
                with self.subTest(field=field, value=value):
                    kwargs = {
                        "engine": "operator_hz_objbound",
                        "operator_exact_budget": 1,
                        "property_residual_budget": 1,
                        "property_residual_time_limit": 1.0,
                        "property_tail_upper": True,
                        "property_tail_suffix_blocks": 1,
                    }
                    kwargs[field] = value
                    with self.assertRaisesRegex(
                        ValueError, "must be an integer"
                    ):
                        HybridZConfig(**kwargs)
        for value in (True, "1"):
            with self.subTest(
                property_residual_time_limit=value
            ):
                with self.assertRaisesRegex(
                    ValueError, "must be numeric"
                ):
                    HybridZConfig(
                        engine="operator_hz_objbound",
                        operator_exact_budget=1,
                        property_residual_budget=1,
                        property_residual_time_limit=value,
                        property_tail_upper=True,
                        property_tail_suffix_blocks=1,
                    )
        with self.assertRaisesRegex(ValueError, "depth 1 or 2"):
            HybridZConfig(
                engine="operator_hz_objbound",
                operator_exact_budget=3,
                property_residual_budget=3,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
            )
        with self.assertRaisesRegex(
            ValueError, "property_residual_budget=operator_exact_budget"
        ):
            HybridZConfig(
                engine="operator_hz_objbound",
                operator_exact_budget=2,
                property_residual_budget=1,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=1,
            )
        with self.assertRaisesRegex(ValueError, "suffix_blocks in"):
            HybridZConfig(
                engine="operator_hz_objbound",
                operator_exact_budget=1,
                property_residual_budget=1,
                property_residual_time_limit=1.0,
                property_tail_upper=True,
                property_tail_suffix_blocks=0,
            )


if __name__ == "__main__":
    unittest.main()
