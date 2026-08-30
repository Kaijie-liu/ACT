#!/usr/bin/env python3
"""Toy gates for the disconnected forward compact exact-ReLU census."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.hybridz_tf import forward_exact_relu_census as census_module
from act.back_end.hybridz_tf.forward_exact_relu_census import (
    ForwardExactReLUCensus,
    MAX_SYNTHETIC_WIDTH,
    PAYLOAD_TRAVERSAL_CONTRACT,
    run_synthetic_exact_relu_census,
    sparse_hz_payload_breakdown,
)


def _record(receipt, scenario: str, label: str):
    matches = [
        row
        for row in receipt["records"]
        if row["scenario"] == scenario and row["label"] == label
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one record for {scenario}/{label}, got {len(matches)}"
        )
    return matches[0]


class ForwardExactReLUCensusTest(unittest.TestCase):
    def test_mixed_stable_unstable_obeys_exact_cell_cost_law(self):
        trace = ForwardExactReLUCensus("mixed_gate")
        before = trace.box(
            "input",
            [0.25, -2.0, -1.0, -3.0],
            [1.0, -0.25, 1.0, 2.0],
        )
        out = trace.relu("relu", before)
        receipt = trace.receipt()
        row = _record(receipt, "mixed_gate", "relu")
        detail = row["details"]

        self.assertEqual(detail["active"], 1)
        self.assertEqual(detail["inactive"], 1)
        self.assertEqual(detail["unstable"], 2)
        self.assertEqual(
            detail["actual_delta"],
            {"C": 4, "B": 2, "E": 2, "U": 4, "constraint_nnz": 16},
        )
        self.assertEqual(detail["actual_delta"], detail["expected_delta"])
        self.assertTrue(detail["invariant_passed"])
        self.assertEqual(out.n_cont - before.n_cont, 4)
        self.assertEqual(out.n_bin - before.n_bin, 2)

    def test_sparse_affine_medium_support_accounts_for_p_plus_seven(self):
        width = 128
        trace = ForwardExactReLUCensus("medium_sparse_affine")
        root = trace.box("input", -np.ones(width), np.ones(width))
        diagonal = sp.eye(width, format="csr", dtype=np.float64)
        neighbor = sp.csr_matrix(
            (
                -0.25 * np.ones(width),
                (
                    np.arange(width),
                    (np.arange(width) + 1) % width,
                ),
            ),
            shape=(width, width),
        )
        pre = trace.affine("affine", root, diagonal + neighbor)
        out = trace.relu("relu", pre)
        row = _record(trace.receipt(), "medium_sparse_affine", "relu")
        detail = row["details"]

        self.assertEqual(detail["unstable"], width)
        self.assertEqual(detail["preactivation_support_nnz"], 2 * width)
        self.assertEqual(detail["actual_delta"]["C"], 2 * width)
        self.assertEqual(detail["actual_delta"]["B"], width)
        self.assertEqual(detail["actual_delta"]["E"], width)
        self.assertEqual(detail["actual_delta"]["U"], 2 * width)
        self.assertEqual(detail["actual_delta"]["constraint_nnz"], 9 * width)
        self.assertEqual(out.constraint_nnz, 9 * width)

    def test_identity_residual_keeps_exact_branch_constraints_once(self):
        receipt = run_synthetic_exact_relu_census(width=8)
        branch = _record(receipt, "identity_residual", "identity_relu_branch")
        joined = _record(receipt, "identity_residual", "identity_join")

        for key in ("C", "B", "E", "U", "constraint_nnz"):
            self.assertEqual(joined[key], branch[key])
        self.assertEqual(joined["details"]["residual_kind"], "identity")

    def test_fork_fork_census_exposes_current_shared_prefix_duplication(self):
        width = 8
        receipt = run_synthetic_exact_relu_census(width=width)
        common = _record(receipt, "fork_fork_residual", "fork_common_relu")
        left = _record(receipt, "fork_fork_residual", "fork_left_relu")
        right = _record(receipt, "fork_fork_residual", "fork_right_relu")
        joined = _record(receipt, "fork_fork_residual", "fork_join")

        self.assertEqual((common["E"], common["U"]), (width, 2 * width))
        self.assertEqual((left["E"], left["U"]), (2 * width, 4 * width))
        self.assertEqual((right["E"], right["U"]), (2 * width, 4 * width))
        # The current whole-prefix merge cannot factor P from [P,L] and [P,R].
        self.assertEqual((joined["E"], joined["U"]), (4 * width, 8 * width))
        self.assertEqual((joined["C"], joined["B"]), (7 * width, 3 * width))
        self.assertGreater(joined["constraint_nnz"], left["constraint_nnz"])
        self.assertEqual(joined["details"]["residual_kind"], "fork_fork")

    def test_receipt_is_json_safe_and_explicitly_non_authoritative(self):
        receipt = run_synthetic_exact_relu_census(width=16)
        encoded = json.dumps(receipt, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        contract = decoded["execution_contract"]

        self.assertFalse(decoded["authoritative"])
        self.assertFalse(decoded["production_integration"])
        self.assertEqual(decoded["scope"], "synthetic_only")
        self.assertEqual(contract["direction"], "forward_only")
        self.assertEqual(contract["relu_semantics"], "compact_exact_binary_hz")
        self.assertTrue(contract["compressed"])
        self.assertFalse(contract["valid_cuts"])
        self.assertFalse(contract["solver_called"])
        self.assertFalse(contract["real_dataset_loaded"])
        self.assertTrue(
            all(
                called is False
                for called in contract["prohibited_mechanisms_called"].values()
            )
        )
        self.assertEqual(
            decoded["scenarios"],
            [
                "mixed_stable_unstable",
                "sparse_affine_then_relu",
                "identity_residual",
                "fork_fork_residual",
            ],
        )
        self.assertTrue(all(math.isfinite(row["wall_seconds"]) for row in decoded["records"]))
        required = {
            "wall_seconds",
            "C",
            "B",
            "E",
            "U",
            "constraint_nnz",
            "value_nnz",
            "payload_bytes",
        }
        self.assertTrue(all(required <= set(row) for row in decoded["records"]))

    def test_all_relu_calls_fix_compressed_exact_options(self):
        real_relu = census_module.sparse_hz_apply_relu_exact
        with mock.patch.object(
            census_module,
            "sparse_hz_apply_relu_exact",
            wraps=real_relu,
        ) as exact_relu:
            run_synthetic_exact_relu_census(width=4)

        self.assertGreater(exact_relu.call_count, 0)
        for call in exact_relu.call_args_list:
            self.assertIs(call.kwargs["compressed"], True)
            self.assertIs(call.kwargs["valid_cuts"], False)
            self.assertIs(call.kwargs["return_info"], True)

    def test_payload_counts_dynamic_arrays_and_deduplicates_aliases(self):
        trace = ForwardExactReLUCensus("payload")
        hz = trace.box("input", [-1.0, -2.0], [1.0, 2.0])
        payload = sparse_hz_payload_breakdown(hz)

        # Independent fixed-width oracle.  The 48 dense bytes are c (16),
        # col_ids (16), and dynamic full_col_ids (16); empty vectors cost 0.
        # Six CSR matrices contribute 64 bytes across their explicit buffers.
        self.assertEqual(payload, {"dense_bytes": 48, "csr_bytes": 64, "payload_bytes": 112})

        hz.full_ids_alias = hz.full_col_ids
        hz.matrix_alias = hz.Gc
        hz.nested_payload = {
            "same_view": hz.full_col_ids[:],
            "fresh": np.ones(3, dtype=np.float64),
        }
        with_aliases = sparse_hz_payload_breakdown(hz)
        self.assertEqual(with_aliases["dense_bytes"], 72)
        self.assertEqual(with_aliases["csr_bytes"], 64)
        self.assertEqual(with_aliases["payload_bytes"], 136)

    def test_payload_traverses_plain_dict_objects_and_namespace_cycles(self):
        class PlainPayload:
            pass

        trace = ForwardExactReLUCensus("object_payload")
        hz = trace.box("input", [-1.0, -2.0], [1.0, 2.0])
        plain = PlainPayload()
        plain.payload = np.ones(2, dtype=np.float64)
        namespace = SimpleNamespace(
            payload=np.ones(3, dtype=np.float64),
            child=plain,
        )
        plain.parent = namespace
        hz.object_payload = namespace

        payload = sparse_hz_payload_breakdown(hz)
        self.assertEqual(payload["dense_bytes"], 88)
        self.assertEqual(payload["csr_bytes"], 64)
        self.assertEqual(payload["payload_bytes"], 152)
        self.assertIsInstance(PAYLOAD_TRAVERSAL_CONTRACT, tuple)

        class OpaqueSlots:
            __slots__ = ("payload",)

        opaque = OpaqueSlots()
        opaque.payload = np.ones(1, dtype=np.float64)
        hz.opaque_payload = opaque
        with self.assertRaisesRegex(ValueError, "opaque hz payload object"):
            sparse_hz_payload_breakdown(hz)

    def test_payload_unions_overlapping_views_of_one_ultimate_backing(self):
        trace = ForwardExactReLUCensus("overlap_payload")
        hz = trace.box("input", [-1.0, -2.0], [1.0, 2.0])
        backing = bytearray(64)
        left = np.frombuffer(backing, dtype=np.float64, count=6, offset=0)
        right = np.frombuffer(backing, dtype=np.float64, count=6, offset=16)
        hz.overlap_payload = SimpleNamespace(left=left, right=right)

        payload = sparse_hz_payload_breakdown(hz)
        # Each view is 48 bytes, but their union on one backing is 64 bytes.
        self.assertEqual(payload["dense_bytes"], 112)
        self.assertEqual(payload["csr_bytes"], 64)
        self.assertEqual(payload["payload_bytes"], 176)
        self.assertNotEqual(payload["payload_bytes"], 208)

    def test_payload_explicit_stack_handles_1200_attribute_layers(self):
        trace = ForwardExactReLUCensus("deep_payload")
        hz = trace.box("input", [-1.0, -2.0], [1.0, 2.0])
        node = SimpleNamespace(payload=np.ones(1, dtype=np.float64))
        for _ in range(1200):
            node = SimpleNamespace(child=node)
        hz.deep_payload = node

        payload = sparse_hz_payload_breakdown(hz)
        self.assertEqual(payload["dense_bytes"], 56)
        self.assertEqual(payload["csr_bytes"], 64)
        self.assertEqual(payload["payload_bytes"], 120)

    def test_payload_rejects_container_and_ndarray_subclasses(self):
        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        class ArraySubclass(np.ndarray):
            pass

        payloads = (
            DictSubclass(payload=np.ones(1, dtype=np.float64)),
            ListSubclass([np.ones(1, dtype=np.float64)]),
            np.ones(1, dtype=np.float64).view(ArraySubclass),
        )
        for index, bad_payload in enumerate(payloads):
            trace = ForwardExactReLUCensus(f"subclass_{index}")
            hz = trace.box("input", [-1.0, -2.0], [1.0, 2.0])
            hz.bad_payload = bad_payload
            with self.subTest(payload_type=type(bad_payload).__name__):
                with self.assertRaisesRegex(ValueError, "subclass of an exact"):
                    sparse_hz_payload_breakdown(hz)

    def test_payload_rejects_negative_stride_even_for_shape_one(self):
        trace = ForwardExactReLUCensus("negative_stride")
        hz = trace.box("input", [-1.0, -2.0], [1.0, 2.0])
        reversed_singleton = np.ones(1, dtype=np.float64)[::-1]
        self.assertEqual(reversed_singleton.shape, (1,))
        self.assertEqual(reversed_singleton.strides, (-8,))
        self.assertTrue(reversed_singleton.flags.c_contiguous)
        hz.negative_stride = reversed_singleton

        with self.assertRaisesRegex(ValueError, "negative strides"):
            sparse_hz_payload_breakdown(hz)

    def test_box_rejects_coercive_or_nonfinite_numeric_inputs(self):
        trace = ForwardExactReLUCensus("bad_box")
        bad_pairs = (
            ([float("nan"), 0.0], [1.0, 2.0]),
            ([-1.0, 0.0], [float("inf"), 2.0]),
            (np.asarray([-1.0, 0.0], dtype=np.float32), np.ones(2, dtype=np.float64)),
            (np.asarray([-1.0 + 1.0j, 0.0]), np.ones(2, dtype=np.float64)),
            ([-1, 0], [1, 2]),
        )
        with mock.patch.object(
            census_module,
            "sparse_hz_from_bounds",
            side_effect=AssertionError("invalid box reached production primitive"),
        ) as from_bounds:
            for lower, upper in bad_pairs:
                with self.subTest(lower=np.asarray(lower).dtype):
                    with self.assertRaises(ValueError):
                        trace.box("bad", lower, upper)
            from_bounds.assert_not_called()

    def test_affine_rejects_complex_wrong_dtype_and_nonfinite_before_call(self):
        trace = ForwardExactReLUCensus("bad_affine")
        root = trace.box("input", [-1.0, -1.0], [1.0, 1.0])
        good = np.eye(2, dtype=np.float64)
        sparse_inf = sp.eye(2, format="csr", dtype=np.float64)
        sparse_inf.data[0] = np.inf
        bad_calls = (
            (np.eye(2, dtype=np.complex128) * (1.0 + 1.0j), None),
            (sp.eye(2, format="csr", dtype=np.complex128) * (1.0 + 1.0j), None),
            (np.asarray([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64), None),
            (sparse_inf, None),
            (np.eye(2, dtype=np.float32), None),
            (good, np.asarray([0.0, np.nan], dtype=np.float64)),
            (good, np.asarray([0.0, 0.0], dtype=np.float32)),
            (good, np.asarray([0.0 + 1.0j, 0.0])),
        )
        with mock.patch.object(
            census_module,
            "sparse_hz_linear",
            side_effect=AssertionError("invalid affine reached production primitive"),
        ) as linear:
            for weight, bias in bad_calls:
                with self.subTest(weight_dtype=weight.dtype, bias=repr(bias)):
                    with self.assertRaises(ValueError):
                        trace.affine("bad", root, weight, bias)
            linear.assert_not_called()

    def test_relu_rejects_invalid_pre_bounds_before_exact_graph_call(self):
        trace = ForwardExactReLUCensus("bad_bounds")
        root = trace.box("input", [-1.0, -1.0], [1.0, 1.0])
        bad_bounds = (
            Bounds(
                torch.tensor([[float("nan"), -1.0]], dtype=torch.float64),
                torch.ones((1, 2), dtype=torch.float64),
            ),
            Bounds(
                -torch.ones((1, 2), dtype=torch.float64),
                torch.tensor([[1.0, float("inf")]], dtype=torch.float64),
            ),
            Bounds(
                -torch.ones((1, 2), dtype=torch.float32),
                torch.ones((1, 2), dtype=torch.float32),
            ),
            Bounds(
                torch.tensor([[-1.0 + 1.0j, -1.0]], dtype=torch.complex128),
                torch.ones((1, 2), dtype=torch.complex128),
            ),
            Bounds(
                torch.tensor([[0.5, -1.0]], dtype=torch.float64),
                torch.tensor([[0.25, 1.0]], dtype=torch.float64),
            ),
        )
        with mock.patch.object(
            census_module,
            "sparse_hz_apply_relu_exact",
            side_effect=AssertionError("invalid bounds reached exact graph primitive"),
        ) as exact_relu:
            for bounds in bad_bounds:
                with self.subTest(dtype=bounds.lb.dtype):
                    with self.assertRaises(ValueError):
                        trace.relu("bad", root, bounds)
            exact_relu.assert_not_called()

    def test_public_hz_inputs_reject_nonfinite_or_complex_payload(self):
        trace = ForwardExactReLUCensus("bad_hz")
        nonfinite = trace.box("finite_input", [-1.0, -1.0], [1.0, 1.0])
        nonfinite.c[0] = np.nan
        with self.assertRaises(ValueError):
            sparse_hz_payload_breakdown(nonfinite)
        with self.assertRaises(ValueError):
            trace.relu("bad_relu", nonfinite)

        complex_hz = trace.box("second_input", [-1.0, -1.0], [1.0, 1.0])
        complex_hz.Gc = complex_hz.Gc.astype(np.complex128)
        with self.assertRaises(ValueError):
            trace.affine("bad_affine", complex_hz, np.eye(2, dtype=np.float64))

        right = trace.box("right_input", [-1.0, -1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            trace.add("bad_add", complex_hz, right, residual_kind="identity")

    def test_synthetic_width_guard_excludes_large_runs(self):
        with self.assertRaises(ValueError):
            run_synthetic_exact_relu_census(width=1)
        with self.assertRaises(ValueError):
            run_synthetic_exact_relu_census(width=MAX_SYNTHETIC_WIDTH + 1)
        with self.assertRaises(ValueError):
            run_synthetic_exact_relu_census(width=True)
        with self.assertRaises(ValueError):
            run_synthetic_exact_relu_census(width=np.int64(8))


if __name__ == "__main__":
    unittest.main()
