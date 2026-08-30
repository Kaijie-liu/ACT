#!/usr/bin/env python3
"""Toy-first gates for the default-off Operator-HZ micro-RLT hook."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Dict, Optional, Sequence, Tuple
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.hybridz_tf import property_micro_rlt as micro_rlt
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuildError,
    _OperatorHZBuilder,
    _PropertySuffixAddSourceSnapshot,
    build_operator_hz,
)
from act.back_end.hybridz_tf.property_micro_rlt import (
    PropertyMicroRLTResult,
    verify_property_micro_rlt_result,
)
from act.back_end.hybridz_tf.test_operator_add_fusion import (
    _assemble_width_toy,
    _dense_matrix,
    _input_layers,
    _layer,
    _wide_layer,
)
from act.back_end.solver.solver_hz import (
    hz_constructively_nonempty,
    hz_known_nonempty,
)


def _duplicate_relu_toy():
    """Return y2-y1 for two exact copies of ReLU(x), x in [-1,1]."""

    input_layer, spec = _input_layers(-1, 1)
    layers = [
        input_layer,
        spec,
        _dense_matrix(2, [[1], [1]], [0, 0]),
        _wide_layer(3, "RELU", 2),
        _dense_matrix(4, [[-1, 1]], [0]),
        _layer(5, "ASSERT"),
    ]
    return _assemble_width_toy(
        layers,
        {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        input_lb=-1,
        input_ub=1,
    )


def _build_kwargs() -> Dict[str, Any]:
    return {
        "exact_budget": 2,
        "materialize_add": True,
        "residual_targets": [
            (3, 0, "none"),
            (3, 1, "none"),
        ],
        # These IDs are selection metadata only in this graph-output toy.
        # They deliberately do not create a property-tail output.
        "property_phase_focus_rivals": {
            (3, 0): (0,),
            (3, 1): (0,),
        },
    }


def _build(cap: Optional[int] = None, **overrides):
    toy = _duplicate_relu_toy()
    kwargs = _build_kwargs()
    kwargs.update(overrides)
    if cap is not None:
        kwargs["property_micro_rlt_product_cap"] = int(cap)
    return build_operator_hz(
        toy.net,
        toy.facts,
        toy.facts,
        **kwargs,
    )


def _csr_bits(matrix: sp.csr_matrix) -> Tuple[Any, ...]:
    value = sp.csr_matrix(matrix, dtype=np.float64, copy=False)
    return (
        value.shape,
        value.indptr.tobytes(),
        value.indices.tobytes(),
        value.data.tobytes(),
    )


def _hz_bits(hz, *, include_ids: bool) -> Tuple[Any, ...]:
    result = (
        np.asarray(hz.c, dtype=np.float64).tobytes(),
        _csr_bits(hz.Gc),
        _csr_bits(hz.Gb),
        _csr_bits(hz.Ac),
        _csr_bits(hz.Ab),
        np.asarray(hz.b, dtype=np.float64).tobytes(),
        _csr_bits(hz.Auc),
        _csr_bits(hz.Aub),
        np.asarray(hz.ub, dtype=np.float64).tobytes(),
    )
    if not include_ids:
        return result
    return result + (
        np.asarray(hz.col_ids, dtype=np.int64).tobytes(),
        np.asarray(hz.bcol_ids, dtype=np.int64).tobytes(),
        np.asarray(hz.full_col_ids, dtype=np.int64).tobytes(),
        np.asarray(
            hz._solver_continuous_column_layer_ids,
            dtype=np.int64,
        ).tobytes(),
        tuple(hz._solver_constraint_row_tags),
        np.asarray(hz.operator_input_center, dtype=np.float64).tobytes(),
        np.asarray(hz.operator_input_radius, dtype=np.float64).tobytes(),
    )


def _deterministic_id_allocator(start: int = 10_000_000):
    next_id = [int(start)]

    def allocate(count: int, device=None) -> torch.Tensor:
        count = int(count)
        first = next_id[0]
        next_id[0] += count
        return torch.arange(
            first,
            first + count,
            dtype=torch.long,
            device=device,
        )

    return allocate


def _relaxed_output_range(
    build,
    *,
    fixed_binary: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    hz = build.hz
    if hz.n_out != 1:
        raise AssertionError("micro-RLT toy requires one output")
    objective = np.concatenate(
        [
            hz.Gc.getrow(0).toarray().reshape(-1),
            hz.Gb.getrow(0).toarray().reshape(-1),
        ]
    )
    upper = sp.hstack([hz.Auc, hz.Aub], format="csr")
    equality = sp.hstack([hz.Ac, hz.Ab], format="csr")
    if fixed_binary is None:
        binary_bounds = [(-1.0, 1.0)] * hz.n_bin
    else:
        values = tuple(float(value) for value in fixed_binary)
        if len(values) != hz.n_bin:
            raise AssertionError("fixed assignment has wrong width")
        binary_bounds = [(value, value) for value in values]
    bounds = [(-1.0, 1.0)] * hz.n_cont + binary_bounds
    common = {
        "A_ub": upper if hz.n_ub else None,
        "b_ub": hz.ub if hz.n_ub else None,
        "A_eq": equality if hz.n_eq else None,
        "b_eq": hz.b if hz.n_eq else None,
        "bounds": bounds,
        "method": "highs",
    }
    minimum = linprog(objective, **common)
    maximum = linprog(-objective, **common)
    if not minimum.success or not maximum.success:
        raise AssertionError(
            "independent relaxed LP failed: "
            f"min={minimum.status}/{minimum.message}; "
            f"max={maximum.status}/{maximum.message}"
        )
    return (
        float(hz.c[0] + minimum.fun),
        float(hz.c[0] - maximum.fun),
    )


def _receipt_hash_valid(receipt: Dict[str, Any]) -> bool:
    expected = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return expected == actual


class OperatorPropertyMicroRLTTests(unittest.TestCase):
    def test_default_and_explicit_off_are_bit_identical(self) -> None:
        target = (
            "act.back_end.hybridz_tf.operator_hz.hz_fresh_col_ids"
        )
        with patch(
            target, new=_deterministic_id_allocator()
        ):
            implicit = _build()
        with patch(
            target, new=_deterministic_id_allocator()
        ):
            explicit = _build(0)

        self.assertEqual(
            _hz_bits(implicit.hz, include_ids=True),
            _hz_bits(explicit.hz, include_ids=True),
        )
        self.assertEqual(
            implicit.metadata["property_micro_rlt"],
            explicit.metadata["property_micro_rlt"],
        )
        receipt = implicit.metadata["property_micro_rlt"]
        self.assertEqual(receipt["status"], "no_op_disabled")
        self.assertFalse(receipt["enabled"])
        self.assertTrue(_receipt_hash_valid(receipt))
        self.assertFalse(
            hasattr(implicit.hz, "_property_micro_rlt_receipt")
        )

    def test_duplicate_relu_parent_gap_closes_and_fixed_phases_match(
        self,
    ) -> None:
        baseline = _build(0)
        lifted = _build(64)
        baseline_lower, baseline_upper = _relaxed_output_range(
            baseline
        )
        lifted_lower, lifted_upper = _relaxed_output_range(lifted)

        self.assertLess(baseline_lower, -0.49)
        self.assertGreater(baseline_upper, 0.49)
        self.assertLess(abs(lifted_lower), 1.0e-10)
        self.assertLess(abs(lifted_upper), 1.0e-10)
        for assignment in itertools.product((-1.0, 1.0), repeat=2):
            self.assertTrue(
                np.allclose(
                    _relaxed_output_range(
                        baseline, fixed_binary=assignment
                    ),
                    _relaxed_output_range(
                        lifted, fixed_binary=assignment
                    ),
                    rtol=0.0,
                    atol=1.0e-12,
                ),
                assignment,
            )

        receipt = lifted.metadata["property_micro_rlt"]
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["requested_packet_mode"], "both")
        self.assertEqual(
            receipt["selected_packet_record_indices"], [0, 1]
        )
        self.assertEqual(receipt["selected_packet_count"], 2)
        self.assertTrue(receipt["proof_authority"])
        self.assertTrue(receipt["live_result_validation_passed"])
        self.assertEqual(receipt["common_focused_rival_id"], 0)
        self.assertEqual(receipt["new_product_factors"], 6)
        self.assertEqual(receipt["new_upper_rows"], 40)
        self.assertEqual(len(receipt["exact_relu_records"]), 2)
        self.assertEqual(
            {
                (
                    item["lower_upper_row"],
                    item["x_branch_upper_row"],
                    item["zero_branch_upper_row"],
                )
                for item in receipt["exact_relu_records"]
            },
            {(0, 2, 4), (1, 3, 5)},
        )
        self.assertEqual(receipt["scope"], "parent_pre_phase_fix")
        self.assertFalse(receipt["fixed_phase_projection_gain"])
        self.assertTrue(receipt["fixed_phase_rows_retained"])
        self.assertTrue(receipt["fixed_phase_solver_overhead_only"])
        self.assertTrue(receipt["excluded_from_early_row_prefixes"])
        self.assertFalse(receipt["claimed_c38_prefix_tightening"])
        self.assertEqual(
            receipt["auxiliary_continuous_provenance_layer_id"], -1
        )
        self.assertTrue(_receipt_hash_valid(receipt))

        live_receipt = lifted.hz._property_micro_rlt_receipt
        self.assertTrue(
            verify_property_micro_rlt_result(
                PropertyMicroRLTResult(lifted.hz, live_receipt)
            )
        )
        self.assertEqual(
            receipt["property_micro_rlt_receipt_sha256"],
            live_receipt["receipt_sha256"],
        )
        base_n_cont = receipt["base_counts"]["n_cont"]
        provenance = lifted.hz._solver_continuous_column_layer_ids
        self.assertEqual(provenance.size, lifted.hz.n_cont)
        self.assertTrue(np.all(provenance[base_n_cont:] == -1))
        self.assertEqual(
            len(lifted.hz._solver_constraint_row_tags),
            lifted.hz.n_eq + lifted.hz.n_ub,
        )
        generated_start = receipt["generated_upper_row_start"]
        self.assertTrue(
            all(
                tag.startswith("property_micro_rlt:")
                for tag in lifted.hz._solver_constraint_row_tags[
                    lifted.hz.n_eq + generated_start :
                ]
            )
        )
        self.assertTrue(hz_known_nonempty(lifted.hz))
        self.assertTrue(hz_constructively_nonempty(lifted.hz))
        self.assertTrue(
            lifted.hz._solver_known_nonempty_reason.startswith(
                "property_micro_rlt_exact_integer_extension:"
            )
        )
        self.assertTrue(
            np.array_equal(lifted.hz.full_col_ids, lifted.input_col_ids)
        )
        self.assertEqual(
            lifted.hz._solver_row_constraint_prefix_frames, {}
        )

    def test_each_complete_directed_packet_is_sound_and_directional(
        self,
    ) -> None:
        baseline = _build(0)
        first = _build(
            64, property_micro_rlt_packet_mode="first"
        )
        second = _build(
            64, property_micro_rlt_packet_mode="second"
        )
        first_range = _relaxed_output_range(first)
        second_range = _relaxed_output_range(second)
        self.assertLess(abs(first_range[0]), 1.0e-10)
        self.assertGreater(first_range[1], 0.49)
        self.assertLess(second_range[0], -0.49)
        self.assertLess(abs(second_range[1]), 1.0e-10)

        for index, (mode, build) in enumerate(
            (("first", first), ("second", second))
        ):
            receipt = build.metadata["property_micro_rlt"]
            self.assertEqual(receipt["requested_packet_mode"], mode)
            self.assertEqual(
                receipt["selected_packet_record_indices"], [index]
            )
            self.assertEqual(receipt["selected_packet_count"], 1)
            self.assertEqual(receipt["new_product_factors"], 3)
            self.assertEqual(receipt["new_upper_rows"], 20)
            self.assertTrue(_receipt_hash_valid(receipt))
            self.assertTrue(
                verify_property_micro_rlt_result(
                    PropertyMicroRLTResult(
                        build.hz,
                        build.hz._property_micro_rlt_receipt,
                    )
                )
            )
            for assignment in itertools.product(
                (-1.0, 1.0), repeat=2
            ):
                self.assertTrue(
                    np.allclose(
                        _relaxed_output_range(
                            baseline, fixed_binary=assignment
                        ),
                        _relaxed_output_range(
                            build, fixed_binary=assignment
                        ),
                        rtol=0.0,
                        atol=1.0e-12,
                    ),
                    (mode, assignment),
                )

    def test_cap_and_eligibility_failures_are_complete_no_ops(
        self,
    ) -> None:
        target = (
            "act.back_end.hybridz_tf.operator_hz.hz_fresh_col_ids"
        )
        with patch(
            target, new=_deterministic_id_allocator()
        ):
            baseline = _build(0)
        with patch(
            target, new=_deterministic_id_allocator()
        ):
            capped = _build(1)
        self.assertEqual(
            _hz_bits(baseline.hz, include_ids=True),
            _hz_bits(capped.hz, include_ids=True),
        )
        cap_receipt = capped.metadata["property_micro_rlt"]
        self.assertEqual(cap_receipt["status"], "no_op_cap_exceeded")
        self.assertIn("cap exceeded", cap_receipt["no_op_reason"])
        self.assertEqual(
            cap_receipt["required_selected_source_row_nnz"], 18
        )
        self.assertEqual(
            cap_receipt["required_product_factors"], 6
        )
        self.assertEqual(
            cap_receipt["selected_source_row_nnz_cap"], 16384
        )
        self.assertEqual(
            cap_receipt["requirement_scan_nnz_cap"], 65536
        )
        self.assertTrue(cap_receipt["requirement_count_complete"])
        self.assertFalse(
            cap_receipt["selected_source_nnz_cap_exceeded"]
        )
        self.assertTrue(cap_receipt["product_factor_cap_exceeded"])
        self.assertEqual(
            cap_receipt["primary_cap_failure"],
            "product_factor_cap_exceeded",
        )
        self.assertEqual(
            cap_receipt["supported_product_factor_cap_max"], 4096
        )
        self.assertEqual(
            cap_receipt["base_counts"], cap_receipt["result_counts"]
        )
        self.assertTrue(_receipt_hash_valid(cap_receipt))
        self.assertFalse(
            hasattr(capped.hz, "_property_micro_rlt_receipt")
        )
        self.assertEqual(
            capped.hz._solver_known_nonempty_reason,
            "operator_hz_outward_transfer_induction_v1",
        )

        ineligible = _build(
            64,
            property_phase_focus_rivals={
                (3, 0): (0,),
                (3, 1): (1,),
            },
        )
        ineligible_receipt = ineligible.metadata[
            "property_micro_rlt"
        ]
        self.assertEqual(
            ineligible_receipt["status"], "no_op_ineligible"
        )
        self.assertEqual(
            ineligible_receipt["no_op_reason"],
            "exact_relu_records_do_not_share_one_focused_rival",
        )
        self.assertEqual(
            _hz_bits(baseline.hz, include_ids=False),
            _hz_bits(ineligible.hz, include_ids=False),
        )
        self.assertTrue(_receipt_hash_valid(ineligible_receipt))

        toy = _duplicate_relu_toy()
        with self.assertRaisesRegex(
            OperatorHZBuildError,
            "property_micro_rlt_product_cap",
        ):
            build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                property_micro_rlt_product_cap=True,
            )
        with self.assertRaisesRegex(
            OperatorHZBuildError,
            r"\[0, 4096\]",
        ):
            build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                property_micro_rlt_product_cap=4097,
            )
        for mode in (None, "", "forward", 1):
            with self.subTest(packet_mode=mode):
                with self.assertRaisesRegex(
                    OperatorHZBuildError,
                    "packet_mode",
                ):
                    build_operator_hz(
                        toy.net,
                        toy.facts,
                        toy.facts,
                        property_micro_rlt_product_cap=64,
                        property_micro_rlt_packet_mode=mode,
                    )
        with self.assertRaisesRegex(
            OperatorHZBuildError,
            "requires a positive product cap",
        ):
            build_operator_hz(
                toy.net,
                toy.facts,
                toy.facts,
                property_micro_rlt_packet_mode="first",
            )

    def test_post_lift_decorations_preserve_conditional_rows_and_prefix(
        self,
    ) -> None:
        original = (
            _OperatorHZBuilder._validate_input_spec_enclosure
        )

        def inject_post_build_decorations(builder, order):
            original(builder, order)
            binary_ids = tuple(int(value) for value in builder.bcol_ids)
            self.assertEqual(len(binary_ids), 2)
            rows = []
            for phases in itertools.product((-1, 1), repeat=2):
                rows.append(
                    {
                        "binary_guards": tuple(
                            {
                                "binary_col_id": binary_ids[index],
                                "phase": int(phase),
                                "layer_id": 3,
                                "row": index,
                            }
                            for index, phase in enumerate(phases)
                        ),
                        "layer_id": 3,
                        "row": 0,
                        "center": np.zeros(1, dtype=np.float64),
                        "generator": sp.csr_matrix(
                            (1, builder.n_cont),
                            dtype=np.float64,
                        ),
                        "error": np.zeros(1, dtype=np.float64),
                        "rival_ids": (0,),
                        "receipt": {"synthetic_sequence_gate": True},
                    }
                )
            builder.property_conditional_suffix_rows = rows

            frame = builder.layer_frame_snapshots[2]
            builder.property_suffix_add_source_snapshot = (
                _PropertySuffixAddSourceSnapshot(
                    add_layer_id=2,
                    expression=builder.exprs[2],
                    n_cont=frame.n_cont,
                    n_bin=frame.n_bin,
                    eq_rows=frame.eq_rows,
                    ub_rows=frame.ub_rows,
                    eq_block_count=frame.eq_block_count,
                    ub_block_count=frame.ub_block_count,
                )
            )
            builder.property_tail_receipt = {
                "shared_suffix_replay": {
                    "status": "applied",
                    "proof_authority": True,
                    "output_form": "synthetic_shared_prefix_gate",
                    "stop_layer_id": 2,
                    "row_start": 0,
                    "row_count": 1,
                }
            }

        with patch.object(
            _OperatorHZBuilder,
            "_validate_input_spec_enclosure",
            new=inject_post_build_decorations,
        ):
            lifted = _build(64)

        conditional = lifted.hz._solver_conditional_property_rows
        self.assertEqual(len(conditional), 4)
        for item in conditional:
            self.assertEqual(
                item["generator"].shape,
                (1, lifted.hz.n_cont),
            )
            self.assertEqual(item["generator"].nnz, 0)
        prefix = lifted.hz._solver_row_constraint_prefix_frames
        self.assertEqual(set(prefix), {0})
        receipt = lifted.metadata["property_micro_rlt"]
        self.assertLessEqual(
            prefix[0]["ub_rows"],
            receipt["generated_upper_row_start"],
        )
        self.assertLess(
            prefix[0]["n_cont"],
            lifted.hz.n_cont,
        )
        self.assertTrue(receipt["excluded_from_early_row_prefixes"])
        self.assertTrue(
            np.array_equal(lifted.hz.full_col_ids, lifted.input_col_ids)
        )
        self.assertTrue(hz_constructively_nonempty(lifted.hz))

    def test_malformed_candidate_receipt_rolls_back_builder_state(
        self,
    ) -> None:
        toy = _duplicate_relu_toy()
        builder = _OperatorHZBuilder(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=2,
            materialize_add=True,
            preactivation_lp_budget=0,
            preactivation_lp_time_limit=0.0,
            residual_targets=[
                (3, 0, "none"),
                (3, 1, "none"),
            ],
            property_phase_focus_rivals={
                (3, 0): (0,),
                (3, 1): (0,),
            },
            property_micro_rlt_product_cap=64,
            deadline=None,
        )
        real_apply = micro_rlt.apply_property_micro_rlt
        observed: Dict[str, Any] = {}

        def malformed_apply(*args, **kwargs):
            observed["n_cont"] = int(builder.n_cont)
            observed["col_ids"] = tuple(builder.col_ids)
            observed["provenance"] = dict(
                builder.cont_column_layer_by_id
            )
            result = real_apply(*args, **kwargs)
            malformed_receipt = dict(result.receipt)
            malformed_receipt["generated_row_names"] = list(
                malformed_receipt["generated_row_names"][:-1]
            )
            return PropertyMicroRLTResult(
                result.hz, malformed_receipt
            )

        with patch.object(
            micro_rlt,
            "apply_property_micro_rlt",
            side_effect=malformed_apply,
        ), patch.object(
            micro_rlt,
            "verify_property_micro_rlt_result",
            return_value=True,
        ):
            with self.assertRaisesRegex(
                OperatorHZBuildError,
                "generated-row tags are incomplete",
            ):
                builder.build()

        self.assertEqual(builder.n_cont, observed["n_cont"])
        self.assertEqual(tuple(builder.col_ids), observed["col_ids"])
        self.assertEqual(
            builder.cont_column_layer_by_id,
            observed["provenance"],
        )
        self.assertEqual(
            sum(block.Ac.shape[0] for block in builder.ub_blocks),
            6,
        )

    def test_live_exact_binary_binding_rejects_id_and_sign_tamper(
        self,
    ) -> None:
        original = _OperatorHZBuilder._maybe_apply_property_micro_rlt

        def tamper_stable_id(builder, hz):
            builder.property_exact_phase_records[0][
                "binary_col_id"
            ] += 10_000
            return original(builder, hz)

        with patch.object(
            _OperatorHZBuilder,
            "_maybe_apply_property_micro_rlt",
            new=tamper_stable_id,
        ):
            with self.assertRaisesRegex(
                OperatorHZBuildError,
                "exact binary mapping is malformed",
            ):
                _build(64)

        def tamper_x_branch_sign(builder, hz):
            record = builder.property_exact_phase_records[0]
            row = int(record["exact_upper_rows"]["x_branch"])
            start = int(hz.Aub.indptr[row])
            end = int(hz.Aub.indptr[row + 1])
            if end - start != 1:
                raise AssertionError("toy x-branch row lost its one-bit form")
            hz.Aub.data[start:end] *= -1.0
            return original(builder, hz)

        with patch.object(
            _OperatorHZBuilder,
            "_maybe_apply_property_micro_rlt",
            new=tamper_x_branch_sign,
        ):
            with self.assertRaisesRegex(
                OperatorHZBuildError,
                "selected-binary coefficient structure",
            ):
                _build(64)


if __name__ == "__main__":
    unittest.main()
