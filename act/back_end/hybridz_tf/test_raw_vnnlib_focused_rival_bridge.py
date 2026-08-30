#!/usr/bin/env python3
"""Controlled gates for the raw full-batch focused-rival bridge."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np
import torch

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    ordered_property_digest,
    rival_spec_binding_digest,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.raw_vnnlib_focused_rival_bridge import (
    ExactRawRivalHardness,
    RankedRawRivalHardness,
    RawFocusedRivalBridgeError,
    RawFocusedRivalSelection,
    RawRivalExactHardnessReceipt,
    issue_raw_rival_exact_hardness_receipt,
    select_raw_focused_rivals,
    verify_raw_focused_rival_selection,
    verify_raw_rival_exact_hardness_receipt,
)
from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    consume_raw_vnnlib_top1_candidate,
    issue_raw_vnnlib_top1_candidate,
    validate_consumed_raw_vnnlib_rival_batch,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
)


_INTERVAL_SHA256 = "ab" * 32


def _raw_source(output_width: int) -> str:
    declarations = "\n".join(
        f"(declare-const Y_{index} Real)"
        for index in range(output_width)
    )
    branches = " ".join(
        f"(<= Y_0 Y_{index})"
        for index in range(1, output_width)
    )
    return f"""
    (set-logic QF_LRA)
    (declare-const X_0 Real)
    {declarations}
    (assert (>= X_0 -1))
    (assert (<= X_0 1))
    (assert (or {branches}))
    """


def _consumed_batch(output_width: int = 5):
    rows = output_width - 1
    live_C = torch.zeros(
        (rows, output_width), dtype=torch.float64
    )
    live_C[:, 0] = -1.0
    for encoded_row in range(rows):
        live_C[encoded_row, encoded_row + 1] = 1.0
    live = {
        "kind": "TOP1_ROBUST",
        "C": live_C,
        "thresholds": torch.zeros(
            (1, rows), dtype=torch.float64
        ),
        "M": rows,
        "y_true": torch.tensor([0], dtype=torch.int64),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "focused.vnnlib"
        path.write_text(
            _raw_source(output_width).strip() + "\n",
            encoding="utf-8",
        )
        source_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        candidate = issue_raw_vnnlib_top1_candidate(
            path,
            expected_vnnlib_sha256=source_sha256,
            live_assert_params=live,
            deadline=time.monotonic() + 10.0,
        )
        batch = consume_raw_vnnlib_top1_candidate(
            candidate,
            live_assert_params=live,
            deadline=time.monotonic() + 10.0,
        )
    if not validate_consumed_raw_vnnlib_rival_batch(batch):
        raise AssertionError("test batch did not retain live ownership")
    return batch


def _hardness_values(rows: int = 4):
    baseline = (
        (1, 2),
        (5, 4),
        (5, 4),
        (-1, 8),
    )
    return baseline[:rows]


def _issue(batch, *, values=None, **kwargs):
    exact = (
        _hardness_values(len(batch.rivals))
        if values is None
        else values
    )
    return issue_raw_rival_exact_hardness_receipt(
        batch,
        exact,
        live_interval_bounds_sha256=_INTERVAL_SHA256,
        deadline=time.monotonic() + 10.0,
        **kwargs,
    )


def _select(batch, hardness, *, expected_values=None, **kwargs):
    exact = (
        _hardness_values(len(batch.rivals))
        if expected_values is None
        else expected_values
    )
    return select_raw_focused_rivals(
        batch,
        hardness,
        expected_exact_upper_violations=exact,
        expected_live_interval_bounds_sha256=_INTERVAL_SHA256,
        **kwargs,
    )


def _verify_hardness(
    batch, hardness, *, expected_values=None, **kwargs
):
    exact = (
        _hardness_values(len(batch.rivals))
        if expected_values is None
        else expected_values
    )
    return verify_raw_rival_exact_hardness_receipt(
        batch,
        hardness,
        expected_exact_upper_violations=exact,
        expected_live_interval_bounds_sha256=_INTERVAL_SHA256,
        **kwargs,
    )


def _verify_selection(
    batch,
    hardness,
    selection,
    *,
    expected_values=None,
    **kwargs,
):
    exact = (
        _hardness_values(len(batch.rivals))
        if expected_values is None
        else expected_values
    )
    return verify_raw_focused_rival_selection(
        batch,
        hardness,
        selection,
        expected_exact_upper_violations=exact,
        expected_live_interval_bounds_sha256=_INTERVAL_SHA256,
        **kwargs,
    )


def _residual_property_sha256(batch) -> str:
    C = np.ascontiguousarray(
        tuple(rival.objective for rival in batch.rivals),
        dtype=np.float64,
    )
    thresholds = np.ascontiguousarray(
        tuple(rival.threshold for rival in batch.rivals),
        dtype=np.float64,
    )
    digest = hashlib.sha256()
    for value in (C, thresholds):
        digest.update(
            np.asarray(value.shape, dtype=np.int64).tobytes()
        )
        digest.update(value.tobytes())
    digest.update(b"TOP1_ROBUST")
    return digest.hexdigest()


def _residual_receipt(batch, *, encoded_row: int):
    competitor_id = batch.rivals[encoded_row].rival_id
    return {
        "schema": "property_residual_selector_v1",
        "status": "selected",
        "candidate_only": True,
        "proof_authority": False,
        "property_sha256": _residual_property_sha256(batch),
        "selection_policy": (
            "facility_first_then_same_rival_joint"
        ),
        "joint_focus_rival_id": encoded_row,
        "rival_ids": list(range(len(batch.rivals))),
        "targets_selected": 3,
        "schedule": [
            {
                "layer_id": 7,
                "row": 2,
                "dominant_rival": encoded_row,
                "raw_competitor_class": competitor_id,
            }
        ],
    }


class RawFocusedRivalPositiveTests(unittest.TestCase):
    def test_default_singleton_exact_ranking_and_full_bindings(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        selection = _select(
            batch,
            hardness,
            deadline=time.monotonic() + 10.0,
        )

        self.assertTrue(
            _verify_hardness(
                batch,
                hardness,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertTrue(
            _verify_selection(
                batch,
                hardness,
                selection,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(hardness.shape, (4,))
        self.assertEqual(
            tuple(entry.encoded_row for entry in hardness.entries),
            (0, 1, 2, 3),
        )
        self.assertEqual(
            tuple(
                entry.competitor_class
                for entry in hardness.entries
            ),
            (1, 2, 3, 4),
        )
        self.assertEqual(
            tuple(
                entry.rival_spec_binding_digest
                for entry in hardness.entries
            ),
            tuple(
                rival_spec_binding_digest(rival)
                for rival in batch.rivals
            ),
        )
        self.assertEqual(
            hardness.full_property_digest,
            ordered_property_digest(batch.rivals),
        )
        self.assertEqual(
            tuple(
                entry.encoded_row
                for entry in selection.ranked_entries
            ),
            (1, 2, 0, 3),
        )
        self.assertEqual(selection.focus_count, 1)
        self.assertEqual(
            selection.focused_entries[0].encoded_row, 1
        )
        self.assertIs(
            selection.focused_rivals[0], batch.rivals[1]
        )
        self.assertIs(selection.rivals, selection.focused_rivals)
        self.assertIsInstance(hardness.receipt, MappingProxyType)
        self.assertIsInstance(selection.receipt, MappingProxyType)
        self.assertFalse(hardness.proof_authority)
        self.assertFalse(selection.proof_authority)
        with self.assertRaises(TypeError):
            hardness.receipt["status"] = "forged"

    def test_pre_registered_four_rival_subset_is_ranked_prefix(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        selection = _select(
            batch,
            hardness,
            focus_count=4,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            _verify_selection(
                batch,
                hardness,
                selection,
                expected_focus_count=4,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(
            tuple(rival.rival_id for rival in selection.rivals),
            (2, 3, 1, 4),
        )
        self.assertEqual(
            tuple(
                item.encoded_row
                for item in selection.focused_entries
            ),
            (1, 2, 0, 3),
        )

    def test_residual_joint_focus_row_overrides_hardness_safely(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        caller_receipt = _residual_receipt(
            batch, encoded_row=3
        )
        selection = _select(
            batch,
            hardness,
            explicit_encoded_focus_row=3,
            residual_selector_receipt=caller_receipt,
            residual_selector_property_sha256=(
                _residual_property_sha256(batch)
            ),
            deadline=time.monotonic() + 10.0,
        )
        caller_receipt["joint_focus_rival_id"] = 1

        self.assertTrue(
            _verify_selection(
                batch,
                hardness,
                selection,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(
            selection.explicit_encoded_focus_row, 3
        )
        self.assertEqual(
            selection.residual_joint_focus_rival_id, 3
        )
        self.assertEqual(
            selection.focused_entries[0].encoded_row, 3
        )
        self.assertEqual(selection.rivals[0].rival_id, 4)
        self.assertNotEqual(
            selection.focused_entries[0].encoded_row,
            selection.ranked_entries[0].encoded_row,
        )
        self.assertIsInstance(
            selection.residual_selector_receipt,
            MappingProxyType,
        )

    def test_focused_singleton_feeds_existing_operator_selector(
        self,
    ) -> None:
        batch = _consumed_batch(output_width=3)
        hardness = _issue(
            batch, values=((0, 1), (3, 2))
        )
        focused = _select(
            batch,
            hardness,
            expected_values=((0, 1), (3, 2)),
            deadline=time.monotonic() + 10.0,
        )
        build = _k4_corner_build()
        operator_selection = (
            derive_operator_exact_relu_property_phase_literals(
                build, focused.rivals
            )
        )
        self.assertEqual(
            tuple(
                mapping.rival_coefficients[0].rival_id
                for mapping in operator_selection.mappings
            ),
            (2, 2, 2, 2),
        )
        self.assertTrue(
            verify_operator_exact_relu_property_phase_selection(
                build,
                focused.rivals,
                operator_selection,
            )
        )


class RawFocusedRivalFailClosedTests(unittest.TestCase):
    def test_exact_vector_shape_types_reduction_and_caps(
        self,
    ) -> None:
        batch = _consumed_batch()
        bad_vectors = (
            list(_hardness_values()),
            _hardness_values(3),
            ((True, 1), (1, 1), (1, 1), (1, 1)),
            ((1, 2), (2, 4), (1, 1), (1, 1)),
            ((0, 2), (1, 1), (1, 1), (1, 1)),
            ((1, -2), (1, 1), (1, 1), (1, 1)),
            ((1, 3), (1, 1), (1, 1), (1, 1)),
        )
        for values in bad_vectors:
            with self.subTest(values=repr(values)[:40]):
                with self.assertRaises(
                    RawFocusedRivalBridgeError
                ):
                    _issue(batch, values=values)
        with self.assertRaisesRegex(
            RawFocusedRivalBridgeError, "rival_cap"
        ):
            _issue(batch, max_rivals=3)
        with self.assertRaises(
            RawFocusedRivalBridgeError
        ):
            _issue(
                batch,
                values=(
                    (1 << 4096, 1),
                    (1, 1),
                    (1, 1),
                    (1, 1),
                ),
            )

    def test_residual_focus_rejects_id_row_property_and_policy_swap(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        mutations = (
            ("joint_focus_rival_id", 1),
            ("property_sha256", "ef" * 32),
            ("selection_policy", "multi_rival_facility"),
            ("proof_authority", True),
            ("rival_ids", [0, 1, 2]),
        )
        for key, value in mutations:
            receipt = _residual_receipt(
                batch, encoded_row=3
            )
            receipt[key] = value
            with self.subTest(key=key):
                with self.assertRaises(
                    RawFocusedRivalBridgeError
                ):
                    _select(
                        batch,
                        hardness,
                        explicit_encoded_focus_row=3,
                        residual_selector_receipt=receipt,
                        residual_selector_property_sha256=(
                            _residual_property_sha256(batch)
                        ),
                        deadline=time.monotonic() + 10.0,
                    )
        with self.assertRaisesRegex(
            RawFocusedRivalBridgeError, "requires_singleton"
        ):
            _select(
                batch,
                hardness,
                focus_count=2,
                explicit_encoded_focus_row=3,
                residual_selector_receipt=_residual_receipt(
                    batch, encoded_row=3
                ),
                residual_selector_property_sha256=(
                    _residual_property_sha256(batch)
                ),
                deadline=time.monotonic() + 10.0,
            )
        other_property = "ef" * 32
        same_shape_other_receipt = _residual_receipt(
            batch, encoded_row=3
        )
        same_shape_other_receipt[
            "property_sha256"
        ] = other_property
        with self.assertRaisesRegex(
            RawFocusedRivalBridgeError,
            "joint_focus_contract_invalid",
        ):
            _select(
                batch,
                hardness,
                explicit_encoded_focus_row=3,
                residual_selector_receipt=(
                    same_shape_other_receipt
                ),
                residual_selector_property_sha256=other_property,
                deadline=time.monotonic() + 10.0,
            )

    def test_live_expected_hardness_and_interval_frame_are_required(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        selection = _select(
            batch,
            hardness,
            deadline=time.monotonic() + 10.0,
        )
        changed = (
            (1, 2),
            (5, 4),
            (3, 2),
            (-1, 8),
        )
        self.assertFalse(
            _verify_hardness(
                batch,
                hardness,
                expected_values=changed,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(
            verify_raw_rival_exact_hardness_receipt(
                batch,
                hardness,
                expected_exact_upper_violations=(
                    _hardness_values()
                ),
                expected_live_interval_bounds_sha256="ef" * 32,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(
            _verify_selection(
                batch,
                hardness,
                selection,
                expected_values=changed,
                deadline=time.monotonic() + 10.0,
            )
        )
        with self.assertRaisesRegex(
            RawFocusedRivalBridgeError,
            "expected_live_vector_mismatch",
        ):
            _select(
                batch,
                hardness,
                expected_values=changed,
                deadline=time.monotonic() + 10.0,
            )

    def test_owner_copy_digest_and_subset_tampering_rejected(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        selection = _select(
            batch,
            hardness,
            deadline=time.monotonic() + 10.0,
        )
        copied_batch = replace(batch)
        self.assertFalse(
            validate_consumed_raw_vnnlib_rival_batch(copied_batch)
        )
        self.assertFalse(
            _verify_hardness(
                copied_batch,
                hardness,
                expected_values=_hardness_values(),
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(
            _verify_hardness(
                batch,
                replace(hardness, vector_digest="00" * 32),
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(
            _verify_selection(
                batch,
                hardness,
                replace(
                    selection,
                    focused_subset_digest="00" * 32,
                ),
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(
            _verify_selection(
                batch,
                hardness,
                replace(
                    selection,
                    focused_rivals=(batch.rivals[0],),
                ),
                deadline=time.monotonic() + 10.0,
            )
        )

    def test_residual_result_method_row_and_receipt_tampering_rejected(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        selection = _select(
            batch,
            hardness,
            explicit_encoded_focus_row=3,
            residual_selector_receipt=_residual_receipt(
                batch, encoded_row=3
            ),
            residual_selector_property_sha256=(
                _residual_property_sha256(batch)
            ),
            deadline=time.monotonic() + 10.0,
        )
        tampered = (
            replace(
                selection,
                explicit_encoded_focus_row=2,
            ),
            replace(
                selection,
                residual_selector_receipt_sha256="00" * 32,
            ),
            replace(selection, method="hardest"),
            replace(
                selection,
                residual_selector_property_sha256="ef" * 32,
            ),
        )
        for candidate in tampered:
            self.assertFalse(
                _verify_selection(
                    batch,
                    hardness,
                    candidate,
                    deadline=time.monotonic() + 10.0,
                )
            )

    def test_verifiers_never_invoke_candidate_equality(self) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)
        selection = _select(
            batch,
            hardness,
            deadline=time.monotonic() + 10.0,
        )

        def explode(_self, _other):
            raise AssertionError("candidate equality invoked")

        classes = (
            ExactRawRivalHardness,
            RankedRawRivalHardness,
            RawRivalExactHardnessReceipt,
            RawFocusedRivalSelection,
            RivalSpec,
        )
        patches = [
            mock.patch.object(cls, "__eq__", explode)
            for cls in classes
        ]
        for patch in patches:
            patch.start()
        try:
            self.assertTrue(
                _verify_hardness(
                    batch,
                    hardness,
                    deadline=time.monotonic() + 10.0,
                )
            )
            self.assertTrue(
                _verify_selection(
                    batch,
                    hardness,
                    selection,
                    deadline=time.monotonic() + 10.0,
                )
            )
        finally:
            for patch in reversed(patches):
                patch.stop()

        class AliasKey:
            calls = 0

            def __hash__(self):
                type(self).calls += 1
                return hash("schema")

            def __eq__(self, other):
                type(self).calls += 1
                return other == "schema"

        aliased_receipt = {
            key: value
            for key, value in hardness.receipt.items()
            if key != "schema"
        }
        alias = AliasKey()
        aliased_receipt[alias] = hardness.receipt["schema"]
        AliasKey.calls = 0
        self.assertFalse(
            _verify_hardness(
                batch,
                replace(
                    hardness,
                    receipt=MappingProxyType(aliased_receipt),
                ),
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(AliasKey.calls, 0)

    def test_oversized_shape_and_receipt_reject_before_walk(
        self,
    ) -> None:
        batch = _consumed_batch()
        hardness = _issue(batch)

        oversized_shape = (1,) * 4_000_000
        started = time.monotonic()
        self.assertFalse(
            _verify_hardness(
                batch,
                replace(hardness, shape=oversized_shape),
                deadline=started + 0.1,
            )
        )
        self.assertLess(time.monotonic() - started, 0.02)

        oversized_receipt = MappingProxyType(
            {
                f"oversized_{index}": index
                for index in range(500_000)
            }
        )
        started = time.monotonic()
        self.assertFalse(
            _verify_hardness(
                batch,
                replace(hardness, receipt=oversized_receipt),
                deadline=started + 0.1,
            )
        )
        self.assertLess(time.monotonic() - started, 0.02)

        large_batch = _consumed_batch(output_width=200)
        large_values = tuple(
            (encoded_row, 1) for encoded_row in range(199)
        )
        large_hardness = _issue(
            large_batch,
            values=large_values,
        )
        started = time.monotonic()
        self.assertFalse(
            _verify_hardness(
                large_batch,
                large_hardness,
                expected_values=large_values,
                max_rivals=1,
                deadline=started + 0.01,
            )
        )
        self.assertLess(time.monotonic() - started, 0.005)

    def test_deadline_focus_limit_and_caps_are_caller_bound(
        self,
    ) -> None:
        batch = _consumed_batch()
        with self.assertRaisesRegex(
            RawFocusedRivalBridgeError, "deadline_expired"
        ):
            issue_raw_rival_exact_hardness_receipt(
                batch,
                _hardness_values(),
                live_interval_bounds_sha256=_INTERVAL_SHA256,
                deadline=time.monotonic() - 1.0,
            )
        hardness = _issue(batch)
        with self.assertRaisesRegex(
            RawFocusedRivalBridgeError, "registered_range"
        ):
            _select(
                batch,
                hardness,
                focus_count=5,
                deadline=time.monotonic() + 10.0,
            )
        self.assertFalse(
            _verify_hardness(
                batch,
                hardness,
                max_focus=3,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertFalse(
            _verify_selection(
                batch,
                hardness,
                _select(
                    batch,
                    hardness,
                    deadline=time.monotonic() + 10.0,
                ),
                deadline=time.monotonic() - 1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
