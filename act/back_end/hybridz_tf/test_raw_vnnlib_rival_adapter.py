#!/usr/bin/env python3
"""Strict controlled toys for the raw VNNLIB TOP1 rival adapter."""

from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import os
from pathlib import Path
import tempfile
import time
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np
import torch

import act.back_end.hybridz_tf.raw_vnnlib_rival_adapter as raw_adapter
from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    ConsumedRivalBatch,
    RawVNNLibRivalAdapterError,
    consume_raw_vnnlib_top1_candidate,
    issue_raw_vnnlib_top1_candidate,
    revoke_raw_vnnlib_top1_candidate,
    validate_consumed_raw_vnnlib_rival_batch,
)
from act.front_end.vnnlib_loader.vnnlib_parser import (
    parse_vnnlib_queries,
)


_DEFAULT_BRANCHES = (
    "(and (>= Y_0 Y_2))",
    "(and (<= Y_2 Y_1))",
    "(and (>= Y_3 Y_2))",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _legacy_source(
    *,
    branches: tuple[str, ...] = _DEFAULT_BRANCHES,
    include_output: bool = True,
    output_body: str | None = None,
    extra_asserts: tuple[str, ...] = (),
) -> str:
    declarations = "\n".join(
        ["(declare-const X_0 Real)"]
        + [
            f"(declare-const Y_{index} Real)"
            for index in range(4)
        ]
    )
    assertions = [
        "(assert (>= X_0 0))",
        "(assert (<= X_0 1))",
    ]
    if include_output:
        body = (
            output_body
            if output_body is not None
            else "(or\n    " + "\n    ".join(branches) + "\n)"
        )
        assertions.append(f"(assert {body})")
    assertions.extend(extra_asserts)
    return "\n".join(
        ["(set-logic QF_LRA)", declarations, *assertions]
    )


def _v2_source() -> str:
    return """
    (vnnlib-version 2.0)
    (declare-network N)
    (declare-input X Real [1])
    (declare-output Y Real [4])
    (assert (>= X[0] 0))
    (assert (<= X[0] 1))
    (assert (or
        (and (>= Y[0] Y[2]))
        (and (<= Y[2] Y[1]))
        (and (>= Y[3] Y[2]))))
    """


def _live_params(
    *,
    dtype: torch.dtype = torch.float64,
) -> dict[str, object]:
    C = torch.zeros((3, 4), dtype=dtype)
    for row, competitor in enumerate((0, 1, 3)):
        C[row, competitor] = 1.0
        C[row, 2] = -1.0
    return {
        "kind": "TOP1_ROBUST",
        "C": C,
        "thresholds": torch.zeros((1, 3), dtype=dtype),
        "M": 3,
        "y_true": torch.tensor([2], dtype=torch.int64),
    }


class RawVNNLibTop1PositiveTests(unittest.TestCase):
    def test_independent_adapter_matches_real_frontend_encoding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "frontend_top1.vnnlib",
                _legacy_source(),
            )
            queries = parse_vnnlib_queries(path)
            self.assertEqual(len(queries), 1)
            output_spec = queries[0][1]
            live_params = output_spec.encode_linear(
                B=1,
                n_out=4,
                device=torch.device("cpu"),
                dtype=torch.float64,
            )
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=live_params,
            )
            self.assertEqual(
                tuple(row.competitor_class for row in candidate.rows),
                (0, 1, 3),
            )
            self.assertFalse(hasattr(candidate, "rivals"))
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=live_params
            )
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )
            self.assertEqual(
                tuple(rival.rival_id for rival in batch.rivals),
                (0, 1, 3),
            )

    def test_legacy_raw_atoms_match_live_rows_and_consume_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib",
                _legacy_source(),
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )

            self.assertEqual(candidate.dialect, "vnnlib-1.0-flat")
            self.assertEqual(candidate.true_class, 2)
            self.assertEqual(candidate.output_assert_ordinal, 2)
            self.assertFalse(candidate.proof_authority)
            self.assertEqual(
                tuple(row.competitor_class for row in candidate.rows),
                (0, 1, 3),
            )
            self.assertEqual(
                tuple(row.encoded_row for row in candidate.rows),
                (0, 1, 2),
            )
            self.assertEqual(
                tuple(row.boolean_path for row in candidate.rows),
                ((0, 0), (1, 0), (2, 0)),
            )
            self.assertEqual(
                tuple(row.competitor_class for row in candidate.rows),
                (0, 1, 3),
            )
            self.assertEqual(
                tuple(len(row.assert_digest) for row in candidate.rows),
                (64, 64, 64),
            )
            self.assertEqual(
                candidate.receipt["status"],
                "raw_live_top1_match_candidate",
            )
            self.assertFalse(candidate.receipt["proof_authority"])
            self.assertFalse(candidate.receipt["usable_rivals_exposed"])

            consumed = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            self.assertIsInstance(consumed, ConsumedRivalBatch)
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(consumed)
            )
            self.assertEqual(
                consumed.rivals[0].objective,
                (1.0, 0.0, -1.0, 0.0),
            )
            self.assertEqual(
                consumed.rivals[1].objective,
                (0.0, 1.0, -1.0, 0.0),
            )
            self.assertEqual(
                consumed.rivals[2].objective,
                (0.0, 0.0, -1.0, 1.0),
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "missing_consumed_or_expired",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )

    def test_or_reordering_keeps_competitor_ids_and_row_order_stable(
        self,
    ) -> None:
        reordered = (
            _DEFAULT_BRANCHES[2],
            _DEFAULT_BRANCHES[0],
            _DEFAULT_BRANCHES[1],
        )
        with tempfile.TemporaryDirectory() as directory:
            first = _write(
                Path(directory) / "first.vnnlib",
                _legacy_source(),
            )
            second = _write(
                Path(directory) / "second.vnnlib",
                _legacy_source(branches=reordered),
            )
            params = _live_params(dtype=torch.float32)
            left = issue_raw_vnnlib_top1_candidate(
                first,
                expected_vnnlib_sha256=_sha256(first),
                live_assert_params=params,
            )
            right = issue_raw_vnnlib_top1_candidate(
                second,
                expected_vnnlib_sha256=_sha256(second),
                live_assert_params=params,
            )

            self.assertEqual(
                tuple(row.competitor_class for row in left.rows),
                tuple(row.competitor_class for row in right.rows),
            )
            self.assertEqual(
                {
                    row.competitor_class: row.boolean_path
                    for row in left.rows
                },
                {0: (0, 0), 1: (1, 0), 3: (2, 0)},
            )
            self.assertEqual(
                {
                    row.competitor_class: row.boolean_path
                    for row in right.rows
                },
                {0: (1, 0), 1: (2, 0), 3: (0, 0)},
            )
            # Full-file SHA and Boolean path are intentionally raw bindings,
            # so stable IDs survive reordering while ASSERT digests change.
            self.assertNotEqual(
                tuple(row.assert_digest for row in left.rows),
                tuple(row.assert_digest for row in right.rows),
            )
            left_batch = consume_raw_vnnlib_top1_candidate(
                left, live_assert_params=params
            )
            right_batch = consume_raw_vnnlib_top1_candidate(
                right, live_assert_params=params
            )
            self.assertEqual(
                tuple(rival.rival_id for rival in left_batch.rivals),
                tuple(rival.rival_id for rival in right_batch.rivals),
            )
            self.assertEqual(
                tuple(rival.objective for rival in left_batch.rivals),
                tuple(rival.objective for rival in right_batch.rivals),
            )
            self.assertEqual(
                tuple(rival.threshold for rival in left_batch.rivals),
                tuple(rival.threshold for rival in right_batch.rivals),
            )

    def test_vnnlib_2_bracket_variables_are_strictly_recognized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1_v2.vnnlib", _v2_source()
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            self.assertEqual(candidate.dialect, "vnnlib-2.0")
            self.assertEqual(
                tuple(row.competitor_class for row in candidate.rows),
                (0, 1, 3),
            )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            self.assertEqual(
                tuple(rival.rival_id for rival in batch.rivals),
                (0, 1, 3),
            )

    def test_copy_has_no_capability_identity_and_does_not_revoke_original(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib",
                _legacy_source(),
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            copied = replace(candidate)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "identity_mismatch",
            ):
                consume_raw_vnnlib_top1_candidate(
                    copied, live_assert_params=params
                )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )


class RawVNNLibTop1FailClosedTests(unittest.TestCase):
    def _assert_source_rejected(
        self,
        source: str,
        pattern: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "rejected.vnnlib", source
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, pattern
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=_sha256(path),
                    live_assert_params=_live_params(),
                )

    def test_swapped_live_rows_and_changed_threshold_fail_bitwise(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib",
                _legacy_source(),
            )
            digest = _sha256(path)

            swapped = _live_params()
            swapped["C"] = swapped["C"][[1, 0, 2]]
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "raw_live_C_bit_mismatch",
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=swapped,
                )

            changed = _live_params()
            changed["thresholds"][0, 1] = 1.0e-12
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "raw_live_threshold_bit_mismatch",
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=changed,
                )

            signed_zero = _live_params()
            signed_zero["thresholds"][0, 0] = -0.0
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "raw_live_threshold_bit_mismatch",
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=signed_zero,
                )

    def test_deleted_and_duplicate_competitors_fail_closed(self) -> None:
        self._assert_source_rejected(
            _legacy_source(branches=_DEFAULT_BRANCHES[:2]),
            "coverage_incomplete",
        )
        self._assert_source_rejected(
            _legacy_source(
                branches=(
                    _DEFAULT_BRANCHES[0],
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "competitor_duplicated",
        )

    def test_expected_hash_and_post_issue_byte_change_fail_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib",
                _legacy_source(),
            )
            params = _live_params()
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "sha256_mismatch",
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256="0" * 64,
                    live_assert_params=params,
                )

            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            path.write_bytes(path.read_bytes() + b"; changed after issue\n")
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "sha256_mismatch",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "missing_consumed_or_expired",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )

    def test_live_assert_change_after_issue_consumes_and_rejects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib",
                _legacy_source(),
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            changed = _live_params()
            changed["C"] = changed["C"].to(torch.float32)
            changed["thresholds"] = changed["thresholds"].to(
                torch.float32
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "live_assert_binding_mismatch",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=changed
                )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "missing_consumed_or_expired",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )

    def test_input_only_mixed_x_and_extra_conjunct_fail_closed(
        self,
    ) -> None:
        self._assert_source_rejected(
            _legacy_source(include_output=False),
            "input_only_property",
        )
        self._assert_source_rejected(
            _legacy_source(
                branches=(
                    "(and (<= (+ Y_2 X_0) Y_0))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "output_atom_references_x",
        )
        self._assert_source_rejected(
            _legacy_source(
                branches=(
                    "(and (>= Y_0 Y_2) (>= Y_1 Y_2))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "extra_conjunct",
        )
        self._assert_source_rejected(
            _legacy_source(
                output_body=(
                    "(and (or "
                    + " ".join(_DEFAULT_BRANCHES)
                    + ") (>= Y_0 Y_2))"
                )
            ),
            "root_not_or",
        )

    def test_additional_output_assert_is_implicit_conjunct_and_rejected(
        self,
    ) -> None:
        self._assert_source_rejected(
            _legacy_source(
                extra_asserts=("(assert (>= Y_0 Y_2))",)
            ),
            "exactly_one_output_assert",
        )


class RawVNNLibExactParserAdversarialTests(unittest.TestCase):
    def _reject(self, source: str, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory) / "bad.vnnlib", source)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, pattern
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=_sha256(path),
                    live_assert_params=_live_params(),
                )

    def test_near_unit_exact_decimal_is_not_rounded_to_one(self) -> None:
        near_one = (
            "1.0000000000000000000000000000000000000000000000000001"
        )
        self._reject(
            _legacy_source(
                branches=(
                    f"(and (>= (* {near_one} Y_0) Y_2))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "classification_atom_not_unit_margin",
        )

    def test_exact_nonzero_underflow_decimal_is_not_zero(self) -> None:
        tiny = "0." + ("0" * 399) + "1"
        self._reject(
            _legacy_source(
                branches=(
                    f"(and (>= (+ Y_0 {tiny}) Y_2))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "classification_atom_threshold_not_zero",
        )

    def test_tiny_exact_nonlinear_term_cannot_disappear(self) -> None:
        tiny = "0." + ("0" * 399) + "1"
        self._reject(
            _legacy_source(
                branches=(
                    (
                        "(and (<= (+ (- Y_2 Y_0) "
                        f"(* (* {tiny} Y_0) Y_1)) 0))"
                    ),
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "affine_nonlinear_product",
        )

    def test_v2_requires_real_and_declared_bracket_symbols(self) -> None:
        self._reject(
            _v2_source().replace(
                "(declare-output Y Real [4])",
                "(declare-output Y Int [4])",
            ),
            "declare_output_must_be_Real",
        )
        flat_aliases = (
            _v2_source()
            .replace("X[0]", "X_0")
            .replace("Y[0]", "Y_0")
            .replace("Y[1]", "Y_1")
            .replace("Y[2]", "Y_2")
            .replace("Y[3]", "Y_3")
        )
        self._reject(flat_aliases, "flat_alias_forbidden")

    def test_dimension_cap_precedes_dense_expression_work(self) -> None:
        self._reject(
            _v2_source().replace(
                "(declare-output Y Real [4])",
                "(declare-output Y Real [100000000])",
            ),
            "output_dimension_exceeds_cap",
        )

    def test_non_lra_logic_is_rejected(self) -> None:
        self._reject(
            _legacy_source().replace("QF_LRA", "QF_NIA"),
            "set_logic_must_be_QF_LRA",
        )

    def test_numeric_and_declaration_resource_boundaries(self) -> None:
        self._reject(
            _legacy_source(
                branches=(
                    "(and (>= (* 1e100000000 Y_0) Y_2))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "numeric_exponent_out_of_range",
        )
        self._reject(
            _legacy_source()
            + "\n(declare-const X_1 Real)\n",
            "declaration_after_assert",
        )
        self._reject(
            _legacy_source().replace(
                "(declare-const Y_0 Real)",
                "(declare-const Y_00 Real)",
            ),
            "name_not_canonical",
        )

    def test_ascii_canonical_numeric_shapes_and_indices_only(self) -> None:
        self._reject(
            _v2_source().replace(
                "(declare-output Y Real [4])",
                "(declare-output Y Real [04])",
            ),
            "output_shape_malformed",
        )
        self._reject(
            _v2_source().replace("Y[0]", "Y[00]"),
            "variable_index_malformed",
        )
        self._reject(
            _v2_source().replace(
                "(declare-output Y Real [4])",
                "(declare-output Y Real [٤])",
            ),
            "output_shape_malformed",
        )
        self._reject(
            _v2_source().replace("Y[0]", "Y[٠]"),
            "variable_index_malformed",
        )
        self._reject(
            _legacy_source(
                branches=(
                    "(and (>= (+ Y_0 00) Y_2))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "assert_expression_unknown_symbol",
        )
        self._reject(
            _legacy_source(
                branches=(
                    "(and (>= (+ Y_0 ٠) Y_2))",
                    _DEFAULT_BRANCHES[1],
                    _DEFAULT_BRANCHES[2],
                )
            ),
            "assert_expression_unknown_symbol",
        )


class RawVNNLibLiveBoundaryAdversarialTests(unittest.TestCase):
    def test_exact_keyset_and_explicit_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            digest = _sha256(path)
            extra = _live_params()
            extra["margin"] = torch.tensor([0.0], dtype=torch.float64)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "keyset_mismatch"
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=extra,
                )

            longdouble = {
                "kind": "TOP1_ROBUST",
                "C": np.asarray(
                    _live_params()["C"].numpy(), dtype=np.longdouble
                ),
                "thresholds": np.zeros(
                    (1, 3), dtype=np.longdouble
                ),
                "M": 3,
                "y_true": np.asarray([2], dtype=np.int64),
            }
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "dtype_unsupported"
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=longdouble,
                )

    def test_raw_float_bytes_and_signed_zero_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            digest = _sha256(path)
            signed_zero = _live_params()
            signed_zero["C"][0, 1] = -0.0
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "C_bit_mismatch"
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=signed_zero,
                )

            original = _live_params(dtype=torch.float64)
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=digest,
                live_assert_params=original,
            )
            changed_storage = _live_params(dtype=torch.float32)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "live_assert_binding_mismatch",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=changed_storage
                )

    def test_receipts_are_recursively_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            with self.assertRaises(TypeError):
                candidate.receipt["status"] = "forged"
            with self.assertRaises(TypeError):
                candidate.receipt["rows"][0]["competitor_class"] = 99
            self.assertIsInstance(
                candidate.receipt["rows"][0]["boolean_path"], tuple
            )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            with self.assertRaises(TypeError):
                batch.receipt["rivals"][0]["rival_id"] = 99

    def test_batch_owner_identity_rejects_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            self.assertFalse(hasattr(candidate, "rivals"))
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            copied = replace(batch)
            self.assertNotEqual(batch, copied)
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )
            self.assertFalse(
                validate_consumed_raw_vnnlib_rival_batch(copied)
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "live_identity_invalid"
            ):
                _ = copied.rivals

    def test_exact_runtime_types_reject_class_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )

            class CandidateSubclass(
                raw_adapter.RawVNNLibTop1Candidate
            ):
                __slots__ = ()

            object.__setattr__(
                candidate, "__class__", CandidateSubclass
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "candidate_wrong_type"
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )
            object.__setattr__(
                candidate,
                "__class__",
                raw_adapter.RawVNNLibTop1Candidate,
            )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )

            class BatchSubclass(ConsumedRivalBatch):
                __slots__ = ()

                @property
                def rivals(self):
                    return ("forged",)

            object.__setattr__(batch, "__class__", BatchSubclass)
            self.assertFalse(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )

    def test_issued_batch_snapshot_rejects_rehash_and_receipt_swap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            object.__setattr__(
                batch, "_rivals", tuple(reversed(batch._rivals))
            )
            object.__setattr__(
                batch,
                "batch_sha256",
                raw_adapter._batch_content_digest(batch),
            )
            self.assertFalse(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )

            second_candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            second_batch = consume_raw_vnnlib_top1_candidate(
                second_candidate, live_assert_params=params
            )
            object.__setattr__(
                second_batch,
                "receipt",
                MappingProxyType(
                    {
                        "status": "forged",
                        "proof_authority": True,
                    }
                ),
            )
            self.assertFalse(
                validate_consumed_raw_vnnlib_rival_batch(second_batch)
            )

    def test_candidate_receipt_identity_is_bound_and_single_use(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            object.__setattr__(
                candidate,
                "receipt",
                MappingProxyType(dict(candidate.receipt)),
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "candidate_runtime_snapshot_mismatch",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "missing_consumed_or_expired",
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )


class RawVNNLibLifecycleAndIOTests(unittest.TestCase):
    def test_abandoned_candidate_is_weakly_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            baseline = len(raw_adapter._LIVE_RECORDS)
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=_live_params(),
            )
            self.assertEqual(
                len(raw_adapter._LIVE_RECORDS), baseline + 1
            )
            del candidate
            gc.collect()
            self.assertEqual(len(raw_adapter._LIVE_RECORDS), baseline)

    def test_ttl_capacity_and_explicit_revoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            digest = _sha256(path)
            params = _live_params()
            expiring = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=digest,
                live_assert_params=params,
                capability_ttl_seconds=0.001,
            )
            time.sleep(0.01)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "expired"
            ):
                consume_raw_vnnlib_top1_candidate(
                    expiring, live_assert_params=params
                )

            with mock.patch.object(
                raw_adapter, "_MAX_LIVE_RECORDS", 2
            ):
                first = issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=params,
                )
                second = issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=params,
                )
                with self.assertRaisesRegex(
                    RawVNNLibRivalAdapterError, "capacity_exceeded"
                ):
                    issue_raw_vnnlib_top1_candidate(
                        path,
                        expected_vnnlib_sha256=digest,
                        live_assert_params=params,
                    )
                self.assertTrue(
                    revoke_raw_vnnlib_top1_candidate(first)
                )
                self.assertFalse(
                    revoke_raw_vnnlib_top1_candidate(first)
                )
                self.assertTrue(
                    revoke_raw_vnnlib_top1_candidate(second)
                )

    def test_same_bytes_new_inode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            payload = path.read_bytes()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            replacement = Path(directory) / "replacement.vnnlib"
            replacement.write_bytes(payload)
            os.replace(replacement, path)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "source_identity"
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate, live_assert_params=params
                )

    def test_deadline_rejects_without_accidentally_consuming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            digest = _sha256(path)
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "deadline_expired"
            ):
                issue_raw_vnnlib_top1_candidate(
                    path,
                    expected_vnnlib_sha256=digest,
                    live_assert_params=params,
                    deadline=time.monotonic() - 1.0,
                )
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=digest,
                live_assert_params=params,
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError, "deadline_expired"
            ):
                consume_raw_vnnlib_top1_candidate(
                    candidate,
                    live_assert_params=params,
                    deadline=time.monotonic() - 1.0,
                )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate, live_assert_params=params
            )
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )

    def test_registry_record_type_and_rivals_identity_are_exact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write(
                Path(directory) / "top1.vnnlib", _legacy_source()
            )
            params = _live_params()
            first = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            first_record = raw_adapter._LIVE_RECORDS[
                id(first._live_capability)
            ]

            class RecordSubclass(raw_adapter._LiveRecord):
                __slots__ = ()

            object.__setattr__(
                first_record, "__class__", RecordSubclass
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "missing_consumed_or_expired",
            ):
                consume_raw_vnnlib_top1_candidate(
                    first, live_assert_params=params
                )

            second = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=_sha256(path),
                live_assert_params=params,
            )
            second_record = raw_adapter._LIVE_RECORDS[
                id(second._live_capability)
            ]
            object.__setattr__(
                second_record,
                "rivals",
                tuple(list(second_record.rivals)),
            )
            with self.assertRaisesRegex(
                RawVNNLibRivalAdapterError,
                "candidate_runtime_snapshot_mismatch",
            ):
                consume_raw_vnnlib_top1_candidate(
                    second, live_assert_params=params
                )

    def test_sparse_legacy_index_avoids_large_contiguity_set(
        self,
    ) -> None:
        source = _legacy_source().replace(
            "(declare-const X_0 Real)",
            (
                "(declare-const X_0 Real)\n"
                "(declare-const X_9999999 Real)"
            ),
        )
        real_set = set
        range_arguments = []

        def tracking_set(value=()):
            if isinstance(value, range):
                range_arguments.append(value)
            return real_set(value)

        with tempfile.TemporaryDirectory() as directory:
            path = _write(Path(directory) / "sparse.vnnlib", source)
            with mock.patch("builtins.set", side_effect=tracking_set):
                with self.assertRaisesRegex(
                    RawVNNLibRivalAdapterError,
                    "X_declarations_not_contiguous",
                ):
                    issue_raw_vnnlib_top1_candidate(
                        path,
                        expected_vnnlib_sha256=_sha256(path),
                        live_assert_params=_live_params(),
                    )
        self.assertEqual(range_arguments, [])


if __name__ == "__main__":
    unittest.main()
