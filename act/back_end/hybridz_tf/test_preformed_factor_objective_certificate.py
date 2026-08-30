from __future__ import annotations

import inspect
import copy
import gc
import unittest
import weakref
from fractions import Fraction
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.test_split_block_lp_certificate import (
    _exact_vertex_maximum,
    _random_case,
    _split_call,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    build_objective_binding,
    verify_objective_binding,
)
from act.back_end.solver import solver_hz as solver_hz_module
from act.back_end.solver.solver_hz import (
    _HZPreformedFactorObjectiveEnvelope,
    _hz_form_exact_factor_objective_envelope_from_live_split_blocks,
    _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope,
    _hz_read_exact_objective_binding_material_from_factor_envelope,
)


_PARENT_DIGEST = "17" * 32


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_factor_objective(case):
    row = case["C_row"]
    continuous = tuple(
        sum(
            _fraction(row[output])
            * _fraction(case["Gc_dense"][output, column])
            for output in range(row.size)
        )
        for column in range(case["Gc_dense"].shape[1])
    )
    binary = tuple(
        sum(
            _fraction(row[output])
            * _fraction(case["Gb_dense"][output, column])
            for output in range(row.size)
        )
        for column in range(case["Gb_dense"].shape[1])
    )
    center = (
        sum(
            _fraction(row[output]) * _fraction(case["c"][output])
            for output in range(row.size)
        )
        - _fraction(case["threshold"])
    )
    return center, continuous, binary


def _form(case, **updates):
    arguments = {
        "c": case["c"],
        "Gc": case["Gc"],
        "Gb": case["Gb"],
        "C_row": case["C_row"],
        "threshold": case["threshold"],
        "continuous_col_ids": np.arange(
            100, 100 + case["Gc"].shape[1], dtype=np.int64
        ),
        "binary_col_ids": np.arange(
            1000, 1000 + case["Gb"].shape[1], dtype=np.int64
        ),
        "objective_id": "test:rival:preformed",
        "parent_semantic_digest": _PARENT_DIGEST,
    }
    arguments.update(updates)
    return _hz_form_exact_factor_objective_envelope_from_live_split_blocks(
        **arguments
    )


def _preformed_call(case, envelope, **updates):
    arguments = {
        "objective_envelope": envelope,
        "expected_parent_semantic_digest": _PARENT_DIGEST,
        "expected_exact_objective_sha256": (
            getattr(envelope, "exact_objective_sha256", "00" * 32)
        ),
        "expected_objective_binding_sha256": (
            getattr(envelope, "objective_binding_sha256", "00" * 32)
        ),
        "Auc": case["Auc"],
        "Aub": case["Aub"],
        "Ac": case["Ac"],
        "Ab": case["Ab"],
        "ub": case["ub"],
        "b": case["b"],
        "continuous_lb": case["continuous_lb"],
        "continuous_ub": case["continuous_ub"],
        "binary_lb": case["binary_lb"],
        "binary_ub": case["binary_ub"],
        "upper_row_dual": case["upper_row_dual"],
        "equality_row_dual": case["equality_row_dual"],
    }
    arguments.update(updates)
    return (
        _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope(
            **arguments
        )
    )


class TestPreformedFactorObjectiveCertificate(unittest.TestCase):
    def test_exact_fraction_components_are_enclosed_and_readonly(self):
        rng = np.random.default_rng(20260808)
        for seed in range(12):
            with self.subTest(seed=seed):
                case = _random_case(seed)
                case["c"] = rng.normal(size=2).astype(np.float64)
                case["C_row"] = rng.normal(size=2).astype(np.float64)
                case["threshold"] = float(rng.normal())
                case["Gc_dense"] = rng.normal(size=(2, 2)).astype(np.float64)
                case["Gb_dense"] = rng.normal(size=(2, 1)).astype(np.float64)
                case["Gc"] = sp.csr_matrix(case["Gc_dense"])
                case["Gb"] = sp.csr_matrix(case["Gb_dense"])
                envelope, receipt = _form(case)
                self.assertIsNotNone(envelope, receipt)
                self.assertEqual(receipt["status"], "formed")
                self.assertFalse(receipt["proof_authority"])
                self.assertFalse(receipt["verdict_authority"])
                self.assertFalse(receipt["production_ready"])
                exact_center, exact_continuous, exact_binary = (
                    _exact_factor_objective(case)
                )
                for exact, nominal, error in zip(
                    exact_continuous,
                    envelope.q_continuous_hat,
                    envelope.q_continuous_error,
                ):
                    self.assertLessEqual(
                        abs(exact - _fraction(nominal)), _fraction(error)
                    )
                for exact, nominal, error in zip(
                    exact_binary,
                    envelope.q_binary_hat,
                    envelope.q_binary_error,
                ):
                    self.assertLessEqual(
                        abs(exact - _fraction(nominal)), _fraction(error)
                    )
                self.assertLessEqual(
                    abs(exact_center - _fraction(envelope.kappa_hat)),
                    _fraction(envelope.kappa_error),
                )
                binding = build_objective_binding(
                    objective_id="test:rival:preformed",
                    parent_semantic_digest=_PARENT_DIGEST,
                    center=exact_center,
                    continuous_terms=tuple(
                        (100 + index, value)
                        for index, value in enumerate(exact_continuous)
                        if value != 0
                    ),
                    binary_terms=tuple(
                        (1000 + index, value)
                        for index, value in enumerate(exact_binary)
                        if value != 0
                    ),
                )
                self.assertTrue(verify_objective_binding(binding))
                self.assertEqual(
                    envelope.objective_binding_sha256,
                    binding.objective_binding_sha256,
                )
                material = (
                    _hz_read_exact_objective_binding_material_from_factor_envelope(
                        envelope,
                        expected_parent_semantic_digest=_PARENT_DIGEST,
                        expected_objective_id="test:rival:preformed",
                    )
                )
                self.assertEqual(material[0], exact_center)
                self.assertEqual(material[1], binding.continuous_terms)
                self.assertEqual(material[2], binding.binary_terms)
                self.assertEqual(
                    material[3], binding.objective_binding_sha256
                )
                for values in (
                    envelope.q_continuous_hat,
                    envelope.q_continuous_error,
                    envelope.q_binary_hat,
                    envelope.q_binary_error,
                ):
                    self.assertFalse(values.flags.writeable)
                    with self.assertRaises(ValueError):
                        values.setflags(write=True)

    def test_random_fraction_vertex_oracle_and_legacy_differential(self):
        saw_nonzero_objective_error = False
        for seed in range(24):
            with self.subTest(seed=seed):
                case = _random_case(seed)
                if seed % 2:
                    rng = np.random.default_rng(9000 + seed)
                    case["c"] = rng.normal(size=2).astype(np.float64)
                    case["C_row"] = rng.normal(size=2).astype(np.float64)
                    case["threshold"] = float(rng.normal())
                    case["Gc_dense"] = rng.normal(size=(2, 2)).astype(
                        np.float64
                    )
                    case["Gb_dense"] = rng.normal(size=(2, 1)).astype(
                        np.float64
                    )
                    case["Gc"] = sp.csr_matrix(case["Gc_dense"])
                    case["Gb"] = sp.csr_matrix(case["Gb_dense"])
                envelope, formation_receipt = _form(case)
                self.assertEqual(formation_receipt["status"], "formed")
                saw_nonzero_objective_error = (
                    saw_nonzero_objective_error
                    or bool(np.any(envelope.q_continuous_error > 0.0))
                    or bool(np.any(envelope.q_binary_error > 0.0))
                    or envelope.kappa_error > 0.0
                )
                preformed, receipt = _preformed_call(case, envelope)
                legacy, legacy_receipt = _split_call(case)
                self.assertIsNotNone(preformed)
                self.assertIsNotNone(legacy)
                self.assertEqual(receipt["status"], "verified_upper")
                self.assertEqual(legacy_receipt["status"], "verified_upper")
                exact = _exact_vertex_maximum(case)
                self.assertGreaterEqual(
                    Fraction.from_float(receipt["upper"]), exact
                )
                self.assertGreaterEqual(
                    Fraction.from_float(legacy_receipt["upper"]), exact
                )
                self.assertTrue(receipt["proof_authority"])
                self.assertFalse(receipt["verdict_authority"])
                self.assertEqual(receipt["generator_source_read_count"], 0)
                self.assertEqual(receipt["envelope_rehash_bytes"], 0)
                self.assertNotIn(
                    "objective_envelope_persistent_bytes", receipt
                )
                self.assertEqual(
                    receipt["packed_factor_persistent_bytes"],
                    16
                    * (
                        case["Gc"].shape[1]
                        + case["Gb"].shape[1]
                    ),
                )
                self.assertTrue(
                    receipt[
                        "packed_factor_persistent_bytes_lower_bound_only"
                    ]
                )
                self.assertFalse(
                    receipt["total_persistent_bytes_bounded"]
                )
                self.assertEqual(
                    receipt["total_persistent_bytes_blocker"],
                    "python_fraction_exact_binding_material_v1",
                )
                self.assertEqual(
                    receipt["trust_boundary"],
                    "process_local_registry_and_solver_module_state_"
                    "trusted_v1",
                )
                self.assertEqual(
                    receipt["route"],
                    "native_hz_preformed_objective_split_csr_"
                    "no_generator_read_v1",
                )
        self.assertTrue(saw_nonzero_objective_error)

    def test_sixteen_replays_cannot_reenter_generator_or_hash_formation(self):
        case = _random_case(31)
        envelope, receipt = _form(case)
        self.assertEqual(receipt["generator_validation_pass_count"], 1)
        self.assertEqual(receipt["source_hash_pass_count"], 1)
        self.assertEqual(receipt["exact_expansion_pass_count"], 1)
        original_weighted = (
            solver_hz_module._hz_ld_sparse_weighted_columns_split
        )

        def constraints_only(matrix, *args, **kwargs):
            self.assertIsNot(matrix, case["Gc"])
            self.assertIsNot(matrix, case["Gb"])
            return original_weighted(matrix, *args, **kwargs)

        signature = inspect.signature(
            _hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
        )
        for forbidden in ("c", "Gc", "Gb", "C_row", "threshold"):
            self.assertNotIn(forbidden, signature.parameters)
        with (
            mock.patch.object(
                solver_hz_module,
                "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
                side_effect=AssertionError("formation must not repeat"),
            ),
            mock.patch.object(
                solver_hz_module,
                "_hz_preformed_hash_array",
                side_effect=AssertionError("envelope must not be rehashed"),
            ),
            mock.patch.object(
                solver_hz_module,
                "_hz_ld_sparse_weighted_columns_split",
                side_effect=constraints_only,
            ),
        ):
            for _ in range(16):
                value, replay_receipt = _preformed_call(case, envelope)
                self.assertIsNotNone(value)
                self.assertEqual(
                    replay_receipt["generator_source_read_count"], 0
                )
                self.assertEqual(replay_receipt["envelope_rehash_bytes"], 0)

    def test_illegal_and_nonfinite_duals_are_safely_zeroed(self):
        case = _random_case(52)
        envelope, _ = _form(case)
        value, receipt = _preformed_call(
            case,
            envelope,
            upper_row_dual=np.asarray([0.5, -0.25], dtype=np.float64),
            equality_row_dual=np.asarray([np.nan], dtype=np.float64),
        )
        self.assertIsNotNone(value)
        self.assertEqual(receipt["illegal_sign_projected"], 1)
        self.assertEqual(receipt["nonfinite_dual_zeroed"], 1)
        self.assertGreaterEqual(
            Fraction.from_float(receipt["upper"]),
            _exact_vertex_maximum(case),
        )

    def test_zero_equality_and_binary_blocks(self):
        case = _random_case(61)
        case.update({
            "Gb_dense": np.zeros((2, 0), dtype=np.float64),
            "Aub_dense": np.zeros((2, 0), dtype=np.float64),
            "Ac_dense": np.zeros((0, 2), dtype=np.float64),
            "Ab_dense": np.zeros((0, 0), dtype=np.float64),
            "b": np.zeros(0, dtype=np.float64),
            "binary_lb": np.zeros(0, dtype=np.float64),
            "binary_ub": np.zeros(0, dtype=np.float64),
            "equality_row_dual": np.zeros(0, dtype=np.float64),
        })
        for name in ("Gb", "Aub", "Ac", "Ab"):
            case[name] = sp.csr_matrix(case[f"{name}_dense"])
        envelope, formation_receipt = _form(case)
        self.assertEqual(formation_receipt["status"], "formed")
        value, receipt = _preformed_call(case, envelope)
        self.assertIsNotNone(value)
        self.assertEqual(receipt["status"], "verified_upper")
        self.assertGreaterEqual(
            Fraction.from_float(receipt["upper"]),
            _exact_vertex_maximum(case),
        )

    def test_tamper_stale_digest_and_naked_object_fail_closed(self):
        case = _random_case(71)
        envelope, _ = _form(case)
        value, receipt = _preformed_call(
            case,
            envelope,
            expected_parent_semantic_digest="22" * 32,
        )
        self.assertIsNone(value)
        self.assertIn("parent digest mismatch", receipt["status"])
        value, receipt = _preformed_call(
            case,
            envelope,
            expected_exact_objective_sha256="33" * 32,
        )
        self.assertIsNone(value)
        self.assertIn("exact digest mismatch", receipt["status"])
        value, receipt = _preformed_call(
            case,
            envelope,
            expected_objective_binding_sha256="44" * 32,
        )
        self.assertIsNone(value)
        self.assertIn("binding digest mismatch", receipt["status"])

        tampered, _ = _form(case)
        object.__setattr__(
            tampered,
            "_q_continuous_hat_bytes",
            bytes(len(tampered._q_continuous_hat_bytes)),
        )
        value, receipt = _preformed_call(case, tampered)
        self.assertIsNone(value)
        self.assertIn("registry identity seal is invalid", receipt["status"])

        # Regression for hostile in-process re-sealing: the old validator
        # trusted a token exposed in ``_seal`` and accepted a freshly rebuilt
        # identity tuple after the packed objective bytes were replaced.
        resealed, _ = _form(case)
        forged_hat = np.zeros(
            resealed.n_continuous, dtype="<f8"
        )
        if forged_hat.size:
            forged_hat[0] = np.nextafter(0.0, np.inf)
        object.__setattr__(
            resealed,
            "_q_continuous_hat_bytes",
            forged_hat.tobytes(),
        )
        forged_values = tuple(
            getattr(resealed, name) for name in resealed.__slots__[:-2]
        )
        object.__setattr__(resealed, "_seal", forged_values)
        value, receipt = _preformed_call(case, resealed)
        self.assertIsNone(value)
        self.assertIn("registry identity seal is invalid", receipt["status"])

        original, _ = _form(case)
        with self.assertRaises(TypeError):
            copy.copy(original)
        with self.assertRaises(TypeError):
            copy.deepcopy(original)

        naked = object.__new__(_HZPreformedFactorObjectiveEnvelope)
        value, receipt = _preformed_call(
            case,
            naked,
            expected_exact_objective_sha256="55" * 32,
            expected_objective_binding_sha256="66" * 32,
        )
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("invalid:"))

    def test_process_registry_is_multiuse_and_weakref_lifetime_bounded(self):
        case = _random_case(75)
        envelope, receipt = _form(case)
        self.assertTrue(receipt["process_local_registry"])
        self.assertFalse(receipt["registry_one_use"])
        identity = id(envelope)
        reference = weakref.ref(envelope)
        self.assertIn(
            identity,
            solver_hz_module._HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY,
        )
        for _ in range(3):
            value, replay_receipt = _preformed_call(case, envelope)
            self.assertIsNotNone(value)
            self.assertEqual(replay_receipt["status"], "verified_upper")
        del envelope
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(
            identity,
            solver_hz_module._HZ_PREFORMED_FACTOR_OBJECTIVE_REGISTRY,
        )

    def test_platform_and_deadlines_fail_closed(self):
        case = _random_case(81)
        envelope, _ = _form(case)
        with mock.patch.object(
            solver_hz_module,
            "_hz_longdouble_certificate_platform",
            return_value=(False, "test_unsupported"),
        ):
            value, receipt = _preformed_call(case, envelope)
        self.assertIsNone(value)
        self.assertEqual(receipt["status"], "platform_unsupported")

        envelope, receipt = _form(case, deadline=0.0)
        self.assertIsNone(envelope)
        self.assertTrue(receipt["status"].startswith("deadline_exhausted:"))
        valid, _ = _form(case)
        value, receipt = _preformed_call(case, valid, deadline=0.0)
        self.assertIsNone(value)
        self.assertTrue(receipt["status"].startswith("deadline_exhausted:"))
        self.assertIsNone(receipt["upper"])

    def test_deadline_crossed_during_final_diagnostics_cannot_authorize(self):
        case = _random_case(82)
        envelope, formation_receipt = _form(case)
        self.assertEqual(formation_receipt["status"], "formed")
        clock = {"now": 1.0}
        original_max = np.max

        def max_and_cross_deadline(*args, **kwargs):
            result = original_max(*args, **kwargs)
            clock["now"] = 11.0
            return result

        with (
            mock.patch.object(
                solver_hz_module.time,
                "monotonic",
                side_effect=lambda: clock["now"],
            ),
            mock.patch.object(
                solver_hz_module.np,
                "max",
                side_effect=max_and_cross_deadline,
            ) as max_mock,
        ):
            value, receipt = _preformed_call(
                case, envelope, deadline=10.0
            )
        self.assertGreaterEqual(max_mock.call_count, 1)
        self.assertIsNone(value)
        self.assertEqual(
            receipt["status"],
            "deadline_exhausted:preformed_checker_after_diagnostics_"
            "before_receipt",
        )
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["pcoh_authorization"])
        self.assertIsNone(receipt["upper"])

    def test_no_sparse_stack_is_reachable(self):
        case = _random_case(91)
        envelope, _ = _form(case)
        with (
            mock.patch.object(
                solver_hz_module._sp,
                "hstack",
                side_effect=AssertionError("hstack forbidden"),
            ),
            mock.patch.object(
                solver_hz_module._sp,
                "vstack",
                side_effect=AssertionError("vstack forbidden"),
            ),
        ):
            value, receipt = _preformed_call(case, envelope)
        self.assertIsNotNone(value)
        self.assertEqual(receipt["assembled_sparse_nnz"], 0)

    def test_hashes_are_deterministic_source_sensitive_and_caps_are_strict(self):
        case = _random_case(101)
        first, first_receipt = _form(
            case,
            continuous_col_ids=np.asarray([901, 3], dtype=np.int64),
            binary_col_ids=np.asarray([77], dtype=np.int64),
        )
        second, second_receipt = _form(
            case,
            continuous_col_ids=np.asarray([901, 3], dtype=np.int64),
            binary_col_ids=np.asarray([77], dtype=np.int64),
        )
        self.assertEqual(first.envelope_sha256, second.envelope_sha256)
        self.assertEqual(
            first.objective_binding_sha256,
            second.objective_binding_sha256,
        )
        changed_row = case["C_row"].copy()
        changed_row[0] = np.nextafter(changed_row[0], np.inf)
        changed, changed_receipt = _form(
            case,
            C_row=changed_row,
            continuous_col_ids=np.asarray([901, 3], dtype=np.int64),
            binary_col_ids=np.asarray([77], dtype=np.int64),
        )
        self.assertNotEqual(
            first_receipt["objective_source_sha256"],
            changed_receipt["objective_source_sha256"],
        )
        self.assertNotEqual(first.envelope_sha256, changed.envelope_sha256)

        too_wide = {
            "c": np.asarray([0.0], dtype=np.float64),
            "Gc": sp.csr_matrix((1, 1_000_001), dtype=np.float64),
            "Gb": sp.csr_matrix((1, 0), dtype=np.float64),
            "C_row": np.asarray([1.0], dtype=np.float64),
            "threshold": 0.0,
        }
        envelope, receipt = _form(
            too_wide,
            continuous_col_ids=(),
            binary_col_ids=(),
        )
        self.assertIsNone(envelope)
        self.assertIn("exceeds the factor cap", receipt["status"])

    def test_packed_memory_slope_250k_500k_1m_columns(self):
        packed_sizes = []
        for columns in (250_000, 500_000, 1_000_000):
            case = {
                "c": np.asarray([0.0], dtype=np.float64),
                "Gc": sp.csr_matrix(
                    (
                        np.asarray([1.0], dtype=np.float64),
                        np.asarray([columns - 1], dtype=np.int32),
                        np.asarray([0, 1], dtype=np.int32),
                    ),
                    shape=(1, columns),
                ),
                "Gb": sp.csr_matrix((1, 0), dtype=np.float64),
                "C_row": np.asarray([1.0], dtype=np.float64),
                "threshold": 0.0,
            }
            envelope, receipt = _form(
                case,
                continuous_col_ids=np.arange(columns, dtype=np.int64),
                binary_col_ids=np.zeros(0, dtype=np.int64),
            )
            self.assertIsNotNone(envelope, receipt)
            self.assertEqual(receipt["exact_term_count"], 3)
            self.assertEqual(receipt["packed_factor_bytes"], 16 * columns)
            self.assertEqual(
                envelope.q_continuous_hat.nbytes
                + envelope.q_continuous_error.nbytes,
                16 * columns,
            )
            packed_sizes.append(receipt["packed_factor_bytes"])
        self.assertEqual(packed_sizes, [4_000_000, 8_000_000, 16_000_000])


if __name__ == "__main__":
    unittest.main()
