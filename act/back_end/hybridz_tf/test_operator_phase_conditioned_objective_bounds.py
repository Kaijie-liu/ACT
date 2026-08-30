#!/usr/bin/env python3
"""Soundness audits for toy-first Operator-HZ conditional PCOH bounds."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import itertools
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Optional, Sequence
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    RivalSpec,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    operator_phase_conditioned_objective_bounds as bound_module,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
    verify_operator_exact_relu_property_phase_selection,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    build_operator_hz,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_bounds import (
    OperatorPhaseConditionedObjectiveBoundError,
    OperatorPhaseConditionedScheduledStop,
    OperatorPhaseConditionedScheduledStopPolicy,
    build_complete_operator_phase_conditioned_objective_bounds,
    build_operator_phase_conditioned_objective_bound,
    build_scheduled_complete_operator_phase_conditioned_objective_bounds,
    replay_complete_operator_phase_conditioned_objective_bounds,
    verify_operator_phase_conditioned_scheduled_stop_record,
    verify_operator_phase_conditioned_objective_bound,
    verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure,
)
from act.back_end.hybridz_tf.operator_phase_conditioned_objective_hull import (
    build_phase_conditioned_objective_hull,
)
from act.back_end.solver.solver_hz import SparseHZono


_DTYPE = torch.float64
_CORNER_WEIGHTS = (
    (Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(-1)),
    (Fraction(-1), Fraction(1)),
    (Fraction(-1), Fraction(-1)),
)


def _layer(
    layer_id: int,
    kind: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    width: int,
) -> Any:
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        in_vars=[],
        out_vars=[(int(layer_id), row) for row in range(int(width))],
    )


def _dense(
    layer_id: int,
    weight: Sequence[Sequence[float]],
    bias: Sequence[float],
) -> Any:
    weight_array = np.asarray(weight, dtype=np.float64)
    bias_array = np.asarray(bias, dtype=np.float64)
    return _layer(
        layer_id,
        "DENSE",
        {
            "weight": torch.tensor(weight_array, dtype=_DTYPE),
            "bias": torch.tensor(bias_array, dtype=_DTYPE),
            "in_features": int(weight_array.shape[1]),
            "out_features": int(weight_array.shape[0]),
        },
        width=int(weight_array.shape[0]),
    )


def _k4_corner_build() -> OperatorHZBuild:
    """Four exact ReLUs with mutually exclusive positive corner phases."""

    lower = torch.tensor([[-1.0, -1.0]], dtype=_DTYPE)
    upper = torch.tensor([[1.0, 1.0]], dtype=_DTYPE)
    layers = [
        _layer(0, "INPUT", {"shape": (1, 2)}, width=2),
        _layer(
            1,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": lower, "ub": upper},
            width=2,
        ),
        _dense(
            2,
            (
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ),
            (-1.5, -1.5, -1.5, -1.5),
        ),
        _layer(3, "RELU", width=4),
        _dense(
            4,
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 1.0),
                (0.5, 0.5, 0.5, 0.5),
            ),
            (0.75, 0.0, 0.0),
        ),
        _layer(5, "ASSERT", width=3),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    succs = {layer.id: [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {}
    for layer in layers:
        width = len(layer.out_vars)
        if layer.kind in {"INPUT", "INPUT_SPEC"}:
            fact_lower, fact_upper = lower.clone(), upper.clone()
        else:
            fact_lower = torch.full((1, width), -1.0e30, dtype=_DTYPE)
            fact_upper = torch.full((1, width), 1.0e30, dtype=_DTYPE)
        facts[layer.id] = Fact(Bounds(fact_lower, fact_upper), ConSet())
    return build_operator_hz(
        net,
        facts,
        facts,
        exact_budget=4,
        materialize_add=True,
    )


def _rivals() -> tuple[RivalSpec, RivalSpec]:
    return (
        RivalSpec(
            rival_id=10,
            objective=(-1.0, 1.0, 0.0),
            threshold=0.0,
            assert_digest="a" * 64,
        ),
        RivalSpec(
            rival_id=20,
            objective=(-1.0, 0.0, 1.0),
            threshold=0.0,
            assert_digest="b" * 64,
        ),
    )


def _clone_hz(source: SparseHZono, **overrides: Any) -> SparseHZono:
    row_tags = tuple(
        overrides.pop("row_tags", source._solver_constraint_row_tags)
    )
    values = {
        "c": source.c.copy(),
        "Gc": source.Gc.copy(),
        "Gb": source.Gb.copy(),
        "Ac": source.Ac.copy(),
        "Ab": source.Ab.copy(),
        "b": source.b.copy(),
        "Auc": source.Auc.copy(),
        "Aub": source.Aub.copy(),
        "ub": source.ub.copy(),
        "col_ids": source.col_ids.copy(),
        "bcol_ids": source.bcol_ids.copy(),
    }
    values.update(overrides)
    result = SparseHZono(**values)
    setattr(result, "_solver_constraint_row_tags", row_tags)
    return result


def _with_unrelated_rows(build: OperatorHZBuild) -> OperatorHZBuild:
    hz = build.hz
    Ac = sp.vstack(
        [sp.csr_matrix((1, hz.n_cont), dtype=np.float64), hz.Ac],
        format="csr",
    )
    Ab = sp.vstack(
        [sp.csr_matrix((1, hz.n_bin), dtype=np.float64), hz.Ab],
        format="csr",
    )
    Auc = sp.vstack(
        [hz.Auc, sp.csr_matrix((1, hz.n_cont), dtype=np.float64)],
        format="csr",
    )
    Aub = sp.vstack(
        [hz.Aub, sp.csr_matrix((1, hz.n_bin), dtype=np.float64)],
        format="csr",
    )
    tags = (
        ("toy_unrelated_equality",)
        + tuple(hz._solver_constraint_row_tags)
        + ("toy_unrelated_upper",)
    )
    cloned = _clone_hz(
        hz,
        Ac=Ac,
        Ab=Ab,
        b=np.concatenate([np.asarray([0.0]), hz.b]),
        Auc=Auc,
        Aub=Aub,
        ub=np.concatenate([hz.ub, np.asarray([100.0])]),
        row_tags=tags,
    )
    return replace(build, hz=cloned)


def _intersection(
    left: tuple[Fraction, Fraction, Fraction],
    right: tuple[Fraction, Fraction, Fraction],
) -> Optional[tuple[Fraction, Fraction]]:
    a, b, d = left
    e, f, g = right
    determinant = a * f - b * e
    if determinant == 0:
        return None
    return ((d * f - b * g) / determinant, (a * g - d * e) / determinant)


def _exact_true_network_conditional_upper(
    k: int,
    pattern: tuple[int, ...],
) -> Optional[Fraction]:
    """Exact-Fraction vertex oracle for the actual two-input ReLU network."""

    halfspaces = [
        (Fraction(1), Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(-1), Fraction(1)),
    ]
    for (a, b), phase in zip(_CORNER_WEIGHTS[:k], pattern):
        if phase == 1:
            halfspaces.append((-a, -b, Fraction(-3, 2)))
        else:
            halfspaces.append((a, b, Fraction(3, 2)))
    points = set()
    for left, right in itertools.combinations(halfspaces, 2):
        point = _intersection(left, right)
        if point is None:
            continue
        x, y = point
        if all(a * x + b * y <= d for a, b, d in halfspaces):
            points.add(point)
    if not points:
        return None

    def objective(point: tuple[Fraction, Fraction]) -> Fraction:
        x, y = point
        preactivations = tuple(
            a * x + b * y - Fraction(3, 2)
            for a, b in _CORNER_WEIGHTS
        )
        return sum(
            (max(Fraction(0), value) for value in preactivations),
            Fraction(-3, 4),
        )

    return max(objective(point) for point in points)


class OperatorPhaseConditionedObjectiveBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build = _k4_corner_build()
        cls.rivals = _rivals()
        cls.selection = derive_operator_exact_relu_property_phase_literals(
            cls.build, cls.rivals
        )
        cls.ids = tuple(
            mapping.stable_bcol_id for mapping in cls.selection.mappings
        )

    def _bound(self, k: int, pattern: tuple[int, ...]):
        return build_operator_phase_conditioned_objective_bound(
            self.build,
            self.rivals,
            self.selection,
            focused_rival_id=10,
            stable_bit_ids=self.ids[:k],
            pattern=pattern,
        )

    def _scheduled(
        self,
        k: int,
        schedule: tuple[tuple[int, ...], ...],
        *,
        stop_policy: OperatorPhaseConditionedScheduledStopPolicy = (
            OperatorPhaseConditionedScheduledStopPolicy()
        ),
    ):
        return build_scheduled_complete_operator_phase_conditioned_objective_bounds(
            self.build,
            self.rivals,
            self.selection,
            focused_rival_id=10,
            stable_bit_ids=self.ids[:k],
            evaluation_schedule=schedule,
            deadline=time.monotonic() + 10.0,
            stop_policy=stop_policy,
        )

    @staticmethod
    def _rehashed_telemetry(
        telemetry: Mapping[str, Any], **changes: Any
    ) -> MappingProxyType:
        payload = dict(telemetry)
        payload.update(changes)
        payload.pop("telemetry_sha256", None)
        payload["telemetry_sha256"] = bound_module._canonical_sha256(
            payload
        )
        frozen = bound_module._deep_freeze(payload)
        if type(frozen) is not MappingProxyType:
            raise AssertionError("test telemetry did not freeze")
        return frozen

    @staticmethod
    def _resealed_scheduled_result(
        result: Any,
        *,
        stop_policy: Optional[
            OperatorPhaseConditionedScheduledStopPolicy
        ] = None,
        telemetry: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        policy = result.stop_policy if stop_policy is None else stop_policy
        trace = result.telemetry if telemetry is None else telemetry
        receipt = bound_module._scheduled_receipt(
            parent_semantic_digest=result.parent_semantic_digest,
            stable_bit_ids=result.stable_bit_ids,
            canonical_patterns=result.canonical_patterns,
            evaluation_schedule=result.evaluation_schedule,
            stop_policy=policy,
            certificates=result.certificates,
            telemetry=trace,
        )
        provisional = replace(
            result,
            stop_policy=policy,
            telemetry=trace,
            receipt=receipt,
            bundle_sha256="",
        )
        return replace(
            provisional,
            bundle_sha256=bound_module._canonical_sha256(
                bound_module._scheduled_bundle_payload(
                    provisional, include_digest=False
                )
            ),
        )

    @staticmethod
    def _resealed_stop_record(record: Any, **changes: Any) -> Any:
        provisional = replace(record, **changes, record_sha256="")
        return replace(
            provisional,
            record_sha256=bound_module._canonical_sha256(
                bound_module._scheduled_stop_payload(
                    provisional, include_digest=False
                )
            ),
        )

    def test_k1_to_k4_all_patterns_dominate_exact_fraction_vertex_oracle(self):
        for k in range(1, 5):
            for pattern in itertools.product((-1, 1), repeat=k):
                with self.subTest(k=k, pattern=pattern):
                    certificate = self._bound(k, tuple(pattern))
                    exact = _exact_true_network_conditional_upper(
                        k, tuple(pattern)
                    )
                    if exact is not None:
                        self.assertGreaterEqual(
                            Fraction.from_float(certificate.upper_stored),
                            exact,
                        )
                    self.assertLessEqual(
                        certificate.upper_stored,
                        certificate.global_cube_upper,
                    )
                    self.assertTrue(
                        verify_operator_phase_conditioned_objective_bound(
                            self.build,
                            self.rivals,
                            self.selection,
                            certificate,
                        )
                    )
                    self.assertFalse(certificate.proof_authority)

    def test_candidate_positive_control_and_strict_outward_rounding(self):
        certificate = self._bound(4, (-1, -1, -1, -1))
        self.assertTrue(certificate.candidate_dual_accepted)
        self.assertEqual(certificate.selected_source, "candidate_local_dual")
        self.assertLess(
            certificate.candidate_checked_upper,
            certificate.zero_dual_fixed_upper,
        )
        # Exact maximum is -3/4.  The strict long-double checker deliberately
        # stores a slightly larger outward float, never the inward neighbor.
        self.assertGreater(certificate.upper_stored, -0.75)
        self.assertGreaterEqual(
            Fraction.from_float(certificate.upper_stored),
            Fraction(-3, 4),
        )

    def test_checker_return_and_outward_receipt_are_cross_bound(self):
        raw = np.longdouble(1.0) + np.finfo(np.longdouble).eps
        parent_sha = "1" * 64
        exact_sha = "2" * 64
        envelope_sha = "3" * 64
        binding_sha = "4" * 64
        base_receipt = {
            "schema": (
                "hz_lp_lagrangian_preformed_objective_"
                "split_blocks_longdouble_v1"
            ),
            "status": "verified_upper",
            "route": (
                "native_hz_preformed_objective_split_csr_"
                "no_generator_read_v1"
            ),
            "proof_authority": True,
            "verdict_authority": False,
            "pcoh_authorization": False,
            "generator_source_read_count": 0,
            "envelope_rehash_bytes": 0,
            "objective_formation_reused": True,
            "objective_binding_cross_checked": True,
            "parent_semantic_digest": parent_sha,
            "exact_objective_sha256": exact_sha,
            "objective_envelope_sha256": envelope_sha,
            "objective_binding_sha256": binding_sha,
            "uses_sparse_hstack": False,
            "uses_sparse_vstack": False,
            "assembled_sparse_nnz": 0,
            "upper": float(np.nextafter(1.0, np.inf)),
            "upper_float64_rounding": (
                "toward_positive_infinity_from_longdouble_v1"
            ),
            "illegal_sign_projected": 0,
            "nonfinite_dual_zeroed": 0,
        }
        stored, _ = bound_module._strict_checker_result(
            raw,
            dict(base_receipt),
            require_clean_dual=True,
            expected_parent_semantic_digest=parent_sha,
            expected_exact_objective_sha256=exact_sha,
            expected_objective_envelope_sha256=envelope_sha,
            expected_objective_binding_sha256=binding_sha,
        )
        self.assertEqual(stored, base_receipt["upper"])

        inward = dict(base_receipt)
        inward["upper"] = 1.0
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            bound_module._strict_checker_result(
                raw,
                inward,
                require_clean_dual=True,
                expected_parent_semantic_digest=parent_sha,
                expected_exact_objective_sha256=exact_sha,
                expected_objective_envelope_sha256=envelope_sha,
                expected_objective_binding_sha256=binding_sha,
            )

        for field, bad_value in (
            ("generator_source_read_count", 1),
            ("envelope_rehash_bytes", 8),
            ("objective_formation_reused", False),
            ("proof_authority", False),
            ("objective_envelope_sha256", "6" * 64),
        ):
            with self.subTest(field=field):
                tampered = dict(base_receipt)
                tampered[field] = bad_value
                with self.assertRaises(
                    OperatorPhaseConditionedObjectiveBoundError
                ):
                    bound_module._strict_checker_result(
                        raw,
                        tampered,
                        require_clean_dual=True,
                        expected_parent_semantic_digest=parent_sha,
                        expected_exact_objective_sha256=exact_sha,
                        expected_objective_envelope_sha256=envelope_sha,
                        expected_objective_binding_sha256=binding_sha,
                    )

        wrong_policy = dict(base_receipt)
        wrong_policy["upper_float64_rounding"] = "nearest_even"
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            bound_module._strict_checker_result(
                raw,
                wrong_policy,
                require_clean_dual=True,
                expected_parent_semantic_digest=parent_sha,
                expected_exact_objective_sha256=exact_sha,
                expected_objective_envelope_sha256=envelope_sha,
                expected_objective_binding_sha256=binding_sha,
            )

        stale_binding = dict(base_receipt)
        stale_binding["objective_binding_sha256"] = "5" * 64
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            bound_module._strict_checker_result(
                raw,
                stale_binding,
                require_clean_dual=True,
                expected_parent_semantic_digest=parent_sha,
                expected_exact_objective_sha256=exact_sha,
                expected_objective_envelope_sha256=envelope_sha,
                expected_objective_binding_sha256=binding_sha,
            )

    def test_exact_objective_binding_applies_threshold_once(self):
        certificate = self._bound(1, (-1,))
        rival = self.rivals[0]
        exact_center = -Fraction.from_float(rival.threshold)
        for weight, center in zip(rival.objective, self.build.hz.c):
            exact_center += Fraction.from_float(weight) * Fraction.from_float(
                float(center)
            )
        self.assertEqual(certificate.objective_binding.center, exact_center)
        self.assertEqual(certificate.receipt["threshold_application_count"], 1)
        self.assertEqual(
            certificate.receipt["objective_equality_substitution"], False
        )

    def test_row_subset_is_exact_and_omissions_are_explicit_relaxation(self):
        build = _with_unrelated_rows(self.build)
        selection = derive_operator_exact_relu_property_phase_literals(
            build, self.rivals
        )
        ids = tuple(mapping.stable_bcol_id for mapping in selection.mappings)
        certificate = build_operator_phase_conditioned_objective_bound(
            build,
            self.rivals,
            selection,
            focused_rival_id=10,
            stable_bit_ids=ids[:2],
            pattern=(-1, 1),
        )
        expected_rows = tuple(
            row
            for mapping in selection.mappings[:2]
            for row in (
                mapping.lower_upper_row,
                mapping.x_branch_upper_row,
                mapping.zero_branch_upper_row,
            )
        )
        self.assertEqual(certificate.local_upper_row_ids, expected_rows)
        self.assertEqual(len(set(expected_rows)), 6)
        self.assertEqual(certificate.omitted_equality_rows, 1)
        self.assertGreater(certificate.omitted_upper_rows, 0)
        self.assertEqual(certificate.receipt["retained_equality_rows"], 0)
        self.assertEqual(
            certificate.receipt["relaxation_relation"],
            "conditioned_parent_subset_of_local_three_rows_box",
        )

    def test_producer_and_replayer_never_call_sparse_stack(self):
        with patch.object(
            sp, "hstack", side_effect=AssertionError("hstack forbidden")
        ), patch.object(
            sp, "vstack", side_effect=AssertionError("vstack forbidden")
        ):
            certificate = self._bound(2, (-1, -1))
            self.assertTrue(
                verify_operator_phase_conditioned_objective_bound(
                    self.build,
                    self.rivals,
                    self.selection,
                    certificate,
                )
            )
        self.assertFalse(certificate.receipt["uses_sparse_hstack"])
        self.assertFalse(certificate.receipt["uses_sparse_vstack"])

    def test_candidate_failures_use_sound_zero_or_global_fallback(self):
        cases = (
            bound_module._CandidateSolve(status="nonoptimal_fallback"),
            bound_module._CandidateSolve(status="deadline_fallback"),
            bound_module._CandidateSolve(
                status="optimal",
                raw_upper_dual=(float("nan"),) * 3,
            ),
            bound_module._CandidateSolve(
                status="optimal",
                raw_upper_dual=(1.0,) * 3,
            ),
        )
        for candidate in cases:
            with self.subTest(candidate=candidate.status), patch.object(
                bound_module,
                "_solve_local_candidate",
                return_value=candidate,
            ):
                certificate = self._bound(1, (-1,))
                self.assertFalse(certificate.candidate_dual_accepted)
                self.assertIsNone(certificate.candidate_checked_upper)
                self.assertFalse(certificate.raw_upper_dual)
                self.assertIn(
                    certificate.selected_source,
                    {"zero_dual_fixed_pattern", "global_cube_baseline"},
                )
                self.assertTrue(
                    verify_operator_phase_conditioned_objective_bound(
                        self.build,
                        self.rivals,
                        self.selection,
                        certificate,
                    )
                )

    def test_noncanonical_inputs_and_stale_selection_fail_closed(self):
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            self._bound(1, (0,))
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            build_operator_phase_conditioned_objective_bound(
                self.build,
                self.rivals,
                self.selection,
                focused_rival_id=10,
                stable_bit_ids=tuple(reversed(self.ids[:2])),
                pattern=(-1, -1),
            )
        stale = replace(self.selection, selection_digest="0" * 64)
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            build_operator_phase_conditioned_objective_bound(
                self.build,
                self.rivals,
                stale,
                focused_rival_id=10,
                stable_bit_ids=self.ids[:1],
                pattern=(-1,),
            )

    def test_preformed_formation_and_binding_mismatch_never_fallback_legacy(self):
        with patch.object(
            bound_module,
            "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
            return_value=(
                None,
                {
                    "status": "invalid:test",
                    "proof_authority": False,
                    "verdict_authority": False,
                },
            ),
        ), patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper",
            side_effect=AssertionError("legacy fallback reached"),
        ):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "exact_objective_envelope_formation_failed",
            ):
                self._bound(1, (-1,))

        original_accessor = (
            bound_module._hz_read_exact_objective_binding_material_from_factor_envelope
        )

        def stale_binding(*args, **kwargs):
            center, continuous, binary, _ = original_accessor(*args, **kwargs)
            return center, continuous, binary, "f" * 64

        with patch.object(
            bound_module,
            "_hz_read_exact_objective_binding_material_from_factor_envelope",
            side_effect=stale_binding,
        ), patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            side_effect=AssertionError("checker ran before binding cross-check"),
        ):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "exact_objective_binding_cross_check_failed",
            ):
                self._bound(1, (-1,))

    def test_row_objective_pattern_dual_receipt_and_checksum_tamper(self):
        certificate = self._bound(2, (-1, -1))
        bad_receipt = dict(certificate.receipt)
        bad_receipt["threshold_application_count"] = 2
        mutations = (
            replace(
                certificate,
                local_upper_row_sha256=("0" * 64,)
                + certificate.local_upper_row_sha256[1:],
            ),
            replace(
                certificate,
                objective_binding=replace(
                    certificate.objective_binding,
                    center=certificate.objective_binding.center + 1,
                ),
            ),
            replace(certificate, pattern=(-1, 1)),
            replace(
                certificate,
                raw_upper_dual=tuple(
                    value - 1.0 for value in certificate.raw_upper_dual
                ),
            ),
            replace(certificate, receipt=MappingProxyType(bad_receipt)),
            replace(certificate, certificate_sha256="0" * 64),
            replace(certificate, checker_bundle_sha256="0" * 64),
            replace(certificate, verified_context_sha256="0" * 64),
            replace(certificate, objective_envelope_sha256="0" * 64),
        )
        for mutation in mutations:
            with self.subTest(field=mutation):
                self.assertFalse(
                    verify_operator_phase_conditioned_objective_bound(
                        self.build,
                        self.rivals,
                        self.selection,
                        mutation,
                    )
                )

    def test_live_row_change_invalidates_selection_and_certificate(self):
        certificate = self._bound(1, (-1,))
        ub = self.build.hz.ub.copy()
        ub[certificate.local_upper_row_ids[0]] = np.nextafter(
            ub[certificate.local_upper_row_ids[0]], np.inf
        )
        changed = replace(self.build, hz=_clone_hz(self.build.hz, ub=ub))
        self.assertFalse(
            verify_operator_exact_relu_property_phase_selection(
                changed, self.rivals, self.selection
            )
        )
        self.assertFalse(
            verify_operator_phase_conditioned_objective_bound(
                changed,
                self.rivals,
                self.selection,
                certificate,
            )
        )

    def test_complete_cover_barrier_is_only_external_authority(self):
        certificates = build_complete_operator_phase_conditioned_objective_bounds(
            self.build,
            self.rivals,
            self.selection,
            focused_rival_id=10,
            stable_bit_ids=self.ids[:3],
        )
        self.assertEqual(len(certificates), 8)
        self.assertTrue(all(not item.proof_authority for item in certificates))
        with self.assertRaises(OperatorPhaseConditionedObjectiveBoundError):
            replay_complete_operator_phase_conditioned_objective_bounds(
                self.build,
                self.rivals,
                self.selection,
                certificates[:-1],
            )
        replayed = replay_complete_operator_phase_conditioned_objective_bounds(
            self.build,
            self.rivals,
            self.selection,
            certificates,
        )
        self.assertTrue(replayed.proof_authority)
        self.assertEqual(len(replayed.pattern_bounds), 8)
        self.assertTrue(
            all(bound.upstream_proof_authority for bound in replayed.pattern_bounds)
        )
        self.assertEqual(
            replayed.baseline_upper_stored,
            max(item.upper_stored for item in certificates),
        )
        descriptor = build_phase_conditioned_objective_hull(
            stable_bit_ids=replayed.stable_bit_ids,
            pattern_bounds=replayed.pattern_bounds,
            objective_binding=replayed.objective_binding,
            parent_semantic_digest=replayed.parent_semantic_digest,
            baseline_upper_stored=replayed.baseline_upper_stored,
        )
        self.assertFalse(descriptor.proof_authority)

    def test_complete_build_and_replay_prepare_and_hash_context_once(self):
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            wraps=bound_module._prepare_verified_bundle_context,
        ) as prepare, patch.object(
            bound_module,
            "verify_operator_exact_relu_property_phase_selection",
            wraps=(
                bound_module.verify_operator_exact_relu_property_phase_selection
            ),
        ) as selection_verify, patch.object(
            bound_module,
            "sparse_hz_semantic_digest",
            wraps=bound_module.sparse_hz_semantic_digest,
        ) as semantic_hash, patch.object(
            bound_module,
            "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
            wraps=(
                bound_module._hz_form_exact_factor_objective_envelope_from_live_split_blocks
            ),
        ) as objective_formation, patch.object(
            bound_module,
            "_hz_read_exact_objective_binding_material_from_factor_envelope",
            wraps=(
                bound_module._hz_read_exact_objective_binding_material_from_factor_envelope
            ),
        ) as exact_accessor, patch.object(
            bound_module,
            "_build_exact_objective_binding",
            side_effect=AssertionError("duplicate objective expansion"),
        ), patch.object(
            bound_module,
            "_objective_position_maps",
            side_effect=AssertionError("per-pattern id map rebuilt"),
        ), patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper",
            side_effect=AssertionError("legacy generator checker reached"),
        ), patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=(
                bound_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
            ),
        ) as preformed_checker, patch.object(
            bound_module,
            "_row_payload",
            wraps=bound_module._row_payload,
        ) as row_hash:
            certificates = (
                build_complete_operator_phase_conditioned_objective_bounds(
                    self.build,
                    self.rivals,
                    self.selection,
                    focused_rival_id=10,
                    stable_bit_ids=self.ids,
                )
            )
        self.assertEqual(len(certificates), 16)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(selection_verify.call_count, 1)
        self.assertEqual(semantic_hash.call_count, 2)
        self.assertEqual(objective_formation.call_count, 1)
        self.assertEqual(exact_accessor.call_count, 1)
        self.assertEqual(row_hash.call_count, 12)
        expected_checker_calls = (
            1
            + len(certificates)
            + sum(
                int(item.candidate_dual_accepted)
                for item in certificates
            )
        )
        self.assertEqual(expected_checker_calls, 22)
        self.assertEqual(preformed_checker.call_count, expected_checker_calls)
        self.assertEqual(
            len({
                id(call.kwargs["objective_envelope"])
                for call in preformed_checker.call_args_list
            }),
            1,
        )
        self.assertEqual(
            len({
                call.kwargs["deadline"]
                for call in preformed_checker.call_args_list
            }),
            1,
        )
        self.assertEqual(
            len({item.verified_context_sha256 for item in certificates}), 1
        )
        self.assertEqual(
            len({item.objective_envelope_sha256 for item in certificates}), 1
        )
        for certificate in certificates:
            self.assertEqual(certificate.receipt["context_preparation_count"], 1)
            self.assertEqual(
                certificate.receipt["selection_verification_count"], 1
            )
            self.assertEqual(
                certificate.receipt["exact_objective_expansion_count"], 1
            )
            self.assertEqual(
                certificate.receipt["objective_envelope_formation_count"], 1
            )
            self.assertEqual(
                certificate.receipt[
                    "preformed_checker_generator_source_read_count"
                ],
                0,
            )
            self.assertFalse(
                certificate.receipt["objective_envelope_production_ready"]
            )
            self.assertEqual(
                certificate.receipt[
                    "global_checker_bundle_evaluation_count"
                ],
                1,
            )
            self.assertTrue(
                certificate.receipt["global_checker_reused_per_pattern"]
            )
            self.assertEqual(
                certificate.receipt["context_semantic_digest_count"], 1
            )
            self.assertEqual(
                certificate.receipt[
                    "terminal_parent_semantic_digest_count"
                ],
                1,
            )
            self.assertEqual(
                certificate.receipt["source_row_hashing_pass_count"], 1
            )

        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            wraps=bound_module._prepare_verified_bundle_context,
        ) as replay_prepare, patch.object(
            bound_module,
            "verify_operator_exact_relu_property_phase_selection",
            wraps=(
                bound_module.verify_operator_exact_relu_property_phase_selection
            ),
        ) as replay_selection, patch.object(
            bound_module,
            "sparse_hz_semantic_digest",
            wraps=bound_module.sparse_hz_semantic_digest,
        ) as replay_semantic, patch.object(
            bound_module,
            "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
            wraps=(
                bound_module._hz_form_exact_factor_objective_envelope_from_live_split_blocks
            ),
        ) as replay_formation, patch.object(
            bound_module,
            "_hz_read_exact_objective_binding_material_from_factor_envelope",
            wraps=(
                bound_module._hz_read_exact_objective_binding_material_from_factor_envelope
            ),
        ) as replay_accessor, patch.object(
            bound_module,
            "_build_exact_objective_binding",
            side_effect=AssertionError("duplicate replay objective expansion"),
        ), patch.object(
            bound_module,
            "_objective_position_maps",
            side_effect=AssertionError("replay rebuilt per-pattern id map"),
        ), patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper",
            side_effect=AssertionError("legacy replay checker reached"),
        ), patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=(
                bound_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
            ),
        ) as replay_checker, patch.object(
            bound_module,
            "_row_payload",
            wraps=bound_module._row_payload,
        ) as replay_rows:
            replayed = replay_complete_operator_phase_conditioned_objective_bounds(
                self.build,
                self.rivals,
                self.selection,
                certificates,
            )
        self.assertEqual(replay_prepare.call_count, 1)
        self.assertEqual(replay_selection.call_count, 1)
        self.assertEqual(replay_semantic.call_count, 2)
        self.assertEqual(replay_formation.call_count, 1)
        self.assertEqual(replay_accessor.call_count, 1)
        self.assertEqual(replay_rows.call_count, 12)
        self.assertEqual(replay_checker.call_count, expected_checker_calls)
        self.assertEqual(
            len({
                id(call.kwargs["objective_envelope"])
                for call in replay_checker.call_args_list
            }),
            1,
        )
        self.assertEqual(replayed.receipt["context_preparation_count"], 1)
        self.assertEqual(replayed.receipt["selection_verification_count"], 1)
        self.assertEqual(
            replayed.receipt["exact_objective_expansion_count"], 1
        )
        self.assertEqual(
            replayed.receipt["objective_envelope_formation_count"], 1
        )
        self.assertEqual(
            replayed.receipt[
                "preformed_checker_generator_source_read_count"
            ],
            0,
        )
        self.assertEqual(replayed.receipt["context_semantic_digest_count"], 1)
        self.assertEqual(
            replayed.receipt["terminal_parent_semantic_digest_count"], 1
        )
        self.assertEqual(replayed.receipt["source_row_hashing_pass_count"], 1)

    def test_shared_context_is_readonly_and_pattern_materialization_is_local(self):
        deadline = bound_module.time.monotonic() + 10.0
        shared = bound_module._prepare_verified_bundle_context(
            self.build,
            self.rivals,
            self.selection,
            focused_rival_id=10,
            stable_bit_ids=self.ids,
            deadline=deadline,
        )
        for matrix in (
            shared.local_Auc,
            shared.local_Aub,
            shared.empty_Ac,
            shared.empty_Ab,
        ):
            self.assertFalse(matrix.data.flags.writeable)
            self.assertFalse(matrix.indices.flags.writeable)
            self.assertFalse(matrix.indptr.flags.writeable)
        for array in (
            shared.local_ub,
            shared.empty_b,
            shared.continuous_lb,
            shared.continuous_ub,
            shared.binary_cube_lb,
            shared.binary_cube_ub,
            shared.candidate_continuous_positions,
            shared.candidate_continuous_q,
            shared.candidate_binary_positions,
            shared.candidate_binary_q,
            shared.objective_envelope.q_continuous_hat,
            shared.objective_envelope.q_continuous_error,
            shared.objective_envelope.q_binary_hat,
            shared.objective_envelope.q_binary_error,
        ):
            self.assertFalse(array.flags.writeable)
        left = bound_module._materialize_pattern_context(
            shared, (-1, -1, -1, -1), deadline=deadline
        )
        right = bound_module._materialize_pattern_context(
            shared, (1, 1, 1, 1), deadline=deadline
        )
        self.assertIs(left.local_Auc, right.local_Auc)
        self.assertIs(left.local_Aub, right.local_Aub)
        self.assertIs(left.objective_envelope, right.objective_envelope)
        self.assertIs(
            left.candidate_continuous_positions,
            right.candidate_continuous_positions,
        )
        self.assertIsNot(left.binary_lb, right.binary_lb)
        self.assertEqual(
            left.verified_context_sha256, right.verified_context_sha256
        )
        for mapping in self.selection.mappings:
            self.assertEqual(left.binary_lb[mapping.binary_position], -1.0)
            self.assertEqual(right.binary_lb[mapping.binary_position], 1.0)

    def test_bundle_absolute_deadlines_cannot_leak_partial_handles(self):
        first = self._bound(1, (-1,))
        second = self._bound(1, (1,))
        shared = bound_module._prepare_verified_bundle_context(
            self.build,
            self.rivals,
            self.selection,
            focused_rival_id=10,
            stable_bit_ids=self.ids[:1],
            deadline=bound_module.time.monotonic() + 10.0,
        )
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            return_value=shared,
        ), patch.object(
            bound_module,
            "_build_bound_from_verified_context",
            return_value=first,
        ), patch.object(
            bound_module.time,
            "monotonic",
            side_effect=(0.0, 0.0, 11.0),
        ):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "no_partial_output",
            ):
                build_complete_operator_phase_conditioned_objective_bounds(
                    self.build,
                    self.rivals,
                    self.selection,
                    focused_rival_id=10,
                    stable_bit_ids=self.ids[:1],
                    certificate_timeout_seconds=10.0,
                )

        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            return_value=shared,
        ), patch.object(
            bound_module,
            "_verify_bound_from_verified_context",
            return_value=True,
        ), patch.object(
            bound_module,
            "bind_external_pattern_upper_bound",
            side_effect=AssertionError("partial handle leaked"),
        ), patch.object(
            bound_module.time,
            "monotonic",
            side_effect=(0.0, 0.0, 11.0),
        ):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "no_partial_output",
            ):
                replay_complete_operator_phase_conditioned_objective_bounds(
                    self.build,
                    self.rivals,
                    self.selection,
                    (first, second),
                    certificate_timeout_seconds=10.0,
                )

    def test_private_bundle_helpers_never_reset_outer_absolute_deadline(self):
        build_deadline = bound_module.time.monotonic() + 10.0
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            wraps=bound_module._prepare_verified_bundle_context,
        ) as prepare, patch.object(
            bound_module,
            "_build_bound_from_verified_context",
            wraps=bound_module._build_bound_from_verified_context,
        ) as build_pattern, patch.object(
            bound_module,
            "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
            wraps=(
                bound_module._hz_form_exact_factor_objective_envelope_from_live_split_blocks
            ),
        ) as formation, patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=(
                bound_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
            ),
        ) as checker:
            certificates = (
                bound_module._build_complete_operator_phase_conditioned_objective_bounds_until(
                    self.build,
                    self.rivals,
                    self.selection,
                    focused_rival_id=10,
                    stable_bit_ids=self.ids[:1],
                    deadline=build_deadline,
                )
            )
        self.assertEqual(prepare.call_args.kwargs["deadline"], build_deadline)
        self.assertEqual(formation.call_args.kwargs["deadline"], build_deadline)
        self.assertTrue(checker.call_args_list)
        self.assertTrue(all(
            call.kwargs["deadline"] == build_deadline
            for call in checker.call_args_list
        ))
        self.assertTrue(build_pattern.call_args_list)
        self.assertTrue(
            all(
                call.kwargs["deadline"] == build_deadline
                for call in build_pattern.call_args_list
            )
        )

        replay_deadline = bound_module.time.monotonic() + 10.0
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            wraps=bound_module._prepare_verified_bundle_context,
        ) as replay_prepare, patch.object(
            bound_module,
            "_verify_bound_from_verified_context",
            wraps=bound_module._verify_bound_from_verified_context,
        ) as replay_pattern, patch.object(
            bound_module,
            "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
            wraps=(
                bound_module._hz_form_exact_factor_objective_envelope_from_live_split_blocks
            ),
        ) as replay_formation, patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=(
                bound_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
            ),
        ) as replay_checker:
            bound_module._replay_complete_operator_phase_conditioned_objective_bounds_until(
                self.build,
                self.rivals,
                self.selection,
                certificates,
                deadline=replay_deadline,
            )
        self.assertEqual(
            replay_prepare.call_args.kwargs["deadline"], replay_deadline
        )
        self.assertEqual(
            replay_formation.call_args.kwargs["deadline"], replay_deadline
        )
        self.assertTrue(all(
            call.kwargs["deadline"] == replay_deadline
            for call in replay_checker.call_args_list
        ))
        self.assertTrue(replay_pattern.call_args_list)
        self.assertTrue(
            all(
                call.kwargs["deadline"] == replay_deadline
                for call in replay_pattern.call_args_list
            )
        )

    def test_terminal_parent_seal_rejects_mutation_after_last_checker(self):
        def fresh():
            build = replace(self.build, hz=_clone_hz(self.build.hz))
            selection = derive_operator_exact_relu_property_phase_literals(
                build, self.rivals
            )
            ids = tuple(
                mapping.stable_bcol_id for mapping in selection.mappings
            )
            return build, selection, ids

        def mutating_checker(build):
            original = bound_module._run_checker
            state = {"calls": 0, "mutated_at": None}

            def run(context, **kwargs):
                result = original(context, **kwargs)
                state["calls"] += 1
                if (
                    state["mutated_at"] is None
                    and context.pattern == (1, 1)
                    and kwargs["use_local_rows"] is False
                    and kwargs["fix_pattern"] is False
                ):
                    build.hz.c[0] = np.nextafter(build.hz.c[0], np.inf)
                    state["mutated_at"] = state["calls"]
                return result

            return run, state

        build, selection, ids = fresh()
        mutate, state = mutating_checker(build)
        with patch.object(bound_module, "_run_checker", side_effect=mutate):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "terminal_parent_semantic_digest_mismatch",
            ):
                build_complete_operator_phase_conditioned_objective_bounds(
                    build,
                    self.rivals,
                    selection,
                    focused_rival_id=10,
                    stable_bit_ids=ids[:2],
                )
        self.assertIsNotNone(state["mutated_at"])
        self.assertGreater(state["mutated_at"], 1)

        build, selection, ids = fresh()
        certificates = build_complete_operator_phase_conditioned_objective_bounds(
            build,
            self.rivals,
            selection,
            focused_rival_id=10,
            stable_bit_ids=ids[:2],
        )
        mutate, state = mutating_checker(build)
        with patch.object(
            bound_module, "_run_checker", side_effect=mutate
        ), patch.object(
            bound_module,
            "bind_external_pattern_upper_bound",
            side_effect=AssertionError("partial external handle leaked"),
        ):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "terminal_parent_semantic_digest_mismatch",
            ):
                replay_complete_operator_phase_conditioned_objective_bounds(
                    build,
                    self.rivals,
                    selection,
                    certificates,
                )
        self.assertIsNotNone(state["mutated_at"])
        self.assertGreater(state["mutated_at"], 1)

    def test_scheduled_k1_to_k4_permutations_return_canonical_complete_cover(self):
        for k in range(1, 5):
            canonical = tuple(
                tuple(pattern)
                for pattern in itertools.product((-1, 1), repeat=k)
            )
            schedule = tuple(reversed(canonical))
            with self.subTest(k=k):
                result = self._scheduled(k, schedule)
                self.assertEqual(result.evaluation_schedule, schedule)
                self.assertEqual(result.canonical_patterns, canonical)
                self.assertEqual(
                    tuple(item.pattern for item in result.certificates),
                    canonical,
                )
                self.assertTrue(
                    verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                        result
                    )
                )
                telemetry = result.telemetry
                self.assertEqual(telemetry["context_formations"], 1)
                self.assertEqual(telemetry["exact_objective_expansions"], 1)
                self.assertEqual(telemetry["source_row_hash_passes"], 1)
                self.assertEqual(telemetry["patterns_completed"], 2**k)
                self.assertEqual(telemetry["local_upper_rows"], 3 * k)
                self.assertEqual(
                    telemetry["completed_patterns_in_execution_order"],
                    schedule,
                )
                self.assertFalse(result.full_parent_lp_called)
                self.assertFalse(result.proof_authority)
                self.assertFalse(result.verdict_authority)

    def test_scheduled_k3_worst_children_execute_first_without_stack(self):
        canonical = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=3)
        )
        worst_children = ((1, 1, -1), (1, 1, 1))
        schedule = worst_children + tuple(
            pattern for pattern in canonical if pattern not in worst_children
        )
        with patch.object(
            bound_module,
            "_build_bound_from_verified_context",
            wraps=bound_module._build_bound_from_verified_context,
        ) as build_pattern, patch.object(
            sp, "hstack", side_effect=AssertionError("hstack forbidden")
        ), patch.object(
            sp, "vstack", side_effect=AssertionError("vstack forbidden")
        ):
            result = self._scheduled(3, schedule)
        self.assertEqual(
            tuple(call.kwargs["pattern"] for call in build_pattern.call_args_list),
            schedule,
        )
        self.assertEqual(
            tuple(item.pattern for item in result.certificates), canonical
        )
        self.assertEqual(result.telemetry["patterns_completed"], 8)
        self.assertEqual(result.telemetry["local_upper_rows"], 9)
        self.assertEqual(result.receipt["external_pattern_bounds_bound"], 0)
        self.assertEqual(
            result.receipt["public_sha_role"],
            "structural_self_consistency_only",
        )
        self.assertFalse(result.receipt["provenance_authority"])
        self.assertFalse(result.receipt["authenticity_authority"])
        self.assertTrue(
            result.receipt["future_live_owner_anchor_required"]
        )
        self.assertTrue(
            result.receipt["no_partial_output_on_failure_or_policy_stop"]
        )

    def test_scheduled_k3_telemetry_counts_actual_call_sites(self):
        schedule = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=3)
        )
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            wraps=bound_module._prepare_verified_bundle_context,
        ) as prepare, patch.object(
            bound_module,
            "_hz_form_exact_factor_objective_envelope_from_live_split_blocks",
            wraps=(
                bound_module._hz_form_exact_factor_objective_envelope_from_live_split_blocks
            ),
        ) as exact_formation, patch.object(
            bound_module,
            "_row_payload",
            wraps=bound_module._row_payload,
        ) as row_hash, patch.object(
            bound_module.spo,
            "linprog",
            wraps=bound_module.spo.linprog,
        ) as linprog, patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=(
                bound_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
            ),
        ) as checker:
            result = self._scheduled(3, schedule)
        telemetry = result.telemetry
        accepted = sum(
            int(item.candidate_dual_accepted)
            for item in result.certificates
        )
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(exact_formation.call_count, 1)
        self.assertEqual(row_hash.call_count, 9)
        self.assertEqual(telemetry["candidate_proposal_invocations"], 8)
        self.assertEqual(telemetry["linprog_actual_calls"], linprog.call_count)
        self.assertEqual(
            telemetry["linprog_completed_calls"], linprog.call_count
        )
        self.assertGreaterEqual(
            telemetry["linprog_attempted"], linprog.call_count
        )
        self.assertEqual(telemetry["candidate_dual_accepted"], accepted)
        self.assertEqual(telemetry["candidate_checker_evaluations"], accepted)
        self.assertEqual(telemetry["zero_checker_evaluations"], 8)
        self.assertEqual(telemetry["global_checker_evaluations"], 1)
        self.assertEqual(telemetry["global_checker_cache_hits"], 8)
        self.assertEqual(
            telemetry["split_checker_evaluations"], 1 + 8 + accepted
        )
        self.assertEqual(checker.call_count, 1 + 8 + accepted)
        self.assertEqual(telemetry["terminal_parent_seal_attempts"], 1)
        self.assertEqual(telemetry["terminal_parent_seal_completions"], 1)
        self.assertEqual(
            sum(telemetry["candidate_status_counts"].values()), 8
        )
        traces = telemetry["candidate_call_trace_in_execution_order"]
        self.assertEqual(len(traces), 8)
        self.assertTrue(
            all(type(trace) is MappingProxyType for trace in traces)
        )
        self.assertEqual(
            tuple(trace["pattern"] for trace in traces), schedule
        )
        self.assertEqual(
            sum(int(trace["linprog_eligible"]) for trace in traces),
            telemetry["linprog_attempted"],
        )
        self.assertEqual(
            sum(int(trace["linprog_called"]) for trace in traces),
            telemetry["linprog_actual_calls"],
        )
        self.assertEqual(
            sum(int(trace["linprog_completed"]) for trace in traces),
            telemetry["linprog_completed_calls"],
        )
        self.assertEqual(
            telemetry["observed_upper_exact_in_execution_order"],
            tuple(
                Fraction.from_float(item.upper_stored)
                for item in result.certificates
            ),
        )

    def test_scheduled_k3_all_accepted_build_plus_replay_checker_formula_34(self):
        schedule = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=3)
        )

        def all_accepted(context, **_kwargs):
            return bound_module._CandidateSolve(
                status="optimal",
                raw_upper_dual=(0.0,) * len(context.local_upper_row_ids),
            )

        with patch.object(
            bound_module,
            "_solve_local_candidate",
            side_effect=all_accepted,
        ):
            result = self._scheduled(3, schedule)
        telemetry = result.telemetry
        self.assertEqual(telemetry["candidate_dual_accepted"], 8)
        self.assertEqual(telemetry["candidate_proposal_invocations"], 8)
        self.assertEqual(telemetry["linprog_attempted"], 0)
        self.assertEqual(telemetry["linprog_actual_calls"], 0)
        self.assertEqual(telemetry["linprog_completed_calls"], 0)
        self.assertEqual(telemetry["split_checker_evaluations"], 17)
        with patch.object(
            bound_module,
            "_hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope",
            wraps=(
                bound_module._hz_independent_split_block_lp_lagrangian_upper_from_factor_envelope
            ),
        ) as replay_checker:
            replay_complete_operator_phase_conditioned_objective_bounds(
                self.build,
                self.rivals,
                self.selection,
                result.certificates,
            )
        self.assertEqual(replay_checker.call_count, 17)
        self.assertEqual(
            telemetry["split_checker_evaluations"]
            + replay_checker.call_count,
            34,
        )

    def test_scheduled_stop_is_sealed_non_authoritative_and_exposes_no_partial(self):
        canonical = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=3)
        )
        schedule = ((1, 1, -1), (1, 1, 1)) + canonical[:-2]
        policy = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=Fraction(-1_000_000),
            threshold_pattern_indices=(0,),
        )
        with patch.object(
            bound_module,
            "bind_external_pattern_upper_bound",
            side_effect=AssertionError("partial external binder called"),
        ) as binder:
            with self.assertRaises(
                OperatorPhaseConditionedScheduledStop
            ) as caught:
                self._scheduled(3, schedule, stop_policy=policy)
        record = caught.exception.record
        self.assertEqual(binder.call_count, 0)
        self.assertTrue(
            verify_operator_phase_conditioned_scheduled_stop_record(record)
        )
        self.assertEqual(record.completed_internal_pattern_count, 1)
        self.assertEqual(record.triggering_pattern, schedule[0])
        self.assertFalse(hasattr(record, "certificates"))
        self.assertFalse(record.partial_certificates_returned)
        self.assertEqual(record.external_pattern_bounds_bound, 0)
        self.assertFalse(record.proof_authority)
        self.assertFalse(record.verdict_authority)
        self.assertTrue(record.structural_self_consistency_only)
        self.assertFalse(record.provenance_authority)
        self.assertFalse(record.authenticity_authority)
        self.assertTrue(record.future_live_owner_anchor_required)
        self.assertEqual(record.telemetry["patterns_completed"], 1)
        self.assertEqual(record.telemetry["terminal_parent_seal_attempts"], 1)
        self.assertEqual(
            record.telemetry["terminal_parent_seal_completions"], 1
        )
        tampered = replace(record, reason="forged")
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(tampered)
        )
        policy_tampered = replace(
            record,
            stop_policy=OperatorPhaseConditionedScheduledStopPolicy(
                stop_after_pattern_indices=(0,)
            ),
        )
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(
                policy_tampered
            )
        )
        forged_telemetry = dict(record.telemetry)
        forged_telemetry["linprog_actual_calls"] = 999
        forged_telemetry.pop("telemetry_sha256")
        forged_telemetry["telemetry_sha256"] = bound_module._canonical_sha256(
            forged_telemetry
        )
        forged_record = replace(
            record,
            telemetry=MappingProxyType(forged_telemetry),
            record_sha256="",
        )
        forged_record = replace(
            forged_record,
            record_sha256=bound_module._canonical_sha256(
                bound_module._scheduled_stop_payload(
                    forged_record, include_digest=False
                )
            ),
        )
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(
                forged_record
            )
        )
        with self.assertRaises(OperatorPhaseConditionedScheduledStop) as plain:
            self._scheduled(
                1,
                ((1,), (-1,)),
                stop_policy=OperatorPhaseConditionedScheduledStopPolicy(
                    stop_after_pattern_indices=(0,)
                ),
            )
        self.assertEqual(
            plain.exception.record.reason,
            "stop_after_pattern_index_reached",
        )
        self.assertIsNone(plain.exception.record.strict_upper_threshold)
        self.assertEqual(
            plain.exception.record.observed_upper_exact,
            plain.exception.record.telemetry[
                "observed_upper_exact_in_execution_order"
            ][0],
        )

    def test_scheduled_threshold_exact_binary64_neighbors(self):
        schedule = ((-1,), (1,))
        baseline = self._scheduled(1, schedule)
        first_upper = baseline.certificates[0].upper_stored
        equal_policy = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=Fraction.from_float(first_upper),
            threshold_pattern_indices=(0,),
        )
        equal_result = self._scheduled(
            1, schedule, stop_policy=equal_policy
        )
        self.assertEqual(equal_result.stop_policy, equal_policy)
        self.assertEqual(
            equal_result.receipt["stop_policy"][
                "strict_upper_threshold"
            ],
            equal_policy.strict_upper_threshold,
        )
        self.assertTrue(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                equal_result
            )
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                replace(
                    equal_result,
                    stop_policy=OperatorPhaseConditionedScheduledStopPolicy(),
                )
            )
        )
        above_policy = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=Fraction.from_float(
                float(np.nextafter(first_upper, np.inf))
            ),
            threshold_pattern_indices=(0,),
        )
        self.assertTrue(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                self._scheduled(1, schedule, stop_policy=above_policy)
            )
        )
        below_policy = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=Fraction.from_float(
                float(np.nextafter(first_upper, -np.inf))
            ),
            threshold_pattern_indices=(0,),
        )
        with self.assertRaises(OperatorPhaseConditionedScheduledStop) as caught:
            self._scheduled(1, schedule, stop_policy=below_policy)
        self.assertEqual(
            caught.exception.record.reason,
            "strict_upper_threshold_exceeded",
        )

    def test_scheduled_policy_stop_seal_crossing_deadline_exposes_no_record(self):
        schedule = ((-1,), (1,))
        policy = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=Fraction(-1_000_000),
            threshold_pattern_indices=(0,),
        )
        original_monotonic = time.monotonic
        original_seal = bound_module._terminal_parent_seal
        state = {"crossed": False, "deadline": None}

        def fake_monotonic():
            if state["crossed"]:
                return state["deadline"] + 1.0
            return original_monotonic()

        def delayed_seal(shared, *, deadline, telemetry=None):
            result = original_seal(
                shared, deadline=deadline, telemetry=telemetry
            )
            state["deadline"] = deadline
            state["crossed"] = True
            return result

        with patch.object(
            bound_module.time, "monotonic", side_effect=fake_monotonic
        ), patch.object(
            bound_module,
            "_terminal_parent_seal",
            side_effect=delayed_seal,
        ), patch.object(
            bound_module,
            "_make_scheduled_stop_record",
            wraps=bound_module._make_scheduled_stop_record,
        ) as make_record, patch.object(
            bound_module,
            "bind_external_pattern_upper_bound",
            side_effect=AssertionError("expired stop bound externally"),
        ) as binder:
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "scheduled_policy_stop_return",
            ) as caught:
                self._scheduled(1, schedule, stop_policy=policy)
        self.assertNotIsInstance(
            caught.exception, OperatorPhaseConditionedScheduledStop
        )
        self.assertEqual(make_record.call_count, 0)
        self.assertEqual(binder.call_count, 0)

    def test_scheduled_complete_verifier_exactly_replays_stop_policy(self):
        schedule = ((-1,), (1,))
        result = self._scheduled(1, schedule)
        unconditional = OperatorPhaseConditionedScheduledStopPolicy(
            stop_after_pattern_indices=(1,)
        )
        forged_unconditional = self._resealed_scheduled_result(
            result, stop_policy=unconditional
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                forged_unconditional
            )
        )
        first_observed = result.telemetry[
            "observed_upper_exact_in_execution_order"
        ][0]
        threshold = OperatorPhaseConditionedScheduledStopPolicy(
            strict_upper_threshold=first_observed - 1,
            threshold_pattern_indices=(0,),
        )
        forged_threshold = self._resealed_scheduled_result(
            result, stop_policy=threshold
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                forged_threshold
            )
        )

    def test_scheduled_policy_disjoint_and_stop_first_trigger_replayed(self):
        schedule = ((-1,), (1,))
        overlap = OperatorPhaseConditionedScheduledStopPolicy(
            stop_after_pattern_indices=(0,),
            strict_upper_threshold=Fraction(0),
            threshold_pattern_indices=(0,),
        )
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            side_effect=AssertionError("overlap reached live context"),
        ), self.assertRaisesRegex(
            OperatorPhaseConditionedObjectiveBoundError, "overlap"
        ):
            self._scheduled(1, schedule, stop_policy=overlap)

        with self.assertRaises(OperatorPhaseConditionedScheduledStop) as caught:
            self._scheduled(
                1,
                schedule,
                stop_policy=OperatorPhaseConditionedScheduledStopPolicy(
                    stop_after_pattern_indices=(1,)
                ),
            )
        record = caught.exception.record
        self.assertTrue(
            verify_operator_phase_conditioned_scheduled_stop_record(record)
        )
        earlier_unconditional = self._resealed_stop_record(
            record,
            stop_policy=OperatorPhaseConditionedScheduledStopPolicy(
                stop_after_pattern_indices=(0, 1)
            ),
        )
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(
                earlier_unconditional
            )
        )
        observed = record.telemetry[
            "observed_upper_exact_in_execution_order"
        ]
        earlier_threshold = min(observed) - 1
        forged_threshold = self._resealed_stop_record(
            record,
            reason="strict_upper_threshold_exceeded",
            stop_policy=OperatorPhaseConditionedScheduledStopPolicy(
                strict_upper_threshold=earlier_threshold,
                threshold_pattern_indices=(0, 1),
            ),
            strict_upper_threshold=earlier_threshold,
        )
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(
                forged_threshold
            )
        )
        observed_mismatch = self._resealed_stop_record(
            record,
            observed_upper_exact=record.observed_upper_exact + 1,
        )
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(
                observed_mismatch
            )
        )
        extra_stop_telemetry = self._rehashed_telemetry(
            record.telemetry, coherent_extra_field=True
        )
        self.assertFalse(
            verify_operator_phase_conditioned_scheduled_stop_record(
                self._resealed_stop_record(
                    record, telemetry=extra_stop_telemetry
                )
            )
        )

    def test_scheduled_telemetry_trace_and_exact_keyset_reject_rehashes(self):
        schedule = ((-1,), (1,))
        result = self._scheduled(1, schedule)

        extra = self._rehashed_telemetry(
            result.telemetry, coherent_extra_field=True
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                self._resealed_scheduled_result(result, telemetry=extra)
            )
        )

        impossible_traces = [
            dict(trace)
            for trace in result.telemetry[
                "candidate_call_trace_in_execution_order"
            ]
        ]
        impossible_traces[0]["linprog_eligible"] = False
        impossible_traces[0]["linprog_called"] = True
        impossible_traces[0]["linprog_completed"] = True
        impossible = self._rehashed_telemetry(
            result.telemetry,
            candidate_call_trace_in_execution_order=tuple(
                impossible_traces
            ),
            linprog_attempted=sum(
                int(trace["linprog_eligible"])
                for trace in impossible_traces
            ),
            linprog_actual_calls=sum(
                int(trace["linprog_called"])
                for trace in impossible_traces
            ),
            linprog_completed_calls=sum(
                int(trace["linprog_completed"])
                for trace in impossible_traces
            ),
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                self._resealed_scheduled_result(result, telemetry=impossible)
            )
        )

        status_traces = [
            dict(trace)
            for trace in result.telemetry[
                "candidate_call_trace_in_execution_order"
            ]
        ]
        status_traces[0].update(
            {
                "linprog_eligible": False,
                "linprog_called": False,
                "linprog_completed": False,
                "normalized_candidate_status": "deadline_fallback",
                "candidate_dual_accepted": False,
            }
        )
        statuses = tuple(
            trace["normalized_candidate_status"]
            for trace in status_traces
        )
        accepted = sum(
            int(trace["candidate_dual_accepted"])
            for trace in status_traces
        )
        coherent_status_mismatch = self._rehashed_telemetry(
            result.telemetry,
            candidate_call_trace_in_execution_order=tuple(status_traces),
            candidate_statuses_in_execution_order=statuses,
            candidate_status_counts={
                status: statuses.count(status)
                for status in sorted(set(statuses))
            },
            candidate_dual_accepted=accepted,
            candidate_checker_evaluations=accepted,
            split_checker_evaluations=1 + len(schedule) + accepted,
            linprog_attempted=sum(
                int(trace["linprog_eligible"])
                for trace in status_traces
            ),
            linprog_actual_calls=sum(
                int(trace["linprog_called"])
                for trace in status_traces
            ),
            linprog_completed_calls=sum(
                int(trace["linprog_completed"])
                for trace in status_traces
            ),
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                self._resealed_scheduled_result(
                    result, telemetry=coherent_status_mismatch
                )
            )
        )

        nested_mutable = dict(result.telemetry)
        nested_mutable["candidate_status_counts"] = dict(
            result.telemetry["candidate_status_counts"]
        )
        nested_mutable.pop("telemetry_sha256")
        nested_mutable["telemetry_sha256"] = (
            bound_module._canonical_sha256(nested_mutable)
        )
        nested_mutable = MappingProxyType(nested_mutable)
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                self._resealed_scheduled_result(
                    result, telemetry=nested_mutable
                )
            )
        )

        impossible_time = self._rehashed_telemetry(
            result.telemetry,
            context_seconds=result.telemetry["total_seconds"] + 1.0,
        )
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                self._resealed_scheduled_result(
                    result, telemetry=impossible_time
                )
            )
        )

    def test_scheduled_deadline_schedule_and_receipt_tamper_fail_closed(self):
        canonical = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=2)
        )
        with patch.object(
            bound_module,
            "_prepare_verified_bundle_context",
            side_effect=AssertionError("context built for bad schedule"),
        ):
            with self.assertRaisesRegex(
                OperatorPhaseConditionedObjectiveBoundError,
                "not_complete_permutation",
            ):
                self._scheduled(
                    2,
                    (canonical[0], canonical[0], canonical[2], canonical[3]),
                )
        with self.assertRaisesRegex(
            OperatorPhaseConditionedObjectiveBoundError,
            "deadline_expired_no_partial_output",
        ):
            build_scheduled_complete_operator_phase_conditioned_objective_bounds(
                self.build,
                self.rivals,
                self.selection,
                focused_rival_id=10,
                stable_bit_ids=self.ids[:2],
                evaluation_schedule=canonical,
                deadline=time.monotonic() - 1.0,
            )

        result = self._scheduled(2, tuple(reversed(canonical)))
        bad_receipt = dict(result.receipt)
        bad_receipt["full_parent_lp_called"] = True
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                replace(result, receipt=MappingProxyType(bad_receipt))
            )
        )
        bad_telemetry = dict(result.telemetry)
        bad_telemetry["context_formations"] = 2
        self.assertFalse(
            verify_scheduled_complete_operator_phase_conditioned_objective_bounds_structure(
                replace(result, telemetry=MappingProxyType(bad_telemetry))
            )
        )

    def test_scheduled_canonical_path_is_bit_compatible_with_legacy_complete(self):
        canonical = tuple(
            tuple(pattern)
            for pattern in itertools.product((-1, 1), repeat=2)
        )
        legacy = build_complete_operator_phase_conditioned_objective_bounds(
            self.build,
            self.rivals,
            self.selection,
            focused_rival_id=10,
            stable_bit_ids=self.ids[:2],
        )
        scheduled = self._scheduled(2, canonical)
        self.assertEqual(
            tuple(item.certificate_sha256 for item in scheduled.certificates),
            tuple(item.certificate_sha256 for item in legacy),
        )
        replayed = replay_complete_operator_phase_conditioned_objective_bounds(
            self.build,
            self.rivals,
            self.selection,
            scheduled.certificates,
        )
        self.assertTrue(replayed.proof_authority)
        self.assertEqual(len(replayed.pattern_bounds), 4)

    def test_blast_radius_parent_and_selection_are_unchanged(self):
        before = sparse_hz_semantic_digest(self.build.hz)
        certificate = self._bound(4, (-1, -1, -1, -1))
        after = sparse_hz_semantic_digest(self.build.hz)
        self.assertEqual(before, after)
        self.assertTrue(
            verify_operator_exact_relu_property_phase_selection(
                self.build, self.rivals, self.selection
            )
        )
        self.assertTrue(
            verify_operator_phase_conditioned_objective_bound(
                self.build,
                self.rivals,
                self.selection,
                certificate,
            )
        )


if __name__ == "__main__":
    unittest.main()
