#!/usr/bin/env python3
"""Controlled and adversarial toys for the default-off Operator E2 adapter."""

from __future__ import annotations

from dataclasses import replace
from contextlib import ExitStack
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    operator_localized_phase_edge_candidate as edge_module,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_localized_phase_edge_candidate import (
    OperatorLocalizedPhaseEdgeError,
    run_operator_localized_phase_edge_candidate,
)
from act.back_end.hybridz_tf.operator_hz import build_operator_hz
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_cliques import (
    _corner_build,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _DTYPE,
    _dense,
    _layer,
    _rivals,
)


def _selection(build):
    rivals = _rivals()
    return rivals, derive_operator_exact_relu_property_phase_literals(
        build, rivals
    )


def _sealed_corner_build(*, bias: float):
    return _corner_build(
        bias=bias,
        issue_constructive_nonempty_seal=True,
    )


def _effect_count_build(count: int):
    """Keep exactly ``count`` property-visible exact-ReLU mappings."""

    if count not in {0, 1}:
        raise AssertionError("effect count toy supports only zero or one")
    lower = torch.tensor([[-1.0, -1.0]], dtype=_DTYPE)
    upper = torch.tensor([[1.0, 1.0]], dtype=_DTYPE)
    baseline = tuple(0.0 if index < count else 0.25 for index in range(4))
    visible = tuple(1.0 if index < count else 0.25 for index in range(4))
    half_visible = tuple(0.5 if index < count else 0.25 for index in range(4))
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
            (baseline, visible, half_visible),
            (0.75, 0.0, 0.0),
        ),
        _layer(5, "ASSERT", width=3),
    ]
    predecessors = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
    successors = {layer.id: [] for layer in layers}
    for child, parents in predecessors.items():
        for parent in parents:
            successors[parent].append(child)
    network = SimpleNamespace(
        layers=layers,
        preds=predecessors,
        succs=successors,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {}
    for layer in layers:
        width = len(layer.out_vars)
        if layer.kind in {"INPUT", "INPUT_SPEC"}:
            fact_lower = lower.clone()
            fact_upper = upper.clone()
        else:
            fact_lower = torch.full((1, width), -1.0e30, dtype=_DTYPE)
            fact_upper = torch.full((1, width), 1.0e30, dtype=_DTYPE)
        facts[layer.id] = Fact(Bounds(fact_lower, fact_upper), ConSet())
    build = build_operator_hz(
        network,
        facts,
        facts,
        exact_budget=4,
        materialize_add=True,
        issue_constructive_nonempty_seal=True,
    )
    rivals = _rivals()
    selection = derive_operator_exact_relu_property_phase_literals(
        build, rivals
    )
    return build, rivals, selection


class OperatorLocalizedPhaseEdgeCandidateTests(unittest.TestCase):
    def test_default_off_is_static_and_reads_no_caller_input(self) -> None:
        class Poison:
            def __getattribute__(self, _name):
                raise AssertionError("disabled path read poison object")

            def __iter__(self):
                raise AssertionError("disabled path iterated poison object")

        poison = Poison()
        forbidden = (
            (edge_module, "_normalize_caps"),
            (edge_module, "sparse_hz_semantic_digest"),
            (edge_module, "_row_tag_digest"),
            (edge_module, "_build_binding"),
            (edge_module.clique_module, "_normalize_deadline"),
            (edge_module.clique_module, "_snapshot_operator_build"),
            (
                edge_module.localized_oracle,
                "run_localized_phase_conflict_oracle_candidate",
            ),
        )
        with ExitStack() as stack:
            for owner, name in forbidden:
                stack.enter_context(
                    patch.object(
                        owner,
                        name,
                        side_effect=AssertionError(
                            f"disabled path called {name}"
                        ),
                    )
                )
            result = run_operator_localized_phase_edge_candidate(
                poison,
                poison,
                poison,
                deadline=poison,
                enabled=False,
                selection_max_rivals=poison,
                selection_max_binaries=poison,
                selection_max_work_items=poison,
                selection_timeout_seconds=poison,
                max_parent_variables=poison,
                max_parent_rows=poison,
                max_parent_nonzeros=poison,
                max_parent_buffer_items=poison,
                max_top_literals=poison,
                max_total_pairs=poison,
                max_source_terms=poison,
                max_multiplier_bits=poison,
                max_exact_bits=poison,
                max_exact_nonzeros=poison,
                localized_row_tiers=poison,
                localized_max_selected_nnz=poison,
                localized_max_source_terms=poison,
            )
        self.assertEqual(result.status, "disabled")
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)
        self.assertIsNone(result.caps)
        self.assertIsNone(result.parent_semantic_digest)
        self.assertFalse(result.producer_nonempty_seal_verified)
        self.assertFalse(result.source_modes)
        self.assertFalse(result.proof_authority)
        self.assertIs(
            result,
            run_operator_localized_phase_edge_candidate(
                poison, poison, poison, deadline=poison, enabled=False
            ),
        )
        self.assertEqual(
            result.result_sha256,
            edge_module._sha256(
                edge_module._result_payload(result, include_digest=False)
            ),
        )

    def test_top2_tie_break_is_deterministic_and_ignores_caller_mapping_order(
        self,
    ) -> None:
        build = _sealed_corner_build(bias=-1.5)
        rivals, selection = _selection(build)
        poisoned = replace(
            selection,
            mappings=tuple(reversed(selection.mappings)),
            literals=tuple(reversed(selection.literals)),
        )
        first = run_operator_localized_phase_edge_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 20.0,
            enabled=True,
        )
        second = run_operator_localized_phase_edge_candidate(
            build,
            rivals,
            poisoned,
            deadline=time.monotonic() + 20.0,
            enabled=True,
        )
        expected = tuple(
            sorted(mapping.stable_bcol_id for mapping in selection.mappings)[:2]
        )
        self.assertEqual(
            tuple(item.stable_bcol_id for item in first.ranked_phases), expected
        )
        self.assertEqual(first.ranked_phases, second.ranked_phases)
        self.assertEqual(first.subset_binding_digest, second.subset_binding_digest)

    def test_k4_top2_gets_exact_full_parent_edge(self) -> None:
        build = _sealed_corner_build(bias=-1.5)
        rivals, selection = _selection(build)
        result = run_operator_localized_phase_edge_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 30.0,
            enabled=True,
        )
        self.assertEqual(result.status, "certified_localized_phase_edge")
        self.assertTrue(result.edge_accepted)
        self.assertIsNotNone(result.certificate)
        self.assertIsNotNone(result.localized_result)
        self.assertEqual(result.certificate.literals, result.literals)
        self.assertEqual(
            result.certificate.property_digest, result.subset_binding_digest
        )
        self.assertLess(result.certificate.contradiction, 0)
        self.assertTrue(result.producer_nonempty_seal_verified)
        self.assertTrue(result.source_modes)
        self.assertEqual(
            result.source_modes_sha256,
            edge_module._source_modes_digest(result.source_modes),
        )

    def test_compatible_top2_is_not_an_edge(self) -> None:
        build = _sealed_corner_build(bias=0.0)
        rivals, selection = _selection(build)
        result = run_operator_localized_phase_edge_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 30.0,
            enabled=True,
        )
        self.assertEqual(result.status, "no_certified_localized_phase_edge")
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)

    def test_stale_constructive_seal_rejects_persistently_mutated_parent(
        self,
    ) -> None:
        build = _sealed_corner_build(bias=0.0)
        original_rivals, original_selection = _selection(build)
        original = run_operator_localized_phase_edge_candidate(
            build,
            original_rivals,
            original_selection,
            deadline=time.monotonic() + 30.0,
            enabled=True,
        )
        self.assertFalse(original.edge_accepted)
        build.hz.ub[0] = np.nextafter(build.hz.ub[0], np.inf)
        rivals, stale_parent_selection = _selection(build)
        with patch.object(
            edge_module.localized_oracle,
            "run_localized_phase_conflict_oracle_candidate",
            side_effect=AssertionError("stale seal reached localized oracle"),
        ) as localized:
            with self.assertRaisesRegex(
                OperatorLocalizedPhaseEdgeError,
                "constructive.*seal|seal.*invalid",
            ):
                run_operator_localized_phase_edge_candidate(
                    build,
                    rivals,
                    stale_parent_selection,
                    deadline=time.monotonic() + 30.0,
                    enabled=True,
                )
        localized.assert_not_called()

    def test_active_source_modes_are_rejected_before_localized(self) -> None:
        cases = (
            (
                "prefix",
                lambda build: setattr(
                    build.hz,
                    "_solver_row_constraint_prefix_frames",
                    {7: {"active": True}},
                ),
            ),
            (
                "full_input",
                lambda build: setattr(
                    build.hz,
                    "_property_full_input_replay_result",
                    object(),
                ),
            ),
            (
                "micro_rlt",
                lambda build: setattr(
                    build.hz,
                    "_property_micro_rlt_receipt",
                    object(),
                ),
            ),
            (
                "query_dual",
                lambda build: build.metadata.__setitem__(
                    "verified_query_dual_active", {"enabled": True}
                ),
            ),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                build = _sealed_corner_build(bias=-1.5)
                mutate(build)
                rivals, selection = _selection(build)
                with patch.object(
                    edge_module.localized_oracle,
                    "run_localized_phase_conflict_oracle_candidate",
                    side_effect=AssertionError(
                        "active source mode reached localized oracle"
                    ),
                ) as localized:
                    with self.assertRaises(OperatorLocalizedPhaseEdgeError):
                        run_operator_localized_phase_edge_candidate(
                            build,
                            rivals,
                            selection,
                            deadline=time.monotonic() + 30.0,
                            enabled=True,
                        )
                localized.assert_not_called()

    def test_zero_and_one_literal_stop_before_localized(self) -> None:
        for count in (0, 1):
            with self.subTest(count=count):
                build, rivals, selection = _effect_count_build(count)
                with patch.object(
                    edge_module.localized_oracle,
                    "run_localized_phase_conflict_oracle_candidate",
                ) as localized:
                    result = run_operator_localized_phase_edge_candidate(
                        build,
                        rivals,
                        selection,
                        deadline=time.monotonic() + 20.0,
                        enabled=True,
                    )
                localized.assert_not_called()
                self.assertEqual(result.status, "insufficient_ranked_literals")
                self.assertEqual(len(result.literals), count)
                self.assertFalse(result.edge_accepted)

    def test_wrong_selection_digest_and_deadline_fail_before_candidate(self) -> None:
        build = _sealed_corner_build(bias=-1.5)
        rivals, selection = _selection(build)
        with self.assertRaisesRegex(
            OperatorLocalizedPhaseEdgeError,
            "selection_digest_mismatch",
        ):
            run_operator_localized_phase_edge_candidate(
                build,
                rivals,
                replace(selection, selection_digest="0" * 64),
                deadline=time.monotonic() + 20.0,
                enabled=True,
            )
        with self.assertRaisesRegex(
            OperatorLocalizedPhaseEdgeError,
            "deadline",
        ):
            run_operator_localized_phase_edge_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() - 1.0,
                enabled=True,
            )
        with self.assertRaisesRegex(
            OperatorLocalizedPhaseEdgeError,
            "source_term_caps_mismatch",
        ):
            run_operator_localized_phase_edge_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                enabled=True,
                max_source_terms=64,
                localized_max_source_terms=128,
            )

    def test_live_parent_mutation_after_localized_discards_edge(self) -> None:
        build = _sealed_corner_build(bias=-1.5)
        rivals, selection = _selection(build)
        real = (
            edge_module.localized_oracle
            .run_localized_phase_conflict_oracle_candidate
        )

        def mutate_after_private_run(*args, **kwargs):
            result = real(*args, **kwargs)
            build.hz.ub[0] = np.nextafter(build.hz.ub[0], np.inf)
            return result

        with patch.object(
            edge_module.localized_oracle,
            "run_localized_phase_conflict_oracle_candidate",
            side_effect=mutate_after_private_run,
        ):
            result = run_operator_localized_phase_edge_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 30.0,
                enabled=True,
            )
        self.assertEqual(result.status, "parent_mutated")
        self.assertFalse(result.parent_unchanged)
        self.assertFalse(result.edge_accepted)
        self.assertIsNone(result.certificate)

    def test_adapter_does_not_mutate_live_parent(self) -> None:
        build = _sealed_corner_build(bias=-1.5)
        rivals, selection = _selection(build)
        before = sparse_hz_semantic_digest(build.hz)
        tags = tuple(build.hz._solver_constraint_row_tags)
        result = run_operator_localized_phase_edge_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 30.0,
            enabled=True,
        )
        self.assertEqual(sparse_hz_semantic_digest(build.hz), before)
        self.assertEqual(tuple(build.hz._solver_constraint_row_tags), tags)
        self.assertTrue(result.parent_unchanged)

    def test_tampered_localized_receipt_is_rejected(self) -> None:
        build = _sealed_corner_build(bias=-1.5)
        rivals, selection = _selection(build)
        real = (
            edge_module.localized_oracle
            .run_localized_phase_conflict_oracle_candidate
        )

        def tamper(*args, **kwargs):
            result = real(*args, **kwargs)
            return replace(result, result_sha256="0" * 64)

        with patch.object(
            edge_module.localized_oracle,
            "run_localized_phase_conflict_oracle_candidate",
            side_effect=tamper,
        ):
            with self.assertRaisesRegex(
                OperatorLocalizedPhaseEdgeError,
                "localized_result_binding_mismatch",
            ):
                run_operator_localized_phase_edge_candidate(
                    build,
                    rivals,
                    selection,
                    deadline=time.monotonic() + 30.0,
                    enabled=True,
                )

    def test_source_has_no_production_or_ground_truth_connection(self) -> None:
        source = Path(edge_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("verify_once(", source)
        self.assertNotIn("operator_exact_relu_phase_clique_materializer", source)
        self.assertNotIn("_copy_parent_with_clique_cut", source)
        self.assertNotIn("ground_truth", source.lower())


if __name__ == "__main__":
    unittest.main()
