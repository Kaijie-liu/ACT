"""Pure-function regression for CLI multi-query aggregation.

Added 2026-05-24 Round 3 per advisor: the inline aggregation logic at
``_run_vnnlib_verify_hybridz`` lacked a regression test. After this
refactor, ``aggregate_query_statuses`` is the canonical pure function
and is the only place the disjunctive UNSAFE-set semantic is encoded.

The four cases from the advisor's review:
    SAT, UNKNOWN → SAT
    UNSAT, UNSAT → UNSAT
    UNSAT, UNKNOWN → UNKNOWN
    multi-query SAT short-circuits but the returned status still SAT
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from act.pipeline.cli import (
    aggregate_query_statuses,
    aggregate_reportable_verdicts,
    select_pairs_by_official_ids,
    compute_run_status,
    IncompleteFormalAuditError,
    remaining_instance_query_budget,
)


class TestAggregateQueryStatuses(unittest.TestCase):
    def test_any_sat_wins(self):
        self.assertEqual(aggregate_query_statuses(["SAT", "UNKNOWN"]), "SAT")
        self.assertEqual(aggregate_query_statuses(["UNKNOWN", "SAT"]), "SAT")
        self.assertEqual(aggregate_query_statuses(["UNSAT", "SAT", "UNSAT"]), "SAT")

    def test_all_unsat_wins(self):
        self.assertEqual(aggregate_query_statuses(["UNSAT"]), "UNSAT")
        self.assertEqual(aggregate_query_statuses(["UNSAT", "UNSAT"]), "UNSAT")
        self.assertEqual(
            aggregate_query_statuses(["UNSAT", "UNSAT", "UNSAT", "UNSAT"]), "UNSAT"
        )

    def test_mixed_unknown_is_unknown(self):
        self.assertEqual(aggregate_query_statuses(["UNSAT", "UNKNOWN"]), "UNKNOWN")
        self.assertEqual(aggregate_query_statuses(["UNKNOWN", "UNSAT"]), "UNKNOWN")
        self.assertEqual(aggregate_query_statuses(["UNKNOWN"]), "UNKNOWN")
        self.assertEqual(
            aggregate_query_statuses(["UNSAT", "UNKNOWN", "UNSAT"]), "UNKNOWN"
        )

    def test_empty_input(self):
        self.assertEqual(aggregate_query_statuses([]), "UNKNOWN")

    def test_sat_dominates_even_mixed(self):
        """Short-circuit semantics: even with UNKNOWN later, SAT wins."""
        self.assertEqual(
            aggregate_query_statuses(["SAT", "UNKNOWN", "UNSAT"]), "SAT"
        )

    def test_unrecognized_status_is_unknown(self):
        """Defensive: garbage status from a misbehaving solver doesn't
        accidentally become SAT or UNSAT."""
        self.assertEqual(aggregate_query_statuses(["TIMEOUT"]), "UNKNOWN")
        self.assertEqual(aggregate_query_statuses(["ERROR", "ERROR"]), "UNKNOWN")
        # SAT still wins over unrecognized
        self.assertEqual(aggregate_query_statuses(["ERROR", "SAT"]), "SAT")


class TestAggregateReportableVerdicts(unittest.TestCase):
    """Round 4 (advisor 2026-05-24): the reportable aggregation must
    preserve FAL when at least one query is genuinely falsified, and
    surface ERROR_RECEIPT_* when no query falsifies but some query has
    a receipt error."""

    def test_falsified_beats_error(self):
        # A real FAL in one query suffices to falsify the spec; an
        # error from a DIFFERENT query is irrelevant.
        self.assertEqual(
            aggregate_reportable_verdicts(
                ["ERROR_RECEIPT_MISSING", "FALSIFIED"]
            ),
            "FALSIFIED",
        )

    def test_error_propagates_when_no_falsified(self):
        # No FAL → first ERROR wins.
        self.assertEqual(
            aggregate_reportable_verdicts(
                ["CERTIFIED", "ERROR_RECEIPT_MISSING", "UNKNOWN"]
            ),
            "ERROR_RECEIPT_MISSING",
        )

    def test_all_certified(self):
        self.assertEqual(
            aggregate_reportable_verdicts(["CERTIFIED", "CERTIFIED"]),
            "CERTIFIED",
        )

    def test_mixed_certified_unknown(self):
        self.assertEqual(
            aggregate_reportable_verdicts(["CERTIFIED", "UNKNOWN"]),
            "UNKNOWN",
        )

    def test_empty_input(self):
        self.assertEqual(aggregate_reportable_verdicts([]), "UNKNOWN")

    def test_error_inconsistency_propagates(self):
        # Internal inconsistency is also an ERROR_ prefix → propagates
        self.assertEqual(
            aggregate_reportable_verdicts(
                ["CERTIFIED", "ERROR_INTERNAL_INCONSISTENCY"]
            ),
            "ERROR_INTERNAL_INCONSISTENCY",
        )

    def test_falsified_short_circuit_query_index_preserved(self):
        # Note: the actual short-circuit happens in the CLI loop (it
        # breaks on the first SAT). The aggregation only sees what was
        # produced. This test documents that a single-element [FALSIFIED]
        # is correctly the aggregate, simulating short-circuit.
        self.assertEqual(
            aggregate_reportable_verdicts(["FALSIFIED"]),
            "FALSIFIED",
        )


class TestSelectPairsByOfficialIds(unittest.TestCase):
    def setUp(self):
        self.pairs = [
            {"official_instance_id": 0, "vnnlib_spec": "p0"},
            {"official_instance_id": 100, "vnnlib_spec": "p100"},
            {"official_instance_id": 181, "vnnlib_spec": "prop_6"},
        ]

    def test_selects_official_id_in_requested_order(self):
        chosen = select_pairs_by_official_ids(self.pairs, "181,100")
        self.assertEqual([p["official_instance_id"] for p in chosen], [181, 100])

    def test_no_selection_preserves_input(self):
        self.assertEqual(select_pairs_by_official_ids(self.pairs, None), self.pairs)

    def test_missing_or_duplicate_ids_fail_loudly(self):
        with self.assertRaises(ValueError):
            select_pairs_by_official_ids(self.pairs, "999")
        with self.assertRaises(ValueError):
            select_pairs_by_official_ids(self.pairs, "100,100")


class TestComputeRunStatus(unittest.TestCase):
    """ROUND 6 (advisor 2026-05-24): unattended runs must distinguish
    a clean formal pass from an INCOMPLETE_FORMAL_AUDIT. Pure function
    surface so the CLI hook is testable without spinning up a real run."""

    def test_non_formal_receipt_noise_does_not_invalidate_math_run(self):
        # Non-formal mode never reports INCOMPLETE for receipt-only noise.
        self.assertEqual(
            compute_run_status({"CERTIFIED": 1, "ERROR_RECEIPT": 3},
                               formal_mode=False),
            "PASSED",
        )

    def test_formal_clean_run_passes(self):
        self.assertEqual(
            compute_run_status({"CERTIFIED": 5, "FALSIFIED": 1, "UNKNOWN": 2},
                               formal_mode=True),
            "PASSED",
        )

    def test_formal_with_receipt_errors_is_incomplete(self):
        self.assertEqual(
            compute_run_status({"FALSIFIED": 2, "ERROR_RECEIPT": 1},
                               formal_mode=True),
            "INCOMPLETE_FORMAL_AUDIT",
        )

    def test_formal_with_internal_inconsistency_is_incomplete(self):
        # ERROR_INTERNAL_INCONSISTENCY would be bucketed under
        # ERROR_RECEIPT by the CLI's current bucketing, but we also
        # accept it as its own key so the contract is independent of
        # bucket-name choice.
        self.assertEqual(
            compute_run_status({"ERROR_INTERNAL_INCONSISTENCY": 1},
                               formal_mode=True),
            "INCOMPLETE_FORMAL_AUDIT",
        )

    def test_generic_error_fails_formal_run(self):
        self.assertEqual(
            compute_run_status({"CERTIFIED": 5, "ERROR": 3},
                               formal_mode=True),
            "FAILED",
        )

    def test_generic_error_fails_non_formal_run(self):
        self.assertEqual(
            compute_run_status({"ERROR": 1}, formal_mode=False),
            "FAILED",
        )

    def test_incomplete_audit_error_carries_count_message(self):
        # Sanity: the exception type is RuntimeError subclass and
        # carries a meaningful message body for operators.
        try:
            raise IncompleteFormalAuditError(
                "INCOMPLETE_FORMAL_AUDIT: ERROR_RECEIPT=2, "
                "ERROR_INTERNAL_INCONSISTENCY=0."
            )
        except IncompleteFormalAuditError as e:
            self.assertIn("ERROR_RECEIPT=2", str(e))
            self.assertIsInstance(e, RuntimeError)


class TestRemainingInstanceQueryBudget(unittest.TestCase):
    """A multi-query instance must share one fail-closed CLI budget."""

    def test_subtracts_elapsed_time(self):
        self.assertAlmostEqual(remaining_instance_query_budget(30.0, 7.5), 22.5)

    def test_exhausted_budget_clamps_to_zero(self):
        self.assertEqual(remaining_instance_query_budget(30.0, 30.0), 0.0)
        self.assertEqual(remaining_instance_query_budget(30.0, 31.0), 0.0)

    def test_negative_elapsed_does_not_increase_budget(self):
        self.assertEqual(remaining_instance_query_budget(30.0, -1.0), 30.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
