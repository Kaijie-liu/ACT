#!/usr/bin/env python3
"""Adversarial gates for the Operator-HZ phase-clique solver handoff.

These tests deliberately mutate objects at the last public/private ownership
boundaries.  A checksummed receipt or an old constructive-nonempty marker must
never authorize a live HZ whose core changed after the corresponding audit.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import multiprocessing
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.adaptive_phase_forest import (
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf import (
    operator_exact_relu_phase_clique_materializer as materializer_module,
)
from act.back_end.hybridz_tf import (
    operator_phase_clique_pipeline as pipeline_module,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_clique_materializer import (
    OperatorPhaseCliqueMaterializationError,
)
from act.back_end.hybridz_tf.operator_phase_clique_pipeline import (
    OperatorPhaseCliquePipelineError,
    consume_operator_phase_clique_pipeline_solver_handoff,
    validate_consumed_operator_phase_clique_solver_build,
    verify_operator_phase_clique_pipeline_result,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_clique_materializer import (
    _chain,
    _materialize,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _k4_corner_build,
)
from act.back_end.hybridz_tf.test_operator_phase_clique_pipeline import (
    _run,
    _write_raw,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    hz_base_feasibility,
    hz_constructively_nonempty,
)


_DENSE_CORE_NAMES = ("c", "b", "ub", "col_ids", "bcol_ids")
_CSR_CORE_NAMES = ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")
_PROVENANCE_NAMES = (
    "full_col_ids",
    "operator_input_center",
    "operator_input_radius",
    "_solver_continuous_column_layer_ids",
)


def _clone_core(hz: SparseHZono) -> SparseHZono:
    """Copy only the public SparseHZono core, never audit attributes."""

    return SparseHZono(
        c=hz.c.copy(),
        Gc=hz.Gc.copy(),
        Gb=hz.Gb.copy(),
        Ac=hz.Ac.copy(),
        Ab=hz.Ab.copy(),
        b=hz.b.copy(),
        Auc=hz.Auc.copy(),
        Aub=hz.Aub.copy(),
        ub=hz.ub.copy(),
        col_ids=hz.col_ids.copy(),
        bcol_ids=hz.bcol_ids.copy(),
    )


def _append_contradictory_zero_row(hz: SparseHZono) -> None:
    """Make the live base exactly empty via the row ``0 <= -1``."""

    hz.Auc = sp.vstack(
        [
            hz.Auc,
            sp.csr_matrix((1, hz.n_cont), dtype=np.float64),
        ],
        format="csr",
    )
    hz.Aub = sp.vstack(
        [
            hz.Aub,
            sp.csr_matrix((1, hz.n_bin), dtype=np.float64),
        ],
        format="csr",
    )
    hz.ub = np.concatenate(
        [hz.ub, np.array([-1.0], dtype=np.float64)]
    )


def _success_result():
    source = _k4_corner_build()
    with tempfile.TemporaryDirectory() as directory:
        path, source_sha256 = _write_raw(directory)
        result = _run(source, path, source_sha256)
    if not result.materialized:
        raise AssertionError("K4 success fixture did not materialize")
    return source, result


def _fallback_result():
    source = _k4_corner_build()
    with tempfile.TemporaryDirectory() as directory:
        path, source_sha256 = _write_raw(directory)
        with mock.patch.object(
            pipeline_module,
            "issue_raw_vnnlib_top1_candidate",
            side_effect=RuntimeError("synthetic fallback"),
        ):
            result = _run(source, path, source_sha256)
    if result.materialized or result.build is not source:
        raise AssertionError("enabled fallback fixture changed identity")
    return source, result


def _owned_arrays(build) -> tuple[np.ndarray, ...]:
    hz = build.hz
    values = [build.input_col_ids]
    values.extend(getattr(hz, name) for name in _DENSE_CORE_NAMES)
    for name in _CSR_CORE_NAMES:
        matrix = getattr(hz, name)
        values.extend((matrix.data, matrix.indices, matrix.indptr))
    values.extend(getattr(hz, name) for name in _PROVENANCE_NAMES)
    if any(type(value) is not np.ndarray for value in values):
        raise AssertionError("solver-owned array inventory is malformed")
    return tuple(values)


def _fork_consume_attempt(source, result, deadline, connection) -> None:
    """Child entry point: a fork must not inherit handoff authority."""

    try:
        consume_operator_phase_clique_pipeline_solver_handoff(
            source, result, deadline=deadline
        )
    except BaseException as exc:
        connection.send((False, type(exc).__name__))
    else:
        connection.send((True, None))
    finally:
        connection.close()


def _rebase_private_parent_digest(snapshot) -> None:
    """Model an attacker updating the visible digest but not its seal."""

    parent = materializer_module._parent_prefix_from_verified_cut(
        snapshot.cut_hz,
        original_parent_n_ub=snapshot.original_parent_n_ub,
    )
    object.__setattr__(
        snapshot,
        "parent_semantic_digest",
        sparse_hz_semantic_digest(parent),
    )


class OperatorPhaseCliqueSolverHandoffAttackTests(unittest.TestCase):
    def test_consumed_snapshot_digest_rebase_after_buffer_mutation_rejects(
        self,
    ) -> None:
        build, rivals, selection, candidate = _chain()
        real_consume = (
            materializer_module
            .consume_verified_operator_phase_clique_snapshot
        )

        def consume_then_mutate(capability, *, deadline):
            snapshot = real_consume(
                capability, deadline=deadline
            )
            # Read-only NumPy ownership is not an integrity boundary: an
            # owning array can have writeability re-enabled.
            snapshot.cut_hz.c.setflags(write=True)
            snapshot.cut_hz.c[0] += 123.0
            _rebase_private_parent_digest(snapshot)
            return snapshot

        with mock.patch.object(
            materializer_module,
            "consume_verified_operator_phase_clique_snapshot",
            side_effect=consume_then_mutate,
        ):
            with self.assertRaises(
                OperatorPhaseCliqueMaterializationError
            ):
                _materialize(
                    build, rivals, selection, candidate
                )

    def test_consumed_snapshot_cut_attribute_replacement_rejects(
        self,
    ) -> None:
        build, rivals, selection, candidate = _chain()
        real_consume = (
            materializer_module
            .consume_verified_operator_phase_clique_snapshot
        )

        def consume_then_replace(capability, *, deadline):
            snapshot = real_consume(
                capability, deadline=deadline
            )
            replacement = _clone_core(snapshot.cut_hz)
            replacement.c[0] += 321.0
            # frozen=True blocks ordinary assignment, but a security replay
            # must also detect low-level attribute replacement.
            object.__setattr__(
                snapshot, "cut_hz", replacement
            )
            _rebase_private_parent_digest(snapshot)
            return snapshot

        with mock.patch.object(
            materializer_module,
            "consume_verified_operator_phase_clique_snapshot",
            side_effect=consume_then_replace,
        ):
            with self.assertRaises(
                OperatorPhaseCliqueMaterializationError
            ):
                _materialize(
                    build, rivals, selection, candidate
                )

    def test_public_success_last_seal_mutation_rejects(self) -> None:
        source = _k4_corner_build()
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            result = _run(source, path, source_sha256)
        self.assertTrue(result.materialized)
        fresh = result.build.hz
        real_terminal_seal = (
            pipeline_module._terminal_semantic_pair_seal
        )

        def seal_then_mutate(*args, **kwargs):
            sealed = real_terminal_seal(*args, **kwargs)
            # This is the final callback before the current verifier returns;
            # it exercises the hash-to-return TOCTOU window directly.
            _append_contradictory_zero_row(fresh)
            return sealed

        with mock.patch.object(
            pipeline_module,
            "_terminal_semantic_pair_seal",
            side_effect=seal_then_mutate,
        ):
            self.assertFalse(
                verify_operator_phase_clique_pipeline_result(
                    source, result
                )
            )

    def test_enabled_fallback_source_mutation_rejects(self) -> None:
        source = _k4_corner_build()
        with tempfile.TemporaryDirectory() as directory:
            path, source_sha256 = _write_raw(directory)
            with mock.patch.object(
                pipeline_module,
                "issue_raw_vnnlib_top1_candidate",
                side_effect=RuntimeError("synthetic fallback"),
            ):
                result = _run(source, path, source_sha256)
        self.assertTrue(result.enabled)
        self.assertFalse(result.materialized)
        self.assertIs(result.build, source)

        _append_contradictory_zero_row(source.hz)
        self.assertFalse(
            verify_operator_phase_clique_pipeline_result(
                source, result
            )
        )

    def test_success_and_enabled_fallback_consume_private_builds(
        self,
    ) -> None:
        for name, factory, expected_materialized in (
            ("success", _success_result, True),
            ("fallback", _fallback_result, False),
        ):
            with self.subTest(name=name):
                source, result = factory()
                self.assertIs(
                    result.materialized, expected_materialized
                )
                self.assertTrue(
                    verify_operator_phase_clique_pipeline_result(
                        source, result
                    )
                )
                self.assertIsNotNone(
                    result.solver_handoff_capability
                )
                self.assertEqual(
                    result.receipt["solver_handoff_status"],
                    "issued",
                )

                private = (
                    consume_operator_phase_clique_pipeline_solver_handoff(
                        source,
                        result,
                        deadline=time.monotonic() + 10.0,
                    )
                )
                self.assertIsNot(private, result.build)
                self.assertIsNot(private.hz, result.build.hz)
                self.assertTrue(
                    validate_consumed_operator_phase_clique_solver_build(
                        result, private
                    )
                )
                self.assertTrue(
                    hz_constructively_nonempty(private.hz)
                )
                self.assertEqual(
                    hz_base_feasibility(
                        private.hz, time_limit=1.0
                    )[0],
                    "FEASIBLE",
                )

    def test_public_mutation_after_consume_cannot_change_private_build(
        self,
    ) -> None:
        for name, factory in (
            ("success", _success_result),
            ("fallback", _fallback_result),
        ):
            with self.subTest(name=name):
                source, result = factory()
                private = (
                    consume_operator_phase_clique_pipeline_solver_handoff(
                        source,
                        result,
                        deadline=time.monotonic() + 10.0,
                    )
                )
                private_digest = sparse_hz_semantic_digest(
                    private.hz
                )
                private_arrays = tuple(
                    value.copy() for value in _owned_arrays(private)
                )

                public = result.build
                public.hz.c[0] += 777.0
                public.input_col_ids[0] += 17
                public.hz.full_col_ids[0] += 19
                public.hz.operator_input_center[0] += 23.0
                public.hz.operator_input_radius[0] += 29.0
                public.hz._solver_continuous_column_layer_ids[0] += 31
                _append_contradictory_zero_row(public.hz)

                self.assertEqual(
                    sparse_hz_semantic_digest(private.hz),
                    private_digest,
                )
                for before, after in zip(
                    private_arrays, _owned_arrays(private)
                ):
                    self.assertTrue(np.array_equal(before, after))
                self.assertTrue(
                    validate_consumed_operator_phase_clique_solver_build(
                        result, private
                    )
                )
                self.assertEqual(
                    hz_base_feasibility(
                        private.hz, time_limit=1.0
                    )[0],
                    "FEASIBLE",
                )

    def test_private_dense_csr_and_provenance_are_readonly_no_alias(
        self,
    ) -> None:
        for name, factory in (
            ("success", _success_result),
            ("fallback", _fallback_result),
        ):
            with self.subTest(name=name):
                source, result = factory()
                private = (
                    consume_operator_phase_clique_pipeline_solver_handoff(
                        source,
                        result,
                        deadline=time.monotonic() + 10.0,
                    )
                )
                public = result.build
                public_arrays = _owned_arrays(public)
                private_arrays = _owned_arrays(private)

                self.assertTrue(private_arrays)
                self.assertTrue(
                    all(
                        value.flags.writeable is False
                        for value in private_arrays
                    )
                )
                self.assertFalse(
                    any(
                        np.shares_memory(left, right)
                        for left in public_arrays
                        for right in private_arrays
                    )
                )
                for value in private_arrays:
                    if value.size:
                        with self.assertRaises(ValueError):
                            value.flat[0] = value.flat[0]

                self.assertIsNot(
                    private.input_col_ids, public.input_col_ids
                )
                self.assertIsNot(private.metadata, public.metadata)
                for core_name in _DENSE_CORE_NAMES:
                    self.assertIsNot(
                        getattr(private.hz, core_name),
                        getattr(public.hz, core_name),
                    )
                for core_name in _CSR_CORE_NAMES:
                    private_matrix = getattr(
                        private.hz, core_name
                    )
                    public_matrix = getattr(public.hz, core_name)
                    self.assertIsNot(private_matrix, public_matrix)
                    for buffer_name in (
                        "data",
                        "indices",
                        "indptr",
                    ):
                        self.assertFalse(
                            np.shares_memory(
                                getattr(
                                    private_matrix, buffer_name
                                ),
                                getattr(public_matrix, buffer_name),
                            )
                        )
                for provenance_name in _PROVENANCE_NAMES:
                    self.assertIsNot(
                        getattr(private.hz, provenance_name),
                        getattr(public.hz, provenance_name),
                    )

    def test_handoff_is_one_shot_and_copied_capability_rejects(
        self,
    ) -> None:
        source, result = _success_result()
        private = consume_operator_phase_clique_pipeline_solver_handoff(
            source,
            result,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                result, private
            )
        )
        with self.assertRaises(OperatorPhaseCliquePipelineError):
            consume_operator_phase_clique_pipeline_solver_handoff(
                source,
                result,
                deadline=time.monotonic() + 10.0,
            )

        source, result = _success_result()
        copied_capability = copy.copy(
            result.solver_handoff_capability
        )
        forged_result = replace(
            result,
            solver_handoff_capability=copied_capability,
        )
        with self.assertRaises(OperatorPhaseCliquePipelineError):
            consume_operator_phase_clique_pipeline_solver_handoff(
                source,
                forged_result,
                deadline=time.monotonic() + 10.0,
            )
        private = consume_operator_phase_clique_pipeline_solver_handoff(
            source,
            result,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                result, private
            )
        )

    def test_handoff_rejects_source_owner_build_and_hz_replacement(
        self,
    ) -> None:
        source, result = _success_result()
        wrong_source_owner = replace(source)
        with self.assertRaises(OperatorPhaseCliquePipelineError):
            consume_operator_phase_clique_pipeline_solver_handoff(
                wrong_source_owner,
                result,
                deadline=time.monotonic() + 10.0,
            )
        private = consume_operator_phase_clique_pipeline_solver_handoff(
            source,
            result,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                result, private
            )
        )

        source, result = _success_result()
        object.__setattr__(
            result, "build", replace(result.build)
        )
        with self.assertRaises(OperatorPhaseCliquePipelineError):
            consume_operator_phase_clique_pipeline_solver_handoff(
                source,
                result,
                deadline=time.monotonic() + 10.0,
            )

        source, result = _success_result()
        object.__setattr__(
            result.build,
            "hz",
            _clone_core(result.build.hz),
        )
        with self.assertRaises(OperatorPhaseCliquePipelineError):
            consume_operator_phase_clique_pipeline_solver_handoff(
                source,
                result,
                deadline=time.monotonic() + 10.0,
            )

    def test_two_threads_can_consume_handoff_only_once(self) -> None:
        source, result = _success_result()
        barrier = threading.Barrier(3)
        successes = []
        failures = []
        deadline = time.monotonic() + 10.0

        def consume() -> None:
            barrier.wait()
            try:
                successes.append(
                    consume_operator_phase_clique_pipeline_solver_handoff(
                        source, result, deadline=deadline
                    )
                )
            except BaseException as exc:
                failures.append(exc)

        threads = [
            threading.Thread(target=consume),
            threading.Thread(target=consume),
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(10.0)

        self.assertTrue(
            all(not thread.is_alive() for thread in threads)
        )
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(
            failures[0], OperatorPhaseCliquePipelineError
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                result, successes[0]
            )
        )

    def test_fork_child_rejects_but_parent_can_still_consume(self) -> None:
        source, result = _success_result()
        context = multiprocessing.get_context("fork")
        receiving, sending = context.Pipe(duplex=False)
        deadline = time.monotonic() + 10.0
        process = context.Process(
            target=_fork_consume_attempt,
            args=(source, result, deadline, sending),
        )
        process.start()
        sending.close()
        self.assertTrue(receiving.poll(10.0))
        child_succeeded, error_type = receiving.recv()
        receiving.close()
        process.join(10.0)

        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        self.assertFalse(child_succeeded)
        self.assertEqual(
            error_type, "OperatorPhaseCliquePipelineError"
        )

        private = consume_operator_phase_clique_pipeline_solver_handoff(
            source,
            result,
            deadline=time.monotonic() + 10.0,
        )
        self.assertTrue(
            validate_consumed_operator_phase_clique_solver_build(
                result, private
            )
        )


if __name__ == "__main__":
    unittest.main()
