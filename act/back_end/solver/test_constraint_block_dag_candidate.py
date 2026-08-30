#!/usr/bin/env python3
# ===- test_constraint_block_dag_candidate.py - exact source DAG gates --===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Independent bounded gates for the disconnected exact RANGE/DAG MVP."""

from __future__ import annotations

from fractions import Fraction
import gc
import math
import random
import threading
import unittest
from unittest import mock
import weakref

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import constraint_block_dag_candidate as core
from act.back_end.solver.constraint_block_dag_candidate import (
    ConstraintArenaMismatch,
    ConstraintBlockDAGCandidateError,
    ExactConstraintOwner,
    StableFactorID,
    benchmark_bounded_c89_ratio,
)

try:
    import highspy
except ImportError:  # pragma: no cover - optional native dependency.
    highspy = None


def _canonical(rows, *, columns=None) -> sp.csr_matrix:
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        raise AssertionError("test CSR input must be rank two")
    shape = (
        array.shape
        if columns is None
        else (int(array.shape[0]), int(columns))
    )
    matrix = sp.csr_matrix(array, shape=shape, dtype=np.float64)
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _csr_bits(matrix: sp.csr_matrix):
    matrix = matrix.tocsr()
    return (
        tuple(int(value) for value in matrix.shape),
        np.asarray(matrix.indptr, dtype=np.int64).tobytes(),
        np.asarray(matrix.indices, dtype=np.int64).tobytes(),
        np.asarray(matrix.data, dtype=np.float64).view(np.uint64).tobytes(),
    )


def _new_arena(n_cont=2, n_bin=1):
    owner = ExactConstraintOwner()
    owner.allocate_continuous(int(n_cont))
    owner.allocate_binary(int(n_bin))
    frame = owner.frame()
    arena = owner.new_arena()
    return owner, frame, arena


def _append(
    arena,
    frame,
    *,
    view=None,
    fc=None,
    fb=None,
    fu=None,
    rc=None,
    rb=None,
    ru=None,
    layer_id=7,
):
    if fc is None:
        fc = _canonical([[1.0, -2.0], [0.5, 0.0]])
    if fb is None:
        fb = _canonical([[3.0], [0.0]])
    if fu is None:
        fu = np.asarray([1.25, 2.0], dtype=np.float64)
    if rc is None:
        rc = -fc
    if rb is None:
        rb = -fb
    if ru is None:
        ru = np.asarray([0.75, 1.0], dtype=np.float64)
    return arena.append_guarded_band(
        arena.empty_view if view is None else view,
        frame=frame,
        forward_cont=fc,
        forward_bin=fb,
        forward_upper=fu,
        reverse_cont=rc,
        reverse_bin=rb,
        reverse_upper=ru,
        layer_id=int(layer_id),
    )


def _exact_row_value(matrix, row, values):
    total = Fraction(0)
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    for position in range(start, stop):
        total += Fraction.from_float(float(matrix.data[position])) * values[
            int(matrix.indices[position])
        ]
    return total


def _collect_stream(program, max_rows):
    batches = list(
        core.iter_virtual_facet_batches(program, max_rows=max_rows)
    )
    if not batches:
        return None, batches
    return (
        {
            "A_cont": sp.vstack(
                [batch.A_cont for batch in batches], format="csr"
            ),
            "A_bin": sp.vstack(
                [batch.A_bin for batch in batches], format="csr"
            ),
            "upper": np.concatenate([batch.upper for batch in batches]),
            "row_ids": tuple(
                row_id for batch in batches for row_id in batch.row_ids
            ),
            "row_tags": tuple(
                tag for batch in batches for tag in batch.row_tags
            ),
        },
        batches,
    )


class ConstraintBlockDAGStructureTests(unittest.TestCase):
    def test_all_pair_source_is_half_and_virtual_replay_is_bitwise_exact(self):
        _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, -2.0], [0.5, 0.0]])
        fb = _canonical([[3.0], [0.0]])
        rc, rb = -fc, -fb
        fu = np.asarray([1.25, 2.0], dtype=np.float64)
        ru = np.asarray([0.75, 1.0], dtype=np.float64)
        appended = _append(
            arena,
            frame,
            fc=fc,
            fb=fb,
            fu=fu,
            rc=rc,
            rb=rb,
            ru=ru,
        )
        program = arena.seal(appended.view, final_frame=frame)

        self.assertEqual(
            (
                appended.ranged_rows,
                appended.fallback_pairs,
                program.source_rows,
                program.virtual_facet_rows,
                program.source_nnz,
            ),
            (2, 0, 2, 4, fc.nnz + fb.nnz),
        )
        native = program.native_blocks()[0]
        np.testing.assert_array_equal(
            native.lower.view(np.uint64),
            np.negative(ru).view(np.uint64),
        )
        np.testing.assert_array_equal(native.upper.view(np.uint64), fu.view(np.uint64))
        self.assertEqual(
            native.row_tags,
            ("range:add_materialize:7", "range:add_materialize:7"),
        )

        replay = program.replay_virtual_facets()
        expected_cont = sp.vstack((fc, rc), format="csr")
        expected_bin = sp.vstack((fb, rb), format="csr")
        self.assertEqual(_csr_bits(replay.A_cont), _csr_bits(expected_cont))
        self.assertEqual(_csr_bits(replay.A_bin), _csr_bits(expected_bin))
        np.testing.assert_array_equal(
            replay.upper.view(np.uint64),
            np.concatenate((fu, ru)).view(np.uint64),
        )
        self.assertEqual(
            replay.row_tags,
            (
                "add_materialize:7:forward",
                "add_materialize:7:forward",
                "add_materialize:7:reverse",
                "add_materialize:7:reverse",
            ),
        )
        self.assertEqual(len(set(replay.row_ids)), 4)
        self.assertTrue(all(row.kind == "facet" for row in replay.row_ids))
        self.assertEqual(replay.continuous_ids, frame.continuous_ids)
        self.assertEqual(replay.binary_ids, frame.binary_ids)

        receipt = dict(program.receipt)
        for key in (
            "proof_authority",
            "verdict_authority",
            "authenticity_authority",
            "production_integration",
            "hash_is_identity_authority",
            "triangle_relaxation_called",
            "branch_and_bound_called",
            "backward_called",
            "dual_called",
            "real_model_called",
            "large_model_called",
        ):
            self.assertIs(receipt[key], False)

    def test_mixed_pair_falls_back_per_row_and_replays_original_bits(self):
        _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, 2.0], [3.0, 4.0]])
        fb = _canonical([[0.5], [-0.25]])
        rc, rb = -fc, -fb
        rc.data[-1] = np.nextafter(rc.data[-1], np.inf)
        fu = np.asarray([1.0, 2.0], dtype=np.float64)
        ru = np.asarray([3.0, 4.0], dtype=np.float64)
        appended = _append(
            arena,
            frame,
            fc=fc,
            fb=fb,
            fu=fu,
            rc=rc,
            rb=rb,
            ru=ru,
            layer_id=11,
        )
        program = arena.seal(appended.view, final_frame=frame)
        self.assertEqual(
            (appended.ranged_rows, appended.fallback_pairs, program.source_rows),
            (1, 1, 3),
        )
        native = program.native_blocks()[0]
        self.assertEqual(
            native.row_tags,
            (
                "range:add_materialize:11",
                "add_materialize:11:forward",
                "add_materialize:11:reverse",
            ),
        )
        replay = program.replay_virtual_facets()
        self.assertEqual(
            _csr_bits(replay.A_cont),
            _csr_bits(sp.vstack((fc, rc), format="csr")),
        )
        self.assertEqual(
            _csr_bits(replay.A_bin),
            _csr_bits(sp.vstack((fb, rb), format="csr")),
        )
        np.testing.assert_array_equal(
            replay.upper.view(np.uint64),
            np.concatenate((fu, ru)).view(np.uint64),
        )

    def test_contradictory_signed_pair_remains_an_exact_empty_range(self):
        owner = ExactConstraintOwner()
        owner.allocate_continuous(1)
        frame = owner.frame()
        arena = owner.new_arena()
        forward = _canonical([[1.0]])
        empty_bin = sp.csr_matrix((1, 0), dtype=np.float64)
        appended = arena.append_guarded_band(
            arena.empty_view,
            frame=frame,
            forward_cont=forward,
            forward_bin=empty_bin,
            forward_upper=np.asarray([-1.0], dtype=np.float64),
            reverse_cont=-forward,
            reverse_bin=empty_bin,
            reverse_upper=np.asarray([-1.0], dtype=np.float64),
            layer_id=5,
        )
        program = arena.seal(appended.view, final_frame=frame)
        native = program.native_blocks()[0]
        self.assertEqual((appended.ranged_rows, appended.fallback_pairs), (1, 0))
        self.assertEqual(float(native.lower[0]), 1.0)
        self.assertEqual(float(native.upper[0]), -1.0)
        replay = program.replay_virtual_facets()
        self.assertEqual(replay.upper.tolist(), [-1.0, -1.0])

    def test_views_have_set_union_laws_and_cross_arena_fails_closed(self):
        _owner, frame, arena = _new_arena()
        first = _append(arena, frame, layer_id=1).view
        second = _append(arena, frame, layer_id=2).view
        third = _append(arena, frame, layer_id=3).view
        left = arena.union(arena.union(first, second), third)
        right = arena.union(first, arena.union(second, third))
        self.assertIs(left, right)
        self.assertIs(arena.union(first, first), first)
        self.assertIs(arena.union(first, second), arena.union(second, first))
        self.assertEqual(len(left.block_ids), 3)

        _other_owner, _other_frame, other = _new_arena()
        with self.assertRaises(ConstraintArenaMismatch):
            arena.union(first, other.empty_view)

    def test_hashes_are_only_buckets_and_full_payload_or_view_keys_decide(self):
        _owner, frame, arena = _new_arena()
        with mock.patch.object(core, "_payload_digest", return_value="0" * 64), mock.patch.object(
            core, "_view_digest", return_value="1" * 64
        ):
            first = _append(arena, frame, layer_id=1)
            second = _append(arena, frame, layer_id=2)
            changed_fc = _canonical([[1.0, -2.0], [0.75, 0.0]])
            third = _append(
                arena,
                frame,
                fc=changed_fc,
                rc=-changed_fc,
                layer_id=3,
            )
            first_payload = arena._blocks[first.block.block_id].payload
            second_payload = arena._blocks[second.block.block_id].payload
            third_payload = arena._blocks[third.block.block_id].payload
            self.assertIs(first_payload, second_payload)
            self.assertIsNot(first_payload, third_payload)
            combined = arena.union(first.view, second.view, third.view)
            self.assertEqual(len(combined.block_ids), 3)
        program = arena.seal(combined, final_frame=frame)
        one_payload_bytes = first_payload.payload.numeric_payload_bytes
        self.assertEqual(
            program.numeric_payload_bytes,
            one_payload_bytes + third_payload.payload.numeric_payload_bytes,
        )
        self.assertEqual(program.block_count, 3)


class ConstraintBlockDAGStreamingReplayTests(unittest.TestCase):
    def _multiblock_mixed_program(self):
        _owner, frame, arena = _new_arena()
        first = _append(arena, frame, layer_id=11)
        fc = _canonical([[0.25, 1.0], [-1.5, 0.5], [2.0, -0.75]])
        fb = _canonical([[1.0], [-2.0], [0.5]])
        rc, rb = -fc, -fb
        start = int(rc.indptr[1])
        rc.data[start] = np.nextafter(rc.data[start], np.inf)
        second = _append(
            arena,
            frame,
            view=first.view,
            fc=fc,
            fb=fb,
            fu=np.asarray([2.0, -0.0, 3.5], dtype=np.float64),
            rc=rc,
            rb=rb,
            ru=np.asarray([1.0, 4.0, 2.5], dtype=np.float64),
            layer_id=12,
        )
        return arena.seal(second.view, final_frame=frame)

    def test_chunk_one_boundaries_multiblock_mixed_order_matches_expanded(self):
        program = self._multiblock_mixed_program()
        expanded = program.replay_virtual_facets()
        for max_rows in (1, 3, 5, 256):
            streamed, batches = _collect_stream(program, max_rows)
            self.assertIsNotNone(streamed)
            self.assertEqual(
                [batch.row_offset for batch in batches],
                list(
                    range(0, program.virtual_facet_rows, max_rows)
                ),
            )
            self.assertTrue(
                all(1 <= batch.row_count <= max_rows for batch in batches)
            )
            self.assertTrue(
                all(
                    batch.total_rows == program.virtual_facet_rows
                    and batch.bytes_backed
                    and batch.proof_authority is False
                    and batch.verdict_authority is False
                    and batch.continuous_ids == program.continuous_ids
                    and batch.binary_ids == program.binary_ids
                    for batch in batches
                )
            )
            self.assertEqual(
                _csr_bits(streamed["A_cont"]), _csr_bits(expanded.A_cont)
            )
            self.assertEqual(
                _csr_bits(streamed["A_bin"]), _csr_bits(expanded.A_bin)
            )
            self.assertEqual(
                streamed["upper"].view(np.uint64).tobytes(),
                expanded.upper.view(np.uint64).tobytes(),
            )
            self.assertEqual(streamed["row_ids"], expanded.row_ids)
            self.assertEqual(streamed["row_tags"], expanded.row_tags)

        first_batch = next(
            core.iter_virtual_facet_batches(program, max_rows=1)
        )
        with self.assertRaises(ValueError):
            first_batch.A_cont.data[0] = 7.0
        with self.assertRaises(ValueError):
            first_batch.upper[0] = 7.0
        with self.assertRaises(AttributeError):
            first_batch._row_offset = 99
        forged_id = first_batch.row_ids[0]
        original_value = forged_id.value
        object.__setattr__(forged_id, "value", original_value + 1)
        self.assertEqual(first_batch.row_ids[0].value, original_value)

    def test_stream_validates_once_avoids_full_builders_and_captures_source(self):
        program = self._multiblock_mixed_program()
        expanded = program.replay_virtual_facets()
        original_blocks = program._blocks
        original_frozen = original_blocks[0].payload.payload.A_cont
        original_data_bytes = original_frozen.data_bytes
        original_handle = original_blocks[0].forward_facets[0]
        original_tag = original_handle.tag
        with mock.patch.object(
            core,
            "_validated_program_record",
            wraps=core._validated_program_record,
        ) as validate, mock.patch.object(
            core,
            "replay_virtual_facets",
            side_effect=AssertionError("full replay forbidden"),
        ), mock.patch.object(
            core.sp,
            "vstack",
            side_effect=AssertionError("full vstack forbidden"),
        ), mock.patch.object(
            core, "_payload_digest", return_value="0" * 64
        ), mock.patch.object(core, "_view_digest", return_value="1" * 64):
            iterator = core.iter_virtual_facet_batches(program, max_rows=2)
            object.__setattr__(program, "_blocks", ())
            object.__setattr__(
                original_frozen,
                "data_bytes",
                np.full(original_frozen.nnz, 9.0, dtype=np.float64).tobytes(),
            )
            object.__setattr__(original_handle, "tag", "mutated:after:capture")
            batches = list(iterator)
        object.__setattr__(program, "_blocks", original_blocks)
        object.__setattr__(original_frozen, "data_bytes", original_data_bytes)
        object.__setattr__(original_handle, "tag", original_tag)
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(
            sum(batch.row_count for batch in batches),
            expanded.upper.size,
        )
        self.assertEqual(
            np.concatenate([batch.upper for batch in batches])
            .view(np.uint64)
            .tobytes(),
            expanded.upper.view(np.uint64).tobytes(),
        )
        self.assertEqual(
            _csr_bits(
                sp.vstack([batch.A_cont for batch in batches], format="csr")
            ),
            _csr_bits(expanded.A_cont),
        )
        self.assertEqual(
            tuple(tag for batch in batches for tag in batch.row_tags),
            expanded.row_tags,
        )

        for invalid in (True, 0, 257, 1.0, np.int64(1)):
            with self.assertRaises(ConstraintBlockDAGCandidateError):
                core.iter_virtual_facet_batches(program, max_rows=invalid)

    def test_close_interrupt_aba_and_parallel_readers_release_capture(self):
        program = self._multiblock_mixed_program()
        iterator = core.iter_virtual_facet_batches(program, max_rows=1)
        capture = iterator._capture
        capture_ref = weakref.ref(capture)
        next(iterator)
        iterator.close()
        self.assertTrue(iterator.closed)
        self.assertIsNone(iterator._capture)
        del capture
        gc.collect()
        self.assertIsNone(capture_ref())

        for failure in (
            RuntimeError("consumer_build_failure"),
            KeyboardInterrupt(),
            SystemExit(9),
        ):
            failed = core.iter_virtual_facet_batches(program, max_rows=2)
            with mock.patch.object(core, "_assemble_rows", side_effect=failure):
                with self.assertRaises(type(failure)):
                    next(failed)
            self.assertTrue(failed.closed)
            self.assertIsNone(failed._capture)

        record = core._PROGRAM_REGISTRY[id(program)]
        callback = record[0].__callback__

        class Replacement:
            pass

        replacement_object = Replacement()
        replacement = (weakref.ref(replacement_object), (), ())
        with core._PROGRAM_REGISTRY_LOCK:
            core._PROGRAM_REGISTRY[id(program)] = replacement
        callback(record[0])
        self.assertIs(core._PROGRAM_REGISTRY[id(program)], replacement)
        with core._PROGRAM_REGISTRY_LOCK:
            core._PROGRAM_REGISTRY[id(program)] = record

        results = []
        failures = []

        def consume(max_rows):
            try:
                streamed, _batches = _collect_stream(program, max_rows)
                results.append(
                    (
                        _csr_bits(streamed["A_cont"]),
                        _csr_bits(streamed["A_bin"]),
                        streamed["upper"].view(np.uint64).tobytes(),
                        streamed["row_ids"],
                        streamed["row_tags"],
                    )
                )
            except BaseException as exc:  # pragma: no cover - diagnostic.
                failures.append(exc)

        workers = [
            threading.Thread(target=consume, args=(value,))
            for value in (1, 4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertFalse(failures)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])


class ConstraintBlockDAGOwnerAndHostileTests(unittest.TestCase):
    def test_imported_floor_and_typed_registry_prevent_reuse_or_kind_collision(self):
        high = 9_000_000
        imported = ExactConstraintOwner(
            imported_continuous_ids=(high,), imported_binary_ids=(high + 1,)
        )
        fresh = imported.allocate_continuous(2)
        self.assertGreater(fresh[0].value, high + 1)
        following = ExactConstraintOwner().allocate_binary(1)[0]
        self.assertGreater(following.value, fresh[-1].value)
        self.assertEqual(following.kind, "binary")
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            ExactConstraintOwner(imported_binary_ids=(fresh[0].value,))
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            ExactConstraintOwner(imported_continuous_ids=(high,))
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            ExactConstraintOwner(
                imported_continuous_ids=(17,), imported_binary_ids=(17,)
            )
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            StableFactorID("row", 0)

    def test_input_and_returned_csr_mutation_cannot_change_source(self):
        _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, -2.0], [0.5, 0.0]])
        fb = _canonical([[3.0], [0.0]])
        rc, rb = -fc, -fb
        fu = np.asarray([1.25, 2.0], dtype=np.float64)
        ru = np.asarray([0.75, 1.0], dtype=np.float64)
        expected = (
            _csr_bits(sp.vstack((fc, rc), format="csr")),
            _csr_bits(sp.vstack((fb, rb), format="csr")),
            np.concatenate((fu, ru)).view(np.uint64).tobytes(),
        )
        appended = _append(
            arena,
            frame,
            fc=fc,
            fb=fb,
            fu=fu,
            rc=rc,
            rb=rb,
            ru=ru,
        )
        fc.data[:] = 99.0
        fb.data[:] = -77.0
        rc.data[:] = 55.0
        rb.data[:] = 44.0
        fu[:] = -100.0
        ru[:] = 100.0
        program = arena.seal(appended.view, final_frame=frame)
        replay = program.replay_virtual_facets()
        self.assertEqual(_csr_bits(replay.A_cont), expected[0])
        self.assertEqual(_csr_bits(replay.A_bin), expected[1])
        self.assertEqual(replay.upper.view(np.uint64).tobytes(), expected[2])

        native = program.native_blocks()[0]
        try:
            native.A_cont.data[0] = np.nextafter(native.A_cont.data[0], np.inf)
        except ValueError:
            pass
        self.assertEqual(
            _csr_bits(program.replay_virtual_facets().A_cont), expected[0]
        )

    def test_private_snapshot_is_validated_after_detaching_from_caller(self):
        vector = np.ones(4, dtype=np.float64)
        matrix = _canonical([[1.0, 2.0]])
        real_array = np.array

        def poison_float_copy(value, *args, **kwargs):
            copied = real_array(value, *args, **kwargs)
            if kwargs.get("copy") is True and copied.dtype == np.float64:
                copied.flat[0] = np.nan
            return copied

        with mock.patch.object(core.np, "array", side_effect=poison_float_copy):
            with self.assertRaises(ConstraintBlockDAGCandidateError):
                core._FrozenFloat64Bytes.from_vector(
                    vector, name="hostile_vector", finite=True
                )
            with self.assertRaises(ConstraintBlockDAGCandidateError):
                core._FrozenCSRBytes.from_csr(matrix, name="hostile_csr")

    def test_mutable_owner_and_arena_are_thread_confined(self):
        owner, frame, arena = _new_arena()
        failures = []

        def try_mutations():
            for operation in (
                lambda: owner.allocate_continuous(1),
                lambda: owner.frame(),
                lambda: arena.union(arena.empty_view),
                lambda: _append(arena, frame),
                lambda: arena.seal(arena.empty_view, final_frame=frame),
            ):
                try:
                    operation()
                except ConstraintBlockDAGCandidateError:
                    failures.append(True)
                else:
                    failures.append(False)

        worker = threading.Thread(target=try_mutations)
        worker.start()
        worker.join()
        self.assertEqual(failures, [True] * 5)
        self.assertFalse(owner._sealed)
        self.assertEqual(len(owner._continuous), len(frame.continuous_ids))

    def test_forged_or_mutated_frame_view_and_program_fail_closed(self):
        _owner, frame, arena = _new_arena()
        object.__setattr__(frame, "_version", frame._version + 1)
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            _append(arena, frame)

        _owner2, frame2, arena2 = _new_arena()
        appended = _append(arena2, frame2)
        object.__setattr__(appended.view, "block_ids", ())
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            arena2.union(appended.view)

        _owner_flip, frame_flip, arena_flip = _new_arena()
        appended_flip = _append(arena_flip, frame_flip)

        class Flip:
            def __init__(self, original):
                self.original = original
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(self.original if self.iterations <= 3 else ())

        object.__setattr__(
            appended_flip.view,
            "block_ids",
            Flip(appended_flip.view.block_ids),
        )
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            arena_flip.union(appended_flip.view)

        _owner3, frame3, arena3 = _new_arena()
        appended3 = _append(arena3, frame3)
        program = arena3.seal(appended3.view, final_frame=frame3)
        program.__dict__["proof_authority"] = True
        program.__dict__["verdict_authority"] = True
        self.assertIs(program.proof_authority, False)
        self.assertIs(program.verdict_authority, False)
        object.__setattr__(program, "_schema", "forged")
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            program.native_blocks()

        _owner4, frame4, arena4 = _new_arena()
        appended4 = _append(arena4, frame4)
        program4 = arena4.seal(appended4.view, final_frame=frame4)
        live_payload = program4._blocks[0].payload.payload
        object.__setattr__(
            live_payload.lower,
            "raw",
            bytes(np.zeros(live_payload.lower.length, dtype=np.float64).tobytes()),
        )
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            program4.replay_virtual_facets()

        _owner5, frame5, arena5 = _new_arena()
        first5 = _append(arena5, frame5, fc=_canonical([[1.0, 0.0], [0.5, 0.0]]))
        program5 = arena5.seal(first5.view, final_frame=frame5)
        honest_block = program5._blocks[0]

        class EvilBlock:
            def __init__(self, source):
                self.payload = source.payload
                self.source_row_handles = source.source_row_handles
                self.forward_facets = source.forward_facets
                self.reverse_facets = source.reverse_facets

            def key(self):
                return honest_block.key()

        object.__setattr__(program5, "_blocks", (EvilBlock(honest_block),))
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            program5.native_blocks()

        _owner6, frame6, arena6 = _new_arena()
        first6 = _append(arena6, frame6)
        program6 = arena6.seal(first6.view, final_frame=frame6)

        class LiarBytes(bytes):
            def __eq__(self, other):
                return True

            def __ne__(self, other):
                return False

        frozen = program6._blocks[0].payload.payload.A_cont
        replacement = LiarBytes(
            np.full(frozen.nnz, 2.0, dtype=np.float64).tobytes()
        )
        object.__setattr__(frozen, "data_bytes", replacement)
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            program6.replay_virtual_facets()

    def test_noncanonical_explicit_zero_and_wrong_family_are_rejected(self):
        _owner, frame, arena = _new_arena()
        duplicate = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2, 2], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        zero_bin = sp.csr_matrix((2, 1), dtype=np.float64)
        upper = np.ones(2, dtype=np.float64)
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            arena.append_guarded_band(
                arena.empty_view,
                frame=frame,
                forward_cont=duplicate,
                forward_bin=zero_bin,
                forward_upper=upper,
                reverse_cont=-duplicate,
                reverse_bin=zero_bin,
                reverse_upper=upper,
                layer_id=1,
            )
        explicit_zero = sp.csr_matrix(
            (
                np.asarray([0.0], dtype=np.float64),
                np.asarray([0], dtype=np.int32),
                np.asarray([0, 1, 1], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            arena.append_guarded_band(
                arena.empty_view,
                frame=frame,
                forward_cont=explicit_zero,
                forward_bin=zero_bin,
                forward_upper=upper,
                reverse_cont=-explicit_zero,
                reverse_bin=zero_bin,
                reverse_upper=upper,
                layer_id=1,
            )
        good = _canonical([[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ConstraintBlockDAGCandidateError):
            arena.append_guarded_band(
                arena.empty_view,
                frame=frame,
                forward_cont=good,
                forward_bin=zero_bin,
                forward_upper=upper,
                reverse_cont=-good,
                reverse_bin=zero_bin,
                reverse_upper=upper,
                layer_id=1,
                family="caller_supplied_tag",
            )


class ConstraintBlockDAGSemanticTests(unittest.TestCase):
    def test_fraction_membership_matches_virtual_facets_for_mixed_rows(self):
        rng = random.Random(20260809)
        rows, n_cont, n_bin = 7, 5, 2
        dense_cont = np.zeros((rows, n_cont), dtype=np.float64)
        dense_bin = np.zeros((rows, n_bin), dtype=np.float64)
        choices = (-1.5, -0.75, -0.25, 0.125, 0.5, 1.25)
        for row in range(rows):
            for col in rng.sample(range(n_cont), 3):
                dense_cont[row, col] = rng.choice(choices)
            dense_bin[row, row % n_bin] = rng.choice(choices)
        fc = sp.csr_matrix(dense_cont)
        fb = sp.csr_matrix(dense_bin)
        rc, rb = -fc, -fb
        # Force two exact fallback rows without changing any other row.
        for row in (2, 5):
            position = int(rc.indptr[row])
            rc.data[position] = np.nextafter(rc.data[position], np.inf)
        fu = np.asarray([1.0 + row / 8.0 for row in range(rows)], dtype=np.float64)
        ru = np.asarray([0.75 + row / 16.0 for row in range(rows)], dtype=np.float64)
        _owner, frame, arena = _new_arena(n_cont, n_bin)
        appended = _append(
            arena,
            frame,
            fc=fc,
            fb=fb,
            fu=fu,
            rc=rc,
            rb=rb,
            ru=ru,
        )
        program = arena.seal(appended.view, final_frame=frame)
        self.assertEqual((appended.ranged_rows, appended.fallback_pairs), (5, 2))
        virtual = program.replay_virtual_facets()
        native = program.native_blocks()[0]

        grid = tuple(Fraction(value, 4) for value in range(-4, 5))
        for _sample in range(500):
            continuous = tuple(rng.choice(grid) for _ in range(n_cont))
            binary = tuple(rng.choice((Fraction(-1), Fraction(1))) for _ in range(n_bin))
            virtual_ok = True
            for row in range(virtual.A_cont.shape[0]):
                value = _exact_row_value(virtual.A_cont, row, continuous)
                value += _exact_row_value(virtual.A_bin, row, binary)
                if value > Fraction.from_float(float(virtual.upper[row])):
                    virtual_ok = False
                    break
            native_ok = True
            for row in range(native.A_cont.shape[0]):
                value = _exact_row_value(native.A_cont, row, continuous)
                value += _exact_row_value(native.A_bin, row, binary)
                lower = float(native.lower[row])
                upper = Fraction.from_float(float(native.upper[row]))
                if (not math.isinf(lower) and value < Fraction.from_float(lower)) or value > upper:
                    native_ok = False
                    break
            self.assertEqual(native_ok, virtual_ok)

    @unittest.skipIf(highspy is None, "highspy is unavailable")
    def test_highs_small_toy_range_and_virtual_source_have_same_optimum(self):
        owner = ExactConstraintOwner()
        owner.allocate_continuous(1)
        frame = owner.frame()
        arena = owner.new_arena()
        fc = _canonical([[1.0]])
        fb = sp.csr_matrix((1, 0), dtype=np.float64)
        result = arena.append_guarded_band(
            arena.empty_view,
            frame=frame,
            forward_cont=fc,
            forward_bin=fb,
            forward_upper=np.asarray([0.75], dtype=np.float64),
            reverse_cont=-fc,
            reverse_bin=fb,
            reverse_upper=np.asarray([0.5], dtype=np.float64),
            layer_id=4,
        )
        program = arena.seal(result.view, final_frame=frame)

        def solve(A, lower, upper):
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("presolve", "on")
            status = h.addCols(
                1,
                np.asarray([-1.0], dtype=np.float64),
                np.asarray([-1.0], dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
                0,
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.float64),
            )
            self.assertEqual(status, highspy.HighsStatus.kOk)
            matrix = A.tocsr()
            status = h.addRows(
                matrix.shape[0],
                np.asarray(lower, dtype=np.float64),
                np.asarray(upper, dtype=np.float64),
                matrix.nnz,
                matrix.indptr.astype(np.int32),
                matrix.indices.astype(np.int32),
                matrix.data.astype(np.float64),
            )
            self.assertEqual(status, highspy.HighsStatus.kOk)
            self.assertEqual(h.run(), highspy.HighsStatus.kOk)
            self.assertEqual(h.getModelStatus(), highspy.HighsModelStatus.kOptimal)
            solution = h.getSolution()
            value = float(solution.col_value[0])
            objective = float(h.getObjectiveValue())
            h.clear()
            return value, objective

        native = program.native_blocks()[0]
        native_value, native_objective = solve(
            native.A_cont, native.lower, native.upper
        )
        virtual = program.replay_virtual_facets()
        virtual_value, virtual_objective = solve(
            virtual.A_cont,
            np.full(virtual.upper.size, -np.inf, dtype=np.float64),
            virtual.upper,
        )
        self.assertAlmostEqual(native_value, 0.75, places=12)
        self.assertAlmostEqual(virtual_value, 0.75, places=12)
        self.assertAlmostEqual(native_objective, virtual_objective, places=12)


class ConstraintBlockDAGPerformanceTests(unittest.TestCase):
    def test_bounded_c89_ratio_gate_is_measured_and_honest(self):
        receipt = benchmark_bounded_c89_ratio(
            scale_divisor=40,
            warmups=2,
            repeats=11,
        )
        expected_pass = bool(
            receipt["range_speedup"] >= receipt["speed_gate"]
            and receipt["payload_ratio"] <= receipt["payload_ratio_gate"]
            and receipt["fallback_slowdown"]
            <= receipt["fallback_slowdown_gate"]
        )
        self.assertEqual(receipt["status"], "passed" if expected_pass else "closed")
        self.assertEqual(bool(receipt["closed_reasons"]), not expected_pass)
        self.assertLessEqual(receipt["payload_ratio"], 0.60)
        self.assertEqual(
            receipt["baseline_source_rows"],
            2 * receipt["range_source_rows"],
        )
        self.assertEqual(
            receipt["baseline_source_nnz"],
            2 * receipt["range_source_nnz"],
        )
        self.assertEqual(
            receipt["virtual_facet_rows"],
            receipt["baseline_source_rows"],
        )
        for key in (
            "proof_authority",
            "verdict_authority",
            "production_integration",
            "rss_measured",
            "full_promotion_gate",
            "promotion_authority",
            "production_baseline",
            "real_model_allowed",
            "large_model_allowed",
            "triangle_relaxation_called",
            "branch_and_bound_called",
            "backward_called",
            "dual_called",
        ):
            self.assertIs(receipt[key], False)
        self.assertEqual(
            receipt["baseline_kind"],
            "candidate_forced_dual_le_same_pipeline",
        )


if __name__ == "__main__":
    unittest.main()
