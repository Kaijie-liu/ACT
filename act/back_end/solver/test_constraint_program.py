#!/usr/bin/env python3
# ===- test_constraint_program.py - production source-program gates -----===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------===#
"""Bounded, offline gates for the exact production constraint program."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import gc
import inspect
import random
import sys
import threading
import unittest
from unittest import mock
import weakref

import numpy as np
import scipy.optimize
import scipy.sparse as sp

from act.back_end.solver import constraint_program as core
from act.back_end.solver.constraint_program import (
    ConstraintArenaMismatch,
    ConstraintFamily,
    ConstraintProgramError,
    ConstraintProgramOwner,
    ConstraintTransactionError,
    ExternalAllocatorContractError,
    ExternalFactorAllocatorAdapter,
    FactorKind,
)


class _FakeAllocator:
    def __init__(self, start: int = 0) -> None:
        self.next_id = int(start)
        self.continuous = []
        self.binary = []
        self.cont_calls = 0
        self.bin_calls = 0

    def allocate_continuous(self, count: int):
        self.cont_calls += 1
        result = tuple(range(self.next_id, self.next_id + count))
        self.next_id += count
        self.continuous.extend(result)
        return result

    def allocate_binary(self, count: int):
        self.bin_calls += 1
        result = tuple(range(self.next_id, self.next_id + count))
        self.next_id += count
        self.binary.extend(result)
        return result

    def snapshot(self):
        return tuple(self.continuous), tuple(self.binary)


def _bind(value: _FakeAllocator):
    return ExternalFactorAllocatorAdapter.bind(
        value,
        allocate_continuous=value.allocate_continuous,
        allocate_binary=value.allocate_binary,
        live_ids_snapshot=value.snapshot,
    )


def _new_arena(n_cont: int = 2, n_bin: int = 1):
    allocator = _FakeAllocator()
    adapter = _bind(allocator)
    owner = ConstraintProgramOwner(adapter)
    owner.allocate_continuous(n_cont)
    owner.allocate_binary(n_bin)
    frame = owner.frame()
    arena = owner.new_arena()
    return allocator, adapter, owner, frame, arena


def _canonical(rows, *, columns=None):
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2:
        raise AssertionError("test input must be rank two")
    shape = array.shape if columns is None else (array.shape[0], int(columns))
    result = sp.csr_matrix(array, shape=shape, dtype=np.float64)
    result.eliminate_zeros()
    result.sort_indices()
    return result


def _csr_bits(value: sp.csr_matrix):
    value = value.tocsr()
    return (
        tuple(int(item) for item in value.shape),
        np.asarray(value.indptr, dtype=np.int64).tobytes(),
        np.asarray(value.indices, dtype=np.int64).tobytes(),
        np.asarray(value.data, dtype=np.float64).view(np.uint64).tobytes(),
    )


def _collect(program, *, native: bool, max_rows: int):
    iterator = (
        program.iter_native_batches(max_rows=max_rows)
        if native
        else program.iter_legacy_facet_batches(max_rows=max_rows)
    )
    batches = list(iterator)
    if not batches:
        return {
            "A_cont": sp.csr_matrix((0, len(program.continuous_ids)), dtype=np.float64),
            "A_bin": sp.csr_matrix((0, len(program.binary_ids)), dtype=np.float64),
            "upper": np.empty(0, dtype=np.float64),
            "lower": np.empty(0, dtype=np.float64) if native else None,
            "row_ids": (),
            "row_tags": (),
            "block_ids": (),
            "append_ordinals": (),
            "batches": (),
        }
    return {
        "A_cont": sp.vstack([item.A_cont for item in batches], format="csr"),
        "A_bin": sp.vstack([item.A_bin for item in batches], format="csr"),
        "upper": np.concatenate([item.upper for item in batches]),
        "lower": np.concatenate([item.lower for item in batches]) if native else None,
        "row_ids": tuple(value for item in batches for value in item.row_ids),
        "row_tags": tuple(value for item in batches for value in item.row_tags),
        "block_ids": tuple(value for item in batches for value in item.block_ids),
        "append_ordinals": tuple(
            value for item in batches for value in item.append_ordinals
        ),
        "batches": tuple(batches),
    }


def _append_band(
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
        layer_id=layer_id,
        family=ConstraintFamily.ADD_MATERIALIZE,
    )


def _row_value(matrix: sp.csr_matrix, row: int, values):
    result = Fraction(0)
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    for position in range(start, stop):
        result += Fraction.from_float(float(matrix.data[position])) * values[
            int(matrix.indices[position])
        ]
    return result


class ConstraintProgramAllocatorTests(unittest.TestCase):
    def test_adapter_captures_callbacks_and_owner_only_delegates(self):
        allocator = _FakeAllocator(start=11)
        adapter = _bind(allocator)
        captured = allocator.allocate_continuous

        def rebound(_count):
            raise AssertionError("public callback field was reread")

        allocator.allocate_continuous = rebound
        owner = ConstraintProgramOwner(adapter)
        allocated = owner.allocate_continuous(3)
        self.assertEqual(tuple(item.raw_id for item in allocated), (11, 12, 13))
        self.assertEqual(allocator.cont_calls, 1)
        self.assertIsNot(captured, rebound)
        self.assertTrue(all(item.kind is FactorKind.CONTINUOUS for item in allocated))
        self.assertTrue(
            all(item.namespace_identity is adapter.namespace_identity for item in allocated)
        )

        binary = owner.allocate_binary(2)
        self.assertEqual(tuple(item.raw_id for item in binary), (14, 15))
        frame = owner.frame()
        self.assertEqual(tuple(item.raw_id for item in frame.continuous_ids), (11, 12, 13))
        self.assertEqual(tuple(item.raw_id for item in frame.binary_ids), (14, 15))

    def test_external_snapshot_growth_is_bound_without_parallel_factor_counter(self):
        allocator = _FakeAllocator(start=1000)
        adapter = _bind(allocator)
        owner = ConstraintProgramOwner(adapter)
        first = owner.allocate_continuous(1)
        externally_added = allocator.allocate_continuous(2)
        frame = owner.frame()
        self.assertEqual(
            tuple(item.raw_id for item in frame.continuous_ids),
            (1000, 1001, 1002),
        )
        self.assertEqual(externally_added, (1001, 1002))
        self.assertFalse(
            any(
                name.startswith("_NEXT_") and "FACTOR" in name
                for name in core.__dict__
            )
        )

    def test_bad_allocator_return_or_baseexception_poison_owner_without_retry(self):
        class Bad(_FakeAllocator):
            def allocate_continuous(self, count):
                self.cont_calls += 1
                result = tuple(range(self.next_id, self.next_id + count))
                self.next_id += count
                self.continuous.extend(result)
                return list(result)

        bad = Bad()
        owner = ConstraintProgramOwner(_bind(bad))
        with self.assertRaises(ExternalAllocatorContractError):
            owner.allocate_continuous(2)
        self.assertEqual(bad.cont_calls, 1)
        with self.assertRaises(ConstraintProgramError):
            owner.allocate_continuous(2)
        self.assertEqual(bad.cont_calls, 1)

        class Interrupted(_FakeAllocator):
            def allocate_continuous(self, count):
                self.cont_calls += 1
                self.next_id += count
                raise SystemExit("after external burn")

        interrupted = Interrupted()
        owner2 = ConstraintProgramOwner(_bind(interrupted))
        with self.assertRaises(SystemExit):
            owner2.allocate_continuous(3)
        with self.assertRaises(ConstraintProgramError):
            owner2.frame()
        self.assertEqual(interrupted.cont_calls, 1)

    def test_reuse_wrong_kind_and_nonappend_snapshot_fail_closed(self):
        class Reuse(_FakeAllocator):
            def allocate_continuous(self, count):
                self.cont_calls += 1
                if not self.continuous:
                    self.continuous.append(8)
                return (8,) * count

        reuse = Reuse()
        owner = ConstraintProgramOwner(_bind(reuse))
        owner.allocate_continuous(1)
        with self.assertRaises(ExternalAllocatorContractError):
            owner.allocate_continuous(1)

        allocator = _FakeAllocator()
        owner2 = ConstraintProgramOwner(_bind(allocator))
        owner2.allocate_continuous(2)
        allocator.continuous.reverse()
        with self.assertRaises(ExternalAllocatorContractError):
            owner2.frame()

    def test_public_self_signing_and_mutable_subclasses_are_rejected(self):
        with self.assertRaises(TypeError):
            core.ExternalFactorAllocatorAdapter()
        with self.assertRaises(TypeError):
            core.ExternalFactorID()
        with self.assertRaises(TypeError):
            core.FactorFrame()
        with self.assertRaises(TypeError):
            core.ConstraintArena()
        with self.assertRaises(TypeError):
            core.ConstraintView()
        with self.assertRaises(TypeError):
            core.PreparedAppend()
        with self.assertRaises(TypeError):
            core.ConstraintProgram()

        class AdapterSubclass(core.ExternalFactorAllocatorAdapter):
            pass

        allocator = _FakeAllocator()
        with self.assertRaises(TypeError):
            AdapterSubclass.bind(
                allocator,
                allocate_continuous=allocator.allocate_continuous,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=allocator.snapshot,
            )

    def test_adapter_and_owner_registry_gc_cleanup(self):
        allocator = _FakeAllocator()
        adapter = _bind(allocator)
        adapter_id = id(adapter)
        adapter_ref = weakref.ref(adapter)
        del adapter
        gc.collect()
        self.assertIsNone(adapter_ref())
        self.assertNotIn(adapter_id, core._ADAPTER_REGISTRY)

        allocator2 = _FakeAllocator()
        adapter2 = _bind(allocator2)
        owner = ConstraintProgramOwner(adapter2)
        owner_id = id(owner)
        owner_ref = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertNotIn(owner_id, core._OWNER_REGISTRY)

    def test_captured_allocator_record_callback_rebinding_fails_closed(self):
        for field in (
            "allocator_obj",
            "allocate_continuous",
            "allocate_binary",
            "live_ids_snapshot",
            "namespace",
            "thread",
        ):
            with self.subTest(field=field):
                allocator = _FakeAllocator()
                adapter = _bind(allocator)
                owner = ConstraintProgramOwner(adapter)
                record = core._adapter_record(adapter)
                called = []

                def evil(*_args):
                    called.append(True)
                    return (999,)

                replacement = (
                    object()
                    if field in {"allocator_obj", "namespace", "thread"}
                    else evil
                )
                object.__setattr__(record, field, replacement)
                with self.assertRaises(ExternalAllocatorContractError):
                    owner.allocate_continuous(1)
                self.assertEqual(called, [])

    def test_callback_time_adapter_rebinding_cannot_cross_seal(self):
        allocator = _FakeAllocator()
        holder = {"record": None, "armed": False}

        def snapshot():
            if holder["armed"]:
                holder["armed"] = False
                object.__setattr__(
                    holder["record"],
                    "allocate_continuous",
                    lambda _count: (999,),
                )
            return allocator.snapshot()

        adapter = ExternalFactorAllocatorAdapter.bind(
            allocator,
            allocate_continuous=allocator.allocate_continuous,
            allocate_binary=allocator.allocate_binary,
            live_ids_snapshot=snapshot,
        )
        owner = ConstraintProgramOwner(adapter)
        owner.allocate_continuous(1)
        frame = owner.frame()
        arena = owner.new_arena()
        holder["record"] = core._adapter_record(adapter)
        holder["armed"] = True
        programs_before = len(core._PROGRAM_REGISTRY)
        with self.assertRaises(ExternalAllocatorContractError):
            arena.seal(arena.empty_view, final_frame=frame)
        self.assertEqual(len(core._PROGRAM_REGISTRY), programs_before)
        self.assertFalse(core._ARENA_REGISTRY[id(arena)].state.sealed)

    def test_unobserved_external_id_cannot_be_reissued_as_fresh(self):
        allocator = _FakeAllocator()
        real_allocate = allocator.allocate_continuous

        def stale_allocate(count):
            allocator.cont_calls += 1
            return (0,) * count

        adapter = ExternalFactorAllocatorAdapter.bind(
            allocator,
            allocate_continuous=stale_allocate,
            allocate_binary=allocator.allocate_binary,
            live_ids_snapshot=allocator.snapshot,
        )
        owner = ConstraintProgramOwner(adapter)
        self.assertEqual(real_allocate(1), (0,))
        with self.assertRaises(ExternalAllocatorContractError):
            owner.allocate_continuous(1)
        self.assertEqual(allocator.snapshot(), ((0,), ()))
        with self.assertRaises(ConstraintProgramError):
            owner.frame()

    def test_allocator_namespace_lease_blocks_aba_without_process_leak(self):
        allocator = _FakeAllocator()
        adapter = _bind(allocator)
        owner = ConstraintProgramOwner(adapter)
        owner.allocate_continuous(1)
        frame = owner.frame()
        arena = owner.new_arena()
        result = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="lease",
        )
        program = arena.seal(result.view, final_frame=frame)
        namespace_reference = weakref.ref(adapter.namespace_identity)
        binding_key = id(allocator)
        del result, arena, frame, owner, adapter
        gc.collect()
        self.assertIsNotNone(namespace_reference())
        self.assertIn(binding_key, core._ALLOCATOR_BINDINGS)
        with self.assertRaises(ExternalAllocatorContractError):
            _bind(allocator)

        del program
        gc.collect()
        self.assertIsNone(namespace_reference())
        self.assertNotIn(binding_key, core._ALLOCATOR_BINDINGS)
        rebound = _bind(allocator)
        self.assertIsNotNone(rebound.namespace_identity)

    def test_all_captured_callbacks_reject_swallowed_reentrant_mutation(self):
        def make_context(*, allocation_callback=False):
            allocator = _FakeAllocator()
            holder = {
                "action": None,
                "caught": [],
                "snapshot_calls": 0,
                "skip_actions": 0,
            }

            def invoke_action():
                action = holder["action"]
                if action is not None:
                    holder["action"] = None
                    try:
                        action()
                    except BaseException as error:
                        holder["caught"].append(error)

            def snapshot():
                holder["snapshot_calls"] += 1
                if not allocation_callback:
                    if holder["action"] is not None and holder["skip_actions"]:
                        holder["skip_actions"] -= 1
                    else:
                        invoke_action()
                return allocator.snapshot()

            real_allocate = allocator.allocate_continuous

            def allocate(count):
                if allocation_callback:
                    invoke_action()
                return real_allocate(count)

            adapter = ExternalFactorAllocatorAdapter.bind(
                allocator,
                allocate_continuous=allocate,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=snapshot,
            )
            owner = ConstraintProgramOwner(adapter)
            owner.allocate_continuous(1)
            frame = owner.frame()
            arena = owner.new_arena()
            return allocator, holder, owner, frame, arena

        for action_name in ("allocate", "frame", "prepare", "seal"):
            with self.subTest(callback="snapshot", action=action_name):
                allocator, holder, owner, frame, arena = make_context()
                empty = arena.empty_view
                kwargs = dict(
                    frame=frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="reentrant",
                )
                actions = {
                    "allocate": lambda: owner.allocate_continuous(1),
                    "frame": owner.frame,
                    "prepare": lambda: arena.prepare_le(empty, **kwargs),
                    "seal": lambda: arena.seal(empty, final_frame=frame),
                }
                holder["action"] = actions[action_name]
                calls_before = allocator.cont_calls
                with self.assertRaises(ExternalAllocatorContractError):
                    owner.frame()
                self.assertEqual(len(holder["caught"]), 1)
                self.assertIsInstance(
                    holder["caught"][0], ConstraintProgramError
                )
                self.assertEqual(allocator.cont_calls, calls_before)
                arena_state = core._arena_state(arena)
                self.assertEqual(arena_state.pending, [])
                self.assertEqual(arena_state.prepared, {})
                self.assertFalse(arena_state.sealed)

        allocator, holder, owner, frame, arena = make_context(
            allocation_callback=True
        )
        holder["action"] = owner.frame
        calls_before = allocator.cont_calls
        with self.assertRaises(ExternalAllocatorContractError):
            owner.allocate_continuous(1)
        self.assertEqual(len(holder["caught"]), 1)
        self.assertEqual(allocator.cont_calls, calls_before + 1)
        self.assertFalse(core._arena_state(arena).sealed)

        allocator, holder, owner, frame, arena = make_context()
        empty = arena.empty_view
        holder["action"] = lambda: arena.seal(empty, final_frame=frame)
        calls_before = allocator.cont_calls
        with self.assertRaises(ExternalAllocatorContractError):
            owner.allocate_continuous(1)
        self.assertEqual(len(holder["caught"]), 1)
        # The pre-snapshot guard rejects before the allocation callback.
        self.assertEqual(allocator.cont_calls, calls_before)
        self.assertFalse(core._arena_state(arena).sealed)

        allocator, holder, owner, frame, arena = make_context()
        empty = arena.empty_view
        holder["action"] = lambda: arena.seal(empty, final_frame=frame)
        holder["skip_actions"] = 1
        calls_before = allocator.cont_calls
        with self.assertRaises(ExternalAllocatorContractError):
            owner.allocate_continuous(1)
        self.assertEqual(len(holder["caught"]), 1)
        # The allocator call ran and burned an ID; the post-snapshot callback
        # still cannot publish a stale-frame program.
        self.assertEqual(allocator.cont_calls, calls_before + 1)
        self.assertFalse(core._arena_state(arena).sealed)

    def test_seal_callback_cannot_prepare_or_recursively_seal(self):
        for nested in ("prepare", "seal"):
            with self.subTest(nested=nested):
                allocator = _FakeAllocator()
                holder = {"action": None, "caught": []}

                def snapshot():
                    action = holder["action"]
                    if action is not None:
                        holder["action"] = None
                        try:
                            action()
                        except BaseException as error:
                            holder["caught"].append(error)
                    return allocator.snapshot()

                adapter = ExternalFactorAllocatorAdapter.bind(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=snapshot,
                )
                owner = ConstraintProgramOwner(adapter)
                owner.allocate_continuous(1)
                frame = owner.frame()
                arena = owner.new_arena()
                empty = arena.empty_view
                kwargs = dict(
                    frame=frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="outer",
                )
                result = arena.append_le(empty, **kwargs)
                holder["action"] = (
                    (lambda: arena.prepare_le(result.view, **kwargs))
                    if nested == "prepare"
                    else (lambda: arena.seal(empty, final_frame=frame))
                )
                programs_before = len(core._PROGRAM_REGISTRY)
                with self.assertRaises(ExternalAllocatorContractError):
                    arena.seal(result.view, final_frame=frame)
                self.assertEqual(len(holder["caught"]), 1)
                self.assertEqual(len(core._PROGRAM_REGISTRY), programs_before)
                state = core._arena_state(arena)
                self.assertEqual(state.pending, [])
                self.assertEqual(state.prepared, {})
                self.assertFalse(state.sealed)


class ConstraintProgramConstructionTests(unittest.TestCase):
    def test_public_two_phase_construction_and_legacy_factories_match(self):
        allocator = _FakeAllocator(start=31)
        snapshot_calls = 0

        def snapshot():
            nonlocal snapshot_calls
            snapshot_calls += 1
            return allocator.snapshot()

        adapter = ExternalFactorAllocatorAdapter.reserve()
        self.assertIs(type(adapter), ExternalFactorAllocatorAdapter)
        self.assertNotIn(id(adapter), core._ADAPTER_REGISTRY)
        self.assertNotIn(id(allocator), core._ALLOCATOR_BINDINGS)
        with self.assertRaises(ExternalAllocatorContractError):
            _ = adapter.namespace_identity
        self.assertIsNone(
            adapter.initialize(
                allocator,
                allocate_continuous=allocator.allocate_continuous,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=snapshot,
            )
        )
        self.assertEqual(snapshot_calls, 0)
        namespace = adapter.namespace_identity
        with self.assertRaises(ExternalAllocatorContractError):
            adapter.initialize(
                allocator,
                allocate_continuous=allocator.allocate_continuous,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=snapshot,
            )
        self.assertIs(adapter.namespace_identity, namespace)

        owner = ConstraintProgramOwner.reserve()
        self.assertIs(type(owner), ConstraintProgramOwner)
        self.assertNotIn(id(owner), core._OWNER_REGISTRY)
        rejected = (
            lambda: owner.representation_authority,
            lambda: owner.proof_authority,
            lambda: owner.discarded,
            lambda: owner.allocate_continuous(0),
            lambda: owner.allocate_binary(0),
            owner.frame,
            owner.new_arena,
        )
        for operation in rejected:
            with self.assertRaises(ConstraintProgramError):
                operation()
        self.assertIsNone(owner.initialize(adapter))
        self.assertEqual(snapshot_calls, 1)
        with self.assertRaises(ConstraintProgramError):
            owner.initialize(adapter)
        self.assertEqual(snapshot_calls, 1)
        arena = owner.new_arena()
        arena.discard()
        self.assertTrue(owner.discarded)

        legacy_allocator = _FakeAllocator()
        legacy_adapter = _bind(legacy_allocator)
        legacy_owner = ConstraintProgramOwner(legacy_adapter)
        self.assertIs(type(legacy_adapter), ExternalFactorAllocatorAdapter)
        self.assertIs(type(legacy_owner), ConstraintProgramOwner)
        legacy_arena = legacy_owner.new_arena()
        legacy_arena.discard()

    def test_initialization_is_one_shot_and_preserves_callback_exception(self):
        allocator = _FakeAllocator()
        adapter = ExternalFactorAllocatorAdapter.reserve()
        with self.assertRaises(ExternalAllocatorContractError):
            adapter.initialize(
                allocator,
                allocate_continuous=None,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=allocator.snapshot,
            )
        self.assertNotIn(id(adapter), core._ADAPTER_REGISTRY)
        self.assertNotIn(id(allocator), core._ALLOCATOR_BINDINGS)
        self.assertIs(
            core._ADAPTER_RESERVATIONS[id(adapter)].state.phase,
            core._ReservationPhase.POISONED,
        )
        with self.assertRaises(ExternalAllocatorContractError):
            adapter.initialize(
                allocator,
                allocate_continuous=allocator.allocate_continuous,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=allocator.snapshot,
            )

        class SnapshotAbort(BaseException):
            pass

        allocator2 = _FakeAllocator()
        marker = SnapshotAbort("snapshot effect")
        calls = 0

        def interrupted_snapshot():
            nonlocal calls
            calls += 1
            allocator2.allocate_continuous(1)
            raise marker

        adapter2 = ExternalFactorAllocatorAdapter.reserve()
        adapter2.initialize(
            allocator2,
            allocate_continuous=allocator2.allocate_continuous,
            allocate_binary=allocator2.allocate_binary,
            live_ids_snapshot=interrupted_snapshot,
        )
        owner = ConstraintProgramOwner.reserve()
        try:
            owner.initialize(adapter2)
        except SnapshotAbort as error:
            self.assertIs(error, marker)
        else:
            self.fail("snapshot BaseException was not preserved")
        marker.__traceback__ = None
        self.assertEqual((calls, allocator2.snapshot()), (1, ((0,), ())))
        self.assertNotIn(id(owner), core._OWNER_REGISTRY)
        self.assertIs(
            core._OWNER_RESERVATIONS[id(owner)].state.phase,
            core._ReservationPhase.POISONED,
        )
        with self.assertRaises(ConstraintProgramError):
            owner.initialize(adapter2)
        self.assertEqual(calls, 1)

        allocator3 = _FakeAllocator()
        unready = ExternalFactorAllocatorAdapter.reserve()
        owner2 = ConstraintProgramOwner.reserve()
        with self.assertRaises(ExternalAllocatorContractError):
            owner2.initialize(unready)
        self.assertIs(
            core._OWNER_RESERVATIONS[id(owner2)].state.phase,
            core._ReservationPhase.POISONED,
        )
        unready.initialize(
            allocator3,
            allocate_continuous=allocator3.allocate_continuous,
            allocate_binary=allocator3.allocate_binary,
            live_ids_snapshot=allocator3.snapshot,
        )
        with self.assertRaises(ConstraintProgramError):
            owner2.initialize(unready)

    def test_thread_forgery_subclass_and_reservation_rebinding_fail_closed(self):
        allocator = _FakeAllocator()
        adapter = ExternalFactorAllocatorAdapter.reserve()
        adapter_errors = []
        start = threading.Event()
        done = threading.Event()

        def initialize_adapter_on_wrong_thread():
            start.wait()
            try:
                adapter.initialize(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=allocator.snapshot,
                )
            except BaseException as error:
                adapter_errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=initialize_adapter_on_wrong_thread)
        thread.start()
        start.set()
        self.assertTrue(done.wait(5.0))
        thread.join()
        self.assertEqual(len(adapter_errors), 1)
        self.assertIsInstance(adapter_errors[0], ExternalAllocatorContractError)
        self.assertIs(
            core._ADAPTER_RESERVATIONS[id(adapter)].state.phase,
            core._ReservationPhase.RESERVED,
        )
        adapter.initialize(
            allocator,
            allocate_continuous=allocator.allocate_continuous,
            allocate_binary=allocator.allocate_binary,
            live_ids_snapshot=allocator.snapshot,
        )

        owner = ConstraintProgramOwner.reserve()
        owner_errors = []
        done.clear()

        def initialize_owner_on_wrong_thread():
            try:
                owner.initialize(adapter)
            except BaseException as error:
                owner_errors.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=initialize_owner_on_wrong_thread)
        thread.start()
        self.assertTrue(done.wait(5.0))
        thread.join()
        self.assertEqual(len(owner_errors), 1)
        self.assertIsInstance(owner_errors[0], ConstraintProgramError)
        self.assertIs(
            core._OWNER_RESERVATIONS[id(owner)].state.phase,
            core._ReservationPhase.RESERVED,
        )
        owner.initialize(adapter)
        arena = owner.new_arena()
        arena.discard()

        foreign = []

        def make_foreign_adapter():
            other = _FakeAllocator()
            foreign.append((other, _bind(other)))

        thread = threading.Thread(target=make_foreign_adapter)
        thread.start()
        thread.join()
        foreign_owner = ConstraintProgramOwner.reserve()
        with self.assertRaises(ExternalAllocatorContractError):
            foreign_owner.initialize(foreign[0][1])
        self.assertIs(
            core._OWNER_RESERVATIONS[id(foreign_owner)].state.phase,
            core._ReservationPhase.POISONED,
        )

        forged_adapter = object.__new__(ExternalFactorAllocatorAdapter)
        with self.assertRaises(ExternalAllocatorContractError):
            forged_adapter.initialize(
                allocator,
                allocate_continuous=allocator.allocate_continuous,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=allocator.snapshot,
            )
        forged_owner = object.__new__(ConstraintProgramOwner)
        with self.assertRaises(ConstraintProgramError):
            forged_owner.initialize(adapter)

        class AdapterSubclass(ExternalFactorAllocatorAdapter):
            pass

        class OwnerSubclass(ConstraintProgramOwner):
            pass

        with self.assertRaises(TypeError):
            AdapterSubclass.reserve()
        with self.assertRaises(TypeError):
            OwnerSubclass.reserve()

        rebound = ExternalFactorAllocatorAdapter.reserve()
        reservation = core._ADAPTER_RESERVATIONS[id(rebound)]
        object.__setattr__(reservation, "state_key", ("substituted",))
        with self.assertRaises(ExternalAllocatorContractError):
            rebound.initialize(
                allocator,
                allocate_continuous=allocator.allocate_continuous,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=allocator.snapshot,
            )

    def test_owner_initialization_callback_swallowed_mutation_is_sticky(self):
        for operation_name in ("initialize", "allocate", "frame", "new_arena"):
            with self.subTest(operation=operation_name):
                allocator = _FakeAllocator()
                holder = {
                    "owner": None,
                    "adapter": None,
                    "caught": [],
                    "calls": 0,
                }

                def snapshot():
                    holder["calls"] += 1
                    owner = holder["owner"]
                    operations = {
                        "initialize": lambda: owner.initialize(holder["adapter"]),
                        "allocate": lambda: owner.allocate_continuous(0),
                        "frame": owner.frame,
                        "new_arena": owner.new_arena,
                    }
                    try:
                        operations[operation_name]()
                    except BaseException as error:
                        holder["caught"].append(error)
                    return allocator.snapshot()

                adapter = ExternalFactorAllocatorAdapter.reserve()
                adapter.initialize(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=snapshot,
                )
                owner = ConstraintProgramOwner.reserve()
                holder["owner"] = owner
                holder["adapter"] = adapter
                with self.assertRaises(ExternalAllocatorContractError):
                    owner.initialize(adapter)
                self.assertEqual(holder["calls"], 1)
                self.assertEqual(len(holder["caught"]), 1)
                self.assertIsInstance(holder["caught"][0], ConstraintProgramError)
                self.assertNotIn(id(owner), core._OWNER_REGISTRY)
                self.assertIs(
                    core._OWNER_RESERVATIONS[id(owner)].state.phase,
                    core._ReservationPhase.POISONED,
                )
                with self.assertRaises(ConstraintProgramError):
                    owner.initialize(adapter)
                self.assertEqual(holder["calls"], 1)

    def test_reserve_return_and_all_construction_phases_gc_cleanly(self):
        class ReserveAbort(BaseException):
            pass

        cases = (
            (
                ExternalFactorAllocatorAdapter.reserve,
                ExternalFactorAllocatorAdapter.reserve.__func__.__code__,
                core._ADAPTER_RESERVATIONS,
                core._ADAPTER_REGISTRY,
            ),
            (
                ConstraintProgramOwner.reserve,
                ConstraintProgramOwner.reserve.__func__.__code__,
                core._OWNER_RESERVATIONS,
                core._OWNER_REGISTRY,
            ),
        )
        for reserve, target_code, reservations, registry in cases:
            with self.subTest(reserve=reserve.__qualname__):
                captured = []

                def interrupt_return(frame_obj, event, value):
                    if frame_obj.f_code is target_code and event == "return":
                        captured.append((id(value), weakref.ref(value)))
                        sys.settrace(None)
                        raise ReserveAbort("public reserve return")
                    return interrupt_return

                sys.settrace(interrupt_return)
                try:
                    reserve()
                except ReserveAbort as error:
                    error.__traceback__ = None
                else:
                    self.fail("reserve return trace did not interrupt")
                finally:
                    sys.settrace(None)
                gc.collect()
                self.assertEqual(len(captured), 1)
                object_id, reference = captured[0]
                self.assertIsNone(reference())
                self.assertNotIn(object_id, reservations)
                self.assertNotIn(object_id, registry)

        allocator = _FakeAllocator()
        handles = []
        reserved_adapter = ExternalFactorAllocatorAdapter.reserve()
        handles.append(
            (
                weakref.ref(reserved_adapter),
                id(reserved_adapter),
                core._ADAPTER_RESERVATIONS,
            )
        )
        del reserved_adapter

        poisoned_adapter = ExternalFactorAllocatorAdapter.reserve()
        with self.assertRaises(ExternalAllocatorContractError):
            poisoned_adapter.initialize(
                allocator,
                allocate_continuous=None,
                allocate_binary=allocator.allocate_binary,
                live_ids_snapshot=allocator.snapshot,
            )
        handles.append(
            (
                weakref.ref(poisoned_adapter),
                id(poisoned_adapter),
                core._ADAPTER_RESERVATIONS,
            )
        )
        del poisoned_adapter

        reserved_owner = ConstraintProgramOwner.reserve()
        handles.append(
            (
                weakref.ref(reserved_owner),
                id(reserved_owner),
                core._OWNER_RESERVATIONS,
            )
        )
        del reserved_owner

        unready = ExternalFactorAllocatorAdapter.reserve()
        poisoned_owner = ConstraintProgramOwner.reserve()
        with self.assertRaises(ExternalAllocatorContractError):
            poisoned_owner.initialize(unready)
        handles.append(
            (
                weakref.ref(poisoned_owner),
                id(poisoned_owner),
                core._OWNER_RESERVATIONS,
            )
        )
        del poisoned_owner
        gc.collect()
        for reference, object_id, registry in handles:
            self.assertIsNone(reference())
            self.assertNotIn(object_id, registry)

        ready_allocator = _FakeAllocator()
        ready_adapter = _bind(ready_allocator)
        adapter_id = id(ready_adapter)
        binding_id = id(ready_allocator)
        reference = weakref.ref(ready_adapter)
        del ready_adapter
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(adapter_id, core._ADAPTER_REGISTRY)
        self.assertNotIn(binding_id, core._ALLOCATOR_BINDINGS)

        ready_adapter = _bind(ready_allocator)
        ready_owner = ConstraintProgramOwner(ready_adapter)
        owner_id = id(ready_owner)
        reference = weakref.ref(ready_owner)
        del ready_owner
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(owner_id, core._OWNER_REGISTRY)

    def test_adapter_publication_fault_matrix_and_partial_read_rejection(self):
        class PublishAbort(BaseException):
            pass

        lines, first_line = inspect.getsourcelines(
            core._publish_adapter_initialization
        )

        def line_containing(text):
            return next(
                first_line + offset
                for offset, source in enumerate(lines)
                if text in source
            )

        boundaries = (
            (
                "pre-intent",
                line_containing("_ALLOCATOR_BINDINGS[allocator_id] = ("),
                False,
            ),
            (
                "post-intent",
                line_containing(
                    "_ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record"
                ),
                True,
            ),
            (
                "post-publishing-guard",
                line_containing("_ADAPTER_REGISTRY[id(adapter)] = entry"),
                True,
            ),
            (
                "post-entry",
                line_containing("_ADAPTER_RESERVATIONS.pop(id(adapter), None)"),
                True,
            ),
        )
        for name, target_line, expect_ready in boundaries:
            with self.subTest(boundary=name):
                allocator = _FakeAllocator()
                adapter = ExternalFactorAllocatorAdapter.reserve()
                marker = PublishAbort(name)

                def interrupt(frame_obj, event, _value):
                    if (
                        frame_obj.f_code
                        is core._publish_adapter_initialization.__code__
                        and event == "line"
                        and frame_obj.f_lineno == target_line
                    ):
                        sys.settrace(None)
                        raise marker
                    return interrupt

                sys.settrace(interrupt)
                try:
                    adapter.initialize(
                        allocator,
                        allocate_continuous=allocator.allocate_continuous,
                        allocate_binary=allocator.allocate_binary,
                        live_ids_snapshot=allocator.snapshot,
                    )
                except PublishAbort as error:
                    self.assertIs(error, marker)
                else:
                    self.fail("adapter publication boundary did not interrupt")
                finally:
                    sys.settrace(None)
                marker.__traceback__ = None
                if expect_ready:
                    self.assertIsNotNone(adapter.namespace_identity)
                    self.assertNotIn(id(adapter), core._ADAPTER_RESERVATIONS)
                    binding = core._ALLOCATOR_BINDINGS[id(allocator)]
                    self.assertIs(binding[0], allocator)
                    self.assertEqual(len(binding), 3)
                else:
                    with self.assertRaises(ExternalAllocatorContractError):
                        _ = adapter.namespace_identity
                    self.assertNotIn(id(adapter), core._ADAPTER_REGISTRY)
                    self.assertNotIn(id(allocator), core._ALLOCATOR_BINDINGS)
                    self.assertIs(
                        core._ADAPTER_RESERVATIONS[id(adapter)].state.phase,
                        core._ReservationPhase.POISONED,
                    )

        allocator = _FakeAllocator()
        adapter = ExternalFactorAllocatorAdapter.reserve()
        target_line = line_containing(
            "_ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record"
        )
        repair_started = threading.Event()
        observer_done = threading.Event()
        observer_errors = []
        marker = PublishAbort("partial visibility")
        original_repair = core._repair_adapter_initialization

        def observer():
            repair_started.wait()
            try:
                _ = adapter.namespace_identity
            except BaseException as error:
                observer_errors.append(error)
            finally:
                observer_done.set()

        thread = threading.Thread(target=observer)
        thread.start()

        def pause_before_repair(*args, **kwargs):
            repair_started.set()
            self.assertTrue(observer_done.wait(5.0))
            return original_repair(*args, **kwargs)

        def interrupt_partial(frame_obj, event, _value):
            if (
                frame_obj.f_code is core._publish_adapter_initialization.__code__
                and event == "line"
                and frame_obj.f_lineno == target_line
            ):
                sys.settrace(None)
                raise marker
            return interrupt_partial

        with mock.patch.object(
            core,
            "_repair_adapter_initialization",
            side_effect=pause_before_repair,
        ):
            sys.settrace(interrupt_partial)
            try:
                adapter.initialize(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=allocator.snapshot,
                )
            except PublishAbort as error:
                self.assertIs(error, marker)
            else:
                self.fail("partial publication did not interrupt")
            finally:
                sys.settrace(None)
        marker.__traceback__ = None
        thread.join()
        self.assertEqual(len(observer_errors), 1)
        self.assertIsInstance(observer_errors[0], ExternalAllocatorContractError)
        self.assertIsNotNone(adapter.namespace_identity)

        allocator3 = _FakeAllocator()
        first_adapter = ExternalFactorAllocatorAdapter.reserve()
        competitor_ready = threading.Event()
        begin_competitor = threading.Event()
        competitor_done = threading.Event()
        competitor = {}
        competitor_errors = []

        def compete_for_allocator():
            second_adapter = ExternalFactorAllocatorAdapter.reserve()
            competitor["adapter"] = second_adapter
            competitor_ready.set()
            begin_competitor.wait()
            try:
                second_adapter.initialize(
                    allocator3,
                    allocate_continuous=allocator3.allocate_continuous,
                    allocate_binary=allocator3.allocate_binary,
                    live_ids_snapshot=allocator3.snapshot,
                )
            except BaseException as error:
                competitor_errors.append(error)
            finally:
                competitor_done.set()

        thread = threading.Thread(target=compete_for_allocator)
        thread.start()
        self.assertTrue(competitor_ready.wait(5.0))
        marker = PublishAbort("post-intent allocator reservation")
        original_repair = core._repair_adapter_initialization

        def let_competitor_run(*args, **kwargs):
            if args[0] is first_adapter:
                begin_competitor.set()
                self.assertTrue(competitor_done.wait(5.0))
            return original_repair(*args, **kwargs)

        def interrupt_after_allocator_intent(frame_obj, event, _value):
            if (
                frame_obj.f_code is core._publish_adapter_initialization.__code__
                and event == "line"
                and frame_obj.f_lineno == target_line
            ):
                sys.settrace(None)
                raise marker
            return interrupt_after_allocator_intent

        with mock.patch.object(
            core,
            "_repair_adapter_initialization",
            side_effect=let_competitor_run,
        ):
            sys.settrace(interrupt_after_allocator_intent)
            try:
                first_adapter.initialize(
                    allocator3,
                    allocate_continuous=allocator3.allocate_continuous,
                    allocate_binary=allocator3.allocate_binary,
                    live_ids_snapshot=allocator3.snapshot,
                )
            except PublishAbort as error:
                self.assertIs(error, marker)
            else:
                self.fail("allocator intent did not interrupt")
            finally:
                sys.settrace(None)
        marker.__traceback__ = None
        thread.join()
        self.assertEqual(len(competitor_errors), 1)
        self.assertIsInstance(
            competitor_errors[0], ExternalAllocatorContractError
        )
        self.assertIsNotNone(first_adapter.namespace_identity)
        with self.assertRaises(ExternalAllocatorContractError):
            _ = competitor["adapter"].namespace_identity
        first_record = core._adapter_record(first_adapter)
        binding = core._ALLOCATOR_BINDINGS[id(allocator3)]
        self.assertIs(binding[2], first_record.commit_token)

        allocator2 = _FakeAllocator()
        adapter2 = ExternalFactorAllocatorAdapter.reserve()
        marker = PublishAbort("initialize public return")

        def interrupt_initialize_return(frame_obj, event, _value):
            if (
                frame_obj.f_code
                is ExternalFactorAllocatorAdapter.initialize.__code__
                and event == "return"
            ):
                sys.settrace(None)
                raise marker
            return interrupt_initialize_return

        sys.settrace(interrupt_initialize_return)
        try:
            adapter2.initialize(
                allocator2,
                allocate_continuous=allocator2.allocate_continuous,
                allocate_binary=allocator2.allocate_binary,
                live_ids_snapshot=allocator2.snapshot,
            )
        except PublishAbort as error:
            self.assertIs(error, marker)
        else:
            self.fail("adapter initialize return did not interrupt")
        finally:
            sys.settrace(None)
        marker.__traceback__ = None
        self.assertIsNotNone(adapter2.namespace_identity)

    def test_owner_publication_and_public_return_faults_leave_recoverable_new(self):
        class PublishAbort(BaseException):
            pass

        lines, first_line = inspect.getsourcelines(
            core._publish_owner_initialization
        )

        def line_containing(text):
            return next(
                first_line + offset
                for offset, source in enumerate(lines)
                if text in source
            )

        boundaries = (
            line_containing("_OWNER_REGISTRY[id(owner)] = entry"),
            line_containing("_OWNER_RESERVATIONS.pop(id(owner), None)"),
        )
        for target_line in boundaries:
            with self.subTest(line=target_line):
                allocator = _FakeAllocator()
                adapter = _bind(allocator)
                owner = ConstraintProgramOwner.reserve()
                marker = PublishAbort(str(target_line))

                def interrupt(frame_obj, event, _value):
                    if (
                        frame_obj.f_code
                        is core._publish_owner_initialization.__code__
                        and event == "line"
                        and frame_obj.f_lineno == target_line
                    ):
                        sys.settrace(None)
                        raise marker
                    return interrupt

                sys.settrace(interrupt)
                try:
                    owner.initialize(adapter)
                except PublishAbort as error:
                    self.assertIs(error, marker)
                else:
                    self.fail("owner registry boundary did not interrupt")
                finally:
                    sys.settrace(None)
                marker.__traceback__ = None
                arena = owner.new_arena()
                arena.discard()
                self.assertTrue(owner.discarded)

        allocator = _FakeAllocator()
        adapter = _bind(allocator)
        owner = ConstraintProgramOwner.reserve()
        marker = PublishAbort("owner initialize public return")

        def interrupt_return(frame_obj, event, _value):
            if (
                frame_obj.f_code is ConstraintProgramOwner.initialize.__code__
                and event == "return"
            ):
                sys.settrace(None)
                raise marker
            return interrupt_return

        sys.settrace(interrupt_return)
        try:
            owner.initialize(adapter)
        except PublishAbort as error:
            self.assertIs(error, marker)
        else:
            self.fail("owner initialize return did not interrupt")
        finally:
            sys.settrace(None)
        marker.__traceback__ = None
        arena = owner.new_arena()
        arena.discard()

    def test_repair_interruptions_preserve_first_exception_and_complete_truth(self):
        class FirstAbort(BaseException):
            pass

        class RepairAbort(BaseException):
            pass

        for after_publish in (False, True):
            with self.subTest(kind="adapter", after_publish=after_publish):
                allocator = _FakeAllocator()
                adapter = ExternalFactorAllocatorAdapter.reserve()
                first = FirstAbort("adapter first")
                second = RepairAbort("adapter repair")
                original_publish = core._publish_adapter_initialization
                original_repair = core._repair_adapter_initialization

                def publish(*args, **kwargs):
                    if after_publish:
                        original_publish(*args, **kwargs)
                    raise first

                def repair(*args, **kwargs):
                    if after_publish:
                        original_repair(*args, **kwargs)
                    raise second

                with mock.patch.object(
                    core,
                    "_publish_adapter_initialization",
                    side_effect=publish,
                ), mock.patch.object(
                    core,
                    "_repair_adapter_initialization",
                    side_effect=repair,
                ):
                    try:
                        adapter.initialize(
                            allocator,
                            allocate_continuous=allocator.allocate_continuous,
                            allocate_binary=allocator.allocate_binary,
                            live_ids_snapshot=allocator.snapshot,
                        )
                    except FirstAbort as error:
                        self.assertIs(error, first)
                    else:
                        self.fail("adapter first exception was not preserved")
                first.__traceback__ = None
                second.__traceback__ = None
                if after_publish:
                    self.assertIsNotNone(adapter.namespace_identity)
                else:
                    with self.assertRaises(ExternalAllocatorContractError):
                        _ = adapter.namespace_identity
                    self.assertNotIn(id(allocator), core._ALLOCATOR_BINDINGS)
                    self.assertIs(
                        core._ADAPTER_RESERVATIONS[id(adapter)].state.phase,
                        core._ReservationPhase.POISONED,
                    )

        allocator = _FakeAllocator()
        adapter = ExternalFactorAllocatorAdapter.reserve()
        first = FirstAbort("adapter intent first")
        second = RepairAbort("adapter intent repair")
        source, first_line = inspect.getsourcelines(
            core._publish_adapter_initialization
        )
        post_intent_line = next(
            first_line + offset
            for offset, line in enumerate(source)
            if "_ADAPTER_REGISTRY[id(adapter)] = entry" in line
        )

        def interrupt_post_intent(frame_obj, event, _value):
            if (
                frame_obj.f_code
                is core._publish_adapter_initialization.__code__
                and event == "line"
                and frame_obj.f_lineno == post_intent_line
            ):
                sys.settrace(None)
                raise first
            return interrupt_post_intent

        with mock.patch.object(
            core,
            "_repair_adapter_initialization",
            side_effect=second,
        ):
            sys.settrace(interrupt_post_intent)
            try:
                adapter.initialize(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=allocator.snapshot,
                )
            except FirstAbort as error:
                self.assertIs(error, first)
            else:
                self.fail("post-intent first exception was not preserved")
            finally:
                sys.settrace(None)
        first.__traceback__ = None
        second.__traceback__ = None
        self.assertIsNotNone(adapter.namespace_identity)

        allocator = _FakeAllocator()
        adapter = ExternalFactorAllocatorAdapter.reserve()
        first = FirstAbort("adapter intent reentrancy")
        nested_errors = []
        intent_guard_line = next(
            first_line + offset
            for offset, line in enumerate(source)
            if "_ADAPTER_RESERVATIONS[id(adapter)] = _make_reservation_record"
            in line
        )
        original_repair = core._repair_adapter_initialization

        def interrupt_before_guard(frame_obj, event, _value):
            if (
                frame_obj.f_code
                is core._publish_adapter_initialization.__code__
                and event == "line"
                and frame_obj.f_lineno == intent_guard_line
            ):
                sys.settrace(None)
                raise first
            return interrupt_before_guard

        def swallow_nested_then_repair(*args, **kwargs):
            try:
                adapter.initialize(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=allocator.snapshot,
                )
            except BaseException as error:
                nested_errors.append(error)
            return original_repair(*args, **kwargs)

        with mock.patch.object(
            core,
            "_repair_adapter_initialization",
            side_effect=swallow_nested_then_repair,
        ):
            sys.settrace(interrupt_before_guard)
            try:
                adapter.initialize(
                    allocator,
                    allocate_continuous=allocator.allocate_continuous,
                    allocate_binary=allocator.allocate_binary,
                    live_ids_snapshot=allocator.snapshot,
                )
            except FirstAbort as error:
                self.assertIs(error, first)
            else:
                self.fail("post-intent reentrancy did not preserve first error")
            finally:
                sys.settrace(None)
        first.__traceback__ = None
        self.assertEqual(len(nested_errors), 1)
        self.assertIsInstance(nested_errors[0], ExternalAllocatorContractError)
        self.assertIsNotNone(adapter.namespace_identity)

        for after_publish in (False, True):
            with self.subTest(kind="owner", after_publish=after_publish):
                allocator = _FakeAllocator()
                adapter = _bind(allocator)
                owner = ConstraintProgramOwner.reserve()
                first = FirstAbort("owner first")
                second = RepairAbort("owner repair")
                original_publish = core._publish_owner_initialization
                original_repair = core._repair_owner_initialization

                def publish(*args, **kwargs):
                    if after_publish:
                        original_publish(*args, **kwargs)
                    raise first

                def repair(*args, **kwargs):
                    if after_publish:
                        original_repair(*args, **kwargs)
                    raise second

                with mock.patch.object(
                    core,
                    "_publish_owner_initialization",
                    side_effect=publish,
                ), mock.patch.object(
                    core,
                    "_repair_owner_initialization",
                    side_effect=repair,
                ):
                    try:
                        owner.initialize(adapter)
                    except FirstAbort as error:
                        self.assertIs(error, first)
                    else:
                        self.fail("owner first exception was not preserved")
                first.__traceback__ = None
                second.__traceback__ = None
                arena = owner.new_arena()
                arena.discard()
                self.assertTrue(owner.discarded)


class ConstraintProgramStructureTests(unittest.TestCase):
    def test_all_range_matches_disconnected_candidate_oracle_fieldwise(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, -2.0], [0.5, 0.0]])
        fb = _canonical([[3.0], [0.0]])
        fu = np.asarray([1.25, 2.0], dtype=np.float64)
        ru = np.asarray([0.75, 1.0], dtype=np.float64)
        result = _append_band(arena, frame, fc=fc, fb=fb, fu=fu, ru=ru)
        program = arena.seal(result.view, final_frame=frame)
        native = _collect(program, native=True, max_rows=256)
        legacy = _collect(program, native=False, max_rows=256)

        from act.back_end.solver import constraint_block_dag_candidate as oracle

        oracle_owner = oracle.ExactConstraintOwner()
        oracle_owner.allocate_continuous(2)
        oracle_owner.allocate_binary(1)
        oracle_frame = oracle_owner.frame()
        oracle_arena = oracle_owner.new_arena()
        appended = oracle_arena.append_guarded_band(
            oracle_arena.empty_view,
            frame=oracle_frame,
            forward_cont=fc,
            forward_bin=fb,
            forward_upper=fu,
            reverse_cont=-fc,
            reverse_bin=-fb,
            reverse_upper=ru,
            layer_id=7,
        )
        oracle_program = oracle_arena.seal(appended.view, final_frame=oracle_frame)
        oracle_native = oracle_program.native_blocks()[0]
        oracle_legacy = oracle_program.replay_virtual_facets()

        self.assertEqual(_csr_bits(native["A_cont"]), _csr_bits(oracle_native.A_cont))
        self.assertEqual(_csr_bits(native["A_bin"]), _csr_bits(oracle_native.A_bin))
        np.testing.assert_array_equal(
            native["lower"].view(np.uint64), oracle_native.lower.view(np.uint64)
        )
        np.testing.assert_array_equal(
            native["upper"].view(np.uint64), oracle_native.upper.view(np.uint64)
        )
        self.assertEqual(_csr_bits(legacy["A_cont"]), _csr_bits(oracle_legacy.A_cont))
        self.assertEqual(_csr_bits(legacy["A_bin"]), _csr_bits(oracle_legacy.A_bin))
        np.testing.assert_array_equal(
            legacy["upper"].view(np.uint64), oracle_legacy.upper.view(np.uint64)
        )
        self.assertEqual(native["row_tags"], oracle_native.row_tags)
        self.assertEqual(legacy["row_tags"], oracle_legacy.row_tags)
        self.assertEqual(
            (
                program.source_rows,
                program.virtual_facet_rows,
                program.source_nnz,
                program.virtual_facet_nnz,
                program.ranged_rows,
                program.fallback_pairs,
            ),
            (2, 4, fc.nnz + fb.nnz, 2 * (fc.nnz + fb.nnz), 2, 0),
        )

    def test_mixed_fallback_is_per_row_and_exactly_replayed(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, 2.0], [3.0, 4.0]])
        fb = _canonical([[0.5], [-0.25]])
        rc, rb = -fc, -fb
        rc.data[-1] = np.nextafter(rc.data[-1], np.inf)
        fu = np.asarray([1.0, 2.0], dtype=np.float64)
        ru = np.asarray([3.0, 4.0], dtype=np.float64)
        result = _append_band(
            arena, frame, fc=fc, fb=fb, fu=fu, rc=rc, rb=rb, ru=ru
        )
        program = arena.seal(result.view, final_frame=frame)
        native = _collect(program, native=True, max_rows=2)
        legacy = _collect(program, native=False, max_rows=2)
        self.assertEqual(
            (result.ranged_rows, result.fallback_pairs, program.source_rows),
            (1, 1, 3),
        )
        self.assertEqual(
            _csr_bits(legacy["A_cont"]),
            _csr_bits(sp.vstack((fc, rc), format="csr")),
        )
        self.assertEqual(
            _csr_bits(legacy["A_bin"]),
            _csr_bits(sp.vstack((fb, rb), format="csr")),
        )
        np.testing.assert_array_equal(
            legacy["upper"].view(np.uint64),
            np.concatenate((fu, ru)).view(np.uint64),
        )
        self.assertTrue(np.isneginf(native["lower"][1]))
        self.assertTrue(np.isneginf(native["lower"][2]))

    def test_le_range_le_global_order_and_all_chunk_sizes(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        empty_bin_1 = sp.csr_matrix((1, 1), dtype=np.float64)
        first = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0, 0.0]]),
            A_bin=empty_bin_1,
            upper=np.asarray([9.0], dtype=np.float64),
            tag="input_le",
            layer_id=1,
        )
        band = _append_band(arena, frame, view=first.view, layer_id=2)
        last = arena.append_le(
            band.view,
            frame=frame,
            A_cont=_canonical([[0.0, 1.0], [-1.0, 1.0]]),
            A_bin=sp.csr_matrix((2, 1), dtype=np.float64),
            upper=np.asarray([8.0, 7.0], dtype=np.float64),
            tag="output_le",
            layer_id=3,
        )
        program = arena.seal(last.view, final_frame=frame)
        self.assertEqual(program.append_ordinals, (0, 1, 2))
        native_reference = _collect(program, native=True, max_rows=256)
        legacy_reference = _collect(program, native=False, max_rows=256)
        self.assertEqual(
            native_reference["row_tags"],
            (
                "input_le:1",
                "range:add_materialize:2",
                "range:add_materialize:2",
                "output_le:3",
                "output_le:3",
            ),
        )
        self.assertEqual(
            legacy_reference["row_tags"],
            (
                "input_le:1",
                "add_materialize:2:forward",
                "add_materialize:2:forward",
                "add_materialize:2:reverse",
                "add_materialize:2:reverse",
                "output_le:3",
                "output_le:3",
            ),
        )
        for chunk in (1, 2, 63, 256):
            for is_native, reference in (
                (True, native_reference),
                (False, legacy_reference),
            ):
                current = _collect(program, native=is_native, max_rows=chunk)
                self.assertEqual(_csr_bits(current["A_cont"]), _csr_bits(reference["A_cont"]))
                self.assertEqual(_csr_bits(current["A_bin"]), _csr_bits(reference["A_bin"]))
                np.testing.assert_array_equal(
                    current["upper"].view(np.uint64),
                    reference["upper"].view(np.uint64),
                )
                if is_native:
                    np.testing.assert_array_equal(
                        current["lower"].view(np.uint64),
                        reference["lower"].view(np.uint64),
                    )
                self.assertEqual(current["row_tags"], reference["row_tags"])
                self.assertEqual(
                    tuple(batch.row_offset for batch in current["batches"]),
                    tuple(range(0, reference["upper"].size, chunk)),
                )
                self.assertTrue(all(0 < batch.row_count <= chunk for batch in current["batches"]))
                self.assertTrue(all(batch.total_rows == reference["upper"].size for batch in current["batches"]))

    def test_digest_collision_is_only_bucket_and_full_bytes_decide(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        A = _canonical([[1.0]])
        empty = sp.csr_matrix((1, 0), dtype=np.float64)
        with mock.patch.object(core, "_digest_key", return_value="forced-collision"):
            first = arena.append_le(
                arena.empty_view,
                frame=frame,
                A_cont=A,
                A_bin=empty,
                upper=np.asarray([1.0], dtype=np.float64),
                tag="a",
            )
            second = arena.append_le(
                first.view,
                frame=frame,
                A_cont=A,
                A_bin=empty,
                upper=np.asarray([2.0], dtype=np.float64),
                tag="b",
            )
            program = arena.seal(second.view, final_frame=frame)
            replay = _collect(program, native=True, max_rows=256)
            np.testing.assert_array_equal(replay["upper"], np.asarray([1.0, 2.0]))
            self.assertEqual(program.block_count, 2)
            self.assertIs(dict(program.receipt)["authenticity_from_digest"], False)

    def test_identical_payload_interning_uses_full_equality(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        A = _canonical([[1.0], [-2.0]])
        empty = sp.csr_matrix((2, 0), dtype=np.float64)
        upper = np.asarray([1.0, 2.0], dtype=np.float64)
        first = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=A,
            A_bin=empty,
            upper=upper,
            tag="first",
        )
        second = arena.append_le(
            first.view,
            frame=frame,
            A_cont=A.copy(),
            A_bin=empty.copy(),
            upper=upper.copy(),
            tag="second",
        )
        program = arena.seal(second.view, final_frame=frame)
        one_payload_bytes = core._program_state(program).occurrences[0].payload.payload_bytes
        self.assertEqual(program.numeric_payload_bytes, one_payload_bytes)
        self.assertEqual(program.block_count, 2)

    def test_signed_zero_subnormal_and_max_finite_are_preserved(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
        maximum = np.finfo(np.float64).max
        fc = sp.csr_matrix(
            (
                np.asarray([tiny, -maximum], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int64),
                np.asarray([0, 1, 2], dtype=np.int64),
            ),
            shape=(2, 1),
        )
        fb = sp.csr_matrix((2, 0), dtype=np.float64)
        fu = np.asarray([0.0, -0.0], dtype=np.float64)
        ru = np.asarray([-0.0, maximum], dtype=np.float64)
        result = _append_band(arena, frame, fc=fc, fb=fb, fu=fu, ru=ru)
        program = arena.seal(result.view, final_frame=frame)
        native = _collect(program, native=True, max_rows=2)
        legacy = _collect(program, native=False, max_rows=2)
        expected_lower = np.asarray([0.0, -maximum], dtype=np.float64)
        np.testing.assert_array_equal(
            native["lower"].view(np.uint64), expected_lower.view(np.uint64)
        )
        np.testing.assert_array_equal(
            legacy["upper"].view(np.uint64),
            np.concatenate((fu, ru)).view(np.uint64),
        )

    def test_empty_program_and_exact_max_rows_contract(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        program = arena.seal(arena.empty_view, final_frame=frame)
        self.assertEqual((program.block_count, program.source_rows), (0, 0))
        self.assertEqual(list(program.iter_native_batches(max_rows=1)), [])
        self.assertEqual(list(program.iter_legacy_facet_batches(max_rows=256)), [])
        for invalid in (0, 257, True, np.int64(2), 1.5):
            with self.assertRaises(ConstraintProgramError):
                program.iter_native_batches(max_rows=invalid)

    def test_seal_rejects_stale_final_frame_after_live_snapshot_growth(self):
        for grow_through_owner in (True, False):
            with self.subTest(grow_through_owner=grow_through_owner):
                allocator, _adapter, owner, old_frame, arena = _new_arena(
                    n_cont=1, n_bin=0
                )
                result = arena.append_le(
                    arena.empty_view,
                    frame=old_frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="stale",
                )
                if grow_through_owner:
                    owner.allocate_continuous(1)
                else:
                    allocator.allocate_continuous(1)
                with self.assertRaises(ConstraintProgramError):
                    arena.seal(result.view, final_frame=old_frame)
                latest = owner.frame()
                program = arena.seal(result.view, final_frame=latest)
                self.assertEqual(
                    tuple(item.raw_id for item in program.continuous_ids),
                    tuple(allocator.continuous),
                )
                self.assertEqual(
                    next(program.iter_native_batches(max_rows=1)).A_cont.shape[1],
                    2,
                )


class ConstraintProgramTransactionTests(unittest.TestCase):
    def test_owner_reserve_activate_and_finalize_are_exception_recoverable(self):
        allocator = _FakeAllocator()
        adapter = _bind(allocator)
        owner = ConstraintProgramOwner(adapter)

        target = core._activate_owner_operation.__code__

        def trace(frame_obj, event, _arg):
            if frame_obj.f_code is target and event == "return":
                sys.settrace(None)
                raise KeyboardInterrupt("activate-return")
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(KeyboardInterrupt):
                owner.frame()
        finally:
            sys.settrace(None)
        entry = core._OWNER_REGISTRY[id(owner)]
        self.assertFalse(entry.operation_active)
        self.assertIsNotNone(owner.frame())

        original_commit = core._commit_owner_operation

        def interrupt_after_commit(*args, **kwargs):
            original_commit(*args, **kwargs)
            raise SystemExit("owner-post-swap")

        with mock.patch.object(
            core, "_commit_owner_operation", side_effect=interrupt_after_commit
        ):
            with self.assertRaises(SystemExit):
                owner.frame()
        entry = core._OWNER_REGISTRY[id(owner)]
        self.assertFalse(entry.operation_active)
        self.assertIsNotNone(owner.frame())

    def test_all_final_owner_swaps_reject_swallowed_reentrancy_and_restore(self):
        def le_kwargs(frame):
            return dict(
                frame=frame,
                A_cont=_canonical([[1.0]]),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="final-swap",
            )

        def run_case(name, owner, operation, verify):
            original_commit = core._commit_owner_operation
            nested_errors = []
            captured_rollbacks = []

            def swallowed_reentrancy(state, epoch):
                if not captured_rollbacks:
                    entry = core._owner_operation_entry(state)
                    self.assertIsNotNone(entry)
                    self.assertTrue(entry.operation_active)
                    captured_rollbacks.append(entry.rollback_data)
                    try:
                        owner.frame()
                    except BaseException as error:
                        nested_errors.append(error)
                return original_commit(state, epoch)

            with self.subTest(operation=name):
                with mock.patch.object(
                    core,
                    "_commit_owner_operation",
                    side_effect=swallowed_reentrancy,
                ):
                    with self.assertRaises(ExternalAllocatorContractError):
                        operation()
                self.assertEqual(len(nested_errors), 1)
                self.assertIsInstance(nested_errors[0], ConstraintProgramError)
                self.assertEqual(len(captured_rollbacks), 1)
                state = core._owner_state(owner)
                entry = core._owner_operation_entry(state)
                self.assertIsNotNone(entry)
                self.assertFalse(entry.operation_active)
                self.assertFalse(entry.reentrancy_detected)
                self.assertIsNone(entry.rollback_data)
                self.assertFalse(entry.external_touched)
                self.assertIs(state._data, captured_rollbacks[0])
                self.assertFalse(state.poisoned)
                verify()

        allocator = _FakeAllocator()
        owner = ConstraintProgramOwner(_bind(allocator))
        run_case(
            "allocate",
            owner,
            lambda: owner.allocate_continuous(1),
            lambda: self.assertEqual(allocator.snapshot(), ((0,), ())),
        )

        allocator = _FakeAllocator()
        owner = ConstraintProgramOwner(_bind(allocator))
        owner.allocate_continuous(1)
        frames_before = set(core._owner_state(owner).frames)
        run_case(
            "frame",
            owner,
            owner.frame,
            lambda: self.assertEqual(
                set(core._owner_state(owner).frames), frames_before
            ),
        )

        _allocator, _adapter, owner, _frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        run_case(
            "new_arena_existing",
            owner,
            owner.new_arena,
            lambda: self.assertIs(core._arena_state(arena).owner, owner),
        )

        _allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        union_before = core._ARENA_REGISTRY[id(arena)].committed_key
        run_case(
            "union",
            owner,
            lambda: arena.union(arena.empty_view),
            lambda: self.assertEqual(
                core._ARENA_REGISTRY[id(arena)].committed_key, union_before
            ),
        )

        _allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        run_case(
            "prepare",
            owner,
            lambda: arena.prepare_le(arena.empty_view, **le_kwargs(frame)),
            lambda: self.assertEqual(
                (
                    core._arena_state(arena).next_sequence,
                    core._arena_state(arena).pending,
                    core._arena_state(arena).prepared,
                ),
                (1, [], {}),
            ),
        )

        _allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        prepared = arena.prepare_le(arena.empty_view, **le_kwargs(frame))
        run_case(
            "commit",
            owner,
            lambda: arena.commit(prepared),
            lambda: self.assertEqual(
                (
                    core._arena_state(arena).blocks,
                    core._arena_state(arena).pending,
                    core._arena_state(arena).prepared,
                ),
                ({}, [], {}),
            ),
        )

        _allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        prepared = arena.prepare_le(arena.empty_view, **le_kwargs(frame))
        run_case(
            "abort",
            owner,
            lambda: arena.abort(prepared),
            lambda: self.assertEqual(
                (
                    core._arena_state(arena).blocks,
                    core._arena_state(arena).pending,
                    core._arena_state(arena).prepared,
                ),
                ({}, [], {}),
            ),
        )

        _allocator, _adapter, owner, _frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        program_ids = set(core._PROGRAM_REGISTRY)
        run_case(
            "discard",
            owner,
            arena.discard,
            lambda: self.assertEqual(
                (
                    core._arena_state(arena).discarded,
                    core._arena_state(arena).sealed,
                    set(core._PROGRAM_REGISTRY),
                ),
                (False, False, program_ids),
            ),
        )

        _allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        staged = []
        original_stage = core._stage_program

        def capture_stage(*args, **kwargs):
            result = original_stage(*args, **kwargs)
            staged.append(result)
            return result

        program_ids = set(core._PROGRAM_REGISTRY)
        with mock.patch.object(core, "_stage_program", side_effect=capture_stage):
            run_case(
                "seal",
                owner,
                lambda: arena.seal(arena.empty_view, final_frame=frame),
                lambda: self.assertEqual(
                    (
                        core._arena_state(arena).sealed,
                        id(arena) in core._SEALED_ARENA_REGISTRY,
                        set(core._PROGRAM_REGISTRY),
                    ),
                    (False, False, program_ids),
                ),
            )
        self.assertEqual(len(staged), 1)
        staged_program, staged_state = staged[0]
        self.assertFalse(core._is_registered_program(staged_program, staged_state))
        with self.assertRaises(ConstraintProgramError):
            _ = staged_program.representation_authority

        allocator = _FakeAllocator()
        owner = ConstraintProgramOwner(_bind(allocator))
        state = core._owner_state(owner)
        epoch = core._begin_owner_operation(state)
        core._activate_owner_operation(state, epoch)
        active = core._owner_operation_entry(state)
        self.assertIsNotNone(active)
        rollback = active.rollback_data
        core._mark_external_call(state, epoch)
        with self.assertRaises(ConstraintProgramError):
            owner.frame()
        with self.assertRaises(ExternalAllocatorContractError):
            core._end_owner_operation(state, epoch)
        entry = core._owner_operation_entry(state)
        self.assertFalse(entry.operation_active)
        self.assertTrue(state.poisoned)
        self.assertIs(state.snapshot, rollback.snapshot)
        self.assertIs(state.id_cache, rollback.id_cache)
        self.assertIs(state.claimed, rollback.claimed)
        self.assertIs(state.frames, rollback.frames)

    def test_reentrant_owner_rollback_tolerates_pre_and_post_swap_interruptions(self):
        allocator = _FakeAllocator()
        owner = ConstraintProgramOwner(_bind(allocator))
        owner.allocate_continuous(1)
        original_commit = core._commit_owner_operation
        original_restore = core._restore_owner_operation
        captured = []
        restore_calls = 0

        def swallowed_reentrancy(state, epoch):
            if not captured:
                captured.append(core._owner_operation_entry(state).rollback_data)
                try:
                    owner.frame()
                except ConstraintProgramError:
                    pass
            return original_commit(state, epoch)

        def interrupt_restore(*args, **kwargs):
            nonlocal restore_calls
            restore_calls += 1
            if restore_calls == 1:
                raise KeyboardInterrupt("owner rollback pre-swap")
            result = original_restore(*args, **kwargs)
            if restore_calls == 2:
                raise SystemExit("owner rollback post-swap")
            return result

        with mock.patch.object(
            core,
            "_commit_owner_operation",
            side_effect=swallowed_reentrancy,
        ), mock.patch.object(
            core,
            "_restore_owner_operation",
            side_effect=interrupt_restore,
        ):
            with self.assertRaises(ExternalAllocatorContractError):
                owner.frame()
        self.assertEqual(restore_calls, 2)
        state = core._owner_state(owner)
        entry = core._owner_operation_entry(state)
        self.assertFalse(entry.operation_active)
        self.assertIs(state._data, captured[0])
        self.assertIsNotNone(owner.frame())

    def test_new_arena_staging_never_leaves_an_orphan_or_occupied_hole(self):
        for helper in ("_register_staged_arena", "_commit_owner_operation"):
            with self.subTest(helper=helper):
                allocator = _FakeAllocator()
                adapter = _bind(allocator)
                owner = ConstraintProgramOwner(adapter)
                original = getattr(core, helper)

                def interrupt_after(*args, _original=original, **kwargs):
                    _original(*args, **kwargs)
                    raise KeyboardInterrupt(helper)

                with mock.patch.object(core, helper, side_effect=interrupt_after):
                    with self.assertRaises(KeyboardInterrupt):
                        owner.new_arena()
                owner_state = core._owner_state(owner)
                self.assertFalse(owner_state.arena_created)
                self.assertIsNone(owner_state.arena_id)
                self.assertFalse(
                    any(
                        entry.state.owner_state is owner_state
                        for entry in core._ARENA_REGISTRY.values()
                    )
                )
                arena = owner.new_arena()
                self.assertIs(owner.new_arena(), arena)

    def test_prepare_abort_is_atomic_and_burns_occurrence_ids(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        kwargs = dict(
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="txn",
        )
        prepared = arena.prepare_le(arena.empty_view, **kwargs)
        private = core._arena_state(arena).prepared[id(prepared)]
        burned_block = private.result.block_id.value
        burned_row = private.result.source_row_ids[0].value
        self.assertEqual(len(core._arena_state(arena).blocks), 0)
        arena.abort(prepared)
        with self.assertRaises(ConstraintTransactionError):
            arena.commit(prepared)
        next_prepared = arena.prepare_le(arena.empty_view, **kwargs)
        next_private = core._arena_state(arena).prepared[id(next_prepared)]
        self.assertGreater(next_private.result.block_id.value, burned_block)
        self.assertGreater(next_private.result.source_row_ids[0].value, burned_row)
        committed = arena.commit(next_prepared)
        program = arena.seal(committed.view, final_frame=frame)
        self.assertEqual(program.source_rows, 1)

    def test_public_append_metadata_is_explicitly_non_authoritative(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        result = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="metadata",
        )
        forged = core.ConstraintAppend(
            result.view,
            result.block_id,
            result.source_row_ids,
            result.legacy_facet_ids,
            999,
            999,
            999,
            999,
            999,
            999,
            999,
        )
        self.assertIs(result.representation_authority, False)
        self.assertIs(result.replay_authority, False)
        self.assertIs(forged.representation_authority, False)
        self.assertIs(forged.replay_authority, False)

    def test_prepare_order_is_linear_and_failed_out_of_order_cap_is_consumed(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        kwargs = dict(
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="ordered",
        )
        first = arena.prepare_le(arena.empty_view, **kwargs)
        second = arena.prepare_le(arena.empty_view, **kwargs)
        with self.assertRaises(ConstraintTransactionError):
            arena.commit(second)
        with self.assertRaises(ConstraintTransactionError):
            arena.commit(second)
        committed = arena.commit(first)
        program = arena.seal(committed.view, final_frame=frame)
        self.assertEqual(program.append_ordinals, (0,))

    def test_baseexception_during_commit_revokes_capability_without_publication(self):
        for exception in (KeyboardInterrupt("interrupt"), SystemExit("exit")):
            with self.subTest(exception=type(exception).__name__):
                _allocator, _adapter, _owner, frame, arena = _new_arena(
                    n_cont=1, n_bin=0
                )
                prepared = arena.prepare_le(
                    arena.empty_view,
                    frame=frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="interrupt",
                )
                with mock.patch.object(core, "_digest_key", side_effect=exception):
                    with self.assertRaises(type(exception)):
                        arena.commit(prepared)
                self.assertEqual(len(core._arena_state(arena).blocks), 0)
                with self.assertRaises(ConstraintTransactionError):
                    arena.commit(prepared)

    def test_commit_has_no_partial_publication_at_each_mutation_boundary(self):
        source, start = inspect.getsourcelines(core._commit_validated)
        markers = (
            "interned_payload = occurrence.payload",
            "block_previous = state.blocks.get",
            "view_previous = state.views.get",
            "view_bucket_key = _digest_key",
            "view_append_attempted = True",
            "state.prepared.pop",
            "if state.pending and state.pending[0]",
            'transaction.status = "committed"',
        )
        targets = {}
        for marker in markers:
            targets[marker] = start + next(
                index for index, line in enumerate(source) if marker in line
            )

        for marker, target in targets.items():
            with self.subTest(marker=marker):
                _allocator, _adapter, _owner, frame, arena = _new_arena(
                    n_cont=1, n_bin=0
                )
                kwargs = dict(
                    frame=frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="commit-boundary",
                )
                prepared = arena.prepare_le(arena.empty_view, **kwargs)

                def trace(frame_obj, event, _arg):
                    if (
                        frame_obj.f_code is core._commit_validated.__code__
                        and event == "line"
                        and frame_obj.f_lineno == target
                    ):
                        sys.settrace(None)
                        raise KeyboardInterrupt(marker)
                    return trace

                sys.settrace(trace)
                try:
                    with self.assertRaises(KeyboardInterrupt):
                        arena.commit(prepared)
                finally:
                    sys.settrace(None)
                state = core._arena_state(arena)
                self.assertEqual(state.blocks, {})
                self.assertEqual(state.payload_buckets, {})
                self.assertEqual(len(state.views), 1)
                self.assertEqual(state.prepared, {})
                self.assertEqual(state.pending, [])
                committed = arena.append_le(arena.empty_view, **kwargs)
                self.assertEqual(committed.append_ordinal, 1)
                program = arena.seal(committed.view, final_frame=frame)
                self.assertEqual(program.source_rows, 1)

    def test_prepare_baseexceptions_leave_only_burned_sequence(self):
        failure_factories = []

        def occurrence_failure():
            return mock.patch.object(
                core,
                "_Occurrence",
                side_effect=KeyboardInterrupt("occurrence"),
            )

        failure_factories.append(occurrence_failure)

        def view_failure():
            original = core._make_view

            def fail_after_view(*args, **kwargs):
                original(*args, **kwargs)
                raise SystemExit("view")

            return mock.patch.object(core, "_make_view", side_effect=fail_after_view)

        failure_factories.append(view_failure)

        def prepared_failure():
            return mock.patch.object(
                core,
                "PreparedAppend",
                side_effect=KeyboardInterrupt("prepared"),
            )

        failure_factories.append(prepared_failure)

        for make_failure in failure_factories:
            with self.subTest(failure=make_failure.__name__):
                _allocator, _adapter, _owner, frame, arena = _new_arena(
                    n_cont=1, n_bin=0
                )
                kwargs = dict(
                    frame=frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="prepare-boundary",
                )
                with make_failure():
                    with self.assertRaises((KeyboardInterrupt, SystemExit)):
                        arena.prepare_le(arena.empty_view, **kwargs)
                state = core._arena_state(arena)
                self.assertEqual(state.next_sequence, 1)
                self.assertEqual(state.prepared, {})
                self.assertEqual(state.pending, [])
                committed = arena.append_le(arena.empty_view, **kwargs)
                self.assertEqual(committed.append_ordinal, 1)

        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        kwargs = dict(
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="sequence-publish",
        )
        original_publish = core._publish_arena_state
        calls = 0

        def interrupt_after_publish(state):
            nonlocal calls
            calls += 1
            original_publish(state)
            if calls == 1:
                raise KeyboardInterrupt("after sequence publish")

        with mock.patch.object(
            core, "_publish_arena_state", side_effect=interrupt_after_publish
        ):
            with self.assertRaises(KeyboardInterrupt):
                arena.prepare_le(arena.empty_view, **kwargs)
        self.assertEqual(core._arena_state(arena).next_sequence, 1)
        committed = arena.append_le(arena.empty_view, **kwargs)
        self.assertEqual(committed.append_ordinal, 1)

    def test_abort_repairs_prepared_and_pending_after_async_boundary(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        kwargs = dict(
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="abort-boundary",
        )
        prepared = arena.prepare_le(arena.empty_view, **kwargs)
        source, start = inspect.getsourcelines(core._abort_transaction)
        target = start + next(
            index
            for index, line in enumerate(source)
            if "while object_id in state.pending" in line
        )

        def trace(frame_obj, event, _arg):
            if (
                frame_obj.f_code.co_name == "repair"
                and frame_obj.f_back is not None
                and frame_obj.f_back.f_code is core._abort_transaction.__code__
                and event == "line"
                and frame_obj.f_lineno == target
            ):
                sys.settrace(None)
                raise KeyboardInterrupt("after prepared removal")
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(KeyboardInterrupt):
                arena.abort(prepared)
        finally:
            sys.settrace(None)
        state = core._arena_state(arena)
        self.assertEqual(state.prepared, {})
        self.assertEqual(state.pending, [])
        committed = arena.append_le(arena.empty_view, **kwargs)
        self.assertEqual(committed.append_ordinal, 1)

    def test_arena_rollback_and_dead_cap_prune_tolerate_two_interruptions(self):
        _allocator, _adapter, _owner, _frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        original_publish = core._publish_arena_state
        calls = 0

        def interrupt_twice(state):
            nonlocal calls
            original_publish(state)
            calls += 1
            if calls <= 2:
                raise KeyboardInterrupt(f"arena-post-swap-{calls}")

        with mock.patch.object(
            core, "_publish_arena_state", side_effect=interrupt_twice
        ):
            with self.assertRaises(ConstraintProgramError):
                arena.union()
        self.assertIs(arena.empty_view, arena.empty_view)

        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        prepared = arena.prepare_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="gc-double-interrupt",
        )
        reference = weakref.ref(prepared)
        del prepared
        gc.collect()
        self.assertIsNone(reference())
        calls = 0
        with mock.patch.object(
            core, "_publish_arena_state", side_effect=interrupt_twice
        ):
            self.assertEqual(core._arena_state(arena).pending, [])
        self.assertEqual(core._arena_state(arena).prepared, {})

    def test_seal_stages_program_and_repairs_async_publication(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        result = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="staged-program",
        )
        original_stage = core._stage_program
        leaked = []

        def interrupt_after_stage(*args, **kwargs):
            program, state = original_stage(*args, **kwargs)
            leaked.append(program)
            raise KeyboardInterrupt("before seal publication")

        with mock.patch.object(
            core, "_stage_program", side_effect=interrupt_after_stage
        ):
            with self.assertRaises(KeyboardInterrupt):
                arena.seal(result.view, final_frame=frame)
        self.assertEqual(len(leaked), 1)
        with self.assertRaises(ConstraintProgramError):
            _ = leaked[0].representation_authority
        self.assertFalse(core._arena_state(arena).sealed)

        original_complete = core._complete_seal_publication
        calls = 0

        def interrupt_after_complete(*args, **kwargs):
            nonlocal calls
            calls += 1
            program = original_complete(*args, **kwargs)
            if calls == 1:
                raise SystemExit("after combined seal publication")
            return program

        with mock.patch.object(
            core,
            "_complete_seal_publication",
            side_effect=interrupt_after_complete,
        ):
            with self.assertRaises(SystemExit):
                arena.seal(result.view, final_frame=frame)
        state = core._arena_state(arena)
        self.assertTrue(state.sealed)
        self.assertTrue(state.owner_state.sealed)
        recovered = arena.seal(result.view, final_frame=frame)
        self.assertTrue(recovered.representation_authority)
        self.assertEqual(recovered.source_rows, 1)
        self.assertEqual(recovered.append_ordinals, (0,))

    def test_seal_pre_registry_failure_rolls_back_and_gc_revival_chooses_one(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        result = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="seal-registry",
        )
        with mock.patch.object(
            core,
            "_register_staged_program",
            side_effect=KeyboardInterrupt("pre-program-swap"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                arena.seal(result.view, final_frame=frame)
        state = core._arena_state(arena)
        self.assertFalse(state.sealed)
        self.assertFalse(state.owner_state.sealed)

        program = arena.seal(result.view, final_frame=frame)
        program_reference = weakref.ref(program)
        del program
        gc.collect()
        self.assertIsNone(program_reference())

        original_register = core._register_staged_program

        def interrupt_after_register(*args, **kwargs):
            original_register(*args, **kwargs)
            raise SystemExit("revival-post-program-swap")

        with mock.patch.object(
            core,
            "_register_staged_program",
            side_effect=interrupt_after_register,
        ):
            with self.assertRaises(SystemExit):
                arena.seal(result.view, final_frame=frame)
        recovered = arena.seal(result.view, final_frame=frame)
        self.assertTrue(recovered.representation_authority)
        record = core._SEALED_ARENA_REGISTRY[id(arena)]
        self.assertIs(record.program(), recovered)
        self.assertIs(core._program_state(recovered), record.program_state)

    def test_prepared_gc_aborts_and_releases_capture(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        prepared = arena.prepare_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="gc",
        )
        reference = weakref.ref(prepared)
        del prepared
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(core._arena_state(arena).pending, [])
        self.assertEqual(core._arena_state(arena).prepared, {})

    def test_seal_rejects_live_transaction_and_consumed_capability(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        prepared = arena.prepare_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="live",
        )
        with self.assertRaises(ConstraintTransactionError):
            arena.seal(arena.empty_view, final_frame=frame)
        result = arena.commit(prepared)
        with self.assertRaises(ConstraintTransactionError):
            arena.commit(prepared)
        arena.seal(result.view, final_frame=frame)


class ConstraintProgramHostileTests(unittest.TestCase):
    def test_input_and_returned_mutation_cannot_change_source(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, -2.0], [0.5, 0.0]])
        fb = _canonical([[3.0], [0.0]])
        fu = np.asarray([1.25, 2.0], dtype=np.float64)
        ru = np.asarray([0.75, 1.0], dtype=np.float64)
        expected_fc = fc.copy()
        expected_fb = fb.copy()
        result = _append_band(arena, frame, fc=fc, fb=fb, fu=fu, ru=ru)
        fc.data[:] = 99.0
        fb.data[:] = 88.0
        fu[:] = 77.0
        ru[:] = 66.0
        program = arena.seal(result.view, final_frame=frame)
        legacy = _collect(program, native=False, max_rows=256)
        self.assertEqual(
            _csr_bits(legacy["A_cont"]),
            _csr_bits(sp.vstack((expected_fc, -expected_fc), format="csr")),
        )
        self.assertEqual(
            _csr_bits(legacy["A_bin"]),
            _csr_bits(sp.vstack((expected_fb, -expected_fb), format="csr")),
        )
        batch = next(program.iter_native_batches(max_rows=2))
        with self.assertRaises(ValueError):
            batch.A_cont.data[0] = 12.0
        detached = batch.A_cont
        detached.data = np.full(detached.nnz, 5.0)
        again = next(program.iter_native_batches(max_rows=2))
        self.assertNotEqual(tuple(again.A_cont.data), tuple(detached.data))

    def test_noncanonical_explicit_zero_mutable_subclass_and_wrong_family_reject(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        explicit_zero = sp.csr_matrix(
            (
                np.asarray([0.0], dtype=np.float64),
                np.asarray([0], dtype=np.int64),
                np.asarray([0, 1], dtype=np.int64),
            ),
            shape=(1, 1),
        )
        with self.assertRaises(ConstraintProgramError):
            arena.prepare_le(
                arena.empty_view,
                frame=frame,
                A_cont=explicit_zero,
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="bad",
            )

        class CSRSubclass(sp.csr_matrix):
            pass

        with self.assertRaises(ConstraintProgramError):
            arena.prepare_le(
                arena.empty_view,
                frame=frame,
                A_cont=CSRSubclass(_canonical([[1.0]])),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="bad",
            )
        with self.assertRaises(ConstraintProgramError):
            arena.prepare_guarded_band(
                arena.empty_view,
                frame=frame,
                forward_cont=_canonical([[1.0]]),
                forward_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                forward_upper=np.asarray([1.0], dtype=np.float64),
                reverse_cont=_canonical([[-1.0]]),
                reverse_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                reverse_upper=np.asarray([1.0], dtype=np.float64),
                layer_id=1,
                family="add_materialize",
            )

    def test_cross_owner_arena_and_mutable_threads_fail_closed(self):
        _a1, _ad1, owner1, frame1, arena1 = _new_arena(n_cont=1, n_bin=0)
        _a2, _ad2, _owner2, frame2, arena2 = _new_arena(n_cont=1, n_bin=0)
        with self.assertRaises(ConstraintArenaMismatch):
            arena1.union(arena2.empty_view)
        with self.assertRaises(ConstraintArenaMismatch):
            arena1.prepare_le(
                arena1.empty_view,
                frame=frame2,
                A_cont=_canonical([[1.0]]),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="foreign",
            )
        errors = []

        def mutate():
            for action in (
                lambda: owner1.allocate_continuous(1),
                lambda: arena1.union(arena1.empty_view),
            ):
                try:
                    action()
                except BaseException as error:
                    errors.append(error)

        thread = threading.Thread(target=mutate)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(type(item) is ConstraintProgramError for item in errors))
        self.assertEqual(len(frame1.continuous_ids), 1)

    def test_owner_and_arena_registry_anchor_rebinding_fail_closed(self):
        allocator1 = _FakeAllocator(start=0)
        adapter1 = _bind(allocator1)
        owner1 = ConstraintProgramOwner(adapter1)
        allocator2 = _FakeAllocator(start=100)
        adapter2 = _bind(allocator2)
        owner2 = ConstraintProgramOwner(adapter2)
        state1 = core._owner_state(owner1)
        state2 = core._owner_state(owner2)
        object.__setattr__(state1, "adapter", adapter2)
        object.__setattr__(state1, "adapter_record", state2.adapter_record)
        object.__setattr__(state1, "thread", state2.thread)
        object.__setattr__(state1, "token", state2.token)
        with self.assertRaises(ConstraintProgramError):
            owner1.allocate_continuous(1)
        self.assertEqual(allocator1.snapshot(), ((), ()))
        self.assertEqual(allocator2.snapshot(), ((), ()))

        _a3, _ad3, _owner3, frame3, arena3 = _new_arena(
            n_cont=1, n_bin=0
        )
        _a4, _ad4, owner4, frame4, _arena4 = _new_arena(
            n_cont=1, n_bin=0
        )
        arena_state = core._arena_state(arena3)
        owner4_state = core._owner_state(owner4)
        object.__setattr__(arena_state, "owner", owner4)
        object.__setattr__(arena_state, "owner_state", owner4_state)
        with self.assertRaises(ConstraintProgramError):
            arena3.prepare_le(
                arena3.empty_view,
                frame=frame4,
                A_cont=_canonical([[1.0]]),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="anchor",
            )
        self.assertEqual(len(frame3.continuous_ids), 1)

    def test_factor_cache_namespace_and_committed_block_swaps_fail_closed(self):
        allocator = _FakeAllocator()
        adapter = _bind(allocator)
        owner = ConstraintProgramOwner(adapter)
        factor = owner.allocate_continuous(1)[0]
        state = core._owner_state(owner)
        state.id_cache[(FactorKind.CONTINUOUS, factor.raw_id)] = core._new_factor_id(
            FactorKind.CONTINUOUS,
            factor.raw_id,
            core._NamespaceIdentity(),
        )
        with self.assertRaises(ConstraintProgramError):
            owner.frame()

        for replacement_kind in ("direct", "same-key-copy"):
            with self.subTest(replacement_kind=replacement_kind):
                _a, _ad, _o, frame, arena = _new_arena(n_cont=1, n_bin=0)
                first = arena.append_le(
                    arena.empty_view,
                    frame=frame,
                    A_cont=_canonical([[1.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([1.0], dtype=np.float64),
                    tag="first",
                )
                second = arena.append_le(
                    first.view,
                    frame=frame,
                    A_cont=_canonical([[2.0]]),
                    A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                    upper=np.asarray([2.0], dtype=np.float64),
                    tag="second",
                )
                arena_state = core._arena_state(arena)
                first_key, second_key = sorted(
                    arena_state.blocks,
                    key=lambda item: item[1],
                )
                replacement = arena_state.blocks[second_key]
                if replacement_kind == "same-key-copy":
                    replacement = replace(replacement, block_key=first_key)
                arena_state.blocks[first_key] = replacement
                with self.assertRaises(ConstraintProgramError):
                    arena.seal(first.view, final_frame=frame)
                self.assertEqual(second.append_ordinal, 1)

    def test_namespace_token_cannot_change_class_and_all_mutable_builders_are_non_authority(self):
        _allocator, adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        result = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="namespace",
        )
        program = arena.seal(result.view, final_frame=frame)

        class EvilNamespace:
            pass

        with self.assertRaises(TypeError):
            object.__setattr__(
                adapter.namespace_identity,
                "__class__",
                EvilNamespace,
            )
        self.assertTrue(program.representation_authority)
        self.assertIs(owner.representation_authority, False)
        self.assertIs(arena.representation_authority, False)

    def test_empty_view_full_graph_is_validated(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        empty = arena.empty_view
        result = arena.append_le(
            empty,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="empty-anchor",
        )
        object.__setattr__(empty, "_view_id", result.view.view_id)
        with self.assertRaises(ConstraintProgramError):
            _ = arena.representation_authority
        with self.assertRaises(ConstraintProgramError):
            _ = arena.empty_view

    def test_arena_full_root_indexes_are_registry_sealed(self):
        def committed_context():
            _allocator, _adapter, _owner, frame, arena = _new_arena(
                n_cont=1, n_bin=0
            )
            result = arena.append_le(
                arena.empty_view,
                frame=frame,
                A_cont=_canonical([[1.0]]),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="root-graph",
            )
            return arena, result

        arena, _result = committed_context()
        core._arena_state(arena).pending.append(999)
        with self.assertRaises(ConstraintProgramError):
            core._arena_state(arena)

        arena, result = committed_context()
        state = core._arena_state(arena)
        state.views.pop(id(result.view))
        with self.assertRaises(ConstraintProgramError):
            core._arena_state(arena)

        arena, _result = committed_context()
        state = core._arena_state(arena)
        next(iter(state.payload_buckets.values())).clear()
        with self.assertRaises(ConstraintProgramError):
            core._arena_state(arena)

        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        prepared = arena.prepare_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="prepared-root-graph",
        )
        core._arena_state(arena).prepared[id(prepared)].status = "committed"
        with self.assertRaises(ConstraintProgramError):
            core._arena_state(arena)

    def test_frame_and_view_equivalent_rebinding_is_detected(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(n_cont=1, n_bin=0)
        object.__setattr__(
            frame, "_continuous_ids", tuple(list(frame.continuous_ids))
        )
        with self.assertRaises(ConstraintArenaMismatch):
            arena.prepare_le(
                arena.empty_view,
                frame=frame,
                A_cont=_canonical([[1.0]]),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="rebind",
            )

        _a2, _ad2, _o2, frame2, arena2 = _new_arena(n_cont=1, n_bin=0)
        committed = arena2.append_le(
            arena2.empty_view,
            frame=frame2,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="view",
        )
        view = committed.view
        object.__setattr__(view, "_block_ids", tuple(list(view.block_ids)))
        with self.assertRaises(ConstraintProgramError):
            arena2.union(view)
        self.assertEqual(len(frame2.continuous_ids), 1)

    def test_iterator_capture_close_interrupt_gc_and_parallel_readers(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)

        iterator = program.iter_legacy_facet_batches(max_rows=1)
        capture_reference = weakref.ref(core._iterator_state(iterator).capture)
        first = next(iterator)
        self.assertEqual(first.row_offset, 0)
        iterator.close()
        gc.collect()
        self.assertIsNone(capture_reference())
        self.assertTrue(iterator.closed)
        self.assertEqual(list(iterator), [])

        iterator2 = program.iter_native_batches(max_rows=1)
        reference = weakref.ref(iterator2)
        del iterator2
        gc.collect()
        self.assertIsNone(reference())

        for exception in (KeyboardInterrupt("kb"), SystemExit("exit")):
            iterator3 = program.iter_native_batches(max_rows=1)
            with mock.patch.object(core, "_assemble_rows", side_effect=exception):
                with self.assertRaises(type(exception)):
                    next(iterator3)
            self.assertTrue(iterator3.closed)

        expected = _collect(program, native=False, max_rows=1)["upper"].view(np.uint64).tobytes()
        outputs = []
        errors = []

        def read():
            try:
                outputs.append(
                    _collect(program, native=False, max_rows=2)["upper"].view(np.uint64).tobytes()
                )
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=read) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(outputs, [expected] * 8)

    def test_iterator_public_handle_cannot_replace_captured_replay_state(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        for native in (True, False):
            with self.subTest(native=native):
                iterator = (
                    program.iter_native_batches(max_rows=1)
                    if native
                    else program.iter_legacy_facet_batches(max_rows=1)
                )
                first = next(iterator)
                forged = _canonical([[123.0, 456.0]])
                for field, value in (
                    ("_active_cont", forged),
                    ("_capture", object()),
                    ("_row_index", 0),
                    ("_offset", 0),
                ):
                    with self.assertRaises(AttributeError):
                        object.__setattr__(iterator, field, value)
                second = next(iterator)
                reference = _collect(
                    program, native=native, max_rows=1
                )["A_cont"]
                self.assertEqual(
                    _csr_bits(first.A_cont), _csr_bits(reference.getrow(0))
                )
                self.assertEqual(
                    _csr_bits(second.A_cont), _csr_bits(reference.getrow(1))
                )

    def test_iterator_registry_state_is_fully_sealed_and_immutable(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)

        for native in (True, False):
            factory = (
                program.iter_native_batches
                if native
                else program.iter_legacy_facet_batches
            )
            for field, replacement in (
                ("max_rows", 2),
                ("block_index", 1),
                ("row_index", 1),
                ("offset", 1),
                ("closed", True),
                ("capture", None),
                ("lock", threading.Lock()),
            ):
                with self.subTest(native=native, field=field):
                    iterator = factory(max_rows=1)
                    state = core._ITERATOR_REGISTRY[id(iterator)][1]
                    original = getattr(state, field)
                    object.__setattr__(state, field, replacement)
                    try:
                        with self.assertRaises(ConstraintProgramError):
                            next(iterator)
                    finally:
                        object.__setattr__(state, field, original)
                    iterator.close()

            with self.subTest(native=native, field="state_identity"):
                iterator = factory(max_rows=1)
                entry = core._ITERATOR_REGISTRY[id(iterator)]
                cloned_state = replace(entry[1])
                core._ITERATOR_REGISTRY[id(iterator)] = (
                    entry[0],
                    cloned_state,
                    *entry[2:],
                )
                try:
                    with self.assertRaises(ConstraintProgramError):
                        next(iterator)
                finally:
                    core._ITERATOR_REGISTRY[id(iterator)] = entry
                iterator.close()

    def test_iterator_async_advance_and_publish_fail_to_closed_state(self):
        source, start = inspect.getsourcelines(core._BatchIterator.__next__)
        target = start + next(
            index
            for index, line in enumerate(source)
            if line.strip() == "_publish_iterator_state(self, working)"
        )

        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        iterator = program.iter_legacy_facet_batches(max_rows=1)

        def trace(frame_obj, event, _arg):
            if (
                frame_obj.f_code is core._BatchIterator.__next__.__code__
                and event == "line"
                and frame_obj.f_lineno == target
            ):
                sys.settrace(None)
                raise KeyboardInterrupt("after cursor advance")
            return trace

        sys.settrace(trace)
        try:
            with self.assertRaises(KeyboardInterrupt):
                next(iterator)
        finally:
            sys.settrace(None)
        self.assertTrue(iterator.closed)
        with self.assertRaises(StopIteration):
            next(iterator)

        iterator2 = program.iter_native_batches(max_rows=1)
        original_publish = core._publish_iterator_state
        calls = 0

        def interrupt_after_publish(*args, **kwargs):
            nonlocal calls
            calls += 1
            original_publish(*args, **kwargs)
            if calls == 1:
                raise SystemExit("after cursor publication")

        with mock.patch.object(
            core,
            "_publish_iterator_state",
            side_effect=interrupt_after_publish,
        ):
            with self.assertRaises(SystemExit):
                next(iterator2)
        self.assertTrue(iterator2.closed)
        with self.assertRaises(StopIteration):
            next(iterator2)

    def test_iterator_capture_is_detached_from_program_occurrence_graph(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        first = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[3.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([3.0], dtype=np.float64),
            tag="three",
        )
        second = arena.append_le(
            first.view,
            frame=frame,
            A_cont=_canonical([[4.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([4.0], dtype=np.float64),
            tag="four",
        )
        program = arena.seal(second.view, final_frame=frame)
        native = program.iter_native_batches(max_rows=1)
        legacy = program.iter_legacy_facet_batches(max_rows=1)
        state = core._program_state(program)
        object.__setattr__(
            state.occurrences[0],
            "payload",
            state.occurrences[1].payload,
        )
        with self.assertRaises(ConstraintProgramError):
            _ = program.representation_authority
        self.assertEqual(next(native).A_cont.data.tolist(), [3.0])
        self.assertEqual(next(legacy).A_cont.data.tolist(), [3.0])

    def test_one_iterator_is_serialized_across_competing_readers(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        iterator = program.iter_legacy_facet_batches(max_rows=1)
        offsets = []
        errors = []

        def consume():
            try:
                while True:
                    try:
                        offsets.append(next(iterator).row_offset)
                    except StopIteration:
                        return
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=consume) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(offsets), [0, 1, 2, 3])
        self.assertTrue(iterator.closed)

    def test_batch_equivalent_rebind_and_nested_bytes_rebind_fail_closed(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        batch = next(program.iter_native_batches(max_rows=2))
        original_state = batch._state
        object.__setattr__(batch, "_state", replace(original_state))
        with self.assertRaises(ConstraintProgramError):
            _ = batch.upper

        batch2 = next(program.iter_native_batches(max_rows=2))
        object.__setattr__(
            batch2._state,
            "upper",
            replace(batch2._state.upper),
        )
        with self.assertRaises(ConstraintProgramError):
            _ = batch2.upper

    def test_program_nested_graph_and_registry_state_rebind_fail_closed(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        state = core._program_state(program)
        occurrence = state.occurrences[0]
        object.__setattr__(occurrence, "payload", replace(occurrence.payload))
        with self.assertRaises(ConstraintProgramError):
            _ = program.source_rows

        _a2, _ad2, _o2, frame2, arena2 = _new_arena()
        result2 = _append_band(arena2, frame2)
        program2 = arena2.seal(result2.view, final_frame=frame2)
        state2 = core._program_state(program2)
        object.__setattr__(state2, "digest", "".join([state2.digest]))
        # CPython may return the identical string for the expression above;
        # force a semantically equal but distinct object.
        if state2.digest is core._PROGRAM_REGISTRY[id(program2)].graph[7]:
            object.__setattr__(state2, "digest", (state2.digest + "x")[:-1])
        with self.assertRaises(ConstraintProgramError):
            _ = program2.digest

    def test_program_authoritative_accounting_is_recomputed_from_occurrences(self):
        fields = (
            "source_rows",
            "virtual_rows",
            "source_nnz",
            "virtual_nnz",
            "ranged_rows",
            "fallback_pairs",
            "numeric_payload_bytes",
        )
        for field in fields:
            with self.subTest(field=field):
                _allocator, _adapter, _owner, frame, arena = _new_arena()
                result = _append_band(arena, frame)
                program = arena.seal(result.view, final_frame=frame)
                state = core._program_state(program)
                object.__setattr__(state, field, getattr(state, field) + 999)
                with self.assertRaises(ConstraintProgramError):
                    _ = program.receipt

    def test_authority_flags_validate_registry_before_returning_true(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        batch = next(program.iter_native_batches(max_rows=1))
        object.__setattr__(core._program_state(program), "source_rows", 999)
        with self.assertRaises(ConstraintProgramError):
            _ = program.representation_authority
        with self.assertRaises(ConstraintProgramError):
            _ = program.replay_authority

        object.__setattr__(batch._state, "row_offset", 999)
        with self.assertRaises(ConstraintProgramError):
            _ = batch.bytes_backed
        with self.assertRaises(ConstraintProgramError):
            _ = batch.representation_authority
        with self.assertRaises(ConstraintProgramError):
            _ = batch.replay_authority

        forged_program = object.__new__(core.ConstraintProgram)
        forged_batch = object.__new__(core.NativeConstraintBatch)
        with self.assertRaises(ConstraintProgramError):
            _ = forged_program.representation_authority
        with self.assertRaises(ConstraintProgramError):
            _ = forged_batch.representation_authority

    def test_raw_batch_construction_cannot_self_sign_authority(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        issued = next(program.iter_native_batches(max_rows=1))
        self.assertIs(issued.representation_authority, False)
        self.assertIs(issued.replay_authority, False)
        with self.assertRaises(TypeError):
            core.NativeConstraintBatch(
                issued._state,
                _token=core._FACTORY_TOKEN,
            )
        forged = object.__new__(core.NativeConstraintBatch)
        object.__setattr__(forged, "_state", issued._state)
        object.__setattr__(forged, "_sealed", True)
        core._register_batch(forged, issued._state)
        self.assertIs(forged.representation_authority, False)
        self.assertIs(forged.replay_authority, False)

    def test_staged_program_cannot_self_register_without_sealed_provenance(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        result = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="program-provenance",
        )
        genuine = arena.seal(result.view, final_frame=frame)
        genuine_state = core._program_state(genuine)
        staged, staged_state = core._stage_program(
            genuine_state.occurrences,
            (genuine_state.frame_cont_keys, genuine_state.frame_bin_keys, 0),
        )
        with self.assertRaises(ConstraintProgramError):
            core._register_staged_program(staged, staged_state)
        with self.assertRaises(ConstraintProgramError):
            _ = staged.representation_authority

    def test_batch_and_program_registry_gc_callbacks_are_aba_guarded(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        batch = next(program.iter_native_batches(max_rows=1))
        batch_id = id(batch)
        program_id = id(program)
        batch_ref = weakref.ref(batch)
        program_ref = weakref.ref(program)
        del batch
        del program
        gc.collect()
        self.assertIsNone(batch_ref())
        self.assertIsNone(program_ref())
        self.assertNotIn(batch_id, core._BATCH_REGISTRY)
        self.assertNotIn(program_id, core._PROGRAM_REGISTRY)

    def test_iterator_captures_program_before_handle_gc(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        result = _append_band(arena, frame)
        program = arena.seal(result.view, final_frame=frame)
        iterator = program.iter_legacy_facet_batches(max_rows=1)
        reference = weakref.ref(program)
        del program
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(sum(batch.row_count for batch in iterator), 4)

    def test_forbidden_proof_paths_are_never_called(self):
        names = (
            "triangle_relaxation",
            "branch_and_bound",
            "backward_propagation",
            "dual_tightening",
            "solver_proof",
        )
        patches = [
            mock.patch.object(core, name, create=True, side_effect=AssertionError(name))
            for name in names
        ]
        mocks = [patch.start() for patch in patches]
        try:
            _allocator, _adapter, _owner, frame, arena = _new_arena()
            result = _append_band(arena, frame)
            program = arena.seal(result.view, final_frame=frame)
            list(program.iter_native_batches(max_rows=2))
            list(program.iter_legacy_facet_batches(max_rows=2))
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertTrue(all(item.call_count == 0 for item in mocks))
        receipt = dict(program.receipt)
        for key in (
            "proof_authority",
            "verdict_authority",
            "solver_status_authority",
            "triangle_relaxation_called",
            "branch_and_bound_called",
            "backward_called",
            "dual_called",
            "solver_proof_called",
            "real_model_called",
            "large_model_called",
        ):
            self.assertIs(receipt[key], False)
        self.assertIs(type(receipt["program_digest"]), str)
        self.assertIs(type(receipt["block_digests"]), tuple)
        self.assertTrue(
            all(type(value) is int for value in (
                receipt["block_count"],
                receipt["source_rows"],
                receipt["virtual_facet_rows"],
                receipt["source_nnz"],
                receipt["virtual_facet_nnz"],
            ))
        )


class ConstraintProgramPhaseBAPITests(unittest.TestCase):
    def test_le_tags_require_strict_utf8_before_staging_and_preserve_valid_text(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        kwargs = dict(
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            layer_id=7,
        )
        state = core._arena_state(arena)
        before_occurrence = core._NEXT_OCCURRENCE_ID
        before_owner_entry = core._OWNER_REGISTRY[id(state.owner)]
        before_arena_entry = core._ARENA_REGISTRY[id(arena)]
        before_program_ids = set(core._PROGRAM_REGISTRY)
        before_sealed_ids = set(core._SEALED_ARENA_REGISTRY)

        with mock.patch.object(core, "_stage_append", wraps=core._stage_append) as stage:
            for prepare in (arena.prepare_le, arena.prepare_le_exact_tag):
                for tag in ("\ud800", "\udfff"):
                    with self.subTest(api=prepare.__name__, tag=repr(tag)):
                        with self.assertRaises(ConstraintProgramError) as caught:
                            prepare(arena.empty_view, tag=tag, **kwargs)
                        self.assertNotIsInstance(caught.exception, UnicodeEncodeError)
        stage.assert_not_called()
        self.assertEqual(core._NEXT_OCCURRENCE_ID, before_occurrence)
        self.assertIs(core._OWNER_REGISTRY[id(state.owner)], before_owner_entry)
        self.assertIs(core._ARENA_REGISTRY[id(arena)], before_arena_entry)
        self.assertEqual(state.next_sequence, 0)
        self.assertEqual(state.pending, [])
        self.assertEqual(state.prepared, {})
        self.assertEqual(state.blocks, {})
        self.assertEqual(state.payload_buckets, {})
        self.assertEqual(set(core._PROGRAM_REGISTRY), before_program_ids)
        self.assertEqual(set(core._SEALED_ARENA_REGISTRY), before_sealed_ids)

        valid_tags = (
            "BMP:\u96ea",
            "astral:\U0001f642",
            "max:\U0010ffff",
            "nul:\x00:tail",
            "multi:::colon",
        )
        for exact in (False, True):
            for tag in valid_tags:
                with self.subTest(exact=exact, tag=repr(tag)):
                    _a, _adapter, _owner, current_frame, current_arena = _new_arena(
                        n_cont=1, n_bin=0
                    )
                    append = (
                        current_arena.append_le_exact_tag
                        if exact
                        else current_arena.append_le
                    )
                    result = append(
                        current_arena.empty_view,
                        frame=current_frame,
                        A_cont=_canonical([[1.0]]),
                        A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                        upper=np.asarray([1.0], dtype=np.float64),
                        tag=tag,
                        layer_id=7,
                    )
                    program = current_arena.seal(
                        result.view, final_frame=current_frame
                    )
                    expected = tag if exact else f"{tag}:7"
                    self.assertEqual(
                        _collect(program, native=True, max_rows=1)["row_tags"],
                        (expected,),
                    )
                    self.assertEqual(
                        _collect(program, native=False, max_rows=1)["row_tags"],
                        (expected,),
                    )
                    self.assertIs(type(program.digest), str)
                    self.assertTrue(program.digest)

    def test_exact_complete_le_tag_replays_unchanged_and_changes_digest(self):
        def build(tag):
            _allocator, _adapter, _owner, frame, arena = _new_arena(
                n_cont=1, n_bin=0
            )
            result = arena.append_le_exact_tag(
                arena.empty_view,
                frame=frame,
                A_cont=_canonical([[1.0], [2.0]]),
                A_bin=sp.csr_matrix((2, 0), dtype=np.float64),
                upper=np.asarray([3.0, 4.0], dtype=np.float64),
                tag=tag,
                layer_id=17,
            )
            program = arena.seal(result.view, final_frame=frame)
            native = _collect(program, native=True, max_rows=1)
            legacy = _collect(program, native=False, max_rows=2)
            self.assertEqual(native["row_tags"], (tag, tag))
            self.assertEqual(legacy["row_tags"], (tag, tag))
            self.assertEqual(core._program_state(program).occurrences[0].layer_id, 17)
            return program.digest

        first = "relu_active:block-17:forward"
        self.assertNotEqual(build(first), build(first + ":different"))

        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        legacy_append = arena.append_le(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="legacy-compatible",
            layer_id=9,
        )
        program = arena.seal(legacy_append.view, final_frame=frame)
        self.assertEqual(
            _collect(program, native=False, max_rows=1)["row_tags"],
            ("legacy-compatible:9",),
        )

    def test_exact_tag_type_subclass_and_rebinding_fail_closed(self):
        class TextSubclass(str):
            pass

        _allocator, _adapter, _owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        kwargs = dict(
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
        )
        for value in ("", TextSubclass("subclass"), None, 3):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ConstraintProgramError):
                    arena.prepare_le_exact_tag(
                        arena.empty_view, tag=value, **kwargs
                    )
                self.assertEqual(core._arena_state(arena).next_sequence, 0)

        result = arena.append_le_exact_tag(
            arena.empty_view, tag="affine_chain_cut:8:reverse", **kwargs
        )
        program = arena.seal(result.view, final_frame=frame)
        occurrence = core._program_state(program).occurrences[0]
        object.__setattr__(
            occurrence,
            "source_tags",
            tuple(list(occurrence.source_tags)),
        )
        with self.assertRaises(ConstraintProgramError):
            _ = program.representation_authority
        with self.assertRaises(ConstraintProgramError):
            next(program.iter_native_batches(max_rows=1))

        _a2, _ad2, _o2, frame2, arena2 = _new_arena(
            n_cont=1, n_bin=0
        )
        result2 = arena2.append_le_exact_tag(
            arena2.empty_view,
            frame=frame2,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="legacy-tag:rebind",
        )
        program2 = arena2.seal(result2.view, final_frame=frame2)
        occurrence2 = core._program_state(program2).occurrences[0]
        object.__setattr__(
            occurrence2,
            "legacy_tags",
            tuple(list(occurrence2.legacy_tags)),
        )
        with self.assertRaises(ConstraintProgramError):
            next(program2.iter_legacy_facet_batches(max_rows=1))

    def test_discard_empty_is_idempotent_and_closes_owner(self):
        allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        empty = arena.empty_view
        before_snapshot = allocator.snapshot()
        before_calls = allocator.cont_calls
        arena.discard()
        self.assertTrue(arena.discarded)
        self.assertTrue(owner.discarded)
        self.assertFalse(arena.representation_authority)
        state = core._arena_state(arena)
        self.assertTrue(state.discarded)
        self.assertTrue(state.owner_state.discarded)
        self.assertFalse(state.sealed)
        self.assertFalse(state.owner_state.sealed)
        self.assertEqual(state.pending, [])
        self.assertEqual(state.prepared, {})
        self.assertEqual(state.next_sequence, 0)
        self.assertEqual(allocator.snapshot(), before_snapshot)
        self.assertEqual(allocator.cont_calls, before_calls)

        arena.discard()
        arena.close()
        for operation in (
            lambda: owner.allocate_continuous(1),
            owner.frame,
            owner.new_arena,
            lambda: arena.prepare_le_exact_tag(
                empty,
                frame=frame,
                A_cont=_canonical([[1.0]]),
                A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
                upper=np.asarray([1.0], dtype=np.float64),
                tag="after-discard",
            ),
            lambda: arena.seal(empty, final_frame=frame),
            lambda: arena.empty_view,
        ):
            with self.assertRaises(ConstraintProgramError):
                operation()
        self.assertEqual(allocator.snapshot(), before_snapshot)
        self.assertEqual(allocator.cont_calls, before_calls)

    def test_discard_committed_and_pending_consumes_caps_but_burns_order(self):
        allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        committed = arena.append_le_exact_tag(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="committed:forward",
        )
        prepared = arena.prepare_le_exact_tag(
            committed.view,
            frame=frame,
            A_cont=_canonical([[2.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([2.0], dtype=np.float64),
            tag="pending:reverse",
        )
        before_snapshot = allocator.snapshot()
        before_program_ids = set(core._PROGRAM_REGISTRY)
        self.assertEqual(core._arena_state(arena).next_sequence, 2)
        arena.discard()

        state = core._arena_state(arena)
        self.assertEqual(len(state.blocks), 0)
        self.assertEqual(len(state.views), 1)
        self.assertEqual(state.payload_buckets, {})
        self.assertEqual(state.next_sequence, 2)
        self.assertEqual(state.pending, [])
        self.assertEqual(state.prepared, {})
        self.assertEqual(allocator.snapshot(), before_snapshot)
        self.assertEqual(set(core._PROGRAM_REGISTRY), before_program_ids)
        self.assertNotIn(id(arena), core._SEALED_ARENA_REGISTRY)
        with self.assertRaises(ConstraintProgramError):
            core._validate_view(state, committed.view)

        forged_program, forged_state = core._stage_program(
            tuple(state.blocks.values()), core._frame_key(frame)
        )
        with self.assertRaises(ConstraintProgramError):
            core._register_staged_program(forged_program, forged_state)
        with self.assertRaises(ConstraintProgramError):
            _ = forged_program.representation_authority
        for operation in (
            lambda: arena.commit(prepared),
            lambda: arena.abort(prepared),
            lambda: arena.union(committed.view),
            lambda: arena.seal(committed.view, final_frame=frame),
            owner.frame,
        ):
            with self.assertRaises(ConstraintProgramError):
                operation()

    def test_discard_rejects_sealed_arena_without_revoking_program(self):
        _allocator, _adapter, owner, frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        result = arena.append_le_exact_tag(
            arena.empty_view,
            frame=frame,
            A_cont=_canonical([[1.0]]),
            A_bin=sp.csr_matrix((1, 0), dtype=np.float64),
            upper=np.asarray([1.0], dtype=np.float64),
            tag="sealed:forward",
        )
        program = arena.seal(result.view, final_frame=frame)
        for operation in (arena.discard, arena.close):
            with self.assertRaises(ConstraintProgramError):
                operation()
        self.assertFalse(arena.discarded)
        self.assertFalse(owner.discarded)
        self.assertTrue(program.representation_authority)
        self.assertEqual(program.source_rows, 1)

    def test_discard_pre_post_swap_baseexceptions_preserve_original_and_recover(self):
        class CustomAbort(BaseException):
            pass

        cases = (
            ("arena-pre", "_publish_arena_state", False, KeyboardInterrupt),
            ("arena-post", "_publish_arena_state", True, SystemExit),
            ("owner-pre", "_commit_owner_operation", False, CustomAbort),
            ("owner-post", "_commit_owner_operation", True, CustomAbort),
        )
        for name, helper_name, after, exception_type in cases:
            with self.subTest(name=name):
                _allocator, _adapter, owner, _frame, arena = _new_arena(
                    n_cont=1, n_bin=0
                )
                helper = getattr(core, helper_name)
                marker = exception_type(name)
                calls = 0

                def interrupt_once(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        if after:
                            helper(*args, **kwargs)
                        raise marker
                    return helper(*args, **kwargs)

                with mock.patch.object(
                    core, helper_name, side_effect=interrupt_once
                ):
                    with self.assertRaises(exception_type) as caught:
                        arena.discard()
                self.assertIs(caught.exception, marker)
                state = core._arena_state(arena)
                self.assertFalse(state.discarded)
                self.assertFalse(state.owner_state.discarded)
                self.assertFalse(state.sealed)
                self.assertFalse(state.owner_state.sealed)
                self.assertFalse(
                    core._owner_operation_entry(state.owner_state).operation_active
                )
                arena.discard()
                self.assertTrue(arena.discarded)
                self.assertTrue(owner.discarded)

        _allocator, _adapter, owner, _frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )

        def interrupt_return(frame_obj, event, _arg):
            if (
                frame_obj.f_code is core.ConstraintArena.discard.__code__
                and event == "return"
            ):
                sys.settrace(None)
                raise KeyboardInterrupt("discard public return")
            return interrupt_return

        sys.settrace(interrupt_return)
        try:
            with self.assertRaises(KeyboardInterrupt):
                arena.discard()
        finally:
            sys.settrace(None)
        state = core._arena_state(arena)
        self.assertTrue(state.discarded)
        self.assertTrue(state.owner_state.discarded)
        self.assertFalse(
            core._owner_operation_entry(state.owner_state).operation_active
        )
        self.assertTrue(owner.discarded)

    def test_poisoned_owner_can_still_discard_without_external_callback(self):
        allocator = _FakeAllocator()
        fail = {"snapshot": False}

        def snapshot():
            if fail["snapshot"]:
                raise KeyboardInterrupt("allocator snapshot failed")
            return allocator.snapshot()

        adapter = ExternalFactorAllocatorAdapter.bind(
            allocator,
            allocate_continuous=allocator.allocate_continuous,
            allocate_binary=allocator.allocate_binary,
            live_ids_snapshot=snapshot,
        )
        owner = ConstraintProgramOwner(adapter)
        owner.allocate_continuous(1)
        owner.frame()
        arena = owner.new_arena()
        fail["snapshot"] = True
        with self.assertRaises(KeyboardInterrupt):
            owner.frame()
        self.assertTrue(core._owner_state(owner).poisoned)
        calls = allocator.cont_calls
        arena.discard()
        self.assertTrue(arena.discarded)
        self.assertTrue(owner.discarded)
        self.assertTrue(core._owner_state(owner).poisoned)
        self.assertEqual(allocator.cont_calls, calls)

    def test_discard_rollback_second_interrupt_keeps_first_exception(self):
        class FirstAbort(BaseException):
            pass

        _allocator, _adapter, owner, _frame, arena = _new_arena(
            n_cont=1, n_bin=0
        )
        original_publish = core._publish_arena_state
        first = FirstAbort("discard publication")
        calls = 0

        def interrupt_publication_and_first_repair(*args, **kwargs):
            nonlocal calls
            calls += 1
            original_publish(*args, **kwargs)
            if calls == 1:
                raise first
            if calls == 2:
                raise KeyboardInterrupt("discard rollback repair")

        with mock.patch.object(
            core,
            "_publish_arena_state",
            side_effect=interrupt_publication_and_first_repair,
        ):
            with self.assertRaises(FirstAbort) as caught:
                arena.discard()
        self.assertIs(caught.exception, first)
        state = core._arena_state(arena)
        self.assertFalse(state.discarded)
        self.assertFalse(state.owner_state.discarded)
        self.assertFalse(
            core._owner_operation_entry(state.owner_state).operation_active
        )
        arena.discard()
        self.assertTrue(arena.discarded)
        self.assertTrue(owner.discarded)

    def test_discard_reentrancy_thread_and_gc_leave_no_program(self):
        allocator = _FakeAllocator()
        holder = {"action": None, "caught": []}

        def snapshot():
            action = holder["action"]
            if action is not None:
                holder["action"] = None
                try:
                    action()
                except BaseException as error:
                    holder["caught"].append(error)
            return allocator.snapshot()

        adapter = ExternalFactorAllocatorAdapter.bind(
            allocator,
            allocate_continuous=allocator.allocate_continuous,
            allocate_binary=allocator.allocate_binary,
            live_ids_snapshot=snapshot,
        )
        owner = ConstraintProgramOwner(adapter)
        owner.allocate_continuous(1)
        frame = owner.frame()
        arena = owner.new_arena()
        empty = arena.empty_view
        holder["action"] = arena.discard
        with self.assertRaises(ExternalAllocatorContractError):
            arena.seal(empty, final_frame=frame)
        self.assertEqual(len(holder["caught"]), 1)
        self.assertIsInstance(holder["caught"][0], ConstraintProgramError)
        state = core._arena_state(arena)
        self.assertFalse(state.sealed)
        self.assertFalse(state.discarded)
        self.assertNotIn(id(arena), core._SEALED_ARENA_REGISTRY)

        errors = []

        def wrong_thread_discard():
            try:
                arena.discard()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=wrong_thread_discard)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ConstraintProgramError)
        self.assertFalse(core._arena_state(arena).discarded)

        object_id = id(arena)
        reference = weakref.ref(arena)
        arena.discard()
        holder["caught"].clear()
        holder["action"] = None
        errors.clear()
        del arena, state
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(object_id, core._ARENA_REGISTRY)
        self.assertTrue(owner.discarded)
        with self.assertRaises(ConstraintProgramError):
            owner.new_arena()


class ConstraintProgramSemanticTests(unittest.TestCase):
    def _mixed_program(self):
        _allocator, _adapter, _owner, frame, arena = _new_arena()
        fc = _canonical([[1.0, 2.0], [3.0, -4.0], [-0.5, 0.25]])
        fb = _canonical([[0.5], [-0.25], [2.0]])
        rc, rb = -fc, -fb
        rc.data[-1] = np.nextafter(rc.data[-1], np.inf)
        fu = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        ru = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)
        result = _append_band(
            arena, frame, fc=fc, fb=fb, fu=fu, rc=rc, rb=rb, ru=ru
        )
        return arena.seal(result.view, final_frame=frame)

    def test_ten_thousand_fraction_memberships_match(self):
        program = self._mixed_program()
        native = _collect(program, native=True, max_rows=2)
        legacy = _collect(program, native=False, max_rows=2)
        rng = random.Random(0xC0FFEE)
        for _ in range(10_000):
            continuous = tuple(
                Fraction(rng.randint(-1024, 1024), 1024)
                for _unused in range(native["A_cont"].shape[1])
            )
            binary = tuple(
                Fraction(rng.choice((-1, 1)))
                for _unused in range(native["A_bin"].shape[1])
            )
            native_ok = True
            for row in range(native["A_cont"].shape[0]):
                value = _row_value(native["A_cont"], row, continuous) + _row_value(
                    native["A_bin"], row, binary
                )
                lower = float(native["lower"][row])
                upper = Fraction.from_float(float(native["upper"][row]))
                if (not np.isneginf(lower) and value < Fraction.from_float(lower)) or value > upper:
                    native_ok = False
                    break
            legacy_ok = True
            for row in range(legacy["A_cont"].shape[0]):
                value = _row_value(legacy["A_cont"], row, continuous) + _row_value(
                    legacy["A_bin"], row, binary
                )
                if value > Fraction.from_float(float(legacy["upper"][row])):
                    legacy_ok = False
                    break
            self.assertEqual(native_ok, legacy_ok)

    def test_small_highs_lp_native_and_legacy_optima_match_without_status_authority(self):
        program = self._mixed_program()
        native = _collect(program, native=True, max_rows=1)
        legacy = _collect(program, native=False, max_rows=1)
        native_A = sp.hstack((native["A_cont"], native["A_bin"]), format="csr")
        legacy_A = sp.hstack((legacy["A_cont"], legacy["A_bin"]), format="csr")
        rows = []
        bounds = []
        for row in range(native_A.shape[0]):
            rows.append(native_A.getrow(row))
            bounds.append(float(native["upper"][row]))
            if np.isfinite(native["lower"][row]):
                rows.append(-native_A.getrow(row))
                bounds.append(float(-native["lower"][row]))
        expanded_A = sp.vstack(rows, format="csr")
        expanded_b = np.asarray(bounds, dtype=np.float64)
        objective = np.asarray([0.75, -0.5, 0.125], dtype=np.float64)
        common_bounds = [(-1.0, 1.0)] * objective.size
        native_result = scipy.optimize.linprog(
            objective,
            A_ub=expanded_A,
            b_ub=expanded_b,
            bounds=common_bounds,
            method="highs",
        )
        legacy_result = scipy.optimize.linprog(
            objective,
            A_ub=legacy_A,
            b_ub=legacy["upper"],
            bounds=common_bounds,
            method="highs",
        )
        self.assertEqual(native_result.success, legacy_result.success)
        self.assertAlmostEqual(native_result.fun, legacy_result.fun, places=11)
        self.assertIs(program.solver_status_authority, False)
        self.assertIs(dict(program.receipt)["solver_status_authority"], False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
